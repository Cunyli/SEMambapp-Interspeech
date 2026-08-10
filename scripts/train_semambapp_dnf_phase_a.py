"""Controlled scratch Standard-versus-DNF Phase A training entry point."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.dnf_controlled_phase_a import (
    ARTIFICIAL_NOISE_ENERGY_POLICY,
    DEPLOYMENT_INPUT_DEFINITION,
    NOISE_PAIRING_CROSS_FAMILY_CYCLE,
    NOISE_PAIRING_SAME_FAMILY_IID,
    ROUTE_CLEAN_REGULAR,
    ROUTE_CLEAN_WEAK,
    ROUTE_NOISY,
    SPEECH_PARTITION_DISJOINT,
    SNR_DEFINITION,
    TRAINING_INPUT_DEFINITION,
    PhaseAControlledStreamDataset,
    gap_worker_init_fn,
    phase_a_collate,
)
from model.dnf_paper import (
    dnf_clean_loss_eq15,
    dnf_noisy_loss_eq13,
    dnf_output_eq14,
    sdr_loss_eq5,
    si_sdr_loss,
)
from model.dnf_phase_a import active_frame_log_rms_loss
from model.dnf_semambapp import DNFSEMambapp
from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled scratch DNF Phase A gate.")
    parser.add_argument("--mode", choices=("standard", "dnf"), required=True)
    parser.add_argument(
        "--loss-variant",
        choices=("paper_exact", "matched_scale"),
        default="paper_exact",
    )
    parser.add_argument(
        "--config",
        default="configs/train/semambapp_shifted_anechoic_online_v1.yaml",
    )
    parser.add_argument(
        "--contract",
        default="configs/train/dnf_phase_ab_v2_contract.json",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs/semambapp_dnf_phase_a"))
    parser.add_argument("--pair-contract-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cut-duration", type=float, default=1.0)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--listening-samples", type=int, default=5)
    parser.add_argument(
        "--checkpoint-steps",
        nargs="*",
        type=int,
        default=(250, 500, 1000, 2000),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-eps", type=float, default=1e-8)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_a_code_surface_sha256() -> dict[str, str]:
    """Freeze every model module plus the Phase-A loader and runner."""

    paths = sorted((REPO_ROOT / "model").rglob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "dataloaders" / "dnf_controlled_phase_a.py",
            Path(__file__).resolve(),
        ]
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase A code surface is incomplete: {missing}")
    return {
        str(path.resolve().relative_to(REPO_ROOT.resolve())): sha256_file(path)
        for path in sorted(set(paths))
    }


def canonical_speech_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(("noise_mag_decoder.", "noise_phase_decoder.")):
            continue
        canonical_name = name.replace("speech_mag_decoder.", "mask_decoder.")
        canonical_name = canonical_name.replace("speech_phase_decoder.", "phase_decoder.")
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def update_uid_digest(digest: "hashlib._Hash", sample_uids: list[str]) -> None:
    for uid in sample_uids:
        encoded = str(uid).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)


def sum_over_total_microbatch(
    loss_vectors: list[torch.Tensor],
    total_microbatch_size: int,
) -> torch.Tensor:
    if total_microbatch_size <= 0:
        raise ValueError("total_microbatch_size must be positive")
    nonempty = [values.reshape(-1) for values in loss_vectors if values.numel()]
    if not nonempty:
        raise ValueError("at least one non-empty loss vector is required")
    return sum(values.sum() for values in nonempty) / int(total_microbatch_size)


def validate_pair_receipts(receipts: dict[str, dict]) -> dict:
    if set(receipts) != {"standard", "dnf"}:
        raise ValueError("pair receipts must contain exactly standard and dnf")
    keys = (
        "train_manifest_sha256",
        "valid_manifest_sha256",
        "train_manifest_length",
        "valid_manifest_length",
        "canonical_speech_init_sha256",
        "uid_sequence_sha256",
        "uid_sequence_count",
        "seed",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "learning_rate",
        "cut_duration_seconds",
        "validation_samples",
        "checkpoint_steps",
        "geometry_eps",
        "loss_variant",
        "active_log_rms_weight",
        "contract_sha256",
        "model_config_sha256",
        "training_script_sha256",
        "code_surface_sha256",
        "noise_pairing_policy",
        "speech_partition_policy",
        "deployment_validation_input",
        "evaluation_input_views",
        "paper_mechanism_gate",
    )
    mismatches = {
        key: {mode: receipts[mode].get(key) for mode in ("standard", "dnf")}
        for key in keys
        if receipts["standard"].get(key) != receipts["dnf"].get(key)
    }
    if mismatches:
        raise ValueError(f"Phase A pair contract mismatch: {mismatches}")
    return {"status": "matched", **{key: receipts["standard"][key] for key in keys}}


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("schema_version") != "dnf_phase_ab_v2" or contract.get("phase") != "A":
        raise ValueError("unexpected Phase A contract")
    return contract


def validate_runtime_contract(args: argparse.Namespace, contract: dict) -> None:
    training = contract["training"]
    if args.mode not in contract["allowed_modes"]:
        raise ValueError(f"mode {args.mode} is not allowed")
    if args.max_steps != int(training["max_steps"]):
        raise ValueError("max_steps differs from the controlled contract")
    if args.seed != int(training["seed"]):
        raise ValueError("seed differs from the controlled contract")
    if args.batch_size != int(training["batch_size"]):
        raise ValueError("batch_size differs from the controlled contract")
    if args.gradient_accumulation_steps != int(training["gradient_accumulation_steps"]):
        raise ValueError("gradient accumulation differs from the controlled contract")
    if abs(args.cut_duration - float(contract["data"]["cut_duration_seconds"])) > 1e-12:
        raise ValueError("cut duration differs from the controlled contract")
    if args.validation_samples != int(training["validation_samples"]):
        raise ValueError("validation sample count differs from the controlled contract")
    if sorted(args.checkpoint_steps) != sorted(training["checkpoint_steps"]):
        raise ValueError("checkpoint steps differ from the controlled contract")
    variants = contract["loss"]["active_log_rms"]["variants"]
    if args.loss_variant not in variants:
        raise ValueError("loss variant is not frozen in the controlled contract")
    expected_weights = {"paper_exact": 0.0, "matched_scale": 1.0}
    if float(variants[args.loss_variant]["weight"]) != expected_weights[
        args.loss_variant
    ]:
        raise ValueError("loss-variant weight differs from the controlled contract")


def active_log_rms_weight(contract: dict, loss_variant: str) -> float:
    return float(
        contract["loss"]["active_log_rms"]["variants"][loss_variant]["weight"]
    )


def load_model_cfg(path: str | Path, args: argparse.Namespace) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["env_setting"]["seed"] = int(args.seed)
    cfg["training_cfg"]["batch_size"] = int(args.batch_size)
    cfg["training_cfg"]["segment_size"] = int(
        round(args.cut_duration * int(cfg["stft_cfg"]["sampling_rate"]))
    )
    if args.learning_rate is not None:
        cfg["training_cfg"]["learning_rate"] = float(args.learning_rate)
    return cfg


def validate_manifest_routes(dataset: PhaseAControlledStreamDataset) -> None:
    counts = dataset.route_summary["route_counts"]
    total = int(dataset.route_summary["sample_count"])
    if total % 20:
        raise ValueError("Phase A manifest length must be a multiple of 20")
    blocks = total // 20
    expected = {
        ROUTE_NOISY: 15 * blocks,
        ROUTE_CLEAN_REGULAR: 4 * blocks,
        ROUTE_CLEAN_WEAK: blocks,
    }
    if counts != expected:
        raise ValueError(f"manifest route counts {counts} != {expected}")
    expected_families = {"hvac", "fan", "vehicle_cabin"}
    if set(dataset.route_summary["noise_families"]) != expected_families:
        raise ValueError("manifest does not contain the three indoor-noise families")


def validate_manifest_speech_sources(
    dataset: PhaseAControlledStreamDataset,
    contract: dict,
    *,
    seed: int,
) -> None:
    allowed = set(contract["data"]["clean_sources"])
    expected_semantics = {
        "snr_definition": SNR_DEFINITION,
        "training_input": TRAINING_INPUT_DEFINITION,
        "deployment_validation_input": DEPLOYMENT_INPUT_DEFINITION,
        "artificial_noise_energy_policy": ARTIFICIAL_NOISE_ENERGY_POLICY,
    }
    for key, expected in expected_semantics.items():
        if contract["data"].get(key) != expected:
            raise ValueError(f"contract {key} differs from loader semantics")
    for row in dataset.rows:
        if int(row.get("manifest_seed", -1)) != int(seed):
            raise ValueError("Phase A manifest seed differs from the runtime seed")
        if row.get("speech_source_category") != "clean_strict":
            raise ValueError("Phase A speech must be labeled clean_strict")
        speech = row.get("speech", {})
        source = str(speech.get("dataset", speech.get("source", "")))
        if source not in allowed:
            raise ValueError(f"Phase A manifest contains forbidden speech source {source!r}")


def validate_manifest_noise_pairing(
    dataset: PhaseAControlledStreamDataset,
) -> str:
    policies = set(dataset.route_summary["noise_pairing_policies"])
    if policies == {"legacy_unlabeled"}:
        same_family = [
            row["noise1"]["family"] == row["noise2"]["family"]
            for row in dataset.rows
        ]
        if all(same_family):
            return "legacy_unlabeled_same_family"
        if not any(same_family):
            return "legacy_unlabeled_cross_family"
        raise ValueError("legacy manifest mixes same- and cross-family noise pairs")
    if len(policies) != 1:
        raise ValueError(f"manifest mixes noise-pairing policies: {policies}")
    policy = next(iter(policies))
    for row in dataset.rows:
        same_family = row["noise1"]["family"] == row["noise2"]["family"]
        if row["noise1"]["seed"] == row["noise2"]["seed"]:
            raise ValueError("n1 and n2 must use distinct realization seeds")
        if policy == NOISE_PAIRING_SAME_FAMILY_IID and not same_family:
            raise ValueError("same_family_iid row contains a cross-family pair")
        if policy == NOISE_PAIRING_CROSS_FAMILY_CYCLE and same_family:
            raise ValueError("cross_family_cycle row contains a same-family pair")
    if policy not in {
        NOISE_PAIRING_SAME_FAMILY_IID,
        NOISE_PAIRING_CROSS_FAMILY_CYCLE,
    }:
        raise ValueError(f"unsupported noise-pairing policy {policy!r}")
    return policy


def validate_manifest_speech_partition(
    dataset: PhaseAControlledStreamDataset,
) -> str:
    policies = set(dataset.route_summary["speech_partition_policies"])
    if policies == {"legacy_unlabeled"}:
        return "legacy_unlabeled"
    if policies != {SPEECH_PARTITION_DISJOINT}:
        raise ValueError(f"unsupported speech-partition policies: {policies}")
    clean_speech = {
        json.dumps(row["speech"], sort_keys=True)
        for row in dataset.rows
        if row["route"] != ROUTE_NOISY
    }
    noisy_speech = {
        json.dumps(row["speech"], sort_keys=True)
        for row in dataset.rows
        if row["route"] == ROUTE_NOISY
    }
    overlap = clean_speech & noisy_speech
    if overlap:
        raise ValueError(
            f"clean/noisy speech partitions overlap by {len(overlap)} items"
        )
    if len(clean_speech) + len(noisy_speech) != len(dataset.rows):
        raise ValueError("speech items must be unique within the Phase A manifest")
    return SPEECH_PARTITION_DISJOINT


def validate_seen_route_counts(route_counter: Counter[str]) -> None:
    total = sum(route_counter.values())
    if total == 0 or total % 20:
        return
    blocks = total // 20
    expected = {
        ROUTE_NOISY: 15 * blocks,
        ROUTE_CLEAN_REGULAR: 4 * blocks,
        ROUTE_CLEAN_WEAK: blocks,
    }
    if dict(route_counter) != expected:
        raise RuntimeError(f"seen route counts {dict(route_counter)} != {expected}")


def build_loader(
    manifest: Path,
    split: str,
    args: argparse.Namespace,
    cfg: dict,
    expose_clean_for_eval: bool,
) -> DataLoader:
    limit = int(args.validation_samples) if expose_clean_for_eval else None
    dataset = PhaseAControlledStreamDataset(
        split_root=args.split_root,
        contract_path=manifest,
        split=split,
        samples_per_epoch=limit,
        target_sample_rate=int(cfg["stft_cfg"]["sampling_rate"]),
        cut_duration=float(args.cut_duration),
        seed=int(args.seed),
        expose_clean_for_eval=expose_clean_for_eval,
    )
    validate_manifest_routes(dataset)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=not expose_clean_for_eval,
        collate_fn=phase_a_collate,
        worker_init_fn=gap_worker_init_fn if int(args.num_workers) else None,
    )


def stft_features(waveform: torch.Tensor, cfg: dict):
    return mag_phase_stft(
        waveform,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
        addeps=True,
    )


def istft_waveform(magnitude: torch.Tensor, phase: torch.Tensor, cfg: dict):
    return mag_phase_istft(
        magnitude,
        phase,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
    )


def crop_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(left.size(-1), right.size(-1))
    return left[..., :length], right[..., :length]


def decode_model_outputs(
    model: torch.nn.Module,
    mode: str,
    model_input: torch.Tensor,
    cfg: dict,
    eps: float,
) -> dict[str, torch.Tensor]:
    magnitude, phase, _ = stft_features(model_input, cfg)
    if mode == "standard":
        output_magnitude, output_phase, _ = model(magnitude, phase)
        return {"standard": istft_waveform(output_magnitude, output_phase, cfg)}
    branches = model(magnitude, phase)
    speech = istft_waveform(branches["speech_mag"], branches["speech_pha"], cfg)
    noise = istft_waveform(branches["noise_mag"], branches["noise_pha"], cfg)
    speech, noise = crop_pair(speech, noise)
    projection = dnf_output_eq14(speech, noise, eps=eps)
    return {"eq14": projection.enhanced, "speech_head": speech, "noise_head": noise}


def _valid_values(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask, values, torch.zeros_like(values))


def nonzero_energy_mask(waveform: torch.Tensor, eps: float) -> torch.Tensor:
    return waveform.float().square().sum(dim=-1) >= float(eps)


def eq13_input_valid_mask(
    speech_estimate: torch.Tensor,
    noise_estimate: torch.Tensor,
    noisy_target: torch.Tensor,
    artificial_noise: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    artificial_energy = artificial_noise.float().square().sum(dim=-1)
    speech_inner = (
        artificial_noise.float() * speech_estimate.float()
    ).sum(dim=-1)
    noise_inner = (
        artificial_noise.float() * noise_estimate.float()
    ).sum(dim=-1)
    return (
        (artificial_energy >= float(eps))
        & nonzero_energy_mask(noisy_target, eps)
        & (speech_inner.abs() >= float(eps))
        & (noise_inner.abs() >= float(eps))
    )


def run_standard_step(
    model: torch.nn.Module,
    batch: dict,
    cfg: dict,
    contract: dict,
    eps: float,
    scale_weight: float,
) -> tuple[torch.Tensor, dict]:
    outputs = decode_model_outputs(model, "standard", batch["model_input_wav"], cfg, eps)
    estimate = outputs["standard"]
    clean_estimate = estimate.index_select(0, batch["clean_indices"])
    noisy_estimate = estimate.index_select(0, batch["noisy_indices"])
    clean_estimate, clean_target = crop_pair(clean_estimate, batch["clean_speech_wav"])
    noisy_estimate, noisy_target = crop_pair(
        noisy_estimate, batch["noisy_speech_target_wav"]
    )
    clean_loss = si_sdr_loss(clean_estimate, clean_target, eps=eps)
    noisy_loss = si_sdr_loss(noisy_estimate, noisy_target, eps=eps)
    if not clean_loss.valid_mask.all() or not noisy_loss.valid_mask.all():
        raise RuntimeError("Standard Phase A received a silent supervision target")
    base = sum_over_total_microbatch(
        [
            clean_loss.value,
            noisy_loss.value,
        ],
        estimate.size(0),
    )
    clean_rms = active_frame_log_rms_loss(
        clean_estimate,
        clean_target,
        frame_length=int(contract["loss"]["active_log_rms"]["frame_length"]),
        hop_length=int(contract["loss"]["active_log_rms"]["hop_length"]),
        relative_activity_db=float(
            contract["loss"]["active_log_rms"]["relative_activity_db"]
        ),
    )
    noisy_rms = active_frame_log_rms_loss(
        noisy_estimate,
        noisy_target,
        frame_length=int(contract["loss"]["active_log_rms"]["frame_length"]),
        hop_length=int(contract["loss"]["active_log_rms"]["hop_length"]),
        relative_activity_db=float(
            contract["loss"]["active_log_rms"]["relative_activity_db"]
        ),
    )
    clean_rms_sum = clean_rms.per_sample_loss[
        clean_rms.active_sample_mask
    ].sum()
    noisy_rms_sum = noisy_rms.per_sample_loss[
        noisy_rms.active_sample_mask
    ].sum()
    rms_term = (clean_rms_sum + noisy_rms_sum) / estimate.size(0)
    total = base + scale_weight * rms_term
    return total, {
        "sample_count": int(estimate.size(0)),
        "clean_count": int(clean_estimate.size(0)),
        "noisy_count": int(noisy_estimate.size(0)),
        "standard_invalid_count": int((~clean_loss.valid_mask).sum().item())
        + int((~noisy_loss.valid_mask).sum().item()),
        "eq13_invalid_count": 0,
        "eq14_invalid_count": 0,
        "eq15_invalid_count": 0,
        "rms_invalid_count": int(
            clean_rms.empty_sample_mask.sum().item()
            + noisy_rms.empty_sample_mask.sum().item()
        ),
        "base_loss": float(base.detach().item()),
        "active_log_rms_loss": float(rms_term.detach().item()),
        "clean_active_log_rms_loss": float(
            (clean_rms_sum / estimate.size(0)).detach().item()
        ),
        "noisy_active_log_rms_loss": float(
            (noisy_rms_sum / estimate.size(0)).detach().item()
        ),
        "loss": float(total.detach().item()),
    }


def run_dnf_step(
    model: torch.nn.Module,
    batch: dict,
    cfg: dict,
    contract: dict,
    eps: float,
    scale_weight: float,
) -> tuple[torch.Tensor, dict]:
    outputs = decode_model_outputs(model, "dnf", batch["model_input_wav"], cfg, eps)
    speech = outputs["speech_head"]
    noise = outputs["noise_head"]
    clean_speech = speech.index_select(0, batch["clean_indices"])
    clean_noise = noise.index_select(0, batch["clean_indices"])
    noisy_speech = speech.index_select(0, batch["noisy_indices"])
    noisy_noise = noise.index_select(0, batch["noisy_indices"])
    clean_length = min(
        clean_speech.size(-1),
        batch["clean_speech_wav"].size(-1),
        batch["mixture_noise_wav"].size(-1),
    )
    noisy_length = min(
        noisy_speech.size(-1),
        batch["noisy_speech_target_wav"].size(-1),
        batch["artificial_noise_wav"].size(-1),
    )
    clean_speech = clean_speech[..., :clean_length]
    clean_noise = clean_noise[..., :clean_length]
    clean_target = batch["clean_speech_wav"][..., :clean_length]
    mixture_noise = batch["mixture_noise_wav"][..., :clean_length]
    clean_input_valid = (
        nonzero_energy_mask(clean_target, eps)
        & nonzero_energy_mask(mixture_noise, eps)
        & nonzero_energy_mask(clean_target + 0.5 * mixture_noise, eps)
        & nonzero_energy_mask(clean_noise, eps)
    )
    if clean_input_valid.any():
        clean_loss = dnf_clean_loss_eq15(
            clean_speech[clean_input_valid],
            clean_noise[clean_input_valid],
            clean_target[clean_input_valid],
            mixture_noise[clean_input_valid],
            eps=eps,
        )
        if not clean_loss.valid_mask.all():
            raise RuntimeError("Pre-filtered equation (15) sample became invalid")
        clean_total = clean_loss.total
    else:
        clean_total = speech.new_empty(0)

    noisy_speech = noisy_speech[..., :noisy_length]
    noisy_noise = noisy_noise[..., :noisy_length]
    noisy_target = batch["noisy_speech_target_wav"][..., :noisy_length]
    artificial_noise = batch["artificial_noise_wav"][..., :noisy_length]
    noisy_input_valid = eq13_input_valid_mask(
        noisy_speech,
        noisy_noise,
        noisy_target,
        artificial_noise,
        eps,
    )
    if noisy_input_valid.any():
        noisy_loss = dnf_noisy_loss_eq13(
            noisy_speech[noisy_input_valid],
            noisy_noise[noisy_input_valid],
            noisy_target[noisy_input_valid],
            artificial_noise[noisy_input_valid],
            eps=eps,
            scale_clamp=None,
        )
        if not noisy_loss.valid_mask.all() or not noisy_loss.faithful_mask.all():
            raise RuntimeError("Pre-filtered equation (13) sample became invalid")
        noisy_total = noisy_loss.total
    else:
        noisy_total = speech.new_empty(0)
    base = sum_over_total_microbatch(
        [clean_total, noisy_total],
        speech.size(0),
    )
    clean_projection = dnf_output_eq14(clean_speech, clean_noise, eps=eps)
    final_clean = clean_projection.enhanced
    clean_rms = active_frame_log_rms_loss(
        final_clean,
        clean_target,
        frame_length=int(contract["loss"]["active_log_rms"]["frame_length"]),
        hop_length=int(contract["loss"]["active_log_rms"]["hop_length"]),
        relative_activity_db=float(
            contract["loss"]["active_log_rms"]["relative_activity_db"]
        ),
    )
    noisy_rms = active_frame_log_rms_loss(
        noisy_speech,
        noisy_target,
        frame_length=int(contract["loss"]["active_log_rms"]["frame_length"]),
        hop_length=int(contract["loss"]["active_log_rms"]["hop_length"]),
        relative_activity_db=float(
            contract["loss"]["active_log_rms"]["relative_activity_db"]
        ),
    )
    clean_rms_sum = clean_rms.per_sample_loss[
        clean_rms.active_sample_mask
    ].sum()
    noisy_rms_sum = noisy_rms.per_sample_loss[
        noisy_rms.active_sample_mask
    ].sum()
    rms_term = (clean_rms_sum + noisy_rms_sum) / speech.size(0)
    total = base + scale_weight * rms_term
    projection_all = dnf_output_eq14(speech, noise, eps=eps)
    return total, {
        "sample_count": int(speech.size(0)),
        "clean_count": int(clean_speech.size(0)),
        "noisy_count": int(noisy_speech.size(0)),
        "standard_invalid_count": 0,
        "eq13_invalid_count": int((~noisy_input_valid).sum().item()),
        "eq14_invalid_count": int((~projection_all.valid_mask).sum().item()),
        "eq15_invalid_count": int((~clean_input_valid).sum().item()),
        "rms_invalid_count": int(
            clean_rms.empty_sample_mask.sum().item()
            + noisy_rms.empty_sample_mask.sum().item()
        ),
        "base_loss": float(base.detach().item()),
        "active_log_rms_loss": float(rms_term.detach().item()),
        "clean_active_log_rms_loss": float(
            (clean_rms_sum / speech.size(0)).detach().item()
        ),
        "noisy_active_log_rms_loss": float(
            (noisy_rms_sum / speech.size(0)).detach().item()
        ),
        "loss": float(total.detach().item()),
    }


def finite_gradient_summary(model: torch.nn.Module) -> dict:
    total = 0
    finite = 0
    maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        total += gradient.numel()
        finite += int(torch.isfinite(gradient).sum().item())
        if gradient.numel():
            maximum = max(maximum, float(gradient.abs().max().item()))
    return {"total": total, "finite": finite, "max_abs": maximum}


def aggregate_microbatch_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate an empty microbatch window")
    additive = {
        "sample_count",
        "clean_count",
        "noisy_count",
        "standard_invalid_count",
        "eq13_invalid_count",
        "eq14_invalid_count",
        "eq15_invalid_count",
        "rms_invalid_count",
    }
    keys = set().union(*(row.keys() for row in rows))
    output = {}
    for key in sorted(keys):
        values = [row[key] for row in rows if key in row]
        output[key] = (
            int(sum(values))
            if key in additive
            else float(sum(values) / len(values))
        )
    return output


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    cfg: dict,
    step: int,
    counters: dict,
    metadata: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "cfg": cfg,
            "step": int(step),
            "counters": counters,
            "metadata": metadata,
            "scratch_init": True,
            "init_checkpoint": None,
            "resume": None,
        },
        path,
    )


def rms_gain_db(estimate: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    estimate_rms = estimate.float().square().mean(dim=-1).sqrt()
    reference_rms = reference.float().square().mean(dim=-1).sqrt()
    return 20.0 * torch.log10((estimate_rms + eps) / (reference_rms + eps))


def active_frame_gain_db(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    frame_length: int,
    hop_length: int,
    relative_activity_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = active_frame_log_rms_loss(
        estimate,
        reference,
        frame_length=frame_length,
        hop_length=hop_length,
        relative_activity_db=relative_activity_db,
    )
    floor = torch.finfo(result.estimate_frame_rms.dtype).tiny
    signed_log_ratio = (
        result.estimate_frame_rms.clamp_min(floor).log()
        - result.reference_frame_rms.clamp_min(floor).log()
    )
    weights = result.active_frame_mask.to(signed_log_ratio.dtype)
    mean_log_ratio = (signed_log_ratio * weights).sum(dim=-1) / (
        result.active_frame_count.clamp_min(1).to(signed_log_ratio.dtype)
    )
    gain_db = mean_log_ratio * (20.0 / math.log(10.0))
    return gain_db, result.active_sample_mask


def summarize(values: list[torch.Tensor]) -> dict:
    tensor = torch.cat(values).float()
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p05": float(torch.quantile(tensor, 0.05).item()),
        "p50": float(torch.quantile(tensor, 0.5).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def safe_uid(uid: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in uid)[:160]


def select_listening_uids(
    rows: list[dict],
    *,
    limit: int,
) -> set[str]:
    """Select a deterministic route/family/SNR-stratified listening panel."""

    if limit <= 0:
        return set()
    buckets: dict[tuple[str, str, float], list[str]] = {}
    for row in rows:
        key = (
            str(row["route"]),
            str(row["noise1"]["family"]),
            float(row["target_snr_db"]),
        )
        buckets.setdefault(key, []).append(str(row["uid"]))
    for uids in buckets.values():
        uids.sort()
    ordered_buckets = sorted(buckets)
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in ordered_buckets:
            uids = buckets[key]
            if depth < len(uids):
                selected.append(uids[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return set(selected)


def dnf_geometry_diagnostics(
    speech: torch.Tensor,
    noise: torch.Tensor,
    clean: torch.Tensor,
    model_input: torch.Tensor,
    *,
    eps: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Record deployment geometry without turning uncalibrated values into gates."""

    projection = dnf_output_eq14(speech, noise, eps=eps)
    true_noise = model_input - clean
    speech = speech.float()
    noise = noise.float()
    clean = clean.float()
    true_noise = true_noise.float()
    enhanced = projection.enhanced.float()
    floor = max(float(eps), float(torch.finfo(speech.dtype).tiny))

    def energy(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum(dim=-1)

    def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * right).sum(dim=-1)

    true_noise_energy = energy(true_noise).clamp_min(floor)
    clean_energy = energy(clean).clamp_min(floor)
    speech_energy = energy(speech).clamp_min(floor)
    noise_energy = energy(noise).clamp_min(floor)
    speech_noise_cosine = inner(speech, noise) / torch.sqrt(
        speech_energy * noise_energy
    )
    speech_true_noise_coefficient = inner(speech, true_noise) / true_noise_energy
    noise_true_noise_coefficient = inner(noise, true_noise) / true_noise_energy
    clean_leakage_energy_fraction = (
        inner(noise, clean).square() / (clean_energy * noise_energy)
    )

    def residual_true_noise_coefficient(
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        return inner(estimate - clean, true_noise) / true_noise_energy

    speech_residual = residual_true_noise_coefficient(speech)
    eq14_residual = residual_true_noise_coefficient(enhanced)
    diagnostics = {
        "eq14_projection_coefficient": projection.projection_coefficient,
        "speech_noise_head_cosine": speech_noise_cosine,
        "speech_head_true_noise_coefficient": speech_true_noise_coefficient,
        "noise_head_true_noise_coefficient": noise_true_noise_coefficient,
        "branch_true_noise_coefficient_abs_gap": (
            speech_true_noise_coefficient - noise_true_noise_coefficient
        ).abs(),
        "noise_head_clean_leakage_energy_fraction": (
            clean_leakage_energy_fraction
        ),
        "speech_head_residual_true_noise_coefficient_abs": (
            speech_residual.abs()
        ),
        "eq14_residual_true_noise_coefficient_abs": eq14_residual.abs(),
    }
    return diagnostics, projection.valid_mask


def evaluate_end(
    model: torch.nn.Module,
    mode: str,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    eps: float,
    listening_limit: int,
    contract: dict,
) -> dict:
    model.eval()
    values: dict[str, list[torch.Tensor]] = {}
    per_sample_rows = []
    listening_rows = []
    listening_dir = output_dir / "listening"
    listening_uids = select_listening_uids(
        loader.dataset.rows,
        limit=listening_limit,
    )
    eq14_invalid_count = 0
    eq14_evaluated_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            clean = batch["eval_clean_wav"]
            evaluation_views = {
                "single_noise_s_plus_n1": batch["eval_model_input_wav"],
                "identity_clean_s": clean,
            }
            for view_name, model_input in evaluation_views.items():
                decoded = decode_model_outputs(
                    model,
                    mode,
                    model_input,
                    cfg,
                    eps,
                )
                noise_head = decoded.pop("noise_head", None)
                outputs = decoded
                length = min(
                    model_input.size(-1),
                    clean.size(-1),
                    *(waveform.size(-1) for waveform in outputs.values()),
                )
                model_input = model_input[..., :length]
                clean_view = clean[..., :length]
                outputs = {
                    name: waveform[..., :length]
                    for name, waveform in outputs.items()
                }
                geometry = {}
                if mode == "dnf":
                    if noise_head is None:
                        raise RuntimeError("DNF validation misses the noise head")
                    speech = outputs["speech_head"]
                    noise_head = noise_head[..., :length]
                    if view_name == "single_noise_s_plus_n1":
                        geometry, eq14_valid = dnf_geometry_diagnostics(
                            speech,
                            noise_head,
                            clean_view,
                            model_input,
                            eps=eps,
                        )
                    else:
                        eq14_valid = dnf_output_eq14(
                            speech,
                            noise_head,
                            eps=eps,
                        ).valid_mask
                    eq14_invalid_count += int((~eq14_valid).sum().item())
                    eq14_evaluated_count += int(eq14_valid.numel())
                    for metric, tensor in geometry.items():
                        values.setdefault(
                            f"{view_name}/geometry/{metric}",
                            [],
                        ).append(tensor.detach().cpu())

                input_si = si_sdr_loss(model_input, clean_view, eps=eps)
                input_sdr = sdr_loss_eq5(model_input, clean_view, eps=eps)
                for name, waveform in outputs.items():
                    output_si = si_sdr_loss(waveform, clean_view, eps=eps)
                    output_sdr = sdr_loss_eq5(waveform, clean_view, eps=eps)
                    if (
                        not output_si.valid_mask.all()
                        or not torch.isfinite(output_sdr).all()
                    ):
                        raise RuntimeError(
                            f"invalid validation output {view_name}/{name}"
                        )
                    metric_prefix = f"{view_name}/{name}"
                    values.setdefault(
                        f"{metric_prefix}/si_sdri_db",
                        [],
                    ).append((input_si.value - output_si.value).cpu())
                    values.setdefault(
                        f"{metric_prefix}/sdri_db",
                        [],
                    ).append((input_sdr - output_sdr).cpu())
                    values.setdefault(
                        f"{metric_prefix}/si_sdr_db",
                        [],
                    ).append((-output_si.value).cpu())
                    values.setdefault(
                        f"{metric_prefix}/sdr_db",
                        [],
                    ).append((-output_sdr).cpu())
                    values.setdefault(
                        f"{metric_prefix}/gain_db_to_clean",
                        [],
                    ).append(rms_gain_db(waveform, clean_view, eps).cpu())
                    values.setdefault(
                        f"{metric_prefix}/gain_db_to_input",
                        [],
                    ).append(rms_gain_db(waveform, model_input, eps).cpu())
                    active_gain_clean, active_gain_clean_valid = (
                        active_frame_gain_db(
                            waveform,
                            clean_view,
                            frame_length=int(
                                contract["loss"]["active_log_rms"][
                                    "frame_length"
                                ]
                            ),
                            hop_length=int(
                                contract["loss"]["active_log_rms"][
                                    "hop_length"
                                ]
                            ),
                            relative_activity_db=float(
                                contract["loss"]["active_log_rms"][
                                    "relative_activity_db"
                                ]
                            ),
                        )
                    )
                    active_gain_input, active_gain_input_valid = (
                        active_frame_gain_db(
                            waveform,
                            model_input,
                            frame_length=int(
                                contract["loss"]["active_log_rms"][
                                    "frame_length"
                                ]
                            ),
                            hop_length=int(
                                contract["loss"]["active_log_rms"][
                                    "hop_length"
                                ]
                            ),
                            relative_activity_db=float(
                                contract["loss"]["active_log_rms"][
                                    "relative_activity_db"
                                ]
                            ),
                        )
                    )
                    if (
                        not active_gain_clean_valid.all()
                        or not active_gain_input_valid.all()
                    ):
                        raise RuntimeError(
                            f"{view_name}/{name} has no active reference frames"
                        )
                    values.setdefault(
                        f"{metric_prefix}/active_gain_db_to_clean",
                        [],
                    ).append(active_gain_clean.cpu())
                    values.setdefault(
                        f"{metric_prefix}/active_gain_db_to_input",
                        [],
                    ).append(active_gain_input.cpu())

                    row_tensors = {
                        "si_sdri_db": input_si.value - output_si.value,
                        "sdri_db": input_sdr - output_sdr,
                        "si_sdr_db": -output_si.value,
                        "sdr_db": -output_sdr,
                        "gain_db_to_clean": rms_gain_db(
                            waveform,
                            clean_view,
                            eps,
                        ),
                        "gain_db_to_input": rms_gain_db(
                            waveform,
                            model_input,
                            eps,
                        ),
                        "active_gain_db_to_clean": active_gain_clean,
                        "active_gain_db_to_input": active_gain_input,
                    }
                    row_tensors = {
                        key: value.detach().cpu()
                        for key, value in row_tensors.items()
                    }
                    for index, uid in enumerate(batch["sample_uid"]):
                        row = {
                            "sample_uid": str(uid),
                            "evaluation_input_view": view_name,
                            "route": str(batch["route"][index]),
                            "noise_family": str(
                                batch["info"][index]["noise1"]["family"]
                            ),
                            "target_snr_db": float(
                                batch["info"][index]["target_snr_db"]
                            ),
                            "weak_degradation": bool(
                                batch["weak_mask"][index].item()
                            ),
                            "output_name": name,
                            **{
                                key: float(value[index].item())
                                for key, value in row_tensors.items()
                            },
                        }
                        if mode == "dnf":
                            row["eq14_valid"] = bool(
                                eq14_valid[index].item()
                            )
                        if geometry:
                            row["dnf_geometry"] = {
                                key: float(value[index].item())
                                for key, value in geometry.items()
                            }
                        per_sample_rows.append(row)

                for index, uid in enumerate(batch["sample_uid"]):
                    uid = str(uid)
                    if uid not in listening_uids:
                        continue
                    filename = f"{safe_uid(uid)}.wav"
                    sample_rate = int(batch["sample_rate"][index].item())
                    for name, waveform in {
                        "input": model_input,
                        "clean": clean_view,
                        **outputs,
                    }.items():
                        target = (
                            listening_dir
                            / view_name
                            / name
                            / filename
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        torchaudio.save(
                            str(target),
                            waveform[index].detach().cpu().unsqueeze(0),
                            sample_rate,
                            encoding="PCM_F",
                            bits_per_sample=32,
                        )
                    listening_rows.append(
                        {
                            "sample_uid": uid,
                            "evaluation_input_view": view_name,
                            "route": str(batch["route"][index]),
                            "noise_family": str(
                                batch["info"][index]["noise1"]["family"]
                            ),
                            "target_snr_db": float(
                                batch["info"][index]["target_snr_db"]
                            ),
                            "sample_rate": sample_rate,
                            "outputs": sorted(outputs),
                        }
                    )
    invalid_rate = eq14_invalid_count / max(eq14_evaluated_count, 1)
    invalid_rate_exceeds_contract = (
        mode == "dnf"
        and invalid_rate
        > float(contract["stop_gates"]["max_eq14_invalid_rate"])
    )
    summary = {
        "mode": mode,
        "evaluation_input_views": [
            "single_noise_s_plus_n1",
            "identity_clean_s",
        ],
        "metrics": {key: summarize(rows) for key, rows in sorted(values.items())},
        "eq14_invalid_count": eq14_invalid_count,
        "eq14_evaluated_count": eq14_evaluated_count,
        "eq14_invalid_rate": invalid_rate,
        "eq14_invalid_rate_exceeds_contract": (
            invalid_rate_exceeds_contract
        ),
        "listening_unique_samples": len(listening_uids),
        "listening_rows": len(listening_rows),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    per_sample_rows.sort(
        key=lambda row: (
            row["evaluation_input_view"],
            row["sample_uid"],
            row["output_name"],
        )
    )
    write_jsonl(
        output_dir / "validation_per_sample.jsonl",
        per_sample_rows,
    )
    listening_dir.mkdir(parents=True, exist_ok=True)
    with (listening_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in listening_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if invalid_rate_exceeds_contract:
        raise RuntimeError(
            "deployment/identity Eq.14 invalid rate "
            f"{invalid_rate:.6f} exceeds contract"
        )
    model.train()
    return summary


def write_pair_receipt(pair_dir: Path, mode: str, receipt: dict) -> None:
    receipt_dir = pair_dir / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{mode}.json"
    receipt_tmp = receipt_dir / f".{mode}.{os.getpid()}.tmp"
    receipt_tmp.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(receipt_tmp, receipt_path)
    paths = {name: receipt_dir / f"{name}.json" for name in ("standard", "dnf")}
    if all(path.exists() for path in paths.values()):
        receipts = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        verification = validate_pair_receipts(receipts)
        verification_path = pair_dir / "pair_verification.json"
        verification_tmp = pair_dir / f".pair_verification.{os.getpid()}.tmp"
        verification_tmp.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(verification_tmp, verification_path)


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    validate_runtime_contract(args, contract)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cfg = load_model_cfg(args.config, args)
    configured_learning_rate = float(cfg["training_cfg"]["learning_rate"])
    if abs(
        configured_learning_rate - float(contract["training"]["learning_rate"])
    ) > 1.0e-15:
        raise ValueError("learning rate differs from the controlled contract")
    scale_weight = active_log_rms_weight(contract, args.loss_variant)
    device = torch.device(args.device)
    output_dir = args.output_root / args.run_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    train_loader = build_loader(args.train_manifest, "train", args, cfg, False)
    valid_loader = build_loader(args.valid_manifest, "valid", args, cfg, True)
    validate_manifest_speech_sources(
        train_loader.dataset,
        contract,
        seed=args.seed,
    )
    validate_manifest_speech_sources(
        valid_loader.dataset,
        contract,
        seed=args.seed,
    )
    train_noise_pairing_policy = validate_manifest_noise_pairing(
        train_loader.dataset
    )
    valid_noise_pairing_policy = validate_manifest_noise_pairing(
        valid_loader.dataset
    )
    if train_noise_pairing_policy != valid_noise_pairing_policy:
        raise ValueError(
            "train and validation noise-pairing policies differ: "
            f"{train_noise_pairing_policy!r} != {valid_noise_pairing_policy!r}"
        )
    train_speech_partition_policy = validate_manifest_speech_partition(
        train_loader.dataset
    )
    valid_speech_partition_policy = validate_manifest_speech_partition(
        valid_loader.dataset
    )
    if train_speech_partition_policy != valid_speech_partition_policy:
        raise ValueError(
            "train and validation speech-partition policies differ: "
            f"{train_speech_partition_policy!r} != "
            f"{valid_speech_partition_policy!r}"
        )
    paper_mechanism_gate = (
        train_noise_pairing_policy == NOISE_PAIRING_SAME_FAMILY_IID
        and train_speech_partition_policy == SPEECH_PARTITION_DISJOINT
    )
    if train_noise_pairing_policy != contract["data"]["noise_pairing_policy"]:
        raise ValueError(
            "manifest noise-pairing policy differs from the Phase A contract"
        )
    if (
        train_speech_partition_policy
        != contract["data"]["speech_partition_policy"]
    ):
        raise ValueError(
            "manifest speech-partition policy differs from the Phase A contract"
        )
    if not paper_mechanism_gate:
        raise ValueError(
            "the primary Phase A runner accepts only the paper-mechanism "
            "noise and speech partition policies"
        )
    model = (SEMambapp(cfg) if args.mode == "standard" else DNFSEMambapp(cfg)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training_cfg"]["learning_rate"]),
        betas=(
            float(cfg["training_cfg"]["adam_b1"]),
            float(cfg["training_cfg"]["adam_b2"]),
        ),
    )
    canonical_hash = canonical_speech_state_sha256(model)
    code_surface_hashes = phase_a_code_surface_sha256()
    metadata = {
        "phase": "A",
        "mode": args.mode,
        "scratch_init": True,
        "init_checkpoint": None,
        "resume": None,
        "gan": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": configured_learning_rate,
        "cut_duration_seconds": args.cut_duration,
        "validation_samples": args.validation_samples,
        "checkpoint_steps": sorted(args.checkpoint_steps),
        "geometry_eps": args.geometry_eps,
        "loss_variant": args.loss_variant,
        "active_log_rms_weight": scale_weight,
        "deployment_validation_input": contract["data"][
            "deployment_validation_input"
        ],
        "evaluation_input_views": [
            "single_noise_s_plus_n1",
            "identity_clean_s",
        ],
        "train_manifest": str(args.train_manifest.resolve()),
        "valid_manifest": str(args.valid_manifest.resolve()),
        "train_manifest_sha256": train_loader.dataset.manifest_sha256,
        "valid_manifest_sha256": valid_loader.dataset.manifest_sha256,
        "train_manifest_length": len(train_loader.dataset),
        "valid_manifest_length": len(valid_loader.dataset),
        "noise_pairing_policy": train_noise_pairing_policy,
        "speech_partition_policy": train_speech_partition_policy,
        "paper_mechanism_gate": paper_mechanism_gate,
        "contract_sha256": sha256_file(args.contract),
        "model_config_sha256": sha256_file(args.config),
        "training_script_sha256": sha256_file(__file__),
        "code_surface_sha256": code_surface_hashes,
        "canonical_speech_init_sha256": canonical_hash,
        "pair_contract_dir": str(args.pair_contract_dir.resolve()),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_step = run_standard_step if args.mode == "standard" else run_dnf_step
    # Model construction consumes a different number of random draws in the
    # one-head and two-head arms.  Reset the runtime RNGs after initialization
    # so shared stochastic layers start from the same stream.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint_steps = set(args.checkpoint_steps)
    route_counter: Counter[str] = Counter()
    accumulation_route_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    uid_digest = hashlib.sha256()
    uid_count = 0
    optimizer.zero_grad(set_to_none=True)
    microbatch = 0
    accumulation_metrics = []
    step = 0
    start = time.time()
    while step < args.max_steps:
        batches = 0
        for batch in train_loader:
            batches += 1
            update_uid_digest(uid_digest, batch["sample_uid"])
            uid_count += len(batch["sample_uid"])
            route_counter.update(batch["route"])
            accumulation_route_counter.update(batch["route"])
            validate_seen_route_counts(route_counter)
            source_counter.update(
                str(info["speech"].get("dataset", info["speech"].get("source", "")))
                for info in batch["info"]
            )
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss, metrics = run_step(
                model,
                batch,
                cfg,
                contract,
                args.geometry_eps,
                scale_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {metrics}")
            (loss / args.gradient_accumulation_steps).backward()
            counters.update(
                {
                    key: int(metrics[key])
                    for key in (
                        "sample_count",
                        "clean_count",
                        "noisy_count",
                        "standard_invalid_count",
                        "eq13_invalid_count",
                        "eq14_invalid_count",
                        "eq15_invalid_count",
                        "rms_invalid_count",
                    )
                }
            )
            accumulation_metrics.append(metrics)
            microbatch += 1
            if counters["sample_count"] >= int(
                contract["stop_gates"]["invalid_rate_warmup_samples"]
            ):
                for key, denominator_key, threshold_key in (
                    ("eq13_invalid_count", "noisy_count", "max_eq13_invalid_rate"),
                    ("eq14_invalid_count", "sample_count", "max_eq14_invalid_rate"),
                    ("eq15_invalid_count", "clean_count", "max_eq15_invalid_rate"),
                    (
                        "rms_invalid_count",
                        "sample_count",
                        "max_active_log_rms_invalid_rate",
                    ),
                ):
                    rate = counters[key] / max(counters[denominator_key], 1)
                    if rate > float(contract["stop_gates"][threshold_key]):
                        raise RuntimeError(f"{key} rate {rate:.6f} exceeds contract")
            if microbatch % args.gradient_accumulation_steps:
                continue
            expected_accumulation_routes = {
                ROUTE_NOISY: 15,
                ROUTE_CLEAN_REGULAR: 4,
                ROUTE_CLEAN_WEAK: 1,
            }
            if dict(accumulation_route_counter) != expected_accumulation_routes:
                raise RuntimeError(
                    "optimizer update did not consume one exact 20-row route block: "
                    f"{dict(accumulation_route_counter)}"
                )
            accumulation_route_counter.clear()
            step_metrics = aggregate_microbatch_metrics(accumulation_metrics)
            accumulation_metrics = []
            gradients = finite_gradient_summary(model)
            if gradients["total"] == 0 or gradients["finite"] != gradients["total"]:
                raise RuntimeError(f"invalid gradients: {gradients}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % args.log_interval == 0:
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step,
                            **step_metrics,
                            "routes": dict(sorted(route_counter.items())),
                            "sources": dict(sorted(source_counter.items())),
                            "uid_count": uid_count,
                            "elapsed_seconds": time.time() - start,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step in checkpoint_steps or step == args.max_steps:
                path = checkpoint_dir / f"step_{step:08d}.pt"
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    args,
                    cfg,
                    step,
                    {
                        **dict(counters),
                        "routes": dict(route_counter),
                        "sources": dict(source_counter),
                        "uid_sequence_sha256": uid_digest.hexdigest(),
                        "uid_sequence_count": uid_count,
                    },
                    metadata,
                )
            if step >= args.max_steps:
                break
        if batches == 0:
            raise RuntimeError("training manifest emitted no batches")

    evaluation = evaluate_end(
        model,
        args.mode,
        valid_loader,
        cfg,
        device,
        output_dir,
        args.geometry_eps,
        args.listening_samples,
        contract,
    )
    summary = {
        **metadata,
        "step": step,
        "uid_sequence_sha256": uid_digest.hexdigest(),
        "uid_sequence_count": uid_count,
        "routes": dict(sorted(route_counter.items())),
        "sources": dict(sorted(source_counter.items())),
        "counters": dict(counters),
        "end_eval": evaluation,
        "final_checkpoint": str(
            (checkpoint_dir / f"step_{step:08d}.pt").resolve()
        ),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "mode": args.mode,
        "train_manifest_sha256": metadata["train_manifest_sha256"],
        "valid_manifest_sha256": metadata["valid_manifest_sha256"],
        "train_manifest_length": metadata["train_manifest_length"],
        "valid_manifest_length": metadata["valid_manifest_length"],
        "canonical_speech_init_sha256": canonical_hash,
        "uid_sequence_sha256": uid_digest.hexdigest(),
        "uid_sequence_count": uid_count,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": configured_learning_rate,
        "cut_duration_seconds": args.cut_duration,
        "validation_samples": args.validation_samples,
        "checkpoint_steps": sorted(args.checkpoint_steps),
        "geometry_eps": args.geometry_eps,
        "loss_variant": args.loss_variant,
        "active_log_rms_weight": scale_weight,
        "contract_sha256": metadata["contract_sha256"],
        "model_config_sha256": metadata["model_config_sha256"],
        "training_script_sha256": metadata["training_script_sha256"],
        "code_surface_sha256": code_surface_hashes,
        "noise_pairing_policy": train_noise_pairing_policy,
        "speech_partition_policy": train_speech_partition_policy,
        "deployment_validation_input": metadata[
            "deployment_validation_input"
        ],
        "evaluation_input_views": metadata["evaluation_input_views"],
        "paper_mechanism_gate": paper_mechanism_gate,
        "output_dir": str(output_dir.resolve()),
    }
    write_pair_receipt(args.pair_contract_dir, args.mode, receipt)
    print(json.dumps({"event": "done", **receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
