"""Scratch NyTT versus paper-exact noisy-target DNF training.

This entry point is intentionally narrow: both arms consume the same routed
``x, s_noisy, n2`` stream.  ``nytt`` trains one SeMamba++ output with SI-SDR;
``dnf_exact`` trains the shared two-output model with paper equation (13) and
uses equation (14) for output diagnostics.  There is no checkpoint loading,
resume path, GAN, identity batch, L1, magnitude MSE, or gain regularizer.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dataloaders.dnf_controlled_webdataset import (
    ControlledDNFAdditiveStreamDataset,
    controlled_dnf_collate,
    gap_worker_init_fn,
)

from model.dnf_paper import (
    dnf_noisy_loss_eq13,
    dnf_output_eq14,
    sdr_loss_eq5,
    si_sdr_loss,
)
from model.dnf_semambapp import DNFSEMambapp
from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scratch NyTT versus exact equation (13)/(14) DNF."
    )
    parser.add_argument("--mode", choices=("nytt", "dnf_exact"), required=True)
    parser.add_argument(
        "--config",
        default="configs/train/semambapp_shifted_anechoic_online_v1.yaml",
    )
    parser.add_argument(
        "--split-root",
        default=(
            "/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/"
            "gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="runs/semambapp_dnf_paper_noisy",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--listening-samples", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--checkpoint-steps",
        nargs="*",
        type=int,
        default=(250, 500, 750, 1000),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cut-duration", type=float, default=1.0)
    parser.add_argument("--clean-shuffle-buffer", type=int, default=128)
    parser.add_argument("--noise-buffer-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--tiny-hid-feature", type=int, default=16)
    parser.add_argument("--tiny-num-tfmamba", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-eps", type=float, default=1e-8)
    parser.add_argument("--invalid-rate-warmup-samples", type=int, default=256)
    parser.add_argument("--max-eq13-invalid-rate", type=float, default=0.001)
    parser.add_argument("--max-eq14-invalid-rate", type=float, default=0.001)
    return parser.parse_args()


def load_cfg(path, args):
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    cfg["env_setting"]["seed"] = int(args.seed)
    cfg["training_cfg"]["batch_size"] = int(args.batch_size)
    cfg["training_cfg"]["segment_size"] = int(
        round(float(args.cut_duration) * int(cfg["stft_cfg"]["sampling_rate"]))
    )
    if args.learning_rate is not None:
        cfg["training_cfg"]["learning_rate"] = float(args.learning_rate)
    if args.tiny_model:
        cfg["model_cfg"]["hid_feature"] = int(args.tiny_hid_feature)
        cfg["model_cfg"]["num_tfmamba"] = int(args.tiny_num_tfmamba)
    return cfg


def build_loader(args, cfg, split, samples_per_epoch, expose_clean_for_eval):
    dataset = ControlledDNFAdditiveStreamDataset(
        split_root=args.split_root,
        split=split,
        target_sample_rate=int(cfg["stft_cfg"]["sampling_rate"]),
        cut_duration=float(args.cut_duration),
        samples_per_epoch=int(samples_per_epoch),
        clean_shuffle_buffer=int(args.clean_shuffle_buffer),
        noise_buffer_size=int(args.noise_buffer_size),
        shard_shuffle_seed=int(args.seed),
        expose_clean_for_eval=bool(expose_clean_for_eval),
    )
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=not expose_clean_for_eval,
        collate_fn=controlled_dnf_collate,
        worker_init_fn=gap_worker_init_fn if int(args.num_workers) > 0 else None,
    )


def stft_features(waveform, cfg):
    return mag_phase_stft(
        waveform,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
        addeps=True,
    )


def istft_waveform(magnitude, phase, cfg):
    return mag_phase_istft(
        magnitude,
        phase,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
    )


def crop_waveforms(*waveforms):
    length = min(waveform.size(-1) for waveform in waveforms)
    return tuple(waveform[..., :length] for waveform in waveforms)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model, canonical_speech_only=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        canonical_name = name
        if canonical_speech_only:
            if name.startswith(("noise_mag_decoder.", "noise_phase_decoder.")):
                continue
            canonical_name = canonical_name.replace("speech_mag_decoder.", "mask_decoder.")
            canonical_name = canonical_name.replace("speech_phase_decoder.", "phase_decoder.")
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finite_grad_summary(model):
    total = 0
    finite = 0
    max_abs = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        total += gradient.numel()
        finite += int(torch.isfinite(gradient).sum().item())
        if gradient.numel():
            max_abs = max(max_abs, float(gradient.abs().max().item()))
    return {"finite": finite, "total": total, "max_abs": max_abs}


def rms_gain_db(estimate, reference, eps):
    estimate_rms = estimate.float().square().mean(dim=-1).sqrt()
    reference_rms = reference.float().square().mean(dim=-1).sqrt()
    return 20.0 * torch.log10((estimate_rms + eps) / (reference_rms + eps))


def summarize_vector(prefix, values):
    values = values.detach().float()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_p05": float(torch.quantile(values, 0.05).item()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
    }


def normalized_inner_product(left, right, eps):
    numerator = (left.float() * right.float()).sum(dim=-1).abs()
    denominator = (
        left.float().square().sum(dim=-1)
        * right.float().square().sum(dim=-1)
    ).sqrt()
    return numerator / denominator.clamp_min(eps)


def eq13_valid_mask(speech_estimate, noise_estimate, noisy_target, artificial_noise, eps):
    artificial_energy = artificial_noise.float().square().sum(dim=-1)
    target_energy = noisy_target.float().square().sum(dim=-1)
    speech_inner = (artificial_noise.float() * speech_estimate.float()).sum(dim=-1)
    noise_inner = (artificial_noise.float() * noise_estimate.float()).sum(dim=-1)
    return (
        (artificial_energy >= eps)
        & (target_energy >= eps)
        & (speech_inner.abs() >= eps)
        & (noise_inner.abs() >= eps)
    )


def run_nytt(model, batch, cfg, eps):
    input_mag, input_phase, _ = stft_features(batch["degraded_wav"], cfg)
    magnitude, phase, _ = model(input_mag, input_phase)
    estimate = istft_waveform(magnitude, phase, cfg)
    estimate, noisy_target, model_input = crop_waveforms(
        estimate,
        batch["s_noisy_wav"],
        batch["degraded_wav"],
    )
    loss_result = si_sdr_loss(estimate, noisy_target, eps=eps)
    invalid_count = int((~loss_result.valid_mask).sum().item())
    if invalid_count:
        raise RuntimeError(f"NyTT has {invalid_count} invalid SI-SDR targets")
    loss = loss_result.value.mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "nytt_si_sdr_loss": float(loss.detach().item()),
        "sample_count": int(estimate.size(0)),
        "eq13_invalid_count": 0,
        "eq14_invalid_count": 0,
    }
    metrics.update(summarize_vector("gain_db_to_input", rms_gain_db(estimate, model_input, eps)))
    metrics.update(summarize_vector("gain_db_to_target", rms_gain_db(estimate, noisy_target, eps)))
    return loss, metrics


def run_dnf_exact(model, batch, cfg, eps):
    input_mag, input_phase, _ = stft_features(batch["degraded_wav"], cfg)
    outputs = model(input_mag, input_phase)
    speech_estimate = istft_waveform(outputs["speech_mag"], outputs["speech_pha"], cfg)
    noise_estimate = istft_waveform(outputs["noise_mag"], outputs["noise_pha"], cfg)
    (
        speech_estimate,
        noise_estimate,
        noisy_target,
        artificial_noise,
        model_input,
    ) = crop_waveforms(
        speech_estimate,
        noise_estimate,
        batch["s_noisy_wav"],
        batch["added_noise_wav"],
        batch["degraded_wav"],
    )

    valid_mask = eq13_valid_mask(
        speech_estimate,
        noise_estimate,
        noisy_target,
        artificial_noise,
        eps,
    )
    invalid_count = int((~valid_mask).sum().item())
    if not valid_mask.any():
        raise RuntimeError("Every sample in this microbatch is invalid for exact equation (13)")

    loss_result = dnf_noisy_loss_eq13(
        speech_estimate[valid_mask],
        noise_estimate[valid_mask],
        noisy_target[valid_mask],
        artificial_noise[valid_mask],
        eps=eps,
        scale_clamp=None,
    )
    if not loss_result.valid_mask.all() or not loss_result.faithful_mask.all():
        raise RuntimeError("Pre-filtered equation (13) samples were not faithful")
    loss = loss_result.total.mean()

    projection = dnf_output_eq14(speech_estimate, noise_estimate, eps=eps)
    final_estimate = projection.enhanced
    projection_invalid_count = int((~projection.valid_mask).sum().item())
    speech_normalized_denom = normalized_inner_product(
        artificial_noise, speech_estimate, eps
    )
    noise_normalized_denom = normalized_inner_product(
        artificial_noise, noise_estimate, eps
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "eq13_speech_loss": float(loss_result.noisy_speech.mean().detach().item()),
        "eq13_noise_loss": float(loss_result.noise.mean().detach().item()),
        "sample_count": int(speech_estimate.size(0)),
        "eq13_invalid_count": invalid_count,
        "eq14_invalid_count": projection_invalid_count,
        "eq13_speech_normalized_denom_min": float(speech_normalized_denom.min().detach().item()),
        "eq13_noise_normalized_denom_min": float(noise_normalized_denom.min().detach().item()),
        "eq14_projection_coefficient_abs_mean": float(
            projection.projection_coefficient.abs().mean().detach().item()
        ),
        "eq14_orthogonality_abs_mean": float(
            projection.enhanced_noise_inner_product.abs().mean().detach().item()
        ),
    }
    metrics.update(
        summarize_vector("gain_db_to_input", rms_gain_db(final_estimate, model_input, eps))
    )
    metrics.update(
        summarize_vector("gain_db_to_target", rms_gain_db(final_estimate, noisy_target, eps))
    )
    raw_subtraction = speech_estimate - noise_estimate
    metrics.update(
        summarize_vector(
            "raw_subtract_gain_db_to_input",
            rms_gain_db(raw_subtraction, model_input, eps),
        )
    )
    return loss, metrics


def update_data_counters(batch, route_counter, source_counter):
    for info in batch["info"]:
        route_counter.update((str(info["route_category"]),))
        source_counter.update(
            str(source) for source in info.get("speech_sources", ()) if source
        )
        noise_source = str(info.get("noise_source", ""))
        if noise_source:
            source_counter.update((noise_source,))


def average_metric_window(metric_window):
    additive_keys = {"sample_count", "eq13_invalid_count", "eq14_invalid_count"}
    keys = set().union(*(row.keys() for row in metric_window))
    averaged = {}
    for key in sorted(keys):
        values = [row[key] for row in metric_window if key in row]
        if key in additive_keys:
            averaged[key] = int(sum(values))
        else:
            averaged[key] = float(sum(values) / len(values))
    return averaged


def cuda_memory_summary(device):
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    gibibyte = 1024**3
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated(device) / gibibyte,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / gibibyte,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / gibibyte,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / gibibyte,
    }


def infer_outputs(model, mode, model_input, cfg, eps):
    input_magnitude, input_phase, _ = stft_features(model_input, cfg)
    if mode == "nytt":
        magnitude, phase, _ = model(input_magnitude, input_phase)
        enhanced = istft_waveform(magnitude, phase, cfg)
        return {"nytt": enhanced}, 0

    branch_outputs = model(input_magnitude, input_phase)
    speech_estimate = istft_waveform(
        branch_outputs["speech_mag"], branch_outputs["speech_pha"], cfg
    )
    noise_estimate = istft_waveform(
        branch_outputs["noise_mag"], branch_outputs["noise_pha"], cfg
    )
    speech_estimate, noise_estimate = crop_waveforms(speech_estimate, noise_estimate)
    projection = dnf_output_eq14(speech_estimate, noise_estimate, eps=eps)
    return {
        "eq14": projection.enhanced,
        "speech_head": speech_estimate,
        "raw_subtract": speech_estimate - noise_estimate,
    }, int((~projection.valid_mask).sum().item())


def safe_audio_id(value):
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))
    return cleaned[:160] or "sample"


def summarize_eval_values(values):
    tensor = torch.cat(values).float()
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p05": float(torch.quantile(tensor, 0.05).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def save_listening_waveform(path, waveform, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path),
        waveform.detach().float().cpu().unsqueeze(0),
        int(sample_rate),
        encoding="PCM_F",
        bits_per_sample=32,
    )


def evaluate_end(model, mode, loader, cfg, device, output_dir, eps, listening_samples):
    model.eval()
    metric_values = {}
    projection_invalid = 0
    sample_count = 0
    listening_count = 0
    listening_dir = output_dir / "listening"
    listening_manifest = []

    with torch.inference_mode():
        loader.dataset.set_epoch(0)
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if "eval_clean_wav" not in batch:
                raise RuntimeError("Validation loader did not expose eval_clean_wav")
            model_input = batch["degraded_wav"]
            clean = batch["eval_clean_wav"]
            outputs, invalid_count = infer_outputs(model, mode, model_input, cfg, eps)
            projection_invalid += invalid_count
            length = min(
                model_input.size(-1),
                clean.size(-1),
                *(waveform.size(-1) for waveform in outputs.values()),
            )
            model_input = model_input[..., :length]
            clean = clean[..., :length]
            outputs = {name: waveform[..., :length] for name, waveform in outputs.items()}

            input_si_loss = si_sdr_loss(model_input, clean, eps=eps)
            input_sdr_loss = sdr_loss_eq5(model_input, clean, eps=eps)
            if not input_si_loss.valid_mask.all() or not torch.isfinite(input_sdr_loss).all():
                raise RuntimeError("Validation input/clean reference is invalid")
            metric_values.setdefault("input/si_sdr_db", []).append(
                -input_si_loss.value.detach().cpu()
            )
            metric_values.setdefault("input/sdr_db", []).append(
                -input_sdr_loss.detach().cpu()
            )
            metric_values.setdefault("input/gain_db_to_clean", []).append(
                rms_gain_db(model_input, clean, eps).detach().cpu()
            )

            for output_name, waveform in outputs.items():
                output_si_loss = si_sdr_loss(waveform, clean, eps=eps)
                output_sdr_loss = sdr_loss_eq5(waveform, clean, eps=eps)
                if not output_si_loss.valid_mask.all() or not torch.isfinite(output_sdr_loss).all():
                    raise RuntimeError(f"Validation output {output_name} is invalid")
                prefix = f"outputs/{output_name}"
                metric_values.setdefault(f"{prefix}/si_sdr_db", []).append(
                    -output_si_loss.value.detach().cpu()
                )
                metric_values.setdefault(f"{prefix}/si_sdri_db", []).append(
                    (input_si_loss.value - output_si_loss.value).detach().cpu()
                )
                metric_values.setdefault(f"{prefix}/sdr_db", []).append(
                    -output_sdr_loss.detach().cpu()
                )
                metric_values.setdefault(f"{prefix}/sdri_db", []).append(
                    (input_sdr_loss - output_sdr_loss).detach().cpu()
                )
                metric_values.setdefault(f"{prefix}/gain_db_to_clean", []).append(
                    rms_gain_db(waveform, clean, eps).detach().cpu()
                )
                metric_values.setdefault(f"{prefix}/gain_db_to_input", []).append(
                    rms_gain_db(waveform, model_input, eps).detach().cpu()
                )

            batch_size = int(clean.size(0))
            sample_count += batch_size
            for index in range(batch_size):
                if listening_count >= int(listening_samples):
                    break
                utterance_id = safe_audio_id(batch["utterance_id"][index])
                sample_rate = int(batch["sample_rate"][index].item())
                save_listening_waveform(
                    listening_dir / "input" / f"{utterance_id}.wav",
                    model_input[index],
                    sample_rate,
                )
                save_listening_waveform(
                    listening_dir / "clean" / f"{utterance_id}.wav",
                    clean[index],
                    sample_rate,
                )
                for output_name, waveform in outputs.items():
                    save_listening_waveform(
                        listening_dir / output_name / f"{utterance_id}.wav",
                        waveform[index],
                        sample_rate,
                    )
                listening_manifest.append(
                    {
                        "utterance_id": batch["utterance_id"][index],
                        "sample_rate": sample_rate,
                        "info": batch["info"][index],
                        "outputs": sorted(outputs),
                    }
                )
                listening_count += 1

    summary = {
        "mode": mode,
        "sample_count": sample_count,
        "eq14_invalid_count": projection_invalid,
        "eq14_invalid_rate": projection_invalid / max(sample_count, 1),
        "metrics": {
            key: summarize_eval_values(values)
            for key, values in sorted(metric_values.items())
        },
        "listening_samples": listening_count,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (listening_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in listening_manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    model.train()
    return summary


def save_checkpoint(path, model, optimizer, args, cfg, step, counters, metadata):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "cfg": cfg,
        "step": int(step),
        "counters": counters,
        "metadata": metadata,
        "scratch_init": True,
        "init_checkpoint": None,
        "resume": None,
        "created_at": datetime.now().isoformat(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    if args.geometry_eps <= 0:
        raise ValueError("--geometry-eps must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    cfg = load_cfg(args.config, args)
    run_name = args.run_name or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}__{args.mode}__scratch"
    )
    output_dir = Path(args.output_root) / run_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    loader = build_loader(
        args,
        cfg,
        split="train",
        samples_per_epoch=int(args.samples_per_epoch),
        expose_clean_for_eval=False,
    )
    validation_loader = build_loader(
        args,
        cfg,
        split="valid",
        samples_per_epoch=int(args.validation_samples),
        expose_clean_for_eval=True,
    )
    model = (
        SEMambapp(cfg)
        if args.mode == "nytt"
        else DNFSEMambapp(cfg)
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training_cfg"]["learning_rate"]),
        betas=(
            float(cfg["training_cfg"]["adam_b1"]),
            float(cfg["training_cfg"]["adam_b2"]),
        ),
    )
    run_step = run_nytt if args.mode == "nytt" else run_dnf_exact

    script_path = Path(__file__).resolve()
    metadata = {
        "run_name": run_name,
        "mode": args.mode,
        "scratch_init": True,
        "init_checkpoint": None,
        "resume": None,
        "output_dir": str(output_dir.resolve()),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "split_root": args.split_root,
        "train_data": getattr(loader.dataset, "route_summary", {}),
        "validation_data": getattr(validation_loader.dataset, "route_summary", {}),
        "data_contract": "controlled EARS/VCTK + WHAM pure additive s+n1+n2",
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size)
        * int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_initial_state_sha256": state_sha256(model),
        "canonical_speech_path_initial_sha256": state_sha256(
            model, canonical_speech_only=True
        ),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "args": vars(args),
        "model_cfg": cfg["model_cfg"],
        "training_cfg": cfg["training_cfg"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"event": "metadata", **metadata}, sort_keys=True), flush=True)

    grad_accumulation = max(1, int(args.gradient_accumulation_steps))
    checkpoint_steps = {int(value) for value in args.checkpoint_steps if int(value) > 0}
    route_counter = Counter()
    source_counter = Counter()
    metric_window = []
    step = 0
    microbatch_index = 0
    epoch = 0
    total_samples = 0
    total_eq13_invalid = 0
    total_eq14_invalid = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < int(args.max_steps):
        loader.dataset.set_epoch(epoch)
        batches_seen_this_epoch = 0
        for batch in loader:
            batches_seen_this_epoch += 1
            update_data_counters(batch, route_counter, source_counter)
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if device.type == "cuda" and microbatch_index % grad_accumulation == 0:
                torch.cuda.reset_peak_memory_stats(device)

            loss, metrics = run_step(model, batch, cfg, float(args.geometry_eps))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at optimizer_step={step}: {metrics}")
            (loss / grad_accumulation).backward()
            metric_window.append(metrics)
            microbatch_index += 1
            total_samples += int(metrics["sample_count"])
            total_eq13_invalid += int(metrics["eq13_invalid_count"])
            total_eq14_invalid += int(metrics["eq14_invalid_count"])

            if total_samples >= int(args.invalid_rate_warmup_samples):
                eq13_invalid_rate = total_eq13_invalid / total_samples
                eq14_invalid_rate = total_eq14_invalid / total_samples
                if eq13_invalid_rate > float(args.max_eq13_invalid_rate):
                    raise RuntimeError(
                        f"Eq.13 invalid rate {eq13_invalid_rate:.6f} exceeds "
                        f"{args.max_eq13_invalid_rate}"
                    )
                if eq14_invalid_rate > float(args.max_eq14_invalid_rate):
                    raise RuntimeError(
                        f"Eq.14 invalid rate {eq14_invalid_rate:.6f} exceeds "
                        f"{args.max_eq14_invalid_rate}"
                    )

            if microbatch_index % grad_accumulation:
                continue

            gradient = finite_grad_summary(model)
            if gradient["total"] == 0 or gradient["finite"] != gradient["total"]:
                raise RuntimeError(f"Non-finite gradient at step={step + 1}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % int(args.log_interval) == 0:
                averaged = average_metric_window(metric_window)
                elapsed = max(time.time() - start_time, 1e-6)
                log_row = {
                    "event": "train",
                    "step": step,
                    **averaged,
                    "eq13_invalid_rate_cumulative": total_eq13_invalid
                    / max(total_samples, 1),
                    "eq14_invalid_rate_cumulative": total_eq14_invalid
                    / max(total_samples, 1),
                    "route_count_cumulative": dict(sorted(route_counter.items())),
                    "source_count_cumulative": dict(sorted(source_counter.items())),
                    "samples_per_second": total_samples / elapsed,
                    "grad_finite": gradient["finite"],
                    "grad_total": gradient["total"],
                    "grad_max_abs": gradient["max_abs"],
                    **cuda_memory_summary(device),
                }
                print(json.dumps(log_row, sort_keys=True), flush=True)
                metric_window = []

            if step in checkpoint_steps or step == int(args.max_steps):
                counters = {
                    "total_samples": total_samples,
                    "eq13_invalid": total_eq13_invalid,
                    "eq14_invalid": total_eq14_invalid,
                    "routes": dict(sorted(route_counter.items())),
                    "sources": dict(sorted(source_counter.items())),
                }
                checkpoint_path = checkpoint_dir / f"step_{step:08d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args,
                    cfg,
                    step,
                    counters,
                    metadata,
                )
                print(
                    json.dumps(
                        {"event": "checkpoint", "step": step, "path": str(checkpoint_path)},
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if step >= int(args.max_steps):
                break

        if batches_seen_this_epoch == 0:
            raise RuntimeError(f"DataLoader emitted no batches at epoch={epoch}")
        epoch += 1

    final_checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
    if not final_checkpoint.exists():
        counters = {
            "total_samples": total_samples,
            "eq13_invalid": total_eq13_invalid,
            "eq14_invalid": total_eq14_invalid,
            "routes": dict(sorted(route_counter.items())),
            "sources": dict(sorted(source_counter.items())),
        }
        save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            args,
            cfg,
            step,
            counters,
            metadata,
        )
    evaluation = evaluate_end(
        model,
        args.mode,
        validation_loader,
        cfg,
        device,
        output_dir,
        float(args.geometry_eps),
        int(args.listening_samples),
    )
    print(json.dumps({"event": "end_eval", **evaluation}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "event": "done",
                "step": step,
                "output_dir": str(output_dir.resolve()),
                "final_checkpoint": str(final_checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
