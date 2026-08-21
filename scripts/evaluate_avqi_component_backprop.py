#!/usr/bin/env python3
"""Speaker-disjoint diagnostic for two AVQI-component backpropagation routes.

This script deliberately stops before generator optimization. It compares two
small mechanisms, evaluates all six AVQI v03.01 terms, and verifies the intended
gradient paths:

1. shared SeMamba++ heads at late-backbone and enhanced-spectral points;
2. separately trained waveform predictors that are frozen before backprop;
3. direct differentiable signal-processing formulas with no neural predictor.

Architecture selection and affine calibration use only the calibration split.
The speaker-disjoint holdout and 24-speaker external panel remain evaluation-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.avqi_components import (
    AVQI_COMPONENT_LOSS_WEIGHTS,
    AVQI_COMPONENT_NAMES,
    CompactTFGridSharedComponentHead,
    CompactTFGridWaveformComponentPredictor,
    ComponentAffineCalibrator,
    DifferentiableAVQIComponentEstimator,
    FrequencyAwareSharedComponentHead,
    FrequencyAwareWaveformComponentPredictor,
    PhaseAwareCompactTFGridWaveformComponentPredictor,
    PhaseAwareFrequencyAwareWaveformComponentPredictor,
    PretrainedFullTFGridWaveformComponentPredictor,
    PraatDifferentiableAVQIComponentEstimator,
    SharedComponentHead,
    WaveformComponentPredictor,
    denormalize_components,
    freeze_module,
    freeze_module_for_input_gradient,
    phase_aware_spectral_features,
    pool_frequency_aware_shared_feature_map,
    pool_shared_feature_map,
    standardized_component_loss,
)
from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft
from utils import load_config


SAMPLE_RATE = 16_000
DEFAULT_EXPECTED_SPLIT_SPEAKERS = {
    "surrogate_train": 70,
    "surrogate_calibration": 14,
    "surrogate_holdout": 14,
}
MIN_LABEL_BANK_COVERAGE = 0.95
MIN_LABEL_BANK_SLICE_COVERAGE = 0.90
PRIMARY_GATE_COMPONENTS = ("cpps", "hnr")
COMPONENT_FAMILIES = {
    "cpps": "periodicity_noise",
    "hnr": "periodicity_noise",
    "shimmer_percent": "amplitude_modulation",
    "shimmer_db": "amplitude_modulation",
    "slope": "spectral_shape",
    "tilt": "spectral_shape",
}
LEVEL_SPEARMAN_GATE = 0.70
DELTA_SPEARMAN_GATE = 0.60
PAIRED_STABILITY_NMAE_GATE = 0.25
NORMALIZED_MAE_GATE = 0.50
CALIBRATION_SLOPE_RANGE = (0.75, 1.25)
COMPONENT_INPUT_GRADIENT_MAX = 1e4
EXTERNAL_COVERAGE_GATE = 0.99
SCREEN_BATCH_SIZE = 16
SCREEN_LEARNING_RATE = 3e-4
SCREEN_MIN_EPOCHS = 15
SCREEN_GRADIENT_CLIP_NORM = 5.0
FULL_TFGRID_BACKBONE_LEARNING_RATE = 3e-5
TRAINING_SEGMENT_SAMPLES = 48_000
SEGMENT_TRANSFER_NMAE_GATE = NORMALIZED_MAE_GATE
EXTERNAL_CANDIDATES = ("B0_250", "S3_500", "S3_2000")
EXTERNAL_PRIMARY_CANDIDATE = "S3_500"
EXTERNAL_CONDITIONS = (
    "clean",
    "rir_only",
    "snr30",
    "snr20",
    "snr15",
    "snr10",
)
FREQUENCY_BINS = 8
TFGRID_FREQUENCY_BINS = 32
TFGRID_TIME_BINS = 64
EXPECTED_EXTERNAL_SPEAKERS = 24
EXTERNAL_REQUIRED_SLICES = (
    "view=cs",
    "view=sv",
    "label=patient",
    "view=sv&sample_group=pathological_severe",
    "condition=snr10",
)


@dataclass(frozen=True)
class Example:
    speaker_id: str
    sample_id: str
    split: str
    condition: str
    view: str
    label: str
    sample_group: str
    path: Path
    audio_sha256: str
    waveform: torch.Tensor
    own_target: torch.Tensor
    clean_target: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--external-exact-csv", type=Path, required=True)
    parser.add_argument("--external-exact-csv-sha256", required=True)
    parser.add_argument("--vctk-external-label-bank", type=Path)
    parser.add_argument("--vctk-external-label-bank-sha256")
    parser.add_argument("--full-tfgrid-checkpoint", type=Path)
    parser.add_argument("--full-tfgrid-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--shared-head-epochs", type=int, default=60)
    parser.add_argument("--waveform-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="matched per-candidate cap; zero keeps the historical epoch budget",
    )
    parser.add_argument(
        "--expected-train-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_train"],
    )
    parser.add_argument(
        "--expected-calibration-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_calibration"],
    )
    parser.add_argument(
        "--expected-holdout-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_holdout"],
    )
    parser.add_argument(
        "--shared-candidates",
        default="late_global,late_frequency,late_tfgrid",
        help="comma-separated shared-head candidates selected using calibration only",
    )
    parser.add_argument(
        "--waveform-architectures",
        default="global_stats,frequency_aware,compact_tfgrid",
        help="comma-separated independent predictors selected using calibration only",
    )
    return parser.parse_args()


def comma_separated_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("candidate list must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate candidates are not allowed: {values}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def component_tensor(row: dict[str, str], prefix: str = "") -> torch.Tensor:
    values = [float(row[f"{prefix}{name}"]) for name in AVQI_COMPONENT_NAMES]
    tensor = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("non-finite exact component label")
    return tensor


def row_sample_id(row: dict[str, str]) -> str:
    """Return a stable within-speaker sample key with v1/v2 compatibility."""
    return (
        row.get("sample_id", "").strip()
        or row.get("pair_id", "").strip()
        or row["speaker_id"]
    )


def load_waveform(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"invalid 16 kHz mono audio: {path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite audio: {path}")
    return waveform


def clean_target_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["speaker_id"],
        row_sample_id(row),
        row["split"],
        row["view"],
    )


def select_usable_label_rows(
    all_task_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    exact_rows = [
        row for row in all_task_rows if row["scoring_status"] == "ok"
    ]
    clean_keys = {
        clean_target_key(row)
        for row in exact_rows
        if row["condition_id"] == "clean"
    }
    usable_rows = [
        row for row in exact_rows if clean_target_key(row) in clean_keys
    ]
    missing_clean_target_rows = [
        row for row in exact_rows if clean_target_key(row) not in clean_keys
    ]
    return exact_rows, usable_rows, missing_clean_target_rows


def load_examples(
    label_bank: Path,
    expected_split_speakers: dict[str, int],
) -> tuple[list[Example], dict[str, Any]]:
    with label_bank.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    all_task_rows = [row for row in rows if row["view"] in {"cs", "sv"}]
    if not all_task_rows:
        raise ValueError("label bank contains no CS/SV task rows")
    row_keys = [
        (
            row["speaker_id"],
            row_sample_id(row),
            row["split"],
            row["condition_id"],
            row["view"],
        )
        for row in all_task_rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(
            "duplicate speaker/sample/split/condition/view rows in label bank"
        )
    actual_splits = {row["split"] for row in all_task_rows}
    if actual_splits != set(expected_split_speakers):
        raise ValueError(
            "label-bank split mismatch: "
            f"expected {sorted(expected_split_speakers)}, found {sorted(actual_splits)}"
        )
    exact_rows, task_rows, missing_clean_target_rows = (
        select_usable_label_rows(all_task_rows)
    )
    exact_coverage_fraction = len(exact_rows) / len(all_task_rows)
    if exact_coverage_fraction < MIN_LABEL_BANK_COVERAGE:
        raise ValueError(
            "label-bank exact-score coverage below gate: "
            f"{exact_coverage_fraction:.6f} < {MIN_LABEL_BANK_COVERAGE:.2f}"
        )
    usable_coverage_fraction = len(task_rows) / len(all_task_rows)
    if usable_coverage_fraction < MIN_LABEL_BANK_COVERAGE:
        raise ValueError(
            "label-bank clean-target-compatible coverage below gate: "
            f"{usable_coverage_fraction:.6f} < {MIN_LABEL_BANK_COVERAGE:.2f}"
        )
    invalid_rows = [
        {
            "speaker_id": row.get("speaker_id", ""),
            "split": row.get("split", ""),
            "condition": row.get("condition_id", ""),
            "view": row.get("view", ""),
            "scoring_status": row.get("scoring_status", ""),
            "error_type": row.get("error_type", ""),
            "error_message": row.get("error_message", ""),
        }
        for row in all_task_rows
        if row["scoring_status"] != "ok"
    ]
    missing_clean_target_cases = [
        {
            "speaker_id": row["speaker_id"],
            "sample_id": row_sample_id(row),
            "split": row["split"],
            "condition": row["condition_id"],
            "view": row["view"],
        }
        for row in missing_clean_target_rows
    ]
    split_speakers = {
        split: len({row["speaker_id"] for row in task_rows if row["split"] == split})
        for split in expected_split_speakers
    }
    if split_speakers != expected_split_speakers:
        raise ValueError(f"speaker split mismatch: {split_speakers}")
    speaker_sets = {
        split: {row["speaker_id"] for row in task_rows if row["split"] == split}
        for split in expected_split_speakers
    }
    for first, first_speakers in speaker_sets.items():
        for second, second_speakers in speaker_sets.items():
            if first < second and first_speakers & second_speakers:
                raise ValueError(f"speaker leakage between {first} and {second}")
    split_condition_coverage: dict[str, Any] = {}
    for split, condition in sorted(
        {(row["split"], row["condition_id"]) for row in all_task_rows}
    ):
        eligible = [
            row
            for row in all_task_rows
            if row["split"] == split and row["condition_id"] == condition
        ]
        exact = [
            row
            for row in exact_rows
            if row["split"] == split and row["condition_id"] == condition
        ]
        usable = [
            row
            for row in task_rows
            if row["split"] == split and row["condition_id"] == condition
        ]
        usable_fraction = len(usable) / len(eligible)
        slice_name = f"{split}/{condition}"
        split_condition_coverage[slice_name] = {
            "eligible_rows": len(eligible),
            "exact_rows": len(exact),
            "usable_rows": len(usable),
            "exact_fraction": len(exact) / len(eligible),
            "usable_fraction": usable_fraction,
            "minimum_usable_fraction": MIN_LABEL_BANK_SLICE_COVERAGE,
            "decision": (
                "PASS"
                if usable_fraction >= MIN_LABEL_BANK_SLICE_COVERAGE
                else "FAIL"
            ),
        }
        if usable_fraction < MIN_LABEL_BANK_SLICE_COVERAGE:
            raise ValueError(
                "label-bank clean-target-compatible slice coverage below gate: "
                f"{slice_name}={usable_fraction:.6f} < "
                f"{MIN_LABEL_BANK_SLICE_COVERAGE:.2f}"
            )
    clean_targets = {
        clean_target_key(row): component_tensor(row)
        for row in exact_rows
        if row["condition_id"] == "clean"
    }
    examples: list[Example] = []
    for index, row in enumerate(task_rows, start=1):
        view = row["view"]
        path = Path(row[f"{view}_path"])
        expected_hash = row[f"{view}_sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"audio hash mismatch: {path}")
        sample_id = row_sample_id(row)
        key = clean_target_key(row)
        examples.append(
            Example(
                speaker_id=row["speaker_id"],
                sample_id=sample_id,
                split=row["split"],
                condition=row["condition_id"],
                view=view,
                label=row["label"],
                sample_group=row["sample_group"],
                path=path,
                audio_sha256=expected_hash,
                waveform=load_waveform(path),
                own_target=component_tensor(row),
                clean_target=clean_targets[key],
            )
        )
        if index % 50 == 0 or index == len(task_rows):
            print(f"loaded_examples={index}/{len(task_rows)}", flush=True)
    coverage = {
        "total_rows": len(all_task_rows),
        "exact_rows": len(exact_rows),
        "usable_rows": len(task_rows),
        "invalid_rows": len(invalid_rows),
        "missing_clean_target_rows": len(missing_clean_target_rows),
        "exact_fraction": exact_coverage_fraction,
        "fraction": usable_coverage_fraction,
        "minimum_fraction": MIN_LABEL_BANK_COVERAGE,
        "minimum_slice_fraction": MIN_LABEL_BANK_SLICE_COVERAGE,
        "split_speakers": split_speakers,
        "split_condition_coverage": split_condition_coverage,
        "invalid_cases": invalid_rows,
        "missing_clean_target_cases": missing_clean_target_cases,
    }
    return examples, coverage


def target_stats(
    examples: list[Example], target_attribute: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.stack(
        [
            getattr(example, target_attribute)
            for example in examples
            if example.split == "surrogate_train"
        ]
    )
    mean = targets.mean(dim=0).to(device)
    scale = targets.std(dim=0, unbiased=False).clamp_min(1e-6).to(device)
    return mean, scale


def set_model_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generator(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> SEMambapp:
    model = SEMambapp(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("generator", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def normalized_stft_input(
    waveform: torch.Tensor, config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    scale = 0.9 / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    normalized = waveform * scale
    stft_cfg = config["stft_cfg"]
    magnitude, phase, _ = mag_phase_stft(
        normalized,
        stft_cfg["n_fft"],
        stft_cfg["hop_size"],
        stft_cfg["win_size"],
        config["model_cfg"]["compress_factor"],
    )
    return magnitude, phase, scale


def shared_feature_maps(
    model: SEMambapp,
    waveform: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    magnitude, phase, _ = normalized_stft_input(waveform, config)
    input_phase_spectral = phase_aware_spectral_features(
        magnitude,
        phase,
        time_dim=-1,
    ).transpose(-2, -1)
    magnitude = magnitude.permute(0, 2, 1).unsqueeze(1)
    phase = phase.permute(0, 2, 1).unsqueeze(1)
    shared = model.dense_encoder(torch.cat((magnitude, phase), dim=1))
    encoder = shared
    for block in model.TSMamba:
        shared = block(shared)
    enhanced_magnitude = model.mask_decoder(shared).squeeze(1)
    enhanced_phase = model.phase_decoder(shared).squeeze(1)
    enhanced_spectral = torch.stack(
        (
            torch.log1p(enhanced_magnitude.clamp_min(0.0)),
            torch.cos(enhanced_phase),
            torch.sin(enhanced_phase),
        ),
        dim=1,
    )
    output_phase_spectral = phase_aware_spectral_features(
        enhanced_magnitude,
        enhanced_phase,
        time_dim=-2,
    )
    return {
        "encoder": encoder,
        "late": shared,
        "enhanced_spectral": enhanced_spectral,
        "input_phase_spectral": input_phase_spectral,
        "output_phase_spectral": output_phase_spectral,
    }


def input_phase_feature_maps(
    waveform: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Build the scorer-training representation without running the generator."""
    magnitude, phase, _ = normalized_stft_input(waveform, config)
    return {
        "input_phase_spectral": phase_aware_spectral_features(
            magnitude,
            phase,
            time_dim=-1,
        ).transpose(-2, -1)
    }


def pool_shared_candidate(
    feature_maps: dict[str, torch.Tensor],
    candidate: str,
    *,
    training: bool = False,
) -> torch.Tensor:
    if candidate == "late_global":
        return pool_shared_feature_map(feature_maps["late"])
    if candidate == "late_frequency":
        return pool_frequency_aware_shared_feature_map(
            feature_maps["late"],
            frequency_bins=FREQUENCY_BINS,
        )
    if candidate == "late_tfgrid":
        # Store a bounded grid for repeatable head training. The live gradient
        # path uses the same adaptive grid inside CompactTFGridSharedComponentHead.
        frequency_time = feature_maps["late"].transpose(-2, -1)
        bounded = torch.nn.functional.adaptive_avg_pool2d(
            frequency_time,
            (TFGRID_FREQUENCY_BINS, TFGRID_TIME_BINS),
        )
        return bounded.transpose(-2, -1)
    if candidate == "enhanced_spectral":
        return pool_frequency_aware_shared_feature_map(
            feature_maps["enhanced_spectral"],
            frequency_bins=FREQUENCY_BINS,
        )
    if candidate == "output_phase_tfgrid":
        key = "input_phase_spectral" if training else "output_phase_spectral"
        frequency_time = feature_maps[key].transpose(-2, -1)
        bounded = torch.nn.functional.adaptive_avg_pool2d(
            frequency_time,
            (TFGRID_FREQUENCY_BINS, TFGRID_TIME_BINS),
        )
        return bounded.transpose(-2, -1)
    raise ValueError(f"unknown shared candidate: {candidate}")


def shared_head_forward(
    head: torch.nn.Module,
    features: torch.Tensor,
    candidate: str,
) -> torch.Tensor:
    if candidate in {"late_tfgrid", "output_phase_tfgrid"}:
        return head(features)
    return head.forward_pooled(features)


def enhance_waveform(
    model: SEMambapp,
    waveform: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    magnitude, phase, scale = normalized_stft_input(waveform, config)
    enhanced_magnitude, enhanced_phase, _ = model(magnitude, phase)
    stft_cfg = config["stft_cfg"]
    enhanced = mag_phase_istft(
        enhanced_magnitude,
        enhanced_phase,
        stft_cfg["n_fft"],
        stft_cfg["hop_size"],
        stft_cfg["win_size"],
        config["model_cfg"]["compress_factor"],
    )
    return enhanced / scale


def extract_shared_features(
    model: SEMambapp,
    examples: list[Example],
    config: dict[str, Any],
    device: torch.device,
    candidates: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {candidate: [] for candidate in candidates}
    with torch.no_grad():
        for index, example in enumerate(examples, start=1):
            if candidates == ("output_phase_tfgrid",):
                maps = input_phase_feature_maps(
                    example.waveform.to(device),
                    config,
                )
            else:
                maps = shared_feature_maps(model, example.waveform.to(device), config)
            for candidate in rows:
                rows[candidate].append(
                    pool_shared_candidate(
                        maps,
                        candidate,
                        training=True,
                    ).cpu()[0]
                )
            if index % 25 == 0 or index == len(examples):
                print(f"shared_feature_rows={index}/{len(examples)}", flush=True)
    return {name: torch.stack(values) for name, values in rows.items()}


def calibration_loss(
    model: torch.nn.Module,
    predict: Callable[[torch.nn.Module, int], torch.Tensor],
    examples: list[Example],
    target_attribute: str,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> float:
    losses = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            if example.split != "surrogate_calibration":
                continue
            prediction = predict(model, index)
            target = getattr(example, target_attribute).to(prediction.device).unsqueeze(0)
            losses.append(
                standardized_component_loss(
                    prediction, target, target_mean, target_scale
                ).item()
            )
    return float(np.mean(losses))


def train_shared_head(
    pooled_features: torch.Tensor,
    examples: list[Example],
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
    candidate: str,
    max_optimizer_steps: int = 0,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    set_model_seed(seed)
    target_attribute = (
        "own_target" if candidate == "output_phase_tfgrid" else "clean_target"
    )
    target_mean, target_scale = target_stats(examples, target_attribute, device)
    features = pooled_features.to(device)
    if candidate == "late_global":
        head: torch.nn.Module = SharedComponentHead(
            feature_channels=pooled_features.shape[1] // 2
        )
    elif candidate in {"late_frequency", "enhanced_spectral"}:
        feature_channels = pooled_features.shape[1] // (FREQUENCY_BINS * 2)
        head = FrequencyAwareSharedComponentHead(
            feature_channels=feature_channels,
            frequency_bins=FREQUENCY_BINS,
        )
    elif candidate in {"late_tfgrid", "output_phase_tfgrid"}:
        head = CompactTFGridSharedComponentHead(
            feature_channels=pooled_features.shape[1]
        )
    else:
        raise ValueError(f"unknown shared candidate: {candidate}")
    head = head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=SCREEN_LEARNING_RATE,
        weight_decay=1e-4,
    )
    train_indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_train"
    ]
    train_targets = torch.stack(
        [getattr(example, target_attribute) for example in examples]
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        for start in range(0, len(order), SCREEN_BATCH_SIZE):
            stop = min(start + SCREEN_BATCH_SIZE, len(order))
            batch = [
                train_indices[order[position]] for position in range(start, stop)
            ]
            prediction = shared_head_forward(head, features[batch], candidate)
            loss = standardized_component_loss(
                prediction,
                train_targets[batch],
                target_mean,
                target_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                SCREEN_GRADIENT_CLIP_NORM,
            )
            optimizer.step()
            optimizer_steps += 1
            if max_optimizer_steps and optimizer_steps >= max_optimizer_steps:
                break
        head.eval()
        value = calibration_loss(
            head,
            lambda current, index: shared_head_forward(
                current,
                features[index : index + 1],
                candidate,
            ),
            examples,
            target_attribute,
            target_mean,
            target_scale,
        )
        history.append({"epoch": epoch, "calibration_loss": value})
        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= SCREEN_MIN_EPOCHS and stale >= patience:
            break
        if max_optimizer_steps and optimizer_steps >= max_optimizer_steps:
            break
    if best_state is None:
        raise RuntimeError("shared head did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return (
        head,
        {
            "best_epoch": best_epoch,
            "best_calibration_loss": best_loss,
            "epochs_ran": len(history),
            "optimizer_steps": optimizer_steps,
            "target_attribute": target_attribute,
            "history": history,
        },
        target_mean,
        target_scale,
    )


def train_waveform_predictor(
    examples: list[Example],
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
    architecture: str,
    cached_spectrograms: torch.Tensor | None = None,
    full_tfgrid_checkpoint: Path | None = None,
    max_optimizer_steps: int = 0,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    set_model_seed(seed)
    target_mean, target_scale = target_stats(examples, "own_target", device)
    pretrained_receipt: dict[str, Any] | None = None
    if architecture == "global_stats":
        predictor: torch.nn.Module = WaveformComponentPredictor()
    elif architecture == "frequency_aware":
        predictor = FrequencyAwareWaveformComponentPredictor(
            frequency_bins=FREQUENCY_BINS
        )
    elif architecture == "phase_frequency_aware":
        predictor = PhaseAwareFrequencyAwareWaveformComponentPredictor(
            frequency_bins=FREQUENCY_BINS
        )
    elif architecture == "compact_tfgrid":
        predictor = CompactTFGridWaveformComponentPredictor()
    elif architecture == "phase_compact_tfgrid":
        predictor = PhaseAwareCompactTFGridWaveformComponentPredictor()
    elif architecture == "pretrained_full_tfgrid":
        if full_tfgrid_checkpoint is None:
            raise ValueError(
                "pretrained_full_tfgrid requires --full-tfgrid-checkpoint"
            )
        predictor = PretrainedFullTFGridWaveformComponentPredictor()
        pretrained_receipt = predictor.load_hybrid_discriminative_checkpoint(
            full_tfgrid_checkpoint
        )
    elif architecture == "direct_exact_inspired":
        predictor = DifferentiableAVQIComponentEstimator()
    elif architecture == "direct_praat_soft_v2":
        predictor = PraatDifferentiableAVQIComponentEstimator(peak_mode="soft")
    elif architecture == "direct_praat_hard_v2":
        predictor = PraatDifferentiableAVQIComponentEstimator(peak_mode="hard")
    elif architecture == "direct_praat_hard_shimmer_rms_v3":
        predictor = PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            shimmer_mode="hann_rms_v3",
        )
    elif architecture == "direct_praat_hard_shimmer_raw_cc_surrogate_v4":
        predictor = PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            shimmer_mode="hann_rms_raw_cc_surrogate_v4",
        )
    elif architecture == "direct_praat_hard_shimmer_pulse_chain_v5":
        predictor = PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            shimmer_mode="praat_pulse_chain_v5",
        )
    elif architecture == "direct_praat_hard_shimmer_pulse_path_v6":
        predictor = PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            shimmer_mode="praat_pulse_path_v6",
        )
    else:
        raise ValueError(f"unknown waveform architecture: {architecture}")
    predictor = predictor.to(device)
    if isinstance(predictor, DifferentiableAVQIComponentEstimator):
        if cached_spectrograms is not None:
            raise ValueError(
                "direct exact-inspired estimator uses cached formula outputs, "
                "not log-spectrogram cache"
            )
        cached_inputs = cache_direct_component_features(
            predictor,
            examples,
            device,
        )
        train_indices = [
            index
            for index, example in enumerate(examples)
            if example.split == "surrogate_train"
        ]
        raw_targets = torch.stack(
            [examples[index].own_target for index in train_indices]
        ).to(device)
        alignment = predictor.fit_alignment(
            cached_inputs[train_indices].to(device),
            raw_targets,
            target_mean,
            target_scale,
        )

        def direct_predict(current: torch.nn.Module, index: int) -> torch.Tensor:
            return current.forward_proxy_features(
                cached_inputs[index : index + 1].to(device)
            )

        value = calibration_loss(
            predictor,
            direct_predict,
            examples,
            "own_target",
            target_mean,
            target_scale,
        )
        return (
            predictor,
            {
                "best_epoch": 0,
                "best_calibration_loss": value,
                "epochs_ran": 0,
                "optimizer_steps": 0,
                "history": [],
                "pretrained_backbone": None,
                "direct_alignment": alignment,
                "trainable_parameter_count": 0,
            },
            target_mean,
            target_scale,
            cached_inputs,
        )
    cached_inputs = cached_spectrograms
    if isinstance(predictor, PretrainedFullTFGridWaveformComponentPredictor):
        if cached_spectrograms is not None:
            raise ValueError(
                "full TF-GridNet uses cached pretrained-prefix features, "
                "not log-spectrogram cache"
            )
        cached_inputs = cache_full_tfgrid_prefix_features(
            predictor,
            examples,
            device,
        )
        adapted_blocks = list(
            predictor.blocks[predictor.prefix_blocks :].parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": adapted_blocks,
                    "lr": FULL_TFGRID_BACKBONE_LEARNING_RATE,
                },
                {
                    "params": predictor.regressor.parameters(),
                    "lr": SCREEN_LEARNING_RATE,
                },
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=SCREEN_LEARNING_RATE,
            weight_decay=1e-4,
        )
    train_indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_train"
    ]
    generator = random.Random(seed)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    history: list[dict[str, float]] = []

    def predict(current: torch.nn.Module, index: int) -> torch.Tensor:
        if cached_inputs is None:
            return current(examples[index].waveform.to(device))
        return cached_waveform_prediction(
            current,
            cached_inputs[index : index + 1].to(device),
        )

    for epoch in range(1, epochs + 1):
        predictor.train()
        order = list(train_indices)
        generator.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), SCREEN_BATCH_SIZE):
            batch = order[start : start + SCREEN_BATCH_SIZE]
            if cached_inputs is None:
                batch_losses = []
                for index in batch:
                    prediction = predict(predictor, index)
                    target = examples[index].own_target.to(device).unsqueeze(0)
                    batch_losses.append(
                        standardized_component_loss(
                            prediction,
                            target,
                            target_mean,
                            target_scale,
                        )
                    )
                loss = torch.stack(batch_losses).mean()
            else:
                prediction = cached_waveform_prediction(
                    predictor,
                    cached_inputs[batch].to(device),
                )
                target = torch.stack(
                    [examples[index].own_target for index in batch]
                ).to(device)
                loss = standardized_component_loss(
                    prediction,
                    target,
                    target_mean,
                    target_scale,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(),
                SCREEN_GRADIENT_CLIP_NORM,
            )
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if max_optimizer_steps and optimizer_steps >= max_optimizer_steps:
                break
        predictor.eval()
        value = calibration_loss(
            predictor,
            predict,
            examples,
            "own_target",
            target_mean,
            target_scale,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "calibration_loss": value,
            }
        )
        print(
            f"waveform_epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
            f"calibration_loss={value:.6f}",
            flush=True,
        )
        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(predictor.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= SCREEN_MIN_EPOCHS and stale >= patience:
            break
        if max_optimizer_steps and optimizer_steps >= max_optimizer_steps:
            break
    if best_state is None:
        raise RuntimeError("waveform predictor did not produce a checkpoint")
    predictor.load_state_dict(best_state)
    predictor.eval()
    return (
        predictor,
        {
            "best_epoch": best_epoch,
            "best_calibration_loss": best_loss,
            "epochs_ran": len(history),
            "optimizer_steps": optimizer_steps,
            "history": history,
            "pretrained_backbone": pretrained_receipt,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in predictor.parameters()
                if parameter.requires_grad
            ),
        },
        target_mean,
        target_scale,
        cached_inputs,
    )


def predict_shared(
    head: torch.nn.Module,
    pooled_features: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
    candidate: str,
) -> torch.Tensor:
    predictions = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(pooled_features), SCREEN_BATCH_SIZE):
            features = pooled_features[
                start : start + SCREEN_BATCH_SIZE
            ].to(device)
            normalized = shared_head_forward(head, features, candidate)
            predictions.append(
                denormalize_components(
                    normalized,
                    target_mean,
                    target_scale,
                ).cpu()
            )
    return torch.cat(predictions)


def predict_waveforms(
    predictor: torch.nn.Module,
    examples: list[Example],
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
    cached_inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    predictions = []
    predictor.eval()
    with torch.inference_mode():
        for index, example in enumerate(examples):
            if cached_inputs is None:
                normalized = predictor(example.waveform.to(device))
            else:
                normalized = cached_waveform_prediction(
                    predictor,
                    cached_inputs[index : index + 1].to(device),
                )
            predictions.append(
                denormalize_components(normalized, target_mean, target_scale).cpu()[0]
            )
    return torch.stack(predictions)


def cache_waveform_spectrograms(
    predictor: torch.nn.Module,
    examples: list[Example],
) -> torch.Tensor:
    spectrograms = []
    with torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            spectrograms.append(predictor.cache_features(example.waveform).cpu()[0])
            if index % 50 == 0 or index == len(examples):
                print(f"waveform_spectrogram_rows={index}/{len(examples)}", flush=True)
    shapes = {tuple(spectrogram.shape) for spectrogram in spectrograms}
    if len(shapes) != 1:
        raise ValueError(f"waveform spectrogram shapes differ: {sorted(shapes)}")
    return torch.stack(spectrograms)


def cache_full_tfgrid_prefix_features(
    predictor: PretrainedFullTFGridWaveformComponentPredictor,
    examples: list[Example],
    device: torch.device,
) -> torch.Tensor:
    prefix_features = []
    predictor.eval()
    with torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            spectrogram = predictor.spectrogram_features(
                example.waveform.to(device)
            )
            prefix = predictor.encode_frozen_prefix(spectrogram)
            prefix_features.append(prefix.cpu()[0])
            if index % 25 == 0 or index == len(examples):
                print(
                    f"full_tfgrid_prefix_rows={index}/{len(examples)}",
                    flush=True,
                )
    shapes = {tuple(features.shape) for features in prefix_features}
    if len(shapes) != 1:
        raise ValueError(
            f"full TF-GridNet prefix shapes differ: {sorted(shapes)}"
        )
    return torch.stack(prefix_features)


def cache_direct_component_features(
    predictor: DifferentiableAVQIComponentEstimator,
    examples: list[Example],
    device: torch.device,
) -> torch.Tensor:
    components = []
    predictor.eval()
    with torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            raw = predictor.raw_components(example.waveform.to(device))
            components.append(raw.cpu()[0])
            if index % 50 == 0 or index == len(examples):
                print(
                    f"direct_component_rows={index}/{len(examples)}",
                    flush=True,
                )
    return torch.stack(components)


def cached_waveform_prediction(
    predictor: torch.nn.Module,
    cached_inputs: torch.Tensor,
) -> torch.Tensor:
    if isinstance(predictor, PretrainedFullTFGridWaveformComponentPredictor):
        return predictor.forward_cached_prefix(cached_inputs)
    if isinstance(predictor, DifferentiableAVQIComponentEstimator):
        return predictor.forward_proxy_features(cached_inputs)
    return predictor.forward_spectrogram(cached_inputs)


def fit_component_calibrator(
    examples: list[Example],
    predictions: torch.Tensor,
    target_attribute: str,
    device: torch.device,
) -> ComponentAffineCalibrator:
    indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_calibration"
    ]
    estimate = predictions[indices].to(device)
    target = torch.stack(
        [getattr(examples[index], target_attribute) for index in indices]
    ).to(device)
    estimate_centered = estimate - estimate.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    variance = estimate_centered.square().mean(dim=0).clamp_min(1e-8)
    scale = (
        (estimate_centered * target_centered).mean(dim=0) / variance
    ).clamp_min(1e-4)
    bias = target.mean(dim=0) - scale * estimate.mean(dim=0)
    return ComponentAffineCalibrator(scale, bias).to(device)


def apply_component_calibrator(
    predictions: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
) -> torch.Tensor:
    with torch.inference_mode():
        device = calibrator.scale.device
        return calibrator(predictions.to(device)).cpu()


def calibrated_normalized_prediction(
    normalized_prediction: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
) -> torch.Tensor:
    raw_prediction = denormalize_components(
        normalized_prediction,
        target_mean,
        target_scale,
    )
    calibrated = calibrator(raw_prediction)
    return (calibrated - target_mean) / target_scale.clamp_min(1e-8)


def safe_spearman(reference: np.ndarray, estimate: np.ndarray) -> float:
    value = float(stats.spearmanr(reference, estimate).statistic)
    return value if math.isfinite(value) else -1.0


def component_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    train_scale: torch.Tensor,
    include_delta_gate: bool = False,
    delta_spearman: dict[str, float] | None = None,
    include_stability_gate: bool = False,
    paired_stability_nmae: dict[str, float] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    reference_np = reference.numpy()
    estimate_np = estimate.numpy()
    train_scale_np = train_scale.detach().cpu().numpy()
    for index, name in enumerate(AVQI_COMPONENT_NAMES):
        truth = reference_np[:, index]
        prediction = estimate_np[:, index]
        mae = float(np.mean(np.abs(prediction - truth)))
        variance = float(np.sum((truth - truth.mean()) ** 2))
        slope = float(
            np.sum((truth - truth.mean()) * (prediction - prediction.mean()))
            / max(variance, 1e-12)
        )
        rho = safe_spearman(truth, prediction)
        nmae = mae / max(float(train_scale_np[index]), 1e-8)
        gates = {
            "level_spearman_ge_0_70": rho >= LEVEL_SPEARMAN_GATE,
            "normalized_mae_le_0_50": nmae <= NORMALIZED_MAE_GATE,
            "calibration_slope_0_75_to_1_25": (
                CALIBRATION_SLOPE_RANGE[0]
                <= slope
                <= CALIBRATION_SLOPE_RANGE[1]
            ),
        }
        if include_delta_gate:
            if delta_spearman is None:
                raise ValueError("delta metrics are required by this gate")
            gates["delta_spearman_ge_0_60"] = (
                delta_spearman[name] >= DELTA_SPEARMAN_GATE
            )
        if include_stability_gate:
            if paired_stability_nmae is None:
                raise ValueError("paired stability metrics are required by this gate")
            gates["paired_clean_target_stability_nmae_le_0_25"] = (
                paired_stability_nmae[name] <= PAIRED_STABILITY_NMAE_GATE
            )
        output[name] = {
            "rows": len(truth),
            "level_spearman": rho,
            "normalized_mae": nmae,
            "calibration_slope": slope,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    return output


def paired_delta_spearman(
    examples: list[Example], predictions: torch.Tensor
) -> dict[str, float]:
    holdout = [
        (index, example)
        for index, example in enumerate(examples)
        if example.split == "surrogate_holdout"
    ]
    grouped: dict[
        tuple[str, str, str],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
    ] = {}
    for index, example in holdout:
        grouped.setdefault(
            (example.speaker_id, example.sample_id, example.view), {}
        )[example.condition] = (example.own_target, predictions[index])
    exact_deltas = []
    predicted_deltas = []
    for conditions in grouped.values():
        if "clean" not in conditions or len(conditions) < 2:
            raise ValueError("holdout clean/degraded pairing failed")
        for condition, values in conditions.items():
            if condition == "clean":
                continue
            exact_deltas.append(values[0] - conditions["clean"][0])
            predicted_deltas.append(values[1] - conditions["clean"][1])
    exact = torch.stack(exact_deltas).numpy()
    predicted = torch.stack(predicted_deltas).numpy()
    return {
        name: safe_spearman(exact[:, index], predicted[:, index])
        for index, name in enumerate(AVQI_COMPONENT_NAMES)
    }


def paired_clean_target_stability(
    examples: list[Example],
    predictions: torch.Tensor,
    train_scale: torch.Tensor,
) -> dict[str, float]:
    holdout = [
        (index, example)
        for index, example in enumerate(examples)
        if example.split == "surrogate_holdout"
    ]
    grouped: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    for index, example in holdout:
        grouped.setdefault(
            (example.speaker_id, example.sample_id, example.view), {}
        )[example.condition] = predictions[index]
    deltas = []
    for conditions in grouped.values():
        if "clean" not in conditions or len(conditions) < 2:
            raise ValueError("holdout clean/degraded pairing failed")
        for condition, prediction in conditions.items():
            if condition == "clean":
                continue
            deltas.append(
                (prediction - conditions["clean"]).abs()
                / train_scale.cpu().clamp_min(1e-8)
            )
    mean_delta = torch.stack(deltas).mean(dim=0)
    return {
        name: float(mean_delta[index])
        for index, name in enumerate(AVQI_COMPONENT_NAMES)
    }


def route_metrics(
    examples: list[Example],
    predictions: torch.Tensor,
    target_attribute: str,
    train_scale: torch.Tensor,
    primary_filter: Callable[[Example], bool],
    include_delta_gate: bool,
    include_stability_gate: bool = False,
) -> dict[str, Any]:
    indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_holdout" and primary_filter(example)
    ]
    reference = torch.stack([getattr(examples[index], target_attribute) for index in indices])
    delta = paired_delta_spearman(examples, predictions) if include_delta_gate else None
    stability = (
        paired_clean_target_stability(examples, predictions, train_scale)
        if include_stability_gate
        else None
    )
    primary = component_metrics(
        reference,
        predictions[indices],
        train_scale,
        include_delta_gate=include_delta_gate,
        delta_spearman=delta,
        include_stability_gate=include_stability_gate,
        paired_stability_nmae=stability,
    )
    slices: dict[str, Any] = {}
    for key, predicate in {
        "clean": lambda example: example.condition == "clean",
        "phone": lambda example: example.condition == "aug16k_phone",
        "cs": lambda example: example.view == "cs",
        "sv": lambda example: example.view == "sv",
        "healthy": lambda example: example.label == "healthy",
        "patient": lambda example: example.label == "patient",
    }.items():
        selected = [
            index
            for index, example in enumerate(examples)
            if example.split == "surrogate_holdout" and predicate(example)
        ]
        if len(selected) >= 4:
            slice_reference = torch.stack(
                [getattr(examples[index], target_attribute) for index in selected]
            )
            slices[key] = component_metrics(
                slice_reference,
                predictions[selected],
                train_scale,
            )
    return {
        "primary": primary,
        "paired_delta_spearman": delta,
        "paired_clean_target_stability_nmae": stability,
        "slices": slices,
    }


def perturbations(waveform: torch.Tensor) -> dict[str, torch.Tensor]:
    waveform = waveform.reshape(-1)
    rms = waveform.square().mean().sqrt().clamp_min(1e-8)
    generator = torch.Generator(device=waveform.device).manual_seed(20260811)
    noise = torch.randn(
        waveform.shape,
        generator=generator,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    noise = noise * (rms / math.sqrt(10.0)) / noise.square().mean().sqrt().clamp_min(1e-8)
    spectrum = torch.fft.rfft(waveform)
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / SAMPLE_RATE).to(
        waveform.device
    )
    lowpass = torch.fft.irfft(
        spectrum * (frequencies <= 3_000.0), n=waveform.numel()
    )
    shift = int(0.1 * SAMPLE_RATE)
    # A circular shift tests time-position invariance without introducing a
    # silent prefix, which previously confounded the anti-shortcut gate.
    shifted = torch.roll(waveform, shifts=shift)
    time = torch.arange(waveform.numel(), device=waveform.device) / SAMPLE_RATE
    amplitude_modulated = waveform * (
        1.0 + 0.5 * torch.sin(2.0 * math.pi * 5.0 * time)
    )
    amplitude_modulated = (
        amplitude_modulated
        * rms
        / amplitude_modulated.square().mean().sqrt().clamp_min(1e-8)
    )
    tone = torch.sin(2.0 * math.pi * 150.0 * time)
    tone = tone * rms / tone.square().mean().sqrt().clamp_min(1e-8)
    return {
        "clean": waveform,
        "gain_minus12db": waveform * 0.25,
        "circular_shift_100ms": shifted,
        "noise_10db": waveform + noise,
        "rms_matched_am_5hz": amplitude_modulated,
        "lowpass_3khz": lowpass,
        "rms_matched_150hz_tone": tone,
        "silence": torch.zeros_like(waveform),
    }


def anti_shortcut_report(
    examples: list[Example],
    predict: Callable[[torch.Tensor], torch.Tensor],
    train_scale: torch.Tensor,
    expect_degradation_sensitivity: bool,
) -> dict[str, Any]:
    selected = [
        example
        for example in examples
        if example.split == "surrogate_holdout"
        and example.condition == "clean"
        and example.view == "sv"
        and example.label == "patient"
    ][:4]
    if len(selected) < 2:
        raise ValueError("insufficient pathological holdout cases for anti-shortcut")
    distances: dict[str, list[torch.Tensor]] = {}
    for example in selected:
        variants = perturbations(example.waveform)
        clean_prediction = predict(variants["clean"])
        for name, waveform in variants.items():
            prediction = predict(waveform)
            distance = (prediction - clean_prediction).abs() / train_scale.cpu()
            distances.setdefault(name, []).append(distance)
    means = {
        variant: torch.stack(values).mean(dim=0)
        for variant, values in distances.items()
    }
    components: dict[str, Any] = {}
    for index, component in enumerate(AVQI_COMPONENT_NAMES):
        values = {name: float(value[index]) for name, value in means.items()}
        gates: dict[str, bool] = {
            "silence_moves_away": values["silence"] >= 0.25,
            "tone_moves_away": values["rms_matched_150hz_tone"] >= 0.25,
            "gain_nearly_invariant": values["gain_minus12db"] <= 0.10,
            "circular_shift_nearly_invariant": (
                values["circular_shift_100ms"] <= 0.10
            ),
        }
        if expect_degradation_sensitivity:
            family = COMPONENT_FAMILIES[component]
            if family == "periodicity_noise":
                gates.update(
                    {
                        "noise_moves_away": values["noise_10db"] >= 0.10,
                        "gain_less_than_noise": (
                            values["gain_minus12db"] < values["noise_10db"]
                        ),
                        "circular_shift_less_than_noise": (
                            values["circular_shift_100ms"]
                            < values["noise_10db"]
                        ),
                    }
                )
            elif family == "amplitude_modulation":
                gates.update(
                    {
                        "amplitude_modulation_moves_away": (
                            values["rms_matched_am_5hz"] >= 0.10
                        ),
                        "gain_less_than_amplitude_modulation": (
                            values["gain_minus12db"]
                            < values["rms_matched_am_5hz"]
                        ),
                        "circular_shift_less_than_amplitude_modulation": (
                            values["circular_shift_100ms"]
                            < values["rms_matched_am_5hz"]
                        ),
                    }
                )
            elif family == "spectral_shape":
                gates.update(
                    {
                        "lowpass_moves_away": values["lowpass_3khz"] >= 0.10,
                        "gain_less_than_lowpass": (
                            values["gain_minus12db"] < values["lowpass_3khz"]
                        ),
                        "circular_shift_less_than_lowpass": (
                            values["circular_shift_100ms"]
                            < values["lowpass_3khz"]
                        ),
                    }
                )
            else:
                raise ValueError(f"unknown component family: {family}")
        components[component] = {
            "mean_standardized_distance": values,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    return {"case_count": len(selected), "components": components}


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(gradients)).cpu())


def component_input_gradient_report(
    input_tensor: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for index, component in enumerate(AVQI_COMPONENT_NAMES):
        current_input = input_tensor.detach().clone().requires_grad_(True)
        prediction = predict(current_input)
        gradient = torch.autograd.grad(
            prediction[:, index].sum(),
            current_input,
            retain_graph=False,
            create_graph=False,
        )[0]
        norm = float(torch.linalg.vector_norm(gradient.detach()).cpu())
        finite = bool(torch.isfinite(gradient).all()) and math.isfinite(norm)
        gates = {
            "finite": finite,
            "nonzero": norm > 1e-10,
            "bounded": norm <= COMPONENT_INPUT_GRADIENT_MAX,
        }
        report[component] = {
            "gradient_norm": norm,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    return report


def fixed_segment(waveform: torch.Tensor, samples: int = 48_000) -> torch.Tensor:
    if waveform.numel() >= samples:
        return waveform[:samples]
    return torch.nn.functional.pad(waveform, (0, samples - waveform.numel()))


def deterministic_training_segments(
    waveform: torch.Tensor,
    samples: int = TRAINING_SEGMENT_SAMPLES,
) -> list[torch.Tensor]:
    """Return start/middle/end crops matching the generator training length."""
    waveform = waveform.reshape(-1)
    if waveform.numel() <= samples:
        return [fixed_segment(waveform, samples)]
    last_start = waveform.numel() - samples
    starts = sorted({0, last_start // 2, last_start})
    return [waveform[start : start + samples] for start in starts]


def training_segment_transfer_report(
    examples: list[Example],
    predict: Callable[[torch.Tensor], torch.Tensor],
    train_scale: torch.Tensor,
    target_attribute: str,
) -> dict[str, Any]:
    """Check full-utterance exact targets on the 3 s deployment input domain."""
    normalized_errors: list[torch.Tensor] = []
    example_count = 0
    segment_count = 0
    for example in examples:
        if example.split != "surrogate_holdout":
            continue
        target = getattr(example, target_attribute).cpu()
        example_count += 1
        for segment in deterministic_training_segments(example.waveform):
            prediction = predict(segment)
            normalized_errors.append(
                (prediction - target).abs()
                / train_scale.cpu().clamp_min(1e-8)
            )
            segment_count += 1
    if not normalized_errors:
        raise ValueError("no holdout examples for training-segment transfer gate")
    mean_error = torch.stack(normalized_errors).mean(dim=0)
    components: dict[str, Any] = {}
    for index, component in enumerate(AVQI_COMPONENT_NAMES):
        nmae = float(mean_error[index])
        passed = math.isfinite(nmae) and nmae <= SEGMENT_TRANSFER_NMAE_GATE
        components[component] = {
            "normalized_mae": nmae,
            "gate": SEGMENT_TRANSFER_NMAE_GATE,
            "decision": "PASS" if passed else "FAIL",
        }
    return {
        "training_segment_samples": TRAINING_SEGMENT_SAMPLES,
        "crop_positions": ["start", "middle", "end"],
        "example_count": example_count,
        "segment_count": segment_count,
        "target_attribute": target_attribute,
        "components": components,
    }


def independent_gradient_smoke(
    generator: SEMambapp,
    config: dict[str, Any],
    waveform_predictor: torch.nn.Module,
    waveform_mean: torch.Tensor,
    waveform_scale: torch.Tensor,
    waveform_calibrator: ComponentAffineCalibrator,
    examples: list[Example],
    device: torch.device,
) -> dict[str, Any]:
    """Verify one frozen scorer's gradients through the decoded waveform path."""
    example = next(
        item
        for item in examples
        if item.split == "surrogate_holdout"
        and item.condition == "aug16k_phone"
        and item.view == "sv"
        and item.label == "patient"
    )
    waveform = fixed_segment(example.waveform).to(device)
    clean_target = example.clean_target.to(device).unsqueeze(0)
    backbone_parameters = [
        parameter
        for name, parameter in generator.named_parameters()
        if name.startswith("dense_encoder") or name.startswith("TSMamba")
    ]
    decoder_parameters = [
        parameter
        for name, parameter in generator.named_parameters()
        if name.startswith("mask_decoder") or name.startswith("phase_decoder")
    ]

    generator.zero_grad(set_to_none=True)
    waveform_predictor.zero_grad(set_to_none=True)
    freeze_module_for_input_gradient(waveform_predictor)
    enhanced = enhance_waveform(generator, waveform, config)
    normalized_prediction = waveform_predictor(enhanced)
    normalized_prediction = calibrated_normalized_prediction(
        normalized_prediction,
        waveform_mean,
        waveform_scale,
        waveform_calibrator,
    )
    loss = standardized_component_loss(
        normalized_prediction,
        clean_target,
        waveform_mean,
        waveform_scale,
    )
    loss.backward()
    backbone_norm = gradient_norm(backbone_parameters)
    decoder_norm = gradient_norm(decoder_parameters)
    predictor_gradients_absent = all(
        parameter.grad is None for parameter in waveform_predictor.parameters()
    )
    finite = all(
        math.isfinite(value) for value in (backbone_norm, decoder_norm)
    )

    with torch.no_grad():
        predictor_input = enhance_waveform(generator, waveform, config)

    def component_prediction(output_waveform: torch.Tensor) -> torch.Tensor:
        normalized = waveform_predictor(output_waveform)
        return calibrated_normalized_prediction(
            normalized,
            waveform_mean,
            waveform_scale,
            waveform_calibrator,
        )

    component_gradients = component_input_gradient_report(
        predictor_input,
        component_prediction,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "backbone_gradient_norm": backbone_norm,
        "decoder_gradient_norm": decoder_norm,
        "predictor_gradients_absent": predictor_gradients_absent,
        "component_input_gradients": component_gradients,
        "input_gradient_execution_mode": (
            "eval_except_zero_dropout_recurrent_modules"
        ),
        "decision": (
            "PASS"
            if finite
            and backbone_norm > 1e-8
            and decoder_norm > 1e-8
            and predictor_gradients_absent
            else "FAIL"
        ),
    }


def gradient_smokes(
    generator: SEMambapp,
    config: dict[str, Any],
    selected_head: torch.nn.Module,
    selected_candidate: str,
    shared_mean: torch.Tensor,
    shared_scale: torch.Tensor,
    shared_calibrator: ComponentAffineCalibrator,
    waveform_predictor: torch.nn.Module,
    waveform_mean: torch.Tensor,
    waveform_scale: torch.Tensor,
    waveform_calibrator: ComponentAffineCalibrator,
    examples: list[Example],
    device: torch.device,
) -> dict[str, Any]:
    example = next(
        item
        for item in examples
        if item.split == "surrogate_holdout"
        and item.condition == "aug16k_phone"
        and item.view == "sv"
        and item.label == "patient"
    )
    waveform = fixed_segment(example.waveform).to(device)
    clean_target = example.clean_target.to(device).unsqueeze(0)

    generator.zero_grad(set_to_none=True)
    selected_head.zero_grad(set_to_none=True)
    freeze_module_for_input_gradient(selected_head)
    maps = shared_feature_maps(generator, waveform, config)
    shared_pooled = pool_shared_candidate(maps, selected_candidate)
    shared_prediction = shared_head_forward(
        selected_head,
        shared_pooled,
        selected_candidate,
    )
    shared_prediction = calibrated_normalized_prediction(
        shared_prediction,
        shared_mean,
        shared_scale,
        shared_calibrator,
    )
    shared_loss = standardized_component_loss(
        shared_prediction,
        clean_target,
        shared_mean,
        shared_scale,
    )
    shared_loss.backward()
    backbone_parameters = [
        parameter
        for name, parameter in generator.named_parameters()
        if name.startswith("dense_encoder") or name.startswith("TSMamba")
    ]
    decoder_parameters = [
        parameter
        for name, parameter in generator.named_parameters()
        if name.startswith("mask_decoder") or name.startswith("phase_decoder")
    ]
    shared_backbone_norm = gradient_norm(backbone_parameters)
    shared_head_norm = gradient_norm(list(selected_head.parameters()))
    head_gradients_absent = all(
        parameter.grad is None for parameter in selected_head.parameters()
    )
    shared_decoder_norm = gradient_norm(decoder_parameters)
    output_conditioned = selected_candidate in {
        "enhanced_spectral",
        "output_phase_tfgrid",
    }
    shared_decoder_path_valid = (
        shared_decoder_norm > 1e-8
        if output_conditioned
        else shared_decoder_norm == 0.0
    )
    shared_finite = all(
        math.isfinite(value)
        for value in (shared_backbone_norm, shared_head_norm, shared_decoder_norm)
    )

    generator.zero_grad(set_to_none=True)
    with torch.no_grad():
        shared_maps = shared_feature_maps(generator, waveform, config)
        shared_head_input = pool_shared_candidate(
            shared_maps,
            selected_candidate,
        )

    def shared_component_prediction(head_input: torch.Tensor) -> torch.Tensor:
        normalized = shared_head_forward(
            selected_head,
            head_input,
            selected_candidate,
        )
        return calibrated_normalized_prediction(
            normalized,
            shared_mean,
            shared_scale,
            shared_calibrator,
        )

    shared_component_gradients = component_input_gradient_report(
        shared_head_input,
        shared_component_prediction,
    )

    independent_report = independent_gradient_smoke(
        generator,
        config,
        waveform_predictor,
        waveform_mean,
        waveform_scale,
        waveform_calibrator,
        examples,
        device,
    )
    return {
        "shared_dual_head": {
            "loss": float(shared_loss.detach().cpu()),
            "backbone_gradient_norm": shared_backbone_norm,
            "head_gradient_norm": shared_head_norm,
            "head_gradients_absent": head_gradients_absent,
            "decoder_gradient_norm": shared_decoder_norm,
            "decoder_gradient_expected": (
                "nonzero"
                if output_conditioned
                else "zero"
            ),
            "component_input_gradients": shared_component_gradients,
            "input_gradient_execution_mode": (
                "eval_except_zero_dropout_recurrent_modules"
            ),
            "decision": (
                "PASS"
                if shared_finite
                and shared_backbone_norm > 1e-8
                and head_gradients_absent
                and shared_decoder_path_valid
                else "FAIL"
            ),
        },
        "frozen_independent_predictor": independent_report,
    }


def external_coverage_report(
    eligible_rows: list[dict[str, str]],
    usable_rows: list[dict[str, str]],
    expected_rows: int,
) -> dict[str, Any]:
    if len(eligible_rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} external rows, found {len(eligible_rows)}"
        )
    invalid_rows = [row for row in eligible_rows if row["scoring_status"] != "ok"]
    return {
        "total_rows": len(eligible_rows),
        "usable_rows": len(usable_rows),
        "invalid_rows": len(invalid_rows),
        "fraction": len(usable_rows) / len(eligible_rows),
        "gate": EXTERNAL_COVERAGE_GATE,
        "decision": (
            "PASS"
            if len(usable_rows) / len(eligible_rows) >= EXTERNAL_COVERAGE_GATE
            else "FAIL"
        ),
        "invalid_cases": [
            {
                "candidate": row.get("candidate", ""),
                "condition": row.get("condition", ""),
                "speaker_id": row.get("speaker_id", ""),
                "view": row.get("view", ""),
                "scoring_status": row.get("scoring_status", ""),
                "error_type": row.get("error_type", ""),
                "error_message": row.get("error_message", ""),
            }
            for row in invalid_rows
        ],
    }


def external_stress_test(
    csv_path: Path,
    predictor: torch.nn.Module,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
    forbidden_speaker_ids: set[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if row["source_type"] == "enhanced"
        and row["candidate"] in EXTERNAL_CANDIDATES
        and row["condition"] in EXTERNAL_CONDITIONS
        and row["view"] in {"cs", "sv"}
    ]
    selected = [row for row in eligible if row["scoring_status"] == "ok"]
    expected_rows = (
        EXPECTED_EXTERNAL_SPEAKERS
        * 2
        * len(EXTERNAL_CONDITIONS)
        * len(EXTERNAL_CANDIDATES)
    )
    coverage = external_coverage_report(eligible, selected, expected_rows)
    speaker_ids = {row["speaker_id"] for row in eligible}
    overlap = speaker_ids & forbidden_speaker_ids
    if overlap:
        raise ValueError(f"external speaker leakage: {sorted(overlap)}")
    if len(speaker_ids) != EXPECTED_EXTERNAL_SPEAKERS:
        raise ValueError(
            f"expected {EXPECTED_EXTERNAL_SPEAKERS} external speakers, "
            f"found {len(speaker_ids)}"
        )
    predictions = []
    references = []
    output_rows: list[dict[str, Any]] = []
    predictor.eval()
    with torch.inference_mode():
        for index, row in enumerate(selected, start=1):
            path = Path(row[f"{row['view']}_path"])
            waveform = load_waveform(path).to(device)
            normalized = predictor(waveform)
            raw_prediction = denormalize_components(
                normalized,
                target_mean,
                target_scale,
            )
            prediction = calibrator(raw_prediction).cpu()[0]
            reference = component_tensor(row, prefix="audio_")
            predictions.append(prediction)
            references.append(reference)
            item: dict[str, Any] = {
                "candidate": row["candidate"],
                "condition": row["condition"],
                "speaker_id": row["speaker_id"],
                "view": row["view"],
                "label": row["label"],
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
            }
            for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
                item[f"exact_{component}"] = float(reference[component_index])
                item[f"predicted_{component}"] = float(prediction[component_index])
            output_rows.append(item)
            if index % 50 == 0 or index == len(selected):
                print(f"external_rows={index}/{len(selected)}", flush=True)
    reference_tensor = torch.stack(references)
    prediction_tensor = torch.stack(predictions)
    primary_indices = [
        index
        for index, row in enumerate(selected)
        if row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
    ]
    expected_primary_rows = (
        EXPECTED_EXTERNAL_SPEAKERS * 2 * len(EXTERNAL_CONDITIONS)
    )
    primary_eligible = [
        row for row in eligible if row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
    ]
    primary_selected = [
        row for row in selected if row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
    ]
    primary_coverage = external_coverage_report(
        primary_eligible,
        primary_selected,
        expected_primary_rows,
    )
    if not primary_indices:
        raise ValueError("no usable primary external rows")
    primary = component_metrics(
        reference_tensor[primary_indices],
        prediction_tensor[primary_indices],
        target_scale,
    )
    stress_overall = component_metrics(
        reference_tensor,
        prediction_tensor,
        target_scale,
    )
    slices: dict[str, Any] = {}
    slice_coverage: dict[str, Any] = {}
    for field, values in {
        "condition": EXTERNAL_CONDITIONS,
        "view": ("cs", "sv"),
        "label": ("healthy", "patient"),
    }.items():
        for value in values:
            eligible_slice = [
                row
                for row in primary_eligible
                if row[field] == value
            ]
            selected_slice = [
                row
                for row in primary_selected
                if row[field] == value
            ]
            if eligible_slice:
                slice_coverage[f"{field}={value}"] = external_coverage_report(
                    eligible_slice,
                    selected_slice,
                    len(eligible_slice),
                )
            indices = [
                i
                for i, row in enumerate(selected)
                if row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
                and row[field] == value
            ]
            if len(indices) >= 4:
                slices[f"{field}={value}"] = component_metrics(
                    reference_tensor[indices], prediction_tensor[indices], target_scale
                )
    severe_sv_key = "view=sv&sample_group=pathological_severe"
    severe_sv_eligible = [
        row
        for row in primary_eligible
        if row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    severe_sv_selected = [
        row
        for row in primary_selected
        if row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    if severe_sv_eligible:
        slice_coverage[severe_sv_key] = external_coverage_report(
            severe_sv_eligible,
            severe_sv_selected,
            len(severe_sv_eligible),
        )
    severe_sv_indices = [
        index
        for index, row in enumerate(selected)
        if row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
        and row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    if len(severe_sv_indices) >= 4:
        slices[severe_sv_key] = component_metrics(
            reference_tensor[severe_sv_indices],
            prediction_tensor[severe_sv_indices],
            target_scale,
        )
    stress_candidate_slices = {}
    for candidate in EXTERNAL_CANDIDATES:
        indices = [
            index
            for index, row in enumerate(selected)
            if row["candidate"] == candidate
        ]
        stress_candidate_slices[candidate] = component_metrics(
            reference_tensor[indices],
            prediction_tensor[indices],
            target_scale,
        )
    report = {
        "rows": len(selected),
        "speaker_count": len(speaker_ids),
        "speaker_overlap_with_surrogate": 0,
        "coverage": coverage,
        "primary_coverage": primary_coverage,
        "primary_candidate": EXTERNAL_PRIMARY_CANDIDATE,
        "primary": primary,
        "slices": slices,
        "slice_coverage": slice_coverage,
        "stress_overall": stress_overall,
        "stress_candidate_slices": stress_candidate_slices,
    }
    return report, output_rows


def shared_external_stress_test(
    csv_path: Path,
    generator: SEMambapp,
    config: dict[str, Any],
    head: torch.nn.Module,
    candidate: str,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
    forbidden_speaker_ids: set[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_conditioned = candidate == "output_phase_tfgrid"
    eligible = [
        row
        for row in rows
        if row["source_type"] == "input"
        and row["condition"] in EXTERNAL_CONDITIONS
        and row["view"] in {"cs", "sv"}
    ]
    enhanced_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    if output_conditioned:
        for row in rows:
            if (
                row["source_type"] == "enhanced"
                and row["candidate"] == EXTERNAL_PRIMARY_CANDIDATE
                and row["condition"] in EXTERNAL_CONDITIONS
                and row["view"] in {"cs", "sv"}
            ):
                key = (row["speaker_id"], row["condition"], row["view"])
                if key in enhanced_by_key:
                    raise ValueError(f"duplicate enhanced external row: {key}")
                enhanced_by_key[key] = row

    def external_row_is_usable(row: dict[str, str]) -> bool:
        if row["scoring_status"] != "ok":
            return False
        if not output_conditioned:
            return True
        key = (row["speaker_id"], row["condition"], row["view"])
        return (
            key in enhanced_by_key
            and enhanced_by_key[key]["scoring_status"] == "ok"
        )

    selected = [row for row in eligible if external_row_is_usable(row)]
    expected_rows = EXPECTED_EXTERNAL_SPEAKERS * 2 * len(EXTERNAL_CONDITIONS)
    coverage = external_coverage_report(eligible, selected, expected_rows)
    speaker_ids = {row["speaker_id"] for row in eligible}
    overlap = speaker_ids & forbidden_speaker_ids
    if overlap:
        raise ValueError(f"external speaker leakage: {sorted(overlap)}")
    if len(speaker_ids) != EXPECTED_EXTERNAL_SPEAKERS:
        raise ValueError(
            f"expected {EXPECTED_EXTERNAL_SPEAKERS} external speakers, "
            f"found {len(speaker_ids)}"
        )
    if not selected:
        raise ValueError("no usable shared external rows")
    predictions = []
    references = []
    output_rows: list[dict[str, Any]] = []
    generator.eval()
    head.eval()
    with torch.inference_mode():
        for index, row in enumerate(selected, start=1):
            path = Path(row[f"{row['view']}_path"])
            waveform = load_waveform(path).to(device)
            maps = shared_feature_maps(generator, waveform, config)
            pooled = pool_shared_candidate(maps, candidate)
            normalized = shared_head_forward(head, pooled, candidate)
            raw_prediction = denormalize_components(
                normalized,
                target_mean,
                target_scale,
            )
            prediction = calibrator(raw_prediction).cpu()[0]
            if output_conditioned:
                key = (row["speaker_id"], row["condition"], row["view"])
                reference = component_tensor(enhanced_by_key[key], prefix="audio_")
                reference_source = "exact_enhanced_S3_500"
            else:
                reference = component_tensor(row, prefix="clean_")
                reference_source = "same_speaker_clean"
            predictions.append(prediction)
            references.append(reference)
            item: dict[str, Any] = {
                "condition": row["condition"],
                "speaker_id": row["speaker_id"],
                "view": row["view"],
                "label": row["label"],
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
                "reference_source": reference_source,
            }
            for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
                item[f"exact_{component}"] = float(reference[component_index])
                item[f"predicted_{component}"] = float(
                    prediction[component_index]
                )
            output_rows.append(item)
            if index % 50 == 0 or index == len(selected):
                print(f"shared_external_rows={index}/{len(selected)}", flush=True)
    reference_tensor = torch.stack(references)
    prediction_tensor = torch.stack(predictions)
    primary = component_metrics(reference_tensor, prediction_tensor, target_scale)
    slices: dict[str, Any] = {}
    slice_coverage: dict[str, Any] = {}
    for field, values in {
        "condition": EXTERNAL_CONDITIONS,
        "view": ("cs", "sv"),
        "label": ("healthy", "patient"),
    }.items():
        for value in values:
            eligible_slice = [row for row in eligible if row[field] == value]
            selected_slice = [row for row in selected if row[field] == value]
            if eligible_slice:
                slice_coverage[f"{field}={value}"] = external_coverage_report(
                    eligible_slice,
                    selected_slice,
                    len(eligible_slice),
                )
            indices = [i for i, row in enumerate(selected) if row[field] == value]
            if len(indices) >= 4:
                slices[f"{field}={value}"] = component_metrics(
                    reference_tensor[indices],
                    prediction_tensor[indices],
                    target_scale,
                )
    severe_sv_key = "view=sv&sample_group=pathological_severe"
    severe_sv_eligible = [
        row
        for row in eligible
        if row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    severe_sv_selected = [
        row
        for row in selected
        if row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    if severe_sv_eligible:
        slice_coverage[severe_sv_key] = external_coverage_report(
            severe_sv_eligible,
            severe_sv_selected,
            len(severe_sv_eligible),
        )
    severe_sv_indices = [
        index
        for index, row in enumerate(selected)
        if row["view"] == "sv"
        and row["sample_group"] == "pathological_severe"
    ]
    if len(severe_sv_indices) >= 4:
        slices[severe_sv_key] = component_metrics(
            reference_tensor[severe_sv_indices],
            prediction_tensor[severe_sv_indices],
            target_scale,
        )
    report = {
        "rows": len(selected),
        "speaker_count": len(speaker_ids),
        "speaker_overlap_with_surrogate": 0,
        "coverage": coverage,
        "primary_coverage": coverage,
        "primary_candidate": EXTERNAL_PRIMARY_CANDIDATE,
        "primary": primary,
        "slices": slices,
        "slice_coverage": slice_coverage,
        "stress_overall": primary,
        "reference_source": (
            "exact enhanced S3_500 waveform components"
            if output_conditioned
            else "same-speaker clean waveform components"
        ),
    }
    return report, output_rows


def vctk_external_test(
    csv_path: Path,
    predict: Callable[[torch.Tensor], torch.Tensor],
    target_scale: torch.Tensor,
    forbidden_speaker_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate an already-frozen scorer on the held-out VCTK speakers."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if row["split"] == "vctk_external" and row["view"] == "cs"
    ]
    selected = [row for row in eligible if row["scoring_status"] == "ok"]
    coverage = external_coverage_report(eligible, selected, 12 * 4 * 4)
    speaker_ids = {row["speaker_id"] for row in eligible}
    overlap = speaker_ids & forbidden_speaker_ids
    if overlap:
        raise ValueError(f"VCTK external speaker leakage: {sorted(overlap)}")
    if len(speaker_ids) != 12:
        raise ValueError(f"expected 12 VCTK external speakers, found {len(speaker_ids)}")
    predictions: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        path = Path(row["cs_path"])
        if sha256_file(path) != row["cs_sha256"]:
            raise ValueError(f"VCTK external audio hash mismatch: {path}")
        prediction = predict(load_waveform(path))
        reference = component_tensor(row)
        predictions.append(prediction)
        references.append(reference)
        item: dict[str, Any] = {
            "speaker_id": row["speaker_id"],
            "sample_id": row_sample_id(row),
            "condition": row["condition_id"],
            "view": row["view"],
            "label": row["label"],
            "audio_path": str(path),
            "audio_sha256": row["cs_sha256"],
        }
        for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
            item[f"exact_{component}"] = float(reference[component_index])
            item[f"predicted_{component}"] = float(prediction[component_index])
        output_rows.append(item)
        if index % 50 == 0 or index == len(selected):
            print(f"vctk_external_rows={index}/{len(selected)}", flush=True)
    reference_tensor = torch.stack(references)
    prediction_tensor = torch.stack(predictions)
    primary = component_metrics(reference_tensor, prediction_tensor, target_scale)
    slices: dict[str, Any] = {}
    slice_coverage: dict[str, Any] = {}
    for condition in ("clean", "rir_only", "snr20", "snr10"):
        eligible_slice = [row for row in eligible if row["condition_id"] == condition]
        selected_slice = [row for row in selected if row["condition_id"] == condition]
        slice_coverage[f"condition={condition}"] = external_coverage_report(
            eligible_slice,
            selected_slice,
            12 * 4,
        )
        indices = [
            index for index, row in enumerate(selected)
            if row["condition_id"] == condition
        ]
        slices[f"condition={condition}"] = component_metrics(
            reference_tensor[indices],
            prediction_tensor[indices],
            target_scale,
        )
    return (
        {
            "rows": len(selected),
            "speaker_count": len(speaker_ids),
            "speaker_overlap_with_surrogate": 0,
            "primary_coverage": coverage,
            "primary": primary,
            "slices": slices,
            "slice_coverage": slice_coverage,
            "required_slices": list(slices),
        },
        output_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(
    examples: list[Example], routes: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        row: dict[str, Any] = {
            "speaker_id": example.speaker_id,
            "sample_id": example.sample_id,
            "split": example.split,
            "condition": example.condition,
            "view": example.view,
            "label": example.label,
            "audio_path": str(example.path),
            "audio_sha256": example.audio_sha256,
        }
        for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
            row[f"own_exact_{component}"] = float(example.own_target[component_index])
            row[f"clean_exact_{component}"] = float(example.clean_target[component_index])
            for route_name, predictions in routes.items():
                row[f"{route_name}_{component}"] = float(
                    predictions[index, component_index]
                )
        rows.append(row)
    return rows


def component_passes(report: dict[str, Any], component: str) -> bool:
    return report["primary"][component]["decision"] == "PASS"


def external_component_passes(report: dict[str, Any], component: str) -> bool:
    if report["primary_coverage"]["decision"] != "PASS":
        return False
    if report["primary"][component]["decision"] != "PASS":
        return False
    return all(
        slice_name in report["slices"]
        and slice_name in report["slice_coverage"]
        and report["slice_coverage"][slice_name]["decision"] == "PASS"
        and report["slices"][slice_name][component]["decision"] == "PASS"
        for slice_name in EXTERNAL_REQUIRED_SLICES
    )


def vctk_external_component_passes(
    report: dict[str, Any] | None,
    component: str,
) -> bool:
    if report is None:
        return True
    if report["primary_coverage"]["decision"] != "PASS":
        return False
    if report["primary"][component]["decision"] != "PASS":
        return False
    return all(
        report["slice_coverage"][slice_name]["decision"] == "PASS"
        and report["slices"][slice_name][component]["decision"] == "PASS"
        for slice_name in report["required_slices"]
    )


def eligible_components(
    metrics: dict[str, Any],
    external_metrics: dict[str, Any],
    anti_shortcut: dict[str, Any],
    gradient: dict[str, Any],
    segment_transfer: dict[str, Any],
    vctk_external: dict[str, Any] | None = None,
) -> list[str]:
    if gradient["decision"] != "PASS":
        return []
    return [
        component
        for component in AVQI_COMPONENT_NAMES
        if component_passes(metrics, component)
        and external_component_passes(external_metrics, component)
        and vctk_external_component_passes(vctk_external, component)
        and anti_shortcut["components"][component]["decision"] == "PASS"
        and gradient["component_input_gradients"][component]["decision"] == "PASS"
        and segment_transfer["components"][component]["decision"] == "PASS"
    ]


def route_has_minimum_component_coverage(components: list[str]) -> bool:
    families = {COMPONENT_FAMILIES[component] for component in components}
    return "periodicity_noise" in families and len(families) >= 2


def route_decision(
    metrics: dict[str, Any],
    external_metrics: dict[str, Any],
    anti_shortcut: dict[str, Any],
    gradient: dict[str, Any],
    segment_transfer: dict[str, Any],
    vctk_external: dict[str, Any] | None = None,
) -> str:
    components = eligible_components(
        metrics,
        external_metrics,
        anti_shortcut,
        gradient,
        segment_transfer,
        vctk_external,
    )
    return (
        "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
        if route_has_minimum_component_coverage(components)
        else "NO_GO_GENERATOR_TRAINING"
    )


def human_summary(report: dict[str, Any]) -> str:
    shared = report["routes"]["shared_dual_head"]
    independent = report["routes"]["frozen_independent_predictor"]
    lines = [
        "# AVQI component backpropagation diagnostic",
        "",
        "## One-line result",
        "",
        report["plain_language_conclusion"],
        "",
        "## Minimal comparison",
        "",
        "| Route | Chosen form | Eligible components | Gradient | Decision |",
        "|---|---|---|---:|---|",
        (
            f"| Shared dual head | {shared['selected_candidate']} | "
            f"{', '.join(shared['eligible_components']) or 'none'} | "
            f"{shared['gradient']['decision']} | {shared['decision']} |"
        ),
        (
            f"| Frozen independent predictor | {independent['selected_architecture']} | "
            f"{', '.join(independent['eligible_components']) or 'none'} | "
            f"{independent['gradient']['decision']} | {independent['decision']} |"
        ),
        "",
        "All six AVQI v03.01 terms are reported in `diagnostic_report.json`; "
        "jitter is not part of this task. A passing gradient only proves that "
        "the route is differentiable. Exact Praat output metrics remain the "
        "promotion evidence.",
        "",
        "No generator optimizer step or formal pathology training was run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    expected_split_speakers = {
        "surrogate_train": args.expected_train_speakers,
        "surrogate_calibration": args.expected_calibration_speakers,
        "surrogate_holdout": args.expected_holdout_speakers,
    }
    if any(count <= 0 for count in expected_split_speakers.values()):
        raise ValueError(
            f"expected split speaker counts must be positive: {expected_split_speakers}"
        )
    shared_candidates = comma_separated_values(args.shared_candidates)
    waveform_architectures = comma_separated_values(args.waveform_architectures)
    allowed_shared = {
        "late_global",
        "late_frequency",
        "late_tfgrid",
        "enhanced_spectral",
        "output_phase_tfgrid",
    }
    allowed_waveform = {
        "global_stats",
        "frequency_aware",
        "phase_frequency_aware",
        "compact_tfgrid",
        "phase_compact_tfgrid",
        "pretrained_full_tfgrid",
        "direct_exact_inspired",
        "direct_praat_soft_v2",
        "direct_praat_hard_v2",
        "direct_praat_hard_shimmer_rms_v3",
        "direct_praat_hard_shimmer_raw_cc_surrogate_v4",
        "direct_praat_hard_shimmer_pulse_chain_v5",
        "direct_praat_hard_shimmer_pulse_path_v6",
    }
    if not set(shared_candidates) <= allowed_shared:
        raise ValueError(
            f"unknown shared candidates: {sorted(set(shared_candidates) - allowed_shared)}"
        )
    if not set(waveform_architectures) <= allowed_waveform:
        raise ValueError(
            "unknown waveform architectures: "
            f"{sorted(set(waveform_architectures) - allowed_waveform)}"
        )
    if "output_phase_tfgrid" in shared_candidates and len(shared_candidates) != 1:
        raise ValueError(
            "output_phase_tfgrid has own-waveform targets and must be screened "
            "separately from legacy clean-target shared heads"
        )
    if args.max_optimizer_steps < 0:
        raise ValueError("max optimizer steps cannot be negative")
    uses_vctk_external = args.vctk_external_label_bank is not None
    if uses_vctk_external != (args.vctk_external_label_bank_sha256 is not None):
        raise ValueError(
            "VCTK external label bank path and SHA256 must be supplied together"
        )
    uses_full_tfgrid = "pretrained_full_tfgrid" in waveform_architectures
    if uses_full_tfgrid and (
        args.full_tfgrid_checkpoint is None
        or args.full_tfgrid_checkpoint_sha256 is None
    ):
        raise ValueError(
            "pretrained_full_tfgrid requires both checkpoint path and SHA256"
        )
    if not uses_full_tfgrid and (
        args.full_tfgrid_checkpoint is not None
        or args.full_tfgrid_checkpoint_sha256 is not None
    ):
        raise ValueError(
            "full TF-GridNet checkpoint arguments require the matching architecture"
        )
    if args.shared_head_epochs != args.waveform_epochs:
        raise ValueError(
            "matched architecture screen requires equal shared and waveform epochs"
        )
    if args.checkpoint.parent.name != EXTERNAL_PRIMARY_CANDIDATE:
        raise ValueError(
            "external comparison is locked to checkpoint directory "
            f"{EXTERNAL_PRIMARY_CANDIDATE}, got {args.checkpoint.parent.name}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.checkpoint_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoints: {args.checkpoint_dir}"
        )
    for path, expected_hash in (
        (args.label_bank, args.label_bank_sha256),
        (args.config, args.config_sha256),
        (args.checkpoint, args.checkpoint_sha256),
        (args.external_exact_csv, args.external_exact_csv_sha256),
    ):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source hash mismatch: {path}")
    if uses_vctk_external and (
        sha256_file(args.vctk_external_label_bank)
        != args.vctk_external_label_bank_sha256
    ):
        raise ValueError(
            f"source hash mismatch: {args.vctk_external_label_bank}"
        )
    if uses_full_tfgrid:
        if sha256_file(args.full_tfgrid_checkpoint) != args.full_tfgrid_checkpoint_sha256:
            raise ValueError(
                f"source hash mismatch: {args.full_tfgrid_checkpoint}"
            )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True)
    args.checkpoint_dir.mkdir(parents=True)
    contract = {
        "schema_version": "avqi-component-predictor-screen-v3",
        "purpose": "single_seed_architecture_screen_no_generator_update",
        "components": list(AVQI_COMPONENT_NAMES),
        "component_loss_weights": dict(
            zip(AVQI_COMPONENT_NAMES, AVQI_COMPONENT_LOSS_WEIGHTS, strict=True)
        ),
        "periodicity_anchor_components": list(PRIMARY_GATE_COMPONENTS),
        "minimum_route_coverage": (
            "at least one CPPS/HNR component and one component from shimmer or LTAS"
        ),
        "jitter_in_primary_task": False,
        "speaker_split": expected_split_speakers,
        "routes": {
            "shared_dual_head": {
                "candidates": list(shared_candidates),
                "target": (
                    "own waveform exact Praat components; frozen before "
                    "attachment to decoded enhanced magnitude and phase"
                    if shared_candidates == ("output_phase_tfgrid",)
                    else "same-speaker clean exact Praat components"
                ),
            },
            "frozen_independent_predictor": {
                "architectures": list(waveform_architectures),
                "target": "input waveform exact Praat components",
                "pretrained_full_tfgrid": (
                    {
                        "backbone": "Hybrid-UniSE discriminative branch",
                        "depth": 8,
                        "embedding": 64,
                        "lstm_hidden": 256,
                        "adaptation_blocks": 1,
                        "time_pool_position": "after_frozen_prefix",
                        "backbone_learning_rate": (
                            FULL_TFGRID_BACKBONE_LEARNING_RATE
                        ),
                        "head_learning_rate": SCREEN_LEARNING_RATE,
                    }
                    if uses_full_tfgrid
                    else None
                ),
                "direct_exact_inspired": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "formulas": [
                            "soft cepstral prominence",
                            "soft autocorrelation HNR",
                            "adjacent-frame shimmer percent and dB",
                            "LTAS band slope and trend tilt",
                        ],
                    }
                    if "direct_exact_inspired" in waveform_architectures
                    else None
                ),
                "direct_praat_soft_v2": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "soft expectation",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "cycle-lag analytic-envelope shimmer",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_soft_v2" in waveform_architectures
                    else None
                ),
                "direct_praat_hard_v2": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "piecewise-differentiable maximum",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "cycle-lag analytic-envelope shimmer",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_hard_v2" in waveform_architectures
                    else None
                ),
                "direct_praat_hard_shimmer_rms_v3": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "piecewise-differentiable maximum",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "period-scaled Hann-RMS shimmer with validity gates",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_hard_shimmer_rms_v3"
                    in waveform_architectures
                    else None
                ),
                "direct_praat_hard_shimmer_raw_cc_surrogate_v4": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "piecewise-differentiable maximum",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "v3 Hann-RMS shimmer forward value",
                            "raw-CC paired-delta shimmer surrogate gradient",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_hard_shimmer_raw_cc_surrogate_v4"
                    in waveform_architectures
                    else None
                ),
                "direct_praat_hard_shimmer_pulse_chain_v5": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "detached hard pulse topology",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "independent 50--400 Hz raw-AC pitch contour",
                            "recursive cross-correlation pulse chain",
                            "asymmetric Hann-RMS shimmer with hard screening",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_hard_shimmer_pulse_chain_v5"
                    in waveform_architectures
                    else None
                ),
                "direct_praat_hard_shimmer_pulse_path_v6": (
                    {
                        "neural_predictor": False,
                        "trainable_parameters": 0,
                        "alignment": "positive per-component affine on train only",
                        "peak_mode": "detached hard pulse topology",
                        "formulas": [
                            "Praat 34 Hz stop-Hann high-pass",
                            "overlap-normalized linear autocorrelation HNR",
                            "periodicity-aware soft voiced mask",
                            "smoothed robust-baseline CPPS",
                            "independent 50--400 Hz raw-AC candidate lattice",
                            "candidate-strength and unvoiced-state pitch path",
                            "sample-aligned recursive cross-correlation pulses",
                            "asymmetric Hann-RMS shimmer with hard screening",
                            "global LTAS slope and trend tilt",
                        ],
                    }
                    if "direct_praat_hard_shimmer_pulse_path_v6"
                    in waveform_architectures
                    else None
                ),
            },
        },
        "calibration": {
            "method": "per-component positive-scale affine",
            "fit_split": "surrogate_calibration",
            "holdout_used_for_fit_or_selection": False,
        },
        "architecture_screen_seed": args.seed,
        "matched_training_budget": {
            "batch_size": SCREEN_BATCH_SIZE,
            "learning_rate": SCREEN_LEARNING_RATE,
            "gradient_clip_norm": SCREEN_GRADIENT_CLIP_NORM,
            "shared_max_epochs": args.shared_head_epochs,
            "independent_max_epochs": args.waveform_epochs,
            "minimum_epochs": SCREEN_MIN_EPOCHS,
            "early_stop_patience": args.patience,
            "maximum_optimizer_steps_per_candidate": args.max_optimizer_steps,
        },
        "multiseed_confirmation": {
            "seeds": [args.seed + 1, args.seed + 2, args.seed + 3],
            "architecture_locked_from": "screen calibration loss only",
            "component_rule": (
                "full component gate in at least two of three locked seeds; "
                "no post-hoc threshold changes"
            ),
            "generator_updates_allowed": False,
        },
        "matched_external_primary_candidate": EXTERNAL_PRIMARY_CANDIDATE,
        "additional_external_stress_candidates": [
            candidate
            for candidate in EXTERNAL_CANDIDATES
            if candidate != EXTERNAL_PRIMARY_CANDIDATE
        ],
        "anti_shortcut": {
            "common_invariance": ["gain_minus12db", "circular_shift_100ms"],
            "common_ood": ["silence", "rms_matched_150hz_tone"],
            "periodicity_noise": "noise_10db",
            "amplitude_modulation": "rms_matched_am_5hz",
            "spectral_shape": "lowpass_3khz",
        },
        "gates": {
            "level_spearman": LEVEL_SPEARMAN_GATE,
            "paired_delta_spearman": DELTA_SPEARMAN_GATE,
            "paired_clean_target_stability_nmae": PAIRED_STABILITY_NMAE_GATE,
            "normalized_mae": NORMALIZED_MAE_GATE,
            "calibration_slope": list(CALIBRATION_SLOPE_RANGE),
            "component_input_gradient_norm": [1e-10, COMPONENT_INPUT_GRADIENT_MAX],
            "external_coverage": EXTERNAL_COVERAGE_GATE,
            "required_external_slices": list(EXTERNAL_REQUIRED_SLICES),
            "training_segment_samples": TRAINING_SEGMENT_SAMPLES,
            "training_segment_transfer_normalized_mae": (
                SEGMENT_TRANSFER_NMAE_GATE
            ),
        },
        "source_sha256": {
            "label_bank": args.label_bank_sha256,
            "config": args.config_sha256,
            "generator_checkpoint": args.checkpoint_sha256,
            "external_exact_csv": args.external_exact_csv_sha256,
            "vctk_external_label_bank": (
                args.vctk_external_label_bank_sha256
                if uses_vctk_external
                else None
            ),
            "full_tfgrid_checkpoint": (
                args.full_tfgrid_checkpoint_sha256 if uses_full_tfgrid else None
            ),
        },
        "source_commit": args.source_commit,
        "artifact_layout": {
            "run_output_dir": str(args.output_dir.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        },
    }
    write_json(args.output_dir / "experiment_contract.json", contract)

    examples, label_bank_coverage = load_examples(
        args.label_bank,
        expected_split_speakers,
    )
    config = load_config(args.config)
    generator = load_generator(config, args.checkpoint, device)
    pooled = extract_shared_features(
        generator,
        examples,
        config,
        device,
        shared_candidates,
    )
    torch.save(pooled, args.output_dir / "shared_features.pt")

    shared_models: dict[str, torch.nn.Module] = {}
    shared_training: dict[str, Any] = {}
    shared_raw_predictions: dict[str, torch.Tensor] = {}
    shared_predictions: dict[str, torch.Tensor] = {}
    shared_calibrators: dict[str, ComponentAffineCalibrator] = {}
    shared_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for candidate in pooled:
        head, training, mean, scale = train_shared_head(
            pooled[candidate],
            examples,
            device,
            args.shared_head_epochs,
            args.patience,
            args.seed,
            candidate,
            args.max_optimizer_steps,
        )
        shared_models[candidate] = head
        training["parameter_count"] = sum(
            parameter.numel() for parameter in head.parameters()
        )
        shared_training[candidate] = training
        shared_stats[candidate] = (mean, scale)
        raw_predictions = predict_shared(
            head,
            pooled[candidate],
            mean,
            scale,
            device,
            candidate,
        )
        shared_target_attribute = training["target_attribute"]
        calibrator = fit_component_calibrator(
            examples,
            raw_predictions,
            shared_target_attribute,
            device,
        )
        shared_raw_predictions[candidate] = raw_predictions
        shared_calibrators[candidate] = calibrator
        shared_predictions[candidate] = apply_component_calibrator(
            raw_predictions,
            calibrator,
        )
        torch.save(
            {
                "state_dict": head.state_dict(),
                "target_mean": mean.cpu(),
                "target_scale": scale.cpu(),
                "calibration_scale": calibrator.scale.cpu(),
                "calibration_bias": calibrator.bias.cpu(),
                "candidate": candidate,
                "target_attribute": shared_target_attribute,
                "parameter_count": sum(
                    parameter.numel() for parameter in head.parameters()
                ),
            },
            args.checkpoint_dir / f"shared_{candidate}_head.pt",
        )
    selected_candidate = min(
        shared_training,
        key=lambda name: shared_training[name]["best_calibration_loss"],
    )

    waveform_models: dict[str, torch.nn.Module] = {}
    waveform_training: dict[str, Any] = {}
    waveform_raw_predictions: dict[str, torch.Tensor] = {}
    waveform_predictions: dict[str, torch.Tensor] = {}
    waveform_calibrators: dict[str, ComponentAffineCalibrator] = {}
    waveform_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    cacheless_architectures = {
        "pretrained_full_tfgrid",
        "direct_exact_inspired",
        "direct_praat_soft_v2",
        "direct_praat_hard_v2",
        "direct_praat_hard_shimmer_rms_v3",
        "direct_praat_hard_shimmer_raw_cc_surrogate_v4",
        "direct_praat_hard_shimmer_pulse_chain_v5",
        "direct_praat_hard_shimmer_pulse_path_v6",
    }
    magnitude_architectures = {
        "global_stats",
        "frequency_aware",
        "compact_tfgrid",
    }
    phase_architectures = {
        "phase_frequency_aware",
        "phase_compact_tfgrid",
    }
    standard_architectures = set(waveform_architectures) - cacheless_architectures
    cached_magnitude_spectrograms: torch.Tensor | None = None
    cached_phase_spectrograms: torch.Tensor | None = None
    if standard_architectures & magnitude_architectures:
        spectrogram_template = WaveformComponentPredictor()
        cached_magnitude_spectrograms = cache_waveform_spectrograms(
            spectrogram_template,
            examples,
        )
    if standard_architectures & phase_architectures:
        phase_template = PhaseAwareFrequencyAwareWaveformComponentPredictor()
        cached_phase_spectrograms = cache_waveform_spectrograms(
            phase_template,
            examples,
        )
    for architecture in waveform_architectures:
        if architecture in cacheless_architectures:
            architecture_cache = None
        elif architecture in phase_architectures:
            architecture_cache = cached_phase_spectrograms
        else:
            architecture_cache = cached_magnitude_spectrograms
        predictor, training, mean, scale, trained_cache = train_waveform_predictor(
            examples,
            device,
            args.waveform_epochs,
            args.patience,
            args.seed,
            architecture,
            architecture_cache,
            args.full_tfgrid_checkpoint,
            args.max_optimizer_steps,
        )
        raw_predictions = predict_waveforms(
            predictor,
            examples,
            mean,
            scale,
            device,
            trained_cache,
        )
        calibrator = fit_component_calibrator(
            examples,
            raw_predictions,
            "own_target",
            device,
        )
        waveform_models[architecture] = predictor
        training["parameter_count"] = sum(
            parameter.numel() for parameter in predictor.parameters()
        )
        waveform_training[architecture] = training
        waveform_stats[architecture] = (mean, scale)
        waveform_raw_predictions[architecture] = raw_predictions
        waveform_calibrators[architecture] = calibrator
        waveform_predictions[architecture] = apply_component_calibrator(
            raw_predictions,
            calibrator,
        )
        torch.save(
            {
                "state_dict": predictor.state_dict(),
                "target_mean": mean.cpu(),
                "target_scale": scale.cpu(),
                "calibration_scale": calibrator.scale.cpu(),
                "calibration_bias": calibrator.bias.cpu(),
                "components": AVQI_COMPONENT_NAMES,
                "architecture": architecture,
                "parameter_count": sum(
                    parameter.numel() for parameter in predictor.parameters()
                ),
                "trainable_parameter_count": training[
                    "trainable_parameter_count"
                ],
                "pretrained_backbone": training["pretrained_backbone"],
            },
            args.checkpoint_dir / f"waveform_{architecture}_predictor.pt",
        )
    selected_architecture = min(
        waveform_training,
        key=lambda name: waveform_training[name]["best_calibration_loss"],
    )

    shared_candidate_reports: dict[str, Any] = {}
    for candidate in pooled:
        _, scale = shared_stats[candidate]
        output_conditioned = candidate == "output_phase_tfgrid"
        shared_target_attribute = (
            "own_target" if output_conditioned else "clean_target"
        )
        shared_primary_filter = (
            (lambda example: True)
            if output_conditioned
            else (lambda example: example.condition == "aug16k_phone")
        )
        shared_candidate_reports[candidate] = {
            "raw": route_metrics(
                examples,
                shared_raw_predictions[candidate],
                shared_target_attribute,
                scale,
                primary_filter=shared_primary_filter,
                include_delta_gate=output_conditioned,
                include_stability_gate=not output_conditioned,
            ),
            "calibrated": route_metrics(
                examples,
                shared_predictions[candidate],
                shared_target_attribute,
                scale,
                primary_filter=shared_primary_filter,
                include_delta_gate=output_conditioned,
                include_stability_gate=not output_conditioned,
            ),
        }
    waveform_architecture_reports: dict[str, Any] = {}
    for architecture in waveform_models:
        _, scale = waveform_stats[architecture]
        waveform_architecture_reports[architecture] = {
            "raw": route_metrics(
                examples,
                waveform_raw_predictions[architecture],
                "own_target",
                scale,
                primary_filter=lambda example: True,
                include_delta_gate=True,
            ),
            "calibrated": route_metrics(
                examples,
                waveform_predictions[architecture],
                "own_target",
                scale,
                primary_filter=lambda example: True,
                include_delta_gate=True,
            ),
        }

    selected_head = shared_models[selected_candidate]
    shared_mean, shared_scale = shared_stats[selected_candidate]
    shared_calibrator = shared_calibrators[selected_candidate]
    waveform_predictor = waveform_models[selected_architecture]
    waveform_mean, waveform_scale = waveform_stats[selected_architecture]
    waveform_calibrator = waveform_calibrators[selected_architecture]
    freeze_module(waveform_predictor)

    def shared_predict(waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if selected_candidate == "output_phase_tfgrid":
                maps = input_phase_feature_maps(
                    waveform.to(device),
                    config,
                )
            else:
                maps = shared_feature_maps(generator, waveform.to(device), config)
            pooled_features = pool_shared_candidate(
                maps,
                selected_candidate,
                training=selected_candidate == "output_phase_tfgrid",
            )
            normalized = shared_head_forward(
                selected_head,
                pooled_features,
                selected_candidate,
            )
            raw = denormalize_components(
                normalized,
                shared_mean,
                shared_scale,
            )
            return shared_calibrator(raw).cpu()[0]

    shared_anti = anti_shortcut_report(
        examples,
        shared_predict,
        shared_scale,
        expect_degradation_sensitivity=(
            selected_candidate == "output_phase_tfgrid"
        ),
    )
    gradients = gradient_smokes(
        generator,
        config,
        selected_head,
        selected_candidate,
        shared_mean,
        shared_scale,
        shared_calibrator,
        waveform_predictor,
        waveform_mean,
        waveform_scale,
        waveform_calibrator,
        examples,
        device,
    )
    shared_segment_transfer = training_segment_transfer_report(
        examples,
        shared_predict,
        shared_scale,
        (
            "own_target"
            if selected_candidate == "output_phase_tfgrid"
            else "clean_target"
        ),
    )
    surrogate_speaker_ids = {example.speaker_id for example in examples}
    independent_anti_by_architecture: dict[str, dict[str, Any]] = {}
    independent_gradient_by_architecture: dict[str, dict[str, Any]] = {}
    independent_segment_transfer_by_architecture: dict[
        str, dict[str, Any]
    ] = {}
    independent_external_by_architecture: dict[str, dict[str, Any]] = {}
    independent_external_rows_by_architecture: dict[
        str, list[dict[str, Any]]
    ] = {}
    independent_vctk_external_by_architecture: dict[
        str, dict[str, Any] | None
    ] = {}
    independent_vctk_rows_by_architecture: dict[
        str, list[dict[str, Any]]
    ] = {}
    independent_eligible_by_architecture: dict[str, list[str]] = {}
    independent_decision_by_architecture: dict[str, str] = {}
    for architecture, candidate_predictor in waveform_models.items():
        candidate_mean, candidate_scale = waveform_stats[architecture]
        candidate_calibrator = waveform_calibrators[architecture]
        freeze_module(candidate_predictor)

        def candidate_waveform_predict(
            waveform: torch.Tensor,
            current_predictor: torch.nn.Module = candidate_predictor,
            current_mean: torch.Tensor = candidate_mean,
            current_scale: torch.Tensor = candidate_scale,
            current_calibrator: ComponentAffineCalibrator = candidate_calibrator,
        ) -> torch.Tensor:
            with torch.inference_mode():
                normalized = current_predictor(waveform.to(device))
                raw = denormalize_components(
                    normalized,
                    current_mean,
                    current_scale,
                )
                return current_calibrator(raw).cpu()[0]

        candidate_anti = anti_shortcut_report(
            examples,
            candidate_waveform_predict,
            candidate_scale,
            expect_degradation_sensitivity=True,
        )
        candidate_segment_transfer = training_segment_transfer_report(
            examples,
            candidate_waveform_predict,
            candidate_scale,
            "own_target",
        )
        if architecture == selected_architecture:
            candidate_gradient = gradients["frozen_independent_predictor"]
        else:
            candidate_gradient = independent_gradient_smoke(
                generator,
                config,
                candidate_predictor,
                candidate_mean,
                candidate_scale,
                candidate_calibrator,
                examples,
                device,
            )
        candidate_external, candidate_external_rows = external_stress_test(
            args.external_exact_csv,
            candidate_predictor,
            candidate_mean,
            candidate_scale,
            candidate_calibrator,
            surrogate_speaker_ids,
            device,
        )
        independent_external_by_architecture[architecture] = candidate_external
        independent_external_rows_by_architecture[architecture] = (
            candidate_external_rows
        )
        if uses_vctk_external:
            candidate_vctk, candidate_vctk_rows = vctk_external_test(
                args.vctk_external_label_bank,
                candidate_waveform_predict,
                candidate_scale,
                surrogate_speaker_ids,
            )
        else:
            candidate_vctk = None
            candidate_vctk_rows = []
        independent_vctk_external_by_architecture[architecture] = candidate_vctk
        independent_vctk_rows_by_architecture[architecture] = (
            candidate_vctk_rows
        )
        candidate_metrics = waveform_architecture_reports[architecture][
            "calibrated"
        ]
        candidate_eligible = eligible_components(
            candidate_metrics,
            candidate_external,
            candidate_anti,
            candidate_gradient,
            candidate_segment_transfer,
            candidate_vctk,
        )
        independent_anti_by_architecture[architecture] = candidate_anti
        independent_gradient_by_architecture[architecture] = candidate_gradient
        independent_segment_transfer_by_architecture[architecture] = (
            candidate_segment_transfer
        )
        independent_eligible_by_architecture[architecture] = candidate_eligible
        independent_decision_by_architecture[architecture] = (
            "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
            if route_has_minimum_component_coverage(candidate_eligible)
            else "NO_GO_GENERATOR_TRAINING"
        )
    independent_anti = independent_anti_by_architecture[selected_architecture]
    independent_segment_transfer = (
        independent_segment_transfer_by_architecture[selected_architecture]
    )
    independent_external = independent_external_by_architecture[
        selected_architecture
    ]
    independent_external_rows = independent_external_rows_by_architecture[
        selected_architecture
    ]
    independent_vctk_external = independent_vctk_external_by_architecture[
        selected_architecture
    ]
    independent_vctk_external_rows = independent_vctk_rows_by_architecture[
        selected_architecture
    ]
    shared_external, shared_external_rows = shared_external_stress_test(
        args.external_exact_csv,
        generator,
        config,
        selected_head,
        selected_candidate,
        shared_mean,
        shared_scale,
        shared_calibrator,
        surrogate_speaker_ids,
        device,
    )
    if uses_vctk_external:
        shared_vctk_external, shared_vctk_external_rows = vctk_external_test(
            args.vctk_external_label_bank,
            shared_predict,
            shared_scale,
            surrogate_speaker_ids,
        )
    else:
        shared_vctk_external = None
        shared_vctk_external_rows = []

    shared_metrics = shared_candidate_reports[selected_candidate]["calibrated"]
    independent_metrics = waveform_architecture_reports[selected_architecture][
        "calibrated"
    ]
    shared_decision = route_decision(
        shared_metrics,
        shared_external,
        shared_anti,
        gradients["shared_dual_head"],
        shared_segment_transfer,
        shared_vctk_external,
    )
    independent_decision = independent_decision_by_architecture[
        selected_architecture
    ]
    shared_eligible_components = eligible_components(
        shared_metrics,
        shared_external,
        shared_anti,
        gradients["shared_dual_head"],
        shared_segment_transfer,
        shared_vctk_external,
    )
    independent_eligible_components = independent_eligible_by_architecture[
        selected_architecture
    ]
    if (
        independent_decision == "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
        and shared_decision != independent_decision
    ):
        conclusion = (
            "The frozen independent predictor met the six-component coverage "
            "rule and advances to multi-seed confirmation; the shared dual "
            "head remains ineligible."
        )
    elif (
        shared_decision == "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
        and independent_decision != shared_decision
    ):
        conclusion = (
            "The shared dual head met the six-component coverage rule and "
            "advances to multi-seed confirmation; the independent predictor "
            "remains ineligible."
        )
    elif (
        shared_decision
        == independent_decision
        == "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
    ):
        conclusion = (
            "Both routes met the six-component coverage rule and advance to "
            "multi-seed confirmation; no generator training starts yet."
        )
    else:
        conclusion = (
            "Neither route produced enough individually qualified AVQI "
            "components across two concept families; generator training remains blocked."
        )

    report = {
        "decision": "COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE",
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "plain_language_conclusion": conclusion,
        "contract": contract,
        "routes": {
            "shared_dual_head": {
                "selected_candidate": selected_candidate,
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": shared_training,
                "all_candidate_metrics": shared_candidate_reports,
                "metrics": shared_metrics,
                "calibration": {
                    "scale": shared_calibrator.scale.detach().cpu().tolist(),
                    "bias": shared_calibrator.bias.detach().cpu().tolist(),
                },
                "anti_shortcut": shared_anti,
                "training_segment_transfer": shared_segment_transfer,
                "gradient": gradients["shared_dual_head"],
                "external_clean_target_stress": shared_external,
                "vctk_external_own_target_stress": shared_vctk_external,
                "eligible_components": shared_eligible_components,
                "decision": shared_decision,
                "interpretation_limit": (
                    "head accuracy does not prove exact components of the final waveform improve"
                ),
            },
            "frozen_independent_predictor": {
                "selected_architecture": selected_architecture,
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": waveform_training,
                "all_architecture_metrics": waveform_architecture_reports,
                "metrics": independent_metrics,
                "calibration": {
                    "scale": waveform_calibrator.scale.detach().cpu().tolist(),
                    "bias": waveform_calibrator.bias.detach().cpu().tolist(),
                },
                "anti_shortcut": independent_anti,
                "training_segment_transfer": independent_segment_transfer,
                "gradient": gradients["frozen_independent_predictor"],
                "external_enhancement_stress": independent_external,
                "vctk_external_own_target_stress": independent_vctk_external,
                "external_evaluation_by_architecture": {
                    architecture: {
                        "pathology": independent_external_by_architecture[
                            architecture
                        ],
                        "vctk": independent_vctk_external_by_architecture[
                            architecture
                        ],
                    }
                    for architecture in waveform_models
                },
                "qualification_by_architecture": {
                    architecture: {
                        "anti_shortcut": independent_anti_by_architecture[
                            architecture
                        ],
                        "training_segment_transfer": (
                            independent_segment_transfer_by_architecture[
                                architecture
                            ]
                        ),
                        "gradient": independent_gradient_by_architecture[
                            architecture
                        ],
                        "eligible_components": (
                            independent_eligible_by_architecture[architecture]
                        ),
                        "decision": independent_decision_by_architecture[
                            architecture
                        ],
                    }
                    for architecture in waveform_models
                },
                "eligible_components": independent_eligible_components,
                "decision": independent_decision,
            },
        },
        "coverage": label_bank_coverage,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    write_json(args.output_dir / "diagnostic_report.json", report)
    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows(
            examples,
            {
                **{
                    f"shared_{name}": prediction
                    for name, prediction in shared_predictions.items()
                },
                **{
                    f"independent_{name}": prediction
                    for name, prediction in waveform_predictions.items()
                },
            },
        ),
    )
    write_csv(
        args.output_dir / "external_independent_predictions.csv",
        independent_external_rows,
    )
    for architecture, rows in independent_external_rows_by_architecture.items():
        write_csv(
            args.output_dir
            / f"external_independent_{architecture}_predictions.csv",
            rows,
        )
    write_csv(
        args.output_dir / "external_shared_predictions.csv",
        shared_external_rows,
    )
    if uses_vctk_external:
        write_csv(
            args.output_dir / "vctk_external_independent_predictions.csv",
            independent_vctk_external_rows,
        )
        for architecture, rows in independent_vctk_rows_by_architecture.items():
            write_csv(
                args.output_dir
                / f"vctk_external_independent_{architecture}_predictions.csv",
                rows,
            )
        write_csv(
            args.output_dir / "vctk_external_shared_predictions.csv",
            shared_vctk_external_rows,
        )
    (args.output_dir / "SUMMARY.md").write_text(
        human_summary(report), encoding="utf-8"
    )
    receipt = {
        "decision": report["decision"],
        "shared_route": shared_decision,
        "independent_route": independent_decision,
        "selected_shared_candidate": selected_candidate,
        "selected_independent_architecture": selected_architecture,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "experiment_contract.json",
                "diagnostic_report.json",
                "predictions.csv",
                "external_independent_predictions.csv",
                "external_shared_predictions.csv",
                "SUMMARY.md",
                *(
                    (
                        "vctk_external_independent_predictions.csv",
                        "vctk_external_shared_predictions.csv",
                    )
                    if uses_vctk_external
                    else ()
                ),
                *tuple(
                    f"external_independent_{architecture}_predictions.csv"
                    for architecture in waveform_models
                ),
                *(
                    tuple(
                        "vctk_external_independent_"
                        f"{architecture}_predictions.csv"
                        for architecture in waveform_models
                    )
                    if uses_vctk_external
                    else ()
                ),
            )
        },
        "checkpoint_sha256": {
            path.name: sha256_file(path)
            for path in sorted(args.checkpoint_dir.glob("*.pt"))
        },
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
