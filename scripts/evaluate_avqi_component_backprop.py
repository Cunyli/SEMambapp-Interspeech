#!/usr/bin/env python3
"""Speaker-disjoint diagnostic for two AVQI-component backpropagation routes.

This script deliberately stops before generator optimization. It compares two
small mechanisms, evaluates all six AVQI v03.01 terms, and verifies the intended
gradient paths:

1. shared SeMamba++ heads at late-backbone and enhanced-spectral points;
2. separately trained waveform predictors that are frozen before backprop.

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
    ComponentAffineCalibrator,
    FrequencyAwareSharedComponentHead,
    FrequencyAwareWaveformComponentPredictor,
    SharedComponentHead,
    WaveformComponentPredictor,
    denormalize_components,
    freeze_module,
    pool_frequency_aware_shared_feature_map,
    pool_shared_feature_map,
    standardized_component_loss,
)
from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft
from utils import load_config


SAMPLE_RATE = 16_000
EXPECTED_ROWS = 390
EXPECTED_SPLIT_SPEAKERS = {
    "surrogate_train": 70,
    "surrogate_calibration": 14,
    "surrogate_holdout": 14,
}
PRIMARY_GATE_COMPONENTS = ("cpps", "hnr")
LEVEL_SPEARMAN_GATE = 0.70
DELTA_SPEARMAN_GATE = 0.60
NORMALIZED_MAE_GATE = 0.50
CALIBRATION_SLOPE_RANGE = (0.75, 1.25)
EXTERNAL_CANDIDATES = ("B0_250", "S3_500", "S3_2000")
EXTERNAL_CONDITIONS = (
    "clean",
    "rir_only",
    "snr30",
    "snr20",
    "snr15",
    "snr10",
)
FREQUENCY_BINS = 8
EXPECTED_EXTERNAL_SPEAKERS = 24
EXTERNAL_REQUIRED_SLICES = (
    "view=cs",
    "view=sv",
    "label=patient",
    "condition=snr10",
)


@dataclass(frozen=True)
class Example:
    speaker_id: str
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--shared-head-epochs", type=int, default=250)
    parser.add_argument("--waveform-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


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


def load_waveform(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"invalid 16 kHz mono audio: {path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite audio: {path}")
    return waveform


def load_examples(label_bank: Path) -> list[Example]:
    with label_bank.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    task_rows = [
        row
        for row in rows
        if row["view"] in {"cs", "sv"} and row["scoring_status"] == "ok"
    ]
    if len(task_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} usable rows, found {len(task_rows)}")
    split_speakers = {
        split: len({row["speaker_id"] for row in task_rows if row["split"] == split})
        for split in EXPECTED_SPLIT_SPEAKERS
    }
    if split_speakers != EXPECTED_SPLIT_SPEAKERS:
        raise ValueError(f"speaker split mismatch: {split_speakers}")
    speaker_sets = {
        split: {row["speaker_id"] for row in task_rows if row["split"] == split}
        for split in EXPECTED_SPLIT_SPEAKERS
    }
    for first, first_speakers in speaker_sets.items():
        for second, second_speakers in speaker_sets.items():
            if first < second and first_speakers & second_speakers:
                raise ValueError(f"speaker leakage between {first} and {second}")
    clean_targets = {
        (row["speaker_id"], row["view"]): component_tensor(row)
        for row in task_rows
        if row["condition_id"] == "clean"
    }
    examples: list[Example] = []
    for index, row in enumerate(task_rows, start=1):
        view = row["view"]
        path = Path(row[f"{view}_path"])
        expected_hash = row[f"{view}_sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"audio hash mismatch: {path}")
        key = (row["speaker_id"], view)
        if key not in clean_targets:
            raise ValueError(f"missing clean target for {key}")
        examples.append(
            Example(
                speaker_id=row["speaker_id"],
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
    return examples


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
    return {
        "encoder": encoder,
        "late": shared,
        "enhanced_spectral": enhanced_spectral,
    }


def pool_shared_candidate(
    feature_maps: dict[str, torch.Tensor],
    candidate: str,
) -> torch.Tensor:
    if candidate == "late_global":
        return pool_shared_feature_map(feature_maps["late"])
    if candidate == "late_frequency":
        return pool_frequency_aware_shared_feature_map(
            feature_maps["late"],
            frequency_bins=FREQUENCY_BINS,
        )
    if candidate == "enhanced_spectral":
        return pool_frequency_aware_shared_feature_map(
            feature_maps["enhanced_spectral"],
            frequency_bins=FREQUENCY_BINS,
        )
    raise ValueError(f"unknown shared candidate: {candidate}")


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
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {
        "late_global": [],
        "late_frequency": [],
        "enhanced_spectral": [],
    }
    with torch.no_grad():
        for index, example in enumerate(examples, start=1):
            maps = shared_feature_maps(model, example.waveform.to(device), config)
            for candidate in rows:
                rows[candidate].append(
                    pool_shared_candidate(maps, candidate).cpu()[0]
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
) -> tuple[
    SharedComponentHead | FrequencyAwareSharedComponentHead,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    target_mean, target_scale = target_stats(examples, "clean_target", device)
    features = pooled_features.to(device)
    if candidate == "late_global":
        head: SharedComponentHead | FrequencyAwareSharedComponentHead = (
            SharedComponentHead(feature_channels=pooled_features.shape[1] // 2)
        )
    else:
        feature_channels = pooled_features.shape[1] // (FREQUENCY_BINS * 2)
        head = FrequencyAwareSharedComponentHead(
            feature_channels=feature_channels,
            frequency_bins=FREQUENCY_BINS,
        )
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    train_indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_train"
    ]
    train_targets = torch.stack(
        [example.clean_target for example in examples]
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        for start in range(0, len(order), 32):
            stop = min(start + 32, len(order))
            batch = [
                train_indices[order[position]] for position in range(start, stop)
            ]
            prediction = head.forward_pooled(features[batch])
            loss = standardized_component_loss(
                prediction,
                train_targets[batch],
                target_mean,
                target_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        value = calibration_loss(
            head,
            lambda current, index: current.forward_pooled(features[index : index + 1]),
            examples,
            "clean_target",
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
        if epoch >= 20 and stale >= patience:
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
) -> tuple[
    WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    target_mean, target_scale = target_stats(examples, "own_target", device)
    if architecture == "global_stats":
        predictor: WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor = (
            WaveformComponentPredictor()
        )
    elif architecture == "frequency_aware":
        predictor = FrequencyAwareWaveformComponentPredictor(
            frequency_bins=FREQUENCY_BINS
        )
    else:
        raise ValueError(f"unknown waveform architecture: {architecture}")
    predictor = predictor.to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=1e-4)
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
    history: list[dict[str, float]] = []

    def predict(current: torch.nn.Module, index: int) -> torch.Tensor:
        return current(examples[index].waveform.to(device))

    for epoch in range(1, epochs + 1):
        predictor.train()
        order = list(train_indices)
        generator.shuffle(order)
        epoch_losses = []
        for index in order:
            prediction = predictor(examples[index].waveform.to(device))
            target = examples[index].own_target.to(device).unsqueeze(0)
            loss = standardized_component_loss(
                prediction, target, target_mean, target_scale
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
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
        if epoch >= 10 and stale >= patience:
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
            "history": history,
        },
        target_mean,
        target_scale,
    )


def predict_shared(
    head: SharedComponentHead | FrequencyAwareSharedComponentHead,
    pooled_features: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        normalized = head.forward_pooled(pooled_features.to(device))
        return denormalize_components(normalized, target_mean, target_scale).cpu()


def predict_waveforms(
    predictor: WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor,
    examples: list[Example],
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    predictions = []
    predictor.eval()
    with torch.inference_mode():
        for example in examples:
            normalized = predictor(example.waveform.to(device))
            predictions.append(
                denormalize_components(normalized, target_mean, target_scale).cpu()[0]
            )
    return torch.stack(predictions)


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
    scale = (estimate_centered * target_centered).mean(dim=0) / variance
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
    grouped: dict[tuple[str, str], dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for index, example in holdout:
        grouped.setdefault((example.speaker_id, example.view), {})[example.condition] = (
            example.own_target,
            predictions[index],
        )
    exact_deltas = []
    predicted_deltas = []
    for conditions in grouped.values():
        if set(conditions) != {"clean", "aug16k_phone"}:
            raise ValueError("holdout clean/phone pairing failed")
        exact_deltas.append(conditions["aug16k_phone"][0] - conditions["clean"][0])
        predicted_deltas.append(
            conditions["aug16k_phone"][1] - conditions["clean"][1]
        )
    exact = torch.stack(exact_deltas).numpy()
    predicted = torch.stack(predicted_deltas).numpy()
    return {
        name: safe_spearman(exact[:, index], predicted[:, index])
        for index, name in enumerate(AVQI_COMPONENT_NAMES)
    }


def route_metrics(
    examples: list[Example],
    predictions: torch.Tensor,
    target_attribute: str,
    train_scale: torch.Tensor,
    primary_filter: Callable[[Example], bool],
    include_delta_gate: bool,
) -> dict[str, Any]:
    indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "surrogate_holdout" and primary_filter(example)
    ]
    reference = torch.stack([getattr(examples[index], target_attribute) for index in indices])
    delta = paired_delta_spearman(examples, predictions) if include_delta_gate else None
    primary = component_metrics(
        reference,
        predictions[indices],
        train_scale,
        include_delta_gate=include_delta_gate,
        delta_spearman=delta,
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
    return {"primary": primary, "paired_delta_spearman": delta, "slices": slices}


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
    shifted = torch.nn.functional.pad(waveform[:-shift], (shift, 0))
    time = torch.arange(waveform.numel(), device=waveform.device) / SAMPLE_RATE
    tone = torch.sin(2.0 * math.pi * 150.0 * time)
    tone = tone * rms / tone.square().mean().sqrt().clamp_min(1e-8)
    return {
        "clean": waveform,
        "gain_minus12db": waveform * 0.25,
        "time_shift_100ms": shifted,
        "noise_10db": waveform + noise,
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
        }
        if expect_degradation_sensitivity:
            gates.update(
                {
                    "noise_moves_away": values["noise_10db"] >= 0.10,
                    "lowpass_moves_away": values["lowpass_3khz"] >= 0.10,
                    "gain_less_than_noise": (
                        values["gain_minus12db"] < values["noise_10db"]
                    ),
                    "shift_less_than_noise": (
                        values["time_shift_100ms"] < values["noise_10db"]
                    ),
                }
            )
        else:
            gates.update(
                {
                    "gain_nearly_invariant": values["gain_minus12db"] <= 0.10,
                    "shift_nearly_invariant": values["time_shift_100ms"] <= 0.10,
                }
            )
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


def fixed_segment(waveform: torch.Tensor, samples: int = 48_000) -> torch.Tensor:
    if waveform.numel() >= samples:
        return waveform[:samples]
    return torch.nn.functional.pad(waveform, (0, samples - waveform.numel()))


def gradient_smokes(
    generator: SEMambapp,
    config: dict[str, Any],
    selected_head: SharedComponentHead | FrequencyAwareSharedComponentHead,
    selected_candidate: str,
    shared_mean: torch.Tensor,
    shared_scale: torch.Tensor,
    shared_calibrator: ComponentAffineCalibrator,
    waveform_predictor: WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor,
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
    maps = shared_feature_maps(generator, waveform, config)
    shared_pooled = pool_shared_candidate(maps, selected_candidate)
    shared_prediction = selected_head.forward_pooled(shared_pooled)
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
    shared_decoder_norm = gradient_norm(decoder_parameters)
    shared_decoder_path_valid = (
        shared_decoder_norm > 1e-8
        if selected_candidate == "enhanced_spectral"
        else shared_decoder_norm == 0.0
    )
    shared_finite = all(
        math.isfinite(value)
        for value in (shared_backbone_norm, shared_head_norm, shared_decoder_norm)
    )

    generator.zero_grad(set_to_none=True)
    waveform_predictor.zero_grad(set_to_none=True)
    freeze_module(waveform_predictor)
    enhanced = enhance_waveform(generator, waveform, config)
    normalized_prediction = waveform_predictor(enhanced)
    normalized_prediction = calibrated_normalized_prediction(
        normalized_prediction,
        waveform_mean,
        waveform_scale,
        waveform_calibrator,
    )
    independent_loss = standardized_component_loss(
        normalized_prediction,
        clean_target,
        waveform_mean,
        waveform_scale,
    )
    independent_loss.backward()
    independent_backbone_norm = gradient_norm(backbone_parameters)
    independent_decoder_norm = gradient_norm(decoder_parameters)
    predictor_gradients_absent = all(
        parameter.grad is None for parameter in waveform_predictor.parameters()
    )
    independent_finite = all(
        math.isfinite(value)
        for value in (independent_backbone_norm, independent_decoder_norm)
    )
    return {
        "shared_dual_head": {
            "loss": float(shared_loss.detach().cpu()),
            "backbone_gradient_norm": shared_backbone_norm,
            "head_gradient_norm": shared_head_norm,
            "decoder_gradient_norm": shared_decoder_norm,
            "decoder_gradient_expected": (
                "nonzero"
                if selected_candidate == "enhanced_spectral"
                else "zero"
            ),
            "decision": (
                "PASS"
                if shared_finite
                and shared_backbone_norm > 1e-8
                and shared_head_norm > 1e-8
                and shared_decoder_path_valid
                else "FAIL"
            ),
        },
        "frozen_independent_predictor": {
            "loss": float(independent_loss.detach().cpu()),
            "backbone_gradient_norm": independent_backbone_norm,
            "decoder_gradient_norm": independent_decoder_norm,
            "predictor_gradients_absent": predictor_gradients_absent,
            "decision": (
                "PASS"
                if independent_finite
                and independent_backbone_norm > 1e-8
                and independent_decoder_norm > 1e-8
                and predictor_gradients_absent
                else "FAIL"
            ),
        },
    }


def external_stress_test(
    csv_path: Path,
    predictor: WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
    forbidden_speaker_ids: set[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["source_type"] == "enhanced"
        and row["candidate"] in EXTERNAL_CANDIDATES
        and row["condition"] in EXTERNAL_CONDITIONS
        and row["view"] in {"cs", "sv"}
        and row["scoring_status"] == "ok"
    ]
    speaker_ids = {row["speaker_id"] for row in selected}
    overlap = speaker_ids & forbidden_speaker_ids
    if overlap:
        raise ValueError(f"external speaker leakage: {sorted(overlap)}")
    if len(speaker_ids) != EXPECTED_EXTERNAL_SPEAKERS:
        raise ValueError(
            f"expected {EXPECTED_EXTERNAL_SPEAKERS} external speakers, "
            f"found {len(speaker_ids)}"
        )
    expected_rows = (
        EXPECTED_EXTERNAL_SPEAKERS
        * 2
        * len(EXTERNAL_CONDITIONS)
        * len(EXTERNAL_CANDIDATES)
    )
    if len(selected) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} external enhanced rows, found {len(selected)}"
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
    overall = component_metrics(reference_tensor, prediction_tensor, target_scale)
    slices: dict[str, Any] = {}
    for field, values in {
        "candidate": EXTERNAL_CANDIDATES,
        "condition": EXTERNAL_CONDITIONS,
        "view": ("cs", "sv"),
        "label": ("healthy", "patient"),
    }.items():
        for value in values:
            indices = [i for i, row in enumerate(selected) if row[field] == value]
            if len(indices) >= 4:
                slices[f"{field}={value}"] = component_metrics(
                    reference_tensor[indices], prediction_tensor[indices], target_scale
                )
    report = {
        "rows": len(selected),
        "speaker_count": len(speaker_ids),
        "speaker_overlap_with_surrogate": 0,
        "overall": overall,
        "slices": slices,
    }
    return report, output_rows


def shared_external_stress_test(
    csv_path: Path,
    generator: SEMambapp,
    config: dict[str, Any],
    head: SharedComponentHead | FrequencyAwareSharedComponentHead,
    candidate: str,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    calibrator: ComponentAffineCalibrator,
    forbidden_speaker_ids: set[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["source_type"] == "input"
        and row["condition"] in EXTERNAL_CONDITIONS
        and row["view"] in {"cs", "sv"}
        and row["scoring_status"] == "ok"
    ]
    speaker_ids = {row["speaker_id"] for row in selected}
    overlap = speaker_ids & forbidden_speaker_ids
    if overlap:
        raise ValueError(f"external speaker leakage: {sorted(overlap)}")
    if len(speaker_ids) != EXPECTED_EXTERNAL_SPEAKERS:
        raise ValueError(
            f"expected {EXPECTED_EXTERNAL_SPEAKERS} external speakers, "
            f"found {len(speaker_ids)}"
        )
    expected_rows = EXPECTED_EXTERNAL_SPEAKERS * 2 * len(EXTERNAL_CONDITIONS)
    if len(selected) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} shared external rows, found {len(selected)}"
        )
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
            normalized = head.forward_pooled(pooled)
            raw_prediction = denormalize_components(
                normalized,
                target_mean,
                target_scale,
            )
            prediction = calibrator(raw_prediction).cpu()[0]
            reference = component_tensor(row, prefix="clean_")
            predictions.append(prediction)
            references.append(reference)
            item: dict[str, Any] = {
                "condition": row["condition"],
                "speaker_id": row["speaker_id"],
                "view": row["view"],
                "label": row["label"],
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
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
    overall = component_metrics(reference_tensor, prediction_tensor, target_scale)
    slices: dict[str, Any] = {}
    for field, values in {
        "condition": EXTERNAL_CONDITIONS,
        "view": ("cs", "sv"),
        "label": ("healthy", "patient"),
    }.items():
        for value in values:
            indices = [i for i, row in enumerate(selected) if row[field] == value]
            if len(indices) >= 4:
                slices[f"{field}={value}"] = component_metrics(
                    reference_tensor[indices],
                    prediction_tensor[indices],
                    target_scale,
                )
    report = {
        "rows": len(selected),
        "speaker_count": len(speaker_ids),
        "speaker_overlap_with_surrogate": 0,
        "overall": overall,
        "slices": slices,
    }
    return report, output_rows


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
    if report["overall"][component]["decision"] != "PASS":
        return False
    return all(
        report["slices"][slice_name][component]["decision"] == "PASS"
        for slice_name in EXTERNAL_REQUIRED_SLICES
    )


def external_primary_gate_passed(report: dict[str, Any]) -> bool:
    return all(
        external_component_passes(report, component)
        for component in PRIMARY_GATE_COMPONENTS
    )


def route_decision(
    metrics: dict[str, Any],
    external_metrics: dict[str, Any],
    anti_shortcut: dict[str, Any],
    gradient: dict[str, Any],
) -> str:
    primary_metrics = all(
        component_passes(metrics, name) for name in PRIMARY_GATE_COMPONENTS
    )
    primary_external = external_primary_gate_passed(external_metrics)
    primary_anti = all(
        anti_shortcut["components"][name]["decision"] == "PASS"
        for name in PRIMARY_GATE_COMPONENTS
    )
    return (
        "ELIGIBLE_FOR_BOUNDED_PILOT"
        if primary_metrics
        and primary_external
        and primary_anti
        and gradient["decision"] == "PASS"
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
        "| Route | Chosen form | CPPS gate | HNR gate | External | Gradient | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
        (
            f"| Shared dual head | {shared['selected_candidate']} | "
            f"{shared['metrics']['primary']['cpps']['decision']} | "
            f"{shared['metrics']['primary']['hnr']['decision']} | "
            f"{shared['external_primary_gate_passed']} | "
            f"{shared['gradient']['decision']} | {shared['decision']} |"
        ),
        (
            f"| Frozen independent predictor | {independent['selected_architecture']} | "
            f"{independent['metrics']['primary']['cpps']['decision']} | "
            f"{independent['metrics']['primary']['hnr']['decision']} | "
            f"{independent['external_primary_gate_passed']} | "
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
        "schema_version": "avqi-component-backprop-v2",
        "purpose": "diagnostic_only_no_generator_update",
        "components": list(AVQI_COMPONENT_NAMES),
        "component_loss_weights": dict(
            zip(AVQI_COMPONENT_NAMES, AVQI_COMPONENT_LOSS_WEIGHTS, strict=True)
        ),
        "primary_gate_components": list(PRIMARY_GATE_COMPONENTS),
        "jitter_in_primary_task": False,
        "speaker_split": EXPECTED_SPLIT_SPEAKERS,
        "routes": {
            "shared_dual_head": {
                "candidates": [
                    "late_global",
                    "late_frequency",
                    "enhanced_spectral",
                ],
                "target": "same-speaker clean exact Praat components",
            },
            "frozen_independent_predictor": {
                "architectures": ["global_stats", "frequency_aware"],
                "target": "input waveform exact Praat components",
            },
        },
        "calibration": {
            "method": "per-component affine",
            "fit_split": "surrogate_calibration",
            "holdout_used_for_fit_or_selection": False,
        },
        "gates": {
            "level_spearman": LEVEL_SPEARMAN_GATE,
            "paired_delta_spearman": DELTA_SPEARMAN_GATE,
            "normalized_mae": NORMALIZED_MAE_GATE,
            "calibration_slope": list(CALIBRATION_SLOPE_RANGE),
            "required_external_slices": list(EXTERNAL_REQUIRED_SLICES),
        },
        "source_sha256": {
            "label_bank": args.label_bank_sha256,
            "config": args.config_sha256,
            "generator_checkpoint": args.checkpoint_sha256,
            "external_exact_csv": args.external_exact_csv_sha256,
        },
        "source_commit": args.source_commit,
        "artifact_layout": {
            "run_output_dir": str(args.output_dir.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        },
    }
    write_json(args.output_dir / "experiment_contract.json", contract)

    examples = load_examples(args.label_bank)
    config = load_config(args.config)
    generator = load_generator(config, args.checkpoint, device)
    pooled = extract_shared_features(generator, examples, config, device)
    torch.save(pooled, args.output_dir / "shared_features.pt")

    shared_models: dict[
        str, SharedComponentHead | FrequencyAwareSharedComponentHead
    ] = {}
    shared_training: dict[str, Any] = {}
    shared_raw_predictions: dict[str, torch.Tensor] = {}
    shared_predictions: dict[str, torch.Tensor] = {}
    shared_calibrators: dict[str, ComponentAffineCalibrator] = {}
    shared_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for candidate_index, candidate in enumerate(pooled):
        head, training, mean, scale = train_shared_head(
            pooled[candidate],
            examples,
            device,
            args.shared_head_epochs,
            args.patience,
            args.seed + candidate_index,
            candidate,
        )
        shared_models[candidate] = head
        shared_training[candidate] = training
        shared_stats[candidate] = (mean, scale)
        raw_predictions = predict_shared(
            head,
            pooled[candidate],
            mean,
            scale,
            device,
        )
        calibrator = fit_component_calibrator(
            examples,
            raw_predictions,
            "clean_target",
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
            },
            args.checkpoint_dir / f"shared_{candidate}_head.pt",
        )
    selected_candidate = min(
        shared_training,
        key=lambda name: shared_training[name]["best_calibration_loss"],
    )

    waveform_models: dict[
        str, WaveformComponentPredictor | FrequencyAwareWaveformComponentPredictor
    ] = {}
    waveform_training: dict[str, Any] = {}
    waveform_raw_predictions: dict[str, torch.Tensor] = {}
    waveform_predictions: dict[str, torch.Tensor] = {}
    waveform_calibrators: dict[str, ComponentAffineCalibrator] = {}
    waveform_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for architecture_index, architecture in enumerate(
        ("global_stats", "frequency_aware")
    ):
        predictor, training, mean, scale = train_waveform_predictor(
            examples,
            device,
            args.waveform_epochs,
            args.patience,
            args.seed + architecture_index,
            architecture,
        )
        raw_predictions = predict_waveforms(
            predictor,
            examples,
            mean,
            scale,
            device,
        )
        calibrator = fit_component_calibrator(
            examples,
            raw_predictions,
            "own_target",
            device,
        )
        waveform_models[architecture] = predictor
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
        shared_candidate_reports[candidate] = {
            "raw": route_metrics(
                examples,
                shared_raw_predictions[candidate],
                "clean_target",
                scale,
                primary_filter=lambda example: example.condition == "aug16k_phone",
                include_delta_gate=False,
            ),
            "calibrated": route_metrics(
                examples,
                shared_predictions[candidate],
                "clean_target",
                scale,
                primary_filter=lambda example: example.condition == "aug16k_phone",
                include_delta_gate=False,
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

    def independent_predict(waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            normalized = waveform_predictor(waveform.to(device))
            raw = denormalize_components(
                normalized,
                waveform_mean,
                waveform_scale,
            )
            return waveform_calibrator(raw).cpu()[0]

    def shared_predict(waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            maps = shared_feature_maps(generator, waveform.to(device), config)
            pooled_features = pool_shared_candidate(maps, selected_candidate)
            normalized = selected_head.forward_pooled(pooled_features)
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
        expect_degradation_sensitivity=False,
    )
    independent_anti = anti_shortcut_report(
        examples,
        independent_predict,
        waveform_scale,
        expect_degradation_sensitivity=True,
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
    surrogate_speaker_ids = {example.speaker_id for example in examples}
    independent_external, independent_external_rows = external_stress_test(
        args.external_exact_csv,
        waveform_predictor,
        waveform_mean,
        waveform_scale,
        waveform_calibrator,
        surrogate_speaker_ids,
        device,
    )
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

    shared_metrics = shared_candidate_reports[selected_candidate]["calibrated"]
    independent_metrics = waveform_architecture_reports[selected_architecture][
        "calibrated"
    ]
    shared_decision = route_decision(
        shared_metrics,
        shared_external,
        shared_anti,
        gradients["shared_dual_head"],
    )
    independent_decision = route_decision(
        independent_metrics,
        independent_external,
        independent_anti,
        gradients["frozen_independent_predictor"],
    )
    shared_external_primary_pass = external_primary_gate_passed(shared_external)
    independent_external_primary_pass = external_primary_gate_passed(
        independent_external
    )
    if (
        independent_decision == "ELIGIBLE_FOR_BOUNDED_PILOT"
        and shared_decision != independent_decision
    ):
        conclusion = (
            "The frozen independent predictor passed the pre-registered "
            "CPPS+HNR gates; the shared dual head remains ineligible."
        )
    elif (
        shared_decision == "ELIGIBLE_FOR_BOUNDED_PILOT"
        and independent_decision != shared_decision
    ):
        conclusion = (
            "The shared dual head passed the pre-registered CPPS+HNR gates; "
            "the independent predictor remains ineligible."
        )
    elif shared_decision == independent_decision == "ELIGIBLE_FOR_BOUNDED_PILOT":
        conclusion = (
            "Both routes passed the CPPS+HNR diagnostic and may enter one "
            "small, matched-budget generator comparison."
        )
    else:
        conclusion = (
            "Neither route passed every CPPS+HNR accuracy, external-domain, "
            "anti-shortcut, and gradient gate; generator training remains blocked."
        )

    report = {
        "decision": "COMPLETED_DIAGNOSTIC_NO_GENERATOR_UPDATE",
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
                "gradient": gradients["shared_dual_head"],
                "external_clean_target_stress": shared_external,
                "external_primary_gate_passed": shared_external_primary_pass,
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
                "gradient": gradients["frozen_independent_predictor"],
                "external_enhancement_stress": independent_external,
                "external_primary_gate_passed": independent_external_primary_pass,
                "decision": independent_decision,
            },
        },
        "coverage": {
            "usable_rows": len(examples),
            "expected_rows": EXPECTED_ROWS,
            "fraction": len(examples) / EXPECTED_ROWS,
        },
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
    write_csv(
        args.output_dir / "external_shared_predictions.csv",
        shared_external_rows,
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
