#!/usr/bin/env python3
"""Compare CPPS candidates on frozen train and calibration rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator
from scripts.evaluate_avqi_component_backprop import load_examples, target_stats


EXPECTED_SPLIT_SPEAKERS = {
    "surrogate_train": 197,
    "surrogate_calibration": 26,
    "surrogate_holdout": 26,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--modes",
        default=(
            "praat_relative_log1p_v10,praat_pow2_highpass_v11,"
            "praat_view_input_v12"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def affine_fit(estimate: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    estimate_centered = estimate - estimate.mean()
    target_centered = target - target.mean()
    variance = estimate_centered.square().mean().clamp_min(1e-8)
    scale = ((estimate_centered * target_centered).mean() / variance).clamp_min(
        1e-4
    )
    bias = target.mean() - scale * estimate.mean()
    return float(scale), float(bias)


def safe_spearman(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    value = float(stats.spearmanr(reference, estimate).statistic)
    return value if math.isfinite(value) else None


def metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    train_scale: float,
) -> dict[str, Any]:
    truth = reference.detach().cpu().numpy()
    prediction = estimate.detach().cpu().numpy()
    error = prediction - truth
    variance = float(np.sum((truth - truth.mean()) ** 2))
    slope = float(
        np.sum((truth - truth.mean()) * (prediction - prediction.mean()))
        / max(variance, 1e-12)
    )
    return {
        "rows": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "normalized_mae": float(np.mean(np.abs(error)) / train_scale),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "level_spearman": safe_spearman(truth, prediction),
        "calibration_slope": slope,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    actual_hash = sha256_file(args.label_bank)
    if actual_hash != args.label_bank_sha256:
        raise ValueError(
            f"label-bank hash mismatch: {actual_hash} != {args.label_bank_sha256}"
        )
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not modes:
        raise ValueError("at least one CPPS mode is required")
    device = torch.device(args.device)
    examples, coverage = load_examples(args.label_bank, EXPECTED_SPLIT_SPEAKERS)
    train_mean_tensor, train_scale_tensor = target_stats(
        examples,
        "own_target",
        device,
    )
    train_mean = float(train_mean_tensor[0].cpu())
    train_scale = float(train_scale_tensor[0].cpu())
    split_indices = {
        split: [
            index
            for index, example in enumerate(examples)
            if example.split == split
        ]
        for split in ("surrogate_train", "surrogate_calibration")
    }
    targets = torch.stack([example.own_target[0] for example in examples]).to(device)
    reports: dict[str, Any] = {}
    for mode in modes:
        estimator = PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            cpps_mode=mode,
            cpps_power_floor=1e-6,
        ).to(device)
        values: list[tuple[int, torch.Tensor]] = []
        with torch.inference_mode():
            for index, example in enumerate(examples):
                if example.split == "surrogate_holdout":
                    continue
                values.append(
                    (
                        index,
                        estimator.raw_cpps(
                            example.waveform.to(device),
                            speaking_type=example.view,
                        )[0],
                    )
                )
                if len(values) % 100 == 0:
                    print(f"mode={mode} rows={len(values)}", flush=True)
        row_indices = [index for index, _ in values]
        raw = torch.stack([value for _, value in values])
        target = targets[row_indices]
        positions = {
            row_index: position
            for position, row_index in enumerate(row_indices)
        }
        train_positions = [
            positions[index] for index in split_indices["surrogate_train"]
        ]
        calibration_positions = [
            positions[index]
            for index in split_indices["surrogate_calibration"]
        ]
        normalized_train_target = (
            target[train_positions] - train_mean
        ) / train_scale
        alignment_scale, alignment_bias = affine_fit(
            raw[train_positions],
            normalized_train_target,
        )
        aligned_normalized = raw * alignment_scale + alignment_bias
        aligned = aligned_normalized * train_scale + train_mean
        calibration_scale, calibration_bias = affine_fit(
            aligned[calibration_positions],
            target[calibration_positions],
        )
        calibrated = aligned * calibration_scale + calibration_bias
        normalized_error = (
            aligned[calibration_positions] - target[calibration_positions]
        ) / train_scale
        calibration_loss = float(
            torch.nn.functional.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
            ).cpu()
        )
        calibration_examples = [
            examples[index]
            for index in split_indices["surrogate_calibration"]
        ]
        per_view = {}
        for view in ("cs", "sv"):
            selected = [
                calibration_positions[index]
                for index, example in enumerate(calibration_examples)
                if example.view == view
            ]
            per_view[view] = metrics(
                target[selected],
                calibrated[selected],
                train_scale,
            )
        reports[mode] = {
            "alignment": {
                "scale": alignment_scale,
                "bias": alignment_bias,
            },
            "calibrator": {
                "scale": calibration_scale,
                "bias": calibration_bias,
            },
            "best_calibration_loss": calibration_loss,
            "raw_calibration": metrics(
                target[calibration_positions],
                raw[calibration_positions],
                train_scale,
            ),
            "aligned_calibration": metrics(
                target[calibration_positions],
                aligned[calibration_positions],
                train_scale,
            ),
            "calibrated_calibration": metrics(
                target[calibration_positions],
                calibrated[calibration_positions],
                train_scale,
            ),
            "calibrated_by_view": per_view,
        }
    selected_mode = min(
        reports,
        key=lambda mode: reports[mode]["best_calibration_loss"],
    )
    output = {
        "schema_version": "cpps-route-c-calibration-audit-v1",
        "decision": "CALIBRATION_ONLY_CANDIDATE_SELECTED",
        "selected_mode": selected_mode,
        "evaluated_splits": ["surrogate_train", "surrogate_calibration"],
        "holdout_evaluated": False,
        "external_evaluated": False,
        "label_bank": str(args.label_bank),
        "label_bank_sha256": actual_hash,
        "source_commit": args.source_commit,
        "coverage": {
            "total_rows": coverage["total_rows"],
            "usable_rows": coverage["usable_rows"],
            "fraction": coverage["fraction"],
            "split_speakers": coverage["split_speakers"],
        },
        "train_cpps_mean": train_mean,
        "train_cpps_scale": train_scale,
        "modes": reports,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
