#!/usr/bin/env python3
"""Reproduce exact-label, legacy NumPy, and frozen Torch HNR parity.

The input prediction bank is the sealed non-final ``raw_cc_v3`` diagnostic.
All usable internal train/calibration/holdout rows are rescored by the legacy
NumPy pipeline.  No candidate, waveform, or generator parameter is updated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avqi_code import python_version as legacy_python


LEVEL_SPEARMAN_GATE = 0.70
NORMALIZED_MAE_GATE = 0.50
CALIBRATION_SLOPE_RANGE = (0.75, 1.25)
DELTA_SPEARMAN_GATE = 0.60
EXPECTED_SPLIT_ROWS = {
    "surrogate_train": 1_618,
    "surrogate_calibration": 244,
    "surrogate_holdout": 244,
}


@dataclass(frozen=True)
class ParityRow:
    speaker_id: str
    sample_id: str
    split: str
    condition: str
    view: str
    label: str
    sample_group: str
    audio_path: str
    audio_sha256: str
    exact_hnr: float
    torch_raw_hnr: float
    torch_frozen_calibrated_hnr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--torch-predictions", type=Path, required=True)
    parser.add_argument("--torch-predictions-sha256", required=True)
    parser.add_argument("--torch-report", type=Path, required=True)
    parser.add_argument("--torch-report-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash drift for {path}: {actual} != {expected}")
    return actual


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_key(row: dict[str, str] | ParityRow) -> tuple[str, str, str, str, str]:
    if isinstance(row, ParityRow):
        return (
            row.speaker_id,
            row.sample_id,
            row.split,
            row.condition,
            row.view,
        )
    return (
        row["speaker_id"],
        row.get("sample_id", "") or row.get("pair_id", "") or row["speaker_id"],
        row["split"],
        row["condition_id"],
        row["view"],
    )


def load_rows(label_bank: Path, predictions: Path) -> list[ParityRow]:
    label_rows = [
        row
        for row in read_csv(label_bank)
        if row["view"] in {"cs", "sv"} and row["scoring_status"] == "ok"
    ]
    label_by_key = {row_key(row): row for row in label_rows}
    if len(label_by_key) != len(label_rows):
        raise ValueError("duplicate HNR rows in exact label bank")

    prediction_rows = [
        row
        for row in read_csv(predictions)
        if row["panel"] == "internal_nonfinal"
    ]
    rows: list[ParityRow] = []
    for prediction in prediction_rows:
        key = (
            prediction["speaker_id"],
            prediction["sample_id"],
            prediction["split"],
            prediction["condition"],
            prediction["view"],
        )
        label = label_by_key.get(key)
        if label is None:
            raise ValueError(f"prediction row missing from exact bank: {key}")
        if prediction["audio_sha256"] != label[f"{prediction['view']}_sha256"]:
            raise ValueError(f"audio hash label mismatch: {key}")
        if abs(float(prediction["exact_hnr"]) - float(label["hnr"])) > 1e-12:
            raise ValueError(f"exact HNR drift in prediction bank: {key}")
        rows.append(
            ParityRow(
                speaker_id=prediction["speaker_id"],
                sample_id=prediction["sample_id"],
                split=prediction["split"],
                condition=prediction["condition"],
                view=prediction["view"],
                label=prediction["label"],
                sample_group=prediction["sample_group"],
                audio_path=prediction["audio_path"],
                audio_sha256=prediction["audio_sha256"],
                exact_hnr=float(prediction["exact_hnr"]),
                torch_raw_hnr=float(prediction["raw_cc_hard_v3_raw_hnr"]),
                torch_frozen_calibrated_hnr=float(
                    prediction["raw_cc_hard_v3_calibrated_hnr"]
                ),
            )
        )
    split_counts = {
        split: sum(row.split == split for row in rows) for split in EXPECTED_SPLIT_ROWS
    }
    if split_counts != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"unexpected parity rows: {split_counts}")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prediction rows")
    return rows


def _disable_legacy_filter_plot(*_: Any, **__: Any) -> None:
    """Remove only the legacy diagnostic plot side effect, not its filter math."""


legacy_python.visualize_highpass = _disable_legacy_filter_plot


def legacy_numpy_hnr(row: ParityRow) -> tuple[float, str]:
    path = Path(row.audio_path)
    if sha256_file(path) != row.audio_sha256:
        raise ValueError(f"audio hash mismatch: {path}")
    signal = legacy_python.read_and_resample_signal(path, 16_000)
    signal = legacy_python.highpass_filter(signal, 16_000)
    if row.view == "sv":
        metric_input = legacy_python.length_normalize_sv(signal, 16_000)
    elif row.view == "cs":
        metric_input = legacy_python.get_voiced_segments(signal, 16_000)
    else:
        raise ValueError(f"unsupported HNR view: {row.view}")
    value = float(legacy_python.get_hnr(metric_input, 16_000))
    return value, sha256_file(path)


def positive_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source_centered = source - source.mean()
    target_centered = target - target.mean()
    variance = max(float(np.mean(source_centered**2)), 1e-12)
    scale = max(
        float(np.mean(source_centered * target_centered)) / variance,
        1e-4,
    )
    bias = float(target.mean() - scale * source.mean())
    return scale, bias


def safe_spearman(reference: np.ndarray, estimate: np.ndarray) -> float:
    value = float(stats.spearmanr(reference, estimate).statistic)
    return value if math.isfinite(value) else -1.0


def metric_report(
    reference: np.ndarray,
    estimate: np.ndarray,
    train_scale: float,
) -> dict[str, Any]:
    finite = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[finite]
    estimate = estimate[finite]
    if reference.size < 2:
        raise ValueError("fewer than two finite HNR rows in metric slice")
    mae = float(np.mean(np.abs(estimate - reference)))
    variance = float(np.sum((reference - reference.mean()) ** 2))
    slope = float(
        np.sum((reference - reference.mean()) * (estimate - estimate.mean()))
        / max(variance, 1e-12)
    )
    spearman = safe_spearman(reference, estimate)
    normalized_mae = mae / max(train_scale, 1e-8)
    gates = {
        "level_spearman_ge_0_70": spearman >= LEVEL_SPEARMAN_GATE,
        "normalized_mae_le_0_50": normalized_mae <= NORMALIZED_MAE_GATE,
        "calibration_slope_0_75_to_1_25": (
            CALIBRATION_SLOPE_RANGE[0]
            <= slope
            <= CALIBRATION_SLOPE_RANGE[1]
        ),
    }
    return {
        "rows": int(reference.size),
        "level_spearman": spearman,
        "normalized_mae": normalized_mae,
        "calibration_slope": slope,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def paired_delta_report(
    rows: list[ParityRow],
    exact: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, Any]:
    clean: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(rows):
        if row.split == "surrogate_holdout" and row.condition == "clean":
            clean[(row.speaker_id, row.sample_id, row.view)] = index
    exact_delta: list[float] = []
    estimate_delta: list[float] = []
    for index, row in enumerate(rows):
        if row.split != "surrogate_holdout" or row.condition == "clean":
            continue
        clean_index = clean.get((row.speaker_id, row.sample_id, row.view))
        if clean_index is None:
            continue
        values = (exact[index], exact[clean_index], estimate[index], estimate[clean_index])
        if not all(math.isfinite(float(value)) for value in values):
            continue
        exact_delta.append(float(exact[index] - exact[clean_index]))
        estimate_delta.append(float(estimate[index] - estimate[clean_index]))
    spearman = safe_spearman(
        np.asarray(exact_delta, dtype=np.float64),
        np.asarray(estimate_delta, dtype=np.float64),
    )
    return {
        "rows": len(exact_delta),
        "spearman": spearman,
        "gate": DELTA_SPEARMAN_GATE,
        "decision": "PASS" if spearman >= DELTA_SPEARMAN_GATE else "FAIL",
    }


def formula_report(
    name: str,
    rows: list[ParityRow],
    exact: np.ndarray,
    raw: np.ndarray,
    train_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row.split == split],
            dtype=np.int64,
        )
        for split in EXPECTED_SPLIT_ROWS
    }
    train_index = split_indices["surrogate_train"]
    calibration_index = split_indices["surrogate_calibration"]
    holdout_index = split_indices["surrogate_holdout"]
    finite_train = np.isfinite(raw[train_index])
    alignment_scale, alignment_bias = positive_affine(
        raw[train_index][finite_train],
        exact[train_index][finite_train],
    )
    aligned = raw * alignment_scale + alignment_bias
    finite_calibration = np.isfinite(aligned[calibration_index])
    calibration_scale, calibration_bias = positive_affine(
        aligned[calibration_index][finite_calibration],
        exact[calibration_index][finite_calibration],
    )
    calibrated = aligned * calibration_scale + calibration_bias
    normalized_calibration_mse = float(
        np.mean(
            (
                (
                    calibrated[calibration_index][finite_calibration]
                    - exact[calibration_index][finite_calibration]
                )
                / train_scale
            )
            ** 2
        )
    )
    holdout_views = {}
    for view in ("cs", "sv"):
        index = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if row.split == "surrogate_holdout" and row.view == view
            ],
            dtype=np.int64,
        )
        holdout_views[view] = metric_report(exact[index], calibrated[index], train_scale)
    coverage = {
        split: {
            view: {
                "rows": sum(row.split == split and row.view == view for row in rows),
                "finite_rows": sum(
                    row.split == split
                    and row.view == view
                    and math.isfinite(float(raw[index]))
                    for index, row in enumerate(rows)
                ),
            }
            for view in ("cs", "sv")
        }
        for split in EXPECTED_SPLIT_ROWS
    }
    return calibrated, {
        "formula": name,
        "train_alignment": {
            "scale": alignment_scale,
            "bias": alignment_bias,
            "finite_rows": int(finite_train.sum()),
        },
        "calibration_alignment": {
            "scale": calibration_scale,
            "bias": calibration_bias,
            "finite_rows": int(finite_calibration.sum()),
        },
        "normalized_calibration_mse": normalized_calibration_mse,
        "holdout": metric_report(
            exact[holdout_index], calibrated[holdout_index], train_scale
        ),
        "holdout_views": holdout_views,
        "holdout_delta": paired_delta_report(rows, exact, calibrated),
        "coverage": coverage,
    }


def write_predictions(
    path: Path,
    rows: list[ParityRow],
    numpy_raw: np.ndarray,
    numpy_calibrated: np.ndarray,
    torch_recalibrated: np.ndarray,
) -> None:
    fields = [
        "speaker_id",
        "sample_id",
        "split",
        "condition",
        "view",
        "label",
        "sample_group",
        "audio_path",
        "audio_sha256",
        "exact_hnr",
        "legacy_numpy_raw_hnr",
        "legacy_numpy_calibrated_hnr",
        "raw_cc_v3_raw_hnr",
        "raw_cc_v3_recalibrated_hnr",
        "raw_cc_v3_frozen_calibrated_hnr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "speaker_id": row.speaker_id,
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "condition": row.condition,
                    "view": row.view,
                    "label": row.label,
                    "sample_group": row.sample_group,
                    "audio_path": row.audio_path,
                    "audio_sha256": row.audio_sha256,
                    "exact_hnr": row.exact_hnr,
                    "legacy_numpy_raw_hnr": numpy_raw[index],
                    "legacy_numpy_calibrated_hnr": numpy_calibrated[index],
                    "raw_cc_v3_raw_hnr": row.torch_raw_hnr,
                    "raw_cc_v3_recalibrated_hnr": torch_recalibrated[index],
                    "raw_cc_v3_frozen_calibrated_hnr": (
                        row.torch_frozen_calibrated_hnr
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_hashes = {
        "label_bank": validate_hash(args.label_bank, args.label_bank_sha256),
        "torch_predictions": validate_hash(
            args.torch_predictions, args.torch_predictions_sha256
        ),
        "torch_report": validate_hash(args.torch_report, args.torch_report_sha256),
    }
    torch_report = json.loads(args.torch_report.read_text(encoding="utf-8"))
    train_scale = float(torch_report["train_hnr_scale"])
    rows = load_rows(args.label_bank, args.torch_predictions)

    values: list[float] = []
    verified_hashes: list[str] = []
    if args.workers == 1:
        iterator = map(legacy_numpy_hnr, rows)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(legacy_numpy_hnr, rows, chunksize=4)
    try:
        for index, (value, audio_hash) in enumerate(iterator, start=1):
            values.append(value)
            verified_hashes.append(audio_hash)
            if index % 50 == 0 or index == len(rows):
                print(f"hnr_numpy_parity rows={index}/{len(rows)}", flush=True)
    finally:
        if args.workers != 1:
            executor.shutdown()
    numpy_raw = np.asarray(values, dtype=np.float64)
    if len(verified_hashes) != len(rows):
        raise ValueError("legacy NumPy scoring ended before all rows were returned")
    torch_raw = np.asarray([row.torch_raw_hnr for row in rows], dtype=np.float64)
    exact = np.asarray([row.exact_hnr for row in rows], dtype=np.float64)
    numpy_calibrated, numpy_report = formula_report(
        "legacy_numpy_python_version_get_hnr",
        rows,
        exact,
        numpy_raw,
        train_scale,
    )
    torch_recalibrated, current_torch_report = formula_report(
        "raw_cc_v3",
        rows,
        exact,
        torch_raw,
        train_scale,
    )
    frozen_torch = np.asarray(
        [row.torch_frozen_calibrated_hnr for row in rows], dtype=np.float64
    )
    maximum_reproduction_error = float(
        np.max(np.abs(frozen_torch - torch_recalibrated))
    )
    if maximum_reproduction_error > 1e-9:
        raise ValueError(
            "frozen raw_cc_v3 calibration was not reproduced: "
            f"max error {maximum_reproduction_error}"
        )

    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "hnr_baseline_parity_predictions.csv"
    write_predictions(
        predictions_path,
        rows,
        numpy_raw,
        numpy_calibrated,
        torch_recalibrated,
    )
    report = {
        "schema_version": "avqi-route-c-hnr-baseline-parity-v7",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_hashes": source_hashes,
        "rows": len(rows),
        "split_rows": EXPECTED_SPLIT_ROWS,
        "train_hnr_scale": train_scale,
        "legacy_numpy": numpy_report,
        "current_torch_raw_cc_v3": current_torch_report,
        "frozen_torch_calibration_max_abs_reproduction_error": (
            maximum_reproduction_error
        ),
        "contract": {
            "exact_praat_labels_are_authority": True,
            "legacy_numpy_is_secondary_reference": True,
            "final_or_fresh_waveform_panel_accessed": False,
            "candidate_selection_performed": False,
            "waveform_modified": False,
            "formal_generator_training_authorized": False,
        },
        "waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
    }
    report_path = args.output_dir / "hnr_baseline_parity_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "avqi-route-c-hnr-baseline-parity-receipt-v7",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
