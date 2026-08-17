#!/usr/bin/env python3
"""Compare the frozen v2 HNR proxy with one Praat-timed raw-CC candidate.

This is a Route C formula diagnostic, not a generator experiment. Candidate
selection uses only the speaker-disjoint calibration split. The previously
used pathological waveform-pilot panel is deliberately not an input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator


SAMPLE_RATE = 16_000
FORMULAS = {
    "linear_ac_hard_v2": "linear_ac_v2",
    "raw_cc_hard_v3": "raw_cc_v3",
}
LEVEL_SPEARMAN_GATE = 0.70
NORMALIZED_MAE_GATE = 0.50
CALIBRATION_SLOPE_RANGE = (0.75, 1.25)
DELTA_SPEARMAN_GATE = 0.60
GRADIENT_MAX_ABS_GATE = 1e4


@dataclass(frozen=True)
class HNRRow:
    panel: str
    speaker_id: str
    sample_id: str
    split: str
    condition: str
    view: str
    label: str
    sample_group: str
    path: Path
    audio_sha256: str
    exact_hnr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--vctk-external-label-bank", type=Path, required=True)
    parser.add_argument("--vctk-external-label-bank-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-internal-valid-rows", type=int, default=2_134)
    parser.add_argument("--expected-vctk-valid-rows", type=int, default=192)
    parser.add_argument("--expected-train-speakers", type=int, default=197)
    parser.add_argument("--expected-calibration-speakers", type=int, default=26)
    parser.add_argument("--expected-holdout-speakers", type=int, default=26)
    parser.add_argument("--expected-vctk-speakers", type=int, default=12)
    parser.add_argument(
        "--minimum-relative-calibration-improvement",
        type=float,
        default=0.05,
    )
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


def sample_id(row: dict[str, str]) -> str:
    return (
        row.get("sample_id", "").strip()
        or row.get("pair_id", "").strip()
        or row["speaker_id"]
    )


def to_hnr_row(row: dict[str, str], panel: str) -> HNRRow:
    view = row["view"]
    if view not in {"cs", "sv"}:
        raise ValueError(f"unsupported HNR view: {view}")
    exact_hnr = float(row["hnr"])
    if not math.isfinite(exact_hnr):
        raise ValueError("non-finite exact HNR label")
    return HNRRow(
        panel=panel,
        speaker_id=row["speaker_id"],
        sample_id=sample_id(row),
        split=row["split"],
        condition=row["condition_id"],
        view=view,
        label=row["label"],
        sample_group=row["sample_group"],
        path=Path(row[f"{view}_path"]),
        audio_sha256=row[f"{view}_sha256"],
        exact_hnr=exact_hnr,
    )


def load_label_rows(
    path: Path,
    panel: str,
    expected_valid_rows: int,
) -> tuple[list[HNRRow], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    task_rows = [row for row in source_rows if row["view"] in {"cs", "sv"}]
    valid_rows = [row for row in task_rows if row["scoring_status"] == "ok"]
    if len(valid_rows) != expected_valid_rows:
        raise ValueError(
            f"{panel} valid-row mismatch: {len(valid_rows)} != {expected_valid_rows}"
        )
    keys = [
        (
            row["speaker_id"],
            sample_id(row),
            row["split"],
            row["condition_id"],
            row["view"],
        )
        for row in valid_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate task rows in {panel} label bank")
    rows = [to_hnr_row(row, panel) for row in valid_rows]
    return rows, {
        "eligible_rows": len(task_rows),
        "valid_rows": len(rows),
        "invalid_rows": len(task_rows) - len(rows),
        "valid_fraction": len(rows) / len(task_rows),
        "speakers": len({row.speaker_id for row in rows}),
    }


def validate_speaker_contract(rows: list[HNRRow], args: argparse.Namespace) -> None:
    expected = {
        "surrogate_train": args.expected_train_speakers,
        "surrogate_calibration": args.expected_calibration_speakers,
        "surrogate_holdout": args.expected_holdout_speakers,
    }
    actual = {
        split: len({row.speaker_id for row in rows if row.split == split})
        for split in expected
    }
    if actual != expected:
        raise ValueError(f"internal speaker split mismatch: {actual} != {expected}")
    speaker_splits: dict[str, set[str]] = {}
    for row in rows:
        speaker_splits.setdefault(row.speaker_id, set()).add(row.split)
    overlap = {
        speaker: sorted(splits)
        for speaker, splits in speaker_splits.items()
        if len(splits) != 1
    }
    if overlap:
        raise ValueError(f"speaker-disjoint split violation: {overlap}")


def load_waveform(row: HNRRow, verified_hashes: dict[Path, str]) -> torch.Tensor:
    if row.path not in verified_hashes:
        actual_hash = sha256_file(row.path)
        if actual_hash != row.audio_sha256:
            raise ValueError(f"audio hash mismatch: {row.path}")
        verified_hashes[row.path] = actual_hash
    elif verified_hashes[row.path] != row.audio_sha256:
        raise ValueError(f"conflicting audio hash labels: {row.path}")
    audio, sample_rate = sf.read(row.path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"invalid 16 kHz mono audio: {row.path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {row.path}")
    return waveform


def score_rows(
    rows: list[HNRRow],
    estimators: dict[str, PraatDifferentiableAVQIComponentEstimator],
    device: torch.device,
) -> dict[str, np.ndarray]:
    values = {name: [] for name in estimators}
    verified_hashes: dict[Path, str] = {}
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            waveform = load_waveform(row, verified_hashes).to(device)
            for name, estimator in estimators.items():
                value = float(estimator.raw_hnr(waveform).cpu()[0])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} HNR for {row.path}")
                values[name].append(value)
            if index % 50 == 0 or index == len(rows):
                print(
                    f"hnr_formula_rows panel={rows[0].panel} "
                    f"rows={index}/{len(rows)}",
                    flush=True,
                )
    return {
        name: np.asarray(candidate_values, dtype=np.float64)
        for name, candidate_values in values.items()
    }


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


def indexed_metric(
    rows: list[HNRRow],
    exact: np.ndarray,
    estimate: np.ndarray,
    train_scale: float,
    predicate: Callable[[HNRRow], bool],
) -> dict[str, Any]:
    indices = np.asarray(
        [index for index, row in enumerate(rows) if predicate(row)],
        dtype=np.int64,
    )
    if indices.size < 2:
        raise ValueError("metric slice has fewer than two rows")
    return metric_report(exact[indices], estimate[indices], train_scale)


def holdout_delta_spearman(
    rows: list[HNRRow],
    exact: np.ndarray,
    estimate: np.ndarray,
) -> tuple[float, int]:
    holdout_indices = [
        index for index, row in enumerate(rows)
        if row.split == "surrogate_holdout"
    ]
    clean = {
        (rows[index].speaker_id, rows[index].sample_id, rows[index].view): index
        for index in holdout_indices
        if rows[index].condition == "clean"
    }
    exact_delta: list[float] = []
    estimate_delta: list[float] = []
    for index in holdout_indices:
        row = rows[index]
        if row.condition == "clean":
            continue
        key = (row.speaker_id, row.sample_id, row.view)
        if key not in clean:
            raise ValueError(f"holdout degradation lacks clean pair: {key}")
        clean_index = clean[key]
        exact_delta.append(float(exact[index] - exact[clean_index]))
        estimate_delta.append(float(estimate[index] - estimate[clean_index]))
    return (
        safe_spearman(np.asarray(exact_delta), np.asarray(estimate_delta)),
        len(exact_delta),
    )


def gradient_report(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    device: torch.device,
) -> dict[str, Any]:
    time = torch.arange(SAMPLE_RATE, device=device, dtype=torch.float32) / SAMPLE_RATE
    waveform = (
        torch.sin(2.0 * math.pi * 180.0 * time)
        + 0.07 * torch.sin(2.0 * math.pi * 2_300.0 * time)
    ).requires_grad_()
    value = estimator.raw_hnr(waveform).sum()
    gradient = torch.autograd.grad(value, waveform)[0]
    finite = bool(torch.isfinite(gradient).all())
    norm = float(gradient.norm().detach().cpu())
    maximum = float(gradient.abs().max().detach().cpu())
    gates = {
        "finite": finite,
        "nonzero": norm > 0.0,
        "max_abs_le_1e4": maximum <= GRADIENT_MAX_ABS_GATE,
    }
    return {
        "value": float(value.detach().cpu()),
        "gradient_norm": norm,
        "gradient_max_abs": maximum,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def write_predictions(
    path: Path,
    internal_rows: list[HNRRow],
    external_rows: list[HNRRow],
    raw_internal: dict[str, np.ndarray],
    raw_external: dict[str, np.ndarray],
    calibrated_internal: dict[str, np.ndarray],
    calibrated_external: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "panel",
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
    ]
    for formula in FORMULAS:
        fieldnames.extend((f"{formula}_raw_hnr", f"{formula}_calibrated_hnr"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rows, raw, calibrated in (
            (internal_rows, raw_internal, calibrated_internal),
            (external_rows, raw_external, calibrated_external),
        ):
            for index, row in enumerate(rows):
                output: dict[str, Any] = {
                    "panel": row.panel,
                    "speaker_id": row.speaker_id,
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "condition": row.condition,
                    "view": row.view,
                    "label": row.label,
                    "sample_group": row.sample_group,
                    "audio_path": str(row.path),
                    "audio_sha256": row.audio_sha256,
                    "exact_hnr": row.exact_hnr,
                }
                for formula in FORMULAS:
                    output[f"{formula}_raw_hnr"] = float(raw[formula][index])
                    output[f"{formula}_calibrated_hnr"] = float(
                        calibrated[formula][index]
                    )
                writer.writerow(output)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not 0.0 < args.minimum_relative_calibration_improvement < 1.0:
        raise ValueError("relative calibration improvement gate must be in (0, 1)")
    for path, expected_hash in (
        (args.label_bank, args.label_bank_sha256),
        (args.vctk_external_label_bank, args.vctk_external_label_bank_sha256),
    ):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source hash mismatch: {path}")

    internal_rows, internal_coverage = load_label_rows(
        args.label_bank,
        "internal_nonfinal",
        args.expected_internal_valid_rows,
    )
    external_rows, external_coverage = load_label_rows(
        args.vctk_external_label_bank,
        "vctk_external",
        args.expected_vctk_valid_rows,
    )
    validate_speaker_contract(internal_rows, args)
    if external_coverage["speakers"] != args.expected_vctk_speakers:
        raise ValueError(
            "VCTK speaker-count mismatch: "
            f"{external_coverage['speakers']} != {args.expected_vctk_speakers}"
        )
    internal_speakers = {row.speaker_id for row in internal_rows}
    external_speakers = {row.speaker_id for row in external_rows}
    if internal_speakers & external_speakers:
        raise ValueError("VCTK speakers overlap the internal formula panel")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    estimators = {
        name: PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            hnr_mode=mode,
        ).to(device).eval()
        for name, mode in FORMULAS.items()
    }
    raw_internal = score_rows(internal_rows, estimators, device)
    raw_external = score_rows(external_rows, estimators, device)
    gradient = {
        name: gradient_report(estimator, device)
        for name, estimator in estimators.items()
    }

    exact_internal = np.asarray(
        [row.exact_hnr for row in internal_rows], dtype=np.float64
    )
    exact_external = np.asarray(
        [row.exact_hnr for row in external_rows], dtype=np.float64
    )
    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(internal_rows) if row.split == split],
            dtype=np.int64,
        )
        for split in (
            "surrogate_train",
            "surrogate_calibration",
            "surrogate_holdout",
        )
    }
    train_exact = exact_internal[split_indices["surrogate_train"]]
    train_scale = max(float(train_exact.std(ddof=0)), 1e-8)
    calibrated_internal: dict[str, np.ndarray] = {}
    calibrated_external: dict[str, np.ndarray] = {}
    formula_reports: dict[str, Any] = {}

    for name in FORMULAS:
        train_index = split_indices["surrogate_train"]
        calibration_index = split_indices["surrogate_calibration"]
        holdout_index = split_indices["surrogate_holdout"]
        alignment_scale, alignment_bias = positive_affine(
            raw_internal[name][train_index],
            exact_internal[train_index],
        )
        aligned_internal = raw_internal[name] * alignment_scale + alignment_bias
        aligned_external = raw_external[name] * alignment_scale + alignment_bias
        normalized_calibration_mse = float(
            np.mean(
                (
                    (
                        aligned_internal[calibration_index]
                        - exact_internal[calibration_index]
                    )
                    / train_scale
                )
                ** 2
            )
        )
        calibration_scale, calibration_bias = positive_affine(
            aligned_internal[calibration_index],
            exact_internal[calibration_index],
        )
        calibrated_internal[name] = (
            aligned_internal * calibration_scale + calibration_bias
        )
        calibrated_external[name] = (
            aligned_external * calibration_scale + calibration_bias
        )
        delta_spearman, delta_rows = holdout_delta_spearman(
            internal_rows,
            exact_internal,
            calibrated_internal[name],
        )
        internal_slices = {
            view: indexed_metric(
                internal_rows,
                exact_internal,
                calibrated_internal[name],
                train_scale,
                lambda row, expected=view: (
                    row.split == "surrogate_holdout" and row.view == expected
                ),
            )
            for view in ("cs", "sv")
        }
        external_slices = {
            condition: indexed_metric(
                external_rows,
                exact_external,
                calibrated_external[name],
                train_scale,
                lambda row, expected=condition: row.condition == expected,
            )
            for condition in ("clean", "rir_only", "snr20", "snr10")
        }
        formula_reports[name] = {
            "formula": FORMULAS[name],
            "train_alignment": {
                "fit_split": "surrogate_train",
                "scale": alignment_scale,
                "bias": alignment_bias,
            },
            "selection_metric": {
                "split": "surrogate_calibration",
                "normalized_mse": normalized_calibration_mse,
            },
            "post_selection_calibration": {
                "fit_split": "surrogate_calibration",
                "scale": calibration_scale,
                "bias": calibration_bias,
            },
            "holdout": metric_report(
                exact_internal[holdout_index],
                calibrated_internal[name][holdout_index],
                train_scale,
            ),
            "holdout_delta": {
                "rows": delta_rows,
                "spearman": delta_spearman,
                "gate": DELTA_SPEARMAN_GATE,
                "decision": "PASS" if delta_spearman >= DELTA_SPEARMAN_GATE else "FAIL",
            },
            "holdout_views": internal_slices,
            "vctk_external": metric_report(
                exact_external,
                calibrated_external[name],
                train_scale,
            ),
            "vctk_conditions": external_slices,
            "gradient": gradient[name],
        }

    selected = min(
        FORMULAS,
        key=lambda name: formula_reports[name]["selection_metric"]["normalized_mse"],
    )
    baseline_loss = formula_reports["linear_ac_hard_v2"]["selection_metric"][
        "normalized_mse"
    ]
    raw_cc_loss = formula_reports["raw_cc_hard_v3"]["selection_metric"][
        "normalized_mse"
    ]
    relative_improvement = (baseline_loss - raw_cc_loss) / max(baseline_loss, 1e-12)
    raw_cc_report = formula_reports["raw_cc_hard_v3"]
    integration_gates = {
        "selected_by_calibration_only": selected == "raw_cc_hard_v3",
        "relative_calibration_mse_improvement_ge_0_05": (
            relative_improvement >= args.minimum_relative_calibration_improvement
        ),
        "holdout_primary": raw_cc_report["holdout"]["decision"] == "PASS",
        "holdout_delta": raw_cc_report["holdout_delta"]["decision"] == "PASS",
        "holdout_cs_and_sv": all(
            report["decision"] == "PASS"
            for report in raw_cc_report["holdout_views"].values()
        ),
        "vctk_external_primary": (
            raw_cc_report["vctk_external"]["decision"] == "PASS"
        ),
        "vctk_condition_slices": all(
            report["decision"] == "PASS"
            for report in raw_cc_report["vctk_conditions"].values()
        ),
        "finite_nonzero_bounded_gradient": (
            raw_cc_report["gradient"]["decision"] == "PASS"
        ),
    }
    if selected != "raw_cc_hard_v3":
        decision = "KEEP_LINEAR_AC_V2_NO_RAW_CC_INTEGRATION"
    elif all(integration_gates.values()):
        decision = "GO_INTEGRATE_RAW_CC_V3_INTO_FULL_ROUTE_C_SCREEN"
    else:
        decision = "NO_GO_RAW_CC_V3_FULL_ROUTE_C_SCREEN"

    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "hnr_formula_predictions.csv"
    report_path = args.output_dir / "hnr_formula_report.json"
    write_predictions(
        predictions_path,
        internal_rows,
        external_rows,
        raw_internal,
        raw_external,
        calibrated_internal,
        calibrated_external,
    )
    report = {
        "schema_version": "avqi-direct-hnr-formula-diagnostic-v1",
        "purpose": "route_c_hnr_formula_selection_no_waveform_or_generator_update",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "data": {
            "internal_label_bank": str(args.label_bank),
            "internal_label_bank_sha256": args.label_bank_sha256,
            "internal_coverage": internal_coverage,
            "vctk_external_label_bank": str(args.vctk_external_label_bank),
            "vctk_external_label_bank_sha256": args.vctk_external_label_bank_sha256,
            "vctk_external_coverage": external_coverage,
            "final_pathological_waveform_pilot_panel_accessed": False,
        },
        "contract": {
            "selection_split": "surrogate_calibration",
            "holdout_used_for_selection": False,
            "exact_praat_labels_are_authority": True,
            "metric_branch_only": True,
            "full_band_waveform_modified": False,
            "minimum_relative_calibration_improvement": (
                args.minimum_relative_calibration_improvement
            ),
            "raw_cc_timing": {
                "pitch_floor_hz": 75.0,
                "pitch_ceiling_hz": 600.0,
                "periods_per_window": 1.0,
                "time_step_period_fraction": 0.25,
                "silence_threshold": 0.03,
                "voicing_threshold": 0.45,
            },
        },
        "train_hnr_scale": train_scale,
        "formulas": formula_reports,
        "selection": {
            "selected_formula": selected,
            "linear_ac_hard_v2_normalized_calibration_mse": baseline_loss,
            "raw_cc_hard_v3_normalized_calibration_mse": raw_cc_loss,
            "raw_cc_relative_improvement": relative_improvement,
        },
        "integration_gates": integration_gates,
        "decision": decision,
        "authorization": {
            "full_route_c_screen_authorized": (
                decision == "GO_INTEGRATE_RAW_CC_V3_INTO_FULL_ROUTE_C_SCREEN"
            ),
            "bounded_waveform_pilot_authorized": False,
            "formal_generator_training_authorized": False,
        },
        "waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
    }
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-direct-hnr-formula-completion-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "decision": decision,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
