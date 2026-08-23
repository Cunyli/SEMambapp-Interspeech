#!/usr/bin/env python3
"""Audit exact Praat, NumPy, and current Torch CPPS on shared AVQI inputs.

The exact Praat label remains authoritative. ``prepare`` runs in the locked
AVQI environment and saves exact-preprocessed inputs plus exact/NumPy values;
``torch`` runs in the Slurm ``semambapp`` environment and adds current Torch
values and gradients. Neither stage trains a model or optimizes a waveform.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
CURRENT_TORCH_PEAK_MODE = "hard"
CPPS_CANDIDATE_MODE = "praat_topology_v7"
CPPS_CANDIDATE_POWER_FLOOR = 1e-6
EXACT_BANK_TOLERANCE = 1e-4
DEFAULT_GRADIENT_ROW_INDICES = "3,16,20"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--avqi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "torch", "gradient"),
        required=True,
        help=(
            "prepare exact/NumPy inputs, add Torch parity results, or audit "
            "selected CPPS gradient terms"
        ),
    )
    parser.add_argument(
        "--splits",
        default="surrogate_calibration,surrogate_holdout",
        help="comma-separated frozen splits to audit",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12,
        help="maximum rows in prepare stage; zero audits every eligible row",
    )
    parser.add_argument(
        "--views",
        default="cs,sv",
        help="comma-separated task views to include; default excludes combined both",
    )
    parser.add_argument(
        "--row-indices",
        default=DEFAULT_GRADIENT_ROW_INDICES,
        help="comma-separated prepared row indices for the gradient stage",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"label bank is empty: {path}")
    return rows


def select_rows(
    rows: Iterable[dict[str, str]],
    splits: tuple[str, ...],
    max_rows: int,
    views: tuple[str, ...],
) -> list[dict[str, str]]:
    if max_rows < 0:
        raise ValueError("max rows must be non-negative")
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("view") in views
            and row.get("scoring_status") == "ok"
            and row.get("split") in splits
        ),
        key=lambda row: (
            splits.index(row["split"]),
            row["speaker_id"],
            row["condition_id"],
            row["view"],
        ),
    )
    if not eligible:
        raise ValueError(f"no eligible rows for splits={splits}")
    if max_rows == 0 or max_rows >= len(eligible):
        return eligible
    grouped = {
        split: [row for row in eligible if row["split"] == split]
        for split in splits
    }
    present_splits = sum(bool(grouped[split]) for split in splits)
    base_quota, remainder = divmod(max_rows, present_splits)
    selected: list[dict[str, str]] = []
    for split_index, split in enumerate(splits):
        if not grouped[split]:
            continue
        quota = base_quota + int(split_index < remainder)
        by_speaker: dict[str, list[dict[str, str]]] = {}
        for row in grouped[split]:
            by_speaker.setdefault(row["speaker_id"], []).append(row)
        speakers = sorted(by_speaker)
        for speaker_index, speaker in enumerate(speakers[:quota]):
            speaker_rows = sorted(
                by_speaker[speaker],
                key=lambda row: (row["condition_id"], row["view"], row["sample_id"]),
            )
            selected.append(speaker_rows[speaker_index % len(speaker_rows)])
    return selected


def prepare_exact_input(
    row: dict[str, str],
    *,
    read_and_resample_signal: Any,
    highpass_filter: Any,
    length_normalize_sv: Any,
    get_voiced_segments: Any,
    concatenate_signals: Any,
) -> np.ndarray:
    view = row["view"]
    cs_path = Path(row["cs_path"])
    sv_path = Path(row["sv_path"])
    if view == "cs":
        signal = read_and_resample_signal(cs_path, SAMPLE_RATE)
        signal = highpass_filter(signal, SAMPLE_RATE)
        return np.asarray(get_voiced_segments(signal, SAMPLE_RATE), dtype=np.float64)
    if view == "sv":
        signal = read_and_resample_signal(sv_path, SAMPLE_RATE)
        signal = highpass_filter(signal, SAMPLE_RATE)
        return np.asarray(length_normalize_sv(signal, SAMPLE_RATE), dtype=np.float64)
    if view != "both":
        raise ValueError(f"unsupported label-bank view: {view}")
    signal_sv = read_and_resample_signal(sv_path, SAMPLE_RATE)
    signal_cs = read_and_resample_signal(cs_path, SAMPLE_RATE)
    signal_sv = highpass_filter(signal_sv, SAMPLE_RATE)
    signal_cs = highpass_filter(signal_cs, SAMPLE_RATE)
    voiced_sv = length_normalize_sv(signal_sv, SAMPLE_RATE)
    voiced_cs = get_voiced_segments(signal_cs, SAMPLE_RATE)
    return np.asarray(
        concatenate_signals(voiced_sv, voiced_cs, SAMPLE_RATE),
        dtype=np.float64,
    )


def safe_spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = stats.spearmanr(left, right).statistic
    return float(value) if math.isfinite(float(value)) else None


def summarize_pairs(
    exact: list[float],
    candidate: list[float],
) -> dict[str, float | None]:
    error = np.asarray(candidate, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return {
        "n": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "spearman": safe_spearman(exact, candidate),
    }


def operation_mapping() -> list[dict[str, str]]:
    return [
        {
            "exact_praat_operation": "Resample to 2 * maximum frequency = 10 kHz",
            "existing_numpy_approximation": "No resampling; keeps input sampling rate",
            "current_torch_operation": "No resampling; normally operates at 16 kHz",
            "mismatch": "Frequency grid, filter coefficient, window samples, and quefrency bins differ",
            "differentiability_treatment": "Use a fixed differentiable resampler or equivalent fixed linear map with recorded boundary convention",
        },
        {
            "exact_praat_operation": "60 Hz pitch floor gives a 50 ms effective Gaussian analysis window",
            "existing_numpy_approximation": "40 ms periodic Hann window",
            "current_torch_operation": "50 ms periodic Hann window at 16 kHz",
            "mismatch": "Window family and exact centre geometry differ",
            "differentiability_treatment": "Use a fixed Gaussian-like window with explicit sample-centre convention",
        },
        {
            "exact_praat_operation": "Frame centres separated by 0.002 s",
            "existing_numpy_approximation": "2 ms hop",
            "current_torch_operation": "2 ms hop plus optional max-frame subsampling",
            "mismatch": "Hop agrees, but frame limits/subsampling can change the average",
            "differentiability_treatment": "Keep every frame or use a fixed, documented subset only as a bounded approximation",
        },
        {
            "exact_praat_operation": "50 Hz pre-emphasis after resampling",
            "existing_numpy_approximation": "No pre-emphasis",
            "current_torch_operation": "Causal 50 Hz pre-emphasis at original rate after separate preparation",
            "mismatch": "Filter rate and placement differ; NumPy omits it",
            "differentiability_treatment": "Use the exact causal recurrence after the 10 kHz-equivalent stage",
        },
        {
            "exact_praat_operation": "log-power Spectrum -> real inverse Fourier transform -> square to PowerCepstrum",
            "existing_numpy_approximation": "real IFFT of log power without squaring the inverse result",
            "current_torch_operation": "real IFFT of log power with a hard power floor, without squaring",
            "mismatch": "PowerCepstrum units and normalization are not aligned",
            "differentiability_treatment": "Implement fixed log floor, squared inverse, and dB conversion explicitly",
        },
        {
            "exact_praat_operation": "Smooth before trend fitting: 0.01 s time and 0.001 s quefrency windows",
            "existing_numpy_approximation": "No cepstrogram smoothing; median-filter final frame CPPS values",
            "current_torch_operation": "Average-pool five frames and 17 quefrency bins",
            "mismatch": "NumPy smooths the wrong object; Torch widths are tied to 16 kHz kernels",
            "differentiability_treatment": "Use explicit rectangular averaging with Praat-equivalent widths and edge handling",
        },
        {
            "exact_praat_operation": "60--330 Hz search, tolerance 0.05, Parabolic interpolation",
            "existing_numpy_approximation": "Hard argmax plus local parabolic value",
            "current_torch_operation": "Soft expectation or hard bin maximum; hard mode has no parabolic correction",
            "mismatch": "Hard Torch mode misses requested interpolation; soft mode changes topology",
            "differentiability_treatment": "Detach only selected bin/topology; keep neighbouring peak values differentiable",
        },
        {
            "exact_praat_operation": "Straight trend over 0.001--0.05 s, incomplete Theil Robust fit",
            "existing_numpy_approximation": "Ordinary least squares over only the search band",
            "current_torch_operation": "Iteratively weighted least-squares-like line over only the search band",
            "mismatch": "Trend range, fit family, and peak influence differ",
            "differentiability_treatment": "Freeze robust pair/order topology if needed; retain gradients through selected amplitudes",
        },
        {
            "exact_praat_operation": "Average individual frame CPPS values after smoothing/trend/peak operations",
            "existing_numpy_approximation": "Median-filter frame CPPS, then unweighted mean",
            "current_torch_operation": "Soft power/ZCR selection-weighted mean",
            "mismatch": "Frame inclusion and weighting differ; selection can create unrelated large gradients",
            "differentiability_treatment": "Use a fixed valid-frame mask and unweighted mean",
        },
    ]


def mapping_markdown(
    mapping: list[dict[str, str]],
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> str:
    environment = summary.get("exact_environment", {})
    lines = [
        "# CPPS Route C operation mapping and baseline parity",
        "",
        "## Scope",
        "",
        "This artifact audits CPPS only. Exact Praat output is authoritative; no generator optimizer step or formal pathology training was run.",
        "",
        f"- source commit: `{args.source_commit}`",
        f"- label bank: `{args.label_bank}`",
        f"- label bank SHA256: `{args.label_bank_sha256}`",
        f"- exact environment: `{environment.get('python', 'not recorded')}`; Parselmouth `{environment.get('parselmouth', 'not recorded')}`",
        f"- audited rows: `{summary['rows']}`",
        "",
        "## Operation mapping",
        "",
        "| Exact Praat operation | Existing NumPy approximation | Current Torch operation | Mismatch | Differentiability treatment |",
        "|---|---|---|---|---|",
    ]
    for row in mapping:
        lines.append(
            "| "
            + " | ".join(
                row[key].replace("|", "\\|")
                for key in (
                    "exact_praat_operation",
                    "existing_numpy_approximation",
                    "current_torch_operation",
                    "mismatch",
                    "differentiability_treatment",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Baseline parity",
            "",
            "| Candidate | MAE vs exact runtime CPPS | RMSE | Median absolute error | Max absolute error | Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate, metrics in summary["candidate_metrics"].items():
        lines.append(
            f"| {candidate} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
            f"{metrics['median_absolute_error']:.6f} | {metrics['max_absolute_error']:.6f} | "
            f"{metrics['spearman'] if metrics['spearman'] is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The exact-vs-label-bank check is a scorer/data integrity check, not a candidate promotion gate. NumPy and Torch values quantify approximation mismatch on the same exact-preprocessed inputs. Candidate implementation is deferred until this evidence is reviewed.",
            "",
            "Praat references: [Sound: To PowerCepstrogram](https://praat.org/manual/Sound__To_PowerCepstrogram___.html), [Spectrum: To PowerCepstrum](https://praat.org/manual/Spectrum__To_PowerCepstrum.html), [PowerCepstrogram: Smooth](https://praat.org/manual/PowerCepstrogram__Smooth___.html), and [PowerCepstrogram: Get CPPS](https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html).",
            "",
        ]
    )
    return "\n".join(lines)


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty parity CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_common_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, str]]]:
    if not args.label_bank.is_file():
        raise FileNotFoundError(args.label_bank)
    if not args.avqi_root.is_dir():
        raise FileNotFoundError(args.avqi_root)
    actual_label_hash = sha256_file(args.label_bank)
    if actual_label_hash != args.label_bank_sha256:
        raise ValueError(
            f"label-bank hash mismatch: expected {args.label_bank_sha256}, got {actual_label_hash}"
        )
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    if not splits:
        raise ValueError("at least one split is required")
    views = tuple(value.strip() for value in args.views.split(",") if value.strip())
    if not views:
        raise ValueError("at least one view is required")
    return splits, views, select_rows(read_rows(args.label_bank), splits, args.max_rows, views)


def prepare_stage(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    splits, views, rows = validate_common_inputs(args)

    # The exact scorer and Torch live in separate locked environments. These
    # stage-local imports are intentional and avoid package shadowing.
    sys.path.insert(0, str(args.avqi_root))
    import parselmouth

    from avqi_code.praat_version import concatenate_signals
    from avqi_code.praat_version import get_cpps as get_praat_cpps
    from avqi_code.praat_version import get_voiced_segments
    from avqi_code.praat_version import highpass_filter
    from avqi_code.praat_version import length_normalize_sv
    from avqi_code.python_version import get_cpps as get_numpy_cpps
    from avqi_code.python_version import read_and_resample_signal

    records: list[dict[str, Any]] = []
    inputs: dict[str, np.ndarray] = {}
    exact_values: list[float] = []
    numpy_values: list[float] = []
    for index, row in enumerate(rows):
        exact_input = prepare_exact_input(
            row,
            read_and_resample_signal=read_and_resample_signal,
            highpass_filter=highpass_filter,
            length_normalize_sv=length_normalize_sv,
            get_voiced_segments=get_voiced_segments,
            concatenate_signals=concatenate_signals,
        )
        if exact_input.ndim != 1 or exact_input.size < 2:
            raise ValueError(f"invalid exact input for row {row['speaker_id']}")
        exact_value = float(get_praat_cpps(exact_input, SAMPLE_RATE))
        bank_value = float(row["cpps"])
        numpy_value = float(get_numpy_cpps(exact_input, SAMPLE_RATE))
        input_key = f"input_{index:04d}"
        inputs[input_key] = exact_input
        records.append(
            {
                "row_index": index,
                "input_key": input_key,
                "speaker_id": row["speaker_id"],
                "sample_id": row.get("sample_id", ""),
                "split": row["split"],
                "condition_id": row["condition_id"],
                "view": row["view"],
                "sample_count": int(exact_input.size),
                "bank_cpps": bank_value,
                "exact_runtime_cpps": exact_value,
                "exact_runtime_minus_bank": exact_value - bank_value,
                "numpy_cpps": numpy_value,
                "numpy_minus_exact_runtime": numpy_value - exact_value,
            }
        )
        exact_values.append(exact_value)
        numpy_values.append(numpy_value)
        print(
            f"row={index + 1}/{len(rows)} speaker={row['speaker_id']} "
            f"split={row['split']} view={row['view']} exact={exact_value:.6f}",
            flush=True,
        )

    bank_errors = [float(record["exact_runtime_minus_bank"]) for record in records]
    summary = {
        "rows": len(records),
        "splits": list(splits),
        "views": list(views),
        "exact_bank_max_absolute_error": float(np.max(np.abs(bank_errors))),
        "exact_bank_reproduction": (
            "PASS" if np.max(np.abs(bank_errors)) <= EXACT_BANK_TOLERANCE else "FAIL"
        ),
        "candidate_metrics": {
            "existing_numpy": summarize_pairs(exact_values, numpy_values),
        },
        "exact_environment": {
            "python": sys.executable,
            "parselmouth": parselmouth.__version__,
            "numpy": np.__version__,
        },
    }
    mapping = operation_mapping()
    report = {
        "schema_version": "avqi-route-c-cpps-parity-v1",
        "stage": "prepare",
        "decision": "BASELINE_EXACT_NUMPY_PREPARED",
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "source_commit": args.source_commit,
        "label_bank": str(args.label_bank),
        "label_bank_sha256": args.label_bank_sha256,
        "exact_environment": summary["exact_environment"],
        "operation_mapping": mapping,
        "summary": summary,
        "records": records,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "cpps_parity_inputs.npz", **inputs)
    (args.output_dir / "cpps_parity_prepare.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_records_csv(args.output_dir / "cpps_parity_prepare_rows.csv", records)
    (args.output_dir / "cpps_operation_mapping.md").write_text(
        mapping_markdown(mapping, args=args, summary=summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def torch_stage(args: argparse.Namespace) -> None:
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"prepare output is missing: {args.output_dir}")
    report_path = args.output_dir / "cpps_parity_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {report_path}")
    prepare_path = args.output_dir / "cpps_parity_prepare.json"
    inputs_path = args.output_dir / "cpps_parity_inputs.npz"
    if not prepare_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError("prepare stage artifacts are incomplete")
    if not args.label_bank.is_file():
        raise FileNotFoundError(args.label_bank)
    actual_label_hash = sha256_file(args.label_bank)
    if actual_label_hash != args.label_bank_sha256:
        raise ValueError(
            f"label-bank hash mismatch: expected {args.label_bank_sha256}, "
            f"got {actual_label_hash}"
        )

    # The semambapp Torch import is intentionally isolated to the Slurm stage.
    import torch

    from model.avqi_components import PraatDifferentiableAVQIComponentEstimator

    prepare_report = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare_report["label_bank_sha256"] != args.label_bank_sha256:
        raise ValueError("prepared inputs use a different label-bank hash")
    records = prepare_report["records"]
    input_archive = np.load(inputs_path, allow_pickle=False)
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode=CURRENT_TORCH_PEAK_MODE,
    )
    candidate_estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode=CURRENT_TORCH_PEAK_MODE,
        cpps_mode=CPPS_CANDIDATE_MODE,
        cpps_power_floor=CPPS_CANDIDATE_POWER_FLOOR,
    )
    exact_values: list[float] = []
    torch_direct_values: list[float] = []
    torch_full_values: list[float] = []
    candidate_values: list[float] = []
    candidate_full_values: list[float] = []
    candidate_gradient_norms: list[float] = []
    for index, record in enumerate(records):
        exact_input = np.asarray(input_archive[record["input_key"]], dtype=np.float64)
        waveform = torch.from_numpy(exact_input.copy())
        with torch.inference_mode():
            direct_value = float(estimator._cpps(waveform))
            full_value = float(estimator.raw_components(waveform.unsqueeze(0))[0, 0])
            candidate_value = float(candidate_estimator._cpps(waveform))
            candidate_full_value = float(
                candidate_estimator.raw_components(waveform.unsqueeze(0))[0, 0]
            )
        gradient_waveform = waveform.clone().requires_grad_()
        gradient_value = estimator._cpps(gradient_waveform)
        gradient = torch.autograd.grad(gradient_value, gradient_waveform)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite CPPS gradient at row {index}")
        candidate_gradient_waveform = waveform.clone().requires_grad_()
        candidate_gradient_value = candidate_estimator._cpps(
            candidate_gradient_waveform
        )
        candidate_gradient = torch.autograd.grad(
            candidate_gradient_value,
            candidate_gradient_waveform,
        )[0]
        if not torch.isfinite(candidate_gradient).all():
            raise FloatingPointError(
                f"non-finite candidate CPPS gradient at row {index}"
            )
        exact_value = float(record["exact_runtime_cpps"])
        record.update(
            {
                "torch_direct_current_cpps": direct_value,
                "torch_direct_minus_exact_runtime": direct_value - exact_value,
                "torch_full_current_cpps": full_value,
                "torch_full_minus_exact_runtime": full_value - exact_value,
                "torch_direct_input_gradient_norm": float(gradient.norm()),
                "torch_direct_input_gradient_max_abs": float(gradient.abs().max()),
                "torch_candidate_cpps": candidate_value,
                "torch_candidate_minus_exact_runtime": candidate_value - exact_value,
                "torch_candidate_full_raw_components": candidate_full_value,
                "torch_candidate_full_minus_exact_runtime": (
                    candidate_full_value - exact_value
                ),
                "torch_candidate_input_gradient_norm": float(
                    candidate_gradient.norm()
                ),
                "torch_candidate_input_gradient_max_abs": float(
                    candidate_gradient.abs().max()
                ),
            }
        )
        exact_values.append(exact_value)
        torch_direct_values.append(direct_value)
        torch_full_values.append(full_value)
        candidate_values.append(candidate_value)
        candidate_full_values.append(candidate_full_value)
        candidate_gradient_norms.append(float(candidate_gradient.norm()))
        print(
            f"torch_row={index + 1}/{len(records)} speaker={record['speaker_id']} "
            f"split={record['split']} view={record['view']} "
            f"direct={direct_value:.6f} candidate={candidate_value:.6f}",
            flush=True,
        )

    summary = dict(prepare_report["summary"])
    summary["candidate_metrics"] = {
        "existing_numpy": summary["candidate_metrics"]["existing_numpy"],
        "current_torch_direct_on_exact_input": summarize_pairs(
            exact_values,
            torch_direct_values,
        ),
        "current_torch_full_raw_components": summarize_pairs(
            exact_values,
            torch_full_values,
        ),
        "torch_candidate_direct_praat_topology_v7": summarize_pairs(
            exact_values,
            candidate_values,
        ),
        "torch_candidate_full_raw_components_praat_topology_v7": summarize_pairs(
            exact_values,
            candidate_full_values,
        ),
    }
    gradient_norms = [record["torch_direct_input_gradient_norm"] for record in records]
    summary["gradient"] = {
        "min_norm": float(min(gradient_norms)),
        "median_norm": float(np.median(gradient_norms)),
        "max_norm": float(max(gradient_norms)),
        "candidate_min_norm": float(min(candidate_gradient_norms)),
        "candidate_median_norm": float(np.median(candidate_gradient_norms)),
        "candidate_max_norm": float(max(candidate_gradient_norms)),
    }
    summary["exact_environment"] = dict(summary["exact_environment"])
    summary["exact_environment"]["torch"] = torch.__version__
    report = dict(prepare_report)
    report["stage"] = "complete"
    report["decision"] = "BASELINE_PARITY_COMPLETE"
    report["source_commit"] = args.source_commit
    report["prepare_source_commit"] = prepare_report["source_commit"]
    report["candidate_configuration"] = {
        "cpps_mode": CPPS_CANDIDATE_MODE,
        "cpps_power_floor": CPPS_CANDIDATE_POWER_FLOOR,
    }
    report["exact_environment"] = summary["exact_environment"]
    report["summary"] = summary
    report["records"] = records
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_records_csv(args.output_dir / "cpps_parity_torch_rows.csv", records)
    (args.output_dir / "cpps_operation_mapping.md").write_text(
        mapping_markdown(report["operation_mapping"], args=args, summary=summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_gradient_row_indices(raw_indices: str, row_count: int) -> tuple[int, ...]:
    values = tuple(
        int(value.strip())
        for value in raw_indices.split(",")
        if value.strip()
    )
    if not values:
        raise ValueError("gradient stage requires at least one row index")
    if len(set(values)) != len(values):
        raise ValueError(f"gradient row indices must be unique: {values}")
    invalid = tuple(value for value in values if value < 0 or value >= row_count)
    if invalid:
        raise IndexError(
            f"gradient row indices {invalid} are outside prepared row count {row_count}"
        )
    return values


def tensor_distribution(values: Any) -> dict[str, float | int]:
    flat = values.detach().reshape(-1).double()
    finite = flat[flat.isfinite()]
    if finite.numel() != flat.numel():
        raise FloatingPointError("gradient audit encountered a non-finite tensor")
    quantile_levels = finite.new_tensor((0.0, 0.001, 0.01, 0.5, 0.99, 1.0))
    quantiles = finite.quantile(quantile_levels)
    positive = finite[finite > 0.0]
    return {
        "count": int(finite.numel()),
        "min": float(quantiles[0]),
        "q001": float(quantiles[1]),
        "q01": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(quantiles[5]),
        "min_positive": float(positive.min()) if positive.numel() else 0.0,
        "mean": float(finite.mean()),
        "mean_abs": float(finite.abs().mean()),
        "rms": float(finite.square().mean().sqrt()),
    }


def vector_stats(vector: Any) -> dict[str, float | int]:
    flat = vector.detach().reshape(-1).double()
    if not flat.isfinite().all():
        raise FloatingPointError("gradient audit produced a non-finite vector")
    energy = flat.square()
    total_energy = energy.sum()
    statistics: dict[str, float | int] = {
        "count": int(flat.numel()),
        "norm": float(flat.norm()),
        "max_abs": float(flat.abs().max()),
        "mean_abs": float(flat.abs().mean()),
        "max_abs_index": int(flat.abs().argmax()),
    }
    for top_k in (1, 10, 100):
        selected = min(top_k, flat.numel())
        fraction = energy.topk(selected).values.sum() / total_energy.clamp_min(1e-300)
        statistics[f"top_{top_k}_energy_fraction"] = float(fraction)
    return statistics


def gradient_vector(torch_module: Any, scalar: Any, waveform: Any) -> Any:
    gradient = torch_module.autograd.grad(
        scalar,
        waveform,
        retain_graph=True,
    )[0]
    if not gradient.isfinite().all():
        raise FloatingPointError("CPPS decomposition produced a non-finite gradient")
    return gradient


def gradient_pair_metrics(peak_gradient: Any, baseline_gradient: Any) -> dict[str, float]:
    peak_flat = peak_gradient.detach().reshape(-1).double()
    baseline_flat = baseline_gradient.detach().reshape(-1).double()
    difference = peak_flat - baseline_flat
    peak_norm = peak_flat.norm()
    baseline_norm = baseline_flat.norm()
    denominator = (peak_norm * baseline_norm).clamp_min(1e-300)
    return {
        "cosine_similarity": float(peak_flat.dot(baseline_flat) / denominator),
        "difference_norm": float(difference.norm()),
        "cancellation_ratio": float(
            difference.norm() / (peak_norm + baseline_norm).clamp_min(1e-300)
        ),
    }


def intermediate_gradient_stats(
    torch_module: Any,
    scalar: Any,
    terms: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    statistics: dict[str, dict[str, float | int]] = {}
    for name in (
        "spectrum_power",
        "log_power",
        "real_cepstrum",
        "power_cepstrum",
        "cepstrum_db",
    ):
        gradient = torch_module.autograd.grad(
            scalar,
            terms[name],
            retain_graph=True,
        )[0]
        statistics[name] = vector_stats(gradient)
    return statistics


def topology_gradient_record(
    torch_module: Any,
    *,
    waveform: Any,
    peak_scalar: Any,
    baseline_scalar: Any,
    cpps_scalar: Any,
) -> tuple[dict[str, Any], Any]:
    peak_gradient = gradient_vector(torch_module, peak_scalar, waveform)
    baseline_gradient = gradient_vector(torch_module, baseline_scalar, waveform)
    cpps_gradient = gradient_vector(torch_module, cpps_scalar, waveform)
    reconstructed = peak_gradient - baseline_gradient
    reconstruction_error = (
        reconstructed - cpps_gradient
    ).norm() / cpps_gradient.norm().clamp_min(1e-300)
    return (
        {
            "peak": vector_stats(peak_gradient),
            "baseline": vector_stats(baseline_gradient),
            "cpps": vector_stats(cpps_gradient),
            "peak_minus_baseline": gradient_pair_metrics(
                peak_gradient,
                baseline_gradient,
            ),
            "gradient_reconstruction_relative_error": float(reconstruction_error),
        },
        cpps_gradient,
    )


def gradient_stage(args: argparse.Namespace) -> None:
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"prepare output is missing: {args.output_dir}")
    result_path = args.output_dir / "cpps_gradient_decomposition.json"
    if result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite completed gradient audit: {result_path}"
        )
    prepare_path = args.output_dir / "cpps_parity_prepare.json"
    inputs_path = args.output_dir / "cpps_parity_inputs.npz"
    if not prepare_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError("prepare stage artifacts are incomplete")
    if not args.label_bank.is_file():
        raise FileNotFoundError(args.label_bank)
    actual_label_hash = sha256_file(args.label_bank)
    if actual_label_hash != args.label_bank_sha256:
        raise ValueError(
            f"label-bank hash mismatch: expected {args.label_bank_sha256}, "
            f"got {actual_label_hash}"
        )

    # The semambapp Torch import is intentionally isolated to the Slurm stage.
    import torch

    from model.avqi_components import PraatDifferentiableAVQIComponentEstimator

    prepare_report = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare_report["label_bank_sha256"] != args.label_bank_sha256:
        raise ValueError("prepared inputs use a different label-bank hash")
    prepared_records = prepare_report["records"]
    row_indices = parse_gradient_row_indices(
        args.row_indices,
        len(prepared_records),
    )
    input_archive = np.load(inputs_path, allow_pickle=False)
    exact_backward_estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode=CURRENT_TORCH_PEAK_MODE,
        cpps_mode=CPPS_CANDIDATE_MODE,
        cpps_power_floor=1e-30,
    )
    bounded_floor_estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode=CURRENT_TORCH_PEAK_MODE,
        cpps_mode=CPPS_CANDIDATE_MODE,
        cpps_power_floor=CPPS_CANDIDATE_POWER_FLOOR,
    )

    records: list[dict[str, Any]] = []
    for row_index in row_indices:
        prepared_record = prepared_records[row_index]
        exact_input = np.asarray(
            input_archive[prepared_record["input_key"]],
            dtype=np.float64,
        )

        exact_waveform = torch.from_numpy(exact_input.copy()).requires_grad_()
        exact_terms = exact_backward_estimator._cpps_praat_topology_v7_terms(
            exact_waveform
        )
        exact_topology, exact_gradient = topology_gradient_record(
            torch,
            waveform=exact_waveform,
            peak_scalar=exact_terms["parabolic_peak_value"].mean(),
            baseline_scalar=exact_terms["exact_baseline"].mean(),
            cpps_scalar=exact_terms["exact_cpps"],
        )
        exact_intermediate = intermediate_gradient_stats(
            torch,
            exact_terms["exact_cpps"],
            exact_terms,
        )

        bounded_waveform = torch.from_numpy(exact_input.copy()).requires_grad_()
        bounded_terms = bounded_floor_estimator._cpps_praat_topology_v7_terms(
            bounded_waveform
        )
        floor_only_topology, floor_only_gradient = topology_gradient_record(
            torch,
            waveform=bounded_waveform,
            peak_scalar=bounded_terms["parabolic_peak_value"].mean(),
            baseline_scalar=bounded_terms["exact_baseline"].mean(),
            cpps_scalar=bounded_terms["exact_cpps"],
        )
        detached_topology, detached_gradient = topology_gradient_record(
            torch,
            waveform=bounded_waveform,
            peak_scalar=bounded_terms["stable_peak_value"].mean(),
            baseline_scalar=bounded_terms["baseline"].mean(),
            cpps_scalar=bounded_terms["current_cpps"],
        )
        detached_intermediate = intermediate_gradient_stats(
            torch,
            bounded_terms["current_cpps"],
            bounded_terms,
        )

        forward_values = {
            "exact_topology_floor_1e_30": float(exact_terms["exact_cpps"]),
            "exact_topology_floor_1e_6": float(bounded_terms["exact_cpps"]),
            "detached_topology_floor_1e_6": float(bounded_terms["current_cpps"]),
        }
        if max(forward_values.values()) - min(forward_values.values()) > 1e-10:
            raise AssertionError(
                f"gradient variants changed CPPS forward value at row {row_index}: "
                f"{forward_values}"
            )

        search_power = bounded_terms["power_cepstrum"][
            :, bounded_terms["search_mask"]
        ]
        trend_power = bounded_terms["power_cepstrum"][
            :, bounded_terms["trend_mask"]
        ]
        record = {
            "row_index": row_index,
            "input_key": prepared_record["input_key"],
            "speaker_id": prepared_record["speaker_id"],
            "sample_id": prepared_record["sample_id"],
            "split": prepared_record["split"],
            "condition_id": prepared_record["condition_id"],
            "view": prepared_record["view"],
            "sample_count": prepared_record["sample_count"],
            "exact_runtime_cpps": prepared_record["exact_runtime_cpps"],
            "forward_values": forward_values,
            "input_distribution": tensor_distribution(bounded_waveform),
            "intermediate_distributions": {
                "spectrum_power": tensor_distribution(
                    bounded_terms["spectrum_power"]
                ),
                "power_cepstrum": tensor_distribution(
                    bounded_terms["power_cepstrum"]
                ),
                "search_power_cepstrum": tensor_distribution(search_power),
                "trend_power_cepstrum": tensor_distribution(trend_power),
                "parabolic_denominator": tensor_distribution(
                    bounded_terms["parabolic_denominator"]
                ),
                "peak_offset": tensor_distribution(bounded_terms["peak_offset"]),
            },
            "gradient_modes": {
                "exact_topology_floor_1e_30": exact_topology,
                "exact_topology_floor_1e_6": floor_only_topology,
                "detached_topology_floor_1e_6": detached_topology,
            },
            "intermediate_gradient_sensitivity": {
                "exact_topology_floor_1e_30": exact_intermediate,
                "detached_topology_floor_1e_6": detached_intermediate,
            },
            "cross_mode_gradient": {
                "exact_vs_floor_only_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        exact_gradient.reshape(1, -1),
                        floor_only_gradient.reshape(1, -1),
                    )[0]
                ),
                "exact_vs_detached_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        exact_gradient.reshape(1, -1),
                        detached_gradient.reshape(1, -1),
                    )[0]
                ),
                "floor_only_vs_detached_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        floor_only_gradient.reshape(1, -1),
                        detached_gradient.reshape(1, -1),
                    )[0]
                ),
            },
        }
        records.append(record)
        print(
            f"gradient_row={row_index} speaker={record['speaker_id']} "
            f"view={record['view']} exact_norm="
            f"{exact_topology['cpps']['norm']:.6f} floor_norm="
            f"{floor_only_topology['cpps']['norm']:.6f} detached_norm="
            f"{detached_topology['cpps']['norm']:.6f}",
            flush=True,
        )

    mode_names = tuple(records[0]["gradient_modes"])
    summary = {
        "rows": len(records),
        "row_indices": list(row_indices),
        "gradient_norm_by_mode": {
            mode: {
                "min": float(
                    min(record["gradient_modes"][mode]["cpps"]["norm"] for record in records)
                ),
                "median": float(
                    np.median(
                        [
                            record["gradient_modes"][mode]["cpps"]["norm"]
                            for record in records
                        ]
                    )
                ),
                "max": float(
                    max(record["gradient_modes"][mode]["cpps"]["norm"] for record in records)
                ),
            }
            for mode in mode_names
        },
    }
    report = {
        "schema_version": "avqi-route-c-cpps-gradient-decomposition-v1",
        "stage": "gradient",
        "decision": "GRADIENT_DECOMPOSITION_COMPLETE_NO_PROMOTION",
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "source_commit": args.source_commit,
        "prepare_source_commit": prepare_report["source_commit"],
        "prepare_artifact_sha256": sha256_file(prepare_path),
        "input_archive_sha256": sha256_file(inputs_path),
        "label_bank": str(args.label_bank),
        "label_bank_sha256": args.label_bank_sha256,
        "candidate_configuration": {
            "cpps_mode": CPPS_CANDIDATE_MODE,
            "bounded_floor": CPPS_CANDIDATE_POWER_FLOOR,
        },
        "iteration_compact": {
            "goal": "preserve exact CPPS forward values while localizing gradient outliers",
            "dataset": "frozen 24-row exact parity inputs",
            "baseline": "praat_topology_v7 at ab695ec",
            "primary_metric": "waveform gradient norm and peak/baseline cancellation",
            "guardrails": "forward equality, fixed rows, no scorer promotion or optimizer step",
            "next_experiment": "choose one bounded derivative from measured singular layer",
        },
        "summary": summary,
        "records": records,
    }
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "torch":
        torch_stage(args)
    else:
        gradient_stage(args)


if __name__ == "__main__":
    main()
