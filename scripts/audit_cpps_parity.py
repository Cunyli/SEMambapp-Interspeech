#!/usr/bin/env python3
"""Audit exact Praat, NumPy, and current Torch CPPS on shared AVQI inputs.

The exact Praat label remains authoritative.  This runner reconstructs the
same AVQI preprocessing for frozen label-bank rows, verifies the stored exact
label with the live Praat bridge, and compares it with the existing NumPy and
current differentiable Torch CPPS formulas.  It deliberately does not train a
model or optimize a waveform.
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
import parselmouth
import torch
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
CURRENT_TORCH_PEAK_MODE = "hard"
EXACT_BANK_TOLERANCE = 1e-4

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
        "--splits",
        default="surrogate_calibration,surrogate_holdout",
        help="comma-separated frozen splits to audit",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12,
        help="maximum rows; zero audits every eligible row",
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
) -> list[dict[str, str]]:
    if max_rows < 0:
        raise ValueError("max rows must be non-negative")
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("view") in {"cs", "sv", "both"}
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

    selected: list[dict[str, str]] = []
    grouped = {
        split: [row for row in eligible if row["split"] == split]
        for split in splits
    }
    split_count = sum(bool(grouped[split]) for split in splits)
    quota = max(1, max_rows // split_count)
    for split in splits:
        selected.extend(grouped[split][:quota])
    if len(selected) < max_rows:
        selected_ids = {id(row) for row in selected}
        selected.extend(
            row for row in eligible if id(row) not in selected_ids
        )
    return selected[:max_rows]


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
    exact_array = np.asarray(exact, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    error = candidate_array - exact_array
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
            "exact_praat_operation": "Resample Sound to 2 * maximum frequency = 10 kHz",
            "existing_numpy_approximation": "No resampling; keeps the input sampling rate",
            "current_torch_operation": "No resampling; operates at the input rate, normally 16 kHz",
            "mismatch": "Frequency grid, pre-emphasis coefficient, window samples, and quefrency bin spacing differ",
            "differentiability_treatment": "Resample with a fixed differentiable low-pass/interpolation operator or an equivalent fixed linear map; record its boundary convention",
        },
        {
            "exact_praat_operation": "Pitch floor 60 Hz gives a 3 / 60 = 50 ms effective Gaussian analysis window",
            "existing_numpy_approximation": "40 ms periodic Hann window",
            "current_torch_operation": "50 ms periodic Hann window at 16 kHz",
            "mismatch": "Window family and exact effective geometry differ; current Torch length is closer but not Gaussian",
            "differentiability_treatment": "Use a fixed Gaussian-like window with explicit sample-centre convention",
        },
        {
            "exact_praat_operation": "Frame centres separated by 0.002 s",
            "existing_numpy_approximation": "2 ms hop",
            "current_torch_operation": "2 ms hop, then optional frame subsampling via max_frames",
            "mismatch": "The hop agrees, but frame limits/subsampling can change the average",
            "differentiability_treatment": "Keep every frame for CPPS or use a fixed, documented frame subset only as a bounded diagnostic approximation",
        },
        {
            "exact_praat_operation": "Pre-emphasize from 50 Hz after resampling",
            "existing_numpy_approximation": "No pre-emphasis",
            "current_torch_operation": "Causal 50 Hz pre-emphasis at the original input rate after a separate high-pass/RMS preparation",
            "mismatch": "The filter rate and its placement differ; the NumPy path omits it",
            "differentiability_treatment": "Use the exact causal recurrence after the 10 kHz-equivalent stage; keep it differentiable",
        },
        {
            "exact_praat_operation": "Gaussian-window Spectrum, then log-power spectrum, inverse Fourier transform, and square to PowerCepstrum",
            "existing_numpy_approximation": "rFFT power plus real IFFT of log power; does not square the inverse result",
            "current_torch_operation": "rFFT power plus real IFFT of log power; adds a hard floor and does not square the inverse result",
            "mismatch": "PowerCepstrum units and normalization are not aligned; the existing 10 / ln(10) conversion is not an exact substitute",
            "differentiability_treatment": "Implement the fixed log-power floor and squared real inverse explicitly, then convert the resulting power to dB before prominence",
        },
        {
            "exact_praat_operation": "Smooth PowerCepstrogram before trend fitting: 0.01 s time and 0.001 s quefrency windows",
            "existing_numpy_approximation": "No cepstrogram smoothing; median-filter CPPS frame values with size 5 afterward",
            "current_torch_operation": "Average-pool five frames and 17 quefrency bins before peak/trend fitting",
            "mismatch": "Existing NumPy smooths the final scalar; current Torch is qualitatively closer but uses fixed kernels at 16 kHz",
            "differentiability_treatment": "Use explicit rectangular averaging with Praat-equivalent widths and edge handling; keep smoothing before topology/trend",
        },
        {
            "exact_praat_operation": "Search 60--330 Hz, tolerance 0.05, Parabolic interpolation",
            "existing_numpy_approximation": "Hard argmax in the band plus local parabolic value",
            "current_torch_operation": "Soft expectation or hard bin maximum; no parabolic correction in hard mode",
            "mismatch": "Hard mode misses the requested interpolation; soft expectation changes the peak topology and value",
            "differentiability_treatment": "Detach only the selected bin/topology and calculate the parabolic peak height from differentiable neighbouring values",
        },
        {
            "exact_praat_operation": "Straight trend over 0.001--0.05 s, incomplete Theil Robust fit",
            "existing_numpy_approximation": "Ordinary least squares over only the 60--330 Hz search band",
            "current_torch_operation": "Iteratively weighted least-squares-like line over only the search band",
            "mismatch": "Trend range, fit family, and peak influence differ; neither current path is exact Robust",
            "differentiability_treatment": "Freeze robust pair/order topology if needed, while retaining waveform gradients through the selected cepstral amplitudes and line evaluation",
        },
        {
            "exact_praat_operation": "Average the individual frame CPPS values after the configured smoothing/trend/peak operations",
            "existing_numpy_approximation": "Median-filter frame CPPS, then unweighted mean",
            "current_torch_operation": "Soft power/ZCR selection-weighted mean",
            "mismatch": "Frame inclusion and weighting differ; current selection can create large gradients unrelated to CPPS definition",
            "differentiability_treatment": "Use a fixed valid-frame mask and an unweighted mean; do not introduce an unrelated voicing weight into CPPS",
        },
    ]


def mapping_markdown(
    mapping: list[dict[str, str]],
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# CPPS Route C operation mapping and baseline parity",
        "",
        "## Scope",
        "",
        "This artifact audits CPPS only. Exact Praat output is the label authority; no generator optimizer step or formal pathology training was run.",
        "",
        f"- source commit: `{args.source_commit}`",
        f"- label bank: `{args.label_bank}`",
        f"- label bank SHA256: `{args.label_bank_sha256}`",
        f"- exact environment: `{sys.executable}`; Parselmouth `{parselmouth.__version__}`",
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
            "The runtime exact-vs-label-bank check is a data/scorer integrity check, not a candidate promotion gate. The NumPy and Torch rows quantify approximation mismatch on the same exact-preprocessed inputs. Candidate implementation is intentionally deferred until this mapping and parity evidence are reviewed.",
            "",
            "Praat references: [Sound: To PowerCepstrogram](https://praat.org/manual/Sound__To_PowerCepstrogram___.html), [Spectrum: To PowerCepstrum](https://praat.org/manual/Spectrum__To_PowerCepstrum.html), [PowerCepstrogram: Smooth](https://praat.org/manual/PowerCepstrogram__Smooth___.html), and [PowerCepstrogram: Get CPPS](https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
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
    rows = select_rows(read_rows(args.label_bank), splits, args.max_rows)

    # The exact AVQI checkout is selected at runtime so that this audit cannot
    # silently import a shadowed local copy.  These imports must therefore stay
    # after the explicit path insertion.
    sys.path.insert(0, str(args.avqi_root))
    from avqi_code.praat_version import concatenate_signals
    from avqi_code.praat_version import get_cpps as get_praat_cpps
    from avqi_code.praat_version import get_voiced_segments
    from avqi_code.praat_version import highpass_filter
    from avqi_code.praat_version import length_normalize_sv
    from avqi_code.python_version import get_cpps as get_numpy_cpps
    from avqi_code.python_version import read_and_resample_signal
    from model.avqi_components import PraatDifferentiableAVQIComponentEstimator

    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode=CURRENT_TORCH_PEAK_MODE,
    )
    records: list[dict[str, Any]] = []
    exact_runtime_values: list[float] = []
    numpy_values: list[float] = []
    torch_direct_values: list[float] = []
    torch_full_values: list[float] = []

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

        exact_runtime = float(get_praat_cpps(exact_input, SAMPLE_RATE))
        bank_value = float(row["cpps"])
        numpy_value = float(get_numpy_cpps(exact_input, SAMPLE_RATE))
        waveform = torch.from_numpy(exact_input.copy())
        with torch.inference_mode():
            torch_direct = float(estimator._cpps(waveform))
            torch_full = float(estimator.raw_components(waveform.unsqueeze(0))[0, 0])
        gradient_waveform = waveform.clone().requires_grad_()
        gradient_value = estimator._cpps(gradient_waveform)
        gradient = torch.autograd.grad(gradient_value, gradient_waveform)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite CPPS gradient at row {index}")

        record = {
            "row_index": index,
            "speaker_id": row["speaker_id"],
            "sample_id": row.get("sample_id", ""),
            "split": row["split"],
            "condition_id": row["condition_id"],
            "view": row["view"],
            "sample_count": int(exact_input.size),
            "bank_cpps": bank_value,
            "exact_runtime_cpps": exact_runtime,
            "exact_runtime_minus_bank": exact_runtime - bank_value,
            "numpy_cpps": numpy_value,
            "numpy_minus_exact_runtime": numpy_value - exact_runtime,
            "torch_direct_current_cpps": torch_direct,
            "torch_direct_minus_exact_runtime": torch_direct - exact_runtime,
            "torch_full_current_cpps": torch_full,
            "torch_full_minus_exact_runtime": torch_full - exact_runtime,
            "torch_direct_input_gradient_norm": float(gradient.norm()),
            "torch_direct_input_gradient_max_abs": float(gradient.abs().max()),
        }
        records.append(record)
        exact_runtime_values.append(exact_runtime)
        numpy_values.append(numpy_value)
        torch_direct_values.append(torch_direct)
        torch_full_values.append(torch_full)
        print(
            f"row={index + 1}/{len(rows)} speaker={row['speaker_id']} "
            f"split={row['split']} view={row['view']} exact={exact_runtime:.6f}",
            flush=True,
        )

    bank_errors = [
        float(record["exact_runtime_minus_bank"])
        for record in records
    ]
    summary = {
        "rows": len(records),
        "splits": list(splits),
        "exact_bank_max_absolute_error": float(np.max(np.abs(bank_errors))),
        "exact_bank_reproduction": (
            "PASS" if np.max(np.abs(bank_errors)) <= EXACT_BANK_TOLERANCE else "FAIL"
        ),
        "candidate_metrics": {
            "existing_numpy": summarize_pairs(exact_runtime_values, numpy_values),
            "current_torch_direct_on_exact_input": summarize_pairs(
                exact_runtime_values,
                torch_direct_values,
            ),
            "current_torch_full_raw_components": summarize_pairs(
                exact_runtime_values,
                torch_full_values,
            ),
        },
        "gradient": {
            "min_norm": float(
                min(record["torch_direct_input_gradient_norm"] for record in records)
            ),
            "median_norm": float(
                np.median(
                    [record["torch_direct_input_gradient_norm"] for record in records]
                )
            ),
            "max_norm": float(
                max(record["torch_direct_input_gradient_norm"] for record in records)
            ),
        },
    }
    mapping = operation_mapping()
    report = {
        "schema_version": "avqi-route-c-cpps-parity-v1",
        "decision": "BASELINE_PARITY_COMPLETE",
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "source_commit": args.source_commit,
        "label_bank": str(args.label_bank),
        "label_bank_sha256": actual_label_hash,
        "exact_environment": {
            "python": sys.executable,
            "parselmouth": parselmouth.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "operation_mapping": mapping,
        "summary": summary,
        "records": records,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "cpps_parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cpps_parity_rows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "cpps_operation_mapping.md").write_text(
        mapping_markdown(mapping, args=args, summary=summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
