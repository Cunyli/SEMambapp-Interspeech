#!/usr/bin/env python3
"""Compare the deployable v6 Shimmer pulse path with exact Praat pulses.

This is a read-only Route C Shimmer dB diagnosis.  It freezes exact Praat
pulses as the authority, extracts the current v6 detached pulse chain from
the same waveform, and compares pulse timing plus the same differentiable
Hann-RMS dB tier under both topologies.  It does not run CPPS/HNR, alter a
model, change gates, or authorize a waveform pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator
from scripts.evaluate_avqi_shimmer_pulse_oracle_pilot import EXACT_SCORER


SAMPLE_RATE = 16_000
METRIC_SAMPLE_COUNT = 3 * SAMPLE_RATE
CONFIDENCE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MATERIAL_DB_GAP = 0.05
CONFIDENCE_MATCH_AUC_GATE = 0.65
CONFIDENCE_RETAINED_FRACTION_GATE = 0.75
CONFIDENCE_EXACT_COVERAGE_GATE = 0.90
CONFIDENCE_MEDIAN_GAP_INCREASE_MAX = 0.005
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument(
        "--shimmer-mode",
        choices=("praat_pulse_chain_v5", "praat_pulse_path_v6"),
        default="praat_pulse_path_v6",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", default="")
    parser.add_argument("--device", default="cpu")
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


def read_rows(path: Path, expected_hash: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_hash:
        raise ValueError("VCTK label-bank SHA-256 mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "speaker_id",
        "sample_id",
        "condition_id",
        "view",
        "split",
        "scoring_status",
        "cs_path",
        "cs_sha256",
    }
    if not rows or not required <= rows[0].keys():
        raise ValueError("unexpected VCTK label-bank schema")
    return rows


def select_records(
    rows: list[dict[str, str]], max_speakers: int
) -> list[dict[str, str]]:
    external = [
        row
        for row in rows
        if row["split"] == "vctk_external"
        and row["view"] == "cs"
        and row["scoring_status"] == "ok"
    ]
    speakers = sorted({row["speaker_id"] for row in external})
    if max_speakers < 1 or max_speakers > len(speakers):
        raise ValueError(f"invalid max-speakers={max_speakers}")
    by_key = {(row["speaker_id"], row["sample_id"], row["condition_id"]): row for row in external}
    records: list[dict[str, str]] = []
    for speaker in speakers[:max_speakers]:
        clean_rows = sorted(
            (row for row in external if row["speaker_id"] == speaker and row["condition_id"] == "clean"),
            key=lambda row: row["sample_id"],
        )
        if len(clean_rows) != 4:
            raise ValueError(f"expected four clean rows for {speaker}")
        target = clean_rows[0]
        for condition in ("rir_only", "snr20"):
            candidate = by_key.get((speaker, target["sample_id"], condition))
            if candidate is None:
                raise ValueError(f"missing paired record {speaker}/{target['sample_id']}/{condition}")
            records.append(
                {
                    "speaker_id": speaker,
                    "sample_id": target["sample_id"],
                    "condition": condition,
                    "path": candidate["cs_path"],
                    "audio_sha256": candidate["cs_sha256"],
                }
            )
    return records


def load_audio(record: dict[str, str]) -> np.ndarray:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["audio_sha256"]:
        raise ValueError(f"VCTK audio hash mismatch: {path}")
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"invalid mono 16 kHz VCTK waveform: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite VCTK waveform: {path}")
    return audio


def run_exact(items: list[dict[str, str]], exact_python: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, ""],
        input=json.dumps({"items": items, "include_pulses": True}, sort_keys=True),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_SHIMMER_EXACT_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"exact Shimmer scorer emitted {len(lines)} JSON records")
    return json.loads(lines[0][len(marker) :])


def nearest_errors(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.size == 0 or candidate.size == 0:
        return np.full(reference.shape, np.inf, dtype=np.float64)
    return np.abs(reference[:, None] - candidate[None, :]).min(axis=1)


def binary_rank_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Return tie-aware rank AUC without adding a statistics dependency."""
    if scores.ndim != 1 or positive.shape != scores.shape:
        raise ValueError("confidence AUC expects aligned one-dimensional arrays")
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[positive].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def optional_median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def validate_fresh_output_dir(output_dir: Path, slurm_job_id: str) -> None:
    if not output_dir.exists():
        return
    if not slurm_job_id or not output_dir.is_dir() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    entries = list(output_dir.iterdir())
    if len(entries) != 1 or entries[0].name != "logs":
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    log_dir = entries[0]
    if not log_dir.is_dir() or log_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    allowed_logs = {
        f"slurm_{slurm_job_id}.out",
        f"slurm_{slurm_job_id}.err",
        f"shimmer_confidence_{slurm_job_id}.log",
    }
    if any(
        child.name not in allowed_logs or not child.is_file() or child.is_symlink()
        for child in log_dir.iterdir()
    ):
        raise FileExistsError(f"refusing to overwrite {output_dir}")


def main() -> None:
    args = parse_args()
    validate_fresh_output_dir(args.output_dir, args.slurm_job_id)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = read_rows(args.label_bank, args.label_bank_sha256)
    records = select_records(rows, args.max_speakers)
    items = [
        {"id": f"{record['speaker_id']}:{record['condition']}", "path": record["path"]}
        for record in records
    ]
    exact = run_exact(items, args.exact_python)
    print(f"event=exact_complete records={len(exact['rows'])}", flush=True)
    exact_index = {row["id"]: row for row in exact["rows"]}
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode=args.shimmer_mode,
    ).to(device).eval()
    result_rows: list[dict[str, Any]] = []
    all_confidence: list[np.ndarray] = []
    all_matches: list[np.ndarray] = []
    for record in records:
        identifier = f"{record['speaker_id']}:{record['condition']}"
        print(f"event=record_start id={identifier}", flush=True)
        exact_row = exact_index[identifier]
        audio = load_audio(record)
        waveform = torch.from_numpy(audio).to(device)
        prepared = estimator._prepare(waveform)
        if prepared.numel() > METRIC_SAMPLE_COUNT:
            prepared = prepared[-METRIC_SAMPLE_COUNT:]
        current_pulse_tensor, current_confidence_tensor = (
            estimator._praat_shimmer_pulse_chain_with_confidence(prepared)
        )
        current_pulses = current_pulse_tensor.detach().cpu().numpy()
        current_confidence = current_confidence_tensor.detach().cpu().numpy()
        if current_confidence.shape != current_pulses.shape:
            raise RuntimeError("pulse confidence shape does not match pulse positions")
        exact_pulses = np.asarray(exact_row["pulse_positions_samples"], dtype=np.float64)
        if current_pulses.size == 0 or exact_pulses.size == 0:
            raise RuntimeError(f"empty pulse topology for {identifier}")
        current_pulse_db = float(
            estimator.raw_shimmer_from_pulse_positions(
                waveform,
                torch.from_numpy(current_pulses.astype(np.float32)).to(device),
                metric_sample_count=METRIC_SAMPLE_COUNT,
            )[1]
        )
        exact_pulse_db = float(
            estimator.raw_shimmer_from_pulse_positions(
                waveform,
                torch.from_numpy(exact_pulses.astype(np.float32)).to(device),
                metric_sample_count=METRIC_SAMPLE_COUNT,
            )[1]
        )
        exact_to_current = nearest_errors(exact_pulses, current_pulses)
        current_to_exact = nearest_errors(current_pulses, exact_pulses)
        reference_period = float(np.median(np.diff(exact_pulses))) if exact_pulses.size > 1 else 0.0
        tolerance = 0.25 * reference_period
        current_matches = current_to_exact <= tolerance
        all_confidence.append(current_confidence)
        all_matches.append(current_matches)
        threshold_diagnostics: list[dict[str, Any]] = []
        for threshold in CONFIDENCE_THRESHOLDS:
            retained = current_confidence >= threshold
            retained_pulses = current_pulses[retained]
            exact_to_retained = nearest_errors(exact_pulses, retained_pulses)
            retained_db = float(
                estimator.raw_shimmer_from_pulse_positions(
                    waveform,
                    torch.from_numpy(retained_pulses.astype(np.float32)).to(device),
                    metric_sample_count=METRIC_SAMPLE_COUNT,
                )[1]
            )
            if threshold == CONFIDENCE_THRESHOLDS[0] and not np.isclose(
                retained_db,
                current_pulse_db,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("baseline confidence threshold changed v6 shimmer")
            threshold_diagnostics.append(
                {
                    "threshold": threshold,
                    "retained_count": int(retained.sum()),
                    "retained_fraction": float(np.mean(retained)),
                    "retained_match_fraction": float(
                        np.mean(current_matches[retained])
                    ),
                    "exact_coverage_fraction": float(
                        np.mean(exact_to_retained <= tolerance)
                    ),
                    "shimmer_db": retained_db,
                    "shimmer_db_gap": abs(
                        retained_db - float(exact_row["shimmer_db"])
                    ),
                }
            )
        unmatched_confidence = current_confidence[~current_matches]
        result_rows.append(
            {
                "speaker_id": record["speaker_id"],
                "sample_id": record["sample_id"],
                "condition": record["condition"],
                "path": record["path"],
                "exact_shimmer_db": float(exact_row["shimmer_db"]),
                "exact_pulse_count": int(exact_row["pulse_count"]),
                "v6_pulse_count": int(current_pulses.size),
                "exact_to_v6_median_abs_samples": float(np.median(exact_to_current)),
                "v6_to_exact_median_abs_samples": float(np.median(current_to_exact)),
                "exact_to_v6_fraction_within_quarter_period": float(
                    np.mean(exact_to_current <= tolerance)
                ),
                "v6_to_exact_fraction_within_quarter_period": float(
                    np.mean(current_matches)
                ),
                "v6_pulse_confidence_median": float(
                    np.median(current_confidence)
                ),
                "v6_matched_pulse_confidence_median": optional_median(
                    current_confidence[current_matches]
                ),
                "v6_unmatched_pulse_confidence_median": optional_median(
                    unmatched_confidence
                ),
                "v6_shimmer_db_with_v6_pulses": current_pulse_db,
                "v6_shimmer_db_with_exact_pulses": exact_pulse_db,
                "v6_pulse_db_gap": abs(current_pulse_db - float(exact_row["shimmer_db"])),
                "exact_pulse_db_gap": abs(exact_pulse_db - float(exact_row["shimmer_db"])),
                "confidence_threshold_diagnostics": threshold_diagnostics,
            }
        )
        print(f"event=record_complete id={identifier}", flush=True)

    def median(field: str) -> float:
        return float(np.median([row[field] for row in result_rows]))

    confidence = np.concatenate(all_confidence)
    matches = np.concatenate(all_matches)
    threshold_aggregates: list[dict[str, Any]] = []
    for threshold in CONFIDENCE_THRESHOLDS:
        diagnostics = [
            next(
                item
                for item in row["confidence_threshold_diagnostics"]
                if item["threshold"] == threshold
            )
            for row in result_rows
        ]
        material = [
            (item, row)
            for item, row in zip(diagnostics, result_rows)
            if row["v6_pulse_db_gap"] >= MATERIAL_DB_GAP
        ]
        threshold_aggregates.append(
            {
                "threshold": threshold,
                "median_retained_fraction": float(
                    np.median([item["retained_fraction"] for item in diagnostics])
                ),
                "median_retained_match_fraction": float(
                    np.median(
                        [item["retained_match_fraction"] for item in diagnostics]
                    )
                ),
                "median_exact_coverage_fraction": float(
                    np.median(
                        [item["exact_coverage_fraction"] for item in diagnostics]
                    )
                ),
                "median_shimmer_db_gap": float(
                    np.median([item["shimmer_db_gap"] for item in diagnostics])
                ),
                "median_gap_change_from_v6": float(
                    np.median(
                        [
                            item["shimmer_db_gap"] - row["v6_pulse_db_gap"]
                            for item, row in zip(diagnostics, result_rows)
                        ]
                    )
                ),
                "improved_rows": int(
                    sum(
                        item["shimmer_db_gap"] < row["v6_pulse_db_gap"]
                        for item, row in zip(diagnostics, result_rows)
                    )
                ),
                "material_outlier_rows": len(material),
                "material_outlier_improved_rows": sum(
                    item["shimmer_db_gap"] < row["v6_pulse_db_gap"]
                    for item, row in material
                ),
            }
        )

    confidence_match_auc = binary_rank_auc(confidence, matches)
    viable_thresholds = [
        item["threshold"]
        for item in threshold_aggregates
        if item["threshold"] > CONFIDENCE_THRESHOLDS[0]
        and item["median_retained_fraction"]
        >= CONFIDENCE_RETAINED_FRACTION_GATE
        and item["median_exact_coverage_fraction"]
        >= CONFIDENCE_EXACT_COVERAGE_GATE
        and item["median_gap_change_from_v6"]
        <= CONFIDENCE_MEDIAN_GAP_INCREASE_MAX
        and item["material_outlier_rows"] > 0
        and item["material_outlier_improved_rows"]
        == item["material_outlier_rows"]
    ]
    confidence_hypothesis_supported = (
        confidence_match_auc is not None
        and confidence_match_auc >= CONFIDENCE_MATCH_AUC_GATE
        and bool(viable_thresholds)
    )

    report = {
        "schema_version": "avqi-route-c-shimmer-vctk-topology-audit-v2",
        "decision": "COMPLETED_SHIMMER_TOPOLOGY_DIAGNOSTIC_NO_PROMOTION",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id or None,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "source_files_sha256": {
            "model/avqi_components.py": sha256_file(
                REPO_ROOT / "model/avqi_components.py"
            ),
            "scripts/evaluate_avqi_shimmer_vctk_topology_audit.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "label_bank": str(args.label_bank.resolve()),
        "label_bank_sha256": args.label_bank_sha256,
        "exact_python": str(args.exact_python),
        "parselmouth_version": exact["parselmouth_version"],
        "praat_version": exact["praat_version"],
        "records": len(result_rows),
        "speaker_count": args.max_speakers,
        "shimmer_mode": args.shimmer_mode,
        "conditions": ["rir_only", "snr20"],
        "aggregates": {
            "median_exact_to_v6_abs_samples": median("exact_to_v6_median_abs_samples"),
            "median_exact_to_v6_fraction_within_quarter_period": median(
                "exact_to_v6_fraction_within_quarter_period"
            ),
            "median_v6_shimmer_db_gap": median("v6_pulse_db_gap"),
            "median_exact_pulse_shimmer_db_gap": median("exact_pulse_db_gap"),
            "exact_pulse_gap_minus_v6_gap": median("exact_pulse_db_gap")
            - median("v6_pulse_db_gap"),
            "confidence_match_auc": confidence_match_auc,
            "matched_pulse_confidence_median": optional_median(
                confidence[matches]
            ),
            "unmatched_pulse_confidence_median": optional_median(
                confidence[~matches]
            ),
            "confidence_thresholds": threshold_aggregates,
        },
        "confidence_hypothesis": {
            "decision": (
                "SUPPORTED_FOR_CALIBRATION_ONLY_FOLLOWUP"
                if confidence_hypothesis_supported
                else "FALSIFIED_NO_CONFIDENCE_CANDIDATE"
            ),
            "viable_thresholds_not_selected": viable_thresholds,
            "gates": {
                "confidence_match_auc_min": CONFIDENCE_MATCH_AUC_GATE,
                "median_retained_fraction_min": CONFIDENCE_RETAINED_FRACTION_GATE,
                "median_exact_coverage_fraction_min": CONFIDENCE_EXACT_COVERAGE_GATE,
                "median_gap_increase_max_db": CONFIDENCE_MEDIAN_GAP_INCREASE_MAX,
                "material_db_gap_min": MATERIAL_DB_GAP,
                "all_material_outliers_must_improve": True,
            },
            "interpretation": (
                "A pass authorizes only a fresh calibration-only threshold study; "
                "it is not a scorer, external, waveform, or training promotion."
            ),
        },
        "rows": result_rows,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "scope": "Shimmer pulse topology only; CPPS/HNR untouched",
    }
    output_root = args.output_dir / "outputs"
    output_root.mkdir(parents=True)
    report_path = output_root / "diagnostic_report.json"
    write_json(report_path, report)
    write_json(output_root / "predictions.json", result_rows)
    receipt = {
        "decision": report["decision"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "predictions.json": sha256_file(output_root / "predictions.json"),
        },
    }
    write_json(output_root / "completion_receipt.json", receipt)
    (output_root / "SUMMARY.md").write_text(
        "# Shimmer dB VCTK pulse topology audit\n\n"
        f"Decision: `{report['decision']}`\n\n"
        "This diagnostic does not authorize a waveform pilot or generator update.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
