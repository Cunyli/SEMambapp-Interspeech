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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_rows(args.label_bank, args.label_bank_sha256)
    records = select_records(rows, args.max_speakers)
    items = [
        {"id": f"{record['speaker_id']}:{record['condition']}", "path": record["path"]}
        for record in records
    ]
    exact = run_exact(items, args.exact_python)
    exact_index = {row["id"]: row for row in exact["rows"]}
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode=args.shimmer_mode,
    ).eval()
    result_rows: list[dict[str, Any]] = []
    for record in records:
        identifier = f"{record['speaker_id']}:{record['condition']}"
        exact_row = exact_index[identifier]
        audio = load_audio(record)
        waveform = torch.from_numpy(audio)
        prepared = estimator._prepare(waveform)
        if prepared.numel() > METRIC_SAMPLE_COUNT:
            prepared = prepared[-METRIC_SAMPLE_COUNT:]
        current_pulses = estimator._praat_shimmer_pulse_chain(prepared).detach().cpu().numpy()
        exact_pulses = np.asarray(exact_row["pulse_positions_samples"], dtype=np.float64)
        current_pulse_db = float(
            estimator.raw_shimmer_from_pulse_positions(
                waveform,
                torch.from_numpy(current_pulses.astype(np.float32)),
                metric_sample_count=METRIC_SAMPLE_COUNT,
            )[1]
        )
        exact_pulse_db = float(
            estimator.raw_shimmer_from_pulse_positions(
                waveform,
                torch.from_numpy(exact_pulses.astype(np.float32)),
                metric_sample_count=METRIC_SAMPLE_COUNT,
            )[1]
        )
        exact_to_current = nearest_errors(exact_pulses, current_pulses)
        current_to_exact = nearest_errors(current_pulses, exact_pulses)
        reference_period = float(np.median(np.diff(exact_pulses))) if exact_pulses.size > 1 else 0.0
        tolerance = 0.25 * reference_period
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
                "v6_shimmer_db_with_v6_pulses": current_pulse_db,
                "v6_shimmer_db_with_exact_pulses": exact_pulse_db,
                "v6_pulse_db_gap": abs(current_pulse_db - float(exact_row["shimmer_db"])),
                "exact_pulse_db_gap": abs(exact_pulse_db - float(exact_row["shimmer_db"])),
            }
        )

    def median(field: str) -> float:
        return float(np.median([row[field] for row in result_rows]))

    report = {
        "schema_version": "avqi-route-c-shimmer-vctk-topology-audit-v1",
        "decision": "COMPLETED_SHIMMER_TOPOLOGY_DIAGNOSTIC_NO_PROMOTION",
        "source_commit": args.source_commit,
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
