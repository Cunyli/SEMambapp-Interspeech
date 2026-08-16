#!/usr/bin/env python3
"""Exact-score VCTK v4 audio and merge the internal rows with the v2 bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import parselmouth
import soundfile as sf

from avqi_code import run_avqi


METRICS = (
    "avqi",
    "cpps",
    "hnr",
    "jitter_local",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
STEP_VERSIONS = {
    "highpass": "praat",
    "read_and_resample": "praat",
    "sv_length_norm": "praat",
    "cs_voiced_segments": "praat",
    "concatenate": "praat",
    "cpps": "praat",
    "slope": "praat",
    "tilt": "praat",
    "shimmer": "praat",
    "hnr": "praat",
    "pitch": "praat",
}
VCTK_SPLIT_COUNTS = {
    "surrogate_train": 72,
    "surrogate_calibration": 12,
    "surrogate_holdout": 12,
    "vctk_external": 12,
}


@dataclass(frozen=True)
class ScoringTask:
    speaker_id: str
    sample_id: str
    split: str
    condition: str
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-label-bank", type=Path, required=True)
    parser.add_argument("--base-label-bank-sha256", required=True)
    parser.add_argument("--vctk-metadata", type=Path, required=True)
    parser.add_argument("--vctk-metadata-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exact-runner", type=Path, required=True)
    parser.add_argument("--exact-runner-sha256", required=True)
    parser.add_argument("--avqi-main", type=Path, required=True)
    parser.add_argument("--avqi-main-sha256", required=True)
    parser.add_argument("--expected-base-rows", type=int, default=918)
    parser.add_argument("--expected-vctk-rows", type=int, default=1_728)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-split-condition-coverage", type=float, default=0.90)
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
        raise ValueError(f"source hash mismatch for {path}: {actual} != {expected}")
    return actual


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows, list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def score_task(task: ScoringTask) -> dict[str, Any]:
    row: dict[str, Any] = asdict(task)
    try:
        metrics = run_avqi(
            task.path,
            task.path,
            target_sr=16_000,
            speaking_type="cs",
            step_versions=STEP_VERSIONS,
            remove_sv_silence_with_sox=False,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        subprocess.CalledProcessError,
        parselmouth.PraatError,
    ) as error:
        row.update(
            {
                "scoring_status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        )
        for metric in METRICS:
            row[metric] = ""
        return row
    row.update({"scoring_status": "ok", "error_type": "", "error_message": ""})
    for metric in METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric}: {task}")
        row[metric] = value
    return row


def coverage(rows: list[dict[str, Any]]) -> float:
    return sum(row["scoring_status"] == "ok" for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_hashes = {
        "base_label_bank": validate_hash(
            args.base_label_bank, args.base_label_bank_sha256
        ),
        "vctk_metadata": validate_hash(
            args.vctk_metadata, args.vctk_metadata_sha256
        ),
        "exact_runner": validate_hash(args.exact_runner, args.exact_runner_sha256),
        "avqi_main": validate_hash(args.avqi_main, args.avqi_main_sha256),
    }
    base_rows, base_fields = read_csv(args.base_label_bank)
    vctk_rows, _ = read_csv(args.vctk_metadata)
    if len(base_rows) != args.expected_base_rows:
        raise ValueError(f"expected {args.expected_base_rows} base rows, found {len(base_rows)}")
    if len(vctk_rows) != args.expected_vctk_rows:
        raise ValueError(f"expected {args.expected_vctk_rows} VCTK rows, found {len(vctk_rows)}")
    row_keys = [
        (row["speaker_id"], row["sample_id"], row["condition_id"])
        for row in vctk_rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("duplicate VCTK speaker/sample/condition rows")
    speaker_sets = {
        split: {row["speaker_id"] for row in vctk_rows if row["split"] == split}
        for split in VCTK_SPLIT_COUNTS
    }
    actual_split_counts = {split: len(speakers) for split, speakers in speaker_sets.items()}
    if actual_split_counts != VCTK_SPLIT_COUNTS:
        raise ValueError(f"VCTK speaker split mismatch: {actual_split_counts}")
    for first, first_speakers in speaker_sets.items():
        for second, second_speakers in speaker_sets.items():
            if first < second and first_speakers & second_speakers:
                raise ValueError(f"VCTK speaker leakage between {first} and {second}")
    base_speakers = {row["speaker_id"] for row in base_rows}
    vctk_speakers = set().union(*speaker_sets.values())
    if base_speakers & vctk_speakers:
        raise ValueError("VCTK speakers overlap the base label bank")

    tasks: list[ScoringTask] = []
    metadata_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in vctk_rows:
        key = (row["speaker_id"], row["sample_id"], row["condition_id"])
        path = Path(row["audio_path"])
        if not path.is_file() or sha256_file(path) != row["audio_sha256"]:
            raise ValueError(f"VCTK audio receipt mismatch: {path}")
        info = sf.info(path)
        if info.samplerate != 16_000 or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"invalid VCTK output audio: {path}: {info}")
        metadata_by_key[key] = row
        tasks.append(
            ScoringTask(
                speaker_id=row["speaker_id"],
                sample_id=row["sample_id"],
                split=row["split"],
                condition=row["condition_id"],
                path=str(path.resolve()),
            )
        )
    if args.workers == 1:
        scored_rows = [score_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            scored_rows = list(executor.map(score_task, tasks))
    overall_coverage = coverage(scored_rows)
    if overall_coverage < args.minimum_coverage:
        raise ValueError(
            f"exact-score coverage {overall_coverage:.6f} below {args.minimum_coverage}"
        )
    slice_coverage: dict[str, float] = {}
    for split in VCTK_SPLIT_COUNTS:
        for condition in ("clean", "rir_only", "snr20", "snr10"):
            selected = [
                row for row in scored_rows
                if row["split"] == split and row["condition"] == condition
            ]
            value = coverage(selected)
            slice_coverage[f"{split}/{condition}"] = value
            if value < args.minimum_split_condition_coverage:
                raise ValueError(
                    f"slice coverage {split}/{condition}={value:.6f} below gate"
                )

    fields = list(base_fields)
    fields.insert(fields.index("pair_id") + 1, "sample_id")
    normalized_base: list[dict[str, Any]] = []
    for row in base_rows:
        normalized = dict(row)
        normalized["sample_id"] = row.get("pair_id") or row["speaker_id"]
        normalized_base.append(normalized)
    converted: list[dict[str, Any]] = []
    for scored in scored_rows:
        key = (scored["speaker_id"], scored["sample_id"], scored["condition"])
        metadata = metadata_by_key[key]
        path = Path(scored["path"])
        native_snr = {
            "clean": "",
            "rir_only": "",
            "snr20": 20,
            "snr10": 10,
        }[scored["condition"]]
        row: dict[str, Any] = {field: "" for field in fields}
        row.update(
            {
                "schema_version": "avqi_component_label_bank_v4_vctk",
                "speaker_id": scored["speaker_id"],
                "pair_id": scored["sample_id"],
                "sample_id": scored["sample_id"],
                "condition_id": scored["condition"],
                "view": "cs",
                "sample_group": "healthy_vctk",
                "label": "healthy",
                "source": "VCTK",
                "split_version": "avqi-component-phaseaware-v4",
                "split": scored["split"],
                "cs_uid": scored["sample_id"],
                "cs_path": str(path.resolve()),
                "cs_sha256": metadata["audio_sha256"],
                "cs_sample_rate": 16_000,
                "cs_frames": metadata["frames"],
                "sv_uid": scored["sample_id"],
                "sv_path": str(path.resolve()),
                "sv_sha256": metadata["audio_sha256"],
                "sv_sample_rate": 16_000,
                "sv_frames": metadata["frames"],
                "cs_native_snr_db": native_snr,
                "sv_native_snr_db": native_snr,
                "same_noise_rir_seed_across_cs_sv": 1,
                "target_sr": 16_000,
                "speaking_type": "cs",
                "all_praat": 1,
                "remove_sv_silence_with_sox": 0,
                "scoring_status": scored["scoring_status"],
                "error_type": scored["error_type"],
                "error_message": scored["error_message"],
            }
        )
        for metric in METRICS:
            row[metric] = scored[metric]
        converted.append(row)

    internal_vctk = [row for row in converted if row["split"] != "vctk_external"]
    external_vctk = [row for row in converted if row["split"] == "vctk_external"]
    expected_internal = 96 * 4 * 4
    expected_external = 12 * 4 * 4
    if len(internal_vctk) != expected_internal or len(external_vctk) != expected_external:
        raise ValueError(
            f"unexpected internal/external rows: {len(internal_vctk)}/{len(external_vctk)}"
        )
    args.output_dir.mkdir(parents=True)
    internal_path = args.output_dir / "exact_component_label_bank_v4.csv"
    external_path = args.output_dir / "vctk_external_exact_components_v4.csv"
    write_csv(internal_path, normalized_base + internal_vctk, fields)
    write_csv(external_path, external_vctk, fields)
    receipt = {
        "schema_version": "avqi-component-label-bank-v4",
        "source_hashes": source_hashes,
        "base_rows": len(base_rows),
        "vctk_scored_rows": len(scored_rows),
        "vctk_valid_rows": sum(row["scoring_status"] == "ok" for row in scored_rows),
        "vctk_overall_coverage": overall_coverage,
        "vctk_split_condition_coverage": slice_coverage,
        "internal_vctk_rows": len(internal_vctk),
        "external_vctk_rows": len(external_vctk),
        "merged_internal_rows": len(normalized_base) + len(internal_vctk),
        "speaker_overlap_with_base": 0,
        "scoring_errors": dict(Counter(
            row["error_type"] for row in scored_rows if row["scoring_status"] != "ok"
        )),
        "internal_label_bank": str(internal_path.resolve()),
        "internal_label_bank_sha256": sha256_file(internal_path),
        "external_label_bank": str(external_path.resolve()),
        "external_label_bank_sha256": sha256_file(external_path),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
