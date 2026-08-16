#!/usr/bin/env python3
"""Cross-check AVQI v4 data receipts and write one scorer-ready receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPEAKERS = {
    "surrogate_train": 72,
    "surrogate_calibration": 12,
    "surrogate_holdout": 12,
    "vctk_external": 12,
}
EXPECTED_CONDITIONS = ("clean", "rir_only", "snr20", "snr10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--label-receipt", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--data-source-commit", required=True)
    parser.add_argument("--finalizer-source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_artifact(path: str, expected_hash: str) -> None:
    artifact = Path(path)
    if not artifact.is_file() or sha256_file(artifact) != expected_hash:
        raise ValueError(f"artifact hash mismatch: {artifact}")


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_path}")
    if repository_head() != args.finalizer_source_commit:
        raise ValueError("declared finalizer commit differs from repository HEAD")
    if not args.slurm_job_id.isdigit():
        raise ValueError(f"invalid Slurm job ID: {args.slurm_job_id}")
    data = load_json(args.data_receipt)
    labels = load_json(args.label_receipt)
    if data["schema_version"] != "avqi-component-vctk-v4":
        raise ValueError("unexpected VCTK data receipt schema")
    if labels["schema_version"] != "avqi-component-label-bank-v4":
        raise ValueError("unexpected exact label receipt schema")
    if data["speaker_counts"] != EXPECTED_SPEAKERS:
        raise ValueError(f"speaker split mismatch: {data['speaker_counts']}")
    if data["speaker_overlap"] != 0:
        raise ValueError("VCTK speaker splits overlap")
    if tuple(data["conditions"]) != EXPECTED_CONDITIONS:
        raise ValueError(f"condition contract differs: {data['conditions']}")
    if data["row_count"] != 1_824:
        raise ValueError(f"expected 1824 prepared candidate rows: {data['row_count']}")
    if data["utterances_per_speaker"] != 4:
        raise ValueError("selected utterance count differs from frozen contract")
    if data["external_reserve_utterances_per_speaker"] != 2:
        raise ValueError("external reserve utterance count differs")
    expected_candidate_rows_by_split = {
        "surrogate_train": 1_152,
        "surrogate_calibration": 192,
        "surrogate_holdout": 192,
        "vctk_external": 288,
    }
    if data["row_counts_by_split"] != expected_candidate_rows_by_split:
        raise ValueError(
            f"candidate split row counts differ: {data['row_counts_by_split']}"
        )
    expected_candidate_rows_by_condition = {
        condition: 456 for condition in EXPECTED_CONDITIONS
    }
    if data["row_counts_by_condition"] != expected_candidate_rows_by_condition:
        raise ValueError(
            "candidate condition row counts differ: "
            f"{data['row_counts_by_condition']}"
        )
    if data["full_band_audio_preserved"] is not True:
        raise ValueError("prepared audio is not declared full-band")
    if data["avqi_metric_branch_highpass_applied"] is not False:
        raise ValueError("prepared waveform unexpectedly contains AVQI high-pass")
    if data["max_open_shards"] != 4:
        raise ValueError(f"tar cache contract differs: {data['max_open_shards']}")
    validate_artifact(data["metadata_csv"], data["metadata_sha256"])
    if labels["source_hashes"]["vctk_metadata"] != data["metadata_sha256"]:
        raise ValueError("label scorer did not consume the prepared metadata receipt")
    expected_label_counts = {
        "base_rows": 918,
        "vctk_candidate_scored_rows": 1_824,
        "vctk_scored_rows": 1_728,
        "internal_vctk_rows": 1_536,
        "external_vctk_rows": 192,
        "merged_internal_rows": 2_454,
        "speaker_overlap_with_base": 0,
    }
    for key, expected in expected_label_counts.items():
        if labels[key] != expected:
            raise ValueError(f"label count {key}={labels[key]} != {expected}")
    if labels["vctk_valid_rows"] / labels["vctk_scored_rows"] < 0.95:
        raise ValueError("valid exact-score fraction is below 0.95")
    if labels["vctk_overall_coverage"] < 0.95:
        raise ValueError("reported exact-score coverage is below 0.95")
    expected_slice_keys = {
        f"{split}/{condition}"
        for split in EXPECTED_SPEAKERS
        for condition in EXPECTED_CONDITIONS
    }
    observed_slices = labels["vctk_split_condition_coverage"]
    if set(observed_slices) != expected_slice_keys:
        raise ValueError("split-condition coverage keys differ")
    failed_slices = {
        key: value for key, value in observed_slices.items() if value < 0.90
    }
    if failed_slices:
        raise ValueError(f"split-condition exact coverage failed: {failed_slices}")
    external_selection = labels["external_selection"]
    if external_selection["metric_values_used_for_selection"] is not False:
        raise ValueError("external reserve selection used component values")
    if external_selection["speaker_count"] != 12:
        raise ValueError("external reserve selection speaker count differs")
    if external_selection["candidate_external_rows"] != 288:
        raise ValueError("external reserve candidate row count differs")
    if external_selection["selected_external_rows"] != 192:
        raise ValueError("external selected row count differs")
    if external_selection["selected_external_valid_rows"] != 192:
        raise ValueError("external selected rows are not all exact-valid")
    validate_artifact(
        labels["internal_label_bank"],
        labels["internal_label_bank_sha256"],
    )
    validate_artifact(
        labels["external_label_bank"],
        labels["external_label_bank_sha256"],
    )
    receipt = {
        "schema_version": "avqi-component-v4-data-completion-v1",
        "decision": "DATA_READY_FOR_SCORER_SCREENS",
        "slurm_job_id": args.slurm_job_id,
        "data_source_commit": args.data_source_commit,
        "finalizer_source_commit": args.finalizer_source_commit,
        "prepared_candidate_rows": data["row_count"],
        "prepared_rows": labels["vctk_scored_rows"],
        "exact_scored_rows": labels["vctk_scored_rows"],
        "exact_valid_rows": labels["vctk_valid_rows"],
        "exact_coverage": labels["vctk_overall_coverage"],
        "split_condition_coverage": observed_slices,
        "speaker_counts": data["speaker_counts"],
        "speaker_overlap": 0,
        "merged_internal_rows": labels["merged_internal_rows"],
        "external_rows": labels["external_vctk_rows"],
        "external_replacement_count": external_selection["replacement_count"],
        "external_selection": external_selection,
        "full_band_audio_preserved": True,
        "waveform_highpass_applied": False,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "source_receipt_sha256": {
            "data": sha256_file(args.data_receipt),
            "labels": sha256_file(args.label_receipt),
        },
        "artifact_sha256": {
            "metadata_csv": data["metadata_sha256"],
            "internal_label_bank": labels["internal_label_bank_sha256"],
            "external_label_bank": labels["external_label_bank_sha256"],
        },
        "artifact_paths": {
            "metadata_csv": data["metadata_csv"],
            "internal_label_bank": labels["internal_label_bank"],
            "external_label_bank": labels["external_label_bank"],
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
