#!/usr/bin/env python3
"""Diagnose FD23 Shimmer-dB pulse-path drift without opening exact outcomes.

This dev-only audit compares the exact Praat pulse times of the hash-bound v16
base waveform and its fixed four-level trust-region candidates.  It requests
topology only (``score_components=false``), reports pulse-time offsets and
mismatch locations, and routes v17 to one source-informed Candidate-D family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    topology_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
CASE_ID = "sealed_final__FD23__cs__rir_only"
ALPHA_LADDER = (0.001, 0.0005, 0.00025, 0.000125)
MATCH_BANDS_SAMPLES = (4.0, 8.0, 16.0, 32.0)
TOPOLOGY_MATCH_TOLERANCE_SAMPLES = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--attempts-csv-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--runtime-worker-script-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")
    return observed


def repository_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_range_layout(topology: dict[str, Any]) -> list[dict[str, Any]]:
    cursor = int(topology["metric_constant_prefix_samples"])
    layout = []
    for index, (source_start, length) in enumerate(
        topology["metric_source_ranges"]
    ):
        length = int(length)
        layout.append(
            {
                "range_index": index,
                "metric_start_sample": cursor,
                "metric_end_sample": cursor + length,
                "source_start_sample": int(source_start),
                "source_end_sample": int(source_start) + length,
                "length_samples": length,
            }
        )
        cursor += length
    if cursor != int(topology["metric_sample_count"]):
        raise ValueError("metric range layout does not cover exact metric waveform")
    return layout


def locate_metric_position(
    position: float,
    layout: list[dict[str, Any]],
) -> dict[str, Any]:
    for row in layout:
        start = float(row["metric_start_sample"])
        end = float(row["metric_end_sample"])
        if start <= position < end:
            return {
                "range_index": int(row["range_index"]),
                "source_position_sample": float(
                    row["source_start_sample"] + position - start
                ),
            }
    return {"range_index": None, "source_position_sample": None}


def contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return [group for group in np.split(indices, split_points) if group.size]


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    percentiles = (0, 10, 25, 50, 75, 90, 95, 99, 100)
    observed = np.percentile(values, percentiles)
    return {
        f"p{percentile:02d}": float(value)
        for percentile, value in zip(percentiles, observed, strict=True)
    }


def pulse_alignment(
    base_topology: dict[str, Any],
    candidate_topology: dict[str, Any],
    *,
    alpha: float,
    backtrack_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base = np.asarray(base_topology["pulse_positions_samples"], dtype=np.float64)
    candidate = np.asarray(
        candidate_topology["pulse_positions_samples"],
        dtype=np.float64,
    )
    if base.size < 3 or candidate.size < 3:
        raise ValueError("pulse-time diagnosis requires at least three pulses")
    distances = np.abs(base[:, None] - candidate[None, :])
    nearest_candidate = np.argmin(distances, axis=1)
    nearest_base = np.argmin(distances, axis=0)
    signed_offsets = candidate[nearest_candidate] - base
    absolute_offsets = np.abs(signed_offsets)
    matched = absolute_offsets <= TOPOLOGY_MATCH_TOLERANCE_SAMPLES
    candidate_matched = (
        np.min(distances, axis=0) <= TOPOLOGY_MATCH_TOLERANCE_SAMPLES
    )
    mutual = (
        nearest_base[nearest_candidate] == np.arange(base.size)
    ) & matched

    layout = metric_range_layout(base_topology)
    boundaries = np.asarray(
        sorted(
            {
                float(row["metric_start_sample"])
                for row in layout
            }
            | {
                float(row["metric_end_sample"])
                for row in layout
            }
        ),
        dtype=np.float64,
    )
    pulse_rows = []
    for index, (base_position, candidate_index, offset) in enumerate(
        zip(base, nearest_candidate, signed_offsets, strict=True)
    ):
        location = locate_metric_position(float(base_position), layout)
        pulse_rows.append(
            {
                "case_id": CASE_ID,
                "alpha": alpha,
                "backtrack_index": backtrack_index,
                "base_pulse_index": index,
                "base_metric_sample": float(base_position),
                "base_metric_seconds": float(base_position / SAMPLE_RATE),
                "base_range_index": location["range_index"],
                "base_source_sample": location["source_position_sample"],
                "nearest_candidate_pulse_index": int(candidate_index),
                "nearest_candidate_metric_sample": float(
                    candidate[candidate_index]
                ),
                "signed_offset_samples": float(offset),
                "absolute_offset_samples": float(abs(offset)),
                "within_4_samples": bool(abs(offset) <= 4.0),
                "within_8_samples": bool(abs(offset) <= 8.0),
                "within_16_samples": bool(abs(offset) <= 16.0),
                "within_32_samples": bool(abs(offset) <= 32.0),
                "mutual_nearest_within_16": bool(mutual[index]),
                "distance_to_nearest_range_boundary_samples": float(
                    np.min(np.abs(boundaries - base_position))
                ),
            }
        )

    mismatch_rows = []
    bad_indices = np.flatnonzero(~matched)
    for run_index, run in enumerate(contiguous_runs(bad_indices)):
        run_positions = base[run]
        first = float(run_positions[0])
        last = float(run_positions[-1])
        previous_boundaries = boundaries[boundaries <= first]
        following_boundaries = boundaries[boundaries >= last]
        internal_boundaries = boundaries[(boundaries > first) & (boundaries < last)]
        touched_ranges = sorted(
            {
                int(pulse_rows[index]["base_range_index"])
                for index in run
                if pulse_rows[index]["base_range_index"] is not None
            }
        )
        mismatch_rows.append(
            {
                "case_id": CASE_ID,
                "alpha": alpha,
                "backtrack_index": backtrack_index,
                "run_index": run_index,
                "base_pulse_index_start": int(run[0]),
                "base_pulse_index_end": int(run[-1]),
                "base_pulse_count": int(run.size),
                "metric_start_sample": first,
                "metric_end_sample": last,
                "duration_samples": last - first,
                "metric_start_seconds": first / SAMPLE_RATE,
                "metric_end_seconds": last / SAMPLE_RATE,
                "signed_offset_median_samples": float(
                    np.median(signed_offsets[run])
                ),
                "signed_offset_min_samples": float(signed_offsets[run].min()),
                "signed_offset_max_samples": float(signed_offsets[run].max()),
                "start_distance_from_previous_boundary_samples": float(
                    first - previous_boundaries[-1]
                ),
                "end_distance_to_next_boundary_samples": float(
                    following_boundaries[0] - last
                ),
                "internal_range_boundary_count": int(internal_boundaries.size),
                "internal_range_boundaries_samples": json.dumps(
                    internal_boundaries.tolist(),
                    separators=(",", ":"),
                ),
                "touched_range_indices": json.dumps(
                    touched_ranges,
                    separators=(",", ":"),
                ),
            }
        )

    outside = absolute_offsets[matched]
    summary = {
        "alpha": alpha,
        "backtrack_index": backtrack_index,
        "base_pulse_count": int(base.size),
        "candidate_pulse_count": int(candidate.size),
        "pulse_count_delta": int(candidate.size - base.size),
        "source_ranges_equal": (
            candidate_topology["metric_source_ranges"]
            == base_topology["metric_source_ranges"]
        ),
        "metric_sample_count_equal": (
            int(candidate_topology["metric_sample_count"])
            == int(base_topology["metric_sample_count"])
        ),
        "base_to_candidate_match_rate_16_samples": float(np.mean(matched)),
        "candidate_to_base_match_rate_16_samples": float(
            np.mean(candidate_matched)
        ),
        "mutual_nearest_match_rate_16_samples": float(np.mean(mutual)),
        "unmatched_base_pulse_count": int(np.count_nonzero(~matched)),
        "unmatched_candidate_pulse_count": int(
            np.count_nonzero(~candidate_matched)
        ),
        "signed_offset_quantiles_samples": percentile_dict(signed_offsets),
        "absolute_offset_quantiles_samples": percentile_dict(absolute_offsets),
        "match_band_fractions": {
            str(int(band)): float(np.mean(absolute_offsets <= band))
            for band in MATCH_BANDS_SAMPLES
        },
        "matched_region_absolute_offset_median_samples": (
            float(np.median(outside)) if outside.size else None
        ),
        "mismatch_run_count": len(mismatch_rows),
        "mismatch_base_pulse_fraction": float(np.mean(~matched)),
        "mismatch_runs": mismatch_rows,
    }
    return summary, pulse_rows, mismatch_rows


def route_candidate_d(alignment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_ranges = all(row["source_ranges_equal"] for row in alignment_rows)
    same_single_run = (
        all(row["mismatch_run_count"] == 1 for row in alignment_rows)
        and len(
            {
                (
                    row["mismatch_runs"][0]["base_pulse_index_start"],
                    row["mismatch_runs"][0]["base_pulse_index_end"],
                )
                for row in alignment_rows
            }
        )
        == 1
    )
    crosses_boundaries = all(
        row["mismatch_runs"][0]["internal_range_boundary_count"] > 0
        for row in alignment_rows
        if row["mismatch_run_count"] == 1
    )
    if fixed_ranges and same_single_run and crosses_boundaries:
        return {
            "diagnostic_class": (
                "localized_contiguous_alternate_pulse_path_with_fixed_source_ranges"
            ),
            "candidate_d_family": (
                "pitch_synchronous_zero_crossing_shape_preserving_gain_projection"
            ),
            "support_mask_family_rejected_by_diagnosis": True,
            "reason": (
                "all four alphas retain the same long contiguous mismatch run "
                "across multiple exact source-range joins while source ranges stay "
                "identical"
            ),
        }
    return {
        "diagnostic_class": "boundary_or_mixed_pulse_topology_drift",
        "candidate_d_family": "detached_topology_aware_interior_support_mask",
        "support_mask_family_rejected_by_diagnosis": False,
        "reason": (
            "the fixed-source-range, same-run alternate-path signature was absent"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head() != args.source_commit:
        raise ValueError("source HEAD drifted after v17 diagnostic submission")
    input_hashes = {
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "opened panel contract",
        ),
        "attempts_csv": validate_hash(
            args.attempts_csv,
            args.attempts_csv_sha256,
            "v16 attempts CSV",
        ),
        "runtime_worker": validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "exact topology worker",
        ),
    }
    panel = read_json(args.panel_contract)
    panel_rows = [row for row in panel.get("rows", []) if row["case_id"] == CASE_ID]
    if len(panel_rows) != 1:
        raise ValueError("FD23/CS diagnostic base coverage drift")
    panel_row = panel_rows[0]
    if panel_row["view"] != "cs" or panel_row["condition"] != "rir_only":
        raise ValueError("FD23/CS diagnostic identity drift")
    base_path = Path(panel_row["base_path"])
    validate_hash(base_path, panel_row["base_sha256"], "FD23 base waveform")

    attempts = [
        row for row in read_csv(args.attempts_csv) if row["case_id"] == CASE_ID
    ]
    attempts.sort(key=lambda row: int(row["backtrack_index"]))
    if len(attempts) != len(ALPHA_LADDER):
        raise ValueError("v17 diagnosis requires the complete v16 alpha ladder")
    candidate_paths = []
    for index, (row, expected_alpha) in enumerate(
        zip(attempts, ALPHA_LADDER, strict=True)
    ):
        if int(row["backtrack_index"]) != index or not math.isclose(
            float(row["alpha"]),
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("v16 attempts alpha order drift")
        path = Path(row["candidate_path"])
        validate_hash(path, row["candidate_sha256"], f"candidate alpha {expected_alpha}")
        candidate_paths.append(path)

    args.output_dir.mkdir(parents=True)
    worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    items = [
        {
            "id": f"v17-pulse-diagnostic:{index}",
            "case_id": CASE_ID,
            "role": "current_output_topology",
            "path": str(path.resolve()),
            "view": "cs",
            "score_components": False,
            "exact_metric_topology": True,
            "highpass_mode": NUMPY_HIGHPASS_MODE,
        }
        for index, path in enumerate([base_path, *candidate_paths])
    ]
    refresh_started = time.perf_counter()
    try:
        topologies, request_wall_ms = worker.refresh(items)
        worker_startup = dict(worker.startup)
        worker_startup_ms = worker.startup_ms
    finally:
        worker.close()
    refresh_total_ms = 1000.0 * (time.perf_counter() - refresh_started)
    if len(topologies) != 1 + len(ALPHA_LADDER):
        raise ValueError("exact topology diagnostic coverage drift")

    base_topology = topologies[0]
    alignment_rows = []
    pulse_rows = []
    mismatch_rows = []
    for index, (alpha, topology) in enumerate(
        zip(ALPHA_LADDER, topologies[1:], strict=True)
    ):
        summary, per_pulse, per_run = pulse_alignment(
            base_topology,
            topology,
            alpha=alpha,
            backtrack_index=index,
        )
        summary["candidate_path"] = str(candidate_paths[index].resolve())
        summary["candidate_sha256"] = sha256_file(candidate_paths[index])
        summary["candidate_topology_sha256"] = topology_sha256(topology)
        alignment_rows.append(summary)
        pulse_rows.extend(per_pulse)
        mismatch_rows.extend(per_run)

    routing = route_candidate_d(alignment_rows)
    report = {
        "schema_version": "avqi-route-c-shimmer-db-pulse-alignment-v17-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "dev_only": True,
        "case_id": CASE_ID,
        "candidate_exact_outcomes_opened": False,
        "exact_component_scoring_requested": False,
        "input_hashes": input_hashes,
        "base_path": str(base_path.resolve()),
        "base_sha256": sha256_file(base_path),
        "base_topology_sha256": topology_sha256(base_topology),
        "base_pulse_count": int(base_topology["pulse_count"]),
        "base_metric_sample_count": int(base_topology["metric_sample_count"]),
        "base_metric_source_ranges": base_topology["metric_source_ranges"],
        "base_metric_range_layout": metric_range_layout(base_topology),
        "alpha_ladder": list(ALPHA_LADDER),
        "match_bands_samples": list(MATCH_BANDS_SAMPLES),
        "worker_startup": worker_startup,
        "worker_startup_ms": worker_startup_ms,
        "topology_batch_request_wall_ms": request_wall_ms,
        "topology_refresh_total_ms": refresh_total_ms,
        "alignment": alignment_rows,
        "routing": routing,
        "decision": "SELECT_ONE_SOURCE_INFORMED_CANDIDATE_D_FROM_TOPOLOGY_ONLY",
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_optimizer_steps": 0,
    }
    report_path = args.output_dir / "pulse_alignment_report.json"
    pulse_path = args.output_dir / "pulse_alignment_offsets.csv"
    mismatch_path = args.output_dir / "pulse_alignment_mismatch_runs.csv"
    write_json(report_path, report)
    write_csv(pulse_path, pulse_rows)
    write_csv(mismatch_path, mismatch_rows)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-pulse-alignment-v17-receipt-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "decision": report["decision"],
        "candidate_exact_outcomes_opened": False,
        "candidate_d_family": routing["candidate_d_family"],
        "artifacts": {
            report_path.name: sha256_file(report_path),
            pulse_path.name: sha256_file(pulse_path),
            mismatch_path.name: sha256_file(mismatch_path),
        },
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_optimizer_steps": 0,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
