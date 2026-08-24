#!/usr/bin/env python3
"""Falsify the exact-equivalent v15 high-pass on two frozen runtime outliers.

The opened v14 waveforms are development-only.  This probe compares the
official-source NumPy implementation with the frozen Praat 6.1.38 command at
PCM16, metric-waveform, source-range, and pulse-topology levels.  It never
loads a generator or authorizes a new speaker panel by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    PRAAT_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    require_exact_topology_equal,
    topology_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
FORMAL_REFRESH_GATE_MS = 500.0
DEV_ENGINEERING_MARGIN_MS = 450.0
DEFAULT_WARM_REPEATS = 7
FROZEN_OUTLIER_CASE_IDS = (
    "sealed_final__SD05__cs__rir_only",
    "sealed_final__ÄHH16__cs__rir_only",
)
PASS_DECISION = (
    "PASS_SHIMMER_DB_RUNTIME_V15_HIGHPASS_OUTLIERS_"
    "AUTHORIZE_FULL_DEV_EQUIVALENCE"
)
FAIL_DECISION = "NO_GO_SHIMMER_DB_RUNTIME_V15_HIGHPASS_OUTLIERS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "avqi_shimmer_exact_topology_worker.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--warm-repeats", type=int, default=DEFAULT_WARM_REPEATS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty high-pass profile CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_cases(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("v14 panel-contract hash drift")
    panel = json.loads(path.read_text(encoding="utf-8"))
    rows_by_case = {row["case_id"]: dict(row) for row in panel["rows"]}
    if not set(FROZEN_OUTLIER_CASE_IDS).issubset(rows_by_case):
        raise ValueError("v14 panel no longer contains both frozen outliers")
    cases = [rows_by_case[case_id] for case_id in FROZEN_OUTLIER_CASE_IDS]
    for row in cases:
        if row["view"] != "cs" or row["condition"] != "rir_only":
            raise ValueError("frozen high-pass outlier slice drift")
        path = Path(row["base_path"])
        if sha256_file(path) != row["base_sha256"]:
            raise ValueError(f"frozen base-waveform hash drift: {row['case_id']}")
    return cases


def topology_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"runtime_highpass:{row['case_id']}",
        "case_id": row["case_id"],
        "role": "current_s3_500_output_topology",
        "path": str(Path(row["base_path"]).resolve()),
        "view": row["view"],
        "score_components": False,
        "exact_metric_topology": True,
    }


def compare_to_praat(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    topology_equal = True
    topology_error = ""
    try:
        require_exact_topology_equal(reference, candidate, label)
    except ValueError as error:
        topology_equal = False
        topology_error = str(error)
    highpass_equal = (
        reference["highpass_pcm16_sha256"]
        == candidate["highpass_pcm16_sha256"]
    )
    metric_equal = (
        reference["metric_pcm16_sha256"]
        == candidate["metric_pcm16_sha256"]
    )
    return {
        "highpass_pcm16_equal": highpass_equal,
        "metric_pcm16_equal": metric_equal,
        "topology_equal": topology_equal,
        "topology_error": topology_error,
        "exact_equivalent": highpass_equal and metric_equal and topology_equal,
    }


def flatten_row(
    case: dict[str, Any],
    phase: str,
    repeat_index: int,
    topology: dict[str, Any],
    request_wall_ms: float,
    staging: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    timing = topology["timing_ms"]
    return {
        "case_id": case["case_id"],
        "speaker_id": case["speaker_id"],
        "view": case["view"],
        "condition": case["condition"],
        "sample_group": case["sample_group"],
        "phase": phase,
        "repeat_index": repeat_index,
        "highpass_mode": topology["metric_highpass"],
        "topology_sha256": topology_sha256(topology),
        "highpass_pcm16_sha256": topology["highpass_pcm16_sha256"],
        "metric_pcm16_sha256": topology["metric_pcm16_sha256"],
        "source_ranges_sha256": topology["source_ranges_sha256"],
        "pulse_positions_sha256": topology["pulse_positions_sha256"],
        "pulse_count": topology["pulse_count"],
        "client_tmpfs_staging_ms": staging["staging_ms"],
        "input_read_ms": timing["input_read"],
        "highpass_ms": timing["highpass"],
        "highpass_input_pcm16_roundtrip_ms": timing[
            "highpass_input_pcm16_roundtrip"
        ],
        "highpass_sound_construct_ms": timing["highpass_sound_construct"],
        "highpass_stop_hann_filter_ms": timing["highpass_stop_hann_filter"],
        "highpass_peak_extremum_ms": timing["highpass_peak_extremum"],
        "highpass_scale_peak_ms": timing["highpass_scale_peak"],
        "highpass_quantize_ms": timing["highpass_quantize"],
        "textgrid_ms": timing["textgrid"],
        "source_selection_ms": timing["source_selection"],
        "metric_gather_ms": timing["metric_gather"],
        "pointprocess_construct_ms": timing["pointprocess_construct"],
        "pulse_enumeration_ms": timing["pulse_enumeration"],
        "internal_refresh_ms": topology["pulse_runtime_ms"],
        "request_wall_ms": request_wall_ms,
        "end_to_end_refresh_ms": float(staging["staging_ms"]) + request_wall_ms,
        **parity,
    }


def summarize_candidate_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] == "candidate_warm"]
    if not selected:
        raise ValueError("candidate runtime summary requires warm rows")
    internal = [float(row["internal_refresh_ms"]) for row in selected]
    end_to_end = [float(row["end_to_end_refresh_ms"]) for row in selected]
    exact = all(bool(row["exact_equivalent"]) for row in selected)
    formal = max(internal) <= FORMAL_REFRESH_GATE_MS and max(
        end_to_end
    ) <= FORMAL_REFRESH_GATE_MS
    development = max(internal) <= DEV_ENGINEERING_MARGIN_MS and max(
        end_to_end
    ) <= DEV_ENGINEERING_MARGIN_MS
    return {
        "measurement_count": len(selected),
        "all_exact_equivalent": exact,
        "internal_refresh_ms": {
            "minimum": min(internal),
            "median": median(internal),
            "maximum": max(internal),
        },
        "end_to_end_refresh_ms": {
            "minimum": min(end_to_end),
            "median": median(end_to_end),
            "maximum": max(end_to_end),
            "includes_client_tmpfs_staging": True,
        },
        "formal_500ms_pass": formal,
        "development_450ms_margin_pass": development,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("high-pass profile source commit differs from HEAD")
    if args.warm_repeats < 5:
        raise ValueError("high-pass profile requires at least five warm repeats")
    cases = load_cases(args.panel_contract, args.panel_contract_sha256)
    args.output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    worker_receipts: list[dict[str, Any]] = []
    for case in cases:
        waveform, sample_rate = sf.read(case["base_path"], dtype="float32")
        if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
            raise ValueError(f"invalid frozen waveform: {case['case_id']}")
        waveform = np.asarray(waveform, dtype=np.float32)
        item = topology_item(case)
        with ExactShimmerTopologyWorker(
            args.exact_python,
            args.worker_script,
            args.avqi_code_root,
            args.avqi_code_tree_sha256,
        ) as worker:
            synthetic_warmup, synthetic_wall_ms = worker.warmup()
            candidate_rows, candidate_wall, candidate_staging = (
                worker.refresh_current_waveforms(
                    [item],
                    [waveform],
                    highpass_mode=NUMPY_HIGHPASS_MODE,
                )
            )
            candidate_first = candidate_rows[0]
            reference_rows, reference_wall, reference_staging = (
                worker.refresh_current_waveforms(
                    [item],
                    [waveform],
                    highpass_mode=PRAAT_HIGHPASS_MODE,
                )
            )
            reference = reference_rows[0]
            reference_parity = {
                "highpass_pcm16_equal": True,
                "metric_pcm16_equal": True,
                "topology_equal": True,
                "topology_error": "",
                "exact_equivalent": True,
            }
            rows.append(
                flatten_row(
                    case,
                    "praat_reference",
                    0,
                    reference,
                    reference_wall,
                    reference_staging[0],
                    reference_parity,
                )
            )
            first_parity = compare_to_praat(
                reference,
                candidate_first,
                f"{case['case_id']}:candidate-post-synthetic-first",
            )
            rows.append(
                flatten_row(
                    case,
                    "candidate_post_synthetic_first",
                    0,
                    candidate_first,
                    candidate_wall,
                    candidate_staging[0],
                    first_parity,
                )
            )
            for repeat_index in range(1, args.warm_repeats + 1):
                candidate_rows, candidate_wall, candidate_staging = (
                    worker.refresh_current_waveforms(
                        [item],
                        [waveform],
                        highpass_mode=NUMPY_HIGHPASS_MODE,
                    )
                )
                candidate = candidate_rows[0]
                parity = compare_to_praat(
                    reference,
                    candidate,
                    f"{case['case_id']}:candidate-warm:{repeat_index}",
                )
                rows.append(
                    flatten_row(
                        case,
                        "candidate_warm",
                        repeat_index,
                        candidate,
                        candidate_wall,
                        candidate_staging[0],
                        parity,
                    )
                )
            worker_receipts.append(
                {
                    "case_id": case["case_id"],
                    "startup": worker.startup,
                    "startup_ms": worker.startup_ms,
                    "synthetic_warmup": synthetic_warmup,
                    "synthetic_warmup_wall_ms": synthetic_wall_ms,
                }
            )
        print(f"highpass_profile_complete={case['case_id']}", flush=True)

    runtime = summarize_candidate_runtime(rows)
    first_rows = [
        row for row in rows if row["phase"] == "candidate_post_synthetic_first"
    ]
    first_exact = all(bool(row["exact_equivalent"]) for row in first_rows)
    passed = (
        first_exact
        and runtime["all_exact_equivalent"]
        and runtime["formal_500ms_pass"]
        and runtime["development_450ms_margin_pass"]
    )
    decision = PASS_DECISION if passed else FAIL_DECISION

    results_path = args.output_dir / "highpass_profile_results.csv"
    write_csv(results_path, rows)
    report = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v15-highpass-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "scope": "opened_v14_two_outlier_mechanism_probe_only",
        "case_ids": list(FROZEN_OUTLIER_CASE_IDS),
        "candidate_highpass_mode": NUMPY_HIGHPASS_MODE,
        "reference_highpass_mode": PRAAT_HIGHPASS_MODE,
        "official_source_contract": {
            "praat_version": "6.1.38",
            "fft_padding": "next_power_of_two_starting_at_two",
            "stop_hann_zero_through_hz": 33.9,
            "stop_hann_transition_end_hz": 34.1,
            "inverse_output": "truncate_to_original_sample_count",
            "sinc70_peak_and_pcm16_roundtrip_remain_praat": True,
        },
        "fixed_alpha": 0.001,
        "scientific_gates_changed": False,
        "formal_refresh_gate_ms": FORMAL_REFRESH_GATE_MS,
        "development_engineering_margin_ms": DEV_ENGINEERING_MARGIN_MS,
        "metric_highpass_only": True,
        "emitted_waveform_full_band": True,
        "waveform_dependent_topology_cache": False,
        "current_output_topology_refreshed_each_call": True,
        "cold_definition": "first_numpy_call_after_synthetic_praat_warmup",
        "warm_repeats_per_case": args.warm_repeats,
        "first_candidate_calls_exact_equivalent": first_exact,
        "runtime": runtime,
        "worker_receipts": worker_receipts,
        "full_12_case_dev_equivalence_authorized": passed,
        "new_speaker_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "source_artifacts": {
            "panel_contract": str(args.panel_contract.resolve()),
            "panel_contract_sha256": args.panel_contract_sha256,
            "worker_script": str(args.worker_script.resolve()),
            "worker_script_sha256": sha256_file(args.worker_script),
            "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
        },
        "artifacts": {"results_csv": results_path.name},
    }
    report_path = args.output_dir / "highpass_profile_report.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v15-highpass-receipt-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "results_csv": str(results_path.resolve()),
        "results_csv_sha256": sha256_file(results_path),
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
