#!/usr/bin/env python3
"""Stage-profile the two immutable Candidate-C v14 refresh outliers.

This opened-panel diagnostic is runtime-only.  It never scores waveform effect,
selects alpha, creates a generator optimizer, or authorizes promotion.  Each CS
waveform receives a dedicated persistent exact-Praat worker, one disclosed cold
refresh, repeated warm refreshes, and a separate authority-parity refresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

from scripts.profile_avqi_shimmer_exact_topology_runtime import (
    ExactWorker,
    require_same_topology,
    sha256_file,
    sha256_tree,
    topology_identity,
    validate_hash,
    write_csv,
    write_json,
)


SAMPLE_RATE = 16_000
PULSE_REFRESH_GATE_MS = 500.0
DEV_ENGINEERING_MARGIN_MS = 450.0
DEFAULT_WARM_REPEATS = 7
INPUT_LOADER = "soundfile_float32_exact_16khz_mono"
FROZEN_IMPLEMENTATION = "frozen_praat_per_frame_and_point"
FASTPATH_IMPLEMENTATION = "exact_bulk_frame_and_pointprocess_matrix"
IMPLEMENTATION_CONFIGS = {
    FROZEN_IMPLEMENTATION: {
        "frame_scan_mode": "praat_per_frame",
        "pulse_enumeration_mode": "praat_per_point",
        "wav_roundtrip_mode": "praat_temp_wav",
        "sounding_assembly_mode": "praat_extract_and_concatenate",
    },
    FASTPATH_IMPLEMENTATION: {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_temp_wav",
        "sounding_assembly_mode": "praat_extract_and_concatenate",
    },
}
EXTRA_PROBE_CONFIGS = {
    "probe_in_memory_pcm16_roundtrip": {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "soundfile_in_memory_pcm16",
        "sounding_assembly_mode": "praat_extract_and_concatenate",
    },
    "probe_numpy_sounding_assembly": {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_temp_wav",
        "sounding_assembly_mode": "numpy_exact_interval_slices",
    },
    "probe_combined_roundtrip_and_sounding": {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "soundfile_in_memory_pcm16",
        "sounding_assembly_mode": "numpy_exact_interval_slices",
    },
    "probe_reused_tmpfs_praat_wav": {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_reused_tmpfs_wav",
        "sounding_assembly_mode": "praat_extract_and_concatenate",
    },
    "probe_reused_tmpfs_wav_and_numpy_sounding": {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_reused_tmpfs_wav",
        "sounding_assembly_mode": "numpy_exact_interval_slices",
    },
    "probe_vectorized_frame_scan": {
        "frame_scan_mode": "numpy_vectorized_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_temp_wav",
        "sounding_assembly_mode": "praat_extract_and_concatenate",
    },
    "probe_vectorized_frames_tmpfs_wav_numpy_sounding": {
        "frame_scan_mode": "numpy_vectorized_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "praat_reused_tmpfs_wav",
        "sounding_assembly_mode": "numpy_exact_interval_slices",
    },
}
FROZEN_OUTLIER_CASE_IDS = (
    "sealed_final__SD05__cs__rir_only",
    "sealed_final__ÄHH16__cs__rir_only",
)
STAGE_FIELDS = (
    "input_read",
    "highpass",
    "highpass_filter_compute",
    "highpass_quantize",
    "textgrid",
    "source_selection",
    "textgrid_range",
    "metric_gather",
    "pointprocess_construct",
    "pulse_enumeration",
    "total_refresh",
)
DOMINANT_STAGE_FIELDS = (
    "highpass",
    "textgrid",
    "source_selection",
    "metric_gather",
    "pointprocess_construct",
    "pulse_enumeration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--warm-repeats", type=int, default=DEFAULT_WARM_REPEATS)
    return parser.parse_args()


def load_outlier_rows(path: Path) -> list[dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    rows_by_case = {row["case_id"]: dict(row) for row in contract["rows"]}
    missing = set(FROZEN_OUTLIER_CASE_IDS) - set(rows_by_case)
    if missing:
        raise ValueError(f"fresh-panel contract misses outliers: {sorted(missing)}")
    selected = [rows_by_case[case_id] for case_id in FROZEN_OUTLIER_CASE_IDS]
    for row in selected:
        if row["view"] != "cs":
            raise ValueError(f"runtime outlier is not CS: {row['case_id']}")
        base_path = Path(row["base_path"])
        validate_hash(base_path, row["base_sha256"], "sealed base waveform")
        info = sf.info(base_path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"invalid sealed base waveform: {base_path}")
        row["base_frame_count"] = int(info.frames)
        row["base_duration_seconds"] = float(info.frames / SAMPLE_RATE)
    return selected


def flatten_runtime_row(
    case: dict[str, Any],
    implementation: str,
    phase: str,
    repeat_index: int,
    worker_startup_ms: float,
    response: dict[str, Any],
    request_wall_ms: float,
) -> dict[str, Any]:
    timings = response["timing_ms"]
    return {
        "case_id": case["case_id"],
        "speaker_id": case["speaker_id"],
        "view": case["view"],
        "condition": case["condition"],
        "sample_group": case["sample_group"],
        "implementation": implementation,
        "phase": phase,
        "repeat_index": repeat_index,
        "worker_startup_ms": worker_startup_ms,
        "request_wall_ms": request_wall_ms,
        "wall_minus_internal_ms": request_wall_ms
        - float(timings["total_refresh"]),
        "input_loader": response["topology_input_loader"],
        "frame_scan_mode": response["frame_scan_mode"],
        "pulse_enumeration_mode": response["pulse_enumeration_mode"],
        "wav_roundtrip_mode": response["wav_roundtrip_mode"],
        "sounding_assembly_mode": response["sounding_assembly_mode"],
        "base_frame_count": case["base_frame_count"],
        "base_duration_seconds": case["base_duration_seconds"],
        "metric_sample_count": response["metric_sample_count"],
        "metric_source_range_count": response["metric_source_range_count"],
        "pulse_count": response["pulse_count"],
        "source_ranges_sha256": response["source_ranges_sha256"],
        "pulse_positions_sha256": response["pulse_positions_sha256"],
        **{f"{field}_ms": float(timings[field]) for field in STAGE_FIELDS},
    }


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        field: {
            "median_ms": median(float(row[f"{field}_ms"]) for row in rows),
            "maximum_ms": max(float(row[f"{field}_ms"]) for row in rows),
        }
        for field in STAGE_FIELDS
    }


def topology_identity_difference(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    reference_identity = topology_identity(reference)
    candidate_identity = topology_identity(candidate)
    return {
        key: {
            "reference": reference_identity[key],
            "candidate": candidate_identity[key],
        }
        for key in reference_identity
        if reference_identity[key] != candidate_identity[key]
    }


def case_summary(
    rows: list[dict[str, Any]],
    implementation: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["phase"] == "warm" and row["implementation"] == implementation
    ]
    if not selected:
        raise ValueError("case summary requires warm rows")
    stage_medians = {
        field: median(float(row[f"{field}_ms"]) for row in selected)
        for field in DOMINANT_STAGE_FIELDS
    }
    dominant_stage = max(stage_medians, key=stage_medians.__getitem__)
    totals = [float(row["total_refresh_ms"]) for row in selected]
    walls = [float(row["request_wall_ms"]) for row in selected]
    scheduler = [float(row["wall_minus_internal_ms"]) for row in selected]
    return {
        "case_id": selected[0]["case_id"],
        "speaker_id": selected[0]["speaker_id"],
        "implementation": implementation,
        "base_frame_count": int(selected[0]["base_frame_count"]),
        "base_duration_seconds": float(selected[0]["base_duration_seconds"]),
        "metric_sample_count": int(selected[0]["metric_sample_count"]),
        "metric_source_range_count": int(
            selected[0]["metric_source_range_count"]
        ),
        "pulse_count": int(selected[0]["pulse_count"]),
        "warm_repeat_count": len(selected),
        "warm_total_median_ms": median(totals),
        "warm_total_minimum_ms": min(totals),
        "warm_total_maximum_ms": max(totals),
        "warm_total_p95_ms": float(np.percentile(totals, 95)),
        "warm_request_wall_maximum_ms": max(walls),
        "warm_wall_minus_internal_maximum_ms": max(scheduler),
        "dominant_stage": dominant_stage,
        "dominant_stage_median_ms": stage_medians[dominant_stage],
        "stage_medians_ms": stage_medians,
        "formal_500ms_pass": max(totals) <= PULSE_REFRESH_GATE_MS,
        "development_450ms_margin_pass": max(totals)
        <= DEV_ENGINEERING_MARGIN_MS,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.warm_repeats < 5:
        raise ValueError("runtime profile requires at least five warm repeats")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    if sha256_tree(args.avqi_code_root) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    panel_hash = validate_hash(
        args.panel_contract,
        args.panel_contract_sha256,
        "sealed fresh-panel contract",
    )
    cases = load_outlier_rows(args.panel_contract)

    args.output_dir.mkdir(parents=True)
    runtime_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    fastpath_warmup_rows: list[dict[str, Any]] = []
    extra_probe_rows: list[dict[str, Any]] = []
    exact_versions: dict[str, str] | None = None
    for case in cases:
        with ExactWorker(args.exact_python, args.avqi_code_root) as worker:
            if exact_versions is None:
                exact_versions = dict(worker.startup)
            elif exact_versions != worker.startup:
                raise ValueError("exact worker version drift")
            base_payload = {
                "op": "refresh",
                "path": case["base_path"],
                "verify_authority": False,
                "input_loader": INPUT_LOADER,
            }
            frozen_payload = {
                **base_payload,
                **IMPLEMENTATION_CONFIGS[FROZEN_IMPLEMENTATION],
            }
            cold, cold_wall_ms = worker.request(frozen_payload)
            reference = cold
            runtime_rows.append(
                flatten_runtime_row(
                    case,
                    FROZEN_IMPLEMENTATION,
                    "cold",
                    0,
                    worker.startup_ms,
                    cold,
                    cold_wall_ms,
                )
            )
            fastpath_payload = {
                **base_payload,
                **IMPLEMENTATION_CONFIGS[FASTPATH_IMPLEMENTATION],
            }
            fastpath_warmup, fastpath_warmup_wall_ms = worker.request(
                fastpath_payload
            )
            require_same_topology(
                reference,
                fastpath_warmup,
                f"{case['case_id']}:fastpath-command-warmup",
            )
            fastpath_warmup_rows.append(
                {
                    "case_id": case["case_id"],
                    "internal_ms": float(
                        fastpath_warmup["timing_ms"]["total_refresh"]
                    ),
                    "request_wall_ms": fastpath_warmup_wall_ms,
                    "used_for_runtime_gate": False,
                    "role": "opened_dev_command_initialization_only",
                }
            )
            for probe_name, probe_config in EXTRA_PROBE_CONFIGS.items():
                probe_response, probe_wall_ms = worker.request(
                    {**base_payload, **probe_config}
                )
                identity_difference = topology_identity_difference(
                    reference,
                    probe_response,
                )
                extra_probe_rows.append(
                    {
                        "case_id": case["case_id"],
                        "probe": probe_name,
                        "config": probe_config,
                        "topology_identity_equal": not identity_difference,
                        "identity_difference": identity_difference,
                        "internal_ms": float(
                            probe_response["timing_ms"]["total_refresh"]
                        ),
                        "request_wall_ms": probe_wall_ms,
                        "used_for_runtime_gate": False,
                    }
                )
            for implementation, config in IMPLEMENTATION_CONFIGS.items():
                payload = {**base_payload, **config}
                for repeat_index in range(1, args.warm_repeats + 1):
                    warm, warm_wall_ms = worker.request(payload)
                    require_same_topology(
                        reference,
                        warm,
                        f"{case['case_id']}:{implementation}:{repeat_index}",
                    )
                    runtime_rows.append(
                        flatten_runtime_row(
                            case,
                            implementation,
                            "warm",
                            repeat_index,
                            worker.startup_ms,
                            warm,
                            warm_wall_ms,
                        )
                    )
            authority, authority_wall_ms = worker.request(
                {
                    **base_payload,
                    **IMPLEMENTATION_CONFIGS[FASTPATH_IMPLEMENTATION],
                    "verify_authority": True,
                }
            )
            require_same_topology(
                reference,
                authority,
                f"{case['case_id']}:authority",
            )
            parity = dict(authority["authority_parity"])
            if not parity["pass"]:
                raise ValueError(f"authority parity failed: {case['case_id']}")
            authority_rows.append(
                {
                    "case_id": case["case_id"],
                    "speaker_id": case["speaker_id"],
                    "authority_request_wall_ms": authority_wall_ms,
                    **parity,
                }
            )

    warm_rows_by_implementation = {
        implementation: [
            row
            for row in runtime_rows
            if row["phase"] == "warm"
            and row["implementation"] == implementation
        ]
        for implementation in IMPLEMENTATION_CONFIGS
    }
    cold_rows = [row for row in runtime_rows if row["phase"] == "cold"]
    per_case_by_implementation = {
        implementation: [
            case_summary(
                [
                    row
                    for row in runtime_rows
                    if row["case_id"] == case["case_id"]
                ],
                implementation,
            )
            for case in cases
        ]
        for implementation in IMPLEMENTATION_CONFIGS
    }
    runtime_by_implementation = {}
    for implementation, selected_rows in warm_rows_by_implementation.items():
        warm_maximum_ms = max(
            float(row["total_refresh_ms"]) for row in selected_rows
        )
        runtime_by_implementation[implementation] = {
            "config": IMPLEMENTATION_CONFIGS[implementation],
            "warm_stage_summary": stage_summary(selected_rows),
            "per_case": per_case_by_implementation[implementation],
            "warm_maximum_total_refresh_ms": warm_maximum_ms,
            "formal_500ms_pass_on_development_repeats": warm_maximum_ms
            <= PULSE_REFRESH_GATE_MS,
            "development_450ms_margin_pass": warm_maximum_ms
            <= DEV_ENGINEERING_MARGIN_MS,
        }
    candidate_runtime = runtime_by_implementation[FASTPATH_IMPLEMENTATION]
    report = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v15-outlier-profile-v3",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "scope": "opened_panel_runtime_only_not_promotion",
        "candidate": "current_output_exact_topology_refresh_alpha_0p001",
        "fixed_alpha": 0.001,
        "pure_torch_estimator": False,
        "exact_praat_semantics_changed": False,
        "scientific_gates_changed": False,
        "formal_generator_training_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "source_artifacts": {
            "panel_contract": str(args.panel_contract),
            "panel_contract_sha256": panel_hash,
            "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
        },
        "exact_versions": exact_versions,
        "contract": {
            "input_loader": INPUT_LOADER,
            "one_dedicated_persistent_worker_per_case": True,
            "one_disclosed_cold_refresh": True,
            "warm_repeats": args.warm_repeats,
            "implementation_configs": IMPLEMENTATION_CONFIGS,
            "extra_probe_configs": EXTRA_PROBE_CONFIGS,
            "fastpath_command_warmup_per_worker": True,
            "production_warmup_must_not_use_panel_or_training_waveforms": True,
            "formal_refresh_gate_ms_unchanged": PULSE_REFRESH_GATE_MS,
            "development_engineering_margin_ms": DEV_ENGINEERING_MARGIN_MS,
            "opened_outlier_case_ids": list(FROZEN_OUTLIER_CASE_IDS),
        },
        "runtime": {
            "cold_stage_summary": stage_summary(cold_rows),
            "opened_dev_fastpath_warmup_rows": fastpath_warmup_rows,
            "by_implementation": runtime_by_implementation,
            "candidate_implementation": FASTPATH_IMPLEMENTATION,
            "candidate_formal_500ms_pass_on_development_repeats": (
                candidate_runtime[
                    "formal_500ms_pass_on_development_repeats"
                ]
            ),
            "candidate_development_450ms_margin_pass": candidate_runtime[
                "development_450ms_margin_pass"
            ],
        },
        "extra_exact_equivalence_probes": {
            "rows": extra_probe_rows,
            "all_equal_by_probe": {
                probe_name: all(
                    row["topology_identity_equal"]
                    for row in extra_probe_rows
                    if row["probe"] == probe_name
                )
                for probe_name in EXTRA_PROBE_CONFIGS
            },
        },
        "fastpath_topology_equivalence": {
            "all_repeated_calls_equal_frozen": True,
            "identity_fields": [
                "highpass_pcm16_sha256",
                "metric_pcm16_sha256",
                "source_ranges_sha256",
                "pulse_positions_sha256",
                "metric_sample_count",
                "metric_constant_prefix_samples",
                "metric_mapped_sample_count",
                "pulse_count",
            ],
        },
        "authority_parity": {
            "all_pass": all(row["pass"] for row in authority_rows),
            "rows": authority_rows,
        },
        "promotion_authorized": False,
        "next_action": (
            "freeze no implementation from this opened-panel diagnostic; "
            "use stage evidence to choose the smallest exact-equivalent change"
        ),
    }
    runtime_path = args.output_dir / "runtime_profile.csv"
    report_path = args.output_dir / "diagnostic_report.json"
    write_csv(runtime_path, runtime_rows)
    write_json(report_path, report)
    receipt = {
        "schema_version": (
            "avqi-route-c-shimmer-db-runtime-v15-outlier-profile-receipt-v3"
        ),
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "report_sha256": sha256_file(report_path),
        "runtime_profile_sha256": sha256_file(runtime_path),
        "panel_contract_sha256": panel_hash,
        "authority_parity_pass": report["authority_parity"]["all_pass"],
        "fastpath_topology_equivalence_pass": True,
        "extra_probe_all_equal_by_probe": report[
            "extra_exact_equivalence_probes"
        ]["all_equal_by_probe"],
        "formal_500ms_pass_on_development_repeats": report["runtime"][
            "candidate_formal_500ms_pass_on_development_repeats"
        ],
        "development_450ms_margin_pass": report["runtime"][
            "candidate_development_450ms_margin_pass"
        ],
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
