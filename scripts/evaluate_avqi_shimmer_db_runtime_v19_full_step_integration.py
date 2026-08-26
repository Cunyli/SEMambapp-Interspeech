#!/usr/bin/env python3
"""Audit the v19 paired certificate inside the frozen full v18 selector step.

This opened-development probe executes the unchanged D-then-C selector on all
24 v14+v15 cases for at least three repeats.  Candidate exact AVQI components
remain closed.  Every attempted candidate must be byte-identical to the
immutable v18 PCM24 candidate, topology/proxy certificates must remain
equivalent, and the complete timed step must retain both the frozen 500-ms
gate and the pre-registered 450-ms engineering margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from scripts.avqi_shimmer_exact_topology_runtime_v19 import (
    TMPFS_ROOT,
    PairedPeakCertificateTopologyWorker,
    float32_payload,
    sha256_file,
)
from scripts import (
    evaluate_avqi_shimmer_db_topology_family_selector_opened24 as opened24,
)
from scripts import (
    evaluate_avqi_shimmer_db_topology_family_selector_v18 as selector_core,
)


PASS_DECISION = "PASS_SHIMMER_DB_RUNTIME_V19_FULL_STEP_INTEGRATION"
FAIL_DECISION = "NO_GO_SHIMMER_DB_RUNTIME_V19_FULL_STEP_INTEGRATION"
V18_NO_GO_DECISION = "NO_GO_SHIMMER_DB_V18_OPENED24_PRESELECTION"
V18_SOURCE_COMMIT = "cb29d05ec073649b5d11beb7d5813f445d38eb43"
V18_SLURM_JOB_ID = "19943414"
V19_TOPOLOGY_PASS_DECISION = (
    "PASS_SHIMMER_DB_RUNTIME_V19_PAIRED_PEAK_EQUIVALENCE"
)
V19_TOPOLOGY_SOURCE_COMMIT = (
    "89ef9fc7466603eef1461ad8c17347e30943a1da"
)
V19_TOPOLOGY_SLURM_JOB_ID = "19943522"
EXPECTED_CASE_COUNT = 24
EXPECTED_SPEAKER_COUNT = 12
EXPECTED_REFERENCE_ATTEMPT_COUNT = 36
DEFAULT_REPEATS = 3
FORMAL_RUNTIME_GATE_MS = 500.0
ENGINEERING_MARGIN_MS = 450.0
PROXY_EQUIVALENCE_ABS_TOLERANCE = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for panel_label in ("v14", "v15"):
        for artifact in ("panel_contract", "target_contract", "fresh_results"):
            option = artifact.replace("_", "-")
            parser.add_argument(
                f"--{panel_label}-{option}",
                type=Path,
                required=True,
            )
            parser.add_argument(
                f"--{panel_label}-{option}-sha256",
                required=True,
            )
    for artifact in ("report", "preselection", "seal", "results", "receipt"):
        parser.add_argument(f"--selector4-{artifact}", type=Path, required=True)
        parser.add_argument(f"--selector4-{artifact}-sha256", required=True)
    parser.add_argument("--selector-core-script", type=Path, required=True)
    parser.add_argument("--selector-core-script-sha256", required=True)
    parser.add_argument("--v16-family-source", type=Path, required=True)
    parser.add_argument("--v16-family-source-sha256", required=True)
    parser.add_argument("--v17-family-source", type=Path, required=True)
    parser.add_argument("--v17-family-source-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--runtime-worker-script-sha256", required=True)

    for artifact in ("report", "preselection", "receipt"):
        parser.add_argument(f"--v18-{artifact}", type=Path, required=True)
        parser.add_argument(f"--v18-{artifact}-sha256", required=True)
    for artifact in (
        "report",
        "equivalence",
        "runtime",
        "pcm24_equivalence",
        "receipt",
    ):
        option = artifact.replace("_", "-")
        parser.add_argument(f"--v19-topology-{option}", type=Path, required=True)
        parser.add_argument(f"--v19-topology-{option}-sha256", required=True)

    implementation_names = (
        "peak_certificate_helper",
        "phase1_evaluator",
        "frozen_worker",
        "v19_worker",
        "v19_runtime_client",
        "integration_evaluator",
        "integration_runner",
    )
    for name in implementation_names:
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)

    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser.parse_args()


def git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_repository_provenance(
    args: argparse.Namespace,
) -> dict[str, Any]:
    repository_root = args.repository_root.resolve()
    expected_root = Path(__file__).resolve().parents[1]
    if repository_root != expected_root:
        raise ValueError("repository root does not contain this evaluator")
    observed_head = git_output(repository_root, "rev-parse", "HEAD")
    if observed_head != args.source_commit:
        raise ValueError("v19 integration repository HEAD/source drift")
    observed_status = git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if observed_status:
        raise ValueError("v19 integration requires a clean repository")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            V19_TOPOLOGY_SOURCE_COMMIT,
            observed_head,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("v19 integration does not descend from topology PASS")

    implementation_names = (
        "peak_certificate_helper",
        "phase1_evaluator",
        "frozen_worker",
        "v19_worker",
        "v19_runtime_client",
        "integration_evaluator",
        "integration_runner",
    )
    implementation_hashes = {
        name: opened24.validate_hash(
            getattr(args, name),
            getattr(args, f"{name}_sha256"),
            f"v19 integration {name}",
        )
        for name in implementation_names
    }
    if args.integration_evaluator.resolve() != Path(__file__).resolve():
        raise ValueError("v19 integration evaluator path drift")
    if args.frozen_worker.resolve() != args.runtime_worker_script.resolve():
        raise ValueError("frozen worker provenance/path binding drift")
    if (
        args.frozen_worker_sha256
        != args.runtime_worker_script_sha256
    ):
        raise ValueError("frozen worker provenance/hash binding drift")
    if args.v19_worker.resolve() == args.runtime_worker_script.resolve():
        raise ValueError("v19 worker must not replace the frozen worker binding")
    return {
        "repository_head": observed_head,
        "repository_tree_clean": True,
        "implementation_sha256": implementation_hashes,
    }


def validate_artifact_group(
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
    label: str,
) -> dict[str, str]:
    return {
        name: opened24.validate_hash(
            path,
            expected_hashes[name],
            f"{label} {name}",
        )
        for name, path in paths.items()
    }


def validate_v18_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[dict[str, str]], list[str]]:
    paths = {
        "report": args.v18_report,
        "preselection": args.v18_preselection,
        "receipt": args.v18_receipt,
    }
    hashes = validate_artifact_group(
        paths,
        {
            name: getattr(args, f"v18_{name}_sha256") for name in paths
        },
        "immutable v18 opened24",
    )
    report = opened24.read_json(paths["report"])
    receipt = opened24.read_json(paths["receipt"])
    rows = opened24.read_csv(paths["preselection"])
    if report.get("decision") != V18_NO_GO_DECISION:
        raise ValueError("immutable v18 decision drift")
    if report.get("source_commit") != V18_SOURCE_COMMIT:
        raise ValueError("immutable v18 report source drift")
    if receipt.get("source_commit") != V18_SOURCE_COMMIT:
        raise ValueError("immutable v18 receipt source drift")
    if receipt.get("slurm_job_id") != V18_SLURM_JOB_ID:
        raise ValueError("immutable v18 Slurm job drift")
    if report.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("immutable v18 unexpectedly opened exact outcomes")
    if report.get("exact_component_scoring_requested") is not False:
        raise ValueError("immutable v18 unexpectedly requested exact scoring")
    if report.get("new_sealed_panel_authorized") is not False:
        raise ValueError("immutable v18 unexpectedly authorized a fresh panel")
    if receipt.get("decision") != report.get("decision"):
        raise ValueError("immutable v18 report/receipt decision drift")
    if receipt.get("generator_optimizer_steps") != 0:
        raise ValueError("immutable v18 training boundary drift")
    if receipt.get("authoritative_training_decision") != (
        "NO_GO_AVQI_T2_TRAINING"
    ):
        raise ValueError("immutable v18 authoritative decision drift")
    if receipt.get("artifact_sha256") != {
        "diagnostic_report.json": hashes["report"],
        "family_selector_preselection.csv": hashes["preselection"],
    }:
        raise ValueError("immutable v18 receipt binding drift")
    if len(rows) != EXPECTED_REFERENCE_ATTEMPT_COUNT:
        raise ValueError("immutable v18 attempt coverage drift")
    failures = [str(value) for value in report.get("selector_failures", [])]
    if len(failures) != 2:
        raise ValueError("immutable v18 runtime-failure coverage drift")
    return hashes, rows, failures


def validate_v19_topology_evidence(
    args: argparse.Namespace,
) -> dict[str, str]:
    paths = {
        "report": args.v19_topology_report,
        "equivalence": args.v19_topology_equivalence,
        "runtime": args.v19_topology_runtime,
        "pcm24_equivalence": args.v19_topology_pcm24_equivalence,
        "receipt": args.v19_topology_receipt,
    }
    hashes = validate_artifact_group(
        paths,
        {
            name: getattr(args, f"v19_topology_{name}_sha256")
            for name in paths
        },
        "immutable v19 topology-only",
    )
    report = opened24.read_json(paths["report"])
    receipt = opened24.read_json(paths["receipt"])
    for value, label in ((report, "report"), (receipt, "receipt")):
        if value.get("decision") != V19_TOPOLOGY_PASS_DECISION:
            raise ValueError(f"v19 topology {label} decision drift")
        if value.get("source_commit") != V19_TOPOLOGY_SOURCE_COMMIT:
            raise ValueError(f"v19 topology {label} source drift")
        if value.get("slurm_job_id") != V19_TOPOLOGY_SLURM_JOB_ID:
            raise ValueError(f"v19 topology {label} Slurm job drift")
        if value.get("v19_integration_probe_authorized") is not True:
            raise ValueError(f"v19 topology {label} did not authorize integration")
        if value.get("opened24_rerun_authorized") is not False:
            raise ValueError(f"v19 topology {label} over-authorized opened24")
        if value.get("candidate_exact_avqi_components_opened") is not False:
            raise ValueError(f"v19 topology {label} opened exact components")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"v19 topology {label} training boundary drift")
    expected_receipt_hashes = {
        "diagnostic_report.json": hashes["report"],
        "peak_certificate_equivalence.csv": hashes["equivalence"],
        "paired_runtime_repeats.csv": hashes["runtime"],
        "pcm24_tmpfs_equivalence.csv": hashes["pcm24_equivalence"],
    }
    if receipt.get("artifact_sha256") != expected_receipt_hashes:
        raise ValueError("v19 topology receipt artifact binding drift")
    expected_implementation_hashes = {
        "peak_certificate_helper": args.peak_certificate_helper_sha256,
        "evaluator": args.phase1_evaluator_sha256,
        "frozen_worker": args.frozen_worker_sha256,
    }
    if report.get("implementation_sha256") != expected_implementation_hashes:
        raise ValueError("v19 topology report implementation binding drift")
    if receipt.get("implementation_sha256") != expected_implementation_hashes:
        raise ValueError("v19 topology receipt implementation binding drift")
    if report.get("equivalence", {}).get(
        "all_24_post_highpass_pcm16_equal"
    ) is not True:
        raise ValueError("v19 topology post-highpass equivalence drift")
    if report.get("paired_runtime", {}).get(
        "may_authorize_only_full_step_integration_probe"
    ) is not True:
        raise ValueError("v19 topology phase-boundary drift")
    return hashes


def attempt_id_from_reference(row: dict[str, str]) -> str:
    attempt_index = int(row["attempt_index"])
    if attempt_index == 0:
        if row["backtrack_index"] not in {"", "None"}:
            raise ValueError("v18 Candidate D backtrack index drift")
        return "candidate_d"
    backtrack_index = int(row["backtrack_index"])
    if attempt_index != backtrack_index + 1:
        raise ValueError("v18 Candidate C attempt ordering drift")
    return f"candidate_c_bt{backtrack_index}"


def load_reference_attempts(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        attempt_id = attempt_id_from_reference(row)
        candidate_path = Path(row["candidate_path"])
        observed_sha256 = opened24.validate_hash(
            candidate_path,
            row["candidate_sha256"],
            "immutable v18 attempted candidate",
        )
        stored, sample_rate = sf.read(
            candidate_path,
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != 16_000 or stored.ndim != 1 or stored.size == 0:
            raise ValueError("immutable v18 candidate PCM24 format drift")
        _, _, raw_float32_sha256 = float32_payload(stored)
        key = (row["case_id"], attempt_id)
        if key in references:
            raise ValueError("duplicate immutable v18 candidate attempt")
        references[key] = {
            **row,
            "attempt_id": attempt_id,
            "candidate_sha256": observed_sha256,
            "raw_float32_sha256": raw_float32_sha256,
        }
    if len(references) != EXPECTED_REFERENCE_ATTEMPT_COUNT:
        raise ValueError("immutable v18 candidate reference coverage drift")
    if len({case_id for case_id, _ in references}) != EXPECTED_CASE_COUNT:
        raise ValueError("immutable v18 candidate case coverage drift")
    return references


def reference_for_record(
    references: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    key = (case_id, str(record["attempt_id"]))
    if key not in references:
        raise ValueError(f"missing immutable v18 attempt reference: {key}")
    reference = references[key]
    if record["candidate_sha256"] != reference["candidate_sha256"]:
        raise ValueError(
            f"v19 candidate PCM24 differs from frozen v18 bytes: {key}"
        )
    return reference


def make_paired_refresh(
    references: dict[tuple[str, str], dict[str, Any]],
):
    def refresh_candidate_records_v19(
        context: dict[str, Any],
        records: list[dict[str, Any]],
        workers: list[PairedPeakCertificateTopologyWorker],
        executor: ThreadPoolExecutor,
    ) -> dict[str, float]:
        if not records or len(workers) != selector_core.WORKER_COUNT:
            raise ValueError("v19 candidate refresh worker contract drift")
        groups: list[list[dict[str, Any]]] = [[] for _ in workers]
        for index, record in enumerate(records):
            groups[index % len(workers)].append(record)

        def refresh_group(
            worker: PairedPeakCertificateTopologyWorker,
            grouped_records: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
            items = [
                selector_core.candidate_topology_item(
                    context["case_id"],
                    context["panel_row"]["view"],
                    record["candidate_path"],
                    record["attempt_id"],
                )
                for record in grouped_records
            ]
            grouped_references = [
                reference_for_record(
                    references,
                    context["case_id"],
                    record,
                )
                for record in grouped_records
            ]
            return worker.refresh_current_pcm24_candidates_paired(
                items,
                [record["candidate_path"] for record in grouped_records],
                [
                    reference["candidate_sha256"]
                    for reference in grouped_references
                ],
                [
                    reference["raw_float32_sha256"]
                    for reference in grouped_references
                ],
                [record["stored_waveform"] for record in grouped_records],
                case_id=context["case_id"],
                base_waveform=context["base_values"],
                base_topology=context["base_topology"],
                base_topology_sha256=context["base_topology_sha256"],
            )

        started = time.perf_counter()
        futures = [
            executor.submit(refresh_group, worker, grouped_records)
            for worker, grouped_records in zip(workers, groups, strict=True)
            if grouped_records
        ]
        topology_by_id: dict[
            str,
            tuple[dict[str, Any], float, dict[str, Any]],
        ] = {}
        request_sum_ms = 0.0
        internal_sum_ms = 0.0
        staging_sum_ms = 0.0
        for future in futures:
            rows, request_wall_ms, staging_rows = future.result()
            request_sum_ms += request_wall_ms
            staging_by_id = {row["id"]: row for row in staging_rows}
            for topology in rows:
                item_id = str(topology["id"])
                staging = staging_by_id[item_id]
                topology_by_id[item_id] = (
                    topology,
                    request_wall_ms,
                    staging,
                )
                internal_sum_ms += float(topology["pulse_runtime_ms"])
                staging_sum_ms += float(staging["staging_ms"])
                staging_sum_ms += float(staging["base_staging_ms"])
        wall_ms = 1000.0 * (time.perf_counter() - started)

        for record in records:
            item_id = (
                f"v18-topology:{context['case_id']}:{record['attempt_id']}"
            )
            topology, request_wall_ms, staging = topology_by_id[item_id]
            stability = selector_core.topology_stability(
                context["base_topology"],
                topology,
            )
            reference = reference_for_record(
                references,
                context["case_id"],
                record,
            )
            raw_hash_pass = (
                staging["raw_float32_sha256"]
                == reference["raw_float32_sha256"]
            )
            pcm24_hash_pass = (
                staging["candidate_pcm24_sha256"]
                == reference["candidate_sha256"]
            )
            base_case_pass = (
                topology["paired_base_case_id"] == context["case_id"]
                and topology["paired_base_view"]
                == context["panel_row"]["view"]
            )
            base_topology_pass = (
                topology["paired_base_topology_sha256"]
                == context["base_topology_sha256"]
                and topology[
                    "paired_base_source_waveform_float32_sha256"
                ]
                == context["base_topology"][
                    "source_waveform_float32_sha256"
                ]
            )
            expected_base_timing_sha256 = hashlib.sha256(
                json.dumps(
                    context["base_topology"]["timing_ms"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            base_timing_pass = (
                topology["paired_base_highpass_timing_sha256"]
                == expected_base_timing_sha256
            )
            worker_pcm24_echo_pass = (
                topology["paired_candidate_pcm24_sha256"]
                == reference["candidate_sha256"]
            )
            if not all(
                (
                    raw_hash_pass,
                    pcm24_hash_pass,
                    worker_pcm24_echo_pass,
                    base_case_pass,
                    base_topology_pass,
                    base_timing_pass,
                    staging[
                        "pcm24_readback_equals_frozen_stored_waveform"
                    ],
                )
            ):
                raise ValueError("v19 paired refresh provenance drift")
            certificate = topology["paired_peak_certificate"]
            record.update(
                {
                    "candidate_topology": topology,
                    "candidate_topology_sha256": (
                        selector_core.topology_sha256(topology)
                    ),
                    "candidate_refresh_group_request_wall_ms": (
                        request_wall_ms
                    ),
                    "candidate_refresh_client_staging_ms": float(
                        staging["staging_ms"]
                    )
                    + float(staging["base_staging_ms"]),
                    "candidate_refresh_internal_ms": float(
                        topology["pulse_runtime_ms"]
                    ),
                    "candidate_pcm24_reference_bound": pcm24_hash_pass,
                    "candidate_raw_float32_reference_bound": raw_hash_pass,
                    "candidate_pcm24_readback_used_for_refresh": True,
                    "paired_base_case_binding_pass": base_case_pass,
                    "paired_base_topology_binding_pass": (
                        base_topology_pass
                    ),
                    "paired_base_timing_binding_pass": base_timing_pass,
                    "worker_candidate_pcm24_echo_pass": (
                        worker_pcm24_echo_pass
                    ),
                    "paired_candidate_sinc70_search_skipped": certificate[
                        "candidate_sinc70_search_may_be_skipped"
                    ],
                    "paired_candidate_sinc70_peak_upper_bound": certificate[
                        "candidate_sinc70_peak_upper_bound"
                    ],
                    **stability,
                }
            )
        return {
            "candidate_refresh_concurrent_wall_ms": wall_ms,
            "candidate_refresh_request_wall_sum_ms": request_sum_ms,
            "candidate_refresh_internal_sum_ms": internal_sum_ms,
            "candidate_refresh_client_staging_sum_ms": staging_sum_ms,
        }

    return refresh_candidate_records_v19


def parse_optional_int(value: str) -> int | None:
    return None if value in {"", "None"} else int(value)


def parse_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid frozen boolean: {value}")
    return value == "True"


def float_equivalent(current: float, reference: str) -> bool:
    return math.isclose(
        float(current),
        float(reference),
        rel_tol=0.0,
        abs_tol=PROXY_EQUIVALENCE_ABS_TOLERANCE,
    )


def compare_attempt_rows(
    panel_row: dict[str, Any],
    case_record: dict[str, Any],
    references: dict[tuple[str, str], dict[str, Any]],
    repeat_index: int,
) -> list[dict[str, Any]]:
    current_rows = selector_core.preselection_rows(
        [panel_row],
        [case_record],
    )
    output: list[dict[str, Any]] = []
    for attempt, current in zip(
        case_record["attempts"],
        current_rows,
        strict=True,
    ):
        reference = references[(current["case_id"], attempt["attempt_id"])]
        scalar_equal = all(
            float_equivalent(current[field], reference[field])
            for field in (
                "proxy_before",
                "proxy_after_frozen_topology",
                "normalized_proxy_gap_before",
                "normalized_proxy_gap_after",
                "reference_to_candidate_match_rate_16_samples",
                "candidate_to_reference_match_rate_16_samples",
            )
        )
        boolean_equal = all(
            bool(current[field]) == parse_bool(reference[field])
            for field in (
                "proxy_nonregression_pass",
                "topology_stability_pass",
                "finite_safety_pass",
                "pcm24_effective_step_pass",
                "selected_attempt",
            )
        )
        identity_equal = all(
            (
                str(current["family"]) == reference["family"],
                float(current["alpha"]) == float(reference["alpha"]),
                current["backtrack_index"]
                == parse_optional_int(reference["backtrack_index"]),
                current["candidate_sha256"]
                == reference["candidate_sha256"],
                current["candidate_topology_sha256"]
                == reference["candidate_topology_sha256"],
                int(current["candidate_pulse_count"])
                == int(reference["candidate_pulse_count"]),
                str(current["selected_family"])
                == reference["selected_family"],
                float(current["selected_alpha"])
                == float(reference["selected_alpha"]),
            )
        )
        provenance_pass = all(
            (
                attempt["candidate_pcm24_reference_bound"],
                attempt["candidate_raw_float32_reference_bound"],
                attempt["candidate_pcm24_readback_used_for_refresh"],
                attempt["paired_base_case_binding_pass"],
                attempt["paired_base_topology_binding_pass"],
                attempt["paired_base_timing_binding_pass"],
                attempt["worker_candidate_pcm24_echo_pass"],
            )
        )
        output.append(
            {
                "case_id": current["case_id"],
                "opened_panel": panel_row["opened_panel"],
                "speaker_id": panel_row["speaker_id"],
                "view": panel_row["view"],
                "condition": panel_row["condition"],
                "sample_group": panel_row["sample_group"],
                "repeat_index": repeat_index,
                "attempt_index": current["attempt_index"],
                "attempt_id": attempt["attempt_id"],
                "family": current["family"],
                "alpha": current["alpha"],
                "backtrack_index": current["backtrack_index"],
                "current_candidate_sha256": current["candidate_sha256"],
                "reference_candidate_sha256": reference[
                    "candidate_sha256"
                ],
                "current_raw_float32_sha256": attempt[
                    "candidate_topology"
                ]["source_waveform_float32_sha256"],
                "reference_raw_float32_sha256": reference[
                    "raw_float32_sha256"
                ],
                "worker_candidate_pcm24_sha256": attempt[
                    "candidate_topology"
                ]["paired_candidate_pcm24_sha256"],
                "paired_base_topology_sha256": attempt[
                    "candidate_topology"
                ]["paired_base_topology_sha256"],
                "paired_base_highpass_timing_sha256": attempt[
                    "candidate_topology"
                ]["paired_base_highpass_timing_sha256"],
                "current_topology_sha256": current[
                    "candidate_topology_sha256"
                ],
                "reference_topology_sha256": reference[
                    "candidate_topology_sha256"
                ],
                "identity_equivalence_pass": identity_equal,
                "proxy_scalar_equivalence_pass": scalar_equal,
                "certificate_boolean_equivalence_pass": boolean_equal,
                "pcm24_readback_provenance_pass": provenance_pass,
                "paired_candidate_sinc70_search_skipped": attempt[
                    "paired_candidate_sinc70_search_skipped"
                ],
                "paired_candidate_sinc70_peak_upper_bound": attempt[
                    "paired_candidate_sinc70_peak_upper_bound"
                ],
                "attempt_equivalence_pass": (
                    identity_equal
                    and scalar_equal
                    and boolean_equal
                    and provenance_pass
                ),
            }
        )
    return output


def copy_selected_pcm24(
    case_record: dict[str, Any],
    durable_root: Path,
    repeat_index: int,
) -> dict[str, Any]:
    selected = case_record.get("selected_record")
    if not isinstance(selected, dict):
        return {
            "case_id": case_record["case_id"],
            "repeat_index": repeat_index,
            "selected_candidate_present": False,
            "durable_byte_equivalence_pass": False,
        }
    source = Path(selected["candidate_path"]).resolve()
    tmpfs_root = TMPFS_ROOT.resolve()
    if tmpfs_root != source and tmpfs_root not in source.parents:
        raise ValueError("selected v19 PCM24 source is not node-local tmpfs")
    source_sha256 = sha256_file(source)
    if source_sha256 != selected["candidate_sha256"]:
        raise ValueError("selected v19 tmpfs PCM24 hash drift")
    repeat_root = durable_root / f"repeat_{repeat_index}"
    repeat_root.mkdir(exist_ok=True)
    destination = repeat_root / f"{case_record['case_id']}__selected.wav"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selected PCM24: {destination}")
    copy_started = time.perf_counter()
    shutil.copyfile(source, destination)
    copy_ms = 1000.0 * (time.perf_counter() - copy_started)
    destination_sha256 = sha256_file(destination)
    byte_equal = source.read_bytes() == destination.read_bytes()
    if not byte_equal or destination_sha256 != source_sha256:
        raise ValueError("selected durable PCM24 copy is not byte-identical")
    selected["candidate_path"] = destination
    case_record["selected_path"] = destination
    return {
        "case_id": case_record["case_id"],
        "repeat_index": repeat_index,
        "selected_candidate_present": True,
        "selected_family": case_record["selected_family"],
        "selected_alpha": case_record["selected_alpha"],
        "tmpfs_source_path": str(source),
        "durable_selected_path": str(destination.resolve()),
        "expected_candidate_sha256": selected["candidate_sha256"],
        "tmpfs_source_sha256": source_sha256,
        "durable_selected_sha256": destination_sha256,
        "durable_byte_equivalence_pass": True,
        "selected_path_updated_to_durable_before_future_seal": (
            Path(case_record["selected_path"]).resolve()
            == destination.resolve()
        ),
        "durable_copy_after_timed_step": True,
        "durable_copy_ms_outside_metric_step": copy_ms,
    }


def case_runtime_row(
    panel_row: dict[str, Any],
    case_record: dict[str, Any],
    repeat_index: int,
    v18_failures: list[str],
) -> dict[str, Any]:
    runtime_ms = float(case_record["total_metric_step_runtime_ms"])
    d_batch = case_record["candidate_d_batch_runtime"]
    d_refresh = case_record["candidate_d_refresh_runtime"]
    c_batch = case_record["candidate_c_batch_runtime"]
    c_refresh = case_record["candidate_c_refresh_runtime"]
    return {
        "case_id": case_record["case_id"],
        "opened_panel": panel_row["opened_panel"],
        "speaker_id": panel_row["speaker_id"],
        "view": panel_row["view"],
        "condition": panel_row["condition"],
        "sample_group": panel_row["sample_group"],
        "repeat_index": repeat_index,
        "v18_runtime_failure_case": case_record["case_id"] in v18_failures,
        "selected_family": case_record["selected_family"],
        "selected_alpha": case_record["selected_alpha"],
        "attempted_family_count": case_record["attempted_family_count"],
        "candidate_topology_refresh_count": case_record[
            "candidate_topology_refresh_count"
        ],
        "base_refresh_runtime_ms": case_record["base_refresh_runtime_ms"],
        "gradient_runtime_ms": case_record["gradient_runtime_ms"],
        "candidate_d_plan_runtime_ms": case_record[
            "candidate_d_plan_runtime_ms"
        ],
        "candidate_d_projection_runtime_ms": case_record[
            "candidate_d_projection_runtime_ms"
        ],
        "candidate_d_pcm24_transfer_ms": d_batch[
            "candidate_gpu_to_cpu_batch_ms"
        ],
        "candidate_d_pcm24_write_read_hash_safety_wall_ms": d_batch[
            "candidate_pcm24_io_concurrent_wall_ms"
        ],
        "candidate_d_proxy_ms": d_batch[
            "candidate_frozen_proxy_batch_ms"
        ],
        "candidate_d_staging_worker_wall_ms": d_refresh[
            "candidate_refresh_concurrent_wall_ms"
        ],
        "candidate_d_worker_request_wall_sum_ms": d_refresh[
            "candidate_refresh_request_wall_sum_ms"
        ],
        "candidate_d_candidate_and_base_staging_sum_ms": d_refresh[
            "candidate_refresh_client_staging_sum_ms"
        ],
        "candidate_c_pcm24_transfer_ms": (
            c_batch["candidate_gpu_to_cpu_batch_ms"]
            if c_batch is not None
            else 0.0
        ),
        "candidate_c_pcm24_write_read_hash_safety_wall_ms": (
            c_batch["candidate_pcm24_io_concurrent_wall_ms"]
            if c_batch is not None
            else 0.0
        ),
        "candidate_c_proxy_ms": (
            c_batch["candidate_frozen_proxy_batch_ms"]
            if c_batch is not None
            else 0.0
        ),
        "candidate_c_staging_worker_wall_ms": (
            c_refresh["candidate_refresh_concurrent_wall_ms"]
            if c_refresh is not None
            else 0.0
        ),
        "candidate_c_worker_request_wall_sum_ms": (
            c_refresh["candidate_refresh_request_wall_sum_ms"]
            if c_refresh is not None
            else 0.0
        ),
        "candidate_c_candidate_and_base_staging_sum_ms": (
            c_refresh["candidate_refresh_client_staging_sum_ms"]
            if c_refresh is not None
            else 0.0
        ),
        "device_sync_call_count": case_record["device_sync_call_count"],
        "device_sync_wall_ms": case_record["device_sync_wall_ms"],
        "phase_accounted_sequential_wall_ms": case_record[
            "phase_accounted_sequential_wall_ms"
        ],
        "total_metric_step_runtime_ms": runtime_ms,
        "selector_call_external_wall_ms": case_record[
            "selector_call_external_wall_ms"
        ],
        "full_step_timer_contract_pass": case_record[
            "full_step_timer_contract_pass"
        ],
        "formal_500ms_pass": runtime_ms <= FORMAL_RUNTIME_GATE_MS,
        "engineering_450ms_margin_pass": runtime_ms <= ENGINEERING_MARGIN_MS,
        "selector_pass": case_record["selector_pass"],
        "candidate_exact_avqi_components_opened": False,
    }


def write_completion_receipt(
    args: argparse.Namespace,
    decision: str,
    artifact_paths: list[Path],
    source_provenance: dict[str, Any],
    opened24_rerun_authorized: bool,
) -> None:
    receipt = {
        "schema_version": (
            "avqi-route-c-shimmer-db-runtime-v19-full-step-integration-receipt-v1"
        ),
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "candidate_exact_avqi_components_opened": False,
        "opened24_rerun_authorized": opened24_rerun_authorized,
        "new_sealed_panel_authorized": False,
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    opened24.write_json(args.output_dir / "completion_receipt.json", receipt)


def integration_authorized(gates: dict[str, bool]) -> bool:
    expected = {
        "all_attempts_equal_frozen_v18",
        "complete_24case_three_repeat_coverage",
        "all_full_steps_within_frozen_500ms",
        "all_full_steps_within_450ms_engineering_margin",
        "all_selectors_pass",
        "selected_pcm24_durable_byte_equivalence",
        "full_step_timer_envelope_and_phase_accounting",
    }
    if set(gates) != expected:
        raise ValueError("v19 integration gate coverage drift")
    return all(gates.values())


def repeated_key_sets_equal(
    rows: list[dict[str, Any]],
    repeat_count: int,
    expected_keys: set[Any],
    key_fields: tuple[str, ...],
) -> bool:
    for repeat_index in range(1, repeat_count + 1):
        repeat_rows = [
            row for row in rows if row["repeat_index"] == repeat_index
        ]
        observed_keys = {
            tuple(row[field] for field in key_fields) for row in repeat_rows
        }
        normalized_keys = {
            key if isinstance(key, tuple) else (key,) for key in expected_keys
        }
        if observed_keys != normalized_keys or len(repeat_rows) != len(
            normalized_keys
        ):
            return False
    return True


def full_step_timer_contract(
    case_record: dict[str, Any],
    external_wall_ms: float,
    device_sync_call_count: int,
    device_sync_wall_ms: float,
) -> tuple[float, bool]:
    d_batch = case_record["candidate_d_batch_runtime"]
    d_refresh = case_record["candidate_d_refresh_runtime"]
    c_batch = case_record["candidate_c_batch_runtime"]
    c_refresh = case_record["candidate_c_refresh_runtime"]
    accounted_ms = sum(
        float(value)
        for value in (
            case_record["base_refresh_runtime_ms"],
            case_record["gradient_runtime_ms"],
            case_record["candidate_d_plan_runtime_ms"],
            case_record["candidate_d_projection_runtime_ms"],
            d_batch["candidate_gpu_to_cpu_batch_ms"],
            d_batch["candidate_pcm24_io_concurrent_wall_ms"],
            d_batch["candidate_frozen_proxy_batch_ms"],
            d_refresh["candidate_refresh_concurrent_wall_ms"],
            (
                c_batch["candidate_gpu_to_cpu_batch_ms"]
                if c_batch is not None
                else 0.0
            ),
            (
                c_batch["candidate_pcm24_io_concurrent_wall_ms"]
                if c_batch is not None
                else 0.0
            ),
            (
                c_batch["candidate_frozen_proxy_batch_ms"]
                if c_batch is not None
                else 0.0
            ),
            (
                c_refresh["candidate_refresh_concurrent_wall_ms"]
                if c_refresh is not None
                else 0.0
            ),
        )
    )
    total_ms = float(case_record["total_metric_step_runtime_ms"])
    phase_values_nonnegative = all(
        math.isfinite(value) and value >= 0.0
        for value in (accounted_ms, total_ms, external_wall_ms, device_sync_wall_ms)
    )
    timer_encloses_phases = total_ms + 2.0 >= accounted_ms
    external_encloses_frozen_timer = external_wall_ms + 0.1 >= total_ms
    candidate_staging_present = all(
        float(attempt["candidate_refresh_client_staging_ms"]) >= 0.0
        and attempt["candidate_pcm24_readback_used_for_refresh"]
        for attempt in case_record["attempts"]
    )
    worker_wall_present = (
        float(d_refresh["candidate_refresh_request_wall_sum_ms"]) > 0.0
        and (
            c_refresh is None
            or float(c_refresh["candidate_refresh_request_wall_sum_ms"]) > 0.0
        )
    )
    passed = all(
        (
            phase_values_nonnegative,
            timer_encloses_phases,
            external_encloses_frozen_timer,
            candidate_staging_present,
            worker_wall_present,
            device_sync_call_count > 0,
            case_record["selector_uses_no_candidate_exact_outcome"],
        )
    )
    return accounted_ms, passed


def main() -> None:
    args = parse_args()
    source_provenance = validate_repository_provenance(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.repeats < DEFAULT_REPEATS:
        raise ValueError("v19 full-step integration requires three repeats")
    if not TMPFS_ROOT.is_dir():
        raise FileNotFoundError("v19 full-step integration requires /dev/shm")

    panel_rows, input_by_case, source_hashes, panel_metadata = (
        opened24.validate_sources_and_inputs(args)
    )
    if len(panel_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("v19 integration opened-case coverage drift")
    if len({row["speaker_id"] for row in panel_rows}) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("v19 integration speaker coverage drift")
    v18_hashes, v18_rows, v18_failures = validate_v18_evidence(args)
    v19_topology_hashes = validate_v19_topology_evidence(args)
    references = load_reference_attempts(v18_rows)
    source_hashes.update(
        {
            f"immutable_v18_{name}": value
            for name, value in v18_hashes.items()
        }
    )
    source_hashes.update(
        {
            f"immutable_v19_topology_{name}": value
            for name, value in v19_topology_hashes.items()
        }
    )
    source_hashes.update(source_provenance["implementation_sha256"])

    args.output_dir.mkdir(parents=True)
    durable_root = args.output_dir / "durable_selected_pcm24"
    durable_root.mkdir()
    device = torch.device(args.device)
    predictor, _, _, target_scale = opened24.load_predictor(
        args.predictor_checkpoint,
        device,
    )
    torch_warmup = opened24.synthetic_torch_warmup(
        predictor,
        target_scale,
        device,
    )
    candidate_d_warmup = opened24.synthetic_candidate_d_warmup(device)
    selector_warmup = opened24.synthetic_v18_warmup(device)

    workers: list[PairedPeakCertificateTopologyWorker] = []
    worker_startups: list[dict[str, Any]] = []
    worker_warmups: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    durable_rows: list[dict[str, Any]] = []
    paired_refresh = make_paired_refresh(references)
    original_refresh = selector_core.refresh_candidate_records
    original_sync = selector_core.synchronize
    sync_state = {"call_count": 0, "wall_ms": 0.0}

    def timed_synchronize(sync_device: torch.device) -> None:
        started = time.perf_counter()
        original_sync(sync_device)
        sync_state["call_count"] += 1
        sync_state["wall_ms"] += 1000.0 * (
            time.perf_counter() - started
        )

    try:
        for worker_index in range(selector_core.WORKER_COUNT):
            worker = PairedPeakCertificateTopologyWorker(
                args.exact_python,
                args.v19_worker,
                args.avqi_code_root,
                args.avqi_code_tree_sha256,
            )
            workers.append(worker)
            warmup, warmup_ms = worker.warmup()
            worker_startups.append(
                {
                    "worker_index": worker_index,
                    "startup_ms": worker.startup_ms,
                    **worker.startup,
                }
            )
            worker_warmups.append(
                {
                    "worker_index": worker_index,
                    "request_wall_ms": warmup_ms,
                    **warmup,
                }
            )
        selector_core.refresh_candidate_records = paired_refresh
        selector_core.synchronize = timed_synchronize
        with tempfile.TemporaryDirectory(
            prefix="avqi-shimmer-v19-full-step-",
            dir=TMPFS_ROOT,
        ) as tmpfs_directory:
            waveform_root = Path(tmpfs_directory)
            with ThreadPoolExecutor(
                max_workers=selector_core.WORKER_COUNT
            ) as executor:
                for repeat_index in range(1, args.repeats + 1):
                    for case_index, panel_row in enumerate(panel_rows, start=1):
                        sync_state["call_count"] = 0
                        sync_state["wall_ms"] = 0.0
                        selector_call_started = time.perf_counter()
                        case_record = selector_core.evaluate_selector_case(
                            panel_row,
                            float(
                                input_by_case[panel_row["case_id"]][
                                    "exact_target_shimmer_db"
                                ]
                            ),
                            predictor,
                            target_scale,
                            device,
                            workers,
                            executor,
                            waveform_root,
                        )
                        selector_call_wall_ms = 1000.0 * (
                            time.perf_counter() - selector_call_started
                        )
                        accounted_ms, timer_contract_pass = (
                            full_step_timer_contract(
                                case_record,
                                selector_call_wall_ms,
                                int(sync_state["call_count"]),
                                float(sync_state["wall_ms"]),
                            )
                        )
                        case_record.update(
                            {
                                "selector_call_external_wall_ms": (
                                    selector_call_wall_ms
                                ),
                                "device_sync_call_count": int(
                                    sync_state["call_count"]
                                ),
                                "device_sync_wall_ms": float(
                                    sync_state["wall_ms"]
                                ),
                                "phase_accounted_sequential_wall_ms": (
                                    accounted_ms
                                ),
                                "full_step_timer_contract_pass": (
                                    timer_contract_pass
                                ),
                            }
                        )
                        runtime_rows.append(
                            case_runtime_row(
                                panel_row,
                                case_record,
                                repeat_index,
                                v18_failures,
                            )
                        )
                        durable_rows.append(
                            copy_selected_pcm24(
                                case_record,
                                durable_root,
                                repeat_index,
                            )
                        )
                        attempt_rows.extend(
                            compare_attempt_rows(
                                panel_row,
                                case_record,
                                references,
                                repeat_index,
                            )
                        )
                        print(
                            "v19_full_step_integration="
                            f"repeat_{repeat_index}:"
                            f"{case_index}/{EXPECTED_CASE_COUNT}",
                            flush=True,
                        )
    finally:
        selector_core.refresh_candidate_records = original_refresh
        selector_core.synchronize = original_sync
        for worker in workers:
            worker.close()

    attempts_path = args.output_dir / "full_step_attempt_equivalence.csv"
    runtime_path = args.output_dir / "full_step_case_runtime_repeats.csv"
    durable_path = args.output_dir / "durable_selected_equivalence.csv"
    opened24.write_csv(attempts_path, attempt_rows)
    opened24.write_csv(runtime_path, runtime_rows)
    opened24.write_csv(durable_path, durable_rows)

    expected_attempt_rows = EXPECTED_REFERENCE_ATTEMPT_COUNT * args.repeats
    expected_runtime_rows = EXPECTED_CASE_COUNT * args.repeats
    reference_keys = set(references)
    expected_case_ids = {row["case_id"] for row in panel_rows}
    attempt_key_coverage_pass = repeated_key_sets_equal(
        attempt_rows,
        args.repeats,
        reference_keys,
        ("case_id", "attempt_id"),
    )
    runtime_key_coverage_pass = repeated_key_sets_equal(
        runtime_rows,
        args.repeats,
        expected_case_ids,
        ("case_id",),
    )
    durable_key_coverage_pass = repeated_key_sets_equal(
        durable_rows,
        args.repeats,
        expected_case_ids,
        ("case_id",),
    )
    attempt_equivalence_pass = (
        len(attempt_rows) == expected_attempt_rows
        and attempt_key_coverage_pass
        and all(row["attempt_equivalence_pass"] for row in attempt_rows)
    )
    full_coverage_pass = (
        len(runtime_rows) == expected_runtime_rows
        and len(durable_rows) == expected_runtime_rows
        and runtime_key_coverage_pass
        and durable_key_coverage_pass
    )
    formal_runtime_pass = all(
        row["formal_500ms_pass"] for row in runtime_rows
    )
    engineering_margin_pass = all(
        row["engineering_450ms_margin_pass"] for row in runtime_rows
    )
    selector_pass = all(row["selector_pass"] for row in runtime_rows)
    timer_contract_pass = all(
        row["full_step_timer_contract_pass"] for row in runtime_rows
    )
    durable_pass = all(
        row["selected_candidate_present"]
        and row["durable_byte_equivalence_pass"]
        and row["selected_path_updated_to_durable_before_future_seal"]
        for row in durable_rows
    )
    gates = {
        "all_attempts_equal_frozen_v18": attempt_equivalence_pass,
        "complete_24case_three_repeat_coverage": full_coverage_pass,
        "all_full_steps_within_frozen_500ms": formal_runtime_pass,
        "all_full_steps_within_450ms_engineering_margin": (
            engineering_margin_pass
        ),
        "all_selectors_pass": selector_pass,
        "selected_pcm24_durable_byte_equivalence": durable_pass,
        "full_step_timer_envelope_and_phase_accounting": timer_contract_pass,
    }
    all_gates_pass = integration_authorized(gates)
    decision = PASS_DECISION if all_gates_pass else FAIL_DECISION
    runtime_values = [
        float(row["total_metric_step_runtime_ms"]) for row in runtime_rows
    ]
    failure_runtime = {
        case_id: {
            "repeat_runtime_ms": [
                float(row["total_metric_step_runtime_ms"])
                for row in runtime_rows
                if row["case_id"] == case_id
            ],
            "maximum_ms": max(
                float(row["total_metric_step_runtime_ms"])
                for row in runtime_rows
                if row["case_id"] == case_id
            ),
        }
        for case_id in v18_failures
    }
    report = {
        "schema_version": (
            "avqi-route-c-shimmer-db-runtime-v19-full-step-integration-v1"
        ),
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "phase": "opened24_full_selector_step_integration_only",
        "dev_only": True,
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "v18_immutable_job_id": V18_SLURM_JOB_ID,
        "v19_topology_only_job_id": V19_TOPOLOGY_SLURM_JOB_ID,
        "v18_artifacts_mutated": False,
        "v19_topology_artifacts_mutated": False,
        "scientific_gates_changed": False,
        "selector_contract": selector_core.selector_contract(),
        "formal_runtime_gate_ms": FORMAL_RUNTIME_GATE_MS,
        "engineering_margin_ms": ENGINEERING_MARGIN_MS,
        "repeat_count_per_case": args.repeats,
        "case_count": EXPECTED_CASE_COUNT,
        "speaker_count": EXPECTED_SPEAKER_COUNT,
        "source_sha256": source_hashes,
        "source_provenance": source_provenance,
        "panel_bindings": panel_metadata,
        "runtime_environment": {
            "torch_synthetic_warmup": torch_warmup,
            "candidate_d_synthetic_warmup": candidate_d_warmup,
            "selector_synthetic_warmup": selector_warmup,
            "worker_startups": worker_startups,
            "worker_synthetic_warmups": worker_warmups,
            "worker_count": selector_core.WORKER_COUNT,
            "candidate_pcm24_staging": "node_local_dev_shm",
            "selected_durable_copy_after_timed_step": True,
            "warmups_outside_case_timer": True,
        },
        "gates": gates,
        "runtime": {
            "measurement_count": len(runtime_rows),
            "median_ms": median(runtime_values),
            "maximum_ms": max(runtime_values),
            "v18_failure_cases": failure_runtime,
            "timer_contract": {
                "outer_timer": (
                    "frozen evaluate_selector_case total_started through "
                    "final device synchronize"
                ),
                "external_selector_call_wall_encloses_outer_timer": True,
                "base_refresh_included": True,
                "gradient_plan_projection_included": True,
                "pcm24_write_read_hash_proxy_safety_included": True,
                "candidate_and_base_staging_included": True,
                "worker_request_wall_included": True,
                "selector_final_device_sync_included": True,
                "durable_copy_outside_timer": True,
                "parallel_refresh_uses_concurrent_wall_not_request_sum": True,
            },
        },
        "coverage": {
            "attempt_reference_key_set_equal_each_repeat": (
                attempt_key_coverage_pass
            ),
            "runtime_case_set_unique_and_complete_each_repeat": (
                runtime_key_coverage_pass
            ),
            "durable_case_set_unique_and_complete_each_repeat": (
                durable_key_coverage_pass
            ),
        },
        "candidate_input_contract": {
            "candidate_float32_source": "tmpfs_pcm24_file_readback_only",
            "candidate_pcm24_hash_bound_to_frozen_v18": True,
            "candidate_raw_float32_hash_bound_to_frozen_v18": True,
            "prequantization_tensor_sent_to_worker": False,
        },
        "paired_base_contract": {
            "timing_source": "current_case_base_topology_timing_ms",
            "base_float32_source": "current_case_context_base_values",
            "case_view_topology_source_hash_bound": True,
            "cross_case_or_stale_timing_fail_closed": True,
        },
        "durable_output_contract": {
            "only_selected_pcm24_copied_to_durable_output": True,
            "copy_occurs_after_timed_selector_step": True,
            "selected_path_updated_before_future_seal": True,
            "byte_identical_sha_verified": durable_pass,
        },
        "opened24_rerun_authorized": all_gates_pass,
        "new_sealed_panel_authorized": False,
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    report_path = args.output_dir / "diagnostic_report.json"
    opened24.write_json(report_path, report)
    write_completion_receipt(
        args,
        decision,
        [report_path, attempts_path, runtime_path, durable_path],
        source_provenance,
        all_gates_pass,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "maximum_full_step_ms": max(runtime_values),
                "opened24_rerun_authorized": all_gates_pass,
                "candidate_exact_avqi_components_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
