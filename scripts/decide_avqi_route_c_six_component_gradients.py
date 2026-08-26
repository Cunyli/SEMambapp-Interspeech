#!/usr/bin/env python3
"""Apply the frozen Route C six-component code-gradient decision contract.

This is a JSON-only decision layer over an immutable raw measurement report.
It does not load audio or models, recompute gradients, invoke exact scoring, or
authorize a joint panel.  The numeric contract is frozen from the accepted
five-component audit before any six-component holdout is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping


DECISION_SCHEMA_VERSION = "avqi-route-c-six-gradient-decision-v1"
DECISION_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-gradient-decision-receipt-v1"
)
RAW_SCHEMA_VERSION = "dev-avqi-route-c-six-gradient-raw-measurement-v1"
RAW_RECEIPT_SCHEMA_VERSION = (
    "dev-avqi-route-c-six-gradient-raw-measurement-receipt-v1"
)
RAW_PENDING_DECISION = (
    "PENDING_ROUTE_C_SIX_COMPONENT_GRADIENT_GATES_UNFROZEN"
)
PASS_DECISION = "PASS_ROUTE_C_SIX_ACTIVE_CODE_GRADIENT_AUDIT"
NO_GO_DECISION = "NO_GO_ROUTE_C_SIX_ACTIVE_CODE_GRADIENT_AUDIT"
JOINT_PANEL_NO_GO = "NO_GO_ROUTE_C_SIX_JOINT_PANEL"
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"

FROZEN_FIVE_JOB_ID = "19906556"
FROZEN_FIVE_REPORT_SHA256 = (
    "b752377dbb14d4f91616b5179457f3246ffa173df5d423c2ee503cc19a519847"
)
FROZEN_FIVE_RECEIPT_SHA256 = (
    "34ad15ea001ffada940b74edf06b4b71ad98a4756d121e4f98361192470f6bc6"
)
ACCEPTED_DECISION_BASE_COMMIT = "868520cdd4d38cc9246cc89514d85164975ef7c1"
FIVE_REPORT_SCHEMA_VERSION = "avqi_route_c_five_component_gradient_audit_v1"
FIVE_PASS_DECISION = "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT"

ACTIVE_COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
FIVE_ACTIVE_COMPONENTS = tuple(
    component for component in ACTIVE_COMPONENTS if component != "shimmer_db"
)
AUDIT_SPLITS = ("surrogate_calibration", "surrogate_holdout")
SELECTION_STRATA = (
    "pathological_mild/cs",
    "pathological_mild/sv",
    "pathological_severe/cs",
    "pathological_severe/sv",
)
PAIRWISE_COMPONENT_KEYS = tuple(
    f"{left}__{right}"
    for left_index, left in enumerate(ACTIVE_COMPONENTS)
    for right in ACTIVE_COMPONENTS[left_index + 1 :]
)
EXPECTED_CASES_PER_SPLIT = 4
EXPECTED_CASES = 8
NONZERO_GRADIENT_NORM_MIN = 1e-10
MAXIMUM_GRADIENT_NORM = 1e4
MAXIMUM_WEIGHTED_COMPONENT_SHARE = 0.80
MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO = 1.000001
MINIMUM_COMPONENT_TO_JOINT_COSINE = 0.0

LOSS_TARGET = (
    "normalized bidirectional gap to same-speaker clean pathological CS/SV target"
)
WEIGHT_RULE = (
    "minimum calibration median gradient norm / component median gradient norm"
)
TOPOLOGY_IMPLEMENTATION = "exact_paired_peak_certificate_tmpfs_v19"
TOPOLOGY_HIGHPASS = "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
TOPOLOGY_LOADER = "client_tmpfs_raw_float32_current_output"

FIVE_COMPONENT_EVIDENCE_KEYS = (
    "cpps_report",
    "cpps_receipt",
    "hnr_report",
    "hnr_receipt",
    "shimmer_percent_report",
    "shimmer_percent_receipt",
    "slope_report",
    "slope_receipt",
    "slope_final_panel_seal",
    "slope_final_results",
    "tilt_report",
    "tilt_receipt",
)
RAW_SOURCE_EVIDENCE_KEYS = (
    *FIVE_COMPONENT_EVIDENCE_KEYS,
    "five_gradient_report",
    "five_gradient_receipt",
    "cpps_checkpoint",
    "hnr_checkpoint",
    "shimmer_percent_checkpoint",
    "slope_checkpoint",
    "tilt_checkpoint",
    "label_bank",
    "v19_evidence_manifest",
    "topology_manifest",
    "focused_test_evidence",
)
READINESS_SOURCE_EVIDENCE_KEYS = (
    *FIVE_COMPONENT_EVIDENCE_KEYS,
    "five_gradient_report",
    "five_gradient_receipt",
    "v19_runtime_evidence_manifest",
)
RAW_IMPLEMENTATION_KEYS = (
    "evaluate_avqi_route_c_six_component_gradients.py",
    "run_avqi_route_c_six_component_gradient_audit.sh",
)
RAW_IMPLEMENTATION_PATHS = {
    "evaluate_avqi_route_c_six_component_gradients.py": Path(
        "scripts/evaluate_avqi_route_c_six_component_gradients.py"
    ),
    "run_avqi_route_c_six_component_gradient_audit.sh": Path(
        "scripts/run_avqi_route_c_six_component_gradient_audit.sh"
    ),
}
DECISION_IMPLEMENTATION_KEYS = (
    "decide_avqi_route_c_six_component_gradients.py",
    "run_avqi_route_c_six_component_gradient_decision.sh",
)
FROZEN_GATE_KEYS = (
    "accepted_five_gradient_precedent_bound",
    "raw_pending_measurement_bound_without_rewrite",
    "exact_component_order",
    "exact_eight_case_dev_selection",
    "calibration_holdout_speaker_disjoint",
    "mild_severe_cs_sv_coverage_each_split",
    "final_and_fresh_panels_closed",
    "current_waveform_v19_base_topology_coverage_8_of_8",
    "slot2_slot3_separation",
    "zero_scorer_parameters",
    "same_speaker_clean_pathological_bidirectional_target",
    "avqi_coefficient_direction_unused",
    "all_component_gradients_finite_nonzero_bounded",
    "all_joint_gradients_finite_nonzero_bounded",
    "calibration_only_inverse_gradient_weights",
    "calibration_weighted_median_ratio_le_1_000001",
    "all_weighted_component_shares_le_0_80",
    "all_15_pairwise_cosines_finite_reported",
    "all_6_component_to_joint_cosines_finite_reported",
    "all_component_to_joint_cosines_nonnegative",
    "generator_optimizer_steps_zero",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite")
    return parsed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 is invalid")
    resolved = path.resolve()
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def _require_boundary(
    value: Mapping[str, Any], key: str, expected: Any, label: str
) -> None:
    if value.get(key) != expected or type(value.get(key)) is not type(expected):
        raise ValueError(f"{label} {key} differs")


def _require_no_go_boundaries(value: Mapping[str, Any], label: str) -> None:
    for key in (
        "joint_scientific_promotion_granted",
        "combined_final_panel_opened",
    ):
        if key in value:
            _require_boundary(value, key, False, label)
    for key in (
        "scientific_promotion_granted",
        "joint_panel_authorized",
        "fresh_panel_opened",
        "exact_candidate_scoring_requested",
        "waveform_generation_performed",
        "waveform_mutation_performed",
        "generator_loaded",
        "formal_generator_training_submitted",
    ):
        if key in value:
            _require_boundary(value, key, False, label)
    _require_boundary(value, "generator_optimizer_steps", 0, label)
    _require_boundary(value, "authoritative_training_decision", TRAINING_NO_GO, label)


def _require_raw_pending_boundaries(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    common_false = (
        "scientific_promotion_granted",
        "joint_panel_authorized",
        "combined_final_panel_opened",
        "fresh_panel_opened",
        "exact_candidate_scoring_requested",
        "waveform_generation_performed",
        "formal_generator_training_submitted",
    )
    for value, label in ((report, "raw report"), (receipt, "raw receipt")):
        for key in common_false:
            _require_boundary(value, key, False, label)
        _require_boundary(value, "generator_optimizer_steps", 0, label)
        _require_boundary(
            value, "authoritative_training_decision", TRAINING_NO_GO, label
        )
    for key in ("waveform_mutation_performed", "generator_loaded"):
        _require_boundary(report, key, False, "raw report")


def _case_selector(row: Mapping[str, Any], *, raw: bool) -> tuple[str, ...]:
    digest_key = "source_audio_file_sha256" if raw else "audio_sha256"
    required = (
        "split",
        "speaker_id",
        "sample_id",
        "sample_group",
        "view",
        "condition",
        digest_key,
    )
    if any(not isinstance(row.get(key), str) or not row[key] for key in required):
        raise ValueError("gradient case selector fields differ")
    if not _is_sha256(row[digest_key]):
        raise ValueError("gradient case audio SHA-256 differs")
    return tuple(str(row[key]) for key in required)


def _selection_case_contract(
    rows: list[Mapping[str, Any]], label: str, *, raw: bool
) -> set[tuple[str, ...]]:
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"{label} does not contain exactly eight cases")
    selectors = {_case_selector(row, raw=raw) for row in rows}
    if len(selectors) != EXPECTED_CASES:
        raise ValueError(f"{label} cases are not unique")
    for split in AUDIT_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        strata = {
            f"{row['sample_group']}/{row['view']}" for row in split_rows
        }
        if len(split_rows) != EXPECTED_CASES_PER_SPLIT or strata != set(
            SELECTION_STRATA
        ):
            raise ValueError(f"{label} {split} strata differ")
    return selectors


def validate_five_gradient_precedent(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    report_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the sole frozen numeric precedent and return its selection."""
    if report_sha256 != FROZEN_FIVE_REPORT_SHA256:
        raise ValueError("accepted five-gradient report SHA-256 differs")
    if receipt_sha256 != FROZEN_FIVE_RECEIPT_SHA256:
        raise ValueError("accepted five-gradient receipt SHA-256 differs")
    if report.get("schema_version") != FIVE_REPORT_SCHEMA_VERSION:
        raise ValueError("accepted five-gradient schema differs")
    if report.get("decision") != FIVE_PASS_DECISION:
        raise ValueError("accepted five-gradient report did not pass")
    if receipt.get("decision") != FIVE_PASS_DECISION:
        raise ValueError("accepted five-gradient receipt did not pass")
    if str(receipt.get("slurm_job_id")) != FROZEN_FIVE_JOB_ID:
        raise ValueError("accepted five-gradient job ID differs")
    runtime = _mapping(report.get("runtime"), "accepted five-gradient runtime")
    if str(runtime.get("slurm_job_id")) != FROZEN_FIVE_JOB_ID:
        raise ValueError("accepted five-gradient report job ID differs")
    receipt_hashes = _mapping(
        receipt.get("artifact_sha256"), "accepted five-gradient receipt hashes"
    )
    if receipt_hashes.get("gradient_interference_report.json") != report_sha256:
        raise ValueError("accepted five-gradient receipt does not bind report")
    gates = _mapping(report.get("gates"), "accepted five-gradient gates")
    if not gates or any(value is not True for value in gates.values()):
        raise ValueError("accepted five-gradient gates differ")
    contract = _mapping(report.get("contract"), "accepted five-gradient contract")
    if (
        contract.get("loss_target") != LOSS_TARGET
        or contract.get("avqi_scalar_coefficient_used_for_direction") is not False
        or contract.get("calibration_only_weight_selection") is not True
        or contract.get("weight_rule") != WEIGHT_RULE
    ):
        raise ValueError("accepted five-gradient target/weight contract differs")
    selection = _mapping(report.get("selection"), "accepted five-gradient selection")
    expected_selection = {
        "allowed_splits": list(AUDIT_SPLITS),
        "cases": EXPECTED_CASES,
        "cases_by_split": {
            split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS
        },
        "speaker_overlap": 0,
        "strata": list(SELECTION_STRATA),
        "final_panel_opened": False,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"accepted five-gradient selection {key} differs")
    speaker_sets = []
    speakers_by_split = _mapping(
        selection.get("speakers_by_split"), "accepted five-gradient speakers"
    )
    for split in AUDIT_SPLITS:
        speakers = speakers_by_split.get(split)
        if (
            not isinstance(speakers, list)
            or len(speakers) != EXPECTED_CASES_PER_SPLIT
            or len(set(speakers)) != len(speakers)
        ):
            raise ValueError(f"accepted five-gradient {split} speakers differ")
        speaker_sets.append(set(speakers))
    if speaker_sets[0] & speaker_sets[1]:
        raise ValueError("accepted five-gradient speakers overlap")
    case_results = report.get("case_results")
    if not isinstance(case_results, list) or any(
        not isinstance(row, dict) for row in case_results
    ):
        raise ValueError("accepted five-gradient cases differ")
    selectors = _selection_case_contract(
        case_results, "accepted five-gradient", raw=False
    )
    for split in ("calibration", "holdout"):
        summary = _mapping(report.get(split), f"accepted five-gradient {split}")
        if summary.get("cases") != EXPECTED_CASES_PER_SPLIT:
            raise ValueError(f"accepted five-gradient {split} coverage differs")
        components = _mapping(
            summary.get("components"), f"accepted five-gradient {split} components"
        )
        if set(components) != set(FIVE_ACTIVE_COMPONENTS):
            raise ValueError(f"accepted five-gradient {split} component order differs")
        if any(
            _mapping(components[name], f"accepted {name}").get(
                "opposed_to_joint_cases"
            )
            != 0
            for name in FIVE_ACTIVE_COMPONENTS
        ):
            raise ValueError("accepted five-gradient opposed-to-joint precedent differs")
    calibration = _mapping(report.get("calibration"), "five-gradient calibration")
    ratio = _number(
        calibration.get("weighted_median_norm_ratio"),
        "five-gradient weighted median ratio",
    )
    if ratio > MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO:
        raise ValueError("accepted five-gradient balance precedent differs")
    if not any(
        item.get("direction_conflict_cases", 0) > 0
        for split in ("calibration", "holdout")
        for item in _mapping(
            _mapping(report[split], f"five-gradient {split}").get(
                "pairwise_component_cosines"
            ),
            f"five-gradient {split} pairwise cosines",
        ).values()
        if isinstance(item, dict)
    ):
        raise ValueError("accepted five-gradient negative-pair precedent is absent")
    _require_no_go_boundaries(report, "accepted five-gradient report")
    _require_no_go_boundaries(receipt, "accepted five-gradient receipt")
    return {
        "selection": dict(selection),
        "case_selectors": selectors,
        "speaker_sets": {
            split: list(speakers_by_split[split]) for split in AUDIT_SPLITS
        },
    }


def _validate_source_evidence(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, str]:
    evidence = _mapping(report.get("source_evidence"), "raw source evidence")
    evidence_hashes = _mapping(
        report.get("source_evidence_sha256"), "raw source evidence hashes"
    )
    if set(evidence) != set(RAW_SOURCE_EVIDENCE_KEYS) or set(
        evidence_hashes
    ) != set(RAW_SOURCE_EVIDENCE_KEYS):
        raise ValueError("raw source evidence keys differ")
    parsed: dict[str, str] = {}
    for key in RAW_SOURCE_EVIDENCE_KEYS:
        binding = _mapping(evidence[key], f"raw source evidence {key}")
        if set(binding) != {"path", "sha256"}:
            raise ValueError(f"raw source evidence {key} binding differs")
        if not isinstance(binding["path"], str) or not Path(
            binding["path"]
        ).is_absolute():
            raise ValueError(f"raw source evidence {key} path differs")
        digest = binding["sha256"]
        if not _is_sha256(digest) or evidence_hashes[key] != digest:
            raise ValueError(f"raw source evidence {key} SHA-256 differs")
        parsed[key] = digest
    if parsed["five_gradient_report"] != FROZEN_FIVE_REPORT_SHA256:
        raise ValueError("raw report does not bind accepted five-gradient report")
    if parsed["five_gradient_receipt"] != FROZEN_FIVE_RECEIPT_SHA256:
        raise ValueError("raw report does not bind accepted five-gradient receipt")
    if receipt.get("source_evidence_sha256") != evidence_hashes:
        raise ValueError("raw receipt source evidence differs")
    return parsed


def _validate_raw_envelope(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    report_name: str,
    report_sha256: str,
    precedent: Mapping[str, Any],
    execution_source: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    if not _is_sha256(report_sha256):
        raise ValueError("raw six-gradient report SHA-256 is invalid")
    if report.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("raw six-gradient report schema differs")
    if receipt.get("schema_version") != RAW_RECEIPT_SCHEMA_VERSION:
        raise ValueError("raw six-gradient receipt schema differs")
    if report.get("decision") != RAW_PENDING_DECISION:
        raise ValueError("raw six-gradient report decision differs")
    if receipt.get("decision") != RAW_PENDING_DECISION:
        raise ValueError("raw six-gradient receipt decision differs")
    if report.get("joint_panel_decision") != JOINT_PANEL_NO_GO:
        raise ValueError("raw report joint-panel decision differs")
    if receipt.get("joint_panel_decision") != JOINT_PANEL_NO_GO:
        raise ValueError("raw receipt joint-panel decision differs")
    receipt_hashes = _mapping(receipt.get("artifact_sha256"), "raw receipt hashes")
    if receipt_hashes != {report_name: report_sha256}:
        raise ValueError("raw receipt does not exactly bind raw report")
    implementation = _mapping(
        receipt.get("implementation_sha256"), "raw implementation hashes"
    )
    if set(implementation) != set(RAW_IMPLEMENTATION_KEYS) or any(
        not _is_sha256(value) for value in implementation.values()
    ):
        raise ValueError("raw implementation provenance differs")
    if implementation != execution_source.get("raw_implementation_sha256"):
        raise ValueError("raw implementation hashes differ from clean source tree")
    if receipt.get("launcher_submitted_slurm_job") is not False:
        raise ValueError("raw launcher submitted an unreviewed job")
    if (
        receipt.get("scientific_schema_frozen") is not False
        or receipt.get("numeric_scientific_gates_applied") is not False
    ):
        raise ValueError("raw receipt incorrectly applies scientific gates")
    if tuple(receipt.get("active_components", ())) != ACTIVE_COMPONENTS:
        raise ValueError("raw receipt component order differs")
    if (
        receipt.get("calibration_cases") != EXPECTED_CASES_PER_SPLIT
        or receipt.get("holdout_cases") != EXPECTED_CASES_PER_SPLIT
    ):
        raise ValueError("raw receipt coverage differs")
    source_hashes = _validate_source_evidence(report, receipt)
    contract = _mapping(report.get("contract"), "raw six-gradient contract")
    raw_source = _mapping(contract.get("source"), "raw six-gradient source")
    if (
        raw_source.get("head") != execution_source.get("head")
        or receipt.get("source_commit") != execution_source.get("head")
        or raw_source.get("branch") != execution_source.get("branch")
        or receipt.get("source_branch") != execution_source.get("branch")
        or raw_source.get("accepted_base_commit")
        != receipt.get("accepted_base_commit")
    ):
        raise ValueError("raw report/receipt/source provenance differs")
    if tuple(contract.get("component_order", ())) != ACTIVE_COMPONENTS:
        raise ValueError("raw six-gradient component order differs")
    if (
        contract.get("loss_target") != LOSS_TARGET
        or contract.get("avqi_scalar_coefficient_used_for_direction") is not False
        or contract.get("weight_fit_split") != "surrogate_calibration"
        or contract.get("weight_rule") != WEIGHT_RULE
        or contract.get("scientific_schema_frozen") is not False
        or contract.get("numeric_scientific_gates_applied") is not False
    ):
        raise ValueError("raw six-gradient target/weight semantics differ")
    selection = _mapping(report.get("selection"), "raw six-gradient selection")
    for key, expected in {
        "allowed_splits": list(AUDIT_SPLITS),
        "cases": EXPECTED_CASES,
        "cases_by_split": {
            split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS
        },
        "speaker_overlap": 0,
        "strata": list(SELECTION_STRATA),
        "final_panel_opened": False,
        "component_and_joint_share_split": True,
        "topology_manifest_uses_same_selection": True,
    }.items():
        if selection.get(key) != expected:
            raise ValueError(f"raw six-gradient selection {key} differs")
    for split, key in (
        ("surrogate_calibration", "calibration_speaker_ids"),
        ("surrogate_holdout", "holdout_speaker_ids"),
    ):
        if selection.get(key) != precedent["speaker_sets"][split]:
            raise ValueError(f"raw six-gradient {key} differs from precedent")
    rows = report.get("case_results")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("raw six-gradient cases differ")
    raw_selectors = _selection_case_contract(rows, "raw six-gradient", raw=True)
    if raw_selectors != precedent["case_selectors"]:
        raise ValueError("raw six-gradient case selection differs from precedent")
    coverage = _mapping(report.get("coverage"), "raw six-gradient coverage")
    expected_coverage = {
        "selected_cases": EXPECTED_CASES,
        "cases_by_split": {
            split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS
        },
        "component_gradient_measurements": EXPECTED_CASES * len(ACTIVE_COMPONENTS),
        "pairwise_cosine_measurements": EXPECTED_CASES
        * len(PAIRWISE_COMPONENT_KEYS),
        "component_to_joint_cosine_measurements": EXPECTED_CASES
        * len(ACTIVE_COMPONENTS),
        "expected_pairwise_cosines_per_case": len(PAIRWISE_COMPONENT_KEYS),
        "expected_component_to_joint_cosines_per_case": len(ACTIVE_COMPONENTS),
    }
    if any(coverage.get(key) != value for key, value in expected_coverage.items()):
        raise ValueError("raw six-gradient measurement coverage differs")
    topology = _mapping(report.get("topology_coverage"), "raw topology coverage")
    expected_topology = {
        "expected_cases": EXPECTED_CASES,
        "observed_cases": EXPECTED_CASES,
        "unique_case_ids": EXPECTED_CASES,
        "cases_by_split": {
            split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS
        },
        "cases_by_view": {"cs": 4, "sv": 4},
        "exact_selection_coverage": True,
    }
    if any(topology.get(key) != value for key, value in expected_topology.items()):
        raise ValueError("raw v19 topology coverage differs")
    separation = _mapping(
        report.get("slot2_slot3_separation"), "raw slot separation"
    )
    slot2 = _mapping(separation.get("slot2_shimmer_percent"), "raw slot 2")
    slot3 = _mapping(separation.get("slot3_shimmer_db"), "raw slot 3")
    expected_slot2 = {
        "component_index": 2,
        "source": "sealed_shimmer_percent_checkpoint",
        "checkpoint_output_preserved": True,
        "v19_topology_used": False,
    }
    expected_slot3 = {
        "component_index": 3,
        "source": "current_waveform_with_detached_v19_base_topology",
        "checkpoint_affine_used": False,
        "v19_topology_used": True,
        "topology_role": "base_current_output",
        "implementation": TOPOLOGY_IMPLEMENTATION,
        "metric_highpass": TOPOLOGY_HIGHPASS,
        "topology_input_loader": TOPOLOGY_LOADER,
        "scientific_promotion_granted": False,
    }
    if (
        any(slot2.get(key) != value for key, value in expected_slot2.items())
        or any(slot3.get(key) != value for key, value in expected_slot3.items())
        or separation.get("slots_are_independent") is not True
    ):
        raise ValueError("raw slot 2/slot 3 contract differs")
    integrity = _mapping(report.get("measurement_integrity"), "raw integrity")
    required_true = (
        "six_component_order_exact",
        "all_15_pairwise_cosines_per_case",
        "all_6_component_to_joint_cosines_per_case",
        "calibration_holdout_speaker_disjoint",
        "topology_exact_selection_coverage",
        "slot2_slot3_sources_independent",
        "scorer_has_zero_parameters",
        "shimmer_db_scientific_status_pending",
        "v19_runtime_evidence_does_not_grant_promotion",
    )
    if any(integrity.get(key) is not True for key in required_true):
        raise ValueError("raw six-gradient integrity differs")
    for key in ("numeric_scientific_gates_applied", "final_or_fresh_panel_opened"):
        if integrity.get(key) is not False:
            raise ValueError(f"raw six-gradient integrity {key} differs")
    if integrity.get("generator_optimizer_steps") != 0:
        raise ValueError("raw six-gradient integrity optimizer steps differ")
    for row in rows:
        row_topology = _mapping(row.get("topology"), "raw case topology")
        if (
            row_topology.get("role") != "base_current_output"
            or row_topology.get("slot2_shimmer_percent_uses_topology") is not False
            or row_topology.get("slot3_shimmer_db_uses_topology") is not True
            or not _is_sha256(row_topology.get("topology_sha256"))
        ):
            raise ValueError("raw case topology contract differs")
    _require_raw_pending_boundaries(report, receipt)
    return rows, source_hashes


def _finite_mapping(
    value: Any, keys: tuple[str, ...], label: str, *, positive: bool = False
) -> dict[str, float]:
    mapping = _mapping(value, label)
    if set(mapping) != set(keys):
        raise ValueError(f"{label} keys differ")
    parsed = {key: _number(mapping[key], f"{label} {key}") for key in keys}
    if positive and any(number <= 0.0 for number in parsed.values()):
        raise ValueError(f"{label} values are not positive")
    return parsed


def _isclose(left: float, right: float) -> bool:
    # Raw weighted norms are produced by float32 tensor arithmetic.
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-8)


def _evaluate_numeric_gates(
    rows: list[Mapping[str, Any]], report: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    calibration_rows = [
        row for row in rows if row["split"] == "surrogate_calibration"
    ]
    component_norms: dict[str, list[float]] = {
        component: [] for component in ACTIVE_COMPONENTS
    }
    joint_norms: list[float] = []
    maximum_shares: list[float] = []
    joint_cosines: list[float] = []
    pairwise_cosines: list[float] = []
    weights_by_case: list[dict[str, float]] = []
    component_metadata_valid = True
    joint_metadata_valid = True
    all_shares_valid = True
    all_pairwise_reported = True
    all_joint_cosines_reported = True
    all_joint_cosines_nonnegative = True
    for row in rows:
        components = _mapping(row.get("components"), "raw case components")
        if set(components) != set(ACTIVE_COMPONENTS):
            raise ValueError("raw case component keys differ")
        for component in ACTIVE_COMPONENTS:
            item = _mapping(components[component], f"raw {component} measurement")
            norm = _number(item.get("gradient_norm"), f"raw {component} norm")
            component_norms[component].append(norm)
            component_metadata_valid &= (
                item.get("finite_observed") is True
                and item.get("strictly_positive_norm_observed") is True
                and item.get("scientific_gate_applied") is False
            )
            for field in ("prediction", "clean_pathological_target"):
                _number(item.get(field), f"raw {component} {field}")
            signed_error = _number(
                item.get("normalized_signed_error"),
                f"raw {component} normalized signed error",
            )
            gap = _number(
                item.get("normalized_bidirectional_gap"),
                f"raw {component} normalized bidirectional gap",
            )
            loss = _number(
                item.get("smooth_l1_loss"), f"raw {component} SmoothL1 loss"
            )
            if gap < 0.0 or loss < 0.0 or not _isclose(gap, abs(signed_error)):
                raise ValueError("raw normalized bidirectional target semantics differ")
        joint = _mapping(row.get("joint"), "raw case joint measurement")
        joint_norms.append(_number(joint.get("gradient_norm"), "raw joint norm"))
        joint_metadata_valid &= (
            joint.get("all_values_finite_observed") is True
            and joint.get("scientific_gate_applied") is False
        )
        weights_by_case.append(
            _finite_mapping(
                joint.get("calibration_only_inverse_gradient_weights"),
                ACTIVE_COMPONENTS,
                "raw case inverse-gradient weights",
                positive=True,
            )
        )
        weighted_norms = _finite_mapping(
            joint.get("weighted_component_gradient_norms"),
            ACTIVE_COMPONENTS,
            "raw case weighted gradient norms",
            positive=True,
        )
        shares = _finite_mapping(
            joint.get("weighted_component_norm_shares"),
            ACTIVE_COMPONENTS,
            "raw case weighted component shares",
        )
        if any(value < 0.0 or value > 1.0 for value in shares.values()):
            all_shares_valid = False
        if not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-6):
            all_shares_valid = False
        weighted_norm_sum = sum(weighted_norms.values())
        if weighted_norm_sum <= 0.0 or any(
            not _isclose(
                shares[component],
                weighted_norms[component] / weighted_norm_sum,
            )
            for component in ACTIVE_COMPONENTS
        ):
            raise ValueError("raw weighted component shares are inconsistent")
        observed_max = _number(
            joint.get("maximum_component_norm_share"), "raw maximum share"
        )
        if not _isclose(observed_max, max(shares.values())):
            raise ValueError("raw maximum weighted share is inconsistent")
        expected_dominant = max(shares, key=shares.__getitem__)
        if joint.get("dominant_component") != expected_dominant:
            raise ValueError("raw dominant component is inconsistent")
        maximum_shares.append(observed_max)
        pairwise = _mapping(
            joint.get("pairwise_component_cosines"), "raw pairwise cosines"
        )
        if set(pairwise) != set(PAIRWISE_COMPONENT_KEYS):
            raise ValueError("raw pairwise cosine keys differ")
        for pair in PAIRWISE_COMPONENT_KEYS:
            item = _mapping(pairwise[pair], f"raw pairwise cosine {pair}")
            cosine = _number(item.get("cosine"), f"raw pairwise cosine {pair}")
            pairwise_cosines.append(cosine)
            all_pairwise_reported &= (
                -1.0 <= cosine <= 1.0
                and item.get("negative_direction_observed") is (cosine < 0.0)
                and item.get("scientific_gate_applied") is False
            )
        to_joint = _mapping(
            joint.get("component_to_joint_cosines"), "raw component-to-joint cosines"
        )
        if set(to_joint) != set(ACTIVE_COMPONENTS):
            raise ValueError("raw component-to-joint cosine keys differ")
        for component in ACTIVE_COMPONENTS:
            item = _mapping(to_joint[component], f"raw {component}-to-joint cosine")
            cosine = _number(
                item.get("cosine"), f"raw {component}-to-joint cosine"
            )
            joint_cosines.append(cosine)
            all_joint_cosines_reported &= (
                -1.0 <= cosine <= 1.0
                and item.get("negative_direction_observed") is (cosine < 0.0)
                and item.get("scientific_gate_applied") is False
            )
            all_joint_cosines_nonnegative &= (
                cosine >= MINIMUM_COMPONENT_TO_JOINT_COSINE
            )
        for component in ACTIVE_COMPONENTS:
            expected_weighted = (
                component_norms[component][-1] * weights_by_case[-1][component]
            )
            if not _isclose(weighted_norms[component], expected_weighted):
                raise ValueError("raw weighted component norm is inconsistent")

    medians = {
        component: statistics.median(
            component_norms[component][index]
            for index, row in enumerate(rows)
            if row["split"] == "surrogate_calibration"
        )
        for component in ACTIVE_COMPONENTS
    }
    minimum_median = min(medians.values())
    expected_weights = {
        component: minimum_median / medians[component]
        for component in ACTIVE_COMPONENTS
    }
    calibration = _mapping(report.get("calibration"), "raw calibration summary")
    recorded_medians = _finite_mapping(
        calibration.get("median_component_gradient_norms"),
        ACTIVE_COMPONENTS,
        "raw calibration medians",
        positive=True,
    )
    recorded_weights = _finite_mapping(
        calibration.get("frozen_inverse_gradient_weights"),
        ACTIVE_COMPONENTS,
        "raw frozen inverse-gradient weights",
        positive=True,
    )
    recorded_weighted_medians = _finite_mapping(
        calibration.get("weighted_median_gradient_norms"),
        ACTIVE_COMPONENTS,
        "raw weighted calibration medians",
        positive=True,
    )
    weights_consistent = calibration.get("weights_selected_on_holdout") is False
    for component in ACTIVE_COMPONENTS:
        weights_consistent &= (
            _isclose(recorded_medians[component], medians[component])
            and _isclose(recorded_weights[component], expected_weights[component])
            and _isclose(
                recorded_weighted_medians[component],
                medians[component] * expected_weights[component],
            )
            and all(
                _isclose(weights[component], expected_weights[component])
                for weights in weights_by_case
            )
        )
    ratio = max(recorded_weighted_medians.values()) / min(
        recorded_weighted_medians.values()
    )
    for split, expected_rows in (
        ("calibration", calibration_rows),
        (
            "holdout",
            [row for row in rows if row["split"] == "surrogate_holdout"],
        ),
    ):
        summary = _mapping(report.get(split), f"raw {split} summary")
        if (
            summary.get("cases") != len(expected_rows)
            or summary.get("component_gradient_measurements")
            != len(expected_rows) * len(ACTIVE_COMPONENTS)
            or summary.get("pairwise_cosine_measurements")
            != len(expected_rows) * len(PAIRWISE_COMPONENT_KEYS)
            or summary.get("component_to_joint_cosine_measurements")
            != len(expected_rows) * len(ACTIVE_COMPONENTS)
            or summary.get("all_values_finite_observed") is not True
            or summary.get("scientific_gate_applied") is not False
        ):
            raise ValueError(f"raw {split} aggregate coverage differs")
    component_values = [
        value for values in component_norms.values() for value in values
    ]
    gates = {
        "all_component_gradients_finite_nonzero_bounded": (
            component_metadata_valid
            and all(
                value > NONZERO_GRADIENT_NORM_MIN
                and value <= MAXIMUM_GRADIENT_NORM
                for value in component_values
            )
        ),
        "all_joint_gradients_finite_nonzero_bounded": (
            joint_metadata_valid
            and all(
                value > NONZERO_GRADIENT_NORM_MIN
                and value <= MAXIMUM_GRADIENT_NORM
                for value in joint_norms
            )
        ),
        "calibration_only_inverse_gradient_weights": bool(weights_consistent),
        "calibration_weighted_median_ratio_le_1_000001": (
            ratio <= MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO
        ),
        "all_weighted_component_shares_le_0_80": (
            all_shares_valid
            and all(
                value <= MAXIMUM_WEIGHTED_COMPONENT_SHARE
                for value in maximum_shares
            )
        ),
        "all_15_pairwise_cosines_finite_reported": bool(all_pairwise_reported),
        "all_6_component_to_joint_cosines_finite_reported": bool(
            all_joint_cosines_reported
        ),
        "all_component_to_joint_cosines_nonnegative": bool(
            all_joint_cosines_nonnegative
        ),
    }
    metrics = {
        "calibration_cases": len(calibration_rows),
        "holdout_cases": len(rows) - len(calibration_rows),
        "component_gradient_norm_min": min(component_values),
        "component_gradient_norm_max": max(component_values),
        "joint_gradient_norm_min": min(joint_norms),
        "joint_gradient_norm_max": max(joint_norms),
        "calibration_inverse_gradient_weights": expected_weights,
        "calibration_weighted_median_norm_ratio": ratio,
        "maximum_weighted_component_norm_share": max(maximum_shares),
        "minimum_component_to_joint_cosine": min(joint_cosines),
        "negative_pairwise_cosine_observations": sum(
            value < 0.0 for value in pairwise_cosines
        ),
        "pairwise_negative_values_are_diagnostic_only": True,
    }
    return gates, metrics


def evaluate_six_gradient_decision(
    raw_report: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    five_report: Mapping[str, Any],
    five_receipt: Mapping[str, Any],
    *,
    raw_report_name: str,
    raw_report_sha256: str,
    raw_receipt_sha256: str,
    five_report_sha256: str,
    five_receipt_sha256: str,
    execution_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not _is_sha256(raw_receipt_sha256):
        raise ValueError("raw six-gradient receipt SHA-256 is invalid")
    precedent = validate_five_gradient_precedent(
        five_report,
        five_receipt,
        five_report_sha256,
        five_receipt_sha256,
    )
    rows, raw_source_hashes = _validate_raw_envelope(
        raw_report,
        raw_receipt,
        report_name=raw_report_name,
        report_sha256=raw_report_sha256,
        precedent=precedent,
        execution_source=execution_source,
    )
    numeric_gates, metrics = _evaluate_numeric_gates(rows, raw_report)
    structural_gates = {
        "accepted_five_gradient_precedent_bound": True,
        "raw_pending_measurement_bound_without_rewrite": True,
        "exact_component_order": True,
        "exact_eight_case_dev_selection": True,
        "calibration_holdout_speaker_disjoint": True,
        "mild_severe_cs_sv_coverage_each_split": True,
        "final_and_fresh_panels_closed": True,
        "current_waveform_v19_base_topology_coverage_8_of_8": True,
        "slot2_slot3_separation": True,
        "zero_scorer_parameters": True,
        "same_speaker_clean_pathological_bidirectional_target": True,
        "avqi_coefficient_direction_unused": True,
        "generator_optimizer_steps_zero": True,
    }
    gate_values = {**structural_gates, **numeric_gates}
    if set(gate_values) != set(FROZEN_GATE_KEYS):
        raise ValueError("six-gradient frozen gate keys differ")
    gates = {key: gate_values[key] for key in FROZEN_GATE_KEYS}
    decision = PASS_DECISION if all(gates.values()) else NO_GO_DECISION
    readiness_evidence = {
        key: raw_source_hashes[key] for key in FIVE_COMPONENT_EVIDENCE_KEYS
    }
    readiness_evidence.update(
        {
            "five_gradient_report": FROZEN_FIVE_REPORT_SHA256,
            "five_gradient_receipt": FROZEN_FIVE_RECEIPT_SHA256,
            "v19_runtime_evidence_manifest": raw_source_hashes[
                "v19_evidence_manifest"
            ],
        }
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "active_components": list(ACTIVE_COMPONENTS),
        "frozen_contract": decision_requirements()["frozen_contract"],
        "accepted_numeric_precedent": {
            "slurm_job_id": FROZEN_FIVE_JOB_ID,
            "report_sha256": FROZEN_FIVE_REPORT_SHA256,
            "receipt_sha256": FROZEN_FIVE_RECEIPT_SHA256,
        },
        "raw_measurement_evidence": {
            "report_sha256": raw_report_sha256,
            "receipt_sha256": raw_receipt_sha256,
            "raw_decision": RAW_PENDING_DECISION,
            "raw_artifacts_rewritten": False,
        },
        "source_evidence_sha256": readiness_evidence,
        "measurement_summary": metrics,
        "gates": gates,
        "scientific_contract_frozen_before_six_holdout_open": True,
        "raw_measurement_recomputed": False,
        "scientific_promotion_granted": False,
        "joint_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def decision_requirements() -> dict[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": "NO_GO_ROUTE_C_SIX_GRADIENT_DECISION_NOT_EXECUTED",
        "frozen_contract": {
            "frozen_from": {
                "slurm_job_id": FROZEN_FIVE_JOB_ID,
                "report_sha256": FROZEN_FIVE_REPORT_SHA256,
                "receipt_sha256": FROZEN_FIVE_RECEIPT_SHA256,
                "sole_numeric_precedent": True,
            },
            "component_order": list(ACTIVE_COMPONENTS),
            "splits": list(AUDIT_SPLITS),
            "cases_by_split": {
                split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS
            },
            "strata_per_split": list(SELECTION_STRATA),
            "speaker_overlap": 0,
            "topology_coverage": "8/8 current-waveform v19 base topology",
            "component_gradient_norm": {
                "finite": True,
                "strictly_greater_than": NONZERO_GRADIENT_NORM_MIN,
                "less_than_or_equal_to": MAXIMUM_GRADIENT_NORM,
            },
            "joint_gradient_norm": {
                "finite": True,
                "strictly_greater_than": NONZERO_GRADIENT_NORM_MIN,
                "less_than_or_equal_to": MAXIMUM_GRADIENT_NORM,
            },
            "weight_selection_split": "surrogate_calibration_only",
            "weight_rule": WEIGHT_RULE,
            "maximum_calibration_weighted_median_ratio": (
                MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO
            ),
            "maximum_weighted_component_norm_share_per_case": (
                MAXIMUM_WEIGHTED_COMPONENT_SHARE
            ),
            "pairwise_cosines": {
                "required_per_case": len(PAIRWISE_COMPONENT_KEYS),
                "finite_and_reported": True,
                "negative_values": "diagnostic_only",
            },
            "component_to_joint_cosines": {
                "required_per_case": len(ACTIVE_COMPONENTS),
                "finite_and_reported": True,
                "minimum": MINIMUM_COMPONENT_TO_JOINT_COSINE,
            },
            "loss_target": LOSS_TARGET,
            "avqi_scalar_coefficient_used_for_direction": False,
            "slot2_slot3_separated": True,
            "zero_scorer_parameters": True,
            "final_or_fresh_panel_opened": False,
            "generator_optimizer_steps": 0,
        },
        "required_raw_decision": RAW_PENDING_DECISION,
        "possible_code_gradient_decisions": [PASS_DECISION, NO_GO_DECISION],
        "joint_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "fresh_panel_opened": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def _repository_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source(root: Path, expected_commit: str) -> dict[str, Any]:
    resolved = root.resolve()
    head = _repository_value(resolved, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("six-gradient decision source HEAD differs")
    if _repository_value(resolved, "status", "--porcelain"):
        raise ValueError("six-gradient decision requires a clean source tree")
    subprocess.run(
        [
            "git",
            "-C",
            str(resolved),
            "merge-base",
            "--is-ancestor",
            ACCEPTED_DECISION_BASE_COMMIT,
            head,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    implementation_hashes = {}
    for name, relative_path in RAW_IMPLEMENTATION_PATHS.items():
        path = resolved / relative_path
        if not path.is_file():
            raise ValueError(f"raw implementation is unavailable: {name}")
        implementation_hashes[name] = sha256_file(path)
    return {
        "root": str(resolved),
        "head": head,
        "branch": _repository_value(resolved, "branch", "--show-current"),
        "accepted_base_commit": ACCEPTED_DECISION_BASE_COMMIT,
        "raw_implementation_sha256": implementation_hashes,
    }


def _post_evaluation_immutability(
    bindings: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    observed = {name: sha256_file(path) for name, (path, _) in bindings.items()}
    expected = {name: digest for name, (_, digest) in bindings.items()}
    if observed != expected:
        raise ValueError("decision input changed during evaluation")
    return {"verified": True, "artifact_sha256": observed}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-only", action="store_true")
    parser.add_argument("--raw-report", type=Path)
    parser.add_argument("--raw-report-sha256")
    parser.add_argument("--raw-receipt", type=Path)
    parser.add_argument("--raw-receipt-sha256")
    parser.add_argument("--five-precedent-report", type=Path)
    parser.add_argument("--five-precedent-report-sha256")
    parser.add_argument("--five-precedent-receipt", type=Path)
    parser.add_argument("--five-precedent-receipt-sha256")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    execution_inputs = (
        args.raw_report,
        args.raw_report_sha256,
        args.raw_receipt,
        args.raw_receipt_sha256,
        args.five_precedent_report,
        args.five_precedent_report_sha256,
        args.five_precedent_receipt,
        args.five_precedent_receipt_sha256,
        args.source_root,
        args.source_commit,
        args.output_dir,
    )
    if args.requirements_only:
        if any(value is not None for value in execution_inputs):
            raise ValueError("requirements-only mode accepts no evidence inputs")
        print(json.dumps(decision_requirements(), indent=2, sort_keys=True))
        return
    if any(value is None for value in execution_inputs):
        raise ValueError("six-gradient decision inputs are incomplete")
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
    raw_report_path = _verified_file(
        args.raw_report, args.raw_report_sha256, "raw six-gradient report"
    )
    raw_receipt_path = _verified_file(
        args.raw_receipt, args.raw_receipt_sha256, "raw six-gradient receipt"
    )
    five_report_path = _verified_file(
        args.five_precedent_report,
        args.five_precedent_report_sha256,
        "accepted five-gradient report",
    )
    five_receipt_path = _verified_file(
        args.five_precedent_receipt,
        args.five_precedent_receipt_sha256,
        "accepted five-gradient receipt",
    )
    source = _validate_source(args.source_root, args.source_commit)
    report = evaluate_six_gradient_decision(
        _read_json(raw_report_path, "raw six-gradient report"),
        _read_json(raw_receipt_path, "raw six-gradient receipt"),
        _read_json(five_report_path, "accepted five-gradient report"),
        _read_json(five_receipt_path, "accepted five-gradient receipt"),
        raw_report_name=raw_report_path.name,
        raw_report_sha256=args.raw_report_sha256,
        raw_receipt_sha256=args.raw_receipt_sha256,
        five_report_sha256=args.five_precedent_report_sha256,
        five_receipt_sha256=args.five_precedent_receipt_sha256,
        execution_source=source,
    )
    immutability = _post_evaluation_immutability(
        {
            "raw_report": (raw_report_path, args.raw_report_sha256),
            "raw_receipt": (raw_receipt_path, args.raw_receipt_sha256),
            "five_precedent_report": (
                five_report_path,
                args.five_precedent_report_sha256,
            ),
            "five_precedent_receipt": (
                five_receipt_path,
                args.five_precedent_receipt_sha256,
            ),
        }
    )
    report["decision_source"] = source
    report["post_evaluation_immutability"] = immutability
    report["raw_measurement_evidence"].update(
        {
            "report_path": str(raw_report_path),
            "receipt_path": str(raw_receipt_path),
        }
    )
    evaluator_path = Path(__file__).resolve()
    launcher_path = evaluator_path.with_name(
        "run_avqi_route_c_six_component_gradient_decision.sh"
    )
    if not launcher_path.is_file():
        raise ValueError("six-gradient decision launcher is unavailable")
    implementation_hashes = {
        evaluator_path.name: sha256_file(evaluator_path),
        launcher_path.name: sha256_file(launcher_path),
    }
    if tuple(implementation_hashes) != DECISION_IMPLEMENTATION_KEYS:
        raise ValueError("six-gradient decision implementation names differ")
    report["implementation_sha256"] = implementation_hashes
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "six_gradient_decision_report.json"
    _write_json(report_path, report)
    receipt = {
        "schema_version": DECISION_RECEIPT_SCHEMA_VERSION,
        "decision": report["decision"],
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "active_components": list(ACTIVE_COMPONENTS),
        "raw_measurement_sha256": {
            "report": args.raw_report_sha256,
            "receipt": args.raw_receipt_sha256,
        },
        "accepted_numeric_precedent": report["accepted_numeric_precedent"],
        "implementation_sha256": implementation_hashes,
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
        "raw_artifacts_rewritten": False,
        "post_evaluation_immutability": immutability,
        "launcher_submitted_slurm_job": False,
        "scientific_promotion_granted": False,
        "joint_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(
            json.dumps(
                {
                    "decision": NO_GO_DECISION,
                    "joint_panel_decision": JOINT_PANEL_NO_GO,
                    "reason": str(error),
                    "joint_panel_authorized": False,
                    "fresh_panel_opened": False,
                    "generator_optimizer_steps": 0,
                    "authoritative_training_decision": TRAINING_NO_GO,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2) from None
