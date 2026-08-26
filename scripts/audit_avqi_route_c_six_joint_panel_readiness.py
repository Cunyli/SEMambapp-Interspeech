#!/usr/bin/env python3
"""Fail-closed preflight for a future Route C six-component joint panel.

This script only audits immutable metadata and file hashes.  It cannot create
candidate waveforms, open a fresh panel, invoke Praat, or authorize training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


# Direct-path execution cannot import the project package until it re-enters
# through the supported module path. This narrow re-entry precedes local imports.
if __name__ == "__main__" and __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_avqi_route_c_six_joint_panel_readiness",
            *sys.argv[1:],
        ],
        cwd=project_root,
        check=False,
    )
    raise SystemExit(completed.returncode)

from model.avqi_route_c import (
    ROUTE_C_FIVE_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_SCIENTIFIC_STATUS,
    route_c_six_registry_records,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.decide_avqi_route_c_six_component_gradients import (
    ACTIVE_COMPONENTS as FROZEN_SIX_GRADIENT_COMPONENTS,
    DECISION_RECEIPT_SCHEMA_VERSION as SIX_GRADIENT_RECEIPT_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION as SIX_GRADIENT_SCHEMA_VERSION,
    DECISION_IMPLEMENTATION_KEYS as SIX_GRADIENT_DECISION_IMPLEMENTATION_KEYS,
    FROZEN_FIVE_JOB_ID,
    FROZEN_FIVE_RECEIPT_SHA256,
    FROZEN_FIVE_REPORT_SHA256,
    FROZEN_GATE_KEYS as SIX_GRADIENT_FROZEN_GATE_KEYS,
    JOINT_PANEL_NO_GO as SIX_GRADIENT_JOINT_PANEL_NO_GO,
    MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO,
    MAXIMUM_WEIGHTED_COMPONENT_SHARE,
    PASS_DECISION as SIX_GRADIENT_PASS_DECISION,
    RAW_PENDING_DECISION as SIX_GRADIENT_RAW_PENDING_DECISION,
    READINESS_SOURCE_EVIDENCE_KEYS,
    TRAINING_NO_GO,
    decision_requirements as six_gradient_decision_requirements,
)


READINESS_SCHEMA_VERSION = "avqi-route-c-six-joint-panel-readiness-v1"
DRAFT_SPLIT_SEAL_SCHEMA_VERSION = "draft-avqi-route-c-six-joint-split-seal-v1"
SHIMMER_DB_REQUIRED_STATUS = "fresh_speaker_panel_pass"
REQUIRED_SPLITS = ("calibration", "final")
REQUIRED_VIEWS = ("cs", "sv")
REQUIRED_CONDITIONS = ("clean", "rir_only", "snr20", "snr10")
REQUIRED_GUARDRAILS = (
    "full_band_low_frequency",
    "pause",
    "airflow",
    "cs_sv_pathology",
    "residual",
    "cosine",
    "clipping",
    "snr",
    "si_sdr",
)
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
FIVE_COMPONENT_REPORT_CONTRACTS = {
    "cpps": ("cpps_report", "cpps_receipt", "PASS_WAVEFORM_OPTIMIZATION"),
    "hnr": ("hnr_report", "hnr_receipt", "PASS_HNR_FRESH_SPEAKER_PANEL"),
    "shimmer_percent": (
        "shimmer_percent_report",
        "shimmer_percent_receipt",
        "PASS_SHIMMER_FRESH_SPEAKER_PANEL",
    ),
    "slope": (
        "slope_report",
        "slope_receipt",
        "PASS_LTAS_SLOPE_FRESH_SPEAKER_PANEL",
    ),
    "tilt": ("tilt_report", "tilt_receipt", "FAIL_WAVEFORM_OPTIMIZATION"),
}
REQUIRED_ARTIFACT_KEYS = (
    *FIVE_COMPONENT_EVIDENCE_KEYS,
    "five_gradient_report",
    "five_gradient_receipt",
    "shimmer_db_promotion_report",
    "shimmer_db_promotion_receipt",
    "six_gradient_raw_report",
    "six_gradient_raw_receipt",
    "six_gradient_report",
    "six_gradient_receipt",
    "fresh_panel_split_seal",
    "fresh_speaker_source_manifest",
    "prior_panel_speaker_ledger",
    "joint_gate_contract",
    "target_value_protocol_contract",
    "clean_target_label_bank",
    "cpps_checkpoint",
    "hnr_checkpoint",
    "shimmer_percent_checkpoint",
    "slope_checkpoint",
    "tilt_checkpoint",
    "v19_runtime_evidence_manifest",
    "v19_worker",
    "v19_runtime_client",
    "generator_config",
    "generator_checkpoint",
    "fixed_recipes",
    "simulation_config",
    "simulation_source",
    "exact_avqi_code_tree_manifest",
    "exact_runtime_manifest",
)
SIX_GRADIENT_SOURCE_EVIDENCE_KEYS = (
    *READINESS_SOURCE_EVIDENCE_KEYS,
)
MISSING_CODE_STAGES = (
    "joint waveform preparation and immutable sealing runner",
    "post-seal exact-Praat six-component evaluator/decision runner",
)
UNFROZEN_SCIENTIFIC_CONTRACTS = (
    "Shimmer dB fresh-promotion evidence schema and decision",
    "fresh speaker source-manifest schema",
    "fresh panel split-seal schema",
    "clean pathological target-bank schema",
    "six-component joint waveform/exact gate contract",
)
DRAFT_PANEL_DATA_REQUIREMENTS = (
    "each speaker has clean/RIR/SNR x CS/SV rows",
    "calibration and final speakers are disjoint",
    "each split contains pathological and healthy speakers",
    "pathological rows alone use same-speaker clean pathological targets",
    "healthy rows are guardrail-only and never optimization targets",
    "source manifest binds every selected case and waveform hash",
    "target bank covers every pathological case and all six exact columns",
    "selected speakers do not overlap the prior-panel ledger",
)
SOURCE_REQUIREMENT_MATRIX = (
    {
        "requirement": "six-slot composed scorer",
        "current_evidence": "model.avqi_route_c.load_route_c_six_active_scorer",
        "status": "present_fail_closed_scaffold",
    },
    {
        "requirement": "current-output v19 topology binding for slot 3",
        "current_evidence": (
            "model.avqi_route_c_v19_contracts.validate_v19_exact_topology"
        ),
        "status": "present_fail_closed_scaffold",
    },
    {
        "requirement": "same-speaker normalized bidirectional six-slot loss",
        "current_evidence": "model.avqi_route_c.six_active_bidirectional_gap_losses",
        "status": "present",
    },
    {
        "requirement": "six-component gradient evaluator/runner",
        "current_evidence": (
            "scripts.evaluate_avqi_route_c_six_component_gradients + "
            "scripts.decide_avqi_route_c_six_component_gradients"
        ),
        "status": "present_dev_only_raw_measurement_plus_frozen_code_decision",
    },
    {
        "requirement": "two-stage sealed joint waveform evaluator/runner",
        "current_evidence": None,
        "status": "missing",
    },
    {
        "requirement": "five promoted-component scientific evidence bundle",
        "current_evidence": (
            "frozen source_evidence set in five-component gradient audit"
        ),
        "status": "external_immutable_bindings_required",
    },
    {
        "requirement": "fresh-panel source/split/target schemas",
        "current_evidence": "draft structural validators only",
        "status": "unfrozen_scientific_contract_blocker",
    },
    {
        "requirement": "joint waveform/exact gate thresholds",
        "current_evidence": (
            "six-component code-gradient thresholds frozen separately from "
            "future waveform/exact gates"
        ),
        "status": "downstream_scientific_contract_blocker",
    },
)


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_optimizer_zero(value: Mapping[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} contains generator optimizer steps")


def _require_all_true_gates(value: Mapping[str, Any], label: str) -> None:
    gates = value.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"{label} has no frozen gates")
    if any(gate is not True for gate in gates.values()):
        raise ValueError(f"{label} contains a failed gate")


def _finite_mapping(
    value: Any,
    expected_keys: tuple[str, ...],
    label: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError(f"{label} keys differ")
    parsed = {key: float(value[key]) for key in expected_keys}
    if any(
        not math.isfinite(number)
        or (positive and number <= 0.0)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
        for number in parsed.values()
    ):
        raise ValueError(f"{label} values are invalid")
    return parsed


def readiness_requirements() -> dict[str, Any]:
    """Describe current blockers without reading any future panel artifact."""
    registry = route_c_six_registry_records()
    shimmer = next(row for row in registry if row["name"] == "shimmer_db")
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "decision": "NO_GO_SIX_JOINT_PANEL_EXECUTION",
        "source_requirement_matrix": list(SOURCE_REQUIREMENT_MATRIX),
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "required_artifacts": list(REQUIRED_ARTIFACT_KEYS),
        "required_splits": list(REQUIRED_SPLITS),
        "required_views": list(REQUIRED_VIEWS),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "required_guardrails": list(REQUIRED_GUARDRAILS),
        "current_shimmer_db_scientific_status": shimmer["scientific_status"],
        "required_shimmer_db_scientific_status": SHIMMER_DB_REQUIRED_STATUS,
        "missing_code_stages": list(MISSING_CODE_STAGES),
        "unfrozen_scientific_contracts": list(UNFROZEN_SCIENTIFIC_CONTRACTS),
        "draft_panel_data_requirements": list(DRAFT_PANEL_DATA_REQUIREMENTS),
        "execution_authorized": False,
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }


def current_blockers() -> list[str]:
    requirements = readiness_requirements()
    blockers = list(requirements["missing_code_stages"])
    blockers.extend(requirements["unfrozen_scientific_contracts"])
    if requirements["current_shimmer_db_scientific_status"] == (
        ROUTE_C_SIX_SCIENTIFIC_STATUS
    ):
        blockers.insert(0, "Shimmer dB scientific promotion remains pending")
    blockers.extend(
        f"not yet bound into a six-joint manifest: {key}"
        for key in REQUIRED_ARTIFACT_KEYS
    )
    return blockers


def _receipt_binds_report(
    receipt: Mapping[str, Any],
    report_path: Path,
    report_sha256: str,
    label: str,
) -> None:
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or receipt_hashes.get(report_path.name) != report_sha256
    ):
        raise ValueError(f"{label} receipt does not bind its report")


def _validate_five_component_evidence(
    artifacts: Mapping[str, Mapping[str, str]],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    """Validate the evidence set frozen by the accepted five-active audit."""
    required = {
        *FIVE_COMPONENT_EVIDENCE_KEYS,
        "five_gradient_report",
        "five_gradient_receipt",
    }
    if set(artifacts) != required or set(paths) != required:
        raise ValueError("five-component frozen evidence keys differ")
    for key in required:
        path = paths[key]
        binding = artifacts[key]
        if (
            not path.is_absolute()
            or not path.is_file()
            or set(binding) != {"path", "sha256"}
            or binding["path"] != str(path)
            or not _is_sha256(binding["sha256"])
            or sha256_file(path) != binding["sha256"]
        ):
            raise ValueError(f"five-component frozen evidence hash differs: {key}")
    for component, (report_key, receipt_key, decision) in (
        FIVE_COMPONENT_REPORT_CONTRACTS.items()
    ):
        report = _read_json_mapping(paths[report_key], f"{component} report")
        receipt = _read_json_mapping(paths[receipt_key], f"{component} receipt")
        if report.get("decision") != decision or receipt.get("decision") != decision:
            raise ValueError(f"{component} frozen evidence decision differs")
        waveform_schema = {
            "cpps": "direct-avqi-waveform-optimization-v3",
            "tilt": "direct-avqi-waveform-optimization-v1",
        }.get(component)
        if waveform_schema is not None and report.get("schema_version") != (
            waveform_schema
        ):
            raise ValueError(f"{component} frozen evidence schema differs")
        _require_optimizer_zero(report, f"{component} report")
        _require_optimizer_zero(receipt, f"{component} receipt")
        if report.get("formal_pathology_training_submitted") is not False:
            raise ValueError(f"{component} report overclaims training")
        _receipt_binds_report(
            receipt,
            paths[report_key],
            artifacts[report_key]["sha256"],
            component,
        )
        if component in {"hnr", "shimmer_percent", "slope"}:
            final = report.get("final")
            if (
                report.get("final_exact_panel_opened") is not True
                or not isinstance(final, dict)
                or final.get("decision") != "PASS"
            ):
                raise ValueError(f"{component} fresh-panel evidence differs")
            _require_all_true_gates(final, f"{component} final panel")
        elif component in {"cpps", "tilt"}:
            summary = report.get("summary")
            component_gates = (
                summary.get("component_gates")
                if isinstance(summary, dict)
                else None
            )
            component_gate = (
                component_gates.get(component)
                if isinstance(component_gates, dict)
                else None
            )
            safety = summary.get("safety") if isinstance(summary, dict) else None
            if (
                not isinstance(component_gate, dict)
                or component_gate.get("decision") != "PASS"
                or not isinstance(safety, dict)
                or safety.get("decision") != "PASS"
            ):
                raise ValueError(f"{component} component-level evidence differs")
            _require_all_true_gates(component_gate, f"{component} component")

    slope_receipt = _read_json_mapping(paths["slope_receipt"], "slope receipt")
    slope_hashes = slope_receipt.get("artifact_sha256")
    if not isinstance(slope_hashes, dict):
        raise ValueError("slope receipt artifact bindings differ")
    for key in ("slope_final_panel_seal", "slope_final_results"):
        if slope_hashes.get(paths[key].name) != artifacts[key]["sha256"]:
            raise ValueError(f"slope receipt does not bind {key}")

    report = _read_json_mapping(paths["five_gradient_report"], "five-gradient report")
    receipt = _read_json_mapping(
        paths["five_gradient_receipt"], "five-gradient receipt"
    )
    if (
        report.get("schema_version")
        != "avqi_route_c_five_component_gradient_audit_v1"
        or report.get("decision") != "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT"
        or receipt.get("decision") != report.get("decision")
    ):
        raise ValueError("five-gradient frozen decision differs")
    _require_all_true_gates(report, "five-gradient audit")
    _require_optimizer_zero(report, "five-gradient report")
    _require_optimizer_zero(receipt, "five-gradient receipt")
    if tuple(receipt.get("active_components", ())) != ROUTE_C_FIVE_ACTIVE_COMPONENTS:
        raise ValueError("five-gradient active components differ")
    if receipt.get("inactive_slots") != ["shimmer_db"]:
        raise ValueError("five-gradient inactive slot differs")
    _receipt_binds_report(
        receipt,
        paths["five_gradient_report"],
        artifacts["five_gradient_report"]["sha256"],
        "five-gradient",
    )
    source_evidence = report.get("source_evidence")
    if not isinstance(source_evidence, dict) or set(source_evidence) != set(
        FIVE_COMPONENT_EVIDENCE_KEYS
    ):
        raise ValueError("five-gradient source-evidence keys differ")
    for key in FIVE_COMPONENT_EVIDENCE_KEYS:
        if source_evidence[key] != artifacts[key]:
            raise ValueError(f"five-gradient source evidence differs: {key}")
    return {key: artifacts[key]["sha256"] for key in FIVE_COMPONENT_EVIDENCE_KEYS}


def _validate_six_gradient(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    report_sha256: str,
    expected_source_evidence: Mapping[str, str],
) -> dict[str, float]:
    if report.get("schema_version") != SIX_GRADIENT_SCHEMA_VERSION:
        raise ValueError("six-component gradient schema differs")
    if report.get("decision") != SIX_GRADIENT_PASS_DECISION:
        raise ValueError("six-component gradient audit did not pass")
    if receipt.get("schema_version") != SIX_GRADIENT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("six-component gradient receipt schema differs")
    if receipt.get("decision") != report["decision"]:
        raise ValueError("six-component gradient receipt decision differs")
    if (
        report.get("joint_panel_decision") != SIX_GRADIENT_JOINT_PANEL_NO_GO
        or receipt.get("joint_panel_decision")
        != SIX_GRADIENT_JOINT_PANEL_NO_GO
    ):
        raise ValueError("six-component gradient joint-panel decision differs")
    if (
        tuple(report.get("active_components", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or tuple(receipt.get("active_components", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or tuple(report.get("active_components", ()))
        != FROZEN_SIX_GRADIENT_COMPONENTS
    ):
        raise ValueError("six-component gradient active order differs")
    if report.get("source_evidence_sha256") != expected_source_evidence:
        raise ValueError("six-component gradient source-evidence binding differs")
    if report.get("frozen_contract") != six_gradient_decision_requirements().get(
        "frozen_contract"
    ):
        raise ValueError("six-component gradient frozen contract differs")
    precedent = report.get("accepted_numeric_precedent")
    expected_precedent = {
        "slurm_job_id": FROZEN_FIVE_JOB_ID,
        "report_sha256": FROZEN_FIVE_REPORT_SHA256,
        "receipt_sha256": FROZEN_FIVE_RECEIPT_SHA256,
    }
    if precedent != expected_precedent or receipt.get(
        "accepted_numeric_precedent"
    ) != expected_precedent:
        raise ValueError("six-component gradient numeric precedent differs")
    raw = report.get("raw_measurement_evidence")
    raw_receipt_hashes = receipt.get("raw_measurement_sha256")
    if not isinstance(raw, dict) or not isinstance(raw_receipt_hashes, dict):
        raise ValueError("six-component raw measurement binding is unavailable")
    if (
        raw.get("raw_decision") != SIX_GRADIENT_RAW_PENDING_DECISION
        or raw.get("raw_artifacts_rewritten") is not False
        or receipt.get("raw_artifacts_rewritten") is not False
        or raw_receipt_hashes
        != {"report": raw.get("report_sha256"), "receipt": raw.get("receipt_sha256")}
        or not _is_sha256(raw.get("report_sha256"))
        or not _is_sha256(raw.get("receipt_sha256"))
    ):
        raise ValueError("six-component raw measurement binding differs")
    decision_source = report.get("decision_source")
    if (
        not isinstance(decision_source, dict)
        or receipt.get("source_commit") != decision_source.get("head")
        or receipt.get("source_branch") != decision_source.get("branch")
    ):
        raise ValueError("six-component gradient decision source differs")
    implementation = report.get("implementation_sha256")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(SIX_GRADIENT_DECISION_IMPLEMENTATION_KEYS)
        or implementation != receipt.get("implementation_sha256")
        or any(not _is_sha256(value) for value in implementation.values())
    ):
        raise ValueError("six-component gradient decision implementation differs")
    immutability = report.get("post_evaluation_immutability")
    expected_immutability_hashes = {
        "raw_report": raw["report_sha256"],
        "raw_receipt": raw["receipt_sha256"],
        "five_precedent_report": FROZEN_FIVE_REPORT_SHA256,
        "five_precedent_receipt": FROZEN_FIVE_RECEIPT_SHA256,
    }
    if (
        not isinstance(immutability, dict)
        or immutability.get("verified") is not True
        or immutability.get("artifact_sha256") != expected_immutability_hashes
        or receipt.get("post_evaluation_immutability") != immutability
    ):
        raise ValueError("six-component gradient input immutability differs")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != set(SIX_GRADIENT_FROZEN_GATE_KEYS)
        or any(value is not True for value in gates.values())
    ):
        raise ValueError("six-component gradient frozen gates failed")
    summary = report.get("measurement_summary")
    if not isinstance(summary, dict):
        raise ValueError("six-component gradient measurement summary is unavailable")
    weights = _finite_mapping(
        summary.get("calibration_inverse_gradient_weights"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-component frozen weights",
        positive=True,
    )
    ratio = summary.get("calibration_weighted_median_norm_ratio")
    maximum_share = summary.get("maximum_weighted_component_norm_share")
    minimum_joint_cosine = summary.get("minimum_component_to_joint_cosine")
    if (
        not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio > MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO
        or not isinstance(maximum_share, (int, float))
        or not math.isfinite(maximum_share)
        or maximum_share > MAXIMUM_WEIGHTED_COMPONENT_SHARE
        or not isinstance(minimum_joint_cosine, (int, float))
        or not math.isfinite(minimum_joint_cosine)
        or minimum_joint_cosine < 0.0
        or summary.get("calibration_cases") != 4
        or summary.get("holdout_cases") != 4
        or summary.get("pairwise_negative_values_are_diagnostic_only") is not True
    ):
        raise ValueError("six-component gradient frozen summary differs")
    required_false = (
        "scientific_promotion_granted",
        "joint_scientific_promotion_granted",
        "joint_panel_authorized",
        "combined_final_panel_opened",
        "fresh_panel_opened",
        "exact_candidate_scoring_requested",
        "waveform_generation_performed",
        "formal_generator_training_submitted",
    )
    if any(report.get(key) is not False for key in required_false):
        raise ValueError("six-component gradient report overclaims science")
    if any(receipt.get(key) is not False for key in required_false):
        raise ValueError("six-component gradient receipt overclaims science")
    if (
        report.get("scientific_contract_frozen_before_six_holdout_open") is not True
        or report.get("raw_measurement_recomputed") is not False
        or report.get("authoritative_training_decision") != TRAINING_NO_GO
        or receipt.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("six-component gradient decision boundaries differ")
    _require_optimizer_zero(report, "six-component gradient report")
    _require_optimizer_zero(receipt, "six-component gradient receipt")
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or len(receipt_hashes) != 1
        or report_sha256 not in receipt_hashes.values()
    ):
        raise ValueError("six-component gradient receipt does not bind its report")
    return weights


PATHOLOGICAL_ROLE = "same_speaker_clean_pathological_target"
HEALTHY_ROLE = "guardrail_only_no_optimization_target"
PANEL_ROW_FIELDS = {
    "case_id",
    "speaker_id",
    "split",
    "view",
    "condition",
    "label",
    "optimization_role",
}


def _validate_panel_rows(
    rows: Any,
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} has no rows")
    if any(
        not isinstance(row, dict) or not PANEL_ROW_FIELDS <= set(row)
        for row in rows
    ):
        raise ValueError(f"{label} row fields differ")
    rows_by_case = {str(row["case_id"]): row for row in rows}
    if len(rows_by_case) != len(rows) or "" in rows_by_case:
        raise ValueError(f"{label} case IDs are not unique")
    role_by_label = {"patient": PATHOLOGICAL_ROLE, "healthy": HEALTHY_ROLE}
    for row in rows:
        if (
            row["split"] not in REQUIRED_SPLITS
            or row["view"] not in REQUIRED_VIEWS
            or row["condition"] not in REQUIRED_CONDITIONS
            or role_by_label.get(row["label"]) != row["optimization_role"]
        ):
            raise ValueError(f"{label} row semantics differ")

    expected_matrix = {
        (condition, view)
        for condition in REQUIRED_CONDITIONS
        for view in REQUIRED_VIEWS
    }
    speakers = {str(row["speaker_id"]) for row in rows}
    if "" in speakers:
        raise ValueError(f"{label} has an empty speaker ID")
    for speaker in speakers:
        speaker_rows = [row for row in rows if row["speaker_id"] == speaker]
        if (
            len({row["split"] for row in speaker_rows}) != 1
            or len({row["label"] for row in speaker_rows}) != 1
            or len(speaker_rows) != len(expected_matrix)
            or {(row["condition"], row["view"]) for row in speaker_rows}
            != expected_matrix
        ):
            raise ValueError(f"{label} speaker matrix differs: {speaker}")
    speakers_by_split = {
        split: {str(row["speaker_id"]) for row in rows if row["split"] == split}
        for split in REQUIRED_SPLITS
    }
    if speakers_by_split["calibration"] & speakers_by_split["final"]:
        raise ValueError(f"{label} calibration/final speakers overlap")
    for split in REQUIRED_SPLITS:
        split_labels = {row["label"] for row in rows if row["split"] == split}
        if split_labels != {"patient", "healthy"}:
            raise ValueError(f"{label} {split} strata differ")
    return rows_by_case, {
        split: sorted(values) for split, values in speakers_by_split.items()
    }


def _validate_split_seal(
    seal: Mapping[str, Any],
    *,
    gate_sha256: str,
    target_sha256: str,
    ledger_sha256: str,
    source_sha256: str,
) -> dict[str, list[str]]:
    if seal.get("schema_version") != DRAFT_SPLIT_SEAL_SCHEMA_VERSION:
        raise ValueError("six-joint draft split-seal schema differs")
    required_values = {
        "exact_scores_opened": False,
        "speaker_split_before_simulation": True,
        "selection_or_tuning_on_this_panel": False,
    }
    if any(seal.get(key) is not value for key, value in required_values.items()):
        raise ValueError("six-joint split seal semantics differ")
    bindings = {
        "joint_gate_contract_sha256": gate_sha256,
        "target_value_protocol_sha256": target_sha256,
        "prior_panel_speaker_ledger_sha256": ledger_sha256,
        "fresh_speaker_source_manifest_sha256": source_sha256,
    }
    if any(seal.get(key) != value for key, value in bindings.items()):
        raise ValueError("six-joint split seal hash binding differs")
    _, speakers_by_split = _validate_panel_rows(
        seal.get("rows"), "six-joint split seal"
    )
    _require_optimizer_zero(seal, "six-joint split seal")
    return speakers_by_split


def validate_readiness_manifest(
    manifest: Mapping[str, Any],
    *,
    registry_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate evidence, then remain NO-GO until missing runners exist."""
    if manifest.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError("six-joint readiness schema differs")
    if manifest.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("six-joint readiness opened candidate outcomes")
    if manifest.get("fresh_panel_opened") is not False:
        raise ValueError("six-joint readiness opened a fresh panel")
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("six-joint source commit binding differs")
    _require_optimizer_zero(manifest, "six-joint readiness manifest")

    registry = (
        route_c_six_registry_records()
        if registry_records is None
        else registry_records
    )
    if tuple(row.get("name") for row in registry) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("six-joint live registry component order differs")
    shimmer = registry[3]
    if shimmer.get("scientific_status") != SHIMMER_DB_REQUIRED_STATUS:
        raise ValueError(
            "Shimmer dB scientific promotion is still pending; joint panel closed"
        )
    if UNFROZEN_SCIENTIFIC_CONTRACTS:
        raise ValueError(
            "six-joint scientific schemas remain unfrozen: "
            + "; ".join(UNFROZEN_SCIENTIFIC_CONTRACTS)
        )


def repository_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(root: Path, expected_commit: str) -> dict[str, str]:
    resolved = root.resolve()
    head = repository_value(resolved, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("six-joint source HEAD differs")
    if repository_value(resolved, "status", "--porcelain"):
        raise ValueError("six-joint preflight requires a clean worktree")
    return {
        "root": str(resolved),
        "head": head,
        "branch": repository_value(resolved, "branch", "--show-current"),
    }


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("six-joint readiness manifest hash differs")
    return _read_json_mapping(path, "six-joint readiness manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execution_inputs = (
        args.manifest,
        args.manifest_sha256,
        args.source_root,
        args.source_commit,
    )
    if args.requirements_only:
        if any(value is not None for value in execution_inputs):
            raise ValueError("requirements-only mode accepts no execution inputs")
        report = readiness_requirements()
        report["blockers"] = current_blockers()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    if any(value is None for value in execution_inputs):
        raise ValueError("six-joint preflight inputs are incomplete")
    source = validate_source(args.source_root, args.source_commit)
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    if manifest.get("source_commit") != source["head"]:
        raise ValueError("six-joint manifest/source commit binding differs")
    validate_readiness_manifest(manifest)
    raise ValueError("six-joint executable runner remains unavailable")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(
            json.dumps(
                {
                    "decision": "NO_GO_SIX_JOINT_PANEL_EXECUTION",
                    "execution_authorized": False,
                    "reason": str(error),
                    "generator_optimizer_steps": 0,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2) from None
