#!/usr/bin/env python3
"""Interpret a passing Candidate-E receipt as scientific component readiness.

This successor never rewrites the Candidate-D/v23 closure or the legacy
registry. It emits an evidence-bound six-component scientific-readiness
snapshot while joint execution and formal generator training remain closed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from model.avqi_route_c import (
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    route_c_six_registry_records,
)
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    SHIMMER_DB_COMPONENT_NO_GO_DECISION,
    SHIMMER_DB_COMPONENT_NO_GO_STATUS,
    load_shimmer_db_component_no_go_closure,
    sha256_file,
)


CONTRACT_SCHEMA = "avqi-route-c-six-joint-candidate-e-readiness-contract-v4"
REPORT_SCHEMA = "avqi-route-c-six-joint-candidate-e-readiness-v4"
RECEIPT_SCHEMA = "avqi-route-c-six-joint-candidate-e-readiness-receipt-v4"
PROMOTION_REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-exact-promotion-v32r8"
)
PROMOTION_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-receipt-v32r8"
)
PROMOTION_PASS = "PASS_CANDIDATE_E_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V32R8"
COMPONENT_PASS = "PASS_SHIMMER_DB_CANDIDATE_E_COMPONENT_READINESS_V4"
COMPONENT_STATUS = "fresh_speaker_panel_pass"
SHIMMER_READINESS = "READY_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
JOINT_NO_GO = "NO_GO_SIX_JOINT_PANEL_OTHER_INPUTS_UNBOUND_V4"
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
PREDECESSOR_SOURCE_COMMIT = "5d33561fd1324e6e9370a7ff815d34973a262d01"
PROMOTION_SOURCE_COMMIT = "109f398d607bce936a9576c826ff74ce0ea9f636"
PROMOTION_JOB_ID = "20044559"
PROMOTION_RUNNER_SHA256 = (
    "45dde78d00845d378c81ca863104cd19e39001c527c3ddee1224e9a644ff7429"
)
RUNTIME_CONFIG_SHA256 = (
    "4dec4b018b6cd9f7a5a7f87966cc7f2dde057f152df256f65fc397faefb53b98"
)
V32R7_REPORT_SHA256 = (
    "1f727fb4b766af4a8387b33828a3ad4b21bf7af80f4e167aa44996557f91e3d5"
)
V32R7_RECEIPT_SHA256 = (
    "a10895951faf85d7abaeeb9657737d1fe66b9e6c30d20072abbad75f28c1a4c6"
)
PROMOTION_REPORT_SHA256 = (
    "4a5b49dbdf3a1324d6b0f1e084d5153469f7c178afae0ded9cefd9060206e632"
)
PROMOTION_RECEIPT_SHA256 = (
    "d256073901fd708cfbad10687c8e281007f98a64b46e902bd8eeb528e06cc84f"
)
PANEL_SEAL_SHA256 = (
    "0f1147b107ae848b6914916ef0fec87ed2e115e6868c4014d132aae9b86487a2"
)
PANEL_RECEIPT_SHA256 = (
    "b4f570c8ed8fd2084469d2a998bdb5aa604b70cd52e5e72b16f95c420c78c711"
)
UPDATED_LEDGER_SHA256 = (
    "f5831ecd455f21bed626ebd4302881626553db2c378d45330faa75137179e4ee"
)
TARGET_SEAL_SHA256 = (
    "b3e33ef80e2f0454f55c4b045a644a6394dd29806889b6b186fc96757f06db2d"
)
TARGET_RECEIPT_SHA256 = (
    "71a5e616a407675f2336d758acb3bffa457945862768f67942721ddd053d60a2"
)
V23_CLOSURE_SHA256 = (
    "56a314b65b7a1272f34ad300253ed86e5618c33d5bd9c958b4515d4e12719492"
)
READINESS_V2_SHA256 = (
    "78b2769af014ed0bc2cfa6e4c70a235f4eab5e4266bcef1729f08cb74b64bc66"
)
PREEXACT_GATES = (
    "complete_selector_coverage",
    "candidate_pool_frozen_serial_equivalence",
    "total_metric_step_runtime_le_500ms",
    "selector_uses_no_candidate_exact_outcome",
    "selector_uses_no_identity",
    "candidate_e_remains_frozen",
    "generator_optimizer_steps_zero",
)
RECEIPT_ARTIFACTS = (
    "external_svd_report_v32r8.json",
    "external_svd_exact_results_v32r8.csv",
    "selector_seal_pre_exact_v32r8.json",
    "candidate_e_attempts_pre_exact_v32r8.csv",
    "candidate_pool_equivalence_v32r8.json",
)
PRESERVED_PREEXACT_VERSIONS = (
    "v32r2",
    "v32r3",
    "v32r4",
    "v32r5",
    "v32r6",
    "v32r7",
)
UNBOUND_JOINT_INPUTS = (
    "candidate_e_joint_runtime_binding",
    "six_gradient_report",
    "six_gradient_receipt",
    "fresh_panel_split_seal",
    "fresh_speaker_source_manifest",
    "clean_target_label_bank",
    "joint_gradient_manifest",
)


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not is_sha256(expected):
        raise ValueError(f"{label} expected hash is not lowercase SHA-256")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash differs: {observed} != {expected}")
    return observed


def validate_receipt_artifact_files(
    output_dir: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, str]:
    if set(artifacts) != set(RECEIPT_ARTIFACTS):
        raise ValueError("Candidate-E promotion receipt artifact names differ")
    observed = {}
    for name in RECEIPT_ARTIFACTS:
        expected = artifacts[name]
        observed[name] = validate_hash(
            output_dir / name,
            expected,
            f"Candidate-E promotion artifact {name}",
        )
    return observed


def require_all_true(value: Any, label: str) -> Mapping[str, bool]:
    if (
        not isinstance(value, dict)
        or not value
        or any(item is not True for item in value.values())
    ):
        raise ValueError(f"{label} did not pass completely")
    return value


def require_finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} is not finite")
    return float(value)


def require_training_closed(value: Mapping[str, Any], label: str) -> None:
    expected = {
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    if any(
        value.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise ValueError(f"{label} training boundary differs")


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("Candidate-E readiness contract schema differs")
    promotion_binding = contract.get("external_promotion_binding")
    if promotion_binding != {
        "slurm_job_id": PROMOTION_JOB_ID,
        "observed_state": "COMPLETED",
        "exit_code": "0:0",
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "scientific_result": PROMOTION_PASS,
        "promotion_report_sha256": PROMOTION_REPORT_SHA256,
        "promotion_receipt_sha256": PROMOTION_RECEIPT_SHA256,
    }:
        raise ValueError("Candidate-E readiness promotion binding differs")
    history = contract.get("historical_boundary")
    if (
        not isinstance(history, dict)
        or history.get("predecessor_source_commit") != PREDECESSOR_SOURCE_COMMIT
        or history.get("readiness_v2_sha256") != READINESS_V2_SHA256
        or history.get("candidate_d_v23_closure_sha256") != V23_CLOSURE_SHA256
        or history.get("candidate_d_v23_no_go_remains_immutable") is not True
        or history.get("candidate_d_v23_no_go_reinterpreted_as_pass") is not False
    ):
        raise ValueError("Candidate-D historical boundary differs")
    promotion = contract.get("candidate_e_external_promotion")
    expected_promotion = {
        "report_schema": PROMOTION_REPORT_SCHEMA,
        "receipt_schema": PROMOTION_RECEIPT_SCHEMA,
        "pass_decision": PROMOTION_PASS,
        "readiness_status": SHIMMER_READINESS,
        "source_commit": PROMOTION_SOURCE_COMMIT,
        "runner_sha256": PROMOTION_RUNNER_SHA256,
        "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
        "v32r7_no_go_report_sha256": V32R7_REPORT_SHA256,
        "v32r7_no_go_receipt_sha256": V32R7_RECEIPT_SHA256,
        "v30r2_panel_seal_sha256": PANEL_SEAL_SHA256,
        "v30r2_panel_receipt_sha256": PANEL_RECEIPT_SHA256,
        "v30r2_updated_ledger_sha256": UPDATED_LEDGER_SHA256,
        "v31r2_target_seal_sha256": TARGET_SEAL_SHA256,
        "v31r2_target_receipt_sha256": TARGET_RECEIPT_SHA256,
        "required_preexact_gates": list(PREEXACT_GATES),
        "required_summary_gate_families": [
            "mechanism_gates",
            "integration_gates",
        ],
        "required_receipt_artifacts": list(RECEIPT_ARTIFACTS),
        "exact_praat_is_final_judge": True,
        "candidate_exact_must_follow_selector_seal": True,
        "candidate_exact_outcomes_used_for_selection": False,
        "speaker_or_case_identity_used_for_routing": False,
        "preserved_preexact_no_go_versions": list(
            PRESERVED_PREEXACT_VERSIONS
        ),
        "thresholds_changed": False,
    }
    if promotion != expected_promotion:
        raise ValueError("Candidate-E promotion contract differs")
    interpretation = contract.get("readiness_interpretation")
    if interpretation != {
        "component_decision": COMPONENT_PASS,
        "component_scientific_status": COMPONENT_STATUS,
        "shimmer_db_six_component_readiness_eligible": True,
        "six_component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "all_six_scientific_components_ready": True,
        "readiness_scope": (
            "component_scientific_status_only_not_joint_execution_authorization"
        ),
        "joint_panel_decision": JOINT_NO_GO,
        "joint_panel_authorized": False,
        "execution_authorized": False,
        "joint_scientific_promotion_granted": False,
        "unbound_joint_inputs": list(UNBOUND_JOINT_INPUTS),
    }:
        raise ValueError("Candidate-E readiness interpretation differs")
    require_training_closed(
        contract.get("immutable_training_boundary", {}),
        "Candidate-E readiness contract",
    )


def validate_source(root: Path, expected_commit: str) -> dict[str, str]:
    root = root.resolve()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("Candidate-E readiness source commit differs")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Candidate-E readiness source worktree is dirty")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            PREDECESSOR_SOURCE_COMMIT,
            head,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Candidate-E readiness predecessor is not an ancestor")
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"root": str(root), "head": head, "branch": branch}


def validate_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    if (
        ledger.get("schema_version")
        != "avqi-route-c-prior-panel-speaker-ledger-v1"
        or ledger.get("exact_outcomes_used_for_selection") is not False
        or ledger.get("target_component_scorability_boolean_used_for_selection")
        is not True
        or ledger.get("target_scalar_values_used_for_selection") is not False
    ):
        raise ValueError("Candidate-E prior speaker ledger boundary differs")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or len(entries) != 35:
        raise ValueError("Candidate-E prior speaker ledger coverage differs")
    canonical = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Candidate-E prior speaker ledger entry differs")
        dataset = str(entry.get("dataset", "")).strip().upper()
        speaker_id = str(entry.get("speaker_id", "")).strip()
        identity = str(entry.get("canonical_speaker_id", ""))
        if not dataset or not speaker_id or identity != f"{dataset}:{speaker_id}":
            raise ValueError("Candidate-E prior speaker identity differs")
        canonical.append(identity)
    if len(set(canonical)) != len(canonical):
        raise ValueError("Candidate-E prior speaker identities are duplicated")
    return {"entry_count": len(entries), "unique_speakers": len(set(canonical))}


def validate_promotion(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    report_sha256: str,
    ledger_sha256: str,
) -> dict[str, Any]:
    expected_identity = {
        "schema_version": PROMOTION_REPORT_SCHEMA,
        "decision": PROMOTION_PASS,
        "component": "shimmer_db",
        "component_status": PROMOTION_PASS,
        "readiness_status": SHIMMER_READINESS,
        "source_commit": PROMOTION_SOURCE_COMMIT,
        "slurm_job_id": PROMOTION_JOB_ID,
    }
    if any(report.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Candidate-E external promotion identity differs")
    required_report_true = (
        "candidate_exact_outcomes_opened_after_selector_seal",
        "exact_scoring_complete",
        "result_blind_external_three_stage_chain_complete",
        "old_v23_no_go_preserved",
        "candidate_e_frozen",
        "external_speaker_gate_pass",
        "bounded_waveform_promotion_pass",
        "scientific_promotion_granted",
        "six_component_readiness_eligible",
        *(
            f"{version}_preexact_no_go_preserved"
            for version in PRESERVED_PREEXACT_VERSIONS
        ),
    )
    if any(report.get(key) is not True for key in required_report_true):
        raise ValueError("Candidate-E external promotion boundary differs")
    if report.get("retuning_authorized") is not False:
        raise ValueError("Candidate-E external promotion permits retuning")
    if report.get("joint_panel_authorized") is not False:
        raise ValueError("Candidate-E external promotion over-authorizes joint panel")
    require_training_closed(report, "Candidate-E promotion report")
    preexact = report.get("preexact_gates")
    if not isinstance(preexact, dict) or set(preexact) != set(PREEXACT_GATES):
        raise ValueError("Candidate-E pre-exact gate keys differ")
    require_all_true(preexact, "Candidate-E pre-exact gates")
    equivalence = report.get("candidate_pool_equivalence")
    proxy_error = equivalence.get(
        "maximum_current_topology_proxy_absolute_error"
    ) if isinstance(equivalence, dict) else None
    if (
        not isinstance(equivalence, dict)
        or equivalence.get("all_equal") is not True
        or equivalence.get("candidate_grid_waveform_byte_equal") is not True
        or equivalence.get("candidate_grid_topology_hash_equal") is not True
        or equivalence.get("selector_choice_equal") is not True
        or equivalence.get("candidate_grid_row_count") != 120
        or isinstance(proxy_error, bool)
        or not isinstance(proxy_error, (int, float))
        or not math.isfinite(float(proxy_error))
        or not 0.0 <= float(proxy_error) <= 1e-12
        or equivalence.get("candidate_exact_outcomes_used") is not False
    ):
        raise ValueError("Candidate-E frozen serial equivalence differs")
    summary = report.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("all_gates_pass") is not True
        or summary.get("external_effect_slices", {}).get("decision") != "PASS"
    ):
        raise ValueError("Candidate-E external summary did not pass")
    improvement_fraction = require_finite_number(
        summary.get("exact_db_improvement_fraction"),
        "Candidate-E exact improvement fraction",
    )
    median_reduction = require_finite_number(
        summary.get("median_exact_db_normalized_gap_reduction"),
        "Candidate-E median exact reduction",
    )
    runtime_summary = summary.get("total_metric_step_runtime_ms")
    if not isinstance(runtime_summary, dict):
        raise ValueError("Candidate-E runtime summary is unavailable")
    maximum_runtime = require_finite_number(
        runtime_summary.get("maximum"),
        "Candidate-E maximum runtime",
    )
    if (
        summary.get("rows") != 12
        or summary.get("material_rows") != 12
        or not 0.8 <= improvement_fraction <= 1.0
        or median_reduction < 0.02
        or not 0.0 <= maximum_runtime <= 500.0
    ):
        raise ValueError("Candidate-E external effect threshold differs")
    require_all_true(summary.get("mechanism_gates"), "Candidate-E mechanism gates")
    require_all_true(
        summary.get("integration_gates"),
        "Candidate-E integration gates",
    )
    source = report.get("source_sha256")
    runtime = source.get("runtime_successor") if isinstance(source, dict) else None
    expected_runtime = {
        "runtime_config": RUNTIME_CONFIG_SHA256,
        "v32r7_report": V32R7_REPORT_SHA256,
        "v32r7_receipt": V32R7_RECEIPT_SHA256,
    }
    expected_source = {
        "panel_seal": PANEL_SEAL_SHA256,
        "panel_receipt": PANEL_RECEIPT_SHA256,
        "updated_speaker_ledger": UPDATED_LEDGER_SHA256,
        "target_contract": TARGET_SEAL_SHA256,
        "target_receipt": TARGET_RECEIPT_SHA256,
    }
    if runtime != expected_runtime or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise ValueError("Candidate-E external source hashes differ")
    provenance = report.get("source_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("runner_sha256") != PROMOTION_RUNNER_SHA256
    ):
        raise ValueError("Candidate-E promotion runner hash differs")
    evidence = report.get("evidence_bindings")
    if (
        not isinstance(evidence, dict)
        or evidence.get("v32r7_preexact_no_go_preserved") is not True
        or evidence.get("v32r7_preexact_no_go_report_sha256")
        != V32R7_REPORT_SHA256
        or evidence.get("v32r7_preexact_no_go_receipt_sha256")
        != V32R7_RECEIPT_SHA256
        or evidence.get("updated_speaker_ledger_sha256") != ledger_sha256
    ):
        raise ValueError("Candidate-E predecessor evidence binding differs")
    thresholds = report.get("fixed_scientific_thresholds")
    if not isinstance(thresholds, dict) or {
        "alpha_ladder": thresholds.get("alpha_ladder"),
        "improvement_fraction": thresholds.get("improvement_fraction"),
        "material_normalized_gap": thresholds.get("material_normalized_gap"),
        "median_normalized_reduction": thresholds.get(
            "median_normalized_reduction"
        ),
        "maximum_clip_fraction": thresholds.get("maximum_clip_fraction"),
        "minimum_cosine": thresholds.get("minimum_cosine"),
        "residual_ceiling_db": thresholds.get("residual_ceiling_db"),
        "runtime_gate_ms": thresholds.get("runtime_gate_ms"),
        "target_reproduction_abs_tolerance": thresholds.get(
            "target_reproduction_abs_tolerance"
        ),
    } != {
        "alpha_ladder": [0.001, 0.0005, 0.00025, 0.000125],
        "improvement_fraction": 0.8,
        "material_normalized_gap": 0.02,
        "median_normalized_reduction": 0.02,
        "maximum_clip_fraction": 0.0,
        "minimum_cosine": 0.99999,
        "residual_ceiling_db": -50.0,
        "runtime_gate_ms": 500.0,
        "target_reproduction_abs_tolerance": 1e-9,
    }:
        raise ValueError("Candidate-E frozen scientific thresholds differ")

    expected_receipt_identity = {
        "schema_version": PROMOTION_RECEIPT_SCHEMA,
        "decision": PROMOTION_PASS,
        "component": "shimmer_db",
        "source_commit": PROMOTION_SOURCE_COMMIT,
        "slurm_job_id": PROMOTION_JOB_ID,
    }
    if any(
        receipt.get(key) != value
        for key, value in expected_receipt_identity.items()
    ):
        raise ValueError("Candidate-E promotion receipt identity differs")
    required_receipt_true = (
        "candidate_exact_outcomes_opened_after_selector_seal",
        "exact_scoring_complete",
        "result_blind_external_three_stage_chain_complete",
        "old_v23_no_go_preserved",
        "candidate_e_frozen",
        "scientific_promotion_granted",
        "six_component_readiness_eligible",
        *(
            f"{version}_preexact_no_go_preserved"
            for version in PRESERVED_PREEXACT_VERSIONS
        ),
    )
    if any(receipt.get(key) is not True for key in required_receipt_true):
        raise ValueError("Candidate-E promotion receipt boundary differs")
    if (
        receipt.get("retuning_authorized") is not False
        or receipt.get("joint_panel_authorized") is not False
    ):
        raise ValueError("Candidate-E promotion receipt over-authorizes execution")
    require_training_closed(receipt, "Candidate-E promotion receipt")
    artifacts = receipt.get("artifact_sha256")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(RECEIPT_ARTIFACTS)
        or artifacts.get("external_svd_report_v32r8.json") != report_sha256
        or any(not is_sha256(value) for value in artifacts.values())
    ):
        raise ValueError("Candidate-E promotion receipt artifacts differ")
    return {
        "preexact_gate_count": len(preexact),
        "mechanism_gate_count": len(summary["mechanism_gates"]),
        "integration_gate_count": len(summary["integration_gates"]),
        "receipt_artifact_count": len(artifacts),
    }


def six_component_readiness_snapshot() -> list[dict[str, Any]]:
    registry = route_c_six_registry_records()
    if tuple(row.get("name") for row in registry) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("six-component readiness registry order differs")
    snapshot = []
    for expected_index, row in enumerate(registry):
        if (
            row.get("index") != expected_index
            or row.get("active_in_six_component_scorer") is not True
        ):
            raise ValueError("six-component readiness registry slot differs")
        name = str(row["name"])
        registry_status = str(row["scientific_status"])
        if name == "shimmer_db":
            if registry_status != "pending_v19_component_promotion":
                raise ValueError("legacy Shimmer dB registry status differs")
            evidence = PROMOTION_PASS
        else:
            if registry_status != COMPONENT_STATUS:
                raise ValueError(f"{name} scientific readiness differs")
            evidence = "existing_fresh_speaker_panel_pass"
        snapshot.append(
            {
                "index": expected_index,
                "name": name,
                "scientific_status": COMPONENT_STATUS,
                "scientific_readiness_eligible": True,
                "evidence": evidence,
                "registry_scientific_status_preserved": registry_status,
                "registry_code_status": row["code_status"],
            }
        )
    return snapshot


def build_readiness_report(
    source: Mapping[str, str],
    promotion_report_sha256: str,
    promotion_receipt_sha256: str,
    ledger_sha256: str,
    contract_sha256: str,
    validation: Mapping[str, Any],
    ledger_summary: Mapping[str, Any],
) -> dict[str, Any]:
    closure = load_shimmer_db_component_no_go_closure()
    if (
        closure.get("decision") != SHIMMER_DB_COMPONENT_NO_GO_DECISION
        or closure.get("scientific_status") != SHIMMER_DB_COMPONENT_NO_GO_STATUS
    ):
        raise ValueError("Candidate-D/v23 closure was not preserved")
    component_readiness = six_component_readiness_snapshot()
    return {
        "schema_version": REPORT_SCHEMA,
        "decision": COMPONENT_PASS,
        "component": "shimmer_db",
        "component_scientific_status": COMPONENT_STATUS,
        "shimmer_db_six_component_readiness_eligible": True,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "component_readiness": component_readiness,
        "all_six_scientific_components_ready": all(
            row["scientific_readiness_eligible"]
            for row in component_readiness
        ),
        "readiness_scope": (
            "component_scientific_status_only_not_joint_execution_authorization"
        ),
        "candidate_e_external_promotion": PROMOTION_PASS,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "input_sha256": {
            "candidate_e_readiness_contract": contract_sha256,
            "candidate_e_external_report": promotion_report_sha256,
            "candidate_e_external_receipt": promotion_receipt_sha256,
            "candidate_e_prior_panel_speaker_ledger": ledger_sha256,
            "candidate_d_v23_closure": V23_CLOSURE_SHA256,
        },
        "candidate_e_validation": dict(validation),
        "candidate_e_prior_ledger": dict(ledger_summary),
        "historical_candidate_d_v23": {
            "decision": SHIMMER_DB_COMPONENT_NO_GO_DECISION,
            "scientific_status": SHIMMER_DB_COMPONENT_NO_GO_STATUS,
            "closure_sha256": V23_CLOSURE_SHA256,
            "remains_immutable": True,
            "reinterpreted_as_pass": False,
        },
        "joint_panel_decision": JOINT_NO_GO,
        "unbound_joint_inputs": list(UNBOUND_JOINT_INPUTS),
        "execution_authorized": False,
        "joint_panel_authorized": False,
        "joint_scientific_promotion_granted": False,
        "six_joint_candidate_exact_outcomes_opened": False,
        "six_joint_fresh_panel_opened": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "contract",
        "promotion-report",
        "promotion-receipt",
        "speaker-ledger",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source = validate_source(args.source_root, args.source_commit)
    contract_sha256 = validate_hash(
        args.contract,
        args.contract_sha256,
        "Candidate-E readiness contract",
    )
    report_sha256 = validate_hash(
        args.promotion_report,
        args.promotion_report_sha256,
        "Candidate-E external report",
    )
    receipt_sha256 = validate_hash(
        args.promotion_receipt,
        args.promotion_receipt_sha256,
        "Candidate-E external receipt",
    )
    if report_sha256 != PROMOTION_REPORT_SHA256:
        raise ValueError("Candidate-E external report frozen hash differs")
    if receipt_sha256 != PROMOTION_RECEIPT_SHA256:
        raise ValueError("Candidate-E external receipt frozen hash differs")
    ledger_sha256 = validate_hash(
        args.speaker_ledger,
        args.speaker_ledger_sha256,
        "Candidate-E prior speaker ledger",
    )
    validate_hash(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_shimmer_db_component_no_go_v23.json",
        V23_CLOSURE_SHA256,
        "Candidate-D/v23 closure",
    )
    validate_hash(
        Path(__file__).resolve().parent
        / "audit_avqi_route_c_six_joint_panel_readiness.py",
        READINESS_V2_SHA256,
        "six-joint readiness v2 source",
    )
    contract = read_json(args.contract, "Candidate-E readiness contract")
    validate_contract(contract)
    ledger = read_json(args.speaker_ledger, "Candidate-E prior speaker ledger")
    ledger_summary = validate_ledger(ledger)
    promotion_report = read_json(
        args.promotion_report,
        "Candidate-E external report",
    )
    promotion_receipt = read_json(
        args.promotion_receipt,
        "Candidate-E external receipt",
    )
    validation = validate_promotion(
        promotion_report,
        promotion_receipt,
        report_sha256=report_sha256,
        ledger_sha256=ledger_sha256,
    )
    validate_receipt_artifact_files(
        args.promotion_report.parent,
        promotion_receipt["artifact_sha256"],
    )
    report = build_readiness_report(
        source,
        report_sha256,
        receipt_sha256,
        ledger_sha256,
        contract_sha256,
        validation,
        ledger_summary,
    )
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "candidate_e_component_readiness_report_v4.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": COMPONENT_PASS,
        "component": "shimmer_db",
        "source_commit": source["head"],
        "shimmer_db_six_component_readiness_eligible": True,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "all_six_scientific_components_ready": True,
        "readiness_scope": (
            "component_scientific_status_only_not_joint_execution_authorization"
        ),
        "joint_panel_decision": JOINT_NO_GO,
        "execution_authorized": False,
        "joint_panel_authorized": False,
        "joint_scientific_promotion_granted": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
    }
    receipt_path = args.output_dir / "completion_receipt_v4.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": COMPONENT_PASS,
                "joint_panel_decision": JOINT_NO_GO,
                "joint_panel_authorized": False,
                "completion_receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
