#!/usr/bin/env python3
"""Apply the frozen dominance-capped fusion rule to a sealed raw panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from model.avqi_route_c_gradient_fusion import (
    CAP_POLICY,
    CONFLICT_POLICY,
    FUSION_SCHEMA_VERSION,
    JOINT_NORMALIZATION,
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
    fusion_from_gram,
)
from scripts import decide_avqi_route_c_six_component_gradients as legacy
from scripts.seal_avqi_route_c_six_gradient_fusion_panel_v1 import (
    AUDIT_SPLITS,
    EXPECTED_CASES_PER_SPLIT,
    EXPECTED_STRATA,
    OPENED_DECISION_RECEIPT_SHA256,
    OPENED_DECISION_REPORT_SHA256,
    OPENED_RAW_RECEIPT_SHA256,
    OPENED_RAW_REPORT_SHA256,
    PANEL_DECISION,
    PANEL_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION as PANEL_RECEIPT_SCHEMA_VERSION,
    SOURCE_LABEL_BANK_SHA256,
    TRAINING_NO_GO,
    validate_contract,
)


DECISION_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-decision-v1"
DECISION_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-gradient-fusion-decision-receipt-v1"
)
PASS_DECISION = "PASS_ROUTE_C_SIX_GRADIENT_DOMINANCE_CAPPED_FUSION_V1"
NO_GO_DECISION = "NO_GO_ROUTE_C_SIX_GRADIENT_DOMINANCE_CAPPED_FUSION_V1"
JOINT_PANEL_NO_GO = "NO_GO_ROUTE_C_SIX_JOINT_PANEL_PACKAGE_NOT_BOUND"
NONZERO_GRADIENT_NORM_MIN = 1e-10
MAXIMUM_GRADIENT_NORM = 1e4
ACTIVE_COMPONENTS = tuple(legacy.ACTIVE_COMPONENTS)
PAIRWISE_COMPONENT_KEYS = tuple(legacy.PAIRWISE_COMPONENT_KEYS)
EXPECTED_CASES = EXPECTED_CASES_PER_SPLIT * len(AUDIT_SPLITS)
IMPLEMENTATION_PATHS = {
    "avqi_route_c_gradient_fusion.py": Path(
        "model/avqi_route_c_gradient_fusion.py"
    ),
    "decide_avqi_route_c_six_gradient_fusion_v1.py": Path(
        "scripts/decide_avqi_route_c_six_gradient_fusion_v1.py"
    ),
    "seal_avqi_route_c_six_gradient_fusion_panel_v1.py": Path(
        "scripts/seal_avqi_route_c_six_gradient_fusion_panel_v1.py"
    ),
    "avqi_route_c_six_gradient_fusion_contract_v1.json": Path(
        "configs/avqi_route_c_six_gradient_fusion_contract_v1.json"
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--panel-seal", type=Path, required=True)
    parser.add_argument("--panel-seal-sha256", required=True)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--panel-receipt-sha256", required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--raw-report-sha256", required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--raw-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


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


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 is invalid")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_source(root: Path, expected_commit: str) -> dict[str, Any]:
    source = legacy._validate_source(root, expected_commit)
    implementation_sha256 = {}
    for name, relative_path in IMPLEMENTATION_PATHS.items():
        path = root.resolve() / relative_path
        if not path.is_file():
            raise ValueError(f"fusion implementation is unavailable: {name}")
        implementation_sha256[name] = sha256_file(path)
    source["fusion_implementation_sha256"] = implementation_sha256
    return source


def validate_panel_seal(
    panel: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    panel_sha256: str,
    contract_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    if panel.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError("fusion panel schema differs")
    if panel.get("decision") != PANEL_DECISION:
        raise ValueError("fusion panel was not sealed")
    if receipt.get("schema_version") != PANEL_RECEIPT_SCHEMA_VERSION:
        raise ValueError("fusion panel receipt schema differs")
    if receipt.get("decision") != PANEL_DECISION:
        raise ValueError("fusion panel receipt decision differs")
    artifacts = receipt.get("artifact_sha256")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("fusion_panel_seal_v1.json") != panel_sha256
    ):
        raise ValueError("fusion panel receipt does not bind its seal")
    filtered_sha256 = panel.get("filtered_label_bank_sha256")
    if (
        not _is_sha256(filtered_sha256)
        or artifacts.get("filtered_exact_component_label_bank_v1.csv")
        != filtered_sha256
    ):
        raise ValueError("fusion panel receipt does not bind filtered label bank")
    source = panel.get("source")
    if not isinstance(source, dict) or (
        source.get("head") != source_commit
        or receipt.get("source_commit") != source_commit
        or source.get("branch") != receipt.get("source_branch")
    ):
        raise ValueError("fusion panel source provenance differs")
    if panel.get("contract_sha256") != contract_sha256:
        raise ValueError("fusion panel contract binding differs")
    if panel.get("source_label_bank_sha256") != SOURCE_LABEL_BANK_SHA256:
        raise ValueError("fusion panel source label bank differs")
    opened_hashes = panel.get("opened_evidence_sha256")
    if opened_hashes != {
        "opened_raw_report": OPENED_RAW_REPORT_SHA256,
        "opened_raw_receipt": OPENED_RAW_RECEIPT_SHA256,
        "opened_decision_report": OPENED_DECISION_REPORT_SHA256,
        "opened_decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
    }:
        raise ValueError("fusion panel opened-evidence binding differs")
    if receipt.get("input_sha256") != {
        "contract": contract_sha256,
        "source_label_bank": SOURCE_LABEL_BANK_SHA256,
        "opened_raw_report": OPENED_RAW_REPORT_SHA256,
        "opened_raw_receipt": OPENED_RAW_RECEIPT_SHA256,
        "opened_decision_report": OPENED_DECISION_REPORT_SHA256,
        "opened_decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
    }:
        raise ValueError("fusion panel receipt input binding differs")
    if panel.get("opened_speaker_overlap") != 0:
        raise ValueError("fusion panel reused an opened speaker")
    if panel.get("fusion_rule_frozen_before_panel_selection") is not True:
        raise ValueError("fusion rule was not frozen before panel selection")
    for key in (
        "gradient_measurement_performed",
        "candidate_exact_outcomes_opened",
        "fresh_or_final_joint_panel_opened",
        "waveform_generation_performed",
        "joint_panel_authorized",
        "formal_generator_training_submitted",
    ):
        if panel.get(key) is not False:
            raise ValueError(f"fusion panel boundary differs: {key}")
    for key in (
        "candidate_exact_outcomes_opened",
        "joint_panel_authorized",
        "formal_generator_training_submitted",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"fusion panel receipt boundary differs: {key}")
    if (
        panel.get("generator_optimizer_steps") != 0
        or panel.get("authoritative_training_decision") != TRAINING_NO_GO
        or receipt.get("generator_optimizer_steps") != 0
        or receipt.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("fusion panel training boundary differs")
    cases = panel.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError("fusion panel case coverage differs")
    selectors = set()
    case_ids = set()
    target_vectors = {}
    speaker_sets = {split: set() for split in AUDIT_SPLITS}
    strata = {split: set() for split in AUDIT_SPLITS}
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("fusion panel case is not a mapping")
        split = row.get("split")
        if split not in AUDIT_SPLITS:
            raise ValueError("fusion panel split differs")
        required = (
            "case_id",
            "speaker_id",
            "sample_id",
            "sample_group",
            "view",
            "condition",
            "source_audio_file_sha256",
        )
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError("fusion panel case selector differs")
        if not _is_sha256(row["source_audio_file_sha256"]):
            raise ValueError("fusion panel source audio SHA-256 differs")
        selector = (
            split,
            row["speaker_id"],
            row["sample_id"],
            row["sample_group"],
            row["view"],
            row["condition"],
            row["source_audio_file_sha256"],
        )
        selectors.add(selector)
        case_ids.add(row["case_id"])
        speaker_sets[split].add(row["speaker_id"])
        strata[split].add(f"{row['sample_group']}/{row['view']}")
        target = row.get("same_speaker_clean_pathological_target")
        if (
            not isinstance(target, list)
            or len(target) != len(ACTIVE_COMPONENTS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in target
            )
        ):
            raise ValueError("fusion panel target vector differs")
        target_payload = json.dumps(
            [float(value) for value in target],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if row.get("target_vector_sha256") != hashlib.sha256(
            target_payload
        ).hexdigest():
            raise ValueError("fusion panel target vector SHA-256 differs")
        target_vectors[selector] = tuple(float(value) for value in target)
    if len(selectors) != EXPECTED_CASES or len(case_ids) != EXPECTED_CASES:
        raise ValueError("fusion panel selectors are not unique")
    for split in AUDIT_SPLITS:
        if (
            len(speaker_sets[split]) != EXPECTED_CASES_PER_SPLIT
            or strata[split] != set(EXPECTED_STRATA)
        ):
            raise ValueError(f"fusion panel {split} coverage differs")
    if speaker_sets[AUDIT_SPLITS[0]] & speaker_sets[AUDIT_SPLITS[1]]:
        raise ValueError("fusion panel calibration/holdout speakers overlap")
    excluded = set(panel.get("excluded_speaker_union", []))
    if not excluded or excluded & set.union(*speaker_sets.values()):
        raise ValueError("fusion panel opened-speaker exclusion differs")
    return {
        "case_selectors": selectors,
        "speaker_sets": {
            split: sorted(speaker_sets[split]) for split in AUDIT_SPLITS
        },
        "filtered_label_bank_sha256": filtered_sha256,
        "selection_salt": panel.get("selection_salt"),
        "excluded_speakers": sorted(excluded),
        "target_vectors": target_vectors,
    }


def validate_raw_targets(
    rows: list[Mapping[str, Any]],
    expected_targets: Mapping[tuple[str, ...], tuple[float, ...]],
) -> None:
    if len(rows) != len(expected_targets):
        raise ValueError("raw target coverage differs from sealed panel")
    for row in rows:
        selector = legacy._case_selector(row, raw=True)
        expected = expected_targets.get(selector)
        if expected is None:
            raise ValueError("raw target selector is absent from sealed panel")
        components = row.get("components")
        if (
            not isinstance(components, dict)
            or set(components) != set(ACTIVE_COMPONENTS)
        ):
            raise ValueError("raw target component keys differ")
        actual = []
        for component in ACTIVE_COMPONENTS:
            component_row = components[component]
            if not isinstance(component_row, dict):
                raise ValueError("raw target component is not a mapping")
            value = component_row.get("clean_pathological_target")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("raw target value is invalid")
            actual.append(float(value))
        if tuple(actual) != expected:
            raise ValueError("raw targets differ from sealed same-speaker targets")


def evaluate_fusion(
    rows: list[Mapping[str, Any]],
    raw_report: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any], list[dict[str, Any]]]:
    baseline_gates, baseline_metrics = legacy._evaluate_numeric_gates(
        rows,
        raw_report,
    )
    required_baseline_integrity = (
        "all_component_gradients_finite_nonzero_bounded",
        "all_joint_gradients_finite_nonzero_bounded",
        "calibration_only_inverse_gradient_weights",
        "calibration_weighted_median_ratio_le_1_000001",
        "all_15_pairwise_cosines_finite_reported",
        "all_6_component_to_joint_cosines_finite_reported",
    )
    baseline_integrity = all(baseline_gates[key] for key in required_baseline_integrity)
    calibration = raw_report.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("raw calibration summary is unavailable")
    weights = calibration.get("frozen_inverse_gradient_weights")
    if not isinstance(weights, dict):
        raise ValueError("raw calibration weights are unavailable")

    case_results = []
    for row in rows:
        components = row.get("components")
        pairwise = row.get("joint", {}).get("pairwise_component_cosines")
        if not isinstance(components, dict) or not isinstance(pairwise, dict):
            raise ValueError("raw fusion inputs are unavailable")
        norms = {
            component: float(components[component]["gradient_norm"])
            for component in ACTIVE_COMPONENTS
        }
        cosine_values = {
            key: float(pairwise[key]["cosine"])
            for key in PAIRWISE_COMPONENT_KEYS
        }
        result = fusion_from_gram(
            ACTIVE_COMPONENTS,
            norms,
            cosine_values,
            weights,
        )
        case_results.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "sample_id": row["sample_id"],
                "sample_group": row["sample_group"],
                "view": row["view"],
                "condition": row["condition"],
                "source_audio_file_sha256": row["source_audio_file_sha256"],
                "fusion": result,
            }
        )

    post_shares = [
        row["fusion"]["post_cap_maximum_share"] for row in case_results
    ]
    joint_norms = [row["fusion"]["joint_gradient_norm"] for row in case_results]
    joint_cosines = [
        cosine
        for row in case_results
        for cosine in row["fusion"]["component_to_joint_cosines"].values()
    ]
    cap_applications = [
        row for row in case_results if row["fusion"]["cap_applied"]
    ]
    gates = {
        "legacy_raw_measurement_integrity": baseline_integrity,
        "all_post_cap_weighted_component_shares_le_0_80": all(
            share <= MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE for share in post_shares
        ),
        "all_post_cap_joint_gradients_finite_nonzero_bounded": all(
            math.isfinite(norm)
            and norm > NONZERO_GRADIENT_NORM_MIN
            and norm <= MAXIMUM_GRADIENT_NORM
            for norm in joint_norms
        ),
        "all_post_cap_component_to_joint_cosines_nonnegative": all(
            math.isfinite(cosine) and cosine >= 0.0 for cosine in joint_cosines
        ),
        "no_component_amplified": all(
            row["fusion"]["no_component_amplified"] for row in case_results
        ),
        "only_unique_dominant_component_may_be_attenuated": all(
            row["fusion"]["only_dominant_component_attenuated"]
            for row in case_results
        ),
    }
    metrics = {
        "cases": len(case_results),
        "calibration_cases": sum(
            row["split"] == "surrogate_calibration" for row in case_results
        ),
        "holdout_cases": sum(
            row["split"] == "surrogate_holdout" for row in case_results
        ),
        "baseline": baseline_metrics,
        "post_cap_maximum_weighted_component_norm_share": max(post_shares),
        "post_cap_minimum_component_to_joint_cosine": min(joint_cosines),
        "post_cap_joint_gradient_norm_min": min(joint_norms),
        "post_cap_joint_gradient_norm_max": max(joint_norms),
        "cap_application_count": len(cap_applications),
        "cap_application_case_ids": [row["case_id"] for row in cap_applications],
    }
    return gates, metrics, case_results


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite fusion decision: {args.output_dir}"
        )
    paths = {
        "contract": _verified_file(
            args.contract,
            args.contract_sha256,
            "fusion contract",
        ),
        "panel_seal": _verified_file(
            args.panel_seal,
            args.panel_seal_sha256,
            "fusion panel seal",
        ),
        "panel_receipt": _verified_file(
            args.panel_receipt,
            args.panel_receipt_sha256,
            "fusion panel receipt",
        ),
        "raw_report": _verified_file(
            args.raw_report,
            args.raw_report_sha256,
            "fusion raw report",
        ),
        "raw_receipt": _verified_file(
            args.raw_receipt,
            args.raw_receipt_sha256,
            "fusion raw receipt",
        ),
    }
    source = verify_source(args.source_root, args.source_commit)
    if (
        source["fusion_implementation_sha256"][
            "avqi_route_c_six_gradient_fusion_contract_v1.json"
        ]
        != args.contract_sha256
    ):
        raise ValueError("fusion contract hash differs from clean source tree")
    contract = _read_json(paths["contract"], "fusion contract")
    policy = validate_contract(contract)
    fusion_contract = contract.get("fusion_rule")
    if not isinstance(fusion_contract, dict) or fusion_contract != {
        "schema_version": FUSION_SCHEMA_VERSION,
        "maximum_weighted_component_norm_share": (
            MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE
        ),
        "threshold_source": "unchanged frozen five-component numeric precedent",
        "cap_scope": "per_case_before_joint_sum",
        "cap_policy": CAP_POLICY,
        "component_amplification_allowed": False,
        "joint_normalization": JOINT_NORMALIZATION,
        "pairwise_negative_cosines": "diagnostic_only",
        "direction_conflict_policy": CONFLICT_POLICY,
        "speaker_or_case_identity_used_for_fusion": False,
        "exact_candidate_outcome_used_for_fusion": False,
        "learned_fusion_parameters": 0,
    }:
        raise ValueError("frozen fusion rule differs from implementation")
    panel = _read_json(paths["panel_seal"], "fusion panel seal")
    panel_receipt = _read_json(paths["panel_receipt"], "fusion panel receipt")
    precedent = validate_panel_seal(
        panel,
        panel_receipt,
        panel_sha256=args.panel_seal_sha256,
        contract_sha256=args.contract_sha256,
        source_commit=args.source_commit,
    )
    if policy["selection_salt"] != precedent["selection_salt"]:
        raise ValueError("fusion panel salt differs from frozen contract")
    contract_exclusions = sorted(
        speaker
        for speakers in policy["excluded_speakers_by_split"].values()
        for speaker in speakers
    )
    if precedent["excluded_speakers"] != contract_exclusions:
        raise ValueError("fusion panel exclusions differ from frozen contract")

    raw_report = _read_json(paths["raw_report"], "fusion raw report")
    raw_receipt = _read_json(paths["raw_receipt"], "fusion raw receipt")
    rows, raw_source_hashes = legacy._validate_raw_envelope(
        raw_report,
        raw_receipt,
        report_name=paths["raw_report"].name,
        report_sha256=args.raw_report_sha256,
        precedent=precedent,
        execution_source=source,
    )
    validate_raw_targets(rows, precedent["target_vectors"])
    if raw_source_hashes["label_bank"] != precedent["filtered_label_bank_sha256"]:
        raise ValueError("raw measurement does not bind sealed filtered label bank")
    selection = raw_report.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("selection_salt") != precedent["selection_salt"]
    ):
        raise ValueError("raw measurement does not bind sealed selection salt")

    numeric_gates, metrics, case_results = evaluate_fusion(rows, raw_report)
    structural_gates = {
        "fusion_contract_frozen_before_new_panel": True,
        "sealed_new_panel_bound_before_gradient_measurement": True,
        "all_opened_speakers_excluded": True,
        "new_calibration_and_holdout_speaker_disjoint": True,
        "same_speaker_clean_pathological_targets_bound": True,
        "raw_targets_match_sealed_panel": True,
        "candidate_exact_outcomes_closed": True,
        "fresh_and_final_joint_panels_closed": True,
        "generator_optimizer_steps_zero": True,
    }
    gates = {**structural_gates, **numeric_gates}
    decision = PASS_DECISION if all(gates.values()) else NO_GO_DECISION
    report = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "fusion_rule": fusion_contract,
        "component_order": list(ACTIVE_COMPONENTS),
        "contract_sha256": args.contract_sha256,
        "panel_evidence_sha256": {
            "seal": args.panel_seal_sha256,
            "receipt": args.panel_receipt_sha256,
            "filtered_label_bank": precedent["filtered_label_bank_sha256"],
        },
        "raw_measurement_sha256": {
            "report": args.raw_report_sha256,
            "receipt": args.raw_receipt_sha256,
        },
        "opened_evidence_role": "diagnostic_and_exclusion_only",
        "opened_evidence_sha256": {
            "decision_report": OPENED_DECISION_REPORT_SHA256,
            "decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
            "raw_report": OPENED_RAW_REPORT_SHA256,
            "raw_receipt": OPENED_RAW_RECEIPT_SHA256,
        },
        "excluded_speakers": precedent["excluded_speakers"],
        "new_panel_speakers_by_split": precedent["speaker_sets"],
        "metrics": metrics,
        "case_results": case_results,
        "gates": gates,
        "source": source,
        "fusion_validation_holdout_opened": True,
        "fusion_scientific_promotion_granted": decision == PASS_DECISION,
        "eligible_for_joint_package_binding": decision == PASS_DECISION,
        "joint_panel_authorized": False,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "candidate_exact_outcomes_opened": False,
        "fresh_or_final_joint_panel_opened": False,
        "waveform_generation_performed": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "six_gradient_fusion_decision_report_v1.json"
    _write_json(report_path, report)
    input_bindings = {
        "contract": args.contract_sha256,
        "panel_seal": args.panel_seal_sha256,
        "panel_receipt": args.panel_receipt_sha256,
        "raw_report": args.raw_report_sha256,
        "raw_receipt": args.raw_receipt_sha256,
    }
    observed_after = {name: sha256_file(path) for name, path in paths.items()}
    if observed_after != input_bindings:
        raise ValueError("fusion decision input changed during evaluation")
    receipt = {
        "schema_version": DECISION_RECEIPT_SCHEMA_VERSION,
        "decision": decision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
        "input_sha256": input_bindings,
        "implementation_sha256": source["fusion_implementation_sha256"],
        "post_evaluation_immutability_verified": True,
        "fusion_scientific_promotion_granted": decision == PASS_DECISION,
        "joint_panel_authorized": False,
        "candidate_exact_outcomes_opened": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    _write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": decision,
                "report_sha256": sha256_file(report_path),
                "receipt_sha256": sha256_file(receipt_path),
                "maximum_post_cap_share": metrics[
                    "post_cap_maximum_weighted_component_norm_share"
                ],
                "minimum_post_cap_component_to_joint_cosine": metrics[
                    "post_cap_minimum_component_to_joint_cosine"
                ],
                "joint_panel_authorized": False,
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
