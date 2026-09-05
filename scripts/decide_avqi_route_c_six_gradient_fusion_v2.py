#!/usr/bin/env python3
"""Decide the sealed SVD six-gradient fusion validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_gradient_fusion import (
    CAP_POLICY,
    CONFLICT_POLICY,
    FUSION_SCHEMA_VERSION,
    JOINT_NORMALIZATION,
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts import decide_avqi_route_c_six_component_gradients as legacy
from scripts.decide_avqi_route_c_six_gradient_fusion_v1 import (
    evaluate_fusion,
    validate_raw_targets,
)
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    load_svd_fusion_label_bank,
)
from scripts.materialize_avqi_route_c_six_gradient_svd_panel_v2 import (
    MATERIALIZED_DECISION,
    MATERIALIZED_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION as MATERIALIZED_RECEIPT_SCHEMA_VERSION,
    TARGET_METRICS,
    validate_source_panel,
    validate_updated_ledger,
)
from scripts.seal_avqi_route_c_six_gradient_svd_source_panel_v2 import (
    AUDIT_SPLITS,
    EXPECTED_CASES,
    STRATA,
    TRAINING_NO_GO,
    validate_contract,
)


DECISION_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-decision-v2"
DECISION_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-gradient-fusion-decision-receipt-v2"
)
PASS_DECISION = "PASS_ROUTE_C_SIX_GRADIENT_DOMINANCE_CAPPED_FUSION_V2"
NO_GO_DECISION = "NO_GO_ROUTE_C_SIX_GRADIENT_DOMINANCE_CAPPED_FUSION_V2"
JOINT_PANEL_NO_GO = "NO_GO_ROUTE_C_SIX_JOINT_PANEL_PACKAGE_NOT_BOUND"
EXPECTED_STRATA = tuple(f"{sex}/{view}" for sex, view in STRATA)
IMPLEMENTATION_PATHS = {
    "avqi_route_c_gradient_fusion.py": Path(
        "model/avqi_route_c_gradient_fusion.py"
    ),
    "avqi_route_c_candidate_e.py": Path("model/avqi_route_c_candidate_e.py"),
    "evaluate_avqi_route_c_multicomponent_gradients.py": Path(
        "scripts/evaluate_avqi_route_c_multicomponent_gradients.py"
    ),
    "evaluate_avqi_route_c_six_component_gradients.py": Path(
        "scripts/evaluate_avqi_route_c_six_component_gradients.py"
    ),
    "seal_avqi_route_c_six_gradient_svd_source_panel_v2.py": Path(
        "scripts/seal_avqi_route_c_six_gradient_svd_source_panel_v2.py"
    ),
    "materialize_avqi_route_c_six_gradient_svd_panel_v2.py": Path(
        "scripts/materialize_avqi_route_c_six_gradient_svd_panel_v2.py"
    ),
    "decide_avqi_route_c_six_gradient_fusion_v2.py": Path(
        "scripts/decide_avqi_route_c_six_gradient_fusion_v2.py"
    ),
    "avqi_route_c_six_gradient_fusion_contract_v2.json": Path(
        "configs/avqi_route_c_six_gradient_fusion_contract_v2.json"
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "contract",
        "source-panel-seal",
        "source-panel-receipt",
        "updated-speaker-ledger",
        "materialized-panel-seal",
        "materialized-panel-receipt",
        "label-bank",
        "raw-report",
        "raw-receipt",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
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


def verify_source(root: Path, expected_commit: str) -> dict[str, Any]:
    source = legacy._validate_source(root.resolve(), expected_commit)
    implementation_sha256 = {}
    for name, relative_path in IMPLEMENTATION_PATHS.items():
        path = root.resolve() / relative_path
        if not path.is_file():
            raise ValueError(f"fusion v2 implementation is unavailable: {name}")
        implementation_sha256[name] = sha256_file(path)
    source["fusion_v2_implementation_sha256"] = implementation_sha256
    return source


def _require_boundaries(value: Mapping[str, Any], label: str) -> None:
    expected = {
        "base_or_candidate_exact_outcomes_opened": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{label} boundary differs: {key}")


def _audio_binding(row: Mapping[str, Any], prefix: str) -> tuple[Path, str]:
    path = Path(str(row.get(f"{prefix}_path", "")))
    digest = row.get(f"{prefix}_sha256")
    if not path.is_absolute() or not _is_sha256(digest):
        raise ValueError(f"materialized {prefix} binding differs")
    if sha256_file(path) != digest:
        raise ValueError(f"materialized {prefix} waveform SHA-256 differs")
    return path.resolve(), str(digest)


def validate_materialized_panel(
    contract: Mapping[str, Any],
    source_seal: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    updated_ledger: Mapping[str, Any],
    materialized: Mapping[str, Any],
    materialized_receipt: Mapping[str, Any],
    *,
    contract_sha256: str,
    source_seal_sha256: str,
    source_receipt_sha256: str,
    ledger_sha256: str,
    materialized_sha256: str,
    label_bank_path: Path,
    label_bank_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    source_rows = validate_source_panel(
        source_seal,
        source_receipt,
        seal_sha256=source_seal_sha256,
        ledger_sha256=ledger_sha256,
        contract_sha256=contract_sha256,
    )
    validate_updated_ledger(updated_ledger, source_rows, source_commit)
    if (
        source_seal.get("source", {}).get("head") != source_commit
        or materialized.get("schema_version") != MATERIALIZED_SCHEMA_VERSION
        or materialized.get("decision") != MATERIALIZED_DECISION
        or materialized_receipt.get("schema_version")
        != MATERIALIZED_RECEIPT_SCHEMA_VERSION
        or materialized_receipt.get("decision") != MATERIALIZED_DECISION
        or materialized.get("contract_sha256") != contract_sha256
        or materialized.get("label_bank_sha256") != label_bank_sha256
        or materialized.get("source", {}).get("head") != source_commit
        or materialized_receipt.get("source_commit") != source_commit
        or materialized_receipt.get("source_branch")
        != materialized.get("source", {}).get("branch")
    ):
        raise ValueError("materialized SVD panel identity differs")
    if materialized.get("source_panel_sha256") != {
        "seal": source_seal_sha256,
        "receipt": source_receipt_sha256,
        "updated_speaker_ledger": ledger_sha256,
    }:
        raise ValueError("materialized SVD panel source binding differs")
    artifacts = materialized_receipt.get("artifact_sha256")
    if artifacts != {
        label_bank_path.name: label_bank_sha256,
        "svd_materialized_panel_seal_v2.json": materialized_sha256,
    }:
        raise ValueError("materialized SVD receipt artifact binding differs")
    if materialized_receipt.get("input_sha256") != materialized.get(
        "input_sha256"
    ):
        raise ValueError("materialized SVD receipt input binding differs")
    expected_inputs = {
        "contract": contract_sha256,
        "source_panel_seal": source_seal_sha256,
        "source_panel_receipt": source_receipt_sha256,
        "updated_speaker_ledger": ledger_sha256,
        "base_label_bank": contract["external_svd_source_panel"][
            "normalization_label_bank_sha256"
        ],
        "fixed_recipes": contract["waveform_materialization"][
            "fixed_recipes_sha256"
        ],
        "generator_config": contract["waveform_materialization"][
            "generator_config_sha256"
        ],
        "generator_checkpoint": contract["waveform_materialization"][
            "generator_checkpoint_sha256"
        ],
        "simulation_config": contract["waveform_materialization"][
            "simulation_config_sha256"
        ],
        "simulation_source": contract["waveform_materialization"][
            "simulation_source_sha256"
        ],
        "exact_avqi_code_tree": contract["exact_authority"][
            "avqi_code_tree_sha256"
        ],
    }
    if materialized.get("input_sha256") != expected_inputs:
        raise ValueError("materialized SVD frozen input binding differs")
    for value, label in (
        (materialized, "materialized SVD panel"),
        (materialized_receipt, "materialized SVD receipt"),
    ):
        _require_boundaries(value, label)
    if (
        materialized.get("clean_target_scalars_sealed_before_gradient_measurement")
        is not True
        or materialized.get("target_scalar_values_opened") is not True
        or materialized.get("emitted_waveform_highpass") is not False
        or materialized.get("generator_mode") != "frozen_inference_only"
        or materialized.get("generator_optimizer_created") is not False
        or materialized_receipt.get("target_scalar_values_opened") is not True
    ):
        raise ValueError("materialized SVD scientific boundary differs")

    source_by_case = {row["case_id"]: row for row in source_rows}
    rows = materialized.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_CASES
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("materialized SVD row coverage differs")
    if len({row.get("case_id") for row in rows}) != EXPECTED_CASES:
        raise ValueError("materialized SVD case IDs differ")
    cases, _, _, selection = load_svd_fusion_label_bank(
        label_bank_path,
        label_bank_sha256,
        contract["external_svd_source_panel"]["selection_salt"],
    )
    case_lookup = {
        (
            case.split,
            case.speaker_id,
            case.sample_id,
            case.sample_group,
            case.view,
        ): case
        for case in cases
    }
    selectors: set[tuple[str, ...]] = set()
    targets: dict[tuple[str, ...], tuple[float, ...]] = {}
    speaker_sets = {split: set() for split in AUDIT_SPLITS}
    peak_source_rows = {}
    for row in rows:
        case_id = row.get("case_id")
        source_row = source_by_case.get(case_id)
        if source_row is None:
            raise ValueError("materialized SVD case is absent from source seal")
        for key in (
            "speaker_id",
            "canonical_speaker_id",
            "session_id",
            "sample_id",
            "split",
            "sample_group",
            "sex",
            "view",
            "condition",
            "recipe_index",
        ):
            if row.get(key) != source_row.get(key):
                raise ValueError(f"materialized SVD source field differs: {key}")
        target_path, target_sha256 = _audio_binding(row, "target")
        degraded_path, degraded_sha256 = _audio_binding(row, "degraded")
        base_path, base_sha256 = _audio_binding(row, "base")
        if not all(path.is_file() for path in (target_path, degraded_path, base_path)):
            raise ValueError("materialized SVD waveform is unavailable")
        exact_target_values = row.get("exact_target_components")
        if (
            not isinstance(exact_target_values, Mapping)
            or set(exact_target_values) != set(TARGET_METRICS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in exact_target_values.values()
            )
        ):
            raise ValueError("materialized SVD exact target components differ")
        exact_target_payload = json.dumps(
            {name: float(exact_target_values[name]) for name in TARGET_METRICS},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if row.get("exact_target_components_sha256") != hashlib.sha256(
            exact_target_payload
        ).hexdigest():
            raise ValueError("materialized SVD exact target hash differs")
        target_values = row.get("target_components")
        if (
            not isinstance(target_values, list)
            or len(target_values) != len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in target_values
            )
        ):
            raise ValueError("materialized SVD target vector differs")
        target_tuple = tuple(float(value) for value in target_values)
        expected_gradient_target = tuple(
            float(value)
            for value in torch.tensor(
                [exact_target_values[name] for name in ROUTE_C_SIX_ACTIVE_COMPONENTS],
                dtype=torch.float32,
            ).tolist()
        )
        if target_tuple != expected_gradient_target:
            raise ValueError("materialized SVD gradient target precision differs")
        target_payload = json.dumps(
            list(target_tuple), separators=(",", ":")
        ).encode("utf-8")
        if row.get("target_vector_sha256") != hashlib.sha256(
            target_payload
        ).hexdigest():
            raise ValueError("materialized SVD target vector hash differs")
        case_key = (
            row["split"],
            row["canonical_speaker_id"],
            row["sample_id"],
            row["sample_group"],
            row["view"],
        )
        case = case_lookup.get(case_key)
        if case is None or case.waveform_path.resolve() != base_path:
            raise ValueError("materialized SVD label-bank case differs")
        if case.waveform_sha256 != base_sha256 or tuple(
            float(value) for value in case.clean_target.tolist()
        ) != target_tuple:
            raise ValueError("materialized SVD label-bank target differs")
        selector = (
            row["split"],
            row["canonical_speaker_id"],
            row["sample_id"],
            row["sample_group"],
            row["view"],
            row["condition"],
            base_sha256,
        )
        selectors.add(selector)
        targets[selector] = target_tuple
        speaker_sets[row["split"]].add(row["canonical_speaker_id"])
        peak_source_rows[case_id] = {
            "base_waveform_sha256": base_sha256,
            "target_waveform_sha256": target_sha256,
        }
    if len(selectors) != EXPECTED_CASES:
        raise ValueError("materialized SVD selectors differ")
    if speaker_sets[AUDIT_SPLITS[0]] & speaker_sets[AUDIT_SPLITS[1]]:
        raise ValueError("materialized SVD speakers overlap")
    return {
        "case_selectors": selectors,
        "speaker_sets": {
            split: sorted(speaker_sets[split]) for split in AUDIT_SPLITS
        },
        "target_vectors": targets,
        "label_bank_sha256": label_bank_sha256,
        "selection": selection,
        "waveform_bindings": peak_source_rows,
    }


def _peak_hash_gate(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        topology = row.get("topology")
        components = row.get("components")
        if not isinstance(topology, Mapping) or not isinstance(components, Mapping):
            return False
        shimmer = components.get("shimmer_db")
        if not isinstance(shimmer, Mapping):
            return False
        projection = shimmer.get("candidate_e_projection")
        if not isinstance(projection, Mapping):
            return False
        digest = topology.get("highpass_pcm16_sha256")
        scaled = topology.get("highpass_peak_scaled")
        if (
            not _is_sha256(digest)
            or projection.get("candidate_e_exact_highpass_pcm16_sha256")
            != digest
            or projection.get("candidate_e_peak_scale_support_pass") is not True
            or projection.get("candidate_e_peak_handling_pass") is not True
            or not isinstance(scaled, bool)
            or projection.get("candidate_e_peak_scale_abstention_pass")
            is not (not scaled)
        ):
            return False
    return True


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite fusion v2 decision: {args.output_dir}"
        )
    input_args = {
        "contract": (args.contract, args.contract_sha256),
        "source_panel_seal": (
            args.source_panel_seal,
            args.source_panel_seal_sha256,
        ),
        "source_panel_receipt": (
            args.source_panel_receipt,
            args.source_panel_receipt_sha256,
        ),
        "updated_speaker_ledger": (
            args.updated_speaker_ledger,
            args.updated_speaker_ledger_sha256,
        ),
        "materialized_panel_seal": (
            args.materialized_panel_seal,
            args.materialized_panel_seal_sha256,
        ),
        "materialized_panel_receipt": (
            args.materialized_panel_receipt,
            args.materialized_panel_receipt_sha256,
        ),
        "label_bank": (args.label_bank, args.label_bank_sha256),
        "raw_report": (args.raw_report, args.raw_report_sha256),
        "raw_receipt": (args.raw_receipt, args.raw_receipt_sha256),
    }
    paths = {
        name: _verified_file(path, digest, name)
        for name, (path, digest) in input_args.items()
    }
    source = verify_source(args.source_root, args.source_commit)
    if (
        source["fusion_v2_implementation_sha256"][
            "avqi_route_c_six_gradient_fusion_contract_v2.json"
        ]
        != args.contract_sha256
    ):
        raise ValueError("fusion v2 contract differs from clean source tree")
    contract = _read_json(paths["contract"], "fusion v2 contract")
    validate_contract(contract)
    source_seal = _read_json(paths["source_panel_seal"], "SVD source seal")
    source_receipt = _read_json(
        paths["source_panel_receipt"], "SVD source receipt"
    )
    ledger = _read_json(paths["updated_speaker_ledger"], "SVD speaker ledger")
    materialized = _read_json(
        paths["materialized_panel_seal"], "materialized SVD panel"
    )
    materialized_receipt = _read_json(
        paths["materialized_panel_receipt"], "materialized SVD receipt"
    )
    precedent = validate_materialized_panel(
        contract,
        source_seal,
        source_receipt,
        ledger,
        materialized,
        materialized_receipt,
        contract_sha256=args.contract_sha256,
        source_seal_sha256=args.source_panel_seal_sha256,
        source_receipt_sha256=args.source_panel_receipt_sha256,
        ledger_sha256=args.updated_speaker_ledger_sha256,
        materialized_sha256=args.materialized_panel_seal_sha256,
        label_bank_path=paths["label_bank"],
        label_bank_sha256=args.label_bank_sha256,
        source_commit=args.source_commit,
    )
    raw_report = _read_json(paths["raw_report"], "six-gradient raw report")
    raw_receipt = _read_json(paths["raw_receipt"], "six-gradient raw receipt")
    rows, raw_source_hashes = legacy._validate_raw_envelope(
        raw_report,
        raw_receipt,
        report_name=paths["raw_report"].name,
        report_sha256=args.raw_report_sha256,
        precedent=precedent,
        execution_source=source,
        expected_strata=EXPECTED_STRATA,
    )
    validate_raw_targets(rows, precedent["target_vectors"])
    if raw_source_hashes["label_bank"] != args.label_bank_sha256:
        raise ValueError("raw gradients do not bind materialized SVD label bank")
    selection = raw_report.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_mode") != "sealed_external_svd_v2"
        or selection.get("selection_salt")
        != contract["external_svd_source_panel"]["selection_salt"]
        or selection.get("base_or_candidate_exact_outcomes_opened") is not False
    ):
        raise ValueError("raw SVD selection boundary differs")

    numeric_gates, metrics, case_results = evaluate_fusion(rows, raw_report)
    peak_hash_gate = _peak_hash_gate(rows)
    structural_gates = {
        "fusion_contract_frozen_before_source_selection": True,
        "source_split_sealed_before_waveform_materialization": True,
        "clean_target_scalars_sealed_before_gradient_measurement": True,
        "all_prior_ledger_speakers_excluded": True,
        "calibration_and_holdout_speaker_disjoint": True,
        "same_speaker_clean_pathological_targets_bound": True,
        "raw_targets_match_sealed_target_vectors": True,
        "all_candidate_e_peak_paths_pcm16_hash_bound": peak_hash_gate,
        "candidate_exact_outcomes_closed": True,
        "generator_optimizer_steps_zero": True,
    }
    gates = {**structural_gates, **numeric_gates}
    decision = PASS_DECISION if all(gates.values()) else NO_GO_DECISION
    report = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "fusion_rule": contract["fusion_rule"],
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "contract_sha256": args.contract_sha256,
        "source_panel_sha256": {
            "seal": args.source_panel_seal_sha256,
            "receipt": args.source_panel_receipt_sha256,
            "updated_speaker_ledger": args.updated_speaker_ledger_sha256,
        },
        "materialized_panel_sha256": {
            "seal": args.materialized_panel_seal_sha256,
            "receipt": args.materialized_panel_receipt_sha256,
            "label_bank": args.label_bank_sha256,
        },
        "raw_measurement_sha256": {
            "report": args.raw_report_sha256,
            "receipt": args.raw_receipt_sha256,
        },
        "predecessor_failure_role": "diagnostic_and_exclusion_only",
        "predecessor_evidence": contract["predecessor_evidence"],
        "panel_speakers_by_split": precedent["speaker_sets"],
        "metrics": metrics,
        "case_results": case_results,
        "gates": gates,
        "source": source,
        "fusion_validation_holdout_opened": True,
        "fusion_scientific_promotion_granted": decision == PASS_DECISION,
        "eligible_for_joint_package_binding": decision == PASS_DECISION,
        "joint_panel_authorized": False,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "base_or_candidate_exact_outcomes_opened": False,
        "fresh_or_final_joint_panel_opened": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "six_gradient_fusion_decision_report_v2.json"
    _write_json(report_path, report)
    input_hashes = {name: digest for name, (_, digest) in input_args.items()}
    observed_after = {name: sha256_file(path) for name, path in paths.items()}
    if observed_after != input_hashes:
        raise ValueError("fusion v2 decision input changed during evaluation")
    receipt = {
        "schema_version": DECISION_RECEIPT_SCHEMA_VERSION,
        "decision": decision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
        "input_sha256": input_hashes,
        "implementation_sha256": source[
            "fusion_v2_implementation_sha256"
        ],
        "post_evaluation_immutability_verified": True,
        "fusion_scientific_promotion_granted": decision == PASS_DECISION,
        "eligible_for_joint_package_binding": decision == PASS_DECISION,
        "joint_panel_authorized": False,
        "base_or_candidate_exact_outcomes_opened": False,
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
                "all_candidate_e_peak_paths_pcm16_hash_bound": peak_hash_gate,
                "joint_panel_authorized": False,
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
