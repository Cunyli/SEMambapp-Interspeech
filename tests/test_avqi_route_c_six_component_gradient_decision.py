from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import decide_avqi_route_c_six_component_gradients as decision
from scripts.decide_avqi_route_c_six_component_gradients import (
    ACTIVE_COMPONENTS,
    AUDIT_SPLITS,
    DECISION_SCHEMA_VERSION,
    FIVE_ACTIVE_COMPONENTS,
    FROZEN_FIVE_RECEIPT_SHA256,
    FROZEN_FIVE_REPORT_SHA256,
    JOINT_PANEL_NO_GO,
    NO_GO_DECISION,
    PAIRWISE_COMPONENT_KEYS,
    PASS_DECISION,
    RAW_IMPLEMENTATION_KEYS,
    RAW_PENDING_DECISION,
    RAW_RECEIPT_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    RAW_SOURCE_EVIDENCE_KEYS,
    SELECTION_STRATA,
    TOPOLOGY_HIGHPASS,
    TOPOLOGY_IMPLEMENTATION,
    TOPOLOGY_LOADER,
    TRAINING_NO_GO,
    decision_requirements,
    evaluate_six_gradient_decision,
)


def _speakers() -> dict[str, list[str]]:
    return {
        "surrogate_calibration": ["cal-mild-cs", "cal-mild-sv", "cal-severe-cs", "cal-severe-sv"],
        "surrogate_holdout": ["hold-mild-cs", "hold-mild-sv", "hold-severe-cs", "hold-severe-sv"],
    }


def _case_rows(*, raw: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for split in AUDIT_SPLITS:
        for speaker, stratum in zip(_speakers()[split], SELECTION_STRATA):
            index += 1
            sample_group, view = stratum.split("/")
            row: dict[str, object] = {
                "split": split,
                "speaker_id": speaker,
                "sample_id": f"sample-{index}",
                "sample_group": sample_group,
                "view": view,
                "condition": "aug16k_phone",
                ("source_audio_file_sha256" if raw else "audio_sha256"): (
                    f"{index:064x}"
                ),
            }
            if raw:
                weights = {name: 1.0 for name in ACTIVE_COMPONENTS}
                norms = {name: 1.0 for name in ACTIVE_COMPONENTS}
                shares = {name: 1.0 / len(ACTIVE_COMPONENTS) for name in ACTIVE_COMPONENTS}
                pairwise = {}
                for pair_index, pair in enumerate(PAIRWISE_COMPONENT_KEYS):
                    cosine = -0.2 if pair_index == 0 else 0.1
                    pairwise[pair] = {
                        "cosine": cosine,
                        "negative_direction_observed": cosine < 0.0,
                        "scientific_gate_applied": False,
                    }
                row.update(
                    {
                        "case_id": f"case-{index}",
                        "source_waveform_float32_sha256": f"{index + 20:064x}",
                        "segment_samples": 48_000,
                        "topology": {
                            "role": "base_current_output",
                            "worker_role": "current_output_topology",
                            "topology_sha256": f"{index + 40:064x}",
                            "pulse_count": 4,
                            "metric_source_range_count": 1,
                            "metric_constant_prefix_samples": 0,
                            "slot2_shimmer_percent_uses_topology": False,
                            "slot3_shimmer_db_uses_topology": True,
                        },
                        "components": {
                            name: {
                                "prediction": 1.0,
                                "clean_pathological_target": 2.0,
                                "normalized_signed_error": -1.0,
                                "normalized_bidirectional_gap": 1.0,
                                "smooth_l1_loss": 0.5,
                                "gradient_norm": norms[name],
                                "finite_observed": True,
                                "strictly_positive_norm_observed": True,
                                "scientific_gate_applied": False,
                            }
                            for name in ACTIVE_COMPONENTS
                        },
                        "joint": {
                            "gradient_norm": 1.0,
                            "calibration_only_inverse_gradient_weights": weights,
                            "weighted_component_gradient_norms": norms,
                            "weighted_component_norm_shares": shares,
                            "maximum_component_norm_share": 1.0
                            / len(ACTIVE_COMPONENTS),
                            "dominant_component": ACTIVE_COMPONENTS[0],
                            "pairwise_component_cosines": pairwise,
                            "component_to_joint_cosines": {
                                name: {
                                    "cosine": 0.5,
                                    "negative_direction_observed": False,
                                    "scientific_gate_applied": False,
                                }
                                for name in ACTIVE_COMPONENTS
                            },
                            "all_values_finite_observed": True,
                            "scientific_gate_applied": False,
                        },
                    }
                )
            rows.append(row)
    return rows


def _five_evidence() -> tuple[dict[str, object], dict[str, object]]:
    selection = {
        "allowed_splits": list(AUDIT_SPLITS),
        "cases": 8,
        "cases_by_split": {split: 4 for split in AUDIT_SPLITS},
        "final_panel_opened": False,
        "selection_salt": "frozen-five-selection",
        "speaker_overlap": 0,
        "speakers_by_split": _speakers(),
        "strata": list(SELECTION_STRATA),
        "target_stat_rows": 100,
    }
    component_summary = {
        name: {"opposed_to_joint_cases": 0} for name in FIVE_ACTIVE_COMPONENTS
    }
    report: dict[str, object] = {
        "schema_version": "avqi_route_c_five_component_gradient_audit_v1",
        "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
        "contract": {
            "loss_target": decision.LOSS_TARGET,
            "avqi_scalar_coefficient_used_for_direction": False,
            "calibration_only_weight_selection": True,
            "weight_rule": decision.WEIGHT_RULE,
        },
        "selection": selection,
        "case_results": _case_rows(raw=False),
        "calibration": {
            "cases": 4,
            "components": deepcopy(component_summary),
            "pairwise_component_cosines": {
                "cpps__hnr": {"direction_conflict_cases": 1}
            },
            "weighted_median_norm_ratio": 1.0,
        },
        "holdout": {
            "cases": 4,
            "components": deepcopy(component_summary),
            "pairwise_component_cosines": {
                "cpps__hnr": {"direction_conflict_cases": 1}
            },
        },
        "gates": {"accepted": True},
        "runtime": {"slurm_job_id": "19906556"},
        "joint_scientific_promotion_granted": False,
        "combined_final_panel_opened": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt: dict[str, object] = {
        "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
        "slurm_job_id": "19906556",
        "artifact_sha256": {
            "gradient_interference_report.json": FROZEN_FIVE_REPORT_SHA256
        },
        "joint_scientific_promotion_granted": False,
        "combined_final_panel_opened": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    return report, receipt


def _aggregate_summary() -> dict[str, object]:
    return {
        "cases": 4,
        "components": {
            name: {
                "gradient_norm_min": 1.0,
                "gradient_norm_median": 1.0,
                "gradient_norm_max": 1.0,
                "weighted_norm_share_median": 1.0 / 6.0,
                "weighted_norm_share_max": 1.0 / 6.0,
                "joint_cosine_min": 0.5,
                "joint_cosine_median": 0.5,
                "joint_cosine_max": 0.5,
                "negative_to_joint_observations": 0,
            }
            for name in ACTIVE_COMPONENTS
        },
        "pairwise_component_cosines": {
            pair: {
                "cosine_min": -0.2 if index == 0 else 0.1,
                "cosine_median": -0.2 if index == 0 else 0.1,
                "cosine_max": -0.2 if index == 0 else 0.1,
                "negative_direction_observations": 4 if index == 0 else 0,
                "negative_direction_fraction": 1.0 if index == 0 else 0.0,
            }
            for index, pair in enumerate(PAIRWISE_COMPONENT_KEYS)
        },
        "joint_gradient_norm_min": 1.0,
        "joint_gradient_norm_median": 1.0,
        "joint_gradient_norm_max": 1.0,
        "maximum_component_norm_share_observed": 1.0 / 6.0,
        "component_gradient_measurements": 24,
        "pairwise_cosine_measurements": 60,
        "component_to_joint_cosine_measurements": 24,
        "all_values_finite_observed": True,
        "scientific_gate_applied": False,
    }


def _raw_implementation_hashes() -> dict[str, str]:
    return {
        key: f"{index + 300:064x}"
        for index, key in enumerate(RAW_IMPLEMENTATION_KEYS)
    }


def _execution_source() -> dict[str, object]:
    return {
        "root": "/clean/source",
        "head": "a" * 40,
        "branch": "feat/avqi-route-c-six-component-scaffold-v1",
        "accepted_base_commit": decision.ACCEPTED_DECISION_BASE_COMMIT,
        "raw_implementation_sha256": _raw_implementation_hashes(),
    }


def _raw_evidence() -> tuple[dict[str, object], dict[str, object], str, str]:
    source_hashes = {
        key: f"{index + 100:064x}"
        for index, key in enumerate(RAW_SOURCE_EVIDENCE_KEYS)
    }
    source_hashes["five_gradient_report"] = FROZEN_FIVE_REPORT_SHA256
    source_hashes["five_gradient_receipt"] = FROZEN_FIVE_RECEIPT_SHA256
    source_evidence = {
        key: {"path": f"/immutable/{key}", "sha256": digest}
        for key, digest in source_hashes.items()
    }
    rows = _case_rows(raw=True)
    calibration = _aggregate_summary()
    calibration.update(
        {
            "median_component_gradient_norms": {
                name: 1.0 for name in ACTIVE_COMPONENTS
            },
            "frozen_inverse_gradient_weights": {
                name: 1.0 for name in ACTIVE_COMPONENTS
            },
            "weighted_median_gradient_norms": {
                name: 1.0 for name in ACTIVE_COMPONENTS
            },
            "weights_selected_on_holdout": False,
        }
    )
    report: dict[str, object] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "decision": RAW_PENDING_DECISION,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "contract": {
            "source": {
                "head": "a" * 40,
                "branch": "feat/avqi-route-c-six-component-scaffold-v1",
                "accepted_base_commit": "f" * 40,
            },
            "component_order": list(ACTIVE_COMPONENTS),
            "loss_target": decision.LOSS_TARGET,
            "avqi_scalar_coefficient_used_for_direction": False,
            "weight_fit_split": "surrogate_calibration",
            "weight_rule": decision.WEIGHT_RULE,
            "scientific_schema_frozen": False,
            "numeric_scientific_gates_applied": False,
        },
        "slot2_slot3_separation": {
            "slot2_shimmer_percent": {
                "component_index": 2,
                "source": "sealed_shimmer_percent_checkpoint",
                "checkpoint_output_preserved": True,
                "v19_topology_used": False,
            },
            "slot3_shimmer_db": {
                "component_index": 3,
                "source": "current_waveform_with_detached_v19_base_topology",
                "checkpoint_affine_used": False,
                "v19_topology_used": True,
                "topology_role": "base_current_output",
                "implementation": TOPOLOGY_IMPLEMENTATION,
                "metric_highpass": TOPOLOGY_HIGHPASS,
                "topology_input_loader": TOPOLOGY_LOADER,
                "scientific_promotion_granted": False,
            },
            "slots_are_independent": True,
        },
        "source_evidence": source_evidence,
        "source_evidence_sha256": source_hashes,
        "selection": {
            "allowed_splits": list(AUDIT_SPLITS),
            "cases": 8,
            "cases_by_split": {split: 4 for split in AUDIT_SPLITS},
            "final_panel_opened": False,
            "speaker_overlap": 0,
            "speakers_by_split": _speakers(),
            "strata": list(SELECTION_STRATA),
            "calibration_speaker_ids": _speakers()["surrogate_calibration"],
            "holdout_speaker_ids": _speakers()["surrogate_holdout"],
            "component_and_joint_share_split": True,
            "topology_manifest_uses_same_selection": True,
        },
        "topology_coverage": {
            "expected_cases": 8,
            "observed_cases": 8,
            "unique_case_ids": 8,
            "cases_by_split": {split: 4 for split in AUDIT_SPLITS},
            "cases_by_view": {"cs": 4, "sv": 4},
            "topology_roles": {"current_output_topology": 8},
            "exact_selection_coverage": True,
        },
        "coverage": {
            "selected_cases": 8,
            "cases_by_split": {split: 4 for split in AUDIT_SPLITS},
            "component_gradient_measurements": 48,
            "pairwise_cosine_measurements": 120,
            "component_to_joint_cosine_measurements": 48,
            "expected_pairwise_cosines_per_case": 15,
            "expected_component_to_joint_cosines_per_case": 6,
        },
        "calibration": calibration,
        "holdout": _aggregate_summary(),
        "case_results": rows,
        "measurement_integrity": {
            "six_component_order_exact": True,
            "all_15_pairwise_cosines_per_case": True,
            "all_6_component_to_joint_cosines_per_case": True,
            "calibration_holdout_speaker_disjoint": True,
            "topology_exact_selection_coverage": True,
            "slot2_slot3_sources_independent": True,
            "scorer_has_zero_parameters": True,
            "shimmer_db_scientific_status_pending": True,
            "v19_runtime_evidence_does_not_grant_promotion": True,
            "numeric_scientific_gates_applied": False,
            "final_or_fresh_panel_opened": False,
            "generator_optimizer_steps": 0,
        },
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "waveform_mutation_performed": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    report_sha = "d" * 64
    receipt_sha = "e" * 64
    receipt: dict[str, object] = {
        "schema_version": RAW_RECEIPT_SCHEMA_VERSION,
        "decision": RAW_PENDING_DECISION,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "active_components": list(ACTIVE_COMPONENTS),
        "calibration_cases": 4,
        "holdout_cases": 4,
        "source_evidence_sha256": source_hashes,
        "implementation_sha256": _raw_implementation_hashes(),
        "artifact_sha256": {"six_gradient_measurement_report.json": report_sha},
        "source_commit": "a" * 40,
        "source_branch": "feat/avqi-route-c-six-component-scaffold-v1",
        "accepted_base_commit": "f" * 40,
        "launcher_submitted_slurm_job": False,
        "scientific_schema_frozen": False,
        "numeric_scientific_gates_applied": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    return report, receipt, report_sha, receipt_sha


def _evaluate(
    raw_report: dict[str, object], raw_receipt: dict[str, object]
) -> dict[str, object]:
    five_report, five_receipt = _five_evidence()
    return evaluate_six_gradient_decision(
        raw_report,
        raw_receipt,
        five_report,
        five_receipt,
        raw_report_name="six_gradient_measurement_report.json",
        raw_report_sha256="d" * 64,
        raw_receipt_sha256="e" * 64,
        five_report_sha256=FROZEN_FIVE_REPORT_SHA256,
        five_receipt_sha256=FROZEN_FIVE_RECEIPT_SHA256,
        execution_source=_execution_source(),
    )


def test_valid_raw_measurement_passes_only_code_gradient_contract() -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()

    report = _evaluate(raw_report, raw_receipt)

    assert report["schema_version"] == DECISION_SCHEMA_VERSION
    assert report["decision"] == PASS_DECISION
    assert all(report["gates"].values())
    assert report["measurement_summary"]["negative_pairwise_cosine_observations"] > 0
    assert report["measurement_summary"]["pairwise_negative_values_are_diagnostic_only"] is True
    assert report["joint_scientific_promotion_granted"] is False
    assert report["joint_panel_authorized"] is False
    assert report["fresh_panel_opened"] is False
    assert report["generator_optimizer_steps"] == 0
    assert report["authoritative_training_decision"] == TRAINING_NO_GO


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    (
        ("component_min", 1e-10, "all_component_gradients_finite_nonzero_bounded"),
        ("component_max", 10_000.1, "all_component_gradients_finite_nonzero_bounded"),
        ("joint_min", 1e-10, "all_joint_gradients_finite_nonzero_bounded"),
        ("joint_max", 10_000.1, "all_joint_gradients_finite_nonzero_bounded"),
        ("joint_cosine", -0.01, "all_component_to_joint_cosines_nonnegative"),
    ),
)
def test_frozen_numeric_boundaries_emit_no_go(
    field: str, value: float, gate: str
) -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()
    row = raw_report["case_results"][0]
    if field.startswith("component"):
        row["components"]["cpps"]["gradient_norm"] = value
        row["joint"]["weighted_component_gradient_norms"]["cpps"] = value
        total = value + 5.0
        shares = {
            name: (value if name == "cpps" else 1.0) / total
            for name in ACTIVE_COMPONENTS
        }
        row["joint"]["weighted_component_norm_shares"] = shares
        row["joint"]["maximum_component_norm_share"] = max(shares.values())
        row["joint"]["dominant_component"] = max(shares, key=shares.__getitem__)
    elif field.startswith("joint_") and field != "joint_cosine":
        row["joint"]["gradient_norm"] = value
    else:
        item = row["joint"]["component_to_joint_cosines"]["cpps"]
        item["cosine"] = value
        item["negative_direction_observed"] = value < 0.0

    report = _evaluate(raw_report, raw_receipt)

    assert report["decision"] == NO_GO_DECISION
    assert report["gates"][gate] is False
    assert report["joint_panel_authorized"] is False


def test_weighted_share_above_point_eight_emits_no_go() -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()
    row = next(
        value
        for value in raw_report["case_results"]
        if value["split"] == "surrogate_holdout"
    )
    row["components"]["cpps"]["gradient_norm"] = 100.0
    row["joint"]["weighted_component_gradient_norms"]["cpps"] = 100.0
    shares = {
        name: (100.0 if name == "cpps" else 1.0) / 105.0
        for name in ACTIVE_COMPONENTS
    }
    row["joint"]["weighted_component_norm_shares"] = shares
    row["joint"]["maximum_component_norm_share"] = shares["cpps"]
    row["joint"]["dominant_component"] = "cpps"

    report = _evaluate(raw_report, raw_receipt)

    assert report["decision"] == NO_GO_DECISION
    assert report["gates"]["all_weighted_component_shares_le_0_80"] is False


def test_calibration_weight_ratio_is_recomputed_not_trusted() -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()
    raw_report["calibration"]["weighted_median_gradient_norms"]["cpps"] = 2.0

    report = _evaluate(raw_report, raw_receipt)

    assert report["decision"] == NO_GO_DECISION
    assert report["gates"]["calibration_only_inverse_gradient_weights"] is False
    assert report["gates"]["calibration_weighted_median_ratio_le_1_000001"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("raw_decision", "report decision differs"),
        ("raw_receipt_hash", "does not exactly bind"),
        ("component_order", "component order differs"),
        ("speaker_overlap", "selection speaker_overlap differs"),
        ("topology_coverage", "topology coverage differs"),
        ("slot3_loader", "slot 2/slot 3 contract differs"),
        ("scorer_parameters", "integrity differs"),
        ("target_semantics", "target/weight semantics differ"),
        ("opened_panel", "fresh_panel_opened differs"),
        ("missing_panel_boundary", "fresh_panel_opened differs"),
        ("pairwise_missing", "pairwise cosine keys differ"),
        ("source_commit", "source provenance differs"),
        ("report_source_commit", "source provenance differs"),
        ("source_branch", "source provenance differs"),
        ("raw_implementation", "implementation hashes differ from clean source"),
    ),
)
def test_structural_or_provenance_tampering_fails_closed(
    mutation: str, message: str
) -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()
    if mutation == "raw_decision":
        raw_report["decision"] = PASS_DECISION
    elif mutation == "raw_receipt_hash":
        raw_receipt["artifact_sha256"]["six_gradient_measurement_report.json"] = "f" * 64
    elif mutation == "component_order":
        raw_report["contract"]["component_order"] = list(reversed(ACTIVE_COMPONENTS))
    elif mutation == "speaker_overlap":
        raw_report["selection"]["speaker_overlap"] = 1
    elif mutation == "topology_coverage":
        raw_report["topology_coverage"]["observed_cases"] = 7
    elif mutation == "slot3_loader":
        raw_report["slot2_slot3_separation"]["slot3_shimmer_db"][
            "topology_input_loader"
        ] = "candidate_loader"
    elif mutation == "scorer_parameters":
        raw_report["measurement_integrity"]["scorer_has_zero_parameters"] = False
    elif mutation == "target_semantics":
        raw_report["contract"]["avqi_scalar_coefficient_used_for_direction"] = True
    elif mutation == "opened_panel":
        raw_report["fresh_panel_opened"] = True
    elif mutation == "missing_panel_boundary":
        raw_receipt.pop("fresh_panel_opened")
    elif mutation == "pairwise_missing":
        raw_report["case_results"][0]["joint"]["pairwise_component_cosines"].pop(
            PAIRWISE_COMPONENT_KEYS[0]
        )
    elif mutation == "source_commit":
        raw_receipt["source_commit"] = "b" * 40
    elif mutation == "report_source_commit":
        raw_report["contract"]["source"]["head"] = "b" * 40
    elif mutation == "source_branch":
        raw_report["contract"]["source"]["branch"] = "wrong-branch"
    elif mutation == "raw_implementation":
        raw_receipt["implementation_sha256"][RAW_IMPLEMENTATION_KEYS[0]] = "f" * 64

    with pytest.raises(ValueError, match=message):
        _evaluate(raw_report, raw_receipt)


def test_five_precedent_hash_and_opposed_to_joint_are_immutable() -> None:
    raw_report, raw_receipt, _, _ = _raw_evidence()
    five_report, five_receipt = _five_evidence()
    with pytest.raises(ValueError, match="report SHA-256 differs"):
        evaluate_six_gradient_decision(
            raw_report,
            raw_receipt,
            five_report,
            five_receipt,
            raw_report_name="six_gradient_measurement_report.json",
            raw_report_sha256="d" * 64,
            raw_receipt_sha256="e" * 64,
            five_report_sha256="f" * 64,
            five_receipt_sha256=FROZEN_FIVE_RECEIPT_SHA256,
            execution_source=_execution_source(),
        )

    five_report["holdout"]["components"]["cpps"]["opposed_to_joint_cases"] = 1
    with pytest.raises(ValueError, match="opposed-to-joint precedent differs"):
        evaluate_six_gradient_decision(
            raw_report,
            raw_receipt,
            five_report,
            five_receipt,
            raw_report_name="six_gradient_measurement_report.json",
            raw_report_sha256="d" * 64,
            raw_receipt_sha256="e" * 64,
            five_report_sha256=FROZEN_FIVE_REPORT_SHA256,
            five_receipt_sha256=FROZEN_FIVE_RECEIPT_SHA256,
            execution_source=_execution_source(),
        )


def test_requirements_freeze_contract_without_opening_evidence() -> None:
    requirements = decision_requirements()
    contract = requirements["frozen_contract"]

    assert requirements["decision"].startswith("NO_GO_")
    assert contract["frozen_from"]["slurm_job_id"] == "19906556"
    assert contract["frozen_from"]["report_sha256"] == FROZEN_FIVE_REPORT_SHA256
    assert contract["component_gradient_norm"]["strictly_greater_than"] == 1e-10
    assert contract["component_gradient_norm"]["less_than_or_equal_to"] == 1e4
    assert contract["maximum_weighted_component_norm_share_per_case"] == 0.80
    assert contract["pairwise_cosines"]["negative_values"] == "diagnostic_only"
    assert contract["component_to_joint_cosines"]["minimum"] == 0.0
    assert requirements["joint_panel_authorized"] is False


def test_source_validation_binds_accepted_ancestor_and_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name, relative_path in decision.RAW_IMPLEMENTATION_PATHS.items():
        (tmp_path / relative_path).write_text(name, encoding="utf-8")
    git_calls: list[list[str]] = []

    def fake_repository_value(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("branch", "--show-current"):
            return "feat/avqi-route-c-six-component-scaffold-v1"
        raise AssertionError(arguments)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        git_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(decision, "_repository_value", fake_repository_value)
    monkeypatch.setattr(decision.subprocess, "run", fake_run)

    source = decision._validate_source(tmp_path, "a" * 40)

    assert source["accepted_base_commit"] == decision.ACCEPTED_DECISION_BASE_COMMIT
    assert source["raw_implementation_sha256"] == {
        name: decision.sha256_file(tmp_path / relative_path)
        for name, relative_path in decision.RAW_IMPLEMENTATION_PATHS.items()
    }
    assert decision.ACCEPTED_DECISION_BASE_COMMIT in git_calls[0]


def test_source_validation_rejects_missing_accepted_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name, relative_path in decision.RAW_IMPLEMENTATION_PATHS.items():
        (tmp_path / relative_path).write_text(name, encoding="utf-8")
    monkeypatch.setattr(
        decision,
        "_repository_value",
        lambda _root, *arguments: {
            ("rev-parse", "HEAD"): "a" * 40,
            ("status", "--porcelain"): "",
        }.get(arguments, "feat/avqi-route-c-six-component-scaffold-v1"),
    )

    def fail_ancestor(command: list[str], **_: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(decision.subprocess, "run", fail_ancestor)

    with pytest.raises(subprocess.CalledProcessError):
        decision._validate_source(tmp_path, "a" * 40)


def test_post_evaluation_rehash_detects_artifact_tampering(tmp_path: Path) -> None:
    paths = {}
    for name in (
        "raw_report",
        "raw_receipt",
        "five_precedent_report",
        "five_precedent_receipt",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = (path, decision.sha256_file(path))

    verified = decision._post_evaluation_immutability(paths)
    assert verified["verified"] is True

    paths["raw_report"][0].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during evaluation"):
        decision._post_evaluation_immutability(paths)


@pytest.mark.parametrize("mode", ("direct", "module"))
def test_requirements_cli_works_without_caller_pythonpath(mode: str) -> None:
    root = Path(decision.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = (
        [sys.executable, str(Path(decision.__file__).resolve()), "--requirements-only"]
        if mode == "direct"
        else [
            sys.executable,
            "-m",
            "scripts.decide_avqi_route_c_six_component_gradients",
            "--requirements-only",
        ]
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["decision"].startswith("NO_GO_")
    assert report["joint_panel_authorized"] is False


def test_execution_cli_and_launcher_fail_closed_without_reviewed_inputs() -> None:
    root = Path(decision.__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(Path(decision.__file__).resolve())],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["decision"] == NO_GO_DECISION

    launcher = root / "scripts/run_avqi_route_c_six_component_gradient_decision.sh"
    environment = os.environ.copy()
    environment.pop("RUNTIME_PYTHON", None)
    launched = subprocess.run(
        [str(launcher)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert launched.returncode == 2
    assert "RUNTIME_PYTHON" in launched.stderr


def test_decision_layer_has_no_measurement_or_experiment_execution_path() -> None:
    evaluator_source = Path(decision.__file__).read_text(encoding="utf-8")
    launcher_source = (
        Path(decision.__file__)
        .with_name("run_avqi_route_c_six_component_gradient_decision.sh")
        .read_text(encoding="utf-8")
    )

    for forbidden in (
        "import torch",
        "import soundfile",
        "import parselmouth",
        "torch.optim",
        "sbatch",
        "scipy",
    ):
        assert forbidden not in evaluator_source
        assert forbidden not in launcher_source
    assert "scripts.decide_avqi_route_c_six_component_gradients" in launcher_source
    assert "--raw-report-sha256" in launcher_source
    assert "--raw-receipt-sha256" in launcher_source
    assert "scripts.evaluate_avqi_route_c_six_component_gradients" not in launcher_source
