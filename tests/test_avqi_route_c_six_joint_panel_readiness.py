from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from model.avqi_route_c import (
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_SCIENTIFIC_STATUS,
    route_c_six_registry_records,
)
from scripts import audit_avqi_route_c_six_joint_panel_readiness as readiness
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    CLEAN_PATHOLOGICAL_ROLE,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TOTAL_SPEAKERS,
    FIVE_COMPONENT_EVIDENCE_KEYS,
    FROZEN_PANEL_DATA_REQUIREMENTS,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    FROZEN_SPLIT_SEAL_SCHEMA_VERSION,
    GAP_NOISE_MANIFEST_SHA256,
    GAP_RECIPE_ASSIGNMENT_SALT,
    GAP_RIR_MANIFEST_SHA256,
    GAP_SIMULATION_INVENTORY_SHA256,
    HEALTHY_ROLE,
    MISSING_CODE_STAGES,
    NORMALIZATION_SOURCE,
    NORMALIZATION_TARGET_MEAN_FIELD,
    NORMALIZATION_TARGET_SCALE_FIELD,
    PANEL_ROW_FIELDS,
    PATHOLOGICAL_ROLE,
    READINESS_SCHEMA_VERSION,
    REQUIRED_ARTIFACT_KEYS,
    REQUIRED_CONDITIONS,
    REQUIRED_EFFICACY_SLICES,
    REQUIRED_SPLITS,
    REQUIRED_VIEWS,
    SHIMMER_DB_REQUIRED_STATUS,
    SIX_GRADIENT_FROZEN_GATE_KEYS,
    SIX_GRADIENT_RECEIPT_SCHEMA_VERSION,
    SIX_GRADIENT_SCHEMA_VERSION,
    SIX_GRADIENT_PASS_DECISION,
    SIX_GRADIENT_SOURCE_EVIDENCE_KEYS,
    SOURCE_DATASET,
    SVD_CS_METADATA_SHA256,
    SVD_SPEAKER_SELECTION_SALT,
    SVD_SV_METADATA_SHA256,
    UNBOUND_EXECUTION_INPUTS,
    UNFROZEN_SCIENTIFIC_CONTRACTS,
    _validate_five_component_evidence,
    _validate_panel_rows,
    _validate_six_gradient,
    _validate_split_seal,
    current_blockers,
    frozen_svd_speaker_rank,
    readiness_requirements,
    sha256_file,
    validate_readiness_manifest,
)
from scripts.decide_avqi_route_c_six_component_gradients import (
    FROZEN_FIVE_JOB_ID,
    FROZEN_FIVE_RECEIPT_SHA256,
    FROZEN_FIVE_REPORT_SHA256,
    JOINT_PANEL_NO_GO,
    PASS_DECISION,
    RAW_PENDING_DECISION,
    TRAINING_NO_GO,
    decision_requirements as six_gradient_decision_requirements,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _bind(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _promoted_registry() -> list[dict[str, object]]:
    records = route_c_six_registry_records()
    records[3]["scientific_status"] = SHIMMER_DB_REQUIRED_STATUS
    records[3]["scientific_promotion_granted"] = True
    return records


def _readiness_manifest(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "source_commit": "a" * 40,
        "generator_optimizer_steps": 0,
    }
    value.update(overrides)
    return value


def _panel_rows() -> list[dict[str, str]]:
    rows = []
    for split in REQUIRED_SPLITS:
        for label in ("patient", "healthy"):
            for speaker_index in range(3):
                speaker = f"{split}-{label}-speaker-{speaker_index}"
                for condition in REQUIRED_CONDITIONS:
                    for view in REQUIRED_VIEWS:
                        if label == "healthy":
                            role = HEALTHY_ROLE
                        elif condition == "clean":
                            role = CLEAN_PATHOLOGICAL_ROLE
                        else:
                            role = PATHOLOGICAL_ROLE
                        rows.append(
                            {
                                "case_id": f"{speaker}-{condition}-{view}",
                                "dataset": SOURCE_DATASET,
                                "speaker_id": speaker,
                                "split": split,
                                "view": view,
                                "condition": condition,
                                "label": label,
                                "optimization_role": role,
                                "source_waveform_sha256": "a" * 64,
                            }
                        )
    return rows


def _split_seal(
    *,
    result_blind: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": FROZEN_SPLIT_SEAL_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "source_dataset": SOURCE_DATASET,
        "sv_metadata_sha256": SVD_SV_METADATA_SHA256,
        "cs_metadata_sha256": SVD_CS_METADATA_SHA256,
        "health_status_mapping": {"1": "patient", "0": "healthy"},
        "paired_cs_sv_same_session_required": True,
        "minimum_raw_mono_duration_seconds": {"sv": 1.0, "cs": 3.0},
        "eligible_session_per_speaker": "minimum_numeric_session_id",
        "speaker_selection_salt": SVD_SPEAKER_SELECTION_SALT,
        "speaker_rank_digest": "SHA256(salt:speaker_id:session_id)",
        "prior_ledger_excluded_before_hash_ranking": True,
        "diagnosis_used_for_selection": False,
        "exact_scores_opened": False,
        "speaker_split_before_simulation": True,
        "gap_simulation_inventory_sha256": GAP_SIMULATION_INVENTORY_SHA256,
        "gap_rir_manifest_sha256": GAP_RIR_MANIFEST_SHA256,
        "gap_noise_manifest_sha256": GAP_NOISE_MANIFEST_SHA256,
        "recipe_assignment_salt": GAP_RECIPE_ASSIGNMENT_SALT,
        "recipe_uid_unique_per_row": True,
        "recipe_uid_reused_across_splits": False,
        "condition_recipe_semantics": {
            "clean": "no_simulation",
            "rir_only": "rir_only",
            "snr20": "rir_plus_noise_fixed_target_snr_20db",
            "snr10": "rir_plus_noise_fixed_target_snr_10db",
        },
        "metadata_only_result_blind_selection": result_blind,
        "mild_severe_labels_created": False,
        "prior_panel_speaker_overlap": 0,
        "waveform_steps": 1,
        "one_global_alpha": True,
        "gradient_normalization": "waveform_rms_normalized",
        "alpha_grid": [
            0.0,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
        ],
        "zero_alpha_selectable": False,
        "alpha_required_gate_families": [
            "all_six_components",
            "equal_weight_joint",
            "all_required_efficacy_slices",
            "waveform_safety",
            "full_band_pathology",
            "denoising",
        ],
        "alpha_required_gate_split": "calibration",
        "alpha_selection_objective": (
            "maximize_equal_weight_joint_exact_median_normalized_gap_reduction"
        ),
        "alpha_selection_tie_break": "smaller_alpha",
        "alpha_selection_split": "calibration",
        "final_tuning_permitted": False,
        "optimization_weight_source_decision": SIX_GRADIENT_PASS_DECISION,
        "optimization_weights_calibration_only": True,
        "optimization_weights_used_for_exact_joint_decision": False,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization_target_mean_field": NORMALIZATION_TARGET_MEAN_FIELD,
        "normalization_target_scale_field": NORMALIZATION_TARGET_SCALE_FIELD,
        "normalization_refit_permitted": False,
        "healthy_no_step_does_not_establish_optimized_healthy_safety": True,
        "joint_gate_contract_sha256": "a" * 64,
        "target_value_protocol_sha256": "b" * 64,
        "prior_panel_speaker_ledger_sha256": "c" * 64,
        "fresh_speaker_source_manifest_sha256": "d" * 64,
        "rows": [
            {field: row[field] for field in PANEL_ROW_FIELDS}
            for row in _panel_rows()
        ],
        "generator_optimizer_steps": 0,
    }


def _five_evidence_bundle(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    paths: dict[str, Path] = {}
    contracts = {
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
    paths["slope_final_panel_seal"] = _write_json(
        tmp_path / "final_panel_seal.json", {"sealed": True}
    )
    paths["slope_final_results"] = tmp_path / "final_results.csv"
    paths["slope_final_results"].write_text("case_id,decision\n", encoding="utf-8")
    for component, (report_key, receipt_key, decision) in contracts.items():
        if component in {"hnr", "shimmer_percent", "slope"}:
            report = {
                "decision": decision,
                "final_exact_panel_opened": True,
                "final": {"decision": "PASS", "gates": {"frozen": True}},
                "formal_pathology_training_submitted": False,
                "generator_optimizer_steps": 0,
            }
        else:
            report = {
                "schema_version": (
                    "direct-avqi-waveform-optimization-v3"
                    if component == "cpps"
                    else "direct-avqi-waveform-optimization-v1"
                ),
                "decision": decision,
                "summary": {
                    "component_gates": {
                        component: {
                            "decision": "PASS",
                            "gates": {"frozen": True},
                        }
                    },
                    "safety": {"decision": "PASS"},
                },
                "formal_pathology_training_submitted": False,
                "generator_optimizer_steps": 0,
            }
        paths[report_key] = _write_json(tmp_path / f"{report_key}.json", report)
        receipt_hashes = {
            paths[report_key].name: sha256_file(paths[report_key])
        }
        if component == "slope":
            receipt_hashes.update(
                {
                    paths["slope_final_panel_seal"].name: sha256_file(
                        paths["slope_final_panel_seal"]
                    ),
                    paths["slope_final_results"].name: sha256_file(
                        paths["slope_final_results"]
                    ),
                }
            )
        paths[receipt_key] = _write_json(
            tmp_path / f"{receipt_key}.json",
            {
                "decision": decision,
                "artifact_sha256": receipt_hashes,
                "generator_optimizer_steps": 0,
            },
        )

    artifacts = {key: _bind(paths[key]) for key in FIVE_COMPONENT_EVIDENCE_KEYS}
    five_report = {
        "schema_version": "avqi_route_c_five_component_gradient_audit_v1",
        "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
        "source_evidence": {
            key: artifacts[key] for key in FIVE_COMPONENT_EVIDENCE_KEYS
        },
        "gates": {"frozen": True},
        "generator_optimizer_steps": 0,
    }
    paths["five_gradient_report"] = _write_json(
        tmp_path / "gradient_interference_report.json", five_report
    )
    paths["five_gradient_receipt"] = _write_json(
        tmp_path / "five_completion_receipt.json",
        {
            "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
            "active_components": [
                "cpps",
                "hnr",
                "shimmer_percent",
                "slope",
                "tilt",
            ],
            "inactive_slots": ["shimmer_db"],
            "artifact_sha256": {
                paths["five_gradient_report"].name: sha256_file(
                    paths["five_gradient_report"]
                )
            },
            "generator_optimizer_steps": 0,
        },
    )
    artifacts.update(
        {
            key: _bind(paths[key])
            for key in ("five_gradient_report", "five_gradient_receipt")
        }
    )
    return artifacts, paths


def _six_gradient_evidence() -> tuple[dict[str, object], dict[str, object], str]:
    source_evidence = {
        key: "a" * 64 for key in SIX_GRADIENT_SOURCE_EVIDENCE_KEYS
    }
    report: dict[str, object] = {
        "schema_version": SIX_GRADIENT_SCHEMA_VERSION,
        "decision": SIX_GRADIENT_PASS_DECISION,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "active_components": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "source_evidence_sha256": source_evidence,
        "frozen_contract": six_gradient_decision_requirements()[
            "frozen_contract"
        ],
        "accepted_numeric_precedent": {
            "slurm_job_id": FROZEN_FIVE_JOB_ID,
            "report_sha256": FROZEN_FIVE_REPORT_SHA256,
            "receipt_sha256": FROZEN_FIVE_RECEIPT_SHA256,
        },
        "raw_measurement_evidence": {
            "report_sha256": "c" * 64,
            "receipt_sha256": "d" * 64,
            "raw_decision": RAW_PENDING_DECISION,
            "raw_artifacts_rewritten": False,
        },
        "decision_source": {
            "head": "e" * 40,
            "branch": "feat/avqi-route-c-six-component-scaffold-v1",
        },
        "implementation_sha256": {
            "decide_avqi_route_c_six_component_gradients.py": "f" * 64,
            "run_avqi_route_c_six_component_gradient_decision.sh": "1" * 64,
        },
        "post_evaluation_immutability": {
            "verified": True,
            "artifact_sha256": {
                "raw_report": "c" * 64,
                "raw_receipt": "d" * 64,
                "five_precedent_report": FROZEN_FIVE_REPORT_SHA256,
                "five_precedent_receipt": FROZEN_FIVE_RECEIPT_SHA256,
            },
        },
        "measurement_summary": {
            "calibration_cases": 4,
            "holdout_cases": 4,
            "calibration_inverse_gradient_weights": {
                name: 1.0 for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
            },
            "calibration_weighted_median_norm_ratio": 1.0,
            "maximum_weighted_component_norm_share": 0.5,
            "minimum_component_to_joint_cosine": 0.0,
            "pairwise_negative_values_are_diagnostic_only": True,
        },
        "gates": {key: True for key in SIX_GRADIENT_FROZEN_GATE_KEYS},
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
    report_sha256 = "b" * 64
    receipt = {
        "schema_version": SIX_GRADIENT_RECEIPT_SCHEMA_VERSION,
        "decision": SIX_GRADIENT_PASS_DECISION,
        "joint_panel_decision": JOINT_PANEL_NO_GO,
        "active_components": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "raw_measurement_sha256": {
            "report": "c" * 64,
            "receipt": "d" * 64,
        },
        "accepted_numeric_precedent": report["accepted_numeric_precedent"],
        "source_commit": "e" * 40,
        "source_branch": "feat/avqi-route-c-six-component-scaffold-v1",
        "implementation_sha256": report["implementation_sha256"],
        "post_evaluation_immutability": report[
            "post_evaluation_immutability"
        ],
        "artifact_sha256": {"gradient_report.json": report_sha256},
        "raw_artifacts_rewritten": False,
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
    return report, receipt, report_sha256


def test_current_requirements_are_explicitly_no_go() -> None:
    requirements = readiness_requirements()

    assert requirements["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert requirements["execution_authorized"] is False
    assert requirements["current_shimmer_db_scientific_status"] == (
        ROUTE_C_SIX_SCIENTIFIC_STATUS
    )
    assert requirements["required_conditions"] == list(REQUIRED_CONDITIONS)
    assert "clean" in requirements["required_conditions"]
    assert requirements["missing_code_stages"] == list(MISSING_CODE_STAGES)
    assert requirements["scientific_contract_frozen"] is True
    assert requirements["unfrozen_scientific_contracts"] == []
    assert UNFROZEN_SCIENTIFIC_CONTRACTS == ()
    assert requirements["unbound_execution_inputs"] == list(
        UNBOUND_EXECUTION_INPUTS
    )
    assert requirements["actual_manifests_bound"] is False
    assert requirements["joint_scientific_promotion_granted"] is False
    assert requirements["joint_panel_authorized"] is False
    assert not any(
        "gradient decision evaluator" in item
        for item in requirements["missing_code_stages"]
    )
    assert requirements["frozen_panel_data_requirements"] == list(
        FROZEN_PANEL_DATA_REQUIREMENTS
    )
    matrix = {
        row["requirement"]: row for row in requirements["source_requirement_matrix"]
    }
    assert matrix["fresh-panel source/split/target schemas"]["status"] == (
        "frozen_contract_actual_manifests_required"
    )
    assert matrix["joint waveform/exact gate thresholds"]["status"] == (
        "frozen_contract_evaluator_runner_present"
    )
    assert matrix["two-stage sealed joint waveform evaluator/runner"][
        "status"
    ] == "present_fail_closed_hash_bound_runners"
    assert matrix[
        "GAP simulation source and recipe assignment contract"
    ]["status"] == (
        "frozen_contract_actual_manifests_and_assignment_required"
    )
    assert set(FIVE_COMPONENT_EVIDENCE_KEYS) <= set(REQUIRED_ARTIFACT_KEYS)
    assert {
        "svd_sv_metadata",
        "svd_cs_metadata",
        "gap_simulation_inventory",
        "gap_v1_arni_rir_manifest",
        "gap_v1_dns5_noise_manifest",
        "joint_recipe_assignment_manifest",
        "joint_gradient_manifest",
    } <= set(
        requirements["required_artifacts"]
    )
    assert (
        current_blockers()[0]
        == "Shimmer dB scientific promotion remains pending"
    )
    assert not any(
        "schemas remain unfrozen" in blocker for blocker in current_blockers()
    )


def test_frozen_scientific_contract_matches_preregistered_values() -> None:
    contract = readiness_requirements()["frozen_scientific_contract"]

    assert contract["schema_version"] == FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
    assert contract["frozen_before_source_selection"] is True
    source = contract["source"]
    assert source["dataset"] == SOURCE_DATASET
    assert source["metadata_sha256"] == {
        "sv": SVD_SV_METADATA_SHA256,
        "cs": SVD_CS_METADATA_SHA256,
    }
    assert source["selection_mode"] == "metadata_only_result_blind"
    assert source["health_status_mapping"] == {"1": "patient", "0": "healthy"}
    assert source["paired_cs_sv_same_session_required"] is True
    assert source["minimum_raw_mono_duration_seconds"] == {
        "sv": 1.0,
        "cs": 3.0,
    }
    assert source["eligible_session_per_speaker"] == "minimum_numeric_session_id"
    assert source["speaker_selection_salt"] == (
        "avqi-route-c-six-joint-svd-v1-20260826"
    )
    assert source["selection_operation_order"] == [
        "map_health_status",
        "pair_cs_sv_by_same_session",
        "filter_raw_mono_minimum_duration",
        "retain_minimum_numeric_eligible_session_per_speaker",
        "exclude_prior_ledger_speakers",
        "bucket_by_health_status_and_gender",
        "rank_by_salted_sha256",
        "allocate_calibration_then_final_by_frozen_gender_quota",
    ]
    assert source["ranking"] == {
        "bucket_fields": ["health_status", "gender"],
        "digest": "SHA256(salt:speaker_id:session_id)",
        "order": "ascending_hex_digest",
        "collision_tie_break": ["speaker_id", "session_id"],
        "split_allocation_order": ["calibration", "final"],
        "quota_source": "source.gender_allocation",
    }
    assert source["prior_ledger_exclusion_stage"] == "before_hash_ranking"
    assert source["prior_panel_speaker_overlap"] == 0
    assert source["mild_severe_labels_created"] is False
    assert source["record_only_fields"] == ["diagnosis"]
    assert {
        "diagnosis",
        "avqi",
        "exact_avqi",
        "component_values",
        "exact_component_values",
        "mild_severe",
    } <= set(
        source["forbidden_selection_inputs"]
    )
    assert source["selection_performed_at_contract_freeze"] is False
    assert source["gender_allocation"] == {
        "calibration": {
            "patient": {"female": 2, "male": 1},
            "healthy": {"female": 1, "male": 2},
        },
        "final": {
            "patient": {"female": 1, "male": 2},
            "healthy": {"female": 2, "male": 1},
        },
    }

    panel = contract["panel"]
    assert panel["total_speakers"] == 12
    assert panel["total_rows"] == 96
    assert panel["rows_per_split"] == 48
    assert panel["patient_degraded_efficacy_rows_per_split"] == 18
    assert panel["patient_clean_control_rows_per_split"] == 6
    assert panel["healthy_guardrail_rows_per_split"] == 24

    simulation = contract["simulation"]
    assert simulation["source_inventory_sha256"] == (
        "859a9e058f4f44c8e15d4b37d992cefa4d1501d1127a374d7e8cb1403c020384"
    )
    assert simulation["rir_source"] == {
        "name": "v1_arni_rir",
        "manifest_sha256": (
            "2bac3a563292a5a0a1377e3e98d29b6cfb8808d81f2e53ec1cbbafb08642d9da"
        ),
    }
    assert simulation["noise_source"] == {
        "name": "v1_dns5_noise",
        "manifest_sha256": (
            "c6f9441cdd76f50b4eb7f4fa5b83b994a509d3a925d7ae9b887059af31794d65"
        ),
    }
    assert simulation["speaker_split_before_recipe_assignment"] is True
    assert simulation["recipe_assignment_salt"] == (
        "avqi-route-c-six-joint-recipes-v1-20260826"
    )
    assert simulation["recipe_uid_required_for_every_row"] is True
    assert simulation["recipe_uid_unique_per_row"] is True
    assert simulation["recipe_uid_reused_across_splits"] is False
    assert simulation["condition_recipes"] == {
        "clean": {"rir": False, "noise": False, "target_snr_db": None},
        "rir_only": {"rir": True, "noise": False, "target_snr_db": None},
        "snr20": {"rir": True, "noise": True, "target_snr_db": 20.0},
        "snr10": {"rir": True, "noise": True, "target_snr_db": 10.0},
    }
    assert simulation["actual_recipes_selected_at_contract_freeze"] is False

    opening = contract["two_stage_opening"]
    assert opening == {
        "calibration_selection_sealed_before_final_open": True,
        "calibration_and_final_speakers_disjoint": True,
        "calibration_may_select": ["one_global_alpha"],
        "final_may_select_or_tune": [],
        "alpha_selection_receipt_sealed_before_final_exact_open": True,
        "final_waveforms_sealed_before_final_exact_open": True,
        "final_exact_outcomes_opened_at_contract_freeze": False,
    }

    step = contract["waveform_step"]
    assert step == {
        "steps": 1,
        "global_alpha": True,
        "gradient_normalization": "waveform_rms_normalized",
        "alpha_grid": [
            0.0,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
        ],
        "zero_alpha_role": "negative_control_only",
        "zero_alpha_selectable": False,
        "nonzero_alpha_required_gate_families": [
            "all_six_components",
            "equal_weight_joint",
            "all_required_efficacy_slices",
            "waveform_safety",
            "full_band_pathology",
            "denoising",
        ],
        "nonzero_alpha_gate_split": "calibration",
        "selection_objective": (
            "maximize_equal_weight_joint_exact_median_normalized_gap_reduction"
        ),
        "selection_tie_break": "smaller_alpha",
        "selection_split": "calibration",
        "final_tuning_permitted": False,
    }

    assert contract["optimization_weights"] == {
        "only_allowed_source": "passed_six_gradient_report",
        "source_report_decision": SIX_GRADIENT_PASS_DECISION,
        "source_report_and_receipt_hash_bound": True,
        "source_field": (
            "measurement_summary.calibration_inverse_gradient_weights"
        ),
        "calibration_only": True,
        "used_for": "waveform_gradient_generation_only",
        "used_for_exact_joint_decision": False,
        "held_fixed_for_final": True,
        "final_reestimation_permitted": False,
    }

    target = contract["target_and_aggregate"]
    assert target["component_order"] == list(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    assert target["patient_degraded_enters_efficacy_denominator"] is True
    assert target["patient_clean_role"] == "no-overprocessing control"
    assert target["patient_clean_enters_degraded_efficacy_denominator"] is False
    assert target["healthy_target"] is None
    assert target["healthy_loss_enabled"] is False
    assert target["healthy_waveform_step_enabled"] is False
    assert target["healthy_enters_degraded_efficacy_denominator"] is False
    assert target["healthy_control_role"] == (
        "routing/source/topology/coverage control only"
    )
    assert target["optimized_healthy_safety_claimed"] is False
    assert "equal-weight" in target["joint_aggregate"]
    assert target["avqi_scalar_coefficient_used_for_direction"] is False
    assert target["avqi_scalar_coefficient_used_for_aggregate"] is False
    assert target["emitted_waveform_highpass"] is False

    gates = contract["efficacy_gates"]
    assert gates["material_normalized_before_gap"] == {
        "comparison": ">",
        "value": 0.02,
    }
    assert gates["material_coverage_fraction"] == {
        "comparison": ">=",
        "value": 0.80,
    }
    assert gates["material_cases_absolute"]["value"] == 5
    assert gates["material_cases_per_18"]["value"] == 15
    assert gates["exact_improvement_fraction"]["value"] == 0.80
    assert gates["median_normalized_gap_reduction"]["value"] == 0.02
    assert gates["applies_to_each_component_and_joint"] is True

    slices = contract["required_efficacy_slices"]
    assert slices["keys"] == list(REQUIRED_EFFICACY_SLICES)
    assert len(slices["keys"]) == 6
    assert slices["expected_rows_per_slice"] == 3
    assert slices["zero_coverage_decision"] == "FAIL"
    assert slices["material_case_present"] is True
    assert slices["applies_to_each_component_and_joint"] is True
    assert slices["improvement_fraction"] == {
        "comparison": ">",
        "value": 0.50,
    }
    assert slices["median_normalized_gap_reduction"]["value"] == 0.0

    assert contract["safety_gates"] == {
        "residual_rms_db": {"comparison": "<=", "value": -50.0},
        "cosine_similarity": {"comparison": ">=", "value": 0.99999},
        "clip_fraction": {"comparison": "=", "value": 0.0},
    }
    full_band = contract["full_band_pathology_denoising_gates"]
    assert full_band["scope"] == {
        "patient_degraded": "efficacy and guardrails",
        "patient_clean": "no-overprocessing control only",
        "healthy": "routing/source/topology/coverage control only",
    }
    assert full_band["reference_by_role"] == {
        "patient_degraded": "same-speaker clean pathological CS or SV waveform",
        "patient_clean": "same-speaker clean pathological CS or SV waveform",
        "healthy": None,
    }
    assert full_band["healthy_candidate_contract"] == (
        "candidate_sha256_equals_base_sha256"
    )
    assert full_band["healthy_pathological_reference_applied"] is False
    assert "no shift, filter, resample" in full_band["alignment"]
    assert full_band["low_frequency_bands_hz"] == [
        [20.0, 80.0],
        [80.0, 300.0],
    ]
    assert full_band["pathology_db_median_gap_increase_max"] == 0.5
    assert full_band["pathology_db_worst_gap_increase_max"] == 1.5
    assert full_band["airflow_flatness_median_gap_increase_max"] == 0.05
    assert full_band["airflow_flatness_worst_gap_increase_max"] == 0.10
    assert full_band["pause_f1_median_decrease_max"] == 0.05
    assert full_band["pause_f1_worst_decrease_max"] == 0.15
    assert full_band["guardrail_pass_fraction_min"] == pytest.approx(2.0 / 3.0)
    assert full_band["denoising_median_change_min_db"] == -0.10
    assert full_band["denoising_worst_change_min_db"] == -0.50
    assert full_band["denoising_metrics"] == ["snr", "si_sdr"]
    assert "not clinical airflow labels" in full_band["airflow_proxy_limit"]
    assert contract["boundaries"] == {
        "generator_optimizer_steps": 0,
        "joint_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "healthy_no_step_does_not_establish_optimized_healthy_safety": True,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def test_frozen_svd_rank_uses_exact_salt_speaker_and_numeric_session() -> None:
    assert SVD_SPEAKER_SELECTION_SALT == (
        "avqi-route-c-six-joint-svd-v1-20260826"
    )
    assert frozen_svd_speaker_rank("101", "7") == (
        "5c9eb8c717cd639c42608e775d69e44560524743685b1f5d2c8b346599845dd1"
    )

    with pytest.raises(ValueError, match="numeric session ID"):
        frozen_svd_speaker_rank("101", "session-seven")


def test_frozen_gap_hashes_and_recipe_salt_are_exact() -> None:
    assert GAP_SIMULATION_INVENTORY_SHA256 == (
        "859a9e058f4f44c8e15d4b37d992cefa4d1501d1127a374d7e8cb1403c020384"
    )
    assert GAP_RIR_MANIFEST_SHA256 == (
        "2bac3a563292a5a0a1377e3e98d29b6cfb8808d81f2e53ec1cbbafb08642d9da"
    )
    assert GAP_NOISE_MANIFEST_SHA256 == (
        "c6f9441cdd76f50b4eb7f4fa5b83b994a509d3a925d7ae9b887059af31794d65"
    )
    assert GAP_RECIPE_ASSIGNMENT_SALT == (
        "avqi-route-c-six-joint-recipes-v1-20260826"
    )


def test_alpha_selection_and_healthy_control_boundaries_are_non_overclaiming() -> None:
    contract = readiness_requirements()["frozen_scientific_contract"]
    step = contract["waveform_step"]
    healthy = contract["target_and_aggregate"]
    full_band = contract["full_band_pathology_denoising_gates"]

    assert step["zero_alpha_role"] == "negative_control_only"
    assert step["zero_alpha_selectable"] is False
    assert step["selection_tie_break"] == "smaller_alpha"
    assert step["selection_objective"].startswith("maximize_equal_weight_joint")
    assert healthy["healthy_waveform_step_enabled"] is False
    assert healthy["optimized_healthy_safety_claimed"] is False
    assert full_band["reference_by_role"]["healthy"] is None
    assert full_band["healthy_candidate_contract"] == (
        "candidate_sha256_equals_base_sha256"
    )
    assert contract["boundaries"][
        "healthy_no_step_does_not_establish_optimized_healthy_safety"
    ] is True


def test_normalization_is_bound_to_the_passed_six_gradient_raw_report() -> None:
    contract = readiness_requirements()["frozen_scientific_contract"]
    normalization = contract["normalization"]

    assert normalization == {
        "only_allowed_source": NORMALIZATION_SOURCE,
        "source_bound_by_passed_six_gradient_decision": True,
        "target_mean_field": NORMALIZATION_TARGET_MEAN_FIELD,
        "target_scale_field": NORMALIZATION_TARGET_SCALE_FIELD,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "target_scales_finite_and_positive": True,
        "joint_panel_refit_permitted": False,
        "joint_panel_rows_used_to_fit_mean_or_scale": False,
    }


def test_existing_cpps_evidence_is_unbound_not_missing() -> None:
    blockers = current_blockers()

    assert "not yet bound into a six-joint manifest: cpps_report" in blockers
    assert not any(
        "missing" in blocker and "cpps_report" in blocker for blocker in blockers
    )


def test_pending_registry_closes_manifest_before_artifacts() -> None:
    with pytest.raises(ValueError, match="scientific promotion is still pending"):
        validate_readiness_manifest(_readiness_manifest())


def test_promoted_registry_still_closes_unbound_execution_inputs() -> None:
    with pytest.raises(ValueError, match="execution remains closed"):
        validate_readiness_manifest(
            _readiness_manifest(),
            registry_records=_promoted_registry(),
        )


def test_opened_exact_outcomes_fail_before_registry_or_artifacts() -> None:
    with pytest.raises(ValueError, match="opened candidate outcomes"):
        validate_readiness_manifest(
            _readiness_manifest(candidate_exact_outcomes_opened=True)
        )


def test_complete_svd_twelve_speaker_ninety_six_row_matrix_passes() -> None:
    rows_by_case, speakers = _validate_panel_rows(_panel_rows(), "test panel")

    assert len(rows_by_case) == EXPECTED_TOTAL_ROWS == 96
    assert sum(len(values) for values in speakers.values()) == (
        EXPECTED_TOTAL_SPEAKERS
    )
    assert set(speakers) == set(REQUIRED_SPLITS)
    assert all(len(values) == 6 for values in speakers.values())


def test_missing_panel_row_fails_exact_count_contract() -> None:
    rows = _panel_rows()
    rows.pop(0)

    with pytest.raises(ValueError, match="row count differs"):
        _validate_panel_rows(rows, "test panel")


def test_missing_panel_speaker_fails_exact_count_contract() -> None:
    rows = _panel_rows()
    missing = rows[0]["speaker_id"]
    rows = [row for row in rows if row["speaker_id"] != missing]

    with pytest.raises(
        ValueError,
        match="row count differs|speaker count differs",
    ):
        _validate_panel_rows(rows, "test panel")


def test_wrong_split_label_count_fails() -> None:
    rows = _panel_rows()
    speaker = next(
        row["speaker_id"]
        for row in rows
        if row["split"] == "calibration" and row["label"] == "healthy"
    )
    for row in rows:
        if row["speaker_id"] == speaker:
            row["label"] = "patient"
            row["optimization_role"] = (
                CLEAN_PATHOLOGICAL_ROLE
                if row["condition"] == "clean"
                else PATHOLOGICAL_ROLE
            )

    with pytest.raises(ValueError, match="label counts differ"):
        _validate_panel_rows(rows, "test panel")


def test_result_derived_severity_field_is_rejected() -> None:
    rows = _panel_rows()
    rows[0]["severity"] = "mild"

    with pytest.raises(ValueError, match="forbidden result-derived"):
        _validate_panel_rows(rows, "test panel")


def test_healthy_rows_cannot_become_optimization_targets() -> None:
    rows = _panel_rows()
    healthy = next(row for row in rows if row["label"] == "healthy")
    healthy["optimization_role"] = PATHOLOGICAL_ROLE

    with pytest.raises(ValueError, match="row semantics differ"):
        _validate_panel_rows(rows, "test panel")


def test_clean_patient_rows_must_use_no_overprocessing_role() -> None:
    rows = _panel_rows()
    clean = next(
        row
        for row in rows
        if row["label"] == "patient" and row["condition"] == "clean"
    )
    clean["optimization_role"] = PATHOLOGICAL_ROLE

    with pytest.raises(ValueError, match="row semantics differ"):
        _validate_panel_rows(rows, "test panel")


def test_non_svd_panel_row_is_rejected() -> None:
    rows = _panel_rows()
    rows[0]["dataset"] = "TAU"

    with pytest.raises(ValueError, match="row semantics differ"):
        _validate_panel_rows(rows, "test panel")


def test_split_seal_requires_complete_stratified_speaker_matrix() -> None:
    seal = _split_seal()

    speakers = _validate_split_seal(
        seal,
        gate_sha256="a" * 64,
        target_sha256="b" * 64,
        ledger_sha256="c" * 64,
        source_sha256="d" * 64,
    )
    assert set(speakers) == set(REQUIRED_SPLITS)

    frozen_alpha_grid = seal["alpha_grid"]
    seal["alpha_grid"] = frozen_alpha_grid[:-1]
    with pytest.raises(ValueError, match="split seal semantics differ"):
        _validate_split_seal(
            seal,
            gate_sha256="a" * 64,
            target_sha256="b" * 64,
            ledger_sha256="c" * 64,
            source_sha256="d" * 64,
        )
    seal["alpha_grid"] = frozen_alpha_grid

    seal["rows"][0]["speaker_id"] = "wrong-speaker"
    with pytest.raises(
        ValueError,
        match="speaker count differs|speaker matrix differs",
    ):
        _validate_split_seal(
            seal,
            gate_sha256="a" * 64,
            target_sha256="b" * 64,
            ledger_sha256="c" * 64,
            source_sha256="d" * 64,
        )


def test_split_seal_rejects_result_aware_selection() -> None:
    seal = _split_seal(result_blind=False)

    with pytest.raises(ValueError, match="split seal semantics differ"):
        _validate_split_seal(
            seal,
            gate_sha256="a" * 64,
            target_sha256="b" * 64,
            ledger_sha256="c" * 64,
            source_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    ("key", "wrong_value"),
    [
        ("speaker_selection_salt", "wrong-speaker-salt"),
        ("gap_simulation_inventory_sha256", "e" * 64),
        ("gap_rir_manifest_sha256", "e" * 64),
        ("gap_noise_manifest_sha256", "e" * 64),
        ("recipe_assignment_salt", "wrong-recipe-salt"),
        ("alpha_selection_tie_break", "larger_alpha"),
        (
            "healthy_no_step_does_not_establish_optimized_healthy_safety",
            False,
        ),
    ],
)
def test_split_seal_rejects_frozen_contract_drift(
    key: str,
    wrong_value: object,
) -> None:
    seal = _split_seal()
    seal[key] = wrong_value

    with pytest.raises(ValueError, match="split seal semantics differ"):
        _validate_split_seal(
            seal,
            gate_sha256="a" * 64,
            target_sha256="b" * 64,
            ledger_sha256="c" * 64,
            source_sha256="d" * 64,
        )


def test_five_component_frozen_evidence_set_accepts_tilt_component_pass(
    tmp_path: Path,
) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)

    evidence = _validate_five_component_evidence(artifacts, paths)

    assert set(evidence) == set(FIVE_COMPONENT_EVIDENCE_KEYS)
    tilt = json.loads(paths["tilt_report"].read_text(encoding="utf-8"))
    assert tilt["decision"] == "FAIL_WAVEFORM_OPTIMIZATION"
    assert tilt["summary"]["component_gates"]["tilt"]["decision"] == "PASS"


def test_tilt_container_fail_does_not_mask_component_gate_failure(
    tmp_path: Path,
) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)
    report = json.loads(paths["tilt_report"].read_text(encoding="utf-8"))
    report["summary"]["component_gates"]["tilt"]["gates"]["frozen"] = False
    _write_json(paths["tilt_report"], report)
    receipt = json.loads(paths["tilt_receipt"].read_text(encoding="utf-8"))
    receipt["artifact_sha256"][paths["tilt_report"].name] = sha256_file(
        paths["tilt_report"]
    )
    _write_json(paths["tilt_receipt"], receipt)
    artifacts["tilt_report"] = _bind(paths["tilt_report"])
    artifacts["tilt_receipt"] = _bind(paths["tilt_receipt"])

    with pytest.raises(ValueError, match="contains a failed gate"):
        _validate_five_component_evidence(artifacts, paths)


def test_slope_sealed_results_must_be_bound_by_receipt(tmp_path: Path) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)
    receipt = json.loads(paths["slope_receipt"].read_text(encoding="utf-8"))
    receipt["artifact_sha256"].pop(paths["slope_final_results"].name)
    _write_json(paths["slope_receipt"], receipt)
    artifacts["slope_receipt"] = _bind(paths["slope_receipt"])

    with pytest.raises(ValueError, match="does not bind slope_final_results"):
        _validate_five_component_evidence(artifacts, paths)


def test_six_gradient_binds_five_evidence_and_reports_all_interference() -> None:
    report, receipt, report_sha256 = _six_gradient_evidence()
    expected = {
        key: "a" * 64 for key in SIX_GRADIENT_SOURCE_EVIDENCE_KEYS
    }

    weights = _validate_six_gradient(
        report,
        receipt,
        report_sha256,
        expected,
    )
    assert set(weights) == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)

    report["source_evidence_sha256"].pop("tilt_receipt")
    with pytest.raises(ValueError, match="source-evidence binding differs"):
        _validate_six_gradient(report, receipt, report_sha256, expected)

    report, receipt, report_sha256 = _six_gradient_evidence()
    report["raw_measurement_evidence"]["raw_decision"] = PASS_DECISION
    with pytest.raises(ValueError, match="raw measurement binding differs"):
        _validate_six_gradient(report, receipt, report_sha256, expected)


def test_six_gradient_sorted_json_round_trip_preserves_mapping_semantics(
    tmp_path: Path,
) -> None:
    report, receipt, _ = _six_gradient_evidence()
    report_path = _write_json(tmp_path / "six_gradient_decision_report.json", report)
    report_sha256 = sha256_file(report_path)
    receipt["artifact_sha256"] = {report_path.name: report_sha256}
    receipt_path = _write_json(tmp_path / "completion_receipt.json", receipt)
    loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
    loaded_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        key: "a" * 64 for key in SIX_GRADIENT_SOURCE_EVIDENCE_KEYS
    }

    assert tuple(loaded_report["gates"]) != SIX_GRADIENT_FROZEN_GATE_KEYS
    weights = _validate_six_gradient(
        loaded_report,
        loaded_receipt,
        report_sha256,
        expected,
    )
    assert set(weights) == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)

    loaded_report["implementation_sha256"] = dict(
        reversed(tuple(loaded_report["implementation_sha256"].items()))
    )
    loaded_receipt["implementation_sha256"] = dict(
        loaded_report["implementation_sha256"]
    )
    _validate_six_gradient(
        loaded_report,
        loaded_receipt,
        report_sha256,
        expected,
    )

    loaded_report["active_components"] = list(
        reversed(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    )
    with pytest.raises(ValueError, match="active order differs"):
        _validate_six_gradient(
            loaded_report,
            loaded_receipt,
            report_sha256,
            expected,
        )


def test_preflight_source_contains_no_panel_execution_path() -> None:
    source = Path(readiness.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "sbatch",
        "parselmouth",
        "soundfile",
        "run_exact_batch",
        "torch.optim",
    ):
        assert forbidden not in source
    assert "anti-healthification guardrail" not in source
    assert "raise SystemExit(2)" in source


def test_preflight_direct_cli_requirements_only_without_pythonpath() -> None:
    root = Path(readiness.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(readiness.__file__).resolve()),
            "--requirements-only",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert report["scientific_contract_frozen"] is True
    assert report["unfrozen_scientific_contracts"] == []
    contract = report["frozen_scientific_contract"]
    assert contract["source"]["speaker_selection_salt"] == (
        "avqi-route-c-six-joint-svd-v1-20260826"
    )
    assert contract["simulation"]["source_inventory_sha256"] == (
        "859a9e058f4f44c8e15d4b37d992cefa4d1501d1127a374d7e8cb1403c020384"
    )
    assert contract["boundaries"][
        "healthy_no_step_does_not_establish_optimized_healthy_safety"
    ] is True
    assert report["execution_authorized"] is False
    assert report["joint_panel_authorized"] is False
    assert report["generator_optimizer_steps"] == 0


def test_preflight_direct_cli_execution_mode_fails_closed() -> None:
    root = Path(readiness.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(readiness.__file__).resolve())],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    assert report["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert report["execution_authorized"] is False
    assert report["generator_optimizer_steps"] == 0
