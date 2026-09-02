from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

import scripts.seal_avqi_shimmer_db_candidate_e_external_svd_target_v31r2 as v31r2


def sealed_panel() -> tuple[dict[str, object], dict[str, object]]:
    speaker_ids = ("101", "102", "104", "201", "203", "204")
    rows = []
    conditions = ("rir_only", "snr20", "snr10")
    for speaker_index, speaker_id in enumerate(speaker_ids):
        for view_index, view in enumerate(v31r2.v31.VIEWS):
            rows.append(
                {
                    "case_id": f"case-{speaker_id}-{view}",
                    "dataset": "SVD",
                    "panel_speaker_id": f"SVD:{speaker_id}",
                    "speaker_id": speaker_id,
                    "session_id": str(1000 + speaker_index),
                    "sex": "female" if speaker_index < 3 else "male",
                    "label": "patient",
                    "view": view,
                    "condition": conditions[(2 * speaker_index + view_index) % 3],
                    "target_path": f"/tmp/case-{speaker_id}-{view}.wav",
                    "target_sha256": "d" * 64,
                }
            )
    panel = {
        "schema_version": v31r2.v30r2.PANEL_SCHEMA,
        "scientific_stage_mapping": "v24_prepare_and_seal",
        "source_commit": "a" * 40,
        "case_count": 12,
        "speaker_count": 6,
        "severity_labels_created": False,
        "selection": {
            "selection_mode": "frozen_rank_then_target_scorability_boolean_only",
            "selection_uses_diagnosis": False,
            "selection_uses_severity": False,
            "selection_uses_target_scalar_values": False,
            "selection_uses_target_scorability_boolean": True,
            "selection_uses_base_or_candidate_exact_outcomes": False,
            "slot_assignment_preserves_retained_v30_recipe_mapping": True,
            "retained_v30_final_artifact_mode": v31r2.v30r2.INHERITANCE_MODE,
            "retained_v30_rerun_outputs_used_for_final_panel": False,
            "prior_ledger_excluded_before_hash_ranking": True,
            "prior_panel_speaker_overlap": 0,
            "paired_cs_sv_same_session_required": True,
            "selected_speakers": [f"SVD:{value}" for value in speaker_ids],
            "retained_v30_speakers": [
                "SVD:101",
                "SVD:102",
                "SVD:201",
                "SVD:203",
            ],
            "rejected_v30_speakers": ["SVD:103", "SVD:202"],
            "replacement_speakers": ["SVD:104", "SVD:204"],
        },
        "authorization": {
            "candidate_e_v29_decision": v31r2.v30r2.v29.PASS_DECISION,
            "v29_report_sha256": "b" * 64,
            "v29_receipt_sha256": "c" * 64,
            "external_panel_prepare_authorized": True,
            "old_v23_no_go_not_reinterpreted": True,
        },
        "scorability_artifact_sha256": {
            "target_scorability_preflight_v30r2.json": "1" * 64,
            "target_scorability_confirmation_v30r2.json": "2" * 64,
        },
        "retained_v30_rerun_diagnostic_sha256": "3" * 64,
        "retained_v30_artifact_inheritance_sha256": "4" * 64,
        "retained_v30_equivalence_sha256": "5" * 64,
        "generator": {
            "mode": "frozen_inference_only",
            "optimizer_created": False,
            "optimizer_steps": 0,
            "retained_rerun_diagnostic_only": True,
            "retained_rerun_outputs_used_for_final_panel": False,
            "replacement_outputs_used_for_final_panel": True,
        },
        "waveform_contract": {
            "emitted_waveform_highpass": False,
            "exact_metric_highpass_branch_only": True,
            "target_is_same_speaker_same_view_clean_pathological": True,
            "retained_v30_waveforms_byte_inherited_from_original_seal": True,
            "replacement_waveforms_newly_generated": True,
            "full_band_pathology_guardrails_required_later": True,
            "denoising_nonregression_required_later": True,
        },
        "failed_v30r2_evidence": {
            "job_id": "20041442",
            "state": "FAILED",
            "failure_not_reinterpreted_as_pass": True,
            "old_artifacts_remain_immutable": True,
        },
        "exact_contract": {
            "target_shimmer_scalar_values_opened": False,
            "target_scorability_boolean_opened": True,
            "base_exact_outcomes_opened": False,
            "candidate_exact_outcomes_opened": False,
            "target_scalar_stage_authorized": True,
            "selector_stage_authorized": False,
            "promotion_authorized": False,
        },
        "rows": rows,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v31r2.TRAINING_DECISION,
    }
    receipt = {
        "schema_version": v31r2.v30r2.RECEIPT_SCHEMA,
        "decision": v31r2.v30r2.PANEL_DECISION,
        "source_commit": "a" * 40,
        "target_shimmer_scalar_values_opened": False,
        "target_scorability_boolean_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "retained_v30_artifact_inheritance_verified": True,
        "retained_v30_rerun_outputs_used_for_final_panel": False,
        "failed_v30r2_evidence": {
            "job_id": "20041442",
            "state": "FAILED",
            "failure_not_reinterpreted_as_pass": True,
            "old_artifacts_remain_immutable": True,
        },
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v31r2.TRAINING_DECISION,
        "artifact_sha256": {
            "panel_seal_v30r2.json": "e" * 64,
            "target_scorability_preflight_v30r2.json": "1" * 64,
            "target_scorability_confirmation_v30r2.json": "2" * 64,
            "retained_v30_rerun_diagnostic_v30r2.json": "3" * 64,
            "retained_v30_artifact_inheritance_v30r2.json": "4" * 64,
            "retained_v30_equivalence_v30r2.json": "5" * 64,
        },
    }
    return panel, receipt


def test_v31r2_accepts_only_bound_boolean_amended_panel() -> None:
    panel, receipt = sealed_panel()
    rows = v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)
    assert len(rows) == 12

    leaked = deepcopy(panel)
    leaked["selection"]["selection_uses_target_scalar_values"] = True
    with pytest.raises(ValueError, match="selection drift"):
        v31r2.validate_panel_binding(leaked, receipt, panel_sha256="e" * 64)

    opened = deepcopy(receipt)
    opened["candidate_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="exact-opening drift"):
        v31r2.validate_panel_binding(panel, opened, panel_sha256="e" * 64)


def test_v31r2_rejects_unbound_scorability_certificate() -> None:
    panel, receipt = sealed_panel()
    receipt["artifact_sha256"][
        "target_scorability_confirmation_v30r2.json"
    ] = "f" * 64
    with pytest.raises(ValueError, match="scorability binding drift"):
        v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)


def test_v31r2_rejects_unbound_or_runtime_retained_rerun() -> None:
    panel, receipt = sealed_panel()
    receipt["artifact_sha256"][
        "retained_v30_artifact_inheritance_v30r2.json"
    ] = "f" * 64
    with pytest.raises(ValueError, match="retained binding drift"):
        v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)

    panel, receipt = sealed_panel()
    panel["generator"]["retained_rerun_outputs_used_for_final_panel"] = True
    with pytest.raises(ValueError, match="generator drift"):
        v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)

    panel, receipt = sealed_panel()
    receipt["failed_v30r2_evidence"]["state"] = "COMPLETED"
    with pytest.raises(ValueError, match="failed-run evidence drift"):
        v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)


def test_v31r2_target_contract_retains_only_db_scalar() -> None:
    panel, receipt = sealed_panel()
    rows = v31r2.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)
    exact = {
        "parselmouth_version": "test-parselmouth",
        "praat_version": "test-praat",
        "rows": [
            {"id": f"target:{row['case_id']}", "shimmer_db": index + 1.0}
            for index, row in enumerate(rows)
        ],
    }
    target = v31r2.build_target_contract(
        panel,
        rows,
        exact,
        panel_sha256="e" * 64,
        panel_receipt_sha256="f" * 64,
        source_commit="1" * 40,
        slurm_job_id="123",
        avqi_tree_sha256="2" * 64,
    )
    assert target["target_exact_components_retained"] == ["shimmer_db"]
    assert target["selection_or_tuning_use"] is False
    assert target["selector_stage_authorized"] is True
    assert target["scientific_promotion_granted"] is False
    assert target["generator_optimizer_steps"] == 0


def test_v31r2_source_has_no_candidate_or_training_execution() -> None:
    source = inspect.getsource(v31r2)
    assert "build_candidate_pool" not in source
    assert "evaluate_selector_case" not in source
    assert "torch.optim" not in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"scientific_promotion_granted": False' in source
    assert '"joint_panel_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
