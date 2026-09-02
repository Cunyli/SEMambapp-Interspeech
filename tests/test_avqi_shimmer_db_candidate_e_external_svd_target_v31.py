from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

import scripts.seal_avqi_shimmer_db_candidate_e_external_svd_target_v31 as v31


def sealed_panel() -> tuple[dict[str, object], dict[str, object]]:
    rows = []
    conditions = ("rir_only", "snr20", "snr10")
    for speaker_index in range(6):
        speaker_id = str(100 + speaker_index)
        for view_index, view in enumerate(v31.VIEWS):
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
        "schema_version": v31.v30.PANEL_SCHEMA,
        "scientific_stage_mapping": "v24_prepare_and_seal",
        "source_commit": "a" * 40,
        "case_count": 12,
        "speaker_count": 6,
        "severity_labels_created": False,
        "selection": {
            "selection_mode": "metadata_only_result_blind",
            "selection_uses_diagnosis": False,
            "selection_uses_shimmer_or_avqi": False,
            "prior_ledger_excluded_before_hash_ranking": True,
            "prior_panel_speaker_overlap": 0,
            "paired_cs_sv_same_session_required": True,
        },
        "authorization": {
            "candidate_e_v29_decision": v31.v30.v29.PASS_DECISION,
            "v29_report_sha256": "b" * 64,
            "v29_receipt_sha256": "c" * 64,
            "external_panel_prepare_authorized": True,
            "old_v23_no_go_not_reinterpreted": True,
        },
        "waveform_contract": {
            "emitted_waveform_highpass": False,
            "exact_metric_highpass_branch_only": True,
            "target_is_same_speaker_same_view_clean_pathological": True,
            "full_band_pathology_guardrails_required_later": True,
            "denoising_nonregression_required_later": True,
        },
        "exact_contract": {
            "target_shimmer_values_opened": False,
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
        "authoritative_training_decision": v31.TRAINING_DECISION,
    }
    receipt = {
        "schema_version": v31.v30.RECEIPT_SCHEMA,
        "decision": v31.v30.PANEL_DECISION,
        "source_commit": "a" * 40,
        "exact_shimmer_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v31.TRAINING_DECISION,
        "artifact_sha256": {"panel_seal_v30.json": "e" * 64},
    }
    return panel, receipt


def test_v31_requires_bound_exact_unopened_candidate_e_panel() -> None:
    panel, receipt = sealed_panel()
    rows = v31.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)
    assert len(rows) == 12

    opened = deepcopy(panel)
    opened["exact_contract"]["base_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="exact-opening drift"):
        v31.validate_panel_binding(opened, receipt, panel_sha256="e" * 64)

    reinterpreted = deepcopy(panel)
    reinterpreted["authorization"]["old_v23_no_go_not_reinterpreted"] = False
    with pytest.raises(ValueError, match="reinterprets v23 NO_GO"):
        v31.validate_panel_binding(
            reinterpreted,
            receipt,
            panel_sha256="e" * 64,
        )

    leaked = deepcopy(panel)
    leaked["selection"]["selection_uses_shimmer_or_avqi"] = True
    with pytest.raises(ValueError, match="result-blind selection drift"):
        v31.validate_panel_binding(leaked, receipt, panel_sha256="e" * 64)


def test_v31_rejects_invented_external_severity() -> None:
    panel, receipt = sealed_panel()
    panel["rows"][0]["sample_group"] = "pathological_mild"
    with pytest.raises(ValueError, match="severity leakage"):
        v31.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)


def test_v31_contract_retains_only_supervised_target_scalar() -> None:
    panel, receipt = sealed_panel()
    rows = v31.validate_panel_binding(panel, receipt, panel_sha256="e" * 64)
    exact_rows = []
    for index, row in enumerate(rows):
        exact_rows.append(
            {
                "id": f"target:{row['case_id']}",
                "shimmer_db": float(index + 1),
            }
        )
    exact = {
        "parselmouth_version": "test-parselmouth",
        "praat_version": "test-praat",
        "rows": exact_rows,
    }
    contract = v31.build_target_contract(
        panel,
        rows,
        exact,
        panel_sha256="e" * 64,
        panel_receipt_sha256="f" * 64,
        source_commit="1" * 40,
        slurm_job_id="123",
        avqi_tree_sha256="2" * 64,
    )
    assert contract["target_exact_components_retained"] == ["shimmer_db"]
    assert contract["source_commit"] == "1" * 40
    assert contract["panel_source_commit"] == "a" * 40
    assert contract["selector_stage_authorized"] is True
    assert contract["scientific_promotion_granted"] is False
    assert contract["generator_optimizer_steps"] == 0
    assert all(
        set(row) == {
            "case_id",
            "panel_speaker_id",
            "speaker_id",
            "session_id",
            "sex",
            "view",
            "condition",
            "target_sha256",
            "exact_target_shimmer_db",
        }
        for row in contract["rows"]
    )


def test_v31_source_has_no_candidate_or_training_execution() -> None:
    source = inspect.getsource(v31)
    assert "build_candidate_pool" not in source
    assert "evaluate_selector_case" not in source
    assert "torch.optim" not in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"scientific_promotion_granted": False' in source
    assert '"joint_panel_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"emitted_waveform_highpass": False' in source
