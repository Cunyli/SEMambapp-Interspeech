from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_avqi_route_c_svd_history_capacity_v3 import (
    CANDIDATE_E_DECISION,
    CONTRACT_DECISION,
    FAILURE_DECISION,
    FRESH_DECISION,
    INPUT_KEYS,
    INVALID_V2_DECISION,
    TRAINING_NO_GO,
    V10_DECISION,
    V9_DECISION,
    existing_scorability_by_speaker,
    history_sources_by_speaker,
    merged_history_ledger,
    validate_contract,
    validate_invalid_v2,
    validate_v9,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/avqi_route_c_svd_history_capacity_contract_v3.json"


def _ledger(*speakers: str) -> dict[str, object]:
    return {
        "schema_version": "avqi-route-c-prior-panel-speaker-ledger-v1",
        "exact_outcomes_used_for_selection": False,
        "entries": [
            {
                "dataset": speaker.split(":", maxsplit=1)[0],
                "speaker_id": speaker.split(":", maxsplit=1)[1],
                "canonical_speaker_id": speaker,
                "panel_role": "existing",
            }
            for speaker in speakers
        ],
    }


def test_real_history_capacity_contract_is_fail_closed() -> None:
    contract = validate_contract(json.loads(CONTRACT.read_text(encoding="utf-8")))

    assert contract["decision"] == CONTRACT_DECISION
    assert set(contract["input_sha256"]) == set(INPUT_KEYS)
    assert contract["failure_decision"] == FAILURE_DECISION
    assert contract["expected_audit_result"] == {
        "candidate_e_prior_ledger_entries": 35,
        "complete_historical_exact_opened_svd_speakers": 55,
        "invalid_v2_selected_speakers": 8,
        "invalid_v2_historical_overlap": 8,
        "metadata_eligible_svd_speakers_before_complete_history_exclusion": 36,
        "remaining_eligible_svd_speakers": [
            "SVD:1301",
            "SVD:1530",
            "SVD:1819",
        ],
        "remaining_female_speakers": 3,
        "remaining_male_speakers": 0,
    }
    assert contract["audit_boundaries"]["new_target_scalar_values_opened"] is False
    assert contract["audit_boundaries"]["new_waveforms_materialized"] is False
    assert contract["audit_boundaries"]["six_gradient_evaluation_submitted"] is False
    assert contract["audit_boundaries"]["joint_panel_submitted"] is False
    assert contract["audit_boundaries"]["generator_optimizer_steps"] == 0
    assert (
        contract["audit_boundaries"]["authoritative_training_decision"]
        == TRAINING_NO_GO
    )


def test_contract_rejects_capacity_threshold_weakening() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["source_panel_capacity_gate"]["required_distinct_speakers"] = 3

    with pytest.raises(ValueError, match="scientific gate"):
        validate_contract(contract)


def test_history_union_preserves_all_source_roles() -> None:
    result = history_sources_by_speaker(
        {
            "v9": {"SVD:1", "SVD:2"},
            "v10": {"SVD:2", "SVD:3"},
            "candidate_e": {"SVD:2"},
        }
    )

    assert result == {
        "SVD:1": ["v9"],
        "SVD:2": ["candidate_e", "v10", "v9"],
        "SVD:3": ["v10"],
    }


def test_merged_history_ledger_preserves_original_entries() -> None:
    original = _ledger("TAU:FD09", "SVD:2")
    sources = {
        "SVD:1": ["v9"],
        "SVD:2": ["v10"],
        "SVD:3": ["fresh"],
    }

    merged = merged_history_ledger(
        original,
        "a" * 64,
        sources,
        {key: "b" * 64 for key in INPUT_KEYS},
        "c" * 40,
    )
    indexed = {
        entry["canonical_speaker_id"]: entry for entry in merged["entries"]
    }

    assert indexed["TAU:FD09"] == original["entries"][0]
    assert indexed["SVD:2"] == original["entries"][1]
    assert indexed["SVD:1"]["historical_exact_opened_sources"] == ["v9"]
    assert indexed["SVD:3"]["historical_exact_opened_sources"] == ["fresh"]
    assert merged["added_speaker_count"] == 2
    assert merged["historical_exact_opened_svd_speaker_count"] == 3
    assert merged["new_source_selection_performed"] is False
    assert merged["generator_optimizer_steps"] == 0


def _v9_evidence() -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    speakers = [f"SVD:{index}" for index in range(1, 25)]
    panel = {
        "exact_scores_opened": False,
        "selection": {"speaker_count": 24, "speakers": speakers},
    }
    predictions = [
        {"panel_speaker_id": speaker, "view": view}
        for speaker in speakers
        for view in ("cs", "sv")
    ]
    report = {
        "decision": V9_DECISION,
        "slurm_job_id": "19901205",
        "exact_coverage": 185 / 192,
        "exact_failures": [{"id": f"failure-{index}"} for index in range(7)],
    }
    hashes = {
        "v9_panel_seal": "a" * 64,
        "v9_diagnostic_report": "b" * 64,
        "v9_predictions": "c" * 64,
    }
    receipt = {
        "decision": V9_DECISION,
        "panel_seal_sha256": hashes["v9_panel_seal"],
        "artifact_sha256": {
            "diagnostic_report.json": hashes["v9_diagnostic_report"],
            "predictions.json": hashes["v9_predictions"],
        },
    }
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return panel, report, predictions, receipt, contract, hashes


def test_v9_validation_proves_every_variant_was_attempted() -> None:
    panel, report, predictions, receipt, contract, hashes = _v9_evidence()

    speakers, evidence = validate_v9(
        panel, report, predictions, receipt, contract, hashes
    )

    assert len(speakers) == 24
    assert evidence["attempted_exact_variants"] == 192
    assert evidence["successful_exact_variants"] == 185
    assert evidence["failed_exact_variants"] == 7


def test_v9_validation_rejects_unaccounted_exact_attempt() -> None:
    panel, report, predictions, receipt, contract, hashes = _v9_evidence()
    report = copy.deepcopy(report)
    report["exact_failures"] = report["exact_failures"][:-1]

    with pytest.raises(ValueError, match="v9 exact-opening evidence"):
        validate_v9(panel, report, predictions, receipt, contract, hashes)


def test_invalid_v2_validation_preserves_original_ledger() -> None:
    original = _ledger("TAU:FD09", "SVD:100")
    selected = [f"SVD:{index}" for index in range(200, 208)]
    updated = copy.deepcopy(original)
    updated["entries"].extend(
        {
            "dataset": "SVD",
            "speaker_id": speaker.removeprefix("SVD:"),
            "canonical_speaker_id": speaker,
            "panel_role": "six_gradient_fusion_svd_source_v2",
        }
        for speaker in selected
    )
    seal = {
        "decision": INVALID_V2_DECISION,
        "selection": {
            "target_scalar_values_used": False,
            "base_or_candidate_exact_outcomes_used": False,
        },
        "rows": [{"canonical_speaker_id": speaker} for speaker in selected],
    }
    hashes = {
        "invalid_v2_source_seal": "d" * 64,
        "invalid_v2_updated_ledger": "e" * 64,
    }
    receipt = {
        "decision": INVALID_V2_DECISION,
        "slurm_job_id": "20083880",
        "target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
        "generator_optimizer_steps": 0,
        "artifact_sha256": {
            "svd_source_panel_seal_v2.json": hashes["invalid_v2_source_seal"],
            "prior_speaker_ledger_after_svd_v2.json": hashes[
                "invalid_v2_updated_ledger"
            ],
        },
    }
    invalid_contract = {
        "decision": "FROZEN_BEFORE_EXTERNAL_SVD_SOURCE_PANEL_SELECTION"
    }

    observed, evidence = validate_invalid_v2(
        invalid_contract, seal, receipt, updated, original, hashes
    )

    assert observed == set(selected)
    assert evidence["waveforms_materialized"] is False
    assert evidence["target_scalar_values_opened"] is False


def test_existing_scorability_is_boolean_only_and_view_complete() -> None:
    seal = {
        "target_scorability_audit": {
            "rows": [
                {
                    "id": f"SVD:1301:{view}",
                    "all_six_components_scorable": True,
                    "failure_class": "none",
                }
                for view in ("cs", "sv")
            ]
        }
    }

    assert existing_scorability_by_speaker(seal) == {
        "SVD:1301": {"cs": True, "sv": True}
    }


def test_decision_constants_do_not_conflate_component_or_training_gates() -> None:
    assert FAILURE_DECISION.startswith("NO_GO_ROUTE_C_SIX_GRADIENT_SVD_SOURCE_PANEL")
    assert INVALID_V2_DECISION.startswith("SEALED_UNUSED_SPEAKER")
    assert V10_DECISION.startswith("PASS_EXTERNAL_SVD_LTAS")
    assert FRESH_DECISION.startswith("PASS_LTAS_SLOPE")
    assert CANDIDATE_E_DECISION.startswith("PASS_CANDIDATE_E_EXACT_PRAAT")
    assert TRAINING_NO_GO == "NO_GO_AVQI_T2_TRAINING"
