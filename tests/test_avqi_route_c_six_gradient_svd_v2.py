from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.decide_avqi_route_c_six_gradient_fusion_v2 import (
    PASS_DECISION,
    _peak_hash_gate,
)
from scripts.materialize_avqi_route_c_six_gradient_svd_panel_v2 import (
    validate_updated_ledger,
)
from scripts.seal_avqi_route_c_six_gradient_svd_source_panel_v2 import (
    AUDIT_SPLITS,
    EXPECTED_CASES,
    STRATA,
    extend_ledger,
    select_cases,
    validate_contract,
)


CONTRACT_PATH = Path("configs/avqi_route_c_six_gradient_fusion_contract_v2.json")


def _eligible_rows() -> list[dict[str, str]]:
    rows = []
    session_id = 10_000
    for sex in ("female", "male"):
        for index in range(6):
            session_id += 1
            speaker_id = f"{sex[0]}-{index}"
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "session_id": str(session_id),
                    "sex": sex,
                    "diagnosis_record_only": "not-used-for-selection",
                    "cs_path": f"/tmp/{speaker_id}-cs.wav",
                    "sv_path": f"/tmp/{speaker_id}-sv.wav",
                }
            )
    return rows


def _scorability(rows: list[dict[str, str]]) -> dict[str, bool]:
    return {
        f"SVD:{row['speaker_id']}:{view}": True
        for row in rows
        for view in ("cs", "sv")
    }


def test_v2_contract_freezes_exact_and_training_boundaries() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    panel = validate_contract(contract)

    assert panel["target_scalar_values_used_for_selection"] is False
    assert panel["base_or_candidate_exact_outcomes_used_for_selection"] is False
    assert contract["exact_authority"]["base_exact_components_opened"] is False
    assert contract["exact_authority"]["candidate_exact_components_opened"] is False
    assert contract["boundaries"]["generator_optimizer_steps"] == 0
    assert (
        contract["boundaries"]["authoritative_training_decision"]
        == "NO_GO_AVQI_T2_TRAINING"
    )
    assert PASS_DECISION == (
        "PASS_ROUTE_C_SIX_GRADIENT_DOMINANCE_CAPPED_FUSION_V2"
    )

    changed = deepcopy(contract)
    changed["fusion_rule"]["maximum_weighted_component_norm_share"] = 0.9
    with pytest.raises(ValueError, match="fusion rule differs"):
        validate_contract(changed)


def test_svd_selection_is_deterministic_stratified_and_speaker_disjoint() -> None:
    rows = _eligible_rows()
    scorability = _scorability(rows)
    first = select_cases(rows, scorability, "sealed-selection-salt")
    second = select_cases(rows, scorability, "sealed-selection-salt")

    assert first == second
    assert len(first) == EXPECTED_CASES
    assert len({row["speaker_id"] for row in first}) == EXPECTED_CASES
    for split in AUDIT_SPLITS:
        assert {
            (row["sex"], row["view"])
            for row in first
            if row["split"] == split
        } == set(STRATA)


def test_updated_ledger_binds_only_selected_svd_speakers() -> None:
    selected = select_cases(
        _eligible_rows(),
        _scorability(_eligible_rows()),
        "sealed-selection-salt",
    )
    prior = {
        "schema_version": "avqi-route-c-prior-panel-speaker-ledger-v1",
        "exact_outcomes_used_for_selection": False,
        "entries": [
            {
                "dataset": "SVD",
                "speaker_id": "historic",
                "canonical_speaker_id": "SVD:historic",
                "panel_role": "historic",
            }
        ],
    }
    source_commit = "1" * 40
    ledger = extend_ledger(prior, selected, source_commit)
    sealed_rows = [
        {
            **row,
            "canonical_speaker_id": f"SVD:{row['speaker_id']}",
        }
        for row in selected
    ]

    validate_updated_ledger(ledger, sealed_rows, source_commit)
    tampered = deepcopy(ledger)
    selected_entry = next(
        entry
        for entry in tampered["entries"]
        if entry.get("panel_role") == "six_gradient_fusion_svd_source_v2"
    )
    selected_entry["target_scalar_values_used"] = True
    with pytest.raises(ValueError, match="selected entry differs"):
        validate_updated_ledger(tampered, sealed_rows, source_commit)


def test_peak_hash_gate_accepts_scaled_and_unscaled_exact_forward_paths() -> None:
    rows = []
    for index, scaled in enumerate((False, True)):
        digest = f"{index + 1:064x}"
        rows.append(
            {
                "topology": {
                    "highpass_pcm16_sha256": digest,
                    "highpass_peak_scaled": scaled,
                },
                "components": {
                    "shimmer_db": {
                        "candidate_e_projection": {
                            "candidate_e_exact_highpass_pcm16_sha256": digest,
                            "candidate_e_peak_scale_support_pass": True,
                            "candidate_e_peak_handling_pass": True,
                            "candidate_e_peak_scale_abstention_pass": not scaled,
                        }
                    }
                },
            }
        )

    assert _peak_hash_gate(rows) is True
    rows[1]["components"]["shimmer_db"]["candidate_e_projection"][
        "candidate_e_exact_highpass_pcm16_sha256"
    ] = "f" * 64
    assert _peak_hash_gate(rows) is False
