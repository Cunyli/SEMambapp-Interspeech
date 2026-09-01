from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30 as v30


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "avqi_route_c_shimmer_db_candidate_e_external_svd_panel_v30.json"
)


def valid_v29_authorization() -> tuple[dict[str, object], dict[str, object]]:
    report_sha256 = "a" * 64
    common = {
        "decision": v30.v29.PASS_DECISION,
        "candidate_e_remains_frozen": True,
        "retuning_authorized": False,
        "external_panel_prepare_authorized": True,
        "external_panel_authorized": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v30.TRAINING_DECISION,
    }
    report = {
        **common,
        "schema_version": v30.v29.REPORT_SCHEMA,
        "gates": {"exact": True, "safety": True},
    }
    receipt = {
        **common,
        "schema_version": v30.v29.RECEIPT_SCHEMA,
        "report_sha256": report_sha256,
    }
    return report, receipt


def test_v30_config_is_result_blind_and_retains_training_boundary() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v30.validate_v30_config(config)
    assert config["panel_selection"]["speaker_split_before_simulation"] is True
    assert config["panel_selection"]["exact_outcomes_used_for_selection"] is False
    assert config["immutable_boundaries"]["old_v23_no_go_receipt_preserved"]
    assert config["immutable_boundaries"]["generator_optimizer_steps"] == 0


def test_v30_requires_bound_passing_frozen_v29_without_overauthorization() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    report, receipt = valid_v29_authorization()
    config["authorization"]["report_sha256"] = "a" * 64
    config["authorization"]["receipt_sha256"] = "b" * 64
    v30.validate_v29_authorization(
        config,
        report,
        receipt,
        report_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )

    no_go = deepcopy(report)
    no_go["decision"] = "NO_GO"
    with pytest.raises(ValueError, match="v29 report is not PASS"):
        v30.validate_v29_authorization(
            config,
            no_go,
            receipt,
            report_sha256="a" * 64,
            receipt_sha256="b" * 64,
        )

    retuning = deepcopy(receipt)
    retuning["retuning_authorized"] = True
    with pytest.raises(ValueError, match="over-authorizes retuning"):
        v30.validate_v29_authorization(
            config,
            report,
            retuning,
            report_sha256="a" * 64,
            receipt_sha256="b" * 64,
        )


def test_v30_prepare_has_no_exact_scoring_or_candidate_selection() -> None:
    source = inspect.getsource(v30)
    assert "run_exact" not in source
    assert "parselmouth" not in source
    assert '"target_shimmer_values_opened": False' in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"selector_stage_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"emitted_waveform_highpass": False' in source


def test_v30_ledger_uses_candidate_e_versioned_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        type("Case", (), {"panel_speaker_id": "SVD:1"})(),
        type("Case", (), {"panel_speaker_id": "SVD:1"})(),
    ]
    inherited = {
        "entries": [
            {
                "canonical_speaker_id": "TAU:SD05",
                "panel_role": "frozen-shimmer-history",
            },
            {
                "canonical_speaker_id": "SVD:1",
                "panel_role": "shimmer_db_external_svd_v24",
            },
        ],
        "added_by": "shimmer_db_external_svd_v24_panel_seal",
    }
    monkeypatch.setattr(v30.v24, "extend_prior_ledger", lambda *_: inherited)
    monkeypatch.setattr(v30.v24, "validate_prior_ledger", lambda *_: set())
    output = v30.extend_prior_ledger_v30({}, cases, "c" * 40)
    selected = next(
        entry for entry in output["entries"] if entry["canonical_speaker_id"] == "SVD:1"
    )
    assert selected["panel_role"] == "shimmer_db_candidate_e_external_svd_v30"
    assert output["added_by"] == (
        "shimmer_db_candidate_e_external_svd_v30_panel_seal"
    )
