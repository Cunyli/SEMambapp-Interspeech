from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.decide_avqi_route_c_six_gradient_fusion_v1 import (
    validate_panel_seal,
    validate_raw_targets,
)
from scripts.seal_avqi_route_c_six_gradient_fusion_panel_v1 import (
    AUDIT_SPLITS,
    OPENED_DECISION_RECEIPT_SHA256,
    OPENED_DECISION_REPORT_SHA256,
    OPENED_RAW_RECEIPT_SHA256,
    OPENED_RAW_REPORT_SHA256,
    PANEL_DECISION,
    PANEL_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    TRAINING_NO_GO,
    filter_label_bank_rows,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/avqi_route_c_six_gradient_fusion_contract_v1.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _target(values: list[float]) -> tuple[list[float], str]:
    payload = json.dumps(
        values,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return values, hashlib.sha256(payload).hexdigest()


def _panel() -> tuple[dict[str, object], dict[str, object], str, str, str]:
    source_commit = "c" * 40
    panel_sha256 = "a" * 64
    contract_sha256 = "b" * 64
    filtered_sha256 = "d" * 64
    cases = []
    strata = (
        ("pathological_mild", "cs"),
        ("pathological_mild", "sv"),
        ("pathological_severe", "cs"),
        ("pathological_severe", "sv"),
    )
    for split_index, split in enumerate(AUDIT_SPLITS):
        for index, (sample_group, view) in enumerate(strata):
            speaker = f"new-{split_index}-{index}"
            target, target_sha256 = _target([float(index + 1)] * 6)
            cases.append(
                {
                    "case_id": f"case-{split_index}-{index}",
                    "split": split,
                    "speaker_id": speaker,
                    "sample_id": f"sample-{split_index}-{index}",
                    "sample_group": sample_group,
                    "view": view,
                    "condition": "aug16k_phone",
                    "source_audio_file_sha256": f"{split_index + 1}{index + 1}" * 32,
                    "same_speaker_clean_pathological_target": target,
                    "target_vector_sha256": target_sha256,
                }
            )
    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "source": {
            "head": source_commit,
            "branch": "research/avqi-route-c-six-gradient-fusion-v1",
        },
        "selection_salt": "route-c-six-gradient-fusion-v1-predeclared-20260903",
        "filtered_label_bank_sha256": filtered_sha256,
        "source_label_bank_sha256": (
            "03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760"
        ),
        "contract_sha256": contract_sha256,
        "opened_evidence_sha256": {
            "opened_raw_report": OPENED_RAW_REPORT_SHA256,
            "opened_raw_receipt": OPENED_RAW_RECEIPT_SHA256,
            "opened_decision_report": OPENED_DECISION_REPORT_SHA256,
            "opened_decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
        },
        "excluded_speaker_union": ["old-a", "old-b"],
        "opened_speaker_overlap": 0,
        "fusion_rule_frozen_before_panel_selection": True,
        "gradient_measurement_performed": False,
        "candidate_exact_outcomes_opened": False,
        "fresh_or_final_joint_panel_opened": False,
        "waveform_generation_performed": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
        "cases": cases,
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "source_commit": source_commit,
        "source_branch": "research/avqi-route-c-six-gradient-fusion-v1",
        "artifact_sha256": {
            "fusion_panel_seal_v1.json": panel_sha256,
            "filtered_exact_component_label_bank_v1.csv": filtered_sha256,
        },
        "input_sha256": {
            "contract": contract_sha256,
            "source_label_bank": (
                "03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760"
            ),
            "opened_raw_report": OPENED_RAW_REPORT_SHA256,
            "opened_raw_receipt": OPENED_RAW_RECEIPT_SHA256,
            "opened_decision_report": OPENED_DECISION_REPORT_SHA256,
            "opened_decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
        },
        "candidate_exact_outcomes_opened": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    return panel, receipt, panel_sha256, contract_sha256, source_commit


def test_real_fusion_contract_is_frozen_and_excludes_opened_speakers() -> None:
    policy = validate_contract(_contract())

    assert policy["selection_salt"] == (
        "route-c-six-gradient-fusion-v1-predeclared-20260903"
    )
    excluded = {
        speaker
        for speakers in policy["excluded_speakers_by_split"].values()
        for speaker in speakers
    }
    assert excluded == {
        "PD_92",
        "SD20",
        "SD21",
        "ÄHH31",
        "FD01",
        "PD_69",
        "PD_94",
        "ÄHH36",
    }


def test_contract_rejects_rule_gate_or_training_boundary_drift() -> None:
    for mutation in ("threshold", "promotion_gate", "opened_use", "optimizer"):
        contract = copy.deepcopy(_contract())
        if mutation == "threshold":
            contract["fusion_rule"]["maximum_weighted_component_norm_share"] = 0.9
        elif mutation == "promotion_gate":
            contract["promotion_gates"][
                "all_post_cap_component_to_joint_cosines_nonnegative"
            ] = False
        elif mutation == "opened_use":
            contract["opened_failure_use"]["allowed"].append(
                "reuse opened holdout for tuning"
            )
        else:
            contract["boundaries"]["generator_optimizer_steps"] = 1
        with pytest.raises(ValueError):
            validate_contract(contract)


def test_label_filter_removes_opened_speakers_only_from_audit_splits() -> None:
    rows = [
        {"split": "surrogate_calibration", "speaker_id": "old-a"},
        {"split": "surrogate_holdout", "speaker_id": "old-b"},
        {"split": "surrogate_train", "speaker_id": "old-a"},
        {"split": "surrogate_holdout", "speaker_id": "new-a"},
    ]

    filtered, removed = filter_label_bank_rows(rows, {"old-a", "old-b"})

    assert removed == 2
    assert filtered == [rows[2], rows[3]]


def test_panel_seal_validation_builds_exact_new_precedent() -> None:
    panel, receipt, panel_sha256, contract_sha256, source_commit = _panel()

    precedent = validate_panel_seal(
        panel,
        receipt,
        panel_sha256=panel_sha256,
        contract_sha256=contract_sha256,
        source_commit=source_commit,
    )

    assert len(precedent["case_selectors"]) == 8
    assert set(precedent["speaker_sets"]) == set(AUDIT_SPLITS)
    assert set(precedent["excluded_speakers"]) == {"old-a", "old-b"}
    assert len(precedent["target_vectors"]) == 8


def test_raw_targets_must_match_the_sealed_panel() -> None:
    panel, receipt, panel_sha256, contract_sha256, source_commit = _panel()
    precedent = validate_panel_seal(
        panel,
        receipt,
        panel_sha256=panel_sha256,
        contract_sha256=contract_sha256,
        source_commit=source_commit,
    )
    rows = []
    for panel_case in panel["cases"]:
        rows.append(
            {
                key: panel_case[key]
                for key in (
                    "split",
                    "speaker_id",
                    "sample_id",
                    "sample_group",
                    "view",
                    "condition",
                    "source_audio_file_sha256",
                )
            }
        )
        rows[-1]["components"] = {
            component: {"clean_pathological_target": value}
            for component, value in zip(
                (
                    "cpps",
                    "hnr",
                    "shimmer_percent",
                    "shimmer_db",
                    "slope",
                    "tilt",
                ),
                panel_case["same_speaker_clean_pathological_target"],
            )
        }

    validate_raw_targets(rows, precedent["target_vectors"])
    rows[0]["components"]["shimmer_db"]["clean_pathological_target"] += 0.01
    with pytest.raises(ValueError, match="raw targets differ"):
        validate_raw_targets(rows, precedent["target_vectors"])


def test_panel_seal_rejects_target_hash_or_opened_speaker_overlap() -> None:
    panel, receipt, panel_sha256, contract_sha256, source_commit = _panel()
    for mutation in ("target_hash", "overlap"):
        changed = copy.deepcopy(panel)
        if mutation == "target_hash":
            changed["cases"][0]["target_vector_sha256"] = "f" * 64
        else:
            changed["cases"][0]["speaker_id"] = "old-a"
        with pytest.raises(ValueError):
            validate_panel_seal(
                changed,
                receipt,
                panel_sha256=panel_sha256,
                contract_sha256=contract_sha256,
                source_commit=source_commit,
            )
