from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.prepare_avqi_shimmer_db_external_svd_panel_v24 as v24


def valid_opened24_authorization() -> tuple[dict[str, object], dict[str, object]]:
    report_hash = "a" * 64
    common = {
        "decision": v24.OPENED24_PASS_DECISION,
        "external_speaker_panel_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v24.TRAINING_DECISION,
    }
    report = {
        **common,
        "schema_version": v24.OPENED24_REPORT_SCHEMA,
        "exact_scoring_complete": True,
        "gates": {"exact": True, "safety": True, "anti_shortcut": True},
    }
    receipt = {
        **common,
        "schema_version": v24.OPENED24_RECEIPT_SCHEMA,
        "artifact_sha256": {"diagnostic_report.json": report_hash},
    }
    return report, receipt


def test_external_panel_requires_bound_v23_pass_without_overauthorization() -> None:
    report, receipt = valid_opened24_authorization()
    v24.validate_opened24_authorization(
        report,
        receipt,
        report_sha256="a" * 64,
    )

    no_go = deepcopy(report)
    no_go["decision"] = "NO_GO"
    with pytest.raises(ValueError, match="report is not PASS"):
        v24.validate_opened24_authorization(
            no_go,
            receipt,
            report_sha256="a" * 64,
        )

    unbound = deepcopy(receipt)
    unbound["artifact_sha256"]["diagnostic_report.json"] = "b" * 64
    with pytest.raises(ValueError, match="receipt/report binding drift"):
        v24.validate_opened24_authorization(
            report,
            unbound,
            report_sha256="a" * 64,
        )

    over_authorized = deepcopy(report)
    over_authorized["scientific_promotion_granted"] = True
    with pytest.raises(ValueError, match="over-authorized promotion"):
        v24.validate_opened24_authorization(
            over_authorized,
            receipt,
            report_sha256="a" * 64,
        )


def prior_ledger() -> dict[str, object]:
    return {
        "schema_version": v24.PRIOR_LEDGER_SCHEMA,
        "exact_outcomes_used_for_selection": False,
        "entries": [
            {
                "dataset": "SVD",
                "speaker_id": "100",
                "canonical_speaker_id": "SVD:100",
                "panel_role": "prior-test-panel",
            },
            {
                "dataset": "TAU",
                "speaker_id": "SD05",
                "canonical_speaker_id": "TAU:SD05",
                "panel_role": "opened24-v14",
            },
        ],
    }


def test_prior_ledger_is_canonical_unique_and_result_blind() -> None:
    assert v24.validate_prior_ledger(prior_ledger()) == {"SVD:100", "TAU:SD05"}

    duplicate = prior_ledger()
    duplicate["entries"].append(dict(duplicate["entries"][0]))
    with pytest.raises(ValueError, match="duplicate prior-ledger speaker"):
        v24.validate_prior_ledger(duplicate)

    result_selected = prior_ledger()
    result_selected["exact_outcomes_used_for_selection"] = True
    with pytest.raises(ValueError, match="selected using exact outcomes"):
        v24.validate_prior_ledger(result_selected)


def write_svd_audio(path: Path, seconds: float) -> None:
    samples = int(v24.SAMPLE_RATE * seconds)
    waveform = np.sin(np.arange(samples, dtype=np.float32) / 23.0) * 0.05
    sf.write(path, waveform, v24.SAMPLE_RATE, subtype="PCM_16")


def metadata_rows(
    tmp_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Path, Path]:
    sv_root = tmp_path / "sv"
    cs_root = tmp_path / "cs"
    sv_root.mkdir()
    cs_root.mkdir()
    sv_rows = []
    cs_rows = []
    speakers = [
        ("100", "female"),
        ("101", "female"),
        ("102", "female"),
        ("103", "female"),
        ("200", "male"),
        ("201", "male"),
        ("202", "male"),
    ]
    session = 1000
    for speaker_id, sex in speakers:
        sessions = (session, session + 100) if speaker_id == "101" else (session,)
        for session_id in sessions:
            sv_name = f"{session_id}-sv.wav"
            cs_name = f"{session_id}-cs.wav"
            write_svd_audio(sv_root / sv_name, 1.2)
            write_svd_audio(cs_root / cs_name, 3.2)
            common = {
                "session_id": str(session_id),
                "speaker id": speaker_id,
                "gender": sex,
                "health status": "1",
                "diagnosis": "record-only-diagnosis",
            }
            sv_rows.append({**common, "filename": sv_name})
            cs_rows.append({**common, "filename": cs_name})
        session += 1
    return sv_rows, cs_rows, sv_root, cs_root


def test_svd_selection_is_ledger_disjoint_balanced_and_metadata_only(
    tmp_path: Path,
) -> None:
    sv_rows, cs_rows, sv_root, cs_root = metadata_rows(tmp_path)
    cases, selection = v24.select_svd_cases(
        sv_rows,
        cs_rows,
        sv_root,
        cs_root,
        {"SVD:100"},
    )

    assert len(cases) == 12
    assert "SVD:100" not in {case.panel_speaker_id for case in cases}
    assert Counter(case.sex for case in cases[::2]) == Counter(
        {"female": 3, "male": 3}
    )
    assert Counter(case.condition for case in cases) == Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    )
    assert tuple(case.recipe_index for case in cases) == v24.RECIPE_ASSIGNMENT
    speaker_101 = [case for case in cases if case.speaker_id == "101"]
    assert {case.session_id for case in speaker_101} == {"1001"}
    assert selection["selection_uses_diagnosis"] is False
    assert selection["selection_uses_shimmer_or_avqi"] is False
    assert selection["prior_panel_speaker_overlap"] == 0


def test_ledger_extension_registers_external_speakers_before_exact(
    tmp_path: Path,
) -> None:
    sv_rows, cs_rows, sv_root, cs_root = metadata_rows(tmp_path)
    cases, _ = v24.select_svd_cases(
        sv_rows,
        cs_rows,
        sv_root,
        cs_root,
        {"SVD:100"},
    )

    updated = v24.extend_prior_ledger(prior_ledger(), cases, "c" * 40)
    speakers = v24.validate_prior_ledger(updated)

    assert len(speakers) == 8
    assert updated["added_speaker_count"] == 6
    additions = [
        entry
        for entry in updated["entries"]
        if entry["panel_role"] == "shimmer_db_external_svd_v24"
    ]
    assert len(additions) == 6
    assert all(
        entry["exact_shimmer_outcomes_opened_at_ledger_update"] is False
        for entry in additions
    )


def test_prepare_stage_has_no_exact_scoring_or_training_path() -> None:
    source = Path(v24.__file__).read_text(encoding="utf-8")

    assert "run_exact" not in source
    assert "parselmouth" not in source
    assert '"target_shimmer_values_opened": False' in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"selector_stage_authorized": False' in source
    assert '"scientific_promotion_granted": False' in source
    assert '"joint_panel_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"emitted_waveform_highpass": False' in source
