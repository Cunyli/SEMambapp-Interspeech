from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.adjudicate_avqi_shimmer_db_deterministic_opened24_v23 as v23


def passing_v22_gates(mode: str) -> dict[str, bool]:
    names = {
        "complete_24case_three_repeat_coverage",
        "all_full_steps_within_frozen_500ms",
        "all_full_steps_within_450ms_engineering_margin",
        "all_selectors_pass",
        "selected_pcm24_durable_byte_equivalence",
        "full_step_timer_envelope_and_phase_accounting",
        "all_attempts_equal_deterministic_reference",
        "historical_v18_comparison_complete",
    }
    if mode == "deterministic_repeat":
        names.add("baseline_capture_receipt_authorized_repeat")
    return {name: True for name in names}


def v22_payloads() -> tuple[dict[str, object], ...]:
    migration_hash = "1" * 64
    manifest_hash = "2" * 64
    capture_report_hash = "3" * 64
    repeat_report_hash = "4" * 64
    durable_hash = "5" * 64
    common_report = {
        "schema_version": v23.V22_REPORT_SCHEMA,
        "source_commit": v23.V22_SOURCE_COMMIT,
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "v18_artifacts_mutated": False,
        "candidate_input_contract": {
            "immutable_v18_comparison_disclosed_separately": True,
        },
        "migration_review": {
            "sha256": migration_hash,
            "immutable_v18_kept_separate": True,
        },
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v23.TRAINING_DECISION,
    }
    capture_report = {
        **common_report,
        "decision": v23.v22.CAPTURE_PASS_DECISION,
        "candidate_reference_mode": "deterministic_capture",
        "gates": passing_v22_gates("deterministic_capture"),
        "deterministic_repeat_authorized": True,
        "new_sealed_panel_authorized": False,
        "deterministic_baseline_output": {
            "sha256": manifest_hash,
            "deterministic_repeat_authorized": True,
        },
    }
    repeat_report = {
        **common_report,
        "decision": v23.v22.REPEAT_PASS_DECISION,
        "candidate_reference_mode": "deterministic_repeat",
        "gates": passing_v22_gates("deterministic_repeat"),
        "deterministic_repeat_authorized": False,
        "new_sealed_panel_authorized": True,
        "deterministic_baseline_binding": {"manifest_sha256": manifest_hash},
    }
    receipt_common = {
        "schema_version": v23.V22_RECEIPT_SCHEMA,
        "source_commit": v23.V22_SOURCE_COMMIT,
        "candidate_exact_avqi_components_opened": False,
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v23.TRAINING_DECISION,
    }
    capture_receipt = {
        **receipt_common,
        "decision": v23.v22.CAPTURE_PASS_DECISION,
        "deterministic_repeat_authorized": True,
        "new_sealed_panel_authorized": False,
        "artifact_sha256": {
            "diagnostic_report.json": capture_report_hash,
            "deterministic_baseline_manifest.json": manifest_hash,
        },
    }
    repeat_receipt = {
        **receipt_common,
        "decision": v23.v22.REPEAT_PASS_DECISION,
        "deterministic_repeat_authorized": False,
        "new_sealed_panel_authorized": True,
        "artifact_sha256": {
            "diagnostic_report.json": repeat_report_hash,
            "durable_selected_equivalence.csv": durable_hash,
        },
    }
    baseline_manifest = {
        "schema_version": v23.v22.DETERMINISTIC_MANIFEST_SCHEMA,
        "source_commit": v23.V22_SOURCE_COMMIT,
        "candidate_reference_mode": "deterministic_capture",
        "migration_review_sha256": migration_hash,
        "deterministic_repeat_authorized": True,
        "historical_v18_kept_separate": True,
        "candidate_exact_avqi_components_opened": False,
        "new_sealed_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v23.TRAINING_DECISION,
        "attempt_references": [
            {"case_id": f"case-{index}"}
            for index in range(v23.v22.EXPECTED_REFERENCE_ATTEMPT_COUNT)
        ],
    }
    hashes = {
        "capture_report_sha256": capture_report_hash,
        "capture_receipt_sha256": "6" * 64,
        "baseline_manifest_sha256": manifest_hash,
        "repeat_report_sha256": repeat_report_hash,
        "repeat_receipt_sha256": "7" * 64,
        "durable_csv_sha256": durable_hash,
    }
    return (
        capture_report,
        capture_receipt,
        baseline_manifest,
        repeat_report,
        repeat_receipt,
        hashes,
    )


def test_v22_chain_passes_only_with_separate_v18_and_bound_receipts() -> None:
    capture, capture_receipt, manifest, repeat, repeat_receipt, hashes = (
        v22_payloads()
    )
    evidence = v23.validate_v22_chain_payloads(
        capture,
        capture_receipt,
        manifest,
        repeat,
        repeat_receipt,
        **hashes,
    )

    assert evidence["v18_evidence_kept_separate"] is True
    assert evidence["candidate_exact_closed_during_selection"] is True
    assert evidence["new_sealed_panel_authorized_by_repeat"] is True

    collapsed = deepcopy(repeat)
    collapsed["candidate_input_contract"][
        "immutable_v18_comparison_disclosed_separately"
    ] = False
    with pytest.raises(ValueError, match="collapsed the v18 comparison"):
        v23.validate_v22_chain_payloads(
            capture,
            capture_receipt,
            manifest,
            collapsed,
            repeat_receipt,
            **hashes,
        )


def test_durable_selected_requires_three_hash_identical_unique_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v23, "EXPECTED_CASE_COUNT", 2)
    case_ids = {"case-a", "case-b"}
    waveform = np.sin(np.arange(512, dtype=np.float32) / 17.0) * 0.05
    rows = []
    for repeat in range(1, v23.EXPECTED_REPEATS + 1):
        repeat_root = tmp_path / f"repeat_{repeat}"
        repeat_root.mkdir()
        for case_id in sorted(case_ids):
            path = repeat_root / f"{case_id}__selected.wav"
            sf.write(path, waveform, v23.hybrid.SAMPLE_RATE, subtype="PCM_24")
            observed_hash = v23.sha256_file(path)
            rows.append(
                {
                    "case_id": case_id,
                    "repeat_index": str(repeat),
                    "selected_candidate_present": "True",
                    "selected_family": "D",
                    "selected_alpha": "0.001",
                    "durable_selected_path": str(path),
                    "durable_selected_sha256": observed_hash,
                    "durable_byte_equivalence_pass": "True",
                    "selected_path_updated_to_durable_before_future_seal": "True",
                    "durable_copy_after_timed_step": "True",
                }
            )

    selected = v23.validate_durable_rows(rows, case_ids)
    assert set(selected) == case_ids
    assert {row["selected_family"] for row in selected.values()} == {"D"}

    drifted = deepcopy(rows)
    drifted[-1]["selected_alpha"] = "0.0005"
    with pytest.raises(ValueError, match="repeat identity drift"):
        v23.validate_durable_rows(drifted, case_ids)


def test_panel_contract_allows_frozen_cross_view_condition_rotation(
    tmp_path: Path,
) -> None:
    waveform = np.sin(np.arange(512, dtype=np.float32) / 19.0) * 0.05
    paths = {}
    for role in ("base", "target", "candidate"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, waveform, v23.hybrid.SAMPLE_RATE, subtype="PCM_24")
        paths[role] = path
    rows = []
    target_rows = []
    result_rows = []
    conditions = ("rir_only", "snr20", "snr10")
    for speaker_index in range(6):
        sample_group = (
            "pathological_mild"
            if speaker_index < 3
            else "pathological_severe"
        )
        for view_index, view in enumerate(("cs", "sv")):
            condition = conditions[(2 * speaker_index + view_index) % 3]
            case_id = f"speaker-{speaker_index}__{view}__{condition}"
            rows.append(
                {
                    "case_id": case_id,
                    "speaker_id": f"speaker-{speaker_index}",
                    "view": view,
                    "condition": condition,
                    "sample_group": sample_group,
                    "base_path": str(paths["base"]),
                    "base_sha256": v23.sha256_file(paths["base"]),
                    "target_path": str(paths["target"]),
                    "target_sha256": v23.sha256_file(paths["target"]),
                }
            )
            target_rows.append(
                {
                    "case_id": case_id,
                    "speaker_id": f"speaker-{speaker_index}",
                    "view": view,
                    "target_sha256": v23.sha256_file(paths["target"]),
                    "exact_target_shimmer_db": 0.5,
                }
            )
            result_rows.append(
                {
                    "case_id": case_id,
                    "gradient_finite": "True",
                    "gradient_l2_norm": "1.0",
                    "fixed_alpha": "0.001",
                    "optimized_component": "shimmer_db",
                    "candidate_path": str(paths["candidate"]),
                    "candidate_sha256": v23.sha256_file(paths["candidate"]),
                }
            )
    panel = {
        "schema_version": v23.PANEL_SCHEMAS["v14"],
        "speaker_split_before_simulation": True,
        "panel_status": "sealed_new_speaker_panel_before_exact_outcomes",
        "rows": rows,
    }
    target = {
        "schema_version": v23.TARGET_CONTRACT_SCHEMA,
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "rows": target_rows,
    }

    validated, _, _ = v23.validate_panel_payloads(
        "v14",
        panel,
        target,
        result_rows,
    )

    assert len(validated) == 12
    assert any(
        first["condition"] != second["condition"]
        for first, second in zip(validated[::2], validated[1::2])
    )


def test_exact_items_open_candidate_only_after_seal_and_hide_target_topology() -> None:
    panel_rows = [
        {
            "case_id": "case-a",
            "view": "cs",
            "target_path": "/tmp/target.wav",
            "base_path": "/tmp/base.wav",
        }
    ]
    durable = {"case-a": {"candidate_path": "/tmp/candidate.wav"}}

    items = v23.build_exact_items(panel_rows, durable)

    assert [item["role"] for item in items] == [
        "same_speaker_clean_pathological_target",
        "current_output_before_step",
        "durable_selected_after_step",
    ]
    assert items[0]["exact_metric_topology"] is False
    assert items[1]["exact_metric_topology"] is True
    assert items[2]["exact_metric_topology"] is True
    assert all(item["score_components"] is True for item in items)


def valid_completion_report() -> dict[str, object]:
    return {
        "decision": v23.PASS_DECISION,
        "external_speaker_panel_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v23.TRAINING_DECISION,
    }


def test_completion_summary_fails_on_missing_or_wrong_authorization_fields() -> None:
    summary = v23.completion_summary(valid_completion_report())
    assert summary["external_speaker_panel_authorized"] is True

    missing = valid_completion_report()
    del missing["external_speaker_panel_authorized"]
    with pytest.raises(KeyError, match="field missing"):
        v23.completion_summary(missing)

    over_authorized = valid_completion_report()
    over_authorized["scientific_promotion_granted"] = True
    with pytest.raises(ValueError, match="scientific promotion"):
        v23.completion_summary(over_authorized)

    inconsistent = valid_completion_report()
    inconsistent["decision"] = v23.FAIL_DECISION
    with pytest.raises(ValueError, match="authorization/decision mismatch"):
        v23.completion_summary(inconsistent)


def test_v23_reuses_frozen_scientific_thresholds_and_training_boundary() -> None:
    assert v23.hybrid.FIXED_ALPHA == 0.001
    assert v23.hybrid.MATERIAL_GAP_THRESHOLD == 0.02
    assert v23.hybrid.MEDIAN_REDUCTION_GATE == 0.02
    assert v23.hybrid.IMPROVEMENT_FRACTION_GATE == 0.80
    assert v23.hybrid.NONSELECTED_MEDIAN_INCREASE_GATE == 0.05
    assert v23.anti_shortcut_contract() == {
        "candidate_exact_closed_during_v22_selection": True,
        "candidate_exact_scored_only_after_durable_pcm24_seal": True,
        "target_scalar_frozen_before_candidate_exact_open": True,
        "clean_target_topology_not_used_by_selector": True,
        "selector_uses_proxy_and_frozen_thresholds_only": True,
        "old_v18_comparison_preserved_as_separate_evidence": True,
        "opened24_not_reused_as_external_promotion_panel": True,
    }

    source = Path(v23.__file__).read_text(encoding="utf-8")
    assert '"generator_optimizer_steps": 0' in source
    assert '"scientific_promotion_granted": False' in source
    assert '"joint_panel_authorized": False' in source
    assert "sf.write(" not in source
    assert "highpass" not in source.lower().replace(
        "candidate_d_fixed_alpha",
        "",
    )
