from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import scripts.evaluate_avqi_shimmer_db_external_svd_selector_v26 as v26
import scripts.prepare_avqi_shimmer_db_external_svd_panel_v24 as v24


def panel_rows() -> list[dict[str, object]]:
    rows = []
    conditions = ("rir_only", "snr20", "snr10")
    for speaker_index in range(6):
        speaker_id = str(100 + speaker_index)
        for view_index, view in enumerate(("cs", "sv")):
            rows.append(
                {
                    "case_id": f"case-{speaker_id}-{view}",
                    "panel_speaker_id": f"SVD:{speaker_id}",
                    "speaker_id": speaker_id,
                    "session_id": str(1000 + speaker_index),
                    "sex": "female" if speaker_index < 3 else "male",
                    "view": view,
                    "condition": conditions[
                        (2 * speaker_index + view_index) % 3
                    ],
                    "target_sha256": f"{speaker_index + view_index + 1:064x}",
                }
            )
    return rows


def target_payloads() -> tuple[dict[str, object], dict[str, object]]:
    rows = panel_rows()
    contract = {
        "schema_version": v26.TARGET_SCHEMA,
        "source_commit": "a" * 40,
        "panel_seal_sha256": "b" * 64,
        "role": "same_speaker_target_scalar_required_by_candidate_loss",
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "target_exact_components_retained": ["shimmer_db"],
        "severity_labels_created": False,
        "emitted_waveform_highpass": False,
        "selector_stage_authorized": True,
        "exact_scorer_versions": {
            "parselmouth": "test-parselmouth",
            "praat": "test-praat",
        },
        "rows": [
            {**row, "exact_target_shimmer_db": 0.25}
            for row in rows
        ],
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v26.TRAINING_DECISION,
    }
    receipt = {
        "schema_version": v26.TARGET_RECEIPT_SCHEMA,
        "decision": v26.TARGET_DECISION,
        "source_commit": "a" * 40,
        "input_sha256": {
            "panel_seal.json": "b" * 64,
            "seal_receipt.json": "c" * 64,
        },
        "target_exact_shimmer_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "selector_stage_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v26.TRAINING_DECISION,
        "artifact_sha256": {"target_label_contract.json": "d" * 64},
    }
    return contract, receipt


def test_target_binding_preserves_preselection_information_boundary() -> None:
    contract, receipt = target_payloads()
    target_by_case = v26.validate_target_binding(
        panel_rows(),
        contract,
        receipt,
        panel_sha256="b" * 64,
        panel_receipt_sha256="c" * 64,
        target_sha256="d" * 64,
        source_commit="a" * 40,
    )
    assert len(target_by_case) == 12

    opened_candidate = deepcopy(receipt)
    opened_candidate["candidate_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="opening-order drift"):
        v26.validate_target_binding(
            panel_rows(),
            contract,
            opened_candidate,
            panel_sha256="b" * 64,
            panel_receipt_sha256="c" * 64,
            target_sha256="d" * 64,
            source_commit="a" * 40,
        )

    leaked_topology = deepcopy(contract)
    leaked_topology[
        "clean_target_pulse_positions_exposed_to_output_branch"
    ] = True
    with pytest.raises(ValueError, match="information boundary drift"):
        v26.validate_target_binding(
            panel_rows(),
            leaked_topology,
            receipt,
            panel_sha256="b" * 64,
            panel_receipt_sha256="c" * 64,
            target_sha256="d" * 64,
            source_commit="a" * 40,
        )


def updated_ledger() -> dict[str, object]:
    entries = [
        {
            "dataset": "TAU",
            "speaker_id": speaker_id,
            "canonical_speaker_id": f"TAU:{speaker_id}",
            "panel_role": "frozen-shimmer-history",
        }
        for speaker_id in sorted(v24.REQUIRED_PRIOR_TAU_SPEAKER_IDS)
    ]
    entries.extend(
        {
            "dataset": "SVD",
            "speaker_id": str(100 + index),
            "canonical_speaker_id": f"SVD:{100 + index}",
            "panel_role": "shimmer_db_external_svd_v24",
            "source_commit": "a" * 40,
            "exact_shimmer_outcomes_opened_at_ledger_update": False,
        }
        for index in range(6)
    )
    return {
        "schema_version": v26.PRIOR_LEDGER_SCHEMA,
        "exact_outcomes_used_for_selection": False,
        "entries": entries,
        "added_speaker_count": 6,
    }


def test_updated_ledger_registers_external_speakers_before_exact() -> None:
    ledger = updated_ledger()
    receipt = {
        "artifact_sha256": {v26.UPDATED_LEDGER_NAME: "e" * 64},
    }
    speakers = v26.validate_updated_ledger(
        ledger,
        panel_rows(),
        receipt,
        ledger_sha256="e" * 64,
        source_commit="a" * 40,
    )
    assert len(speakers) == len(v24.REQUIRED_PRIOR_TAU_SPEAKERS) + 6

    missing = deepcopy(ledger)
    missing["entries"] = [
        entry
        for entry in missing["entries"]
        if entry["canonical_speaker_id"] != "SVD:100"
    ]
    with pytest.raises(ValueError, match="absent from updated ledger"):
        v26.validate_updated_ledger(
            missing,
            panel_rows(),
            receipt,
            ledger_sha256="e" * 64,
            source_commit="a" * 40,
        )


def summary_rows() -> list[dict[str, object]]:
    rows = []
    for row in panel_rows():
        rows.append(
            {
                **row,
                "candidate": v26.SELECTOR_NAME,
                "selected_family": "candidate_d_cycle_projected",
                "material_shimmer_db_gap": True,
                "exact_normalized_gap_reduction_shimmer_db": 0.1,
                "gradient_finite": True,
                "gradient_l2_norm": 1.0,
                "total_metric_step_runtime_ms": 100.0,
                "residual_rms_db": -60.0,
                "cosine_similarity": 1.0,
                "clip_fraction": 0.0,
                "topology_stability_pass": True,
                "selector_pass": True,
                "selector_uses_no_candidate_exact_outcome": True,
                "selected_topology_rebound": True,
                "base_topology_rebound": True,
                "pcm24_effective_step_pass": True,
                "clean_target_topology_drives_output": False,
                "metric_reconstruction_max_pcm16_error": 0,
                "metric_reconstruction_differing_samples": 0,
                "candidate_metric_reconstruction_max_pcm16_error": 0,
                "candidate_metric_reconstruction_differing_samples": 0,
                "target_reproduction_pass": True,
                "emitted_waveform_highpass": False,
                "severity_label_present": False,
            }
        )
    return rows


def test_external_summary_reuses_frozen_thresholds_without_svd_severity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v26,
        "aggregate_candidate",
        lambda candidate, rows: {
            "candidate": candidate,
            "exact_db_improvement_fraction": 1.0,
            "median_exact_db_normalized_gap_reduction": 0.1,
            "nonselected_median_normalized_gap_increase": {"cpps": 0.0},
        },
    )
    monkeypatch.setattr(
        v26,
        "aggregate_pathology_guardrails",
        lambda rows: {"decision": "PASS"},
    )
    monkeypatch.setattr(
        v26,
        "aggregate_denoising",
        lambda rows: {"decision": "PASS"},
    )
    summary = v26.summarize_external(summary_rows())

    assert summary["all_gates_pass"] is True
    assert summary["svd_severity_labels_available"] is False
    assert summary["frozen_core_severity_slice_gate_applied_to_svd"] is False
    assert summary["external_effect_slices"]["decision"] == "PASS"

    target_drift = summary_rows()
    target_drift[0]["target_reproduction_pass"] = False
    assert v26.summarize_external(target_drift)["all_gates_pass"] is False


def test_selector_seal_precedes_exact_and_training_stays_closed() -> None:
    source = Path(v26.__file__).read_text(encoding="utf-8")

    assert source.index("write_json(selector_seal_path") < source.index(
        "rows, exact_versions = build_exact_rows("
    )
    assert '"selection_uses_candidate_exact_outcome": False' in source
    assert '"base_exact_component_outcomes_present": False' in source
    assert '"severity_labels_created": False' in source
    assert '"joint_panel_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert "validate_deterministic_process_contract" in source
    assert "torch.optim" not in source
