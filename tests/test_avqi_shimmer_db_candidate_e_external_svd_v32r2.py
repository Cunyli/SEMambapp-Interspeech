from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from model.avqi_components import AVQI_COMPONENT_NAMES
import scripts.evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r2 as v32


def candidate_row(
    case_id: str,
    variant: str,
    alpha: float,
    *,
    digest: str = "a",
    proxy: float = 1.0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "variant": variant,
        "alpha": alpha,
        "candidate_sha256": digest * 64,
        "current_topology_sha256": "b" * 64,
        "current_topology_proxy_shimmer_db": proxy,
        "topology_stability_pass": True,
    }


def selector(case_id: str, digest: str = "a") -> dict[str, object]:
    return {
        "rows": [
            {
                "case_id": case_id,
                "selected": {
                    "direction_family": v32.VARIANT_E_PROJECTED,
                    "alpha": 0.001,
                    "candidate_sha256": digest * 64,
                },
            }
        ]
    }


def test_v32_parallel_pool_equivalence_is_byte_topology_proxy_and_selector_bound(
) -> None:
    optimized = [
        candidate_row("case", v32.VARIANT_E_PROJECTED, 0.001)
    ]
    reference = [
        candidate_row("case", v32.VARIANT_E_PROJECTED, 0.001)
    ]
    result = v32.compare_candidate_pools(
        optimized,
        reference,
        selector("case"),
        selector("case"),
    )
    assert result["all_equal"] is True
    assert result["candidate_exact_outcomes_used"] is False

    reference[0]["candidate_sha256"] = "c" * 64
    mismatch = v32.compare_candidate_pools(
        optimized,
        reference,
        selector("case"),
        selector("case", digest="c"),
    )
    assert mismatch["all_equal"] is False
    assert mismatch["candidate_grid_waveform_byte_equal"] is False


def passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = ("rir_only", "snr20", "snr10")
    for index in range(12):
        row: dict[str, object] = {
            "case_id": f"case-{index}",
            "panel_speaker_id": f"SVD:{100 + index // 2}",
            "sex": "female" if index < 6 else "male",
            "view": "cs" if index % 2 == 0 else "sv",
            "condition": conditions[index % 3],
            "material_shimmer_db_gap": True,
            "frozen_gradient_finite": True,
            "frozen_gradient_l2_norm": 1.0,
            "total_metric_step_runtime_ms": 300.0,
            "runtime_gate_pass": True,
            "residual_rms_db": -60.0,
            "cosine_similarity": 0.999999,
            "clip_fraction": 0.0,
            "topology_stability_pass": True,
            "selector_proxy_after_shimmer_db": 1.0,
            "exact_after_shimmer_db": 1.0,
            "selector_pass": True,
            "selector_uses_no_candidate_exact_outcome": True,
            "candidate_pool_equivalence_pass": True,
            "selected_topology_rebound": True,
            "base_topology_rebound": True,
            "pcm24_effective_step_pass": True,
            "clean_target_topology_drives_output": False,
            "base_metric_reconstruction_max_pcm16_error": 0,
            "base_metric_reconstruction_differing_samples": 0,
            "candidate_metric_reconstruction_max_pcm16_error": 0,
            "candidate_metric_reconstruction_differing_samples": 0,
            "target_reproduction_pass": True,
            "emitted_waveform_highpass": False,
            "severity_label_present": False,
        }
        for component in AVQI_COMPONENT_NAMES:
            row[f"exact_normalized_gap_reduction_{component}"] = (
                0.03 if component == "shimmer_db" else 0.0
            )
        rows.append(row)
    return rows


def test_v32_summary_keeps_runtime_and_external_slice_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v32.direct,
        "aggregate_pathology_guardrails",
        lambda _: {"decision": "PASS"},
    )
    monkeypatch.setattr(
        v32.direct,
        "aggregate_denoising",
        lambda _: {"decision": "PASS"},
    )
    rows = passing_rows()
    summary = v32.summarize_external(rows)
    assert summary["all_gates_pass"] is True
    assert summary["external_effect_slices"]["decision"] == "PASS"
    assert summary["total_metric_step_runtime_ms"]["maximum"] == 300.0

    rows[0]["runtime_gate_pass"] = False
    rows[0]["total_metric_step_runtime_ms"] = 500.1
    failed = v32.summarize_external(rows)
    assert failed["all_gates_pass"] is False
    assert failed["mechanism_gates"]["total_metric_step_runtime"] is False


def test_v32_exact_scorer_is_six_component_only_and_skips_unrelated_pitch() -> None:
    source = v32.EXACT_COMPONENT_SCORER
    assert "Get shimmer (local)" in source
    assert "Get shimmer (local_dB)" in source
    assert "get_cpps" in source
    assert "get_hnr" in source
    assert "get_slope" in source
    assert "get_tilt" in source
    assert "get_pitch" not in source
    assert set(AVQI_COMPONENT_NAMES) == {
        "cpps",
        "hnr",
        "shimmer_percent",
        "shimmer_db",
        "slope",
        "tilt",
    }


def test_v32r2_ledger_accepts_boolean_amendment_without_scalar_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {"SVD:1", "SVD:3"}
    panel_rows = [
        {"panel_speaker_id": speaker}
        for speaker in sorted(selected)
    ]
    selection = {
        "retained_v30_speakers": ["SVD:1"],
        "rejected_v30_speakers": ["SVD:2"],
        "replacement_speakers": ["SVD:3"],
    }
    entries = [
        {
            "canonical_speaker_id": "SVD:1",
            "panel_role": "shimmer_db_candidate_e_external_svd_v30",
            "candidate_e_v30r2_status": "retained_in_original_recipe_slot",
            "exact_shimmer_outcomes_opened_at_ledger_update": False,
        },
        {
            "canonical_speaker_id": "SVD:2",
            "panel_role": "shimmer_db_candidate_e_external_svd_v30",
            "candidate_e_v30r2_status": (
                "target_component_unscorable_not_selected"
            ),
            "exact_shimmer_outcomes_opened_at_ledger_update": False,
        },
        {
            "canonical_speaker_id": "SVD:3",
            "panel_role": "shimmer_db_candidate_e_external_svd_v30r2",
            "source_commit": "a" * 40,
            "target_component_scorability_boolean_used": True,
            "target_scalar_values_used": False,
            "exact_shimmer_outcomes_opened_at_ledger_update": False,
        },
    ]
    ledger = {
        "entries": entries,
        "added_speaker_count": 1,
        "target_component_scorability_boolean_used_for_selection": True,
        "target_scalar_values_used_for_selection": False,
    }
    receipt = {
        "artifact_sha256": {
            "prior_panel_speaker_ledger_after_v30r2.json": "b" * 64
        }
    }
    monkeypatch.setattr(
        v32.v24,
        "validate_prior_ledger",
        lambda _: {"SVD:1", "SVD:2", "SVD:3"},
    )
    observed = v32.validate_updated_ledger(
        ledger,
        panel_rows,
        selection,
        receipt,
        ledger_sha256="b" * 64,
        panel_source_commit="a" * 40,
    )
    assert selected.issubset(observed)

    ledger["target_scalar_values_used_for_selection"] = True
    with pytest.raises(ValueError, match="information boundary drift"):
        v32.validate_updated_ledger(
            ledger,
            panel_rows,
            selection,
            receipt,
            ledger_sha256="b" * 64,
            panel_source_commit="a" * 40,
        )


def test_v32r2_promotion_receipt_is_readiness_complete_but_training_closed(
    tmp_path,
) -> None:
    artifact = tmp_path / "external_svd_report_v32r2.json"
    artifact.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        output_dir=tmp_path,
        source_commit="a" * 40,
        slurm_job_id="123",
    )
    receipt_path = v32.write_receipt(
        args,
        v32.PASS_DECISION,
        [artifact],
        exact_opened=True,
        exact_scoring_complete=True,
        promotion_granted=True,
    )
    receipt = v32.v23.read_json(receipt_path)
    assert receipt["component"] == "shimmer_db"
    assert receipt["exact_scoring_complete"] is True
    assert receipt["result_blind_external_three_stage_chain_complete"] is True
    assert receipt["old_v23_no_go_preserved"] is True
    assert receipt["six_component_readiness_eligible"] is True
    assert receipt["joint_panel_authorized"] is False
    assert receipt["formal_generator_training_submitted"] is False
    assert receipt["formal_generator_training_authorized"] is False
    assert receipt["generator_optimizer_steps"] == 0


def test_v32_source_seals_selector_before_candidate_exact_and_never_trains() -> None:
    source = inspect.getsource(v32.main)
    assert source.index("v23.write_json(selector_path") < source.index(
        "run_exact_components("
    )
    assert '"candidate_exact_outcomes_present": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert "optimizer.step" not in source
    assert "torch.optim" not in source
