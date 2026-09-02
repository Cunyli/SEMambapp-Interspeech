from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model.avqi_components import AVQI_COMPONENT_NAMES
import scripts.evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r3 as v32


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


def test_v32r3_ledger_accepts_boolean_amendment_without_scalar_leakage(
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


def test_v32r3_promotion_receipt_is_readiness_complete_but_training_closed(
    tmp_path,
) -> None:
    artifact = tmp_path / "external_svd_report_v32r3.json"
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


def test_v32r3_runtime_config_freezes_science_and_binds_preexact_no_go() -> None:
    path = (
        Path(v32.__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_shimmer_db_candidate_e_external_svd_runtime_v32r3.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["schema_version"] == v32.RUNTIME_CONFIG_SCHEMA
    assert config["predecessor_no_go"]["scientific_decision"] == (
        v32.v32r2.PREEXACT_NO_GO
    )
    assert config["predecessor_no_go"]["candidate_exact_outcomes_opened"] is False
    assert config["runtime_successor"]["worker_count"] == 8
    assert config["runtime_successor"]["synthetic_only_warmup"] is True
    assert config["runtime_successor"]["serial_oracle_required"] is True
    scientific = config["frozen_scientific_contract"]
    assert scientific["direction_families"] == list(v32.CANDIDATE_E_VARIANTS)
    assert scientific["alpha_ladder"] == list(v32.ALPHA_LADDER)
    assert scientific["formal_total_metric_step_runtime_ms"] == 500.0
    assert scientific["thresholds_unchanged"] is True
    assert scientific["candidate_e_math_unchanged"] is True
    assert scientific["exact_praat_remains_final_judge"] is True


def test_v32r3_runtime_successor_rejects_predecessor_exact_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        Path(v32.__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_shimmer_db_candidate_e_external_svd_runtime_v32r3.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report_hash = config["predecessor_no_go"]["report_sha256"]
    receipt_hash = config["predecessor_no_go"]["receipt_sha256"]
    training_boundary = {
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v32.TRAINING_DECISION,
    }
    report = {
        "schema_version": v32.v32r2.REPORT_SCHEMA,
        "decision": v32.v32r2.PREEXACT_NO_GO,
        "source_commit": config["predecessor_no_go"]["source_commit"],
        "slurm_job_id": config["predecessor_no_go"]["job_id"],
        "preexact_gates": {
            "complete_selector_coverage": True,
            "candidate_pool_frozen_serial_equivalence": True,
            "total_metric_step_runtime_le_500ms": False,
            "selector_uses_no_candidate_exact_outcome": True,
            "selector_uses_no_identity": True,
            "candidate_e_remains_frozen": True,
            "generator_optimizer_steps_zero": True,
        },
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "scientific_promotion_granted": False,
        **training_boundary,
    }
    receipt = {
        "schema_version": v32.v32r2.RECEIPT_SCHEMA,
        "decision": v32.v32r2.PREEXACT_NO_GO,
        "artifact_sha256": {"external_svd_report_v32r2.json": report_hash},
        "candidate_exact_outcomes_opened_after_selector_seal": False,
        "exact_scoring_complete": False,
        "scientific_promotion_granted": False,
        **training_boundary,
    }
    paths = {
        "runtime": config,
        "report": report,
        "receipt": receipt,
    }
    observed_hashes = {
        "runtime": "c" * 64,
        "report": report_hash,
        "receipt": receipt_hash,
    }
    monkeypatch.setattr(
        v32.v23,
        "validate_hash",
        lambda path, expected, label: observed_hashes[str(path)],
    )
    monkeypatch.setattr(v32.v23, "read_json", lambda path: paths[str(path)])
    args = SimpleNamespace(
        runtime_config=Path("runtime"),
        runtime_config_sha256="c" * 64,
        v32r2_report=Path("report"),
        v32r2_report_sha256=report_hash,
        v32r2_receipt=Path("receipt"),
        v32r2_receipt_sha256=receipt_hash,
    )
    validated, observed = v32.validate_runtime_successor(args)
    assert validated is config
    assert observed["v32r2_report"] == report_hash

    report["candidate_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="opened or promoted exact outcomes"):
        v32.validate_runtime_successor(args)


def test_v32r3_per_case_selector_times_only_its_own_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_selector(rows, target_by_case, scale_value):
        assert scale_value == 2.0
        assert len(target_by_case) == 1
        case_id = next(iter(target_by_case))
        assert {str(row["case_id"]) for row in rows} == {case_id}
        calls.append(case_id)
        return {
            "schema_version": "frozen-selector",
            "candidate_exact_outcomes_present": False,
            "candidate_exact_outcomes_used_for_selection": False,
            "speaker_or_case_identity_used_for_routing": False,
            "rows": [{"case_id": case_id, "selected": {"alpha": 0.001}}],
        }

    clock = iter((1.0, 1.010, 2.0, 2.025))
    monkeypatch.setattr(v32, "dual_direction_selector_seal", fake_selector)
    monkeypatch.setattr(v32.time, "perf_counter", lambda: next(clock))
    runtimes = {
        "case-a": {
            "selector_runtime_per_case_ms": 0.0,
            "total_metric_step_runtime_ms": 100.0,
        },
        "case-b": {
            "selector_runtime_per_case_ms": 0.0,
            "total_metric_step_runtime_ms": 200.0,
        },
    }
    result = v32.seal_selector_per_case(
        [
            {"case_id": "case-b"},
            {"case_id": "case-a"},
        ],
        {"case-b": 1.0, "case-a": 1.0},
        2.0,
        runtimes,
    )
    assert calls == ["case-a", "case-b"]
    assert [row["case_id"] for row in result["rows"]] == ["case-a", "case-b"]
    assert runtimes["case-a"]["selector_runtime_per_case_ms"] == pytest.approx(
        10.0
    )
    assert runtimes["case-b"]["selector_runtime_per_case_ms"] == pytest.approx(
        25.0
    )
    assert runtimes["case-a"]["total_metric_step_runtime_ms"] == pytest.approx(
        110.0
    )
    assert runtimes["case-b"]["total_metric_step_runtime_ms"] == pytest.approx(
        225.0
    )


def test_v32r3_source_uses_in_memory_runtime_path_and_serial_oracle() -> None:
    pool_source = inspect.getsource(v32.build_candidate_pool_parallel)
    main_source = inspect.getsource(v32.main)
    assert "synthetic_runtime_warmup" in pool_source
    assert "refresh_waveform_chunks" in pool_source
    assert "candidate_values = candidate_batch.detach().cpu().numpy()" in pool_source
    assert "materialize_runtime_candidate" in pool_source
    assert "seal_selector_per_case" in main_source
    assert "v29.build_candidate_pool" in main_source
    assert "compare_candidate_pools" in main_source
    assert "selector_runtime_conservative_ms" not in main_source
