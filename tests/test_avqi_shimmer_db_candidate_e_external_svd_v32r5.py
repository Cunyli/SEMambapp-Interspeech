from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import scripts.diagnose_avqi_shimmer_db_candidate_e_direction_v27 as diagnosis
import scripts.evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r5 as v32


def exact_metric_topology(sample_count: int) -> dict[str, object]:
    return {
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": sample_count,
        "metric_source_ranges": [[0, sample_count]],
        "metric_mapped_sample_count": sample_count,
        "metric_constant_prefix_samples": 0,
        "metric_sample_count": sample_count,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": list(
            np.arange(320.0, sample_count - 320.0, 160.0)
        ),
    }


def selector_rows(case_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, variant in enumerate(v32.CANDIDATE_E_VARIANTS):
        for alpha_index, alpha in enumerate((0.0, *v32.ALPHA_LADDER)):
            proxy = 1.5 - 0.01 * alpha_index - 0.001 * family_index
            if alpha == 0.0:
                proxy = 1.5
            rows.append(
                {
                    "case_id": case_id,
                    "variant": variant,
                    "alpha": alpha,
                    "base_path": "/not-read/base.wav",
                    "candidate_path": f"/not-read/{family_index}-{alpha}.wav",
                    "candidate_sha256": f"{family_index}{alpha_index}" * 32,
                    "current_topology_sha256": "b" * 64,
                    "current_topology_proxy_shimmer_db": proxy,
                    "finite_safety_pass": True,
                    "pcm24_effective_step_pass": True,
                    "topology_stability_pass": True,
                    "paired_candidate_sinc70_search_may_be_skipped": True,
                }
            )
    return rows


def test_v32r5_runtime_config_freezes_science_and_binds_v32r4_no_go() -> None:
    path = (
        Path(v32.__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_shimmer_db_candidate_e_external_svd_runtime_v32r5.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["schema_version"] == v32.RUNTIME_CONFIG_SCHEMA
    predecessor = config["predecessor_no_go"]
    assert predecessor["job_id"] == "20043338"
    assert predecessor["scientific_decision"] == v32.v32r4.PREEXACT_NO_GO
    assert predecessor["candidate_exact_outcomes_opened"] is False
    runtime = config["runtime_successor"]
    assert runtime["topology_worker_count"] == 8
    assert runtime["executor_worker_count"] == 8
    assert runtime["pcm24_in_memory_encoding"] is True
    assert runtime["durable_single_write_without_disk_readback"] is True
    assert runtime["zero_step_exact_input_pcm16_identity_asserted"] is True
    assert runtime["zero_step_current_topology_reused_from_base"] is True
    assert runtime["candidate_topology_refresh_count"] == 8
    assert runtime["batched_exact_forward_proxy_fft"] is True
    assert runtime["serial_oracle_required"] is True
    scientific = config["frozen_scientific_contract"]
    assert scientific["direction_families"] == list(v32.CANDIDATE_E_VARIANTS)
    assert scientific["alpha_ladder"] == list(v32.ALPHA_LADDER)
    assert scientific["formal_total_metric_step_runtime_ms"] == 500.0
    assert scientific["thresholds_unchanged"] is True
    assert scientific["candidate_e_math_unchanged"] is True
    assert scientific["exact_praat_remains_final_judge"] is True


def test_v32r5_candidate_layout_refreshes_only_eight_nonzero_waveforms() -> None:
    unique, projected_zero, raw_zero = v32.candidate_layout()
    assert len(unique) == 8
    assert len(set(unique)) == 8
    assert projected_zero == 0
    assert raw_zero == 5
    assert projected_zero not in unique
    assert raw_zero not in unique


def test_v32r5_batched_proxy_matches_frozen_serial_cpu() -> None:
    sample_count = 16_000
    samples = np.arange(sample_count, dtype=np.float64)
    envelope = 0.2 + 0.03 * np.sin(2.0 * np.pi * samples / 800.0)
    first = (envelope * np.sin(2.0 * np.pi * 100.0 * samples / 16_000.0)).astype(
        np.float32
    )
    second = (first * 0.999).astype(np.float32)
    topology = exact_metric_topology(sample_count)
    observed_frozen, observed_current = v32.batched_topology_proxy_values(
        [first, second],
        topology,
        [topology, topology],
        torch.device("cpu"),
    )
    expected = [
        v32.v32r3.candidate_proxy_value(row, topology, torch.device("cpu"))
        for row in (first, second)
    ]
    assert observed_frozen == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert observed_current == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_v32r5_certificate_selector_matches_frozen_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = selector_rows("case")

    class Waveform:
        def numpy(self) -> np.ndarray:
            return np.zeros(16, dtype=np.float32)

    monkeypatch.setattr(diagnosis, "read_waveform", lambda _: Waveform())
    monkeypatch.setattr(
        diagnosis,
        "finite_safety",
        lambda *_: {"finite_safety_pass": True},
    )
    monkeypatch.setattr(
        diagnosis,
        "pcm24_effective_step",
        lambda *_: {"pcm24_effective_step_pass": True},
    )
    expected = v32.v32r3.dual_direction_selector_seal(
        rows,
        {"case": 1.0},
        1.0,
    )
    observed = v32.dual_direction_selector_from_certificates(
        rows,
        {"case": 1.0},
        1.0,
    )
    assert observed == expected


def test_v32r5_in_memory_pcm24_materialization_is_file_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.arange(16_000, dtype=np.float64)
    values = (
        0.2 * np.sin(2.0 * np.pi * 101.0 * samples / 16_000.0)
    ).astype(np.float32)
    candidate_path = tmp_path / "candidate.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(reference_path, values, 16_000, subtype="PCM_24")
    monkeypatch.setattr(
        v32,
        "pcm16_roundtrip_values_to_codes",
        lambda value: np.zeros(value.size, dtype=np.int16),
    )
    monkeypatch.setattr(
        v32,
        "pcm16_roundtrip",
        lambda value: np.asarray(value),
    )
    monkeypatch.setattr(
        v32,
        "paired_candidate_peak_certificate",
        lambda *_: {
            "base_peak_check_mode": "exact",
            "base_highpass_peak_scaled": False,
            "candidate_sinc70_peak_upper_bound": 0.1,
            "candidate_sinc70_search_may_be_skipped": True,
            "failure_mode": None,
        },
    )
    monkeypatch.setattr(v32.direct, "waveform_safety", lambda *_: {})
    monkeypatch.setattr(
        v32,
        "finite_safety",
        lambda *_: {
            "waveform_finite": True,
            "waveform_bound_pass": True,
            "finite_safety_pass": True,
        },
    )
    monkeypatch.setattr(
        v32,
        "pcm24_metrics_from_loaded",
        lambda *_: {"pcm24_effective_step_pass": True},
    )
    row, timing = v32.materialize_runtime_candidate(
        {
            "base_values": np.zeros_like(values),
            "base_codes": np.zeros(values.size, dtype=np.int64),
            "base_sha256": "a" * 64,
            "base_pcm16_codes": np.zeros(values.size, dtype=np.int16),
            "base_highpass_timing": {},
            "stop_hann_impulse_certificate": {},
        },
        {
            "item_id": "candidate",
            "case_id": "case",
            "view": "cs",
            "variant": v32.VARIANT_E_PROJECTED,
            "alpha": 0.001,
            "base_path": str(tmp_path / "base.wav"),
            "candidate_path": str(candidate_path),
        },
        values,
    )
    observed, _ = sf.read(candidate_path, dtype="float32")
    expected, _ = sf.read(reference_path, dtype="float32")
    assert candidate_path.read_bytes() == reference_path.read_bytes()
    np.testing.assert_array_equal(row["_stored_waveform"], observed)
    np.testing.assert_array_equal(observed, expected)
    assert set(timing) == {
        "pcm24_in_memory_encode_ms",
        "pcm24_durable_single_write_ms",
        "pcm24_memory_decode_and_certificates_ms",
    }


def test_v32r5_runtime_successor_rejects_predecessor_exact_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        Path(v32.__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_shimmer_db_candidate_e_external_svd_runtime_v32r5.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predecessor = config["predecessor_no_go"]
    training_boundary = {
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v32.TRAINING_DECISION,
    }
    runtime_by_case = {
        f"case-{index}": {
            "total_metric_step_runtime_ms": (
                600.0 if index < predecessor["failed_case_count"] else 300.0
            )
        }
        for index in range(v32.EXPECTED_CASES)
    }
    runtime_by_case["case-0"]["total_metric_step_runtime_ms"] = predecessor[
        "maximum_total_metric_step_runtime_ms"
    ]
    report = {
        "schema_version": v32.v32r4.REPORT_SCHEMA,
        "decision": v32.v32r4.PREEXACT_NO_GO,
        "source_commit": predecessor["source_commit"],
        "slurm_job_id": predecessor["job_id"],
        "preexact_gates": {
            "complete_selector_coverage": True,
            "candidate_pool_frozen_serial_equivalence": True,
            "total_metric_step_runtime_le_500ms": False,
            "selector_uses_no_candidate_exact_outcome": True,
            "selector_uses_no_identity": True,
            "candidate_e_remains_frozen": True,
            "generator_optimizer_steps_zero": True,
        },
        "runtime_by_case": runtime_by_case,
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "scientific_promotion_granted": False,
        **training_boundary,
    }
    receipt = {
        "schema_version": v32.v32r4.RECEIPT_SCHEMA,
        "decision": v32.v32r4.PREEXACT_NO_GO,
        "artifact_sha256": {
            "external_svd_report_v32r4.json": predecessor["report_sha256"],
            "selector_seal_pre_exact_v32r4.json": predecessor[
                "selector_seal_sha256"
            ],
            "candidate_e_attempts_pre_exact_v32r4.csv": predecessor[
                "candidate_attempts_sha256"
            ],
            "candidate_pool_equivalence_v32r4.json": predecessor[
                "candidate_pool_equivalence_sha256"
            ],
        },
        "candidate_exact_outcomes_opened_after_selector_seal": False,
        "exact_scoring_complete": False,
        "scientific_promotion_granted": False,
        **training_boundary,
    }
    payloads = {"runtime": config, "report": report, "receipt": receipt}
    observed_hashes = {
        "runtime": "c" * 64,
        "report": predecessor["report_sha256"],
        "receipt": predecessor["receipt_sha256"],
    }
    monkeypatch.setattr(
        v32.v23,
        "validate_hash",
        lambda path, expected, label: observed_hashes[str(path)],
    )
    monkeypatch.setattr(
        v32.v23,
        "read_json",
        lambda path: payloads[str(path)],
    )
    args = SimpleNamespace(
        runtime_config=Path("runtime"),
        runtime_config_sha256="c" * 64,
        v32r4_report=Path("report"),
        v32r4_report_sha256=predecessor["report_sha256"],
        v32r4_receipt=Path("receipt"),
        v32r4_receipt_sha256=predecessor["receipt_sha256"],
    )
    validated, observed = v32.validate_runtime_successor(args)
    assert validated is config
    assert observed["v32r4_report"] == predecessor["report_sha256"]

    report["candidate_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="opened or promoted exact outcomes"):
        v32.validate_runtime_successor(args)


def test_v32r5_source_seals_selector_before_exact_and_never_trains() -> None:
    source = inspect.getsource(v32.main)
    preexact_source = inspect.getsource(v32.preexact_no_go)
    assert source.index("v23.write_json(selector_path") < source.index(
        "run_exact_components("
    )
    assert '"candidate_exact_outcomes_present": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert "optimizer.step" not in source
    assert "torch.optim" not in source
    assert '"runtime_environment": runtime_environment' in preexact_source


def test_v32r5_source_uses_pcm24_memory_path_and_keeps_serial_oracle() -> None:
    pool_source = inspect.getsource(v32.build_candidate_pool_pipeline)
    main_source = inspect.getsource(v32.main)
    materialize_source = inspect.getsource(v32.materialize_runtime_candidate)
    assert "io.BytesIO" in materialize_source
    assert "write_bytes(encoded)" in materialize_source
    assert "sf.read" in materialize_source
    assert "raw_zero" in pool_source
    assert "zero-step PCM24 drift" in pool_source
    assert "zero-step exact-input PCM16 drift" in pool_source
    assert "refresh_waveform_chunks" in pool_source
    assert "batched_topology_proxy_values" in pool_source
    assert "candidate_unique_pcm24_count" in pool_source
    assert "v29.build_candidate_pool" in main_source
    assert "compare_candidate_pools" in main_source


def test_v32r5_promotion_receipt_keeps_training_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "external_svd_report_v32r5.json"
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
    assert receipt["result_blind_external_three_stage_chain_complete"] is True
    assert receipt["v32r3_preexact_no_go_preserved"] is True
    assert receipt["v32r4_preexact_no_go_preserved"] is True
    assert receipt["six_component_readiness_eligible"] is True
    assert receipt["joint_panel_authorized"] is False
    assert receipt["formal_generator_training_authorized"] is False
    assert receipt["generator_optimizer_steps"] == 0
