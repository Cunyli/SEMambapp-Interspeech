from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.avqi_shimmer_exact_topology_runtime_v19 as runtime_v19
import scripts.evaluate_avqi_shimmer_db_runtime_v19_full_step_integration as integration
from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    topology_sha256,
)
from scripts.avqi_shimmer_exact_topology_runtime_v19 import (
    PAIRED_CERTIFIED_MODE,
    PairedPeakCertificateTopologyWorker,
    float32_payload,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_base_topology(values: np.ndarray) -> dict[str, object]:
    _, _, source_sha256 = float32_payload(values)
    timing = {
        "highpass_mode": NUMPY_HIGHPASS_MODE,
        "highpass_peak_check_mode": "exact_praat_sinc70",
        "highpass_sample_abs_max": 0.1,
        "highpass_sinc70_peak_upper_bound": 0.52,
        "highpass_sinc70_skipped": False,
        "highpass_sinc70_absolute_weight_bound": 5.2,
        "highpass_peak_value": 0.2,
        "highpass_peak_scaled": False,
    }
    return {
        "id": "base:case-a",
        "case_id": "case-a",
        "role": "current_output_topology",
        "view": "cs",
        "scoring_status": "ok",
        "metric_highpass": NUMPY_HIGHPASS_MODE,
        "source_waveform_float32_sha256": source_sha256,
        "source_sample_count": int(values.size),
        "metric_sample_count": int(values.size),
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": [[0, int(values.size)]],
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": int(values.size),
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": [10.0, 20.0, 30.0],
        "pulse_count": 3,
        "timing_ms": timing,
    }


def test_candidate_payload_is_only_pcm24_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmpfs_root = tmp_path / "dev_shm"
    tmpfs_root.mkdir()
    monkeypatch.setattr(runtime_v19, "TMPFS_ROOT", tmpfs_root)
    pcm24_path = tmpfs_root / "candidate.wav"
    prequantized = np.linspace(-0.1, 0.1, 257, dtype=np.float32)
    sf.write(pcm24_path, prequantized, 16_000, subtype="PCM_24")
    stored, _ = sf.read(pcm24_path, dtype="float32", always_2d=False)
    _, _, raw_sha256 = float32_payload(stored)
    pcm24_sha256 = sha256_file(pcm24_path)

    observed, observed_pcm24_sha256, observed_raw_sha256 = (
        PairedPeakCertificateTopologyWorker._read_tmpfs_pcm24(
            pcm24_path,
            pcm24_sha256,
            raw_sha256,
            stored,
        )
    )
    assert np.array_equal(observed, stored)
    assert observed_pcm24_sha256 == pcm24_sha256
    assert observed_raw_sha256 == raw_sha256

    with pytest.raises(ValueError, match="readback"):
        PairedPeakCertificateTopologyWorker._read_tmpfs_pcm24(
            pcm24_path,
            pcm24_sha256,
            raw_sha256,
            prequantized,
        )
    with pytest.raises(ValueError, match="float32 hash drift"):
        PairedPeakCertificateTopologyWorker._read_tmpfs_pcm24(
            pcm24_path,
            pcm24_sha256,
            "0" * 64,
            stored,
        )


def test_paired_base_binding_rejects_cross_case_stale_or_wrong_hash() -> None:
    values = np.linspace(-0.05, 0.05, 128, dtype=np.float32)
    topology = synthetic_base_topology(values)
    topology_hash = topology_sha256(topology)
    _, _, _, timing = (
        PairedPeakCertificateTopologyWorker._validate_base_binding(
            "case-a",
            "cs",
            values,
            topology,
            topology_hash,
        )
    )
    assert timing is topology["timing_ms"]

    with pytest.raises(ValueError, match="case identity"):
        PairedPeakCertificateTopologyWorker._validate_base_binding(
            "case-b",
            "cs",
            values,
            topology,
            topology_hash,
        )
    with pytest.raises(ValueError, match="topology hash"):
        PairedPeakCertificateTopologyWorker._validate_base_binding(
            "case-a",
            "cs",
            values,
            topology,
            "0" * 64,
        )
    stale = deepcopy(topology)
    stale["timing_ms"]["highpass_mode"] = "stale-mode"
    with pytest.raises(ValueError, match="timing mode"):
        PairedPeakCertificateTopologyWorker._validate_base_binding(
            "case-a",
            "cs",
            values,
            stale,
            topology_sha256(stale),
        )


def test_worker_row_echoes_candidate_pcm24_and_base_timing_hashes() -> None:
    item = {
        "raw_float32_sha256": "1" * 64,
        "paired_base_raw_float32_sha256": "2" * 64,
        "paired_base_case_id": "case-a",
        "paired_base_view": "cs",
        "paired_base_topology_sha256": "3" * 64,
        "paired_base_highpass_timing_sha256": "4" * 64,
        "candidate_pcm24_sha256": "5" * 64,
    }
    row = {
        "topology_input_loader": (
            "client_tmpfs_raw_float32_current_output_paired_v19"
        ),
        "source_waveform_float32_sha256": "1" * 64,
        "paired_base_source_waveform_float32_sha256": "2" * 64,
        "paired_base_case_id": "case-a",
        "paired_base_view": "cs",
        "paired_base_topology_sha256": "3" * 64,
        "paired_base_highpass_timing_sha256": "4" * 64,
        "paired_candidate_pcm24_sha256": "5" * 64,
        "paired_certificate_cache_waveform_dependent": False,
        "paired_peak_certificate": {
            "paired_input_contract": (
                "exact_worker_pcm16_roundtrip_int16_codes"
            ),
            "response_cache_waveform_dependent": False,
        },
    }
    PairedPeakCertificateTopologyWorker._validate_paired_row(row, item)
    bad = deepcopy(row)
    bad["paired_candidate_pcm24_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="candidate_pcm24"):
        PairedPeakCertificateTopologyWorker._validate_paired_row(bad, item)


def test_durable_copy_is_selected_only_byte_equal_and_updates_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmpfs_root = tmp_path / "dev_shm"
    tmpfs_root.mkdir()
    monkeypatch.setattr(integration, "TMPFS_ROOT", tmpfs_root)
    selected_path = tmpfs_root / "selected.wav"
    nonselected_path = tmpfs_root / "nonselected.wav"
    waveform = np.sin(np.arange(512, dtype=np.float32) / 13.0) * 0.05
    sf.write(selected_path, waveform, 16_000, subtype="PCM_24")
    sf.write(nonselected_path, -waveform, 16_000, subtype="PCM_24")
    selected = {
        "candidate_path": selected_path,
        "candidate_sha256": sha256_file(selected_path),
    }
    record = {
        "case_id": "case-a",
        "selected_record": selected,
        "selected_family": "D",
        "selected_alpha": 0.001,
        "selected_path": selected_path,
        "attempts": [selected, {"candidate_path": nonselected_path}],
    }
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    row = integration.copy_selected_pcm24(record, durable_root, 1)
    durable_path = Path(row["durable_selected_path"])
    assert durable_path.read_bytes() == selected_path.read_bytes()
    assert row["durable_selected_sha256"] == selected["candidate_sha256"]
    assert Path(record["selected_path"]) == durable_path
    assert list(durable_root.rglob("*.wav")) == [durable_path]


def test_repeated_key_coverage_requires_exact_unique_reference_sets() -> None:
    references = {("case-a", "candidate_d"), ("case-b", "candidate_d")}
    rows = [
        {"repeat_index": repeat, "case_id": case_id, "attempt_id": attempt_id}
        for repeat in (1, 2, 3)
        for case_id, attempt_id in sorted(references)
    ]
    assert integration.repeated_key_sets_equal(
        rows,
        3,
        references,
        ("case_id", "attempt_id"),
    )
    duplicate = rows + [dict(rows[0])]
    assert not integration.repeated_key_sets_equal(
        duplicate,
        3,
        references,
        ("case_id", "attempt_id"),
    )
    missing = rows[:-1]
    assert not integration.repeated_key_sets_equal(
        missing,
        3,
        references,
        ("case_id", "attempt_id"),
    )


def test_typed_csv_contract_supports_candidate_d_and_c(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "preselection.csv"
    csv_path.write_text(
        "case_id,attempt_index,backtrack_index,selected_attempt\n"
        "case-a,0,,True\n"
        "case-a,1,0,False\n",
        encoding="utf-8",
    )
    candidate_d, candidate_c = integration.opened24.read_csv(csv_path)

    assert candidate_d["attempt_index"] == 0.0
    assert candidate_d["backtrack_index"] is None
    assert candidate_d["selected_attempt"] is True
    assert integration.attempt_id_from_reference(candidate_d) == "candidate_d"
    assert integration.parse_optional_int(candidate_d["backtrack_index"]) is None
    assert integration.parse_bool(candidate_d["selected_attempt"]) is True

    assert candidate_c["attempt_index"] == 1.0
    assert candidate_c["backtrack_index"] == 0.0
    assert candidate_c["selected_attempt"] is False
    assert integration.attempt_id_from_reference(candidate_c) == "candidate_c_bt0"
    assert integration.parse_optional_int(candidate_c["backtrack_index"]) == 0
    assert integration.parse_bool(candidate_c["selected_attempt"]) is False

    for invalid in (0.5, float("nan"), True, "1"):
        with pytest.raises(ValueError, match="optional integer"):
            integration.parse_optional_int(invalid)
    for invalid in (1, "True", None):
        with pytest.raises(ValueError, match="frozen boolean"):
            integration.parse_bool(invalid)


def test_integration_authorization_requires_every_fail_closed_gate() -> None:
    gates = {
        "all_attempts_equal_frozen_v18": True,
        "complete_24case_three_repeat_coverage": True,
        "all_full_steps_within_frozen_500ms": True,
        "all_full_steps_within_450ms_engineering_margin": True,
        "all_selectors_pass": True,
        "selected_pcm24_durable_byte_equivalence": True,
        "full_step_timer_envelope_and_phase_accounting": True,
    }
    assert integration.integration_authorized(gates) is True
    for gate_name in gates:
        failed = {**gates, gate_name: False}
        assert integration.integration_authorized(failed) is False
    with pytest.raises(ValueError, match="gate coverage"):
        integration.integration_authorized({**gates, "extra": True})


def synthetic_timer_record() -> dict[str, object]:
    batch = {
        "candidate_gpu_to_cpu_batch_ms": 1.0,
        "candidate_pcm24_io_concurrent_wall_ms": 2.0,
        "candidate_frozen_proxy_batch_ms": 1.0,
    }
    refresh = {
        "candidate_refresh_concurrent_wall_ms": 5.0,
        "candidate_refresh_request_wall_sum_ms": 5.0,
    }
    return {
        "base_refresh_runtime_ms": 10.0,
        "gradient_runtime_ms": 2.0,
        "candidate_d_plan_runtime_ms": 1.0,
        "candidate_d_projection_runtime_ms": 1.0,
        "candidate_d_batch_runtime": batch,
        "candidate_d_refresh_runtime": refresh,
        "candidate_c_batch_runtime": None,
        "candidate_c_refresh_runtime": None,
        "total_metric_step_runtime_ms": 30.0,
        "attempts": [
            {
                "candidate_refresh_client_staging_ms": 0.5,
                "candidate_pcm24_readback_used_for_refresh": True,
            }
        ],
        "selector_uses_no_candidate_exact_outcome": True,
    }


def test_full_step_timer_contract_covers_all_sequential_phases() -> None:
    accounted, passed = integration.full_step_timer_contract(
        synthetic_timer_record(),
        external_wall_ms=31.0,
        device_sync_call_count=4,
        device_sync_wall_ms=0.5,
    )
    assert accounted == 23.0
    assert passed is True
    _, failed = integration.full_step_timer_contract(
        synthetic_timer_record(),
        external_wall_ms=20.0,
        device_sync_call_count=4,
        device_sync_wall_ms=0.5,
    )
    assert failed is False


def test_integration_probe_keeps_exact_components_closed_and_core_frozen() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_runtime_v19_full_step_integration.py"
    ).read_text(encoding="utf-8")
    for token in (
        '"candidate_exact_avqi_components_opened": False',
        '"exact_component_scoring_requested": False',
        '"opened24_rerun_authorized": all_gates_pass',
        '"new_sealed_panel_authorized": False',
        '"generator_optimizer_steps": 0',
        "candidate_pcm24_readback_used_for_refresh",
        "paired_base_highpass_timing_sha256",
        "repeated_key_sets_equal",
        "selector_call_external_wall_ms",
        "durable_copy_after_timed_step",
    ):
        assert token in source
    for forbidden in ("build_exact_rows(", "exact_components(", "run_exact("):
        assert forbidden not in source

    frozen_hashes = {
        "evaluate_avqi_shimmer_db_topology_family_selector_v18.py": (
            "7401b4b80f6dbb546a4a88886c469bb4df6b4681bad9314f1244a046fbb2b69b"
        ),
        "evaluate_avqi_shimmer_db_trust_region_v16.py": (
            "d8bfb0f31d9d98832d6c4409e5044b5d7cbe0b8b585e72f359fa3119d22aa662"
        ),
        "evaluate_avqi_shimmer_db_source_informed_v17.py": (
            "324660709b2e6a4994d057c4d532cf89613f535ec96490f2cb038d7b33f55b22"
        ),
        "avqi_shimmer_peak_certificate_v19.py": (
            "e77f832423153817917fc903177816c227814df3dd162881266ab5ba49653249"
        ),
        "evaluate_avqi_shimmer_db_runtime_v19_peak_certificate.py": (
            "18f2456b20861772488fa96e2f6bb54374b97c8082b48cfaa47b97c8f5004ad2"
        ),
        "avqi_shimmer_exact_topology_worker.py": (
            "c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f"
        ),
    }
    for filename, expected_sha256 in frozen_hashes.items():
        assert hashlib.sha256(
            (REPO_ROOT / "scripts" / filename).read_bytes()
        ).hexdigest() == expected_sha256


def test_full_step_runner_is_hash_bound_single_submit_and_no_exact_open() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_runtime_v19_full_step_integration.sh"
    ).read_text(encoding="utf-8")
    for token in (
        "CONFIRM_SLURM_SUBMIT",
        "status --porcelain=v1 --untracked-files=all",
        "squeue --noheader",
        "Refusing duplicate v19 integration job",
        "EVALUATOR_SHA256",
        "PEAK_CERTIFICATE_HELPER_SHA256",
        "PHASE1_EVALUATOR_SHA256",
        "FROZEN_WORKER_SHA256",
        "V19_WORKER_SHA256",
        "V19_RUNTIME_CLIENT_SHA256",
        "V18_REPORT_SHA256",
        "V18_PRESELECTION_SHA256",
        "V18_RECEIPT_SHA256",
        "V19_TOPOLOGY_REPORT_SHA256",
        "V19_TOPOLOGY_EQUIVALENCE_SHA256",
        "V19_TOPOLOGY_RUNTIME_SHA256",
        "V19_TOPOLOGY_PCM24_EQUIVALENCE_SHA256",
        "V19_TOPOLOGY_RECEIPT_SHA256",
        "--integration-runner-sha256",
        "phase=full_step_integration",
        "--repeats \"$REPEATS\"",
    ):
        assert token in source
    assert source.count("sbatch \\") == 1
    assert "build_exact_rows" not in source
    assert "generator" not in source.lower()
