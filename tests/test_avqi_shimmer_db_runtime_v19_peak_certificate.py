from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.avqi_shimmer_peak_certificate_v19 import (
    FFT_ROUNDOFF_TRANSFORM_COUNT,
    FROZEN_NUMPY_HIGHPASS_MODE,
    PEAK_SCALE_TRIGGER,
    SINC70_ABSOLUTE_WEIGHT_BOUND,
    paired_candidate_peak_certificate,
    pcm16_roundtrip_values_to_codes,
    stop_hann_impulse_l1_certificate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def frozen_stop_hann_response(fft_length: int) -> np.ndarray:
    frequencies = (
        np.arange(fft_length // 2 + 1, dtype=np.float64)
        * 16000.0
        / fft_length
    )
    response = np.ones(frequencies.size, dtype=np.float64)
    response[frequencies <= 33.9] = 0.0
    transition = (frequencies > 33.9) & (frequencies <= 34.1)
    response[transition] = 0.5 - 0.5 * np.cos(
        np.pi / 0.2 * (frequencies[transition] - 33.9)
    )
    response.setflags(write=False)
    return response


def exact_base_timing(
    peak: float = 0.2,
    *,
    sample_abs_max: float = 0.1,
    scaled: bool = False,
) -> dict[str, object]:
    return {
        "highpass_mode": FROZEN_NUMPY_HIGHPASS_MODE,
        "highpass_peak_check_mode": "exact_praat_sinc70",
        "highpass_sample_abs_max": sample_abs_max,
        "highpass_sinc70_peak_upper_bound": (
            sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
        ),
        "highpass_sinc70_skipped": False,
        "highpass_sinc70_absolute_weight_bound": (
            SINC70_ABSOLUTE_WEIGHT_BOUND
        ),
        "highpass_peak_value": peak,
        "highpass_peak_scaled": scaled,
    }


def safe_base_timing() -> dict[str, object]:
    sample_abs_max = 0.1
    return {
        "highpass_mode": FROZEN_NUMPY_HIGHPASS_MODE,
        "highpass_peak_check_mode": "proven_safe_sinc70_l1_upper_bound",
        "highpass_sample_abs_max": sample_abs_max,
        "highpass_sinc70_peak_upper_bound": (
            sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
        ),
        "highpass_sinc70_skipped": True,
        "highpass_sinc70_absolute_weight_bound": (
            SINC70_ABSOLUTE_WEIGHT_BOUND
        ),
        "highpass_peak_value": None,
        "highpass_peak_scaled": False,
    }


def small_impulse_certificate() -> dict[str, object]:
    return stop_hann_impulse_l1_certificate(
        frozen_stop_hann_response(8),
        8,
    )


@pytest.mark.parametrize(
    ("fft_length", "expected_l1"),
    (
        (131072, 4.395621259052617),
        (262144, 4.396239847980231),
        (524288, 4.403904426505159),
    ),
)
def test_impulse_l1_matches_frozen_stop_hann_receipt(
    fft_length: int,
    expected_l1: float,
) -> None:
    certificate = stop_hann_impulse_l1_certificate(
        frozen_stop_hann_response(fft_length),
        fft_length,
    )
    assert certificate["impulse_l1_observed"] == pytest.approx(
        expected_l1,
        rel=0.0,
        abs=2e-12,
    )
    assert certificate["impulse_l1_irfft_construction_epsilon"] > 0.0
    assert certificate["impulse_l1_summation_epsilon"] > 0.0
    assert certificate["impulse_l1_upper_bound"] > (
        certificate["impulse_l1_observed"]
        + certificate["impulse_l1_irfft_construction_epsilon"]
        + certificate["impulse_l1_summation_epsilon"]
    )
    assert certificate["response_cache_waveform_dependent"] is False
    assert certificate["response_read_only"] is True


def test_impulse_certificate_rejects_writable_or_mismatched_response() -> None:
    response = frozen_stop_hann_response(8)
    writable = response.copy()
    with pytest.raises(ValueError, match="read-only"):
        stop_hann_impulse_l1_certificate(writable, 8)
    with pytest.raises(ValueError, match="mismatch"):
        stop_hann_impulse_l1_certificate(response, 16)


def test_pcm16_contract_rejects_arbitrary_floats() -> None:
    with pytest.raises(ValueError, match="not an exact worker PCM16 roundtrip"):
        pcm16_roundtrip_values_to_codes(np.array([0.1], dtype=np.float64))
    with pytest.raises(ValueError, match="signed PCM16 integer codes"):
        paired_candidate_peak_certificate(
            np.zeros(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            safe_base_timing(),
            small_impulse_certificate(),
        )


def test_pcm16_codes_and_two_independent_fft_errors_are_recorded() -> None:
    base_values = np.array([0.0, 1.0 / 32768.0] * 4, dtype=np.float64)
    candidate_values = np.array([0.0, 2.0 / 32768.0] * 4, dtype=np.float64)
    certificate = paired_candidate_peak_certificate(
        pcm16_roundtrip_values_to_codes(base_values),
        pcm16_roundtrip_values_to_codes(candidate_values),
        safe_base_timing(),
        small_impulse_certificate(),
    )
    assert certificate["paired_input_contract"] == (
        "exact_worker_pcm16_roundtrip_int16_codes"
    )
    assert certificate["paired_pcm16_difference_max_abs"] == 1.0 / 32768.0
    assert (
        certificate["fft_roundoff_transform_count"]
        == FFT_ROUNDOFF_TRANSFORM_COUNT
        == 2
    )
    assert certificate["fft_roundoff_epsilon"] == pytest.approx(
        2.0 * certificate["fft_roundoff_per_transform_epsilon"],
        rel=0.0,
        abs=0.0,
    )
    assert len(certificate["base_pcm16_codes_sha256"]) == 64
    assert len(certificate["candidate_pcm16_codes_sha256"]) == 64


def test_near_trigger_candidate_bound_falls_back_exact() -> None:
    zeros = np.zeros(8, dtype=np.int16)
    certificate = paired_candidate_peak_certificate(
        zeros,
        zeros,
        exact_base_timing(
            peak=PEAK_SCALE_TRIGGER,
            sample_abs_max=0.2,
            scaled=False,
        ),
        small_impulse_certificate(),
    )
    assert certificate["candidate_sinc70_peak_upper_bound"] > PEAK_SCALE_TRIGGER
    assert certificate["candidate_sinc70_search_may_be_skipped"] is False
    assert certificate["failure_mode"] == (
        "fallback_exact_praat_sinc70_bound_not_below_trigger"
    )


def test_scaled_base_always_falls_back_exact() -> None:
    zeros = np.zeros(8, dtype=np.int16)
    certificate = paired_candidate_peak_certificate(
        zeros,
        zeros,
        exact_base_timing(peak=1.0, sample_abs_max=0.2, scaled=True),
        small_impulse_certificate(),
    )
    assert certificate["base_requires_exact_candidate_fallback"] is True
    assert certificate["candidate_sinc70_search_may_be_skipped"] is False
    assert certificate["failure_mode"] == (
        "fallback_exact_praat_sinc70_scaled_base"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"highpass_mode": "praat_stop_hann"}, "frozen NumPy"),
        ({"highpass_sinc70_skipped": True}, "timing combination"),
        ({"highpass_peak_value": None}, "timing combination"),
        ({"highpass_peak_scaled": True}, "peak/scale decision"),
        ({"highpass_sinc70_peak_upper_bound": 0.3}, "maximum/local bound"),
    ),
)
def test_invalid_base_timing_combinations_reject(
    mutation: dict[str, object],
    message: str,
) -> None:
    timing = exact_base_timing()
    timing.update(mutation)
    zeros = np.zeros(8, dtype=np.int16)
    with pytest.raises(ValueError, match=message):
        paired_candidate_peak_certificate(
            zeros,
            zeros,
            timing,
            small_impulse_certificate(),
        )


def test_safe_bound_timing_rejects_inconsistent_flags() -> None:
    timing = deepcopy(safe_base_timing())
    timing["highpass_sinc70_skipped"] = False
    zeros = np.zeros(8, dtype=np.int16)
    with pytest.raises(ValueError, match="timing combination"):
        paired_candidate_peak_certificate(
            zeros,
            zeros,
            timing,
            small_impulse_certificate(),
        )


def test_topology_probe_is_provenance_bound_and_cannot_open_opened24() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_runtime_v19_peak_certificate.py"
    ).read_text(encoding="utf-8")
    for token in (
        "validate_repository_provenance",
        '"status",\n        "--porcelain=v1"',
        "V18_SOURCE_COMMIT",
        "expected_v18_receipt_bindings",
        "--peak-certificate-helper-sha256",
        "--evaluator-sha256",
        "--frozen-worker-sha256",
        '"implementation_sha256": source_provenance["implementation_sha256"]',
        '"candidate_exact_avqi_components_opened": False',
        '"v19_integration_probe_authorized": passed',
        '"opened24_rerun_authorized": False',
        '"topology_refresh_only_not_full_selector_step": True',
        '"generator_optimizer_steps": 0',
    ):
        assert token in source


def test_v19_runner_is_clean_hash_bound_and_topology_only() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_runtime_v19_peak_certificate.sh"
    ).read_text(encoding="utf-8")
    for token in (
        "CONFIRM_SLURM_SUBMIT",
        "status --porcelain=v1 --untracked-files=all",
        "EVALUATOR_SHA256",
        "PEAK_CERTIFICATE_HELPER_SHA256",
        "FROZEN_WORKER_SHA256",
        "V14_PANEL_CONTRACT_SHA256",
        "V15_PANEL_CONTRACT_SHA256",
        "V18_REPORT_SHA256",
        "V18_PRESELECTION_SHA256",
        "V18_RECEIPT_SHA256",
        "Refusing to overwrite v19 peak-certificate output",
        '--job-name="avqi-shim-v19-cert"',
        "phase=topology_only_peak_certificate",
        '--slurm-job-id "$SLURM_JOB_ID"',
    ):
        assert token in source
    assert "opened24" not in source.split("phase=topology_only_peak_certificate")[1]
    assert "generator" not in source.lower()
