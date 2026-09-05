from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.avqi_route_c_candidate_e import (
    CANDIDATE_E_NUMPY_HIGHPASS_MODE,
    CANDIDATE_E_REFERENCE_SHA256,
    SINC70_ABSOLUTE_WEIGHT_BOUND,
    build_cycle_gain_plan,
    candidate_e_proxy,
    exact_numpy_highpass_pcm16,
    pcm16_ste,
    praat_pcm16_ste,
    project_cycle_gain_gradient_fixed_order,
    validate_candidate_e_base_peak_certificate,
)


def _waveform() -> torch.Tensor:
    time = torch.arange(4_000, dtype=torch.float64) / 16_000
    return torch.sin(2.0 * math.pi * 200.0 * time) * (
        0.1 + 0.015 * torch.sin(2.0 * math.pi * 4.0 * time)
    )


def _topology(waveform: torch.Tensor) -> dict[str, object]:
    pulses = [
        float(position)
        for position in range(120, waveform.numel() - 120, 80)
    ]
    return {
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": waveform.numel(),
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": [[0, waveform.numel()]],
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": waveform.numel(),
        "metric_sample_count": waveform.numel(),
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": pulses,
    }


def test_candidate_e_reference_and_pcm16_forward_contract() -> None:
    assert CANDIDATE_E_REFERENCE_SHA256 == (
        "e9266444fa1a8a9589471fb1edd08dbec020368e4dc984551eab304f20d4a9cf"
    )
    values = torch.tensor(
        [-1.1, -0.1, 0.0, 0.1, 1.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    libsndfile = pcm16_ste(values)
    praat = praat_pcm16_ste(values)
    bounded = values.detach().clamp(-1.0, 1.0 - 1.0 / 32768.0)
    assert torch.equal(
        libsndfile.detach(),
        torch.floor(bounded * 32768.0) / 32768.0,
    )
    assert torch.equal(
        praat.detach(),
        torch.round(bounded * 32768.0) / 32768.0,
    )
    (libsndfile.sum() + praat.sum()).backward()
    assert torch.equal(values.grad[1:4], torch.full((3,), 2.0, dtype=torch.float64))


def test_candidate_e_proxy_and_fixed_order_projection_are_deterministic() -> None:
    waveform = _waveform().requires_grad_()
    original = waveform.detach().clone()
    topology = _topology(waveform)
    pulses = waveform.new_tensor(topology["pulse_positions_samples"])
    source_indices = torch.arange(waveform.numel())
    result = candidate_e_proxy(waveform, pulses, source_indices, 0)
    loss = (result.shimmer_db - 0.75).square()
    raw_gradient = torch.autograd.grad(loss, waveform)[0]
    plan = build_cycle_gain_plan(
        waveform.detach().numpy(),
        topology,
    )
    first, first_receipt = project_cycle_gain_gradient_fixed_order(
        waveform.detach(),
        raw_gradient,
        plan,
    )
    second, second_receipt = project_cycle_gain_gradient_fixed_order(
        waveform.detach(),
        raw_gradient,
        plan,
    )

    assert torch.equal(waveform.detach(), original)
    assert result.fft_sample_count == 4_096
    assert result.peak_scale_abstention_pass is True
    assert torch.isfinite(result.shimmer_db)
    assert torch.isfinite(raw_gradient).all()
    assert first_receipt["complete_cycle_count"] >= 16
    assert first_receipt["projected_gradient_valid"] is True
    assert first_receipt["projection_reduction"] == (
        "numpy_float64_fixed_cycle_order"
    )
    assert first_receipt == second_receipt
    assert torch.equal(first, second)
    assert float(torch.linalg.vector_norm(first)) > 0.0
    assert np.isfinite(first.numpy()).all()


def test_candidate_e_peak_certificate_uses_exact_fallback_for_loose_bound() -> None:
    waveform = _waveform() * 3.8
    pulses = waveform.new_tensor(_topology(waveform)["pulse_positions_samples"])
    proxy = candidate_e_proxy(
        waveform,
        pulses,
        torch.arange(waveform.numel()),
        0,
    )
    assert proxy.sinc70_peak_upper_bound > 0.999
    exact_peak = min(0.95, proxy.metric_sample_abs_max * 1.01)
    topology = {
        "metric_highpass": CANDIDATE_E_NUMPY_HIGHPASS_MODE,
        "timing_ms": {
            "highpass_mode": CANDIDATE_E_NUMPY_HIGHPASS_MODE,
            "highpass_sample_abs_max": proxy.metric_sample_abs_max,
            "highpass_sinc70_peak_upper_bound": (
                proxy.metric_sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
            ),
            "highpass_sinc70_absolute_weight_bound": (
                SINC70_ABSOLUTE_WEIGHT_BOUND
            ),
            "highpass_peak_check_mode": "exact_praat_sinc70",
            "highpass_sinc70_skipped": False,
            "highpass_peak_value": exact_peak,
            "highpass_peak_scaled": False,
        },
    }
    certificate = validate_candidate_e_base_peak_certificate(topology, proxy)
    assert certificate["base_exact_sinc70_peak"] == exact_peak
    assert certificate["base_peak_upper_bound"] < 0.999
    assert certificate["base_peak_scale_abstention_pass"] is True

    topology["timing_ms"]["highpass_peak_value"] = 1.0
    topology["timing_ms"]["highpass_peak_scaled"] = True
    with np.testing.assert_raises_regex(
        ValueError,
        "requires Praat peak scaling",
    ):
        validate_candidate_e_base_peak_certificate(topology, proxy)

    _, exact_pcm16_sha256 = exact_numpy_highpass_pcm16(
        waveform,
        peak_scale_required=True,
    )
    scaled_proxy = candidate_e_proxy(
        waveform,
        pulses,
        torch.arange(waveform.numel()),
        0,
        peak_scale_required=True,
        expected_highpass_pcm16_sha256=exact_pcm16_sha256,
    )
    topology["highpass_pcm16_sha256"] = exact_pcm16_sha256
    certificate = validate_candidate_e_base_peak_certificate(
        topology,
        scaled_proxy,
    )
    assert certificate["base_highpass_peak_scaled"] is True
    assert certificate["base_peak_scale_abstention_pass"] is False
    assert certificate["base_peak_scale_support_pass"] is True
    assert certificate["base_peak_handling_pass"] is True
    assert scaled_proxy.exact_highpass_pcm16_sha256 == exact_pcm16_sha256


def test_candidate_e_exact_pcm_hash_mismatch_fails_closed() -> None:
    waveform = _waveform()
    pulses = waveform.new_tensor(_topology(waveform)["pulse_positions_samples"])
    with pytest.raises(ValueError, match="exact high-pass PCM16 hash differs"):
        candidate_e_proxy(
            waveform,
            pulses,
            torch.arange(waveform.numel()),
            0,
            peak_scale_required=True,
            expected_highpass_pcm16_sha256="0" * 64,
        )
