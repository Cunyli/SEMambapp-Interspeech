from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.avqi_shimmer_db_candidate_e_proxy_v27 import (
    SAMPLE_RATE,
    STOP_HANN_HIGH_HZ,
    STOP_HANN_LOW_HZ,
    fixed_pulse_shimmer_db,
    official_stop_hann,
    pcm16_ste,
    project_cycle_gain_gradient_fixed_order,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "avqi_route_c_shimmer_db_candidate_e_directional_diagnostic_v27.json"
)


def test_pcm16_ste_has_exact_forward_grid_and_identity_backward() -> None:
    waveform = torch.tensor(
        [-0.75, -0.123456, 0.0, 0.123456, 0.75],
        dtype=torch.float64,
        requires_grad=True,
    )
    observed = pcm16_ste(waveform)
    expected = torch.round(waveform.detach() * 32768.0) / 32768.0
    torch.testing.assert_close(observed.detach(), expected, rtol=0.0, atol=0.0)
    observed.sum().backward()
    torch.testing.assert_close(
        waveform.grad,
        torch.ones_like(waveform),
        rtol=0.0,
        atol=0.0,
    )


def test_official_stop_hann_matches_frozen_numpy_exact_worker_formula() -> None:
    samples = torch.arange(23_111, dtype=torch.float64)
    waveform = (
        0.12 * torch.sin(2.0 * torch.pi * 23.0 * samples / SAMPLE_RATE)
        + 0.08 * torch.sin(2.0 * torch.pi * 173.0 * samples / SAMPLE_RATE)
    )
    input_pcm16 = pcm16_ste(waveform)
    observed, fft_sample_count = official_stop_hann(input_pcm16)
    values = input_pcm16.detach().numpy()
    frequencies = (
        np.arange(fft_sample_count // 2 + 1, dtype=np.float64)
        * SAMPLE_RATE
        / fft_sample_count
    )
    response = np.ones(frequencies.size, dtype=np.float64)
    response[frequencies <= STOP_HANN_LOW_HZ] = 0.0
    transition = (frequencies > STOP_HANN_LOW_HZ) & (
        frequencies <= STOP_HANN_HIGH_HZ
    )
    response[transition] = 0.5 - 0.5 * np.cos(
        np.pi
        / (STOP_HANN_HIGH_HZ - STOP_HANN_LOW_HZ)
        * (frequencies[transition] - STOP_HANN_LOW_HZ)
    )
    expected = np.fft.irfft(
        np.fft.rfft(values, n=fft_sample_count) * response,
        n=fft_sample_count,
    )[: values.size]
    np.testing.assert_allclose(
        observed.detach().numpy(),
        expected,
        rtol=0.0,
        atol=2e-15,
    )


def test_fixed_pulse_shimmer_has_finite_nonzero_gradient() -> None:
    samples = torch.arange(6_400, dtype=torch.float64)
    envelope = 0.15 + 0.02 * torch.sin(
        2.0 * torch.pi * samples / 1_700.0
    )
    prepared = (
        envelope * torch.sin(2.0 * torch.pi * 100.0 * samples / SAMPLE_RATE)
    ).requires_grad_(True)
    pulses = torch.arange(320.0, 6_080.0, 160.0, dtype=torch.float64)
    shimmer_db, amplitudes, centers, valid_pair, contributions = (
        fixed_pulse_shimmer_db(prepared, pulses)
    )
    gradient = torch.autograd.grad(shimmer_db, prepared)[0]
    assert amplitudes.numel() == centers.numel()
    assert valid_pair.numel() == contributions.numel()
    assert bool(valid_pair.any())
    assert float(shimmer_db) > 0.0
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 0.0


def test_fixed_order_projection_is_repeatable_and_preserves_shape() -> None:
    cycle_count = 20
    cycle_length = 8
    sample_count = cycle_count * cycle_length
    waveform = torch.linspace(
        -0.3,
        0.4,
        sample_count,
        dtype=torch.float64,
    )
    gradient = waveform * torch.linspace(
        -2.0,
        3.0,
        sample_count,
        dtype=torch.float64,
    )
    cycles = [
        {
            "cell_id": index,
            "mapped_start_index": index * cycle_length,
            "mapped_end_index": (index + 1) * cycle_length,
        }
        for index in range(cycle_count)
    ]
    plan = {
        "source_indices": np.arange(sample_count, dtype=np.int64),
        "cycles": cycles,
        "previous_cells": np.maximum(
            np.arange(cycle_count, dtype=np.int64) - 1,
            0,
        ),
        "next_cells": np.minimum(
            np.arange(cycle_count, dtype=np.int64) + 1,
            cycle_count - 1,
        ),
        "summary": {"complete_cycle_count": cycle_count},
    }
    first, first_receipt = project_cycle_gain_gradient_fixed_order(
        waveform,
        gradient,
        plan,
    )
    second, second_receipt = project_cycle_gain_gradient_fixed_order(
        waveform,
        gradient,
        plan,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.shape == waveform.shape
    assert first_receipt == second_receipt
    assert first_receipt["projected_gradient_valid"] is True
    assert first_receipt["projection_reduction"] == (
        "numpy_float64_fixed_cycle_order"
    )


def test_v27_config_is_v14_only_symmetric_and_fail_closed() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = config["dataset_contract"]
    grid = config["frozen_directional_grid"]["alphas"]
    boundaries = config["immutable_boundaries"]
    assert dataset["opened_v15_access_authorized"] is False
    assert dataset["external_panel_access_authorized"] is False
    assert set(grid) == {-value for value in grid}
    assert config["frozen_directional_grid"]["grid_may_not_change_after_exact_scoring"]
    assert "candidate_exact_shimmer_db" in config[
        "forbidden_runtime_selector_inputs"
    ]
    assert "speaker_id" in config["forbidden_runtime_selector_inputs"]
    assert boundaries["no_speaker_or_case_hardcode"] is True
    assert boundaries["no_historical_threshold_change"] is True
    assert boundaries["no_final_waveform_highpass"] is True
    assert boundaries["generator_optimizer_steps"] == 0
    assert boundaries["authoritative_training_decision"] == (
        "NO_GO_AVQI_T2_TRAINING"
    )
