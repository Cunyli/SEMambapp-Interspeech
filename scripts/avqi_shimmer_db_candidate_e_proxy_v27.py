#!/usr/bin/env python3
"""Candidate-E exact-path fixed-topology Shimmer-dB differentiable proxy.

The emitted waveform is never high-passed.  This module mirrors the exact
metric branch only for backward construction: PCM16 input, official Praat
stop-Hann filtering, PCM16 metric-WAV output, detached exact CS/SV mapping,
and exact fixed-pulse Shimmer-dB amplitude statistics.  The two PCM16 stages
use a straight-through gradient while preserving their exact forward values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


SAMPLE_RATE = 16_000
STOP_HANN_LOW_HZ = 33.9
STOP_HANN_HIGH_HZ = 34.1
SINC70_ABSOLUTE_WEIGHT_BOUND = 5.2
PEAK_SCALE_TRIGGER = 0.999
MINIMUM_PERIOD_SECONDS = 0.0001
MAXIMUM_PERIOD_SECONDS = 0.02
MAXIMUM_PERIOD_FACTOR = 1.3
MAXIMUM_AMPLITUDE_FACTOR = 1.6
MINIMUM_COMPLETE_CYCLES = 16


@dataclass(frozen=True)
class CandidateEProxyResult:
    shimmer_db: torch.Tensor
    amplitudes: torch.Tensor
    amplitude_centers: torch.Tensor
    valid_pair_mask: torch.Tensor
    pair_contributions_db: torch.Tensor
    input_pcm16: torch.Tensor
    metric_pcm16: torch.Tensor
    fft_sample_count: int
    metric_sample_abs_max: float
    sinc70_peak_upper_bound: float
    peak_scale_abstention_pass: bool


def pcm16_ste(values: torch.Tensor) -> torch.Tensor:
    """Return exact PCM16 grid values with identity straight-through gradient."""
    bounded = values.clamp(-1.0, 1.0 - 1.0 / 32768.0)
    quantized = torch.round(bounded * 32768.0) / 32768.0
    return bounded + (quantized - bounded).detach()


def next_power_of_two(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    return 1 << (sample_count - 1).bit_length()


def official_stop_hann(
    input_pcm16: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mirror Praat 6.1.38 stop-Hann 0--34 Hz, 0.1-Hz smoothing."""
    if input_pcm16.ndim != 1:
        raise ValueError("stop-Hann input must be one-dimensional")
    fft_sample_count = next_power_of_two(input_pcm16.numel())
    spectrum = torch.fft.rfft(input_pcm16, n=fft_sample_count)
    frequencies = torch.fft.rfftfreq(
        fft_sample_count,
        d=1.0 / SAMPLE_RATE,
        device=input_pcm16.device,
        dtype=input_pcm16.dtype,
    )
    response = torch.ones_like(frequencies)
    response = torch.where(
        frequencies <= STOP_HANN_LOW_HZ,
        torch.zeros_like(response),
        response,
    )
    transition = (
        (frequencies > STOP_HANN_LOW_HZ)
        & (frequencies <= STOP_HANN_HIGH_HZ)
    )
    transition_response = 0.5 - 0.5 * torch.cos(
        math.pi
        / (STOP_HANN_HIGH_HZ - STOP_HANN_LOW_HZ)
        * (frequencies - STOP_HANN_LOW_HZ)
    )
    response = torch.where(transition, transition_response, response)
    filtered = torch.fft.irfft(
        spectrum * response,
        n=fft_sample_count,
    )[: input_pcm16.numel()]
    return filtered, fft_sample_count


def exact_metric_branch_ste(
    waveform: torch.Tensor,
    source_indices: torch.Tensor,
    metric_constant_prefix_samples: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct the exact-forward metric waveform with an STE Jacobian."""
    if waveform.ndim != 1:
        raise ValueError("Candidate-E waveform must be one-dimensional")
    if source_indices.ndim != 1 or source_indices.numel() == 0:
        raise ValueError("Candidate-E source indices must be nonempty and 1-D")
    if metric_constant_prefix_samples < 0:
        raise ValueError("metric constant prefix must be nonnegative")
    source_indices = source_indices.to(
        device=waveform.device,
        dtype=torch.long,
    ).detach()
    if bool((source_indices < 0).any()) or bool(
        (source_indices >= waveform.numel()).any()
    ):
        raise ValueError("metric source indices exceed waveform bounds")

    input_pcm16 = pcm16_ste(waveform)
    filtered, fft_sample_count = official_stop_hann(input_pcm16)
    sample_abs_max = float(filtered.detach().abs().max().cpu())
    sinc70_peak_upper_bound = sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
    peak_scale_abstention_pass = sinc70_peak_upper_bound < PEAK_SCALE_TRIGGER
    metric_pcm16_full = pcm16_ste(filtered)
    mapped = metric_pcm16_full.index_select(0, source_indices)
    if metric_constant_prefix_samples:
        mapped = torch.cat(
            (
                mapped.new_zeros(metric_constant_prefix_samples),
                mapped,
            )
        )
    return mapped, {
        "input_pcm16": input_pcm16,
        "metric_pcm16": metric_pcm16_full,
        "fft_sample_count": fft_sample_count,
        "metric_sample_abs_max": sample_abs_max,
        "sinc70_peak_upper_bound": sinc70_peak_upper_bound,
        "peak_scale_abstention_pass": peak_scale_abstention_pass,
    }


def asymmetric_hann_rms(
    prepared: torch.Tensor,
    centers: torch.Tensor,
    left_periods: torch.Tensor,
    right_periods: torch.Tensor,
) -> torch.Tensor:
    maximum_period_samples = SAMPLE_RATE / 50.0
    maximum_half_width = int(math.ceil(0.2 * maximum_period_samples))
    offsets = torch.arange(
        -maximum_half_width,
        maximum_half_width + 1,
        device=prepared.device,
    )
    anchor = centers.floor().long()
    sample_indices = anchor.unsqueeze(-1) + offsets.unsqueeze(0)
    valid_index = (sample_indices >= 0) & (sample_indices < prepared.numel())
    bounded_indices = sample_indices.clamp(0, prepared.numel() - 1)
    relative_position = (
        sample_indices.to(dtype=prepared.dtype) - centers.unsqueeze(-1)
    )
    left_width = (0.2 * left_periods).clamp_min(1e-12).unsqueeze(-1)
    right_width = (0.2 * right_periods).clamp_min(1e-12).unsqueeze(-1)
    width = torch.where(relative_position < 0.0, left_width, right_width)
    phase = relative_position / width
    support = valid_index & (phase >= -1.0) & (phase <= 1.0)
    window = (
        0.5 + 0.5 * torch.cos(math.pi * phase.clamp(-1.0, 1.0))
    ) * support.to(dtype=prepared.dtype)
    samples = prepared.index_select(0, bounded_indices.reshape(-1)).reshape(
        bounded_indices.shape
    )
    numerator = (samples * window).square().sum(dim=-1)
    denominator = window.square().sum(dim=-1).clamp_min(1e-24)
    return (numerator / denominator).clamp_min(0.0).sqrt()


def fixed_pulse_shimmer_db(
    prepared: torch.Tensor,
    pulse_positions: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    pulses = pulse_positions.to(
        device=prepared.device,
        dtype=prepared.dtype,
    ).detach()
    if pulses.ndim != 1 or pulses.numel() < 3:
        raise ValueError("Candidate-E needs at least three fixed pulses")
    previous_period = pulses[1:-1] - pulses[:-2]
    following_period = pulses[2:] - pulses[1:-1]
    minimum_period = MINIMUM_PERIOD_SECONDS * SAMPLE_RATE
    maximum_period = MAXIMUM_PERIOD_SECONDS * SAMPLE_RATE
    period_factor = torch.maximum(
        previous_period,
        following_period,
    ) / torch.minimum(previous_period, following_period).clamp_min(1e-24)
    valid_tier = (
        (previous_period >= minimum_period)
        & (previous_period <= maximum_period)
        & (following_period >= minimum_period)
        & (following_period <= maximum_period)
        & (period_factor <= MAXIMUM_PERIOD_FACTOR)
    )
    if not bool(valid_tier.any()):
        raise ValueError("Candidate-E fixed topology has no valid amplitudes")
    centers = pulses[1:-1][valid_tier]
    amplitudes = asymmetric_hann_rms(
        prepared,
        centers,
        previous_period[valid_tier],
        following_period[valid_tier],
    )
    positive = amplitudes.detach() > 0.0
    centers = centers[positive]
    amplitudes = amplitudes[positive]
    if amplitudes.numel() < 2:
        raise ValueError("Candidate-E fixed topology has fewer than two amplitudes")
    pair_period = centers[1:] - centers[:-1]
    amplitude_factor = torch.maximum(
        amplitudes[:-1],
        amplitudes[1:],
    ) / torch.minimum(amplitudes[:-1], amplitudes[1:]).clamp_min(1e-24)
    valid_pair = (
        (pair_period >= minimum_period)
        & (pair_period <= maximum_period)
        & (amplitude_factor.detach() <= MAXIMUM_AMPLITUDE_FACTOR)
    )
    if not bool(valid_pair.any()):
        raise ValueError("Candidate-E fixed topology has no valid amplitude pairs")
    contributions = 20.0 * torch.log10(
        amplitudes[1:].clamp_min(1e-24)
        / amplitudes[:-1].clamp_min(1e-24)
    ).abs()
    shimmer_db = contributions[valid_pair].mean()
    return shimmer_db, amplitudes, centers, valid_pair, contributions


def candidate_e_proxy(
    waveform: torch.Tensor,
    pulse_positions: torch.Tensor,
    source_indices: torch.Tensor,
    metric_constant_prefix_samples: int,
) -> CandidateEProxyResult:
    metric, certificate = exact_metric_branch_ste(
        waveform,
        source_indices,
        metric_constant_prefix_samples,
    )
    shimmer_db, amplitudes, centers, valid_pair, contributions = (
        fixed_pulse_shimmer_db(metric, pulse_positions)
    )
    return CandidateEProxyResult(
        shimmer_db=shimmer_db,
        amplitudes=amplitudes,
        amplitude_centers=centers,
        valid_pair_mask=valid_pair,
        pair_contributions_db=contributions,
        input_pcm16=certificate["input_pcm16"],
        metric_pcm16=certificate["metric_pcm16"],
        fft_sample_count=int(certificate["fft_sample_count"]),
        metric_sample_abs_max=float(certificate["metric_sample_abs_max"]),
        sinc70_peak_upper_bound=float(certificate["sinc70_peak_upper_bound"]),
        peak_scale_abstention_pass=bool(
            certificate["peak_scale_abstention_pass"]
        ),
    )


def project_cycle_gain_gradient_fixed_order(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    plan: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply Candidate-D cycle-gain math with a fixed-order CPU reduction."""
    if waveform.ndim != 1 or gradient.shape != waveform.shape:
        raise ValueError("Candidate-E projection expects matching 1-D tensors")
    cycles = list(plan["cycles"])
    cycle_count = len(cycles)
    if cycle_count < MINIMUM_COMPLETE_CYCLES:
        return torch.zeros_like(gradient), {
            **plan["summary"],
            "projection_reduction": "numpy_float64_fixed_cycle_order",
            "projected_gradient_valid": False,
        }
    source_indices = np.asarray(plan["source_indices"], dtype=np.int64)
    waveform_np = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
    coefficients = np.zeros(cycle_count, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for cycle in cycles:
        cell_id = int(cycle["cell_id"])
        start = int(cycle["mapped_start_index"])
        end = int(cycle["mapped_end_index"])
        indices = source_indices[start:end]
        reference = waveform_np[indices]
        raw = gradient_np[indices]
        numerator = np.sum(raw * reference, dtype=np.float64)
        denominator = np.sum(reference * reference, dtype=np.float64)
        coefficients[cell_id] = numerator / max(denominator, tiny)
    previous_cells = np.asarray(plan["previous_cells"], dtype=np.int64)
    next_cells = np.asarray(plan["next_cells"], dtype=np.int64)
    smoothed = (
        coefficients[previous_cells]
        + 2.0 * coefficients
        + coefficients[next_cells]
    ) / 4.0
    projected_np = np.zeros(waveform_np.shape, dtype=np.float64)
    for cycle in cycles:
        cell_id = int(cycle["cell_id"])
        start = int(cycle["mapped_start_index"])
        end = int(cycle["mapped_end_index"])
        indices = source_indices[start:end]
        projected_np[indices] = smoothed[cell_id] * waveform_np[indices]
    projected = torch.from_numpy(projected_np).to(
        device=waveform.device,
        dtype=waveform.dtype,
    )
    norm = float(projected.detach().norm().cpu())
    finite = bool(torch.isfinite(projected).all().detach().cpu())
    return projected, {
        **plan["summary"],
        "projection_reduction": "numpy_float64_fixed_cycle_order",
        "raw_cycle_coefficient_minimum": float(coefficients.min()),
        "raw_cycle_coefficient_median": float(np.median(coefficients)),
        "raw_cycle_coefficient_maximum": float(coefficients.max()),
        "smoothed_cycle_coefficient_minimum": float(smoothed.min()),
        "smoothed_cycle_coefficient_median": float(np.median(smoothed)),
        "smoothed_cycle_coefficient_maximum": float(smoothed.max()),
        "projected_gradient_l2_norm": norm,
        "projected_gradient_finite": finite,
        "projected_gradient_valid": finite and norm > 0.0,
    }


def normalized_gradient_step(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    gradient_rms = gradient.square().mean().sqrt()
    if float(gradient_rms.detach().cpu()) <= 1e-15:
        raise ValueError("Candidate-E projected gradient is numerically zero")
    base_rms = waveform.square().mean().sqrt()
    return waveform.detach() - alpha * base_rms * gradient / gradient_rms
