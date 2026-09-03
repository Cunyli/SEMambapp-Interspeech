"""Candidate-E Shimmer-dB backward used by the Route C joint panel.

The full-band waveform is never high-pass filtered.  The official Praat
stop-Hann branch is represented only inside the metric proxy, and detached
exact pulse topology is supplied by the sealed runtime worker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

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
COEFFICIENT_SMOOTHING_KERNEL = (0.25, 0.5, 0.25)
CANDIDATE_E_REFERENCE_SHA256 = (
    "e9266444fa1a8a9589471fb1edd08dbec020368e4dc984551eab304f20d4a9cf"
)
CANDIDATE_E_SOURCE_COMMIT = "109f398d607bce936a9576c826ff74ce0ea9f636"
CANDIDATE_E_RUNTIME_CLIENT_SHA256 = (
    "28e48fc3de99bb2c7258559f4f58be2760c7804f53a08bab162fff670b36153b"
)
CANDIDATE_E_WORKER_SHA256 = (
    "c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f"
)
CANDIDATE_E_SELECTOR_SHA256 = (
    "7401b4b80f6dbb546a4a88886c469bb4df6b4681bad9314f1244a046fbb2b69b"
)
CANDIDATE_E_RUNTIME_CONFIG_SHA256 = (
    "4dec4b018b6cd9f7a5a7f87966cc7f2dde057f152df256f65fc397faefb53b98"
)
CANDIDATE_E_TOPOLOGY_IMPLEMENTATION = (
    "exact_vectorized_frames_reused_tmpfs_numpy_sounding_v15"
)
CANDIDATE_E_NUMPY_HIGHPASS_MODE = (
    "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
)
PEAK_CERTIFICATE_NUMERICAL_SAFETY_FACTOR = 4096.0


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


def validate_candidate_e_base_peak_certificate(
    topology: Mapping[str, Any],
    proxy: CandidateEProxyResult,
) -> dict[str, Any]:
    """Validate the worker's exact fallback when the local peak bound is loose."""
    timing = topology.get("timing_ms")
    if not isinstance(timing, Mapping):
        raise ValueError("Candidate-E topology lacks a peak certificate")
    mode = timing.get("highpass_peak_check_mode")
    skipped = timing.get("highpass_sinc70_skipped")
    scaled = timing.get("highpass_peak_scaled")
    peak = timing.get("highpass_peak_value")
    sample_abs_max = timing.get("highpass_sample_abs_max")
    local_upper = timing.get("highpass_sinc70_peak_upper_bound")
    if (
        timing.get("highpass_mode") != CANDIDATE_E_NUMPY_HIGHPASS_MODE
        or topology.get("metric_highpass") != CANDIDATE_E_NUMPY_HIGHPASS_MODE
        or timing.get("highpass_sinc70_absolute_weight_bound")
        != SINC70_ABSOLUTE_WEIGHT_BOUND
        or not isinstance(skipped, bool)
        or not isinstance(scaled, bool)
        or isinstance(sample_abs_max, bool)
        or not isinstance(sample_abs_max, (int, float))
        or isinstance(local_upper, bool)
        or not isinstance(local_upper, (int, float))
    ):
        raise ValueError("Candidate-E base peak certificate fields differ")
    sample_abs_max = float(sample_abs_max)
    local_upper = float(local_upper)
    if (
        not math.isfinite(sample_abs_max)
        or sample_abs_max < 0.0
        or not math.isfinite(local_upper)
        or local_upper < 0.0
        or local_upper != sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
        or not math.isclose(
            sample_abs_max,
            proxy.metric_sample_abs_max,
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
        or not math.isclose(
            local_upper,
            proxy.sinc70_peak_upper_bound,
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("Candidate-E base peak certificate/proxy differs")

    exact_peak: float | None
    if mode == "proven_safe_sinc70_l1_upper_bound":
        if skipped is not True or peak is not None or scaled is not False:
            raise ValueError("Candidate-E safe-bound peak certificate differs")
        observed_or_bound = local_upper
        exact_peak = None
    elif mode == "exact_praat_sinc70":
        if (
            skipped is not False
            or isinstance(peak, bool)
            or not isinstance(peak, (int, float))
        ):
            raise ValueError("Candidate-E exact peak certificate differs")
        exact_peak = float(peak)
        if (
            not math.isfinite(exact_peak)
            or exact_peak < 0.0
            or exact_peak > math.nextafter(local_upper, math.inf)
            or scaled != (exact_peak > PEAK_SCALE_TRIGGER)
        ):
            raise ValueError("Candidate-E exact peak/scale decision differs")
        observed_or_bound = exact_peak
    else:
        raise ValueError("Candidate-E peak-check mode differs")

    numerical_epsilon = (
        PEAK_CERTIFICATE_NUMERICAL_SAFETY_FACTOR
        * np.finfo(np.float64).eps
        * max(observed_or_bound, 1.0)
    )
    certified_upper = math.nextafter(
        observed_or_bound + numerical_epsilon,
        math.inf,
    )
    if scaled or certified_upper >= PEAK_SCALE_TRIGGER:
        raise ValueError("Candidate-E base requires Praat peak scaling")
    return {
        "base_peak_check_mode": mode,
        "base_local_sinc70_peak_upper_bound": local_upper,
        "base_exact_sinc70_peak": exact_peak,
        "base_peak_numerical_epsilon": numerical_epsilon,
        "base_peak_upper_bound": certified_upper,
        "base_highpass_peak_scaled": False,
        "base_peak_scale_abstention_pass": True,
    }


def pcm16_ste(values: torch.Tensor) -> torch.Tensor:
    """Mirror libsndfile input PCM16 with an identity backward."""
    bounded = values.clamp(-1.0, 1.0 - 1.0 / 32768.0)
    quantized = torch.floor(bounded * 32768.0) / 32768.0
    return bounded + (quantized - bounded).detach()


def praat_pcm16_ste(values: torch.Tensor) -> torch.Tensor:
    """Mirror the Praat WAV-save PCM16 stage with an identity backward."""
    bounded = values.clamp(-1.0, 1.0 - 1.0 / 32768.0)
    quantized = torch.round(bounded * 32768.0) / 32768.0
    return bounded + (quantized - bounded).detach()


def next_power_of_two(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    return 1 << (sample_count - 1).bit_length()


def official_stop_hann(input_pcm16: torch.Tensor) -> tuple[torch.Tensor, int]:
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
        raise ValueError("Candidate-E source indices exceed waveform bounds")

    input_pcm16 = pcm16_ste(waveform)
    filtered, fft_sample_count = official_stop_hann(input_pcm16)
    sample_abs_max = float(filtered.detach().abs().max().cpu())
    sinc70_peak_upper_bound = sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
    peak_scale_abstention_pass = sinc70_peak_upper_bound < PEAK_SCALE_TRIGGER
    metric_pcm16_full = praat_pcm16_ste(filtered)
    mapped = metric_pcm16_full.index_select(0, source_indices)
    if metric_constant_prefix_samples:
        mapped = torch.cat(
            (mapped.new_zeros(metric_constant_prefix_samples), mapped)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _metric_source_indices(topology: Mapping[str, Any]) -> np.ndarray:
    if topology.get("topology_preprocessing") != (
        "exact_avqi_view_metric_waveform"
    ):
        raise ValueError("Candidate-E topology preprocessing differs")
    ranges = topology.get("metric_source_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Candidate-E topology has no metric source ranges")
    pieces = []
    previous_end = 0
    source_sample_count = int(topology["source_sample_count"])
    for start_value, length_value in ranges:
        start = int(start_value)
        length = int(length_value)
        end = start + length
        if length <= 0 or start < previous_end or end > source_sample_count:
            raise ValueError("Candidate-E metric source range is invalid")
        pieces.append(np.arange(start, end, dtype=np.int64))
        previous_end = end
    indices = np.concatenate(pieces)
    if indices.size != int(topology["metric_mapped_sample_count"]):
        raise ValueError("Candidate-E metric mapped sample count differs")
    expected_metric_samples = (
        int(topology["metric_constant_prefix_samples"]) + indices.size
    )
    if expected_metric_samples != int(topology["metric_sample_count"]):
        raise ValueError("Candidate-E metric waveform length differs")
    if (
        int(topology["metric_reconstruction_max_pcm16_error"]) != 0
        or int(topology["metric_reconstruction_differing_samples"]) != 0
    ):
        raise ValueError("Candidate-E metric source mapping failed parity")
    return indices


def _metric_range_layout(topology: Mapping[str, Any]) -> list[dict[str, int]]:
    cursor = int(topology["metric_constant_prefix_samples"])
    layout = []
    for index, (source_start, length) in enumerate(
        topology["metric_source_ranges"]
    ):
        source_start = int(source_start)
        length = int(length)
        layout.append(
            {
                "range_index": index,
                "metric_start_sample": cursor,
                "metric_end_sample": cursor + length,
                "source_start_sample": source_start,
                "source_end_sample": source_start + length,
                "length_samples": length,
            }
        )
        cursor += length
    if cursor != int(topology["metric_sample_count"]):
        raise ValueError("Candidate-E metric ranges do not cover the metric waveform")
    return layout


def build_cycle_gain_plan(
    waveform: np.ndarray,
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the promoted v18 zero-crossing plan in deterministic CPU order."""
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    source_indices = _metric_source_indices(topology)
    if (
        int(topology["source_sample_count"]) != values.size
        or source_indices.size == 0
        or int(source_indices.max()) >= values.size
    ):
        raise ValueError("Candidate-E cycle plan source indices exceed waveform")
    mapped_values = values[source_indices]
    pulses = np.asarray(topology["pulse_positions_samples"], dtype=np.float64)
    layout = _metric_range_layout(topology)
    prefix = int(topology["metric_constant_prefix_samples"])
    cell_ids = np.full(source_indices.size, -1, dtype=np.int64)
    cycles: list[dict[str, int]] = []
    groups: list[list[int]] = []
    crossing_rows: list[dict[str, Any]] = []
    failed_pulse_pairs = 0

    for range_row in layout:
        range_index = int(range_row["range_index"])
        metric_start = int(range_row["metric_start_sample"])
        metric_end = int(range_row["metric_end_sample"])
        mapped_start = metric_start - prefix
        length = int(range_row["length_samples"])
        range_values = mapped_values[mapped_start : mapped_start + length]
        pulse_indices = np.flatnonzero(
            (pulses >= metric_start) & (pulses < metric_end)
        )
        if pulse_indices.size < 3:
            continue

        adjacent_left = pulses[pulse_indices[:-1]]
        adjacent_right = pulses[pulse_indices[1:]]
        local_left = np.maximum(
            np.floor(adjacent_left - metric_start).astype(np.int64),
            0,
        )
        local_right = np.minimum(
            np.ceil(adjacent_right - metric_start).astype(np.int64),
            range_values.size - 1,
        )
        sample_indices = np.arange(
            max(range_values.size - 1, 0),
            dtype=np.int64,
        )
        left_values = range_values[:-1]
        right_values = range_values[1:]
        crossing_mask = (
            ((left_values <= 0.0) & (right_values > 0.0))
            | ((left_values >= 0.0) & (right_values < 0.0))
        )
        all_crossing_indices = sample_indices[crossing_mask]
        boundaries: list[dict[str, Any] | None] = [
            None for _ in range(pulse_indices.size - 1)
        ]
        if all_crossing_indices.size:
            crossing_left = range_values[all_crossing_indices]
            crossing_right = range_values[all_crossing_indices + 1]
            denominator = crossing_left - crossing_right
            fraction = np.divide(
                crossing_left,
                denominator,
                out=np.zeros_like(crossing_left),
                where=denominator != 0.0,
            )
            all_crossing_positions = (
                metric_start + all_crossing_indices + fraction
            )
            lower = np.searchsorted(all_crossing_indices, local_left, side="left")
            upper = np.searchsorted(all_crossing_indices, local_right, side="left")
            valid_pairs = upper > lower
            midpoints = 0.5 * (adjacent_left + adjacent_right)
            insertion = np.searchsorted(
                all_crossing_positions,
                midpoints,
                side="left",
            )
            maximum_crossing_index = all_crossing_indices.size - 1
            safe_lower = np.minimum(lower, maximum_crossing_index)
            safe_upper = np.minimum(
                np.maximum(upper - 1, 0),
                maximum_crossing_index,
            )
            left_choice = np.minimum(
                np.maximum(insertion - 1, safe_lower),
                safe_upper,
            )
            right_choice = np.minimum(
                np.maximum(insertion, safe_lower),
                safe_upper,
            )
            left_distance = np.abs(
                all_crossing_positions[left_choice] - midpoints
            )
            right_distance = np.abs(
                all_crossing_positions[right_choice] - midpoints
            )
            selected = np.where(
                left_distance <= right_distance,
                left_choice,
                right_choice,
            )
            for pair_offset in np.flatnonzero(valid_pairs):
                selected_crossing = int(selected[pair_offset])
                left_index = int(all_crossing_indices[selected_crossing])
                boundary = {
                    "mapped_right_index": mapped_start + left_index + 1,
                    "metric_position": float(
                        all_crossing_positions[selected_crossing]
                    ),
                    "left_abs_amplitude": float(abs(range_values[left_index])),
                    "right_abs_amplitude": float(
                        abs(range_values[left_index + 1])
                    ),
                }
                boundaries[int(pair_offset)] = boundary
                crossing_rows.append(
                    {
                        "range_index": range_index,
                        "left_pulse_index": int(pulse_indices[pair_offset]),
                        "right_pulse_index": int(
                            pulse_indices[pair_offset + 1]
                        ),
                        **boundary,
                    }
                )
        failed_pulse_pairs += sum(boundary is None for boundary in boundaries)

        boundary_right = np.asarray(
            [
                -1
                if boundary is None
                else int(boundary["mapped_right_index"])
                for boundary in boundaries
            ],
            dtype=np.int64,
        )
        starts = boundary_right[:-1]
        ends = boundary_right[1:]
        cycle_pulses = pulse_indices[1:-1]
        valid_cycles = (starts >= 0) & (ends >= 0)
        starts = starts[valid_cycles]
        ends = ends[valid_cycles]
        cycle_pulses = cycle_pulses[valid_cycles]
        if np.any(ends <= starts):
            raise ValueError("Candidate-E cycle plan overlaps or is empty")
        if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError("Candidate-E cycle plan overlaps or is empty")
        if not starts.size:
            continue

        cell_base = len(cycles)
        local_cell_ids = np.arange(starts.size, dtype=np.int64) + cell_base
        lengths = ends - starts
        repeated_starts = np.repeat(starts, lengths)
        repeated_offsets = np.repeat(
            np.cumsum(lengths) - lengths,
            lengths,
        )
        positions = (
            repeated_starts
            + np.arange(int(lengths.sum()), dtype=np.int64)
            - repeated_offsets
        )
        repeated_cells = np.repeat(local_cell_ids, lengths)
        if np.any(cell_ids[positions] >= 0):
            raise ValueError("Candidate-E cycle plan overlaps or is empty")
        cell_ids[positions] = repeated_cells
        cycles.extend(
            {
                "cell_id": int(cell_id),
                "range_index": range_index,
                "pulse_index": int(pulse_index),
                "mapped_start_index": int(start),
                "mapped_end_index": int(end),
                "sample_count": int(end - start),
            }
            for cell_id, pulse_index, start, end in zip(
                local_cell_ids,
                cycle_pulses,
                starts,
                ends,
                strict=True,
            )
        )
        contiguous = (
            (cycle_pulses[1:] == cycle_pulses[:-1] + 1)
            & (starts[1:] == ends[:-1])
        )
        split_points = np.flatnonzero(~contiguous) + 1
        groups.extend(
            group.astype(np.int64).tolist()
            for group in np.split(local_cell_ids, split_points)
            if group.size
        )

    cycle_count = len(cycles)
    previous_cells = np.arange(cycle_count, dtype=np.int64)
    next_cells = np.arange(cycle_count, dtype=np.int64)
    for group in groups:
        group_array = np.asarray(group, dtype=np.int64)
        previous_cells[group_array[1:]] = group_array[:-1]
        next_cells[group_array[:-1]] = group_array[1:]
    supported = np.flatnonzero(cell_ids >= 0)
    supported_source_indices = source_indices[supported]
    if np.unique(supported_source_indices).size != supported_source_indices.size:
        raise ValueError("Candidate-E plan maps one source sample more than once")
    boundary_amplitudes = np.asarray(
        [
            max(row["left_abs_amplitude"], row["right_abs_amplitude"])
            for row in crossing_rows
        ],
        dtype=np.float64,
    )
    waveform_rms = float(np.sqrt(np.mean(np.square(values))))
    return {
        "source_indices": source_indices,
        "cell_ids": cell_ids,
        "previous_cells": previous_cells,
        "next_cells": next_cells,
        "cycles": cycles,
        "groups": groups,
        "crossings": crossing_rows,
        "summary": {
            "pulse_count": int(pulses.size),
            "exact_source_range_count": len(layout),
            "zero_crossing_boundary_count": len(crossing_rows),
            "zero_crossing_failed_adjacent_pulse_pairs": failed_pulse_pairs,
            "complete_cycle_count": cycle_count,
            "complete_cycle_group_count": len(groups),
            "complete_cycle_pulse_fraction": cycle_count
            / max(int(pulses.size), 1),
            "supported_sample_count": int(supported.size),
            "supported_source_sample_fraction": supported.size
            / max(values.size, 1),
            "boundary_abs_amplitude_median": (
                float(np.median(boundary_amplitudes))
                if boundary_amplitudes.size
                else None
            ),
            "boundary_abs_amplitude_maximum": (
                float(np.max(boundary_amplitudes))
                if boundary_amplitudes.size
                else None
            ),
            "boundary_abs_amplitude_maximum_over_waveform_rms": (
                float(
                    np.max(boundary_amplitudes) / max(waveform_rms, 1e-12)
                )
                if boundary_amplitudes.size
                else None
            ),
            "coefficient_smoothing_kernel": list(
                COEFFICIENT_SMOOTHING_KERNEL
            ),
            "coefficient_smoothing_passes": 1,
            "source_range_joins_bridged": False,
        },
    }


def project_cycle_gain_gradient_fixed_order(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    plan: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the promoted cycle-gain projection with fixed-order reductions."""
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
