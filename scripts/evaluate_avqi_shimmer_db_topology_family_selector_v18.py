#!/usr/bin/env python3
"""Audit and run the frozen D-then-C Shimmer-dB topology-family selector.

Candidate D is the v17 zero-crossing shape-preserving projection at alpha
0.001.  Candidate C is the v16 direct waveform gradient with the frozen
four-level half-step ladder.  Candidate D is always attempted first; Candidate
C is evaluated only when D fails a preregistered non-outcome certificate.
Candidate exact AVQI outcomes remain closed until all four opened-dev cases
select within the unchanged 500-ms end-to-end metric-step contract.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    require_exact_topology_equal,
    topology_sha256,
)
from scripts.diagnose_avqi_shimmer_db_pulse_alignment_v17 import (
    metric_range_layout,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    MATERIAL_GAP_THRESHOLD,
    SAMPLE_RATE,
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    component_fields,
    exact_components,
    load_predictor,
    metric_source_indices_from_topology,
    normalized_gradient_step,
    read_waveform,
    run_exact,
    sha256_file,
    topology_stability,
    waveform_safety,
    write_csv,
    write_json,
)
from scripts.evaluate_avqi_shimmer_db_cycle_projected_backward import (
    exact_vector,
    read_csv,
    repository_head,
    validate_hash,
    validate_panel,
)
from scripts.evaluate_avqi_shimmer_db_source_informed_v17 import (
    CANDIDATE_D_NAME,
    COEFFICIENT_SMOOTHING_KERNEL,
    MINIMUM_COMPLETE_CYCLES,
    build_zero_crossing_cycle_plan as build_legacy_candidate_d_plan,
    synthetic_candidate_d_warmup,
    zero_crossing_shape_preserving_gradient_projection as legacy_candidate_d_projection,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    ALPHA_LADDER,
    PCM24_MIN_CHANGED_SAMPLES,
    PCM24_MIN_RESIDUAL_RMS_LSB,
    PROTOTYPE_CASE_IDS,
    SELECTOR_KEYS as C_SELECTOR_KEYS,
    TRUST_REGION_CANDIDATE_NAME,
    base_topology_item,
    finite_safety,
    pcm24_effective_step,
    select_topology_certified_step,
    selector_view as c_selector_view,
    synthetic_torch_warmup,
    validate_dev_files,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RUNTIME_MAX_MS,
    FIXED_ALPHA,
    GRADIENT_NORM_RANGE,
    IMPROVEMENT_FRACTION_GATE,
    MAXIMUM_CLIP_FRACTION,
    MEDIAN_REDUCTION_GATE,
    MINIMUM_COSINE,
    NONSELECTED_MEDIAN_INCREASE_GATE,
    RESIDUAL_CEILING_DB,
    aggregate_candidate,
)
from scripts.evaluate_direct_avqi_waveform_optimization import (
    aggregate_denoising,
    aggregate_pathology_guardrails,
    full_band_pathology_guardrails,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_NAME = "praat_current_output_topology_family_selector_d_then_c_v18"
V16_SOURCE_SHA256 = "d8bfb0f31d9d98832d6c4409e5044b5d7cbe0b8b585e72f359fa3119d22aa662"
V17_SOURCE_SHA256 = "324660709b2e6a4994d057c4d532cf89613f535ec96490f2cb038d7b33f55b22"
EXPECTED_CASE_COUNT = len(PROTOTYPE_CASE_IDS)
WORKER_COUNT = len(ALPHA_LADDER)
PROXY_GAP_TOLERANCE = 1e-7
EQUIVALENCE_EPSILON_MULTIPLIER = 8.0
D_ROUTING_KEYS = frozenset(
    {
        "projected_gradient_valid",
        "complete_cycle_support_pass",
        "proxy_nonregression_pass",
        "finite_safety_pass",
        "pcm24_effective_step_pass",
        "topology_stability_pass",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("equivalence", "selector4"), required=True)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--fresh-results", type=Path, required=True)
    parser.add_argument("--fresh-results-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--runtime-worker-script-sha256", required=True)
    parser.add_argument("--equivalence-report", type=Path)
    parser.add_argument("--equivalence-report-sha256")
    parser.add_argument("--equivalence-receipt", type=Path)
    parser.add_argument("--equivalence-receipt-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def candidate_topology_item(
    case_id: str,
    view: str,
    path: Path,
    attempt_id: str,
) -> dict[str, Any]:
    return {
        "id": f"v18-topology:{case_id}:{attempt_id}",
        "case_id": case_id,
        "role": "current_output_topology",
        "path": str(path.resolve()),
        "view": view,
        "score_components": False,
        "exact_metric_topology": True,
        "highpass_mode": NUMPY_HIGHPASS_MODE,
    }


def build_zero_crossing_cycle_plan_vectorized(
    waveform: np.ndarray,
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Exact-equivalent v17 plan with vectorized crossing and cell assignment."""
    values = np.asarray(waveform, dtype=np.float64)
    source_indices = metric_source_indices_from_topology(
        topology,
        source_sample_count=values.size,
    )
    mapped_values = values[source_indices]
    pulses = np.asarray(topology["pulse_positions_samples"], dtype=np.float64)
    layout = metric_range_layout(topology)
    prefix = int(topology["metric_constant_prefix_samples"])
    cell_ids = np.full(source_indices.size, -1, dtype=np.int64)
    cycles: list[dict[str, Any]] = []
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
        sample_indices = np.arange(max(range_values.size - 1, 0), dtype=np.int64)
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
            lower = np.searchsorted(
                all_crossing_indices,
                local_left,
                side="left",
            )
            upper = np.searchsorted(
                all_crossing_indices,
                local_right,
                side="left",
            )
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
                        "right_pulse_index": int(pulse_indices[pair_offset + 1]),
                        **boundary,
                    }
                )
        failed_pulse_pairs += sum(boundary is None for boundary in boundaries)

        boundary_right = np.asarray(
            [
                -1 if boundary is None else int(boundary["mapped_right_index"])
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
            raise ValueError("zero-crossing cycle plan overlaps or is empty")
        if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError("zero-crossing cycle plan overlaps or is empty")
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
            raise ValueError("zero-crossing cycle plan overlaps or is empty")
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
        raise ValueError("zero-crossing plan maps one source sample more than once")
    boundary_amplitudes = np.asarray(
        [
            max(row["left_abs_amplitude"], row["right_abs_amplitude"])
            for row in crossing_rows
        ],
        dtype=np.float64,
    )
    waveform_rms = float(np.sqrt(np.mean(np.square(values))))
    summary = {
        "pulse_count": int(pulses.size),
        "exact_source_range_count": len(layout),
        "zero_crossing_boundary_count": len(crossing_rows),
        "zero_crossing_failed_adjacent_pulse_pairs": failed_pulse_pairs,
        "complete_cycle_count": cycle_count,
        "complete_cycle_group_count": len(groups),
        "complete_cycle_pulse_fraction": cycle_count / max(int(pulses.size), 1),
        "supported_sample_count": int(supported.size),
        "supported_source_sample_fraction": supported.size / max(values.size, 1),
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
            float(np.max(boundary_amplitudes) / max(waveform_rms, 1e-12))
            if boundary_amplitudes.size
            else None
        ),
        "coefficient_smoothing_kernel": list(COEFFICIENT_SMOOTHING_KERNEL),
        "coefficient_smoothing_passes": 1,
        "source_range_joins_bridged": False,
    }
    return {
        "source_indices": source_indices,
        "cell_ids": cell_ids,
        "previous_cells": previous_cells,
        "next_cells": next_cells,
        "cycles": cycles,
        "groups": groups,
        "crossings": crossing_rows,
        "summary": summary,
    }


def candidate_d_projection_vectorized(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    plan: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preserve v17 projection math while collecting GPU scalars once."""
    if waveform.ndim != 1 or gradient.shape != waveform.shape:
        raise ValueError("Candidate-D projection expects matching 1-D tensors")
    cell_ids_np = np.asarray(plan["cell_ids"], dtype=np.int64)
    supported_mapped = np.flatnonzero(cell_ids_np >= 0)
    cycle_count = int(plan["summary"]["complete_cycle_count"])
    if cycle_count < MINIMUM_COMPLETE_CYCLES or supported_mapped.size == 0:
        return torch.zeros_like(gradient), {
            **plan["summary"],
            "projected_gradient_valid": False,
        }
    source_indices_np = np.asarray(plan["source_indices"], dtype=np.int64)
    source_indices = torch.as_tensor(
        source_indices_np[supported_mapped],
        dtype=torch.long,
        device=waveform.device,
    )
    cells = torch.as_tensor(
        cell_ids_np[supported_mapped],
        dtype=torch.long,
        device=waveform.device,
    )
    reference = waveform.detach().index_select(0, source_indices)
    raw = gradient.detach().index_select(0, source_indices)
    numerators = waveform.new_zeros(cycle_count)
    denominators = waveform.new_zeros(cycle_count)
    numerators.scatter_add_(0, cells, raw * reference)
    denominators.scatter_add_(0, cells, reference.square())
    coefficients = torch.where(
        denominators > torch.finfo(waveform.dtype).tiny,
        numerators / denominators.clamp_min(torch.finfo(waveform.dtype).tiny),
        torch.zeros_like(numerators),
    )
    previous_cells = torch.as_tensor(
        plan["previous_cells"],
        dtype=torch.long,
        device=waveform.device,
    )
    next_cells = torch.as_tensor(
        plan["next_cells"],
        dtype=torch.long,
        device=waveform.device,
    )
    smoothed = (
        coefficients.index_select(0, previous_cells)
        + 2.0 * coefficients
        + coefficients.index_select(0, next_cells)
    ) / 4.0
    projected = torch.zeros_like(gradient)
    projected.index_copy_(
        0,
        source_indices,
        smoothed.index_select(0, cells) * reference,
    )
    statistics = torch.stack(
        (
            coefficients.min(),
            coefficients.median(),
            coefficients.max(),
            smoothed.min(),
            smoothed.median(),
            smoothed.max(),
            projected.norm(),
            torch.isfinite(projected).all().to(projected.dtype),
        )
    ).detach().cpu().tolist()
    norm = float(statistics[6])
    finite = bool(statistics[7])
    return projected, {
        **plan["summary"],
        "raw_cycle_coefficient_minimum": float(statistics[0]),
        "raw_cycle_coefficient_median": float(statistics[1]),
        "raw_cycle_coefficient_maximum": float(statistics[2]),
        "smoothed_cycle_coefficient_minimum": float(statistics[3]),
        "smoothed_cycle_coefficient_median": float(statistics[4]),
        "smoothed_cycle_coefficient_maximum": float(statistics[5]),
        "projected_gradient_l2_norm": norm,
        "projected_gradient_finite": finite,
        "projected_gradient_valid": finite and norm > 0.0,
    }


def normalized_gradient_steps_shared(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    alphas: tuple[float, ...],
) -> list[torch.Tensor]:
    """Reuse reductions while retaining the frozen normalized-step expression."""
    gradient_rms = gradient.square().mean().sqrt()
    base_rms = waveform.square().mean().sqrt()
    if float(gradient_rms) <= 1e-15:
        return [waveform.detach().clone() for _ in alphas]
    return [
        waveform.detach() - alpha * base_rms * gradient / gradient_rms
        for alpha in alphas
    ]


def plan_equivalence(
    legacy: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    array_keys = ("source_indices", "cell_ids", "previous_cells", "next_cells")
    arrays_equal = {
        key: bool(np.array_equal(legacy[key], optimized[key]))
        for key in array_keys
    }
    object_equal = {
        "cycles": legacy["cycles"] == optimized["cycles"],
        "groups": legacy["groups"] == optimized["groups"],
        "crossings": legacy["crossings"] == optimized["crossings"],
        "summary": legacy["summary"] == optimized["summary"],
    }
    return {
        "array_equal": arrays_equal,
        "object_equal": object_equal,
        "all_equal": all(arrays_equal.values()) and all(object_equal.values()),
    }


def tensor_equivalence(
    reference: torch.Tensor,
    optimized: torch.Tensor,
) -> dict[str, Any]:
    if reference.shape != optimized.shape or reference.dtype != optimized.dtype:
        return {
            "bit_equal": False,
            "numerically_equivalent": False,
            "maximum_abs_difference": math.inf,
            "absolute_tolerance": 0.0,
        }
    difference = (reference.detach() - optimized.detach()).abs()
    maximum_difference = float(difference.max())
    reference_scale = float(reference.detach().abs().max())
    absolute_tolerance = (
        EQUIVALENCE_EPSILON_MULTIPLIER
        * torch.finfo(reference.dtype).eps
        * max(reference_scale, torch.finfo(reference.dtype).eps)
    )
    return {
        "bit_equal": bool(torch.equal(reference, optimized)),
        "numerically_equivalent": maximum_difference <= absolute_tolerance,
        "maximum_abs_difference": maximum_difference,
        "absolute_tolerance": absolute_tolerance,
    }


def projection_report_equivalence(
    reference: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    if set(reference) != set(optimized):
        return {"keys_equal": False, "all_equivalent": False}
    comparisons: dict[str, bool] = {}
    numeric_diagnostics: dict[str, dict[str, float]] = {}
    epsilon = torch.finfo(torch.float32).eps
    for key, reference_value in reference.items():
        optimized_value = optimized[key]
        if isinstance(reference_value, float) and isinstance(
            optimized_value,
            float,
        ):
            tolerance = (
                EQUIVALENCE_EPSILON_MULTIPLIER
                * epsilon
                * max(abs(reference_value), abs(optimized_value), 1.0)
            )
            numeric_diagnostics[key] = {
                "reference": reference_value,
                "optimized": optimized_value,
                "absolute_difference": abs(reference_value - optimized_value),
                "absolute_tolerance": tolerance,
            }
            comparisons[key] = math.isclose(
                reference_value,
                optimized_value,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        else:
            comparisons[key] = reference_value == optimized_value
    return {
        "keys_equal": True,
        "field_equivalence": comparisons,
        "numeric_diagnostics": numeric_diagnostics,
        "all_equivalent": all(comparisons.values()),
    }


def pcm24_codes(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.rint(np.asarray(values, dtype=np.float64) * float(2**23)),
        -(2**23),
        2**23 - 1,
    ).astype(np.int64)


def pcm24_metrics_from_loaded(
    base_codes: np.ndarray,
    base_sha256: str,
    candidate_values: np.ndarray,
    candidate_sha256: str,
) -> dict[str, Any]:
    candidate_codes = pcm24_codes(candidate_values)
    difference_lsb = (candidate_codes - base_codes).astype(np.float64)
    changed = int(np.count_nonzero(difference_lsb))
    residual_rms_lsb = float(np.sqrt(np.mean(np.square(difference_lsb))))
    sha_differs = base_sha256 != candidate_sha256
    passed = (
        sha_differs
        and changed >= PCM24_MIN_CHANGED_SAMPLES
        and residual_rms_lsb >= PCM24_MIN_RESIDUAL_RMS_LSB
    )
    return {
        "pcm24_sha_differs_from_base": sha_differs,
        "pcm24_changed_samples": changed,
        "pcm24_changed_fraction": changed / max(int(base_codes.size), 1),
        "pcm24_residual_rms_lsb": residual_rms_lsb,
        "pcm24_effective_step_pass": passed,
    }


def select_candidate_d(certificates: dict[str, Any]) -> bool:
    if set(certificates) != D_ROUTING_KEYS:
        raise ValueError("v18 Candidate-D routing contract drift")
    return all(bool(certificates[key]) for key in D_ROUTING_KEYS)


def selector_contract() -> dict[str, Any]:
    return {
        "candidate": SELECTOR_NAME,
        "family_order": [CANDIDATE_D_NAME, TRUST_REGION_CANDIDATE_NAME],
        "candidate_d_alpha": FIXED_ALPHA,
        "candidate_c_alpha_ladder": list(ALPHA_LADDER),
        "candidate_d_routing_keys": sorted(D_ROUTING_KEYS),
        "candidate_c_selector_keys": sorted(C_SELECTOR_KEYS),
        "forbidden_information": [
            "candidate_exact_shimmer_db",
            "candidate_exact_avqi_components",
            "candidate_other_exact_outcomes",
            "speaker_id",
            "case_id",
            "severity",
            "condition",
            "view",
        ],
        "base_refreshes_per_case": 1,
        "raw_gradients_per_case": 1,
        "candidate_d_always_attempted": True,
        "candidate_c_complete_ladder_on_fallback": True,
        "formal_total_metric_step_runtime_ms": CACHE_RUNTIME_MAX_MS,
        "equivalence_numeric_contract": {
            "gradient_and_step_absolute_tolerance": (
                "8 * dtype_epsilon * max_abs_reference"
            ),
            "runtime_projection_backend": (
                "CUDA; independent legacy/optimized tensors must pass the "
                "frozen numerical tolerance"
            ),
            "diagnostic_report_backend": (
                "single-thread CPU; legacy/optimized projection tensors and "
                "report fields must be exact-equivalent"
            ),
            "diagnostic_scalar_absolute_tolerance": (
                "8 * float32_epsilon * max(abs(reference), abs(optimized), 1)"
            ),
            "candidate_pcm24_file_sha_must_match": True,
            "candidate_topology_sha_must_match": True,
            "frozen_proxy_must_match": True,
        },
        "runtime_includes": [
            "base_read_and_tmpfs_exact_topology_refresh",
            "shared_raw_gradient_and_frozen_proxy",
            "candidate_d_plan_projection_pcm24_proxy_safety_topology",
            "all_candidate_c_pcm24_proxy_safety_topologies_when_fallback_entered",
            "device_synchronization_and_selector",
        ],
    }


def validate_equivalence_receipt(args: argparse.Namespace) -> dict[str, str]:
    required = (
        args.equivalence_report,
        args.equivalence_report_sha256,
        args.equivalence_receipt,
        args.equivalence_receipt_sha256,
    )
    if any(value is None for value in required):
        raise ValueError("selector4 requires hash-bound equivalence evidence")
    assert args.equivalence_report is not None
    assert args.equivalence_report_sha256 is not None
    assert args.equivalence_receipt is not None
    assert args.equivalence_receipt_sha256 is not None
    hashes = {
        "equivalence_report": validate_hash(
            args.equivalence_report,
            args.equivalence_report_sha256,
            "v18 equivalence report",
        ),
        "equivalence_receipt": validate_hash(
            args.equivalence_receipt,
            args.equivalence_receipt_sha256,
            "v18 equivalence receipt",
        ),
    }
    report = read_json(args.equivalence_report)
    receipt = read_json(args.equivalence_receipt)
    if report.get("decision") != "PASS_SHIMMER_DB_V18_FAMILY_EQUIVALENCE":
        raise ValueError("v18 family equivalence has not passed")
    if receipt.get("decision") != report.get("decision"):
        raise ValueError("v18 equivalence report/receipt decision drift")
    if report.get("source_commit") != args.source_commit:
        raise ValueError("v18 equivalence was run from a different source commit")
    if report.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("v18 equivalence opened forbidden candidate outcomes")
    return hashes


def synthetic_v18_warmup(device: torch.device) -> dict[str, Any]:
    """Prime only v18 plan/projection/step code on a synthetic waveform."""
    started = time.perf_counter()
    sample_count = 5 * SAMPLE_RATE
    timeline = np.arange(sample_count, dtype=np.float32)
    waveform_values = (
        0.08 * np.sin(2.0 * np.pi * 125.0 * timeline / SAMPLE_RATE)
    ).astype(np.float32)
    pulse_positions = np.arange(64, sample_count - 64, 128, dtype=np.float64)
    topology = {
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": sample_count,
        "metric_sample_count": sample_count,
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": [[0, sample_count]],
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": sample_count,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": pulse_positions.tolist(),
        "pulse_count": int(pulse_positions.size),
    }
    plan = build_zero_crossing_cycle_plan_vectorized(
        waveform_values,
        topology,
    )
    waveform = torch.from_numpy(waveform_values).to(device)
    modulation = 1.0 + 0.2 * torch.sin(
        2.0 * torch.pi * torch.arange(sample_count, device=device) / 2048.0
    )
    gradient = waveform * modulation
    projected, projection = candidate_d_projection_vectorized(
        waveform,
        gradient,
        plan,
    )
    candidate = normalized_gradient_steps_shared(
        waveform,
        projected,
        (FIXED_ALPHA,),
    )[0]
    synchronize(device)
    return {
        "synthetic_only": True,
        "panel_or_training_waveform_used": False,
        "sample_count": sample_count,
        "pulse_count": int(pulse_positions.size),
        "complete_cycle_count": projection["complete_cycle_count"],
        "projected_gradient_valid": projection["projected_gradient_valid"],
        "candidate_finite": bool(torch.isfinite(candidate).all()),
        "runtime_ms": 1000.0 * (time.perf_counter() - started),
    }


def validate_sources_and_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, str]]:
    if repository_head() != args.source_commit:
        raise ValueError("v18 source commit differs from repository HEAD")
    source_hashes = {
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "opened panel contract",
        ),
        "fresh_results": validate_hash(
            args.fresh_results,
            args.fresh_results_sha256,
            "opened fresh-panel results",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen Shimmer predictor",
        ),
        "runtime_worker": validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "exact topology worker",
        ),
        "v16_family_source": validate_hash(
            REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_db_trust_region_v16.py",
            V16_SOURCE_SHA256,
            "frozen v16 family source",
        ),
        "v17_family_source": validate_hash(
            REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_db_source_informed_v17.py",
            V17_SOURCE_SHA256,
            "frozen v17 family source",
        ),
    }
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_tree_hash
    if args.phase == "selector4":
        source_hashes.update(validate_equivalence_receipt(args))

    panel = read_json(args.panel_contract)
    input_results = read_csv(args.fresh_results)
    full_panel_rows, input_by_case = validate_panel(panel, input_results)
    full_by_case = {row["case_id"]: row for row in full_panel_rows}
    if not set(PROTOTYPE_CASE_IDS).issubset(full_by_case):
        raise ValueError("v18 four-case coverage drift")
    panel_rows = [full_by_case[case_id] for case_id in PROTOTYPE_CASE_IDS]
    validate_dev_files(panel_rows)
    return panel_rows, input_by_case, source_hashes


def prepare_case_context(
    panel_row: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    worker: ExactShimmerTopologyWorker,
) -> dict[str, Any]:
    base_path = Path(panel_row["base_path"])
    base_waveform = read_waveform(base_path)
    base_values = base_waveform.numpy()
    base_rows, request_wall_ms, staging = worker.refresh_current_waveforms(
        [base_topology_item(panel_row)],
        [base_values],
        highpass_mode=NUMPY_HIGHPASS_MODE,
    )
    base_topology = dict(base_rows[0])
    waveform = base_waveform.to(device).requires_grad_(True)
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            base_topology,
            source_sample_count=waveform.numel(),
        ),
        dtype=torch.long,
        device=device,
    )
    pulses = waveform.new_tensor(base_topology["pulse_positions_samples"])
    synchronize(device)
    gradient_started = time.perf_counter()
    proxy_before = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            base_topology["metric_constant_prefix_samples"]
        ),
    )[1]
    scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    loss = ((proxy_before - target_shimmer_db) / scale).square()
    raw_gradient = torch.autograd.grad(loss, waveform)[0]
    raw_statistics = torch.stack(
        (
            raw_gradient.norm(),
            raw_gradient.square().mean().sqrt(),
            torch.isfinite(raw_gradient).all().to(raw_gradient.dtype),
        )
    ).detach().cpu().tolist()
    if not bool(raw_statistics[2]) or float(raw_statistics[0]) <= 0.0:
        raise ValueError(f"invalid v18 raw gradient: {panel_row['case_id']}")
    return {
        "case_id": panel_row["case_id"],
        "panel_row": panel_row,
        "base_path": base_path,
        "base_waveform": base_waveform,
        "base_values": base_values,
        "base_sha256": sha256_file(base_path),
        "base_codes": pcm24_codes(base_values),
        "base_topology": base_topology,
        "base_topology_sha256": topology_sha256(base_topology),
        "base_refresh_request_wall_ms": request_wall_ms,
        "base_refresh_client_staging_ms": float(staging[0]["staging_ms"]),
        "base_refresh_runtime_ms": request_wall_ms
        + float(staging[0]["staging_ms"]),
        "waveform": waveform,
        "source_indices": source_indices,
        "pulses": pulses,
        "proxy_before_tensor": proxy_before,
        "proxy_before": float(proxy_before.detach()),
        "proxy_target": float(target_shimmer_db),
        "proxy_loss": float(loss.detach()),
        "scale_value": float(scale),
        "raw_gradient": raw_gradient,
        "raw_gradient_l2_norm": float(raw_statistics[0]),
        "raw_gradient_rms": float(raw_statistics[1]),
        "gradient_runtime_ms": 1000.0 * (time.perf_counter() - gradient_started),
    }


def materialize_candidate_pcm24(
    context: dict[str, Any],
    values: np.ndarray,
    path: Path,
    attempt_id: str,
) -> tuple[dict[str, Any], float, float]:
    write_started = time.perf_counter()
    sf.write(path, values, SAMPLE_RATE, subtype="PCM_24")
    write_ms = 1000.0 * (time.perf_counter() - write_started)
    read_started = time.perf_counter()
    stored, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if (
        sample_rate != SAMPLE_RATE
        or stored.ndim != 1
        or stored.shape != context["base_values"].shape
    ):
        raise ValueError("v18 stored PCM24 waveform shape drift")
    stored = np.asarray(stored, dtype=np.float32)
    candidate_sha256 = sha256_file(path)
    record = {
        "attempt_id": attempt_id,
        "candidate_path": path,
        "candidate_sha256": candidate_sha256,
        "stored_waveform": stored,
        **finite_safety(context["base_values"], stored),
        **pcm24_metrics_from_loaded(
            context["base_codes"],
            context["base_sha256"],
            stored,
            candidate_sha256,
        ),
    }
    read_safety_pcm_ms = 1000.0 * (time.perf_counter() - read_started)
    return record, write_ms, read_safety_pcm_ms


def write_candidate_batch(
    context: dict[str, Any],
    candidates: list[torch.Tensor],
    paths: list[Path],
    attempt_ids: list[str],
    predictor: torch.nn.Module,
    device: torch.device,
    executor: ThreadPoolExecutor,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not (len(candidates) == len(paths) == len(attempt_ids)):
        raise ValueError("v18 candidate batch cardinality drift")
    transfer_started = time.perf_counter()
    candidate_batch = torch.stack(candidates).detach().cpu().numpy()
    transfer_ms = 1000.0 * (time.perf_counter() - transfer_started)
    io_started = time.perf_counter()
    inputs = list(zip(candidate_batch, paths, attempt_ids, strict=True))
    if len(inputs) > 1:
        futures = [
            executor.submit(
                materialize_candidate_pcm24,
                context,
                values,
                path,
                attempt_id,
            )
            for values, path, attempt_id in inputs
        ]
        materialized = [future.result() for future in futures]
        io_mode = "persistent_thread_pool"
    else:
        materialized = [
            materialize_candidate_pcm24(
                context,
                values,
                path,
                attempt_id,
            )
            for values, path, attempt_id in inputs
        ]
        io_mode = "direct_single_candidate"
    io_wall_ms = 1000.0 * (time.perf_counter() - io_started)
    records = [row[0] for row in materialized]
    write_ms = sum(row[1] for row in materialized)
    read_safety_pcm_ms = sum(row[2] for row in materialized)

    proxy_started = time.perf_counter()
    proxy_tensors = []
    with torch.inference_mode():
        for record in records:
            stored_tensor = torch.from_numpy(record["stored_waveform"]).to(device)
            proxy_tensors.append(
                predictor.raw_shimmer_from_pulse_positions(
                    stored_tensor,
                    context["pulses"],
                    metric_source_indices=context["source_indices"],
                    metric_constant_prefix_samples=int(
                        context["base_topology"][
                            "metric_constant_prefix_samples"
                        ]
                    ),
                )[1]
            )
    synchronize(device)
    normalized_gap_before = abs(
        context["proxy_before"] - context["proxy_target"]
    ) / max(context["scale_value"], 1e-8)
    for record, proxy_tensor in zip(records, proxy_tensors, strict=True):
        proxy_after = float(proxy_tensor)
        normalized_gap_after = abs(
            proxy_after - context["proxy_target"]
        ) / max(context["scale_value"], 1e-8)
        record.update(
            {
                "proxy_before": context["proxy_before"],
                "proxy_after_frozen_topology": proxy_after,
                "proxy_target": context["proxy_target"],
                "normalized_proxy_gap_before": normalized_gap_before,
                "normalized_proxy_gap_after": normalized_gap_after,
                "proxy_nonregression_pass": (
                    normalized_gap_after
                    <= normalized_gap_before + PROXY_GAP_TOLERANCE
                ),
            }
        )
    proxy_ms = 1000.0 * (time.perf_counter() - proxy_started)
    return records, {
        "candidate_gpu_to_cpu_batch_ms": transfer_ms,
        "candidate_pcm24_io_mode": io_mode,
        "candidate_pcm24_io_concurrent_wall_ms": io_wall_ms,
        "candidate_pcm24_write_total_ms": write_ms,
        "candidate_read_safety_pcm_total_ms": read_safety_pcm_ms,
        "candidate_frozen_proxy_batch_ms": proxy_ms,
    }


def refresh_candidate_records(
    context: dict[str, Any],
    records: list[dict[str, Any]],
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
) -> dict[str, float]:
    if not records or len(workers) != WORKER_COUNT:
        raise ValueError("v18 candidate refresh worker contract drift")
    groups: list[list[dict[str, Any]]] = [[] for _ in workers]
    for index, record in enumerate(records):
        groups[index % len(workers)].append(record)

    def refresh_group(
        worker: ExactShimmerTopologyWorker,
        grouped_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        items = [
            candidate_topology_item(
                context["case_id"],
                context["panel_row"]["view"],
                record["candidate_path"],
                record["attempt_id"],
            )
            for record in grouped_records
        ]
        waveforms = [record["stored_waveform"] for record in grouped_records]
        return worker.refresh_current_waveforms(
            items,
            waveforms,
            highpass_mode=NUMPY_HIGHPASS_MODE,
        )

    started = time.perf_counter()
    futures = [
        executor.submit(refresh_group, worker, grouped_records)
        for worker, grouped_records in zip(workers, groups, strict=True)
        if grouped_records
    ]
    topology_by_id: dict[str, tuple[dict[str, Any], float, float]] = {}
    request_sum_ms = 0.0
    internal_sum_ms = 0.0
    staging_sum_ms = 0.0
    for future in futures:
        rows, request_wall_ms, staging_rows = future.result()
        request_sum_ms += request_wall_ms
        staging_by_id = {row["id"]: row for row in staging_rows}
        for topology in rows:
            item_id = str(topology["id"])
            staging_ms = float(staging_by_id[item_id]["staging_ms"])
            topology_by_id[item_id] = (topology, request_wall_ms, staging_ms)
            internal_sum_ms += float(topology["pulse_runtime_ms"])
            staging_sum_ms += staging_ms
    wall_ms = 1000.0 * (time.perf_counter() - started)
    for record in records:
        item_id = f"v18-topology:{context['case_id']}:{record['attempt_id']}"
        topology, request_wall_ms, staging_ms = topology_by_id[item_id]
        stability = topology_stability(context["base_topology"], topology)
        record.update(
            {
                "candidate_topology": topology,
                "candidate_topology_sha256": topology_sha256(topology),
                "candidate_refresh_group_request_wall_ms": request_wall_ms,
                "candidate_refresh_client_staging_ms": staging_ms,
                "candidate_refresh_internal_ms": float(
                    topology["pulse_runtime_ms"]
                ),
                **stability,
            }
        )
    return {
        "candidate_refresh_concurrent_wall_ms": wall_ms,
        "candidate_refresh_request_wall_sum_ms": request_sum_ms,
        "candidate_refresh_internal_sum_ms": internal_sum_ms,
        "candidate_refresh_client_staging_sum_ms": staging_sum_ms,
    }


def pair_equivalence_row(
    context: dict[str, Any],
    family: str,
    alpha: float,
    reference_tensor: torch.Tensor,
    optimized_tensor: torch.Tensor,
    reference: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    step_equivalence = tensor_equivalence(reference_tensor, optimized_tensor)
    pcm_keys = (
        "pcm24_sha_differs_from_base",
        "pcm24_changed_samples",
        "pcm24_changed_fraction",
        "pcm24_residual_rms_lsb",
        "pcm24_effective_step_pass",
    )
    reference_legacy_pcm = pcm24_effective_step(
        context["base_path"],
        reference["candidate_path"],
    )
    optimized_legacy_pcm = pcm24_effective_step(
        context["base_path"],
        optimized["candidate_path"],
    )
    reference_in_memory_pcm_equal = all(
        reference[key] == reference_legacy_pcm[key] for key in pcm_keys
    )
    optimized_in_memory_pcm_equal = all(
        optimized[key] == optimized_legacy_pcm[key] for key in pcm_keys
    )
    pcm_metrics_equal = all(
        reference[key] == optimized[key] for key in pcm_keys
    )
    file_sha_equal = (
        reference["candidate_sha256"] == optimized["candidate_sha256"]
    )
    stored_waveform_equal = bool(
        np.array_equal(
            reference["stored_waveform"],
            optimized["stored_waveform"],
        )
    )
    proxy_equal = (
        reference["proxy_after_frozen_topology"]
        == optimized["proxy_after_frozen_topology"]
    )
    topology_hash_equal = (
        reference["candidate_topology_sha256"]
        == optimized["candidate_topology_sha256"]
    )
    safety_equal = all(
        reference[key] == optimized[key]
        for key in (
            "residual_rms_db",
            "cosine_similarity",
            "clip_fraction",
            "finite_safety_pass",
        )
    )
    passed = all(
        (
            step_equivalence["numerically_equivalent"],
            file_sha_equal,
            stored_waveform_equal,
            proxy_equal,
            topology_hash_equal,
            pcm_metrics_equal,
            reference_in_memory_pcm_equal,
            optimized_in_memory_pcm_equal,
            safety_equal,
        )
    )
    return {
        "case_id": context["case_id"],
        "family": family,
        "alpha": alpha,
        "step_tensor_bit_equal": step_equivalence["bit_equal"],
        "step_tensor_numerically_equivalent": step_equivalence[
            "numerically_equivalent"
        ],
        "step_tensor_maximum_abs_difference": step_equivalence[
            "maximum_abs_difference"
        ],
        "step_tensor_absolute_tolerance": step_equivalence[
            "absolute_tolerance"
        ],
        "pcm24_file_sha_equal": file_sha_equal,
        "stored_pcm24_waveform_equal": stored_waveform_equal,
        "frozen_proxy_equal": proxy_equal,
        "topology_hash_equal": topology_hash_equal,
        "pcm24_metrics_equal": pcm_metrics_equal,
        "reference_in_memory_pcm_matches_v16_helper": (
            reference_in_memory_pcm_equal
        ),
        "optimized_in_memory_pcm_matches_v16_helper": (
            optimized_in_memory_pcm_equal
        ),
        "safety_metrics_equal": safety_equal,
        "reference_candidate_sha256": reference["candidate_sha256"],
        "optimized_candidate_sha256": optimized["candidate_sha256"],
        "reference_topology_sha256": reference[
            "candidate_topology_sha256"
        ],
        "optimized_topology_sha256": optimized[
            "candidate_topology_sha256"
        ],
        "equivalence_pass": passed,
    }


def run_equivalence_case(
    panel_row: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
) -> dict[str, Any]:
    context = prepare_case_context(
        panel_row,
        target_shimmer_db,
        predictor,
        target_scale,
        device,
        workers[0],
    )
    legacy_plan_started = time.perf_counter()
    legacy_plan = build_legacy_candidate_d_plan(
        context["base_values"],
        context["base_topology"],
    )
    legacy_plan_ms = 1000.0 * (time.perf_counter() - legacy_plan_started)
    optimized_plan_started = time.perf_counter()
    optimized_plan = build_zero_crossing_cycle_plan_vectorized(
        context["base_values"],
        context["base_topology"],
    )
    optimized_plan_ms = 1000.0 * (
        time.perf_counter() - optimized_plan_started
    )
    plan_audit = plan_equivalence(legacy_plan, optimized_plan)

    legacy_projection, legacy_projection_report = legacy_candidate_d_projection(
        context["waveform"],
        context["raw_gradient"],
        legacy_plan,
    )
    optimized_projection, optimized_projection_report = (
        candidate_d_projection_vectorized(
            context["waveform"],
            context["raw_gradient"],
            optimized_plan,
        )
    )
    projection_audit = tensor_equivalence(
        legacy_projection,
        optimized_projection,
    )
    projection_report_audit = projection_report_equivalence(
        legacy_projection_report,
        optimized_projection_report,
    )
    cpu_waveform = context["waveform"].detach().cpu()
    cpu_raw_gradient = context["raw_gradient"].detach().cpu()
    legacy_cpu_projection, legacy_cpu_report = legacy_candidate_d_projection(
        cpu_waveform,
        cpu_raw_gradient,
        legacy_plan,
    )
    optimized_cpu_projection, optimized_cpu_report = (
        candidate_d_projection_vectorized(
            cpu_waveform,
            cpu_raw_gradient,
            optimized_plan,
        )
    )
    deterministic_projection_audit = tensor_equivalence(
        legacy_cpu_projection,
        optimized_cpu_projection,
    )
    deterministic_report_audit = projection_report_equivalence(
        legacy_cpu_report,
        optimized_cpu_report,
    )

    candidate_tensors: list[torch.Tensor] = []
    paths: list[Path] = []
    attempt_ids: list[str] = []
    pair_specs: list[tuple[str, float, int, int]] = []

    legacy_d = normalized_gradient_step(
        context["waveform"],
        optimized_projection,
        FIXED_ALPHA,
    )
    optimized_d = normalized_gradient_steps_shared(
        context["waveform"],
        optimized_projection,
        (FIXED_ALPHA,),
    )[0]
    d_reference_index = len(candidate_tensors)
    candidate_tensors.append(legacy_d)
    paths.append(waveform_root / f"{context['case_id']}__d_v17_reference.wav")
    attempt_ids.append("d_v17_reference")
    d_optimized_index = len(candidate_tensors)
    candidate_tensors.append(optimized_d)
    paths.append(waveform_root / f"{context['case_id']}__d_v18_optimized.wav")
    attempt_ids.append("d_v18_optimized")
    pair_specs.append(
        (
            CANDIDATE_D_NAME,
            FIXED_ALPHA,
            d_reference_index,
            d_optimized_index,
        )
    )

    legacy_c = [
        normalized_gradient_step(
            context["waveform"],
            context["raw_gradient"],
            alpha,
        )
        for alpha in ALPHA_LADDER
    ]
    optimized_c = normalized_gradient_steps_shared(
        context["waveform"],
        context["raw_gradient"],
        ALPHA_LADDER,
    )
    for backtrack_index, (alpha, reference_tensor, optimized_tensor) in enumerate(
        zip(ALPHA_LADDER, legacy_c, optimized_c, strict=True)
    ):
        reference_index = len(candidate_tensors)
        candidate_tensors.append(reference_tensor)
        paths.append(
            waveform_root
            / f"{context['case_id']}__c_v16_bt{backtrack_index}_reference.wav"
        )
        attempt_ids.append(f"c_v16_bt{backtrack_index}_reference")
        optimized_index = len(candidate_tensors)
        candidate_tensors.append(optimized_tensor)
        paths.append(
            waveform_root
            / f"{context['case_id']}__c_v18_bt{backtrack_index}_optimized.wav"
        )
        attempt_ids.append(f"c_v18_bt{backtrack_index}_optimized")
        pair_specs.append(
            (
                TRUST_REGION_CANDIDATE_NAME,
                alpha,
                reference_index,
                optimized_index,
            )
        )

    records, batch_runtime = write_candidate_batch(
        context,
        candidate_tensors,
        paths,
        attempt_ids,
        predictor,
        device,
        executor,
    )
    refresh_runtime = refresh_candidate_records(
        context,
        records,
        workers,
        executor,
    )
    rows = [
        pair_equivalence_row(
            context,
            family,
            alpha,
            candidate_tensors[reference_index],
            candidate_tensors[optimized_index],
            records[reference_index],
            records[optimized_index],
        )
        for family, alpha, reference_index, optimized_index in pair_specs
    ]
    return {
        "case_id": context["case_id"],
        "plan_equivalence": plan_audit,
        "legacy_plan_runtime_ms": legacy_plan_ms,
        "optimized_plan_runtime_ms": optimized_plan_ms,
        "projection_tensor_bit_equal": projection_audit["bit_equal"],
        "projection_tensor_numerically_equivalent": projection_audit[
            "numerically_equivalent"
        ],
        "projection_tensor_maximum_abs_difference": projection_audit[
            "maximum_abs_difference"
        ],
        "projection_tensor_absolute_tolerance": projection_audit[
            "absolute_tolerance"
        ],
        "independent_gpu_projection_report_diagnostics": (
            projection_report_audit
        ),
        "deterministic_cpu_projection_bit_equal": (
            deterministic_projection_audit["bit_equal"]
        ),
        "deterministic_cpu_projection_numerically_equivalent": (
            deterministic_projection_audit["numerically_equivalent"]
        ),
        "deterministic_cpu_projection_maximum_abs_difference": (
            deterministic_projection_audit["maximum_abs_difference"]
        ),
        "deterministic_cpu_projection_absolute_tolerance": (
            deterministic_projection_audit["absolute_tolerance"]
        ),
        "projection_report_equivalence": deterministic_report_audit,
        "candidate_d_pcm_equivalence_projection_input": (
            "shared_optimized_projection_after_independent_legacy_projection_numeric_audit"
        ),
        "independent_projection_repetition_pcm_comparison_not_used": True,
        "legacy_projected_gradient_l2_norm": float(legacy_projection.norm()),
        "optimized_projected_gradient_l2_norm": float(
            optimized_projection.norm()
        ),
        "pair_rows": rows,
        "batch_runtime": batch_runtime,
        "refresh_runtime": refresh_runtime,
        "equivalence_pass": (
            plan_audit["all_equal"]
            and projection_audit["numerically_equivalent"]
            and deterministic_projection_audit["bit_equal"]
            and deterministic_report_audit["all_equivalent"]
            and all(row["equivalence_pass"] for row in rows)
        ),
    }


def evaluate_selector_case(
    panel_row: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    context = prepare_case_context(
        panel_row,
        target_shimmer_db,
        predictor,
        target_scale,
        device,
        workers[0],
    )
    plan_started = time.perf_counter()
    candidate_d_plan = build_zero_crossing_cycle_plan_vectorized(
        context["base_values"],
        context["base_topology"],
    )
    plan_runtime_ms = 1000.0 * (time.perf_counter() - plan_started)
    projection_started = time.perf_counter()
    candidate_d_gradient, projection = candidate_d_projection_vectorized(
        context["waveform"],
        context["raw_gradient"],
        candidate_d_plan,
    )
    candidate_d = normalized_gradient_steps_shared(
        context["waveform"],
        candidate_d_gradient,
        (FIXED_ALPHA,),
    )[0]
    synchronize(device)
    projection_runtime_ms = 1000.0 * (
        time.perf_counter() - projection_started
    )
    candidate_d_path = (
        waveform_root / f"{context['case_id']}__candidate_d_v18.wav"
    )
    d_records, d_batch_runtime = write_candidate_batch(
        context,
        [candidate_d],
        [candidate_d_path],
        ["candidate_d"],
        predictor,
        device,
        executor,
    )
    d_refresh_runtime = refresh_candidate_records(
        context,
        d_records,
        workers,
        executor,
    )
    d_record = d_records[0]
    d_record.update(
        {
            "family": CANDIDATE_D_NAME,
            "alpha": FIXED_ALPHA,
            "backtrack_index": None,
            "gradient_l2_norm": projection["projected_gradient_l2_norm"],
            "gradient_rms": float(
                candidate_d_gradient.square().mean().sqrt()
            ),
            "gradient_finite": projection["projected_gradient_finite"],
            "projected_gradient_valid": projection[
                "projected_gradient_valid"
            ],
            "complete_cycle_support_pass": (
                projection["complete_cycle_count"] >= MINIMUM_COMPLETE_CYCLES
            ),
        }
    )
    d_certificates = {
        key: d_record[key]
        for key in D_ROUTING_KEYS
    }
    d_record["routing_certificates"] = d_certificates
    d_routing_pass = select_candidate_d(d_certificates)
    d_record["family_selector_pass"] = d_routing_pass

    attempts = [d_record]
    c_batch_runtime: dict[str, float] | None = None
    c_refresh_runtime: dict[str, float] | None = None
    selected_family: str | None = None
    selected_record: dict[str, Any] | None = None
    if d_routing_pass:
        selected_family = CANDIDATE_D_NAME
        selected_record = d_record
    else:
        candidate_c_tensors = normalized_gradient_steps_shared(
            context["waveform"],
            context["raw_gradient"],
            ALPHA_LADDER,
        )
        candidate_c_paths = [
            waveform_root
            / (
                f"{context['case_id']}__candidate_c_bt{backtrack_index}"
                f"_alpha_{alpha:.7f}.wav"
            )
            for backtrack_index, alpha in enumerate(ALPHA_LADDER)
        ]
        c_records, c_batch_runtime = write_candidate_batch(
            context,
            candidate_c_tensors,
            candidate_c_paths,
            [f"candidate_c_bt{index}" for index in range(len(ALPHA_LADDER))],
            predictor,
            device,
            executor,
        )
        c_refresh_runtime = refresh_candidate_records(
            context,
            c_records,
            workers,
            executor,
        )
        for backtrack_index, (alpha, record) in enumerate(
            zip(ALPHA_LADDER, c_records, strict=True)
        ):
            record.update(
                {
                    "family": TRUST_REGION_CANDIDATE_NAME,
                    "alpha": alpha,
                    "backtrack_index": backtrack_index,
                    "gradient_l2_norm": context["raw_gradient_l2_norm"],
                    "gradient_rms": context["raw_gradient_rms"],
                    "gradient_finite": True,
                }
            )
        attempts.extend(c_records)
        selected_input = select_topology_certified_step(
            [c_selector_view(record) for record in c_records]
        )
        if selected_input is not None:
            selected_family = TRUST_REGION_CANDIDATE_NAME
            selected_record = c_records[
                int(selected_input["backtrack_index"])
            ]

    synchronize(device)
    total_metric_step_runtime_ms = 1000.0 * (
        time.perf_counter() - total_started
    )
    runtime_gate_pass = (
        total_metric_step_runtime_ms <= CACHE_RUNTIME_MAX_MS
    )
    selector_pass = selected_record is not None and runtime_gate_pass
    return {
        "case_id": context["case_id"],
        "base_topology": context["base_topology"],
        "base_topology_sha256": context["base_topology_sha256"],
        "proxy_before": context["proxy_before"],
        "proxy_target": context["proxy_target"],
        "proxy_loss": context["proxy_loss"],
        "raw_gradient_l2_norm": context["raw_gradient_l2_norm"],
        "raw_gradient_rms": context["raw_gradient_rms"],
        "raw_gradient_finite": True,
        "base_refresh_runtime_ms": context["base_refresh_runtime_ms"],
        "base_refresh_request_wall_ms": context[
            "base_refresh_request_wall_ms"
        ],
        "base_refresh_client_staging_ms": context[
            "base_refresh_client_staging_ms"
        ],
        "gradient_runtime_ms": context["gradient_runtime_ms"],
        "candidate_d_plan_runtime_ms": plan_runtime_ms,
        "candidate_d_projection_runtime_ms": projection_runtime_ms,
        "candidate_d_projection": projection,
        "candidate_d_batch_runtime": d_batch_runtime,
        "candidate_d_refresh_runtime": d_refresh_runtime,
        "candidate_c_batch_runtime": c_batch_runtime,
        "candidate_c_refresh_runtime": c_refresh_runtime,
        "attempts": attempts,
        "attempted_family_count": 1 + int(c_batch_runtime is not None),
        "candidate_topology_refresh_count": len(attempts),
        "selected_family": selected_family,
        "selected_alpha": (
            float(selected_record["alpha"])
            if selected_record is not None
            else None
        ),
        "selected_backtrack_index": (
            selected_record["backtrack_index"]
            if selected_record is not None
            else None
        ),
        "selected_record": selected_record,
        "selected_path": (
            selected_record["candidate_path"]
            if selected_record is not None
            else None
        ),
        "selected_topology": (
            selected_record["candidate_topology"]
            if selected_record is not None
            else None
        ),
        "runtime_gate_pass": runtime_gate_pass,
        "selector_pass": selector_pass,
        "total_metric_step_runtime_ms": total_metric_step_runtime_ms,
        "selector_uses_no_candidate_exact_outcome": True,
    }


def preselection_rows(
    panel_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, records, strict=True):
        for attempt_index, attempt in enumerate(record["attempts"]):
            rows.append(
                {
                    "case_id": panel_row["case_id"],
                    "speaker_id": panel_row["speaker_id"],
                    "view": panel_row["view"],
                    "condition": panel_row["condition"],
                    "sample_group": panel_row["sample_group"],
                    "attempt_index": attempt_index,
                    "family": attempt["family"],
                    "alpha": attempt["alpha"],
                    "backtrack_index": attempt["backtrack_index"],
                    "candidate_path": str(
                        attempt["candidate_path"].resolve()
                    ),
                    "candidate_sha256": attempt["candidate_sha256"],
                    "candidate_topology_sha256": attempt[
                        "candidate_topology_sha256"
                    ],
                    "candidate_pulse_count": int(
                        attempt["candidate_topology"]["pulse_count"]
                    ),
                    "proxy_before": attempt["proxy_before"],
                    "proxy_after_frozen_topology": attempt[
                        "proxy_after_frozen_topology"
                    ],
                    "normalized_proxy_gap_before": attempt[
                        "normalized_proxy_gap_before"
                    ],
                    "normalized_proxy_gap_after": attempt[
                        "normalized_proxy_gap_after"
                    ],
                    "proxy_nonregression_pass": attempt[
                        "proxy_nonregression_pass"
                    ],
                    "topology_stability_pass": attempt[
                        "topology_stability_pass"
                    ],
                    "reference_to_candidate_match_rate_16_samples": attempt[
                        "reference_to_candidate_match_rate_16_samples"
                    ],
                    "candidate_to_reference_match_rate_16_samples": attempt[
                        "candidate_to_reference_match_rate_16_samples"
                    ],
                    "finite_safety_pass": attempt["finite_safety_pass"],
                    "pcm24_effective_step_pass": attempt[
                        "pcm24_effective_step_pass"
                    ],
                    "selected_family": record["selected_family"],
                    "selected_alpha": record["selected_alpha"],
                    "selected_attempt": (
                        record["selected_record"] is attempt
                    ),
                    "runtime_gate_pass": record["runtime_gate_pass"],
                    "selector_pass": record["selector_pass"],
                    "total_metric_step_runtime_ms": record[
                        "total_metric_step_runtime_ms"
                    ],
                }
            )
    return rows


def summarize_exact_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism = aggregate_candidate(SELECTOR_NAME, rows)
    pathology = aggregate_pathology_guardrails(rows)
    denoising = aggregate_denoising(rows)
    mechanism_gates = {
        "complete_prototype_coverage": len(rows) == EXPECTED_CASE_COUNT,
        "exact_db_effect": (
            mechanism["exact_db_improvement_fraction"]
            >= IMPROVEMENT_FRACTION_GATE
            and mechanism["median_exact_db_normalized_gap_reduction"]
            >= MEDIAN_REDUCTION_GATE
        ),
        "required_effect_slices": (
            mechanism["required_slice_gate"]["decision"] == "PASS"
        ),
        "gradient": all(
            row["gradient_finite"]
            and GRADIENT_NORM_RANGE[0]
            <= row["gradient_l2_norm"]
            <= GRADIENT_NORM_RANGE[1]
            for row in rows
        ),
        "total_metric_step_runtime": all(
            row["total_metric_step_runtime_ms"] <= CACHE_RUNTIME_MAX_MS
            for row in rows
        ),
        "nonselected": all(
            value <= NONSELECTED_MEDIAN_INCREASE_GATE
            for value in mechanism[
                "nonselected_median_normalized_gap_increase"
            ].values()
        ),
        "safety": all(
            row["residual_rms_db"] <= RESIDUAL_CEILING_DB
            and row["cosine_similarity"] >= MINIMUM_COSINE
            and row["clip_fraction"] <= MAXIMUM_CLIP_FRACTION
            for row in rows
        ),
        "topology_stability": all(
            row["topology_stability_pass"] for row in rows
        ),
    }
    integration_gates = {
        "mechanism": all(mechanism_gates.values()),
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
        "selector_coverage": all(row["selector_pass"] for row in rows),
        "selector_uses_no_candidate_exact_outcome": all(
            row["selector_uses_no_candidate_exact_outcome"] for row in rows
        ),
        "selected_topology_rebound": all(
            row["selected_topology_rebound"] for row in rows
        ),
        "base_topology_rebound": all(
            row["base_topology_rebound"] for row in rows
        ),
        "pcm24_effective_step": all(
            row["pcm24_effective_step_pass"] for row in rows
        ),
        "target_topology_not_used": all(
            row["clean_target_topology_drives_output"] is False
            for row in rows
        ),
        "exact_metric_mapping_parity": all(
            row["metric_reconstruction_max_pcm16_error"] == 0
            and row["metric_reconstruction_differing_samples"] == 0
            and row["candidate_metric_reconstruction_max_pcm16_error"] == 0
            and row["candidate_metric_reconstruction_differing_samples"] == 0
            for row in rows
        ),
    }
    return {
        "candidate": SELECTOR_NAME,
        "mechanism": mechanism,
        "mechanism_gates": mechanism_gates,
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "integration_gates": integration_gates,
        "selected_family_counts": {
            CANDIDATE_D_NAME: sum(
                row["selected_family"] == CANDIDATE_D_NAME for row in rows
            ),
            TRUST_REGION_CANDIDATE_NAME: sum(
                row["selected_family"] == TRUST_REGION_CANDIDATE_NAME
                for row in rows
            ),
        },
        "total_metric_step_runtime_ms": {
            "median": median(row["total_metric_step_runtime_ms"] for row in rows),
            "maximum": max(row["total_metric_step_runtime_ms"] for row in rows),
            "formal_gate_ms": CACHE_RUNTIME_MAX_MS,
        },
        "all_gates_pass": all(integration_gates.values()),
    }


def write_completion_receipt(
    args: argparse.Namespace,
    decision: str,
    artifact_paths: list[Path],
) -> None:
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-topology-family-selector-v18-receipt-v1",
        "phase": args.phase,
        "decision": decision,
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "dev_only": True,
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)


def run_equivalence_phase(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    input_by_case: dict[str, dict[str, str]],
    source_hashes: dict[str, str],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
    runtime_environment: dict[str, Any],
) -> None:
    case_audits = []
    for index, panel_row in enumerate(panel_rows, start=1):
        case_audits.append(
            run_equivalence_case(
                panel_row,
                float(
                    input_by_case[panel_row["case_id"]][
                        "exact_target_shimmer_db"
                    ]
                ),
                predictor,
                target_scale,
                device,
                workers,
                executor,
                waveform_root,
            )
        )
        print(f"v18_equivalence={index}/{EXPECTED_CASE_COUNT}", flush=True)
    rows = [
        row
        for case_audit in case_audits
        for row in case_audit["pair_rows"]
    ]
    rows_path = args.output_dir / "family_equivalence.csv"
    write_csv(rows_path, rows)
    all_pass = (
        len(case_audits) == EXPECTED_CASE_COUNT
        and all(case_audit["equivalence_pass"] for case_audit in case_audits)
        and all(row["equivalence_pass"] for row in rows)
    )
    decision = (
        "PASS_SHIMMER_DB_V18_FAMILY_EQUIVALENCE"
        if all_pass
        else "FAIL_SHIMMER_DB_V18_FAMILY_EQUIVALENCE"
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-db-topology-family-selector-v18-equivalence-v1",
        "phase": "equivalence",
        "decision": decision,
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "dev_only": True,
        "opened_cases_only": True,
        "candidate_exact_outcomes_opened": False,
        "exact_component_scoring_requested": False,
        "selector_contract": selector_contract(),
        "source_sha256": source_hashes,
        "runtime_environment": runtime_environment,
        "case_audits": [
            {
                key: value
                for key, value in case_audit.items()
                if key != "pair_rows"
            }
            for case_audit in case_audits
        ],
        "pair_count": len(rows),
        "all_equivalence_gates_pass": all_pass,
        "selector4_authorized": all_pass,
        "new_sealed_panel_authorized": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_optimizer_steps": 0,
    }
    report_path = args.output_dir / "equivalence_report.json"
    write_json(report_path, report)
    write_completion_receipt(args, decision, [report_path, rows_path])
    print(json.dumps({"decision": decision, "pairs": len(rows)}), flush=True)


def run_selector_phase(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    input_by_case: dict[str, dict[str, str]],
    source_hashes: dict[str, str],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
    runtime_environment: dict[str, Any],
) -> None:
    case_records = []
    for index, panel_row in enumerate(panel_rows, start=1):
        case_records.append(
            evaluate_selector_case(
                panel_row,
                float(
                    input_by_case[panel_row["case_id"]][
                        "exact_target_shimmer_db"
                    ]
                ),
                predictor,
                target_scale,
                device,
                workers,
                executor,
                waveform_root,
            )
        )
        print(f"v18_selector={index}/{EXPECTED_CASE_COUNT}", flush=True)

    preselection_path = args.output_dir / "family_selector_preselection.csv"
    write_csv(preselection_path, preselection_rows(panel_rows, case_records))
    selector_failures = [
        record["case_id"]
        for record in case_records
        if not record["selector_pass"]
    ]
    common_report = {
        "schema_version": "avqi-route-c-shimmer-db-topology-family-selector-v18-v1",
        "candidate": SELECTOR_NAME,
        "route_type": "hybrid_praat_assisted_topology_family_selector",
        "pure_torch_estimator": False,
        "phase": "selector4",
        "dev_only": True,
        "opened_cases_only": True,
        "selector_contract": selector_contract(),
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_sha256": source_hashes,
        "runtime_environment": runtime_environment,
        "case_runtime": [
            {
                "case_id": record["case_id"],
                "selected_family": record["selected_family"],
                "selected_alpha": record["selected_alpha"],
                "attempted_family_count": record["attempted_family_count"],
                "candidate_topology_refresh_count": record[
                    "candidate_topology_refresh_count"
                ],
                "base_refresh_runtime_ms": record[
                    "base_refresh_runtime_ms"
                ],
                "gradient_runtime_ms": record["gradient_runtime_ms"],
                "candidate_d_plan_runtime_ms": record[
                    "candidate_d_plan_runtime_ms"
                ],
                "candidate_d_projection_runtime_ms": record[
                    "candidate_d_projection_runtime_ms"
                ],
                "candidate_d_batch_runtime": record[
                    "candidate_d_batch_runtime"
                ],
                "candidate_d_refresh_runtime": record[
                    "candidate_d_refresh_runtime"
                ],
                "candidate_c_batch_runtime": record[
                    "candidate_c_batch_runtime"
                ],
                "candidate_c_refresh_runtime": record[
                    "candidate_c_refresh_runtime"
                ],
                "total_metric_step_runtime_ms": record[
                    "total_metric_step_runtime_ms"
                ],
                "runtime_gate_pass": record["runtime_gate_pass"],
                "selector_pass": record["selector_pass"],
            }
            for record in case_records
        ],
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    if selector_failures:
        decision = "NO_GO_SHIMMER_DB_V18_TOPOLOGY_FAMILY_SELECTOR_4CASE"
        report = {
            **common_report,
            "decision": decision,
            "candidate_exact_outcomes_opened": False,
            "exact_component_scoring_requested": False,
            "selector_failures": selector_failures,
            "selector_coverage": (
                EXPECTED_CASE_COUNT - len(selector_failures)
            )
            / EXPECTED_CASE_COUNT,
            "opened_v14_v15_expansion_authorized": False,
            "new_sealed_panel_authorized": False,
        }
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        write_completion_receipt(
            args,
            decision,
            [report_path, preselection_path],
        )
        print(
            json.dumps(
                {"decision": decision, "failures": selector_failures},
                sort_keys=True,
            ),
            flush=True,
        )
        return

    selector_seal = {
        "schema_version": "avqi-route-c-shimmer-db-topology-family-selector-v18-seal-v1",
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_present": False,
        "selection_uses_candidate_exact_outcome": False,
        "selector_contract": selector_contract(),
        "preselection_sha256": sha256_file(preselection_path),
        "rows": [
            {
                "case_id": record["case_id"],
                "selected_family": record["selected_family"],
                "selected_alpha": record["selected_alpha"],
                "selected_backtrack_index": record[
                    "selected_backtrack_index"
                ],
                "candidate_path": str(record["selected_path"].resolve()),
                "candidate_sha256": record["selected_record"][
                    "candidate_sha256"
                ],
                "candidate_topology_sha256": record["selected_record"][
                    "candidate_topology_sha256"
                ],
                "attempted_family_count": record["attempted_family_count"],
                "candidate_topology_refresh_count": record[
                    "candidate_topology_refresh_count"
                ],
                "total_metric_step_runtime_ms": record[
                    "total_metric_step_runtime_ms"
                ],
                "runtime_gate_pass": record["runtime_gate_pass"],
            }
            for record in case_records
        ],
    }
    selector_seal_path = args.output_dir / "selector_seal.json"
    write_json(selector_seal_path, selector_seal)

    exact_items = [
        {
            "id": f"v18-selected:{panel_row['case_id']}",
            "case_id": panel_row["case_id"],
            "role": "topology_family_selected_candidate",
            "path": str(record["selected_path"].resolve()),
            "view": panel_row["view"],
            "score_components": True,
            "exact_metric_topology": True,
        }
        for panel_row, record in zip(panel_rows, case_records, strict=True)
    ]
    exact_after = run_exact(exact_items, args.exact_python, args.avqi_code_root)
    after_by_case = {row["case_id"]: row for row in exact_after["rows"]}
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, case_records, strict=True):
        case_id = panel_row["case_id"]
        input_row = input_by_case[case_id]
        after = after_by_case[case_id]
        selected = record["selected_record"]
        selected_topology_rebound = bool(
            require_exact_topology_equal(
                record["selected_topology"],
                after,
                f"v18 selected topology rebound {case_id}",
            )
        )
        target_components = exact_vector(input_row, "target")
        base_components = exact_vector(input_row, "before")
        after_components = exact_components(after)
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        candidate_waveform = read_waveform(record["selected_path"])
        base_topology = record["base_topology"]
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": SELECTOR_NAME,
            "selected_family": record["selected_family"],
            "optimized_component": "shimmer_db",
            "alpha_max": FIXED_ALPHA,
            "selected_alpha": record["selected_alpha"],
            "selected_backtrack_index": record[
                "selected_backtrack_index"
            ],
            "candidate_path": str(record["selected_path"].resolve()),
            "candidate_sha256": selected["candidate_sha256"],
            "proxy_before": record["proxy_before"],
            "proxy_after_frozen_topology": selected[
                "proxy_after_frozen_topology"
            ],
            "proxy_target": record["proxy_target"],
            "proxy_loss": record["proxy_loss"],
            "gradient_l2_norm": selected["gradient_l2_norm"],
            "gradient_rms": selected["gradient_rms"],
            "gradient_finite": selected["gradient_finite"],
            "pulse_refresh_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "torch_step_runtime_ms": record["gradient_runtime_ms"],
            "total_metric_step_overhead_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "total_metric_step_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "runtime_gate_pass": record["runtime_gate_pass"],
            "base_refresh_runtime_ms": record["base_refresh_runtime_ms"],
            "candidate_topology_refresh_count": record[
                "candidate_topology_refresh_count"
            ],
            "attempted_family_count": record["attempted_family_count"],
            "selector_pass": record["selector_pass"],
            "selector_uses_no_candidate_exact_outcome": True,
            "pcm24_effective_step_pass": selected[
                "pcm24_effective_step_pass"
            ],
            "pcm24_changed_samples": selected["pcm24_changed_samples"],
            "pcm24_changed_fraction": selected[
                "pcm24_changed_fraction"
            ],
            "pcm24_residual_rms_lsb": selected[
                "pcm24_residual_rms_lsb"
            ],
            "base_output_exact_metric_pulse_count": int(
                base_topology["pulse_count"]
            ),
            "candidate_exact_metric_pulse_count": int(after["pulse_count"]),
            "metric_sample_count": int(base_topology["metric_sample_count"]),
            "metric_constant_prefix_samples": int(
                base_topology["metric_constant_prefix_samples"]
            ),
            "metric_source_range_count": int(
                base_topology["metric_source_range_count"]
            ),
            "metric_mapped_sample_count": int(
                base_topology["metric_mapped_sample_count"]
            ),
            "metric_reconstruction_max_pcm16_error": int(
                base_topology["metric_reconstruction_max_pcm16_error"]
            ),
            "metric_reconstruction_differing_samples": int(
                base_topology["metric_reconstruction_differing_samples"]
            ),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                after["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                after["metric_reconstruction_differing_samples"]
            ),
            "selected_topology_rebound": selected_topology_rebound,
            "base_topology_rebound": (
                record["base_topology_sha256"]
                == str(input_row["composite_topology_sha256"])
            ),
            "clean_target_topology_drives_output": False,
        }
        component_fields(
            row,
            target_components,
            base_components,
            after_components,
            target_scale_np,
        )
        row["material_shimmer_db_gap"] = (
            row["exact_absolute_gap_before_shimmer_db"]
            / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
            > MATERIAL_GAP_THRESHOLD
        )
        row["forward_normalized_abs_error_shimmer_db"] = abs(
            row["proxy_before"] - row["exact_before_shimmer_db"]
        ) / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        row.update(topology_stability(base_topology, after))
        row.update(
            waveform_safety(base_waveform.numpy(), candidate_waveform.numpy())
        )
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)

    results_path = args.output_dir / "family_selector_results.csv"
    write_csv(results_path, rows)
    summary = summarize_exact_rows(rows)
    decision = (
        "PASS_SHIMMER_DB_V18_TOPOLOGY_FAMILY_SELECTOR_4CASE_MECHANISM"
        if summary["all_gates_pass"]
        else "NO_GO_SHIMMER_DB_V18_TOPOLOGY_FAMILY_SELECTOR_4CASE_MECHANISM"
    )
    report = {
        **common_report,
        "decision": decision,
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "selector_seal_sha256": sha256_file(selector_seal_path),
        "summary": summary,
        "promotion_authorized": False,
        "opened_v14_v15_expansion_authorized": summary["all_gates_pass"],
        "new_sealed_panel_authorized": False,
        "exact_scorer_versions": {
            "parselmouth": exact_after["parselmouth_version"],
            "praat": exact_after["praat_version"],
        },
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    write_completion_receipt(
        args,
        decision,
        [report_path, preselection_path, selector_seal_path, results_path],
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "runtime": summary["total_metric_step_runtime_ms"],
                "exact_improvement_fraction": summary["mechanism"][
                    "exact_db_improvement_fraction"
                ],
                "median_normalized_reduction": summary["mechanism"][
                    "median_exact_db_normalized_gap_reduction"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    panel_rows, input_by_case, source_hashes = validate_sources_and_inputs(args)
    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    torch_warmup = synthetic_torch_warmup(predictor, target_scale, device)
    candidate_d_warmup = synthetic_candidate_d_warmup(device)
    optimized_v18_warmup = synthetic_v18_warmup(device)
    workers: list[ExactShimmerTopologyWorker] = []
    worker_startups = []
    worker_warmups = []
    try:
        for worker_index in range(WORKER_COUNT):
            worker = ExactShimmerTopologyWorker(
                args.exact_python,
                args.runtime_worker_script,
                args.avqi_code_root,
                args.avqi_code_tree_sha256,
            )
            workers.append(worker)
            warmup, warmup_ms = worker.warmup()
            worker_startups.append(
                {
                    "worker_index": worker_index,
                    "startup_ms": worker.startup_ms,
                    **worker.startup,
                }
            )
            worker_warmups.append(
                {
                    "worker_index": worker_index,
                    "request_wall_ms": warmup_ms,
                    **warmup,
                }
            )
        runtime_environment = {
            "torch_synthetic_warmup": torch_warmup,
            "candidate_d_synthetic_warmup": candidate_d_warmup,
            "optimized_v18_synthetic_warmup": optimized_v18_warmup,
            "worker_startups": worker_startups,
            "worker_synthetic_warmups": worker_warmups,
            "worker_count": WORKER_COUNT,
            "warmups_outside_case_timer": True,
        }
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            if args.phase == "equivalence":
                run_equivalence_phase(
                    args,
                    panel_rows,
                    input_by_case,
                    source_hashes,
                    predictor,
                    target_scale,
                    device,
                    workers,
                    executor,
                    waveform_root,
                    runtime_environment,
                )
            else:
                run_selector_phase(
                    args,
                    panel_rows,
                    input_by_case,
                    source_hashes,
                    predictor,
                    target_scale,
                    device,
                    workers,
                    executor,
                    waveform_root,
                    runtime_environment,
                )
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
