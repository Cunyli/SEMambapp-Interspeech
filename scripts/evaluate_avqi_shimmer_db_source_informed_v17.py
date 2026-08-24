#!/usr/bin/env python3
"""Run the four-case source-informed Shimmer-dB Candidate-D study.

Candidate D keeps the exact current-output pulse topology detached, projects the
live Shimmer-dB waveform gradient onto multiplicative pitch-cycle directions
bounded by full-band zero crossings, and applies one fixed 1:2:1 coefficient
smoothing pass.  Candidate exact component outcomes remain closed until all
four topology/proxy/safety/runtime selectors have produced a hash seal.
"""

from __future__ import annotations

import argparse
import json
import math
import time
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
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    PROTOTYPE_CASE_IDS,
    base_topology_item,
    pcm24_effective_step,
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
CANDIDATE_D_NAME = (
    "praat_current_output_zero_crossing_shape_preserving_gain_db_v17"
)
EXPECTED_CASE_COUNT = len(PROTOTYPE_CASE_IDS)
PROXY_GAP_TOLERANCE = 1e-7
MINIMUM_COMPLETE_CYCLES = 3
COEFFICIENT_SMOOTHING_KERNEL = (0.25, 0.5, 0.25)
SELECTOR_KEYS = frozenset(
    {
        "projected_gradient_valid",
        "complete_cycle_support_pass",
        "proxy_nonregression_pass",
        "finite_safety_pass",
        "pcm24_effective_step_pass",
        "topology_stability_pass",
        "runtime_gate_pass",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--fresh-results", type=Path, required=True)
    parser.add_argument("--fresh-results-sha256", required=True)
    parser.add_argument("--pulse-diagnostic-report", type=Path, required=True)
    parser.add_argument("--pulse-diagnostic-report-sha256", required=True)
    parser.add_argument("--pulse-diagnostic-receipt", type=Path, required=True)
    parser.add_argument("--pulse-diagnostic-receipt-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--runtime-worker-script-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_diagnostic_contract(report: dict[str, Any], receipt: dict[str, Any]) -> None:
    expected_family = (
        "pitch_synchronous_zero_crossing_shape_preserving_gain_projection"
    )
    if report.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("v17 pulse diagnosis opened forbidden candidate outcomes")
    if report.get("exact_component_scoring_requested") is not False:
        raise ValueError("v17 pulse diagnosis requested exact component scoring")
    if report.get("routing", {}).get("candidate_d_family") != expected_family:
        raise ValueError("v17 topology diagnosis did not select Candidate D")
    if receipt.get("candidate_d_family") != expected_family:
        raise ValueError("v17 diagnostic receipt routing drift")
    if receipt.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("v17 diagnostic receipt outcome boundary drift")


def nearest_zero_crossing_boundary(
    values: np.ndarray,
    *,
    metric_start: int,
    mapped_start: int,
    left_pulse: float,
    right_pulse: float,
) -> dict[str, Any] | None:
    local_left = max(int(math.floor(left_pulse - metric_start)), 0)
    local_right = min(int(math.ceil(right_pulse - metric_start)), values.size - 1)
    if local_right <= local_left:
        return None
    indices = np.arange(local_left, local_right, dtype=np.int64)
    left_values = values[indices]
    right_values = values[indices + 1]
    crossing = (
        ((left_values <= 0.0) & (right_values > 0.0))
        | ((left_values >= 0.0) & (right_values < 0.0))
    )
    crossing_indices = indices[crossing]
    if crossing_indices.size == 0:
        return None
    left_crossing_values = values[crossing_indices]
    right_crossing_values = values[crossing_indices + 1]
    denominator = left_crossing_values - right_crossing_values
    fraction = np.divide(
        left_crossing_values,
        denominator,
        out=np.zeros_like(left_crossing_values),
        where=denominator != 0.0,
    )
    crossing_positions = metric_start + crossing_indices + fraction
    midpoint = 0.5 * (left_pulse + right_pulse)
    selected = int(np.argmin(np.abs(crossing_positions - midpoint)))
    left_index = int(crossing_indices[selected])
    return {
        "mapped_right_index": mapped_start + left_index + 1,
        "metric_position": float(crossing_positions[selected]),
        "left_abs_amplitude": float(abs(values[left_index])),
        "right_abs_amplitude": float(abs(values[left_index + 1])),
    }


def build_zero_crossing_cycle_plan(
    waveform: np.ndarray,
    topology: dict[str, Any],
) -> dict[str, Any]:
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
        boundaries: list[dict[str, Any] | None] = []
        for left_index, right_index in zip(
            pulse_indices[:-1],
            pulse_indices[1:],
            strict=True,
        ):
            boundary = nearest_zero_crossing_boundary(
                range_values,
                metric_start=metric_start,
                mapped_start=mapped_start,
                left_pulse=float(pulses[left_index]),
                right_pulse=float(pulses[right_index]),
            )
            boundaries.append(boundary)
            if boundary is None:
                failed_pulse_pairs += 1
            else:
                crossing_rows.append(
                    {
                        "range_index": range_index,
                        "left_pulse_index": int(left_index),
                        "right_pulse_index": int(right_index),
                        **boundary,
                    }
                )

        active_group: list[int] = []
        previous_pulse_index: int | None = None
        previous_end: int | None = None
        for local_index in range(1, pulse_indices.size - 1):
            left_boundary = boundaries[local_index - 1]
            right_boundary = boundaries[local_index]
            if left_boundary is None or right_boundary is None:
                if active_group:
                    groups.append(active_group)
                    active_group = []
                previous_pulse_index = None
                previous_end = None
                continue
            start = int(left_boundary["mapped_right_index"])
            end = int(right_boundary["mapped_right_index"])
            if end <= start or np.any(cell_ids[start:end] >= 0):
                raise ValueError("zero-crossing cycle plan overlaps or is empty")
            cell_id = len(cycles)
            cell_ids[start:end] = cell_id
            pulse_index = int(pulse_indices[local_index])
            cycles.append(
                {
                    "cell_id": cell_id,
                    "range_index": range_index,
                    "pulse_index": pulse_index,
                    "mapped_start_index": start,
                    "mapped_end_index": end,
                    "sample_count": end - start,
                }
            )
            contiguous = (
                previous_pulse_index is not None
                and pulse_index == previous_pulse_index + 1
                and start == previous_end
            )
            if not contiguous and active_group:
                groups.append(active_group)
                active_group = []
            active_group.append(cell_id)
            previous_pulse_index = pulse_index
            previous_end = end
        if active_group:
            groups.append(active_group)

    cycle_count = len(cycles)
    if cycle_count == 0:
        previous_cells = np.empty(0, dtype=np.int64)
        next_cells = np.empty(0, dtype=np.int64)
    else:
        previous_cells = np.arange(cycle_count, dtype=np.int64)
        next_cells = np.arange(cycle_count, dtype=np.int64)
        for group in groups:
            for index, cell_id in enumerate(group):
                previous_cells[cell_id] = group[max(index - 1, 0)]
                next_cells[cell_id] = group[min(index + 1, len(group) - 1)]
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


def zero_crossing_shape_preserving_gradient_projection(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    plan: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
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
    finite = bool(torch.isfinite(projected).all())
    norm = float(projected.norm())
    return projected, {
        **plan["summary"],
        "raw_cycle_coefficient_minimum": float(coefficients.min()),
        "raw_cycle_coefficient_median": float(coefficients.median()),
        "raw_cycle_coefficient_maximum": float(coefficients.max()),
        "smoothed_cycle_coefficient_minimum": float(smoothed.min()),
        "smoothed_cycle_coefficient_median": float(smoothed.median()),
        "smoothed_cycle_coefficient_maximum": float(smoothed.max()),
        "projected_gradient_l2_norm": norm,
        "projected_gradient_finite": finite,
        "projected_gradient_valid": finite and norm > 0.0,
    }


def synthetic_candidate_d_warmup(device: torch.device) -> dict[str, Any]:
    """Prime only Candidate-D plan/projection kernels on a synthetic waveform."""
    started = time.perf_counter()
    sample_count = 5 * SAMPLE_RATE
    timeline_cpu = np.arange(sample_count, dtype=np.float32)
    waveform_cpu = (
        0.08
        * np.sin(2.0 * np.pi * 125.0 * timeline_cpu / SAMPLE_RATE)
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
    plan = build_zero_crossing_cycle_plan(waveform_cpu, topology)
    waveform = torch.from_numpy(waveform_cpu).to(device)
    modulation = 1.0 + 0.2 * torch.sin(
        2.0 * torch.pi * torch.arange(sample_count, device=device) / 2048.0
    )
    synthetic_gradient = waveform * modulation
    projected, projection = zero_crossing_shape_preserving_gradient_projection(
        waveform,
        synthetic_gradient,
        plan,
    )
    candidate = normalized_gradient_step(waveform, projected, FIXED_ALPHA)
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


def selector_contract() -> dict[str, Any]:
    return {
        "candidate": CANDIDATE_D_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "structure_options": 1,
        "selection": "one_fixed_candidate_or_fail_closed",
        "selector_keys": sorted(SELECTOR_KEYS),
        "forbidden_information": [
            "candidate_exact_shimmer_db",
            "candidate_exact_avqi_components",
            "candidate_other_exact_outcomes",
            "speaker_or_case_specific_structure",
            "candidate_specific_alpha",
        ],
        "minimum_complete_cycles": MINIMUM_COMPLETE_CYCLES,
        "proxy_gap_tolerance": PROXY_GAP_TOLERANCE,
        "coefficient_smoothing_kernel": list(COEFFICIENT_SMOOTHING_KERNEL),
        "coefficient_smoothing_passes": 1,
        "formal_total_metric_step_runtime_ms": CACHE_RUNTIME_MAX_MS,
    }


def select_candidate_d(certificates: dict[str, Any]) -> bool:
    if set(certificates) != SELECTOR_KEYS:
        raise ValueError("Candidate-D selector input contract drift")
    return all(bool(certificates[key]) for key in SELECTOR_KEYS)


def candidate_topology_item(case_id: str, view: str, path: Path) -> dict[str, Any]:
    return {
        "id": f"candidate-d-topology:{case_id}",
        "case_id": case_id,
        "role": "current_output_topology",
        "path": str(path.resolve()),
        "view": view,
        "score_components": False,
        "exact_metric_topology": True,
        "highpass_mode": NUMPY_HIGHPASS_MODE,
    }


def evaluate_case(
    panel_row: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    worker: ExactShimmerTopologyWorker,
    waveform_root: Path,
) -> dict[str, Any]:
    case_id = panel_row["case_id"]
    total_started = time.perf_counter()
    base_path = Path(panel_row["base_path"])
    base_waveform = read_waveform(base_path)
    base_values = base_waveform.numpy()
    base_rows, base_request_wall_ms, base_staging = worker.refresh_current_waveforms(
        [base_topology_item(panel_row)],
        [base_values],
        highpass_mode=NUMPY_HIGHPASS_MODE,
    )
    base_topology = dict(base_rows[0])
    base_staging_ms = float(base_staging[0]["staging_ms"])
    base_refresh_ms = base_staging_ms + base_request_wall_ms
    plan_started = time.perf_counter()
    plan = build_zero_crossing_cycle_plan(base_values, base_topology)
    plan_runtime_ms = 1000.0 * (time.perf_counter() - plan_started)

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
    scale_value = float(scale)
    loss = ((proxy_before - target_shimmer_db) / scale).square()
    raw_gradient = torch.autograd.grad(loss, waveform)[0]
    projected_gradient, projection = (
        zero_crossing_shape_preserving_gradient_projection(
            waveform,
            raw_gradient,
            plan,
        )
    )
    candidate = normalized_gradient_step(
        waveform,
        projected_gradient,
        FIXED_ALPHA,
    )
    synchronize(device)
    gradient_runtime_ms = 1000.0 * (time.perf_counter() - gradient_started)

    candidate_path = waveform_root / f"{case_id}__candidate_d_v17.wav"
    write_started = time.perf_counter()
    sf.write(
        candidate_path,
        candidate.detach().cpu().numpy(),
        SAMPLE_RATE,
        subtype="PCM_24",
    )
    write_ms = 1000.0 * (time.perf_counter() - write_started)
    proxy_after_started = time.perf_counter()
    stored = read_waveform(candidate_path)
    stored_values = stored.numpy()
    stored_device = stored.to(device)
    with torch.inference_mode():
        proxy_after = predictor.raw_shimmer_from_pulse_positions(
            stored_device,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=int(
                base_topology["metric_constant_prefix_samples"]
            ),
        )[1]
    synchronize(device)
    stored_proxy_runtime_ms = 1000.0 * (
        time.perf_counter() - proxy_after_started
    )
    normalized_proxy_gap_before = abs(float(proxy_before.detach()) - target_shimmer_db) / max(
        scale_value,
        1e-8,
    )
    normalized_proxy_gap_after = abs(float(proxy_after.detach()) - target_shimmer_db) / max(
        scale_value,
        1e-8,
    )
    safety_started = time.perf_counter()
    safety = waveform_safety(base_values, stored_values)
    finite_safety_pass = (
        bool(np.isfinite(stored_values).all())
        and float(np.max(np.abs(stored_values), initial=0.0)) < 0.999
        and safety["residual_rms_db"] <= RESIDUAL_CEILING_DB
        and safety["cosine_similarity"] >= MINIMUM_COSINE
        and safety["clip_fraction"] <= MAXIMUM_CLIP_FRACTION
    )
    pcm24 = pcm24_effective_step(base_path, candidate_path)
    safety_pcm24_runtime_ms = 1000.0 * (
        time.perf_counter() - safety_started
    )
    candidate_rows, candidate_request_wall_ms = worker.refresh(
        [candidate_topology_item(case_id, panel_row["view"], candidate_path)]
    )
    candidate_topology = dict(candidate_rows[0])
    stability = topology_stability(base_topology, candidate_topology)
    synchronize(device)
    before_selector_ms = 1000.0 * (time.perf_counter() - total_started)
    certificates = {
        "projected_gradient_valid": projection["projected_gradient_valid"],
        "complete_cycle_support_pass": (
            projection["complete_cycle_count"] >= MINIMUM_COMPLETE_CYCLES
        ),
        "proxy_nonregression_pass": (
            normalized_proxy_gap_after
            <= normalized_proxy_gap_before + PROXY_GAP_TOLERANCE
        ),
        "finite_safety_pass": finite_safety_pass,
        "pcm24_effective_step_pass": pcm24["pcm24_effective_step_pass"],
        "topology_stability_pass": stability["topology_stability_pass"],
        "runtime_gate_pass": before_selector_ms <= CACHE_RUNTIME_MAX_MS,
    }
    selector_pass = select_candidate_d(certificates)
    total_runtime_ms = 1000.0 * (time.perf_counter() - total_started)
    if total_runtime_ms > CACHE_RUNTIME_MAX_MS:
        certificates["runtime_gate_pass"] = False
        selector_pass = False
    return {
        "case_id": case_id,
        "base_topology": base_topology,
        "candidate_topology": candidate_topology,
        "base_topology_sha256": topology_sha256(base_topology),
        "candidate_topology_sha256": topology_sha256(candidate_topology),
        "candidate_path": candidate_path,
        "candidate_sha256": sha256_file(candidate_path),
        "proxy_before": float(proxy_before.detach()),
        "proxy_after_frozen_topology": float(proxy_after.detach()),
        "proxy_target": target_shimmer_db,
        "proxy_loss": float(loss.detach()),
        "normalized_proxy_gap_before": normalized_proxy_gap_before,
        "normalized_proxy_gap_after": normalized_proxy_gap_after,
        "raw_gradient_l2_norm": float(raw_gradient.norm()),
        "projected_gradient_l2_norm": float(projected_gradient.norm()),
        "projected_gradient_rms": float(
            projected_gradient.square().mean().sqrt()
        ),
        "projection": projection,
        "certificates": certificates,
        "selector_pass": selector_pass,
        "total_metric_step_runtime_ms": total_runtime_ms,
        "base_refresh_runtime_ms": base_refresh_ms,
        "base_refresh_request_wall_ms": base_request_wall_ms,
        "base_refresh_client_staging_ms": base_staging_ms,
        "gradient_projection_runtime_ms": gradient_runtime_ms,
        "zero_crossing_plan_runtime_ms": plan_runtime_ms,
        "pcm24_write_ms": write_ms,
        "stored_proxy_runtime_ms": stored_proxy_runtime_ms,
        "safety_pcm24_runtime_ms": safety_pcm24_runtime_ms,
        "candidate_refresh_request_wall_ms": candidate_request_wall_ms,
        "candidate_refresh_internal_ms": float(
            candidate_topology["pulse_runtime_ms"]
        ),
        **safety,
        **pcm24,
        **stability,
    }


def preselection_row(panel_row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": panel_row["case_id"],
        "speaker_id": panel_row["speaker_id"],
        "view": panel_row["view"],
        "condition": panel_row["condition"],
        "sample_group": panel_row["sample_group"],
        "candidate": CANDIDATE_D_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "candidate_path": str(record["candidate_path"].resolve()),
        "candidate_sha256": record["candidate_sha256"],
        "base_topology_sha256": record["base_topology_sha256"],
        "candidate_topology_sha256": record["candidate_topology_sha256"],
        "proxy_before": record["proxy_before"],
        "proxy_after_frozen_topology": record["proxy_after_frozen_topology"],
        "proxy_target": record["proxy_target"],
        "normalized_proxy_gap_before": record["normalized_proxy_gap_before"],
        "normalized_proxy_gap_after": record["normalized_proxy_gap_after"],
        "raw_gradient_l2_norm": record["raw_gradient_l2_norm"],
        "projected_gradient_l2_norm": record["projected_gradient_l2_norm"],
        "projected_gradient_rms": record["projected_gradient_rms"],
        "complete_cycle_count": record["projection"]["complete_cycle_count"],
        "complete_cycle_group_count": record["projection"][
            "complete_cycle_group_count"
        ],
        "complete_cycle_pulse_fraction": record["projection"][
            "complete_cycle_pulse_fraction"
        ],
        "supported_sample_count": record["projection"][
            "supported_sample_count"
        ],
        "boundary_abs_amplitude_maximum": record["projection"][
            "boundary_abs_amplitude_maximum"
        ],
        "base_pulse_count": record["base_topology"]["pulse_count"],
        "candidate_pulse_count": record["candidate_topology"]["pulse_count"],
        "reference_to_candidate_match_rate_16_samples": record[
            "reference_to_candidate_match_rate_16_samples"
        ],
        "candidate_to_reference_match_rate_16_samples": record[
            "candidate_to_reference_match_rate_16_samples"
        ],
        "candidate_reference_pulse_count_ratio": record[
            "candidate_reference_pulse_count_ratio"
        ],
        "topology_stability_pass": record["topology_stability_pass"],
        "residual_rms_db": record["residual_rms_db"],
        "cosine_similarity": record["cosine_similarity"],
        "clip_fraction": record["clip_fraction"],
        "pcm24_changed_samples": record["pcm24_changed_samples"],
        "pcm24_residual_rms_lsb": record["pcm24_residual_rms_lsb"],
        **record["certificates"],
        "selector_pass": record["selector_pass"],
        "total_metric_step_runtime_ms": record["total_metric_step_runtime_ms"],
        "base_refresh_runtime_ms": record["base_refresh_runtime_ms"],
        "gradient_projection_runtime_ms": record[
            "gradient_projection_runtime_ms"
        ],
        "zero_crossing_plan_runtime_ms": record[
            "zero_crossing_plan_runtime_ms"
        ],
        "pcm24_write_ms": record["pcm24_write_ms"],
        "stored_proxy_runtime_ms": record["stored_proxy_runtime_ms"],
        "safety_pcm24_runtime_ms": record["safety_pcm24_runtime_ms"],
        "candidate_refresh_request_wall_ms": record[
            "candidate_refresh_request_wall_ms"
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism = aggregate_candidate(CANDIDATE_D_NAME, rows)
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
        "topology_stability": all(row["topology_stability_pass"] for row in rows),
    }
    integration_gates = {
        "mechanism": all(mechanism_gates.values()),
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
        "selector_coverage": all(row["selector_pass"] for row in rows),
        "selector_uses_no_candidate_exact_outcome": all(
            row["selector_uses_no_candidate_exact_outcome"] for row in rows
        ),
        "pcm24_effective_step": all(
            row["pcm24_effective_step_pass"] for row in rows
        ),
        "complete_cycle_support": all(
            row["complete_cycle_count"] >= MINIMUM_COMPLETE_CYCLES
            for row in rows
        ),
        "topology_rebound": all(
            row["candidate_topology_rebound"] for row in rows
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
        "candidate": CANDIDATE_D_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "mechanism": mechanism,
        "mechanism_gates": mechanism_gates,
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "integration_gates": integration_gates,
        "total_metric_step_runtime_ms": {
            "median": median(row["total_metric_step_runtime_ms"] for row in rows),
            "maximum": max(row["total_metric_step_runtime_ms"] for row in rows),
            "formal_gate_ms": CACHE_RUNTIME_MAX_MS,
        },
        "all_gates_pass": all(integration_gates.values()),
    }


def write_receipt(
    args: argparse.Namespace,
    decision: str,
    report_path: Path,
    preselection_path: Path,
    selector_seal_path: Path | None,
    results_path: Path | None,
) -> None:
    artifacts = {
        report_path.name: sha256_file(report_path),
        preselection_path.name: sha256_file(preselection_path),
    }
    for path in (selector_seal_path, results_path):
        if path is not None:
            artifacts[path.name] = sha256_file(path)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-source-informed-v17-receipt-v1",
        "decision": decision,
        "candidate": CANDIDATE_D_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "dev_only": True,
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": artifacts,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head() != args.source_commit:
        raise ValueError("Candidate-D source commit differs from repository HEAD")
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
        "pulse_diagnostic_report": validate_hash(
            args.pulse_diagnostic_report,
            args.pulse_diagnostic_report_sha256,
            "v17 pulse diagnostic report",
        ),
        "pulse_diagnostic_receipt": validate_hash(
            args.pulse_diagnostic_receipt,
            args.pulse_diagnostic_receipt_sha256,
            "v17 pulse diagnostic receipt",
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
    }
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_tree_hash
    diagnostic_report = read_json(args.pulse_diagnostic_report)
    diagnostic_receipt = read_json(args.pulse_diagnostic_receipt)
    validate_diagnostic_contract(diagnostic_report, diagnostic_receipt)

    panel = read_json(args.panel_contract)
    input_results = read_csv(args.fresh_results)
    full_panel_rows, input_by_case = validate_panel(panel, input_results)
    full_by_case = {row["case_id"]: row for row in full_panel_rows}
    if not set(PROTOTYPE_CASE_IDS).issubset(full_by_case):
        raise ValueError("Candidate-D four-case coverage drift")
    panel_rows = [full_by_case[case_id] for case_id in PROTOTYPE_CASE_IDS]
    validate_dev_files(panel_rows)

    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    torch_warmup = synthetic_torch_warmup(predictor, target_scale, device)
    candidate_d_synthetic_warmup = synthetic_candidate_d_warmup(device)
    worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    worker_startup = {"startup_ms": worker.startup_ms, **worker.startup}
    worker_warmup, worker_warmup_ms = worker.warmup()
    records = []
    try:
        for index, panel_row in enumerate(panel_rows, start=1):
            records.append(
                evaluate_case(
                    panel_row,
                    float(
                        input_by_case[panel_row["case_id"]][
                            "exact_target_shimmer_db"
                        ]
                    ),
                    predictor,
                    target_scale,
                    device,
                    worker,
                    waveform_root,
                )
            )
            print(f"candidate_d_step={index}/{EXPECTED_CASE_COUNT}", flush=True)
    finally:
        worker.close()

    preselection_rows = [
        preselection_row(panel_row, record)
        for panel_row, record in zip(panel_rows, records, strict=True)
    ]
    preselection_path = args.output_dir / "candidate_d_preselection.csv"
    write_csv(preselection_path, preselection_rows)
    selector_failures = [
        record["case_id"] for record in records if not record["selector_pass"]
    ]
    common_report = {
        "schema_version": "avqi-route-c-shimmer-db-source-informed-v17-v1",
        "candidate": CANDIDATE_D_NAME,
        "route_type": "hybrid_praat_assisted_source_informed_straight_through",
        "pure_torch_estimator": False,
        "fixed_alpha": FIXED_ALPHA,
        "dev_only": True,
        "opened_cases_only": True,
        "selector_contract": selector_contract(),
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_sha256": source_hashes,
        "torch_synthetic_warmup": torch_warmup,
        "candidate_d_synthetic_warmup": candidate_d_synthetic_warmup,
        "worker_startup": worker_startup,
        "worker_synthetic_warmup": {
            "request_wall_ms": worker_warmup_ms,
            **worker_warmup,
        },
        "case_runtime": [
            {
                "case_id": record["case_id"],
                "total_metric_step_runtime_ms": record[
                    "total_metric_step_runtime_ms"
                ],
                "base_refresh_runtime_ms": record["base_refresh_runtime_ms"],
                "gradient_projection_runtime_ms": record[
                    "gradient_projection_runtime_ms"
                ],
                "zero_crossing_plan_runtime_ms": record[
                    "zero_crossing_plan_runtime_ms"
                ],
                "pcm24_write_ms": record["pcm24_write_ms"],
                "stored_proxy_runtime_ms": record[
                    "stored_proxy_runtime_ms"
                ],
                "safety_pcm24_runtime_ms": record[
                    "safety_pcm24_runtime_ms"
                ],
                "candidate_refresh_request_wall_ms": record[
                    "candidate_refresh_request_wall_ms"
                ],
                "runtime_gate_pass": record["certificates"][
                    "runtime_gate_pass"
                ],
            }
            for record in records
        ],
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    if selector_failures:
        decision = "NO_GO_SHIMMER_DB_SOURCE_INFORMED_V17_SELECTOR_4CASE"
        report = {
            **common_report,
            "decision": decision,
            "candidate_exact_outcomes_opened": False,
            "selector_failures": selector_failures,
            "selector_coverage": (
                EXPECTED_CASE_COUNT - len(selector_failures)
            )
            / EXPECTED_CASE_COUNT,
            "new_sealed_panel_authorized": False,
        }
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        write_receipt(args, decision, report_path, preselection_path, None, None)
        print(json.dumps({"decision": decision, "failures": selector_failures}))
        return

    selector_seal = {
        "schema_version": "avqi-route-c-shimmer-db-source-informed-v17-selector-seal-v1",
        "candidate": CANDIDATE_D_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_present": False,
        "selection_uses_candidate_exact_outcome": False,
        "selector_contract": selector_contract(),
        "preselection_sha256": sha256_file(preselection_path),
        "rows": [
            {
                "case_id": panel_row["case_id"],
                "candidate_path": str(record["candidate_path"].resolve()),
                "candidate_sha256": record["candidate_sha256"],
                "candidate_topology_sha256": record[
                    "candidate_topology_sha256"
                ],
                "certificates": record["certificates"],
                "total_metric_step_runtime_ms": record[
                    "total_metric_step_runtime_ms"
                ],
            }
            for panel_row, record in zip(panel_rows, records, strict=True)
        ],
    }
    selector_seal_path = args.output_dir / "selector_seal.json"
    write_json(selector_seal_path, selector_seal)

    exact_items = [
        {
            "id": f"candidate-d:{panel_row['case_id']}",
            "case_id": panel_row["case_id"],
            "role": "source_informed_candidate_d",
            "path": str(record["candidate_path"].resolve()),
            "view": panel_row["view"],
            "score_components": True,
            "exact_metric_topology": True,
        }
        for panel_row, record in zip(panel_rows, records, strict=True)
    ]
    exact_after = run_exact(exact_items, args.exact_python, args.avqi_code_root)
    after_by_case = {row["case_id"]: row for row in exact_after["rows"]}
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    rows = []
    for panel_row, record in zip(panel_rows, records, strict=True):
        case_id = panel_row["case_id"]
        input_row = input_by_case[case_id]
        after = after_by_case[case_id]
        candidate_topology_rebound = bool(
            require_exact_topology_equal(
                record["candidate_topology"],
                after,
                f"Candidate-D topology rebound {case_id}",
            )
        )
        target_components = exact_vector(input_row, "target")
        base_components = exact_vector(input_row, "before")
        after_components = exact_components(after)
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        candidate_waveform = read_waveform(record["candidate_path"])
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": CANDIDATE_D_NAME,
            "optimized_component": "shimmer_db",
            "fixed_alpha": FIXED_ALPHA,
            "candidate_path": str(record["candidate_path"].resolve()),
            "candidate_sha256": record["candidate_sha256"],
            "proxy_before": record["proxy_before"],
            "proxy_after_frozen_topology": record[
                "proxy_after_frozen_topology"
            ],
            "proxy_target": record["proxy_target"],
            "proxy_loss": record["proxy_loss"],
            "gradient_l2_norm": record["projected_gradient_l2_norm"],
            "raw_gradient_l2_norm": record["raw_gradient_l2_norm"],
            "gradient_rms": record["projected_gradient_rms"],
            "gradient_finite": record["projection"][
                "projected_gradient_finite"
            ],
            "complete_cycle_count": record["projection"][
                "complete_cycle_count"
            ],
            "complete_cycle_pulse_fraction": record["projection"][
                "complete_cycle_pulse_fraction"
            ],
            "pulse_refresh_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "torch_step_runtime_ms": record[
                "gradient_projection_runtime_ms"
            ],
            "total_metric_step_overhead_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "total_metric_step_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "selector_pass": record["selector_pass"],
            "selector_uses_no_candidate_exact_outcome": True,
            "pcm24_effective_step_pass": record[
                "pcm24_effective_step_pass"
            ],
            "pcm24_changed_samples": record["pcm24_changed_samples"],
            "pcm24_residual_rms_lsb": record["pcm24_residual_rms_lsb"],
            "base_output_exact_metric_pulse_count": int(
                record["base_topology"]["pulse_count"]
            ),
            "candidate_exact_metric_pulse_count": int(after["pulse_count"]),
            "metric_reconstruction_max_pcm16_error": int(
                record["base_topology"][
                    "metric_reconstruction_max_pcm16_error"
                ]
            ),
            "metric_reconstruction_differing_samples": int(
                record["base_topology"][
                    "metric_reconstruction_differing_samples"
                ]
            ),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                after["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                after["metric_reconstruction_differing_samples"]
            ),
            "candidate_topology_rebound": candidate_topology_rebound,
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
        row.update(topology_stability(record["base_topology"], after))
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

    results_path = args.output_dir / "candidate_d_results.csv"
    write_csv(results_path, rows)
    summary = summarize(rows)
    decision = (
        "PASS_SHIMMER_DB_SOURCE_INFORMED_V17_4CASE_MECHANISM"
        if summary["all_gates_pass"]
        else "NO_GO_SHIMMER_DB_SOURCE_INFORMED_V17_4CASE_MECHANISM"
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
    write_receipt(
        args,
        decision,
        report_path,
        preselection_path,
        selector_seal_path,
        results_path,
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


if __name__ == "__main__":
    main()
