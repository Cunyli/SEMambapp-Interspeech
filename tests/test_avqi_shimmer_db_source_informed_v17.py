from __future__ import annotations

import numpy as np
import torch

from scripts.evaluate_avqi_shimmer_db_source_informed_v17 import (
    CANDIDATE_D_NAME,
    COEFFICIENT_SMOOTHING_KERNEL,
    SELECTOR_KEYS,
    build_zero_crossing_cycle_plan,
    select_candidate_d,
    selector_contract,
    zero_crossing_shape_preserving_gradient_projection,
)


def synthetic_topology(sample_count: int) -> dict[str, object]:
    return {
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": sample_count,
        "metric_sample_count": sample_count,
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": [[0, sample_count]],
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": sample_count,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": [80.0, 240.0, 400.0, 560.0, 720.0, 880.0],
        "pulse_count": 6,
    }


def test_zero_crossing_plan_builds_complete_pitch_cycles() -> None:
    sample_count = 960
    timeline = np.arange(sample_count, dtype=np.float64)
    waveform = np.sin(2.0 * np.pi * 100.0 * timeline / 16_000.0)
    plan = build_zero_crossing_cycle_plan(
        waveform,
        synthetic_topology(sample_count),
    )
    assert plan["summary"]["complete_cycle_count"] == 4
    assert plan["summary"]["source_range_joins_bridged"] is False
    assert plan["summary"]["coefficient_smoothing_kernel"] == [0.25, 0.5, 0.25]
    assert np.count_nonzero(plan["cell_ids"] >= 0) > 0


def test_projection_is_finite_multiplicative_and_nonzero() -> None:
    sample_count = 960
    timeline = torch.arange(sample_count, dtype=torch.float32)
    waveform = torch.sin(2.0 * torch.pi * 100.0 * timeline / 16_000.0)
    gradient = waveform * torch.linspace(0.5, 1.5, sample_count)
    plan = build_zero_crossing_cycle_plan(
        waveform.numpy(),
        synthetic_topology(sample_count),
    )
    projected, report = zero_crossing_shape_preserving_gradient_projection(
        waveform,
        gradient,
        plan,
    )
    assert report["projected_gradient_valid"] is True
    assert torch.isfinite(projected).all()
    assert float(projected.norm()) > 0.0
    unsupported = torch.as_tensor(plan["cell_ids"] < 0)
    assert torch.count_nonzero(projected[unsupported]) == 0


def test_plan_never_bridges_disjoint_source_ranges() -> None:
    sample_count = 960
    timeline = np.arange(sample_count, dtype=np.float64)
    waveform = np.sin(2.0 * np.pi * 100.0 * timeline / 16_000.0)
    topology = synthetic_topology(sample_count)
    topology["metric_source_ranges"] = [[0, 480], [480, 480]]
    topology["metric_source_range_count"] = 2
    plan = build_zero_crossing_cycle_plan(waveform, topology)
    crossing_cells = [
        cycle
        for cycle in plan["cycles"]
        if cycle["mapped_start_index"] < 480 < cycle["mapped_end_index"]
    ]
    assert crossing_cells == []


def test_selector_and_contract_exclude_exact_outcomes() -> None:
    certificates = {key: True for key in SELECTOR_KEYS}
    assert select_candidate_d(certificates) is True
    certificates["topology_stability_pass"] = False
    assert select_candidate_d(certificates) is False
    contract = selector_contract()
    assert contract["candidate"] == CANDIDATE_D_NAME
    assert contract["coefficient_smoothing_kernel"] == list(
        COEFFICIENT_SMOOTHING_KERNEL
    )
    assert "candidate_exact_shimmer_db" in contract["forbidden_information"]
