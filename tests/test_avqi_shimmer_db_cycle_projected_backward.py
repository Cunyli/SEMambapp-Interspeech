from __future__ import annotations

from pathlib import Path

import torch

from scripts.evaluate_avqi_shimmer_db_cycle_projected_backward import (
    FIXED_ALPHA,
    PROJECTED_CANDIDATE_NAME,
    cycle_multiplicative_gradient_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cycle_projection_is_multiplicative_within_each_detached_cell() -> None:
    waveform = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    gradient = torch.tensor([2.0, -1.0, 3.0, 1.0, -2.0, 4.0, 1.0, 2.0])
    topology = {
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": 8,
        "metric_sample_count": 8,
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": [[0, 8]],
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": 8,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": [1.0, 3.0, 5.0, 7.0],
        "pulse_count": 4,
    }

    projected = cycle_multiplicative_gradient_projection(
        waveform,
        gradient,
        topology,
    )

    for indices in ([0, 1], [2, 3], [4, 5], [6, 7]):
        ratio = projected[indices] / waveform[indices]
        assert torch.allclose(ratio, ratio[:1].expand_as(ratio))
    assert torch.isfinite(projected).all()
    assert float(projected.norm()) > 0.0


def test_cycle_projected_contract_keeps_alpha_and_discloses_route() -> None:
    assert FIXED_ALPHA == 0.001
    assert PROJECTED_CANDIDATE_NAME == (
        "praat_current_output_topology_cycle_projected_backward_db_alpha_0p001"
    )
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_cycle_projected_backward.py"
    ).read_text(encoding="utf-8")
    assert '"forward_scalar_changed": False' in source
    assert '"detached_topology_changed": False' in source
    assert '"backward_projection_has_tunable_parameters": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"NO_GO_AVQI_T2_TRAINING"' in source


def test_cycle_projected_runner_is_hash_bound_and_dev_only() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_cycle_projected_backward.sh"
    ).read_text(encoding="utf-8")
    assert "28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2" in source
    assert "b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62" in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
    assert "load_generator" not in source
    assert "optimizer" not in source.lower()
