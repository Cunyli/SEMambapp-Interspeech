from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.profile_avqi_shimmer_exact_topology_runtime import (
    EXACT_WORKER,
    EXPECTED_CS_CASE_IDS,
    FROZEN_ALPHA_GRID,
    PULSE_REFRESH_GATE_MS,
    metric_source_indices,
    normalized_gradient_step,
    require_same_topology,
)


def test_runtime_profile_contract_is_frozen() -> None:
    assert FROZEN_ALPHA_GRID == (3e-4, 1e-3, 3e-3)
    assert PULSE_REFRESH_GATE_MS == 500.0
    assert len(EXPECTED_CS_CASE_IDS) == 6
    assert all("__cs__" in case_id for case_id in EXPECTED_CS_CASE_IDS)
    for stage in (
        "highpass_ms",
        "textgrid_range_ms",
        "metric_gather_ms",
        "pointprocess_construct_ms",
        "pulse_enumeration_ms",
    ):
        assert stage in EXACT_WORKER


def test_metric_source_ranges_expand_fail_closed() -> None:
    topology = {
        "source_sample_count": 12,
        "metric_source_ranges": [[1, 3], [7, 3]],
        "metric_mapped_sample_count": 6,
        "metric_reconstruction_differing_samples": 0,
    }
    np.testing.assert_array_equal(
        metric_source_indices(topology, 12),
        np.asarray([1, 2, 3, 7, 8, 9]),
    )
    drifted = dict(topology)
    drifted["metric_reconstruction_differing_samples"] = 1
    with pytest.raises(ValueError, match="parity"):
        metric_source_indices(drifted, 12)


def test_topology_reuse_requires_all_identity_hashes() -> None:
    topology = {
        "source_ranges_sha256": "ranges",
        "pulse_positions_sha256": "pulses",
        "highpass_pcm16_sha256": "highpass",
        "metric_pcm16_sha256": "metric",
        "metric_sample_count": 8,
        "metric_constant_prefix_samples": 2,
        "metric_mapped_sample_count": 6,
        "pulse_count": 4,
        "metric_source_ranges": [[1, 3], [7, 3]],
        "pulse_positions_samples": [1.0, 2.0, 3.0, 4.0],
    }
    candidate = dict(topology)
    candidate["source_ranges_sha256"] = "drift"
    with pytest.raises(ValueError, match="topology changed"):
        require_same_topology(topology, candidate, "test")


def test_normalized_step_reuses_one_gradient_across_alpha_grid() -> None:
    waveform = torch.linspace(-0.2, 0.2, 16_000)
    gradient = torch.linspace(-1.0, 1.0, 16_000)
    outputs = [
        normalized_gradient_step(waveform, gradient, alpha)
        for alpha in FROZEN_ALPHA_GRID
    ]
    for alpha, candidate in zip(FROZEN_ALPHA_GRID, outputs, strict=True):
        ratio = (
            (candidate - waveform).square().mean().sqrt()
            / waveform.square().mean().sqrt()
        )
        assert float(ratio) == pytest.approx(alpha, rel=1e-4)
