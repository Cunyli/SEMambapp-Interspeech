from __future__ import annotations

import copy

import pytest

from scripts.avqi_shimmer_exact_topology_runtime import (
    ALLOWED_CURRENT_OUTPUT_ROLES,
    EXPECTED_IMPLEMENTATION,
    require_exact_topology_equal,
    topology_sha256,
)
from scripts.evaluate_avqi_shimmer_db_runtime_v15_equivalence import (
    DEV_ENGINEERING_MARGIN_MS,
    FORMAL_REFRESH_GATE_MS,
    PASS_DECISION,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import FIXED_ALPHA


def topology() -> dict[str, object]:
    return {
        "scoring_status": "ok",
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "source_sample_count": 20,
        "metric_sample_count": 10,
        "metric_constant_prefix_samples": 2,
        "metric_source_range_count": 2,
        "metric_mapped_sample_count": 8,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_count": 3,
        "metric_source_ranges": [[1, 4], [10, 4]],
        "pulse_positions_samples": [1.25, 4.5, 8.75],
    }


def test_v15_scientific_and_runtime_contract_is_frozen() -> None:
    assert FIXED_ALPHA == 0.001
    assert FORMAL_REFRESH_GATE_MS == 500.0
    assert DEV_ENGINEERING_MARGIN_MS == 450.0
    assert PASS_DECISION.endswith("FREEZE_FOR_NEW_PANEL")
    assert EXPECTED_IMPLEMENTATION == (
        "exact_vectorized_frames_reused_tmpfs_numpy_sounding_v15"
    )
    assert ALLOWED_CURRENT_OUTPUT_ROLES == {
        "current_output_topology",
        "current_s3_500_output_topology",
    }


def test_exact_topology_equality_requires_pulse_and_source_identity() -> None:
    reference = topology()
    candidate = copy.deepcopy(reference)
    observed_hash = require_exact_topology_equal(reference, candidate, "same")
    assert observed_hash == topology_sha256(reference)

    candidate["metric_source_ranges"] = [[1, 4], [11, 4]]
    with pytest.raises(ValueError, match="topology drift"):
        require_exact_topology_equal(reference, candidate, "ranges")

    candidate = copy.deepcopy(reference)
    candidate["pulse_positions_samples"] = [1.25, 4.5, 8.76]
    with pytest.raises(ValueError, match="topology drift"):
        require_exact_topology_equal(reference, candidate, "pulses")


def test_exact_topology_equality_fails_without_metric_parity() -> None:
    candidate = topology()
    candidate["metric_reconstruction_differing_samples"] = 1
    with pytest.raises(ValueError, match="lacks parity"):
        require_exact_topology_equal(topology(), candidate, "parity")
