from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_avqi_shimmer_db_source_informed_v17 import (
    build_zero_crossing_cycle_plan as build_legacy_candidate_d_plan,
    zero_crossing_shape_preserving_gradient_projection as legacy_candidate_d_projection,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    ALPHA_LADDER,
    CANDIDATE_D_NAME,
    D_ROUTING_KEYS,
    FIXED_ALPHA,
    SELECTOR_NAME,
    TRUST_REGION_CANDIDATE_NAME,
    build_zero_crossing_cycle_plan_vectorized,
    candidate_d_projection_vectorized,
    normalized_gradient_steps_shared,
    plan_equivalence,
    select_candidate_d,
    selector_contract,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    normalized_gradient_step,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "pulse_positions_samples": [
            80.0,
            240.0,
            400.0,
            560.0,
            720.0,
            880.0,
        ],
        "pulse_count": 6,
    }


def test_vectorized_candidate_d_plan_is_exact_legacy_equivalent() -> None:
    sample_count = 960
    timeline = np.arange(sample_count, dtype=np.float64)
    waveform = (
        np.sin(2.0 * np.pi * 100.0 * timeline / 16_000.0)
        + 0.13 * np.sin(2.0 * np.pi * 211.0 * timeline / 16_000.0)
    )
    topology = synthetic_topology(sample_count)
    legacy = build_legacy_candidate_d_plan(waveform, topology)
    optimized = build_zero_crossing_cycle_plan_vectorized(waveform, topology)
    assert plan_equivalence(legacy, optimized)["all_equal"] is True


def test_vectorized_plan_preserves_disjoint_range_semantics() -> None:
    sample_count = 960
    timeline = np.arange(sample_count, dtype=np.float64)
    waveform = np.sin(2.0 * np.pi * 100.0 * timeline / 16_000.0)
    topology = synthetic_topology(sample_count)
    topology["metric_source_ranges"] = [[0, 480], [480, 480]]
    topology["metric_source_range_count"] = 2
    legacy = build_legacy_candidate_d_plan(waveform, topology)
    optimized = build_zero_crossing_cycle_plan_vectorized(waveform, topology)
    assert plan_equivalence(legacy, optimized)["all_equal"] is True
    assert optimized["summary"]["source_range_joins_bridged"] is False


def test_vectorized_projection_and_shared_steps_are_bit_equal() -> None:
    sample_count = 960
    timeline = torch.arange(sample_count, dtype=torch.float32)
    waveform = torch.sin(2.0 * torch.pi * 100.0 * timeline / 16_000.0)
    gradient = waveform * torch.linspace(0.5, 1.5, sample_count)
    legacy_plan = build_legacy_candidate_d_plan(
        waveform.numpy(),
        synthetic_topology(sample_count),
    )
    optimized_plan = build_zero_crossing_cycle_plan_vectorized(
        waveform.numpy(),
        synthetic_topology(sample_count),
    )
    legacy_projected, legacy_report = legacy_candidate_d_projection(
        waveform,
        gradient,
        legacy_plan,
    )
    optimized_projected, optimized_report = candidate_d_projection_vectorized(
        waveform,
        gradient,
        optimized_plan,
    )
    assert torch.equal(legacy_projected, optimized_projected)
    assert legacy_report == optimized_report

    reference = [
        normalized_gradient_step(waveform, gradient, alpha)
        for alpha in ALPHA_LADDER
    ]
    optimized = normalized_gradient_steps_shared(
        waveform,
        gradient,
        ALPHA_LADDER,
    )
    assert all(
        torch.equal(reference_step, optimized_step)
        for reference_step, optimized_step in zip(
            reference,
            optimized,
            strict=True,
        )
    )


def test_selector_contract_is_d_then_c_and_excludes_outcomes() -> None:
    contract = selector_contract()
    assert FIXED_ALPHA == 0.001
    assert ALPHA_LADDER == (0.001, 0.0005, 0.00025, 0.000125)
    assert contract["family_order"] == [
        CANDIDATE_D_NAME,
        TRUST_REGION_CANDIDATE_NAME,
    ]
    assert contract["candidate_d_always_attempted"] is True
    assert contract["candidate_c_complete_ladder_on_fallback"] is True
    assert contract["formal_total_metric_step_runtime_ms"] == 500.0
    assert "candidate_exact_shimmer_db" in contract["forbidden_information"]
    assert "case_id" in contract["forbidden_information"]


def test_candidate_d_routing_fails_closed_on_schema_or_certificate() -> None:
    certificates = {key: True for key in D_ROUTING_KEYS}
    assert select_candidate_d(certificates) is True
    certificates["topology_stability_pass"] = False
    assert select_candidate_d(certificates) is False
    certificates["candidate_exact_shimmer_db"] = 0.0
    with pytest.raises(ValueError, match="routing contract drift"):
        select_candidate_d(certificates)


def test_selector_seal_precedes_exact_scoring() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_topology_family_selector_v18.py"
    ).read_text(encoding="utf-8")
    seal = "write_json(selector_seal_path, selector_seal)"
    score = "exact_after = run_exact(exact_items"
    assert source.index(seal) < source.index(score)
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert SELECTOR_NAME in source


def test_runner_is_hash_bound_and_phase_gated() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_topology_family_selector_v18.sh"
    ).read_text(encoding="utf-8")
    assert "CONFIRM_SLURM_SUBMIT" in source
    assert "EQUIVALENCE_REPORT_SHA256" in source
    assert "EQUIVALENCE_RECEIPT_SHA256" in source
    assert "RUNTIME_WORKER_SCRIPT_SHA256" in source
    assert 'PHASE="${PHASE:-equivalence}"' in source
