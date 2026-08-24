from __future__ import annotations

import numpy as np

from scripts.diagnose_avqi_shimmer_db_pulse_alignment_v17 import (
    ALPHA_LADDER,
    contiguous_runs,
    locate_metric_position,
    metric_range_layout,
    pulse_alignment,
    route_candidate_d,
)


def topology(pulses: list[float]) -> dict[str, object]:
    return {
        "pulse_positions_samples": pulses,
        "pulse_count": len(pulses),
        "metric_constant_prefix_samples": 2,
        "metric_source_ranges": [[100, 30], [200, 30]],
        "metric_sample_count": 62,
    }


def test_metric_range_layout_maps_back_to_source() -> None:
    layout = metric_range_layout(topology([5.0, 15.0, 25.0]))
    assert layout[0]["metric_start_sample"] == 2
    assert layout[1]["metric_start_sample"] == 32
    assert locate_metric_position(37.5, layout) == {
        "range_index": 1,
        "source_position_sample": 205.5,
    }


def test_contiguous_runs_preserve_disjoint_mismatch_blocks() -> None:
    runs = contiguous_runs(np.asarray([1, 2, 3, 7, 9, 10]))
    assert [run.tolist() for run in runs] == [[1, 2, 3], [7], [9, 10]]


def test_pulse_alignment_reports_one_contiguous_alternate_path() -> None:
    base = topology([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])
    candidate = topology([5.0, 15.0, 62.0, 72.0, 82.0, 92.0])
    summary, rows, runs = pulse_alignment(
        base,
        candidate,
        alpha=ALPHA_LADDER[0],
        backtrack_index=0,
    )
    assert len(rows) == 6
    assert summary["source_ranges_equal"] is True
    assert summary["mismatch_run_count"] == 1
    assert runs[0]["base_pulse_index_start"] == 3
    assert runs[0]["base_pulse_index_end"] == 4


def test_routing_uses_only_fixed_range_and_mismatch_geometry() -> None:
    rows = []
    for index, alpha in enumerate(ALPHA_LADDER):
        rows.append(
            {
                "alpha": alpha,
                "source_ranges_equal": True,
                "mismatch_run_count": 1,
                "mismatch_runs": [
                    {
                        "base_pulse_index_start": 10,
                        "base_pulse_index_end": 40,
                        "internal_range_boundary_count": 2,
                    }
                ],
            }
        )
    routing = route_candidate_d(rows)
    assert routing["diagnostic_class"].startswith("localized_contiguous")
    assert routing["candidate_d_family"].startswith("pitch_synchronous")
