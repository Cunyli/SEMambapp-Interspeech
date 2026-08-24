from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RECORD_MAX_BYTES,
    CANDIDATE_NAMES,
    CURRENT_OUTPUT_REFRESH_ALPHAS,
    CURRENT_OUTPUT_REFRESH_CANDIDATES,
    FIXED_ALPHA,
    GENERATOR_HOP_SIZE,
    REQUIRED_EFFECT_SLICES,
    candidate_alpha,
    cache_record_sha256,
    cache_record_valid,
    finalize_cache_record,
    map_input_metric_pulses_to_output,
    nearest_match_rate,
)


def test_hybrid_candidate_contract_is_fixed() -> None:
    assert FIXED_ALPHA == 1e-3
    assert CANDIDATE_NAMES == (
        "v6_db",
        "praat_input_topology_absolute_db",
        "shimmer_percent_coupled",
        "output_pulse_oracle_db",
        "praat_current_output_topology_refresh_db_alpha_0p0003",
        "praat_current_output_topology_refresh_db_alpha_0p001",
        "praat_current_output_topology_refresh_db_alpha_0p003",
    )
    assert CURRENT_OUTPUT_REFRESH_ALPHAS == (3e-4, 1e-3, 3e-3)
    assert tuple(
        candidate_alpha(candidate)
        for candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
    ) == CURRENT_OUTPUT_REFRESH_ALPHAS
    assert {
        "condition=rir_only",
        "condition=snr20",
        "condition=snr10",
    }.issubset(REQUIRED_EFFECT_SLICES)
    assert CACHE_RECORD_MAX_BYTES == 65_536


def test_nearest_match_rate_uses_sorted_detached_positions() -> None:
    source = np.asarray([10.0, 20.0, 30.0])
    target = np.asarray([9.0, 21.0, 80.0])
    assert nearest_match_rate(source, target, tolerance=2.0) == 2.0 / 3.0
    assert nearest_match_rate(source, target, tolerance=0.5) == 0.0


def test_nearest_match_rate_fails_closed_on_empty_topology() -> None:
    source = np.asarray([], dtype=np.float64)
    target = np.asarray([1.0], dtype=np.float64)
    assert nearest_match_rate(source, target) == 0.0
    assert nearest_match_rate(target, source) == 0.0


def test_cache_record_hash_and_size_fail_closed_on_mutation() -> None:
    record = finalize_cache_record(
        {
            "schema_version": "test-v1",
            "input_sha256": "abc",
            "pulse_positions_samples": [1.5, 2.5, 3.5],
        }
    )
    assert cache_record_valid(record)
    assert record["record_sha256"] == cache_record_sha256(record)

    mutated = dict(record)
    mutated["input_sha256"] = "drifted"
    assert not cache_record_valid(mutated)


def test_metric_pulse_mapping_handles_only_bounded_trailing_truncation() -> None:
    cs = map_input_metric_pulses_to_output(
        np.asarray([10.0, 94_950.0]),
        input_frame_count=94_970,
        output_frame_count=94_900,
        view="cs",
    )
    np.testing.assert_array_equal(cs, np.asarray([10.0]))

    sv = map_input_metric_pulses_to_output(
        np.asarray([10.0, 47_900.0]),
        input_frame_count=48_955,
        output_frame_count=48_900,
        view="sv",
    )
    np.testing.assert_array_equal(sv, np.asarray([65.0, 47_955.0]))
    assert GENERATOR_HOP_SIZE == 100

    with pytest.raises(ValueError, match="timeline drift"):
        map_input_metric_pulses_to_output(
            np.asarray([10.0]),
            input_frame_count=48_100,
            output_frame_count=48_000,
            view="sv",
        )
