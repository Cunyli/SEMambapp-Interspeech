from __future__ import annotations

import numpy as np

from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RECORD_MAX_BYTES,
    CANDIDATE_NAMES,
    FIXED_ALPHA,
    cache_record_sha256,
    cache_record_valid,
    finalize_cache_record,
    nearest_match_rate,
)


def test_hybrid_candidate_contract_is_fixed() -> None:
    assert FIXED_ALPHA == 1e-3
    assert CANDIDATE_NAMES == (
        "v6_db",
        "praat_input_topology_absolute_db",
        "shimmer_percent_coupled",
        "output_pulse_oracle_db",
    )
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
