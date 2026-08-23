from scripts.evaluate_avqi_ltas_slope_gate_alignment import exact_relative_gate


def summaries(lowpass_ratio: float = 1.0) -> dict:
    return {
        "gain_minus12db": {
            "exact_mean_standardized_distance": 0.0,
            "candidate_mean_standardized_distance": 0.0,
        },
        "circular_shift_100ms": {
            "exact_mean_standardized_distance": 0.002,
            "candidate_mean_standardized_distance": 0.003,
        },
        "lowpass_3khz": {
            "exact_mean_standardized_distance": 0.05,
            "candidate_mean_standardized_distance": 0.05 * lowpass_ratio,
            "candidate_to_exact_distance_ratio": lowpass_ratio,
            "signed_direction_agreement": 1.0,
        },
    }


def test_exact_relative_gate_passes_authority_aligned_response() -> None:
    result = exact_relative_gate(summaries())

    assert result["decision"] == "PASS"
    assert result["current_absolute_gate_passes"] is False


def test_exact_relative_gate_rejects_response_ratio_drift() -> None:
    result = exact_relative_gate(summaries(lowpass_ratio=1.5))

    assert result["decision"] == "FAIL"
    assert result["gates"]["candidate_matches_exact_response_ratio"] is False
