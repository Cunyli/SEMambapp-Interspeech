from scripts.evaluate_avqi_ltas_slope_promotion import (
    BASE_EXACT_MATERIAL_DISTANCE_MIN,
    exact_relative_gate_with_floor,
    repeatability_material_floor,
)


def summaries(
    *,
    exact_lowpass: float = 0.05,
    candidate_ratio: float = 1.02,
) -> dict:
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
            "exact_mean_standardized_distance": exact_lowpass,
            "candidate_mean_standardized_distance": (
                exact_lowpass * candidate_ratio
            ),
            "candidate_to_exact_distance_ratio": candidate_ratio,
            "signed_direction_agreement": 1.0,
        },
    }


def test_repeatability_floor_keeps_preregistered_material_minimum() -> None:
    assert repeatability_material_floor(0.0) == (
        BASE_EXACT_MATERIAL_DISTANCE_MIN
    )
    assert repeatability_material_floor(0.0001) == (
        BASE_EXACT_MATERIAL_DISTANCE_MIN
    )


def test_repeatability_floor_rises_only_from_measured_noise() -> None:
    assert repeatability_material_floor(0.003) == 0.03


def test_exact_relative_gate_accepts_authority_aligned_sub_0_10_response() -> None:
    result = exact_relative_gate_with_floor(summaries(), 0.02)

    assert result["decision"] == "PASS"
    assert result["current_absolute_gate_passes"] is False


def test_exact_relative_gate_rejects_nonmaterial_exact_response() -> None:
    result = exact_relative_gate_with_floor(
        summaries(exact_lowpass=0.01),
        0.02,
    )

    assert result["decision"] == "FAIL"
    assert result["gates"]["exact_lowpass_is_material"] is False


def test_exact_relative_gate_rejects_candidate_authority_ratio_drift() -> None:
    result = exact_relative_gate_with_floor(
        summaries(candidate_ratio=1.5),
        0.02,
    )

    assert result["decision"] == "FAIL"
    assert result["gates"]["candidate_matches_exact_response_ratio"] is False
