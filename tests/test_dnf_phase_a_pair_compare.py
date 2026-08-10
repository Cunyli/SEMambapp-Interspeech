import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_dnf_phase_a_pair.py"
)
SPEC = importlib.util.spec_from_file_location("compare_dnf_phase_a_pair", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(offset: float) -> dict[str, dict]:
    return {
        f"uid-{index}": {
            "sample_uid": f"uid-{index}",
            "output_name": "test",
            "route": "noisy",
            "noise_family": "fan",
            "target_snr_db": 5.0,
            "evaluation_input_view": MODULE.DEPLOYMENT_VIEW,
            "weak_degradation": False,
            "si_sdri_db": float(index + offset),
            "sdri_db": float(index + offset),
            "si_sdr_db": float(index + offset),
            "sdr_db": float(index + offset),
            "gain_db_to_clean": 0.0,
            "gain_db_to_input": 0.0,
            "active_gain_db_to_clean": 0.0,
            "active_gain_db_to_input": 0.0,
        }
        for index in range(10)
    }


def test_paired_difference_preserves_constant_offset():
    result = MODULE.paired_difference(
        rows(0.75),
        rows(0.0),
        "si_sdri_db",
        seed=1234,
        samples=1000,
    )
    assert result["mean"] == pytest.approx(0.75)
    assert result["bootstrap_ci95_low"] == pytest.approx(0.75)
    assert result["bootstrap_ci95_high"] == pytest.approx(0.75)


def test_bootstrap_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.bootstrap_summary(
            np.asarray([1.0, np.nan]),
            seed=1,
            samples=10,
        )


def test_paired_difference_rejects_uid_mismatch():
    with pytest.raises(ValueError, match="UID mismatch"):
        MODULE.paired_difference(
            rows(0.0),
            {"other": next(iter(rows(0.0).values()))},
            "si_sdri_db",
            seed=1,
            samples=10,
        )


def test_summarize_output_exposes_gain_tails_without_mean_cancellation():
    sample_rows = rows(0.0)
    for index, row in enumerate(sample_rows.values()):
        row["active_gain_db_to_input"] = -4.0 if index % 2 else 4.0
    summary = MODULE.summarize_output(sample_rows)
    gain = summary["active_gain_db_to_input"]
    assert gain["mean"] == pytest.approx(0.0)
    assert gain["p05"] < -3.0
    assert gain["p95"] > 3.0


def test_validate_row_contract_rejects_route_mismatch():
    standard = rows(0.0)
    eq14 = rows(0.0)
    speech = rows(0.0)
    eq14["uid-0"]["route"] = "clean_weak"
    with pytest.raises(ValueError, match="row contract mismatch"):
        MODULE.validate_row_contract(standard, eq14, speech)


def test_expected_route_counts_are_exact_for_200_rows():
    contract = {"training": {"validation_samples": 200}}
    assert MODULE.expected_route_counts(contract) == {
        "noisy": 150,
        "clean_regular": 40,
        "clean_weak": 10,
    }
