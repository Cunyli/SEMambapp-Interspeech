import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_dnf_phase_a_tau_gate.py"
)
SPEC = importlib.util.spec_from_file_location("dnf_phase_a_tau_compare", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_signal_summary_exposes_worst_attenuation_and_activity():
    rows = [
        {
            "gain_db_to_input": "-0.5",
            "gain_db_to_clean": "-0.4",
            "clean_active_gain_db_to_input": "-0.6",
            "clean_active_gain_db_to_clean": "-0.5",
            "active_ratio_delta_to_clean": "-0.01",
            "si_sdri_db": "1.0",
            "sdri_db": "0.5",
            "si_sdr_db": "5.0",
            "sdr_db": "4.0",
        },
        {
            "gain_db_to_input": "-2.0",
            "gain_db_to_clean": "-1.9",
            "clean_active_gain_db_to_input": "-2.1",
            "clean_active_gain_db_to_clean": "-2.0",
            "active_ratio_delta_to_clean": "-0.08",
            "si_sdri_db": "0.0",
            "sdri_db": "-0.5",
            "si_sdr_db": "4.0",
            "sdr_db": "3.0",
        },
    ]
    summary = MODULE.signal_summary(rows)
    assert summary["n"] == 2
    assert summary["gain_db_to_input"]["mean"] == pytest.approx(-1.25)
    assert summary["clean_active_gain_db_to_clean"]["min"] == pytest.approx(
        -2.0
    )
    assert summary["active_ratio_delta_to_clean"]["min"] == pytest.approx(
        -0.08
    )
