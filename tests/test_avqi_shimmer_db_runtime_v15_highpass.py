from __future__ import annotations

from scripts.profile_avqi_shimmer_db_runtime_v15_highpass import (
    DEV_ENGINEERING_MARGIN_MS,
    FORMAL_REFRESH_GATE_MS,
    FROZEN_OUTLIER_CASE_IDS,
    PASS_DECISION,
    summarize_candidate_runtime,
)


def runtime_row(internal_ms: float, end_to_end_ms: float) -> dict[str, object]:
    return {
        "phase": "candidate_warm",
        "internal_refresh_ms": internal_ms,
        "end_to_end_refresh_ms": end_to_end_ms,
        "exact_equivalent": True,
    }


def test_highpass_probe_keeps_frozen_gate_and_outliers() -> None:
    assert FORMAL_REFRESH_GATE_MS == 500.0
    assert DEV_ENGINEERING_MARGIN_MS == 450.0
    assert FROZEN_OUTLIER_CASE_IDS == (
        "sealed_final__SD05__cs__rir_only",
        "sealed_final__ÄHH16__cs__rir_only",
    )
    assert PASS_DECISION.endswith("AUTHORIZE_FULL_DEV_EQUIVALENCE")


def test_highpass_probe_requires_exactness_and_runtime_margin() -> None:
    rows = [runtime_row(120.0, 125.0), runtime_row(430.0, 435.0)]
    summary = summarize_candidate_runtime(rows)
    assert summary["all_exact_equivalent"] is True
    assert summary["formal_500ms_pass"] is True
    assert summary["development_450ms_margin_pass"] is True

    rows[-1]["exact_equivalent"] = False
    assert summarize_candidate_runtime(rows)["all_exact_equivalent"] is False

    rows[-1]["exact_equivalent"] = True
    rows[-1]["end_to_end_refresh_ms"] = 451.0
    summary = summarize_candidate_runtime(rows)
    assert summary["formal_500ms_pass"] is True
    assert summary["development_450ms_margin_pass"] is False
