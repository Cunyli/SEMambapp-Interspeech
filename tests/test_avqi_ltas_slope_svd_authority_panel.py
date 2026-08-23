from pathlib import Path

from scripts.evaluate_avqi_ltas_slope_svd_authority_panel import (
    EXACT_SCORER,
    PANEL_ROWS,
    PRIMARY_ROWS,
    RESERVE_ROWS,
    V9_PANEL_ROWS,
    exact_failure_receipts,
    exact_speaker_complete,
    level_metrics,
    paired_lowpass_delta,
    preregistered_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows() -> list[dict]:
    rows = []
    for index in range(8):
        clean = -6.0 + index * 0.5
        exact_delta = 0.2 + index * 0.05
        candidate_delta = exact_delta * 1.02
        rows.append(
            {
                "train_slope_scale": 4.0,
                "exact": {
                    "clean": clean,
                    "lowpass_3khz": clean + exact_delta,
                },
                "candidate_frozen_full": {
                    "clean": clean * 1.01,
                    "lowpass_3khz": clean * 1.01 + candidate_delta,
                },
            }
        )
    return rows


def test_panel_freezes_disjoint_primary_and_status_only_reserves() -> None:
    primary_speakers = [row[0] for row in PRIMARY_ROWS]
    primary_sessions = [row[1] for row in PRIMARY_ROWS]
    primary_sexes = [row[2] for row in PRIMARY_ROWS]
    reserve_speakers = [row[0] for row in RESERVE_ROWS]
    reserve_sexes = [row[2] for row in RESERVE_ROWS]
    v9_speakers = {row[0] for row in V9_PANEL_ROWS}

    assert len(PRIMARY_ROWS) == 24
    assert len(set(primary_speakers)) == 24
    assert len(set(primary_sessions)) == 24
    assert primary_sexes.count("female") == 12
    assert primary_sexes.count("male") == 12
    assert len(RESERVE_ROWS) == 8
    assert reserve_sexes.count("female") == 6
    assert reserve_sexes.count("male") == 2
    assert not set(primary_speakers) & set(reserve_speakers)
    assert not {row[0] for row in PANEL_ROWS} & v9_speakers


def test_exact_speaker_completion_uses_status_not_metric_value() -> None:
    rows = []
    for view in ("cs", "sv"):
        for variant in ("clean", "gain", "shift", "lowpass_3khz"):
            rows.append(
                {
                    "id": f"SVD:42:99:{view}:{variant}",
                    "view": view,
                    "scoring_status": "ok",
                    "slope": 1e9,
                }
            )

    assert exact_speaker_complete(rows, "SVD:42") is True
    rows[-1]["scoring_status"] = "error"
    rows[-1]["error_type"] = "PraatError"
    rows[-1]["error_message"] = "LTAS input too short"
    assert exact_speaker_complete(rows, "SVD:42") is False
    receipts = exact_failure_receipts(rows)
    assert receipts == [
        {
            "id": "SVD:42:99:sv:lowpass_3khz",
            "view": "sv",
            "scoring_status": "error",
            "error_type": "PraatError",
            "error_message": "LTAS input too short",
        }
    ]
    assert "slope" not in receipts[0]


def test_exact_scorer_uses_view_correct_authoritative_preprocessing() -> None:
    assert "length_normalize_sv(signal, 16000)" in EXACT_SCORER
    assert "get_voiced_segments(signal, 16000)" in EXACT_SCORER
    assert "highpass_filter(signal, 16000)" in EXACT_SCORER
    assert "get_slope(avqi_input, 16000)" in EXACT_SCORER


def test_external_level_and_lowpass_delta_reuse_frozen_gates() -> None:
    rows = synthetic_rows()

    assert level_metrics(rows)["decision"] == "PASS"
    assert paired_lowpass_delta(rows)["decision"] == "PASS"


def test_contract_keeps_production_gate_unchanged() -> None:
    contract = preregistered_contract()

    assert contract["candidate_to_exact_distance_ratio"] == [0.75, 1.25]
    assert contract["current_absolute_lowpass_min_unchanged"] == 0.10
    assert contract["production_gate_changed"] is False


def test_runner_requires_separate_seal_and_score_stages() -> None:
    source = (
        REPO_ROOT / "scripts/run_avqi_ltas_slope_svd_authority_panel.sh"
    ).read_text(encoding="utf-8")

    assert 'STAGE="${STAGE:-seal}"' in source
    assert "Score stage requires PANEL_SEAL_SHA256" in source
    assert "Refusing to reopen an already scored SVD LTAS panel" in source
    assert "CONFIRM_SLURM_SUBMIT" in source
    assert "avqi_route_c_ltas_slope_svd_authority_v10_20260823_01" in source
    assert (
        "465c15e46c9c9e325c14e5672abead050bbfd9a4bba75d0ace46bf5d58884966"
        in source
    )
