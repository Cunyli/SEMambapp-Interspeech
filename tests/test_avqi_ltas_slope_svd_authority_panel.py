from pathlib import Path

from scripts.evaluate_avqi_ltas_slope_svd_authority_panel import (
    EXACT_SCORER,
    PANEL_ROWS,
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


def test_panel_is_frozen_to_24_unique_speakers_and_sessions() -> None:
    speakers = [row[0] for row in PANEL_ROWS]
    sessions = [row[1] for row in PANEL_ROWS]
    sexes = [row[2] for row in PANEL_ROWS]

    assert len(PANEL_ROWS) == 24
    assert len(set(speakers)) == 24
    assert len(set(sessions)) == 24
    assert sexes.count("female") == 12
    assert sexes.count("male") == 12


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
