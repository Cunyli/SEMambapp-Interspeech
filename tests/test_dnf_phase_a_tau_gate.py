import csv
import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_semambapp_dnf_phase_a_tau_gate.py"
)
SPEC = importlib.util.spec_from_file_location("dnf_phase_a_tau_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_pair_csv(path: Path) -> None:
    rows = []
    groups = (
        ("healthy_low", "healthy", ("h1", "h2", "h3")),
        ("pathological_mild", "patient", ("m1", "m2", "m3")),
        ("pathological_severe", "patient", ("s1", "s2", "s3")),
    )
    for group, label, speakers in groups:
        for speaker in speakers:
            for task in ("cs", "sv"):
                rows.append(
                    {
                        "uid": f"{speaker}-{task}",
                        "speaker_id": speaker,
                        "sample_group": group,
                        "label": label,
                        "task": task,
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_tau_small_gate_is_severity_and_task_stratified(tmp_path):
    pair_csv = tmp_path / "pairs.csv"
    write_pair_csv(pair_csv)
    first, first_receipt = MODULE.stratified_pair_rows(
        pair_csv,
        speakers_per_group=2,
        selection_seed=1234,
    )
    second, second_receipt = MODULE.stratified_pair_rows(
        pair_csv,
        speakers_per_group=2,
        selection_seed=1234,
    )
    assert first == second
    assert first_receipt == second_receipt
    assert len(first) == 12
    assert set(first_receipt["selected_speakers"]) == set(
        MODULE.REQUIRED_SAMPLE_GROUPS
    )
    assert set(first_receipt["stratum_counts"].values()) == {2}


def test_signal_stats_separates_input_and_clean_gain():
    time = torch.arange(16000, dtype=torch.float32) / 16000.0
    clean = (0.1 * torch.sin(2.0 * torch.pi * 200.0 * time)).unsqueeze(0)
    noisy = clean + (
        0.02 * torch.sin(2.0 * torch.pi * 900.0 * time)
    ).unsqueeze(0)
    enhanced = clean * 0.5
    stats = MODULE.signal_stats(noisy, clean, enhanced)
    assert stats["gain_db_to_clean"] == pytest.approx(
        -6.0206,
        abs=1.0e-3,
    )
    assert stats["clean_active_gain_db_to_clean"] == pytest.approx(
        -6.0206,
        abs=1.0e-3,
    )
    assert stats["gain_db_to_input"] < stats["gain_db_to_clean"]
    assert stats["si_sdri_db"] > 0.0
