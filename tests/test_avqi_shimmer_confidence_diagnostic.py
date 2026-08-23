from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_avqi_shimmer_vctk_topology_audit import (
    pulse_period_consistency,
    true_run_lengths,
    validate_fresh_output_dir,
)


def touch_job_logs(output_dir: Path, job_id: str) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True)
    for name in (
        f"slurm_{job_id}.out",
        f"slurm_{job_id}.err",
        f"shimmer_confidence_{job_id}.log",
    ):
        (log_dir / name).touch()
    return log_dir


def test_output_contract_accepts_only_current_slurm_logs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    touch_job_logs(output_dir, "123")

    validate_fresh_output_dir(output_dir, "123")


@pytest.mark.parametrize("intruder", ("outputs", "stale_log"))
def test_output_contract_rejects_existing_run_content(
    tmp_path: Path, intruder: str
) -> None:
    output_dir = tmp_path / "run"
    log_dir = touch_job_logs(output_dir, "123")
    if intruder == "outputs":
        (output_dir / "outputs").mkdir()
    else:
        (log_dir / "slurm_122.out").touch()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_fresh_output_dir(output_dir, "123")


def test_output_contract_keeps_direct_runs_strict(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    touch_job_logs(output_dir, "123")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_fresh_output_dir(output_dir, "")


def test_pulse_period_consistency_tracks_local_period_changes() -> None:
    scores = pulse_period_consistency(np.asarray([0.0, 10.0, 20.0, 31.0]))

    assert scores.tolist() == pytest.approx([1.0, 1.0, 10.0 / 11.0, 1.0])


def test_true_run_lengths_reports_contiguous_mismatch_blocks() -> None:
    lengths = true_run_lengths(np.asarray([False, True, True, False, True]))

    assert lengths.tolist() == [2, 1]
