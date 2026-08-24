from __future__ import annotations

import runpy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_namespace() -> dict[str, object]:
    return runpy.run_path(
        REPO_ROOT
        / "scripts"
        / "finalize_avqi_shimmer_db_candidate_c_fresh_panel.py"
    )


def test_finalizer_binds_original_sealed_job() -> None:
    namespace = load_namespace()

    assert namespace["SEALED_SOURCE_COMMIT"] == (
        "60dd0fe9dc748ebb793937e67aa0e38a7909876f"
    )
    assert namespace["SEALED_JOB_ID"] == "19906678"


def test_finalizer_reconstructs_pcm24_only_in_memory() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "finalize_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    assert "buffer = io.BytesIO()" in source
    assert 'format="WAV"' in source
    assert 'subtype="PCM_24"' in source
    assert '"simulation_rerun": False' in source
    assert '"generator_inference_rerun": False' in source
    assert '"candidate_step_rerun_or_written": False' in source
    assert "apply_degradation_with_wind" not in source
    assert "load_generator" not in source


def test_finalizer_uses_sealed_runtime_for_unchanged_gate() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "finalize_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    assert 'seal_row["pulse_refresh_runtime_ms"]' in source
    assert '"frozen_gate_maximum": CACHE_RUNTIME_MAX_MS' in source
    assert "max(sealed_runtimes) <= CACHE_RUNTIME_MAX_MS" in source


def test_target_does_not_request_live_metric_topology_in_primary_evaluator() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    assert '"exact_metric_topology": role != "target"' in source
    assert 'topology_required = row.get("role") != "target"' in source


def test_finalizer_runner_is_hash_bound_and_training_free() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_candidate_c_fresh_panel_finalize.sh"
    ).read_text(encoding="utf-8")

    assert "28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2" in source
    assert "4d6a4f43d2a982e8d1862abc5bc722f44908d6221b1ff867064fbc44ab53fdd9" in source
    assert "c7ed5dc5aa36ddcd8a807dc77400ba3c6524ff3dd6f8a8873e3f3d1c1fc8ecd6" in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
    assert "load_generator" not in source
    python_lines = [
        line for line in source.splitlines() if line.lstrip().startswith("python ")
    ]
    assert all("train" not in line.lower() for line in python_lines)
