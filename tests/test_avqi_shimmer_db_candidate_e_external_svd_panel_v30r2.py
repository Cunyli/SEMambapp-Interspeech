from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import scripts.prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30r2 as v30r2


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "avqi_route_c_shimmer_db_candidate_e_external_svd_panel_v30r2.json"
)


def speaker(
    speaker_id: str,
    sex: str,
    rank: int,
) -> dict[str, object]:
    return {
        "speaker_id": speaker_id,
        "session_id": str(1000 + int(speaker_id)),
        "sex": sex,
        "diagnosis": "record-only",
        "cs_path": Path(f"/tmp/{speaker_id}-cs.wav"),
        "sv_path": Path(f"/tmp/{speaker_id}-sv.wav"),
        "cs_duration_seconds": 5.0,
        "sv_duration_seconds": 2.0,
        "selection_rank_within_sex": rank,
    }


def original_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sex, ids in (("female", ("1", "2", "3")), ("male", ("5", "6", "7"))):
        for slot, speaker_id in enumerate(ids):
            for view in v30r2.v24.VIEWS:
                rows.append(
                    {
                        "case_id": f"sealed_external_svd__SVD_{speaker_id}__{view}__x",
                        "panel_speaker_id": f"SVD:{speaker_id}",
                        "speaker_id": speaker_id,
                        "sex": sex,
                        "view": view,
                        "recipe_index": 936 + 2 * (slot + (0 if sex == "female" else 3)),
                    }
                )
    return rows


def test_v30r2_config_preregisters_boolean_only_generic_amendment() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v30r2.validate_v30r2_config(config)
    amendment = config["scorability_amendment"]
    assert amendment["target_scalar_values_retained_or_used"] is False
    assert amendment["target_shimmer_percent_scorability_boolean_used"] is True
    assert amendment["target_shimmer_db_scorability_boolean_used"] is True
    assert amendment["speaker_or_case_identity_hardcoded"] is False
    inheritance = config["retained_artifact_contract"]
    assert inheritance["mode"] == v30r2.INHERITANCE_MODE
    assert inheritance["retained_rerun_outputs_diagnostic_only"] is True
    assert inheritance["retained_final_artifacts_byte_inherited"] is True
    assert inheritance["uses_base_or_candidate_exact_outcomes"] is False
    assert inheritance["failed_run_evidence"]["job_id"] == "20041442"
    assert (
        inheritance["failed_run_evidence"]["failure_not_reinterpreted_as_pass"]
        is True
    )
    assert config["immutable_boundaries"]["generator_optimizer_steps"] == 0


def test_scorability_subprocess_accepts_only_boolean_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "parselmouth_version": "test",
        "praat_version": "test",
        "rows": [
            {
                "id": "raw:SVD:1:cs",
                "shimmer_percent_scorable": True,
                "shimmer_db_scorable": True,
                "component_pair_scorable": True,
                "failure_class": "none",
            }
        ],
    }
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=v30r2.EXACT_MARKER + json.dumps(payload),
        stderr="",
    )
    monkeypatch.setattr(v30r2.subprocess, "run", lambda *args, **kwargs: completed)
    observed = v30r2.run_target_scorability(
        [{"id": "raw:SVD:1:cs", "path": "/tmp/x.wav", "view": "cs"}],
        Path("/exact/python"),
        Path("/avqi"),
    )
    assert observed == payload

    leaked = deepcopy(payload)
    leaked["rows"][0]["shimmer_db"] = 1.2
    completed.stdout = v30r2.EXACT_MARKER + json.dumps(leaked)
    with pytest.raises(ValueError, match="forbidden fields"):
        v30r2.run_target_scorability(
            [{"id": "raw:SVD:1:cs", "path": "/tmp/x.wav", "view": "cs"}],
            Path("/exact/python"),
            Path("/avqi"),
        )


def test_slot_assignment_preserves_retained_v30_recipe_slots() -> None:
    selected = {
        "female": [
            speaker("1", "female", 1),
            speaker("3", "female", 3),
            speaker("4", "female", 4),
        ],
        "male": [
            speaker("5", "male", 1),
            speaker("6", "male", 2),
            speaker("7", "male", 3),
        ],
    }
    assigned = v30r2.assign_selected_to_original_slots(selected, original_rows())
    assert [row["speaker_id"] for row in assigned] == ["1", "4", "3", "5", "6", "7"]
    cases = v30r2.build_cases(assigned)
    rank_three = [case for case in cases if case.speaker_id == "3"]
    assert [case.recipe_index for case in rank_three] == [940, 941]
    replacement = [case for case in cases if case.speaker_id == "4"]
    assert [case.recipe_index for case in replacement] == [938, 939]


def test_retained_equivalence_fails_closed_on_waveform_drift() -> None:
    common = {
        "case_id": "case-1",
        "panel_speaker_id": "SVD:1",
        "speaker_id": "1",
        "session_id": "1001",
        "view": "cs",
        "condition": "rir_only",
        "selection_digest": "e" * 64,
        "source_sha256": "a",
        "target_sha256": "b",
        "degraded_sha256": "c",
        "base_sha256": "d",
        "recipe_index": 936,
        "recipe_uid": "recipe",
        "simulation_seed": 12,
        "noise_start_sample": 34,
    }
    result = v30r2.retained_waveform_equivalence([common], [common])
    assert result["all_retained_cases_byte_identical"] is True

    drifted = {**common, "base_sha256": "changed"}
    with pytest.raises(ValueError, match="equivalence failed"):
        v30r2.retained_waveform_equivalence([drifted], [common])


def test_retained_rerun_diagnostic_separates_samples_from_final_artifacts(
    tmp_path: Path,
) -> None:
    samples = np.linspace(-0.2, 0.2, 257, dtype=np.float32)
    paths = {
        name: tmp_path / f"{name}.wav"
        for name in (
            "old-target",
            "new-target",
            "old-degraded",
            "new-degraded",
            "old-base",
            "new-base",
        )
    }
    for name in ("old-target", "new-target", "old-degraded", "new-degraded"):
        sf.write(paths[name], samples, v30r2.v24.SAMPLE_RATE, subtype="FLOAT")
    sf.write(
        paths["old-base"],
        samples,
        v30r2.v24.SAMPLE_RATE,
        subtype="FLOAT",
    )
    sf.write(
        paths["new-base"],
        samples + np.float32(1e-6),
        v30r2.v24.SAMPLE_RATE,
        subtype="FLOAT",
    )
    identity = {
        "panel_speaker_id": "SVD:1",
        "speaker_id": "1",
        "session_id": "1001",
        "view": "cs",
        "condition": "rir_only",
        "selection_digest": "e" * 64,
        "source_sha256": "a" * 64,
        "recipe_index": 936,
        "recipe_uid": "recipe",
        "simulation_seed": 12,
        "noise_start_sample": 34,
    }
    original = {
        "case_id": "case-1",
        **identity,
        "target_path": str(paths["old-target"]),
        "degraded_path": str(paths["old-degraded"]),
        "base_path": str(paths["old-base"]),
    }
    rerun = {
        "case_id": "case-1",
        **identity,
        "target_path": str(paths["new-target"]),
        "degraded_path": str(paths["new-degraded"]),
        "base_path": str(paths["new-base"]),
    }
    result = v30r2.retained_rerun_diagnostic([rerun], [original])
    assert result["all_target_decoded_samples_identical"] is True
    assert result["all_degraded_decoded_samples_identical"] is True
    assert result["base_rerun_sample_identical_count"] == 0
    assert result["retained_rerun_outputs_used_for_final_panel"] is False
    assert result["rows"][0]["rerun_base_used_in_final_panel"] is False


def test_retained_rerun_diagnostic_fails_closed_on_source_or_sample_drift(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.wav"
    rerun_path = tmp_path / "rerun.wav"
    samples = np.linspace(-0.2, 0.2, 257, dtype=np.float32)
    sf.write(old_path, samples, v30r2.v24.SAMPLE_RATE, subtype="FLOAT")
    sf.write(rerun_path, samples, v30r2.v24.SAMPLE_RATE, subtype="FLOAT")
    identity = {
        "panel_speaker_id": "SVD:1",
        "speaker_id": "1",
        "session_id": "1001",
        "view": "cs",
        "condition": "rir_only",
        "selection_digest": "e" * 64,
        "source_sha256": "a" * 64,
        "recipe_index": 936,
        "recipe_uid": "recipe",
        "simulation_seed": 12,
        "noise_start_sample": 34,
    }
    original = {
        "case_id": "case-1",
        **identity,
        "target_path": str(old_path),
        "degraded_path": str(old_path),
        "base_path": str(old_path),
    }
    source_drift = {
        **original,
        "source_sha256": "b" * 64,
        "target_path": str(rerun_path),
        "degraded_path": str(rerun_path),
        "base_path": str(rerun_path),
    }
    with pytest.raises(ValueError, match="identity drift"):
        v30r2.retained_rerun_diagnostic([source_drift], [original])

    sf.write(
        rerun_path,
        samples + np.float32(1e-4),
        v30r2.v24.SAMPLE_RATE,
        subtype="FLOAT",
    )
    sample_drift = {
        **original,
        "target_path": str(rerun_path),
        "degraded_path": str(rerun_path),
        "base_path": str(rerun_path),
    }
    with pytest.raises(ValueError, match="simulation sample drift"):
        v30r2.retained_rerun_diagnostic([sample_drift], [original])


def test_retained_artifacts_are_inherited_by_case_id_without_speaker_hardcode(
    tmp_path: Path,
) -> None:
    old_paths = {
        field: tmp_path / f"old-{field}.wav"
        for field in ("target", "degraded", "base")
    }
    retained_paths = {
        field: tmp_path / f"retained-{field}.wav"
        for field in ("target", "degraded", "base")
    }
    replacement_paths = {
        field: tmp_path / f"replacement-{field}.wav"
        for field in ("target", "degraded", "base")
    }
    for index, field in enumerate(("target", "degraded", "base"), start=1):
        old_paths[field].write_bytes(bytes([index]) * 16)
        retained_paths[field].write_bytes(b"rerun")
        replacement_paths[field].write_bytes(b"replacement")
    original = {
        "case_id": "retained-case",
        **{
            f"{field}_path": str(path)
            for field, path in old_paths.items()
        },
        **{
            f"{field}_sha256": v30r2.v24.sha256_file(path)
            for field, path in old_paths.items()
        },
    }
    prepared = [
        SimpleNamespace(
            spec=SimpleNamespace(case_id="retained-case"),
            **{
                f"{field}_path": path
                for field, path in retained_paths.items()
            },
        ),
        SimpleNamespace(
            spec=SimpleNamespace(case_id="replacement-case"),
            **{
                f"{field}_path": path
                for field, path in replacement_paths.items()
            },
        ),
    ]
    result = v30r2.inherit_retained_v30_artifacts(prepared, [original])
    assert result["retained_case_count"] == 1
    assert result["replacement_case_count"] == 1
    assert result["retained_final_artifacts_byte_inherited"] is True
    assert result["speaker_or_case_identity_hardcoded"] is False
    for field in ("target", "degraded", "base"):
        assert retained_paths[field].read_bytes() == old_paths[field].read_bytes()
        assert replacement_paths[field].read_bytes() == b"replacement"
        artifact = result["rows"][0]["artifacts"][field]
        assert artifact["source_path"] == str(old_paths[field].resolve())
        assert artifact["destination_path"] == str(
            retained_paths[field].resolve()
        )
        assert artifact["sha256"] == v30r2.v24.sha256_file(old_paths[field])


def test_v30r2_ledger_preserves_old_entries_and_adds_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = {
        "entries": [
            {
                "dataset": "SVD",
                "speaker_id": speaker_id,
                "canonical_speaker_id": f"SVD:{speaker_id}",
                "panel_role": "shimmer_db_candidate_e_external_svd_v30",
            }
            for speaker_id in ("1", "2", "3", "5", "6", "7")
        ]
    }
    monkeypatch.setattr(v30r2.v24, "validate_prior_ledger", lambda value: set())
    selected = {
        "female": [
            speaker("1", "female", 1),
            speaker("3", "female", 3),
            speaker("4", "female", 4),
        ],
        "male": [
            speaker("5", "male", 1),
            speaker("6", "male", 2),
            speaker("7", "male", 3),
        ],
    }
    cases = v30r2.build_cases(
        v30r2.assign_selected_to_original_slots(selected, original_rows())
    )
    output = v30r2.extend_prior_ledger_v30r2(
        ledger,
        cases,
        original_rows(),
        "c" * 40,
        "d" * 64,
    )
    by_id = {entry["canonical_speaker_id"]: entry for entry in output["entries"]}
    assert by_id["SVD:2"]["panel_role"] == (
        "shimmer_db_candidate_e_external_svd_v30"
    )
    assert by_id["SVD:2"]["candidate_e_v30r2_status"] == (
        "target_component_unscorable_not_selected"
    )
    assert by_id["SVD:4"]["panel_role"] == (
        "shimmer_db_candidate_e_external_svd_v30r2"
    )
    assert output["target_scalar_values_used_for_selection"] is False


def test_v30r2_source_has_no_candidate_exact_or_training_execution() -> None:
    source = Path(v30r2.__file__).read_text(encoding="utf-8")
    assert "build_candidate_pool" not in source
    assert "torch.optim" not in source
    assert '"target_shimmer_scalar_values_opened": False' in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"generator_optimizer_steps": 0' in source
