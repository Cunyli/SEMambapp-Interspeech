"""Synthetic regression tests for TAU identity, leakage, and capacity gates."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.audit_avqi_route_c_tau_history_capacity_v1 import (
    BOUNDARIES,
    COMPONENTS,
    audit_sources,
    canonical_speaker,
    capacity,
    exact_rows,
    prove_current_universe_opened,
    scan_history,
    sha256_file,
    validate_contract,
    verify_bindings,
    write_json,
)


CONTRACT = Path(__file__).resolve().parents[1] / "configs/avqi_route_c_tau_history_capacity_contract_v1.json"


def exact_row(speaker: str, status: str = "ok") -> dict[str, str]:
    return {"speaker_id": speaker, "scoring_status": status, **{key: "1.0" for key in COMPONENTS}}


def test_unicode_normalization_does_not_merge_different_collections() -> None:
    assert canonical_speaker("A\u0308HH36") == "TAU:ÄHH36"
    assert canonical_speaker(" TAU:ÄHH36 ") == "TAU:ÄHH36"
    assert canonical_speaker("PD08") != canonical_speaker("PD_8")


@pytest.mark.parametrize("value", ["SVD:1301", "", "../FD01", "p123", "FD01_cs"])
def test_invalid_or_foreign_speaker_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        canonical_speaker(value)


def test_failed_exact_attempt_still_consumes_speaker() -> None:
    row = exact_row("SD13", "error")
    row.update({key: "" for key in COMPONENTS})
    result = exact_rows([row])
    assert result["TAU:SD13"]["status_counts"] == {"error": 1}
    assert result["TAU:SD13"]["successful_six_component_rows"] == 0


@pytest.mark.parametrize("mutation", ["missing_component", "unattempted_status"])
def test_incomplete_exact_schema_cannot_prove_opening(mutation: str) -> None:
    row = exact_row("SD13")
    if mutation == "missing_component":
        del row["hnr"]
    else:
        row["scoring_status"] = "pending"
    with pytest.raises(ValueError):
        exact_rows([row])


def evidence_fixture() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "expanded_labels": exact_rows([exact_row("FD01")]),
        "external_exact": exact_rows([exact_row("SD20")]),
        "sd13_exact": exact_rows([exact_row("SD13")]),
        "tau_paired_scores": exact_rows([exact_row(s) for s in ("FD01", "SD20", "SD13")]),
    }


def test_saturation_proof_rejects_partial_ledger_missing_sd13() -> None:
    evidence = evidence_fixture()
    speakers = set(evidence["tau_paired_scores"])
    proof = prove_current_universe_opened(speakers, evidence)
    assert proof["TAU:SD13"] == ["sd13_exact", "tau_paired_scores"]
    evidence["sd13_exact"] = {}
    with pytest.raises(ValueError, match="independent"):
        prove_current_universe_opened(speakers, evidence)


def test_nonfinite_record_cannot_prove_successful_six_component_coverage() -> None:
    evidence = evidence_fixture()
    row = exact_row("SD13")
    row["cpps"] = "nan"
    evidence["tau_paired_scores"].update(exact_rows([row]))
    with pytest.raises(ValueError, match="paired-score"):
        prove_current_universe_opened(set(evidence["tau_paired_scores"]), evidence)


def test_capacity_excludes_opened_and_unknown_sex_without_guessing() -> None:
    rows = [
        {"canonical_speaker_id": f"TAU:FD{i:02d}", "label": "patient", "sex": sex, "source_metadata_eligible": sex != "unknown"}
        for i, sex in enumerate(["female"] * 4 + ["male"] * 4 + ["unknown"], start=1)
    ]
    result = capacity(rows, set())
    assert result["six_gradient_capacity_pass"] is True
    assert result["joint_capacity_pass"] is False
    assert result["disjoint_gradient_and_joint_capacity_pass"] is False
    result = capacity(rows, {row["canonical_speaker_id"] for row in rows})
    assert result["remaining_unopened_speakers"] == []
    assert result["six_gradient_capacity_pass"] is False


def test_coupled_capacity_does_not_reuse_gradient_speakers_in_joint_panel() -> None:
    rows = []
    for label in ("patient", "healthy"):
        for sex in ("female", "male"):
            for _ in range(4):
                rows.append({"canonical_speaker_id": f"TAU:FD{len(rows) + 1:02d}", "label": label, "sex": sex, "source_metadata_eligible": True})
    result = capacity(rows, set())
    assert result["six_gradient_capacity_pass"] is True
    assert result["joint_capacity_pass"] is True
    assert result["disjoint_gradient_and_joint_capacity_pass"] is False


def source_fixture(tmp_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    root = tmp_path / "Elina"
    speaker_root = root / "FD01"
    speaker_root.mkdir(parents=True)
    row = {"speaker_id": "FD01", "source": "Elina", "label": "patient", "sex": "female"}
    inventory = []
    for view, task, seconds in (("cs", "reading", 3), ("sv", "sustained_vowel", 1)):
        path = speaker_root / f"FD01_{view}.wav"
        sf.write(path, np.zeros(16000 * seconds, dtype=np.float32), 16000, subtype="PCM_16")
        row[f"{view}_audio_path"] = str(path)
        inventory.append({"dataset": "TAU", "source": "Elina", "speaker_id": "FD01", "task": task,
                          "audio_path": str(path), "label": "patient", "sex": "female", "channels": "1",
                          "sample_rate": "16000", "duration_sec": str(seconds)})
    return [row], inventory, {"Elina": str(root)}


def test_same_speaker_cs_sv_source_headers_and_hashes_verified(tmp_path: Path) -> None:
    manifest, inventory, roots = source_fixture(tmp_path)
    result = audit_sources(manifest, inventory, roots)
    assert result[0]["same_speaker_cs_sv_verified"] is True
    assert result[0]["source_metadata_eligible"] is True
    assert len(result[0]["sources"]["cs"]["sha256"]) == 64
    assert "clean_avqi" not in result[0]


@pytest.mark.parametrize("mutation", ["extra_speaker", "wrong_view", "outside_root", "metadata_label", "duration"])
def test_source_integrity_mismatches_fail_closed(tmp_path: Path, mutation: str) -> None:
    manifest, inventory, roots = source_fixture(tmp_path)
    if mutation == "extra_speaker":
        (Path(roots["Elina"]) / "FD02").mkdir()
    elif mutation == "wrong_view":
        manifest[0]["sv_audio_path"] = manifest[0]["cs_audio_path"]
    elif mutation == "outside_root":
        manifest[0]["sv_audio_path"] = str(tmp_path / "FD01_sv.wav")
    elif mutation == "metadata_label":
        inventory[0]["label"] = "healthy"
    else:
        inventory[0]["duration_sec"] = "9"
    with pytest.raises(ValueError):
        audit_sources(manifest, inventory, roots)


def test_history_mentions_do_not_become_false_exact_openings(tmp_path: Path) -> None:
    write_json(tmp_path / "panel.json", {"speaker_id": "FD02", "exact_scores_opened": False})
    rows = [exact_row("FD01"), {**exact_row("FD03"), "dataset": "foreign"}]
    with (tmp_path / "exact.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *rows[0]])
        writer.writeheader()
        writer.writerow({"dataset": "TAU", **rows[0]})
        writer.writerow(rows[1])
    inventory, evidence = scan_history([tmp_path])
    assert len(inventory) == 2
    assert set(evidence) == {"TAU:FD01"}
    assert evidence["TAU:FD01"][0]["row_number"] == 2


def test_history_scan_excludes_current_run(tmp_path: Path) -> None:
    own = tmp_path / "current_run"
    own.mkdir()
    write_json(own / "report.json", {"speaker_id": "FD02"})
    write_json(tmp_path / "old.json", {"speaker_id": "FD01"})
    inventory, _ = scan_history([tmp_path], own)
    assert [Path(row["path"]).name for row in inventory] == ["old.json"]


def test_pinned_contract_preserves_tau_only_and_zero_optimizer_boundary() -> None:
    contract = json.loads(CONTRACT.read_text())
    validate_contract(contract)
    assert contract["boundaries"] == BOUNDARIES
    assert contract["source_policy"]["dataset"] == "TAU"


@pytest.mark.parametrize("mutation", ["optimizer", "svd", "quota", "relative_path"])
def test_contract_rejects_unauthorized_changes(mutation: str) -> None:
    contract = copy.deepcopy(json.loads(CONTRACT.read_text()))
    if mutation == "optimizer":
        contract["boundaries"]["generator_optimizer_steps"] = 1
    elif mutation == "svd":
        contract["source_policy"]["dataset"] = "SVD"
    elif mutation == "quota":
        contract["capacity_gates"]["six_gradient_distinct_speaker_quotas"]["patient/male"] = 0
    else:
        contract["inputs"]["tau_manifest"]["path"] = "manifest.csv"
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_input_drift_and_output_overwrite_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    write_json(path, {"decision": "historical"})
    binding = {"evidence": {"path": str(path), "sha256": sha256_file(path)}}
    assert verify_bindings(binding)["evidence"] == path
    with pytest.raises(FileExistsError):
        write_json(path, {"decision": "replacement"})
    path.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        verify_bindings(binding)
