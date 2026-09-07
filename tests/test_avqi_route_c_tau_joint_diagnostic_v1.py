"""Contract and leakage regressions for the explicitly diagnostic TAU path."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.avqi_route_c_tau_joint_diagnostic_v1 import (
    AVQI_COMPONENT_NAMES,
    BOUNDARIES,
    GRADIENT_SPLITS,
    GRADIENT_STRATA,
    HISTORICAL_FRESH_NO_GO,
    JOINT_ALLOCATION,
    SOURCE_DECISION,
    aggregate_measurements,
    binding,
    calibration_inverse_gradient_weights,
    evaluate_fusion,
    finalize_case_measurement,
    load_stage,
    receipt,
    select_sources,
    target_component_rows,
    validate_contract,
    validate_source_record,
    write_audio,
    write_json,
)


@pytest.fixture
def contract():
    path = Path(__file__).resolve().parents[1] / "configs/avqi_route_c_tau_joint_diagnostic_contract_v1.json"
    return json.loads(path.read_text())


def metadata_rows():
    rows = []
    for label in ("patient", "healthy"):
        for sex in ("female", "male"):
            for _ in range(12):
                speaker = f"FD{len(rows) + 1:02d}"
                rows.append({"speaker_id": speaker, "canonical_speaker_id": "TAU:" + speaker,
                             "dataset": "TAU", "label": label, "sex": sex,
                             "source_metadata_eligible": True})
    return rows


def test_contract_keeps_fresh_failure_and_numeric_gates(contract):
    validate_contract(contract)
    assert contract["boundaries"]["independent_fresh_validation"] is False
    assert contract["boundaries"]["historical_fresh_no_go_preserved"] is True
    assert HISTORICAL_FRESH_NO_GO.startswith("NO_GO_")


@pytest.mark.parametrize("group,key,value", [
    ("boundaries", "independent_fresh_validation", True),
    ("boundaries", "generator_optimizer_steps", 1),
    ("boundaries", "scientific_promotion_granted", True),
    ("boundaries", "svd_used_for_new_testing", True),
    ("selection", "failed_speaker_replacement_allowed", True),
    ("selection", "exact_values_used", True),
    ("selection", "target_scorability_used_for_selection", True),
    ("selection", "exclude_every_historically_opened_speaker", True),
    ("selection", "speaker_disjoint_across_all_current_roles", False),
    ("selection", "unknown_sex_ineligible", False),
    ("selection", "seed", 20260909),
    ("gradient_gates", "maximum_weighted_share", 0.81),
    ("gradient_gates", "minimum_component_to_joint_cosine", -0.01),
    ("materialization", "healthy_waveform_step_enabled", True),
    ("materialization", "final_waveform_highpass", True),
    ("materialization", "target_exact_sealed_before_any_candidate", False),
])
def test_contract_rejects_scope_or_threshold_drift(contract, group, key, value):
    contract[group][key] = value
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_contract_rejects_joint_threshold_drift(contract):
    contract["joint_gates"]["efficacy"]["exact_improvement_fraction_minimum"] = 0.7
    with pytest.raises(ValueError, match="joint or fusion"):
        validate_contract(contract)


def test_selection_is_metadata_only_and_invariant_to_input_order(contract):
    rows = metadata_rows()
    selected = select_sources(rows, contract["selection"]["salt"])
    altered = copy.deepcopy(rows[::-1])
    for i, row in enumerate(altered):
        row.update(exact_avqi=i, cpps=-i, diagnosis="ignored", severity="ignored")
    repeated = select_sources(altered, contract["selection"]["salt"])
    assert [row["canonical_speaker_id"] for row in selected] == [row["canonical_speaker_id"] for row in repeated]
    assert len({row["canonical_speaker_id"] for row in selected}) == 20
    for split in GRADIENT_SPLITS:
        strata = [row["sex"] + "/" + row["view"] for row in selected if row["role"] == "gradient" and row["split"] == split]
        assert strata == list(GRADIENT_STRATA)
    for split, label, sex, count in JOINT_ALLOCATION:
        assert sum(row["role"] == "joint_reserve" and row["split"] == split and row["label"] == label and row["sex"] == sex for row in selected) == count


def test_selection_rejects_duplicate_speaker(contract):
    rows = metadata_rows()
    with pytest.raises(ValueError, match="duplicate"):
        select_sources(rows + [rows[0]], contract["selection"]["salt"])


def test_selection_cannot_reuse_gradient_speakers_for_joint_capacity(contract):
    rows = metadata_rows()
    male_patients = [row for row in rows if row["label"] == "patient" and row["sex"] == "male"]
    reduced = [row for row in rows if row not in male_patients[4:]]
    with pytest.raises(ValueError, match="insufficient"):
        select_sources(reduced, contract["selection"]["salt"])


def test_unknown_sex_is_not_inferred(contract):
    rows = metadata_rows()
    for row in rows:
        if row["sex"] == "male":
            row["sex"] = "unknown"
            row["source_metadata_eligible"] = False
    with pytest.raises(ValueError, match="insufficient"):
        select_sources(rows, contract["selection"]["salt"])


def source_fixture(tmp_path):
    directory = tmp_path / "Elina" / "FD01"
    directory.mkdir(parents=True)
    sources = {}
    for view, seconds in (("cs", 3), ("sv", 1)):
        path = directory / f"FD01_{view}.wav"
        sf.write(path, np.linspace(-0.1, 0.1, 16000 * seconds), 16000, subtype="FLOAT")
        sources[view] = {**binding(path), "channels": 1, "frames": 16000 * seconds,
                         "sample_rate": 16000, "mono_duration_eligible": True}
    row = {"speaker_id": "FD01", "canonical_speaker_id": "TAU:FD01", "dataset": "TAU",
           "label": "patient", "sex": "male", "source": "Elina",
           "sources": sources, "same_speaker_cs_sv_verified": True, "source_metadata_eligible": True}
    return row, {"Elina": str(tmp_path / "Elina")}


def test_source_identity_pairing_and_hashes_are_verified(tmp_path):
    row, roots = source_fixture(tmp_path)
    validate_source_record(row, roots)
    row["sources"]["sv"] = copy.deepcopy(row["sources"]["cs"])
    with pytest.raises(ValueError, match="same-speaker"):
        validate_source_record(row, roots)


def test_source_rejects_svd_or_pathlike_identity(tmp_path):
    row, roots = source_fixture(tmp_path)
    row["dataset"] = "SVD"
    with pytest.raises(ValueError, match="identity"):
        validate_source_record(row, roots)
    row["dataset"] = "TAU"
    row["speaker_id"] = "../FD01"
    with pytest.raises(ValueError):
        validate_source_record(row, roots)


def test_source_rejects_content_drift(tmp_path):
    row, roots = source_fixture(tmp_path)
    Path(row["sources"]["sv"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_source_record(row, roots)


def test_target_vectors_follow_component_order():
    values = np.arange(6, dtype=np.float64)
    rows = target_component_rows({"a": values}, ["a"])
    assert list(rows["a"]) == list(AVQI_COMPONENT_NAMES)
    assert rows["a"]["shimmer_db"] == 3


@pytest.mark.parametrize("values", [np.ones(5), np.ones((1, 6)), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_target_vector_is_rejected(values):
    with pytest.raises(ValueError, match="vector"):
        target_component_rows({"a": values}, ["a"])


def test_target_cannot_silently_drop_failed_speaker():
    with pytest.raises(ValueError, match="coverage"):
        target_component_rows({"a": np.ones(6)}, ["a", "b"])


def test_float_audio_roundtrip_and_no_overwrite(tmp_path):
    audio = np.sin(np.arange(16000, dtype=np.float32) * 0.1) * 0.1
    bound = write_audio(tmp_path / "a.wav", audio)
    assert bound["samples"] == len(audio)
    assert bound["subtype"] == "FLOAT"
    with pytest.raises(FileExistsError):
        write_audio(tmp_path / "a.wav", audio)


def test_receipt_rejects_failed_scope_and_artifact_drift(tmp_path):
    write_json(tmp_path / "source.json", {"scope": "diagnostic"})
    receipt(tmp_path, "seal", SOURCE_DECISION, {"path": "/unused", "sha256": "a" * 64},
            {"commit": "b" * 40}, dependencies=[])
    receipt_binding = binding(tmp_path / "completion_receipt.json")
    _, parsed = load_stage(receipt_binding, "a" * 64, "seal", SOURCE_DECISION)
    assert all(parsed[key] == value for key, value in BOUNDARIES.items())
    with pytest.raises(ValueError, match="scope, decision"):
        load_stage(receipt_binding, "a" * 64, "seal", "PASS_FRESH")
    (tmp_path / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        load_stage(receipt_binding, "a" * 64, "seal", SOURCE_DECISION)


def synthetic_gradient_records(conflict=False):
    records = []
    for index in range(8):
        gradients = {name: torch.eye(6, dtype=torch.float64)[j].clone() for j, name in enumerate(AVQI_COMPONENT_NAMES)}
        if conflict and index == 7:
            gradients["shimmer_db"] = -2 * gradients["cpps"]
        records.append({
            "case_id": str(index), "split": GRADIENT_SPLITS[index // 4],
            "speaker_id": f"TAU:FD{index + 1:02d}", "sample_id": str(index),
            "sample_group": "patient", "view": "cs" if index % 2 == 0 else "sv",
            "condition": "rir_plus_noise", "source_audio_file_sha256": "a" * 64,
            "components": {name: {
                "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                "finite_observed": True, "strictly_positive_norm_observed": True,
                "scientific_gate_applied": False, "prediction": 1.0, "clean_pathological_target": 0.0,
                "normalized_signed_error": 1.0, "normalized_bidirectional_gap": 1.0, "smooth_l1_loss": 0.5,
            } for name, gradient in gradients.items()}, "_gradients": gradients,
        })
    return records


@pytest.mark.parametrize("conflict", [False, True])
def test_real_fusion_gate_preserves_direction_conflicts(conflict):
    records = synthetic_gradient_records(conflict)
    medians, weights = calibration_inverse_gradient_weights(records[:4])
    finalized = [finalize_case_measurement(row, weights) for row in records]
    raw = {"calibration": {**aggregate_measurements(finalized[:4]), "median_component_gradient_norms": medians,
                           "frozen_inverse_gradient_weights": weights,
                           "weighted_median_gradient_norms": {name: medians[name] * weights[name] for name in weights},
                           "weights_selected_on_holdout": False},
           "holdout": aggregate_measurements(finalized[4:])}
    gates, _, rows = evaluate_fusion(finalized, raw)
    assert gates["all_post_cap_component_to_joint_cosines_nonnegative"] is (not conflict)
    assert all(gates.values()) is (not conflict)
    if conflict:
        assert rows[-1]["fusion"]["fusion_authorized"] is False
