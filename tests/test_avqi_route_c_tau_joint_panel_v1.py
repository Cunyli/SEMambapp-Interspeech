"""Leakage and real waveform regressions for the historical TAU joint adapter."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients
import scripts.avqi_route_c_tau_joint_panel_v1 as panel
from scripts.avqi_route_c_tau_joint_diagnostic_v1 import BOUNDARIES, SCOPE, write_json
from scripts.prepare_avqi_route_c_six_joint_waveforms import _write_pcm24


def reserved_rows():
    rows = []
    for index in range(12):
        speaker = f"TAU:test-{index}"
        for view in ("cs", "sv"):
            for condition in ("clean", "rir_only", "snr20", "snr10"):
                rows.append({
                    "case_id": f"joint-{index}-{view}-{condition}",
                    "canonical_speaker_id": speaker, "speaker_id": speaker.removeprefix("TAU:"),
                    "dataset": "TAU", "historically_exact_opened": True,
                    "split": "calibration" if index < 6 else "final",
                    "label": "patient" if index % 6 < 3 else "healthy",
                    "sex": "female" if index % 2 == 0 else "male",
                    "view": view, "condition": condition, "recipe_index": None, "recipe_uid": None,
                })
    return rows


def test_reserve_preserves_all_96_predeclared_rows():
    panel.validate_reserved_rows(reserved_rows(), [{"canonical_speaker_id": "TAU:gradient"}])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "overlap", "role", "dataset", "history"])
def test_reserve_rejects_role_leakage_and_sample_replacement(mutation):
    rows = reserved_rows()
    gradients = [{"canonical_speaker_id": "TAU:gradient"}]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "overlap":
        gradients[0]["canonical_speaker_id"] = rows[0]["canonical_speaker_id"]
    elif mutation == "role":
        rows[0]["split"] = "final"
    elif mutation == "dataset":
        rows[0]["dataset"] = "SVD"
    else:
        rows[0]["historically_exact_opened"] = False
    with pytest.raises(ValueError):
        panel.validate_reserved_rows(rows, gradients)


def test_healthy_controls_have_exact_pcm_identity_without_step(tmp_path, monkeypatch):
    base = np.linspace(-0.1, 0.1, 9876, dtype=np.float32)
    sealed = _write_pcm24(tmp_path / "base.wav", base)
    def forbidden(*args, **kwargs):
        raise AssertionError("healthy waveform step executed")
    monkeypatch.setattr(panel, "candidate_from_gradient", forbidden)
    candidates = panel.generate_candidates({"label": "healthy", "target": None}, base, None, sealed, (0.0, 1e-3), tmp_path)
    assert len(candidates) == 2
    assert all(c["sha256"] == sealed["sha256"] for c in candidates)
    assert all(c["samples"] == base.size for c in candidates)
    with pytest.raises(ValueError, match="healthy"):
        panel.generate_candidates({"label": "healthy", "target": None}, base, np.ones_like(base), sealed, (0.0,), tmp_path)


def test_patient_candidates_keep_full_length_and_frozen_zero(tmp_path):
    base = (0.05 * np.sin(np.arange(52037) / 13)).astype(np.float32)
    gradient = np.linspace(-1.0, 1.0, base.size)
    sealed = _write_pcm24(tmp_path / "base.wav", base)
    candidates = panel.generate_candidates({"label": "patient", "case_id": "full"}, base, gradient, sealed, (0.0, 1e-3), tmp_path)
    assert candidates[0]["sha256"] == sealed["sha256"]
    assert candidates[1]["samples"] == 52037
    assert candidates[1]["sha256"] != sealed["sha256"]
    with pytest.raises(ValueError, match="complete baseline"):
        panel.generate_candidates({"label": "patient", "case_id": "bad"}, base, gradient[:48000], sealed, (1e-3,), tmp_path)


def test_full_waveform_direction_conflict_abstains_without_relaxation():
    gradients = {name: torch.ones(53001, dtype=torch.float64) for name in AVQI_COMPONENT_NAMES}
    gradients["hnr"] = -gradients["hnr"]
    _, fusion = fuse_tensor_gradients(AVQI_COMPONENT_NAMES, gradients, {name: 1.0 for name in AVQI_COMPONENT_NAMES})
    record = {"components": {name: {"gradient_norm": float(torch.linalg.vector_norm(value))} for name, value in gradients.items()},
              "topology": {"highpass_pcm16_sha256": "a" * 64}}
    record["components"]["shimmer_db"]["candidate_e_projection"] = {
        "candidate_e_peak_handling_pass": True, "candidate_e_exact_highpass_pcm16_sha256": "a" * 64,
    }
    gates = panel.full_gradient_gates(record, fusion)
    assert not gates["all_component_to_joint_cosines_nonnegative"]
    assert gates["post_cap_share_le_0_80"]
    assert gates["candidate_e_peak_pcm16_bound"]


def test_final_judge_request_contains_only_one_frozen_alpha():
    rows = [{
        "split": "final", "label": "patient", "speaker_id": "test", "view": "cs", "case_id": "case",
        "target": {"path": "/target.wav"}, "base": {"path": "/base.wav"},
        "candidates": [{"alpha": value, "available": True, "path": f"/alpha-{index}.wav"} for index, value in enumerate((0.0, 1e-4, 1e-3))],
    }]
    items = panel.final_exact_items(rows, 1e-4, (0.0, 1e-4, 1e-3))
    assert {item["path"] for item in items} == {"/target.wav", "/base.wav", "/alpha-1.wav"}
    with pytest.raises(ValueError, match="nonzero"):
        panel.final_exact_items(rows, 0.0, (0.0, 1e-4))
    with pytest.raises(ValueError, match="frozen"):
        panel.final_exact_items(rows, 3e-4, (0.0, 1e-4))


def test_failed_preparation_cannot_reach_waveform_generation_or_exact_judge(tmp_path, monkeypatch):
    source = tmp_path / "source"
    gradient = tmp_path / "gradient"
    output = tmp_path / "outputs"
    for root in (source, gradient, output):
        root.mkdir()
    write_json(source / "tau_diagnostic_split_seal.json", {
        "joint_reserved_rows": reserved_rows(), "gradient_rows": [{"canonical_speaker_id": "TAU:gradient"}],
    })
    write_json(source / "tau_speaker_source_manifest.json", {})
    write_json(gradient / "six_gradient_fusion_report.json", {})
    monkeypatch.setattr(panel, "load_joint_inputs", lambda *args: (source, gradient))
    monkeypatch.setattr(panel, "materialize_joint", lambda *args: [])
    def measure(*args):
        for filename in ("candidate_e_joint_runtime_binding.json", "clean_target_label_bank.json",
                         "joint_gradient_manifest.json", "full_waveform_gradient_report.json"):
            write_json(output / filename, {"full_waveform_gradient_prerequisites_pass": False})
        return {"full_waveform_gradient_prerequisites_pass": False}
    monkeypatch.setattr(panel, "measure_full_gradients", measure)
    monkeypatch.setattr(panel, "validate_execution_package", lambda *args: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("failed full-gradient gate allowed candidate evaluation")
    monkeypatch.setattr(panel, "generate_candidates", forbidden)
    monkeypatch.setattr(panel, "run_exact", forbidden)
    decision = panel.prepare({}, {}, output, {"path": "/receipt", "sha256": "a" * 64},
                             {"path": "/contract", "sha256": "b" * 64}, torch.device("cpu"))
    assert decision == panel.PREPARATION_NO_GO
    assert not (output / "joint_waveform_seal.json").exists()
    package = panel.read_json(output / "joint_execution_package.json")
    assert package["execution_authorized"] is False
    assert package["scientific_promotion_granted"] is False
    assert package["generator_optimizer_steps"] == 0
    assert set(package["inputs"]) == set(panel.UNBOUND_JOINT_INPUTS)


@pytest.mark.parametrize("change", [{"scope": "fresh_TAU"}, {"generator_optimizer_steps": 1}, {"scientific_promotion_granted": True}])
def test_execution_package_rejects_scope_escalation(change):
    package = {"scope": SCOPE, "contract": {"sha256": "a" * 64}, **BOUNDARIES, **change}
    with pytest.raises(ValueError, match="scope"):
        panel.validate_execution_package(package, "a" * 64)
