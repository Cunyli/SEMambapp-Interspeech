from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    AVQI_V0301_COEFFICIENTS,
    AVQI_V0301_EXPANDED_COEFFICIENTS,
    AVQI_V0301_SCALE,
    PraatDifferentiableAVQIComponentEstimator,
)
from model.avqi_route_c import (
    ROUTE_C_ACTIVE_COMPONENTS,
    ROUTE_C_COMPONENT_REGISTRY,
    ROUTE_C_SOURCE_ARCHITECTURES,
    ROUTE_C_SOURCE_COMPONENT_INDICES,
    active_bidirectional_gap_losses,
    build_route_c_four_active_estimator,
    load_route_c_four_active_scorer,
    sha256_file,
)
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    MAX_WEIGHTED_COMPONENT_NORM_SHARE,
    finalize_case,
    frozen_inverse_gradient_weights,
    load_label_bank,
)


def test_route_c_registry_freezes_six_slots_and_four_active_components() -> None:
    assert tuple(slot.name for slot in ROUTE_C_COMPONENT_REGISTRY) == (
        AVQI_COMPONENT_NAMES
    )
    assert tuple(
        slot.name
        for slot in ROUTE_C_COMPONENT_REGISTRY
        if slot.active_in_four_component_scorer
    ) == ROUTE_C_ACTIVE_COMPONENTS
    assert tuple(slot.avqi_coefficient for slot in ROUTE_C_COMPONENT_REGISTRY) == (
        AVQI_V0301_COEFFICIENTS
    )
    assert tuple(
        slot.expanded_avqi_coefficient for slot in ROUTE_C_COMPONENT_REGISTRY
    ) == AVQI_V0301_EXPANDED_COEFFICIENTS
    assert AVQI_V0301_EXPANDED_COEFFICIENTS == tuple(
        value * AVQI_V0301_SCALE for value in AVQI_V0301_COEFFICIENTS
    )
    status = {slot.name: slot.scientific_status for slot in ROUTE_C_COMPONENT_REGISTRY}
    assert status["shimmer_db"] == "unresolved"
    assert status["slope"] == "fresh_speaker_panel_pass"
    slope_slot = next(
        slot for slot in ROUTE_C_COMPONENT_REGISTRY if slot.name == "slope"
    )
    assert not slope_slot.active_in_four_component_scorer


def test_four_active_estimator_preserves_each_source_formula() -> None:
    torch.manual_seed(20260824)
    sample_rate = 16_000
    time = torch.arange(sample_rate // 2, dtype=torch.float32) / sample_rate
    waveform = (
        torch.sin(2.0 * math.pi * 178.0 * time)
        * (1.0 + 0.08 * torch.sin(2.0 * math.pi * 4.0 * time))
        + 0.08 * torch.sin(2.0 * math.pi * 2_300.0 * time)
        + 0.02 * torch.randn_like(time)
    )
    common = {"max_frames": 48, "cpps_max_frames": 96, "hnr_max_frames": 96}
    combined = build_route_c_four_active_estimator(**common)
    cpps = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        cpps_mode="praat_view_input_v12",
        cpps_power_floor=1e-6,
        **common,
    )
    hnr = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        hnr_mode="praat_pitch_path_v7",
        **common,
    )
    shimmer = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
        **common,
    )
    baseline = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        **common,
    )
    combined_value = combined.raw_components(waveform, speaking_type="sv")[0]
    cpps_value = cpps.raw_components(waveform, speaking_type="sv")[0]
    hnr_value = hnr.raw_components(waveform, speaking_type="sv")[0]
    shimmer_value = shimmer.raw_components(waveform, speaking_type="sv")[0]
    baseline_value = baseline.raw_components(waveform, speaking_type="sv")[0]

    assert torch.equal(combined_value[0], cpps_value[0])
    assert torch.equal(combined_value[1], hnr_value[1])
    assert torch.equal(combined_value[2:4], shimmer_value[2:4])
    assert torch.equal(combined_value[4:], baseline_value[4:])
    assert sum(parameter.numel() for parameter in combined.parameters()) == 0

    gradient_waveform = waveform.clone().requires_grad_()
    prediction = combined.raw_components(
        gradient_waveform,
        speaking_type="sv",
    )
    target = prediction.detach().clone()
    for offset, component in enumerate(ROUTE_C_ACTIVE_COMPONENTS):
        index = AVQI_COMPONENT_NAMES.index(component)
        target[0, index] += 1.0 if offset % 2 == 0 else -1.0
    losses = active_bidirectional_gap_losses(
        prediction,
        target,
        torch.zeros(6),
        torch.ones(6),
    )[0]
    gradients = []
    for offset in range(len(ROUTE_C_ACTIVE_COMPONENTS)):
        gradient = torch.autograd.grad(
            losses[offset],
            gradient_waveform,
            retain_graph=offset < len(ROUTE_C_ACTIVE_COMPONENTS) - 1,
        )[0]
        assert torch.isfinite(gradient).all()
        assert 0.0 < float(torch.linalg.vector_norm(gradient)) <= 1e4
        gradients.append(gradient)
    joint_gradient = sum(gradients) / len(gradients)
    assert torch.isfinite(joint_gradient).all()
    assert 0.0 < float(torch.linalg.vector_norm(joint_gradient)) <= 1e4


def test_four_active_bidirectional_losses_move_toward_target_without_avqi_signs() -> None:
    prediction = torch.tensor(
        [[2.0, -3.0, 4.0, 100.0, 200.0, -5.0]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    losses = active_bidirectional_gap_losses(
        prediction,
        target,
        torch.zeros(6),
        torch.ones(6),
    )
    losses.sum().backward()
    assert prediction.grad is not None
    active_indices = [
        AVQI_COMPONENT_NAMES.index(name) for name in ROUTE_C_ACTIVE_COMPONENTS
    ]
    assert torch.equal(
        prediction.grad[0, active_indices].sign(),
        prediction.detach()[0, active_indices].sign(),
    )
    assert torch.equal(
        prediction.grad[0, [3, 4]],
        torch.zeros(2),
    )
    assert AVQI_V0301_COEFFICIENTS[0] < 0.0
    assert prediction.grad[0, 0] > 0.0


def _checkpoint(path: Path, key: str, offset: float) -> tuple[Path, str]:
    alignment_scale = torch.arange(1.0, 7.0) + offset
    alignment_bias = torch.arange(11.0, 17.0) + offset
    calibration_scale = torch.arange(21.0, 27.0) + offset
    calibration_bias = torch.arange(31.0, 37.0) + offset
    torch.save(
        {
            "state_dict": {
                "alignment_scale": alignment_scale,
                "alignment_bias": alignment_bias,
            },
            "target_mean": torch.arange(6, dtype=torch.float32),
            "target_scale": torch.arange(1, 7, dtype=torch.float32),
            "calibration_scale": calibration_scale,
            "calibration_bias": calibration_bias,
            "components": AVQI_COMPONENT_NAMES,
            "architecture": ROUTE_C_SOURCE_ARCHITECTURES[key],
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "optimizer_steps": 0,
            "speaking_type_required": key == "cpps",
        },
        path,
    )
    return path, sha256_file(path)


def test_four_active_scorer_composes_only_authorized_checkpoint_slots(
    tmp_path: Path,
) -> None:
    offsets = {
        "cpps": 100.0,
        "hnr": 200.0,
        "shimmer_percent": 300.0,
        "tilt": 400.0,
    }
    created = {
        key: _checkpoint(tmp_path / f"{key}.pt", key, offset)
        for key, offset in offsets.items()
    }
    bundle = load_route_c_four_active_scorer(
        {key: value[0] for key, value in created.items()},
        {key: value[1] for key, value in created.items()},
        max_frames=32,
        cpps_max_frames=64,
        hnr_max_frames=64,
    )
    scorer = bundle.scorer
    for key, indices in ROUTE_C_SOURCE_COMPONENT_INDICES.items():
        for index in indices:
            assert scorer.estimator.alignment_scale[index] == index + 1 + offsets[key]
            assert scorer.estimator.alignment_bias[index] == index + 11 + offsets[key]
            assert scorer.calibrator.scale[index] == index + 21 + offsets[key]
            assert scorer.calibrator.bias[index] == index + 31 + offsets[key]
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 0


def test_inverse_gradient_weights_balance_calibration_and_report_conflict() -> None:
    records = []
    for multiplier in (1.0, 2.0):
        components = {
            name: {"gradient_norm": multiplier * (index + 1)}
            for index, name in enumerate(ROUTE_C_ACTIVE_COMPONENTS)
        }
        records.append({"components": components})
    medians, weights = frozen_inverse_gradient_weights(records)
    weighted = {
        component: medians[component] * weights[component]
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    assert max(weighted.values()) == min(weighted.values())

    gradients = {
        "cpps": torch.tensor([1.0, 0.0]),
        "hnr": torch.tensor([-1.0, 0.0]),
        "shimmer_percent": torch.tensor([0.0, 1.0]),
        "tilt": torch.tensor([0.0, 1.0]),
    }
    record = {
        "components": {
            name: {
                "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                "decision": "PASS",
            }
            for name, gradient in gradients.items()
        },
        "_gradients": gradients,
    }
    finalized = finalize_case(record, {name: 1.0 for name in gradients})
    assert finalized["joint"]["pairwise_component_cosines"]["cpps__hnr"] == {
        "cosine": -1.0,
        "direction_conflict": True,
    }
    assert finalized["joint"]["maximum_component_norm_share"] == 0.25
    assert MAX_WEIGHTED_COMPONENT_NORM_SHARE == 0.80


def _write_label_bank(path: Path, include_final: bool = False) -> str:
    fields = [
        "speaker_id",
        "sample_id",
        "split",
        "condition_id",
        "view",
        "scoring_status",
        "label",
        "sample_group",
        "cs_path",
        "cs_sha256",
        "sv_path",
        "sv_sha256",
        *AVQI_COMPONENT_NAMES,
    ]
    rows = []
    for split in ("surrogate_train", "surrogate_calibration", "surrogate_holdout"):
        strata = (
            ("pathological_mild", "cs"),
            ("pathological_mild", "sv"),
            ("pathological_severe", "cs"),
            ("pathological_severe", "sv"),
        )
        for index, (group, view) in enumerate(strata):
            speaker = f"{split}_{index}"
            common = {
                "speaker_id": speaker,
                "sample_id": "sample",
                "split": split,
                "view": view,
                "scoring_status": "ok",
                "label": "patient",
                "sample_group": group,
                "cs_path": "/tmp/cs.wav",
                "cs_sha256": "a" * 64,
                "sv_path": "/tmp/sv.wav",
                "sv_sha256": "b" * 64,
            }
            clean = {**common, "condition_id": "clean"}
            augmented = {**common, "condition_id": "aug16k_phone"}
            for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
                clean[component] = str(index + component_index)
                augmented[component] = str(index + component_index + 0.5)
            rows.extend((clean, augmented))
    if include_final:
        row = dict(rows[0])
        row["speaker_id"] = "forbidden"
        row["split"] = "sealed_final"
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def test_dev_selection_is_balanced_and_rejects_final_rows(tmp_path: Path) -> None:
    label_bank = tmp_path / "labels.csv"
    digest = _write_label_bank(label_bank)
    cases, _, _, selection = load_label_bank(label_bank, digest, "test-salt")
    assert len(cases) == 8
    assert selection["cases_by_split"] == {
        "surrogate_calibration": 4,
        "surrogate_holdout": 4,
    }
    assert selection["speaker_overlap"] == 0
    assert selection["final_panel_opened"] is False

    forbidden_bank = tmp_path / "labels_with_final.csv"
    forbidden_digest = _write_label_bank(forbidden_bank, include_final=True)
    with pytest.raises(ValueError, match="forbidden final splits"):
        load_label_bank(forbidden_bank, forbidden_digest, "test-salt")


def test_multicomponent_slurm_wrapper_preserves_training_boundary() -> None:
    source = Path(
        "scripts/run_avqi_route_c_multicomponent_gradient_audit.sh"
    ).read_text(encoding="utf-8")
    assert "NO_GO_AVQI_T2_TRAINING" not in source
    assert "generator" not in source.lower()
    assert "--device cuda" in source
    assert "tests/test_avqi_route_c_multicomponent.py" in source
    assert "tests/test_avqi_hnr_fresh_panel.py" in source
    assert "tests/test_avqi_shimmer_fresh_panel.py" in source
