from __future__ import annotations

import math

import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    SharedComponentHead,
    WaveformComponentPredictor,
    avqi_v0301,
    denormalize_components,
    freeze_module,
    standardized_component_loss,
)


def test_avqi_v0301_matches_verified_formula() -> None:
    values = torch.tensor([[10.0, 15.0, 5.0, 0.5, -20.0, -10.0]])
    expected = (
        4.152
        - 0.177 * 10.0
        - 0.006 * 15.0
        - 0.037 * 5.0
        + 0.941 * 0.5
        + 0.01 * -20.0
        + 0.093 * -10.0
    ) * 2.8902
    assert AVQI_COMPONENT_NAMES == (
        "cpps",
        "hnr",
        "shimmer_percent",
        "shimmer_db",
        "slope",
        "tilt",
    )
    assert math.isclose(float(avqi_v0301(values)), expected, rel_tol=1e-6)


def test_shared_head_has_finite_feature_gradient() -> None:
    feature_map = torch.randn(2, 48, 12, 25, requires_grad=True)
    head = SharedComponentHead()
    prediction = head(feature_map)
    assert prediction.shape == (2, 6)
    prediction.square().mean().backward()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert float(feature_map.grad.norm()) > 0.0


def test_waveform_predictor_supports_frozen_input_gradient() -> None:
    waveform = torch.randn(1, 4096, requires_grad=True)
    predictor = WaveformComponentPredictor()
    freeze_module(predictor)
    prediction = predictor(waveform)
    assert prediction.shape == (1, 6)
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_normalized_loss_and_inverse_transform() -> None:
    target_mean = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    target_scale = torch.tensor([2.0, 2.0, 1.0, 1.0, 4.0, 4.0])
    target = target_mean + target_scale
    normalized_prediction = torch.ones(1, 6)
    loss = standardized_component_loss(
        normalized_prediction,
        target.unsqueeze(0),
        target_mean,
        target_scale,
    )
    restored = denormalize_components(
        normalized_prediction,
        target_mean,
        target_scale,
    )
    assert float(loss) == 0.0
    assert torch.equal(restored, target.unsqueeze(0))
