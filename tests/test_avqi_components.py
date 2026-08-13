from __future__ import annotations

import math

import torch

from model.avqi_components import (
    AVQI_COMPONENT_LOSS_WEIGHTS,
    AVQI_COMPONENT_NAMES,
    CompactTFGridComponentEncoder,
    CompactTFGridSharedComponentHead,
    CompactTFGridWaveformComponentPredictor,
    ComponentAffineCalibrator,
    FrequencyAwareSharedComponentHead,
    FrequencyAwareWaveformComponentPredictor,
    PretrainedFullTFGridWaveformComponentPredictor,
    SharedComponentHead,
    WaveformComponentPredictor,
    avqi_v0301,
    denormalize_components,
    freeze_module,
    freeze_module_for_input_gradient,
    pool_frequency_aware_shared_feature_map,
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


def test_frequency_aware_shared_head_retains_frequency_profile() -> None:
    feature_map = torch.randn(2, 48, 12, 25, requires_grad=True)
    pooled = pool_frequency_aware_shared_feature_map(feature_map)
    head = FrequencyAwareSharedComponentHead()
    prediction = head(feature_map)
    assert pooled.shape == (2, 48 * 8 * 2)
    assert prediction.shape == (2, 6)
    prediction.square().mean().backward()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert float(feature_map.grad.norm()) > 0.0


def test_compact_tfgrid_shared_head_has_feature_gradient() -> None:
    feature_map = torch.randn(2, 48, 24, 33, requires_grad=True)
    head = CompactTFGridSharedComponentHead()
    prediction = head(feature_map)
    assert prediction.shape == (2, 6)
    prediction.square().mean().backward()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert float(feature_map.grad.norm()) > 0.0


def test_compact_tfgrid_encoder_rejects_invalid_shape_config() -> None:
    for kwargs in (
        {"input_channels": 1, "embedding": 22},
        {"input_channels": 1, "num_blocks": 0},
        {"input_channels": 1, "frequency_bins": 0},
    ):
        try:
            CompactTFGridComponentEncoder(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid TF-GridNet config was accepted: {kwargs}")


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


def test_frequency_aware_waveform_predictor_supports_input_gradient() -> None:
    waveform = torch.randn(1, 4096, requires_grad=True)
    predictor = FrequencyAwareWaveformComponentPredictor()
    freeze_module(predictor)
    prediction = predictor(waveform)
    assert prediction.shape == (1, 6)
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_waveform_predictors_accept_cached_spectrograms() -> None:
    waveform = torch.randn(1, 4096)
    for predictor in (
        WaveformComponentPredictor(),
        FrequencyAwareWaveformComponentPredictor(),
        CompactTFGridWaveformComponentPredictor(),
    ):
        predictor.eval()
        spectrogram = predictor.log_spectrogram(waveform)
        direct = predictor(waveform)
        cached = predictor.forward_spectrogram(spectrogram)
        assert torch.allclose(direct, cached)


def test_waveform_frontend_uses_fixed_time_grid_for_variable_lengths() -> None:
    predictor = WaveformComponentPredictor()
    short = predictor.log_spectrogram(torch.randn(1, 4096))
    long = predictor.log_spectrogram(torch.randn(1, 8192))
    assert short.shape == long.shape
    assert short.shape[-1] == predictor.time_bins


def test_compact_tfgrid_waveform_predictor_supports_frozen_input_gradient() -> None:
    waveform = torch.randn(1, 4096, requires_grad=True)
    predictor = CompactTFGridWaveformComponentPredictor()
    predictor.eval()
    with torch.no_grad():
        eval_prediction = predictor(waveform.detach())
    freeze_module_for_input_gradient(predictor)
    prediction = predictor(waveform)
    assert prediction.shape == (1, 6)
    assert torch.allclose(
        prediction.detach(),
        eval_prediction,
        atol=1e-6,
        rtol=1e-5,
    )
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in predictor.parameters())
    assert all(
        child.training
        for child in predictor.modules()
        if isinstance(child, torch.nn.LSTM)
    )


def test_compact_tfgrid_frozen_cuda_input_gradient() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    waveform = torch.randn(1, 48_000, device=device, requires_grad=True)
    predictor = CompactTFGridWaveformComponentPredictor().to(device)
    predictor.eval()
    with torch.no_grad():
        eval_prediction = predictor(waveform.detach())
    freeze_module_for_input_gradient(predictor)
    prediction = predictor(waveform)
    assert torch.allclose(
        prediction.detach(),
        eval_prediction,
        atol=1e-6,
        rtol=1e-5,
    )
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_pretrained_full_tfgrid_checkpoint_mapping_and_freeze(tmp_path) -> None:
    config = {
        "n_fft": 32,
        "hop_length": 16,
        "time_bins": 8,
        "embedding": 16,
        "lstm_hidden": 16,
        "num_blocks": 2,
        "adaptation_blocks": 1,
        "attention_heads": 4,
    }
    source = PretrainedFullTFGridWaveformComponentPredictor(**config)
    hybrid_state = {}
    replacements = {
        ".attention_norm.": ".attn_norm.",
        ".frame_attention.": ".frame_attn.",
        ".attention_ffn.": ".attn_ffn.",
    }
    for key, value in source.state_dict().items():
        if not (key.startswith("encoder.") or key.startswith("blocks.")):
            continue
        hybrid_key = key
        for current, original in replacements.items():
            hybrid_key = hybrid_key.replace(current, original)
        hybrid_state[f"discriminative.{hybrid_key}"] = value.clone()
    hybrid_state["discriminative.mask_head.weight"] = torch.randn(2, 16, 1, 1)
    hybrid_state["discriminative.mask_head.bias"] = torch.randn(2)
    checkpoint_path = tmp_path / "hybrid_disc.ckpt"
    torch.save(
        {
            "state_dict": hybrid_state,
            "hybrid_stage": "disc",
            "hybrid_architecture_config": {
                "discriminative": {
                    "num_blocks": 2,
                    "embedding": 16,
                    "lstm_hidden": 16,
                    "attention_heads": 4,
                    "dropout": 0.0,
                }
            },
        },
        checkpoint_path,
    )

    restored = PretrainedFullTFGridWaveformComponentPredictor(**config)
    receipt = restored.load_hybrid_discriminative_checkpoint(checkpoint_path)
    assert receipt["checkpoint_stage"] == "disc"
    assert receipt["adaptation_blocks"] == 1
    assert receipt["time_pool_position"] == "after_frozen_prefix"
    assert receipt["loaded_tensor_count"] == len(hybrid_state) - 2
    assert torch.equal(restored.encoder[0].weight, source.encoder[0].weight)
    assert all(
        not parameter.requires_grad
        for parameter in restored.encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in restored.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in restored.blocks[1].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in restored.regressor.parameters()
    )


def test_pretrained_full_tfgrid_supports_cached_and_input_gradient() -> None:
    predictor = PretrainedFullTFGridWaveformComponentPredictor(
        n_fft=32,
        hop_length=16,
        time_bins=8,
        embedding=16,
        lstm_hidden=16,
        num_blocks=2,
        adaptation_blocks=1,
        attention_heads=4,
    )
    predictor.eval()
    waveform = torch.randn(1, 512, requires_grad=True)
    spectrogram = predictor.spectrogram_features(waveform)
    prefix = predictor.encode_frozen_prefix(spectrogram)
    direct = predictor.forward_spectrogram(spectrogram)
    cached = predictor.forward_cached_prefix(prefix)
    assert direct.shape == (1, 6)
    assert torch.allclose(direct, cached)

    freeze_module_for_input_gradient(predictor)
    prediction = predictor(waveform)
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_component_affine_calibrator_is_fixed_and_differentiable() -> None:
    prediction = torch.randn(2, 6, requires_grad=True)
    scale = torch.linspace(0.8, 1.3, 6)
    bias = torch.linspace(-0.2, 0.3, 6)
    calibrator = ComponentAffineCalibrator(scale, bias)
    calibrated = calibrator(prediction)
    assert torch.allclose(calibrated, prediction * scale + bias)
    calibrated.square().mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert not list(calibrator.parameters())


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


def test_component_loss_balances_correlated_pairs() -> None:
    assert AVQI_COMPONENT_LOSS_WEIGHTS == (1.0, 1.0, 0.5, 0.5, 0.5, 0.5)
    target = torch.zeros(1, 6)
    target_mean = torch.zeros(6)
    target_scale = torch.ones(6)
    hnr_error = torch.zeros(1, 6)
    hnr_error[0, 1] = 1.0
    shimmer_pair_error = torch.zeros(1, 6)
    shimmer_pair_error[0, 2:4] = 1.0

    hnr_loss = standardized_component_loss(
        hnr_error, target, target_mean, target_scale
    )
    shimmer_pair_loss = standardized_component_loss(
        shimmer_pair_error, target, target_mean, target_scale
    )

    assert torch.isclose(hnr_loss, shimmer_pair_loss)
