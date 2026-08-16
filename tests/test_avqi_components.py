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
    DifferentiableAVQIComponentEstimator,
    FrequencyAwareSharedComponentHead,
    FrequencyAwareWaveformComponentPredictor,
    PhaseAwareCompactTFGridWaveformComponentPredictor,
    PhaseAwareFrequencyAwareWaveformComponentPredictor,
    PretrainedFullTFGridWaveformComponentPredictor,
    PraatDifferentiableAVQIComponentEstimator,
    SharedComponentHead,
    WaveformComponentPredictor,
    avqi_v0301,
    denormalize_components,
    freeze_module,
    freeze_module_for_input_gradient,
    phase_aware_spectral_features,
    pool_frequency_aware_shared_feature_map,
    standardized_component_loss,
)
from scripts.avqi_vctk_selection import (
    CONDITIONS,
    select_exact_complete_external_rows,
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


def test_compact_tfgrid_shared_head_supports_frozen_input_gradient() -> None:
    feature_map = torch.randn(1, 3, 24, 33, requires_grad=True)
    head = CompactTFGridSharedComponentHead(feature_channels=3)
    freeze_module_for_input_gradient(head)
    prediction = head(feature_map)
    prediction.square().mean().backward()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert float(feature_map.grad.norm()) > 0.0
    assert all(parameter.grad is None for parameter in head.parameters())


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


def test_phase_aware_features_ignore_constant_phase_offset() -> None:
    magnitude = torch.rand(2, 17, 13)
    phase = torch.randn(2, 17, 13)
    baseline = phase_aware_spectral_features(
        magnitude,
        phase,
        time_dim=-1,
        magnitude_compression=0.3,
    )
    shifted = phase_aware_spectral_features(
        magnitude,
        phase + 1.25,
        time_dim=-1,
        magnitude_compression=0.3,
    )
    assert baseline.shape == (2, 3, 17, 13)
    assert torch.isfinite(baseline).all()
    assert torch.allclose(baseline, shifted, atol=1e-6, rtol=1e-6)


def test_phase_aware_predictors_accept_cached_features() -> None:
    waveform = torch.randn(1, 4096)
    for predictor in (
        PhaseAwareFrequencyAwareWaveformComponentPredictor(),
        PhaseAwareCompactTFGridWaveformComponentPredictor(),
    ):
        predictor.eval()
        features = predictor.cache_features(waveform)
        direct = predictor(waveform)
        cached = predictor.forward_spectrogram(features)
        assert features.shape[1] == 3
        assert torch.allclose(direct, cached)


def test_phase_aware_predictors_support_frozen_input_gradient() -> None:
    for predictor in (
        PhaseAwareFrequencyAwareWaveformComponentPredictor(),
        PhaseAwareCompactTFGridWaveformComponentPredictor(),
    ):
        waveform = torch.randn(1, 4096, requires_grad=True)
        freeze_module_for_input_gradient(predictor)
        prediction = predictor(waveform)
        assert prediction.shape == (1, 6)
        prediction.square().mean().backward()
        assert waveform.grad is not None
        assert torch.isfinite(waveform.grad).all()
        assert float(waveform.grad.norm()) > 0.0
        assert all(parameter.grad is None for parameter in predictor.parameters())


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


def test_direct_exact_inspired_components_are_invariant_and_sensitive() -> None:
    sample_rate = 16_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    carrier = torch.sin(2.0 * math.pi * 170.0 * time)
    waveform = carrier * (
        1.0 + 0.1 * torch.sin(2.0 * math.pi * 3.0 * time)
    )
    estimator = DifferentiableAVQIComponentEstimator(max_frames=64)
    clean = estimator.raw_components(waveform)[0]
    gained = estimator.raw_components(waveform * 0.25)[0]
    modulated = estimator.raw_components(
        carrier * (1.0 + 0.5 * torch.sin(2.0 * math.pi * 5.0 * time))
    )[0]
    noisy = estimator.raw_components(
        waveform + 0.35 * torch.randn_like(waveform)
    )[0]
    spectrum = torch.fft.rfft(waveform)
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / sample_rate)
    lowpassed = torch.fft.irfft(
        spectrum * (frequencies <= 1_000.0),
        n=waveform.numel(),
    )
    lowpass_components = estimator.raw_components(lowpassed)[0]

    assert torch.allclose(clean, gained, atol=2e-4, rtol=2e-4)
    assert abs(float(noisy[1] - clean[1])) > 0.1
    assert abs(float(modulated[2] - clean[2])) > 0.1
    assert abs(float(lowpass_components[4] - clean[4])) > 0.1


def test_direct_exact_inspired_alignment_and_input_gradient() -> None:
    estimator = DifferentiableAVQIComponentEstimator(max_frames=32)
    raw = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    target_mean = torch.arange(6, dtype=torch.float32)
    target_scale = torch.ones(6)
    targets = raw * 2.0 + target_mean
    receipt = estimator.fit_alignment(
        raw,
        targets,
        target_mean,
        target_scale,
    )
    assert all(value > 0.0 for value in receipt["scale"])
    assert torch.allclose(
        estimator.forward_proxy_features(raw),
        (targets - target_mean) / target_scale,
        atol=1e-6,
    )

    waveform = torch.randn(1, 8_000, requires_grad=True)
    prediction = estimator(waveform)
    assert prediction.shape == (1, 6)
    prediction.square().mean().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert float(waveform.grad.norm()) > 0.0
    assert not list(estimator.parameters())


def test_praat_differentiable_v2_rejects_unknown_peak_mode() -> None:
    try:
        PraatDifferentiableAVQIComponentEstimator(peak_mode="unknown")
    except ValueError:
        return
    raise AssertionError("unknown differentiable AVQI peak mode was accepted")


def test_praat_differentiable_v2_is_invariant_and_family_sensitive() -> None:
    sample_rate = 16_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    carrier = torch.sin(2.0 * math.pi * 170.0 * time)
    waveform = carrier * (
        1.0 + 0.15 * torch.sin(2.0 * math.pi * 4.0 * time)
    ) + 0.15 * torch.sin(2.0 * math.pi * 2_500.0 * time)
    noisy = waveform + 0.3 * torch.randn_like(waveform)
    spectrum = torch.fft.rfft(waveform)
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / sample_rate)
    lowpassed = torch.fft.irfft(
        spectrum * (frequencies <= 1_000.0),
        n=waveform.numel(),
    )

    for peak_mode in ("soft", "hard"):
        estimator = PraatDifferentiableAVQIComponentEstimator(
            peak_mode=peak_mode,
            max_frames=128,
            cpps_max_frames=256,
        )
        clean = estimator.raw_components(waveform)[0]
        gained = estimator.raw_components(waveform * 0.25)[0]
        shifted = estimator.raw_components(torch.roll(waveform, 1_600))[0]
        noise_components = estimator.raw_components(noisy)[0]
        lowpass_components = estimator.raw_components(lowpassed)[0]

        assert torch.allclose(clean, gained, atol=2e-3, rtol=2e-3)
        assert abs(float(noise_components[1] - clean[1])) > 0.1
        assert abs(float(lowpass_components[4] - clean[4])) > 0.1
        assert abs(float(shifted[4] - clean[4])) < 0.1


def test_praat_differentiable_v2_has_finite_input_gradient() -> None:
    sample_rate = 16_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    base = torch.sin(2.0 * math.pi * 180.0 * time)
    for peak_mode in ("soft", "hard"):
        waveform = (
            base * (1.0 + 0.2 * torch.sin(2.0 * math.pi * 5.0 * time))
        ).unsqueeze(0).requires_grad_()
        estimator = PraatDifferentiableAVQIComponentEstimator(
            peak_mode=peak_mode,
            max_frames=128,
            cpps_max_frames=256,
        )
        prediction = estimator(waveform)
        assert prediction.shape == (1, 6)
        prediction.square().mean().backward()
        assert waveform.grad is not None
        assert torch.isfinite(waveform.grad).all()
        assert float(waveform.grad.norm()) > 0.0
        assert not list(estimator.parameters())


def test_praat_differentiable_v2_component_gradients_survive_zero_energy_regions() -> None:
    sample_rate = 16_000
    time = torch.arange(sample_rate // 2, dtype=torch.float32) / sample_rate
    voiced = torch.sin(2.0 * math.pi * 180.0 * time)
    waveforms = (
        torch.zeros(sample_rate),
        torch.cat((voiced, torch.zeros_like(voiced))),
    )
    for peak_mode in ("soft", "hard"):
        estimator = PraatDifferentiableAVQIComponentEstimator(
            peak_mode=peak_mode,
            max_frames=128,
            cpps_max_frames=256,
        )
        for source in waveforms:
            waveform = source.unsqueeze(0).requires_grad_()
            prediction = estimator(waveform)
            for component_index in range(prediction.shape[-1]):
                gradient = torch.autograd.grad(
                    prediction[:, component_index].sum(),
                    waveform,
                    retain_graph=component_index + 1 < prediction.shape[-1],
                )[0]
                assert torch.isfinite(gradient).all()


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


def _external_candidate_rows(
    sample_id: str,
    *,
    failed_condition: str | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "split": "vctk_external",
            "speaker_id": "p001",
            "sample_id": sample_id,
            "condition": condition,
            "scoring_status": (
                "error" if condition == failed_condition else "ok"
            ),
            "error_type": (
                "PraatError" if condition == failed_condition else ""
            ),
        }
        for condition in CONDITIONS
    ]


def test_external_exact_selection_uses_reserve_without_metric_values() -> None:
    internal = {
        "split": "surrogate_train",
        "speaker_id": "p900",
        "sample_id": "internal",
        "condition": "clean",
        "scoring_status": "error",
    }
    rows = [
        internal,
        *_external_candidate_rows("primary_bad", failed_condition="clean"),
        *_external_candidate_rows("primary_good"),
        *_external_candidate_rows("reserve_good"),
    ]
    selected, receipt = select_exact_complete_external_rows(
        rows,
        required_utterances_per_speaker=2,
    )
    selected_samples = {
        row["sample_id"]
        for row in selected
        if row["split"] == "vctk_external"
    }
    assert internal in selected
    assert selected_samples == {"primary_good", "reserve_good"}
    assert receipt["metric_values_used_for_selection"] is False
    assert receipt["replacement_count"] == 1
    assert receipt["selected_external_rows"] == 8
    assert receipt["speakers"]["p001"]["replaced_sample_ids"] == [
        "primary_bad"
    ]
    assert receipt["speakers"]["p001"]["replacement_sample_ids"] == [
        "reserve_good"
    ]


def test_external_exact_selection_fails_without_enough_valid_reserves() -> None:
    rows = [
        *_external_candidate_rows("bad", failed_condition="clean"),
        *_external_candidate_rows("only_good"),
    ]
    try:
        select_exact_complete_external_rows(
            rows,
            required_utterances_per_speaker=2,
        )
    except ValueError as error:
        assert "only 1 exact-complete external utterances" in str(error)
        return
    raise AssertionError("insufficient exact-valid reserve was accepted")
