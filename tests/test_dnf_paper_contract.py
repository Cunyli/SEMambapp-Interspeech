import math

import torch

from model.dnf_paper import (
    calibrate_half_noise_projection_eq13,
    dnf_clean_loss_eq15,
    dnf_noisy_loss_eq13,
    dnf_output_eq14,
    sdr_loss_eq5,
    si_sdr_loss,
)


DTYPE = torch.float64


def _orthogonal_batch() -> tuple[torch.Tensor, torch.Tensor]:
    speech = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    noise = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    return speech, noise


def test_eq5_matches_scale_dependent_sdr_definition() -> None:
    target = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    estimate = torch.tensor([[0.5, 0.0]], dtype=DTYPE)

    loss = sdr_loss_eq5(estimate, target)

    expected = torch.tensor([-10.0 * math.log10(4.0)], dtype=DTYPE)
    torch.testing.assert_close(loss, expected)


def test_eq13_calibrates_both_branches_to_half_n2_projection() -> None:
    speech, n2 = _orthogonal_batch()
    noisy_speech_estimate = speech + 0.2 * n2
    noise_estimate = -0.8 * n2
    noisy_speech_target = speech + n2

    speech_calibration = calibrate_half_noise_projection_eq13(noisy_speech_estimate, n2)
    noise_calibration = calibrate_half_noise_projection_eq13(noise_estimate, n2)
    loss = dnf_noisy_loss_eq13(
        noisy_speech_estimate,
        noise_estimate,
        noisy_speech_target,
        n2,
    )

    expected_half = torch.full((2,), 0.5, dtype=DTYPE)
    torch.testing.assert_close(speech_calibration.calibrated_noise_coefficient, expected_half)
    torch.testing.assert_close(noise_calibration.calibrated_noise_coefficient, expected_half)
    torch.testing.assert_close(speech_calibration.scale, torch.full((2,), 2.5, dtype=DTYPE))
    torch.testing.assert_close(noise_calibration.scale, torch.full((2,), -0.625, dtype=DTYPE))
    assert loss.total.shape == (2,)
    assert torch.isfinite(loss.total).all()


def test_eq13_invalid_dot_is_explicit_and_cannot_silently_average() -> None:
    speech, n2 = _orthogonal_batch()

    calibration = calibrate_half_noise_projection_eq13(
        speech,
        n2,
        eps=1e-8,
    )

    assert calibration.denominator_was_guarded.all()
    assert calibration.invalid_mask.all()
    assert not calibration.valid_mask.any()
    assert not calibration.faithful_mask.any()
    assert torch.isnan(calibration.scale).all()
    assert torch.isnan(calibration.scaled_estimate).all()


def test_eq13_optional_clamp_marks_valid_samples_nonfaithful() -> None:
    _, n2 = _orthogonal_batch()
    estimate = 1e-4 * n2

    calibration = calibrate_half_noise_projection_eq13(estimate, n2, scale_clamp=3.0)

    assert calibration.valid_mask.all()
    assert not calibration.invalid_mask.any()
    assert calibration.scale_was_clamped.all()
    assert not calibration.faithful_mask.any()
    torch.testing.assert_close(calibration.scale, torch.full((2,), 3.0, dtype=DTYPE))


def test_eq13_calibration_is_scale_and_sign_invariant() -> None:
    speech, n2 = _orthogonal_batch()
    base_estimate = speech + 0.25 * n2
    baseline = calibrate_half_noise_projection_eq13(base_estimate, n2)

    for scale in (0.1, 10.0, -2.0):
        result = calibrate_half_noise_projection_eq13(scale * base_estimate, n2)
        torch.testing.assert_close(result.scaled_estimate, baseline.scaled_estimate)
        torch.testing.assert_close(
            result.calibrated_noise_coefficient,
            torch.full((2,), 0.5, dtype=DTYPE),
        )
        assert result.valid_mask.all()
        assert result.faithful_mask.all()


def test_eq14_is_noise_scale_invariant_and_orthogonal() -> None:
    speech, noise = _orthogonal_batch()
    noisy_speech_estimate = speech + 0.5 * noise

    positive = dnf_output_eq14(noisy_speech_estimate, 7.0 * noise)
    negative = dnf_output_eq14(noisy_speech_estimate, -3.0 * noise)

    torch.testing.assert_close(positive.enhanced, speech)
    torch.testing.assert_close(negative.enhanced, speech)
    torch.testing.assert_close(positive.enhanced, negative.enhanced)
    torch.testing.assert_close(positive.enhanced_noise_inner_product, torch.zeros(2, dtype=DTYPE), atol=1e-12, rtol=0)
    torch.testing.assert_close(negative.enhanced_noise_inner_product, torch.zeros(2, dtype=DTYPE), atol=1e-12, rtol=0)
    assert positive.valid_mask.all()
    assert negative.valid_mask.all()
    assert not positive.fallback_mask.any()


def test_eq14_uses_passthrough_fallback_for_invalid_noise_energy() -> None:
    speech, noise = _orthogonal_batch()
    zero_noise = torch.zeros_like(noise)

    result = dnf_output_eq14(speech, zero_noise)

    torch.testing.assert_close(result.enhanced, speech)
    torch.testing.assert_close(result.projection_coefficient, torch.zeros(2, dtype=DTYPE))
    assert not result.valid_mask.any()
    assert result.fallback_mask.all()
    assert result.denominator_was_guarded.all()


def test_eq15_oracle_clean_targets_minimize_all_three_terms() -> None:
    speech, noise = _orthogonal_batch()
    noisy_speech_estimate = speech + 0.5 * noise
    interference = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=DTYPE,
    )

    oracle = dnf_clean_loss_eq15(noisy_speech_estimate, noise, speech, noise)
    perturbed = dnf_clean_loss_eq15(
        noisy_speech_estimate + 0.25 * interference,
        noise + 0.25 * interference,
        speech,
        noise,
    )

    torch.testing.assert_close(oracle.noisy_speech_target, noisy_speech_estimate)
    torch.testing.assert_close(oracle.projection.enhanced, speech)
    assert oracle.total.shape == (2,)
    assert torch.isfinite(oracle.total).all()
    assert torch.all(oracle.total < perturbed.total)
    assert oracle.valid_mask.all()
    assert torch.all(oracle.noisy_speech.value < perturbed.noisy_speech.value)
    assert torch.all(oracle.noise.value < perturbed.noise.value)
    assert torch.all(oracle.final.value < perturbed.final.value)


def test_si_sdr_zero_target_is_explicitly_invalid() -> None:
    estimate = torch.ones(2, 8, dtype=DTYPE)
    zero_target = torch.zeros_like(estimate)

    result = si_sdr_loss(estimate, zero_target)

    assert not result.valid_mask.any()
    assert torch.isnan(result.value).all()
    assert torch.isnan(result.value.mean())


def test_eq15_projection_fallback_cannot_silently_enter_default_mean() -> None:
    speech, noise = _orthogonal_batch()
    zero_noise_estimate = torch.zeros_like(noise)

    result = dnf_clean_loss_eq15(speech + 0.5 * noise, zero_noise_estimate, speech, noise)

    assert result.projection.fallback_mask.all()
    assert not result.valid_mask.any()
    assert torch.isnan(result.total).all()
    assert torch.isnan(result.total.mean())


def test_raw_subtraction_is_not_equation14() -> None:
    speech, noise = _orthogonal_batch()
    noisy_speech_estimate = speech + 0.5 * noise

    raw_subtraction = noisy_speech_estimate - noise
    projected = dnf_output_eq14(noisy_speech_estimate, noise).enhanced

    torch.testing.assert_close(projected, speech)
    assert not torch.allclose(raw_subtraction, speech)
    torch.testing.assert_close(raw_subtraction - speech, -0.5 * noise)


def test_eq13_and_eq15_have_finite_nonzero_gradients() -> None:
    generator = torch.Generator().manual_seed(3407)
    noisy_speech_target = torch.randn(3, 32, generator=generator, dtype=DTYPE)
    artificial_noise = torch.randn(3, 32, generator=generator, dtype=DTYPE)
    clean_speech = torch.randn(3, 32, generator=generator, dtype=DTYPE)
    mixture_noise = torch.randn(3, 32, generator=generator, dtype=DTYPE)
    noisy_speech_estimate = torch.randn(3, 32, generator=generator, dtype=DTYPE, requires_grad=True)
    noise_estimate = torch.randn(3, 32, generator=generator, dtype=DTYPE, requires_grad=True)

    noisy_loss = dnf_noisy_loss_eq13(
        noisy_speech_estimate,
        noise_estimate,
        noisy_speech_target,
        artificial_noise,
    )
    clean_loss = dnf_clean_loss_eq15(
        noisy_speech_estimate,
        noise_estimate,
        clean_speech,
        mixture_noise,
    )
    total = noisy_loss.total.mean() + clean_loss.total.mean()
    total.backward()

    assert torch.isfinite(total)
    assert noisy_speech_estimate.grad is not None
    assert noise_estimate.grad is not None
    assert torch.isfinite(noisy_speech_estimate.grad).all()
    assert torch.isfinite(noise_estimate.grad).all()
    assert noisy_speech_estimate.grad.abs().sum() > 0
    assert noise_estimate.grad.abs().sum() > 0


def test_eq13_passes_gradcheck_without_scale_clamp() -> None:
    generator = torch.Generator().manual_seed(17)
    artificial_noise = torch.randn(2, 8, generator=generator, dtype=DTYPE)
    noisy_speech_target = torch.randn(2, 8, generator=generator, dtype=DTYPE)
    noisy_speech_estimate = (
        torch.randn(2, 8, generator=generator, dtype=DTYPE) + 0.5 * artificial_noise
    ).requires_grad_()
    noise_estimate = (
        torch.randn(2, 8, generator=generator, dtype=DTYPE) + artificial_noise
    ).requires_grad_()

    def loss_function(speech_branch: torch.Tensor, noise_branch: torch.Tensor) -> torch.Tensor:
        return dnf_noisy_loss_eq13(
            speech_branch,
            noise_branch,
            noisy_speech_target,
            artificial_noise,
        ).total

    assert torch.autograd.gradcheck(
        loss_function,
        (noisy_speech_estimate, noise_estimate),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_public_losses_promote_half_geometry_to_float32() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        speech = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)
        noise = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)
        noisy_speech_estimate = speech + 0.5 * noise
        noisy_target = speech + noise

        noisy_loss = dnf_noisy_loss_eq13(
            noisy_speech_estimate,
            noise,
            noisy_target,
            noise,
        )
        clean_loss = dnf_clean_loss_eq15(
            noisy_speech_estimate,
            noise,
            speech,
            noise,
        )

        assert noisy_loss.total.dtype == torch.float32
        assert clean_loss.total.dtype == torch.float32
        assert clean_loss.projection.enhanced.dtype == torch.float32
        total = noisy_loss.total.mean() + clean_loss.total.mean()
        total.backward()
        assert speech.grad is not None
        assert noise.grad is not None
        assert torch.isfinite(speech.grad).all()
        assert torch.isfinite(noise.grad).all()
