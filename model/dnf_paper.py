"""Paper-faithful waveform equations for Differential Noise Filtering.

This module implements equations (5), (13), (14), and (15) from
arXiv:2606.02327 without defining a network architecture.  All waveform
tensors use the final dimension as time, and every loss is returned without
batch reduction so callers can retain per-sample diagnostics.

The signed safe division, epsilon floors, and optional scale clamp are
implementation guardrails for finite computation.  They are not additional
terms in the paper equations.  SI-SDR follows the paper's zero-mean waveform
assumption and therefore does not implicitly remove the mean.  Public geometry
and loss APIs promote float16 and bfloat16 waveforms to float32 before dot
products and norms; this promotion preserves autograd and avoids silent
low-precision accumulation.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


DEFAULT_EPS = 1e-8


@dataclass(frozen=True)
class HalfNoiseCalibration:
    """Equation (13) rescaling result and per-sample diagnostics."""

    scaled_estimate: Tensor
    scale: Tensor
    artificial_noise_energy: Tensor
    estimate_noise_inner_product: Tensor
    calibrated_noise_coefficient: Tensor
    valid_mask: Tensor
    faithful_mask: Tensor
    invalid_mask: Tensor
    denominator_was_guarded: Tensor
    scale_was_clamped: Tensor


@dataclass(frozen=True)
class NoisyTargetDNFLoss:
    """Unreduced equation (13) loss terms."""

    total: Tensor
    noisy_speech: Tensor
    noise: Tensor
    valid_mask: Tensor
    faithful_mask: Tensor
    noisy_speech_calibration: HalfNoiseCalibration
    noise_calibration: HalfNoiseCalibration


@dataclass(frozen=True)
class ProjectionSubtraction:
    """Equation (14) output and per-sample projection diagnostics."""

    enhanced: Tensor
    projection_coefficient: Tensor
    noisy_speech_noise_inner_product: Tensor
    noise_energy: Tensor
    enhanced_noise_inner_product: Tensor
    valid_mask: Tensor
    fallback_mask: Tensor
    denominator_was_guarded: Tensor


@dataclass(frozen=True)
class ScaleInvariantSDRLoss:
    """Unreduced SI-SDR loss with explicit target validity."""

    value: Tensor
    valid_mask: Tensor
    target_energy: Tensor
    target_projection_energy: Tensor
    error_energy: Tensor


@dataclass(frozen=True)
class CleanTargetDNFLoss:
    """Unreduced equation (15) loss terms."""

    total: Tensor
    noisy_speech: ScaleInvariantSDRLoss
    noise: ScaleInvariantSDRLoss
    final: ScaleInvariantSDRLoss
    valid_mask: Tensor
    noisy_speech_target: Tensor
    projection: ProjectionSubtraction


def _validate_waveform_pair(left: Tensor, right: Tensor, names: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{names} must have identical shapes, got {left.shape} and {right.shape}")
    if left.ndim < 1:
        raise ValueError(f"{names} must include a time dimension")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError(f"{names} must be floating-point tensors")
    if left.dtype != right.dtype:
        raise TypeError(f"{names} must have the same dtype, got {left.dtype} and {right.dtype}")
    if left.device != right.device:
        raise ValueError(f"{names} must be on the same device, got {left.device} and {right.device}")


def _effective_eps(reference: Tensor, eps: float) -> float:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return max(float(eps), float(torch.finfo(reference.dtype).tiny))


def _energy(waveform: Tensor) -> Tensor:
    return waveform.square().sum(dim=-1)


def _inner_product(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).sum(dim=-1)


def _geometry_waveform(waveform: Tensor) -> Tensor:
    if waveform.dtype in {torch.float16, torch.bfloat16}:
        return waveform.float()
    return waveform


def signed_safe_divide(numerator: Tensor, denominator: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Divide while preserving the sign of a near-zero denominator.

    Equation (13) divides by an inner product, which can be negative.  A
    normal ``clamp_min`` would incorrectly flip negative denominators to a
    positive epsilon.  This guardrail clamps magnitude while preserving sign;
    an exact zero uses the positive sign convention.
    """

    if numerator.shape != denominator.shape:
        raise ValueError(
            "numerator and denominator must have identical shapes, "
            f"got {numerator.shape} and {denominator.shape}"
        )
    if not numerator.is_floating_point() or not denominator.is_floating_point():
        raise TypeError("numerator and denominator must be floating-point tensors")
    if numerator.dtype != denominator.dtype:
        raise TypeError(f"numerator and denominator must have the same dtype, got {numerator.dtype} and {denominator.dtype}")
    if numerator.device != denominator.device:
        raise ValueError("numerator and denominator must be on the same device")
    numerator = _geometry_waveform(numerator)
    denominator = _geometry_waveform(denominator)
    effective_eps = _effective_eps(denominator, eps)
    sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
    safe_denominator = sign * denominator.abs().clamp_min(effective_eps)
    return numerator / safe_denominator


def sdr_loss_eq5(estimate: Tensor, target: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Return equation (5) scale-dependent SDR loss per sample.

    ``-10 log10(||target||^2 / ||target - estimate||^2)``
    """

    _validate_waveform_pair(estimate, target, "estimate and target")
    estimate = _geometry_waveform(estimate)
    target = _geometry_waveform(target)
    effective_eps = _effective_eps(estimate, eps)
    target_energy = _energy(target)
    valid_mask = target_energy >= effective_eps
    safe_target_energy = torch.where(valid_mask, target_energy, torch.ones_like(target_energy))
    error_energy = _energy(target - estimate).clamp_min(effective_eps)
    loss = -10.0 * torch.log10(safe_target_energy / error_energy)
    return torch.where(valid_mask, loss, torch.full_like(loss, torch.nan))


def calibrate_half_noise_projection_eq13(
    estimate: Tensor,
    artificial_noise: Tensor,
    eps: float = DEFAULT_EPS,
    scale_clamp: float | None = None,
) -> HalfNoiseCalibration:
    """Rescale an estimate so its projection on ``n2`` is ``0.5 * n2``.

    This is the rescaling inside each equation (13) SDR term.  ``scale_clamp``
    is an optional symmetric numerical guardrail; ``None`` preserves the paper
    equation exactly apart from the epsilon-protected division.
    """

    _validate_waveform_pair(estimate, artificial_noise, "estimate and artificial_noise")
    estimate = _geometry_waveform(estimate)
    artificial_noise = _geometry_waveform(artificial_noise)
    effective_eps = _effective_eps(estimate, eps)
    noise_energy = _energy(artificial_noise)
    inner_product = _inner_product(artificial_noise, estimate)
    valid_mask = (noise_energy >= effective_eps) & (inner_product.abs() >= effective_eps)
    safe_inner_product = torch.where(valid_mask, inner_product, torch.ones_like(inner_product))
    exact_scale = signed_safe_divide(0.5 * noise_energy, safe_inner_product, eps=effective_eps)

    if scale_clamp is None:
        guarded_scale = exact_scale
        scale_was_clamped = torch.zeros_like(exact_scale, dtype=torch.bool)
    else:
        if scale_clamp <= 0:
            raise ValueError(f"scale_clamp must be positive when provided, got {scale_clamp}")
        guarded_scale = exact_scale.clamp(min=-float(scale_clamp), max=float(scale_clamp))
        scale_was_clamped = valid_mask & (guarded_scale != exact_scale)

    scale = torch.where(valid_mask, guarded_scale, torch.full_like(guarded_scale, torch.nan))

    scaled_estimate = scale.unsqueeze(-1) * estimate
    safe_noise_energy = torch.where(valid_mask, noise_energy, torch.ones_like(noise_energy))
    calibrated_noise_coefficient = torch.where(
        valid_mask,
        _inner_product(artificial_noise, scaled_estimate) / safe_noise_energy,
        torch.full_like(noise_energy, torch.nan),
    )
    return HalfNoiseCalibration(
        scaled_estimate=scaled_estimate,
        scale=scale,
        artificial_noise_energy=noise_energy,
        estimate_noise_inner_product=inner_product,
        calibrated_noise_coefficient=calibrated_noise_coefficient,
        valid_mask=valid_mask,
        faithful_mask=valid_mask & ~scale_was_clamped,
        invalid_mask=~valid_mask,
        denominator_was_guarded=inner_product.abs() < effective_eps,
        scale_was_clamped=scale_was_clamped,
    )


def dnf_noisy_loss_eq13(
    noisy_speech_estimate: Tensor,
    noise_estimate: Tensor,
    noisy_speech_target: Tensor,
    artificial_noise: Tensor,
    eps: float = DEFAULT_EPS,
    scale_clamp: float | None = None,
) -> NoisyTargetDNFLoss:
    """Return the two scale-dependent SDR terms in equation (13)."""

    _validate_waveform_pair(noisy_speech_estimate, noise_estimate, "branch estimates")
    _validate_waveform_pair(noisy_speech_estimate, noisy_speech_target, "estimate and noisy_speech_target")
    _validate_waveform_pair(noisy_speech_estimate, artificial_noise, "estimate and artificial_noise")
    noisy_speech_estimate = _geometry_waveform(noisy_speech_estimate)
    noise_estimate = _geometry_waveform(noise_estimate)
    noisy_speech_target = _geometry_waveform(noisy_speech_target)
    artificial_noise = _geometry_waveform(artificial_noise)

    noisy_speech_calibration = calibrate_half_noise_projection_eq13(
        noisy_speech_estimate,
        artificial_noise,
        eps=eps,
        scale_clamp=scale_clamp,
    )
    noise_calibration = calibrate_half_noise_projection_eq13(
        noise_estimate,
        artificial_noise,
        eps=eps,
        scale_clamp=scale_clamp,
    )
    noisy_speech_loss = sdr_loss_eq5(
        noisy_speech_calibration.scaled_estimate,
        noisy_speech_target,
        eps=eps,
    )
    noise_loss = sdr_loss_eq5(noise_calibration.scaled_estimate, artificial_noise, eps=eps)
    target_valid_mask = _energy(noisy_speech_target) >= _effective_eps(noisy_speech_target, eps)
    valid_mask = noisy_speech_calibration.valid_mask & noise_calibration.valid_mask & target_valid_mask
    faithful_mask = (
        valid_mask
        & noisy_speech_calibration.faithful_mask
        & noise_calibration.faithful_mask
    )
    return NoisyTargetDNFLoss(
        total=noisy_speech_loss + noise_loss,
        noisy_speech=noisy_speech_loss,
        noise=noise_loss,
        valid_mask=valid_mask,
        faithful_mask=faithful_mask,
        noisy_speech_calibration=noisy_speech_calibration,
        noise_calibration=noise_calibration,
    )


def dnf_output_eq14(
    noisy_speech_estimate: Tensor,
    noise_estimate: Tensor,
    eps: float = DEFAULT_EPS,
) -> ProjectionSubtraction:
    """Return equation (14) projection subtraction and diagnostics."""

    _validate_waveform_pair(noisy_speech_estimate, noise_estimate, "branch estimates")
    noisy_speech_estimate = _geometry_waveform(noisy_speech_estimate)
    noise_estimate = _geometry_waveform(noise_estimate)
    effective_eps = _effective_eps(noisy_speech_estimate, eps)
    inner_product = _inner_product(noise_estimate, noisy_speech_estimate)
    noise_energy = _energy(noise_estimate)
    valid_mask = noise_energy >= effective_eps
    safe_noise_energy = torch.where(valid_mask, noise_energy, torch.ones_like(noise_energy))
    exact_projection_coefficient = inner_product / safe_noise_energy
    projection_coefficient = torch.where(valid_mask, exact_projection_coefficient, torch.zeros_like(inner_product))
    enhanced = noisy_speech_estimate - projection_coefficient.unsqueeze(-1) * noise_estimate
    return ProjectionSubtraction(
        enhanced=enhanced,
        projection_coefficient=projection_coefficient,
        noisy_speech_noise_inner_product=inner_product,
        noise_energy=noise_energy,
        enhanced_noise_inner_product=_inner_product(noise_estimate, enhanced),
        valid_mask=valid_mask,
        fallback_mask=~valid_mask,
        denominator_was_guarded=~valid_mask,
    )


def si_sdr_loss(
    estimate: Tensor,
    target: Tensor,
    eps: float = DEFAULT_EPS,
) -> ScaleInvariantSDRLoss:
    """Return the standard scale-invariant SDR loss per sample.

    No implicit de-meaning is applied because the DNF derivation assumes the
    waveform signals are already approximately zero mean.
    """

    _validate_waveform_pair(estimate, target, "estimate and target")
    estimate = _geometry_waveform(estimate)
    target = _geometry_waveform(target)
    effective_eps = _effective_eps(estimate, eps)
    target_energy = _energy(target)
    valid_mask = target_energy >= effective_eps
    safe_target_energy = torch.where(valid_mask, target_energy, torch.ones_like(target_energy))
    target_scale = _inner_product(estimate, target) / safe_target_energy
    target_projection = target_scale.unsqueeze(-1) * target
    error = estimate - target_projection
    projection_energy = _energy(target_projection).clamp_min(effective_eps)
    error_energy = _energy(error).clamp_min(effective_eps)
    loss = -10.0 * torch.log10(projection_energy / error_energy)
    return ScaleInvariantSDRLoss(
        value=torch.where(valid_mask, loss, torch.full_like(loss, torch.nan)),
        valid_mask=valid_mask,
        target_energy=target_energy,
        target_projection_energy=projection_energy,
        error_energy=error_energy,
    )


def dnf_clean_loss_eq15(
    noisy_speech_estimate: Tensor,
    noise_estimate: Tensor,
    clean_speech: Tensor,
    mixture_noise: Tensor,
    eps: float = DEFAULT_EPS,
) -> CleanTargetDNFLoss:
    """Return the three SI-SDR terms in equation (15), unreduced."""

    _validate_waveform_pair(noisy_speech_estimate, noise_estimate, "branch estimates")
    _validate_waveform_pair(noisy_speech_estimate, clean_speech, "estimate and clean_speech")
    _validate_waveform_pair(noisy_speech_estimate, mixture_noise, "estimate and mixture_noise")
    noisy_speech_estimate = _geometry_waveform(noisy_speech_estimate)
    noise_estimate = _geometry_waveform(noise_estimate)
    clean_speech = _geometry_waveform(clean_speech)
    mixture_noise = _geometry_waveform(mixture_noise)

    noisy_speech_target = clean_speech + 0.5 * mixture_noise
    projection = dnf_output_eq14(noisy_speech_estimate, noise_estimate, eps=eps)
    noisy_speech_loss = si_sdr_loss(noisy_speech_estimate, noisy_speech_target, eps=eps)
    noise_loss = si_sdr_loss(noise_estimate, mixture_noise, eps=eps)
    final_loss = si_sdr_loss(projection.enhanced, clean_speech, eps=eps)
    valid_mask = (
        noisy_speech_loss.valid_mask
        & noise_loss.valid_mask
        & final_loss.valid_mask
        & projection.valid_mask
    )
    total = noisy_speech_loss.value + noise_loss.value + final_loss.value
    return CleanTargetDNFLoss(
        total=torch.where(valid_mask, total, torch.full_like(total, torch.nan)),
        noisy_speech=noisy_speech_loss,
        noise=noise_loss,
        final=final_loss,
        valid_mask=valid_mask,
        noisy_speech_target=noisy_speech_target,
        projection=projection,
    )


__all__ = [
    "CleanTargetDNFLoss",
    "DEFAULT_EPS",
    "HalfNoiseCalibration",
    "NoisyTargetDNFLoss",
    "ProjectionSubtraction",
    "ScaleInvariantSDRLoss",
    "calibrate_half_noise_projection_eq13",
    "dnf_clean_loss_eq15",
    "dnf_noisy_loss_eq13",
    "dnf_output_eq14",
    "sdr_loss_eq5",
    "si_sdr_loss",
    "signed_safe_divide",
]
