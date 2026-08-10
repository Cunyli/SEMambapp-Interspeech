from __future__ import annotations

import math
from collections.abc import Sequence
from operator import index as integer_index

import torch
import torch.nn.functional as F


DEFAULT_FRAME_LENGTH = 512
DEFAULT_HOP_LENGTH = 160
DEFAULT_ACTIVE_RELATIVE_DB = -40.0
DEFAULT_ACTIVE_ABSOLUTE_RMS = 1e-5
DEFAULT_SURVIVAL_FLOOR_DB = -6.0
DEFAULT_EPSILON = 1e-8
DEFAULT_ANCHOR_MAX_LAG_SAMPLES = 1600
DEFAULT_RESIDUAL_LAG_RADIUS_SAMPLES = 32
DEFAULT_LAG_TIE_TOLERANCE = 1e-7


def _validate_waveform_pair(reference: torch.Tensor, estimate: torch.Tensor) -> None:
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError(
            "reference and estimate must both have shape [batch, samples], "
            f"got {tuple(reference.shape)} and {tuple(estimate.shape)}"
        )
    if reference.size(0) != estimate.size(0):
        raise ValueError(
            "reference and estimate batch sizes differ: "
            f"{reference.size(0)} != {estimate.size(0)}"
        )
    if reference.device != estimate.device:
        raise ValueError(
            f"reference and estimate must share a device, got {reference.device} and {estimate.device}"
        )


def _validate_sample_mask(sample_mask: torch.Tensor, batch_size: int) -> None:
    if sample_mask.dtype != torch.bool:
        raise TypeError(f"sample_mask must be bool, got {sample_mask.dtype}")
    if sample_mask.ndim != 1 or sample_mask.numel() != batch_size:
        raise ValueError(
            f"sample_mask must have shape [{batch_size}], got {tuple(sample_mask.shape)}"
        )


def _differentiable_zero(estimate: torch.Tensor) -> torch.Tensor:
    return estimate.float().sum() * 0.0


def _crop_waveform_pair(
    reference: torch.Tensor,
    estimate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_waveform_pair(reference, estimate)
    sample_count = min(reference.size(-1), estimate.size(-1))
    if sample_count == 0:
        raise ValueError("reference and estimate must contain at least one sample")
    return reference[..., :sample_count], estimate[..., :sample_count]


def frame_rms(
    waveform: torch.Tensor,
    *,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Return frame RMS with shape [batch, frames] for [batch, samples] audio."""
    if waveform.ndim != 2:
        raise ValueError(f"waveform must have shape [batch, samples], got {tuple(waveform.shape)}")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if waveform.size(-1) < frame_length:
        raise ValueError(
            f"waveform has {waveform.size(-1)} samples, shorter than frame_length={frame_length}"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    frames = waveform.float().unfold(-1, frame_length, hop_length)
    mean_square = frames.square().mean(dim=-1)
    return torch.sqrt(mean_square.clamp_min(epsilon**2))


def reference_active_frame_mask(
    reference_rms: torch.Tensor,
    *,
    relative_threshold_db: float = DEFAULT_ACTIVE_RELATIVE_DB,
    absolute_rms_floor: float = DEFAULT_ACTIVE_ABSOLUTE_RMS,
) -> torch.Tensor:
    """Build a detached relative-to-peak active mask for frame RMS values."""
    if reference_rms.ndim != 2:
        raise ValueError(
            f"reference_rms must have shape [batch, frames], got {tuple(reference_rms.shape)}"
        )
    if relative_threshold_db > 0:
        raise ValueError("relative_threshold_db must be non-positive")
    if absolute_rms_floor < 0:
        raise ValueError("absolute_rms_floor must be non-negative")

    detached_rms = reference_rms.detach()
    peak_rms = detached_rms.amax(dim=-1, keepdim=True)
    relative_ratio = 10.0 ** (relative_threshold_db / 20.0)
    relative_floor = peak_rms * relative_ratio
    absolute_floor = torch.full_like(relative_floor, absolute_rms_floor)
    threshold = torch.maximum(relative_floor, absolute_floor)
    non_silent = peak_rms > absolute_rms_floor
    return (detached_rms >= threshold) & non_silent


def _selected_sample_weights(
    sample_weights: torch.Tensor | None,
    sample_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    device_mask = sample_mask.to(device=device)
    if sample_weights is None:
        return torch.ones(int(device_mask.sum().item()), dtype=dtype, device=device)
    if sample_weights.ndim != 1 or sample_weights.numel() != sample_mask.numel():
        raise ValueError(
            f"sample_weights must have shape [{sample_mask.numel()}], "
            f"got {tuple(sample_weights.shape)}"
        )
    weights = sample_weights.detach().to(device=device, dtype=dtype)
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("sample_weights must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError("sample_weights must be non-negative")
    return weights[device_mask]


def build_sv_guard_masks(
    sv_mask: torch.Tensor,
    identity_mask: torch.Tensor,
    pathological_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exclusive v2 masks from already-decoded batch metadata."""
    for name, mask in (
        ("sv_mask", sv_mask),
        ("identity_mask", identity_mask),
        ("pathological_mask", pathological_mask),
    ):
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional bool tensor")
    if sv_mask.shape != identity_mask.shape or sv_mask.shape != pathological_mask.shape:
        raise ValueError("SV, identity, and pathological masks must have identical shapes")
    if sv_mask.device != identity_mask.device or sv_mask.device != pathological_mask.device:
        raise ValueError("SV, identity, and pathological masks must share a device")

    survival_mask = sv_mask & ~identity_mask
    pathology_mask = survival_mask & pathological_mask
    if bool((survival_mask & identity_mask).any().item()):
        raise AssertionError("identity rows must not enter the v2 survival mask")
    if bool((pathology_mask & ~survival_mask).any().item()):
        raise AssertionError("pathology mask must be a subset of survival mask")
    return {
        "survival": survival_mask,
        "pathology": pathology_mask,
    }


@torch.no_grad()
def best_normalized_cross_correlation_lag(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    max_lag_samples: int,
    epsilon: float = DEFAULT_EPSILON,
    tie_tolerance: float = DEFAULT_LAG_TIE_TOLERANCE,
) -> int:
    """Return a detached signed lag, preferring zero in numerical ties."""
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("reference and estimate must be one-dimensional")
    if reference.device != estimate.device:
        raise ValueError("reference and estimate must share a device")
    if max_lag_samples < 0:
        raise ValueError("max_lag_samples must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")

    sample_count = min(reference.numel(), estimate.numel())
    if sample_count <= max_lag_samples:
        raise ValueError("waveforms must be longer than max_lag_samples")
    reference = reference[:sample_count].detach().float()
    estimate = estimate[:sample_count].detach().float()
    if not bool(torch.isfinite(reference).all().item()) or not bool(
        torch.isfinite(estimate).all().item()
    ):
        raise ValueError("waveforms must be finite")

    fft_size = 1 << (2 * sample_count - 1).bit_length()
    reference_fft = torch.fft.rfft(reference, n=fft_size)
    estimate_fft = torch.fft.rfft(estimate, n=fft_size)
    circular_correlation = torch.fft.irfft(
        estimate_fft * reference_fft.conj(),
        n=fft_size,
    )
    lags = torch.arange(
        -max_lag_samples,
        max_lag_samples + 1,
        dtype=torch.long,
        device=reference.device,
    )
    indices = torch.remainder(lags, fft_size)
    numerator = circular_correlation.index_select(0, indices)

    reference_prefix = torch.cat(
        [torch.zeros_like(reference[:1]), reference.square().cumsum(dim=0)]
    )
    estimate_prefix = torch.cat(
        [torch.zeros_like(estimate[:1]), estimate.square().cumsum(dim=0)]
    )
    nonnegative = lags >= 0
    positive_lags = lags.clamp_min(0)
    negative_magnitudes = (-lags).clamp_min(0)
    reference_energy = torch.where(
        nonnegative,
        reference_prefix.index_select(0, sample_count - positive_lags),
        reference_prefix[-1] - reference_prefix.index_select(0, negative_magnitudes),
    )
    estimate_energy = torch.where(
        nonnegative,
        estimate_prefix[-1] - estimate_prefix.index_select(0, positive_lags),
        estimate_prefix.index_select(0, sample_count - negative_magnitudes),
    )
    denominator = torch.sqrt((reference_energy * estimate_energy).clamp_min(epsilon**2))
    normalized = (numerator / denominator).clamp(min=-1.0, max=1.0)

    maximum = normalized.max()
    tied = torch.isclose(normalized, maximum, rtol=0.0, atol=tie_tolerance)
    tie_indices = torch.nonzero(tied, as_tuple=False).flatten()
    tie_lags = lags.index_select(0, tie_indices)
    centered_index = tie_indices[torch.argmin(tie_lags.abs())]
    return int(lags[centered_index].item())


def align_waveform_pair(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    lag_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a one-dimensional pair at a fixed integer lag without detaching it."""
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("reference and estimate must be one-dimensional")
    sample_count = min(reference.numel(), estimate.numel())
    if sample_count == 0:
        raise ValueError("reference and estimate must contain samples")
    if abs(lag_samples) >= sample_count:
        raise ValueError("absolute lag must be shorter than the waveforms")
    reference = reference[:sample_count]
    estimate = estimate[:sample_count]
    if lag_samples > 0:
        return reference[: sample_count - lag_samples], estimate[lag_samples:]
    if lag_samples < 0:
        magnitude = -lag_samples
        return reference[magnitude:], estimate[: sample_count - magnitude]
    return reference, estimate


def sv_survival_loss(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    floor_db: float = DEFAULT_SURVIVAL_FLOOR_DB,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    active_relative_db: float = DEFAULT_ACTIVE_RELATIVE_DB,
    active_absolute_rms: float = DEFAULT_ACTIVE_ABSOLUTE_RMS,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, torch.Tensor]:
    """Penalize active SV frames whose target-coherent output falls below a floor.

    The clean pathological target is the reference. Each sample is reduced
    before the weighted batch mean so frame count cannot change its dose. The
    signed projection onto the detached reference gives exact silence a useful
    gradient and prevents unrelated noise energy from satisfying the guard.
    """
    reference, estimate = _crop_waveform_pair(reference, estimate)
    _validate_sample_mask(sample_mask, reference.size(0))
    sample_mask = sample_mask.to(device=reference.device)
    if not bool(torch.isfinite(reference).all().item()):
        raise ValueError("reference waveform must be finite")
    if not bool(torch.isfinite(estimate).all().item()):
        raise ValueError("estimate waveform must be finite")
    if not math.isfinite(float(floor_db)):
        raise ValueError("floor_db must be finite")
    if floor_db > 0:
        raise ValueError("floor_db must be non-positive")
    zero = _differentiable_zero(estimate)
    scalar_zero = reference.new_zeros((), dtype=torch.float32)
    if not bool(sample_mask.any().item()):
        return {
            "loss": zero,
            "sample_count": scalar_zero,
            "active_frame_count": scalar_zero,
            "active_frame_fraction": scalar_zero,
            "violation_fraction": scalar_zero,
            "mean_coherent_ratio": scalar_zero,
            "mean_coherent_gain_db": scalar_zero,
            "mean_rms_gain_db": scalar_zero,
        }

    selected_reference = reference[sample_mask]
    selected_estimate = estimate[sample_mask]
    reference_frames = selected_reference.float().unfold(-1, frame_length, hop_length).detach()
    estimate_frames = selected_estimate.float().unfold(-1, frame_length, hop_length)
    reference_rms = frame_rms(
        selected_reference,
        frame_length=frame_length,
        hop_length=hop_length,
        epsilon=epsilon,
    )
    estimate_rms = frame_rms(
        selected_estimate,
        frame_length=frame_length,
        hop_length=hop_length,
        epsilon=epsilon,
    )
    active_mask = reference_active_frame_mask(
        reference_rms,
        relative_threshold_db=active_relative_db,
        absolute_rms_floor=active_absolute_rms,
    )
    valid_samples = active_mask.any(dim=-1)
    if not bool(valid_samples.any().item()):
        return {
            "loss": zero,
            "sample_count": scalar_zero,
            "active_frame_count": scalar_zero,
            "active_frame_fraction": scalar_zero,
            "violation_fraction": scalar_zero,
            "mean_coherent_ratio": scalar_zero,
            "mean_coherent_gain_db": scalar_zero,
            "mean_rms_gain_db": scalar_zero,
        }

    reference_power = reference_frames.square().mean(dim=-1)
    coherent_ratio = (estimate_frames * reference_frames).mean(dim=-1) / reference_power.clamp_min(
        epsilon**2
    )
    floor_ratio = 10.0 ** (float(floor_db) / 20.0)
    normalized_shortfall = F.relu(floor_ratio - coherent_ratio) / floor_ratio
    element_loss = F.smooth_l1_loss(
        normalized_shortfall,
        torch.zeros_like(normalized_shortfall),
        reduction="none",
    )

    rms_gain_db = 20.0 * (
        torch.log10(estimate_rms + epsilon) - torch.log10(reference_rms + epsilon)
    )
    active_float = active_mask.to(dtype=coherent_ratio.dtype)
    active_count_per_sample = active_float.sum(dim=-1).clamp_min(1.0)
    per_sample_loss = (element_loss * active_float).sum(dim=-1) / active_count_per_sample

    selected_weights = _selected_sample_weights(
        sample_weights,
        sample_mask,
        dtype=per_sample_loss.dtype,
        device=per_sample_loss.device,
    )[valid_samples]
    if not bool((selected_weights > 0).any().item()):
        loss = zero
    else:
        loss = (per_sample_loss[valid_samples] * selected_weights).sum() / reference.size(0)

    active_coherent_ratio = coherent_ratio[active_mask]
    active_coherent_gain_db = 20.0 * torch.log10(active_coherent_ratio.clamp_min(epsilon))
    active_rms_gain_db = rms_gain_db[active_mask]
    active_shortfall = normalized_shortfall[active_mask]
    return {
        "loss": loss,
        "sample_count": valid_samples.float().sum().detach(),
        "active_frame_count": active_float.sum().detach(),
        "active_frame_fraction": active_float.mean().detach(),
        "violation_fraction": (active_shortfall > 0).float().mean().detach(),
        "mean_coherent_ratio": active_coherent_ratio.mean().detach(),
        "mean_coherent_gain_db": active_coherent_gain_db.mean().detach(),
        "mean_rms_gain_db": active_rms_gain_db.mean().detach(),
    }


def lag_robust_sv_survival_loss(
    reference: torch.Tensor,
    degraded_input: torch.Tensor,
    estimate: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    anchor_max_lag_samples: int = DEFAULT_ANCHOR_MAX_LAG_SAMPLES,
    residual_lag_radius_samples: int = DEFAULT_RESIDUAL_LAG_RADIUS_SAMPLES,
    floor_db: float = DEFAULT_SURVIVAL_FLOOR_DB,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    active_relative_db: float = DEFAULT_ACTIVE_RELATIVE_DB,
    active_absolute_rms: float = DEFAULT_ACTIVE_ABSOLUTE_RMS,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, torch.Tensor]:
    """Apply survival after a frozen input anchor and a tiny detached residual search.

    The input-to-clean lag is selected within ``anchor_max_lag_samples``. The
    enhanced output may then select only a local residual lag around that
    anchor. Both integer choices are detached; gradients flow solely through
    the final signed coherent projection loss. Each selected sample is reduced
    independently and divided by the original local batch size.
    """
    _validate_waveform_pair(reference, degraded_input)
    _validate_waveform_pair(reference, estimate)
    _validate_sample_mask(sample_mask, reference.size(0))
    if anchor_max_lag_samples < 0:
        raise ValueError("anchor_max_lag_samples must be non-negative")
    if residual_lag_radius_samples < 0:
        raise ValueError("residual_lag_radius_samples must be non-negative")

    sample_count = min(
        reference.size(-1),
        degraded_input.size(-1),
        estimate.size(-1),
    )
    if sample_count <= anchor_max_lag_samples + residual_lag_radius_samples:
        raise ValueError("waveforms are too short for the requested lag searches")
    reference = reference[..., :sample_count]
    degraded_input = degraded_input[..., :sample_count]
    estimate = estimate[..., :sample_count]
    sample_mask = sample_mask.to(device=reference.device)
    if not bool(torch.isfinite(reference).all().item()):
        raise ValueError("reference waveform must be finite")
    if not bool(torch.isfinite(degraded_input).all().item()):
        raise ValueError("degraded_input waveform must be finite")
    if not bool(torch.isfinite(estimate).all().item()):
        raise ValueError("estimate waveform must be finite")

    zero = _differentiable_zero(estimate)
    scalar_zero = reference.new_zeros((), dtype=torch.float32)
    empty_result = {
        "loss": zero,
        "sample_count": scalar_zero,
        "active_frame_count": scalar_zero,
        "active_frame_fraction": scalar_zero,
        "violation_fraction": scalar_zero,
        "mean_coherent_ratio": scalar_zero,
        "mean_coherent_gain_db": scalar_zero,
        "mean_rms_gain_db": scalar_zero,
        "mean_input_anchor_lag_samples": scalar_zero,
        "mean_abs_input_anchor_lag_samples": scalar_zero,
        "mean_residual_lag_samples": scalar_zero,
        "mean_abs_residual_lag_samples": scalar_zero,
        "residual_boundary_fraction": scalar_zero,
    }
    if not bool(sample_mask.any().item()):
        return empty_result

    selected_weights = _selected_sample_weights(
        sample_weights,
        sample_mask,
        dtype=torch.float32,
        device=reference.device,
    )
    selected_indices = torch.nonzero(sample_mask, as_tuple=False).flatten().tolist()
    sample_results: list[dict[str, torch.Tensor]] = []
    weighted_losses: list[torch.Tensor] = []
    input_lags: list[int] = []
    residual_lags: list[int] = []
    for selected_position, batch_index in enumerate(selected_indices):
        sample_reference = reference[batch_index]
        sample_input = degraded_input[batch_index]
        sample_estimate = estimate[batch_index]
        input_lag = best_normalized_cross_correlation_lag(
            sample_reference,
            sample_input,
            max_lag_samples=anchor_max_lag_samples,
            epsilon=epsilon,
        )
        anchor_reference, anchor_estimate = align_waveform_pair(
            sample_reference,
            sample_estimate,
            input_lag,
        )
        residual_lag = best_normalized_cross_correlation_lag(
            anchor_reference,
            anchor_estimate,
            max_lag_samples=residual_lag_radius_samples,
            epsilon=epsilon,
        )
        aligned_reference, aligned_estimate = align_waveform_pair(
            anchor_reference,
            anchor_estimate,
            residual_lag,
        )
        sample_result = sv_survival_loss(
            aligned_reference.unsqueeze(0),
            aligned_estimate.unsqueeze(0),
            torch.ones(1, dtype=torch.bool, device=reference.device),
            floor_db=floor_db,
            frame_length=frame_length,
            hop_length=hop_length,
            active_relative_db=active_relative_db,
            active_absolute_rms=active_absolute_rms,
            epsilon=epsilon,
        )
        sample_results.append(sample_result)
        weighted_losses.append(sample_result["loss"] * selected_weights[selected_position])
        input_lags.append(input_lag)
        residual_lags.append(residual_lag)

    loss = torch.stack(weighted_losses).sum() / reference.size(0)
    active_frame_counts = torch.stack(
        [result["active_frame_count"] for result in sample_results]
    )
    total_active_frames = active_frame_counts.sum()
    valid_mask = torch.stack([result["sample_count"] for result in sample_results]) > 0

    def active_frame_weighted_mean(key: str) -> torch.Tensor:
        if not bool((total_active_frames > 0).item()):
            return scalar_zero
        values = torch.stack([result[key] for result in sample_results])
        return ((values * active_frame_counts).sum() / total_active_frames).detach()

    if bool(valid_mask.any().item()):
        active_frame_fraction = torch.stack(
            [result["active_frame_fraction"] for result in sample_results]
        )[valid_mask].mean().detach()
    else:
        active_frame_fraction = scalar_zero

    input_lag_tensor = reference.new_tensor(input_lags, dtype=torch.float32)
    residual_lag_tensor = reference.new_tensor(residual_lags, dtype=torch.float32)
    residual_boundary_fraction = (
        residual_lag_tensor.abs() == float(residual_lag_radius_samples)
    ).float().mean()
    return {
        "loss": loss,
        "sample_count": valid_mask.float().sum().detach(),
        "active_frame_count": total_active_frames.detach(),
        "active_frame_fraction": active_frame_fraction,
        "violation_fraction": active_frame_weighted_mean("violation_fraction"),
        "mean_coherent_ratio": active_frame_weighted_mean("mean_coherent_ratio"),
        "mean_coherent_gain_db": active_frame_weighted_mean("mean_coherent_gain_db"),
        "mean_rms_gain_db": active_frame_weighted_mean("mean_rms_gain_db"),
        "mean_input_anchor_lag_samples": input_lag_tensor.mean().detach(),
        "mean_abs_input_anchor_lag_samples": input_lag_tensor.abs().mean().detach(),
        "mean_residual_lag_samples": residual_lag_tensor.mean().detach(),
        "mean_abs_residual_lag_samples": residual_lag_tensor.abs().mean().detach(),
        "residual_boundary_fraction": residual_boundary_fraction.detach(),
    }


def pathology_feature_match_loss(
    reference_features: torch.Tensor,
    estimate_features: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    feature_indices: Sequence[int],
    feature_scales: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    tolerance: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Match selected frozen-estimator features to the clean pathology target.

    Inputs have shape [batch, frames, features]. The reference branch is
    detached deliberately; gradients flow only through estimate_features.
    """
    if reference_features.ndim != 3 or estimate_features.ndim != 3:
        raise ValueError(
            "reference_features and estimate_features must have shape "
            f"[batch, frames, features], got {tuple(reference_features.shape)} "
            f"and {tuple(estimate_features.shape)}"
        )
    if reference_features.shape != estimate_features.shape:
        raise ValueError(
            "reference and estimate feature shapes must match exactly, got "
            f"{tuple(reference_features.shape)} and {tuple(estimate_features.shape)}"
        )
    if reference_features.device != estimate_features.device:
        raise ValueError(
            "reference and estimate features must share a device, got "
            f"{reference_features.device} and {estimate_features.device}"
        )
    if reference_features.size(1) == 0 or reference_features.size(2) == 0:
        raise ValueError("feature tensors must contain at least one frame and one feature")
    if not bool(torch.isfinite(reference_features).all().item()):
        raise ValueError("reference_features must be finite")
    if not bool(torch.isfinite(estimate_features).all().item()):
        raise ValueError("estimate_features must be finite")
    _validate_sample_mask(sample_mask, reference_features.size(0))
    sample_mask = sample_mask.to(device=estimate_features.device)
    if frame_mask is not None:
        if frame_mask.dtype != torch.bool:
            raise TypeError(f"frame_mask must be bool, got {frame_mask.dtype}")
        expected_frame_shape = reference_features.shape[:2]
        if frame_mask.shape != expected_frame_shape:
            raise ValueError(
                f"frame_mask must have shape {tuple(expected_frame_shape)}, "
                f"got {tuple(frame_mask.shape)}"
            )
        frame_mask = frame_mask.to(device=estimate_features.device)
    if not feature_indices:
        raise ValueError("feature_indices must not be empty")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    try:
        normalized_indices = tuple(integer_index(value) for value in feature_indices)
    except TypeError as error:
        raise TypeError("feature_indices must contain only integers") from error
    if len(normalized_indices) != len(set(normalized_indices)):
        raise ValueError("feature_indices must not contain duplicates")
    feature_count = reference_features.size(-1)
    indices = torch.tensor(normalized_indices, dtype=torch.long, device=estimate_features.device)
    if bool((indices < 0).any().item()) or bool((indices >= feature_count).any().item()):
        raise IndexError(
            f"feature index outside [0, {feature_count}): {tuple(feature_indices)}"
        )

    frame_count = reference_features.size(1)
    reference = reference_features.detach().index_select(-1, indices)
    estimate = estimate_features.index_select(-1, indices)
    zero = _differentiable_zero(estimate)
    scalar_zero = reference.new_zeros((), dtype=torch.float32)
    if not bool(sample_mask.any().item()):
        return {
            "loss": zero,
            "sample_count": scalar_zero,
            "frame_count": scalar_zero,
            "mean_abs_standardized_gap": scalar_zero,
            "active_fraction": scalar_zero,
        }

    if feature_scales is None:
        scales = torch.ones(len(feature_indices), dtype=estimate.dtype, device=estimate.device)
    else:
        if feature_scales.ndim != 1 or feature_scales.numel() != len(feature_indices):
            raise ValueError(
                f"feature_scales must have shape [{len(feature_indices)}], "
                f"got {tuple(feature_scales.shape)}"
            )
        scales = feature_scales.detach().to(device=estimate.device, dtype=estimate.dtype)
        if not bool(torch.isfinite(scales).all().item()) or bool((scales <= 0).any().item()):
            raise ValueError("feature_scales must be finite and positive")

    selected_reference = reference[sample_mask]
    selected_estimate = estimate[sample_mask]
    standardized_gap = (selected_estimate - selected_reference) / scales.view(1, 1, -1)
    excess = F.relu(standardized_gap.abs() - float(tolerance))
    element_loss = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    per_frame_loss = element_loss.mean(dim=-1)

    if frame_mask is None:
        selected_frame_mask = torch.ones_like(per_frame_loss, dtype=torch.bool)
    else:
        selected_frame_mask = frame_mask[sample_mask]

    valid_samples = selected_frame_mask.any(dim=-1)
    if not bool(valid_samples.any().item()):
        return {
            "loss": zero,
            "sample_count": scalar_zero,
            "frame_count": scalar_zero,
            "mean_abs_standardized_gap": scalar_zero,
            "active_fraction": scalar_zero,
        }

    frame_float = selected_frame_mask.to(dtype=per_frame_loss.dtype)
    frame_count_per_sample = frame_float.sum(dim=-1).clamp_min(1.0)
    per_sample_loss = (per_frame_loss * frame_float).sum(dim=-1) / frame_count_per_sample
    selected_weights = _selected_sample_weights(
        sample_weights,
        sample_mask,
        dtype=per_sample_loss.dtype,
        device=per_sample_loss.device,
    )[valid_samples]
    if not bool((selected_weights > 0).any().item()):
        loss = zero
    else:
        loss = (per_sample_loss[valid_samples] * selected_weights).sum() / reference_features.size(0)

    selected_gap = standardized_gap[selected_frame_mask]
    return {
        "loss": loss,
        "sample_count": valid_samples.float().sum().detach(),
        "frame_count": frame_float.sum().detach(),
        "mean_abs_standardized_gap": selected_gap.abs().mean().detach(),
        "active_fraction": (selected_gap.abs() > float(tolerance)).float().mean().detach(),
    }
