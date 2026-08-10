"""Pure additive sample construction for the paper-faithful DNF pilot.

This module deliberately contains no WebDataset, augmentation, routing, or
loss code.  It only constructs the additive waveform pairs required by the
DNF paper and returns route-specific supervision plus validity diagnostics.
The snapshot generator must pre-scale ``n1`` and ``n2`` to the contracted SNR
and equal-energy condition; this module never rescales them and performs only
the exact additions defined by the paper protocol.
"""

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor


DEFAULT_SILENCE_RMS_THRESHOLD = 1e-6


@dataclass(frozen=True)
class SegmentDiagnostics:
    """Per-segment waveform statistics over the final (time) dimension."""

    mean: Tensor
    energy: Tensor
    mean_square: Tensor
    rms: Tensor
    finite_mask: Tensor
    silent_mask: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class PaperSupervision:
    """A route-specific model input and the targets visible to its loss."""

    supervision_type: Literal["noisy_target", "clean_target"]
    model_input: Tensor
    targets: Mapping[str, Tensor]
    diagnostics: Mapping[str, SegmentDiagnostics]
    valid_mask: Tensor
    demeaned: bool
    source_ids: Mapping[str, str]


def _validate_threshold(silence_rms_threshold: float) -> float:
    threshold = float(silence_rms_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "silence_rms_threshold must be finite and non-negative, "
            f"got {silence_rms_threshold}"
        )
    return threshold


def _validate_waveforms(waveforms: Mapping[str, Tensor]) -> None:
    first_name, first = next(iter(waveforms.items()))
    if first.ndim < 1 or first.shape[-1] == 0:
        raise ValueError(f"{first_name} must have a non-empty time dimension")
    if not first.is_floating_point():
        raise TypeError(f"{first_name} must be a floating-point tensor")

    for name, waveform in waveforms.items():
        if waveform.shape != first.shape:
            raise ValueError(
                f"all waveforms must have the same shape; "
                f"{first_name}={first.shape}, {name}={waveform.shape}"
            )
        if not waveform.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if waveform.dtype != first.dtype:
            raise TypeError(
                f"all waveforms must have the same dtype; "
                f"{first_name}={first.dtype}, {name}={waveform.dtype}"
            )
        if waveform.device != first.device:
            raise ValueError(
                f"all waveforms must be on the same device; "
                f"{first_name}={first.device}, {name}={waveform.device}"
            )


def _prepare_waveform(waveform: Tensor, demean: bool) -> Tensor:
    if not demean:
        return waveform
    return waveform - waveform.mean(dim=-1, keepdim=True)


def _diagnose(waveform: Tensor, silence_rms_threshold: float) -> SegmentDiagnostics:
    mean = waveform.mean(dim=-1)
    energy = waveform.square().sum(dim=-1)
    mean_square = waveform.square().mean(dim=-1)
    rms = mean_square.sqrt()
    finite_mask = torch.isfinite(waveform).all(dim=-1)
    silent_mask = rms <= silence_rms_threshold
    return SegmentDiagnostics(
        mean=mean,
        energy=energy,
        mean_square=mean_square,
        rms=rms,
        finite_mask=finite_mask,
        silent_mask=silent_mask,
        valid_mask=finite_mask & ~silent_mask,
    )


def _diagnose_all(
    waveforms: Mapping[str, Tensor],
    silence_rms_threshold: float,
) -> tuple[dict[str, SegmentDiagnostics], Tensor]:
    diagnostics = {
        name: _diagnose(waveform, silence_rms_threshold)
        for name, waveform in waveforms.items()
    }
    valid_mask = torch.stack(
        [diagnostic.valid_mask for diagnostic in diagnostics.values()],
        dim=0,
    ).all(dim=0)
    return diagnostics, valid_mask


def build_noisy_target_supervision(
    clean_speech: Tensor,
    noise1: Tensor,
    noise2: Tensor,
    *,
    noise1_source_id: str,
    noise2_source_id: str,
    demean: bool = True,
    silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
) -> PaperSupervision:
    """Construct ``s_noisy=s+n1`` and ``x=s_noisy+n2``.

    Distinct source identifiers are mandatory because tensor values alone
    cannot prove that ``n1`` and ``n2`` were independently sampled.  The
    returned targets expose only ``s_noisy`` and the artificial noise ``n2``;
    clean speech and ``n1`` remain diagnostic-only latent components.
    """

    if not noise1_source_id or not noise2_source_id:
        raise ValueError("noise1_source_id and noise2_source_id must be non-empty")
    if noise1_source_id == noise2_source_id:
        raise ValueError(
            "noise1 and noise2 must come from distinct source recordings"
        )

    threshold = _validate_threshold(silence_rms_threshold)
    components = {
        "clean_speech": clean_speech,
        "noise1": noise1,
        "noise2": noise2,
    }
    _validate_waveforms(components)
    prepared = {
        name: _prepare_waveform(waveform, demean)
        for name, waveform in components.items()
    }
    noisy_speech = prepared["clean_speech"] + prepared["noise1"]
    model_input = noisy_speech + prepared["noise2"]
    diagnostic_waveforms = {
        **prepared,
        "noisy_speech": noisy_speech,
        "model_input": model_input,
    }
    diagnostics, valid_mask = _diagnose_all(diagnostic_waveforms, threshold)
    return PaperSupervision(
        supervision_type="noisy_target",
        model_input=model_input,
        targets={
            "noisy_speech": noisy_speech,
            "artificial_noise": prepared["noise2"],
        },
        diagnostics=diagnostics,
        valid_mask=valid_mask,
        demeaned=bool(demean),
        source_ids={
            "noise1": noise1_source_id,
            "noise2": noise2_source_id,
        },
    )


def build_clean_target_supervision(
    clean_speech: Tensor,
    mixture_noise: Tensor,
    *,
    demean: bool = True,
    silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
) -> PaperSupervision:
    """Construct the clean-target additive mixture ``x=s+n``."""

    threshold = _validate_threshold(silence_rms_threshold)
    components = {
        "clean_speech": clean_speech,
        "mixture_noise": mixture_noise,
    }
    _validate_waveforms(components)
    prepared = {
        name: _prepare_waveform(waveform, demean)
        for name, waveform in components.items()
    }
    model_input = prepared["clean_speech"] + prepared["mixture_noise"]
    diagnostic_waveforms = {**prepared, "model_input": model_input}
    diagnostics, valid_mask = _diagnose_all(diagnostic_waveforms, threshold)
    return PaperSupervision(
        supervision_type="clean_target",
        model_input=model_input,
        targets={
            "clean_speech": prepared["clean_speech"],
            "mixture_noise": prepared["mixture_noise"],
        },
        diagnostics=diagnostics,
        valid_mask=valid_mask,
        demeaned=bool(demean),
        source_ids={},
    )


__all__ = [
    "DEFAULT_SILENCE_RMS_THRESHOLD",
    "PaperSupervision",
    "SegmentDiagnostics",
    "build_clean_target_supervision",
    "build_noisy_target_supervision",
]
