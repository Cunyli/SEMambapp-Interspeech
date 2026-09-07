"""Full-band, utterance-constant output attenuation for training and inference."""

import math

import torch


def attenuate_output_peak(
    waveform: torch.Tensor, peak_limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an unboosted waveform and its differentiable per-utterance gain.

    The last axis is time. A single gain preserves the relative envelope and
    spectral shape within each utterance; this is neither clipping nor a filter.
    Invalid input is an error, never converted into apparently valid silence.
    """
    if not math.isfinite(peak_limit) or not 0 < peak_limit < 1:
        raise ValueError("output peak limit must be finite and between zero and one")
    if (not torch.is_floating_point(waveform) or waveform.ndim < 1
            or waveform.numel() == 0 or not bool(torch.isfinite(waveform).all())):
        raise ValueError("output waveform must be nonempty, finite and floating point")
    peak = waveform.abs().amax(dim=-1, keepdim=True)
    # Reserve two rounding units on attenuated rows so multiplication cannot
    # push a boundary sample over the configured ceiling.
    ceiling = peak_limit * (1 - 2 * torch.finfo(waveform.dtype).eps)
    gain = torch.where(
        peak > peak_limit,
        ceiling / peak.clamp_min(peak_limit),
        torch.ones_like(peak),
    )
    return waveform * gain, gain
