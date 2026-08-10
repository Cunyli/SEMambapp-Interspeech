"""Loss helpers for the controlled scratch DNF Phase A experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ActiveFrameLogRMSLossResult:
    """Structured result for active-frame log-RMS preservation."""

    loss: Tensor
    per_sample_loss: Tensor
    active_sample_mask: Tensor
    empty_sample_mask: Tensor
    active_frame_mask: Tensor
    active_frame_count: Tensor
    estimate_frame_rms: Tensor
    reference_frame_rms: Tensor


def _ensure_batched_audio(audio: Tensor) -> Tensor:
    if audio.ndim == 2:
        return audio
    if audio.ndim == 3:
        return audio
    raise ValueError(
        f"Expected audio shaped [batch,time] or [batch,channel,time], got {audio.shape}"
    )


def _frame_audio(audio: Tensor, frame_length: int, hop_length: int) -> Tensor:
    if audio.shape[-1] < frame_length:
        audio = F.pad(audio, (0, frame_length - audio.shape[-1]))
    return audio.unfold(-1, frame_length, hop_length)


def _frame_rms(frames: Tensor) -> Tensor:
    reduction_dims = (-1,) if frames.ndim == 3 else (1, 3)
    return frames.square().mean(dim=reduction_dims).sqrt()


def active_frame_log_rms_loss(
    estimate: Tensor,
    clean_reference: Tensor,
    *,
    frame_length: int = 320,
    hop_length: int = 160,
    relative_activity_db: float = -40.0,
    absolute_activity_rms: float = 1.0e-5,
) -> ActiveFrameLogRMSLossResult:
    """Compare log RMS on frames active according to a detached clean reference."""

    estimate = _ensure_batched_audio(estimate)
    clean_reference = _ensure_batched_audio(clean_reference)
    if estimate.shape != clean_reference.shape:
        raise ValueError(
            f"Estimate/reference shape mismatch: {estimate.shape} != {clean_reference.shape}"
        )
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if absolute_activity_rms <= 0.0:
        raise ValueError("absolute_activity_rms must be positive")

    reference = clean_reference.detach()
    estimate_frames = _frame_audio(estimate, frame_length, hop_length)
    reference_frames = _frame_audio(reference, frame_length, hop_length)
    estimate_rms = _frame_rms(estimate_frames)
    reference_rms = _frame_rms(reference_frames).detach()

    relative_ratio = math.pow(10.0, relative_activity_db / 20.0)
    relative_threshold = reference_rms.amax(dim=-1, keepdim=True) * relative_ratio
    absolute_threshold = reference_rms.new_tensor(absolute_activity_rms)
    activity_threshold = torch.maximum(relative_threshold, absolute_threshold)
    active_frame_mask = (reference_rms >= activity_threshold).detach()

    log_floor = torch.finfo(estimate_rms.dtype).tiny
    frame_loss = (
        estimate_rms.clamp_min(log_floor).log()
        - reference_rms.clamp_min(log_floor).log()
    ).abs()
    active_weights = active_frame_mask.to(frame_loss.dtype)
    active_frame_count = active_frame_mask.sum(dim=-1)
    active_sample_mask = active_frame_count > 0
    empty_sample_mask = ~active_sample_mask
    per_sample_loss = (frame_loss * active_weights).sum(dim=-1) / active_frame_count.clamp_min(
        1
    ).to(frame_loss.dtype)

    if bool(active_sample_mask.any()):
        loss = per_sample_loss[active_sample_mask].mean()
    else:
        loss = estimate.sum() * 0.0

    return ActiveFrameLogRMSLossResult(
        loss=loss,
        per_sample_loss=per_sample_loss,
        active_sample_mask=active_sample_mask,
        empty_sample_mask=empty_sample_mask,
        active_frame_mask=active_frame_mask,
        active_frame_count=active_frame_count,
        estimate_frame_rms=estimate_rms,
        reference_frame_rms=reference_rms,
    )
