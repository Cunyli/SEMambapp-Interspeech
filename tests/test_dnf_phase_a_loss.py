from __future__ import annotations

import math

import torch

from model.dnf_phase_a import active_frame_log_rms_loss


def test_identity_has_zero_loss() -> None:
    clean = torch.full((2, 640), 0.2)
    estimate = clean.clone().requires_grad_(True)

    result = active_frame_log_rms_loss(estimate, clean)

    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    assert result.active_sample_mask.tolist() == [True, True]


def test_half_scale_is_absolute_log_half() -> None:
    clean = torch.full((1, 640), 0.2)
    estimate = (clean * 0.5).requires_grad_(True)

    result = active_frame_log_rms_loss(estimate, clean)

    torch.testing.assert_close(
        result.loss,
        torch.tensor(abs(math.log(0.5))),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_inactive_reference_frames_do_not_contribute() -> None:
    clean = torch.cat((torch.ones(1, 320), torch.zeros(1, 320)), dim=-1)
    estimate = clean.clone()
    estimate[:, 480:] = 100.0
    estimate.requires_grad_(True)

    result = active_frame_log_rms_loss(estimate, clean)

    assert result.active_frame_mask.tolist() == [[True, True, False]]
    torch.testing.assert_close(result.loss, torch.tensor(0.0))


def test_empty_sample_mask_is_explicit() -> None:
    clean = torch.stack((torch.full((640,), 0.2), torch.zeros(640)))
    estimate = clean.clone().requires_grad_(True)

    result = active_frame_log_rms_loss(estimate, clean)

    assert result.active_sample_mask.tolist() == [True, False]
    assert result.empty_sample_mask.tolist() == [False, True]
    assert result.active_frame_count.tolist() == [3, 0]
    torch.testing.assert_close(result.per_sample_loss[1], torch.tensor(0.0))


def test_gradient_is_finite() -> None:
    generator = torch.Generator().manual_seed(1234)
    clean = torch.randn((3, 960), generator=generator) * 0.1
    estimate = (clean + 0.01 * torch.randn((3, 960), generator=generator)).requires_grad_(
        True
    )

    result = active_frame_log_rms_loss(estimate, clean)
    result.loss.backward()

    assert estimate.grad is not None
    assert torch.isfinite(result.loss)
    assert torch.isfinite(estimate.grad).all()


def test_empty_route_microbatch_is_a_differentiable_zero() -> None:
    estimate = torch.empty((0, 640), requires_grad=True)
    reference = torch.empty((0, 640))
    result = active_frame_log_rms_loss(estimate, reference)
    assert result.per_sample_loss.shape == (0,)
    assert result.active_sample_mask.shape == (0,)
    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    result.loss.backward()
    assert estimate.grad is not None
