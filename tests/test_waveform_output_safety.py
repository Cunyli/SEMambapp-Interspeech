import pytest
import torch

from model.encdec import LearnableSoftplus
from model.waveform_output_safety import attenuate_output_peak
from scripts import evaluate_avqi_component_backprop as helper


def test_peak_attenuation_preserves_shape_and_batch_independence():
    x = torch.tensor([[0.0, 0.2, -0.7, 0.1], [2.0, -1.0, 0.25, 0.0]])
    y, gain = attenuate_output_peak(x, 0.95)
    assert torch.equal(y[0], x[0])
    assert torch.all(y.abs().amax(dim=-1) <= 0.95)
    assert torch.all((gain > 0) & (gain <= 1))
    torch.testing.assert_close(y, x * gain)
    torch.testing.assert_close(torch.fft.rfft(y), torch.fft.rfft(x) * gain)


def test_silence_and_safe_waveform_are_exact_identity():
    for x in (torch.zeros(20), torch.linspace(-0.94, 0.94, 20)):
        x.requires_grad_()
        y, gain = attenuate_output_peak(x, 0.95)
        assert torch.equal(x, y)
        assert torch.equal(gain, torch.ones_like(gain))
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))


def test_peak_gain_remains_in_autograd_graph():
    x = torch.tensor([2.0, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: attenuate_output_peak(value, 0.95)[0], (x,))
    y, _ = attenuate_output_peak(x, 0.95)
    grad = torch.autograd.grad(y[1], x)[0]
    assert torch.isfinite(grad).all() and grad[0] != 0 and grad[1] != 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, 1.0, -1.0])
def test_invalid_ceiling_rejected(value):
    with pytest.raises(ValueError, match="peak limit"):
        attenuate_output_peak(torch.zeros(5), value)


@pytest.mark.parametrize("x", [torch.tensor([]), torch.tensor([float("nan")]),
                              torch.tensor([float("inf")]), torch.tensor([1, 2])])
def test_invalid_audio_rejected(x):
    with pytest.raises(ValueError, match="waveform"):
        attenuate_output_peak(x, 0.95)


def test_softplus_matches_original_activation_and_beta_gradients():
    layer = LearnableSoftplus(3).double()
    layer.beta.data.copy_(torch.tensor([-0.5, 0.0, 0.5]))
    x = torch.linspace(-5, 5, 21, dtype=torch.float64).reshape(1, 3, 7).requires_grad_()
    beta = layer.beta.exp().reshape(1, -1, 1)
    old = (1 / beta + 1e-6) * torch.log1p(torch.exp(beta * x))
    new = layer(x)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(new.sum(), (x, layer.beta), retain_graph=True)
    b = torch.autograd.grad(old.sum(), (x, layer.beta))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_softplus_extreme_magnitudes_have_finite_forward_and_backward():
    layer = LearnableSoftplus(1)
    x = torch.tensor([[[-1000.0, 0.0, 1000.0]]], requires_grad=True)
    result = layer(x)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.beta.grad).all()
    assert x.grad[0, 0, 2] > 0


def test_shared_enhancement_opt_in_and_native_length():
    class Identity(torch.nn.Module):
        def forward(self, magnitude, phase):
            return magnitude, phase, None

    cfg = {"stft_cfg": {"n_fft": 400, "hop_size": 100, "win_size": 400},
           "model_cfg": {"compress_factor": 0.3}}
    torch.manual_seed(7)
    x = (torch.randn(16317) * 0.7).requires_grad_()
    legacy = helper.enhance_waveform(Identity(), x, cfg)
    cfg["signal_safety_cfg"] = {"output_peak_limit": 0.95}
    limited = helper.enhance_waveform(Identity(), x, cfg)
    expected, _ = attenuate_output_peak(legacy, 0.95)
    torch.testing.assert_close(limited, expected)
    assert limited.shape == (1, 16300)
    assert limited.abs().max() <= 0.95 < legacy.abs().max()
    limited.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
