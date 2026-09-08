import pytest
import torch

from model.fidelity_gradient_budget import bound_auxiliary_parameter_gradients


def test_opposing_auxiliary_update_cannot_reverse_raw_fidelity_direction():
    primary = (torch.tensor([1.0, 0.0]),)
    auxiliary = (torch.tensor([-100.0, 0.0]),)
    combined, report = bound_auxiliary_parameter_gradients(primary, auxiliary, 0.5)
    torch.testing.assert_close(combined[0], torch.tensor([0.5, 0.0]))
    assert report["auxiliary_scale"] > 0
    assert report["fidelity_directional_derivative_retained"] == pytest.approx(0.5)


def test_unbounded_case_matches_joint_autograd_chain_rule():
    weight = torch.tensor([[0.2, -0.7], [0.5, 0.3]], requires_grad=True)
    y = weight @ torch.tensor([0.4, -0.6])
    fidelity = y.square().mean()
    extra = torch.tensor([0.001, -0.002])
    expected = torch.autograd.grad((fidelity, y), weight,
        grad_outputs=(torch.ones_like(fidelity), extra), retain_graph=True)[0]
    primary = torch.autograd.grad(fidelity, weight, retain_graph=True)
    auxiliary = torch.autograd.grad(y, weight, grad_outputs=extra)
    combined, report = bound_auxiliary_parameter_gradients(primary, auxiliary, 0.5)
    assert report["auxiliary_scale"] == 1
    torch.testing.assert_close(combined[0], expected)


def test_global_budget_spans_parameter_tensors_and_keeps_auxiliary_direction():
    rng = torch.Generator().manual_seed(31)
    primary = tuple(torch.randn(shape, generator=rng) for shape in ((17, 9), (3,)))
    auxiliary = tuple(30 * torch.randn(shape, generator=rng) for shape in ((17, 9), (3,)))
    combined, report = bound_auxiliary_parameter_gradients(primary, auxiliary, 0.5)
    p = torch.cat([x.flatten().double() for x in primary])
    a = torch.cat([x.flatten().double() for x in auxiliary])
    c = torch.cat([x.flatten().double() for x in combined])
    torch.testing.assert_close(c - p, report["auxiliary_scale"] * a, rtol=1e-5, atol=1e-7)
    assert (c - p).norm() <= 0.500001 * p.norm()
    assert torch.dot(c, p) >= 0.499999 * p.square().sum()


@pytest.mark.parametrize("ratio", [0.0, 1.0, float("nan")])
def test_invalid_budgets_are_rejected(ratio):
    with pytest.raises(ValueError):
        bound_auxiliary_parameter_gradients((torch.ones(2),), (torch.ones(2),), ratio)


@pytest.mark.parametrize("bad", [torch.zeros(2), torch.tensor([float("nan"), 0.0])])
def test_zero_or_nonfinite_objective_is_not_silently_disabled(bad):
    with pytest.raises(ValueError):
        bound_auxiliary_parameter_gradients((torch.ones(2),), (bad,), 0.5)
