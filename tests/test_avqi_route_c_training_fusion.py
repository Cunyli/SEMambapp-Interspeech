import pytest
import torch

from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients
from model.avqi_route_c_training_fusion import fuse_training_gradients


def test_conflict_repair_protects_small_near_target_gradient_without_amplification():
    gradients = {"near_target": torch.tensor([0.005, 0.0], dtype=torch.float64),
                 "other": torch.tensor([-0.3, 1.0], dtype=torch.float64)}
    weights = {n: 1.0 for n in gradients}
    _, old = fuse_tensor_gradients(tuple(gradients), gradients, weights)
    assert not old["fusion_authorized"]
    joint, report = fuse_training_gradients(tuple(gradients), gradients, weights)
    assert report["common_direction_repair_applied"]
    for name, grad in gradients.items():
        assert torch.dot(grad, joint) >= 0
        assert 0 < report["effective_weights"][name] <= weights[name]
    assert report["post_cap_maximum_share"] <= 0.8
    assert not report["legacy_dominance_fusion"]["fusion_authorized"]


def test_valid_joint_is_unchanged():
    gradients = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.2, 0.3])}
    weights = {"a": 0.3, "b": 1.0}
    expected, _ = fuse_tensor_gradients(tuple(gradients), gradients, weights)
    actual, report = fuse_training_gradients(tuple(gradients), gradients, weights)
    assert torch.equal(actual, expected)
    assert not report["common_direction_repair_applied"]


def test_opposed_gradients_fail_without_faking_common_descent():
    gradients = {"a": torch.tensor([1.0, 0.0], dtype=torch.float64),
                 "b": torch.tensor([-0.5, 0.0], dtype=torch.float64)}
    with pytest.raises(ValueError, match="no bounded nonzero"):
        fuse_training_gradients(tuple(gradients), gradients, {"a": 1.0, "b": 1.0})


def test_common_direction_scales_with_waveform_gradient_units():
    gradients = {"a": torch.tensor([0.005, 0.0], dtype=torch.float64),
                 "b": torch.tensor([-0.3, 1.0], dtype=torch.float64)}
    first, _ = fuse_training_gradients(tuple(gradients), gradients, {"a": 1.0, "b": 1.0})
    second, _ = fuse_training_gradients(tuple(gradients), {k:v*100 for k,v in gradients.items()}, {"a": 1.0, "b": 1.0})
    torch.testing.assert_close(second, first*100, rtol=1e-9, atol=1e-12)
