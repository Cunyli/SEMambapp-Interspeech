from __future__ import annotations

import math

import pytest
import torch

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_gradient_fusion import (
    CAP_POLICY,
    CONFLICT_POLICY,
    FUSION_SCHEMA_VERSION,
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
    dominance_capped_effective_weights,
    fuse_tensor_gradients,
    fusion_from_gram,
    pairwise_component_keys,
    require_conflict_free,
)


COMPONENTS = tuple(ROUTE_C_SIX_ACTIVE_COMPONENTS)


def _orthogonal_gradients(scales: tuple[float, ...]) -> dict[str, torch.Tensor]:
    return {
        component: torch.eye(len(COMPONENTS), dtype=torch.float64)[index]
        * scales[index]
        for index, component in enumerate(COMPONENTS)
    }


def test_balanced_fusion_is_an_exact_no_op_on_base_weights() -> None:
    gradients = _orthogonal_gradients((1.0,) * len(COMPONENTS))
    weights = {component: 1.0 for component in COMPONENTS}

    joint, metadata = fuse_tensor_gradients(COMPONENTS, gradients, weights)

    assert metadata["schema_version"] == FUSION_SCHEMA_VERSION
    assert metadata["cap_policy"] == CAP_POLICY
    assert metadata["conflict_policy"] == CONFLICT_POLICY
    assert metadata["cap_applied"] is False
    assert metadata["effective_weights"] == weights
    assert metadata["post_cap_maximum_share"] == pytest.approx(1.0 / 6.0)
    assert metadata["direction_conflict_components"] == []
    assert torch.allclose(joint, torch.full_like(joint, 1.0 / 6.0))


def test_dominance_cap_attenuates_only_the_unique_dominant_component() -> None:
    gradients = _orthogonal_gradients((100.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    weights = {component: 1.0 for component in COMPONENTS}

    _, metadata = fuse_tensor_gradients(COMPONENTS, gradients, weights)

    dominant = COMPONENTS[0]
    assert metadata["cap_applied"] is True
    assert metadata["dominant_component"] == dominant
    assert metadata["attenuation_factors"][dominant] == pytest.approx(0.2)
    assert all(
        metadata["attenuation_factors"][component] == 1.0
        for component in COMPONENTS[1:]
    )
    assert metadata["post_cap_maximum_share"] <= (
        MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE
    )
    assert metadata["post_cap_maximum_share"] == pytest.approx(0.8)
    assert metadata["no_component_amplified"] is True
    assert metadata["only_dominant_component_attenuated"] is True
    assert all(
        metadata["effective_normalized_component_coefficients"][component]
        <= metadata["base_normalized_component_coefficients"][component]
        for component in COMPONENTS
    )


def test_tensor_fusion_restores_the_input_dtype() -> None:
    gradients = {
        component: gradient.to(dtype=torch.float32)
        for component, gradient in _orthogonal_gradients(
            (100.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        ).items()
    }
    weights = {component: 1.0 for component in COMPONENTS}

    joint, metadata = fuse_tensor_gradients(COMPONENTS, gradients, weights)

    assert joint.dtype == torch.float32
    assert metadata["returned_tensor_dtype"] == "torch.float32"


def test_tensor_fusion_rejects_zero_norm_or_mixed_dtype() -> None:
    gradients = _orthogonal_gradients((0.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="norms must be finite and positive"):
        fuse_tensor_gradients(
            COMPONENTS,
            gradients,
            {component: 1.0 for component in COMPONENTS},
        )

    gradients = _orthogonal_gradients((1.0,) * len(COMPONENTS))
    gradients[COMPONENTS[-1]] = gradients[COMPONENTS[-1]].float()
    with pytest.raises(ValueError, match="share shape, device, and type"):
        fuse_tensor_gradients(
            COMPONENTS,
            gradients,
            {component: 1.0 for component in COMPONENTS},
        )


def test_gram_reconstruction_matches_real_tensor_fusion() -> None:
    generator = torch.Generator().manual_seed(20260903)
    gradients = {
        component: torch.randn(64, generator=generator, dtype=torch.float64)
        * (index + 1)
        for index, component in enumerate(COMPONENTS)
    }
    weights = {
        component: 1.0 / (index + 1)
        for index, component in enumerate(COMPONENTS)
    }
    joint, tensor_metadata = fuse_tensor_gradients(COMPONENTS, gradients, weights)
    norms = {
        component: float(torch.linalg.vector_norm(gradient))
        for component, gradient in gradients.items()
    }
    pairwise = {}
    for left_index, left in enumerate(COMPONENTS):
        for right in COMPONENTS[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = float(
                torch.dot(gradients[left], gradients[right])
                / (norms[left] * norms[right])
            )

    gram_metadata = fusion_from_gram(COMPONENTS, norms, pairwise, weights)

    assert torch.isfinite(joint).all()
    assert gram_metadata["joint_gradient_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(joint)),
        rel=1e-10,
        abs=1e-12,
    )
    assert gram_metadata["effective_weights"] == pytest.approx(
        tensor_metadata["effective_weights"]
    )
    assert gram_metadata["component_to_joint_cosines"] == pytest.approx(
        tensor_metadata["component_to_joint_cosines"]
    )


def test_direction_conflict_is_reported_and_fails_closed() -> None:
    positive = torch.tensor([1.0, 0.0], dtype=torch.float64)
    gradients = {
        COMPONENTS[0]: -positive,
        **{component: positive.clone() for component in COMPONENTS[1:]},
    }
    weights = {component: 1.0 for component in COMPONENTS}

    _, metadata = fuse_tensor_gradients(COMPONENTS, gradients, weights)

    assert metadata["direction_conflict_components"] == [COMPONENTS[0]]
    assert metadata["direction_conflict_detected"] is True
    assert metadata["fusion_authorized"] is False
    with pytest.raises(ValueError, match="abstained on direction conflict"):
        require_conflict_free(metadata)


@pytest.mark.parametrize(
    ("norms", "weights", "message"),
    [
        ({component: 1.0 for component in COMPONENTS[:-1]}, None, "keys differ"),
        (
            {
                component: (0.0 if index == 0 else 1.0)
                for index, component in enumerate(COMPONENTS)
            },
            None,
            "finite and positive",
        ),
        (
            {component: 1.0 for component in COMPONENTS},
            {component: math.inf for component in COMPONENTS},
            "finite and positive",
        ),
    ],
)
def test_invalid_norms_and_weights_fail_closed(
    norms: dict[str, float],
    weights: dict[str, float] | None,
    message: str,
) -> None:
    actual_weights = weights or {component: 1.0 for component in COMPONENTS}
    with pytest.raises(ValueError, match=message):
        dominance_capped_effective_weights(COMPONENTS, norms, actual_weights)


def test_pairwise_key_contract_is_exact() -> None:
    expected = 15
    assert len(pairwise_component_keys(COMPONENTS)) == expected
    assert len(set(pairwise_component_keys(COMPONENTS))) == expected
