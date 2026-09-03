"""Deterministic dominance-capped Route C six-gradient fusion.

The fusion rule keeps calibration-frozen component weights, attenuates only a
single component whose weighted gradient norm exceeds the existing 0.80 share
gate, and reports any direction conflict for fail-closed abstention.  It has no
learned parameters and never inspects exact candidate outcomes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


FUSION_SCHEMA_VERSION = "avqi-route-c-six-gradient-dominance-capped-fusion-v1"
MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE = 0.80
INTERNAL_CAP_TARGET = math.nextafter(
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
    0.0,
)
CAP_POLICY = (
    "attenuate_only_the_unique_dominant_weighted_gradient_to_the_existing_"
    "maximum_norm_share"
)
CONFLICT_POLICY = (
    "fail_closed_abstain_if_any_component_to_post_cap_joint_cosine_is_negative"
)
JOINT_NORMALIZATION = (
    "divide_by_sum_of_calibration_frozen_base_component_weights"
)


def _validated_component_order(component_order: Sequence[str]) -> tuple[str, ...]:
    parsed = tuple(component_order)
    if len(parsed) < 2 or any(not name for name in parsed):
        raise ValueError("fusion component order is invalid")
    if len(set(parsed)) != len(parsed):
        raise ValueError("fusion component order contains duplicates")
    return parsed


def _positive_mapping(
    values: Mapping[str, float],
    component_order: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    if set(values) != set(component_order):
        raise ValueError(f"{label} keys differ from component order")
    parsed = {name: float(values[name]) for name in component_order}
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed.values()):
        raise ValueError(f"{label} values must be finite and positive")
    return parsed


def pairwise_component_keys(component_order: Sequence[str]) -> tuple[str, ...]:
    parsed = _validated_component_order(component_order)
    return tuple(
        f"{left}__{right}"
        for left_index, left in enumerate(parsed)
        for right in parsed[left_index + 1 :]
    )


def dominance_capped_effective_weights(
    component_order: Sequence[str],
    gradient_norms: Mapping[str, float],
    base_weights: Mapping[str, float],
    *,
    maximum_share: float = MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
) -> dict[str, Any]:
    """Return attenuation-only effective weights and pre/post-cap evidence."""
    order = _validated_component_order(component_order)
    norms = _positive_mapping(gradient_norms, order, "gradient norms")
    weights = _positive_mapping(base_weights, order, "base weights")
    cap = float(maximum_share)
    if not math.isfinite(cap) or cap < 0.5 or cap >= 1.0:
        raise ValueError("maximum component share must be in [0.5, 1.0)")

    pre_norms = {name: weights[name] * norms[name] for name in order}
    pre_total = math.fsum(pre_norms.values())
    pre_shares = {name: pre_norms[name] / pre_total for name in order}
    dominant = max(order, key=pre_shares.__getitem__)
    effective_weights = dict(weights)
    cap_applied = pre_shares[dominant] > cap

    if cap_applied:
        other_total = pre_total - pre_norms[dominant]
        if not math.isfinite(other_total) or other_total <= 0.0:
            raise ValueError("dominance cap has no positive non-dominant norm")
        internal_target = math.nextafter(cap, 0.0)
        allowed_dominant_norm = (
            internal_target / (1.0 - internal_target) * other_total
        )
        attenuation = allowed_dominant_norm / pre_norms[dominant]
        if not math.isfinite(attenuation) or not 0.0 < attenuation < 1.0:
            raise ValueError("dominance attenuation is invalid")
        effective_weights[dominant] *= attenuation

    attenuation_factors = {
        name: effective_weights[name] / weights[name] for name in order
    }
    normalization_denominator = math.fsum(weights.values())
    base_coefficients = {
        name: weights[name] / normalization_denominator for name in order
    }
    effective_coefficients = {
        name: effective_weights[name] / normalization_denominator for name in order
    }
    post_norms = {name: effective_weights[name] * norms[name] for name in order}
    post_total = math.fsum(post_norms.values())
    post_shares = {name: post_norms[name] / post_total for name in order}
    maximum_post_share = max(post_shares.values())
    if maximum_post_share > cap:
        raise RuntimeError("dominance cap exceeded its frozen maximum share")
    if any(value > 1.0 for value in attenuation_factors.values()):
        raise RuntimeError("dominance cap amplified a component")
    if any(
        effective_coefficients[name] > base_coefficients[name]
        for name in order
    ):
        raise RuntimeError("dominance cap amplified a normalized component")
    attenuated = [name for name in order if attenuation_factors[name] < 1.0]
    if attenuated not in ([], [dominant]):
        raise RuntimeError("dominance cap attenuated non-dominant components")

    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "maximum_weighted_component_norm_share": cap,
        "internal_cap_target": math.nextafter(cap, 0.0),
        "cap_policy": CAP_POLICY,
        "base_weights": weights,
        "effective_weights": effective_weights,
        "normalization_denominator": normalization_denominator,
        "base_normalized_component_coefficients": base_coefficients,
        "effective_normalized_component_coefficients": effective_coefficients,
        "attenuation_factors": attenuation_factors,
        "pre_cap_weighted_gradient_norms": pre_norms,
        "pre_cap_weighted_norm_shares": pre_shares,
        "pre_cap_maximum_share": max(pre_shares.values()),
        "post_cap_weighted_gradient_norms": post_norms,
        "post_cap_weighted_norm_shares": post_shares,
        "post_cap_maximum_share": maximum_post_share,
        "dominant_component": dominant,
        "cap_applied": cap_applied,
        "no_component_amplified": True,
        "only_dominant_component_attenuated": True,
    }


def _validated_pairwise_cosines(
    component_order: tuple[str, ...],
    values: Mapping[str, float],
) -> dict[str, float]:
    expected = pairwise_component_keys(component_order)
    if set(values) != set(expected):
        raise ValueError("pairwise cosine keys differ from component order")
    parsed = {key: float(values[key]) for key in expected}
    if any(
        not math.isfinite(value) or value < -1.0 or value > 1.0
        for value in parsed.values()
    ):
        raise ValueError("pairwise cosines must be finite values in [-1, 1]")
    return parsed


def fusion_from_gram(
    component_order: Sequence[str],
    gradient_norms: Mapping[str, float],
    pairwise_cosines: Mapping[str, float],
    base_weights: Mapping[str, float],
    *,
    maximum_share: float = MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
) -> dict[str, Any]:
    """Reconstruct capped-fusion metrics from norms and a cosine Gram matrix."""
    order = _validated_component_order(component_order)
    norms = _positive_mapping(gradient_norms, order, "gradient norms")
    cosines = _validated_pairwise_cosines(order, pairwise_cosines)
    cap = dominance_capped_effective_weights(
        order,
        norms,
        base_weights,
        maximum_share=maximum_share,
    )
    effective_weights = cap["effective_weights"]
    normalization_denominator = cap["normalization_denominator"]

    def gram(left: str, right: str) -> float:
        if left == right:
            return norms[left] * norms[left]
        left_index = order.index(left)
        right_index = order.index(right)
        key = (
            f"{left}__{right}"
            if left_index < right_index
            else f"{right}__{left}"
        )
        return norms[left] * norms[right] * cosines[key]

    diagonal = [
        effective_weights[name] ** 2 * gram(name, name) for name in order
    ]
    cross = [
        2.0 * effective_weights[left] * effective_weights[right] * gram(left, right)
        for left_index, left in enumerate(order)
        for right in order[left_index + 1 :]
    ]
    unnormalized_norm_squared = math.fsum((*diagonal, *cross))
    numerical_floor = -1e-12 * max(math.fsum(diagonal), 1.0)
    if unnormalized_norm_squared < numerical_floor:
        raise ValueError("pairwise cosine matrix yields a negative joint norm")
    unnormalized_norm_squared = max(0.0, unnormalized_norm_squared)
    joint_norm = math.sqrt(unnormalized_norm_squared) / normalization_denominator
    if not math.isfinite(joint_norm) or joint_norm <= 0.0:
        raise ValueError("capped joint gradient is non-finite or zero")

    component_to_joint_cosines: dict[str, float] = {}
    for component in order:
        dot = math.fsum(
            effective_weights[other] * gram(component, other)
            for other in order
        ) / normalization_denominator
        cosine = dot / (norms[component] * joint_norm)
        if abs(cosine) <= 1.0 + 1e-12:
            cosine = min(1.0, max(-1.0, cosine))
        if not math.isfinite(cosine) or cosine < -1.0 or cosine > 1.0:
            raise ValueError("component-to-joint cosine is invalid")
        component_to_joint_cosines[component] = cosine

    conflicts = [
        name for name in order if component_to_joint_cosines[name] < 0.0
    ]
    return {
        **cap,
        "joint_normalization": JOINT_NORMALIZATION,
        "joint_gradient_norm": joint_norm,
        "pairwise_component_cosines": cosines,
        "component_to_joint_cosines": component_to_joint_cosines,
        "conflict_policy": CONFLICT_POLICY,
        "direction_conflict_components": conflicts,
        "direction_conflict_detected": bool(conflicts),
        "fusion_authorized": not conflicts,
    }


def fuse_tensor_gradients(
    component_order: Sequence[str],
    gradients: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, float],
    *,
    maximum_share: float = MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fuse real gradient tensors and return the joint plus auditable metadata."""
    order = _validated_component_order(component_order)
    if set(gradients) != set(order):
        raise ValueError("gradient tensor keys differ from component order")
    first = gradients[order[0]]
    if not torch.is_floating_point(first):
        raise ValueError("gradient tensors must be floating point")
    shape = first.shape
    device = first.device
    dtype = first.dtype
    canonical: dict[str, torch.Tensor] = {}
    for name in order:
        gradient = gradients[name]
        if (
            gradient.shape != shape
            or gradient.device != device
            or gradient.dtype != dtype
            or not torch.is_floating_point(gradient)
        ):
            raise ValueError("gradient tensors must share shape, device, and type")
        converted = gradient.to(dtype=torch.float64)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError("gradient tensor is non-finite")
        canonical[name] = converted

    norms = {
        name: float(torch.linalg.vector_norm(canonical[name])) for name in order
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise ValueError("gradient tensor norms must be finite and positive")
    pairwise = {}
    for left_index, left in enumerate(order):
        for right in order[left_index + 1 :]:
            denominator = norms[left] * norms[right]
            pairwise[f"{left}__{right}"] = float(
                torch.dot(canonical[left].reshape(-1), canonical[right].reshape(-1))
            ) / denominator
    metadata = fusion_from_gram(
        order,
        norms,
        pairwise,
        base_weights,
        maximum_share=maximum_share,
    )
    effective_weights = metadata["effective_weights"]
    joint_high_precision = torch.zeros_like(canonical[order[0]])
    for name in order:
        joint_high_precision = (
            joint_high_precision + canonical[name] * effective_weights[name]
        )
    joint_high_precision = (
        joint_high_precision / metadata["normalization_denominator"]
    )
    actual_norm = float(torch.linalg.vector_norm(joint_high_precision))
    if not bool(torch.isfinite(joint_high_precision).all()) or actual_norm <= 0.0:
        raise ValueError("capped tensor joint gradient is non-finite or zero")
    if not math.isclose(
        actual_norm,
        metadata["joint_gradient_norm"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError("tensor fusion differs from Gram reconstruction")
    joint = joint_high_precision.to(dtype=dtype)
    returned_norm = float(torch.linalg.vector_norm(joint))
    if (
        not bool(torch.isfinite(joint).all())
        or returned_norm <= 0.0
        or not math.isclose(
            returned_norm,
            metadata["joint_gradient_norm"],
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
    ):
        raise RuntimeError("returned tensor fusion differs after dtype restoration")
    metadata["returned_tensor_dtype"] = str(dtype)
    metadata["returned_tensor_gradient_norm"] = returned_norm
    return joint, metadata


def require_conflict_free(metadata: Mapping[str, Any]) -> None:
    """Enforce the pre-registered fail-closed direction-conflict policy."""
    conflicts = metadata.get("direction_conflict_components")
    if not isinstance(conflicts, list):
        raise ValueError("fusion conflict evidence is unavailable")
    if conflicts:
        raise ValueError(
            "capped fusion abstained on direction conflict: "
            + ", ".join(str(name) for name in conflicts)
        )
