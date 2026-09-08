"""Bound auxiliary parameter gradients without changing component directions."""
from __future__ import annotations

import math

import torch


def bound_auxiliary_parameter_gradients(fidelity, auxiliary, maximum_norm_ratio):
    """Keep at least (1-ratio) of fidelity's directional derivative.

    The bound concerns raw parameter gradients. AdamW's moments and finite
    updates still need empirical before/after evaluation.
    """
    if not 0 < maximum_norm_ratio < 1 or not fidelity or len(fidelity) != len(auxiliary):
        raise ValueError("a nonempty matching gradient pair and ratio in (0, 1) are required")
    for primary, extra in zip(fidelity, auxiliary):
        if (primary.shape != extra.shape or primary.dtype != extra.dtype
                or primary.device != extra.device or not torch.isfinite(primary).all()
                or not torch.isfinite(extra).all()):
            raise ValueError("parameter gradient shape, dtype, device or finiteness differs")
    fidelity_sq = sum(float(g.detach().double().square().sum()) for g in fidelity)
    auxiliary_sq = sum(float(g.detach().double().square().sum()) for g in auxiliary)
    if fidelity_sq <= 0 or auxiliary_sq <= 0:
        raise ValueError("both objectives require nonzero parameter gradients")
    raw_ratio = math.sqrt(auxiliary_sq / fidelity_sq)
    scale = min(1.0, maximum_norm_ratio / raw_ratio)
    combined = tuple((primary + scale * extra).detach()
                     for primary, extra in zip(fidelity, auxiliary))
    dot = sum(float(p.detach().double().mul(g.double()).sum()) for p, g in zip(fidelity, combined))
    combined_sq = sum(float(g.double().square().sum()) for g in combined)
    minimum_dot = (1 - maximum_norm_ratio) * fidelity_sq
    if (not all(torch.isfinite(g).all() for g in combined) or combined_sq <= 0
            or dot < minimum_dot * (1 - 1e-6)):
        raise ValueError("actual assigned gradients violate the fidelity direction bound")
    return combined, dict(
        scope="parameter_gradient_before_clipping_and_AdamW",
        maximum_auxiliary_to_fidelity_norm_ratio=maximum_norm_ratio,
        fidelity_parameter_gradient_norm=math.sqrt(fidelity_sq),
        raw_auxiliary_parameter_gradient_norm=math.sqrt(auxiliary_sq),
        raw_auxiliary_to_fidelity_norm_ratio=raw_ratio,
        auxiliary_scale=scale,
        applied_auxiliary_to_fidelity_norm_ratio=raw_ratio * scale,
        fidelity_to_combined_cosine=dot / math.sqrt(fidelity_sq * combined_sq),
        fidelity_directional_derivative_retained=dot / fidelity_sq,
        all_six_share_one_positive_attenuation=True,
        component_formulas_or_internal_weights_changed=False,
        actual_AdamW_step_descent_guaranteed=False,
    )
