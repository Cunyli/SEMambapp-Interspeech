"""Attenuation-only common-direction fusion for real six-component training."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
import torch

from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients

DIRECTION_MARGIN = 1e-6
MINIMUM_RETAINED_FRACTION = 1e-8


def fuse_training_gradients(component_order, gradients, base_weights):
    """Keep the legacy sum if valid; otherwise retain maximal safe norm mass.

    The six-variable linear program may attenuate any conflicting contribution.
    It cannot amplify a component, change targets, use speaker identities, or
    relax the nonnegative direction and 0.80 share requirements.
    """
    order = tuple(component_order)
    legacy_joint, legacy = fuse_tensor_gradients(order, gradients, base_weights)
    if legacy["fusion_authorized"]:
        return legacy_joint, dict(legacy, training_fusion_schema="attenuation-common-direction-v2",
                                  common_direction_repair_applied=False)
    vectors = torch.stack([gradients[n].reshape(-1).double().cpu() for n in order])
    norms = torch.linalg.vector_norm(vectors, dim=1)
    unit = vectors / norms[:, None]
    cosine = (unit @ unit.T).numpy()
    weights = np.array([legacy["effective_weights"][n] for n in order])
    mass = weights * norms.numpy()
    mass_sum = float(mass.sum())
    ceiling = mass / mass_sum
    count = len(order)
    # Variables are retained weighted gradient norms, normalized for numerical
    # conditioning. Every constraint applies to every current component.
    constraints = np.concatenate((
        -cosine + DIRECTION_MARGIN * np.ones((count, count)),
        np.eye(count) - 0.8 * np.ones((count, count)),
    ))
    solution = linprog(
        -np.ones(count), A_ub=constraints, b_ub=np.zeros(2 * count),
        bounds=[(MINIMUM_RETAINED_FRACTION * v, v) for v in ceiling],
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not solution.success:
        raise ValueError("no bounded nonzero common six-component direction: " + solution.message)
    effective = weights * (solution.x / ceiling)
    if np.any(effective <= 0) or np.any(effective > weights * (1 + 1e-12)):
        raise ValueError("common-direction solver violated attenuation bounds")
    # Remove solver endpoint roundoff without weakening any direction check.
    effective = np.minimum(effective, weights)
    denominator = legacy["normalization_denominator"]
    joint = (vectors * torch.from_numpy(effective)[:, None]).sum(dim=0) / denominator
    norm = float(torch.linalg.vector_norm(joint))
    if not torch.isfinite(joint).all() or norm <= 1e-10:
        raise ValueError("common-direction joint is zero or nonfinite")
    cosines = (unit @ joint / norm).numpy()
    retained_norms = effective * norms.numpy()
    shares = retained_norms / retained_norms.sum()
    if np.any(cosines < 0) or float(shares.max()) > 0.8:
        raise ValueError("actual tensors failed nonnegative direction or share cap")
    attenuation = {n: float(effective[i] / base_weights[n]) for i, n in enumerate(order)}
    changed = [n for n, value in attenuation.items() if value < 1]
    metadata = dict(
        legacy,
        training_fusion_schema="attenuation-common-direction-v2",
        common_direction_repair_applied=True,
        legacy_dominance_fusion=legacy,
        effective_weights=dict(zip(order, effective.tolist())),
        effective_normalized_component_coefficients=dict(zip(order, (effective / denominator).tolist())),
        attenuation_factors=attenuation,
        post_cap_weighted_gradient_norms=dict(zip(order, retained_norms.tolist())),
        post_cap_weighted_norm_shares=dict(zip(order, shares.tolist())),
        post_cap_maximum_share=float(shares.max()),
        component_to_joint_cosines=dict(zip(order, cosines.tolist())),
        joint_gradient_norm=norm,
        returned_tensor_gradient_norm=norm,
        no_component_amplified=True,
        only_dominant_component_attenuated=changed in ([], [legacy["dominant_component"]]),
        fusion_authorized=True, direction_conflict_detected=False, direction_conflict_components=[],
        conflict_policy="maximize_retained_norm_mass_with_all_six_nonnegative_directions",
        cap_policy="any_component_may_be_attenuated_under_common_direction_constraints",
        direction_margin=DIRECTION_MARGIN,
        retained_weighted_norm_fraction=float(retained_norms.sum() / mass_sum),
        solver_status=int(solution.status),
    )
    result = joint.reshape(gradients[order[0]].shape).to(gradients[order[0]])
    actual_norm = float(result.double().norm())
    actual_cosines = unit @ result.double().cpu().reshape(-1) / actual_norm
    if not torch.isfinite(result).all() or bool((actual_cosines < 0).any()):
        raise ValueError("returned dtype violates common direction")
    return result, metadata
