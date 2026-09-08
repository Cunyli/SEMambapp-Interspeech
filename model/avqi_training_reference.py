"""Compare output and clean speech in the same frozen differentiable coordinates."""
from __future__ import annotations

import math

import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c import RouteCFiveActiveScorer


def five_component_clean_prediction(scorer, waveform, view):
    """Exclude the legacy Shimmer dB slot; Candidate-E uses its Exact target."""
    with torch.no_grad():
        values = scorer.denormalized_prediction(RouteCFiveActiveScorer.forward(
            scorer, waveform.to(scorer.target_mean.device)[None], view))[0].cpu().tolist()
    return {name: value if name != "shimmer_db" else None
            for name, value in zip(AVQI_COMPONENT_NAMES, values)}


def training_row_with_same_formula_reference(row, clean_prediction):
    """Return a training-only loss view; never mutate the Exact target bank."""
    if row["training_role"] != "train":
        raise ValueError("same-formula references may be constructed for train rows only")
    if set(clean_prediction) != set(AVQI_COMPONENT_NAMES) or clean_prediction["shimmer_db"] is not None:
        raise ValueError("five clean proxy values and an unmeasured Candidate-E identity slot required")
    if any(not math.isfinite(clean_prediction[n]) for n in AVQI_COMPONENT_NAMES if n != "shimmer_db"):
        raise ValueError("nonfinite clean proxy reference")
    targets = {name: (row["target_components"][name] if name == "shimmer_db" else clean_prediction[name])
               for name in AVQI_COMPONENT_NAMES}
    return dict(row, target_components=targets,
                exact_target_components=dict(row["target_components"]),
                gradient_target_source="same_formula_clean_five_plus_exact_candidate_e_db")
