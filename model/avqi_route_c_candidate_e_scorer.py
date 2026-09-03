"""Versioned six-slot Route C scorer with promoted Candidate-E Shimmer dB.

The legacy v19 six-slot scaffold remains unchanged so its historical audits
stay reproducible.  This module is the explicit Candidate-E execution path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from model.avqi_route_c import (
    ROUTE_C_COMPONENT_REGISTRY,
    ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES,
    ROUTE_C_SIX_SOURCE_ARCHITECTURES,
    ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS,
    ROUTE_C_SIX_SOURCE_COMPONENT_INDICES,
    RouteCFiveActiveScorer,
    RouteCSixActiveScorer,
    _compose_route_c_scorer,
    build_route_c_six_active_estimator,
)
from model.avqi_route_c_candidate_e import (
    CANDIDATE_E_TOPOLOGY_IMPLEMENTATION,
    candidate_e_proxy,
)
from model.avqi_route_c_v19_contracts import (
    validate_v19_exact_topology,
)


ROUTE_C_CANDIDATE_E_SIX_ACTIVE_ARCHITECTURE = (
    "direct_praat_hard_cpps_v12_hnr_v7_shimmer_percent_v6_"
    "shimmer_db_candidate_e_v32r8_exact_path_ltas_slope_tilt"
)
ROUTE_C_CANDIDATE_E_REGISTRY_SCHEMA_VERSION = (
    "avqi-route-c-six-component-candidate-e-registry-v1"
)
ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS = "fresh_speaker_panel_pass"


@dataclass(frozen=True)
class RouteCCandidateESixScorerBundle:
    scorer: "RouteCCandidateESixScorer"
    source_metadata: dict[str, Any]
    scientific_status: str


@dataclass(frozen=True)
class RouteCCandidateEComponentSlot:
    """One promoted Candidate-E six-component slot."""

    index: int
    name: str
    avqi_coefficient: float
    expanded_avqi_coefficient: float
    implementation: str
    code_status: str
    scientific_status: str
    active_in_six_component_scorer: bool


ROUTE_C_CANDIDATE_E_COMPONENT_REGISTRY = tuple(
    RouteCCandidateEComponentSlot(
        index=slot.index,
        name=slot.name,
        avqi_coefficient=slot.avqi_coefficient,
        expanded_avqi_coefficient=slot.expanded_avqi_coefficient,
        implementation=(
            "candidate_e_v32r8_exact_path_fixed_order_projection"
            if slot.name == "shimmer_db"
            else slot.implementation
        ),
        code_status="integrated" if slot.name == "shimmer_db" else slot.code_status,
        scientific_status=(
            ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS
            if slot.name == "shimmer_db"
            else slot.scientific_status
        ),
        active_in_six_component_scorer=True,
    )
    for slot in ROUTE_C_COMPONENT_REGISTRY
)


def route_c_candidate_e_registry_records() -> list[dict[str, Any]]:
    """Return the promoted Candidate-E six-slot registry."""
    return [asdict(slot) for slot in ROUTE_C_CANDIDATE_E_COMPONENT_REGISTRY]


class RouteCCandidateESixScorer(RouteCSixActiveScorer):
    """Six-slot scorer whose Shimmer dB slot follows Candidate-E exactly."""

    def forward(
        self,
        waveform: torch.Tensor,
        speaking_type: str,
        *,
        topology: Mapping[str, Any] | None = None,
        case_id: str | None = None,
        view: str | None = None,
        topology_sha256: str | None = None,
    ) -> torch.Tensor:
        if topology is None:
            raise ValueError("Candidate-E scorer requires exact topology")
        if case_id is None or view is None or topology_sha256 is None:
            raise ValueError("Candidate-E topology binding is incomplete")
        if speaking_type != view:
            raise ValueError("Candidate-E speaking type and view differ")
        if waveform.ndim == 1:
            current_waveform = waveform
            scorer_input = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            current_waveform = waveform[0]
            scorer_input = waveform
        else:
            raise ValueError("Candidate-E scorer requires one current waveform")

        validated = validate_v19_exact_topology(
            current_waveform,
            topology,
            case_id=case_id,
            view=view,
            expected_topology_sha256=topology_sha256,
            sample_rate=self.estimator.sample_rate,
            expected_implementation=CANDIDATE_E_TOPOLOGY_IMPLEMENTATION,
        )
        base_prediction = RouteCFiveActiveScorer.forward(
            self,
            scorer_input,
            speaking_type,
        )
        pulses = torch.as_tensor(
            validated.pulse_positions_samples,
            device=current_waveform.device,
            dtype=torch.float64,
        ).detach()
        source_indices = torch.as_tensor(
            validated.metric_source_indices,
            device=current_waveform.device,
            dtype=torch.long,
        ).detach()
        candidate_result = candidate_e_proxy(
            current_waveform.to(dtype=torch.float64),
            pulses,
            source_indices,
            validated.metric_constant_prefix_samples,
        )
        if candidate_result.peak_scale_abstention_pass is not True:
            raise ValueError(
                "Candidate-E proxy is outside the promoted peak-scale domain"
            )
        raw_shimmer_db = candidate_result.shimmer_db.to(
            dtype=base_prediction.dtype
        )
        normalized_shimmer_db = (
            raw_shimmer_db - self.target_mean[3]
        ) / self.target_scale[3].clamp_min(1e-8)
        return torch.cat(
            (
                base_prediction[..., :3],
                normalized_shimmer_db.reshape(1, 1),
                base_prediction[..., 4:],
            ),
            dim=-1,
        )


def load_route_c_candidate_e_six_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    **estimator_kwargs: Any,
) -> RouteCCandidateESixScorerBundle:
    """Compose the promoted six slots without fitting any parameters."""
    scorer, metadata = _compose_route_c_scorer(
        checkpoint_paths,
        checkpoint_sha256,
        source_keys=ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS,
        source_component_indices=ROUTE_C_SIX_SOURCE_COMPONENT_INDICES,
        source_architectures=ROUTE_C_SIX_SOURCE_ARCHITECTURES,
        estimator_builder=build_route_c_six_active_estimator,
        scorer_class=RouteCCandidateESixScorer,
        estimator_kwargs=estimator_kwargs,
        external_component_indices=ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES[
            "shimmer_db"
        ],
    )
    if not isinstance(scorer, RouteCCandidateESixScorer):
        raise RuntimeError(
            "Candidate-E six-slot scorer composition returned wrong type"
        )
    metadata["shimmer_db"] = {
        "source": "candidate_e_v32r8_current_output_exact_topology",
        "component_indices": [3],
        "checkpoint_affine_used": False,
        "scientific_status": ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS,
        "scientific_promotion_granted": True,
        "optimizer_steps": 0,
    }
    return RouteCCandidateESixScorerBundle(
        scorer=scorer,
        source_metadata=metadata,
        scientific_status=ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS,
    )
