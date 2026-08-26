"""Frozen Route C registry and composed component-scorer configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    AVQI_V0301_COEFFICIENTS,
    AVQI_V0301_EXPANDED_COEFFICIENTS,
    ComponentAffineCalibrator,
    PraatDifferentiableAVQIComponentEstimator,
    freeze_module,
)
from model.avqi_route_c_v19_contracts import (
    ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
    ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
    ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
    ROUTE_C_V19_FULL_STEP_GATE_KEYS,
    ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
    ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
    ROUTE_C_V19_PASS_DECISION,
    ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE,
    ROUTE_C_V19_PAIRED_CANDIDATE_INPUT_LOADER,
    ROUTE_C_V19_RECEIPT_SCHEMA_VERSION,
    ROUTE_C_V19_REPORT_SCHEMA_VERSION,
    ROUTE_C_V19_SAMPLE_RATE,
    ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
    ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS,
    RouteCArtifactBinding,
    RouteCV19EvidenceManifest,
    sha256_file,
    validate_v19_evidence_manifest,
    validate_v19_exact_topology,
)


ROUTE_C_FOUR_ACTIVE_ARCHITECTURE = (
    "direct_praat_hard_cpps_v12_hnr_v7_shimmer_v6_ltas_tilt"
)
ROUTE_C_FIVE_ACTIVE_ARCHITECTURE = (
    "direct_praat_hard_cpps_v12_hnr_v7_shimmer_v6_ltas_slope_tilt"
)
ROUTE_C_SIX_ACTIVE_ARCHITECTURE = (
    "direct_praat_hard_cpps_v12_hnr_v7_shimmer_percent_v6_"
    "shimmer_db_v19_exact_topology_scaffold_ltas_slope_tilt"
)
ROUTE_C_FOUR_ACTIVE_COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "tilt",
)
ROUTE_C_FIVE_ACTIVE_COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "slope",
    "tilt",
)
ROUTE_C_SIX_ACTIVE_COMPONENTS = AVQI_COMPONENT_NAMES
# Compatibility alias for the sealed four-active integration branch.
ROUTE_C_ACTIVE_COMPONENTS = ROUTE_C_FOUR_ACTIVE_COMPONENTS
ROUTE_C_SOURCE_CHECKPOINT_KEYS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "tilt",
)
ROUTE_C_SOURCE_COMPONENT_INDICES = {
    "cpps": (0,),
    "hnr": (1,),
    "shimmer_percent": (2, 3),
    "tilt": (4, 5),
}
ROUTE_C_SOURCE_ARCHITECTURES = {
    "cpps": "direct_praat_hard_cpps_view_input_v12",
    "hnr": "direct_praat_hard_hnr_pitch_path_v7",
    "shimmer_percent": "direct_praat_hard_shimmer_pulse_path_v6",
    "tilt": "direct_praat_hard_v2",
}
ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "slope",
    "tilt",
)
ROUTE_C_FIVE_SOURCE_COMPONENT_INDICES = {
    "cpps": (0,),
    "hnr": (1,),
    "shimmer_percent": (2, 3),
    "slope": (4,),
    "tilt": (5,),
}
ROUTE_C_FIVE_SOURCE_ARCHITECTURES = {
    **ROUTE_C_SOURCE_ARCHITECTURES,
    "slope": "direct_praat_hard_shimmer_pulse_path_v6",
}
ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS = ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS
ROUTE_C_SIX_SOURCE_COMPONENT_INDICES = {
    "cpps": (0,),
    "hnr": (1,),
    "shimmer_percent": (2,),
    "slope": (4,),
    "tilt": (5,),
}
ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES = {"shimmer_db": (3,)}
ROUTE_C_SIX_SOURCE_ARCHITECTURES = {
    **ROUTE_C_FIVE_SOURCE_ARCHITECTURES,
}
ROUTE_C_REGISTRY_SCHEMA_VERSION = "avqi-route-c-component-registry-v2"
ROUTE_C_SIX_REGISTRY_SCHEMA_VERSION = (
    "avqi-route-c-six-component-scaffold-registry-v1"
)


@dataclass(frozen=True)
class RouteCComponentSlot:
    """One ordered AVQI slot with separate code and scientific status."""

    index: int
    name: str
    avqi_coefficient: float
    expanded_avqi_coefficient: float
    implementation: str
    code_status: str
    scientific_status: str
    active_in_four_component_scorer: bool
    active_in_five_component_scorer: bool


ROUTE_C_COMPONENT_REGISTRY = (
    RouteCComponentSlot(
        0,
        "cpps",
        AVQI_V0301_COEFFICIENTS[0],
        AVQI_V0301_EXPANDED_COEFFICIENTS[0],
        "praat_view_input_v12",
        "integrated",
        "fresh_speaker_panel_pass",
        True,
        True,
    ),
    RouteCComponentSlot(
        1,
        "hnr",
        AVQI_V0301_COEFFICIENTS[1],
        AVQI_V0301_EXPANDED_COEFFICIENTS[1],
        "praat_pitch_path_v7",
        "integrated",
        "fresh_speaker_panel_pass",
        True,
        True,
    ),
    RouteCComponentSlot(
        2,
        "shimmer_percent",
        AVQI_V0301_COEFFICIENTS[2],
        AVQI_V0301_EXPANDED_COEFFICIENTS[2],
        "praat_pulse_path_v6",
        "integrated",
        "fresh_speaker_panel_pass",
        True,
        True,
    ),
    RouteCComponentSlot(
        3,
        "shimmer_db",
        AVQI_V0301_COEFFICIENTS[3],
        AVQI_V0301_EXPANDED_COEFFICIENTS[3],
        "praat_pulse_path_v6_observed_output",
        "slot_reserved",
        "unresolved",
        False,
        False,
    ),
    RouteCComponentSlot(
        4,
        "slope",
        AVQI_V0301_COEFFICIENTS[4],
        AVQI_V0301_EXPANDED_COEFFICIENTS[4],
        "global_ltas_authority_v1",
        "integrated",
        "fresh_speaker_panel_pass",
        False,
        True,
    ),
    RouteCComponentSlot(
        5,
        "tilt",
        AVQI_V0301_COEFFICIENTS[5],
        AVQI_V0301_EXPANDED_COEFFICIENTS[5],
        "global_ltas",
        "integrated",
        "fresh_speaker_panel_pass",
        True,
        True,
    ),
)


@dataclass(frozen=True)
class RouteCSixComponentSlot:
    """One six-component scaffold slot without scientific promotion claims."""

    index: int
    name: str
    avqi_coefficient: float
    expanded_avqi_coefficient: float
    implementation: str
    code_status: str
    scientific_status: str
    active_in_six_component_scorer: bool


ROUTE_C_SIX_COMPONENT_REGISTRY = tuple(
    RouteCSixComponentSlot(
        index=slot.index,
        name=slot.name,
        avqi_coefficient=slot.avqi_coefficient,
        expanded_avqi_coefficient=slot.expanded_avqi_coefficient,
        implementation=(
            "v19_output_conditioned_exact_topology_scaffold"
            if slot.name == "shimmer_db"
            else slot.implementation
        ),
        code_status=(
            "fail_closed_scaffold"
            if slot.name == "shimmer_db"
            else slot.code_status
        ),
        scientific_status=(
            "pending_v19_component_promotion"
            if slot.name == "shimmer_db"
            else slot.scientific_status
        ),
        active_in_six_component_scorer=True,
    )
    for slot in ROUTE_C_COMPONENT_REGISTRY
)
ROUTE_C_SIX_SCIENTIFIC_STATUS = "pending_v19_component_promotion"


def route_c_registry_records() -> list[dict[str, Any]]:
    """Return JSON-ready ordered records without exposing mutable globals."""
    return [asdict(slot) for slot in ROUTE_C_COMPONENT_REGISTRY]


def route_c_six_registry_records() -> list[dict[str, Any]]:
    """Return the ordered six-slot scaffold registry as JSON-ready records."""
    return [asdict(slot) for slot in ROUTE_C_SIX_COMPONENT_REGISTRY]


def build_route_c_four_active_estimator(
    **estimator_kwargs: Any,
) -> PraatDifferentiableAVQIComponentEstimator:
    """Build the zero-parameter estimator carrying all four active formulas."""
    reserved = {
        "peak_mode",
        "cpps_mode",
        "cpps_power_floor",
        "hnr_mode",
        "shimmer_mode",
    }
    conflicts = reserved & set(estimator_kwargs)
    if conflicts:
        raise ValueError(
            "Route C formula modes are frozen; conflicting overrides: "
            f"{sorted(conflicts)}"
        )
    return PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        cpps_mode="praat_view_input_v12",
        cpps_power_floor=1e-6,
        hnr_mode="praat_pitch_path_v7",
        shimmer_mode="praat_pulse_path_v6",
        **estimator_kwargs,
    )


def build_route_c_five_active_estimator(
    **estimator_kwargs: Any,
) -> PraatDifferentiableAVQIComponentEstimator:
    """Build the estimator carrying all five scientifically promoted formulas."""
    return build_route_c_four_active_estimator(**estimator_kwargs)


def build_route_c_six_active_estimator(
    **estimator_kwargs: Any,
) -> PraatDifferentiableAVQIComponentEstimator:
    """Build the shared formulas; slot 3 is replaced by external v19 topology."""
    return build_route_c_five_active_estimator(**estimator_kwargs)


def _load_source_checkpoint(
    key: str,
    path: Path,
    expected_sha256: str,
    expected_architecture: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Route C {key} checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Route C {key} checkpoint is not a mapping")
    if tuple(checkpoint.get("components", ())) != AVQI_COMPONENT_NAMES:
        raise ValueError(f"Route C {key} checkpoint component order differs")
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError(f"Route C {key} checkpoint architecture differs")
    if checkpoint.get("optimizer_steps") != 0:
        raise ValueError(f"Route C {key} checkpoint contains optimizer steps")
    if checkpoint.get("trainable_parameter_count") != 0:
        raise ValueError(f"Route C {key} checkpoint has trainable parameters")
    if key == "cpps" and checkpoint.get("speaking_type_required") is not True:
        raise ValueError("Route C CPPS checkpoint does not require speaking type")
    required = {
        "state_dict",
        "target_mean",
        "target_scale",
        "calibration_scale",
        "calibration_bias",
    }
    if not required <= set(checkpoint):
        raise ValueError(f"Route C {key} checkpoint is incomplete")
    state_dict = checkpoint["state_dict"]
    expected_shape = (len(AVQI_COMPONENT_NAMES),)
    tensors = (
        state_dict.get("alignment_scale"),
        state_dict.get("alignment_bias"),
        checkpoint["target_mean"],
        checkpoint["target_scale"],
        checkpoint["calibration_scale"],
        checkpoint["calibration_bias"],
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError(f"Route C {key} checkpoint tensors are missing")
    if any(tuple(value.shape) != expected_shape for value in tensors):
        raise ValueError(f"Route C {key} checkpoint tensor shape differs")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"Route C {key} checkpoint has non-finite tensors")
    if not torch.all(checkpoint["target_scale"] > 0.0):
        raise ValueError(f"Route C {key} checkpoint target scale is non-positive")
    return checkpoint


class RouteCFourActiveScorer(nn.Module):
    """A composed, frozen scorer; outputs calibrated normalized components."""

    def __init__(
        self,
        estimator: PraatDifferentiableAVQIComponentEstimator,
        calibrator: ComponentAffineCalibrator,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.calibrator = calibrator
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_scale", target_scale.detach().clone())
        freeze_module(self.estimator)
        freeze_module(self.calibrator)

    def forward(
        self,
        waveform: torch.Tensor,
        speaking_type: str,
    ) -> torch.Tensor:
        return self.calibrator(self.estimator(waveform, speaking_type))

    def normalized_target(self, raw_target: torch.Tensor) -> torch.Tensor:
        return (raw_target - self.target_mean) / self.target_scale.clamp_min(1e-8)

    def denormalized_prediction(
        self,
        normalized_prediction: torch.Tensor,
    ) -> torch.Tensor:
        return normalized_prediction * self.target_scale + self.target_mean


class RouteCFiveActiveScorer(RouteCFourActiveScorer):
    """Frozen scorer carrying the five scientifically promoted components."""


class RouteCSixActiveScorer(RouteCFiveActiveScorer):
    """Fail-closed scorer whose Shimmer dB slot uses current-output topology."""

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
            raise ValueError("Route C six-active scorer requires v19 topology")
        if case_id is None or view is None or topology_sha256 is None:
            raise ValueError("Route C six-active topology binding is incomplete")
        if speaking_type != view:
            raise ValueError("Route C six-active speaking type and view differ")
        if waveform.ndim == 1:
            current_waveform = waveform
            scorer_input = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            current_waveform = waveform[0]
            scorer_input = waveform
        else:
            raise ValueError(
                "Route C six-active scorer requires one current output waveform"
            )
        validated = validate_v19_exact_topology(
            current_waveform,
            topology,
            case_id=case_id,
            view=view,
            expected_topology_sha256=topology_sha256,
            sample_rate=self.estimator.sample_rate,
        )
        base_prediction = super().forward(scorer_input, speaking_type)
        pulses = current_waveform.new_tensor(
            validated.pulse_positions_samples
        ).detach()
        source_indices = torch.as_tensor(
            validated.metric_source_indices,
            dtype=torch.long,
            device=current_waveform.device,
        ).detach()
        raw_shimmer_db = self.estimator.raw_shimmer_from_pulse_positions(
            current_waveform,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=(
                validated.metric_constant_prefix_samples
            ),
        )[1]
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


@dataclass(frozen=True)
class RouteCScorerBundle:
    scorer: RouteCFourActiveScorer
    source_metadata: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RouteCFiveScorerBundle:
    scorer: RouteCFiveActiveScorer
    source_metadata: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RouteCSixScorerBundle:
    scorer: RouteCSixActiveScorer
    source_metadata: dict[str, dict[str, Any]]
    v19_evidence_metadata: dict[str, Any]
    scientific_status: str = ROUTE_C_SIX_SCIENTIFIC_STATUS
    generator_optimizer_steps: int = 0


def _compose_route_c_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    *,
    source_keys: tuple[str, ...],
    source_component_indices: Mapping[str, tuple[int, ...]],
    source_architectures: Mapping[str, str],
    estimator_builder: Callable[..., PraatDifferentiableAVQIComponentEstimator],
    scorer_class: type[RouteCFourActiveScorer],
    estimator_kwargs: Mapping[str, Any],
    external_component_indices: tuple[int, ...] = (),
) -> tuple[RouteCFourActiveScorer, dict[str, dict[str, Any]]]:
    expected_keys = set(source_keys)
    if set(checkpoint_paths) != expected_keys:
        raise ValueError("Route C checkpoint path keys differ")
    if set(checkpoint_sha256) != expected_keys:
        raise ValueError("Route C checkpoint hash keys differ")
    if set(source_component_indices) != expected_keys:
        raise ValueError("Route C source component-index keys differ")
    if set(source_architectures) != expected_keys:
        raise ValueError("Route C source architecture keys differ")
    selected_indices = [
        index
        for key in source_keys
        for index in source_component_indices[key]
    ]
    if len(set(external_component_indices)) != len(external_component_indices):
        raise ValueError("Route C external source slots contain duplicates")
    if set(selected_indices) & set(external_component_indices):
        raise ValueError("Route C checkpoint and external source slots overlap")
    if sorted((*selected_indices, *external_component_indices)) != list(
        range(len(AVQI_COMPONENT_NAMES))
    ):
        raise ValueError("Route C source slots do not cover each AVQI component once")
    sources = {
        key: _load_source_checkpoint(
            key,
            Path(checkpoint_paths[key]),
            checkpoint_sha256[key],
            source_architectures[key],
        )
        for key in source_keys
    }
    reference_mean = sources["tilt"]["target_mean"].detach().clone()
    reference_scale = sources["tilt"]["target_scale"].detach().clone()
    for key, checkpoint in sources.items():
        if not torch.equal(checkpoint["target_mean"], reference_mean):
            raise ValueError(f"Route C {key} target mean differs")
        if not torch.equal(checkpoint["target_scale"], reference_scale):
            raise ValueError(f"Route C {key} target scale differs")

    alignment_scale = sources["tilt"]["state_dict"][
        "alignment_scale"
    ].detach().clone()
    alignment_bias = sources["tilt"]["state_dict"][
        "alignment_bias"
    ].detach().clone()
    calibration_scale = sources["tilt"]["calibration_scale"].detach().clone()
    calibration_bias = sources["tilt"]["calibration_bias"].detach().clone()
    for key, indices in source_component_indices.items():
        source = sources[key]
        for index in indices:
            alignment_scale[index] = source["state_dict"]["alignment_scale"][index]
            alignment_bias[index] = source["state_dict"]["alignment_bias"][index]
            calibration_scale[index] = source["calibration_scale"][index]
            calibration_bias[index] = source["calibration_bias"][index]
    for index in external_component_indices:
        alignment_scale[index] = 1.0
        alignment_bias[index] = 0.0
        calibration_scale[index] = 1.0
        calibration_bias[index] = 0.0

    estimator = estimator_builder(**estimator_kwargs)
    estimator.alignment_scale.copy_(alignment_scale)
    estimator.alignment_bias.copy_(alignment_bias)
    calibrator = ComponentAffineCalibrator(calibration_scale, calibration_bias)
    scorer = scorer_class(
        estimator,
        calibrator,
        reference_mean,
        reference_scale,
    )
    metadata = {
        key: {
            "path": str(Path(checkpoint_paths[key]).resolve()),
            "sha256": checkpoint_sha256[key],
            "architecture": sources[key]["architecture"],
            "component_indices": list(source_component_indices[key]),
            "optimizer_steps": 0,
        }
        for key in source_keys
    }
    return scorer, metadata


def load_route_c_four_active_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    **estimator_kwargs: Any,
) -> RouteCScorerBundle:
    """Compose the sealed four-active scorer without fitting parameters."""
    scorer, metadata = _compose_route_c_scorer(
        checkpoint_paths,
        checkpoint_sha256,
        source_keys=ROUTE_C_SOURCE_CHECKPOINT_KEYS,
        source_component_indices=ROUTE_C_SOURCE_COMPONENT_INDICES,
        source_architectures=ROUTE_C_SOURCE_ARCHITECTURES,
        estimator_builder=build_route_c_four_active_estimator,
        scorer_class=RouteCFourActiveScorer,
        estimator_kwargs=estimator_kwargs,
    )
    return RouteCScorerBundle(scorer=scorer, source_metadata=metadata)


def load_route_c_five_active_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    **estimator_kwargs: Any,
) -> RouteCFiveScorerBundle:
    """Compose the five promoted slots; Shimmer dB remains inactive."""
    scorer, metadata = _compose_route_c_scorer(
        checkpoint_paths,
        checkpoint_sha256,
        source_keys=ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS,
        source_component_indices=ROUTE_C_FIVE_SOURCE_COMPONENT_INDICES,
        source_architectures=ROUTE_C_FIVE_SOURCE_ARCHITECTURES,
        estimator_builder=build_route_c_five_active_estimator,
        scorer_class=RouteCFiveActiveScorer,
        estimator_kwargs=estimator_kwargs,
    )
    if not isinstance(scorer, RouteCFiveActiveScorer):
        raise RuntimeError("Route C five-active scorer composition returned wrong type")
    return RouteCFiveScorerBundle(scorer=scorer, source_metadata=metadata)


def load_route_c_six_active_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    *,
    v19_evidence_manifest: RouteCV19EvidenceManifest,
    **estimator_kwargs: Any,
) -> RouteCSixScorerBundle:
    """Compose an unpromoted six-slot scaffold with fail-closed v19 evidence."""
    evidence_metadata = validate_v19_evidence_manifest(v19_evidence_manifest)
    scorer, metadata = _compose_route_c_scorer(
        checkpoint_paths,
        checkpoint_sha256,
        source_keys=ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS,
        source_component_indices=ROUTE_C_SIX_SOURCE_COMPONENT_INDICES,
        source_architectures=ROUTE_C_SIX_SOURCE_ARCHITECTURES,
        estimator_builder=build_route_c_six_active_estimator,
        scorer_class=RouteCSixActiveScorer,
        estimator_kwargs=estimator_kwargs,
        external_component_indices=ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES[
            "shimmer_db"
        ],
    )
    if not isinstance(scorer, RouteCSixActiveScorer):
        raise RuntimeError("Route C six-active scorer composition returned wrong type")
    metadata["shimmer_db"] = {
        "source": "v19_current_output_exact_topology",
        "component_indices": [3],
        "checkpoint_affine_used": False,
        "scientific_status": ROUTE_C_SIX_SCIENTIFIC_STATUS,
        "scientific_promotion_granted": False,
        "optimizer_steps": 0,
    }
    return RouteCSixScorerBundle(
        scorer=scorer,
        source_metadata=metadata,
        v19_evidence_metadata=evidence_metadata,
    )


def normalized_bidirectional_component_gaps(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Absolute target distance; AVQI coefficient signs never set direction."""
    if normalized_prediction.shape != raw_target.shape:
        raise ValueError("Route C prediction and target shapes differ")
    normalized_target = (raw_target - target_mean) / target_scale.clamp_min(1e-8)
    return (normalized_prediction - normalized_target).abs()


def component_bidirectional_gap_losses(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    components: tuple[str, ...],
) -> torch.Tensor:
    """Return one smooth, bidirectional normalized loss per selected slot."""
    if normalized_prediction.shape != raw_target.shape:
        raise ValueError("Route C prediction and target shapes differ")
    if not components or len(set(components)) != len(components):
        raise ValueError("Route C selected components must be unique and non-empty")
    unknown = set(components) - set(AVQI_COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"Route C selected components are unknown: {sorted(unknown)}")
    normalized_target = (raw_target - target_mean) / target_scale.clamp_min(1e-8)
    indices = normalized_prediction.new_tensor(
        [AVQI_COMPONENT_NAMES.index(name) for name in components],
        dtype=torch.long,
    )
    error = normalized_prediction.index_select(
        -1, indices
    ) - normalized_target.index_select(
        -1, indices
    )
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")


def active_bidirectional_gap_losses(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return the sealed four-active loss vector for compatibility."""
    return component_bidirectional_gap_losses(
        normalized_prediction,
        raw_target,
        target_mean,
        target_scale,
        ROUTE_C_FOUR_ACTIVE_COMPONENTS,
    )


def five_active_bidirectional_gap_losses(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return one bidirectional normalized loss for each five-active slot."""
    return component_bidirectional_gap_losses(
        normalized_prediction,
        raw_target,
        target_mean,
        target_scale,
        ROUTE_C_FIVE_ACTIVE_COMPONENTS,
    )


def six_active_bidirectional_gap_losses(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return six same-speaker, normalized, bidirectional SmoothL1 losses."""
    return component_bidirectional_gap_losses(
        normalized_prediction,
        raw_target,
        target_mean,
        target_scale,
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
    )
