"""Frozen Route C registry and four-active-component scorer composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
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


ROUTE_C_FOUR_ACTIVE_ARCHITECTURE = (
    "direct_praat_hard_cpps_v12_hnr_v7_shimmer_v6_ltas_tilt"
)
ROUTE_C_ACTIVE_COMPONENTS = ("cpps", "hnr", "shimmer_percent", "tilt")
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
    ),
    RouteCComponentSlot(
        4,
        "slope",
        AVQI_V0301_COEFFICIENTS[4],
        AVQI_V0301_EXPANDED_COEFFICIENTS[4],
        "global_ltas_authority_candidate",
        "candidate_available",
        "authority_qualified_candidate_only",
        False,
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
    ),
)


def route_c_registry_records() -> list[dict[str, Any]]:
    """Return JSON-ready ordered records without exposing mutable globals."""
    return [asdict(slot) for slot in ROUTE_C_COMPONENT_REGISTRY]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_checkpoint(
    key: str,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Route C {key} checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Route C {key} checkpoint is not a mapping")
    if tuple(checkpoint.get("components", ())) != AVQI_COMPONENT_NAMES:
        raise ValueError(f"Route C {key} checkpoint component order differs")
    if checkpoint.get("architecture") != ROUTE_C_SOURCE_ARCHITECTURES[key]:
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


@dataclass(frozen=True)
class RouteCScorerBundle:
    scorer: RouteCFourActiveScorer
    source_metadata: dict[str, dict[str, Any]]


def load_route_c_four_active_scorer(
    checkpoint_paths: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    **estimator_kwargs: Any,
) -> RouteCScorerBundle:
    """Compose approved component calibrations without fitting any parameter."""
    expected_keys = set(ROUTE_C_SOURCE_CHECKPOINT_KEYS)
    if set(checkpoint_paths) != expected_keys:
        raise ValueError("Route C checkpoint path keys differ")
    if set(checkpoint_sha256) != expected_keys:
        raise ValueError("Route C checkpoint hash keys differ")
    sources = {
        key: _load_source_checkpoint(
            key,
            Path(checkpoint_paths[key]),
            checkpoint_sha256[key],
        )
        for key in ROUTE_C_SOURCE_CHECKPOINT_KEYS
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
    for key, indices in ROUTE_C_SOURCE_COMPONENT_INDICES.items():
        source = sources[key]
        for index in indices:
            alignment_scale[index] = source["state_dict"]["alignment_scale"][index]
            alignment_bias[index] = source["state_dict"]["alignment_bias"][index]
            calibration_scale[index] = source["calibration_scale"][index]
            calibration_bias[index] = source["calibration_bias"][index]

    estimator = build_route_c_four_active_estimator(**estimator_kwargs)
    estimator.alignment_scale.copy_(alignment_scale)
    estimator.alignment_bias.copy_(alignment_bias)
    calibrator = ComponentAffineCalibrator(calibration_scale, calibration_bias)
    scorer = RouteCFourActiveScorer(
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
            "component_indices": list(ROUTE_C_SOURCE_COMPONENT_INDICES[key]),
            "optimizer_steps": 0,
        }
        for key in ROUTE_C_SOURCE_CHECKPOINT_KEYS
    }
    return RouteCScorerBundle(scorer=scorer, source_metadata=metadata)


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


def active_bidirectional_gap_losses(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return one smooth, bidirectional normalized loss per active slot."""
    if normalized_prediction.shape != raw_target.shape:
        raise ValueError("Route C prediction and target shapes differ")
    normalized_target = (raw_target - target_mean) / target_scale.clamp_min(1e-8)
    indices = normalized_prediction.new_tensor(
        [AVQI_COMPONENT_NAMES.index(name) for name in ROUTE_C_ACTIVE_COMPONENTS],
        dtype=torch.long,
    )
    error = normalized_prediction.index_select(-1, indices) - normalized_target.index_select(
        -1,
        indices,
    )
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
