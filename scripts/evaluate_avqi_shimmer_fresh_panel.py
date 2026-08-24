#!/usr/bin/env python3
"""Run a fresh speaker-disjoint Route C Shimmer waveform pilot.

The pilot is deliberately narrower than AVQI-T2 training. It creates twelve
new TAU pathological CS/SV cases from hash-locked test noise/RIR recipes,
runs the frozen S3_500 enhancer in inference mode, and applies one bounded
waveform step from the frozen Route C v6 Shimmer-percent gradient.

A global step size is selected only on three calibration speakers. The final
three-speaker panel is written and hash-sealed before its exact Praat values
are computed. Exact Praat scores all six AVQI components; the emitted waveform
is never high-pass filtered, and full-band pathology guardrails remain active.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    ComponentAffineCalibrator,
    PraatDifferentiableAVQIComponentEstimator,
    denormalize_components,
)
from evaluate_direct_avqi_waveform_optimization import (
    STEP_VERSIONS,
    aggregate_denoising,
    aggregate_pathology_guardrails,
    avqi_code_tree_sha256,
    full_band_pathology_guardrails,
    waveform_safety,
)
from prepare_avqi_component_expanded_data import (
    WdsReader,
    crop_or_tile,
    match_length,
    read_clean,
    stable_seed,
)


SAMPLE_RATE = 16_000
SHIMMER_PILOT_PROFILE = "shimmer_pulse_path_v6"
HNR_PILOT_PROFILE = "hnr_pitch_path_v7"
OPTIMIZED_COMPONENT = "shimmer_percent"
COMPONENT_INDEX = AVQI_COMPONENT_NAMES.index(OPTIMIZED_COMPONENT)
SHIMMER_INDEX = COMPONENT_INDEX
COMPANION_COMPONENT: str | None = "shimmer_db"
EXPECTED_ARCHITECTURE = "direct_praat_hard_shimmer_pulse_path_v6"
ESTIMATOR_KWARGS: dict[str, Any] = {
    "peak_mode": "hard",
    "shimmer_mode": "praat_pulse_path_v6",
}
DISPLAY_NAME = "Shimmer percent"
VERSION_LABEL = "shimmer_v6"
PANEL_SCHEMA_VERSION = "avqi-route-c-shimmer-fresh-panel-v1"
FINAL_SEAL_SCHEMA_VERSION = "avqi-route-c-shimmer-final-seal-v1"
PASS_DECISION = "PASS_SHIMMER_FRESH_SPEAKER_PANEL"
FAIL_DECISION = "FAIL_SHIMMER_FRESH_SPEAKER_PANEL"
CALIBRATION_NO_GO_DECISION = "NO_GO_SHIMMER_CALIBRATION_FINAL_UNOPENED"
ALPHA_GRID = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
MATERIAL_GAP_THRESHOLD = 0.02
MEDIAN_REDUCTION_GATE = 0.02
IMPROVEMENT_FRACTION_GATE = 0.80
PROXY_IMPROVEMENT_FRACTION_GATE = 0.80
NONSELECTED_MEDIAN_INCREASE_GATE = 0.05
RESIDUAL_CEILING_DB = -50.0
MINIMUM_COSINE = 0.99999
MAXIMUM_CLIP_FRACTION = 0.0
REQUIRED_FINAL_SLICES = (
    "view=cs",
    "view=sv",
    "severity=pathological_mild",
    "severity=pathological_severe",
)
PREVIOUS_WAVEFORM_PILOT_SPEAKERS = frozenset(
    {
        "PD08",
        "PD_51",
        "SD37",
        "V55",
        "ÄHH05",
        "ÄHH10",
        "ÄHH22",
        "ÄHH25",
        "ÄHH28",
        "ÄHH29",
    }
)

# Speaker split precedes recipe assignment. Within each split, CS/SV and the
# three degradation conditions are balanced exactly (two cases per condition).
SHIMMER_PANEL_ROWS = (
    ("calibration", "FD26", "pathological_mild", "cs", "rir_only", 900),
    ("calibration", "FD26", "pathological_mild", "sv", "snr20", 901),
    ("calibration", "SD36", "pathological_mild", "cs", "snr10", 902),
    ("calibration", "SD36", "pathological_mild", "sv", "rir_only", 903),
    ("calibration", "FD11", "pathological_severe", "cs", "snr20", 904),
    ("calibration", "FD11", "pathological_severe", "sv", "snr10", 905),
    ("final", "ÄHH20", "pathological_mild", "cs", "rir_only", 906),
    ("final", "ÄHH20", "pathological_mild", "sv", "snr20", 907),
    ("final", "FD20", "pathological_severe", "cs", "snr10", 908),
    ("final", "FD20", "pathological_severe", "sv", "rir_only", 909),
    ("final", "SD23", "pathological_severe", "cs", "snr20", 910),
    ("final", "SD23", "pathological_severe", "sv", "snr10", 911),
)

# These HNR cases are newly simulated from unused fixed test recipes.  The
# calibration and final speakers are disjoint.  The final speakers also avoid
# every previously opened HNR, CPPS, or Shimmer final panel speaker.  The
# scorer-external TAU speaker pool has otherwise been exhausted by earlier
# component pilots, so calibration reuses three prior calibration speakers.
HNR_PANEL_ROWS = (
    ("calibration", "FD26", "pathological_mild", "cs", "rir_only", 940),
    ("calibration", "FD26", "pathological_mild", "sv", "snr20", 941),
    ("calibration", "SD36", "pathological_mild", "cs", "snr10", 942),
    ("calibration", "SD36", "pathological_mild", "sv", "rir_only", 943),
    ("calibration", "FD11", "pathological_severe", "cs", "snr20", 944),
    ("calibration", "FD11", "pathological_severe", "sv", "snr10", 945),
    ("final", "SD13", "pathological_mild", "cs", "rir_only", 946),
    ("final", "SD13", "pathological_mild", "sv", "snr20", 947),
    ("final", "PD_51", "pathological_severe", "cs", "snr10", 948),
    ("final", "PD_51", "pathological_severe", "sv", "rir_only", 949),
    ("final", "ÄHH28", "pathological_severe", "cs", "snr20", 950),
    ("final", "ÄHH28", "pathological_severe", "sv", "snr10", 951),
)

PRIOR_FINAL_PANEL_SPEAKERS = frozenset(
    {
        "PD08",
        "SD23",
        "SD37",
        "FD20",
        "ÄHH05",
        "ÄHH10",
        "ÄHH20",
        "ÄHH22",
        "ÄHH25",
        "ÄHH29",
    }
)
ALL_PRIOR_PANEL_SPEAKERS = PREVIOUS_WAVEFORM_PILOT_SPEAKERS | frozenset(
    {"FD11", "FD20", "FD26", "SD23", "SD36", "ÄHH20"}
)
PANEL_ROWS = SHIMMER_PANEL_ROWS
ACTIVE_PILOT_PROFILE = SHIMMER_PILOT_PROFILE

PILOT_PROFILE_CONFIGS = {
    SHIMMER_PILOT_PROFILE: {
        "component": "shimmer_percent",
        "companion_component": "shimmer_db",
        "architecture": "direct_praat_hard_shimmer_pulse_path_v6",
        "estimator_kwargs": {
            "peak_mode": "hard",
            "shimmer_mode": "praat_pulse_path_v6",
        },
        "display_name": "Shimmer percent",
        "version_label": "shimmer_v6",
        "panel_rows": SHIMMER_PANEL_ROWS,
        "panel_schema_version": "avqi-route-c-shimmer-fresh-panel-v1",
        "final_seal_schema_version": "avqi-route-c-shimmer-final-seal-v1",
        "pass_decision": "PASS_SHIMMER_FRESH_SPEAKER_PANEL",
        "fail_decision": "FAIL_SHIMMER_FRESH_SPEAKER_PANEL",
        "calibration_no_go_decision": (
            "NO_GO_SHIMMER_CALIBRATION_FINAL_UNOPENED"
        ),
    },
    HNR_PILOT_PROFILE: {
        "component": "hnr",
        "companion_component": None,
        "architecture": "direct_praat_hard_hnr_pitch_path_v7",
        "estimator_kwargs": {
            "peak_mode": "hard",
            "hnr_mode": "praat_pitch_path_v7",
        },
        "display_name": "HNR",
        "version_label": "hnr_v7",
        "panel_rows": HNR_PANEL_ROWS,
        "panel_schema_version": "avqi-route-c-hnr-fresh-panel-v1",
        "final_seal_schema_version": "avqi-route-c-hnr-final-seal-v1",
        "pass_decision": "PASS_HNR_FRESH_SPEAKER_PANEL",
        "fail_decision": "FAIL_HNR_FRESH_SPEAKER_PANEL",
        "calibration_no_go_decision": "NO_GO_HNR_CALIBRATION_FINAL_UNOPENED",
    },
}


def configure_pilot(profile: str) -> None:
    try:
        config = PILOT_PROFILE_CONFIGS[profile]
    except KeyError as error:
        raise ValueError(f"unknown fresh-panel profile: {profile}") from error
    global ACTIVE_PILOT_PROFILE
    global OPTIMIZED_COMPONENT, COMPONENT_INDEX, SHIMMER_INDEX
    global COMPANION_COMPONENT, EXPECTED_ARCHITECTURE, ESTIMATOR_KWARGS
    global DISPLAY_NAME, VERSION_LABEL, PANEL_ROWS
    global PANEL_SCHEMA_VERSION, FINAL_SEAL_SCHEMA_VERSION
    global PASS_DECISION, FAIL_DECISION, CALIBRATION_NO_GO_DECISION
    ACTIVE_PILOT_PROFILE = profile
    OPTIMIZED_COMPONENT = str(config["component"])
    COMPONENT_INDEX = AVQI_COMPONENT_NAMES.index(OPTIMIZED_COMPONENT)
    SHIMMER_INDEX = COMPONENT_INDEX
    companion = config["companion_component"]
    COMPANION_COMPONENT = None if companion is None else str(companion)
    EXPECTED_ARCHITECTURE = str(config["architecture"])
    ESTIMATOR_KWARGS = dict(config["estimator_kwargs"])
    DISPLAY_NAME = str(config["display_name"])
    VERSION_LABEL = str(config["version_label"])
    PANEL_ROWS = tuple(config["panel_rows"])
    PANEL_SCHEMA_VERSION = str(config["panel_schema_version"])
    FINAL_SEAL_SCHEMA_VERSION = str(config["final_seal_schema_version"])
    PASS_DECISION = str(config["pass_decision"])
    FAIL_DECISION = str(config["fail_decision"])
    CALIBRATION_NO_GO_DECISION = str(config["calibration_no_go_decision"])

EXACT_SCORER = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import parselmouth
from avqi_code import run_avqi

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    metrics = run_avqi(
        item["path"],
        item["path"],
        target_sr=16000,
        speaking_type=item["view"],
        step_versions=request["step_versions"],
        remove_sv_silence_with_sox=False,
    )
    rows.append(
        {
            "id": item["id"],
            "components": {
                name: float(metrics[name]) for name in request["components"]
            },
        }
    )
print(
    "AVQI_FRESH_PANEL_EXACT_JSON="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""


@dataclass(frozen=True)
class PanelSpec:
    split: str
    speaker_id: str
    sample_group: str
    view: str
    condition: str
    recipe_index: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.split}__{self.speaker_id}__{self.view}__{self.condition}"
        )


@dataclass
class PreparedCase:
    spec: PanelSpec
    target_path: Path
    noisy_path: Path
    base_path: Path
    source_path: Path
    recipe: dict[str, Any]
    simulation_seed: int
    noise_start_sample: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-profile",
        choices=tuple(PILOT_PROFILE_CONFIGS),
        default=SHIMMER_PILOT_PROFILE,
    )
    parser.add_argument("--authorization-consensus", type=Path, required=True)
    parser.add_argument("--authorization-consensus-sha256", required=True)
    parser.add_argument(
        "--authorization-consensus-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--authorization-consensus-receipt-sha256",
        required=True,
    )
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--screen-report-sha256", required=True)
    parser.add_argument("--screen-completion-receipt", type=Path, required=True)
    parser.add_argument("--screen-completion-receipt-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--generator-config-sha256", required=True)
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--generator-checkpoint-sha256", required=True)
    parser.add_argument("--tau-manifest", type=Path, required=True)
    parser.add_argument("--tau-manifest-sha256", required=True)
    parser.add_argument("--fixed-recipes", type=Path, required=True)
    parser.add_argument("--fixed-recipes-sha256", required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--simulation-config-sha256", required=True)
    parser.add_argument("--simulation-source-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def panel_specs() -> tuple[PanelSpec, ...]:
    return tuple(PanelSpec(*row) for row in PANEL_ROWS)


def validate_panel_specs(specs: tuple[PanelSpec, ...]) -> dict[str, Any]:
    if len(specs) != 12 or len({spec.case_id for spec in specs}) != 12:
        raise ValueError("fresh component panel must contain twelve unique cases")
    split_speakers = {
        split: {spec.speaker_id for spec in specs if spec.split == split}
        for split in ("calibration", "final")
    }
    if any(len(speakers) != 3 for speakers in split_speakers.values()):
        raise ValueError(f"expected three speakers per split: {split_speakers}")
    if split_speakers["calibration"] & split_speakers["final"]:
        raise ValueError("calibration and final speakers overlap")
    all_speakers = set.union(*split_speakers.values())
    prior_speakers = (
        PREVIOUS_WAVEFORM_PILOT_SPEAKERS
        if ACTIVE_PILOT_PROFILE == SHIMMER_PILOT_PROFILE
        else ALL_PRIOR_PANEL_SPEAKERS
    )
    previous_overlap = all_speakers & prior_speakers
    if ACTIVE_PILOT_PROFILE == SHIMMER_PILOT_PROFILE and previous_overlap:
        raise ValueError("fresh panel overlaps a previous waveform pilot speaker")
    prior_final_overlap = split_speakers["final"] & PRIOR_FINAL_PANEL_SPEAKERS
    if ACTIVE_PILOT_PROFILE == HNR_PILOT_PROFILE and prior_final_overlap:
        raise ValueError(
            "HNR final panel overlaps a previously opened final-panel speaker"
        )
    for split in split_speakers:
        selected = [spec for spec in specs if spec.split == split]
        if {spec.view for spec in selected} != {"cs", "sv"}:
            raise ValueError(f"missing CS/SV coverage in {split}")
        view_counts = {
            view: sum(spec.view == view for spec in selected)
            for view in ("cs", "sv")
        }
        condition_counts = {
            condition: sum(spec.condition == condition for spec in selected)
            for condition in ("rir_only", "snr20", "snr10")
        }
        if view_counts != {"cs": 3, "sv": 3}:
            raise ValueError(f"view balance drift in {split}: {view_counts}")
        if condition_counts != {"rir_only": 2, "snr20": 2, "snr10": 2}:
            raise ValueError(
                f"condition balance drift in {split}: {condition_counts}"
            )
        if {spec.sample_group for spec in selected} != {
            "pathological_mild",
            "pathological_severe",
        }:
            raise ValueError(f"severity coverage drift in {split}")
    recipe_indices = [spec.recipe_index for spec in specs]
    if len(recipe_indices) != len(set(recipe_indices)):
        raise ValueError("fixed recipe reuse detected within the fresh panel")
    return {
        "case_count": len(specs),
        "split_speakers": {
            split: sorted(speakers)
            for split, speakers in split_speakers.items()
        },
        "previous_waveform_pilot_overlap": sorted(previous_overlap),
        "previous_final_panel_overlap": sorted(prior_final_overlap),
        "recipe_indices": recipe_indices,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")
    return actual


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def repository_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_file_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return sha256_bytes(result.stdout)


def validate_authorization(args: argparse.Namespace) -> dict[str, Any]:
    consensus_hash = validate_file_hash(
        args.authorization_consensus,
        args.authorization_consensus_sha256,
        "Route C consensus",
    )
    consensus_receipt_hash = validate_file_hash(
        args.authorization_consensus_receipt,
        args.authorization_consensus_receipt_sha256,
        "Route C consensus completion receipt",
    )
    screen_hash = validate_file_hash(
        args.screen_report,
        args.screen_report_sha256,
        "Route C screen report",
    )
    screen_receipt_hash = validate_file_hash(
        args.screen_completion_receipt,
        args.screen_completion_receipt_sha256,
        "Route C screen completion receipt",
    )
    predictor_hash = validate_file_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "Route C v6 checkpoint",
    )
    generator_config_hash = validate_file_hash(
        args.generator_config,
        args.generator_config_sha256,
        "S3_500 generator config",
    )
    generator_checkpoint_hash = validate_file_hash(
        args.generator_checkpoint,
        args.generator_checkpoint_sha256,
        "S3_500 generator checkpoint",
    )
    consensus = load_json(args.authorization_consensus)
    consensus_receipt = load_json(args.authorization_consensus_receipt)
    screen = load_json(args.screen_report)
    screen_receipt = load_json(args.screen_completion_receipt)
    promotion = consensus.get("promotion", {})
    if consensus.get("schema_version") != "avqi-component-multiseed-consensus-v2":
        raise ValueError("unexpected Route C consensus schema")
    if promotion.get("decision") != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT":
        raise ValueError("Route C consensus does not authorize a bounded pilot")
    if OPTIMIZED_COMPONENT not in promotion.get("components", []):
        raise ValueError(
            f"{DISPLAY_NAME} is absent from the authorized components"
        )
    route_consensus = consensus.get("routes", {}).get(
        "direct_differentiable_estimator",
        {},
    )
    if route_consensus.get("decision") != "RELIABLE":
        raise ValueError("Route C is not reliable in the multi-seed consensus")
    if route_consensus.get("component_pass_counts", {}).get(
        OPTIMIZED_COMPONENT
    ) != 3:
        raise ValueError(
            f"{DISPLAY_NAME} did not pass all three confirmation seeds"
        )
    if route_consensus.get("selected_form") != EXPECTED_ARCHITECTURE:
        raise ValueError("Route C consensus selected form differs")
    if consensus.get("generator_optimizer_steps") != 0:
        raise ValueError("consensus contains generator updates")
    if consensus.get("formal_pathology_training_submitted") is not False:
        raise ValueError("consensus formal-training state differs")
    if consensus.get("source_report_sha256", {}).get("screen") != screen_hash:
        raise ValueError("consensus does not bind the supplied screen report")
    if consensus_receipt.get("decision") != promotion["decision"]:
        raise ValueError("consensus receipt decision differs")
    if consensus_receipt.get("artifact_sha256", {}).get(
        args.authorization_consensus.name
    ) != consensus_hash:
        raise ValueError("consensus receipt does not bind its consensus report")
    if consensus_receipt.get("source_report_sha256") != consensus.get(
        "source_report_sha256"
    ):
        raise ValueError("consensus receipt source hashes differ")
    if consensus_receipt.get("generator_optimizer_steps") != 0:
        raise ValueError("consensus receipt contains generator updates")
    if consensus_receipt.get("formal_pathology_training_submitted") is not False:
        raise ValueError("consensus receipt formal-training state differs")

    expected_screen_decision = (
        "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
    )
    if screen.get("decision") != expected_screen_decision:
        raise ValueError("Route C screen is incomplete")
    if screen.get("generator_optimizer_steps") != 0:
        raise ValueError("screen contains generator updates")
    if screen.get("contract", {}).get("route_scope") != "direct_only":
        raise ValueError("screen is not Route-C-only")
    route = screen.get("routes", {}).get("direct_differentiable_estimator", {})
    if route.get("selected_architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("screen did not select the expected component formula")
    if OPTIMIZED_COMPONENT not in route.get("eligible_components", []):
        raise ValueError(f"screen did not qualify {DISPLAY_NAME}")
    component_gradient = route.get("gradient", {}).get(
        "component_input_gradients",
        {},
    ).get(OPTIMIZED_COMPONENT, {})
    if component_gradient.get("decision") != "PASS":
        raise ValueError(f"screen {DISPLAY_NAME} gradient did not pass")
    if screen.get("contract", {}).get("source_sha256", {}).get("config") != (
        generator_config_hash
    ):
        raise ValueError("generator config differs from the scorer screen")
    if screen.get("contract", {}).get("source_sha256", {}).get(
        "generator_checkpoint"
    ) != generator_checkpoint_hash:
        raise ValueError("generator checkpoint differs from the scorer screen")

    if screen_receipt.get("decision") != screen["decision"]:
        raise ValueError("screen receipt decision differs")
    if screen_receipt.get("artifact_sha256", {}).get(
        "diagnostic_report.json"
    ) != screen_hash:
        raise ValueError("screen receipt does not bind its diagnostic report")
    if screen_receipt.get("checkpoint_sha256", {}).get(
        args.predictor_checkpoint.name
    ) != predictor_hash:
        raise ValueError("screen receipt does not bind the v6 checkpoint")

    screen_commit = str(screen["contract"]["source_commit"])
    screen_model_hash = git_file_sha256(
        screen_commit,
        "model/avqi_components.py",
    )
    live_model_hash = sha256_file(REPO_ROOT / "model/avqi_components.py")
    if live_model_hash != screen_model_hash:
        raise ValueError("v6 estimator source changed after the scorer screen")
    return {
        "decision": promotion["decision"],
        "authorized_components": list(promotion["components"]),
        "isolated_component": OPTIMIZED_COMPONENT,
        "consensus_sha256": consensus_hash,
        "consensus_completion_receipt_sha256": consensus_receipt_hash,
        "screen_report_sha256": screen_hash,
        "screen_completion_receipt_sha256": screen_receipt_hash,
        "predictor_checkpoint_sha256": predictor_hash,
        "screen_source_commit": screen_commit,
        "screen_model_source_sha256": screen_model_hash,
        "pilot_model_source_sha256": live_model_hash,
        "pilot_profile": ACTIVE_PILOT_PROFILE,
        "selected_architecture": EXPECTED_ARCHITECTURE,
        "screen_gradient_norm": float(component_gradient["gradient_norm"]),
    }


def load_predictor(
    path: Path,
    device: torch.device,
) -> tuple[
    PraatDifferentiableAVQIComponentEstimator,
    ComponentAffineCalibrator,
    torch.Tensor,
    torch.Tensor,
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"unexpected Route C checkpoint: {checkpoint.get('architecture')}")
    if tuple(checkpoint.get("components", ())) != AVQI_COMPONENT_NAMES:
        raise ValueError("Route C checkpoint component order differs")
    predictor = PraatDifferentiableAVQIComponentEstimator(
        **ESTIMATOR_KWARGS,
    ).to(device)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.eval()
    calibrator = ComponentAffineCalibrator(
        checkpoint["calibration_scale"],
        checkpoint["calibration_bias"],
    ).to(device)
    calibrator.eval()
    target_mean = checkpoint["target_mean"].to(device)
    target_scale = checkpoint["target_scale"].to(device)
    return predictor, calibrator, target_mean, target_scale


def predict_components(
    predictor: PraatDifferentiableAVQIComponentEstimator,
    calibrator: ComponentAffineCalibrator,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    waveform: torch.Tensor,
) -> torch.Tensor:
    normalized = predictor(waveform)
    raw = denormalize_components(normalized, target_mean, target_scale)
    return calibrator(raw)


def read_waveform(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"expected mono 16 kHz waveform: {path}")
    waveform = torch.from_numpy(audio.copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {path}")
    return waveform


def safe_case_name(spec: PanelSpec) -> str:
    return re.sub(r"[^0-9A-Za-z._ÄÖÅäöåÜüÉé_-]", "_", spec.case_id)


def read_tau_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    wanted = {spec.speaker_id for spec in panel_specs()}
    selected = {row["speaker_id"]: row for row in rows if row["speaker_id"] in wanted}
    if set(selected) != wanted:
        raise ValueError(f"TAU manifest misses panel speakers: {sorted(wanted - set(selected))}")
    for speaker_id, row in selected.items():
        if row.get("label") != "patient":
            raise ValueError(f"fresh panel speaker is not pathological: {speaker_id}")
        expected_group = {
            spec.sample_group
            for spec in panel_specs()
            if spec.speaker_id == speaker_id
        }
        if expected_group != {row.get("sample_group")}:
            raise ValueError(
                f"fresh panel severity drift for {speaker_id}: "
                f"{row.get('sample_group')} != {sorted(expected_group)}"
            )
        for view in ("cs", "sv"):
            source = Path(row[f"{view}_audio_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
    return selected


def read_fixed_recipes(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("fixed recipe row is not an object")
                rows.append(value)
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 fixed test recipes, found {len(rows)}")
    return rows


def recipe_wds_row(recipe: dict[str, Any], role: str) -> dict[str, Any]:
    row = dict(recipe[role])
    row["_root"] = str(row.get("_shard_dir", ""))
    if not row["_root"]:
        raise ValueError(f"fixed recipe lacks {role} shard directory")
    return row


def run_exact_batch(
    items: list[dict[str, str]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    request = {
        "items": items,
        "components": list(AVQI_COMPONENT_NAMES),
        "step_versions": STEP_VERSIONS,
    }
    result = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, str(avqi_code_root)],
        input=json.dumps(request, ensure_ascii=False),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_FRESH_PANEL_EXACT_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"exact Praat emitted {len(lines)} result records")
    payload = json.loads(lines[0][len(marker) :])
    observed_ids = [row["id"] for row in payload["rows"]]
    expected_ids = [item["id"] for item in items]
    if observed_ids != expected_ids:
        raise ValueError("exact Praat result order or coverage differs")
    for row in payload["rows"]:
        values = np.asarray(
            [row["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite exact components for {row['id']}")
    return payload


def exact_index(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        row["id"]: np.asarray(
            [row["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        for row in payload["rows"]
    }


def candidate_from_gradient(
    base: torch.Tensor,
    gradient: torch.Tensor,
    alpha: float,
) -> torch.Tensor | None:
    if alpha == 0.0:
        return base.detach().clone()
    gradient_rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient_rms) or float(gradient_rms) <= 1e-15:
        return None
    base_rms = base.square().mean().sqrt().clamp_min(1e-12)
    candidate = base.detach() - alpha * base_rms * gradient / gradient_rms
    if not torch.isfinite(candidate).all() or float(candidate.abs().max()) >= 0.999:
        return None
    return candidate


def component_fields(
    row: dict[str, Any],
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    scales: np.ndarray,
) -> None:
    for index, component in enumerate(AVQI_COMPONENT_NAMES):
        before_gap = abs(float(before[index] - target[index]))
        after_gap = abs(float(after[index] - target[index]))
        row[f"exact_target_{component}"] = float(target[index])
        row[f"exact_before_{component}"] = float(before[index])
        row[f"exact_after_{component}"] = float(after[index])
        row[f"exact_absolute_gap_before_{component}"] = before_gap
        row[f"exact_absolute_gap_after_{component}"] = after_gap
        row[f"exact_normalized_gap_reduction_{component}"] = (
            before_gap - after_gap
        ) / max(float(scales[index]), 1e-8)


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values)))


def improvement_fraction_or_none(before: list[float], after: list[float]) -> float | None:
    if not before:
        return None
    return float(np.mean(np.asarray(after) < np.asarray(before)))


def summarize_slice(
    rows: list[dict[str, Any]],
    *,
    require_gate: bool,
) -> dict[str, Any]:
    material_key = f"material_{OPTIMIZED_COMPONENT}_gap"
    exact_before_key = f"exact_absolute_gap_before_{OPTIMIZED_COMPONENT}"
    exact_after_key = f"exact_absolute_gap_after_{OPTIMIZED_COMPONENT}"
    exact_reduction_key = (
        f"exact_normalized_gap_reduction_{OPTIMIZED_COMPONENT}"
    )
    material = [row for row in rows if row[material_key]]
    before = [row[exact_before_key] for row in material]
    after = [row[exact_after_key] for row in material]
    reductions = [row[exact_reduction_key] for row in material]
    improvement = improvement_fraction_or_none(before, after)
    median_reduction = median_or_none(reductions)
    gates = {
        "material_case_present": len(material) >= 1,
        "improvement_fraction_ge_half": improvement is not None and improvement >= 0.5,
        "median_normalized_reduction_nonnegative": (
            median_reduction is not None and median_reduction >= 0.0
        ),
    }
    return {
        "rows": len(rows),
        "material_rows": len(material),
        "improvement_fraction_material": improvement,
        "median_normalized_gap_reduction_material": median_reduction,
        "gates": gates,
        "decision": (
            "PASS"
            if not require_gate or all(gates.values())
            else "FAIL"
        ),
    }


def slice_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = {
        "view=cs": lambda row: row["view"] == "cs",
        "view=sv": lambda row: row["view"] == "sv",
        "severity=pathological_mild": (
            lambda row: row["sample_group"] == "pathological_mild"
        ),
        "severity=pathological_severe": (
            lambda row: row["sample_group"] == "pathological_severe"
        ),
        "condition=rir_only": lambda row: row["condition"] == "rir_only",
        "condition=snr20": lambda row: row["condition"] == "snr20",
        "condition=snr10": lambda row: row["condition"] == "snr10",
    }
    return {
        name: summarize_slice(
            [row for row in rows if predicate(row)],
            require_gate=name in REQUIRED_FINAL_SLICES,
        )
        for name, predicate in predicates.items()
    }


def summarize_rows(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    coverage_passed = len(rows) == expected_rows
    material_key = f"material_{OPTIMIZED_COMPONENT}_gap"
    exact_before_key = f"exact_absolute_gap_before_{OPTIMIZED_COMPONENT}"
    exact_after_key = f"exact_absolute_gap_after_{OPTIMIZED_COMPONENT}"
    exact_reduction_key = (
        f"exact_normalized_gap_reduction_{OPTIMIZED_COMPONENT}"
    )
    proxy_before_key = f"proxy_absolute_gap_before_{OPTIMIZED_COMPONENT}"
    proxy_after_key = f"proxy_absolute_gap_after_{OPTIMIZED_COMPONENT}"
    proxy_reduction_key = (
        f"proxy_normalized_gap_reduction_{OPTIMIZED_COMPONENT}"
    )
    material = [row for row in rows if row[material_key]]
    exact_before = [row[exact_before_key] for row in material]
    exact_after = [row[exact_after_key] for row in material]
    exact_reductions = [row[exact_reduction_key] for row in material]
    proxy_before = [row[proxy_before_key] for row in material]
    proxy_after = [row[proxy_after_key] for row in material]
    proxy_reductions = [row[proxy_reduction_key] for row in material]
    exact_improvement = improvement_fraction_or_none(exact_before, exact_after)
    proxy_improvement = improvement_fraction_or_none(proxy_before, proxy_after)
    exact_median_reduction = median_or_none(exact_reductions)
    proxy_median_reduction = median_or_none(proxy_reductions)
    companion_reduction = (
        None
        if COMPANION_COMPONENT is None
        else median_or_none(
            [
                row[f"exact_normalized_gap_reduction_{COMPANION_COMPONENT}"]
                for row in rows
            ]
        )
    )
    nonselected = {}
    for component in AVQI_COMPONENT_NAMES:
        if component == OPTIMIZED_COMPONENT:
            continue
        values = [
            -row[f"exact_normalized_gap_reduction_{component}"] for row in rows
        ]
        median_increase = median_or_none(values)
        nonselected[component] = {
            "median_normalized_gap_increase": median_increase,
            "decision": (
                "PASS"
                if median_increase is not None
                and median_increase <= NONSELECTED_MEDIAN_INCREASE_GATE
                else "FAIL"
            ),
        }
    worst_residual = max((row["residual_rms_db"] for row in rows), default=math.inf)
    minimum_cosine = min((row["cosine_similarity"] for row in rows), default=-math.inf)
    maximum_clip = max((row["clip_fraction"] for row in rows), default=math.inf)
    pathology = aggregate_pathology_guardrails(rows) if coverage_passed else {
        "decision": "FAIL",
        "reason": "incomplete candidate coverage",
    }
    denoising = aggregate_denoising(rows) if coverage_passed else {
        "decision": "FAIL",
        "reason": "incomplete candidate coverage",
    }
    slices = slice_summaries(rows) if coverage_passed else {}
    gates: dict[str, bool] = {
        "complete_case_coverage": coverage_passed,
        "material_cases_ge_5": len(material) >= 5,
        "exact_improvement_fraction_ge_0_80": (
            exact_improvement is not None
            and exact_improvement >= IMPROVEMENT_FRACTION_GATE
        ),
        "exact_median_normalized_reduction_ge_0_02": (
            exact_median_reduction is not None
            and exact_median_reduction >= MEDIAN_REDUCTION_GATE
        ),
        "proxy_improvement_fraction_ge_0_80": (
            proxy_improvement is not None
            and proxy_improvement >= PROXY_IMPROVEMENT_FRACTION_GATE
        ),
        "proxy_median_normalized_reduction_ge_0_02": (
            proxy_median_reduction is not None
            and proxy_median_reduction >= MEDIAN_REDUCTION_GATE
        ),
        "all_nonselected_component_medians_within_0_05": all(
            item["decision"] == "PASS" for item in nonselected.values()
        ),
        "residual_rms_le_minus_50_db": worst_residual <= RESIDUAL_CEILING_DB,
        "cosine_similarity_ge_0_99999": minimum_cosine >= MINIMUM_COSINE,
        "clip_fraction_zero": maximum_clip <= MAXIMUM_CLIP_FRACTION,
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
    }
    if COMPANION_COMPONENT is not None:
        gates[f"exact_{COMPANION_COMPONENT}_median_reduction_nonnegative"] = (
            companion_reduction is not None and companion_reduction >= 0.0
        )
    exact_summary_key = f"exact_{OPTIMIZED_COMPONENT}"
    proxy_summary_key = f"proxy_{OPTIMIZED_COMPONENT}"
    summary = {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "material_rows": len(material),
        exact_summary_key: {
            "improvement_fraction_material": exact_improvement,
            "median_normalized_gap_reduction_material": exact_median_reduction,
        },
        proxy_summary_key: {
            "improvement_fraction_material": proxy_improvement,
            "median_normalized_gap_reduction_material": proxy_median_reduction,
        },
        "companion_component": COMPANION_COMPONENT,
        "exact_companion_median_normalized_gap_reduction": companion_reduction,
        "nonselected_components": nonselected,
        "safety": {
            "worst_residual_rms_db": worst_residual,
            "minimum_cosine_similarity": minimum_cosine,
            "maximum_clip_fraction": maximum_clip,
        },
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "slices": slices,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }
    if COMPANION_COMPONENT is not None:
        summary[
            f"exact_{COMPANION_COMPONENT}_median_normalized_gap_reduction"
        ] = companion_reduction
    return summary


def choose_calibration_alpha(
    summaries: dict[float, dict[str, Any]],
) -> float | None:
    passing = [
        alpha
        for alpha, summary in summaries.items()
        if alpha > 0.0 and summary.get("decision") == "PASS"
    ]
    if not passing:
        return None
    exact_summary_key = f"exact_{OPTIMIZED_COMPONENT}"
    return min(
        passing,
        key=lambda alpha: (
            -float(
                summaries[alpha][exact_summary_key]
                ["median_normalized_gap_reduction_material"]
            ),
            alpha,
        ),
    )


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    required_slices = {
        name: summary["slices"].get(name, {"decision": "FAIL"})
        for name in REQUIRED_FINAL_SLICES
    }
    slice_gate = all(item["decision"] == "PASS" for item in required_slices.values())
    output = dict(summary)
    output["required_slice_gate"] = {
        "slices": required_slices,
        "decision": "PASS" if slice_gate else "FAIL",
    }
    output["decision"] = (
        "PASS" if summary["decision"] == "PASS" and slice_gate else "FAIL"
    )
    return output


def write_completion(
    output_dir: Path,
    report: dict[str, Any],
    summary_text: str,
) -> None:
    report_path = output_dir / "fresh_panel_report.json"
    summary_path = output_dir / "SUMMARY.md"
    write_json(report_path, report)
    summary_path.write_text(summary_text, encoding="utf-8")
    receipt = {
        "decision": report["decision"],
        "authoritative_training_decision": report[
            "authoritative_training_decision"
        ],
        "selected_alpha": report.get("selected_alpha"),
        "generator_loaded_for_inference": True,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "final_exact_panel_opened": report["final_exact_panel_opened"],
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
            "panel_contract.json": sha256_file(output_dir / "panel_contract.json"),
            "alpha_selection.json": sha256_file(output_dir / "alpha_selection.json"),
        },
    }
    if (output_dir / "final_panel_seal.json").is_file():
        receipt["artifact_sha256"]["final_panel_seal.json"] = sha256_file(
            output_dir / "final_panel_seal.json"
        )
    for name in ("calibration_alpha_results.csv", "final_results.csv"):
        path = output_dir / name
        if path.is_file():
            receipt["artifact_sha256"][name] = sha256_file(path)
    write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


def markdown_summary(report: dict[str, Any]) -> str:
    calibration = report["calibration"]
    final = report.get("final")
    exact_summary_key = f"exact_{OPTIMIZED_COMPONENT}"
    lines = [
        f"# Route C {DISPLAY_NAME} fresh speaker-disjoint waveform pilot",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        (
            f"This pilot isolates {DISPLAY_NAME}. S3_500 was used only for "
            "frozen inference; no generator parameter was updated."
        ),
        "",
        "| Panel | Material cases | Exact improvement rate | Median normalized reduction |",
        "|---|---:|---:|---:|",
        (
            f"| calibration | {calibration['material_rows']} | "
            f"{calibration[exact_summary_key]['improvement_fraction_material']} | "
            f"{calibration[exact_summary_key]['median_normalized_gap_reduction_material']} |"
        ),
    ]
    if final is not None:
        lines.append(
            f"| final | {final['material_rows']} | "
            f"{final[exact_summary_key]['improvement_fraction_material']} | "
            f"{final[exact_summary_key]['median_normalized_gap_reduction_material']} |"
        )
    lines.extend(
        [
            "",
            f"Selected calibration-only alpha: `{report.get('selected_alpha')}`.",
            "",
            (
                "Exact Praat scored all six components. Full-band low-frequency, "
                "pause, airflow, residual, cosine, clipping, SNR, and SI-SDR "
                "guardrails remained outside the AVQI-compatible metric branch."
            ),
            "",
            "Formal AVQI-T2 generator training remains blocked by contract.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_pilot(args.pilot_profile)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("pilot source commit differs from repository HEAD")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    if not args.simulation_root.is_dir():
        raise FileNotFoundError(args.simulation_root)
    if not args.avqi_code_root.is_dir():
        raise FileNotFoundError(args.avqi_code_root)

    authorization = validate_authorization(args)
    source_hashes = {
        "tau_manifest": validate_file_hash(
            args.tau_manifest,
            args.tau_manifest_sha256,
            "TAU AVQI sampling manifest",
        ),
        "fixed_test_recipes": validate_file_hash(
            args.fixed_recipes,
            args.fixed_recipes_sha256,
            "fixed test recipes",
        ),
        "simulation_config": validate_file_hash(
            args.simulation_config,
            args.simulation_config_sha256,
            "phone-room simulation config",
        ),
        "simulation_source": validate_file_hash(
            args.simulation_root / "simulate_degradation.py",
            args.simulation_source_sha256,
            "simulation source",
        ),
    }
    observed_avqi_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError(
            "exact AVQI code tree drift: "
            f"{observed_avqi_tree_hash} != {args.avqi_code_tree_sha256}"
        )
    source_hashes["avqi_code_tree"] = observed_avqi_tree_hash

    specs = panel_specs()
    panel_validation = validate_panel_specs(specs)
    tau_rows = read_tau_manifest(args.tau_manifest)
    recipes = read_fixed_recipes(args.fixed_recipes)
    simulation_config = yaml.safe_load(
        args.simulation_config.read_text(encoding="utf-8")
    )
    simulation_config["stft_cfg"]["sampling_rate"] = SAMPLE_RATE

    # The simulator and SeMamba++ generator require remote-only dependencies
    # and a CLI-selected external root, so load them only in the GPU execution.
    if str(args.simulation_root) not in sys.path:
        sys.path.insert(0, str(args.simulation_root))
    from simulate_degradation import apply_degradation_with_wind
    from evaluate_avqi_component_backprop import (
        enhance_waveform,
        load_generator,
    )
    from utils import load_config

    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    target_root = waveform_root / "target_clean"
    noisy_root = waveform_root / "degraded"
    base_root = waveform_root / "s3_500_base"
    calibration_candidate_root = waveform_root / "calibration_candidates"
    final_candidate_root = waveform_root / "final_selected"
    for path in (
        target_root,
        noisy_root,
        base_root,
        calibration_candidate_root,
        final_candidate_root,
    ):
        path.mkdir(parents=True)

    reader = WdsReader()
    prepared: list[PreparedCase] = []
    try:
        for spec in specs:
            manifest_row = tau_rows[spec.speaker_id]
            source_path = Path(manifest_row[f"{spec.view}_audio_path"])
            clean = read_clean(source_path)
            recipe = recipes[spec.recipe_index]
            if recipe.get("split") != "test" or recipe.get("target_sample_rate") != SAMPLE_RATE:
                raise ValueError(f"fixed recipe contract drift at {spec.recipe_index}")
            simulation_seed = stable_seed(
                args.seed,
                f"avqi_{VERSION_LABEL}_fresh_panel_v1",
                spec.split,
                spec.speaker_id,
                spec.view,
                recipe["uid"],
            )
            rng = random.Random(simulation_seed)
            noise_row = recipe_wds_row(recipe, "noise")
            rir_row = recipe_wds_row(recipe, "rir")
            noise, noise_start = crop_or_tile(
                reader.read(noise_row),
                clean.shape[1],
                rng,
            )
            rir = reader.read(rir_row)
            selected_degradations = ["reverb"]
            snr = None
            if spec.condition.startswith("snr"):
                selected_degradations.append("noise")
                snr = int(spec.condition.removeprefix("snr"))
            degradation_config = {"snr": 20 if snr is None else snr}
            clean_output, degraded = apply_degradation_with_wind(
                simulation_config,
                clean,
                noise,
                rir,
                None,
                degradation_config,
                selected_degradations,
                seed=simulation_seed,
            )
            clean_output = match_length(clean_output, clean.shape[1]).astype(np.float32)
            degraded = match_length(degraded, clean.shape[1]).astype(np.float32)
            name = safe_case_name(spec)
            target_path = target_root / f"{name}__target_clean.wav"
            noisy_path = noisy_root / f"{name}__degraded.wav"
            base_path = base_root / f"{name}__s3_500.wav"
            sf.write(target_path, clean_output[0], SAMPLE_RATE, subtype="FLOAT")
            sf.write(noisy_path, degraded[0], SAMPLE_RATE, subtype="FLOAT")
            prepared.append(
                PreparedCase(
                    spec=spec,
                    target_path=target_path,
                    noisy_path=noisy_path,
                    base_path=base_path,
                    source_path=source_path,
                    recipe=recipe,
                    simulation_seed=simulation_seed,
                    noise_start_sample=noise_start,
                )
            )
    finally:
        reader.close()

    device = torch.device(args.device)
    generator_config = load_config(args.generator_config)
    generator = load_generator(generator_config, args.generator_checkpoint, device)
    with torch.inference_mode():
        for index, case in enumerate(prepared, start=1):
            degraded = read_waveform(case.noisy_path).to(device)
            enhanced = enhance_waveform(generator, degraded, generator_config)
            enhanced = enhanced.detach().cpu().reshape(-1)
            if not torch.isfinite(enhanced).all() or float(enhanced.abs().max()) >= 1.0:
                raise ValueError(f"invalid S3_500 output for {case.spec.case_id}")
            sf.write(
                case.base_path,
                enhanced.numpy(),
                SAMPLE_RATE,
                subtype="FLOAT",
            )
            print(f"prepared_base={index}/{len(prepared)}", flush=True)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    panel_rows = []
    for case in prepared:
        spec = case.spec
        recipe = case.recipe
        panel_rows.append(
            {
                "case_id": spec.case_id,
                "split": spec.split,
                "speaker_id": spec.speaker_id,
                "sample_group": spec.sample_group,
                "label": "patient",
                "view": spec.view,
                "condition": spec.condition,
                "recipe_index": spec.recipe_index,
                "recipe_uid": recipe["uid"],
                "recipe_seed": recipe["seed"],
                "simulation_seed": case.simulation_seed,
                "source_path": str(case.source_path.resolve()),
                "source_sha256": sha256_file(case.source_path),
                "target_path": str(case.target_path.resolve()),
                "target_sha256": sha256_file(case.target_path),
                "degraded_path": str(case.noisy_path.resolve()),
                "degraded_sha256": sha256_file(case.noisy_path),
                "base_path": str(case.base_path.resolve()),
                "base_sha256": sha256_file(case.base_path),
                "noise_shard_dir": recipe["noise"]["_shard_dir"],
                "noise_shard": recipe["noise"]["shard"],
                "noise_audio_member": recipe["noise"]["audio_member"],
                "noise_start_sample": case.noise_start_sample,
                "rir_shard_dir": recipe["rir"]["_shard_dir"],
                "rir_shard": recipe["rir"]["shard"],
                "rir_audio_member": recipe["rir"]["audio_member"],
            }
        )
    panel_contract = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": args.seed,
        "pilot_profile": ACTIVE_PILOT_PROFILE,
        "isolated_component": OPTIMIZED_COMPONENT,
        "selected_architecture": EXPECTED_ARCHITECTURE,
        "loss_target": (
            "normalized bidirectional gap to same-speaker clean pathological "
            "CS/SV target"
        ),
        "avqi_scalar_coefficient_used_for_direction": False,
        "speaker_split_before_simulation": True,
        "panel_validation": panel_validation,
        "conditions": ["rir_only", "snr20", "snr10"],
        "views": ["cs", "sv"],
        "sample_groups": ["pathological_mild", "pathological_severe"],
        "generator": {
            "candidate": "S3_500",
            "mode": "frozen_inference_only",
            "optimizer_steps": 0,
            "config": str(args.generator_config.resolve()),
            "config_sha256": args.generator_config_sha256,
            "checkpoint": str(args.generator_checkpoint.resolve()),
            "checkpoint_sha256": args.generator_checkpoint_sha256,
        },
        "metric_branch": {
            "exact_praat_is_final_judge": True,
            "avqi_compatible_preprocessing_is_metric_only": True,
            "emitted_waveform_highpass": False,
            "remove_sv_silence_with_sox": False,
        },
        "alpha_contract": {
            "grid": list(ALPHA_GRID),
            "selection_split": "calibration",
            "final_exact_unopened_until_alpha_and_waveforms_are_sealed": True,
            "one_normalized_gradient_step": True,
        },
        "authorization": authorization,
        "source_sha256": source_hashes,
        "rows": panel_rows,
    }
    write_json(args.output_dir / "panel_contract.json", panel_contract)

    predictor, calibrator, target_mean, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    scales = target_scale.detach().cpu().numpy().astype(np.float64)
    calibration_cases = [case for case in prepared if case.spec.split == "calibration"]
    calibration_candidates: dict[tuple[str, float], dict[str, Any]] = {}
    calibration_exact_items: list[dict[str, str]] = []
    for case in calibration_cases:
        calibration_exact_items.extend(
            [
                {
                    "id": f"target:{case.spec.case_id}",
                    "path": str(case.target_path.resolve()),
                    "view": case.spec.view,
                },
                {
                    "id": f"base:{case.spec.case_id}",
                    "path": str(case.base_path.resolve()),
                    "view": case.spec.view,
                },
            ]
        )
        target = read_waveform(case.target_path).to(device)
        base = read_waveform(case.base_path).to(device).requires_grad_(True)
        with torch.inference_mode():
            target_proxy = predict_components(
                predictor,
                calibrator,
                target_mean,
                target_scale,
                target,
            ).detach()
        base_proxy = predict_components(
            predictor,
            calibrator,
            target_mean,
            target_scale,
            base,
        )
        loss = (
            (base_proxy[0, COMPONENT_INDEX] - target_proxy[0, COMPONENT_INDEX])
            / target_scale[COMPONENT_INDEX].clamp_min(1e-8)
        ).square()
        gradient = torch.autograd.grad(loss, base)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"non-finite {DISPLAY_NAME} gradient: {case.spec.case_id}"
            )
        gradient_rms = float(gradient.square().mean().sqrt())
        for alpha_index, alpha in enumerate(ALPHA_GRID):
            candidate = candidate_from_gradient(base, gradient, alpha)
            if candidate is None:
                continue
            if alpha == 0.0:
                candidate_path = case.base_path
            else:
                candidate_path = calibration_candidate_root / (
                    f"alpha_{alpha_index:02d}__{safe_case_name(case.spec)}.wav"
                )
                sf.write(
                    candidate_path,
                    candidate.detach().cpu().numpy(),
                    SAMPLE_RATE,
                    subtype="PCM_24",
                )
            stored = read_waveform(candidate_path)
            with torch.inference_mode():
                stored_proxy = predict_components(
                    predictor,
                    calibrator,
                    target_mean,
                    target_scale,
                    stored.to(device),
                ).detach().cpu()[0].numpy()
            calibration_candidates[(case.spec.case_id, alpha)] = {
                "path": candidate_path,
                "proxy_target": target_proxy.cpu()[0].numpy(),
                "proxy_before": base_proxy.detach().cpu()[0].numpy(),
                "proxy_after": stored_proxy,
                "gradient_rms": gradient_rms,
            }
            calibration_exact_items.append(
                {
                    "id": f"candidate:{alpha_index}:{case.spec.case_id}",
                    "path": str(candidate_path.resolve()),
                    "view": case.spec.view,
                }
            )
        print(f"calibration_gradient={case.spec.case_id}", flush=True)

    calibration_exact = run_exact_batch(
        calibration_exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    calibration_exact_by_id = exact_index(calibration_exact)
    calibration_rows_by_alpha: dict[float, list[dict[str, Any]]] = {
        alpha: [] for alpha in ALPHA_GRID
    }
    for case in calibration_cases:
        target_exact = calibration_exact_by_id[f"target:{case.spec.case_id}"]
        base_exact = calibration_exact_by_id[f"base:{case.spec.case_id}"]
        target_waveform = read_waveform(case.target_path)
        base_waveform = read_waveform(case.base_path)
        for alpha_index, alpha in enumerate(ALPHA_GRID):
            candidate_record = calibration_candidates.get((case.spec.case_id, alpha))
            if candidate_record is None:
                continue
            candidate_exact = calibration_exact_by_id[
                f"candidate:{alpha_index}:{case.spec.case_id}"
            ]
            candidate_waveform = read_waveform(candidate_record["path"])
            row: dict[str, Any] = {
                "case_id": case.spec.case_id,
                "split": case.spec.split,
                "speaker_id": case.spec.speaker_id,
                "sample_group": case.spec.sample_group,
                "view": case.spec.view,
                "condition": case.spec.condition,
                "alpha": alpha,
                "candidate_path": str(candidate_record["path"].resolve()),
                "candidate_sha256": sha256_file(candidate_record["path"]),
                "gradient_rms": candidate_record["gradient_rms"],
            }
            proxy_target = candidate_record["proxy_target"]
            proxy_before = candidate_record["proxy_before"]
            proxy_after = candidate_record["proxy_after"]
            proxy_before_key = f"proxy_absolute_gap_before_{OPTIMIZED_COMPONENT}"
            proxy_after_key = f"proxy_absolute_gap_after_{OPTIMIZED_COMPONENT}"
            proxy_reduction_key = (
                f"proxy_normalized_gap_reduction_{OPTIMIZED_COMPONENT}"
            )
            row[proxy_before_key] = abs(
                float(
                    proxy_before[COMPONENT_INDEX]
                    - proxy_target[COMPONENT_INDEX]
                )
            )
            row[proxy_after_key] = abs(
                float(
                    proxy_after[COMPONENT_INDEX]
                    - proxy_target[COMPONENT_INDEX]
                )
            )
            row[proxy_reduction_key] = (
                row[proxy_before_key] - row[proxy_after_key]
            ) / max(float(scales[COMPONENT_INDEX]), 1e-8)
            component_fields(row, target_exact, base_exact, candidate_exact, scales)
            row[f"material_{OPTIMIZED_COMPONENT}_gap"] = (
                row[f"exact_absolute_gap_before_{OPTIMIZED_COMPONENT}"]
                / max(float(scales[COMPONENT_INDEX]), 1e-8)
                > MATERIAL_GAP_THRESHOLD
            )
            row.update(waveform_safety(base_waveform, candidate_waveform))
            row.update(
                full_band_pathology_guardrails(
                    target_waveform,
                    base_waveform,
                    candidate_waveform,
                )
            )
            calibration_rows_by_alpha[alpha].append(row)

    calibration_summaries = {
        alpha: summarize_rows(rows, expected_rows=len(calibration_cases))
        for alpha, rows in calibration_rows_by_alpha.items()
    }
    selected_alpha = choose_calibration_alpha(calibration_summaries)
    alpha_selection = {
        "selection_split": "calibration",
        "calibration_speakers": panel_validation["split_speakers"]["calibration"],
        "exact_scorer_versions": {
            "parselmouth": calibration_exact["parselmouth_version"],
            "praat": calibration_exact["praat_version"],
        },
        "alpha_grid": list(ALPHA_GRID),
        "summaries": {str(alpha): value for alpha, value in calibration_summaries.items()},
        "selection_rule": (
            "among nonzero alphas passing every calibration exact, proxy, "
            "non-target, full-band, residual, and denoising gate, maximize "
            f"median exact {DISPLAY_NAME} normalized gap reduction; tie to "
            "the smaller alpha"
        ),
        "selected_alpha": selected_alpha,
        "decision": "PASS" if selected_alpha is not None else "FAIL",
    }
    write_json(args.output_dir / "alpha_selection.json", alpha_selection)
    calibration_csv_rows = [
        row
        for alpha in ALPHA_GRID
        for row in calibration_rows_by_alpha[alpha]
    ]
    write_csv(args.output_dir / "calibration_alpha_results.csv", calibration_csv_rows)

    if selected_alpha is None:
        exact_summary_key = f"exact_{OPTIMIZED_COMPONENT}"
        report = {
            "decision": CALIBRATION_NO_GO_DECISION,
            "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
            "final_exact_panel_opened": False,
            "selected_alpha": None,
            "calibration": max(
                calibration_summaries.values(),
                key=lambda item: (
                    item[exact_summary_key][
                        "median_normalized_gap_reduction_material"
                    ]
                    if item[exact_summary_key][
                        "median_normalized_gap_reduction_material"
                    ]
                    is not None
                    else -math.inf
                ),
            ),
            "final": None,
            "generator_loaded_for_inference": True,
            "generator_optimizer_steps": 0,
            "formal_pathology_training_submitted": False,
            "authorization": authorization,
        }
        write_completion(args.output_dir, report, markdown_summary(report))
        return

    calibration_summary = calibration_summaries[selected_alpha]
    final_cases = [case for case in prepared if case.spec.split == "final"]
    final_candidates: dict[str, dict[str, Any]] = {}
    for case in final_cases:
        target = read_waveform(case.target_path).to(device)
        base = read_waveform(case.base_path).to(device).requires_grad_(True)
        with torch.inference_mode():
            target_proxy = predict_components(
                predictor,
                calibrator,
                target_mean,
                target_scale,
                target,
            ).detach()
        base_proxy = predict_components(
            predictor,
            calibrator,
            target_mean,
            target_scale,
            base,
        )
        loss = (
            (base_proxy[0, COMPONENT_INDEX] - target_proxy[0, COMPONENT_INDEX])
            / target_scale[COMPONENT_INDEX].clamp_min(1e-8)
        ).square()
        gradient = torch.autograd.grad(loss, base)[0]
        candidate = candidate_from_gradient(base, gradient, selected_alpha)
        if candidate is None:
            raise RuntimeError(
                f"selected alpha is invalid on sealed final case: {case.spec.case_id}"
            )
        candidate_path = final_candidate_root / (
            f"{safe_case_name(case.spec)}__{VERSION_LABEL}.wav"
        )
        sf.write(
            candidate_path,
            candidate.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        stored = read_waveform(candidate_path)
        with torch.inference_mode():
            stored_proxy = predict_components(
                predictor,
                calibrator,
                target_mean,
                target_scale,
                stored.to(device),
            ).detach().cpu()[0].numpy()
        final_candidates[case.spec.case_id] = {
            "path": candidate_path,
            "proxy_target": target_proxy.cpu()[0].numpy(),
            "proxy_before": base_proxy.detach().cpu()[0].numpy(),
            "proxy_after": stored_proxy,
            "gradient_rms": float(gradient.square().mean().sqrt()),
        }

    final_panel_seal = {
        "schema_version": FINAL_SEAL_SCHEMA_VERSION,
        "selected_alpha": selected_alpha,
        "alpha_selection_sha256": sha256_file(args.output_dir / "alpha_selection.json"),
        "exact_final_scoring_started_after_this_seal": True,
        "rows": [
            {
                "case_id": case.spec.case_id,
                "speaker_id": case.spec.speaker_id,
                "view": case.spec.view,
                "condition": case.spec.condition,
                "target_path": str(case.target_path.resolve()),
                "target_sha256": sha256_file(case.target_path),
                "base_path": str(case.base_path.resolve()),
                "base_sha256": sha256_file(case.base_path),
                "candidate_path": str(
                    final_candidates[case.spec.case_id]["path"].resolve()
                ),
                "candidate_sha256": sha256_file(
                    final_candidates[case.spec.case_id]["path"]
                ),
            }
            for case in final_cases
        ],
    }
    write_json(args.output_dir / "final_panel_seal.json", final_panel_seal)

    final_exact_items = []
    for case in final_cases:
        final_exact_items.extend(
            [
                {
                    "id": f"target:{case.spec.case_id}",
                    "path": str(case.target_path.resolve()),
                    "view": case.spec.view,
                },
                {
                    "id": f"base:{case.spec.case_id}",
                    "path": str(case.base_path.resolve()),
                    "view": case.spec.view,
                },
                {
                    "id": f"candidate:{case.spec.case_id}",
                    "path": str(
                        final_candidates[case.spec.case_id]["path"].resolve()
                    ),
                    "view": case.spec.view,
                },
            ]
        )
    final_exact = run_exact_batch(
        final_exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    if (
        final_exact["parselmouth_version"]
        != calibration_exact["parselmouth_version"]
        or final_exact["praat_version"] != calibration_exact["praat_version"]
    ):
        raise ValueError("exact scorer version drift between calibration and final")
    final_exact_by_id = exact_index(final_exact)
    final_rows: list[dict[str, Any]] = []
    for case in final_cases:
        candidate_record = final_candidates[case.spec.case_id]
        target_exact = final_exact_by_id[f"target:{case.spec.case_id}"]
        base_exact = final_exact_by_id[f"base:{case.spec.case_id}"]
        candidate_exact = final_exact_by_id[f"candidate:{case.spec.case_id}"]
        target_waveform = read_waveform(case.target_path)
        base_waveform = read_waveform(case.base_path)
        candidate_waveform = read_waveform(candidate_record["path"])
        row = {
            "case_id": case.spec.case_id,
            "split": case.spec.split,
            "speaker_id": case.spec.speaker_id,
            "sample_group": case.spec.sample_group,
            "view": case.spec.view,
            "condition": case.spec.condition,
            "alpha": selected_alpha,
            "candidate_path": str(candidate_record["path"].resolve()),
            "candidate_sha256": sha256_file(candidate_record["path"]),
            "gradient_rms": candidate_record["gradient_rms"],
        }
        proxy_target = candidate_record["proxy_target"]
        proxy_before = candidate_record["proxy_before"]
        proxy_after = candidate_record["proxy_after"]
        proxy_before_key = f"proxy_absolute_gap_before_{OPTIMIZED_COMPONENT}"
        proxy_after_key = f"proxy_absolute_gap_after_{OPTIMIZED_COMPONENT}"
        proxy_reduction_key = (
            f"proxy_normalized_gap_reduction_{OPTIMIZED_COMPONENT}"
        )
        row[proxy_before_key] = abs(
            float(
                proxy_before[COMPONENT_INDEX]
                - proxy_target[COMPONENT_INDEX]
            )
        )
        row[proxy_after_key] = abs(
            float(
                proxy_after[COMPONENT_INDEX]
                - proxy_target[COMPONENT_INDEX]
            )
        )
        row[proxy_reduction_key] = (
            row[proxy_before_key] - row[proxy_after_key]
        ) / max(float(scales[COMPONENT_INDEX]), 1e-8)
        component_fields(row, target_exact, base_exact, candidate_exact, scales)
        row[f"material_{OPTIMIZED_COMPONENT}_gap"] = (
            row[f"exact_absolute_gap_before_{OPTIMIZED_COMPONENT}"]
            / max(float(scales[COMPONENT_INDEX]), 1e-8)
            > MATERIAL_GAP_THRESHOLD
        )
        row.update(waveform_safety(base_waveform, candidate_waveform))
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        final_rows.append(row)
    write_csv(args.output_dir / "final_results.csv", final_rows)
    final_summary = finalize_summary(
        summarize_rows(final_rows, expected_rows=len(final_cases))
    )
    decision = PASS_DECISION if final_summary["decision"] == "PASS" else FAIL_DECISION
    report = {
        "decision": decision,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "training_boundary_reason": (
            f"This isolated bounded waveform pilot can promote {DISPLAY_NAME} "
            "as a Route C component, but it does not test a combined AVQI loss "
            "inside generator optimization."
        ),
        "final_exact_panel_opened": True,
        "selected_alpha": selected_alpha,
        "calibration": calibration_summary,
        "final": final_summary,
        "exact_scorer_versions": {
            "parselmouth": final_exact["parselmouth_version"],
            "praat": final_exact["praat_version"],
        },
        "generator_loaded_for_inference": True,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "authorization": authorization,
        "artifacts": {
            "panel_contract": "panel_contract.json",
            "alpha_selection": "alpha_selection.json",
            "calibration_results": "calibration_alpha_results.csv",
            "final_panel_seal": "final_panel_seal.json",
            "final_results": "final_results.csv",
        },
    }
    write_completion(args.output_dir, report, markdown_summary(report))


if __name__ == "__main__":
    main()
