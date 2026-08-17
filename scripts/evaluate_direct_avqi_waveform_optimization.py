#!/usr/bin/env python3
"""Test whether direct AVQI-component gradients improve exact Praat gaps.

This is a no-generator diagnostic. It optimizes a bounded residual on a fixed
S3_500 pathology panel, then recomputes all six exact Praat components. The
test distinguishes a useful gradient from a surrogate-only adversarial move.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    ComponentAffineCalibrator,
    PraatDifferentiableAVQIComponentEstimator,
    denormalize_components,
)


SAMPLE_RATE = 16_000
CANDIDATE = "S3_500"
CONDITION = "snr10"
SEVERITY_GROUPS = ("pathological_mild", "pathological_severe")
VIEWS = ("cs", "sv")
# Keep the first backprop test minimal and family-diverse.  Shimmer percent is
# still reported as a validated component, but is not co-weighted with HNR and
# LTAS tilt in this two-term waveform diagnostic.
OPTIMIZED_COMPONENTS = ("hnr", "tilt")
EXACT_IMPROVEMENT_FRACTION_GATE = 2.0 / 3.0
SURROGATE_IMPROVEMENT_FRACTION_GATE = 0.75
NORMALIZED_GAP_REDUCTION_GATE = 0.02
SELECTED_TOTAL_RELATIVE_REDUCTION_GATE = 0.10
NONSELECTED_MEDIAN_INCREASE_GATE = 0.05
MINIMUM_COSINE_GATE = 0.99
MAXIMUM_CLIP_FRACTION = 1e-4
FRAME_LENGTH = 400
FRAME_HOP = 160
FULL_BAND_FREQUENCY_RANGES = {
    "low_20_80hz": (20.0, 80.0),
    "low_80_300hz": (80.0, 300.0),
}
AIRFLOW_PROXY_FREQUENCY_RANGE = (500.0, 4_000.0)
LOW_ENERGY_QUANTILE = 0.25
PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX = 0.50
PATHOLOGY_DB_WORST_GAP_INCREASE_MAX = 1.50
AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX = 0.05
AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX = 0.10
PAUSE_F1_MEDIAN_DECREASE_MAX = 0.05
PAUSE_F1_WORST_DECREASE_MAX = 0.15
GUARDRAIL_PASS_FRACTION_MIN = 2.0 / 3.0
DENOISING_MEDIAN_CHANGE_MIN_DB = -0.10
DENOISING_WORST_CHANGE_MIN_DB = -0.50
STEP_VERSIONS = {
    "highpass": "praat",
    "read_and_resample": "praat",
    "sv_length_norm": "praat",
    "cs_voiced_segments": "praat",
    "concatenate": "praat",
    "cpps": "praat",
    "slope": "praat",
    "tilt": "praat",
    "shimmer": "praat",
    "hnr": "praat",
    "pitch": "praat",
}


@dataclass(frozen=True)
class Case:
    speaker_id: str
    view: str
    sample_group: str
    path: Path
    reference_path: Path
    target: torch.Tensor
    exact_before: torch.Tensor

    @property
    def case_id(self) -> str:
        return f"{self.sample_group}__{self.speaker_id}__{self.view}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-exact-csv", type=Path, required=True)
    parser.add_argument("--external-exact-csv-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--authorization-consensus", type=Path, required=True)
    parser.add_argument("--authorization-consensus-sha256", required=True)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--screen-report-sha256", required=True)
    parser.add_argument(
        "--screen-completion-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--screen-completion-receipt-sha256", required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--speakers-per-severity", type=int, default=3)
    parser.add_argument("--speaker-offset", type=int, default=0)
    parser.add_argument("--expected-cases", type=int, default=12)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate-scale", type=float, default=0.05)
    parser.add_argument("--fidelity-weight", type=float, default=0.05)
    parser.add_argument("--residual-ceiling-db", type=float, default=-30.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")
    return actual


def validate_route_c_authorization(
    consensus_path: Path,
    consensus_sha256: str,
    screen_report_path: Path,
    screen_report_sha256: str,
    screen_completion_receipt_path: Path,
    screen_completion_receipt_sha256: str,
    predictor_checkpoint_path: Path,
    predictor_checkpoint_sha256: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    consensus_hash = validate_file_hash(
        consensus_path,
        consensus_sha256,
        "Route C multi-seed consensus",
    )
    screen_hash = validate_file_hash(
        screen_report_path,
        screen_report_sha256,
        "Route C screen report",
    )
    screen_receipt_hash = validate_file_hash(
        screen_completion_receipt_path,
        screen_completion_receipt_sha256,
        "Route C screen completion receipt",
    )
    predictor_hash = validate_file_hash(
        predictor_checkpoint_path,
        predictor_checkpoint_sha256,
        "Route C predictor checkpoint",
    )
    consensus = load_json_object(consensus_path)
    screen = load_json_object(screen_report_path)
    screen_receipt = load_json_object(screen_completion_receipt_path)

    if consensus.get("schema_version") != "avqi-component-multiseed-consensus-v2":
        raise ValueError("unexpected Route C consensus schema")
    if consensus.get("route_scope") != "direct_only":
        raise ValueError("waveform pilot requires a direct-only consensus")
    if consensus.get("active_routes") != ["direct_differentiable_estimator"]:
        raise ValueError("waveform pilot consensus contains a non-Route-C route")
    if consensus.get("generator_optimizer_steps") != 0:
        raise ValueError("authorization consensus contains generator updates")
    if consensus.get("bounded_waveform_pilot_submitted") is not False:
        raise ValueError("authorization consensus already submitted a waveform pilot")
    if consensus.get("formal_pathology_training_submitted") is not False:
        raise ValueError("authorization consensus contains formal pathology training")
    promotion = consensus.get("promotion", {})
    if promotion.get("decision") != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT":
        raise ValueError("Route C consensus does not authorize a bounded pilot")
    if promotion.get("routes") != ["direct_differentiable_estimator"]:
        raise ValueError("Route C promotion route differs")
    if tuple(promotion.get("components", ())) != OPTIMIZED_COMPONENTS:
        raise ValueError("Route C promotion components differ")
    route_consensus = consensus.get("routes", {}).get(
        "direct_differentiable_estimator",
        {},
    )
    if route_consensus.get("decision") != "RELIABLE":
        raise ValueError("Route C multi-seed result is not reliable")
    if tuple(route_consensus.get("consensus_components", ())) != OPTIMIZED_COMPONENTS:
        raise ValueError("Route C consensus component list differs")
    pass_counts = route_consensus.get("component_pass_counts", {})
    if any(pass_counts.get(component) != 3 for component in OPTIMIZED_COMPONENTS):
        raise ValueError("Route C components did not pass all three locked seeds")
    if consensus.get("source_report_sha256", {}).get("screen") != screen_hash:
        raise ValueError("Route C consensus does not bind the supplied screen report")
    consensus_screen_path = Path(consensus.get("screen_report", "")).resolve()
    if consensus_screen_path != screen_report_path.resolve():
        raise ValueError("Route C consensus screen path differs")

    expected_screen_decision = (
        "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
    )
    if screen.get("decision") != expected_screen_decision:
        raise ValueError("Route C screen is incomplete")
    if screen.get("generator_optimizer_steps") != 0:
        raise ValueError("Route C screen contains generator updates")
    if screen.get("bounded_waveform_pilot_submitted") is not False:
        raise ValueError("Route C screen already submitted a waveform pilot")
    if screen.get("formal_pathology_training_submitted") is not False:
        raise ValueError("Route C screen contains formal pathology training")
    if screen.get("contract", {}).get("route_scope") != "direct_only":
        raise ValueError("Route C screen scope differs")
    route = screen.get("routes", {}).get("direct_differentiable_estimator", {})
    if route.get("selected_architecture") != "direct_praat_hard_v2":
        raise ValueError("Route C selected estimator differs")
    if route.get("decision") != "ELIGIBLE_FOR_MULTISEED_CONFIRMATION":
        raise ValueError("Route C screen did not pass its scorer gates")
    if tuple(route.get("eligible_components", ())) != OPTIMIZED_COMPONENTS:
        raise ValueError("Route C screen eligible components differ")
    gradient = route.get("gradient", {})
    if gradient.get("decision") != "PASS":
        raise ValueError("Route C screen gradient gate failed")
    component_gradients = gradient.get("component_input_gradients", {})
    screen_component_gradient_norms: dict[str, float] = {}
    for component in OPTIMIZED_COMPONENTS:
        item = component_gradients.get(component, {})
        norm = float(item.get("gradient_norm", math.nan))
        if item.get("decision") != "PASS" or not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"invalid authorized gradient for {component}")
        screen_component_gradient_norms[component] = norm

    if screen_receipt.get("decision") != screen["decision"]:
        raise ValueError("Route C screen receipt decision differs")
    if screen_receipt.get("route_scope") != "direct_only":
        raise ValueError("Route C screen receipt scope differs")
    if screen_receipt.get("route_c") != route["decision"]:
        raise ValueError("Route C screen receipt route decision differs")
    if tuple(screen_receipt.get("eligible_components", ())) != OPTIMIZED_COMPONENTS:
        raise ValueError("Route C screen receipt components differ")
    if screen_receipt.get("generator_optimizer_steps") != 0:
        raise ValueError("Route C screen receipt contains generator updates")
    if screen_receipt.get("bounded_waveform_pilot_submitted") is not False:
        raise ValueError("Route C screen receipt already submitted a waveform pilot")
    if screen_receipt.get("formal_pathology_training_submitted") is not False:
        raise ValueError("Route C screen receipt contains formal pathology training")
    recorded_screen_hash = screen_receipt.get("artifact_sha256", {}).get(
        "diagnostic_report.json"
    )
    if recorded_screen_hash != screen_hash:
        raise ValueError("Route C screen receipt does not bind its report")
    checkpoint_hashes = screen_receipt.get("checkpoint_sha256", {})
    if checkpoint_hashes.get(predictor_checkpoint_path.name) != predictor_hash:
        raise ValueError(
            "Route C screen receipt does not bind the predictor checkpoint"
        )
    receipt_checkpoint_dir = Path(
        screen_receipt.get("checkpoint_dir", "")
    ).resolve()
    if receipt_checkpoint_dir != predictor_checkpoint_path.parent.resolve():
        raise ValueError("Route C predictor checkpoint directory differs")

    minimum_norm = min(screen_component_gradient_norms.values())
    optimization_component_weights = {
        component: minimum_norm / screen_component_gradient_norms[component]
        for component in OPTIMIZED_COMPONENTS
    }
    authorization = {
        "decision": promotion["decision"],
        "route": "direct_differentiable_estimator",
        "components": list(OPTIMIZED_COMPONENTS),
        "consensus_sha256": consensus_hash,
        "screen_report_sha256": screen_hash,
        "screen_completion_receipt_sha256": screen_receipt_hash,
        "predictor_checkpoint_sha256": predictor_hash,
        "screen_source_commit": screen["contract"]["source_commit"],
    }
    return (
        screen_component_gradient_norms,
        optimization_component_weights,
        authorization,
    )


def repository_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def avqi_code_tree_sha256(root: Path) -> str:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".praat"}
    )
    if not files:
        raise ValueError(f"AVQI code tree contains no Python or Praat files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def component_tensor(row: dict[str, str], prefix: str) -> torch.Tensor:
    tensor = torch.tensor(
        [float(row[f"{prefix}{name}"]) for name in AVQI_COMPONENT_NAMES],
        dtype=torch.float32,
    )
    if not torch.isfinite(tensor).all():
        raise ValueError("external exact row contains a non-finite component")
    return tensor


def load_cases(
    path: Path,
    speakers_per_severity: int,
    expected_cases: int,
    speaker_offset: int = 0,
) -> list[Case]:
    if speakers_per_severity <= 0 or speaker_offset < 0:
        raise ValueError("speaker count must be positive and offset non-negative")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if row["source_type"] == "enhanced"
        and row["candidate"] == CANDIDATE
        and row["condition"] == CONDITION
        and row["view"] in VIEWS
        and row["sample_group"] in SEVERITY_GROUPS
        and row["label"] == "patient"
        and row["scoring_status"] == "ok"
    ]
    clean_rows = [
        row
        for row in all_rows
        if row["source_type"] == "clean_reference"
        and row["sample_group"] in SEVERITY_GROUPS
        and row["label"] == "patient"
        and row["scoring_status"] == "ok"
    ]
    clean_by_speaker: dict[str, dict[str, str]] = {}
    for row in clean_rows:
        speaker_id = row["speaker_id"]
        if speaker_id in clean_by_speaker:
            raise ValueError(f"duplicate clean reference row: {speaker_id}")
        clean_by_speaker[speaker_id] = row
    selected_speakers: dict[str, list[str]] = {}
    for group in SEVERITY_GROUPS:
        speakers = sorted(
            {row["speaker_id"] for row in rows if row["sample_group"] == group}
        )
        stop = speaker_offset + speakers_per_severity
        if len(speakers) < stop:
            raise ValueError(
                f"insufficient {group} speakers for offset {speaker_offset}: "
                f"{len(speakers)}"
            )
        selected_speakers[group] = speakers[speaker_offset:stop]
    required_clean_speakers = {
        speaker_id
        for speakers in selected_speakers.values()
        for speaker_id in speakers
    }
    missing_clean_speakers = required_clean_speakers - set(clean_by_speaker)
    if missing_clean_speakers:
        raise ValueError(
            "missing same-speaker clean pathological references: "
            f"{sorted(missing_clean_speakers)}"
        )
    selected = [
        row
        for row in rows
        if row["speaker_id"] in selected_speakers[row["sample_group"]]
    ]
    selected.sort(
        key=lambda row: (
            SEVERITY_GROUPS.index(row["sample_group"]),
            row["speaker_id"],
            VIEWS.index(row["view"]),
        )
    )
    keys = [
        (row["sample_group"], row["speaker_id"], row["view"])
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate fixed-panel case")
    cases = [
        Case(
            speaker_id=row["speaker_id"],
            view=row["view"],
            sample_group=row["sample_group"],
            path=Path(row[f"{row['view']}_path"]),
            reference_path=Path(
                clean_by_speaker[row["speaker_id"]][f"{row['view']}_path"]
            ),
            target=component_tensor(row, "clean_"),
            exact_before=component_tensor(row, "audio_"),
        )
        for row in selected
    ]
    if len(cases) != expected_cases:
        raise ValueError(
            f"fixed panel differs: expected {expected_cases}, found {len(cases)}"
        )
    missing = [
        str(path)
        for case in cases
        for path in (case.path, case.reference_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing fixed-panel waveforms: {missing[:3]}")
    return cases


def load_waveform(path: Path) -> tuple[torch.Tensor, str]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"expected non-empty 16 kHz mono audio: {path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {path}")
    return waveform, info.subtype


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
    if checkpoint["architecture"] != "direct_praat_hard_v2":
        raise ValueError(
            f"expected direct_praat_hard_v2, got {checkpoint['architecture']}"
        )
    if tuple(checkpoint["components"]) != AVQI_COMPONENT_NAMES:
        raise ValueError("predictor component order differs")
    predictor = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard"
    ).to(device)
    predictor.load_state_dict(checkpoint["state_dict"])
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


def project_residual(
    base: torch.Tensor,
    residual: torch.Tensor,
    maximum_rms: torch.Tensor,
) -> None:
    with torch.no_grad():
        residual.sub_(residual.mean())
        residual_rms = residual.square().mean().sqrt().clamp_min(1e-12)
        scale = torch.minimum(
            residual.new_tensor(1.0),
            maximum_rms / residual_rms,
        )
        residual.mul_(scale)
        residual.copy_((base + residual).clamp(-0.999, 0.999) - base)


def waveform_safety(
    base: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    base = base.reshape(1, -1)
    candidate = candidate.reshape(1, -1)
    if base.shape != candidate.shape:
        raise ValueError(
            f"waveform shapes differ: {base.shape} != {candidate.shape}"
        )
    base_rms = base.square().mean().sqrt().clamp_min(1e-12)
    residual_rms = (candidate - base).square().mean().sqrt().clamp_min(1e-12)
    return {
        "residual_rms_db": float(20.0 * torch.log10(residual_rms / base_rms)),
        "cosine_similarity": float(
            F.cosine_similarity(base, candidate, dim=-1)[0]
        ),
        "clip_fraction": float(
            (candidate.abs() >= 0.9989).to(torch.float32).mean()
        ),
    }


def align_reference_waveforms(
    reference: torch.Tensor,
    base: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Tail-crop only for metrics; never shift or filter the waveforms."""
    waveforms = [item.reshape(-1) for item in (reference, base, candidate)]
    minimum_samples = min(item.numel() for item in waveforms)
    maximum_samples = max(item.numel() for item in waveforms)
    if minimum_samples < FRAME_LENGTH:
        raise ValueError("waveform is too short for full-band guardrails")
    trim_samples = maximum_samples - minimum_samples
    if trim_samples > FRAME_LENGTH:
        raise ValueError(
            f"reference alignment would discard {trim_samples} samples"
        )
    aligned = tuple(item[:minimum_samples] for item in waveforms)
    return aligned[0], aligned[1], aligned[2], trim_samples


def waveform_frames(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.reshape(-1)
    if waveform.numel() < FRAME_LENGTH:
        raise ValueError("waveform is too short for framing")
    return waveform.unfold(0, FRAME_LENGTH, FRAME_HOP)


def frame_power_spectrum(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(
        FRAME_LENGTH,
        device=frames.device,
        dtype=frames.dtype,
    )
    spectrum = torch.fft.rfft(frames * window, n=512, dim=-1)
    frequencies = torch.fft.rfftfreq(
        512,
        d=1.0 / SAMPLE_RATE,
        device=frames.device,
    )
    return spectrum.abs().square(), frequencies


def band_energy_db(
    power: torch.Tensor,
    frequencies: torch.Tensor,
    lower_hz: float,
    upper_hz: float,
    frame_mask: torch.Tensor | None = None,
) -> float:
    frequency_mask = (frequencies >= lower_hz) & (frequencies < upper_hz)
    if not bool(frequency_mask.any()):
        raise ValueError(f"empty frequency band: {lower_hz}-{upper_hz} Hz")
    selected = power if frame_mask is None else power[frame_mask]
    if selected.shape[0] == 0:
        raise ValueError("empty frame mask for band energy")
    energy = selected[:, frequency_mask].mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(energy))


def spectral_flatness(
    power: torch.Tensor,
    frequencies: torch.Tensor,
    lower_hz: float,
    upper_hz: float,
    frame_mask: torch.Tensor,
) -> float:
    frequency_mask = (frequencies >= lower_hz) & (frequencies < upper_hz)
    selected = power[frame_mask][:, frequency_mask].clamp_min(1e-12)
    if selected.shape[0] == 0 or selected.shape[1] == 0:
        raise ValueError("empty airflow-proxy spectrum")
    per_frame = torch.exp(torch.log(selected).mean(dim=-1)) / selected.mean(
        dim=-1
    ).clamp_min(1e-12)
    return float(per_frame.median())


def pause_f1(reference_pause: torch.Tensor, estimate_pause: torch.Tensor) -> float:
    true_positive = (reference_pause & estimate_pause).sum().to(torch.float32)
    denominator = (
        reference_pause.sum() + estimate_pause.sum()
    ).to(torch.float32)
    return float(2.0 * true_positive / denominator.clamp_min(1.0))


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    signal = reference.square().mean().clamp_min(1e-12)
    error = (estimate - reference).square().mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(signal / error))


def si_sdr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = torch.dot(estimate, reference) / torch.dot(
        reference, reference
    ).clamp_min(1e-12)
    projected = scale * reference
    residual = estimate - projected
    ratio = projected.square().sum().clamp_min(1e-12) / residual.square().sum().clamp_min(
        1e-12
    )
    return float(10.0 * torch.log10(ratio))


def full_band_pathology_guardrails(
    reference: torch.Tensor,
    base: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    reference, base, candidate, trim_samples = align_reference_waveforms(
        reference,
        base,
        candidate,
    )
    reference_frames = waveform_frames(reference)
    base_frames = waveform_frames(base)
    candidate_frames = waveform_frames(candidate)
    reference_power, frequencies = frame_power_spectrum(reference_frames)
    base_power, _ = frame_power_spectrum(base_frames)
    candidate_power, _ = frame_power_spectrum(candidate_frames)
    output: dict[str, float | int] = {
        "guardrail_aligned_samples": reference.numel(),
        "guardrail_tail_trim_samples": trim_samples,
    }
    for name, (lower_hz, upper_hz) in FULL_BAND_FREQUENCY_RANGES.items():
        reference_db = band_energy_db(
            reference_power,
            frequencies,
            lower_hz,
            upper_hz,
        )
        base_db = band_energy_db(
            base_power,
            frequencies,
            lower_hz,
            upper_hz,
        )
        candidate_db = band_energy_db(
            candidate_power,
            frequencies,
            lower_hz,
            upper_hz,
        )
        before_gap = abs(base_db - reference_db)
        after_gap = abs(candidate_db - reference_db)
        output[f"{name}_gap_before_db"] = before_gap
        output[f"{name}_gap_after_db"] = after_gap
        output[f"{name}_gap_increase_db"] = after_gap - before_gap

    reference_frame_power = reference_frames.square().mean(dim=-1)
    base_frame_power = base_frames.square().mean(dim=-1)
    candidate_frame_power = candidate_frames.square().mean(dim=-1)
    pause_threshold = torch.quantile(
        reference_frame_power,
        LOW_ENERGY_QUANTILE,
    )
    reference_pause = reference_frame_power <= pause_threshold
    base_pause = base_frame_power <= pause_threshold
    candidate_pause = candidate_frame_power <= pause_threshold
    reference_pause_db = float(
        10.0
        * torch.log10(
            reference_frame_power[reference_pause].mean().clamp_min(1e-12)
        )
    )
    base_pause_db = float(
        10.0
        * torch.log10(base_frame_power[reference_pause].mean().clamp_min(1e-12))
    )
    candidate_pause_db = float(
        10.0
        * torch.log10(
            candidate_frame_power[reference_pause].mean().clamp_min(1e-12)
        )
    )
    pause_gap_before = abs(base_pause_db - reference_pause_db)
    pause_gap_after = abs(candidate_pause_db - reference_pause_db)
    pause_f1_before = pause_f1(reference_pause, base_pause)
    pause_f1_after = pause_f1(reference_pause, candidate_pause)
    output.update(
        {
            "reference_low_energy_frame_fraction": float(
                reference_pause.to(torch.float32).mean()
            ),
            "pause_energy_gap_before_db": pause_gap_before,
            "pause_energy_gap_after_db": pause_gap_after,
            "pause_energy_gap_increase_db": pause_gap_after - pause_gap_before,
            "pause_f1_before": pause_f1_before,
            "pause_f1_after": pause_f1_after,
            "pause_f1_change": pause_f1_after - pause_f1_before,
        }
    )

    airflow_lower, airflow_upper = AIRFLOW_PROXY_FREQUENCY_RANGE
    reference_airflow_db = band_energy_db(
        reference_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    base_airflow_db = band_energy_db(
        base_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    candidate_airflow_db = band_energy_db(
        candidate_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    reference_flatness = spectral_flatness(
        reference_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    base_flatness = spectral_flatness(
        base_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    candidate_flatness = spectral_flatness(
        candidate_power,
        frequencies,
        airflow_lower,
        airflow_upper,
        reference_pause,
    )
    airflow_gap_before = abs(base_airflow_db - reference_airflow_db)
    airflow_gap_after = abs(candidate_airflow_db - reference_airflow_db)
    flatness_gap_before = abs(base_flatness - reference_flatness)
    flatness_gap_after = abs(candidate_flatness - reference_flatness)
    output.update(
        {
            "airflow_proxy_energy_gap_before_db": airflow_gap_before,
            "airflow_proxy_energy_gap_after_db": airflow_gap_after,
            "airflow_proxy_energy_gap_increase_db": (
                airflow_gap_after - airflow_gap_before
            ),
            "airflow_proxy_flatness_gap_before": flatness_gap_before,
            "airflow_proxy_flatness_gap_after": flatness_gap_after,
            "airflow_proxy_flatness_gap_increase": (
                flatness_gap_after - flatness_gap_before
            ),
        }
    )
    snr_before = snr_db(reference, base)
    snr_after = snr_db(reference, candidate)
    si_sdr_before = si_sdr_db(reference, base)
    si_sdr_after = si_sdr_db(reference, candidate)
    output.update(
        {
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "snr_change_db": snr_after - snr_before,
            "si_sdr_before_db": si_sdr_before,
            "si_sdr_after_db": si_sdr_after,
            "si_sdr_change_db": si_sdr_after - si_sdr_before,
        }
    )
    return output


def optimize_waveform(
    base: torch.Tensor,
    target: torch.Tensor,
    predictor: PraatDifferentiableAVQIComponentEstimator,
    calibrator: ComponentAffineCalibrator,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    optimization_component_weights: dict[str, float],
    steps: int,
    learning_rate_scale: float,
    fidelity_weight: float,
    residual_ceiling_db: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if steps <= 0 or learning_rate_scale <= 0.0 or fidelity_weight < 0.0:
        raise ValueError("invalid optimization hyperparameters")
    if residual_ceiling_db >= 0.0:
        raise ValueError("residual ceiling must be below 0 dB")
    if set(optimization_component_weights) != set(OPTIMIZED_COMPONENTS):
        raise ValueError(
            "optimization component weights differ from authorized components"
        )
    if any(
        not math.isfinite(weight) or weight <= 0.0
        for weight in optimization_component_weights.values()
    ):
        raise ValueError("optimization component weights must be finite and positive")
    selected_indices = torch.tensor(
        [AVQI_COMPONENT_NAMES.index(name) for name in OPTIMIZED_COMPONENTS],
        device=base.device,
    )
    component_weights = base.new_tensor(
        [optimization_component_weights[name] for name in OPTIMIZED_COMPONENTS]
    )
    base = base.reshape(1, -1)
    target = target.to(base.device).reshape(1, -1)
    base_rms = base.square().mean().sqrt().clamp_min(1e-6)
    maximum_rms = base_rms * (10.0 ** (residual_ceiling_db / 20.0))
    residual = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam(
        [residual],
        lr=float(base_rms.detach()) * learning_rate_scale,
    )
    trajectory: list[dict[str, Any]] = []
    log_steps = {0, 1, 2, 5, 10, 20, steps}
    before = predict_components(
        predictor,
        calibrator,
        target_mean,
        target_scale,
        base,
    ).detach()
    for step in range(1, steps + 1):
        candidate = (base + residual).clamp(-0.999, 0.999)
        prediction = predict_components(
            predictor,
            calibrator,
            target_mean,
            target_scale,
            candidate,
        )
        normalized_gap = (
            prediction.index_select(-1, selected_indices)
            - target.index_select(-1, selected_indices)
        ) / target_scale.index_select(0, selected_indices).clamp_min(1e-8)
        component_element_loss = F.smooth_l1_loss(
            normalized_gap,
            torch.zeros_like(normalized_gap),
            reduction="none",
        )
        component_loss = (
            component_element_loss * component_weights.unsqueeze(0)
        ).sum() / component_weights.sum()
        residual_power_ratio = residual.square().mean() / base_rms.square()
        residual_difference = residual[:, 1:] - residual[:, :-1]
        base_difference = base[:, 1:] - base[:, :-1]
        difference_ratio = residual_difference.square().mean() / base_difference.square().mean().clamp_min(
            1e-8
        )
        fidelity_loss = residual_power_ratio + 0.25 * difference_ratio
        loss = component_loss + fidelity_weight * fidelity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if residual.grad is None or not torch.isfinite(residual.grad).all():
            raise RuntimeError("waveform optimization produced an invalid gradient")
        torch.nn.utils.clip_grad_norm_([residual], max_norm=5.0)
        optimizer.step()
        project_residual(base, residual, maximum_rms)
        if step in log_steps:
            with torch.inference_mode():
                current = predict_components(
                    predictor,
                    calibrator,
                    target_mean,
                    target_scale,
                    (base + residual).clamp(-0.999, 0.999),
                )
                trajectory.append(
                    {
                        "step": step,
                        "component_loss": float(component_loss.detach()),
                        "fidelity_loss": float(fidelity_loss.detach()),
                        "prediction": current.cpu()[0].tolist(),
                        "residual_rms_db": float(
                            20.0
                            * torch.log10(
                                residual.square().mean().sqrt()
                                / base_rms
                            ).cpu()
                        ),
                    }
                )
    optimized = (base + residual.detach()).clamp(-0.999, 0.999)
    after = predict_components(
        predictor,
        calibrator,
        target_mean,
        target_scale,
        optimized,
    ).detach()
    safety = waveform_safety(base.cpu(), optimized.cpu())
    return optimized[0], {
        "surrogate_before": before.cpu()[0].tolist(),
        "surrogate_after": after.cpu()[0].tolist(),
        "trajectory": trajectory,
        "safety": safety,
    }


def score_exact(
    path: Path,
    view: str,
    exact_python: Path,
    avqi_code_root: Path,
) -> torch.Tensor:
    # PyTorch and Praat deliberately live in separate locked environments on
    # Triton.  Keep the optimizer in semambapp and invoke exact Praat with the
    # established AVQI interpreter instead of mixing their dependencies.
    scorer = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from avqi_code import run_avqi

step_versions = json.loads(sys.argv[4])
metrics = run_avqi(
    sys.argv[2],
    sys.argv[2],
    target_sr=16000,
    speaking_type=sys.argv[3],
    step_versions=step_versions,
    remove_sv_silence_with_sox=False,
)
print("AVQI_EXACT_JSON=" + json.dumps(metrics, sort_keys=True))
"""
    result = subprocess.run(
        [
            str(exact_python),
            "-c",
            scorer,
            str(avqi_code_root),
            str(path),
            view,
            json.dumps(STEP_VERSIONS, sort_keys=True),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_EXACT_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"exact Praat emitted {len(lines)} JSON records")
    metrics = json.loads(lines[0][len(marker) :])
    tensor = torch.tensor(
        [float(metrics[name]) for name in AVQI_COMPONENT_NAMES],
        dtype=torch.float32,
    )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"exact Praat returned non-finite components: {path}")
    return tensor


def aggregate_component(
    rows: list[dict[str, Any]],
    component: str,
    domain: str,
    target_scale: float,
) -> dict[str, Any]:
    before = np.array(
        [row[f"{domain}_absolute_gap_before_{component}"] for row in rows],
        dtype=np.float64,
    )
    after = np.array(
        [row[f"{domain}_absolute_gap_after_{component}"] for row in rows],
        dtype=np.float64,
    )
    reduction = (before - after) / max(target_scale, 1e-8)
    return {
        "rows": len(rows),
        "median_gap_before": float(np.median(before)),
        "median_gap_after": float(np.median(after)),
        "improvement_fraction": float(np.mean(after < before)),
        "median_normalized_gap_reduction": float(np.median(reduction)),
        "mean_normalized_gap_reduction": float(np.mean(reduction)),
    }


def aggregate_upper_bounded_change(
    rows: list[dict[str, Any]],
    field: str,
    median_max: float,
    worst_max: float,
) -> dict[str, Any]:
    values = np.array([row[field] for row in rows], dtype=np.float64)
    gates = {
        "median_within_tolerance": float(np.median(values)) <= median_max,
        "worst_within_tolerance": float(np.max(values)) <= worst_max,
        "case_fraction_within_tolerance_ge_two_thirds": (
            float(np.mean(values <= median_max)) >= GUARDRAIL_PASS_FRACTION_MIN
        ),
    }
    return {
        "rows": len(rows),
        "median": float(np.median(values)),
        "worst": float(np.max(values)),
        "case_fraction_within_tolerance": float(np.mean(values <= median_max)),
        "median_max": median_max,
        "worst_max": worst_max,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def aggregate_pathology_guardrails(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        field: aggregate_upper_bounded_change(
            rows,
            field,
            PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX,
            PATHOLOGY_DB_WORST_GAP_INCREASE_MAX,
        )
        for field in (
            "low_20_80hz_gap_increase_db",
            "low_80_300hz_gap_increase_db",
            "pause_energy_gap_increase_db",
            "airflow_proxy_energy_gap_increase_db",
        )
    }
    metrics["airflow_proxy_flatness_gap_increase"] = (
        aggregate_upper_bounded_change(
            rows,
            "airflow_proxy_flatness_gap_increase",
            AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX,
            AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX,
        )
    )
    pause_rows = [
        {"pause_f1_decrease": -float(row["pause_f1_change"])}
        for row in rows
    ]
    metrics["pause_f1_decrease"] = aggregate_upper_bounded_change(
        pause_rows,
        "pause_f1_decrease",
        PAUSE_F1_MEDIAN_DECREASE_MAX,
        PAUSE_F1_WORST_DECREASE_MAX,
    )
    return {
        "metrics": metrics,
        "decision": (
            "PASS"
            if all(metric["decision"] == "PASS" for metric in metrics.values())
            else "FAIL"
        ),
        "interpretation_limit": (
            "Airflow is represented by low-energy 500-4000 Hz energy and "
            "spectral-flatness proxies, not a clinical airflow label."
        ),
    }


def aggregate_denoising(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, field in {
        "snr": "snr_change_db",
        "si_sdr": "si_sdr_change_db",
    }.items():
        values = np.array([row[field] for row in rows], dtype=np.float64)
        gates = {
            "median_change_ge_minus_0_10_db": (
                float(np.median(values)) >= DENOISING_MEDIAN_CHANGE_MIN_DB
            ),
            "worst_change_ge_minus_0_50_db": (
                float(np.min(values)) >= DENOISING_WORST_CHANGE_MIN_DB
            ),
            "case_fraction_non_regressed_ge_two_thirds": (
                float(np.mean(values >= DENOISING_MEDIAN_CHANGE_MIN_DB))
                >= GUARDRAIL_PASS_FRACTION_MIN
            ),
        }
        metrics[name] = {
            "rows": len(rows),
            "median_change_db": float(np.median(values)),
            "worst_change_db": float(np.min(values)),
            "case_fraction_non_regressed": float(
                np.mean(values >= DENOISING_MEDIAN_CHANGE_MIN_DB)
            ),
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    return {
        "metrics": metrics,
        "decision": (
            "PASS"
            if all(metric["decision"] == "PASS" for metric in metrics.values())
            else "FAIL"
        ),
    }


def nonselected_normalized_gap_increase(
    rows: list[dict[str, Any]],
    target_scale: torch.Tensor,
) -> float:
    nonselected = [
        component
        for component in AVQI_COMPONENT_NAMES
        if component not in OPTIMIZED_COMPONENTS
    ]
    values = [
        np.mean(
            [
                (
                    row[f"exact_absolute_gap_after_{component}"]
                    - row[f"exact_absolute_gap_before_{component}"]
                )
                / float(target_scale[AVQI_COMPONENT_NAMES.index(component)])
                for component in nonselected
            ]
        )
        for row in rows
    ]
    return float(np.median(np.asarray(values, dtype=np.float64)))


def slice_reports(
    rows: list[dict[str, Any]],
    target_scale: torch.Tensor,
) -> dict[str, Any]:
    predicates = {
        "view=cs": lambda row: row["view"] == "cs",
        "view=sv": lambda row: row["view"] == "sv",
        "severity=pathological_mild": (
            lambda row: row["sample_group"] == "pathological_mild"
        ),
        "severity=pathological_severe": (
            lambda row: row["sample_group"] == "pathological_severe"
        ),
    }
    output: dict[str, Any] = {}
    for slice_name, predicate in predicates.items():
        selected = [row for row in rows if predicate(row)]
        if not selected:
            raise ValueError(f"empty required waveform slice: {slice_name}")
        components: dict[str, Any] = {}
        for component in OPTIMIZED_COMPONENTS:
            aggregate = aggregate_component(
                selected,
                component,
                "exact",
                float(target_scale[AVQI_COMPONENT_NAMES.index(component)]),
            )
            gates = {
                "exact_improvement_fraction_ge_two_thirds": (
                    aggregate["improvement_fraction"]
                    >= EXACT_IMPROVEMENT_FRACTION_GATE
                ),
                "exact_median_normalized_reduction_ge_0_02": (
                    aggregate["median_normalized_gap_reduction"]
                    >= NORMALIZED_GAP_REDUCTION_GATE
                ),
            }
            components[component] = {
                "aggregate": aggregate,
                "gates": gates,
                "decision": "PASS" if all(gates.values()) else "FAIL",
            }
        nonselected_increase = nonselected_normalized_gap_increase(
            selected,
            target_scale,
        )
        nonselected_passed = (
            nonselected_increase <= NONSELECTED_MEDIAN_INCREASE_GATE
        )
        pathology = aggregate_pathology_guardrails(selected)
        denoising = aggregate_denoising(selected)
        output[slice_name] = {
            "rows": len(selected),
            "components": components,
            "median_nonselected_normalized_gap_increase": nonselected_increase,
            "nonselected_decision": (
                "PASS" if nonselected_passed else "FAIL"
            ),
            "pathology_guardrails": pathology,
            "denoising": denoising,
            "decision": (
                "PASS"
                if all(
                    report["decision"] == "PASS"
                    for report in components.values()
                )
                and nonselected_passed
                and pathology["decision"] == "PASS"
                and denoising["decision"] == "PASS"
                else "FAIL"
            ),
        }
    return output


def summarize(
    rows: list[dict[str, Any]],
    target_scale: torch.Tensor,
    residual_ceiling_db: float,
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {}
    for domain in ("surrogate", "exact"):
        aggregates[domain] = {
            component: aggregate_component(
                rows,
                component,
                domain,
                float(target_scale[AVQI_COMPONENT_NAMES.index(component)]),
            )
            for component in AVQI_COMPONENT_NAMES
        }
    selected_indices = [
        AVQI_COMPONENT_NAMES.index(component)
        for component in OPTIMIZED_COMPONENTS
    ]
    exact_total_before = np.array(
        [
            sum(
                row[f"exact_absolute_gap_before_{component}"]
                / float(target_scale[index])
                for component, index in zip(
                    OPTIMIZED_COMPONENTS,
                    selected_indices,
                    strict=True,
                )
            )
            for row in rows
        ]
    )
    exact_total_after = np.array(
        [
            sum(
                row[f"exact_absolute_gap_after_{component}"]
                / float(target_scale[index])
                for component, index in zip(
                    OPTIMIZED_COMPONENTS,
                    selected_indices,
                    strict=True,
                )
            )
            for row in rows
        ]
    )
    nonselected_increase = nonselected_normalized_gap_increase(
        rows,
        target_scale,
    )
    selected_relative_reduction = float(
        1.0
        - exact_total_after.mean()
        / max(exact_total_before.mean(), 1e-8)
    )
    component_gates = {}
    for component in OPTIMIZED_COMPONENTS:
        surrogate = aggregates["surrogate"][component]
        exact = aggregates["exact"][component]
        gates = {
            "surrogate_improvement_fraction_ge_0_75": (
                surrogate["improvement_fraction"]
                >= SURROGATE_IMPROVEMENT_FRACTION_GATE
            ),
            "surrogate_median_normalized_reduction_ge_0_02": (
                surrogate["median_normalized_gap_reduction"]
                >= NORMALIZED_GAP_REDUCTION_GATE
            ),
            "exact_improvement_fraction_ge_two_thirds": (
                exact["improvement_fraction"]
                >= EXACT_IMPROVEMENT_FRACTION_GATE
            ),
            "exact_median_normalized_reduction_ge_0_02": (
                exact["median_normalized_gap_reduction"]
                >= NORMALIZED_GAP_REDUCTION_GATE
            ),
        }
        component_gates[component] = {
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    safety_values = {
        "worst_residual_rms_db": max(row["residual_rms_db"] for row in rows),
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "maximum_clip_fraction": max(row["clip_fraction"] for row in rows),
        "median_nonselected_normalized_gap_increase": nonselected_increase,
        "selected_total_relative_gap_reduction": selected_relative_reduction,
    }
    safety_gates = {
        "residual_within_ceiling": (
            safety_values["worst_residual_rms_db"] <= residual_ceiling_db + 0.25
        ),
        "cosine_similarity_ge_0_99": (
            safety_values["minimum_cosine_similarity"] >= MINIMUM_COSINE_GATE
        ),
        "clip_fraction_le_1e_4": (
            safety_values["maximum_clip_fraction"] <= MAXIMUM_CLIP_FRACTION
        ),
        "nonselected_median_increase_le_0_05": (
            safety_values["median_nonselected_normalized_gap_increase"]
            <= NONSELECTED_MEDIAN_INCREASE_GATE
        ),
        "selected_total_relative_reduction_ge_0_10": (
            selected_relative_reduction
            >= SELECTED_TOTAL_RELATIVE_REDUCTION_GATE
        ),
    }
    pathology_guardrails = aggregate_pathology_guardrails(rows)
    denoising = aggregate_denoising(rows)
    slices = slice_reports(rows, target_scale)
    complete_slices = all(
        report["decision"] == "PASS" for report in slices.values()
    )
    decision = (
        "PASS_WAVEFORM_OPTIMIZATION"
        if all(
            report["decision"] == "PASS"
            for report in component_gates.values()
        )
        and all(safety_gates.values())
        and pathology_guardrails["decision"] == "PASS"
        and denoising["decision"] == "PASS"
        and complete_slices
        else "FAIL_WAVEFORM_OPTIMIZATION"
    )
    return {
        "aggregates": aggregates,
        "component_gates": component_gates,
        "safety": {
            "values": safety_values,
            "gates": safety_gates,
            "decision": "PASS" if all(safety_gates.values()) else "FAIL",
        },
        "full_band_pathology_guardrails": pathology_guardrails,
        "denoising": denoising,
        "required_slices": slices,
        "decision": decision,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Direct AVQI waveform-optimization diagnostic",
        "",
        f"**Decision:** `{summary['decision']}`",
        "",
        "| Component | Exact improved cases | Median normalized reduction | Decision |",
        "|---|---:|---:|---|",
    ]
    for component in OPTIMIZED_COMPONENTS:
        aggregate = summary["aggregates"]["exact"][component]
        decision = summary["component_gates"][component]["decision"]
        lines.append(
            f"| {component} | {aggregate['improvement_fraction']:.3f} | "
            f"{aggregate['median_normalized_gap_reduction']:.3f} | {decision} |"
        )
    values = summary["safety"]["values"]
    failed_slices = [
        name
        for name, item in summary["required_slices"].items()
        if item["decision"] != "PASS"
    ]
    lines.extend(
        [
            "",
            f"Residual ceiling: `{report['contract']['residual_ceiling_db']} dB`; "
            f"worst observed: `{values['worst_residual_rms_db']:.2f} dB`.",
            "",
            (
                "Full-band pathology guardrails: "
                f"`{summary['full_band_pathology_guardrails']['decision']}`; "
                f"denoising non-regression: `{summary['denoising']['decision']}`."
            ),
            "",
            (
                "CS/SV and severity slices all passed."
                if not failed_slices
                else "Failed required slices: " + ", ".join(failed_slices) + "."
            ),
            "",
            "All exact values were recomputed with the hash-locked Praat implementation. "
            "No generator parameter was updated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("declared source commit differs from repository HEAD")
    if not args.slurm_job_id.isdigit():
        raise ValueError(f"invalid Slurm job ID: {args.slurm_job_id}")
    if sha256_file(args.external_exact_csv) != args.external_exact_csv_sha256:
        raise ValueError("external exact CSV hash drift")
    if sha256_file(args.predictor_checkpoint) != args.predictor_checkpoint_sha256:
        raise ValueError("predictor checkpoint hash drift")
    if avqi_code_tree_sha256(args.avqi_code_root) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code tree hash drift")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"exact AVQI interpreter missing: {args.exact_python}")
    (
        screen_component_gradient_norms,
        optimization_component_weights,
        authorization,
    ) = validate_route_c_authorization(
        args.authorization_consensus,
        args.authorization_consensus_sha256,
        args.screen_report,
        args.screen_report_sha256,
        args.screen_completion_receipt,
        args.screen_completion_receipt_sha256,
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cases = load_cases(
        args.external_exact_csv,
        args.speakers_per_severity,
        args.expected_cases,
        args.speaker_offset,
    )
    predictor, calibrator, target_mean, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    args.output_dir.mkdir(parents=True)
    wav_dir = args.output_dir / "wav"
    wav_dir.mkdir()
    rows: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}
    for index, case in enumerate(cases, start=1):
        waveform, subtype = load_waveform(case.path)
        reference, _ = load_waveform(case.reference_path)
        optimized, optimization = optimize_waveform(
            waveform.to(device),
            case.target,
            predictor,
            calibrator,
            target_mean,
            target_scale,
            optimization_component_weights,
            args.steps,
            args.learning_rate_scale,
            args.fidelity_weight,
            args.residual_ceiling_db,
        )
        output_path = wav_dir / f"{case.case_id}.wav"
        sf.write(
            output_path,
            optimized.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype=subtype,
        )
        written, _ = load_waveform(output_path)
        written_safety = waveform_safety(waveform, written)
        pathology_guardrails = full_band_pathology_guardrails(
            reference,
            waveform,
            written,
        )
        with torch.inference_mode():
            surrogate_after_written = predict_components(
                predictor,
                calibrator,
                target_mean,
                target_scale,
                written.to(device).unsqueeze(0),
            ).cpu()[0]
        exact_after = score_exact(
            output_path,
            case.view,
            args.exact_python,
            args.avqi_code_root,
        )
        surrogate_before = torch.tensor(optimization["surrogate_before"])
        target = case.target
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "speaker_id": case.speaker_id,
            "view": case.view,
            "sample_group": case.sample_group,
            "condition": CONDITION,
            "candidate": CANDIDATE,
            "source_path": str(case.path.resolve()),
            "source_sha256": sha256_file(case.path),
            "clean_pathological_reference_path": str(
                case.reference_path.resolve()
            ),
            "clean_pathological_reference_sha256": sha256_file(
                case.reference_path
            ),
            "optimized_path": str(output_path.resolve()),
            "optimized_sha256": sha256_file(output_path),
            **written_safety,
            **pathology_guardrails,
        }
        for component_index, component in enumerate(AVQI_COMPONENT_NAMES):
            target_value = float(target[component_index])
            surrogate_before_value = float(surrogate_before[component_index])
            surrogate_after_value = float(surrogate_after_written[component_index])
            exact_before_value = float(case.exact_before[component_index])
            exact_after_value = float(exact_after[component_index])
            row[f"target_{component}"] = target_value
            row[f"surrogate_before_{component}"] = surrogate_before_value
            row[f"surrogate_after_{component}"] = surrogate_after_value
            row[f"surrogate_absolute_gap_before_{component}"] = abs(
                surrogate_before_value - target_value
            )
            row[f"surrogate_absolute_gap_after_{component}"] = abs(
                surrogate_after_value - target_value
            )
            row[f"exact_before_{component}"] = exact_before_value
            row[f"exact_after_{component}"] = exact_after_value
            row[f"exact_absolute_gap_before_{component}"] = abs(
                exact_before_value - target_value
            )
            row[f"exact_absolute_gap_after_{component}"] = abs(
                exact_after_value - target_value
            )
        rows.append(row)
        trajectories[case.case_id] = optimization["trajectory"]
        print(f"optimized_cases={index}/{len(cases)} case={case.case_id}", flush=True)

    summary = summarize(rows, target_scale.cpu(), args.residual_ceiling_db)
    report = {
        "schema_version": "direct-avqi-waveform-optimization-v2",
        "decision": summary["decision"],
        "waveform_optimizer_steps": len(rows) * args.steps,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "authorization": authorization,
        "contract": {
            "source_commit": args.source_commit,
            "slurm_job_id": args.slurm_job_id,
            "candidate": CANDIDATE,
            "condition": CONDITION,
            "severity_groups": list(SEVERITY_GROUPS),
            "views": list(VIEWS),
            "speakers_per_severity": args.speakers_per_severity,
            "speaker_offset": args.speaker_offset,
            "case_selection": (
                "lexicographic severity slice beginning at speaker_offset"
            ),
            "expected_cases": args.expected_cases,
            "optimized_components": list(OPTIMIZED_COMPONENTS),
            "screen_component_gradient_norms": screen_component_gradient_norms,
            "optimization_component_weights": optimization_component_weights,
            "steps": args.steps,
            "learning_rate_scale": args.learning_rate_scale,
            "fidelity_weight": args.fidelity_weight,
            "residual_ceiling_db": args.residual_ceiling_db,
            "gates": {
                "exact_improvement_fraction": EXACT_IMPROVEMENT_FRACTION_GATE,
                "surrogate_improvement_fraction": (
                    SURROGATE_IMPROVEMENT_FRACTION_GATE
                ),
                "median_normalized_gap_reduction": (
                    NORMALIZED_GAP_REDUCTION_GATE
                ),
                "selected_total_relative_gap_reduction": (
                    SELECTED_TOTAL_RELATIVE_REDUCTION_GATE
                ),
                "nonselected_median_normalized_gap_increase": (
                    NONSELECTED_MEDIAN_INCREASE_GATE
                ),
                "minimum_cosine_similarity": MINIMUM_COSINE_GATE,
                "maximum_clip_fraction": MAXIMUM_CLIP_FRACTION,
                "pathology_db_median_gap_increase_max": (
                    PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX
                ),
                "pathology_db_worst_gap_increase_max": (
                    PATHOLOGY_DB_WORST_GAP_INCREASE_MAX
                ),
                "airflow_flatness_median_gap_increase_max": (
                    AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX
                ),
                "airflow_flatness_worst_gap_increase_max": (
                    AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX
                ),
                "pause_f1_median_decrease_max": (
                    PAUSE_F1_MEDIAN_DECREASE_MAX
                ),
                "pause_f1_worst_decrease_max": PAUSE_F1_WORST_DECREASE_MAX,
                "guardrail_pass_fraction_min": GUARDRAIL_PASS_FRACTION_MIN,
                "denoising_median_change_min_db": (
                    DENOISING_MEDIAN_CHANGE_MIN_DB
                ),
                "denoising_worst_change_min_db": (
                    DENOISING_WORST_CHANGE_MIN_DB
                ),
            },
            "full_band_guardrail_contract": {
                "reference": "same-speaker clean pathological CS or SV waveform",
                "alignment": (
                    "tail crop to shortest waveform only; no shift, filter, "
                    "resample, or metric-branch high-pass"
                ),
                "low_frequency_bands_hz": FULL_BAND_FREQUENCY_RANGES,
                "low_energy_quantile": LOW_ENERGY_QUANTILE,
                "airflow_proxy_frequency_range_hz": (
                    AIRFLOW_PROXY_FREQUENCY_RANGE
                ),
                "airflow_proxy_limit": (
                    "low-energy band energy and spectral flatness are signal "
                    "proxies, not clinical airflow labels"
                ),
                "required_slices": [
                    "view=cs",
                    "view=sv",
                    "severity=pathological_mild",
                    "severity=pathological_severe",
                ],
                "denoising_metrics": ["snr", "si_sdr"],
            },
            "source_sha256": {
                "external_exact_csv": args.external_exact_csv_sha256,
                "predictor_checkpoint": args.predictor_checkpoint_sha256,
                "authorization_consensus": (
                    args.authorization_consensus_sha256
                ),
                "screen_report": args.screen_report_sha256,
                "screen_completion_receipt": (
                    args.screen_completion_receipt_sha256
                ),
                "avqi_code_tree": args.avqi_code_tree_sha256,
            },
            "exact_python": str(args.exact_python.resolve()),
        },
        "summary": summary,
        "trajectories": trajectories,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    results_path = args.output_dir / "results.csv"
    report_path = args.output_dir / "waveform_optimization_report.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_csv(results_path, rows)
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    receipt = {
        "decision": report["decision"],
        "waveform_optimizer_steps": report["waveform_optimizer_steps"],
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "authorization_decision": authorization["decision"],
        "bounded_waveform_pilot_completed": True,
        "authorization_sha256": {
            "consensus": authorization["consensus_sha256"],
            "screen_report": authorization["screen_report_sha256"],
            "screen_completion_receipt": authorization[
                "screen_completion_receipt_sha256"
            ],
            "predictor_checkpoint": authorization[
                "predictor_checkpoint_sha256"
            ],
        },
        "speaker_offset": args.speaker_offset,
        "slurm_job_id": args.slurm_job_id,
        "case_count": len(rows),
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (results_path, report_path, summary_path)
        },
        "audio_sha256": {
            path.name: sha256_file(path)
            for path in sorted(wav_dir.glob("*.wav"))
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
