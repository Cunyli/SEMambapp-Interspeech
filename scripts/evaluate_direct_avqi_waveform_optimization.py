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
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
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
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["candidate"] == CANDIDATE
            and row["condition"] == CONDITION
            and row["view"] in VIEWS
            and row["sample_group"] in SEVERITY_GROUPS
            and row["label"] == "patient"
            and row["scoring_status"] == "ok"
        ]
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
            target=component_tensor(row, "clean_"),
            exact_before=component_tensor(row, "audio_"),
        )
        for row in selected
    ]
    if len(cases) != expected_cases:
        raise ValueError(
            f"fixed panel differs: expected {expected_cases}, found {len(cases)}"
        )
    missing = [str(case.path) for case in cases if not case.path.is_file()]
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


def optimize_waveform(
    base: torch.Tensor,
    target: torch.Tensor,
    predictor: PraatDifferentiableAVQIComponentEstimator,
    calibrator: ComponentAffineCalibrator,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    steps: int,
    learning_rate_scale: float,
    fidelity_weight: float,
    residual_ceiling_db: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if steps <= 0 or learning_rate_scale <= 0.0 or fidelity_weight < 0.0:
        raise ValueError("invalid optimization hyperparameters")
    if residual_ceiling_db >= 0.0:
        raise ValueError("residual ceiling must be below 0 dB")
    selected_indices = torch.tensor(
        [AVQI_COMPONENT_NAMES.index(name) for name in OPTIMIZED_COMPONENTS],
        device=base.device,
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
        component_loss = F.smooth_l1_loss(
            normalized_gap,
            torch.zeros_like(normalized_gap),
        )
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
    nonselected = [
        component
        for component in AVQI_COMPONENT_NAMES
        if component not in OPTIMIZED_COMPONENTS
    ]
    nonselected_change = np.array(
        [
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
        "median_nonselected_normalized_gap_increase": float(
            np.median(nonselected_change)
        ),
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
    decision = (
        "PASS_WAVEFORM_OPTIMIZATION"
        if all(
            report["decision"] == "PASS"
            for report in component_gates.values()
        )
        and all(safety_gates.values())
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
    lines.extend(
        [
            "",
            f"Residual ceiling: `{report['contract']['residual_ceiling_db']} dB`; "
            f"worst observed: `{values['worst_residual_rms_db']:.2f} dB`.",
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
    if sha256_file(args.external_exact_csv) != args.external_exact_csv_sha256:
        raise ValueError("external exact CSV hash drift")
    if sha256_file(args.predictor_checkpoint) != args.predictor_checkpoint_sha256:
        raise ValueError("predictor checkpoint hash drift")
    if avqi_code_tree_sha256(args.avqi_code_root) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code tree hash drift")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"exact AVQI interpreter missing: {args.exact_python}")
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
        optimized, optimization = optimize_waveform(
            waveform.to(device),
            case.target,
            predictor,
            calibrator,
            target_mean,
            target_scale,
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
            "optimized_path": str(output_path.resolve()),
            "optimized_sha256": sha256_file(output_path),
            **written_safety,
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
        "schema_version": "direct-avqi-waveform-optimization-v1",
        "decision": summary["decision"],
        "waveform_optimizer_steps": len(rows) * args.steps,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "contract": {
            "source_commit": args.source_commit,
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
            },
            "source_sha256": {
                "external_exact_csv": args.external_exact_csv_sha256,
                "predictor_checkpoint": args.predictor_checkpoint_sha256,
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
