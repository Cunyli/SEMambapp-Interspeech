#!/usr/bin/env python3
"""Audit cached-input-pulse and coupled Shimmer gradients on an opened panel.

This is a mechanism diagnostic, not a promotion panel.  Exact Praat extracts
pulse positions from each degraded input once.  PyTorch then evaluates the
existing live asymmetric-Hann amplitude tier at those detached positions on
the time-aligned frozen S3_500 output.  Exact Praat independently relocates
pulses and scores all six components after every fixed bounded waveform step.

The script also evaluates the already-promoted Shimmer-percent gradient, the
frozen v6 Shimmer-dB gradient, and a non-deployable output-pulse oracle upper
bound.  The alpha, cases, gates, and candidate names are fixed; opened exact
results never select a hyperparameter.  No generator optimizer is loaded or
run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    ComponentAffineCalibrator,
    PraatDifferentiableAVQIComponentEstimator,
    denormalize_components,
)


SAMPLE_RATE = 16_000
SV_METRIC_SAMPLES = 3 * SAMPLE_RATE
GENERATOR_HOP_SIZE = 100
SHIMMER_PERCENT_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_percent")
SHIMMER_DB_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_db")
FIXED_ALPHA = 1e-3
MATERIAL_GAP_THRESHOLD = 0.02
MEDIAN_REDUCTION_GATE = 0.02
IMPROVEMENT_FRACTION_GATE = 0.80
NONSELECTED_MEDIAN_INCREASE_GATE = 0.05
GRADIENT_NORM_RANGE = (1e-10, 1e4)
RESIDUAL_CEILING_DB = -50.0
MINIMUM_COSINE = 0.99999
MAXIMUM_CLIP_FRACTION = 0.0
TOPOLOGY_MATCH_TOLERANCE_SAMPLES = 16.0
TOPOLOGY_MATCH_DROP_MAX = 0.05
TOPOLOGY_COUNT_RATIO_DRIFT_MAX = 0.05
CACHE_COVERAGE_MIN = 0.99
CACHE_RUNTIME_MAX_MS = 500.0
CACHE_RECORD_MAX_BYTES = 65_536
REQUIRED_EFFECT_SLICES = (
    "view=cs",
    "view=sv",
    "severity=pathological_mild",
    "severity=pathological_severe",
)
CANDIDATE_NAMES = (
    "v6_db",
    "praat_input_topology_absolute_db",
    "shimmer_percent_coupled",
    "output_pulse_oracle_db",
)
COMPONENT_PREFIXES = {
    "cpps": "cpps",
    "hnr": "hnr",
    "shimmer_percent": "shimmer_percent",
    "shimmer_db": "shimmer_db",
    "slope": "slope",
    "tilt": "tilt",
}


EXACT_SCORER = r"""
import json
import sys
import time

import parselmouth
from parselmouth.praat import call

sys.path.insert(0, sys.argv[1])
from avqi_code import run_avqi


def pulse_positions(path, view):
    started = time.perf_counter()
    sound = parselmouth.Sound(path)
    sound = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
    if view == "sv":
        duration = float(call(sound, "Get total duration"))
        if duration > 3.0:
            sound = call(
                sound,
                "Extract part",
                duration - 3.0,
                duration,
                "rectangular",
                1.0,
                "no",
            )
    elif view != "cs":
        raise ValueError("unsupported view: " + view)
    point_process = call(sound, "To PointProcess (periodic, cc)", 50, 400)
    count = int(call(point_process, "Get number of points"))
    positions = [
        (
            float(call(point_process, "Get time from index", index))
            - float(sound.x1)
        )
        / float(sound.dx)
        for index in range(1, count + 1)
    ]
    return positions, 1000.0 * (time.perf_counter() - started)


request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    row = {
        "id": item["id"],
        "case_id": item["case_id"],
        "role": item["role"],
        "view": item["view"],
    }
    try:
        positions, runtime_ms = pulse_positions(item["path"], item["view"])
        row.update(
            {
                "scoring_status": "ok",
                "pulse_positions_samples": positions,
                "pulse_count": len(positions),
                "pulse_runtime_ms": runtime_ms,
                "error_type": "",
                "error_message": "",
            }
        )
        if item["score_components"]:
            metrics = run_avqi(
                item["path"],
                item["path"],
                target_sr=16000,
                speaking_type=item["view"],
                step_versions=request["step_versions"],
                remove_sv_silence_with_sox=False,
            )
            row["components"] = {
                name: float(metrics[name]) for name in request["components"]
            }
    except Exception as exc:
        row.update(
            {
                "scoring_status": "error",
                "pulse_positions_samples": [],
                "pulse_count": 0,
                "pulse_runtime_ms": 0.0,
                "components": {},
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
        )
    rows.append(row)
print(
    "AVQI_SHIMMER_HYBRID_EXACT_JSON="
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--calibration-results", type=Path, required=True)
    parser.add_argument("--calibration-results-sha256", required=True)
    parser.add_argument("--final-results", type=Path, required=True)
    parser.add_argument("--final-results-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
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


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")
    return actual


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._ÄÖÅäöåÜüÉé_-]", "_", value)


def read_waveform(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"expected mono 16 kHz waveform: {path}")
    waveform = torch.from_numpy(audio.copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {path}")
    return waveform


def validate_panel_contract(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in contract["rows"]]
    if len(rows) != 12 or len({row["case_id"] for row in rows}) != 12:
        raise ValueError("opened hybrid audit requires twelve unique cases")
    if {row["view"] for row in rows} != {"cs", "sv"}:
        raise ValueError("opened hybrid audit requires CS and SV")
    if {row["split"] for row in rows} != {"calibration", "final"}:
        raise ValueError("opened hybrid audit split drift")
    for row in rows:
        for role in ("degraded", "base", "target"):
            artifact = Path(row[f"{role}_path"])
            validate_hash(artifact, row[f"{role}_sha256"], f"{role} waveform")
        degraded = sf.info(row["degraded_path"])
        base = sf.info(row["base_path"])
        target = sf.info(row["target_path"])
        if (
            degraded.samplerate != SAMPLE_RATE
            or base.samplerate != SAMPLE_RATE
            or target.samplerate != SAMPLE_RATE
            or degraded.channels != 1
            or base.channels != 1
            or target.channels != 1
            or degraded.frames != target.frames
        ):
            raise ValueError(f"panel waveform contract drift: {row['case_id']}")
        trailing_truncation = degraded.frames - base.frames
        if not 0 <= trailing_truncation < GENERATOR_HOP_SIZE:
            raise ValueError(
                f"unsupported input/output timeline drift: {row['case_id']} "
                f"({degraded.frames} -> {base.frames})"
            )
        row["degraded_frame_count"] = degraded.frames
        row["base_frame_count"] = base.frames
        row["trailing_truncation_samples"] = trailing_truncation
    return contract, rows


def exact_reference_rows(
    calibration_path: Path,
    final_path: Path,
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in read_csv(calibration_path):
        if float(row["alpha"]) == 0.0:
            selected[row["case_id"]] = row
    for row in read_csv(final_path):
        selected[row["case_id"]] = row
    return selected


def run_exact(
    items: list[dict[str, Any]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, str(avqi_code_root)],
        input=json.dumps(
            {
                "items": items,
                "components": list(AVQI_COMPONENT_NAMES),
                "step_versions": {
                    "highpass": "praat",
                    "read_and_resample": "praat",
                    "sv_length_norm": "praat",
                    "cs_voiced_segments": "praat",
                    "concatenate": "praat",
                    "cpps": "praat",
                    "hnr": "praat",
                    "shimmer": "praat",
                    "slope": "praat",
                    "tilt": "praat",
                    "pitch": "praat",
                },
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("AVQI_SHIMMER_HYBRID_EXACT_JSON=")
    ]
    if len(lines) != 1:
        raise RuntimeError("exact Shimmer hybrid scorer emitted no unique marker")
    report = json.loads(lines[0].split("=", 1)[1])
    if len(report["rows"]) != len(items):
        raise ValueError("exact Shimmer hybrid row count drift")
    if [row["id"] for row in report["rows"]] != [item["id"] for item in items]:
        raise ValueError("exact Shimmer hybrid row order drift")
    return report


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
    expected_architecture = "direct_praat_hard_shimmer_pulse_path_v6"
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError("unexpected frozen Route C checkpoint architecture")
    predictor = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
    ).to(device)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.eval()
    calibrator = ComponentAffineCalibrator(
        checkpoint["calibration_scale"],
        checkpoint["calibration_bias"],
    ).to(device)
    calibrator.eval()
    return (
        predictor,
        calibrator,
        checkpoint["target_mean"].to(device),
        checkpoint["target_scale"].to(device),
    )


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


def candidate_proxy(
    candidate: str,
    predictor: PraatDifferentiableAVQIComponentEstimator,
    calibrator: ComponentAffineCalibrator,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    waveform: torch.Tensor,
    input_pulses: torch.Tensor,
    output_pulses: torch.Tensor,
    view: str,
) -> tuple[torch.Tensor, str]:
    if candidate == "praat_input_topology_absolute_db":
        proxy = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            input_pulses,
            metric_sample_count=SV_METRIC_SAMPLES if view == "sv" else None,
        )[1]
        return proxy, "shimmer_db"
    if candidate == "output_pulse_oracle_db":
        proxy = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            output_pulses,
            metric_sample_count=SV_METRIC_SAMPLES if view == "sv" else None,
        )[1]
        return proxy, "shimmer_db"
    components = predict_components(
        predictor,
        calibrator,
        target_mean,
        target_scale,
        waveform.unsqueeze(0),
    )[0]
    if candidate == "v6_db":
        return components[SHIMMER_DB_INDEX], "shimmer_db"
    if candidate == "shimmer_percent_coupled":
        return components[SHIMMER_PERCENT_INDEX], "shimmer_percent"
    raise ValueError(f"unknown Shimmer hybrid candidate: {candidate}")


def normalized_gradient_step(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    gradient_rms = gradient.square().mean().sqrt()
    base_rms = waveform.square().mean().sqrt()
    if float(gradient_rms) <= 1e-15:
        return waveform.detach().clone()
    return (
        waveform.detach()
        - FIXED_ALPHA * base_rms * gradient / gradient_rms
    )


def db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-15) / max(denominator, 1e-15))


def waveform_safety(base: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    residual = candidate.astype(np.float64) - base.astype(np.float64)
    base_rms = math.sqrt(float(np.mean(np.square(base, dtype=np.float64))))
    residual_rms = math.sqrt(float(np.mean(np.square(residual, dtype=np.float64))))
    cosine = float(np.dot(base, candidate)) / max(
        float(np.linalg.norm(base) * np.linalg.norm(candidate)),
        1e-15,
    )
    return {
        "residual_rms_db": db_ratio(residual_rms, base_rms),
        "cosine_similarity": cosine,
        "clip_fraction": float(np.mean(np.abs(candidate) >= 1.0)),
    }


def nearest_match_rate(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float = TOPOLOGY_MATCH_TOLERANCE_SAMPLES,
) -> float:
    if source.size == 0 or target.size == 0:
        return 0.0
    right = np.searchsorted(target, source, side="left")
    right_bounded = np.minimum(right, target.size - 1)
    left_bounded = np.maximum(right - 1, 0)
    distance = np.minimum(
        np.abs(target[right_bounded] - source),
        np.abs(target[left_bounded] - source),
    )
    return float(np.mean(distance <= tolerance))


def map_input_metric_pulses_to_output(
    pulse_positions: np.ndarray,
    *,
    input_frame_count: int,
    output_frame_count: int,
    view: str,
) -> np.ndarray:
    trailing_truncation = input_frame_count - output_frame_count
    if not 0 <= trailing_truncation < GENERATOR_HOP_SIZE:
        raise ValueError("unsupported input/output timeline drift")
    if view == "sv":
        input_metric_start = max(input_frame_count - SV_METRIC_SAMPLES, 0)
        output_metric_start = max(output_frame_count - SV_METRIC_SAMPLES, 0)
        output_metric_frames = min(output_frame_count, SV_METRIC_SAMPLES)
    elif view == "cs":
        input_metric_start = 0
        output_metric_start = 0
        output_metric_frames = output_frame_count
    else:
        raise ValueError(f"unsupported view: {view}")
    mapped = np.asarray(pulse_positions, dtype=np.float64) + (
        input_metric_start - output_metric_start
    )
    return mapped[(mapped >= 0.0) & (mapped < output_metric_frames)]


def exact_component_fields(
    reference: dict[str, str],
    after: dict[str, float],
    target_scale: np.ndarray,
) -> dict[str, float]:
    fields: dict[str, float] = {}
    for index, name in enumerate(AVQI_COMPONENT_NAMES):
        prefix = COMPONENT_PREFIXES[name]
        target = float(reference[f"exact_target_{prefix}"])
        before = float(reference[f"exact_before_{prefix}"])
        after_value = float(after[name])
        before_gap = abs(before - target)
        after_gap = abs(after_value - target)
        fields[f"exact_target_{prefix}"] = target
        fields[f"exact_before_{prefix}"] = before
        fields[f"exact_after_{prefix}"] = after_value
        fields[f"exact_absolute_gap_before_{prefix}"] = before_gap
        fields[f"exact_absolute_gap_after_{prefix}"] = after_gap
        fields[f"exact_normalized_gap_reduction_{prefix}"] = (
            before_gap - after_gap
        ) / max(float(target_scale[index]), 1e-8)
    return fields


def cache_record_bytes(record: dict[str, Any]) -> int:
    return len(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )


def cache_record_sha256(record: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"record_sha256", "record_bytes"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def finalize_cache_record(record: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(record)
    finalized["record_sha256"] = cache_record_sha256(finalized)
    finalized["record_bytes"] = 0
    for _ in range(4):
        observed = cache_record_bytes(finalized)
        if finalized["record_bytes"] == observed:
            break
        finalized["record_bytes"] = observed
    if finalized["record_bytes"] != cache_record_bytes(finalized):
        raise RuntimeError("cache record byte count did not converge")
    return finalized


def cache_record_valid(record: dict[str, Any]) -> bool:
    return (
        record.get("record_sha256") == cache_record_sha256(record)
        and record.get("record_bytes") == cache_record_bytes(record)
    )


def summarize_effect_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    material = [row for row in rows if row["material_shimmer_db_gap"]]
    reductions = [
        row["exact_normalized_gap_reduction_shimmer_db"] for row in material
    ]
    improvement_fraction = (
        sum(value > 0.0 for value in reductions) / len(reductions)
        if reductions
        else 0.0
    )
    median_reduction = median(reductions) if reductions else None
    gates = {
        "material_case_present": bool(material),
        "improvement_fraction_ge_half": improvement_fraction >= 0.5,
        "median_normalized_reduction_nonnegative": (
            median_reduction is not None and median_reduction >= 0.0
        ),
    }
    return {
        "rows": len(rows),
        "material_rows": len(material),
        "improvement_fraction": improvement_fraction,
        "median_normalized_gap_reduction": median_reduction,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def aggregate_candidate(
    candidate: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == candidate]
    material = [row for row in selected if row["material_shimmer_db_gap"]]
    d_b_reductions = [
        row["exact_normalized_gap_reduction_shimmer_db"] for row in material
    ]
    nonselected = [
        name
        for name in AVQI_COMPONENT_NAMES
        if name
        != (
            "shimmer_percent"
            if candidate == "shimmer_percent_coupled"
            else "shimmer_db"
        )
    ]
    nonselected_medians = {
        name: median(
            -row[f"exact_normalized_gap_reduction_{COMPONENT_PREFIXES[name]}"]
            for row in selected
        )
        for name in nonselected
    }
    slice_predicates = {
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
    slices = {
        name: summarize_effect_slice(
            [row for row in selected if predicate(row)]
        )
        for name, predicate in slice_predicates.items()
    }
    required_slice_gate = all(
        slices[name]["decision"] == "PASS" for name in REQUIRED_EFFECT_SLICES
    )
    effect_pass = (
        bool(material)
        and median(d_b_reductions) >= MEDIAN_REDUCTION_GATE
        and sum(value > 0.0 for value in d_b_reductions) / len(d_b_reductions)
        >= IMPROVEMENT_FRACTION_GATE
    )
    result = {
        "candidate": candidate,
        "rows": len(selected),
        "material_rows": len(material),
        "median_exact_db_normalized_gap_reduction": (
            median(d_b_reductions) if d_b_reductions else None
        ),
        "exact_db_improvement_fraction": (
            sum(value > 0.0 for value in d_b_reductions) / len(d_b_reductions)
            if d_b_reductions
            else 0.0
        ),
        "median_exact_db_absolute_gap_after": (
            median(
                row["exact_absolute_gap_after_shimmer_db"] for row in material
            )
            if material
            else None
        ),
        "slices": slices,
        "required_slice_gate": {
            "required": list(REQUIRED_EFFECT_SLICES),
            "decision": "PASS" if required_slice_gate else "FAIL",
        },
        "gradient_l2_min": min(row["gradient_l2_norm"] for row in selected),
        "gradient_l2_median": median(
            row["gradient_l2_norm"] for row in selected
        ),
        "gradient_l2_max": max(row["gradient_l2_norm"] for row in selected),
        "all_gradients_finite": all(row["gradient_finite"] for row in selected),
        "nonselected_median_normalized_gap_increase": nonselected_medians,
        "minimum_residual_rms_db": min(
            row["residual_rms_db"] for row in selected
        ),
        "maximum_residual_rms_db": max(
            row["residual_rms_db"] for row in selected
        ),
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in selected
        ),
        "maximum_clip_fraction": max(row["clip_fraction"] for row in selected),
        "topology_stability_fraction": sum(
            row["topology_stability_pass"] for row in selected
        )
        / len(selected),
        "gates": {
            "complete_case_coverage": len(selected) == 12,
            "material_cases_ge_5": len(material) >= 5,
            "exact_db_effect": effect_pass,
            "required_effect_slices": required_slice_gate,
            "gradient": all(
                row["gradient_finite"]
                and GRADIENT_NORM_RANGE[0]
                <= row["gradient_l2_norm"]
                <= GRADIENT_NORM_RANGE[1]
                for row in selected
            ),
            "nonselected": all(
                value <= NONSELECTED_MEDIAN_INCREASE_GATE
                for value in nonselected_medians.values()
            ),
            "safety": all(
                row["residual_rms_db"] <= RESIDUAL_CEILING_DB
                and row["cosine_similarity"] >= MINIMUM_COSINE
                and row["clip_fraction"] <= MAXIMUM_CLIP_FRACTION
                for row in selected
            ),
            "topology_stability": all(
                row["topology_stability_pass"] for row in selected
            ),
        },
    }
    result["all_gates_pass"] = all(result["gates"].values())
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    if sha256_tree(args.avqi_code_root) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")

    source_hashes = {
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "opened panel contract",
        ),
        "calibration_results": validate_hash(
            args.calibration_results,
            args.calibration_results_sha256,
            "opened calibration results",
        ),
        "final_results": validate_hash(
            args.final_results,
            args.final_results_sha256,
            "opened final results",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen Route C checkpoint",
        ),
    }
    _, panel_rows = validate_panel_contract(args.panel_contract)
    references = exact_reference_rows(
        args.calibration_results,
        args.final_results,
    )
    if set(references) != {row["case_id"] for row in panel_rows}:
        raise ValueError("opened exact reference coverage drift")

    output_root = args.output_dir
    waveform_root = output_root / "waveforms"
    waveform_root.mkdir(parents=True)

    topology_items = []
    for row in panel_rows:
        for role in ("degraded_input", "base_output"):
            path_key = "degraded_path" if role == "degraded_input" else "base_path"
            topology_items.append(
                {
                    "id": f"{role}:{row['case_id']}",
                    "case_id": row["case_id"],
                    "role": role,
                    "path": row[path_key],
                    "view": row["view"],
                    "score_components": False,
                }
            )
    topology_exact = run_exact(
        topology_items,
        args.exact_python,
        args.avqi_code_root,
    )
    topology_by_id = {row["id"]: row for row in topology_exact["rows"]}

    cache_rows = []
    for row in panel_rows:
        exact = topology_by_id[f"degraded_input:{row['case_id']}"]
        cache_record = {
            "schema_version": "avqi-route-c-shimmer-input-pulse-cache-v1",
            "dataset": "TAU_opened_fresh_panel_mechanism_only",
            "case_id": row["case_id"],
            "speaker_id": row["speaker_id"],
            "split": row["split"],
            "sample_group": row["sample_group"],
            "view": row["view"],
            "condition": row["condition"],
            "input_path": row["degraded_path"],
            "input_sha256": row["degraded_sha256"],
            "sample_rate": SAMPLE_RATE,
            "frame_count": sf.info(row["degraded_path"]).frames,
            "metric_crop_start_sample": (
                max(sf.info(row["degraded_path"]).frames - SV_METRIC_SAMPLES, 0)
                if row["view"] == "sv"
                else 0
            ),
            "metric_preprocessing": (
                "exact Praat 34 Hz high-pass, full timeline"
                if row["view"] == "cs"
                else "exact Praat 34 Hz high-pass, then final 3 seconds"
            ),
            "pulse_positions_samples": exact["pulse_positions_samples"],
            "pulse_count": exact["pulse_count"],
            "pulse_runtime_ms": exact["pulse_runtime_ms"],
            "scoring_status": exact["scoring_status"],
            "parselmouth_version": topology_exact["parselmouth_version"],
            "praat_version": topology_exact["praat_version"],
            "clean_target_pulse_topology_present": False,
        }
        cache_rows.append(finalize_cache_record(cache_record))
    cache_path = output_root / "input_pulse_cache.json"
    write_json(
        cache_path,
        {
            "schema_version": "avqi-route-c-shimmer-input-pulse-cache-bank-v1",
            "source_commit": args.source_commit,
            "exact_versions": {
                "parselmouth": topology_exact["parselmouth_version"],
                "praat": topology_exact["praat_version"],
            },
            "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
            "records": cache_rows,
        },
    )
    cache_by_case = {row["case_id"]: row for row in cache_rows}
    output_topology_by_case = {
        row["case_id"]: topology_by_id[f"base_output:{row['case_id']}"]
        for row in panel_rows
    }

    device = torch.device(args.device)
    predictor, calibrator, target_mean, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    candidate_records = []
    exact_candidate_items = []
    for panel_row in panel_rows:
        case_id = panel_row["case_id"]
        reference = references[case_id]
        base_cpu = read_waveform(Path(panel_row["base_path"]))
        cache = cache_by_case[case_id]
        output_topology = output_topology_by_case[case_id]
        if cache["scoring_status"] != "ok" or cache["pulse_count"] < 3:
            continue
        if output_topology["scoring_status"] != "ok" or output_topology["pulse_count"] < 3:
            continue
        for candidate in CANDIDATE_NAMES:
            waveform = base_cpu.to(device).requires_grad_(True)
            mapped_input_positions = map_input_metric_pulses_to_output(
                np.asarray(cache["pulse_positions_samples"], dtype=np.float64),
                input_frame_count=panel_row["degraded_frame_count"],
                output_frame_count=panel_row["base_frame_count"],
                view=panel_row["view"],
            )
            input_pulses = waveform.new_tensor(mapped_input_positions)
            output_pulses = waveform.new_tensor(
                output_topology["pulse_positions_samples"]
            )
            proxy, optimized_component = candidate_proxy(
                candidate,
                predictor,
                calibrator,
                target_mean,
                target_scale,
                waveform,
                input_pulses,
                output_pulses,
                panel_row["view"],
            )
            optimized_index = AVQI_COMPONENT_NAMES.index(optimized_component)
            target_value = float(reference[f"exact_target_{optimized_component}"])
            loss = (
                (proxy - target_value)
                / target_scale[optimized_index].clamp_min(1e-8)
            ).square()
            gradient = torch.autograd.grad(loss, waveform)[0]
            candidate_waveform = normalized_gradient_step(waveform, gradient)
            output_path = waveform_root / (
                f"{safe_name(case_id)}__{candidate}.wav"
            )
            sf.write(
                output_path,
                candidate_waveform.detach().cpu().numpy(),
                SAMPLE_RATE,
                subtype="PCM_24",
            )
            stored = read_waveform(output_path)
            safety = waveform_safety(base_cpu.numpy(), stored.numpy())
            record = {
                "case_id": case_id,
                "split": panel_row["split"],
                "speaker_id": panel_row["speaker_id"],
                "sample_group": panel_row["sample_group"],
                "view": panel_row["view"],
                "condition": panel_row["condition"],
                "candidate": candidate,
                "optimized_component": optimized_component,
                "fixed_alpha": FIXED_ALPHA,
                "proxy_before": float(proxy.detach()),
                "proxy_target": target_value,
                "proxy_loss": float(loss.detach()),
                "gradient_l2_norm": float(gradient.norm()),
                "gradient_rms": float(gradient.square().mean().sqrt()),
                "gradient_finite": bool(torch.isfinite(gradient).all()),
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "cache_record_sha256": cache["record_sha256"],
                "cache_pulse_count": cache["pulse_count"],
                "mapped_cache_pulse_count": int(mapped_input_positions.size),
                "base_output_pulse_count": output_topology["pulse_count"],
                "degraded_frame_count": panel_row["degraded_frame_count"],
                "base_frame_count": panel_row["base_frame_count"],
                "trailing_truncation_samples": panel_row[
                    "trailing_truncation_samples"
                ],
                "input_to_output_metric_position_shift_samples": (
                    max(
                        panel_row["degraded_frame_count"] - SV_METRIC_SAMPLES,
                        0,
                    )
                    - max(
                        panel_row["base_frame_count"] - SV_METRIC_SAMPLES,
                        0,
                    )
                    if panel_row["view"] == "sv"
                    else 0
                ),
                **safety,
            }
            candidate_records.append(record)
            exact_candidate_items.append(
                {
                    "id": f"candidate:{case_id}:{candidate}",
                    "case_id": case_id,
                    "role": candidate,
                    "path": str(output_path.resolve()),
                    "view": panel_row["view"],
                    "score_components": True,
                }
            )

    if len(candidate_records) != len(panel_rows) * len(CANDIDATE_NAMES):
        raise ValueError("candidate waveform coverage drift")
    candidate_exact = run_exact(
        exact_candidate_items,
        args.exact_python,
        args.avqi_code_root,
    )
    if (
        candidate_exact["parselmouth_version"]
        != topology_exact["parselmouth_version"]
        or candidate_exact["praat_version"] != topology_exact["praat_version"]
    ):
        raise ValueError("exact runtime drift within hybrid diagnostic")
    candidate_exact_by_id = {row["id"]: row for row in candidate_exact["rows"]}

    csv_rows = []
    for record in candidate_records:
        case_id = record["case_id"]
        candidate = record["candidate"]
        exact = candidate_exact_by_id[f"candidate:{case_id}:{candidate}"]
        if exact["scoring_status"] != "ok":
            raise RuntimeError(
                f"exact candidate scoring failed for {case_id}/{candidate}: "
                f"{exact['error_type']} {exact['error_message']}"
            )
        cache_positions = map_input_metric_pulses_to_output(
            np.asarray(
                cache_by_case[case_id]["pulse_positions_samples"],
                dtype=np.float64,
            ),
            input_frame_count=record["degraded_frame_count"],
            output_frame_count=record["base_frame_count"],
            view=record["view"],
        )
        base_positions = np.asarray(
            output_topology_by_case[case_id]["pulse_positions_samples"],
            dtype=np.float64,
        )
        after_positions = np.asarray(
            exact["pulse_positions_samples"],
            dtype=np.float64,
        )
        base_match = nearest_match_rate(cache_positions, base_positions)
        after_match = nearest_match_rate(cache_positions, after_positions)
        base_count_ratio = base_positions.size / max(cache_positions.size, 1)
        after_count_ratio = after_positions.size / max(cache_positions.size, 1)
        topology_pass = (
            after_match >= base_match - TOPOLOGY_MATCH_DROP_MAX
            and abs(after_count_ratio - base_count_ratio)
            <= TOPOLOGY_COUNT_RATIO_DRIFT_MAX
        )
        component_fields = exact_component_fields(
            references[case_id],
            exact["components"],
            target_scale_np,
        )
        material = (
            abs(
                component_fields["exact_before_shimmer_db"]
                - component_fields["exact_target_shimmer_db"]
            )
            / max(target_scale_np[SHIMMER_DB_INDEX], 1e-8)
            > MATERIAL_GAP_THRESHOLD
        )
        csv_rows.append(
            {
                **record,
                **component_fields,
                "material_shimmer_db_gap": material,
                "candidate_exact_pulse_count": exact["pulse_count"],
                "base_cache_match_rate_16_samples": base_match,
                "candidate_cache_match_rate_16_samples": after_match,
                "base_cache_pulse_count_ratio": base_count_ratio,
                "candidate_cache_pulse_count_ratio": after_count_ratio,
                "topology_stability_pass": topology_pass,
            }
        )

    candidate_aggregates = {
        candidate: aggregate_candidate(candidate, csv_rows)
        for candidate in CANDIDATE_NAMES
    }
    cache_coverage = sum(
        row["scoring_status"] == "ok" and row["pulse_count"] >= 3
        for row in cache_rows
    ) / len(cache_rows)
    cache_records_hash_valid = all(cache_record_valid(row) for row in cache_rows)
    timeline_contract_valid = all(
        0 <= row["trailing_truncation_samples"] < GENERATOR_HOP_SIZE
        for row in panel_rows
    )
    cache_gate = (
        cache_coverage >= CACHE_COVERAGE_MIN
        and cache_records_hash_valid
        and timeline_contract_valid
        and all(row["pulse_runtime_ms"] <= CACHE_RUNTIME_MAX_MS for row in cache_rows)
        and all(row["record_bytes"] <= CACHE_RECORD_MAX_BYTES for row in cache_rows)
        and all(not row["clean_target_pulse_topology_present"] for row in cache_rows)
    )
    v6_aggregate = candidate_aggregates["v6_db"]
    hybrid_aggregate = candidate_aggregates[
        "praat_input_topology_absolute_db"
    ]
    hybrid_vs_v6 = {
        "exact_direction_not_worse": (
            hybrid_aggregate["exact_db_improvement_fraction"]
            >= v6_aggregate["exact_db_improvement_fraction"]
        ),
        "exact_level_error_not_worse": (
            hybrid_aggregate["median_exact_db_absolute_gap_after"]
            <= v6_aggregate["median_exact_db_absolute_gap_after"]
        ),
    }
    hybrid_vs_v6["gate_pass"] = all(hybrid_vs_v6.values())
    hybrid_pass = (
        cache_gate
        and hybrid_aggregate["all_gates_pass"]
        and hybrid_vs_v6["gate_pass"]
    )
    coupled_pass = candidate_aggregates[
        "shimmer_percent_coupled"
    ]["all_gates_pass"]
    report = {
        "schema_version": "avqi-route-c-shimmer-hybrid-opened-diagnostic-v1",
        "decision": (
            "PASS_ABSOLUTE_CACHE_MECHANISM_ONLY_FRESH_REQUIRED"
            if hybrid_pass
            else "FAIL_ABSOLUTE_CACHE_MECHANISM_DO_NOT_PROMOTE"
        ),
        "coupled_baseline_decision": (
            "PASS_COUPLED_DB_COIMPROVEMENT_MECHANISM_ONLY"
            if coupled_pass
            else "FAIL_COUPLED_DB_COIMPROVEMENT_MECHANISM"
        ),
        "panel_status": "already_opened_mechanism_diagnostic_only",
        "fresh_panel_authorized": hybrid_pass or coupled_pass,
        "promotion_authorized": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "device": str(device),
        "candidate_names": list(CANDIDATE_NAMES),
        "fixed_alpha": FIXED_ALPHA,
        "alpha_selected_from_opened_panel": False,
        "cache_contract": {
            "degraded_input_only": True,
            "clean_target_pulse_topology_present": False,
            "input_hash_bound": True,
            "output_start_time_aligned": True,
            "output_trailing_truncation_lt_generator_hop": True,
            "generator_hop_size_samples": GENERATOR_HOP_SIZE,
            "maximum_observed_trailing_truncation_samples": max(
                row["trailing_truncation_samples"] for row in panel_rows
            ),
            "timeline_contract_valid": timeline_contract_valid,
            "sv_metric_coordinates_compensate_trailing_truncation": True,
            "exact_output_relocates_pulses": True,
            "coverage": cache_coverage,
            "coverage_min": CACHE_COVERAGE_MIN,
            "all_record_hashes_valid": cache_records_hash_valid,
            "median_runtime_ms": median(
                row["pulse_runtime_ms"] for row in cache_rows
            ),
            "maximum_runtime_ms": max(
                row["pulse_runtime_ms"] for row in cache_rows
            ),
            "runtime_max_ms": CACHE_RUNTIME_MAX_MS,
            "maximum_record_bytes": max(row["record_bytes"] for row in cache_rows),
            "record_bytes_max": CACHE_RECORD_MAX_BYTES,
            "gate_pass": cache_gate,
        },
        "hybrid_vs_frozen_v6": hybrid_vs_v6,
        "gates": {
            "material_gap_threshold": MATERIAL_GAP_THRESHOLD,
            "median_reduction_min": MEDIAN_REDUCTION_GATE,
            "improvement_fraction_min": IMPROVEMENT_FRACTION_GATE,
            "nonselected_median_increase_max": NONSELECTED_MEDIAN_INCREASE_GATE,
            "gradient_l2_range": list(GRADIENT_NORM_RANGE),
            "residual_ceiling_db": RESIDUAL_CEILING_DB,
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_clip_fraction": MAXIMUM_CLIP_FRACTION,
            "topology_match_tolerance_samples": TOPOLOGY_MATCH_TOLERANCE_SAMPLES,
            "topology_match_drop_max": TOPOLOGY_MATCH_DROP_MAX,
            "topology_count_ratio_drift_max": TOPOLOGY_COUNT_RATIO_DRIFT_MAX,
        },
        "candidate_aggregates": candidate_aggregates,
        "exact_runtime": {
            "parselmouth": candidate_exact["parselmouth_version"],
            "praat": candidate_exact["praat_version"],
        },
        "source_sha256": source_hashes,
        "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
        "artifacts": {},
        "limitations": [
            "The twelve cases and their exact references were opened by the prior Shimmer-percent pilot.",
            "This run can falsify a mechanism but cannot select alpha or promote a component.",
            "The output-pulse oracle is non-deployable and is reported only as an upper bound.",
            "Basic waveform safety is measured here; full pathology and denoising gates remain mandatory on a fresh panel.",
        ],
    }
    rows_path = output_root / "candidate_results.csv"
    report_path = output_root / "diagnostic_report.json"
    write_csv(rows_path, csv_rows)
    report["artifacts"] = {
        "input_pulse_cache": {
            "path": str(cache_path.resolve()),
            "sha256": sha256_file(cache_path),
        },
        "candidate_results": {
            "path": str(rows_path.resolve()),
            "sha256": sha256_file(rows_path),
        },
    }
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-hybrid-opened-receipt-v1",
        "decision": report["decision"],
        "coupled_baseline_decision": report["coupled_baseline_decision"],
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "generator_optimizer_steps": 0,
        "report_sha256": sha256_file(report_path),
        "candidate_results_sha256": sha256_file(rows_path),
        "input_pulse_cache_sha256": sha256_file(cache_path),
        "waveform_count": len(list(waveform_root.glob("*.wav"))),
    }
    receipt_path = output_root / "completion_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
