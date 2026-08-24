#!/usr/bin/env python3
"""Audit Praat-assisted and coupled Shimmer gradients on an opened panel.

This is a mechanism diagnostic, not a promotion panel.  Exact Praat extracts
pulse positions from each degraded input once.  PyTorch then evaluates the
existing live asymmetric-Hann amplitude tier at those detached positions on
the time-aligned frozen S3_500 output.  Exact Praat independently relocates
pulses and scores all six components after every fixed bounded waveform step.

The script also evaluates the already-promoted Shimmer-percent gradient, the
frozen v6 Shimmer-dB gradient, and Candidate C: exact Praat refreshes the
current output topology once per waveform/step and reuses that immutable
topology across the fixed alpha mechanism grid.  For CS this includes exact
AVQI sounding/30-ms concatenation ranges; Torch gathers the corresponding live
metric-high-passed samples, after which only the amplitude and dB path receives
gradients.  A supervisor-frozen three-point alpha grid may freeze a mechanism
parameter on this opened panel, but cannot promote the component.  No generator
optimizer is loaded or run.
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
CURRENT_OUTPUT_REFRESH_ALPHAS = (3e-4, 1e-3, 3e-3)
CURRENT_OUTPUT_REFRESH_PREFIX = "praat_current_output_topology_refresh_db"
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
RUNTIME_PROFILE_JOB_ID = "19906297"
RUNTIME_PROFILE_SOURCE_COMMIT = "718a530041c5f0195dbedc29b4d52c75880186df"
RUNTIME_PROFILE_REPORT_SHA256 = (
    "87a2f03b1e92c89ab317fbf27a82863a2b4d92ada5941939a810643766579d00"
)
RUNTIME_PROFILE_REUSE_SHA256 = (
    "27dccd990084b625f7ee459cd2d02447aea48aaaf23c01f0605dd9cd1aa3a578"
)
REQUIRED_EFFECT_SLICES = (
    "view=cs",
    "view=sv",
    "severity=pathological_mild",
    "severity=pathological_severe",
    "condition=rir_only",
    "condition=snr20",
    "condition=snr10",
)
CURRENT_OUTPUT_REFRESH_CANDIDATES = (
    f"{CURRENT_OUTPUT_REFRESH_PREFIX}_alpha_0p0003",
    f"{CURRENT_OUTPUT_REFRESH_PREFIX}_alpha_0p001",
    f"{CURRENT_OUTPUT_REFRESH_PREFIX}_alpha_0p003",
)
BASELINE_CANDIDATE_NAMES = (
    "v6_db",
    "praat_input_topology_absolute_db",
    "shimmer_percent_coupled",
    "output_pulse_oracle_db",
)
CANDIDATE_NAMES = BASELINE_CANDIDATE_NAMES + CURRENT_OUTPUT_REFRESH_CANDIDATES
COMPONENT_PREFIXES = {
    "cpps": "cpps",
    "hnr": "hnr",
    "shimmer_percent": "shimmer_percent",
    "shimmer_db": "shimmer_db",
    "slope": "slope",
    "tilt": "tilt",
}


EXACT_SCORER = r"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

sys.path.insert(0, sys.argv[1])
from avqi_code import run_avqi


SAMPLE_RATE = 16000


def pcm16(values):
    bounded = np.clip(
        np.asarray(values, dtype=np.float64),
        -1.0,
        1.0 - 1.0 / 32768.0,
    )
    return np.rint(bounded * 32768.0).astype(np.int32)


def pcm16_roundtrip(values):
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.asarray(values, dtype=np.float64),
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    buffer.seek(0)
    result, sample_rate = sf.read(buffer, dtype="float64")
    if sample_rate != SAMPLE_RATE:
        raise ValueError("PCM16 roundtrip changed sample rate")
    return result


def praat_wav_roundtrip(sound):
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = handle.name
    handle.close()
    try:
        call(sound, "Save as WAV file", path)
        result, sample_rate = sf.read(path, dtype="float64")
    finally:
        os.unlink(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Praat WAV roundtrip changed sample rate")
    return result


def read_exact_16khz_waveform(path):
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("exact topology input must be mono 16 kHz")
    return waveform.astype(np.float64)


def exact_metric_highpass(waveform):
    input_pcm16 = pcm16_roundtrip(waveform)
    sound = parselmouth.Sound(input_pcm16, SAMPLE_RATE)
    filtered = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
    peak = float(call(filtered, "Get absolute extremum", 0, 0, "Sinc70"))
    if peak > 0.999:
        call(filtered, "Scale peak", 0.99)
    return praat_wav_roundtrip(filtered)


def exact_zero_crossing_rate(part):
    values = np.asarray(part.values[0], dtype=np.float64)
    left = values[:-1]
    right = values[1:]
    indices = np.flatnonzero(
        ((left <= 0.0) & (right > 0.0))
        | ((left >= 0.0) & (right < 0.0))
    )
    if indices.size < 2:
        return float("nan")
    denominator = left[indices] - right[indices]
    fraction = np.divide(
        left[indices],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    crossings = (
        float(part.x1)
        + (indices.astype(np.float64) + fraction) * float(part.dx)
    )
    first = int(np.argmin(np.abs(crossings - 0.0025)))
    last_candidates = np.flatnonzero(crossings[first:] >= 0.0275)
    if last_candidates.size == 0:
        return float("nan")
    last = first + int(last_candidates[0])
    distance = crossings[last] - crossings[first]
    return (last - first) / distance


def compress_source_indices(indices):
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, split_points)
    return [[int(group[0]), int(group.size)] for group in groups]


def exact_cs_metric_waveform(highpassed):
    highpassed_pcm = pcm16(highpassed)
    highpassed_sound = parselmouth.Sound(highpassed, SAMPLE_RATE)

    textgrid = call(
        highpassed_sound,
        "To TextGrid (silences)",
        50,
        0.003,
        -25,
        0.1,
        0.1,
        "silence",
        "sounding",
    )
    interval_count = int(call(textgrid, "Get number of intervals", 1))
    sounding_parts = []
    sounding_source_indices = []
    for index in range(1, interval_count + 1):
        label = call(textgrid, "Get label of interval", 1, index)
        if "silence" in label:
            continue
        start = float(call(textgrid, "Get start point", 1, index))
        end = float(call(textgrid, "Get end point", 1, index))
        part = call(
            highpassed_sound,
            "Extract part",
            start,
            end,
            "rectangular",
            1.0,
            "no",
        )
        part_values = np.asarray(part.values[0], dtype=np.float64)
        source_start = int(round(start * SAMPLE_RATE))
        source_end = source_start + part_values.size
        if (
            source_end > highpassed_pcm.size
            or np.max(
                np.abs(
                    highpassed_pcm[source_start:source_end]
                    - pcm16(part_values)
                )
            )
            != 0
        ):
            raise ValueError("sounding interval failed exact source parity")
        sounding_parts.append(part)
        sounding_source_indices.append(
            np.arange(source_start, source_end, dtype=np.int64)
        )
    if not sounding_parts:
        raise ValueError("exact CS preprocessing found no sounding interval")
    only_loud = call(sounding_parts, "Concatenate")
    only_loud_indices = np.concatenate(sounding_source_indices)
    if only_loud_indices.size != only_loud.n_samples:
        raise ValueError("only-loud source mapping length drift")

    global_power = float(call(only_loud, "Get power in air"))
    left = float(only_loud.xmin)
    width = 0.03
    right = left + width
    extreme_right = float(only_loud.xmax) - width
    kept_parts = []
    kept_source_indices = []
    while right < extreme_right:
        part = call(
            only_loud,
            "Extract part",
            left,
            right,
            "rectangular",
            1.0,
            "no",
        )
        partial_power = float(call(part, "Get power in air"))
        if partial_power > global_power * 0.30:
            zero_crossing_rate = exact_zero_crossing_rate(part)
            if np.isfinite(zero_crossing_rate) and zero_crossing_rate < 3000.0:
                part_values = np.asarray(part.values[0], dtype=np.float64)
                local_start = int(
                    round((left - float(only_loud.xmin)) * SAMPLE_RATE)
                )
                local_end = local_start + part_values.size
                if local_end > only_loud_indices.size:
                    raise ValueError("exact CS frame mapping exceeds only-loud data")
                kept_parts.append(part_values)
                kept_source_indices.append(
                    only_loud_indices[local_start:local_end]
                )
        left += width
        right = left + width
    if not kept_parts:
        raise ValueError("exact CS preprocessing retained no 30-ms frame")

    constant_prefix = round(0.001 * SAMPLE_RATE)
    metric_values = np.concatenate(
        [np.zeros(constant_prefix, dtype=np.float64)] + kept_parts
    )
    metric = praat_wav_roundtrip(
        parselmouth.Sound(metric_values, SAMPLE_RATE)
    )
    selected_indices = np.concatenate(kept_source_indices)
    ranges = compress_source_indices(selected_indices)
    reconstructed = np.concatenate(
        [np.zeros(constant_prefix, dtype=np.float64)]
        + [highpassed[start : start + length] for start, length in ranges]
    )
    difference = np.abs(pcm16(reconstructed) - pcm16(metric))
    maximum_error = int(difference.max(initial=0))
    differing_samples = int(np.count_nonzero(difference))
    if reconstructed.size != metric.size or maximum_error != 0:
        raise ValueError("exact CS metric source mapping failed waveform parity")
    return (
        metric,
        constant_prefix,
        ranges,
        maximum_error,
        differing_samples,
    )


def exact_sv_metric_waveform(highpassed):
    metric_sample_count = min(highpassed.size, 3 * SAMPLE_RATE)
    crop_start = highpassed.size - metric_sample_count
    metric = highpassed[crop_start:].copy()
    constant_prefix = 0
    ranges = [[int(crop_start), int(metric.size)]]
    reconstructed = highpassed[crop_start : crop_start + metric.size]
    difference = np.abs(pcm16(reconstructed) - pcm16(metric))
    maximum_error = int(difference.max(initial=0))
    differing_samples = int(np.count_nonzero(difference))
    if maximum_error != 0:
        raise ValueError("exact SV metric crop failed waveform parity")
    return metric, constant_prefix, ranges, maximum_error, differing_samples


def point_process_positions(sound):
    point_process = call(sound, "To PointProcess (periodic, cc)", 50, 400)
    count = int(call(point_process, "Get number of points"))
    return [
        (
            float(call(point_process, "Get time from index", index))
            - float(sound.x1)
        )
        / float(sound.dx)
        for index in range(1, count + 1)
    ]


def pulse_positions(path, view, exact_metric_topology):
    started = time.perf_counter()
    if exact_metric_topology:
        waveform = read_exact_16khz_waveform(path)
        highpassed = exact_metric_highpass(waveform)
        if view == "sv":
            (
                metric,
                constant_prefix,
                ranges,
                maximum_error,
                differing_samples,
            ) = exact_sv_metric_waveform(highpassed)
        elif view == "cs":
            (
                metric,
                constant_prefix,
                ranges,
                maximum_error,
                differing_samples,
            ) = exact_cs_metric_waveform(highpassed)
        else:
            raise ValueError("unsupported view: " + view)
        sound = parselmouth.Sound(metric, SAMPLE_RATE)
        positions = point_process_positions(sound)
        metadata = {
            "topology_preprocessing": "exact_avqi_view_metric_waveform",
            "source_sample_count": int(highpassed.size),
            "metric_sample_count": int(metric.size),
            "metric_constant_prefix_samples": int(constant_prefix),
            "metric_source_ranges": ranges,
            "metric_source_range_count": len(ranges),
            "metric_mapped_sample_count": int(
                sum(length for _, length in ranges)
            ),
            "metric_reconstruction_max_pcm16_error": maximum_error,
            "metric_reconstruction_differing_samples": differing_samples,
            "topology_input_loader": "soundfile_float32_exact_16khz_mono",
            "metric_highpass": "exact_in_process_praat_stop_hann_0_34_0p1",
        }
        return (
            positions,
            1000.0 * (time.perf_counter() - started),
            metadata,
        )

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
    positions = point_process_positions(sound)
    return (
        positions,
        1000.0 * (time.perf_counter() - started),
        {
            "topology_preprocessing": "legacy_direct_full_cs_or_final_3s_sv",
            "source_sample_count": int(sound.n_samples),
            "metric_sample_count": int(sound.n_samples),
            "metric_constant_prefix_samples": 0,
            "metric_source_ranges": [],
            "metric_source_range_count": 0,
            "metric_mapped_sample_count": 0,
            "metric_reconstruction_max_pcm16_error": 0,
            "metric_reconstruction_differing_samples": 0,
        },
    )


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
        positions, runtime_ms, topology_metadata = pulse_positions(
            item["path"],
            item["view"],
            bool(item.get("exact_metric_topology", False)),
        )
        row.update(
            {
                "scoring_status": "ok",
                "pulse_positions_samples": positions,
                "pulse_count": len(positions),
                "pulse_runtime_ms": runtime_ms,
                "error_type": "",
                "error_message": "",
                **topology_metadata,
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
                "topology_preprocessing": "error",
                "source_sample_count": 0,
                "metric_sample_count": 0,
                "metric_constant_prefix_samples": 0,
                "metric_source_ranges": [],
                "metric_source_range_count": 0,
                "metric_mapped_sample_count": 0,
                "metric_reconstruction_max_pcm16_error": -1,
                "metric_reconstruction_differing_samples": -1,
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
    metric_source_indices: torch.Tensor | None,
    metric_constant_prefix_samples: int,
) -> tuple[torch.Tensor, str]:
    if candidate == "praat_input_topology_absolute_db":
        proxy = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            input_pulses,
            metric_sample_count=SV_METRIC_SAMPLES if view == "sv" else None,
        )[1]
        return proxy, "shimmer_db"
    if (
        candidate == "output_pulse_oracle_db"
        or candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
    ):
        proxy = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            output_pulses,
            metric_source_indices=metric_source_indices,
            metric_constant_prefix_samples=metric_constant_prefix_samples,
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


def candidate_alpha(candidate: str) -> float:
    if candidate not in CURRENT_OUTPUT_REFRESH_CANDIDATES:
        return FIXED_ALPHA
    return CURRENT_OUTPUT_REFRESH_ALPHAS[
        CURRENT_OUTPUT_REFRESH_CANDIDATES.index(candidate)
    ]


def shared_refresh_topology_aliases(
    panel_rows: list[dict[str, Any]],
    metric_output_topology_by_case: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Alias every frozen alpha to one detached topology per waveform."""
    return {
        (candidate, row["case_id"]): metric_output_topology_by_case[
            row["case_id"]
        ]
        for candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
        for row in panel_rows
    }


def normalized_gradient_step(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    gradient_rms = gradient.square().mean().sqrt()
    base_rms = waveform.square().mean().sqrt()
    if float(gradient_rms) <= 1e-15:
        return waveform.detach().clone()
    return (
        waveform.detach()
        - alpha * base_rms * gradient / gradient_rms
    )


def pulse_positions_sha256(positions: list[float] | np.ndarray) -> str:
    values = np.asarray(positions, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


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


def metric_source_indices_from_topology(
    topology: dict[str, Any],
    *,
    source_sample_count: int,
) -> np.ndarray:
    if topology.get("topology_preprocessing") != "exact_avqi_view_metric_waveform":
        raise ValueError("topology does not describe an exact AVQI metric waveform")
    if int(topology["source_sample_count"]) != source_sample_count:
        raise ValueError("exact metric topology source length drift")
    ranges = topology["metric_source_ranges"]
    pieces: list[np.ndarray] = []
    previous_end = 0
    for start_value, length_value in ranges:
        start = int(start_value)
        length = int(length_value)
        end = start + length
        if length <= 0 or start < previous_end or end > source_sample_count:
            raise ValueError("invalid exact metric source range")
        pieces.append(np.arange(start, end, dtype=np.int64))
        previous_end = end
    if not pieces:
        raise ValueError("exact metric topology contains no source ranges")
    indices = np.concatenate(pieces)
    if indices.size != int(topology["metric_mapped_sample_count"]):
        raise ValueError("exact metric mapped sample count drift")
    expected_metric_samples = (
        int(topology["metric_constant_prefix_samples"]) + indices.size
    )
    if expected_metric_samples != int(topology["metric_sample_count"]):
        raise ValueError("exact metric waveform length drift")
    if (
        int(topology["metric_reconstruction_max_pcm16_error"]) != 0
        or int(topology["metric_reconstruction_differing_samples"]) != 0
    ):
        raise ValueError("exact metric source mapping failed waveform parity")
    return indices


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
    forward_errors = [
        row["forward_normalized_abs_error_shimmer_db"]
        for row in selected
        if row["forward_normalized_abs_error_shimmer_db"] is not None
    ]
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
        "pulse_refresh_runtime_ms": {
            "median": median(
                row["pulse_refresh_runtime_ms"] for row in selected
            ),
            "maximum": max(
                row["pulse_refresh_runtime_ms"] for row in selected
            ),
        },
        "torch_step_runtime_ms": {
            "median": median(row["torch_step_runtime_ms"] for row in selected),
            "maximum": max(row["torch_step_runtime_ms"] for row in selected),
        },
        "total_metric_step_overhead_ms": {
            "median": median(
                row["total_metric_step_overhead_ms"] for row in selected
            ),
            "maximum": max(
                row["total_metric_step_overhead_ms"] for row in selected
            ),
        },
        "forward_normalized_abs_error_shimmer_db": {
            "median": median(forward_errors) if forward_errors else None,
            "maximum": max(forward_errors) if forward_errors else None,
        },
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
            "pulse_refresh_runtime": max(
                row["pulse_refresh_runtime_ms"] for row in selected
            )
            <= CACHE_RUNTIME_MAX_MS,
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
    metric_output_topology_items = []
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
        metric_output_topology_items.append(
            {
                "id": f"metric_base_output:{row['case_id']}",
                "case_id": row["case_id"],
                "role": "output_pulse_oracle_db",
                "path": row["base_path"],
                "view": row["view"],
                "score_components": False,
                "exact_metric_topology": True,
            }
        )
    topology_exact = run_exact(
        topology_items,
        args.exact_python,
        args.avqi_code_root,
    )
    refresh_batch_started = time.perf_counter()
    refresh_topology_exact = run_exact(
        metric_output_topology_items,
        args.exact_python,
        args.avqi_code_root,
    )
    refresh_batch_wall_ms = 1000.0 * (
        time.perf_counter() - refresh_batch_started
    )
    if (
        refresh_topology_exact["parselmouth_version"]
        != topology_exact["parselmouth_version"]
        or refresh_topology_exact["praat_version"]
        != topology_exact["praat_version"]
    ):
        raise ValueError("exact runtime drift during Candidate-C refresh")
    topology_by_id = {
        row["id"]: row
        for row in (
            topology_exact["rows"]
            + refresh_topology_exact["rows"]
        )
    }

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
    metric_output_topology_by_case = {
        row["case_id"]: topology_by_id[
            f"metric_base_output:{row['case_id']}"
        ]
        for row in panel_rows
    }
    refresh_topology_by_candidate_case = shared_refresh_topology_aliases(
        panel_rows,
        metric_output_topology_by_case,
    )

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
        metric_output_topology = metric_output_topology_by_case[case_id]
        if cache["scoring_status"] != "ok" or cache["pulse_count"] < 3:
            continue
        if output_topology["scoring_status"] != "ok" or output_topology["pulse_count"] < 3:
            continue
        if (
            metric_output_topology["scoring_status"] != "ok"
            or metric_output_topology["pulse_count"] < 3
        ):
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
            if candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES:
                step_topology = refresh_topology_by_candidate_case[
                    (candidate, case_id)
                ]
            elif candidate == "output_pulse_oracle_db":
                step_topology = metric_output_topology
            else:
                step_topology = output_topology
            if (
                step_topology["scoring_status"] != "ok"
                or step_topology["pulse_count"] < 3
            ):
                continue
            output_pulses = waveform.new_tensor(
                step_topology["pulse_positions_samples"]
            )
            exact_metric_branch = (
                candidate == "output_pulse_oracle_db"
                or candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
            )
            if exact_metric_branch:
                source_indices_np = metric_source_indices_from_topology(
                    step_topology,
                    source_sample_count=waveform.numel(),
                )
                metric_source_indices = torch.as_tensor(
                    source_indices_np,
                    device=waveform.device,
                    dtype=torch.long,
                )
                metric_constant_prefix_samples = int(
                    step_topology["metric_constant_prefix_samples"]
                )
            else:
                metric_source_indices = None
                metric_constant_prefix_samples = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            torch_step_started = time.perf_counter()
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
                metric_source_indices,
                metric_constant_prefix_samples,
            )
            optimized_index = AVQI_COMPONENT_NAMES.index(optimized_component)
            target_value = float(reference[f"exact_target_{optimized_component}"])
            loss = (
                (proxy - target_value)
                / target_scale[optimized_index].clamp_min(1e-8)
            ).square()
            gradient = torch.autograd.grad(loss, waveform)[0]
            alpha = candidate_alpha(candidate)
            candidate_waveform = normalized_gradient_step(
                waveform,
                gradient,
                alpha,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            torch_step_runtime_ms = 1000.0 * (
                time.perf_counter() - torch_step_started
            )
            pulse_refresh_runtime_ms = (
                float(step_topology["pulse_runtime_ms"])
                if candidate == "output_pulse_oracle_db"
                or candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
                else 0.0
            )
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
                "fixed_alpha": alpha,
                "proxy_before": float(proxy.detach()),
                "proxy_target": target_value,
                "proxy_loss": float(loss.detach()),
                "gradient_l2_norm": float(gradient.norm()),
                "gradient_rms": float(gradient.square().mean().sqrt()),
                "gradient_finite": bool(torch.isfinite(gradient).all()),
                "pulse_topology_role": (
                    candidate if exact_metric_branch else str(step_topology["role"])
                ),
                "pulse_topology_source_role": str(step_topology["role"]),
                "unique_topology_refresh_key": (
                    f"metric_base_output:{case_id}" if exact_metric_branch else ""
                ),
                "topology_reused_across_alpha_grid": bool(
                    candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
                ),
                "pulse_topology_sha256": pulse_positions_sha256(
                    step_topology["pulse_positions_samples"]
                ),
                "topology_preprocessing": step_topology[
                    "topology_preprocessing"
                ],
                "metric_sample_count": step_topology["metric_sample_count"],
                "metric_constant_prefix_samples": step_topology[
                    "metric_constant_prefix_samples"
                ],
                "metric_source_range_count": step_topology[
                    "metric_source_range_count"
                ],
                "metric_mapped_sample_count": step_topology[
                    "metric_mapped_sample_count"
                ],
                "metric_reconstruction_max_pcm16_error": step_topology[
                    "metric_reconstruction_max_pcm16_error"
                ],
                "metric_reconstruction_differing_samples": step_topology[
                    "metric_reconstruction_differing_samples"
                ],
                "pulse_refresh_runtime_ms": pulse_refresh_runtime_ms,
                "torch_step_runtime_ms": torch_step_runtime_ms,
                "total_metric_step_overhead_ms": (
                    pulse_refresh_runtime_ms + torch_step_runtime_ms
                ),
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "cache_record_sha256": cache["record_sha256"],
                "cache_pulse_count": cache["pulse_count"],
                "mapped_cache_pulse_count": int(mapped_input_positions.size),
                "base_output_pulse_count": output_topology["pulse_count"],
                "base_output_exact_metric_pulse_count": metric_output_topology[
                    "pulse_count"
                ],
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
                    "exact_metric_topology": exact_metric_branch,
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
        if candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES:
            topology_reference = refresh_topology_by_candidate_case[
                (candidate, case_id)
            ]
        elif candidate == "output_pulse_oracle_db":
            topology_reference = metric_output_topology_by_case[case_id]
        else:
            topology_reference = output_topology_by_case[case_id]
        reference_positions = np.asarray(
            topology_reference["pulse_positions_samples"],
            dtype=np.float64,
        )
        after_positions = np.asarray(
            exact["pulse_positions_samples"],
            dtype=np.float64,
        )
        cache_to_base_match = nearest_match_rate(cache_positions, base_positions)
        cache_to_candidate_match = nearest_match_rate(
            cache_positions,
            after_positions,
        )
        reference_to_candidate_match = nearest_match_rate(
            reference_positions,
            after_positions,
        )
        candidate_to_reference_match = nearest_match_rate(
            after_positions,
            reference_positions,
        )
        candidate_reference_count_ratio = after_positions.size / max(
            reference_positions.size,
            1,
        )
        topology_pass = (
            reference_to_candidate_match >= 1.0 - TOPOLOGY_MATCH_DROP_MAX
            and candidate_to_reference_match >= 1.0 - TOPOLOGY_MATCH_DROP_MAX
            and abs(candidate_reference_count_ratio - 1.0)
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
        forward_normalized_error = (
            abs(
                record["proxy_before"]
                - component_fields["exact_before_shimmer_db"]
            )
            / max(target_scale_np[SHIMMER_DB_INDEX], 1e-8)
            if record["optimized_component"] == "shimmer_db"
            else None
        )
        csv_rows.append(
            {
                **record,
                **component_fields,
                "material_shimmer_db_gap": material,
                "candidate_exact_pulse_count": exact["pulse_count"],
                "cache_to_base_match_rate_16_samples": cache_to_base_match,
                "cache_to_candidate_match_rate_16_samples": (
                    cache_to_candidate_match
                ),
                "topology_reference_role": topology_reference["role"],
                "reference_to_candidate_match_rate_16_samples": (
                    reference_to_candidate_match
                ),
                "candidate_to_reference_match_rate_16_samples": (
                    candidate_to_reference_match
                ),
                "candidate_reference_pulse_count_ratio": (
                    candidate_reference_count_ratio
                ),
                "candidate_metric_sample_count": exact[
                    "metric_sample_count"
                ],
                "candidate_metric_to_reference_sample_count_ratio": (
                    int(exact["metric_sample_count"])
                    / max(int(topology_reference["metric_sample_count"]), 1)
                ),
                "candidate_metric_reconstruction_max_pcm16_error": exact[
                    "metric_reconstruction_max_pcm16_error"
                ],
                "candidate_metric_reconstruction_differing_samples": exact[
                    "metric_reconstruction_differing_samples"
                ],
                "forward_normalized_abs_error_shimmer_db": (
                    forward_normalized_error
                ),
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
    passing_refresh_candidates = [
        candidate
        for candidate in CURRENT_OUTPUT_REFRESH_CANDIDATES
        if candidate_aggregates[candidate]["all_gates_pass"]
    ]
    selected_refresh_candidate = (
        min(
            passing_refresh_candidates,
            key=lambda candidate: (
                -candidate_aggregates[candidate][
                    "median_exact_db_normalized_gap_reduction"
                ],
                candidate_alpha(candidate),
            ),
        )
        if passing_refresh_candidates
        else None
    )
    selected_refresh_alpha = (
        candidate_alpha(selected_refresh_candidate)
        if selected_refresh_candidate is not None
        else None
    )
    oracle_rows = {
        row["case_id"]: row
        for row in csv_rows
        if row["candidate"] == "output_pulse_oracle_db"
    }
    refresh_alpha_001 = CURRENT_OUTPUT_REFRESH_CANDIDATES[1]
    refresh_alpha_001_rows = {
        row["case_id"]: row
        for row in csv_rows
        if row["candidate"] == refresh_alpha_001
    }
    oracle_coverage_equal = set(oracle_rows) == set(refresh_alpha_001_rows)
    oracle_alias_equivalence = {
        "case_coverage_equal": oracle_coverage_equal,
        "pulse_topology_hash_equal": oracle_coverage_equal
        and all(
            oracle_rows[case_id]["pulse_topology_sha256"]
            == refresh_alpha_001_rows[case_id]["pulse_topology_sha256"]
            for case_id in oracle_rows
        ),
        "waveform_hash_equal": oracle_coverage_equal
        and all(
            oracle_rows[case_id]["output_sha256"]
            == refresh_alpha_001_rows[case_id]["output_sha256"]
            for case_id in oracle_rows
        ),
        "exact_six_component_values_equal": oracle_coverage_equal
        and all(
            all(
                oracle_rows[case_id][f"exact_after_{COMPONENT_PREFIXES[name]}"]
                == refresh_alpha_001_rows[case_id][
                    f"exact_after_{COMPONENT_PREFIXES[name]}"
                ]
                for name in AVQI_COMPONENT_NAMES
            )
            for case_id in oracle_rows
        ),
    }
    oracle_alias_equivalence["proved_equal"] = all(
        oracle_alias_equivalence.values()
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-hybrid-opened-diagnostic-v3",
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
        "candidate_c_decision": (
            "PASS_CURRENT_OUTPUT_EXACT_TOPOLOGY_REFRESH_FREEZE_FOR_FRESH_PANEL"
            if selected_refresh_candidate is not None
            else "FAIL_CURRENT_OUTPUT_EXACT_TOPOLOGY_REFRESH_DO_NOT_PROMOTE"
        ),
        "panel_status": "already_opened_mechanism_diagnostic_only",
        "fresh_panel_authorized": selected_refresh_candidate is not None,
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
        "candidate_c": {
            "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
            "pure_torch_estimator": False,
            "topology_detached": True,
            "exact_avqi_view_preprocessing": True,
            "cs_topology_waveform": (
                "metric high-pass then exact sounding/30-ms concatenation"
            ),
            "torch_backward_waveform": (
                "live metric-high-passed samples gathered by detached exact ranges"
            ),
            "metric_mapping_parity": {
                "all_refresh_rows_zero_pcm16_error": all(
                    row["metric_reconstruction_max_pcm16_error"] == 0
                    and row["metric_reconstruction_differing_samples"] == 0
                    for row in refresh_topology_exact["rows"]
                ),
                "cs_constant_prefix_samples": sorted(
                    {
                        row["metric_constant_prefix_samples"]
                        for row in refresh_topology_exact["rows"]
                        if row["view"] == "cs"
                    }
                ),
                "cs_source_range_count_min": min(
                    row["metric_source_range_count"]
                    for row in refresh_topology_exact["rows"]
                    if row["view"] == "cs"
                ),
                "cs_source_range_count_max": max(
                    row["metric_source_range_count"]
                    for row in refresh_topology_exact["rows"]
                    if row["view"] == "cs"
                ),
            },
            "pulse_extractor_called_per_candidate_step": False,
            "pulse_extractor_called_once_per_waveform_step": True,
            "unique_pulse_refresh_calls": len(metric_output_topology_items),
            "diagnostic_alpha_candidate_count": len(
                CURRENT_OUTPUT_REFRESH_ALPHAS
            ),
            "alpha_grid_topology_reused": True,
            "pulse_refresh_calls": len(metric_output_topology_items),
            "pulse_refresh_batch_wall_ms": refresh_batch_wall_ms,
            "pulse_refresh_amortized_wall_ms_per_unique_waveform_step": (
                refresh_batch_wall_ms / len(metric_output_topology_items)
            ),
            "pulse_refresh_internal_runtime_ms": {
                "median": median(
                    row["pulse_runtime_ms"]
                    for row in refresh_topology_exact["rows"]
                ),
                "maximum": max(
                    row["pulse_runtime_ms"]
                    for row in refresh_topology_exact["rows"]
                ),
            },
            "frozen_runtime_profile_evidence": {
                "job_id": RUNTIME_PROFILE_JOB_ID,
                "source_commit": RUNTIME_PROFILE_SOURCE_COMMIT,
                "report_sha256": RUNTIME_PROFILE_REPORT_SHA256,
                "reuse_equivalence_sha256": RUNTIME_PROFILE_REUSE_SHA256,
                "cold_total_refresh_maximum_ms": 1158.4908701479435,
                "warm_total_refresh_maximum_ms": 425.17857486382127,
                "shared_refresh_maximum_ms": 420.25938304141164,
                "authority_parity_pass": True,
                "separate_vs_shared_exact_equivalence_count": 18,
                "separate_vs_shared_exact_equivalence_pass": True,
            },
            "alpha_grid_frozen_before_run": list(CURRENT_OUTPUT_REFRESH_ALPHAS),
            "passing_candidates": passing_refresh_candidates,
            "selected_candidate": selected_refresh_candidate,
            "selected_alpha": selected_refresh_alpha,
            "oracle_alias_at_alpha_0p001": oracle_alias_equivalence,
        },
        "alpha_selected_from_opened_panel": selected_refresh_alpha is not None,
        "opened_panel_alpha_use": (
            "mechanism_parameter_freeze_only; never component promotion"
        ),
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
            "required_effect_slices": list(REQUIRED_EFFECT_SLICES),
            "pulse_refresh_runtime_max_ms": CACHE_RUNTIME_MAX_MS,
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
            "The supervisor-authorized frozen three-point grid can freeze a Candidate-C mechanism alpha here but cannot promote the component.",
            "Candidate C is Praat-assisted and cannot be reported as a pure-PyTorch estimator.",
            "The prior v10 CS topology used the full high-passed recording; v11 corrects it to the exact AVQI concatenated metric waveform without changing alpha or gates.",
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
        "schema_version": "avqi-route-c-shimmer-hybrid-opened-receipt-v3",
        "decision": report["decision"],
        "coupled_baseline_decision": report["coupled_baseline_decision"],
        "candidate_c_decision": report["candidate_c_decision"],
        "candidate_c_selected_alpha": report["candidate_c"]["selected_alpha"],
        "candidate_c_oracle_alias_proved_equal": report["candidate_c"]
        ["oracle_alias_at_alpha_0p001"]["proved_equal"],
        "fresh_panel_authorized": report["fresh_panel_authorized"],
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
