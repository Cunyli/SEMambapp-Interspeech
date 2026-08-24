#!/usr/bin/env python3
"""Profile exact-output Shimmer topology refresh and alpha-grid reuse.

This is an opened-panel runtime/mechanism audit, not a promotion experiment.
Six frozen CS base outputs are profiled in dedicated persistent exact-Praat
workers.  Each worker reports a first (cold) and second (warm) refresh with
stage-level timing.  A separate-vs-shared topology audit then proves that one
detached refresh can be reused by the frozen three-alpha mechanism grid without
changing the proxy, gradient, waveform, or independently recomputed exact AVQI
components.  No generator optimizer is loaded or stepped.
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
    PraatDifferentiableAVQIComponentEstimator,
)


SAMPLE_RATE = 16_000
FROZEN_ALPHA_GRID = (3e-4, 1e-3, 3e-3)
SHIMMER_DB_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_db")
PULSE_REFRESH_GATE_MS = 500.0
EXPECTED_CS_CASE_IDS = (
    "calibration__FD26__cs__rir_only",
    "calibration__SD36__cs__snr10",
    "calibration__FD11__cs__snr20",
    "final__ÄHH20__cs__rir_only",
    "final__FD20__cs__snr10",
    "final__SD23__cs__snr20",
)
EXACT_READY_MARKER = "AVQI_SHIMMER_RUNTIME_READY="
EXACT_RESULT_MARKER = "AVQI_SHIMMER_RUNTIME_RESULT="
RUNTIME_PHASES = ("cold", "warm")
COMPONENT_STEP_VERSIONS = {
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
}


EXACT_WORKER = r"""
import hashlib
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
from avqi_code.main import (
    get_voiced_segments,
    highpass_filter,
    read_and_resample_signal,
)


SAMPLE_RATE = 16000
MAPPING_WINDOW = 32
RESULT_MARKER = "AVQI_SHIMMER_RUNTIME_RESULT="


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


def bytes_sha256(values):
    return hashlib.sha256(values).hexdigest()


def pcm16_sha256(values):
    return bytes_sha256(pcm16(values).astype("<i4", copy=False).tobytes())


def ranges_sha256(ranges):
    encoded = json.dumps(ranges, separators=(",", ":")).encode("utf-8")
    return bytes_sha256(encoded)


def pulses_sha256(positions):
    values = np.asarray(positions, dtype="<f8")
    return bytes_sha256(values.tobytes())


def exact_zero_crossing_rate_values(values, x1, dx):
    values = np.asarray(values, dtype=np.float64)
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
        float(x1)
        + (indices.astype(np.float64) + fraction) * float(dx)
    )
    first = int(np.argmin(np.abs(crossings - 0.0025)))
    last_candidates = np.flatnonzero(crossings[first:] >= 0.0275)
    if last_candidates.size == 0:
        return float("nan")
    last = first + int(last_candidates[0])
    distance = crossings[last] - crossings[first]
    return (last - first) / distance


def exact_zero_crossing_rate(part):
    return exact_zero_crossing_rate_values(
        part.values[0],
        part.x1,
        part.dx,
    )


def compress_source_indices(indices):
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, split_points)
    return [[int(group[0]), int(group.size)] for group in groups]


def find_monotonic_window(source, target, source_start, target_start):
    width = min(MAPPING_WINDOW, target.size - target_start)
    if width <= 0:
        raise ValueError("empty exact metric mapping window")
    candidates = (
        np.flatnonzero(source[source_start:] == target[target_start])
        + source_start
    )
    for tolerance in (0, 1):
        for candidate in candidates:
            if candidate + width > source.size:
                continue
            error = np.max(
                np.abs(
                    source[candidate : candidate + width]
                    - target[target_start : target_start + width]
                )
            )
            if error <= tolerance:
                return int(candidate), tolerance
    raise ValueError("no monotonic authoritative source window")


def monotonic_ranges(source_values, target_values):
    source = pcm16(source_values)
    target = pcm16(target_values)
    constant_prefix = 0
    while constant_prefix < target.size and target[constant_prefix] == 0:
        constant_prefix += 1
    ranges = []
    target_index = constant_prefix
    source_cursor = 0
    while target_index < target.size:
        source_index, tolerance = find_monotonic_window(
            source,
            target,
            source_cursor,
            target_index,
        )
        run = 0
        while (
            target_index + run < target.size
            and source_index + run < source.size
            and abs(
                int(source[source_index + run])
                - int(target[target_index + run])
            )
            <= tolerance
        ):
            run += 1
        if run < min(MAPPING_WINDOW, target.size - target_index):
            raise ValueError("authoritative source mapping produced a short run")
        ranges.append([source_index, run])
        target_index += run
        source_cursor = source_index + run
    reconstructed = np.concatenate(
        [np.zeros(constant_prefix, dtype=np.float64)]
        + [source_values[start : start + length] for start, length in ranges]
    )
    if not np.array_equal(pcm16(reconstructed), target):
        raise ValueError("authoritative source mapping failed PCM16 parity")
    return constant_prefix, ranges


def enumerate_point_process(point_process, sound):
    count = int(call(point_process, "Get number of points"))
    return [
        (
            float(call(point_process, "Get time from index", index))
            - float(sound.x1)
        )
        / float(sound.dx)
        for index in range(1, count + 1)
    ]


def enumerate_point_process_matrix(point_process, sound):
    matrix = call(point_process, "To Matrix")
    times = np.asarray(matrix.values, dtype=np.float64).reshape(-1)
    return ((times - float(sound.x1)) / float(sound.dx)).tolist()


def read_soundfile_exact(path):
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("exact topology input must be mono 16 kHz")
    return waveform.astype(np.float64)


def direct_refresh(
    path,
    verify_authority,
    input_loader,
    frame_scan_mode,
    pulse_enumeration_mode,
    wav_roundtrip_mode,
    sounding_assembly_mode,
):
    total_started = time.perf_counter()
    input_started = time.perf_counter()
    if input_loader == "authoritative_read_and_resample":
        waveform = read_and_resample_signal(path, SAMPLE_RATE)
    elif input_loader == "soundfile_float32_exact_16khz_mono":
        waveform = read_soundfile_exact(path)
    else:
        raise ValueError("unsupported exact input loader: " + input_loader)
    input_read_ms = 1000.0 * (time.perf_counter() - input_started)

    highpass_started = time.perf_counter()
    input_pcm16 = pcm16_roundtrip(waveform)
    sound = parselmouth.Sound(input_pcm16, SAMPLE_RATE)
    filter_started = time.perf_counter()
    filtered = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
    peak = float(call(filtered, "Get absolute extremum", 0, 0, "Sinc70"))
    if peak > 0.999:
        call(filtered, "Scale peak", 0.99)
    highpass_filter_compute_ms = 1000.0 * (
        time.perf_counter() - filter_started
    )
    quantize_started = time.perf_counter()
    if wav_roundtrip_mode == "praat_temp_wav":
        highpassed = praat_wav_roundtrip(filtered)
    elif wav_roundtrip_mode == "soundfile_in_memory_pcm16":
        highpassed = pcm16_roundtrip(filtered.values[0])
    else:
        raise ValueError("unsupported WAV roundtrip: " + wav_roundtrip_mode)
    highpass_quantize_ms = 1000.0 * (
        time.perf_counter() - quantize_started
    )
    highpass_ms = 1000.0 * (time.perf_counter() - highpass_started)

    textgrid_started = time.perf_counter()
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
    textgrid_ms = 1000.0 * (time.perf_counter() - textgrid_started)

    selection_started = time.perf_counter()
    interval_count = int(call(textgrid, "Get number of intervals", 1))
    sounding_parts = []
    sounding_values = []
    sounding_source_indices = []
    highpassed_pcm = pcm16(highpassed)
    for index in range(1, interval_count + 1):
        label = call(textgrid, "Get label of interval", 1, index)
        if "silence" in label:
            continue
        start = float(call(textgrid, "Get start point", 1, index))
        end = float(call(textgrid, "Get end point", 1, index))
        source_start = int(
            round((start - float(highpassed_sound.xmin)) * SAMPLE_RATE)
        )
        if sounding_assembly_mode == "praat_extract_and_concatenate":
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
            source_end = source_start + part_values.size
            if source_end > highpassed_pcm.size or not np.array_equal(
                highpassed_pcm[source_start:source_end],
                pcm16(part_values),
            ):
                raise ValueError("sounding interval failed exact source parity")
            sounding_parts.append(part)
        elif sounding_assembly_mode == "numpy_exact_interval_slices":
            source_end = int(
                round((end - float(highpassed_sound.xmin)) * SAMPLE_RATE)
            )
            if source_end <= source_start or source_end > highpassed.size:
                raise ValueError("invalid exact sounding interval slice")
            part_values = highpassed[source_start:source_end]
            sounding_values.append(part_values)
        else:
            raise ValueError(
                "unsupported sounding assembly: " + sounding_assembly_mode
            )
        sounding_source_indices.append(
            np.arange(source_start, source_end, dtype=np.int64)
        )
    if not sounding_source_indices:
        raise ValueError("exact CS preprocessing found no sounding interval")
    if sounding_assembly_mode == "praat_extract_and_concatenate":
        only_loud = call(sounding_parts, "Concatenate")
    else:
        only_loud = parselmouth.Sound(
            np.concatenate(sounding_values),
            SAMPLE_RATE,
        )
    only_loud_indices = np.concatenate(sounding_source_indices)
    if only_loud_indices.size != only_loud.n_samples:
        raise ValueError("only-loud source mapping length drift")

    global_power = float(call(only_loud, "Get power in air"))
    only_loud_values = np.asarray(only_loud.values[0], dtype=np.float64)
    left = float(only_loud.xmin)
    width = 0.03
    right = left + width
    extreme_right = float(only_loud.xmax) - width
    kept_parts = []
    kept_source_indices = []
    while right < extreme_right:
        local_start = int(
            round((left - float(only_loud.xmin)) * SAMPLE_RATE)
        )
        if frame_scan_mode == "praat_per_frame":
            part = call(
                only_loud,
                "Extract part",
                left,
                right,
                "rectangular",
                1.0,
                "no",
            )
            part_values = np.asarray(part.values[0], dtype=np.float64)
            partial_power = float(call(part, "Get power in air"))
            zero_crossing_rate = None
        elif frame_scan_mode == "numpy_exact_aligned_frames":
            frame_sample_count = int(round(width * SAMPLE_RATE))
            part_values = only_loud_values[
                local_start : local_start + frame_sample_count
            ]
            if part_values.size != frame_sample_count:
                raise ValueError("NumPy exact frame scan exceeded only-loud data")
            partial_power = float(np.mean(np.square(part_values)) / 400.0)
            zero_crossing_rate = None
        else:
            raise ValueError("unsupported exact frame scan: " + frame_scan_mode)
        if partial_power > global_power * 0.30:
            if frame_scan_mode == "praat_per_frame":
                zero_crossing_rate = exact_zero_crossing_rate(part)
            else:
                zero_crossing_rate = exact_zero_crossing_rate_values(
                    part_values,
                    0.5 / SAMPLE_RATE,
                    1.0 / SAMPLE_RATE,
                )
            if np.isfinite(zero_crossing_rate) and zero_crossing_rate < 3000.0:
                local_end = local_start + part_values.size
                if local_end > only_loud_indices.size:
                    raise ValueError("exact frame mapping exceeds only-loud data")
                kept_parts.append(part_values)
                kept_source_indices.append(
                    only_loud_indices[local_start:local_end]
                )
        left += width
        right = left + width
    if not kept_parts:
        raise ValueError("exact CS preprocessing retained no 30-ms frame")
    source_selection_ms = 1000.0 * (
        time.perf_counter() - selection_started
    )

    gather_started = time.perf_counter()
    constant_prefix = round(0.001 * SAMPLE_RATE)
    metric_values = np.concatenate(
        [np.zeros(constant_prefix, dtype=np.float64)] + kept_parts
    )
    if wav_roundtrip_mode == "praat_temp_wav":
        metric = praat_wav_roundtrip(
            parselmouth.Sound(metric_values, SAMPLE_RATE)
        )
    else:
        metric = pcm16_roundtrip(metric_values)
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
        raise ValueError("direct metric gather failed exact PCM16 parity")
    metric_gather_ms = 1000.0 * (time.perf_counter() - gather_started)

    construct_started = time.perf_counter()
    metric_sound = parselmouth.Sound(metric, SAMPLE_RATE)
    point_process = call(
        metric_sound,
        "To PointProcess (periodic, cc)",
        50,
        400,
    )
    pointprocess_construct_ms = 1000.0 * (
        time.perf_counter() - construct_started
    )
    enumeration_started = time.perf_counter()
    if pulse_enumeration_mode == "praat_per_point":
        pulse_positions = enumerate_point_process(point_process, metric_sound)
    elif pulse_enumeration_mode == "praat_pointprocess_to_matrix":
        pulse_positions = enumerate_point_process_matrix(
            point_process,
            metric_sound,
        )
    else:
        raise ValueError(
            "unsupported pulse enumeration: " + pulse_enumeration_mode
        )
    pulse_enumeration_ms = 1000.0 * (
        time.perf_counter() - enumeration_started
    )
    total_refresh_ms = 1000.0 * (time.perf_counter() - total_started)

    result = {
        "status": "ok",
        "path": path,
        "topology_input_loader": input_loader,
        "frame_scan_mode": frame_scan_mode,
        "pulse_enumeration_mode": pulse_enumeration_mode,
        "wav_roundtrip_mode": wav_roundtrip_mode,
        "sounding_assembly_mode": sounding_assembly_mode,
        "source_sample_count": int(highpassed.size),
        "metric_sample_count": int(metric.size),
        "metric_constant_prefix_samples": int(constant_prefix),
        "metric_source_ranges": ranges,
        "metric_source_range_count": len(ranges),
        "metric_mapped_sample_count": int(selected_indices.size),
        "metric_reconstruction_max_pcm16_error": maximum_error,
        "metric_reconstruction_differing_samples": differing_samples,
        "pulse_positions_samples": pulse_positions,
        "pulse_count": len(pulse_positions),
        "highpass_pcm16_sha256": pcm16_sha256(highpassed),
        "metric_pcm16_sha256": pcm16_sha256(metric),
        "source_ranges_sha256": ranges_sha256(ranges),
        "pulse_positions_sha256": pulses_sha256(pulse_positions),
        "timing_ms": {
            "input_read": input_read_ms,
            "highpass": highpass_ms,
            "highpass_filter_compute": highpass_filter_compute_ms,
            "highpass_quantize": highpass_quantize_ms,
            "textgrid": textgrid_ms,
            "source_selection": source_selection_ms,
            "textgrid_range": textgrid_ms + source_selection_ms,
            "metric_gather": metric_gather_ms,
            "pointprocess_construct": pointprocess_construct_ms,
            "pulse_enumeration": pulse_enumeration_ms,
            "total_refresh": total_refresh_ms,
        },
    }

    if verify_authority:
        authority_started = time.perf_counter()
        authoritative_highpassed = highpass_filter(
            "praat",
            waveform,
            sampling_rate=SAMPLE_RATE,
        )
        authoritative_metric = get_voiced_segments(
            "praat",
            authoritative_highpassed,
            sampling_rate=SAMPLE_RATE,
        )
        authoritative_prefix, authoritative_ranges = monotonic_ranges(
            authoritative_highpassed,
            authoritative_metric,
        )
        authoritative_sound = parselmouth.Sound(
            authoritative_metric,
            SAMPLE_RATE,
        )
        authoritative_point_process = call(
            authoritative_sound,
            "To PointProcess (periodic, cc)",
            50,
            400,
        )
        authoritative_pulses = enumerate_point_process(
            authoritative_point_process,
            authoritative_sound,
        )
        highpass_difference = np.abs(
            pcm16(highpassed) - pcm16(authoritative_highpassed)
        )
        metric_difference = (
            np.abs(pcm16(metric) - pcm16(authoritative_metric))
            if metric.size == authoritative_metric.size
            else None
        )
        authority = {
            "highpass_max_pcm16_error": int(
                highpass_difference.max(initial=0)
            ),
            "highpass_differing_samples": int(
                np.count_nonzero(highpass_difference)
            ),
            "metric_length_difference": int(
                metric.size - authoritative_metric.size
            ),
            "metric_max_pcm16_error": (
                int(metric_difference.max(initial=0))
                if metric_difference is not None
                else -1
            ),
            "metric_differing_samples": (
                int(np.count_nonzero(metric_difference))
                if metric_difference is not None
                else -1
            ),
            "constant_prefix_equal": constant_prefix == authoritative_prefix,
            "source_ranges_equal": ranges == authoritative_ranges,
            "authoritative_source_ranges_sha256": ranges_sha256(
                authoritative_ranges
            ),
            "authoritative_pulse_positions_sha256": pulses_sha256(
                authoritative_pulses
            ),
            "pulse_positions_equal": np.array_equal(
                np.asarray(pulse_positions, dtype=np.float64),
                np.asarray(authoritative_pulses, dtype=np.float64),
            ),
            "authority_check_ms": 1000.0 * (
                time.perf_counter() - authority_started
            ),
        }
        authority["pass"] = (
            authority["highpass_differing_samples"] == 0
            and authority["metric_length_difference"] == 0
            and authority["metric_differing_samples"] == 0
            and authority["constant_prefix_equal"]
            and authority["source_ranges_equal"]
            and authority["pulse_positions_equal"]
        )
        result["authority_parity"] = authority
    return result


def score_components(path, view, step_versions):
    metrics = run_avqi(
        path,
        path,
        target_sr=SAMPLE_RATE,
        speaking_type=view,
        step_versions=step_versions,
        remove_sv_silence_with_sox=False,
    )
    names = ("cpps", "hnr", "shimmer_percent", "shimmer_db", "slope", "tilt")
    return {name: float(metrics[name]) for name in names}


print(
    "AVQI_SHIMMER_RUNTIME_READY="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
        },
        sort_keys=True,
    ),
    flush=True,
)
for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    try:
        if request["op"] == "refresh":
            response = direct_refresh(
                request["path"],
                bool(request.get("verify_authority", False)),
                request.get(
                    "input_loader",
                    "authoritative_read_and_resample",
                ),
                request.get("frame_scan_mode", "praat_per_frame"),
                request.get(
                    "pulse_enumeration_mode",
                    "praat_per_point",
                ),
                request.get("wav_roundtrip_mode", "praat_temp_wav"),
                request.get(
                    "sounding_assembly_mode",
                    "praat_extract_and_concatenate",
                ),
            )
        elif request["op"] == "score":
            response = {
                "status": "ok",
                "components": score_components(
                    request["path"],
                    request["view"],
                    request["step_versions"],
                ),
            }
        elif request["op"] == "quit":
            response = {"status": "ok", "quitting": True}
            print(
                RESULT_MARKER + json.dumps(response, sort_keys=True),
                flush=True,
            )
            break
        else:
            raise ValueError("unknown exact worker operation")
    except Exception as error:
        response = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error)[:1000],
        }
    print(RESULT_MARKER + json.dumps(response, sort_keys=True), flush=True)
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
        raise ValueError(f"AVQI code tree contains no Python/Praat files: {root}")
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def ranges_sha256(ranges: list[list[int]]) -> str:
    encoded = json.dumps(ranges, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pulses_sha256(positions: list[float]) -> str:
    values = np.asarray(positions, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def metric_source_indices(
    topology: dict[str, Any],
    source_sample_count: int,
) -> np.ndarray:
    if int(topology["source_sample_count"]) != source_sample_count:
        raise ValueError("topology source length drift")
    pieces: list[np.ndarray] = []
    previous_end = 0
    for start_value, length_value in topology["metric_source_ranges"]:
        start = int(start_value)
        length = int(length_value)
        end = start + length
        if length <= 0 or start < previous_end or end > source_sample_count:
            raise ValueError("invalid detached source range")
        pieces.append(np.arange(start, end, dtype=np.int64))
        previous_end = end
    if not pieces:
        raise ValueError("empty detached source ranges")
    indices = np.concatenate(pieces)
    if indices.size != int(topology["metric_mapped_sample_count"]):
        raise ValueError("detached source range count drift")
    if int(topology["metric_reconstruction_differing_samples"]) != 0:
        raise ValueError("topology lacks exact metric waveform parity")
    return indices


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


class ExactWorker:
    def __init__(self, exact_python: Path, avqi_code_root: Path) -> None:
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [
                str(exact_python),
                "-u",
                "-c",
                EXACT_WORKER,
                str(avqi_code_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.startup = self._read_marker(EXACT_READY_MARKER)
        self.startup_ms = 1000.0 * (time.perf_counter() - started)

    def _read_marker(self, marker: str) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("exact worker stdout is unavailable")
        transcript: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "exact worker exited before marker: " + "".join(transcript)[-2000:]
                )
            transcript.append(line)
            if line.startswith(marker):
                return json.loads(line.split("=", 1)[1])

    def request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self.process.stdin is None:
            raise RuntimeError("exact worker stdin is unavailable")
        started = time.perf_counter()
        self.process.stdin.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()
        response = self._read_marker(EXACT_RESULT_MARKER)
        wall_ms = 1000.0 * (time.perf_counter() - started)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"exact worker failed: {response.get('error_type')} "
                f"{response.get('error_message')}"
            )
        return response, wall_ms

    def close(self) -> None:
        if self.process.poll() is None:
            self.request({"op": "quit"})
        self.process.wait(timeout=30)

    def __enter__(self) -> "ExactWorker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_panel_rows(path: Path) -> list[dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in contract["rows"]]
    selected = [row for row in rows if row["view"] == "cs"]
    if tuple(row["case_id"] for row in selected) != EXPECTED_CS_CASE_IDS:
        raise ValueError("six-CS profiling panel identity/order drift")
    for row in selected:
        base_path = Path(row["base_path"])
        validate_hash(base_path, row["base_sha256"], "CS base waveform")
        info = sf.info(base_path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1:
            raise ValueError(f"invalid CS base waveform: {base_path}")
        row["base_frame_count"] = info.frames
    return selected


def load_exact_references(
    calibration_path: Path,
    final_path: Path,
) -> dict[str, dict[str, str]]:
    references = {
        row["case_id"]: row
        for row in read_csv(calibration_path)
        if float(row["alpha"]) == 0.0
    }
    references.update({row["case_id"]: row for row in read_csv(final_path)})
    return references


def topology_identity(topology: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_ranges_sha256": topology["source_ranges_sha256"],
        "pulse_positions_sha256": topology["pulse_positions_sha256"],
        "highpass_pcm16_sha256": topology["highpass_pcm16_sha256"],
        "metric_pcm16_sha256": topology["metric_pcm16_sha256"],
        "metric_sample_count": topology["metric_sample_count"],
        "metric_constant_prefix_samples": topology[
            "metric_constant_prefix_samples"
        ],
        "metric_mapped_sample_count": topology["metric_mapped_sample_count"],
        "pulse_count": topology["pulse_count"],
    }


def require_same_topology(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
) -> None:
    if topology_identity(reference) != topology_identity(candidate):
        raise ValueError(f"detached topology changed across reuse audit: {label}")
    if ranges_sha256(candidate["metric_source_ranges"]) != candidate[
        "source_ranges_sha256"
    ]:
        raise ValueError(f"source range hash self-check failed: {label}")
    if pulses_sha256(candidate["pulse_positions_samples"]) != candidate[
        "pulse_positions_sha256"
    ]:
        raise ValueError(f"pulse hash self-check failed: {label}")


def read_waveform(path: Path) -> torch.Tensor:
    values, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or values.ndim != 1 or values.size == 0:
        raise ValueError(f"invalid waveform: {path}")
    waveform = torch.from_numpy(values.copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {path}")
    return waveform


def load_predictor(
    path: Path,
    device: torch.device,
) -> tuple[PraatDifferentiableAVQIComponentEstimator, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "direct_praat_hard_shimmer_pulse_path_v6":
        raise ValueError("unexpected frozen Route C checkpoint architecture")
    predictor = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
    ).to(device)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.eval()
    return predictor, checkpoint["target_scale"].to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_gradient(
    predictor: PraatDifferentiableAVQIComponentEstimator,
    target_scale: torch.Tensor,
    base_cpu: torch.Tensor,
    topology: dict[str, Any],
    target_value: float,
    device: torch.device,
) -> dict[str, Any]:
    waveform = base_cpu.to(device).requires_grad_(True)
    source_indices_np = metric_source_indices(topology, waveform.numel())
    source_indices = torch.as_tensor(
        source_indices_np,
        dtype=torch.long,
        device=device,
    )
    pulses = waveform.new_tensor(topology["pulse_positions_samples"])
    synchronize(device)
    started = time.perf_counter()
    proxy = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            topology["metric_constant_prefix_samples"]
        ),
    )[1]
    loss = (
        (proxy - target_value)
        / target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    ).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    synchronize(device)
    runtime_ms = 1000.0 * (time.perf_counter() - started)
    if not bool(torch.isfinite(gradient).all()) or float(gradient.norm()) <= 0.0:
        raise ValueError("reuse audit produced an invalid Shimmer-dB gradient")
    return {
        "waveform": waveform,
        "proxy": float(proxy.detach()),
        "proxy_hex": float(proxy.detach()).hex(),
        "loss": float(loss.detach()),
        "gradient": gradient,
        "gradient_sha256": tensor_sha256(gradient),
        "gradient_l2_norm": float(gradient.norm()),
        "gradient_l2_norm_hex": float(gradient.norm()).hex(),
        "runtime_ms": runtime_ms,
    }


def write_candidate(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    alpha: float,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    synchronize(device)
    started = time.perf_counter()
    candidate = normalized_gradient_step(waveform, gradient, alpha)
    synchronize(device)
    step_ms = 1000.0 * (time.perf_counter() - started)
    candidate_cpu = candidate.detach().cpu()
    write_started = time.perf_counter()
    sf.write(path, candidate_cpu.numpy(), SAMPLE_RATE, subtype="PCM_24")
    write_ms = 1000.0 * (time.perf_counter() - write_started)
    return {
        "candidate_tensor_sha256": tensor_sha256(candidate_cpu),
        "candidate_file_sha256": sha256_file(path),
        "candidate_path": str(path.resolve()),
        "candidate_step_ms": step_ms,
        "candidate_write_ms": write_ms,
    }


def flatten_runtime_row(
    case: dict[str, Any],
    phase: str,
    worker_startup_ms: float,
    response: dict[str, Any],
    request_wall_ms: float,
) -> dict[str, Any]:
    timing = response["timing_ms"]
    authority_check_ms = float(
        response.get("authority_parity", {}).get("authority_check_ms", 0.0)
    )
    row = {
        "case_id": case["case_id"],
        "speaker_id": case["speaker_id"],
        "condition": case["condition"],
        "phase": phase,
        "worker_startup_ms": worker_startup_ms,
        "request_wall_ms": request_wall_ms - authority_check_ms,
        "request_wall_including_authority_ms": request_wall_ms,
        "authority_check_ms": authority_check_ms,
        "source_samples": response["source_sample_count"],
        "metric_samples": response["metric_sample_count"],
        "source_range_count": response["metric_source_range_count"],
        "pulse_count": response["pulse_count"],
        "source_ranges_sha256": response["source_ranges_sha256"],
        "pulse_positions_sha256": response["pulse_positions_sha256"],
    }
    row.update({f"{name}_ms": value for name, value in timing.items()})
    return row


def stage_summary(
    runtime_rows: list[dict[str, Any]],
    phase: str,
) -> dict[str, dict[str, float]]:
    selected = [row for row in runtime_rows if row["phase"] == phase]
    fields = (
        "input_read_ms",
        "highpass_ms",
        "textgrid_range_ms",
        "metric_gather_ms",
        "pointprocess_construct_ms",
        "pulse_enumeration_ms",
        "total_refresh_ms",
        "request_wall_ms",
    )
    return {
        field.removesuffix("_ms"): {
            "median_ms": median(float(row[field]) for row in selected),
            "maximum_ms": max(float(row[field]) for row in selected),
        }
        for field in fields
    }


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
        "avqi_code_tree": args.avqi_code_tree_sha256,
    }
    panel_rows = load_panel_rows(args.panel_contract)
    references = load_exact_references(
        args.calibration_results,
        args.final_results,
    )
    if not set(EXPECTED_CS_CASE_IDS).issubset(references):
        raise ValueError("exact CS target reference coverage drift")

    output_root = args.output_dir
    waveform_root = output_root / "waveforms"
    waveform_root.mkdir(parents=True)

    runtime_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    refresh_cost_rows: list[dict[str, Any]] = []
    shared_topologies: dict[str, dict[str, Any]] = {}
    separate_topologies: dict[tuple[str, float], dict[str, Any]] = {}
    exact_versions: dict[str, str] | None = None

    for case in panel_rows:
        with ExactWorker(args.exact_python, args.avqi_code_root) as worker:
            if exact_versions is None:
                exact_versions = dict(worker.startup)
            elif exact_versions != worker.startup:
                raise ValueError("exact worker version drift")
            cold, cold_wall_ms = worker.request(
                {
                    "op": "refresh",
                    "path": case["base_path"],
                    "verify_authority": False,
                }
            )
            warm, warm_wall_ms = worker.request(
                {
                    "op": "refresh",
                    "path": case["base_path"],
                    "verify_authority": True,
                }
            )
            require_same_topology(cold, warm, f"{case['case_id']}:cold-warm")
            authority = warm["authority_parity"]
            if not authority["pass"]:
                raise ValueError(f"authority parity failed: {case['case_id']}")
            runtime_rows.extend(
                (
                    flatten_runtime_row(
                        case,
                        "cold",
                        worker.startup_ms,
                        cold,
                        cold_wall_ms,
                    ),
                    flatten_runtime_row(
                        case,
                        "warm",
                        worker.startup_ms,
                        warm,
                        warm_wall_ms,
                    ),
                )
            )
            authority_rows.append(
                {
                    "case_id": case["case_id"],
                    "speaker_id": case["speaker_id"],
                    **authority,
                }
            )

            shared, shared_wall_ms = worker.request(
                {
                    "op": "refresh",
                    "path": case["base_path"],
                    "verify_authority": False,
                }
            )
            require_same_topology(warm, shared, f"{case['case_id']}:shared")
            shared_topologies[case["case_id"]] = shared
            separate_wall_total_ms = 0.0
            separate_internal_total_ms = 0.0
            for alpha in FROZEN_ALPHA_GRID:
                separate, separate_wall_ms = worker.request(
                    {
                        "op": "refresh",
                        "path": case["base_path"],
                        "verify_authority": False,
                    }
                )
                require_same_topology(
                    shared,
                    separate,
                    f"{case['case_id']}:alpha={alpha}",
                )
                separate_topologies[(case["case_id"], alpha)] = separate
                separate_wall_total_ms += separate_wall_ms
                separate_internal_total_ms += float(
                    separate["timing_ms"]["total_refresh"]
                )
            refresh_cost_rows.append(
                {
                    "case_id": case["case_id"],
                    "speaker_id": case["speaker_id"],
                    "shared_unique_refresh_calls": 1,
                    "diagnostic_separate_refresh_calls": len(FROZEN_ALPHA_GRID),
                    "shared_refresh_internal_ms": shared["timing_ms"][
                        "total_refresh"
                    ],
                    "shared_refresh_request_wall_ms": shared_wall_ms,
                    "three_separate_refresh_internal_ms": (
                        separate_internal_total_ms
                    ),
                    "three_separate_refresh_request_wall_ms": (
                        separate_wall_total_ms
                    ),
                    "topology_call_reduction_factor": len(FROZEN_ALPHA_GRID),
                }
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime profile requested but unavailable")
    predictor, target_scale = load_predictor(args.predictor_checkpoint, device)

    candidate_records: dict[tuple[str, float, str], dict[str, Any]] = {}
    torch_cost_rows: list[dict[str, Any]] = []
    for case in panel_rows:
        case_id = case["case_id"]
        base_cpu = read_waveform(Path(case["base_path"]))
        target_value = float(references[case_id]["exact_target_shimmer_db"])

        shared_topology = shared_topologies[case_id]
        shared_gradient = compute_gradient(
            predictor,
            target_scale,
            base_cpu,
            shared_topology,
            target_value,
            device,
        )
        shared_step_total_ms = 0.0
        for alpha in FROZEN_ALPHA_GRID:
            output_path = waveform_root / (
                f"{safe_name(case_id)}__shared_alpha_{alpha:.4g}.wav"
            )
            candidate = write_candidate(
                shared_gradient["waveform"],
                shared_gradient["gradient"],
                alpha,
                output_path,
                device,
            )
            shared_step_total_ms += float(candidate["candidate_step_ms"])
            candidate_records[(case_id, alpha, "shared")] = {
                "case_id": case_id,
                "speaker_id": case["speaker_id"],
                "alpha": alpha,
                "route": "shared",
                "source_ranges_sha256": shared_topology[
                    "source_ranges_sha256"
                ],
                "pulse_positions_sha256": shared_topology[
                    "pulse_positions_sha256"
                ],
                "proxy": shared_gradient["proxy"],
                "proxy_hex": shared_gradient["proxy_hex"],
                "gradient_sha256": shared_gradient["gradient_sha256"],
                "gradient_l2_norm": shared_gradient["gradient_l2_norm"],
                "gradient_l2_norm_hex": shared_gradient[
                    "gradient_l2_norm_hex"
                ],
                "gradient_runtime_ms": shared_gradient["runtime_ms"],
                **candidate,
            }

        separate_gradient_total_ms = 0.0
        separate_step_total_ms = 0.0
        for alpha in FROZEN_ALPHA_GRID:
            topology = separate_topologies[(case_id, alpha)]
            gradient = compute_gradient(
                predictor,
                target_scale,
                base_cpu,
                topology,
                target_value,
                device,
            )
            output_path = waveform_root / (
                f"{safe_name(case_id)}__separate_alpha_{alpha:.4g}.wav"
            )
            candidate = write_candidate(
                gradient["waveform"],
                gradient["gradient"],
                alpha,
                output_path,
                device,
            )
            separate_gradient_total_ms += float(gradient["runtime_ms"])
            separate_step_total_ms += float(candidate["candidate_step_ms"])
            candidate_records[(case_id, alpha, "separate")] = {
                "case_id": case_id,
                "speaker_id": case["speaker_id"],
                "alpha": alpha,
                "route": "separate",
                "source_ranges_sha256": topology["source_ranges_sha256"],
                "pulse_positions_sha256": topology["pulse_positions_sha256"],
                "proxy": gradient["proxy"],
                "proxy_hex": gradient["proxy_hex"],
                "gradient_sha256": gradient["gradient_sha256"],
                "gradient_l2_norm": gradient["gradient_l2_norm"],
                "gradient_l2_norm_hex": gradient["gradient_l2_norm_hex"],
                "gradient_runtime_ms": gradient["runtime_ms"],
                **candidate,
            }
        torch_cost_rows.append(
            {
                "case_id": case_id,
                "speaker_id": case["speaker_id"],
                "shared_proxy_gradient_calls": 1,
                "separate_proxy_gradient_calls": len(FROZEN_ALPHA_GRID),
                "shared_proxy_gradient_ms": shared_gradient["runtime_ms"],
                "shared_three_alpha_candidate_step_ms": shared_step_total_ms,
                "shared_three_alpha_torch_ms": (
                    float(shared_gradient["runtime_ms"])
                    + shared_step_total_ms
                ),
                "separate_three_gradient_ms": separate_gradient_total_ms,
                "separate_three_alpha_candidate_step_ms": separate_step_total_ms,
                "separate_three_alpha_torch_ms": (
                    separate_gradient_total_ms + separate_step_total_ms
                ),
            }
        )

    with ExactWorker(args.exact_python, args.avqi_code_root) as score_worker:
        if exact_versions != score_worker.startup:
            raise ValueError("exact scorer version drift after runtime profiling")
        for record in candidate_records.values():
            scored, score_wall_ms = score_worker.request(
                {
                    "op": "score",
                    "path": record["candidate_path"],
                    "view": "cs",
                    "step_versions": COMPONENT_STEP_VERSIONS,
                }
            )
            record["exact_components"] = scored["components"]
            record["exact_score_wall_ms"] = score_wall_ms

    equivalence_rows: list[dict[str, Any]] = []
    all_equivalent = True
    for case in panel_rows:
        case_id = case["case_id"]
        for alpha in FROZEN_ALPHA_GRID:
            shared = candidate_records[(case_id, alpha, "shared")]
            separate = candidate_records[(case_id, alpha, "separate")]
            comparisons = {
                "source_ranges_hash_equal": shared["source_ranges_sha256"]
                == separate["source_ranges_sha256"],
                "pulse_hash_equal": shared["pulse_positions_sha256"]
                == separate["pulse_positions_sha256"],
                "proxy_equal": shared["proxy_hex"] == separate["proxy_hex"],
                "gradient_hash_equal": shared["gradient_sha256"]
                == separate["gradient_sha256"],
                "gradient_norm_equal": shared["gradient_l2_norm_hex"]
                == separate["gradient_l2_norm_hex"],
                "candidate_tensor_hash_equal": shared[
                    "candidate_tensor_sha256"
                ]
                == separate["candidate_tensor_sha256"],
                "candidate_file_hash_equal": shared["candidate_file_sha256"]
                == separate["candidate_file_sha256"],
                "exact_components_equal": shared["exact_components"]
                == separate["exact_components"],
            }
            proved_equal = all(comparisons.values())
            all_equivalent = all_equivalent and proved_equal
            equivalence_rows.append(
                {
                    "case_id": case_id,
                    "speaker_id": case["speaker_id"],
                    "alpha": alpha,
                    **comparisons,
                    "proved_equal": proved_equal,
                    "source_ranges_sha256": shared["source_ranges_sha256"],
                    "pulse_positions_sha256": shared[
                        "pulse_positions_sha256"
                    ],
                    "proxy": shared["proxy"],
                    "gradient_sha256": shared["gradient_sha256"],
                    "gradient_l2_norm": shared["gradient_l2_norm"],
                    "candidate_file_sha256": shared[
                        "candidate_file_sha256"
                    ],
                    **{
                        f"exact_{name}": shared["exact_components"][name]
                        for name in AVQI_COMPONENT_NAMES
                    },
                }
            )

    authority_pass = all(row["pass"] for row in authority_rows)
    warm_maximum_ms = max(
        float(row["total_refresh_ms"])
        for row in runtime_rows
        if row["phase"] == "warm"
    )
    shared_maximum_ms = max(
        float(row["shared_refresh_internal_ms"])
        for row in refresh_cost_rows
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-runtime-profile-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_hashes": source_hashes,
        "exact_versions": exact_versions,
        "scope": {
            "component": "shimmer_db",
            "view": "cs",
            "case_ids": list(EXPECTED_CS_CASE_IDS),
            "case_count": len(panel_rows),
            "opened_panel_role": "runtime_and_reuse_mechanism_only",
            "production_code_modified_by_profile": False,
            "formal_pulse_refresh_gate_ms": PULSE_REFRESH_GATE_MS,
            "formal_gate_changed": False,
            "frozen_alpha_grid": list(FROZEN_ALPHA_GRID),
            "scientifically_valid_candidate_alpha": 1e-3,
        },
        "runtime_profile": {
            "rows": runtime_rows,
            "cold_summary": stage_summary(runtime_rows, "cold"),
            "warm_summary": stage_summary(runtime_rows, "warm"),
            "authority_rows": authority_rows,
            "authority_parity_pass": authority_pass,
            "warm_maximum_total_refresh_ms": warm_maximum_ms,
            "warm_profile_within_500ms": warm_maximum_ms
            <= PULSE_REFRESH_GATE_MS,
        },
        "topology_reuse": {
            "contract": "one_current_output_refresh_per_waveform_per_optimizer_step",
            "refresh_cost_rows": refresh_cost_rows,
            "torch_cost_rows": torch_cost_rows,
            "equivalence_rows": equivalence_rows,
            "equivalence_case_alpha_count": len(equivalence_rows),
            "all_hash_value_and_exact_component_assertions_pass": all_equivalent,
            "shared_refresh_maximum_internal_ms": shared_maximum_ms,
            "shared_refresh_profile_within_500ms": shared_maximum_ms
            <= PULSE_REFRESH_GATE_MS,
            "three_alpha_candidates_count_as_training_refreshes": False,
        },
        "decision": (
            "PASS_PROFILE_AUTHORITY_AND_REUSE_EQUIVALENCE"
            if authority_pass and all_equivalent
            else "FAIL_PROFILE_AUTHORITY_OR_REUSE_EQUIVALENCE"
        ),
        "candidate_c_formal_runtime_gate_evaluated": False,
        "fresh_speaker_panel_authorized": False,
        "formal_generator_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_optimizer_steps": 0,
    }

    runtime_csv = output_root / "runtime_profile.csv"
    equivalence_csv = output_root / "reuse_equivalence.csv"
    report_path = output_root / "diagnostic_report.json"
    write_csv(runtime_csv, runtime_rows)
    write_csv(equivalence_csv, equivalence_rows)
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-runtime-profile-receipt-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "decision": report["decision"],
        "authority_parity_pass": authority_pass,
        "reuse_equivalence_pass": all_equivalent,
        "warm_maximum_total_refresh_ms": warm_maximum_ms,
        "shared_refresh_maximum_internal_ms": shared_maximum_ms,
        "formal_pulse_refresh_gate_ms": PULSE_REFRESH_GATE_MS,
        "formal_gate_changed": False,
        "fresh_speaker_panel_authorized": False,
        "formal_generator_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "generator_optimizer_steps": 0,
        "artifact_sha256": {
            "diagnostic_report": sha256_file(report_path),
            "runtime_profile": sha256_file(runtime_csv),
            "reuse_equivalence": sha256_file(equivalence_csv),
        },
    }
    write_json(output_root / "completion_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not authority_pass or not all_equivalent:
        raise RuntimeError("exact authority or topology reuse equivalence failed")


if __name__ == "__main__":
    main()
