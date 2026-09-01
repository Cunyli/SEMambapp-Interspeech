#!/usr/bin/env python3
"""Score fixed-topology Shimmer dB and per-pulse evidence with exact Praat.

This worker is a development-only adjudicator.  It consumes already frozen
pulse topologies and waveform paths, computes the exact AVQI metric high-pass
branch, and returns both Praat's local-dB scalar and AmplitudeTier evidence.
Its output is prohibited from runtime selector inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

from scripts.avqi_shimmer_exact_topology_worker import (
    ExactTopologyEngine,
    NUMPY_HIGHPASS_MODE,
    SAMPLE_RATE,
    pcm16_sha256,
    sha256_tree,
)


MINIMUM_PERIOD_SECONDS = 0.0001
MAXIMUM_PERIOD_SECONDS = 0.02
MAXIMUM_PERIOD_FACTOR = 1.3
MAXIMUM_AMPLITUDE_FACTOR = 1.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def metric_source_indices(
    topology: dict[str, Any],
    source_sample_count: int,
) -> np.ndarray:
    if int(topology["source_sample_count"]) != source_sample_count:
        raise ValueError("fixed-topology source sample count drift")
    pieces: list[np.ndarray] = []
    previous_end = 0
    for start_value, length_value in topology["metric_source_ranges"]:
        start = int(start_value)
        length = int(length_value)
        end = start + length
        if length <= 0 or start < previous_end or end > source_sample_count:
            raise ValueError("invalid fixed-topology metric source range")
        pieces.append(np.arange(start, end, dtype=np.int64))
        previous_end = end
    if not pieces:
        raise ValueError("fixed topology contains no source ranges")
    indices = np.concatenate(pieces)
    if indices.size != int(topology["metric_mapped_sample_count"]):
        raise ValueError("fixed-topology mapped sample count drift")
    return indices


def point_process_from_positions(
    sound: parselmouth.Sound,
    positions_samples: np.ndarray,
) -> Any:
    point_process = call(
        "Create empty PointProcess",
        "candidate-e-fixed-topology",
        sound.xmin,
        sound.xmax,
    )
    for position in positions_samples:
        time_seconds = sound.x1 + float(position) / SAMPLE_RATE
        if not sound.xmin <= time_seconds <= sound.xmax:
            raise ValueError("fixed pulse lies outside exact metric sound")
        call(point_process, "Add point", time_seconds)
    return point_process


def amplitude_tier_evidence(
    sound: parselmouth.Sound,
    point_process: Any,
    include_arrays: bool,
) -> dict[str, Any]:
    tier = call(
        [point_process, sound],
        "To AmplitudeTier (period)",
        0.0,
        0.0,
        MINIMUM_PERIOD_SECONDS,
        MAXIMUM_PERIOD_SECONDS,
        MAXIMUM_PERIOD_FACTOR,
    )
    point_count = int(call(tier, "Get number of points"))
    times = np.asarray(
        [
            float(call(tier, "Get time from index", index))
            for index in range(1, point_count + 1)
        ],
        dtype=np.float64,
    )
    amplitudes = np.asarray(
        [
            float(call(tier, "Get value at index", index))
            for index in range(1, point_count + 1)
        ],
        dtype=np.float64,
    )
    positions = (times - sound.x1) * SAMPLE_RATE
    if amplitudes.size < 2:
        raise ValueError("exact fixed topology has fewer than two amplitudes")
    pair_period = np.diff(times)
    amplitude_factor = np.maximum(amplitudes[:-1], amplitudes[1:]) / np.maximum(
        np.minimum(amplitudes[:-1], amplitudes[1:]),
        np.finfo(np.float64).tiny,
    )
    valid_pair = (
        (pair_period >= MINIMUM_PERIOD_SECONDS)
        & (pair_period <= MAXIMUM_PERIOD_SECONDS)
        & (amplitude_factor <= MAXIMUM_AMPLITUDE_FACTOR)
    )
    contributions = 20.0 * np.abs(
        np.log10(
            np.maximum(amplitudes[:-1], np.finfo(np.float64).tiny)
            / np.maximum(amplitudes[1:], np.finfo(np.float64).tiny)
        )
    )
    if not np.any(valid_pair):
        raise ValueError("exact fixed topology has no valid amplitude pairs")
    evidence = {
        "amplitude_count": int(amplitudes.size),
        "reconstructed_shimmer_db": float(np.mean(contributions[valid_pair])),
    }
    if include_arrays:
        evidence.update(
            {
                "amplitude_positions_samples": positions.tolist(),
                "amplitudes": amplitudes.tolist(),
                "valid_pair_mask": valid_pair.tolist(),
                "pair_contributions_db": contributions.tolist(),
            }
        )
    return evidence


def score_item(
    engine: ExactTopologyEngine,
    item: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(item["waveform_path"])
    waveform, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
        raise ValueError(f"invalid exact candidate waveform: {path}")
    topology = item["topology"]
    indices = metric_source_indices(topology, waveform.size)
    highpassed, timing = engine.metric_highpass(
        waveform,
        highpass_mode=NUMPY_HIGHPASS_MODE,
    )
    prefix = int(topology["metric_constant_prefix_samples"])
    mapped = highpassed[indices]
    if prefix:
        mapped = np.concatenate((np.zeros(prefix, dtype=np.float64), mapped))
    if mapped.size != int(topology["metric_sample_count"]):
        raise ValueError("exact fixed-topology metric sample count drift")
    sound = parselmouth.Sound(mapped, SAMPLE_RATE)
    positions = np.asarray(
        topology["pulse_positions_samples"],
        dtype=np.float64,
    )
    point_process = point_process_from_positions(sound, positions)
    exact_scalar = float(
        call(
            [sound, point_process],
            "Get shimmer (local_dB)",
            0.0,
            0.0,
            MINIMUM_PERIOD_SECONDS,
            MAXIMUM_PERIOD_SECONDS,
            MAXIMUM_PERIOD_FACTOR,
            MAXIMUM_AMPLITUDE_FACTOR,
        )
    )
    tier = amplitude_tier_evidence(
        sound,
        point_process,
        include_arrays=bool(item.get("include_pulse_evidence", False)),
    )
    if abs(exact_scalar - tier["reconstructed_shimmer_db"]) > 1e-12:
        raise ValueError("exact Shimmer dB and AmplitudeTier reconstruction drift")
    return {
        "item_id": item["item_id"],
        "case_id": item["case_id"],
        "variant": item["variant"],
        "alpha": float(item["alpha"]),
        "waveform_path": str(path.resolve()),
        "waveform_sha256": sha256_file(path),
        "source_pcm16_sha256": pcm16_sha256(waveform),
        "metric_pcm16_sha256": pcm16_sha256(mapped),
        "pulse_count": int(positions.size),
        "exact_shimmer_db": exact_scalar,
        **tier,
        "exact_highpass": timing,
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }


def main() -> None:
    args = parse_args()
    request = read_object(args.request)
    if request.get("scientific_role") != "development_calibration_only":
        raise ValueError("Candidate-E exact worker is development-only")
    avqi_code_root = Path(request["avqi_code_root"])
    observed_tree_sha256 = sha256_tree(avqi_code_root)
    if observed_tree_sha256 != request["avqi_code_tree_sha256"]:
        raise ValueError("exact AVQI code-tree hash drift")
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Candidate-E exact request contains no items")
    engine = ExactTopologyEngine()
    try:
        rows = [score_item(engine, dict(item)) for item in items]
    finally:
        engine.close()
    output = {
        "schema_version": "avqi-route-c-candidate-e-fixed-topology-exact-v27",
        "scientific_role": "development_calibration_only",
        "exact_candidate_outcomes_allowed_as_runtime_selector_inputs": False,
        "avqi_code_root": str(avqi_code_root.resolve()),
        "avqi_code_tree_sha256": observed_tree_sha256,
        "parselmouth_version": parselmouth.__version__,
        "praat_version": parselmouth.PRAAT_VERSION,
        "row_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
