#!/usr/bin/env python3
"""Persistent exact-Praat topology worker for Route-C Shimmer dB.

The worker refreshes each current waveform's own exact AVQI metric topology.
It caches no waveform-dependent ranges or pulses.  Persistent state is limited
to the process, Praat command initialization, and two reusable Praat-generated
WAV slots.  A hash-bound client tmpfs slot may hand off the current output's
float32 samples, but it is reread and revalidated for every refresh.  The
emitted waveform is never high-passed by this worker.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call


SAMPLE_RATE = 16_000
READY_MARKER = "AVQI_SHIMMER_TOPOLOGY_READY="
RESULT_MARKER = "AVQI_SHIMMER_TOPOLOGY_RESULT="
IMPLEMENTATION = "exact_vectorized_frames_reused_tmpfs_numpy_sounding_v15"
PRAAT_HIGHPASS_MODE = "praat_6_1_38_stop_hann_0_34_0p1"
NUMPY_HIGHPASS_MODE = "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
NUMPY_FFT_WARMUP_LENGTHS = tuple(1 << exponent for exponent in range(14, 20))
SINC70_INTERPOLATION_DEPTH = 70
SINC70_ABSOLUTE_WEIGHT_BOUND = 5.2
PEAK_SCALE_TRIGGER = 0.999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    return parser.parse_args()


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


def pcm16(values: np.ndarray) -> np.ndarray:
    bounded = np.clip(
        np.asarray(values, dtype=np.float64),
        -1.0,
        1.0 - 1.0 / 32768.0,
    )
    return np.rint(bounded * 32768.0).astype(np.int32)


def pcm16_roundtrip(values: np.ndarray) -> np.ndarray:
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


def bytes_sha256(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def pcm16_sha256(values: np.ndarray) -> str:
    encoded = pcm16(values).astype("<i4", copy=False).tobytes()
    return bytes_sha256(encoded)


def ranges_sha256(ranges: list[list[int]]) -> str:
    encoded = json.dumps(ranges, separators=(",", ":")).encode("utf-8")
    return bytes_sha256(encoded)


def pulses_sha256(positions: list[float]) -> str:
    encoded = np.asarray(positions, dtype="<f8").tobytes()
    return bytes_sha256(encoded)


def compress_source_indices(indices: np.ndarray) -> list[list[int]]:
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, split_points)
    return [[int(group[0]), int(group.size)] for group in groups]


def exact_zero_crossing_rates_aligned_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    left = values[:, :-1]
    right = values[:, 1:]
    crossing_mask = (
        ((left <= 0.0) & (right > 0.0))
        | ((left >= 0.0) & (right < 0.0))
    )
    denominator = left - right
    fraction = np.divide(
        left,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    sample_indices = np.arange(left.shape[1], dtype=np.float64)[None, :]
    crossings = 0.5 / SAMPLE_RATE + (sample_indices + fraction) / SAMPLE_RATE
    distance_to_first = np.where(
        crossing_mask,
        np.abs(crossings - 0.0025),
        np.inf,
    )
    first_indices = np.argmin(distance_to_first, axis=1)
    columns = np.arange(left.shape[1], dtype=np.int64)[None, :]
    last_mask = (
        crossing_mask
        & (columns >= first_indices[:, None])
        & (crossings >= 0.0275)
    )
    has_enough_crossings = np.count_nonzero(crossing_mask, axis=1) >= 2
    has_last = np.any(last_mask, axis=1)
    last_indices = np.argmax(last_mask, axis=1)
    rows = np.arange(values.shape[0], dtype=np.int64)
    first_crossings = crossings[rows, first_indices]
    last_crossings = crossings[rows, last_indices]
    crossing_ordinals = np.cumsum(crossing_mask, axis=1) - 1
    first_ordinals = crossing_ordinals[rows, first_indices]
    last_ordinals = crossing_ordinals[rows, last_indices]
    crossing_distance = last_crossings - first_crossings
    return np.divide(
        last_ordinals - first_ordinals,
        crossing_distance,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=(has_enough_crossings & has_last & (crossing_distance != 0.0)),
    )


def validate_vectorized_zero_crossing_regression() -> None:
    sample_indices = np.arange(480, dtype=np.float64)
    frames = np.stack(
        [
            np.sin(2.0 * np.pi * frequency * sample_indices / SAMPLE_RATE)
            for frequency in (80.0, 120.0, 440.0, 1200.0, 3200.0)
        ]
    )
    observed = exact_zero_crossing_rates_aligned_frames(frames)
    expected = []
    for frame in frames:
        left = frame[:-1]
        right = frame[1:]
        indices = np.flatnonzero(
            ((left <= 0.0) & (right > 0.0))
            | ((left >= 0.0) & (right < 0.0))
        )
        denominator = left[indices] - right[indices]
        fraction = np.divide(
            left[indices],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0.0,
        )
        crossings = (
            0.5 / SAMPLE_RATE
            + (indices.astype(np.float64) + fraction) / SAMPLE_RATE
        )
        first = int(np.argmin(np.abs(crossings - 0.0025)))
        last_candidates = np.flatnonzero(crossings[first:] >= 0.0275)
        if indices.size < 2 or last_candidates.size == 0:
            expected.append(float("nan"))
            continue
        last = first + int(last_candidates[0])
        expected.append((last - first) / (crossings[last] - crossings[first]))
    expected_array = np.asarray(expected, dtype=np.float64)
    equal = (expected_array == observed) | (
        np.isnan(expected_array) & np.isnan(observed)
    )
    if not np.all(equal):
        raise ValueError("vectorized exact zero-crossing regression failed")


class ExactTopologyEngine:
    def __init__(self) -> None:
        tmpfs_root = Path("/dev/shm")
        runtime_root = str(tmpfs_root) if tmpfs_root.is_dir() else None
        self.runtime_wav_directory = tempfile.TemporaryDirectory(
            prefix="avqi-shimmer-runtime-",
            dir=runtime_root,
        )
        self.runtime_temp_backend = (
            "tmpfs_dev_shm" if runtime_root is not None else "system_temp"
        )
        self.numpy_stop_hann_response_cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self.runtime_wav_directory.cleanup()

    def praat_wav_roundtrip(
        self,
        sound: parselmouth.Sound,
        slot: str,
    ) -> np.ndarray:
        path = os.path.join(self.runtime_wav_directory.name, f"{slot}.wav")
        call(sound, "Save as WAV file", path)
        result, sample_rate = sf.read(path, dtype="float64")
        if sample_rate != SAMPLE_RATE:
            raise ValueError("Praat reused WAV roundtrip changed sample rate")
        return result

    def finish_metric_highpass(
        self,
        filtered: parselmouth.Sound,
        *,
        total_started: float,
        input_roundtrip_ms: float,
        sound_construct_ms: float,
        filter_ms: float,
        highpass_mode: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        peak_started = time.perf_counter()
        sample_abs_max: float | None = None
        sinc70_peak_upper_bound: float | None = None
        peak: float | None = None
        sinc70_skipped = False
        peak_check_mode = "exact_praat_sinc70"
        if highpass_mode == NUMPY_HIGHPASS_MODE:
            sample_abs_max = float(np.max(np.abs(filtered.values)))
            sinc70_peak_upper_bound = (
                sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
            )
            if sinc70_peak_upper_bound < PEAK_SCALE_TRIGGER:
                sinc70_skipped = True
                peak_check_mode = "proven_safe_sinc70_l1_upper_bound"
        if not sinc70_skipped:
            peak = float(
                call(filtered, "Get absolute extremum", 0, 0, "Sinc70")
            )
        peak_ms = 1000.0 * (time.perf_counter() - peak_started)
        scale_started = time.perf_counter()
        scaled = False
        if peak is not None and peak > PEAK_SCALE_TRIGGER:
            call(filtered, "Scale peak", 0.99)
            scaled = True
        scale_ms = 1000.0 * (time.perf_counter() - scale_started)
        quantize_started = time.perf_counter()
        highpassed = self.praat_wav_roundtrip(filtered, "highpass")
        quantize_ms = 1000.0 * (
            time.perf_counter() - quantize_started
        )
        return highpassed, {
            "highpass": 1000.0 * (time.perf_counter() - total_started),
            "highpass_input_pcm16_roundtrip": input_roundtrip_ms,
            "highpass_sound_construct": sound_construct_ms,
            "highpass_stop_hann_filter": filter_ms,
            "highpass_peak_extremum": peak_ms,
            "highpass_peak_check_mode": peak_check_mode,
            "highpass_sample_abs_max": sample_abs_max,
            "highpass_sinc70_peak_upper_bound": sinc70_peak_upper_bound,
            "highpass_sinc70_skipped": sinc70_skipped,
            "highpass_sinc70_absolute_weight_bound": (
                SINC70_ABSOLUTE_WEIGHT_BOUND
            ),
            "highpass_scale_peak": scale_ms,
            "highpass_quantize": quantize_ms,
            "highpass_peak_value": peak,
            "highpass_peak_scaled": scaled,
            "highpass_mode": highpass_mode,
            # Backward-compatible v15 receipt field.  In the frozen worker this
            # covered the filter, Sinc70 extremum, and optional peak scaling.
            "highpass_filter_compute": filter_ms + peak_ms + scale_ms,
        }

    def praat_metric_highpass(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        total_started = time.perf_counter()
        input_started = time.perf_counter()
        input_pcm16 = pcm16_roundtrip(waveform)
        input_roundtrip_ms = 1000.0 * (
            time.perf_counter() - input_started
        )
        construct_started = time.perf_counter()
        sound = parselmouth.Sound(input_pcm16, SAMPLE_RATE)
        sound_construct_ms = 1000.0 * (
            time.perf_counter() - construct_started
        )
        filter_started = time.perf_counter()
        filtered = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
        filter_ms = 1000.0 * (time.perf_counter() - filter_started)
        return self.finish_metric_highpass(
            filtered,
            total_started=total_started,
            input_roundtrip_ms=input_roundtrip_ms,
            sound_construct_ms=sound_construct_ms,
            filter_ms=filter_ms,
            highpass_mode=PRAAT_HIGHPASS_MODE,
        )

    def numpy_official_metric_highpass(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        total_started = time.perf_counter()
        input_started = time.perf_counter()
        input_pcm16 = pcm16_roundtrip(waveform)
        input_roundtrip_ms = 1000.0 * (
            time.perf_counter() - input_started
        )
        filter_started = time.perf_counter()
        filtered_values = self.numpy_official_stop_hann(input_pcm16)
        filter_ms = 1000.0 * (time.perf_counter() - filter_started)
        construct_started = time.perf_counter()
        filtered = parselmouth.Sound(filtered_values, SAMPLE_RATE)
        sound_construct_ms = 1000.0 * (
            time.perf_counter() - construct_started
        )
        return self.finish_metric_highpass(
            filtered,
            total_started=total_started,
            input_roundtrip_ms=input_roundtrip_ms,
            sound_construct_ms=sound_construct_ms,
            filter_ms=filter_ms,
            highpass_mode=NUMPY_HIGHPASS_MODE,
        )

    def numpy_stop_hann_response(
        self,
        number_of_fourier_samples: int,
    ) -> np.ndarray:
        cached = self.numpy_stop_hann_response_cache.get(
            number_of_fourier_samples
        )
        if cached is not None:
            return cached
        frequencies = (
            np.arange(
                number_of_fourier_samples // 2 + 1,
                dtype=np.float64,
            )
            * SAMPLE_RATE
            / number_of_fourier_samples
        )
        f3 = 34.0 - 0.1
        f4 = 34.0 + 0.1
        response = np.ones(frequencies.size, dtype=np.float64)
        response[frequencies <= f3] = 0.0
        transition = (frequencies > f3) & (frequencies <= f4)
        response[transition] = 0.5 - 0.5 * np.cos(
            np.pi / (2.0 * 0.1) * (frequencies[transition] - f3)
        )
        response.setflags(write=False)
        self.numpy_stop_hann_response_cache[number_of_fourier_samples] = (
            response
        )
        return response

    def numpy_official_stop_hann(self, input_pcm16: np.ndarray) -> np.ndarray:
        number_of_fourier_samples = 2
        while number_of_fourier_samples < input_pcm16.size:
            number_of_fourier_samples *= 2
        spectrum = np.fft.rfft(input_pcm16, n=number_of_fourier_samples)
        response = self.numpy_stop_hann_response(number_of_fourier_samples)
        return np.fft.irfft(
            spectrum * response,
            n=number_of_fourier_samples,
        )[: input_pcm16.size]

    def warm_numpy_fft_lengths(self) -> list[dict[str, Any]]:
        rows = []
        for sample_count in NUMPY_FFT_WARMUP_LENGTHS:
            values = np.zeros(sample_count, dtype=np.float64)
            values[0] = 1.0 / 32768.0
            started = time.perf_counter()
            filtered = self.numpy_official_stop_hann(values)
            rows.append(
                {
                    "sample_count": sample_count,
                    "fft_sample_count": sample_count,
                    "wall_ms": 1000.0 * (time.perf_counter() - started),
                    "filtered_pcm16_sha256": pcm16_sha256(filtered),
                    "response_cached_read_only": not self.numpy_stop_hann_response_cache[
                        sample_count
                    ].flags.writeable,
                }
            )
        return rows

    @staticmethod
    def validate_sinc70_safe_bound() -> None:
        # Praat's depth-70 kernel has 70 coefficients on each side.  From
        # NUM_interpolate_sinc, |d| <= |sin(pi*f)| / (pi*distance).  The two
        # nearest terms are each bounded by one; the remaining pairs by
        # 2/(pi*k), k=1..69.  The frozen 5.2 bound is deliberately looser.
        derived_bound = 2.0 + 2.0 / np.pi * sum(
            1.0 / index
            for index in range(1, SINC70_INTERPOLATION_DEPTH)
        )
        if not derived_bound < SINC70_ABSOLUTE_WEIGHT_BOUND:
            raise ValueError("Sinc70 absolute-weight safety bound drift")

    def metric_highpass(
        self,
        waveform: np.ndarray,
        highpass_mode: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if highpass_mode == PRAAT_HIGHPASS_MODE:
            return self.praat_metric_highpass(waveform)
        if highpass_mode == NUMPY_HIGHPASS_MODE:
            return self.numpy_official_metric_highpass(waveform)
        raise ValueError(f"unsupported exact high-pass mode: {highpass_mode}")

    def exact_cs_metric_waveform(
        self,
        highpassed: np.ndarray,
    ) -> tuple[np.ndarray, int, list[list[int]], dict[str, float]]:
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
        sounding_values: list[np.ndarray] = []
        sounding_source_indices: list[np.ndarray] = []
        for index in range(1, interval_count + 1):
            label = call(textgrid, "Get label of interval", 1, index)
            if "silence" in label:
                continue
            start = float(call(textgrid, "Get start point", 1, index))
            end = float(call(textgrid, "Get end point", 1, index))
            source_start = int(round(start * SAMPLE_RATE))
            source_end = int(round(end * SAMPLE_RATE))
            if source_end <= source_start or source_end > highpassed.size:
                raise ValueError("invalid exact sounding interval slice")
            sounding_values.append(highpassed[source_start:source_end])
            sounding_source_indices.append(
                np.arange(source_start, source_end, dtype=np.int64)
            )
        if not sounding_values:
            raise ValueError("exact CS preprocessing found no sounding interval")
        only_loud_values = np.concatenate(sounding_values)
        only_loud_indices = np.concatenate(sounding_source_indices)
        only_loud = parselmouth.Sound(only_loud_values, SAMPLE_RATE)
        global_power = float(call(only_loud, "Get power in air"))

        left = float(only_loud.xmin)
        width = 0.03
        right = left + width
        extreme_right = float(only_loud.xmax) - width
        frame_starts: list[int] = []
        while right < extreme_right:
            frame_starts.append(
                int(round((left - float(only_loud.xmin)) * SAMPLE_RATE))
            )
            left += width
            right = left + width
        frame_sample_count = int(round(width * SAMPLE_RATE))
        frame_indices = (
            np.asarray(frame_starts, dtype=np.int64)[:, None]
            + np.arange(frame_sample_count, dtype=np.int64)[None, :]
        )
        if frame_indices.size == 0 or int(frame_indices.max()) >= only_loud_values.size:
            raise ValueError("vectorized exact frame scan exceeded only-loud data")
        frame_values = only_loud_values[frame_indices]
        partial_powers = np.mean(np.square(frame_values), axis=1) / 400.0
        zero_crossing_rates = exact_zero_crossing_rates_aligned_frames(
            frame_values
        )
        keep = (
            (partial_powers > global_power * 0.30)
            & np.isfinite(zero_crossing_rates)
            & (zero_crossing_rates < 3000.0)
        )
        kept_values = frame_values[keep]
        kept_source_indices = only_loud_indices[frame_indices[keep]]
        if kept_values.size == 0:
            raise ValueError("exact CS preprocessing retained no 30-ms frame")
        source_selection_ms = 1000.0 * (
            time.perf_counter() - selection_started
        )

        gather_started = time.perf_counter()
        constant_prefix = round(0.001 * SAMPLE_RATE)
        metric_values = np.concatenate(
            [np.zeros(constant_prefix, dtype=np.float64), kept_values.reshape(-1)]
        )
        metric = self.praat_wav_roundtrip(
            parselmouth.Sound(metric_values, SAMPLE_RATE),
            "metric",
        )
        selected_indices = kept_source_indices.reshape(-1)
        ranges = compress_source_indices(selected_indices)
        reconstructed = np.concatenate(
            [np.zeros(constant_prefix, dtype=np.float64)]
            + [highpassed[start : start + length] for start, length in ranges]
        )
        difference = np.abs(pcm16(reconstructed) - pcm16(metric))
        if reconstructed.size != metric.size or np.any(difference):
            raise ValueError("exact CS metric source mapping failed parity")
        metric_gather_ms = 1000.0 * (
            time.perf_counter() - gather_started
        )
        return metric, constant_prefix, ranges, {
            "textgrid": textgrid_ms,
            "source_selection": source_selection_ms,
            "textgrid_range": textgrid_ms + source_selection_ms,
            "metric_gather": metric_gather_ms,
        }

    def exact_sv_metric_waveform(
        self,
        highpassed: np.ndarray,
    ) -> tuple[np.ndarray, int, list[list[int]], dict[str, float]]:
        started = time.perf_counter()
        metric_sample_count = min(highpassed.size, 3 * SAMPLE_RATE)
        crop_start = highpassed.size - metric_sample_count
        metric = highpassed[crop_start:].copy()
        ranges = [[int(crop_start), int(metric.size)]]
        return metric, 0, ranges, {
            "textgrid": 0.0,
            "source_selection": 0.0,
            "textgrid_range": 0.0,
            "metric_gather": 1000.0 * (time.perf_counter() - started),
        }

    def point_process_positions(
        self,
        metric: np.ndarray,
    ) -> tuple[list[float], dict[str, float]]:
        metric_sound = parselmouth.Sound(metric, SAMPLE_RATE)
        construct_started = time.perf_counter()
        point_process = call(
            metric_sound,
            "To PointProcess (periodic, cc)",
            50,
            400,
        )
        construct_ms = 1000.0 * (
            time.perf_counter() - construct_started
        )
        enumeration_started = time.perf_counter()
        matrix = call(point_process, "To Matrix")
        times = np.asarray(matrix.values, dtype=np.float64).reshape(-1)
        positions = (
            (times - float(metric_sound.x1)) / float(metric_sound.dx)
        ).tolist()
        enumeration_ms = 1000.0 * (
            time.perf_counter() - enumeration_started
        )
        return positions, {
            "pointprocess_construct": construct_ms,
            "pulse_enumeration": enumeration_ms,
        }

    def refresh_waveform(
        self,
        waveform: np.ndarray,
        view: str,
        input_read_ms: float,
        input_loader: str,
        waveform_float32_sha256: str,
        highpass_mode: str,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        highpassed, highpass_timing = self.metric_highpass(
            waveform,
            highpass_mode,
        )
        if view == "cs":
            metric, constant_prefix, ranges, view_timing = (
                self.exact_cs_metric_waveform(highpassed)
            )
        elif view == "sv":
            metric, constant_prefix, ranges, view_timing = (
                self.exact_sv_metric_waveform(highpassed)
            )
        else:
            raise ValueError(f"unsupported view: {view}")
        positions, point_timing = self.point_process_positions(metric)
        selected_indices = np.concatenate(
            [
                np.arange(start, start + length, dtype=np.int64)
                for start, length in ranges
            ]
        )
        reconstructed = np.concatenate(
            [np.zeros(constant_prefix, dtype=np.float64)]
            + [highpassed[start : start + length] for start, length in ranges]
        )
        difference = np.abs(pcm16(reconstructed) - pcm16(metric))
        maximum_error = int(difference.max(initial=0))
        differing_samples = int(np.count_nonzero(difference))
        if maximum_error != 0 or reconstructed.size != metric.size:
            raise ValueError("metric source mapping failed exact PCM16 parity")
        total_refresh_ms = 1000.0 * (
            time.perf_counter() - total_started
        ) + input_read_ms
        timing = {
            "input_read": input_read_ms,
            **highpass_timing,
            **view_timing,
            **point_timing,
            "total_refresh": total_refresh_ms,
        }
        return {
            "scoring_status": "ok",
            "pulse_positions_samples": positions,
            "pulse_count": len(positions),
            "pulse_runtime_ms": total_refresh_ms,
            "source_sample_count": int(highpassed.size),
            "metric_sample_count": int(metric.size),
            "metric_constant_prefix_samples": int(constant_prefix),
            "metric_source_ranges": ranges,
            "metric_source_range_count": len(ranges),
            "metric_mapped_sample_count": int(selected_indices.size),
            "metric_reconstruction_max_pcm16_error": maximum_error,
            "metric_reconstruction_differing_samples": differing_samples,
            "highpass_pcm16_sha256": pcm16_sha256(highpassed),
            "metric_pcm16_sha256": pcm16_sha256(metric),
            "source_ranges_sha256": ranges_sha256(ranges),
            "pulse_positions_sha256": pulses_sha256(positions),
            "topology_preprocessing": "exact_avqi_view_metric_waveform",
            "topology_input_loader": input_loader,
            "source_waveform_float32_sha256": waveform_float32_sha256,
            "metric_highpass": highpass_mode,
            "frame_scan_mode": "numpy_vectorized_exact_aligned_frames",
            "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
            "wav_roundtrip_mode": "praat_reused_tmpfs_wav",
            "sounding_assembly_mode": "numpy_exact_interval_slices",
            "implementation": IMPLEMENTATION,
            "timing_ms": timing,
        }

    def refresh_path(
        self,
        path: Path,
        view: str,
        highpass_mode: str,
    ) -> dict[str, Any]:
        input_started = time.perf_counter()
        waveform, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != SAMPLE_RATE or waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"exact topology input must be mono 16 kHz: {path}")
        float32_values = np.asarray(waveform, dtype="<f4")
        values = float32_values.astype(np.float64)
        input_read_ms = 1000.0 * (time.perf_counter() - input_started)
        return self.refresh_waveform(
            values,
            view,
            input_read_ms,
            "soundfile_float32_exact_16khz_mono",
            bytes_sha256(float32_values.tobytes()),
            highpass_mode,
        )

    def refresh_raw_float32(
        self,
        path: Path,
        sample_count: int,
        expected_sha256: str,
        view: str,
        highpass_mode: str,
    ) -> dict[str, Any]:
        input_started = time.perf_counter()
        payload = path.read_bytes()
        observed_sha256 = bytes_sha256(payload)
        if observed_sha256 != expected_sha256:
            raise ValueError("current-output tmpfs float32 hash drift")
        if len(payload) != sample_count * np.dtype("<f4").itemsize:
            raise ValueError("current-output tmpfs float32 size drift")
        float32_values = np.frombuffer(payload, dtype="<f4")
        if float32_values.size == 0 or not np.isfinite(float32_values).all():
            raise ValueError("invalid current-output tmpfs float32 waveform")
        values = float32_values.astype(np.float64)
        input_read_ms = 1000.0 * (time.perf_counter() - input_started)
        return self.refresh_waveform(
            values,
            view,
            input_read_ms,
            "client_tmpfs_raw_float32_current_output",
            observed_sha256,
            highpass_mode,
        )

    def warmup(self) -> dict[str, Any]:
        sample_indices = np.arange(4 * SAMPLE_RATE, dtype=np.float64)
        time_axis = sample_indices / SAMPLE_RATE
        envelope = np.zeros_like(time_axis)
        active = (time_axis >= 0.4) & (time_axis < 3.6)
        envelope[active] = 0.8 + 0.2 * np.sin(
            2.0 * np.pi * 2.0 * time_axis[active]
        )
        waveform = envelope * (
            0.08 * np.sin(2.0 * np.pi * 120.0 * time_axis)
            + 0.02 * np.sin(2.0 * np.pi * 240.0 * time_axis)
        )
        started = time.perf_counter()
        numpy_fft_warmup_rows = self.warm_numpy_fft_lengths()
        rows = []
        for view in ("cs", "sv"):
            float32_waveform = waveform.astype("<f4")
            topology = self.refresh_waveform(
                float32_waveform.astype(np.float64),
                view,
                0.0,
                "synthetic_in_memory_float32",
                bytes_sha256(float32_waveform.tobytes()),
                PRAAT_HIGHPASS_MODE,
            )
            rows.append(
                {
                    "view": view,
                    "pulse_count": topology["pulse_count"],
                    "source_ranges_sha256": topology["source_ranges_sha256"],
                    "pulse_positions_sha256": topology[
                        "pulse_positions_sha256"
                    ],
                    "internal_ms": topology["pulse_runtime_ms"],
                }
            )
        return {
            "synthetic_only": True,
            "panel_or_training_waveform_used": False,
            "numpy_fft_synthetic_warmup": numpy_fft_warmup_rows,
            "numpy_fft_warmup_lengths": list(NUMPY_FFT_WARMUP_LENGTHS),
            "numpy_stop_hann_response_cache_waveform_dependent": False,
            "rows": rows,
            "wall_ms": 1000.0 * (time.perf_counter() - started),
        }


def main() -> None:
    args = parse_args()
    actual_tree_hash = sha256_tree(args.avqi_code_root)
    if actual_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError(
            f"exact AVQI code-tree hash drift: {actual_tree_hash} != "
            f"{args.avqi_code_tree_sha256}"
        )
    validate_vectorized_zero_crossing_regression()
    ExactTopologyEngine.validate_sinc70_safe_bound()
    engine = ExactTopologyEngine()
    print(
        READY_MARKER
        + json.dumps(
            {
                "implementation": IMPLEMENTATION,
                "parselmouth_version": parselmouth.__version__,
                "praat_version": parselmouth.PRAAT_VERSION,
                "avqi_code_tree_sha256": actual_tree_hash,
                "runtime_temp_backend": engine.runtime_temp_backend,
                "cpu_affinity": (
                    sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else []
                ),
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
            if request["op"] == "warmup":
                response = {"status": "ok", "warmup": engine.warmup()}
            elif request["op"] == "refresh":
                rows = []
                for item in request["items"]:
                    role = str(item.get("role", ""))
                    if "target" in role.lower():
                        raise ValueError("clean target topology is forbidden")
                    source_path = Path(str(item.get("path", "")))
                    if any(
                        "target_clean" in part.lower()
                        for part in source_path.parts
                    ):
                        raise ValueError("clean target topology path is forbidden")
                    if "raw_float32_path" in item:
                        topology = engine.refresh_raw_float32(
                            Path(item["raw_float32_path"]),
                            int(item["raw_float32_sample_count"]),
                            str(item["raw_float32_sha256"]),
                            str(item["view"]),
                            str(item.get("highpass_mode", PRAAT_HIGHPASS_MODE)),
                        )
                    else:
                        topology = engine.refresh_path(
                            Path(item["path"]),
                            str(item["view"]),
                            str(item.get("highpass_mode", PRAAT_HIGHPASS_MODE)),
                        )
                    rows.append(
                        {
                            "id": item["id"],
                            "case_id": item["case_id"],
                            "role": role,
                            "view": item["view"],
                            **topology,
                        }
                    )
                response = {"status": "ok", "rows": rows}
            elif request["op"] == "quit":
                engine.close()
                response = {"status": "ok", "quitting": True}
                print(
                    RESULT_MARKER + json.dumps(response, sort_keys=True),
                    flush=True,
                )
                break
            else:
                raise ValueError(f"unknown worker operation: {request['op']}")
        except Exception as error:
            response = {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
            }
        print(
            RESULT_MARKER + json.dumps(response, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
