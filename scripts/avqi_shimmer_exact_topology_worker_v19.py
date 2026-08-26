#!/usr/bin/env python3
"""Exact-topology worker with the v19 paired candidate peak certificate.

The base waveform is still refreshed through the frozen NumPy Stop-Hann path.
For a dependent candidate, this worker reads both exact float32 payloads from
tmpfs, derives their exact PCM16 roundtrip codes, and may skip only the
candidate Praat Sinc70 peak search when the frozen paired certificate proves
the peak remains below 0.999.  Pulse/source extraction remains exact Praat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call

from scripts.avqi_shimmer_exact_topology_worker import (
    NUMPY_FFT_WARMUP_LENGTHS,
    NUMPY_HIGHPASS_MODE,
    PEAK_SCALE_TRIGGER,
    PRAAT_HIGHPASS_MODE,
    READY_MARKER,
    RESULT_MARKER,
    SAMPLE_RATE,
    SINC70_ABSOLUTE_WEIGHT_BOUND,
    ExactTopologyEngine,
    bytes_sha256,
    pcm16_roundtrip,
    sha256_tree,
    validate_vectorized_zero_crossing_regression,
)
from scripts.avqi_shimmer_peak_certificate_v19 import (
    paired_candidate_peak_certificate,
    pcm16_roundtrip_values_to_codes,
    power_of_two_fft_length,
    stop_hann_impulse_l1_certificate,
)


IMPLEMENTATION = "exact_paired_peak_certificate_tmpfs_v19"
PAIRED_CERTIFIED_MODE = "numpy_stop_hann_paired_peak_certificate_v19"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    return parser.parse_args()


class PairedPeakCertificateEngine(ExactTopologyEngine):
    def __init__(self) -> None:
        super().__init__()
        self.active_certificate: dict[str, Any] | None = None
        self.active_candidate_pcm16: np.ndarray | None = None
        self.active_candidate_pcm16_roundtrip_ms = 0.0
        self.impulse_certificate_cache: dict[int, dict[str, Any]] = {}

    def refresh_waveform(
        self,
        waveform: np.ndarray,
        view: str,
        input_read_ms: float,
        input_loader: str,
        waveform_float32_sha256: str,
        highpass_mode: str,
    ) -> dict[str, Any]:
        topology = super().refresh_waveform(
            waveform,
            view,
            input_read_ms,
            input_loader,
            waveform_float32_sha256,
            highpass_mode,
        )
        topology["implementation"] = IMPLEMENTATION
        return topology

    def impulse_certificate(self, sample_count: int) -> dict[str, Any]:
        fft_length = power_of_two_fft_length(sample_count)
        certificate = self.impulse_certificate_cache.get(fft_length)
        if certificate is None:
            certificate = stop_hann_impulse_l1_certificate(
                self.numpy_stop_hann_response(fft_length),
                fft_length,
            )
            self.impulse_certificate_cache[fft_length] = certificate
        return certificate

    def _prepare_paired_numpy_filtered(
        self,
        waveform: np.ndarray,
    ) -> tuple[parselmouth.Sound, float, float, float, float]:
        if self.active_candidate_pcm16 is None:
            raise ValueError("paired candidate PCM16 roundtrip is unavailable")
        if self.active_candidate_pcm16.size != waveform.size:
            raise ValueError("paired candidate PCM16/sample-count drift")
        total_started = time.perf_counter()
        filter_started = time.perf_counter()
        filtered_values = self.numpy_official_stop_hann(
            self.active_candidate_pcm16
        )
        filter_ms = 1000.0 * (time.perf_counter() - filter_started)
        construct_started = time.perf_counter()
        filtered = parselmouth.Sound(filtered_values, SAMPLE_RATE)
        construct_ms = 1000.0 * (time.perf_counter() - construct_started)
        return (
            filtered,
            total_started,
            self.active_candidate_pcm16_roundtrip_ms,
            filter_ms,
            construct_ms,
        )

    def _finish_paired_metric_highpass(
        self,
        filtered: parselmouth.Sound,
        *,
        total_started: float,
        input_roundtrip_ms: float,
        filter_ms: float,
        construct_ms: float,
        certificate: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sample_abs_max = float(np.max(np.abs(filtered.values)))
        local_upper_bound = sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
        certified_skip = bool(
            certificate["candidate_sinc70_search_may_be_skipped"]
        )
        peak_started = time.perf_counter()
        if certified_skip:
            peak = None
            peak_check_mode = "paired_base_delta_stop_hann_l1_upper_bound"
        else:
            peak = float(
                call(filtered, "Get absolute extremum", 0, 0, "Sinc70")
            )
            peak_check_mode = "exact_praat_sinc70"
        peak_ms = 1000.0 * (time.perf_counter() - peak_started)
        scale_started = time.perf_counter()
        scaled = False
        if peak is not None and peak > PEAK_SCALE_TRIGGER:
            call(filtered, "Scale peak", 0.99)
            scaled = True
        scale_ms = 1000.0 * (time.perf_counter() - scale_started)
        quantize_started = time.perf_counter()
        highpassed = self.praat_wav_roundtrip(filtered, "highpass")
        quantize_ms = 1000.0 * (time.perf_counter() - quantize_started)
        return highpassed, {
            "highpass": 1000.0 * (time.perf_counter() - total_started),
            "highpass_input_pcm16_roundtrip": input_roundtrip_ms,
            "highpass_sound_construct": construct_ms,
            "highpass_stop_hann_filter": filter_ms,
            "highpass_peak_extremum": peak_ms,
            "highpass_peak_check_mode": peak_check_mode,
            "highpass_sample_abs_max": sample_abs_max,
            "highpass_sinc70_peak_upper_bound": (
                certificate["candidate_sinc70_peak_upper_bound"]
                if certified_skip
                else local_upper_bound
            ),
            "highpass_sinc70_skipped": certified_skip,
            "highpass_sinc70_absolute_weight_bound": (
                SINC70_ABSOLUTE_WEIGHT_BOUND
            ),
            "highpass_scale_peak": scale_ms,
            "highpass_quantize": quantize_ms,
            "highpass_peak_value": peak,
            "highpass_peak_scaled": scaled,
            "highpass_mode": PAIRED_CERTIFIED_MODE,
            "highpass_filter_compute": filter_ms + peak_ms + scale_ms,
            "paired_peak_certificate": certificate,
        }

    def metric_highpass(
        self,
        waveform: np.ndarray,
        highpass_mode: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if highpass_mode != PAIRED_CERTIFIED_MODE:
            return super().metric_highpass(waveform, highpass_mode)
        if self.active_certificate is None:
            raise ValueError("paired candidate refresh lacks a peak certificate")
        filtered, started, input_ms, filter_ms, construct_ms = (
            self._prepare_paired_numpy_filtered(waveform)
        )
        return self._finish_paired_metric_highpass(
            filtered,
            total_started=started,
            input_roundtrip_ms=input_ms,
            filter_ms=filter_ms,
            construct_ms=construct_ms,
            certificate=self.active_certificate,
        )

    @staticmethod
    def _read_raw_float32(
        path: Path,
        sample_count: int,
        expected_sha256: str,
        label: str,
    ) -> tuple[np.ndarray, str, float]:
        started = time.perf_counter()
        payload = path.read_bytes()
        observed_sha256 = bytes_sha256(payload)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"{label} tmpfs float32 hash drift")
        if len(payload) != sample_count * np.dtype("<f4").itemsize:
            raise ValueError(f"{label} tmpfs float32 size drift")
        float32_values = np.frombuffer(payload, dtype="<f4")
        if float32_values.size == 0 or not np.isfinite(float32_values).all():
            raise ValueError(f"invalid {label} tmpfs float32 waveform")
        return (
            float32_values.astype(np.float64),
            observed_sha256,
            1000.0 * (time.perf_counter() - started),
        )

    def refresh_paired_raw_float32(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        candidate, candidate_hash, candidate_read_ms = self._read_raw_float32(
            Path(item["raw_float32_path"]),
            int(item["raw_float32_sample_count"]),
            str(item["raw_float32_sha256"]),
            "candidate",
        )
        base, base_hash, base_read_ms = self._read_raw_float32(
            Path(item["paired_base_raw_float32_path"]),
            int(item["paired_base_raw_float32_sample_count"]),
            str(item["paired_base_raw_float32_sha256"]),
            "paired base",
        )
        if base.shape != candidate.shape:
            raise ValueError("paired base/candidate waveform shape drift")
        if base_hash != item["paired_base_topology_source_float32_sha256"]:
            raise ValueError("paired base waveform/topology provenance drift")
        base_timing = item.get("paired_base_highpass_timing")
        if not isinstance(base_timing, dict):
            raise ValueError("paired base high-pass timing is unavailable")
        base_timing_sha256 = bytes_sha256(
            json.dumps(
                base_timing,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if base_timing_sha256 != item.get(
            "paired_base_highpass_timing_sha256"
        ):
            raise ValueError("paired base high-pass timing hash drift")
        paired_case_id = str(item.get("paired_base_case_id", ""))
        paired_view = str(item.get("paired_base_view", ""))
        paired_topology_sha256 = str(
            item.get("paired_base_topology_sha256", "")
        )
        if paired_case_id != str(item["case_id"]):
            raise ValueError("paired base/candidate case identity drift")
        if paired_view != str(item["view"]):
            raise ValueError("paired base/candidate view drift")
        if len(paired_topology_sha256) != 64:
            raise ValueError("paired base topology hash is unavailable")
        candidate_pcm24_sha256 = str(
            item.get("candidate_pcm24_sha256", "")
        )
        if len(candidate_pcm24_sha256) != 64:
            raise ValueError("paired candidate PCM24 hash is unavailable")

        certificate_started = time.perf_counter()
        base_roundtrip_started = time.perf_counter()
        base_pcm16 = pcm16_roundtrip(base)
        base_roundtrip_ms = 1000.0 * (
            time.perf_counter() - base_roundtrip_started
        )
        candidate_roundtrip_started = time.perf_counter()
        candidate_pcm16 = pcm16_roundtrip(candidate)
        candidate_roundtrip_ms = 1000.0 * (
            time.perf_counter() - candidate_roundtrip_started
        )
        certificate = paired_candidate_peak_certificate(
            pcm16_roundtrip_values_to_codes(base_pcm16),
            pcm16_roundtrip_values_to_codes(candidate_pcm16),
            base_timing,
            self.impulse_certificate(candidate.size),
        )
        certificate_compute_ms = 1000.0 * (
            time.perf_counter() - certificate_started
        )

        if (
            self.active_certificate is not None
            or self.active_candidate_pcm16 is not None
        ):
            raise RuntimeError("nested paired candidate refresh is forbidden")
        self.active_certificate = certificate
        self.active_candidate_pcm16 = candidate_pcm16
        self.active_candidate_pcm16_roundtrip_ms = candidate_roundtrip_ms
        try:
            topology = self.refresh_waveform(
                candidate,
                str(item["view"]),
                candidate_read_ms,
                "client_tmpfs_raw_float32_current_output_paired_v19",
                candidate_hash,
                PAIRED_CERTIFIED_MODE,
            )
        finally:
            self.active_certificate = None
            self.active_candidate_pcm16 = None
            self.active_candidate_pcm16_roundtrip_ms = 0.0

        additional_ms = base_read_ms + certificate_compute_ms
        topology["pulse_runtime_ms"] = (
            float(topology["pulse_runtime_ms"]) + additional_ms
        )
        timing = topology["timing_ms"]
        timing["paired_base_input_read"] = base_read_ms
        timing["paired_base_pcm16_roundtrip"] = base_roundtrip_ms
        timing["paired_candidate_pcm16_roundtrip"] = candidate_roundtrip_ms
        timing["paired_certificate_compute"] = certificate_compute_ms
        timing["total_refresh"] = topology["pulse_runtime_ms"]
        topology.update(
            {
                "paired_peak_certificate": certificate,
                "paired_base_source_waveform_float32_sha256": base_hash,
                "paired_base_case_id": paired_case_id,
                "paired_base_view": paired_view,
                "paired_base_topology_sha256": paired_topology_sha256,
                "paired_base_highpass_timing_sha256": base_timing_sha256,
                "paired_candidate_pcm24_sha256": candidate_pcm24_sha256,
                "paired_certificate_cache_waveform_dependent": False,
            }
        )
        return topology

    def warmup(self) -> dict[str, Any]:
        warmup = super().warmup()
        impulse_rows = []
        for fft_length in NUMPY_FFT_WARMUP_LENGTHS:
            certificate = self.impulse_certificate(fft_length)
            impulse_rows.append(
                {
                    "fft_length": fft_length,
                    "response_sha256": certificate["response_sha256"],
                    "impulse_l1_upper_bound": certificate[
                        "impulse_l1_upper_bound"
                    ],
                }
            )
        warmup.update(
            {
                "paired_peak_certificate_impulse_warmup": impulse_rows,
                "paired_peak_certificate_cache_keyed_only_by_fft_length": True,
                "paired_peak_certificate_cache_waveform_dependent": False,
            }
        )
        return warmup


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
    engine = PairedPeakCertificateEngine()
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
                "paired_candidate_highpass_mode": PAIRED_CERTIFIED_MODE,
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
                    highpass_mode = str(
                        item.get("highpass_mode", PRAAT_HIGHPASS_MODE)
                    )
                    if highpass_mode == PAIRED_CERTIFIED_MODE:
                        topology = engine.refresh_paired_raw_float32(item)
                    elif "raw_float32_path" in item:
                        topology = engine.refresh_raw_float32(
                            Path(item["raw_float32_path"]),
                            int(item["raw_float32_sample_count"]),
                            str(item["raw_float32_sha256"]),
                            str(item["view"]),
                            highpass_mode,
                        )
                    else:
                        topology = engine.refresh_path(
                            Path(item["path"]),
                            str(item["view"]),
                            highpass_mode,
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
