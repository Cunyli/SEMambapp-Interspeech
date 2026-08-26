#!/usr/bin/env python3
"""Thin client for the v19 exact paired-peak topology worker.

Candidate float32 payloads are always re-read from the materialized tmpfs
PCM24 files.  The paired base payload and high-pass timing are bound to the
same case's current base topology before they cross the worker boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    PRAAT_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    topology_sha256,
)


TMPFS_ROOT = Path("/dev/shm")
SAMPLE_RATE = 16_000
IMPLEMENTATION = "exact_paired_peak_certificate_tmpfs_v19"
PAIRED_CERTIFIED_MODE = "numpy_stop_hann_paired_peak_certificate_v19"
ALLOWED_V19_HIGHPASS_MODES = frozenset(
    {PRAAT_HIGHPASS_MODE, NUMPY_HIGHPASS_MODE, PAIRED_CERTIFIED_MODE}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_payload(values: np.ndarray) -> tuple[np.ndarray, bytes, str]:
    array = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("invalid v19 float32 waveform payload")
    payload = array.tobytes()
    return array, payload, hashlib.sha256(payload).hexdigest()


class PairedPeakCertificateTopologyWorker(ExactShimmerTopologyWorker):
    """Persistent v19 worker with case-bound paired candidate refreshes."""

    def __init__(
        self,
        exact_python: Path,
        worker_script: Path,
        avqi_code_root: Path,
        avqi_code_tree_sha256: str,
    ) -> None:
        for label, path in (
            ("exact Python", exact_python),
            ("v19 topology worker", worker_script),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing {label}: {path}")
        if not avqi_code_root.is_dir():
            raise FileNotFoundError(
                f"missing exact AVQI code root: {avqi_code_root}"
            )
        staging_root = str(TMPFS_ROOT) if TMPFS_ROOT.is_dir() else None
        self.staging_directory = tempfile.TemporaryDirectory(
            prefix="avqi-shimmer-v19-client-",
            dir=staging_root,
        )
        self.staging_backend = (
            "tmpfs_dev_shm" if staging_root is not None else "system_temp"
        )
        self._paired_request_index = 0
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [
                str(exact_python),
                "-u",
                str(worker_script),
                "--avqi-code-root",
                str(avqi_code_root),
                "--avqi-code-tree-sha256",
                avqi_code_tree_sha256,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.startup = self._read_marker("AVQI_SHIMMER_TOPOLOGY_READY=")
        self.startup_ms = 1000.0 * (time.perf_counter() - started)
        if self.startup.get("implementation") != IMPLEMENTATION:
            self.close()
            raise ValueError("v19 exact topology worker implementation drift")
        if (
            self.startup.get("avqi_code_tree_sha256")
            != avqi_code_tree_sha256
        ):
            self.close()
            raise ValueError("v19 exact topology worker code-tree hash drift")
        if (
            self.startup.get("paired_candidate_highpass_mode")
            != PAIRED_CERTIFIED_MODE
        ):
            self.close()
            raise ValueError("v19 paired high-pass mode drift")

    def _validate_rows(
        self,
        response: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ids = [str(item["id"]) for item in items]
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) != len(items):
            raise ValueError("v19 exact topology worker row count drift")
        if [row.get("id") for row in rows] != ids:
            raise ValueError("v19 exact topology worker row order drift")
        for row, item in zip(rows, items, strict=True):
            if row.get("case_id") != item.get("case_id"):
                raise ValueError("v19 exact topology worker case identity drift")
            if row.get("implementation") != IMPLEMENTATION:
                raise ValueError("v19 exact topology row implementation drift")
            expected_highpass_mode = str(
                item.get("highpass_mode", PRAAT_HIGHPASS_MODE)
            )
            if expected_highpass_mode not in ALLOWED_V19_HIGHPASS_MODES:
                raise ValueError("unsupported v19 exact high-pass mode")
            if row.get("metric_highpass") != expected_highpass_mode:
                raise ValueError("v19 exact topology high-pass mode drift")
            if row.get("scoring_status") != "ok" or int(
                row.get("pulse_count", 0)
            ) < 3:
                raise ValueError(
                    f"v19 exact topology unavailable: {row.get('case_id')}"
                )
            if expected_highpass_mode == PAIRED_CERTIFIED_MODE:
                self._validate_paired_row(row, item)
        return rows

    @staticmethod
    def _validate_paired_row(
        row: dict[str, Any],
        item: dict[str, Any],
    ) -> None:
        if row.get("topology_input_loader") != (
            "client_tmpfs_raw_float32_current_output_paired_v19"
        ):
            raise ValueError("v19 paired tmpfs loader contract drift")
        expected_fields = {
            "source_waveform_float32_sha256": item["raw_float32_sha256"],
            "paired_base_source_waveform_float32_sha256": item[
                "paired_base_raw_float32_sha256"
            ],
            "paired_base_case_id": item["paired_base_case_id"],
            "paired_base_view": item["paired_base_view"],
            "paired_base_topology_sha256": item[
                "paired_base_topology_sha256"
            ],
            "paired_base_highpass_timing_sha256": item[
                "paired_base_highpass_timing_sha256"
            ],
            "paired_candidate_pcm24_sha256": item[
                "candidate_pcm24_sha256"
            ],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(f"v19 paired row {field} drift")
        if row.get("paired_certificate_cache_waveform_dependent") is not False:
            raise ValueError("v19 paired worker used waveform-dependent cache")
        certificate = row.get("paired_peak_certificate")
        if not isinstance(certificate, dict):
            raise ValueError("v19 paired peak certificate is unavailable")
        if certificate.get("paired_input_contract") != (
            "exact_worker_pcm16_roundtrip_int16_codes"
        ):
            raise ValueError("v19 paired PCM16 certificate contract drift")
        if certificate.get("response_cache_waveform_dependent") is not False:
            raise ValueError("v19 paired response cache contract drift")

    @staticmethod
    def _validate_base_binding(
        case_id: str,
        view: str,
        base_waveform: np.ndarray,
        base_topology: dict[str, Any],
        expected_base_topology_sha256: str,
    ) -> tuple[np.ndarray, bytes, str, dict[str, Any]]:
        if base_topology.get("case_id") != case_id:
            raise ValueError("v19 paired base topology case identity drift")
        if base_topology.get("view") != view:
            raise ValueError("v19 paired base topology view drift")
        observed_topology_sha256 = topology_sha256(base_topology)
        if observed_topology_sha256 != expected_base_topology_sha256:
            raise ValueError("v19 paired base topology hash drift")
        if base_topology.get("metric_highpass") != NUMPY_HIGHPASS_MODE:
            raise ValueError("v19 paired base is not the current NumPy topology")
        if base_topology.get("scoring_status") != "ok":
            raise ValueError("v19 paired base topology is unavailable")
        values, payload, payload_sha256 = float32_payload(base_waveform)
        if int(base_topology["source_sample_count"]) != int(values.size):
            raise ValueError("v19 paired base sample count drift")
        if (
            base_topology.get("source_waveform_float32_sha256")
            != payload_sha256
        ):
            raise ValueError("v19 paired base source waveform hash drift")
        timing = base_topology.get("timing_ms")
        if not isinstance(timing, dict):
            raise ValueError("v19 paired base high-pass timing is unavailable")
        if timing.get("highpass_mode") != NUMPY_HIGHPASS_MODE:
            raise ValueError("v19 paired base high-pass timing mode drift")
        return values, payload, payload_sha256, timing

    @staticmethod
    def _read_tmpfs_pcm24(
        path: Path,
        expected_sha256: str,
        expected_raw_float32_sha256: str,
        expected_stored_waveform: np.ndarray,
    ) -> tuple[np.ndarray, str, str]:
        resolved = path.resolve()
        tmpfs_root = TMPFS_ROOT.resolve()
        if tmpfs_root != resolved and tmpfs_root not in resolved.parents:
            raise ValueError("v19 candidate PCM24 is not staged on node tmpfs")
        observed_pcm24_sha256 = sha256_file(resolved)
        if observed_pcm24_sha256 != expected_sha256:
            raise ValueError("v19 candidate PCM24 hash drift before refresh")
        stored, sample_rate = sf.read(
            resolved,
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != SAMPLE_RATE or stored.ndim != 1 or stored.size == 0:
            raise ValueError("v19 candidate PCM24 format drift")
        values, _, raw_float32_sha256 = float32_payload(stored)
        if raw_float32_sha256 != expected_raw_float32_sha256:
            raise ValueError(
                "v19 candidate PCM24 readback float32 hash drift"
            )
        expected_values = np.ascontiguousarray(
            expected_stored_waveform,
            dtype="<f4",
        ).reshape(-1)
        if not np.array_equal(values, expected_values):
            raise ValueError(
                "v19 candidate refresh did not use frozen PCM24 readback"
            )
        return values, observed_pcm24_sha256, raw_float32_sha256

    def refresh_current_pcm24_candidates_paired(
        self,
        items: list[dict[str, Any]],
        candidate_paths: list[Path],
        expected_pcm24_sha256: list[str],
        expected_raw_float32_sha256: list[str],
        expected_stored_waveforms: list[np.ndarray],
        *,
        case_id: str,
        base_waveform: np.ndarray,
        base_topology: dict[str, Any],
        base_topology_sha256: str,
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Refresh only PCM24-readback candidates against one current base."""

        if not (
            len(items)
            == len(candidate_paths)
            == len(expected_pcm24_sha256)
            == len(expected_raw_float32_sha256)
            == len(expected_stored_waveforms)
        ):
            raise ValueError("v19 paired candidate cardinality drift")
        self._validate_current_output_items(items)
        views = {str(item["view"]) for item in items}
        case_ids = {str(item["case_id"]) for item in items}
        if case_ids != {case_id} or len(views) != 1:
            raise ValueError("v19 paired candidate case/view grouping drift")
        view = next(iter(views))
        base_values, base_payload, base_sha256, base_timing = (
            self._validate_base_binding(
                case_id,
                view,
                base_waveform,
                base_topology,
                base_topology_sha256,
            )
        )
        base_timing_sha256 = hashlib.sha256(
            json.dumps(
                base_timing,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self._paired_request_index += 1
        request_prefix = f"paired_{self._paired_request_index}"
        base_stage_started = time.perf_counter()
        base_slot = (
            Path(self.staging_directory.name) / f"{request_prefix}_base.f32"
        )
        base_slot.write_bytes(base_payload)
        base_staging_ms = 1000.0 * (
            time.perf_counter() - base_stage_started
        )

        staged_items: list[dict[str, Any]] = []
        staging_rows: list[dict[str, Any]] = []
        for index, (
            item,
            candidate_path,
            pcm24_sha256,
            raw_float32_sha256,
            expected_waveform,
        ) in enumerate(
            zip(
                items,
                candidate_paths,
                expected_pcm24_sha256,
                expected_raw_float32_sha256,
                expected_stored_waveforms,
                strict=True,
            )
        ):
            if Path(item["path"]).resolve() != candidate_path.resolve():
                raise ValueError("v19 candidate item/path binding drift")
            staging_started = time.perf_counter()
            values, observed_pcm24_sha256, raw_sha256 = (
                self._read_tmpfs_pcm24(
                    candidate_path,
                    pcm24_sha256,
                    raw_float32_sha256,
                    expected_waveform,
                )
            )
            candidate_payload = values.tobytes()
            candidate_slot = (
                Path(self.staging_directory.name)
                / f"{request_prefix}_candidate_{index}.f32"
            )
            candidate_slot.write_bytes(candidate_payload)
            staging_ms = 1000.0 * (
                time.perf_counter() - staging_started
            )
            staged_items.append(
                {
                    **item,
                    "highpass_mode": PAIRED_CERTIFIED_MODE,
                    "raw_float32_path": str(candidate_slot),
                    "raw_float32_sample_count": int(values.size),
                    "raw_float32_sha256": raw_sha256,
                    "paired_base_raw_float32_path": str(base_slot),
                    "paired_base_raw_float32_sample_count": int(
                        base_values.size
                    ),
                    "paired_base_raw_float32_sha256": base_sha256,
                    "paired_base_topology_source_float32_sha256": (
                        base_topology["source_waveform_float32_sha256"]
                    ),
                    "paired_base_highpass_timing": base_timing,
                    "paired_base_highpass_timing_sha256": (
                        base_timing_sha256
                    ),
                    "paired_base_case_id": case_id,
                    "paired_base_view": view,
                    "paired_base_topology_sha256": base_topology_sha256,
                    "candidate_pcm24_sha256": observed_pcm24_sha256,
                }
            )
            staging_rows.append(
                {
                    "id": item["id"],
                    "sample_count": int(values.size),
                    "candidate_pcm24_sha256": observed_pcm24_sha256,
                    "raw_float32_sha256": raw_sha256,
                    "pcm24_readback_equals_frozen_stored_waveform": True,
                    "staging_ms": staging_ms,
                    "base_staging_ms": (
                        base_staging_ms if index == 0 else 0.0
                    ),
                    "staging_backend": self.staging_backend,
                }
            )

        response, wall_ms = self.request(
            {"op": "refresh", "items": staged_items}
        )
        rows = self._validate_rows(response, staged_items)
        for row, staging, expected_pcm24 in zip(
            rows,
            staging_rows,
            expected_pcm24_sha256,
            strict=True,
        ):
            if (
                row.get("source_waveform_float32_sha256")
                != staging["raw_float32_sha256"]
            ):
                raise ValueError("v19 candidate raw-float32 hash drift")
            if staging["candidate_pcm24_sha256"] != expected_pcm24:
                raise ValueError("v19 candidate PCM24 receipt drift")
        return rows, wall_ms, staging_rows
