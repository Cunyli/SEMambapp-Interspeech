#!/usr/bin/env python3
"""Audit a paired exact-peak certificate on immutable opened v18 dev cases.

This probe never scores candidate AVQI components.  It forces the frozen exact
Praat Sinc70 peak search for reference, then evaluates a fail-closed paired
certificate that may skip only the candidate search.  Source mappings, pulse
arrays, post-high-pass PCM16, and durable PCM24 bytes must remain identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    require_exact_topology_equal,
    topology_sha256,
)
from scripts.avqi_shimmer_exact_topology_worker import (
    ExactTopologyEngine,
    SAMPLE_RATE,
    bytes_sha256,
    pcm16_roundtrip,
)
from scripts.avqi_shimmer_peak_certificate_v19 import (
    FROZEN_NUMPY_HIGHPASS_MODE,
    PEAK_SCALE_TRIGGER,
    SINC70_ABSOLUTE_WEIGHT_BOUND,
    paired_candidate_peak_certificate,
    pcm16_roundtrip_values_to_codes,
    power_of_two_fft_length,
    stop_hann_impulse_l1_certificate,
)


FORCED_EXACT_MODE = "numpy_stop_hann_forced_exact_praat_sinc70_v19_probe"
PAIRED_CERTIFIED_MODE = "numpy_stop_hann_paired_peak_certificate_v19_probe"
PASS_DECISION = "PASS_SHIMMER_DB_RUNTIME_V19_PAIRED_PEAK_EQUIVALENCE"
FAIL_DECISION = "NO_GO_SHIMMER_DB_RUNTIME_V19_PAIRED_PEAK_EQUIVALENCE"
V18_NO_GO_DECISION = "NO_GO_SHIMMER_DB_V18_OPENED24_PRESELECTION"
V18_SOURCE_COMMIT = "cb29d05ec073649b5d11beb7d5813f445d38eb43"
EXPECTED_CASE_COUNT = 24
EXPECTED_PANEL_CASE_COUNT = 12
EXPECTED_SPEAKER_COUNT = 12
FORMAL_RUNTIME_GATE_MS = 500.0
ENGINEERING_MARGIN_MS = 450.0
DEFAULT_REPEATS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for panel_label in ("v14", "v15"):
        parser.add_argument(
            f"--{panel_label}-panel-contract", type=Path, required=True
        )
        parser.add_argument(
            f"--{panel_label}-panel-contract-sha256", required=True
        )
    for artifact in ("report", "preselection", "receipt"):
        parser.add_argument(f"--v18-{artifact}", type=Path, required=True)
        parser.add_argument(f"--v18-{artifact}-sha256", required=True)
    parser.add_argument("--v18-run-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--peak-certificate-helper-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--frozen-worker-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")
    return observed


def git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_repository_provenance(
    args: argparse.Namespace,
) -> dict[str, Any]:
    repository_root = args.repository_root.resolve()
    expected_root = Path(__file__).resolve().parents[1]
    if repository_root != expected_root:
        raise ValueError("repository root does not contain the running evaluator")
    observed_head = git_output(repository_root, "rev-parse", "HEAD")
    if observed_head != args.source_commit:
        raise ValueError(
            f"repository HEAD/source commit drift: {observed_head} != "
            f"{args.source_commit}"
        )
    observed_status = git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if observed_status:
        raise ValueError("v19 evaluator requires a clean repository worktree")
    if NUMPY_HIGHPASS_MODE != FROZEN_NUMPY_HIGHPASS_MODE:
        raise ValueError("helper/frozen worker NumPy high-pass mode drift")

    implementation_paths = {
        "peak_certificate_helper": (
            repository_root / "scripts" / "avqi_shimmer_peak_certificate_v19.py"
        ),
        "evaluator": Path(__file__).resolve(),
        "frozen_worker": (
            repository_root / "scripts" / "avqi_shimmer_exact_topology_worker.py"
        ),
    }
    implementation_expected = {
        "peak_certificate_helper": args.peak_certificate_helper_sha256,
        "evaluator": args.evaluator_sha256,
        "frozen_worker": args.frozen_worker_sha256,
    }
    observed = {
        name: validate_hash(
            path,
            implementation_expected[name],
            f"v19 {name}",
        )
        for name, path in implementation_paths.items()
    }
    return {
        "repository_head": observed_head,
        "repository_tree_clean": True,
        "implementation_sha256": observed,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty v19 evidence CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected mono 16-kHz waveform: {path}")
    float32_values = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(float32_values).all():
        raise ValueError(f"nonfinite waveform: {path}")
    return (
        float32_values,
        float32_values.astype(np.float64),
        bytes_sha256(float32_values.tobytes()),
    )


class PeakCertificateProbeEngine(ExactTopologyEngine):
    def __init__(self) -> None:
        super().__init__()
        self.active_certificate: dict[str, Any] | None = None

    def _prepare_numpy_filtered(
        self,
        waveform: np.ndarray,
    ) -> tuple[parselmouth.Sound, float, float, float]:
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
        construct_ms = 1000.0 * (
            time.perf_counter() - construct_started
        )
        return filtered, total_started, input_roundtrip_ms, filter_ms + construct_ms

    def _finish_probe_highpass(
        self,
        filtered: parselmouth.Sound,
        total_started: float,
        input_roundtrip_ms: float,
        filter_and_construct_ms: float,
        mode: str,
        certificate: dict[str, Any] | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sample_abs_max = float(np.max(np.abs(filtered.values)))
        local_upper_bound = sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
        certified_skip = bool(
            certificate is not None
            and certificate["candidate_sinc70_search_may_be_skipped"]
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
        quantize_ms = 1000.0 * (
            time.perf_counter() - quantize_started
        )
        return highpassed, {
            "highpass": 1000.0 * (time.perf_counter() - total_started),
            "highpass_input_pcm16_roundtrip": input_roundtrip_ms,
            "highpass_sound_construct": 0.0,
            "highpass_stop_hann_filter": filter_and_construct_ms,
            "highpass_peak_extremum": peak_ms,
            "highpass_peak_check_mode": peak_check_mode,
            "highpass_sample_abs_max": sample_abs_max,
            "highpass_sinc70_peak_upper_bound": (
                certificate["candidate_sinc70_peak_upper_bound"]
                if certified_skip and certificate is not None
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
            "highpass_mode": mode,
            "highpass_filter_compute": (
                filter_and_construct_ms + peak_ms + scale_ms
            ),
            "paired_peak_certificate": certificate,
        }

    def metric_highpass(
        self,
        waveform: np.ndarray,
        highpass_mode: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if highpass_mode not in {FORCED_EXACT_MODE, PAIRED_CERTIFIED_MODE}:
            return super().metric_highpass(waveform, highpass_mode)
        filtered, started, input_ms, filter_construct_ms = (
            self._prepare_numpy_filtered(waveform)
        )
        certificate = (
            self.active_certificate
            if highpass_mode == PAIRED_CERTIFIED_MODE
            else None
        )
        if highpass_mode == PAIRED_CERTIFIED_MODE and certificate is None:
            raise ValueError("paired certified high-pass lacks a certificate")
        return self._finish_probe_highpass(
            filtered,
            started,
            input_ms,
            filter_construct_ms,
            highpass_mode,
            certificate,
        )

    def refresh_certified(
        self,
        waveform: np.ndarray,
        view: str,
        waveform_hash: str,
        certificate: dict[str, Any],
    ) -> dict[str, Any]:
        if self.active_certificate is not None:
            raise RuntimeError("nested paired peak certificate is forbidden")
        self.active_certificate = certificate
        try:
            return self.refresh_waveform(
                waveform,
                view,
                0.0,
                "v19_in_memory_float32_opened_dev_probe",
                waveform_hash,
                PAIRED_CERTIFIED_MODE,
            )
        finally:
            self.active_certificate = None


def refresh_probe_waveform(
    engine: PeakCertificateProbeEngine,
    waveform: np.ndarray,
    view: str,
    waveform_hash: str,
    mode: str,
) -> dict[str, Any]:
    return engine.refresh_waveform(
        waveform,
        view,
        0.0,
        "v19_in_memory_float32_opened_dev_probe",
        waveform_hash,
        mode,
    )


def require_equivalent(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
) -> str:
    topology_hash = require_exact_topology_equal(reference, candidate, label)
    exact_fields = (
        "highpass_pcm16_sha256",
        "metric_pcm16_sha256",
        "source_ranges_sha256",
        "pulse_positions_sha256",
    )
    differences = {
        field: {"reference": reference[field], "candidate": candidate[field]}
        for field in exact_fields
        if reference[field] != candidate[field]
    }
    if differences:
        raise ValueError(
            f"{label}: post-highpass/topology hash drift "
            + json.dumps(differences, sort_keys=True)
        )
    return topology_hash


def impulse_certificate_for_waveform(
    engine: PeakCertificateProbeEngine,
    waveform: np.ndarray,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    fft_length = power_of_two_fft_length(waveform.size)
    certificate = cache.get(fft_length)
    if certificate is None:
        response = engine.numpy_stop_hann_response(fft_length)
        certificate = stop_hann_impulse_l1_certificate(
            response,
            fft_length,
        )
        cache[fft_length] = certificate
    return certificate


def evaluate_pair(
    engine: PeakCertificateProbeEngine,
    case_id: str,
    view: str,
    base_values: np.ndarray,
    base_hash: str,
    candidate_values: np.ndarray,
    candidate_hash: str,
    impulse_cache: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_topology = refresh_probe_waveform(
        engine,
        base_values,
        view,
        base_hash,
        NUMPY_HIGHPASS_MODE,
    )
    frozen_candidate = refresh_probe_waveform(
        engine,
        candidate_values,
        view,
        candidate_hash,
        NUMPY_HIGHPASS_MODE,
    )
    forced_candidate = refresh_probe_waveform(
        engine,
        candidate_values,
        view,
        candidate_hash,
        FORCED_EXACT_MODE,
    )
    require_equivalent(
        frozen_candidate,
        forced_candidate,
        f"{case_id}:frozen-vs-forced-exact",
    )

    certificate_started = time.perf_counter()
    base_pcm16_codes = pcm16_roundtrip_values_to_codes(
        pcm16_roundtrip(base_values)
    )
    candidate_pcm16_codes = pcm16_roundtrip_values_to_codes(
        pcm16_roundtrip(candidate_values)
    )
    certificate = paired_candidate_peak_certificate(
        base_pcm16_codes,
        candidate_pcm16_codes,
        base_topology["timing_ms"],
        impulse_certificate_for_waveform(engine, base_values, impulse_cache),
    )
    certificate_compute_ms = 1000.0 * (
        time.perf_counter() - certificate_started
    )
    certified_candidate = engine.refresh_certified(
        candidate_values,
        view,
        candidate_hash,
        certificate,
    )
    topology_hash = require_equivalent(
        forced_candidate,
        certified_candidate,
        f"{case_id}:forced-exact-vs-certified",
    )
    forced_timing = forced_candidate["timing_ms"]
    certified_timing = certified_candidate["timing_ms"]
    exact_candidate_peak = float(forced_timing["highpass_peak_value"])
    bound_valid = (
        float(certificate["candidate_sinc70_peak_upper_bound"])
        >= exact_candidate_peak
    )
    if not bound_valid:
        raise ValueError(f"{case_id}: paired peak certificate underbounded exact peak")
    if certificate["candidate_sinc70_search_may_be_skipped"] and not (
        exact_candidate_peak < PEAK_SCALE_TRIGGER
    ):
        raise ValueError(f"{case_id}: certificate skipped an exact scaling case")
    if bool(forced_timing["highpass_peak_scaled"]) != bool(
        certified_timing["highpass_peak_scaled"]
    ):
        raise ValueError(f"{case_id}: paired certificate scale decision drift")
    row = {
        "case_id": case_id,
        "view": view,
        "fft_length": certificate["fft_length"],
        "impulse_l1_observed": certificate["impulse_l1_observed"],
        "impulse_l1_irfft_construction_epsilon": certificate[
            "impulse_l1_irfft_construction_epsilon"
        ],
        "impulse_l1_summation_epsilon": certificate[
            "impulse_l1_summation_epsilon"
        ],
        "impulse_l1_upper_bound": certificate["impulse_l1_upper_bound"],
        "response_sha256": certificate["response_sha256"],
        "base_peak_check_mode": certificate["base_peak_check_mode"],
        "base_peak_bound_source": certificate["base_peak_bound_source"],
        "base_peak_upper_bound": certificate["base_peak_upper_bound"],
        "base_highpass_peak_scaled": certificate[
            "base_highpass_peak_scaled"
        ],
        "paired_input_contract": certificate["paired_input_contract"],
        "base_pcm16_codes_sha256": certificate["base_pcm16_codes_sha256"],
        "candidate_pcm16_codes_sha256": certificate[
            "candidate_pcm16_codes_sha256"
        ],
        "paired_pcm16_difference_max_abs": certificate[
            "paired_pcm16_difference_max_abs"
        ],
        "fft_roundoff_per_transform_epsilon": certificate[
            "fft_roundoff_per_transform_epsilon"
        ],
        "fft_roundoff_transform_count": certificate[
            "fft_roundoff_transform_count"
        ],
        "fft_roundoff_epsilon": certificate["fft_roundoff_epsilon"],
        "filtered_difference_upper_bound": certificate[
            "filtered_difference_upper_bound"
        ],
        "sinc70_interpolation_difference_upper_bound": certificate[
            "sinc70_interpolation_difference_upper_bound"
        ],
        "candidate_sinc70_peak_upper_bound": certificate[
            "candidate_sinc70_peak_upper_bound"
        ],
        "exact_candidate_sinc70_peak": exact_candidate_peak,
        "certificate_minus_exact_peak": (
            certificate["candidate_sinc70_peak_upper_bound"]
            - exact_candidate_peak
        ),
        "candidate_sinc70_search_may_be_skipped": certificate[
            "candidate_sinc70_search_may_be_skipped"
        ],
        "certificate_failure_mode": certificate["failure_mode"],
        "bound_covers_observed_exact_peak": bound_valid,
        "post_highpass_pcm16_equal": True,
        "metric_pcm16_equal": True,
        "source_mapping_equal": True,
        "pulse_positions_equal": True,
        "topology_sha256": topology_hash,
        "forced_exact_peak_search_ms": forced_timing[
            "highpass_peak_extremum"
        ],
        "certified_peak_search_ms": certified_timing[
            "highpass_peak_extremum"
        ],
        "forced_exact_refresh_ms": forced_candidate["pulse_runtime_ms"],
        "certified_refresh_ms": certified_candidate["pulse_runtime_ms"],
        "certificate_compute_ms": certificate_compute_ms,
        "forced_exact_peak_scaled": forced_timing["highpass_peak_scaled"],
        "certified_peak_scaled": certified_timing["highpass_peak_scaled"],
    }
    return row, forced_candidate, certificate


def pcm24_tmpfs_equivalence(
    case_id: str,
    values: np.ndarray,
    original_path: Path,
    original_hash: str,
    direct_root: Path,
    copied_root: Path,
    tmpfs_root: Path,
) -> dict[str, Any]:
    safe_name = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in case_id
    )
    direct_path = direct_root / f"{safe_name}__direct.wav"
    copied_path = copied_root / f"{safe_name}__copied_from_tmpfs.wav"
    tmpfs_path = tmpfs_root / f"{safe_name}__tmpfs.wav"
    direct_started = time.perf_counter()
    sf.write(direct_path, values, SAMPLE_RATE, subtype="PCM_24")
    direct_write_ms = 1000.0 * (time.perf_counter() - direct_started)
    tmpfs_started = time.perf_counter()
    sf.write(tmpfs_path, values, SAMPLE_RATE, subtype="PCM_24")
    tmpfs_write_ms = 1000.0 * (time.perf_counter() - tmpfs_started)
    direct_hash = sha256_file(direct_path)
    tmpfs_hash = sha256_file(tmpfs_path)
    copy_started = time.perf_counter()
    shutil.copyfile(tmpfs_path, copied_path)
    durable_copy_ms = 1000.0 * (time.perf_counter() - copy_started)
    copied_hash = sha256_file(copied_path)
    observed_original_hash = sha256_file(original_path)
    all_equal = (
        observed_original_hash
        == original_hash
        == direct_hash
        == tmpfs_hash
        == copied_hash
    )
    if not all_equal:
        raise ValueError(f"{case_id}: tmpfs/durable PCM24 byte drift")
    return {
        "case_id": case_id,
        "original_sha256": original_hash,
        "direct_sha256": direct_hash,
        "tmpfs_sha256": tmpfs_hash,
        "copied_durable_sha256": copied_hash,
        "all_pcm24_bytes_equal": all_equal,
        "direct_write_ms": direct_write_ms,
        "tmpfs_write_ms": tmpfs_write_ms,
        "durable_copy_after_timed_step_ms": durable_copy_ms,
        "tmpfs_backend": "node_local_dev_shm",
    }


def synthetic_controls(
    engine: PeakCertificateProbeEngine,
    impulse_cache: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline = np.arange(4 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    specifications = (
        ("safe_local_bound", 0.04, 0.00005, True, False),
        ("paired_certificate_only", 0.20, 0.00005, True, False),
        ("forced_exact_scale_fallback", 0.9999, 0.0, False, True),
    )
    rows: list[dict[str, Any]] = []
    for name, amplitude, delta, expected_skip, expected_scale in specifications:
        base = (
            amplitude * np.sin(2.0 * np.pi * 127.0 * timeline)
        ).astype("<f4")
        candidate = (
            base.astype(np.float64)
            + delta * np.sin(2.0 * np.pi * 233.0 * timeline)
        ).astype("<f4")
        row, _, _ = evaluate_pair(
            engine,
            f"synthetic:{name}",
            "sv",
            base.astype(np.float64),
            bytes_sha256(base.tobytes()),
            candidate.astype(np.float64),
            bytes_sha256(candidate.tobytes()),
            impulse_cache,
        )
        skip_matches = (
            bool(row["candidate_sinc70_search_may_be_skipped"])
            is expected_skip
        )
        scale_matches = bool(row["forced_exact_peak_scaled"]) is expected_scale
        row.update(
            {
                "control_name": name,
                "expected_certificate_skip": expected_skip,
                "expected_exact_scale": expected_scale,
                "certificate_skip_matches_expected": skip_matches,
                "exact_scale_matches_expected": scale_matches,
            }
        )
        if not skip_matches or not scale_matches:
            raise ValueError(f"synthetic paired-peak control failed: {name}")
        rows.append(row)
    return rows


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, str],
    list[str],
]:
    hashes: dict[str, str] = {}
    panel_rows: list[dict[str, Any]] = []
    speakers: set[str] = set()
    for panel_label in ("v14", "v15"):
        panel_path = getattr(args, f"{panel_label}_panel_contract")
        panel_hash = getattr(args, f"{panel_label}_panel_contract_sha256")
        hashes[f"{panel_label}_panel_contract"] = validate_hash(
            panel_path,
            panel_hash,
            f"{panel_label} panel contract",
        )
        panel = read_json(panel_path)
        rows = [dict(row) for row in panel.get("rows", [])]
        if len(rows) != EXPECTED_PANEL_CASE_COUNT:
            raise ValueError(f"{panel_label} panel row-count drift")
        if panel.get("speaker_split_before_simulation") is not True:
            raise ValueError(f"{panel_label} split/simulation contract drift")
        for row in rows:
            row["opened_panel"] = panel_label
            validate_hash(Path(row["base_path"]), row["base_sha256"], "base")
        panel_rows.extend(rows)
        speakers.update(str(row["speaker_id"]) for row in rows)
    case_ids = [str(row["case_id"]) for row in panel_rows]
    if len(panel_rows) != EXPECTED_CASE_COUNT or len(set(case_ids)) != len(case_ids):
        raise ValueError("v19 probe requires 24 unique opened cases")
    if len(speakers) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("v19 probe opened speaker coverage drift")
    if Counter(row["view"] for row in panel_rows) != Counter({"cs": 12, "sv": 12}):
        raise ValueError("v19 probe CS/SV coverage drift")

    artifact_paths = {
        name: getattr(args, f"v18_{name}")
        for name in ("report", "preselection", "receipt")
    }
    expected_artifact_paths = {
        "report": args.v18_run_root / "outputs" / "diagnostic_report.json",
        "preselection": (
            args.v18_run_root
            / "outputs"
            / "family_selector_preselection.csv"
        ),
        "receipt": args.v18_run_root / "outputs" / "completion_receipt.json",
    }
    for name, path in artifact_paths.items():
        if path.resolve() != expected_artifact_paths[name].resolve():
            raise ValueError(f"v18 {name} path is outside the immutable run root")
    for name, path in artifact_paths.items():
        hashes[f"v18_{name}"] = validate_hash(
            path,
            getattr(args, f"v18_{name}_sha256"),
            f"v18 {name}",
        )
    report = read_json(artifact_paths["report"])
    receipt = read_json(artifact_paths["receipt"])
    if report.get("source_commit") != V18_SOURCE_COMMIT:
        raise ValueError("v18 report source commit drift")
    if receipt.get("source_commit") != V18_SOURCE_COMMIT:
        raise ValueError("v18 receipt source commit drift")
    if report.get("decision") != V18_NO_GO_DECISION:
        raise ValueError("v19 probe is not bound to the v18 runtime-only NO-GO")
    if report.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("v18 candidate exact outcomes were unexpectedly opened")
    if report.get("exact_component_scoring_requested") is not False:
        raise ValueError("v18 exact component scoring was unexpectedly requested")
    if report.get("new_sealed_panel_authorized") is not False:
        raise ValueError("v18 unexpectedly authorized a new sealed panel")
    if receipt.get("decision") != V18_NO_GO_DECISION:
        raise ValueError("v18 report/receipt decision drift")
    if receipt.get("slurm_job_id") != "19943414":
        raise ValueError("v18 immutable job binding drift")
    if receipt.get("generator_optimizer_steps") != 0:
        raise ValueError("v18 generator-training boundary drift")
    expected_v18_receipt_bindings = {
        "diagnostic_report.json": hashes["v18_report"],
        "family_selector_preselection.csv": hashes["v18_preselection"],
    }
    if receipt.get("artifact_sha256") != expected_v18_receipt_bindings:
        raise ValueError("v18 receipt artifact binding drift")
    for forbidden in ("selector_seal.json", "family_selector_results.csv"):
        if (args.v18_run_root / "outputs" / forbidden).exists():
            raise ValueError(f"v18 fail-closed artifact unexpectedly exists: {forbidden}")

    selected_rows = [
        row
        for row in read_csv(artifact_paths["preselection"])
        if row.get("selected_attempt") == "True"
    ]
    selected_by_case = {row["case_id"]: row for row in selected_rows}
    if set(selected_by_case) != set(case_ids) or len(selected_rows) != len(case_ids):
        raise ValueError("v18 selected-candidate coverage drift")
    for row in selected_by_case.values():
        validate_hash(
            Path(row["candidate_path"]),
            row["candidate_sha256"],
            "v18 selected candidate",
        )
    failures = [str(value) for value in report.get("selector_failures", [])]
    if not failures or not set(failures).issubset(selected_by_case):
        raise ValueError("v18 selector-failure coverage drift")
    return panel_rows, selected_by_case, hashes, failures


def main() -> None:
    args = parse_args()
    source_provenance = validate_repository_provenance(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.repeats < DEFAULT_REPEATS:
        raise ValueError("v19 peak certificate requires at least three repeats")
    panel_rows, selected_by_case, source_hashes, v18_failures = validate_inputs(args)
    args.output_dir.mkdir(parents=True)
    direct_root = args.output_dir / "pcm24_direct_reference"
    copied_root = args.output_dir / "pcm24_copied_from_tmpfs"
    direct_root.mkdir()
    copied_root.mkdir()

    equivalence_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    pcm24_rows: list[dict[str, Any]] = []
    impulse_cache: dict[int, dict[str, Any]] = {}
    engine = PeakCertificateProbeEngine()
    tmpfs_parent = Path("/dev/shm")
    if not tmpfs_parent.is_dir():
        raise FileNotFoundError("v19 probe requires node-local /dev/shm")
    try:
        synthetic_rows = synthetic_controls(engine, impulse_cache)
        with tempfile.TemporaryDirectory(
            prefix="avqi-shimmer-v19-pcm24-",
            dir=tmpfs_parent,
        ) as tmpfs_directory:
            tmpfs_root = Path(tmpfs_directory)
            for case_index, panel_row in enumerate(panel_rows, start=1):
                case_id = str(panel_row["case_id"])
                selected = selected_by_case[case_id]
                _, base_values, base_hash = load_waveform(Path(panel_row["base_path"]))
                candidate_float32, candidate_values, candidate_hash = load_waveform(
                    Path(selected["candidate_path"])
                )
                row, forced_candidate, _ = evaluate_pair(
                    engine,
                    case_id,
                    str(panel_row["view"]),
                    base_values,
                    base_hash,
                    candidate_values,
                    candidate_hash,
                    impulse_cache,
                )
                row.update(
                    {
                        "opened_panel": panel_row["opened_panel"],
                        "speaker_id": panel_row["speaker_id"],
                        "condition": panel_row["condition"],
                        "sample_group": panel_row["sample_group"],
                        "v18_runtime_failure_case": case_id in v18_failures,
                        "selected_family": selected["selected_family"],
                        "selected_alpha": selected["selected_alpha"],
                        "candidate_sha256": selected["candidate_sha256"],
                    }
                )
                equivalence_rows.append(row)

                for repeat_index in range(1, args.repeats + 1):
                    pair_started = time.perf_counter()
                    base_topology = refresh_probe_waveform(
                        engine,
                        base_values,
                        str(panel_row["view"]),
                        base_hash,
                        NUMPY_HIGHPASS_MODE,
                    )
                    certificate_started = time.perf_counter()
                    certificate = paired_candidate_peak_certificate(
                        pcm16_roundtrip_values_to_codes(
                            pcm16_roundtrip(base_values)
                        ),
                        pcm16_roundtrip_values_to_codes(
                            pcm16_roundtrip(candidate_values)
                        ),
                        base_topology["timing_ms"],
                        impulse_certificate_for_waveform(
                            engine,
                            base_values,
                            impulse_cache,
                        ),
                    )
                    certificate_ms = 1000.0 * (
                        time.perf_counter() - certificate_started
                    )
                    certified_topology = engine.refresh_certified(
                        candidate_values,
                        str(panel_row["view"]),
                        candidate_hash,
                        certificate,
                    )
                    require_equivalent(
                        forced_candidate,
                        certified_topology,
                        f"{case_id}:runtime-repeat={repeat_index}",
                    )
                    pair_ms = 1000.0 * (time.perf_counter() - pair_started)
                    runtime_rows.append(
                        {
                            "case_id": case_id,
                            "opened_panel": panel_row["opened_panel"],
                            "view": panel_row["view"],
                            "condition": panel_row["condition"],
                            "repeat_index": repeat_index,
                            "v18_runtime_failure_case": case_id in v18_failures,
                            "certificate_skip": certificate[
                                "candidate_sinc70_search_may_be_skipped"
                            ],
                            "base_refresh_ms": base_topology["pulse_runtime_ms"],
                            "certificate_compute_ms": certificate_ms,
                            "candidate_refresh_ms": certified_topology[
                                "pulse_runtime_ms"
                            ],
                            "paired_refresh_wall_ms": pair_ms,
                            "paired_refresh_formal_500ms_pass": (
                                pair_ms <= FORMAL_RUNTIME_GATE_MS
                            ),
                            "paired_refresh_engineering_450ms_pass": (
                                pair_ms <= ENGINEERING_MARGIN_MS
                            ),
                        }
                    )
                pcm24_rows.append(
                    pcm24_tmpfs_equivalence(
                        case_id,
                        candidate_float32,
                        Path(selected["candidate_path"]),
                        selected["candidate_sha256"],
                        direct_root,
                        copied_root,
                        tmpfs_root,
                    )
                )
                print(
                    f"v19_peak_certificate={case_index}/{EXPECTED_CASE_COUNT}",
                    flush=True,
                )
    finally:
        engine.close()

    equivalence_path = args.output_dir / "peak_certificate_equivalence.csv"
    runtime_path = args.output_dir / "paired_runtime_repeats.csv"
    pcm24_path = args.output_dir / "pcm24_tmpfs_equivalence.csv"
    write_csv(equivalence_path, equivalence_rows)
    write_csv(runtime_path, runtime_rows)
    write_csv(pcm24_path, pcm24_rows)

    all_equivalent = all(
        row["post_highpass_pcm16_equal"]
        and row["metric_pcm16_equal"]
        and row["source_mapping_equal"]
        and row["pulse_positions_equal"]
        and row["bound_covers_observed_exact_peak"]
        and row["forced_exact_peak_scaled"] == row["certified_peak_scaled"]
        for row in equivalence_rows
    )
    synthetic_pass = all(
        row["certificate_skip_matches_expected"]
        and row["exact_scale_matches_expected"]
        for row in synthetic_rows
    )
    tmpfs_pass = all(row["all_pcm24_bytes_equal"] for row in pcm24_rows)
    failure_rows = [
        row for row in equivalence_rows if row["v18_runtime_failure_case"]
    ]
    failure_certificate_coverage = all(
        row["candidate_sinc70_search_may_be_skipped"] for row in failure_rows
    )
    paired_runtime_max = max(
        float(row["paired_refresh_wall_ms"]) for row in runtime_rows
    )
    paired_runtime_formal_pass = all(
        row["paired_refresh_formal_500ms_pass"] for row in runtime_rows
    )
    paired_runtime_margin_pass = all(
        row["paired_refresh_engineering_450ms_pass"] for row in runtime_rows
    )
    passed = (
        all_equivalent
        and synthetic_pass
        and tmpfs_pass
        and failure_certificate_coverage
        and paired_runtime_formal_pass
        and paired_runtime_margin_pass
    )
    decision = PASS_DECISION if passed else FAIL_DECISION
    report = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v19-peak-certificate-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "phase": "opened24_runtime_only_equivalence_probe",
        "candidate_exact_avqi_components_opened": False,
        "v18_immutable_job_id": "19943414",
        "v18_artifacts_mutated": False,
        "scientific_gates_changed": False,
        "formal_runtime_gate_ms": FORMAL_RUNTIME_GATE_MS,
        "engineering_margin_ms": ENGINEERING_MARGIN_MS,
        "metric_highpass_only": True,
        "emitted_waveform_full_band": True,
        "waveform_dependent_cache": False,
        "source_sha256": source_hashes,
        "source_provenance": source_provenance,
        "implementation_sha256": source_provenance["implementation_sha256"],
        "case_count": len(equivalence_rows),
        "speaker_count": len({row["speaker_id"] for row in equivalence_rows}),
        "v18_runtime_failure_cases": v18_failures,
        "certificate": {
            "formula": (
                "base_peak_upper_bound + 5.2 * "
                "(stop_hann_impulse_l1_upper_bound * max_abs_pcm16_delta + "
                "two_independent_filter_fft_roundoff_epsilon)"
            ),
            "input_contract": "exact_worker_pcm16_roundtrip_int16_codes",
            "irfft_construction_error_term_recorded": True,
            "paired_fft_roundoff_transform_count": 2,
            "impulse_cache_keys": sorted(
                certificate["response_cache_key"]
                for certificate in impulse_cache.values()
            ),
            "cache_keyed_only_by_fft_length": True,
            "all_bounds_cover_forced_exact_candidate_peak": all(
                row["bound_covers_observed_exact_peak"]
                for row in equivalence_rows
            ),
            "certified_skip_count": sum(
                bool(row["candidate_sinc70_search_may_be_skipped"])
                for row in equivalence_rows
            ),
            "v18_failure_certificate_coverage": failure_certificate_coverage,
        },
        "equivalence": {
            "all_24_post_highpass_pcm16_equal": all_equivalent,
            "all_24_metric_pcm16_equal": all_equivalent,
            "all_24_source_mappings_equal": all_equivalent,
            "all_24_pulse_arrays_equal": all_equivalent,
            "all_24_scale_decisions_equal": all_equivalent,
            "synthetic_safe_exact_scale_controls_pass": synthetic_pass,
        },
        "paired_runtime": {
            "repeat_count_per_case": args.repeats,
            "measurement_count": len(runtime_rows),
            "median_ms": median(
                float(row["paired_refresh_wall_ms"]) for row in runtime_rows
            ),
            "maximum_ms": paired_runtime_max,
            "formal_500ms_pass": paired_runtime_formal_pass,
            "engineering_450ms_margin_pass": paired_runtime_margin_pass,
            "topology_refresh_only_not_full_selector_step": True,
            "may_authorize_only_full_step_integration_probe": True,
            "cannot_authorize_opened24_rerun": True,
        },
        "pcm24_tmpfs": {
            "all_24_direct_tmpfs_and_durable_copy_bytes_equal": tmpfs_pass,
            "candidate_materialization_inside_timed_step": "node_local_dev_shm",
            "selected_durable_copy_after_timed_step": True,
        },
        "v19_integration_probe_authorized": passed,
        "opened24_rerun_authorized": False,
        "new_sealed_panel_authorized": False,
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "synthetic_controls": synthetic_rows,
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v19-peak-certificate-receipt-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "implementation_sha256": source_provenance["implementation_sha256"],
        "v18_immutable_job_id": "19943414",
        "candidate_exact_avqi_components_opened": False,
        "v19_integration_probe_authorized": passed,
        "opened24_rerun_authorized": False,
        "new_sealed_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (
                report_path,
                equivalence_path,
                runtime_path,
                pcm24_path,
            )
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(
        json.dumps(
            {
                "decision": decision,
                "certified_skip_count": report["certificate"][
                    "certified_skip_count"
                ],
                "v18_failure_certificate_coverage": (
                    failure_certificate_coverage
                ),
                "paired_runtime_max_ms": paired_runtime_max,
                "tmpfs_equivalence": tmpfs_pass,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
