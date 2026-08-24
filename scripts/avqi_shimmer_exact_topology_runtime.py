#!/usr/bin/env python3
"""Client contract for the persistent exact-Praat Shimmer topology worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


READY_MARKER = "AVQI_SHIMMER_TOPOLOGY_READY="
RESULT_MARKER = "AVQI_SHIMMER_TOPOLOGY_RESULT="
EXPECTED_IMPLEMENTATION = (
    "exact_vectorized_frames_reused_tmpfs_numpy_sounding_v15"
)
ALLOWED_CURRENT_OUTPUT_ROLES = frozenset(
    {
        "current_output_topology",
        "current_s3_500_output_topology",
    }
)
TOPOLOGY_SCALAR_FIELDS = (
    "source_sample_count",
    "metric_sample_count",
    "metric_constant_prefix_samples",
    "metric_source_range_count",
    "metric_mapped_sample_count",
    "metric_reconstruction_max_pcm16_error",
    "metric_reconstruction_differing_samples",
    "pulse_count",
)


def sha256_ranges(ranges: list[list[int]]) -> str:
    encoded = json.dumps(ranges, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_pulses(positions: list[float] | np.ndarray) -> str:
    values = np.asarray(positions, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def topology_sha256(topology: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    scalar_payload = {
        field: int(topology[field]) for field in TOPOLOGY_SCALAR_FIELDS
    }
    digest.update(
        json.dumps(
            scalar_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            topology["metric_source_ranges"],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        np.asarray(
            topology["pulse_positions_samples"],
            dtype="<f8",
        ).tobytes()
    )
    return digest.hexdigest()


def require_exact_topology_equal(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
) -> str:
    for row_name, row in (("reference", reference), ("candidate", candidate)):
        if row.get("scoring_status") != "ok":
            raise ValueError(f"{label}: {row_name} topology is unavailable")
        if row.get("topology_preprocessing") != "exact_avqi_view_metric_waveform":
            raise ValueError(f"{label}: {row_name} preprocessing contract drift")
        if int(row["metric_reconstruction_max_pcm16_error"]) != 0 or int(
            row["metric_reconstruction_differing_samples"]
        ) != 0:
            raise ValueError(f"{label}: {row_name} source mapping lacks parity")

    scalar_differences = {
        field: {
            "reference": reference[field],
            "candidate": candidate[field],
        }
        for field in TOPOLOGY_SCALAR_FIELDS
        if int(reference[field]) != int(candidate[field])
    }
    ranges_equal = reference["metric_source_ranges"] == candidate[
        "metric_source_ranges"
    ]
    pulses_equal = np.array_equal(
        np.asarray(reference["pulse_positions_samples"], dtype=np.float64),
        np.asarray(candidate["pulse_positions_samples"], dtype=np.float64),
    )
    if scalar_differences or not ranges_equal or not pulses_equal:
        raise ValueError(
            f"{label}: exact topology drift: "
            + json.dumps(
                {
                    "scalar_differences": scalar_differences,
                    "source_ranges_equal": ranges_equal,
                    "pulse_positions_equal": pulses_equal,
                },
                sort_keys=True,
            )
        )

    reference_hash = topology_sha256(reference)
    candidate_hash = topology_sha256(candidate)
    if reference_hash != candidate_hash:
        raise ValueError(f"{label}: composite topology hash drift")
    if "source_ranges_sha256" in candidate and candidate[
        "source_ranges_sha256"
    ] != sha256_ranges(candidate["metric_source_ranges"]):
        raise ValueError(f"{label}: source-range hash self-check failed")
    if "pulse_positions_sha256" in candidate and candidate[
        "pulse_positions_sha256"
    ] != sha256_pulses(candidate["pulse_positions_samples"]):
        raise ValueError(f"{label}: pulse hash self-check failed")
    return reference_hash


class ExactShimmerTopologyWorker:
    """One persistent worker with no waveform-dependent client-side cache."""

    def __init__(
        self,
        exact_python: Path,
        worker_script: Path,
        avqi_code_root: Path,
        avqi_code_tree_sha256: str,
    ) -> None:
        for label, path in (
            ("exact Python", exact_python),
            ("topology worker", worker_script),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing {label}: {path}")
        if not avqi_code_root.is_dir():
            raise FileNotFoundError(
                f"missing exact AVQI code root: {avqi_code_root}"
            )
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
        self.startup = self._read_marker(READY_MARKER)
        self.startup_ms = 1000.0 * (time.perf_counter() - started)
        if self.startup.get("implementation") != EXPECTED_IMPLEMENTATION:
            self.close()
            raise ValueError("exact topology worker implementation drift")
        if (
            self.startup.get("avqi_code_tree_sha256")
            != avqi_code_tree_sha256
        ):
            self.close()
            raise ValueError("exact topology worker code-tree hash drift")

    def _read_marker(self, marker: str) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("exact topology worker stdout is unavailable")
        transcript: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "exact topology worker exited before marker: "
                    + "".join(transcript)[-4000:]
                )
            transcript.append(line)
            if line.startswith(marker):
                value = json.loads(line.split("=", 1)[1])
                if not isinstance(value, dict):
                    raise ValueError("exact topology worker returned non-object JSON")
                return value

    def request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self.process.stdin is None:
            raise RuntimeError("exact topology worker stdin is unavailable")
        if self.process.poll() is not None:
            raise RuntimeError("exact topology worker has already exited")
        started = time.perf_counter()
        self.process.stdin.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()
        response = self._read_marker(RESULT_MARKER)
        wall_ms = 1000.0 * (time.perf_counter() - started)
        if response.get("status") != "ok":
            raise RuntimeError(
                "exact topology worker failed: "
                f"{response.get('error_type')} "
                f"{response.get('error_message')}"
            )
        return response, wall_ms

    def warmup(self) -> tuple[dict[str, Any], float]:
        response, wall_ms = self.request({"op": "warmup"})
        warmup = response.get("warmup")
        if not isinstance(warmup, dict):
            raise ValueError("exact topology worker warmup receipt is missing")
        if warmup.get("synthetic_only") is not True or warmup.get(
            "panel_or_training_waveform_used"
        ) is not False:
            raise ValueError("exact topology worker warmup used non-synthetic data")
        return warmup, wall_ms

    def refresh(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float]:
        if not items:
            raise ValueError("exact topology refresh requires at least one item")
        ids = [str(item.get("id", "")) for item in items]
        if any(not item_id for item_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("exact topology refresh IDs must be unique and nonempty")
        refresh_keys: list[tuple[str, str]] = []
        for item in items:
            role = str(item.get("role", ""))
            path = Path(str(item.get("path", "")))
            view = str(item.get("view", ""))
            if role not in ALLOWED_CURRENT_OUTPUT_ROLES:
                raise ValueError(f"forbidden exact topology role: {role}")
            if any("target_clean" in part.lower() for part in path.parts):
                raise ValueError("clean target topology path is forbidden")
            if view not in {"cs", "sv"}:
                raise ValueError(f"unsupported exact topology view: {view}")
            if not path.is_file():
                raise FileNotFoundError(f"missing current-output waveform: {path}")
            refresh_keys.append((str(path.resolve()), view))
        if len(set(refresh_keys)) != len(refresh_keys):
            raise ValueError("duplicate current-waveform refresh in one request")

        response, wall_ms = self.request({"op": "refresh", "items": items})
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) != len(items):
            raise ValueError("exact topology worker row count drift")
        if [row.get("id") for row in rows] != ids:
            raise ValueError("exact topology worker row order drift")
        for row, item in zip(rows, items, strict=True):
            if row.get("case_id") != item.get("case_id"):
                raise ValueError("exact topology worker case identity drift")
            if row.get("implementation") != EXPECTED_IMPLEMENTATION:
                raise ValueError("exact topology row implementation drift")
            if row.get("scoring_status") != "ok" or int(
                row.get("pulse_count", 0)
            ) < 3:
                raise ValueError(
                    f"exact topology unavailable: {row.get('case_id')}"
                )
        return rows, wall_ms

    def close(self) -> None:
        if self.process.poll() is None:
            self.request({"op": "quit"})
        self.process.wait(timeout=30)

    def __enter__(self) -> "ExactShimmerTopologyWorker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
