"""Fail-closed evidence and current-output topology contracts for Route C v19."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import torch


ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION = "avqi-route-c-v19-evidence-manifest-v1"
ROUTE_C_V19_REPORT_SCHEMA_VERSION = (
    "avqi-route-c-shimmer-db-runtime-v19-full-step-integration-v1"
)
ROUTE_C_V19_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-shimmer-db-runtime-v19-full-step-integration-receipt-v1"
)
ROUTE_C_V19_PASS_DECISION = (
    "PASS_SHIMMER_DB_RUNTIME_V19_FULL_STEP_INTEGRATION"
)
ROUTE_C_V19_FULL_STEP_GATE_KEYS = (
    "all_attempts_equal_frozen_v18",
    "complete_24case_three_repeat_coverage",
    "all_full_steps_within_frozen_500ms",
    "all_full_steps_within_450ms_engineering_margin",
    "all_selectors_pass",
    "selected_pcm24_durable_byte_equivalence",
    "full_step_timer_envelope_and_phase_accounting",
)
ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS = (
    "peak_certificate_helper",
    "phase1_evaluator",
    "frozen_worker",
    "v19_worker",
    "v19_runtime_client",
    "integration_evaluator",
    "integration_runner",
)
ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS = (
    "report",
    "attempts_csv",
    "runtime_csv",
    "durable_csv",
    "receipt",
)
ROUTE_C_V19_CURRENT_OUTPUT_ROLES = (
    "current_output_topology",
    "current_s3_500_output_topology",
)
ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION = (
    "exact_paired_peak_certificate_tmpfs_v19"
)
ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE = (
    "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
)
ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER = (
    "client_tmpfs_raw_float32_current_output"
)
ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE = (
    "numpy_stop_hann_paired_peak_certificate_v19"
)
ROUTE_C_V19_PAIRED_CANDIDATE_INPUT_LOADER = (
    "client_tmpfs_raw_float32_current_output_paired_v19"
)
ROUTE_C_V19_SAMPLE_RATE = 16_000
ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS = (
    "source_sample_count",
    "metric_sample_count",
    "metric_constant_prefix_samples",
    "metric_source_range_count",
    "metric_mapped_sample_count",
    "metric_reconstruction_max_pcm16_error",
    "metric_reconstruction_differing_samples",
    "pulse_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteCArtifactBinding:
    """One immutable file path and its expected SHA-256 digest."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class RouteCV19EvidenceManifest:
    """Hash-bound v19 full-step evidence; never implies scientific promotion."""

    schema_version: str
    source_commit: str
    slurm_job_id: str
    decision: str
    implementation_artifacts: Mapping[str, RouteCArtifactBinding]
    full_step_artifacts: Mapping[str, RouteCArtifactBinding]
    candidate_exact_avqi_components_opened: bool
    exact_component_scoring_requested: bool
    opened24_rerun_authorized: bool
    promotion_authorized: bool
    generator_optimizer_steps: int


@dataclass(frozen=True)
class ValidatedRouteCV19Topology:
    """Detached topology values proven to belong to the current waveform."""

    case_id: str
    view: str
    source_waveform_float32_sha256: str
    topology_sha256: str
    pulse_positions_samples: tuple[float, ...]
    metric_source_indices: tuple[int, ...]
    metric_constant_prefix_samples: int


def _is_lower_hex(value: str, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sha256(value: str, label: str) -> None:
    if not _is_lower_hex(value, 64) or value == "0" * 64:
        raise ValueError(f"{label} is not a bound SHA-256 digest")


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _validate_artifact_group(
    bindings: Mapping[str, RouteCArtifactBinding],
    expected_keys: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if set(bindings) != set(expected_keys):
        raise ValueError(f"Route C {label} evidence keys differ")
    observed: dict[str, str] = {}
    for key in expected_keys:
        binding = bindings[key]
        if not isinstance(binding, RouteCArtifactBinding):
            raise ValueError(f"Route C {label} {key} binding has wrong type")
        path = Path(binding.path)
        if not path.is_file():
            raise ValueError(f"Route C {label} {key} file is missing: {path}")
        _validate_sha256(binding.sha256, f"Route C {label} {key}")
        digest = sha256_file(path)
        if digest != binding.sha256:
            raise ValueError(f"Route C {label} {key} hash mismatch: {path}")
        observed[key] = digest
    return observed


def validate_v19_evidence_manifest(
    manifest: RouteCV19EvidenceManifest,
) -> dict[str, Any]:
    """Validate a completed full-step receipt while keeping promotion closed."""
    if not isinstance(manifest, RouteCV19EvidenceManifest):
        raise ValueError("Route C v19 evidence manifest has wrong type")
    if manifest.schema_version != ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Route C v19 evidence schema differs")
    if not _is_lower_hex(manifest.source_commit, 40) or (
        manifest.source_commit == "0" * 40
    ):
        raise ValueError("Route C v19 source commit is not a full Git SHA")
    if (
        not isinstance(manifest.slurm_job_id, str)
        or not manifest.slurm_job_id.isdigit()
        or int(manifest.slurm_job_id) <= 0
    ):
        raise ValueError("Route C v19 Slurm job id is invalid")
    if manifest.decision != ROUTE_C_V19_PASS_DECISION:
        raise ValueError("Route C v19 decision is not the frozen PASS decision")
    if manifest.candidate_exact_avqi_components_opened is not False:
        raise ValueError("Route C v19 evidence opened exact component outcomes")
    if manifest.exact_component_scoring_requested is not False:
        raise ValueError("Route C v19 evidence requested exact component scoring")
    if manifest.opened24_rerun_authorized is not True:
        raise ValueError("Route C v19 evidence did not authorize opened24")
    if manifest.promotion_authorized is not False:
        raise ValueError("Route C v19 full-step evidence overclaims promotion")
    if manifest.generator_optimizer_steps != 0:
        raise ValueError("Route C v19 evidence contains optimizer steps")

    implementation_hashes = _validate_artifact_group(
        manifest.implementation_artifacts,
        ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
        "v19 implementation",
    )
    full_step_hashes = _validate_artifact_group(
        manifest.full_step_artifacts,
        ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
        "v19 full-step",
    )
    report = _read_json_mapping(
        Path(manifest.full_step_artifacts["report"].path),
        "Route C v19 full-step report",
    )
    receipt = _read_json_mapping(
        Path(manifest.full_step_artifacts["receipt"].path),
        "Route C v19 full-step receipt",
    )
    if report.get("schema_version") != ROUTE_C_V19_REPORT_SCHEMA_VERSION:
        raise ValueError("Route C v19 full-step report schema differs")
    if receipt.get("schema_version") != ROUTE_C_V19_RECEIPT_SCHEMA_VERSION:
        raise ValueError("Route C v19 full-step receipt schema differs")
    for value, label in ((report, "report"), (receipt, "receipt")):
        if value.get("decision") != manifest.decision:
            raise ValueError(f"Route C v19 {label} decision differs")
        if value.get("source_commit") != manifest.source_commit:
            raise ValueError(f"Route C v19 {label} source commit differs")
        if str(value.get("slurm_job_id")) != manifest.slurm_job_id:
            raise ValueError(f"Route C v19 {label} Slurm job differs")
        if value.get("candidate_exact_avqi_components_opened") is not False:
            raise ValueError(f"Route C v19 {label} opened exact outcomes")
        if value.get("opened24_rerun_authorized") is not True:
            raise ValueError(f"Route C v19 {label} did not authorize opened24")
        if value.get("promotion_authorized") is not False:
            raise ValueError(f"Route C v19 {label} overclaims promotion")
        if value.get("new_sealed_panel_authorized") is not False:
            raise ValueError(f"Route C v19 {label} authorized a sealed panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"Route C v19 {label} contains optimizer steps")
        if value.get("authoritative_training_decision") != (
            "NO_GO_AVQI_T2_TRAINING"
        ):
            raise ValueError(f"Route C v19 {label} training boundary differs")
    if report.get("exact_component_scoring_requested") is not False:
        raise ValueError("Route C v19 report requested exact scoring")
    if report.get("generator_loaded") is not False:
        raise ValueError("Route C v19 report loaded a generator")
    if report.get("generator_optimizer_created") is not False:
        raise ValueError("Route C v19 report created a generator optimizer")
    report_contract = {
        "phase": "opened24_full_selector_step_integration_only",
        "dev_only": True,
        "repeat_count_per_case": 3,
        "case_count": 24,
        "speaker_count": 12,
        "scientific_gates_changed": False,
        "v18_artifacts_mutated": False,
        "v19_topology_artifacts_mutated": False,
        "new_sealed_panel_authorized": False,
    }
    for field, expected in report_contract.items():
        observed = report.get(field)
        if observed != expected or type(observed) is not type(expected):
            raise ValueError(f"Route C v19 report {field} contract differs")
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(
        ROUTE_C_V19_FULL_STEP_GATE_KEYS
    ):
        raise ValueError("Route C v19 report full-step gate keys differ")
    if any(gates[key] is not True for key in ROUTE_C_V19_FULL_STEP_GATE_KEYS):
        raise ValueError("Route C v19 report full-step gate did not pass")

    report_sources = report.get("source_sha256")
    if not isinstance(report_sources, dict):
        raise ValueError("Route C v19 report source hashes are unavailable")
    receipt_provenance = receipt.get("source_provenance")
    if not isinstance(receipt_provenance, dict):
        raise ValueError("Route C v19 receipt source provenance is unavailable")
    receipt_sources = receipt_provenance.get("implementation_sha256")
    if not isinstance(receipt_sources, dict):
        raise ValueError("Route C v19 receipt implementation hashes are unavailable")
    if set(receipt_sources) != set(ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS):
        raise ValueError("Route C v19 receipt implementation keys differ")
    if receipt_provenance.get("repository_head") != manifest.source_commit:
        raise ValueError("Route C v19 receipt repository HEAD differs")
    if receipt_provenance.get("repository_tree_clean") is not True:
        raise ValueError("Route C v19 receipt repository was not clean")
    if report.get("source_provenance") != receipt_provenance:
        raise ValueError("Route C v19 report/receipt provenance differs")
    for key, digest in implementation_hashes.items():
        if report_sources.get(key) != digest or receipt_sources.get(key) != digest:
            raise ValueError(f"Route C v19 {key} implementation binding differs")

    expected_receipt_artifacts = {
        Path(manifest.full_step_artifacts[key].path).name: full_step_hashes[key]
        for key in ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS
        if key != "receipt"
    }
    if receipt.get("artifact_sha256") != expected_receipt_artifacts:
        raise ValueError("Route C v19 receipt artifact binding differs")
    return {
        "schema_version": manifest.schema_version,
        "source_commit": manifest.source_commit,
        "slurm_job_id": manifest.slurm_job_id,
        "decision": manifest.decision,
        "implementation_sha256": implementation_hashes,
        "full_step_sha256": full_step_hashes,
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "opened24_rerun_authorized": True,
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "scientific_promotion_granted": False,
    }


def _waveform_float32_sha256(waveform: torch.Tensor) -> str:
    values = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = values.numpy().astype("<f4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _source_ranges_sha256(ranges: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps(
        [list(value) for value in ranges],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pulse_positions_sha256(pulses: tuple[float, ...]) -> str:
    payload = struct.pack(f"<{len(pulses)}d", *pulses)
    return hashlib.sha256(payload).hexdigest()


def _v19_topology_sha256(
    topology: Mapping[str, Any],
    ranges: tuple[tuple[int, int], ...],
    pulses: tuple[float, ...],
) -> str:
    scalar_payload = {
        field: int(topology[field])
        for field in ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            scalar_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            [list(value) for value in ranges],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(struct.pack(f"<{len(pulses)}d", *pulses))
    return digest.hexdigest()


def _topology_integer(topology: Mapping[str, Any], field: str) -> int:
    value = topology[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Route C v19 topology {field} is not an integer")
    return value


def validate_v19_exact_topology(
    waveform: torch.Tensor,
    topology: Mapping[str, Any],
    *,
    case_id: str,
    view: str,
    expected_topology_sha256: str,
    sample_rate: int,
    expected_implementation: str = ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
) -> ValidatedRouteCV19Topology:
    """Bind detached exact topology to one current output waveform and view."""
    required_fields = {
        "case_id",
        "view",
        "role",
        "scoring_status",
        "topology_preprocessing",
        "implementation",
        "metric_highpass",
        "topology_input_loader",
        "source_waveform_float32_sha256",
        "source_ranges_sha256",
        "pulse_positions_sha256",
        "metric_source_ranges",
        "pulse_positions_samples",
        *ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS,
    }
    missing = required_fields - set(topology)
    if missing:
        raise ValueError(f"Route C v19 topology fields are missing: {sorted(missing)}")
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Route C v19 topology expects one non-empty waveform")
    if not torch.isfinite(waveform).all():
        raise ValueError("Route C v19 topology waveform is non-finite")
    if not case_id or topology["case_id"] != case_id:
        raise ValueError("Route C v19 topology case binding differs")
    if view not in {"cs", "sv"} or topology["view"] != view:
        raise ValueError("Route C v19 topology view binding differs")
    if topology["role"] not in ROUTE_C_V19_CURRENT_OUTPUT_ROLES:
        raise ValueError("Route C v19 topology is not from a current output")
    if topology["scoring_status"] != "ok":
        raise ValueError("Route C v19 topology is unavailable")
    if topology["topology_preprocessing"] != "exact_avqi_view_metric_waveform":
        raise ValueError("Route C v19 topology preprocessing differs")
    if topology["implementation"] != expected_implementation:
        raise ValueError("Route C v19 topology implementation differs")
    if topology["metric_highpass"] != ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE:
        raise ValueError("Route C v19 base topology high-pass mode differs")
    if (
        topology["topology_input_loader"]
        != ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER
    ):
        raise ValueError("Route C v19 base topology input loader differs")
    if sample_rate != ROUTE_C_V19_SAMPLE_RATE:
        raise ValueError("Route C v19 topology requires a 16000 Hz sample rate")

    source_sample_count = _topology_integer(topology, "source_sample_count")
    if source_sample_count != waveform.numel():
        raise ValueError("Route C v19 topology source sample count differs")
    waveform_sha256 = _waveform_float32_sha256(waveform)
    if topology["source_waveform_float32_sha256"] != waveform_sha256:
        raise ValueError("Route C v19 topology is not bound to the current waveform")

    raw_ranges = topology["metric_source_ranges"]
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError("Route C v19 topology source ranges are unavailable")
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    for raw_range in raw_ranges:
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError("Route C v19 topology source range has wrong shape")
        start, length = raw_range
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_range
        ):
            raise ValueError("Route C v19 topology source range is not integral")
        end = start + length
        if length <= 0 or start < previous_end or end > source_sample_count:
            raise ValueError("Route C v19 topology source range is invalid")
        ranges.append((start, length))
        previous_end = end
    frozen_ranges = tuple(ranges)
    if _topology_integer(topology, "metric_source_range_count") != len(ranges):
        raise ValueError("Route C v19 topology source range count differs")
    mapped_sample_count = sum(length for _, length in ranges)
    if _topology_integer(
        topology,
        "metric_mapped_sample_count",
    ) != mapped_sample_count:
        raise ValueError("Route C v19 topology mapped sample count differs")
    prefix = _topology_integer(topology, "metric_constant_prefix_samples")
    if prefix < 0:
        raise ValueError("Route C v19 topology constant prefix is negative")
    metric_sample_count = _topology_integer(topology, "metric_sample_count")
    if metric_sample_count != prefix + mapped_sample_count:
        raise ValueError("Route C v19 topology metric sample count differs")
    if (
        _topology_integer(topology, "metric_reconstruction_max_pcm16_error") != 0
        or _topology_integer(topology, "metric_reconstruction_differing_samples")
        != 0
    ):
        raise ValueError("Route C v19 topology source mapping lacks PCM16 parity")
    if view == "sv":
        expected_length = min(source_sample_count, 3 * sample_rate)
        expected_range = ((source_sample_count - expected_length, expected_length),)
        if prefix != 0 or frozen_ranges != expected_range:
            raise ValueError("Route C v19 SV topology does not use the final crop")

    raw_pulses = topology["pulse_positions_samples"]
    if not isinstance(raw_pulses, list):
        raise ValueError("Route C v19 topology pulse positions are unavailable")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_pulses
    ):
        raise ValueError("Route C v19 topology pulse positions are not numeric")
    pulses = tuple(float(value) for value in raw_pulses)
    if len(pulses) < 3 or _topology_integer(topology, "pulse_count") != len(pulses):
        raise ValueError("Route C v19 topology pulse count is insufficient")
    if any(not math.isfinite(value) for value in pulses):
        raise ValueError("Route C v19 topology pulse positions are non-finite")
    if any(right <= left for left, right in zip(pulses, pulses[1:])):
        raise ValueError("Route C v19 topology pulse positions are not increasing")
    if pulses[0] < 0.0 or pulses[-1] >= metric_sample_count:
        raise ValueError("Route C v19 topology pulse positions exceed metric bounds")

    source_ranges_sha256 = _source_ranges_sha256(frozen_ranges)
    pulse_positions_sha256 = _pulse_positions_sha256(pulses)
    if topology["source_ranges_sha256"] != source_ranges_sha256:
        raise ValueError("Route C v19 topology source-range hash differs")
    if topology["pulse_positions_sha256"] != pulse_positions_sha256:
        raise ValueError("Route C v19 topology pulse-position hash differs")
    _validate_sha256(expected_topology_sha256, "Route C v19 topology")
    observed_topology_sha256 = _v19_topology_sha256(
        topology,
        frozen_ranges,
        pulses,
    )
    if observed_topology_sha256 != expected_topology_sha256:
        raise ValueError("Route C v19 composite topology hash differs")

    metric_source_indices = tuple(
        index
        for start, length in frozen_ranges
        for index in range(start, start + length)
    )
    return ValidatedRouteCV19Topology(
        case_id=case_id,
        view=view,
        source_waveform_float32_sha256=waveform_sha256,
        topology_sha256=observed_topology_sha256,
        pulse_positions_samples=pulses,
        metric_source_indices=metric_source_indices,
        metric_constant_prefix_samples=prefix,
    )
