"""Collect dev-only six-component Route C waveform-gradient measurements.

This evaluator deliberately has no scientific PASS path.  It reuses the
accepted five-component surrogate calibration/holdout selection, freezes
inverse-gradient weights on calibration rows only, and requires detached exact
base-current-output topology for Candidate-E Shimmer dB.  The raw report stays
PENDING until the independent frozen decision stage accepts it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    AVQI_V0301_COEFFICIENTS,
    AVQI_V0301_EXPANDED_COEFFICIENTS,
    AVQI_V0301_INTERCEPT,
    AVQI_V0301_SCALE,
)
from model.avqi_route_c import (
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    six_active_bidirectional_gap_losses,
)
from model.avqi_route_c_candidate_e import (
    CANDIDATE_E_RUNTIME_CLIENT_SHA256,
    CANDIDATE_E_RUNTIME_CONFIG_SHA256,
    CANDIDATE_E_SELECTOR_SHA256,
    CANDIDATE_E_SOURCE_COMMIT,
    CANDIDATE_E_TOPOLOGY_IMPLEMENTATION,
    CANDIDATE_E_REFERENCE_SHA256,
    CANDIDATE_E_WORKER_SHA256,
    build_cycle_gain_plan,
    candidate_e_proxy,
    project_cycle_gain_gradient_fixed_order,
    validate_candidate_e_base_peak_certificate,
)
from model.avqi_route_c_candidate_e_scorer import (
    ROUTE_C_CANDIDATE_E_REGISTRY_SCHEMA_VERSION,
    ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS,
    ROUTE_C_CANDIDATE_E_SIX_ACTIVE_ARCHITECTURE,
    load_route_c_candidate_e_six_scorer,
    route_c_candidate_e_registry_records,
)
from model.avqi_route_c_v19_contracts import (
    ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
    ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
    ROUTE_C_V19_CURRENT_OUTPUT_ROLES,
    ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
    ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
    ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
    ROUTE_C_V19_SAMPLE_RATE,
    RouteCArtifactBinding,
    RouteCV19EvidenceManifest,
    sha256_file,
)
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    FIVE_COMPONENT_EVIDENCE_KEYS,
    _validate_five_component_evidence,
)
from scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v4 import (
    PROMOTION_RECEIPT_SHA256,
    PROMOTION_REPORT_SHA256,
    UPDATED_LEDGER_SHA256,
    validate_promotion,
    validate_receipt_artifact_files,
)
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    AUDIT_SPLITS,
    SEGMENT_SAMPLES,
    AuditCase,
    cosine,
    load_fixed_segment,
    load_label_bank,
    load_svd_fusion_label_bank,
    verify_source,
    write_json,
)


MEASUREMENT_SCHEMA_VERSION = (
    "dev-avqi-route-c-six-gradient-raw-measurement-v2"
)
TOPOLOGY_INPUT_SCHEMA_VERSION = (
    "dev-avqi-route-c-six-gradient-candidate-e-topology-input-v2"
)
TOPOLOGY_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-gradient-candidate-e-topology-receipt-v2"
)
TOPOLOGY_SEAL_DECISION = (
    "SEALED_ROUTE_C_SIX_GRADIENT_CANDIDATE_E_TOPOLOGIES_V2"
)
MEASUREMENT_DECISION = (
    "PENDING_ROUTE_C_SIX_COMPONENT_GRADIENT_GATES_UNFROZEN"
)
JOINT_PANEL_DECISION = "NO_GO_ROUTE_C_SIX_JOINT_PANEL"
ACCEPTED_SIX_SCAFFOLD_BASE = "f6914366049877181bb9bf5c75a5f9e94d41ffe0"
REQUIRED_FIVE_SOURCE_EVIDENCE = (
    *FIVE_COMPONENT_EVIDENCE_KEYS,
    "five_gradient_report",
    "five_gradient_receipt",
)
CANDIDATE_E_EVIDENCE_KEYS = (
    "candidate_e_promotion_report",
    "candidate_e_promotion_receipt",
    "candidate_e_reference_source",
    "candidate_e_runtime_client",
    "candidate_e_worker",
    "candidate_e_selector_source",
    "candidate_e_runtime_config",
    "exact_code_tree_manifest",
    "exact_runtime_manifest",
    "exact_authority_receipt",
)
PAIRWISE_COMPONENT_KEYS = tuple(
    f"{left}__{right}"
    for left_index, left in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    for right in ROUTE_C_SIX_ACTIVE_COMPONENTS[left_index + 1 :]
)
UNFROZEN_DECISION_FIELDS = (
    "scientific six-gradient report schema",
    "minimum nonzero component-gradient norm",
    "maximum component-gradient norm",
    "maximum joint-gradient norm",
    "pairwise direction-conflict acceptance rule",
    "component-to-joint cosine acceptance rule",
    "maximum weighted-component share",
    "required calibration and holdout coverage",
    "aggregate six-component promotion decision",
)
V19_MANIFEST_FIELDS = {
    "schema_version",
    "source_commit",
    "slurm_job_id",
    "decision",
    "implementation_artifacts",
    "full_step_artifacts",
    "candidate_exact_avqi_components_opened",
    "exact_component_scoring_requested",
    "opened24_rerun_authorized",
    "promotion_authorized",
    "generator_optimizer_steps",
}
TOPOLOGY_MANIFEST_FIELDS = {
    "schema_version",
    "candidate_e_source_commit",
    "candidate_e_evidence_sha256",
    "label_bank_sha256",
    "selection_salt",
    "sample_rate",
    "segment_samples",
    "topology_role",
    "candidate_exact_avqi_components_opened",
    "exact_component_scoring_requested",
    "final_panel_opened",
    "fresh_panel_opened",
    "waveform_generation_performed",
    "generator_optimizer_steps",
    "rows",
}
TOPOLOGY_ROW_FIELDS = {
    "case_id",
    "split",
    "speaker_id",
    "sample_id",
    "sample_group",
    "view",
    "condition",
    "source_waveform_path",
    "source_audio_file_sha256",
    "source_waveform_float32_sha256",
    "source_segment_samples",
    "topology_sha256",
    "topology",
}


@dataclass(frozen=True)
class TopologyAuditInput:
    """One hash-bound detached topology matched to one selected dev case."""

    case_id: str
    topology: Mapping[str, Any]
    topology_sha256: str
    source_waveform_float32_sha256: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--cpps-checkpoint", type=Path, required=True)
    parser.add_argument("--cpps-checkpoint-sha256", required=True)
    parser.add_argument("--hnr-checkpoint", type=Path, required=True)
    parser.add_argument("--hnr-checkpoint-sha256", required=True)
    parser.add_argument("--shimmer-checkpoint", type=Path, required=True)
    parser.add_argument("--shimmer-checkpoint-sha256", required=True)
    parser.add_argument("--slope-checkpoint", type=Path, required=True)
    parser.add_argument("--slope-checkpoint-sha256", required=True)
    parser.add_argument("--tilt-checkpoint", type=Path, required=True)
    parser.add_argument("--tilt-checkpoint-sha256", required=True)
    parser.add_argument(
        "--source-evidence",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument(
        "--candidate-e-evidence",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--topology-manifest-sha256", required=True)
    parser.add_argument("--topology-receipt", type=Path, required=True)
    parser.add_argument("--topology-receipt-sha256", required=True)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument(
        "--selection-mode",
        choices=("legacy_tau", "sealed_external_svd_v2"),
        default="legacy_tau",
    )
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--test-evidence-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--accepted-base-commit",
        default=ACCEPTED_SIX_SCAFFOLD_BASE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def parse_args() -> argparse.Namespace:
    return build_argument_parser().parse_args()


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} has no bound SHA-256")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return resolved


def _artifact_bindings(
    value: Any,
    expected_keys: tuple[str, ...],
    label: str,
) -> dict[str, RouteCArtifactBinding]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError(f"{label} artifact keys differ")
    bindings: dict[str, RouteCArtifactBinding] = {}
    for key in expected_keys:
        item = value[key]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"{label} {key} binding differs")
        path = Path(item["path"])
        digest = item["sha256"]
        resolved = _verified_file(path, digest, f"{label} {key}")
        bindings[key] = RouteCArtifactBinding(resolved, digest)
    return bindings


def load_v19_evidence_manifest(
    path: Path,
    expected_sha256: str,
) -> RouteCV19EvidenceManifest:
    resolved = _verified_file(path, expected_sha256, "v19 evidence manifest")
    value = _read_json_mapping(resolved, "v19 evidence manifest")
    if set(value) != V19_MANIFEST_FIELDS:
        raise ValueError("v19 evidence manifest fields differ")
    if value["schema_version"] != ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("v19 evidence manifest schema differs")
    if not isinstance(value["slurm_job_id"], str):
        raise ValueError("v19 evidence manifest Slurm job ID is not a string")
    return RouteCV19EvidenceManifest(
        schema_version=value["schema_version"],
        source_commit=value["source_commit"],
        slurm_job_id=value["slurm_job_id"],
        decision=value["decision"],
        implementation_artifacts=_artifact_bindings(
            value["implementation_artifacts"],
            ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
            "v19 implementation",
        ),
        full_step_artifacts=_artifact_bindings(
            value["full_step_artifacts"],
            ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
            "v19 full-step",
        ),
        candidate_exact_avqi_components_opened=value[
            "candidate_exact_avqi_components_opened"
        ],
        exact_component_scoring_requested=value[
            "exact_component_scoring_requested"
        ],
        opened24_rerun_authorized=value["opened24_rerun_authorized"],
        promotion_authorized=value["promotion_authorized"],
        generator_optimizer_steps=value["generator_optimizer_steps"],
    )


def validate_five_source_evidence(
    entries: list[list[str]],
) -> dict[str, dict[str, str]]:
    if len(entries) != len(REQUIRED_FIVE_SOURCE_EVIDENCE):
        raise ValueError("five-component source evidence count differs")
    raw = {name: (Path(path), digest) for name, path, digest in entries}
    if set(raw) != set(REQUIRED_FIVE_SOURCE_EVIDENCE):
        raise ValueError("five-component source evidence keys differ")
    paths: dict[str, Path] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for name in REQUIRED_FIVE_SOURCE_EVIDENCE:
        path, digest = raw[name]
        resolved = _verified_file(path, digest, f"source evidence {name}")
        paths[name] = resolved
        artifacts[name] = {"path": str(resolved), "sha256": digest}
    _validate_five_component_evidence(artifacts, paths)
    return artifacts


def validate_candidate_e_evidence(
    entries: list[list[str]],
) -> dict[str, dict[str, str]]:
    if len(entries) != len(CANDIDATE_E_EVIDENCE_KEYS):
        raise ValueError("Candidate-E source evidence count differs")
    raw = {name: (Path(path), digest) for name, path, digest in entries}
    if set(raw) != set(CANDIDATE_E_EVIDENCE_KEYS):
        raise ValueError("Candidate-E source evidence keys differ")
    artifacts = {}
    for name in CANDIDATE_E_EVIDENCE_KEYS:
        path, digest = raw[name]
        resolved = _verified_file(path, digest, f"Candidate-E evidence {name}")
        artifacts[name] = {"path": str(resolved), "sha256": digest}

    expected_hashes = {
        "candidate_e_promotion_report": PROMOTION_REPORT_SHA256,
        "candidate_e_promotion_receipt": PROMOTION_RECEIPT_SHA256,
        "candidate_e_reference_source": CANDIDATE_E_REFERENCE_SHA256,
        "candidate_e_runtime_client": CANDIDATE_E_RUNTIME_CLIENT_SHA256,
        "candidate_e_worker": CANDIDATE_E_WORKER_SHA256,
        "candidate_e_selector_source": CANDIDATE_E_SELECTOR_SHA256,
        "candidate_e_runtime_config": CANDIDATE_E_RUNTIME_CONFIG_SHA256,
    }
    if any(
        artifacts[name]["sha256"] != digest
        for name, digest in expected_hashes.items()
    ):
        raise ValueError("Candidate-E frozen evidence hash differs")
    promotion_report = _read_json_mapping(
        Path(artifacts["candidate_e_promotion_report"]["path"]),
        "Candidate-E promotion report",
    )
    promotion_receipt = _read_json_mapping(
        Path(artifacts["candidate_e_promotion_receipt"]["path"]),
        "Candidate-E promotion receipt",
    )
    validate_promotion(
        promotion_report,
        promotion_receipt,
        report_sha256=PROMOTION_REPORT_SHA256,
        ledger_sha256=UPDATED_LEDGER_SHA256,
    )
    validate_receipt_artifact_files(
        Path(artifacts["candidate_e_promotion_report"]["path"]).parent,
        promotion_receipt["artifact_sha256"],
    )
    return artifacts


def case_selector(case: AuditCase) -> tuple[str, str, str, str, str, str]:
    return (
        case.split,
        case.speaker_id,
        case.sample_id,
        case.sample_group,
        case.view,
        case.condition,
    )


def row_selector(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return tuple(
        str(row[key])
        for key in (
            "split",
            "speaker_id",
            "sample_id",
            "sample_group",
            "view",
            "condition",
        )
    )


def load_topology_inputs(
    path: Path,
    expected_sha256: str,
    cases: list[AuditCase],
    *,
    candidate_e_evidence_sha256: Mapping[str, str],
    label_bank_sha256: str,
    selection_salt: str,
) -> tuple[
    dict[tuple[str, str, str, str, str, str], TopologyAuditInput],
    dict[str, Any],
]:
    resolved = _verified_file(path, expected_sha256, "topology input manifest")
    manifest = _read_json_mapping(resolved, "topology input manifest")
    if set(manifest) != TOPOLOGY_MANIFEST_FIELDS:
        raise ValueError("topology input manifest fields differ")
    expected_contract = {
        "schema_version": TOPOLOGY_INPUT_SCHEMA_VERSION,
        "candidate_e_source_commit": CANDIDATE_E_SOURCE_COMMIT,
        "candidate_e_evidence_sha256": dict(candidate_e_evidence_sha256),
        "label_bank_sha256": label_bank_sha256,
        "selection_salt": selection_salt,
        "sample_rate": ROUTE_C_V19_SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
        "topology_role": "base_current_output",
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "final_panel_opened": False,
        "fresh_panel_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
    }
    for field, expected in expected_contract.items():
        observed = manifest[field]
        if observed != expected or type(observed) is not type(expected):
            raise ValueError(f"topology input manifest {field} differs")

    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("topology input manifest coverage differs")
    expected_cases = {case_selector(case): case for case in cases}
    if len(expected_cases) != len(cases):
        raise ValueError("selected dev cases are not unique")
    observed_inputs: dict[
        tuple[str, str, str, str, str, str], TopologyAuditInput
    ] = {}
    case_ids: set[str] = set()
    role_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != TOPOLOGY_ROW_FIELDS:
            raise ValueError("topology input row fields differ")
        selector = row_selector(row)
        case = expected_cases.get(selector)
        if case is None or selector in observed_inputs:
            raise ValueError("topology input case binding differs")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("topology input case IDs differ")
        case_ids.add(case_id)
        source_path = Path(row["source_waveform_path"])
        if not source_path.is_absolute() or source_path.resolve() != (
            case.waveform_path.resolve()
        ):
            raise ValueError("topology source waveform path differs")
        if row["source_audio_file_sha256"] != case.waveform_sha256:
            raise ValueError("topology source audio-file hash differs")
        _verified_file(
            source_path,
            case.waveform_sha256,
            f"topology source waveform {case_id}",
        )
        if row["source_segment_samples"] != SEGMENT_SAMPLES:
            raise ValueError("topology source segment length differs")
        if not _is_sha256(row["source_waveform_float32_sha256"]):
            raise ValueError("topology source float32 hash is invalid")
        if not _is_sha256(row["topology_sha256"]):
            raise ValueError("topology digest is invalid")
        topology = row["topology"]
        if not isinstance(topology, dict):
            raise ValueError("topology payload is not a JSON mapping")
        if topology.get("case_id") != case_id or topology.get("view") != case.view:
            raise ValueError("topology payload case/view binding differs")
        if topology.get("source_sample_count") != SEGMENT_SAMPLES:
            raise ValueError("topology payload segment length differs")
        if topology.get("source_waveform_float32_sha256") != row[
            "source_waveform_float32_sha256"
        ]:
            raise ValueError("topology payload source hash differs")
        role = topology.get("role")
        if role not in ROUTE_C_V19_CURRENT_OUTPUT_ROLES:
            raise ValueError("topology payload is not a current output")
        if topology.get("implementation") != CANDIDATE_E_TOPOLOGY_IMPLEMENTATION:
            raise ValueError("topology payload implementation differs")
        if topology.get("metric_highpass") != (
            ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE
        ):
            raise ValueError("topology payload is not the base high-pass path")
        if topology.get("topology_input_loader") != (
            ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER
        ):
            raise ValueError("topology payload is not the base current-output loader")
        role_counts[str(role)] = role_counts.get(str(role), 0) + 1
        observed_inputs[selector] = TopologyAuditInput(
            case_id=case_id,
            topology=topology,
            topology_sha256=row["topology_sha256"],
            source_waveform_float32_sha256=row[
                "source_waveform_float32_sha256"
            ],
        )
    if set(observed_inputs) != set(expected_cases):
        raise ValueError("topology input manifest does not exactly cover selection")
    coverage = {
        "manifest_path": str(resolved),
        "manifest_sha256": expected_sha256,
        "expected_cases": len(cases),
        "observed_cases": len(observed_inputs),
        "unique_case_ids": len(case_ids),
        "cases_by_split": {
            split: sum(selector[0] == split for selector in observed_inputs)
            for split in AUDIT_SPLITS
        },
        "cases_by_view": {
            view: sum(selector[4] == view for selector in observed_inputs)
            for view in ("cs", "sv")
        },
        "topology_roles": role_counts,
        "exact_selection_coverage": True,
    }
    return observed_inputs, coverage


def validate_topology_receipt(
    path: Path,
    expected_sha256: str,
    *,
    topology_manifest_sha256: str,
    source_commit: str,
) -> dict[str, str]:
    resolved = _verified_file(path, expected_sha256, "topology receipt")
    receipt = _read_json_mapping(resolved, "topology receipt")
    if (
        receipt.get("schema_version") != TOPOLOGY_RECEIPT_SCHEMA_VERSION
        or receipt.get("decision") != TOPOLOGY_SEAL_DECISION
        or receipt.get("source", {}).get("head") != source_commit
        or receipt.get("topology_count") != 8
        or receipt.get("candidate_exact_outcomes_opened") is not False
        or receipt.get("generator_optimizer_steps") != 0
        or receipt.get("artifact_sha256", {}).get(
            "candidate_e_topology_manifest_v2.json"
        )
        != topology_manifest_sha256
    ):
        raise ValueError("Candidate-E topology receipt differs")
    return {"path": str(resolved), "sha256": expected_sha256}


def extract_case_measurement(
    scorer: torch.nn.Module,
    case: AuditCase,
    topology_input: TopologyAuditInput,
    device: torch.device,
) -> dict[str, Any]:
    waveform = load_fixed_segment(case).to(device).requires_grad_(True)
    raw_target = case.clean_target.to(device).unsqueeze(0)
    prediction = scorer(
        waveform,
        case.view,
        topology=topology_input.topology,
        case_id=topology_input.case_id,
        view=case.view,
        topology_sha256=topology_input.topology_sha256,
    )
    if prediction.shape != (1, len(AVQI_COMPONENT_NAMES)):
        raise ValueError("six-component scorer output shape differs")
    losses = six_active_bidirectional_gap_losses(
        prediction,
        raw_target,
        scorer.target_mean,
        scorer.target_scale,
    )[0]
    normalized_target = scorer.normalized_target(raw_target)[0]
    denormalized_prediction = scorer.denormalized_prediction(prediction)[0]
    gradients: dict[str, torch.Tensor] = {}
    components: dict[str, dict[str, Any]] = {}
    for offset, component in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS):
        gradient = torch.autograd.grad(
            losses[offset],
            waveform,
            retain_graph=offset < len(ROUTE_C_SIX_ACTIVE_COMPONENTS) - 1,
            create_graph=False,
        )[0].detach().cpu().to(dtype=torch.float64)
        projection: dict[str, Any] | None = None
        if component == "shimmer_db":
            plan = build_cycle_gain_plan(
                waveform.detach().cpu().numpy(),
                topology_input.topology,
            )
            proxy = candidate_e_proxy(
                waveform.detach().cpu().to(dtype=torch.float64),
                torch.as_tensor(
                    topology_input.topology["pulse_positions_samples"],
                    dtype=torch.float64,
                ),
                torch.from_numpy(plan["source_indices"]),
                int(
                    topology_input.topology[
                        "metric_constant_prefix_samples"
                    ]
                ),
                peak_scale_required=bool(
                    topology_input.topology.get("timing_ms", {}).get(
                        "highpass_peak_scaled"
                    )
                ),
                expected_highpass_pcm16_sha256=topology_input.topology.get(
                    "highpass_pcm16_sha256"
                ),
            )
            peak_certificate = validate_candidate_e_base_peak_certificate(
                topology_input.topology,
                proxy,
            )
            proxy_value = float(proxy.shimmer_db.detach().cpu())
            prediction_value = float(
                denormalized_prediction[
                    AVQI_COMPONENT_NAMES.index("shimmer_db")
                ].detach().cpu()
            )
            if not math.isclose(
                proxy_value,
                prediction_value,
                rel_tol=1e-6,
                abs_tol=1e-5,
            ):
                raise ValueError("Candidate-E scorer/proxy value differs")
            gradient, projection = project_cycle_gain_gradient_fixed_order(
                waveform.detach().cpu().to(dtype=torch.float64),
                gradient,
                plan,
            )
            if projection.get("projected_gradient_valid") is not True:
                raise ValueError(
                    "invalid Candidate-E projected gradient: "
                    f"{topology_input.case_id}"
                )
            projection = {
                **projection,
                "candidate_e_proxy_shimmer_db": proxy_value,
                "candidate_e_sinc70_peak_upper_bound": (
                    peak_certificate["base_peak_upper_bound"]
                ),
                "candidate_e_local_sinc70_peak_upper_bound": (
                    peak_certificate["base_local_sinc70_peak_upper_bound"]
                ),
                "candidate_e_exact_sinc70_peak": peak_certificate[
                    "base_exact_sinc70_peak"
                ],
                "candidate_e_peak_check_mode": peak_certificate[
                    "base_peak_check_mode"
                ],
                "candidate_e_peak_scale_abstention_pass": peak_certificate[
                    "base_peak_scale_abstention_pass"
                ],
                "candidate_e_peak_scale_support_pass": peak_certificate[
                    "base_peak_scale_support_pass"
                ],
                "candidate_e_peak_handling_pass": peak_certificate[
                    "base_peak_handling_pass"
                ],
                "candidate_e_exact_highpass_pcm16_sha256": (
                    proxy.exact_highpass_pcm16_sha256
                ),
            }
        norm = float(torch.linalg.vector_norm(gradient))
        finite = bool(torch.isfinite(gradient).all()) and math.isfinite(norm)
        if not finite or norm <= 0.0:
            raise ValueError(
                f"invalid raw six-component gradient: "
                f"{topology_input.case_id}/{component}"
            )
        index = AVQI_COMPONENT_NAMES.index(component)
        gradients[component] = gradient
        components[component] = {
            "prediction": float(denormalized_prediction[index].detach().cpu()),
            "clean_pathological_target": float(case.clean_target[index]),
            "normalized_signed_error": float(
                (prediction[0, index] - normalized_target[index]).detach().cpu()
            ),
            "normalized_bidirectional_gap": float(
                (prediction[0, index] - normalized_target[index])
                .abs()
                .detach()
                .cpu()
            ),
            "smooth_l1_loss": float(losses[offset].detach().cpu()),
            "gradient_norm": norm,
            "finite_observed": True,
            "strictly_positive_norm_observed": True,
            "candidate_e_projection": projection,
            "scientific_gate_applied": False,
        }
    return {
        "split": case.split,
        "speaker_id": case.speaker_id,
        "sample_id": case.sample_id,
        "sample_group": case.sample_group,
        "view": case.view,
        "condition": case.condition,
        "case_id": topology_input.case_id,
        "source_audio_file_sha256": case.waveform_sha256,
        "source_waveform_float32_sha256": (
            topology_input.source_waveform_float32_sha256
        ),
        "segment_samples": SEGMENT_SAMPLES,
        "topology": {
            "role": "base_current_output",
            "worker_role": topology_input.topology["role"],
            "topology_sha256": topology_input.topology_sha256,
            "highpass_pcm16_sha256": topology_input.topology.get(
                "highpass_pcm16_sha256"
            ),
            "highpass_peak_scaled": topology_input.topology.get(
                "timing_ms", {}
            ).get("highpass_peak_scaled"),
            "pulse_count": topology_input.topology.get("pulse_count"),
            "metric_source_range_count": topology_input.topology.get(
                "metric_source_range_count"
            ),
            "metric_constant_prefix_samples": topology_input.topology.get(
                "metric_constant_prefix_samples"
            ),
            "slot2_shimmer_percent_uses_topology": False,
            "slot3_shimmer_db_uses_topology": True,
        },
        "components": components,
        "_gradients": gradients,
    }


def calibration_inverse_gradient_weights(
    calibration_records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    if not calibration_records:
        raise ValueError("no six-component calibration gradient records")
    median_norms = {
        component: statistics.median(
            record["components"][component]["gradient_norm"]
            for record in calibration_records
        )
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in median_norms.values()):
        raise ValueError("six-component calibration medians are invalid")
    minimum = min(median_norms.values())
    weights = {
        component: minimum / median_norms[component]
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    return median_norms, weights


def finalize_case_measurement(
    record: dict[str, Any],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    if set(weights) != set(ROUTE_C_SIX_ACTIVE_COMPONENTS):
        raise ValueError("six-component joint-gradient weight keys differ")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("six-component joint-gradient weights are invalid")
    gradients = record.pop("_gradients")
    weighted = {
        component: gradients[component] * weights[component]
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    joint = sum(weighted.values()) / sum(weights.values())
    joint_norm = float(torch.linalg.vector_norm(joint))
    if not bool(torch.isfinite(joint).all()) or not math.isfinite(joint_norm):
        raise ValueError("six-component joint gradient is non-finite")
    if joint_norm <= 0.0:
        raise ValueError("six-component joint gradient is zero")
    weighted_norms = {
        component: float(torch.linalg.vector_norm(weighted[component]))
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    weighted_norm_sum = sum(weighted_norms.values())
    if not math.isfinite(weighted_norm_sum) or weighted_norm_sum <= 0.0:
        raise ValueError("six-component weighted norm sum is invalid")
    shares = {
        component: weighted_norms[component] / weighted_norm_sum
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    pairwise: dict[str, dict[str, Any]] = {}
    for first_index, first in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS):
        for second in ROUTE_C_SIX_ACTIVE_COMPONENTS[first_index + 1 :]:
            value = cosine(gradients[first], gradients[second])
            if not math.isfinite(value):
                raise ValueError(f"undefined six-component cosine: {first}/{second}")
            pairwise[f"{first}__{second}"] = {
                "cosine": value,
                "negative_direction_observed": value < 0.0,
                "scientific_gate_applied": False,
            }
    component_to_joint: dict[str, dict[str, Any]] = {}
    for component in ROUTE_C_SIX_ACTIVE_COMPONENTS:
        value = cosine(gradients[component], joint)
        if not math.isfinite(value):
            raise ValueError(f"undefined component-to-joint cosine: {component}")
        component_to_joint[component] = {
            "cosine": value,
            "negative_direction_observed": value < 0.0,
            "scientific_gate_applied": False,
        }
    record["joint"] = {
        "gradient_norm": joint_norm,
        "calibration_only_inverse_gradient_weights": dict(weights),
        "weighted_component_gradient_norms": weighted_norms,
        "weighted_component_norm_shares": shares,
        "maximum_component_norm_share": max(shares.values()),
        "dominant_component": max(shares, key=shares.__getitem__),
        "pairwise_component_cosines": pairwise,
        "component_to_joint_cosines": component_to_joint,
        "all_values_finite_observed": True,
        "scientific_gate_applied": False,
    }
    return record


def aggregate_measurements(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty six-component measurements")
    component_summary: dict[str, dict[str, Any]] = {}
    for component in ROUTE_C_SIX_ACTIVE_COMPONENTS:
        norms = [row["components"][component]["gradient_norm"] for row in records]
        shares = [
            row["joint"]["weighted_component_norm_shares"][component]
            for row in records
        ]
        joint_cosines = [
            row["joint"]["component_to_joint_cosines"][component]["cosine"]
            for row in records
        ]
        component_summary[component] = {
            "gradient_norm_min": min(norms),
            "gradient_norm_median": statistics.median(norms),
            "gradient_norm_max": max(norms),
            "weighted_norm_share_median": statistics.median(shares),
            "weighted_norm_share_max": max(shares),
            "joint_cosine_min": min(joint_cosines),
            "joint_cosine_median": statistics.median(joint_cosines),
            "joint_cosine_max": max(joint_cosines),
            "negative_to_joint_observations": sum(
                value < 0.0 for value in joint_cosines
            ),
        }
    pairwise_summary: dict[str, dict[str, Any]] = {}
    for pair in PAIRWISE_COMPONENT_KEYS:
        values = [
            row["joint"]["pairwise_component_cosines"][pair]["cosine"]
            for row in records
        ]
        pairwise_summary[pair] = {
            "cosine_min": min(values),
            "cosine_median": statistics.median(values),
            "cosine_max": max(values),
            "negative_direction_observations": sum(value < 0.0 for value in values),
            "negative_direction_fraction": sum(value < 0.0 for value in values)
            / len(values),
        }
    joint_norms = [row["joint"]["gradient_norm"] for row in records]
    maximum_shares = [
        row["joint"]["maximum_component_norm_share"] for row in records
    ]
    return {
        "cases": len(records),
        "components": component_summary,
        "pairwise_component_cosines": pairwise_summary,
        "joint_gradient_norm_min": min(joint_norms),
        "joint_gradient_norm_median": statistics.median(joint_norms),
        "joint_gradient_norm_max": max(joint_norms),
        "maximum_component_norm_share_observed": max(maximum_shares),
        "component_gradient_measurements": (
            len(records) * len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
        ),
        "pairwise_cosine_measurements": len(records) * len(PAIRWISE_COMPONENT_KEYS),
        "component_to_joint_cosine_measurements": (
            len(records) * len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
        ),
        "all_values_finite_observed": True,
        "scientific_gate_applied": False,
    }


def slot_separation_metadata(
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    shimmer_percent = source_metadata.get("shimmer_percent")
    shimmer_db = source_metadata.get("shimmer_db")
    if not isinstance(shimmer_percent, Mapping) or not isinstance(
        shimmer_db, Mapping
    ):
        raise ValueError("six-component Shimmer source metadata is unavailable")
    if shimmer_percent.get("component_indices") != [2]:
        raise ValueError("Shimmer percent is not isolated to slot 2")
    if (
        shimmer_db.get("component_indices") != [3]
        or shimmer_db.get("checkpoint_affine_used") is not False
        or shimmer_db.get("source")
        != "candidate_e_v32r8_current_output_exact_topology"
    ):
        raise ValueError("Shimmer dB is not isolated to external slot 3")
    return {
        "slot2_shimmer_percent": {
            "component_index": 2,
            "source": "sealed_shimmer_percent_checkpoint",
            "checkpoint_output_preserved": True,
            "v19_topology_used": False,
        },
        "slot3_shimmer_db": {
            "component_index": 3,
            "source": "candidate_e_v32r8_current_waveform_exact_path",
            "checkpoint_affine_used": False,
            "v19_topology_used": True,
            "topology_role": "base_current_output",
            "implementation": (
                "candidate_e_exact_path_fixed_order_cycle_gain_projection"
            ),
            "candidate_e_reference_sha256": CANDIDATE_E_REFERENCE_SHA256,
            "topology_implementation": CANDIDATE_E_TOPOLOGY_IMPLEMENTATION,
            "metric_highpass": ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
            "topology_input_loader": ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
            "scientific_promotion_granted": True,
        },
        "slots_are_independent": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA measurement requested but no GPU is visible")
    source = verify_source(
        args.source_root.resolve(),
        args.source_commit,
        args.accepted_base_commit,
    )
    test_evidence_path = _verified_file(
        args.test_evidence,
        args.test_evidence_sha256,
        "focused test evidence",
    )
    if test_evidence_path.stat().st_size == 0:
        raise ValueError("focused test evidence is empty")
    five_source_evidence = validate_five_source_evidence(args.source_evidence)
    candidate_e_evidence = validate_candidate_e_evidence(
        args.candidate_e_evidence
    )
    raw_checkpoint_paths = {
        "cpps": args.cpps_checkpoint,
        "hnr": args.hnr_checkpoint,
        "shimmer_percent": args.shimmer_checkpoint,
        "slope": args.slope_checkpoint,
        "tilt": args.tilt_checkpoint,
    }
    checkpoint_hashes = {
        "cpps": args.cpps_checkpoint_sha256,
        "hnr": args.hnr_checkpoint_sha256,
        "shimmer_percent": args.shimmer_checkpoint_sha256,
        "slope": args.slope_checkpoint_sha256,
        "tilt": args.tilt_checkpoint_sha256,
    }
    checkpoint_paths = {
        component: _verified_file(
            path,
            checkpoint_hashes[component],
            f"{component} checkpoint",
        )
        for component, path in raw_checkpoint_paths.items()
    }
    bundle = load_route_c_candidate_e_six_scorer(
        checkpoint_paths,
        checkpoint_hashes,
    )
    device = torch.device(args.device)
    scorer = bundle.scorer.to(device).eval()
    if sum(parameter.numel() for parameter in scorer.parameters()) != 0:
        raise ValueError("six-component scorer unexpectedly has parameters")
    separation = slot_separation_metadata(bundle.source_metadata)
    label_loader = (
        load_svd_fusion_label_bank
        if args.selection_mode == "sealed_external_svd_v2"
        else load_label_bank
    )
    cases, label_mean, label_scale, selection = label_loader(
        args.label_bank, args.label_bank_sha256, args.selection_salt
    )
    if not torch.equal(scorer.target_mean.detach().cpu(), label_mean):
        raise ValueError("six-component target means differ from label bank")
    if not torch.equal(scorer.target_scale.detach().cpu(), label_scale):
        raise ValueError("six-component target scales differ from label bank")
    topology_inputs, topology_coverage = load_topology_inputs(
        args.topology_manifest,
        args.topology_manifest_sha256,
        cases,
        candidate_e_evidence_sha256={
            key: value["sha256"]
            for key, value in candidate_e_evidence.items()
        },
        label_bank_sha256=args.label_bank_sha256,
        selection_salt=args.selection_salt,
    )
    topology_receipt = validate_topology_receipt(
        args.topology_receipt,
        args.topology_receipt_sha256,
        topology_manifest_sha256=args.topology_manifest_sha256,
        source_commit=args.source_commit,
    )

    extracted = []
    for index, case in enumerate(cases, start=1):
        print(
            f"six_gradient_case={index}/{len(cases)} split={case.split} "
            f"view={case.view} group={case.sample_group}",
            flush=True,
        )
        extracted.append(
            extract_case_measurement(
                scorer,
                case,
                topology_inputs[case_selector(case)],
                device,
            )
        )
    calibration_raw = [
        row for row in extracted if row["split"] == "surrogate_calibration"
    ]
    median_norms, weights = calibration_inverse_gradient_weights(calibration_raw)
    finalized = [finalize_case_measurement(row, weights) for row in extracted]
    calibration_rows = [
        row for row in finalized if row["split"] == "surrogate_calibration"
    ]
    holdout_rows = [
        row for row in finalized if row["split"] == "surrogate_holdout"
    ]
    calibration_summary = aggregate_measurements(calibration_rows)
    holdout_summary = aggregate_measurements(holdout_rows)
    weighted_calibration_medians = {
        component: median_norms[component] * weights[component]
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    selection = {
        **selection,
        "calibration_speaker_ids": selection["speakers_by_split"][
            "surrogate_calibration"
        ],
        "holdout_speaker_ids": selection["speakers_by_split"][
            "surrogate_holdout"
        ],
        "component_and_joint_share_split": True,
        "topology_manifest_uses_same_selection": True,
    }
    registry_records = route_c_candidate_e_registry_records()
    integrity = {
        "six_component_order_exact": tuple(ROUTE_C_SIX_ACTIVE_COMPONENTS)
        == AVQI_COMPONENT_NAMES,
        "all_15_pairwise_cosines_per_case": all(
            set(row["joint"]["pairwise_component_cosines"])
            == set(PAIRWISE_COMPONENT_KEYS)
            for row in finalized
        ),
        "all_6_component_to_joint_cosines_per_case": all(
            set(row["joint"]["component_to_joint_cosines"])
            == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)
            for row in finalized
        ),
        "calibration_holdout_speaker_disjoint": selection["speaker_overlap"] == 0,
        "topology_exact_selection_coverage": topology_coverage[
            "exact_selection_coverage"
        ],
        "slot2_slot3_sources_independent": separation["slots_are_independent"],
        "scorer_has_zero_parameters": sum(
            parameter.numel() for parameter in scorer.parameters()
        )
        == 0,
        "shimmer_db_scientific_status_promoted": bundle.scientific_status
        == ROUTE_C_CANDIDATE_E_SCIENTIFIC_STATUS,
        "candidate_e_promoted_runtime_evidence_bound": (
            set(candidate_e_evidence) == set(CANDIDATE_E_EVIDENCE_KEYS)
        ),
        "numeric_scientific_gates_applied": False,
        "final_or_fresh_panel_opened": False,
        "generator_optimizer_steps": 0,
    }
    required_integrity = {
        key: value
        for key, value in integrity.items()
        if key
        not in {
            "numeric_scientific_gates_applied",
            "final_or_fresh_panel_opened",
            "generator_optimizer_steps",
        }
    }
    if any(value is not True for value in required_integrity.values()):
        raise ValueError("six-component measurement integrity check failed")

    checkpoint_evidence = {
        f"{component}_checkpoint": {
            "path": str(Path(checkpoint_paths[component]).resolve()),
            "sha256": checkpoint_hashes[component],
        }
        for component in checkpoint_paths
    }
    source_evidence_sha256 = {
        **{
            key: value["sha256"]
            for key, value in five_source_evidence.items()
        },
        **{
            key: value["sha256"]
            for key, value in checkpoint_evidence.items()
        },
        "label_bank": args.label_bank_sha256,
        "topology_manifest": args.topology_manifest_sha256,
        "topology_receipt": args.topology_receipt_sha256,
        "focused_test_evidence": args.test_evidence_sha256,
        **{
            key: value["sha256"]
            for key, value in candidate_e_evidence.items()
        },
    }
    report = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "decision": MEASUREMENT_DECISION,
        "joint_panel_decision": JOINT_PANEL_DECISION,
        "contract": {
            "source": source,
            "architecture": ROUTE_C_CANDIDATE_E_SIX_ACTIVE_ARCHITECTURE,
            "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
            "registry_schema_version": (
                ROUTE_C_CANDIDATE_E_REGISTRY_SCHEMA_VERSION
            ),
            "component_registry": registry_records,
            "avqi_v0301": {
                "intercept": AVQI_V0301_INTERCEPT,
                "outer_scale": AVQI_V0301_SCALE,
                "coefficients": list(AVQI_V0301_COEFFICIENTS),
                "expanded_coefficients": list(
                    AVQI_V0301_EXPANDED_COEFFICIENTS
                ),
            },
            "normalization": {
                "source": "surrogate_train exact-component standard deviation",
                "target_mean": {
                    component: float(scorer.target_mean[index].detach().cpu())
                    for index, component in enumerate(AVQI_COMPONENT_NAMES)
                },
                "target_scale": {
                    component: float(scorer.target_scale[index].detach().cpu())
                    for index, component in enumerate(AVQI_COMPONENT_NAMES)
                },
            },
            "loss_target": (
                "normalized bidirectional gap to same-speaker clean "
                "pathological CS/SV target"
            ),
            "avqi_scalar_coefficient_used_for_direction": False,
            "weight_fit_split": "surrogate_calibration",
            "weight_rule": (
                "minimum calibration median gradient norm / component median "
                "gradient norm"
            ),
            "scientific_schema_frozen": False,
            "numeric_scientific_gates_applied": False,
            "unfrozen_decision_fields": list(UNFROZEN_DECISION_FIELDS),
        },
        "slot2_slot3_separation": separation,
        "source_checkpoints": bundle.source_metadata,
        "source_evidence": {
            **five_source_evidence,
            **checkpoint_evidence,
            "label_bank": {
                "path": str(args.label_bank.resolve()),
                "sha256": args.label_bank_sha256,
            },
            **candidate_e_evidence,
            "topology_manifest": {
                "path": str(args.topology_manifest.resolve()),
                "sha256": args.topology_manifest_sha256,
            },
            "topology_receipt": topology_receipt,
            "focused_test_evidence": {
                "path": str(test_evidence_path),
                "sha256": args.test_evidence_sha256,
            },
        },
        "source_evidence_sha256": source_evidence_sha256,
        "candidate_e_runtime_evidence": {
            "source_commit": CANDIDATE_E_SOURCE_COMMIT,
            "promotion_report_sha256": PROMOTION_REPORT_SHA256,
            "promotion_receipt_sha256": PROMOTION_RECEIPT_SHA256,
            "runtime_client_sha256": CANDIDATE_E_RUNTIME_CLIENT_SHA256,
            "worker_sha256": CANDIDATE_E_WORKER_SHA256,
            "selector_sha256": CANDIDATE_E_SELECTOR_SHA256,
            "runtime_config_sha256": CANDIDATE_E_RUNTIME_CONFIG_SHA256,
            "scientific_promotion_granted": True,
            "candidate_exact_outcomes_used_for_selection": False,
            "speaker_or_case_identity_used_for_selection": False,
            "generator_optimizer_steps": 0,
        },
        "selection": selection,
        "topology_coverage": topology_coverage,
        "coverage": {
            "selected_cases": len(cases),
            "cases_by_split": selection["cases_by_split"],
            "component_gradient_measurements": (
                len(cases) * len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
            ),
            "pairwise_cosine_measurements": len(cases)
            * len(PAIRWISE_COMPONENT_KEYS),
            "component_to_joint_cosine_measurements": (
                len(cases) * len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
            ),
            "expected_pairwise_cosines_per_case": len(PAIRWISE_COMPONENT_KEYS),
            "expected_component_to_joint_cosines_per_case": len(
                ROUTE_C_SIX_ACTIVE_COMPONENTS
            ),
        },
        "calibration": {
            **calibration_summary,
            "median_component_gradient_norms": median_norms,
            "frozen_inverse_gradient_weights": weights,
            "weighted_median_gradient_norms": weighted_calibration_medians,
            "weights_selected_on_holdout": False,
        },
        "holdout": holdout_summary,
        "case_results": finalized,
        "measurement_integrity": integrity,
        "runtime": {
            "python_executable": sys.executable,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "waveform_mutation_performed": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }

    evaluator_path = Path(__file__).resolve()
    launcher_path = evaluator_path.with_name(
        "run_avqi_route_c_six_component_gradient_audit.sh"
    )
    if not launcher_path.is_file():
        raise ValueError("six-component measurement launcher is unavailable")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "six_gradient_measurement_report.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": (
            "dev-avqi-route-c-six-gradient-raw-measurement-receipt-v2"
        ),
        "decision": MEASUREMENT_DECISION,
        "joint_panel_decision": JOINT_PANEL_DECISION,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "accepted_base_commit": source["accepted_base_commit"],
        "active_components": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "calibration_cases": selection["cases_by_split"][
            "surrogate_calibration"
        ],
        "holdout_cases": selection["cases_by_split"]["surrogate_holdout"],
        "source_evidence_sha256": source_evidence_sha256,
        "implementation_sha256": {
            evaluator_path.name: sha256_file(evaluator_path),
            launcher_path.name: sha256_file(launcher_path),
        },
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher_submitted_slurm_job": False,
        "scientific_schema_frozen": False,
        "numeric_scientific_gates_applied": False,
        "unfrozen_decision_fields": list(UNFROZEN_DECISION_FIELDS),
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "combined_final_panel_opened": False,
        "fresh_panel_opened": False,
        "exact_candidate_scoring_requested": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
