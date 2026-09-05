#!/usr/bin/env python3
"""Seal fresh Candidate-E exact topologies for the eight-case gradient audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from model.avqi_route_c_candidate_e import (
    CANDIDATE_E_RUNTIME_CLIENT_SHA256,
    CANDIDATE_E_SOURCE_COMMIT,
    CANDIDATE_E_TOPOLOGY_IMPLEMENTATION,
    CANDIDATE_E_WORKER_SHA256,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_panel_readiness import TRAINING_NO_GO
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    SEGMENT_SAMPLES,
    load_fixed_segment,
    load_label_bank,
    load_svd_fusion_label_bank,
    verify_source,
)
from scripts.evaluate_avqi_route_c_six_component_gradients import (
    ACCEPTED_SIX_SCAFFOLD_BASE,
    CANDIDATE_E_EVIDENCE_KEYS,
    TOPOLOGY_INPUT_SCHEMA_VERSION,
    TOPOLOGY_RECEIPT_SCHEMA_VERSION,
    TOPOLOGY_SEAL_DECISION,
    case_selector,
    validate_candidate_e_evidence,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    validate_exact_authority,
)
from scripts.seal_avqi_route_c_exact_authority_v1 import (
    RECEIPT_SCHEMA as EXACT_AUTHORITY_RECEIPT_SCHEMA,
    SEAL_DECISION as EXACT_AUTHORITY_SEAL_DECISION,
)


SAMPLE_RATE = 16_000


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def waveform_float32_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype="<f4").reshape(-1).tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_runtime_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "candidate_e_exact_topology_runtime_v32r8",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load Candidate-E exact topology runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument(
        "--selection-mode",
        choices=("legacy_tau", "sealed_external_svd_v2"),
        default="legacy_tau",
    )
    parser.add_argument(
        "--candidate-e-evidence",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite topology seal: {args.output_dir}"
        )
    source = verify_source(
        args.source_root.resolve(),
        args.source_commit,
        ACCEPTED_SIX_SCAFFOLD_BASE,
    )
    candidate_e_evidence = validate_candidate_e_evidence(
        args.candidate_e_evidence
    )
    evidence_hashes = {
        key: candidate_e_evidence[key]["sha256"]
        for key in CANDIDATE_E_EVIDENCE_KEYS
    }
    exact_authority = validate_exact_authority(
        exact_python=args.exact_python,
        avqi_code_root=args.avqi_code_root,
        code_manifest=read_json(
            Path(candidate_e_evidence["exact_code_tree_manifest"]["path"]),
            "exact code-tree manifest",
        ),
        code_manifest_sha256=evidence_hashes["exact_code_tree_manifest"],
        runtime_manifest=read_json(
            Path(candidate_e_evidence["exact_runtime_manifest"]["path"]),
            "exact runtime manifest",
        ),
    )
    exact_receipt = read_json(
        Path(candidate_e_evidence["exact_authority_receipt"]["path"]),
        "exact authority receipt",
    )
    if (
        exact_receipt.get("schema_version") != EXACT_AUTHORITY_RECEIPT_SCHEMA
        or exact_receipt.get("decision") != EXACT_AUTHORITY_SEAL_DECISION
        or exact_receipt.get("generator_optimizer_steps") != 0
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_avqi_code_tree_manifest.json"
        )
        != evidence_hashes["exact_code_tree_manifest"]
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_runtime_manifest.json"
        )
        != evidence_hashes["exact_runtime_manifest"]
    ):
        raise ValueError("exact authority receipt binding differs")
    label_loader = (
        load_svd_fusion_label_bank
        if args.selection_mode == "sealed_external_svd_v2"
        else load_label_bank
    )
    cases, _, _, selection = label_loader(
        args.label_bank, args.label_bank_sha256, args.selection_salt
    )
    if len(cases) != 8:
        raise ValueError("six-gradient topology selection must contain eight cases")

    runtime_path = Path(
        candidate_e_evidence["candidate_e_runtime_client"]["path"]
    )
    worker_path = Path(candidate_e_evidence["candidate_e_worker"]["path"])
    runtime_module = load_runtime_module(runtime_path)
    if (
        runtime_module.EXPECTED_IMPLEMENTATION
        != CANDIDATE_E_TOPOLOGY_IMPLEMENTATION
        or runtime_module.NUMPY_HIGHPASS_MODE
        != "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
    ):
        raise ValueError("Candidate-E topology runtime contract differs")

    case_ids = []
    items = []
    waveforms = []
    for index, case in enumerate(cases, start=1):
        case_id = (
            f"six-gradient-v2:{index}:{case.split}:{case.speaker_id}:"
            f"{case.sample_id}:{case.sample_group}:{case.view}:{case.condition}"
        )
        waveform = load_fixed_segment(case).detach().cpu().numpy()
        case_ids.append(case_id)
        waveforms.append(waveform)
        items.append(
            {
                "id": f"topology:{case_id}",
                "case_id": case_id,
                "role": "current_output_topology",
                "path": str(case.waveform_path.resolve()),
                "view": case.view,
                "score_components": False,
                "exact_metric_topology": True,
                "highpass_mode": runtime_module.NUMPY_HIGHPASS_MODE,
            }
        )

    with runtime_module.ExactShimmerTopologyWorker(
        args.exact_python.resolve(),
        worker_path,
        args.avqi_code_root.resolve(),
        exact_authority["avqi_code_tree_sha256"],
    ) as worker:
        warmup, warmup_ms = worker.warmup()
        topologies, runtime_ms, staging = worker.refresh_current_waveforms(
            items,
            waveforms,
            highpass_mode=runtime_module.NUMPY_HIGHPASS_MODE,
        )

    rows = []
    for case, case_id, waveform, topology in zip(
        cases,
        case_ids,
        waveforms,
        topologies,
        strict=True,
    ):
        topology_sha256 = runtime_module.topology_sha256(topology)
        source_float32_sha256 = waveform_float32_sha256(waveform)
        if (
            topology["source_waveform_float32_sha256"]
            != source_float32_sha256
        ):
            raise ValueError(f"Candidate-E topology waveform differs: {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "split": case.split,
                "speaker_id": case.speaker_id,
                "sample_id": case.sample_id,
                "sample_group": case.sample_group,
                "view": case.view,
                "condition": case.condition,
                "source_waveform_path": str(case.waveform_path.resolve()),
                "source_audio_file_sha256": case.waveform_sha256,
                "source_waveform_float32_sha256": source_float32_sha256,
                "source_segment_samples": SEGMENT_SAMPLES,
                "topology_sha256": topology_sha256,
                "topology": topology,
            }
        )
    if len({case_selector(case) for case in cases}) != len(rows):
        raise ValueError("six-gradient topology selectors are not unique")

    manifest = {
        "schema_version": TOPOLOGY_INPUT_SCHEMA_VERSION,
        "candidate_e_source_commit": CANDIDATE_E_SOURCE_COMMIT,
        "candidate_e_evidence_sha256": evidence_hashes,
        "label_bank_sha256": args.label_bank_sha256,
        "selection_salt": args.selection_salt,
        "sample_rate": SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
        "topology_role": "base_current_output",
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "final_panel_opened": False,
        "fresh_panel_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "candidate_e_topology_manifest_v2.json"
    write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    receipt = {
        "schema_version": TOPOLOGY_RECEIPT_SCHEMA_VERSION,
        "decision": TOPOLOGY_SEAL_DECISION,
        "source": source,
        "candidate_e_source_commit": CANDIDATE_E_SOURCE_COMMIT,
        "candidate_e_evidence_sha256": evidence_hashes,
        "exact_authority": exact_authority,
        "selection": selection,
        "topology_count": len(rows),
        "topology_runtime_client_sha256": CANDIDATE_E_RUNTIME_CLIENT_SHA256,
        "topology_worker_sha256": CANDIDATE_E_WORKER_SHA256,
        "worker_startup": worker.startup,
        "worker_warmup": warmup,
        "worker_warmup_ms": warmup_ms,
        "topology_batch_runtime_ms": runtime_ms,
        "topology_staging": staging,
        "candidate_exact_outcomes_opened": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
        "artifact_sha256": {manifest_path.name: manifest_sha256},
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": TOPOLOGY_SEAL_DECISION,
                "topology_manifest_sha256": manifest_sha256,
                "completion_receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
