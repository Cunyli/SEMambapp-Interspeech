#!/usr/bin/env python3
"""Seal the exact-Praat code tree and runtime used by the six-joint panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_panel_readiness import TRAINING_NO_GO
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    EXACT_CODE_TREE_MANIFEST_SCHEMA_VERSION,
    EXACT_RUNTIME_MANIFEST_SCHEMA_VERSION,
    STEP_VERSIONS,
    avqi_code_tree_sha256,
    validate_exact_authority,
)


RECEIPT_SCHEMA = "avqi-route-c-exact-authority-seal-receipt-v1"
SEAL_DECISION = "SEALED_EXACT_PRAAT_AUTHORITY_FOR_SIX_JOINT_V1"
RUNTIME_PROBE = r"""
import json
import parselmouth

print(json.dumps({
    "parselmouth_version": parselmouth.__version__,
    "praat_version": parselmouth.PRAAT_VERSION,
}, sort_keys=True))
"""


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-repo-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite exact authority seal: {args.output_dir}"
        )
    exact_python = args.exact_python.resolve()
    code_root = args.avqi_code_root.resolve()
    if not exact_python.is_file() or not code_root.is_dir():
        raise FileNotFoundError("exact-Praat Python or code root is unavailable")
    head = git_value(code_root, "rev-parse", "HEAD")
    if head != args.avqi_repo_commit:
        raise ValueError("exact AVQI repository commit differs")
    if git_value(code_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("exact AVQI repository is dirty")
    main_path = code_root / "avqi_code" / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(main_path)
    tree_sha256 = avqi_code_tree_sha256(code_root)
    code_manifest = {
        "schema_version": EXACT_CODE_TREE_MANIFEST_SCHEMA_VERSION,
        "avqi_code_root": str(code_root),
        "avqi_repo_commit": head,
        "avqi_repo_branch": git_value(code_root, "branch", "--show-current"),
        "avqi_repo_tree": git_value(code_root, "rev-parse", "HEAD^{tree}"),
        "avqi_code_tree_sha256": tree_sha256,
        "main_py_sha256": sha256_file(main_path),
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    args.output_dir.mkdir(parents=True)
    code_path = args.output_dir / "exact_avqi_code_tree_manifest.json"
    write_json(code_path, code_manifest)
    code_sha256 = sha256_file(code_path)
    probe = subprocess.run(
        [str(exact_python), "-c", RUNTIME_PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_versions = json.loads(probe.stdout)
    if not isinstance(runtime_versions, dict):
        raise ValueError("exact-Praat runtime probe did not return a mapping")
    runtime_manifest = {
        "schema_version": EXACT_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "exact_python": str(exact_python),
        "avqi_code_tree_manifest_sha256": code_sha256,
        "avqi_code_tree_sha256": tree_sha256,
        "parselmouth_version": str(runtime_versions["parselmouth_version"]),
        "praat_version": str(runtime_versions["praat_version"]),
        "step_versions": STEP_VERSIONS,
        "candidate_exact_outcomes_opened": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    runtime_path = args.output_dir / "exact_runtime_manifest.json"
    write_json(runtime_path, runtime_manifest)
    validate_exact_authority(
        exact_python=exact_python,
        avqi_code_root=code_root,
        code_manifest=code_manifest,
        code_manifest_sha256=code_sha256,
        runtime_manifest=runtime_manifest,
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": SEAL_DECISION,
        "exact_candidate_outcomes_opened": False,
        "artifact_sha256": {
            code_path.name: code_sha256,
            runtime_path.name: sha256_file(runtime_path),
        },
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": SEAL_DECISION,
                "avqi_repo_commit": head,
                "avqi_code_tree_sha256": tree_sha256,
                "code_manifest_sha256": code_sha256,
                "runtime_manifest_sha256": sha256_file(runtime_path),
                "completion_receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
