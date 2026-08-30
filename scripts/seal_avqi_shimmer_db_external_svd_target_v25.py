#!/usr/bin/env python3
"""Score and seal target scalars for the external SVD Shimmer-dB panel.

The v24 panel must already be hash-sealed with every Shimmer outcome closed.
This stage opens only the exact-Praat Shimmer-dB scalar of each same-speaker
clean pathological CS/SV target.  Base and candidate exact outcomes remain
closed, no candidate waveform is generated, and no optimizer is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.adjudicate_avqi_shimmer_db_deterministic_opened24_v23 import (
    TRAINING_DECISION,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    SHIMMER_DB_INDEX,
)
from scripts.evaluate_avqi_shimmer_fresh_panel import (
    avqi_code_tree_sha256,
    exact_index,
    run_exact_batch,
)
from scripts.prepare_avqi_shimmer_db_external_svd_panel_v24 import (
    CONDITIONS,
    EXPECTED_CASES,
    EXPECTED_SPEAKERS,
    PANEL_SCHEMA,
    SEAL_RECEIPT_SCHEMA,
    VIEWS,
)


TARGET_SCHEMA = "avqi-route-c-shimmer-db-supervised-target-v1"
TARGET_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-target-receipt-v25"
)
TARGET_DECISION = "SEALED_SHIMMER_DB_EXTERNAL_SVD_TARGET_SCALARS_V25"
PANEL_DECISION = "SEALED_SHIMMER_DB_EXTERNAL_SVD_PANEL_EXACT_UNOPENED_V24"


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("panel-seal", "seal-receipt"):
        add_hashed_path(parser, option)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
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
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_repository(args: argparse.Namespace) -> dict[str, str]:
    root = args.repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain the v25 sealer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v25 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v25 target sealing requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "target_sealer_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_panel_binding(
    panel: dict[str, Any],
    receipt: dict[str, Any],
    *,
    panel_sha256: str,
) -> list[dict[str, Any]]:
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("external SVD panel schema drift")
    if receipt.get("schema_version") != SEAL_RECEIPT_SCHEMA:
        raise ValueError("external SVD panel receipt schema drift")
    if receipt.get("decision") != PANEL_DECISION:
        raise ValueError("external SVD panel is not sealed exact-unopened")
    if receipt.get("artifact_sha256", {}).get("panel_seal.json") != (
        panel_sha256
    ):
        raise ValueError("external SVD panel receipt/seal binding drift")
    if panel.get("source_commit") != receipt.get("source_commit"):
        raise ValueError("external SVD panel source commit drift")
    if panel.get("case_count") != EXPECTED_CASES:
        raise ValueError("external SVD panel case count drift")
    if panel.get("speaker_count") != EXPECTED_SPEAKERS:
        raise ValueError("external SVD panel speaker count drift")
    if panel.get("severity_labels_created") is not False:
        raise ValueError("external SVD panel invented severity labels")
    authorization = panel.get("authorization", {})
    if authorization.get("external_speaker_panel_authorized") is not True:
        raise ValueError("external SVD panel lacks v23 authorization")
    waveform_contract = panel.get("waveform_contract", {})
    if (
        waveform_contract.get("emitted_waveform_highpass") is not False
        or waveform_contract.get("exact_metric_highpass_branch_only") is not True
        or waveform_contract.get(
            "target_is_same_speaker_same_view_clean_pathological"
        )
        is not True
    ):
        raise ValueError("external SVD panel waveform contract drift")
    exact_contract = panel.get("exact_contract", {})
    expected_exact = {
        "target_shimmer_values_opened": False,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "promotion_authorized": False,
    }
    if exact_contract != expected_exact:
        raise ValueError("external SVD panel exact-opening contract drift")
    if (
        receipt.get("exact_shimmer_outcomes_opened") is not False
        or receipt.get("target_scalar_stage_authorized") is not True
        or receipt.get("selector_stage_authorized") is not False
    ):
        raise ValueError("external SVD panel receipt exact-opening drift")
    for label, value in (("panel", panel), ("receipt", receipt)):
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"external SVD {label} over-authorized promotion")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"external SVD {label} over-authorized joint panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"external SVD {label} optimizer boundary drift")
        if value.get("authoritative_training_decision") != TRAINING_DECISION:
            raise ValueError(f"external SVD {label} training decision drift")
    rows = panel.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise ValueError("external SVD panel row coverage drift")
    case_ids: set[str] = set()
    speakers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("external SVD panel row is not an object")
        case_id = str(row.get("case_id", ""))
        panel_speaker_id = str(row.get("panel_speaker_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError("external SVD panel case identity drift")
        case_ids.add(case_id)
        if row.get("dataset") != "SVD" or panel_speaker_id != (
            f"SVD:{row.get('speaker_id')}"
        ):
            raise ValueError(f"external SVD speaker identity drift: {case_id}")
        if row.get("label") != "patient":
            raise ValueError(f"external SVD patient mapping drift: {case_id}")
        if "severity" in row or "sample_group" in row:
            raise ValueError(f"external SVD severity label leakage: {case_id}")
        if row.get("view") not in VIEWS or row.get("condition") not in CONDITIONS:
            raise ValueError(f"external SVD slice drift: {case_id}")
        if not str(row.get("target_sha256", "")):
            raise ValueError(f"external SVD target hash missing: {case_id}")
        speakers.add(panel_speaker_id)
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError("external SVD panel speaker coverage drift")
    if Counter(row["view"] for row in rows) != Counter({"cs": 6, "sv": 6}):
        raise ValueError("external SVD panel view balance drift")
    if Counter(row["condition"] for row in rows) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError("external SVD panel condition balance drift")
    return rows


def target_exact_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"target:{row['case_id']}",
            "path": str(Path(row["target_path"]).resolve()),
            "view": row["view"],
        }
        for row in rows
    ]


def build_target_contract(
    panel: dict[str, Any],
    rows: list[dict[str, Any]],
    exact: dict[str, Any],
    *,
    panel_sha256: str,
    source_commit: str,
    slurm_job_id: str,
    avqi_tree_sha256: str,
) -> dict[str, Any]:
    exact_by_id = exact_index(exact)
    expected_ids = {f"target:{row['case_id']}" for row in rows}
    if set(exact_by_id) != expected_ids:
        raise ValueError("external SVD target exact coverage drift")
    return {
        "schema_version": TARGET_SCHEMA,
        "panel_version": "external-svd-v24",
        "role": "same_speaker_target_scalar_required_by_candidate_loss",
        "source_commit": source_commit,
        "slurm_job_id": slurm_job_id,
        "panel_seal_sha256": panel_sha256,
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "target_exact_components_retained": ["shimmer_db"],
        "severity_labels_created": False,
        "exact_metric_highpass_branch_only": True,
        "emitted_waveform_highpass": False,
        "exact_scorer_versions": {
            "parselmouth": exact["parselmouth_version"],
            "praat": exact["praat_version"],
        },
        "avqi_code_tree_sha256": avqi_tree_sha256,
        "rows": [
            {
                "case_id": row["case_id"],
                "panel_speaker_id": row["panel_speaker_id"],
                "speaker_id": row["speaker_id"],
                "session_id": row["session_id"],
                "sex": row["sex"],
                "view": row["view"],
                "condition": row["condition"],
                "target_sha256": row["target_sha256"],
                "exact_target_shimmer_db": float(
                    exact_by_id[f"target:{row['case_id']}"][SHIMMER_DB_INDEX]
                ),
            }
            for row in rows
        ],
        "selector_stage_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
        "panel_authorization": {
            "opened24_v23_report_sha256": panel["authorization"][
                "opened24_report_sha256"
            ],
            "external_speaker_panel_authorized": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_provenance = validate_repository(args)
    panel_sha256 = validate_hash(
        args.panel_seal,
        args.panel_seal_sha256,
        "external SVD panel seal",
    )
    receipt_sha256 = validate_hash(
        args.seal_receipt,
        args.seal_receipt_sha256,
        "external SVD panel seal receipt",
    )
    panel = read_json(args.panel_seal)
    receipt = read_json(args.seal_receipt)
    rows = validate_panel_binding(
        panel,
        receipt,
        panel_sha256=panel_sha256,
    )
    if panel["source_commit"] != args.source_commit:
        raise ValueError("v25 source commit differs from the v24 panel seal")
    for row in rows:
        validate_hash(
            Path(row["target_path"]),
            row["target_sha256"],
            f"target waveform {row['case_id']}",
        )
    observed_avqi_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)

    args.output_dir.mkdir(parents=True)
    exact = run_exact_batch(
        target_exact_items(rows),
        args.exact_python,
        args.avqi_code_root,
    )
    target_contract = build_target_contract(
        panel,
        rows,
        exact,
        panel_sha256=panel_sha256,
        source_commit=args.source_commit,
        slurm_job_id=args.slurm_job_id,
        avqi_tree_sha256=observed_avqi_hash,
    )
    target_path = args.output_dir / "target_label_contract.json"
    write_json(target_path, target_contract)
    target_receipt = {
        "schema_version": TARGET_RECEIPT_SCHEMA,
        "decision": TARGET_DECISION,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "input_sha256": {
            "panel_seal.json": panel_sha256,
            "seal_receipt.json": receipt_sha256,
            "avqi_code_tree": observed_avqi_hash,
        },
        "target_exact_shimmer_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "selector_stage_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {
            target_path.name: sha256_file(target_path),
        },
    }
    receipt_path = args.output_dir / "target_completion_receipt.json"
    write_json(receipt_path, target_receipt)
    print(
        json.dumps(
            {
                "decision": TARGET_DECISION,
                "target_label_contract_sha256": sha256_file(target_path),
                "target_completion_receipt_sha256": sha256_file(receipt_path),
                "target_case_count": len(rows),
                "base_exact_outcomes_opened": False,
                "candidate_exact_outcomes_opened": False,
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
