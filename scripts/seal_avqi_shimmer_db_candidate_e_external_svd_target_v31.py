#!/usr/bin/env python3
"""Seal only Candidate-E external SVD target Shimmer-dB scalars (v25 successor).

The exact-unopened v30 panel is immutable input. This stage opens exact Praat
only for each same-speaker clean pathological target. It creates no candidate,
opens no base/candidate exact result, and creates no optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30 as v30
from scripts.evaluate_avqi_shimmer_fresh_panel import avqi_code_tree_sha256
from scripts.prepare_avqi_shimmer_db_external_svd_panel_v24 import (
    CONDITIONS,
    EXPECTED_CASES,
    EXPECTED_SPEAKERS,
    VIEWS,
    read_json,
    sha256_file,
    validate_hash,
    write_json,
)


TARGET_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-target-v31"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-target-receipt-v31"
)
TARGET_DECISION = (
    "SEALED_CANDIDATE_E_EXTERNAL_SVD_TARGET_SCALARS_SELECTOR_AUTHORIZED_V31"
)
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
EXACT_MARKER = "AVQI_CANDIDATE_E_TARGET_SHIMMER_JSON="
EXACT_TARGET_SCORER = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import parselmouth
from avqi_code.main import (
    get_shimmers,
    get_voiced_segments,
    highpass_filter,
    length_normalize_sv,
    read_and_resample_signal,
)

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    signal = read_and_resample_signal(item["path"], 16000)
    highpassed = highpass_filter("praat", signal, 16000)
    if item["view"] == "sv":
        metric = length_normalize_sv("praat", highpassed, 16000)
    elif item["view"] == "cs":
        metric = get_voiced_segments("praat", highpassed, 16000)
    else:
        raise ValueError(f"unsupported view: {item['view']}")
    _, shimmer_db = get_shimmers("praat", metric, 16000)
    rows.append({"id": item["id"], "shimmer_db": float(shimmer_db)})
print(
    "AVQI_CANDIDATE_E_TARGET_SHIMMER_JSON="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("panel-seal", "panel-receipt"):
        add_hashed_path(parser, option)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


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
        raise ValueError("repository root does not contain the v31 sealer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v31 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v31 target sealing requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "target_sealer_sha256": sha256_file(Path(__file__).resolve()),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")


def validate_panel_binding(
    panel: dict[str, Any],
    receipt: dict[str, Any],
    *,
    panel_sha256: str,
) -> list[dict[str, Any]]:
    if panel.get("schema_version") != v30.PANEL_SCHEMA:
        raise ValueError("Candidate-E external SVD panel schema drift")
    if receipt.get("schema_version") != v30.RECEIPT_SCHEMA:
        raise ValueError("Candidate-E external SVD panel receipt schema drift")
    if receipt.get("decision") != v30.PANEL_DECISION:
        raise ValueError("Candidate-E external SVD panel is not sealed")
    if receipt.get("artifact_sha256", {}).get("panel_seal_v30.json") != (
        panel_sha256
    ):
        raise ValueError("Candidate-E external panel receipt/seal binding drift")
    if panel.get("source_commit") != receipt.get("source_commit"):
        raise ValueError("Candidate-E external panel source commit drift")
    if panel.get("scientific_stage_mapping") != "v24_prepare_and_seal":
        raise ValueError("Candidate-E external panel stage mapping drift")
    if panel.get("case_count") != EXPECTED_CASES:
        raise ValueError("Candidate-E external panel case count drift")
    if panel.get("speaker_count") != EXPECTED_SPEAKERS:
        raise ValueError("Candidate-E external panel speaker count drift")
    if panel.get("severity_labels_created") is not False:
        raise ValueError("Candidate-E external panel invented severity labels")
    selection = panel.get("selection", {})
    expected_selection = {
        "selection_mode": "metadata_only_result_blind",
        "selection_uses_diagnosis": False,
        "selection_uses_shimmer_or_avqi": False,
        "prior_ledger_excluded_before_hash_ranking": True,
        "prior_panel_speaker_overlap": 0,
        "paired_cs_sv_same_session_required": True,
    }
    for field, value in expected_selection.items():
        if selection.get(field) != value:
            raise ValueError(f"Candidate-E result-blind selection drift: {field}")
    authorization = panel.get("authorization", {})
    if authorization.get("candidate_e_v29_decision") != v30.v29.PASS_DECISION:
        raise ValueError("Candidate-E external panel lacks v29 authorization")
    if authorization.get("external_panel_prepare_authorized") is not True:
        raise ValueError("Candidate-E external prepare authorization drift")
    if authorization.get("old_v23_no_go_not_reinterpreted") is not True:
        raise ValueError("Candidate-E external panel reinterprets v23 NO_GO")
    waveform = panel.get("waveform_contract", {})
    expected_waveform = {
        "emitted_waveform_highpass": False,
        "exact_metric_highpass_branch_only": True,
        "target_is_same_speaker_same_view_clean_pathological": True,
        "full_band_pathology_guardrails_required_later": True,
        "denoising_nonregression_required_later": True,
    }
    if waveform != expected_waveform:
        raise ValueError("Candidate-E external panel waveform contract drift")
    exact = panel.get("exact_contract", {})
    expected_exact = {
        "target_shimmer_values_opened": False,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "promotion_authorized": False,
    }
    if exact != expected_exact:
        raise ValueError("Candidate-E external panel exact-opening drift")
    if (
        receipt.get("exact_shimmer_outcomes_opened") is not False
        or receipt.get("target_scalar_stage_authorized") is not True
        or receipt.get("selector_stage_authorized") is not False
    ):
        raise ValueError("Candidate-E external panel receipt exact-opening drift")
    for label, value in (("panel", panel), ("receipt", receipt)):
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"Candidate-E external {label} promotes early")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"Candidate-E external {label} authorizes joint panel")
        require_training_boundary(value, f"Candidate-E external {label}")
    rows = panel.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise ValueError("Candidate-E external panel row coverage drift")
    case_ids: set[str] = set()
    speakers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Candidate-E external panel row is not an object")
        case_id = str(row.get("case_id", ""))
        speaker = str(row.get("panel_speaker_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError("Candidate-E external panel case identity drift")
        case_ids.add(case_id)
        if row.get("dataset") != "SVD" or speaker != f"SVD:{row.get('speaker_id')}":
            raise ValueError(f"Candidate-E external speaker drift: {case_id}")
        if row.get("label") != "patient":
            raise ValueError(f"Candidate-E external patient mapping drift: {case_id}")
        if "severity" in row or "sample_group" in row:
            raise ValueError(f"Candidate-E external severity leakage: {case_id}")
        if row.get("view") not in VIEWS or row.get("condition") not in CONDITIONS:
            raise ValueError(f"Candidate-E external slice drift: {case_id}")
        if not str(row.get("target_sha256", "")):
            raise ValueError(f"Candidate-E external target hash missing: {case_id}")
        speakers.add(speaker)
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError("Candidate-E external speaker coverage drift")
    if Counter(row["view"] for row in rows) != Counter({"cs": 6, "sv": 6}):
        raise ValueError("Candidate-E external view balance drift")
    if Counter(row["condition"] for row in rows) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError("Candidate-E external condition balance drift")
    if Counter(row["sex"] for row in rows) != Counter(
        {"female": 6, "male": 6}
    ):
        raise ValueError("Candidate-E external sex balance drift")
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


def run_target_shimmer_exact(
    items: list[dict[str, Any]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(exact_python), "-c", EXACT_TARGET_SCORER, str(avqi_code_root)],
        input=json.dumps({"items": items}, ensure_ascii=False),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "target-only exact Shimmer scorer failed: "
            + completed.stderr[-4000:]
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(EXACT_MARKER)
    ]
    if len(lines) != 1:
        raise RuntimeError("target-only exact Shimmer scorer marker drift")
    payload = json.loads(lines[0][len(EXACT_MARKER) :])
    expected_ids = [str(item["id"]) for item in items]
    observed_ids = [str(row["id"]) for row in payload.get("rows", [])]
    if observed_ids != expected_ids:
        raise ValueError("target-only exact Shimmer coverage/order drift")
    for row in payload["rows"]:
        if not isinstance(row.get("shimmer_db"), (int, float)):
            raise ValueError(f"target Shimmer scalar is not numeric: {row['id']}")
        if not math.isfinite(float(row["shimmer_db"])):
            raise ValueError(f"target Shimmer scalar is non-finite: {row['id']}")
    return payload


def build_target_contract(
    panel: dict[str, Any],
    rows: list[dict[str, Any]],
    exact: dict[str, Any],
    *,
    panel_sha256: str,
    panel_receipt_sha256: str,
    source_commit: str,
    slurm_job_id: str,
    avqi_tree_sha256: str,
) -> dict[str, Any]:
    exact_by_id = {str(row["id"]): row for row in exact["rows"]}
    expected_ids = {f"target:{row['case_id']}" for row in rows}
    if set(exact_by_id) != expected_ids:
        raise ValueError("Candidate-E external target exact coverage drift")
    return {
        "schema_version": TARGET_SCHEMA,
        "scientific_stage_mapping": "v25_target_scalar_seal",
        "role": "same_speaker_clean_pathological_target_scalar",
        "source_commit": source_commit,
        "panel_source_commit": panel["source_commit"],
        "slurm_job_id": slurm_job_id,
        "panel_seal_sha256": panel_sha256,
        "panel_receipt_sha256": panel_receipt_sha256,
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
                    exact_by_id[f"target:{row['case_id']}"]["shimmer_db"]
                ),
            }
            for row in rows
        ],
        "selector_stage_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "panel_authorization": {
            "v30_panel_decision": v30.PANEL_DECISION,
            "candidate_e_v29_decision": panel["authorization"][
                "candidate_e_v29_decision"
            ],
            "v29_report_sha256": panel["authorization"]["v29_report_sha256"],
            "v29_receipt_sha256": panel["authorization"]["v29_receipt_sha256"],
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
        "Candidate-E external SVD panel seal",
    )
    panel_receipt_sha256 = validate_hash(
        args.panel_receipt,
        args.panel_receipt_sha256,
        "Candidate-E external SVD panel receipt",
    )
    panel = read_json(args.panel_seal)
    panel_receipt = read_json(args.panel_receipt)
    rows = validate_panel_binding(
        panel,
        panel_receipt,
        panel_sha256=panel_sha256,
    )
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
    exact = run_target_shimmer_exact(
        target_exact_items(rows),
        args.exact_python,
        args.avqi_code_root,
    )
    target_contract = build_target_contract(
        panel,
        rows,
        exact,
        panel_sha256=panel_sha256,
        panel_receipt_sha256=panel_receipt_sha256,
        source_commit=args.source_commit,
        slurm_job_id=args.slurm_job_id,
        avqi_tree_sha256=observed_avqi_hash,
    )
    target_path = args.output_dir / "target_scalar_seal_v31.json"
    write_json(target_path, target_contract)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": TARGET_DECISION,
        "source_commit": args.source_commit,
        "panel_source_commit": panel["source_commit"],
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "input_sha256": {
            "panel_seal_v30.json": panel_sha256,
            "seal_receipt_v30.json": panel_receipt_sha256,
            "avqi_code_tree": observed_avqi_hash,
        },
        "target_exact_shimmer_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "selector_stage_authorized": True,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {
            target_path.name: sha256_file(target_path),
        },
    }
    receipt_path = args.output_dir / "target_completion_receipt_v31.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": TARGET_DECISION,
                "target_scalar_seal_sha256": sha256_file(target_path),
                "target_completion_receipt_sha256": sha256_file(receipt_path),
                "target_case_count": len(rows),
                "base_exact_outcomes_opened": False,
                "candidate_exact_outcomes_opened": False,
                "generator_optimizer_steps": 0,
                "authoritative_training_decision": TRAINING_DECISION,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
