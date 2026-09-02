#!/usr/bin/env python3
"""Seal Candidate-E external target Shimmer-dB scalars after v30r2.

This is the versioned successor of scientific v25 and Candidate-E v31.  It
accepts only the sealed v30r2 target-scorability panel, opens exact Praat for
the same-speaker clean pathological target dB scalar, and opens no base or
candidate exact outcome.  It creates no optimizer and performs no selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30r2 as v30r2
from scripts import seal_avqi_shimmer_db_candidate_e_external_svd_target_v31 as v31


TARGET_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-target-v31r2"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-target-receipt-v31r2"
)
TARGET_DECISION = (
    "SEALED_CANDIDATE_E_EXTERNAL_SVD_TARGET_SCALARS_SELECTOR_AUTHORIZED_V31R2"
)
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"


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
        raise ValueError("repository root does not contain the v31r2 sealer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v31r2 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v31r2 target sealing requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "target_sealer_sha256": v31.sha256_file(Path(__file__).resolve()),
        "authoritative_target_scorer_sha256": v31.sha256_file(
            Path(v31.__file__).resolve()
        ),
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
    if panel.get("schema_version") != v30r2.PANEL_SCHEMA:
        raise ValueError("Candidate-E v30r2 panel schema drift")
    if receipt.get("schema_version") != v30r2.RECEIPT_SCHEMA:
        raise ValueError("Candidate-E v30r2 panel receipt schema drift")
    if receipt.get("decision") != v30r2.PANEL_DECISION:
        raise ValueError("Candidate-E v30r2 panel is not sealed")
    if receipt.get("artifact_sha256", {}).get("panel_seal_v30r2.json") != (
        panel_sha256
    ):
        raise ValueError("Candidate-E v30r2 receipt/seal binding drift")
    if panel.get("source_commit") != receipt.get("source_commit"):
        raise ValueError("Candidate-E v30r2 source commit drift")
    if panel.get("scientific_stage_mapping") != "v24_prepare_and_seal":
        raise ValueError("Candidate-E v30r2 stage mapping drift")
    if panel.get("case_count") != v31.EXPECTED_CASES:
        raise ValueError("Candidate-E v30r2 case count drift")
    if panel.get("speaker_count") != v31.EXPECTED_SPEAKERS:
        raise ValueError("Candidate-E v30r2 speaker count drift")
    if panel.get("severity_labels_created") is not False:
        raise ValueError("Candidate-E v30r2 invented severity labels")
    selection = panel.get("selection", {})
    expected_selection = {
        "selection_mode": "frozen_rank_then_target_scorability_boolean_only",
        "selection_uses_diagnosis": False,
        "selection_uses_severity": False,
        "selection_uses_target_scalar_values": False,
        "selection_uses_target_scorability_boolean": True,
        "selection_uses_base_or_candidate_exact_outcomes": False,
        "slot_assignment_preserves_retained_v30_recipe_mapping": True,
        "prior_ledger_excluded_before_hash_ranking": True,
        "prior_panel_speaker_overlap": 0,
        "paired_cs_sv_same_session_required": True,
    }
    for field, value in expected_selection.items():
        if selection.get(field) != value:
            raise ValueError(f"Candidate-E v30r2 selection drift: {field}")
    selected = set(selection.get("selected_speakers", []))
    retained = set(selection.get("retained_v30_speakers", []))
    rejected = set(selection.get("rejected_v30_speakers", []))
    replacements = set(selection.get("replacement_speakers", []))
    if retained | replacements != selected or retained & replacements:
        raise ValueError("Candidate-E v30r2 selected-speaker partition drift")
    if not rejected or len(rejected) != len(replacements):
        raise ValueError("Candidate-E v30r2 amendment replacement drift")
    authorization = panel.get("authorization", {})
    if authorization.get("candidate_e_v29_decision") != v30r2.v29.PASS_DECISION:
        raise ValueError("Candidate-E v30r2 lacks v29 authorization")
    if authorization.get("external_panel_prepare_authorized") is not True:
        raise ValueError("Candidate-E v30r2 prepare authorization drift")
    if authorization.get("old_v23_no_go_not_reinterpreted") is not True:
        raise ValueError("Candidate-E v30r2 reinterprets v23 NO_GO")
    waveform = panel.get("waveform_contract", {})
    expected_waveform = {
        "emitted_waveform_highpass": False,
        "exact_metric_highpass_branch_only": True,
        "target_is_same_speaker_same_view_clean_pathological": True,
        "full_band_pathology_guardrails_required_later": True,
        "denoising_nonregression_required_later": True,
    }
    if waveform != expected_waveform:
        raise ValueError("Candidate-E v30r2 waveform contract drift")
    exact = panel.get("exact_contract", {})
    expected_exact = {
        "target_shimmer_scalar_values_opened": False,
        "target_scorability_boolean_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "promotion_authorized": False,
    }
    if exact != expected_exact:
        raise ValueError("Candidate-E v30r2 exact-opening drift")
    if (
        receipt.get("target_shimmer_scalar_values_opened") is not False
        or receipt.get("target_scorability_boolean_opened") is not True
        or receipt.get("base_exact_outcomes_opened") is not False
        or receipt.get("candidate_exact_outcomes_opened") is not False
        or receipt.get("target_scalar_stage_authorized") is not True
        or receipt.get("selector_stage_authorized") is not False
    ):
        raise ValueError("Candidate-E v30r2 receipt exact-opening drift")
    scorability_hashes = panel.get("scorability_artifact_sha256", {})
    receipt_hashes = receipt.get("artifact_sha256", {})
    for name in (
        "target_scorability_preflight_v30r2.json",
        "target_scorability_confirmation_v30r2.json",
    ):
        if not str(scorability_hashes.get(name, "")) or (
            receipt_hashes.get(name) != scorability_hashes[name]
        ):
            raise ValueError(f"Candidate-E v30r2 scorability binding drift: {name}")
    for label, value in (("panel", panel), ("receipt", receipt)):
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"Candidate-E v30r2 {label} promotes early")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"Candidate-E v30r2 {label} authorizes joint panel")
        require_training_boundary(value, f"Candidate-E v30r2 {label}")
    rows = panel.get("rows")
    if not isinstance(rows, list) or len(rows) != v31.EXPECTED_CASES:
        raise ValueError("Candidate-E v30r2 row coverage drift")
    case_ids: set[str] = set()
    speakers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Candidate-E v30r2 row is not an object")
        case_id = str(row.get("case_id", ""))
        speaker = str(row.get("panel_speaker_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError("Candidate-E v30r2 case identity drift")
        case_ids.add(case_id)
        if row.get("dataset") != "SVD" or speaker != f"SVD:{row.get('speaker_id')}":
            raise ValueError(f"Candidate-E v30r2 speaker drift: {case_id}")
        if row.get("label") != "patient":
            raise ValueError(f"Candidate-E v30r2 patient mapping drift: {case_id}")
        if "severity" in row or "sample_group" in row:
            raise ValueError(f"Candidate-E v30r2 severity leakage: {case_id}")
        if row.get("view") not in v31.VIEWS or row.get("condition") not in (
            v31.CONDITIONS
        ):
            raise ValueError(f"Candidate-E v30r2 slice drift: {case_id}")
        if not str(row.get("target_sha256", "")):
            raise ValueError(f"Candidate-E v30r2 target hash missing: {case_id}")
        speakers.add(speaker)
    if speakers != selected:
        raise ValueError("Candidate-E v30r2 selected-speaker/row drift")
    if Counter(row["view"] for row in rows) != Counter({"cs": 6, "sv": 6}):
        raise ValueError("Candidate-E v30r2 view balance drift")
    if Counter(row["condition"] for row in rows) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError("Candidate-E v30r2 condition balance drift")
    if Counter(row["sex"] for row in rows) != Counter(
        {"female": 6, "male": 6}
    ):
        raise ValueError("Candidate-E v30r2 sex balance drift")
    return rows


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
        raise ValueError("Candidate-E v31r2 target exact coverage drift")
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
            "v30r2_panel_decision": v30r2.PANEL_DECISION,
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
    panel_sha256 = v31.validate_hash(
        args.panel_seal,
        args.panel_seal_sha256,
        "Candidate-E v30r2 external SVD panel seal",
    )
    panel_receipt_sha256 = v31.validate_hash(
        args.panel_receipt,
        args.panel_receipt_sha256,
        "Candidate-E v30r2 external SVD panel receipt",
    )
    panel = v31.read_json(args.panel_seal)
    panel_receipt = v31.read_json(args.panel_receipt)
    rows = validate_panel_binding(
        panel,
        panel_receipt,
        panel_sha256=panel_sha256,
    )
    for row in rows:
        v31.validate_hash(
            Path(row["target_path"]),
            row["target_sha256"],
            f"target waveform {row['case_id']}",
        )
    observed_avqi_hash = v31.avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)

    args.output_dir.mkdir(parents=True)
    exact = v31.run_target_shimmer_exact(
        v31.target_exact_items(rows),
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
    target_path = args.output_dir / "target_scalar_seal_v31r2.json"
    v31.write_json(target_path, target_contract)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": TARGET_DECISION,
        "source_commit": args.source_commit,
        "panel_source_commit": panel["source_commit"],
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "input_sha256": {
            "panel_seal_v30r2.json": panel_sha256,
            "seal_receipt_v30r2.json": panel_receipt_sha256,
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
            target_path.name: v31.sha256_file(target_path),
        },
    }
    receipt_path = args.output_dir / "target_completion_receipt_v31r2.json"
    v31.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": TARGET_DECISION,
                "target_scalar_seal_sha256": v31.sha256_file(target_path),
                "target_completion_receipt_sha256": v31.sha256_file(
                    receipt_path
                ),
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
