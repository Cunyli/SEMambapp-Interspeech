#!/usr/bin/env python3
"""Seal an unused-speaker Route C panel for six-gradient fusion validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_gradient_fusion import (
    CAP_POLICY,
    CONFLICT_POLICY,
    FUSION_SCHEMA_VERSION,
    JOINT_NORMALIZATION,
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.evaluate_avqi_route_c_multicomponent_gradients import load_label_bank


CONTRACT_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-contract-v1"
PANEL_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-panel-seal-v1"
RECEIPT_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-panel-receipt-v1"
PANEL_DECISION = "SEALED_UNUSED_SPEAKER_SIX_GRADIENT_FUSION_PANEL_V1"
OPENED_DECISION = "NO_GO_ROUTE_C_SIX_ACTIVE_CODE_GRADIENT_AUDIT"
OPENED_RAW_DECISION = "PENDING_ROUTE_C_SIX_COMPONENT_GRADIENT_GATES_UNFROZEN"
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
SOURCE_LABEL_BANK_SHA256 = (
    "03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760"
)
OPENED_DECISION_REPORT_SHA256 = (
    "59b0e19a389f2008384b4eb67c37c43a5066e964f38c1a8815d14cd64d3581b7"
)
OPENED_DECISION_RECEIPT_SHA256 = (
    "bee54284f399eb2746c654739f55f9c59ec91f842914a81903fdd7891f3bab54"
)
OPENED_RAW_REPORT_SHA256 = (
    "ece23eadd03c268b7da223fa7a9e50aac8361d52e0fbf5f1bcdd1aa701266874"
)
OPENED_RAW_RECEIPT_SHA256 = (
    "239bfbc98274faa16733468da14bf87f5e4370bc6106a74cf792845d18dd752e"
)
ACCEPTED_SOURCE_BASE = "064a9dcd11443e1447cf0b1257fdd974ebe369a5"
CONTRACT_RELATIVE_PATH = Path(
    "configs/avqi_route_c_six_gradient_fusion_contract_v1.json"
)
AUDIT_SPLITS = ("surrogate_calibration", "surrogate_holdout")
EXPECTED_CASES_PER_SPLIT = 4
EXPECTED_CASES = EXPECTED_CASES_PER_SPLIT * len(AUDIT_SPLITS)
EXPECTED_STRATA = (
    "pathological_mild/cs",
    "pathological_mild/sv",
    "pathological_severe/cs",
    "pathological_severe/sv",
)
LOSS_TARGET = (
    "normalized bidirectional gap to same-speaker clean pathological CS/SV target"
)
BASE_WEIGHT_RULE = (
    "minimum calibration median gradient norm / component median gradient norm"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-label-bank", type=Path, required=True)
    parser.add_argument("--source-label-bank-sha256", required=True)
    parser.add_argument("--opened-raw-report", type=Path, required=True)
    parser.add_argument("--opened-raw-report-sha256", required=True)
    parser.add_argument("--opened-raw-receipt", type=Path, required=True)
    parser.add_argument("--opened-raw-receipt-sha256", required=True)
    parser.add_argument("--opened-decision-report", type=Path, required=True)
    parser.add_argument("--opened-decision-report-sha256", required=True)
    parser.add_argument("--opened-decision-receipt", type=Path, required=True)
    parser.add_argument("--opened-decision-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping")
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
        raise ValueError(f"{label} SHA-256 is invalid")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def _repository_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_source(root: Path, expected_commit: str) -> dict[str, str]:
    resolved = root.resolve()
    head = _repository_value(resolved, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("fusion panel source HEAD differs")
    if _repository_value(resolved, "status", "--porcelain"):
        raise ValueError("fusion panel seal requires a clean source tree")
    subprocess.run(
        [
            "git",
            "-C",
            str(resolved),
            "merge-base",
            "--is-ancestor",
            ACCEPTED_SOURCE_BASE,
            head,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "root": str(resolved),
        "head": head,
        "branch": _repository_value(resolved, "branch", "--show-current"),
        "accepted_base_commit": ACCEPTED_SOURCE_BASE,
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("fusion contract schema differs")
    if contract.get("decision") != "FROZEN_BEFORE_NEW_FUSION_PANEL_OPEN":
        raise ValueError("fusion contract was not frozen before panel open")
    if (
        contract.get("route") != "C"
        or tuple(contract.get("component_order", ()))
        != tuple(ROUTE_C_SIX_ACTIVE_COMPONENTS)
        or contract.get("loss_target") != LOSS_TARGET
        or contract.get("base_weight_rule") != BASE_WEIGHT_RULE
    ):
        raise ValueError("fusion contract task semantics differ")
    fusion_rule = contract.get("fusion_rule")
    if not isinstance(fusion_rule, dict) or fusion_rule != {
        "schema_version": FUSION_SCHEMA_VERSION,
        "maximum_weighted_component_norm_share": (
            MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE
        ),
        "threshold_source": "unchanged frozen five-component numeric precedent",
        "cap_scope": "per_case_before_joint_sum",
        "cap_policy": CAP_POLICY,
        "component_amplification_allowed": False,
        "joint_normalization": JOINT_NORMALIZATION,
        "pairwise_negative_cosines": "diagnostic_only",
        "direction_conflict_policy": CONFLICT_POLICY,
        "speaker_or_case_identity_used_for_fusion": False,
        "exact_candidate_outcome_used_for_fusion": False,
        "learned_fusion_parameters": 0,
    }:
        raise ValueError("fusion contract rule differs")
    if contract.get("opened_failure_use") != {
        "allowed": [
            "identify the general dominance failure class",
            "exclude all previously opened gradient-audit speakers",
        ],
        "forbidden": [
            "choose component-specific multipliers",
            "choose or relax the 0.80 threshold",
            "route by speaker or case identity",
            "reuse the opened holdout as promotion evidence",
        ],
    }:
        raise ValueError("fusion contract opened-failure use differs")
    opened = contract.get("opened_evidence")
    if not isinstance(opened, dict):
        raise ValueError("fusion contract opened evidence is unavailable")
    expected_hashes = {
        "decision_report_sha256": OPENED_DECISION_REPORT_SHA256,
        "decision_receipt_sha256": OPENED_DECISION_RECEIPT_SHA256,
        "raw_report_sha256": OPENED_RAW_REPORT_SHA256,
        "raw_receipt_sha256": OPENED_RAW_RECEIPT_SHA256,
    }
    if opened.get("decision") != OPENED_DECISION or any(
        opened.get(key) != digest for key, digest in expected_hashes.items()
    ):
        raise ValueError("fusion contract opened evidence differs")
    exclusions = opened.get("excluded_speakers_by_split")
    if not isinstance(exclusions, dict) or set(exclusions) != set(AUDIT_SPLITS):
        raise ValueError("fusion contract exclusion splits differ")
    parsed_exclusions = {}
    for split in AUDIT_SPLITS:
        speakers = exclusions[split]
        if (
            not isinstance(speakers, list)
            or len(speakers) != EXPECTED_CASES_PER_SPLIT
            or any(not isinstance(speaker, str) or not speaker for speaker in speakers)
            or len(set(speakers)) != len(speakers)
        ):
            raise ValueError(f"fusion contract {split} exclusions differ")
        parsed_exclusions[split] = sorted(speakers)
    new_panel = contract.get("new_panel")
    if not isinstance(new_panel, dict):
        raise ValueError("fusion contract new panel is unavailable")
    if (
        new_panel.get("source_label_bank_sha256") != SOURCE_LABEL_BANK_SHA256
        or new_panel.get("cases_by_split")
        != {split: EXPECTED_CASES_PER_SPLIT for split in AUDIT_SPLITS}
        or new_panel.get("strata_per_split") != list(EXPECTED_STRATA)
        or new_panel.get("speaker_overlap") != 0
        or new_panel.get("exclude_all_opened_speakers") is not True
        or new_panel.get("seal_before_gradient_measurement") is not True
        or new_panel.get("same_speaker_clean_pathological_targets") is not True
        or not isinstance(new_panel.get("selection_salt"), str)
        or not new_panel["selection_salt"]
    ):
        raise ValueError("fusion contract new-panel policy differs")
    if contract.get("promotion_gates") != {
        "all_component_gradients_finite_nonzero_bounded": True,
        "all_post_cap_joint_gradients_finite_nonzero_bounded": True,
        "all_post_cap_weighted_component_shares_le_0_80": True,
        "all_post_cap_component_to_joint_cosines_nonnegative": True,
        "no_component_amplified": True,
        "only_unique_dominant_component_may_be_attenuated": True,
        "new_calibration_and_holdout_speaker_disjoint": True,
        "all_opened_speakers_excluded": True,
    }:
        raise ValueError("fusion contract promotion gates differ")
    boundaries = contract.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries != {
        "fresh_or_final_joint_panel_opened": False,
        "candidate_exact_outcomes_opened": False,
        "waveform_generation_performed": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }:
        raise ValueError("fusion contract boundaries differ")
    return {
        "excluded_speakers_by_split": parsed_exclusions,
        "selection_salt": new_panel["selection_salt"],
    }


def validate_opened_evidence(
    raw_report: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    decision_report: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    exclusions: Mapping[str, list[str]],
) -> None:
    if raw_report.get("decision") != OPENED_RAW_DECISION:
        raise ValueError("opened raw report decision differs")
    if raw_receipt.get("decision") != OPENED_RAW_DECISION:
        raise ValueError("opened raw receipt decision differs")
    if decision_report.get("decision") != OPENED_DECISION:
        raise ValueError("opened decision report decision differs")
    if decision_receipt.get("decision") != OPENED_DECISION:
        raise ValueError("opened decision receipt decision differs")
    raw_artifacts = raw_receipt.get("artifact_sha256")
    decision_artifacts = decision_receipt.get("artifact_sha256")
    if raw_artifacts != {
        "six_gradient_measurement_report.json": OPENED_RAW_REPORT_SHA256
    }:
        raise ValueError("opened raw receipt does not bind its report")
    if decision_artifacts != {
        "six_gradient_decision_report.json": OPENED_DECISION_REPORT_SHA256
    }:
        raise ValueError("opened decision receipt does not bind its report")
    raw_binding = decision_report.get("raw_measurement_evidence")
    if not isinstance(raw_binding, dict) or (
        raw_binding.get("report_sha256") != OPENED_RAW_REPORT_SHA256
        or raw_binding.get("receipt_sha256") != OPENED_RAW_RECEIPT_SHA256
        or raw_binding.get("raw_artifacts_rewritten") is not False
    ):
        raise ValueError("opened decision does not bind immutable raw evidence")
    selection = raw_report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("opened raw selection is unavailable")
    speakers = selection.get("speakers_by_split")
    if not isinstance(speakers, dict):
        raise ValueError("opened raw speakers are unavailable")
    for split in AUDIT_SPLITS:
        if sorted(speakers.get(split, [])) != exclusions[split]:
            raise ValueError(f"opened {split} speakers differ from exclusions")
    for value, label in (
        (raw_report, "opened raw report"),
        (raw_receipt, "opened raw receipt"),
        (decision_report, "opened decision report"),
        (decision_receipt, "opened decision receipt"),
    ):
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"{label} optimizer boundary differs")
        if value.get("authoritative_training_decision") != TRAINING_NO_GO:
            raise ValueError(f"{label} training boundary differs")


def filter_label_bank_rows(
    rows: list[dict[str, str]],
    excluded_speakers: set[str],
) -> tuple[list[dict[str, str]], int]:
    retained = []
    removed = 0
    for row in rows:
        if (
            row.get("split") in AUDIT_SPLITS
            and row.get("speaker_id") in excluded_speakers
        ):
            removed += 1
            continue
        retained.append(row)
    if removed <= 0:
        raise ValueError("fusion panel filtering removed no opened speaker rows")
    if any(
        row.get("split") in AUDIT_SPLITS
        and row.get("speaker_id") in excluded_speakers
        for row in retained
    ):
        raise RuntimeError("opened speaker survived fusion panel filtering")
    return retained, removed


def serialize_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite fusion panel seal: {args.output_dir}"
        )
    bindings = {
        "contract": (
            _verified_file(args.contract, args.contract_sha256, "fusion contract"),
            args.contract_sha256,
        ),
        "source_label_bank": (
            _verified_file(
                args.source_label_bank,
                args.source_label_bank_sha256,
                "source label bank",
            ),
            args.source_label_bank_sha256,
        ),
        "opened_raw_report": (
            _verified_file(
                args.opened_raw_report,
                args.opened_raw_report_sha256,
                "opened raw report",
            ),
            args.opened_raw_report_sha256,
        ),
        "opened_raw_receipt": (
            _verified_file(
                args.opened_raw_receipt,
                args.opened_raw_receipt_sha256,
                "opened raw receipt",
            ),
            args.opened_raw_receipt_sha256,
        ),
        "opened_decision_report": (
            _verified_file(
                args.opened_decision_report,
                args.opened_decision_report_sha256,
                "opened decision report",
            ),
            args.opened_decision_report_sha256,
        ),
        "opened_decision_receipt": (
            _verified_file(
                args.opened_decision_receipt,
                args.opened_decision_receipt_sha256,
                "opened decision receipt",
            ),
            args.opened_decision_receipt_sha256,
        ),
    }
    expected_cli_hashes = {
        "source_label_bank": SOURCE_LABEL_BANK_SHA256,
        "opened_raw_report": OPENED_RAW_REPORT_SHA256,
        "opened_raw_receipt": OPENED_RAW_RECEIPT_SHA256,
        "opened_decision_report": OPENED_DECISION_REPORT_SHA256,
        "opened_decision_receipt": OPENED_DECISION_RECEIPT_SHA256,
    }
    if any(bindings[name][1] != digest for name, digest in expected_cli_hashes.items()):
        raise ValueError("fusion panel immutable input SHA-256 differs")
    source = verify_source(args.source_root, args.source_commit)
    expected_contract_path = (
        Path(source["root"]) / CONTRACT_RELATIVE_PATH
    ).resolve()
    if bindings["contract"][0] != expected_contract_path:
        raise ValueError("fusion contract is not the clean source-tree contract")
    contract = _read_json(bindings["contract"][0], "fusion contract")
    policy = validate_contract(contract)
    raw_report = _read_json(bindings["opened_raw_report"][0], "opened raw report")
    raw_receipt = _read_json(bindings["opened_raw_receipt"][0], "opened raw receipt")
    decision_report = _read_json(
        bindings["opened_decision_report"][0], "opened decision report"
    )
    decision_receipt = _read_json(
        bindings["opened_decision_receipt"][0], "opened decision receipt"
    )
    validate_opened_evidence(
        raw_report,
        raw_receipt,
        decision_report,
        decision_receipt,
        policy["excluded_speakers_by_split"],
    )

    with bindings["source_label_bank"][0].open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError("source label bank is empty")
    excluded_union = {
        speaker
        for speakers in policy["excluded_speakers_by_split"].values()
        for speaker in speakers
    }
    filtered_rows, removed_rows = filter_label_bank_rows(rows, excluded_union)
    filtered_bytes = serialize_csv(fieldnames, filtered_rows)
    filtered_sha256 = _sha256_bytes(filtered_bytes)

    with tempfile.TemporaryDirectory(prefix="avqi-fusion-panel-") as temp_root:
        temp_path = Path(temp_root) / "filtered_label_bank.csv"
        temp_path.write_bytes(filtered_bytes)
        cases, _, _, selection = load_label_bank(
            temp_path,
            filtered_sha256,
            policy["selection_salt"],
        )
    selected_speakers = {case.speaker_id for case in cases}
    if selected_speakers & excluded_union:
        raise ValueError("new fusion panel selected an opened speaker")
    if len(cases) != EXPECTED_CASES:
        raise ValueError("new fusion panel case count differs")
    if selection.get("speaker_overlap") != 0:
        raise ValueError("new fusion panel speakers overlap across splits")

    observed_after = {
        name: sha256_file(path) for name, (path, _) in bindings.items()
    }
    expected_after = {
        name: digest for name, (_, digest) in bindings.items()
    }
    if observed_after != expected_after:
        raise ValueError("fusion panel input changed during sealing")

    case_rows = []
    for case in cases:
        target_values = [float(value) for value in case.clean_target.tolist()]
        target_payload = json.dumps(
            target_values,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        case_rows.append(
            {
                "case_id": ":".join(
                    (
                        "six-gradient-fusion-v1",
                        case.split,
                        case.speaker_id,
                        case.sample_id,
                        case.sample_group,
                        case.view,
                        case.condition,
                    )
                ),
                "split": case.split,
                "speaker_id": case.speaker_id,
                "sample_id": case.sample_id,
                "sample_group": case.sample_group,
                "view": case.view,
                "condition": case.condition,
                "source_audio_path": str(case.waveform_path.resolve()),
                "source_audio_file_sha256": case.waveform_sha256,
                "same_speaker_clean_pathological_target": target_values,
                "target_vector_sha256": hashlib.sha256(target_payload).hexdigest(),
            }
        )

    args.output_dir.mkdir(parents=True)
    filtered_path = args.output_dir / "filtered_exact_component_label_bank_v1.csv"
    filtered_path.write_bytes(filtered_bytes)
    seal = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "source": source,
        "selection_salt": policy["selection_salt"],
        "source_label_bank_sha256": SOURCE_LABEL_BANK_SHA256,
        "filtered_label_bank_sha256": filtered_sha256,
        "opened_evidence_sha256": {
            name: digest
            for name, digest in expected_cli_hashes.items()
            if name != "source_label_bank"
        },
        "contract_sha256": args.contract_sha256,
        "excluded_speakers_by_split": policy["excluded_speakers_by_split"],
        "excluded_speaker_union": sorted(excluded_union),
        "removed_label_bank_rows": removed_rows,
        "retained_label_bank_rows": len(filtered_rows),
        "selection": selection,
        "cases": case_rows,
        "opened_speaker_overlap": 0,
        "fusion_rule_frozen_before_panel_selection": True,
        "gradient_measurement_performed": False,
        "candidate_exact_outcomes_opened": False,
        "fresh_or_final_joint_panel_opened": False,
        "waveform_generation_performed": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    seal_path = args.output_dir / "fusion_panel_seal_v1.json"
    _write_json(seal_path, seal)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {
            filtered_path.name: sha256_file(filtered_path),
            seal_path.name: sha256_file(seal_path),
        },
        "input_sha256": {
            name: digest for name, (_, digest) in bindings.items()
        },
        "selected_cases": len(case_rows),
        "opened_speaker_overlap": 0,
        "candidate_exact_outcomes_opened": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    _write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": PANEL_DECISION,
                "selected_cases": len(case_rows),
                "selected_speakers": sorted(selected_speakers),
                "filtered_label_bank_sha256": filtered_sha256,
                "panel_seal_sha256": sha256_file(seal_path),
                "receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
