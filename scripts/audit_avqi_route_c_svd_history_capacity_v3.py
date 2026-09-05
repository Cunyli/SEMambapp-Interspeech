#!/usr/bin/env python3
"""Audit opened SVD history and fail closed on successor-panel capacity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from model.avqi_route_c_v19_contracts import sha256_file
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    repository_value,
    verify_source,
)
from scripts.seal_avqi_route_c_six_gradient_svd_source_panel_v2 import (
    eligible_svd_rows,
    validate_ledger,
)


CONTRACT_SCHEMA_VERSION = "avqi-route-c-svd-history-capacity-contract-v3"
REPORT_SCHEMA_VERSION = "avqi-route-c-svd-history-capacity-report-v3"
RECEIPT_SCHEMA_VERSION = "avqi-route-c-svd-history-capacity-receipt-v3"
LEDGER_SCHEMA_VERSION = "avqi-route-c-prior-panel-speaker-ledger-v1"
CONTRACT_DECISION = (
    "FROZEN_DIAGNOSTIC_AUDIT_OF_INCOMPLETE_SVD_HISTORY_BEFORE_ANY_"
    "SUCCESSOR_SELECTION_V3"
)
FAILURE_DECISION = (
    "NO_GO_ROUTE_C_SIX_GRADIENT_SVD_SOURCE_PANEL_INCOMPLETE_HISTORICAL_"
    "LEDGER_AND_INSUFFICIENT_UNUSED_SPEAKERS_V3"
)
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
V9_DECISION = "FAIL_EXTERNAL_SVD_LTAS_EXACT_COVERAGE"
V10_DECISION = "PASS_EXTERNAL_SVD_LTAS_AUTHORITY_PANEL_NO_PRODUCTION_CHANGE"
FRESH_DECISION = "PASS_LTAS_SLOPE_FRESH_SPEAKER_PANEL"
CANDIDATE_E_DECISION = "PASS_CANDIDATE_E_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V32R8"
INVALID_V2_DECISION = "SEALED_UNUSED_SPEAKER_SVD_SIX_GRADIENT_SOURCE_PANEL_V2"
SHIMMER_DB_LEDGER_SOURCE_KEY = "shimmer_db_external_svd_v24"
VIEWS = ("cs", "sv")
VARIANTS_PER_VIEW = 4
INPUT_KEYS = (
    "candidate_e_exact_csv",
    "candidate_e_exact_receipt",
    "candidate_e_exact_report",
    "candidate_e_prior_ledger",
    "cs_metadata",
    "invalid_v2_contract",
    "invalid_v2_receipt",
    "invalid_v2_source_seal",
    "invalid_v2_updated_ledger",
    "ltas_fresh_alpha_selection",
    "ltas_fresh_calibration_csv",
    "ltas_fresh_final_csv",
    "ltas_fresh_final_seal",
    "ltas_fresh_panel_contract",
    "ltas_fresh_receipt",
    "ltas_fresh_report",
    "sv_metadata",
    "v10_diagnostic_report",
    "v10_panel_seal",
    "v10_predictions",
    "v10_receipt",
    "v10_status_audit",
    "v9_diagnostic_report",
    "v9_panel_seal",
    "v9_predictions",
    "v9_receipt",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    for key in INPUT_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--cs-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accepted-base-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return value


def read_json_rows(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a JSON row list")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def canonical_svd_speaker(value: object) -> str:
    normalized = str(value).strip()
    if normalized.startswith("SVD:"):
        normalized = normalized.removeprefix("SVD:")
    if not normalized or ":" in normalized:
        raise ValueError("invalid SVD speaker identity")
    return f"SVD:{normalized}"


def canonical_speakers(values: Iterable[object]) -> set[str]:
    return {canonical_svd_speaker(value) for value in values}


def row_speakers(
    rows: Iterable[Mapping[str, Any]],
    *keys: str,
) -> set[str]:
    speakers = set()
    for row in rows:
        value = next((row.get(key) for key in keys if row.get(key) is not None), None)
        if value is None:
            raise ValueError("speaker identity is unavailable")
        speakers.add(canonical_svd_speaker(value))
    return speakers


def verified_inputs(
    arguments: argparse.Namespace,
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, str]]:
    if set(expected_hashes) != set(INPUT_KEYS):
        raise ValueError("history-capacity input inventory differs")
    paths = {}
    observed_hashes = {}
    for key in INPUT_KEYS:
        path = getattr(arguments, key).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing history-capacity input {key}: {path}")
        observed = sha256_file(path)
        if observed != expected_hashes[key]:
            raise ValueError(f"history-capacity input SHA-256 differs: {key}")
        paths[key] = path
        observed_hashes[key] = observed
    return paths, observed_hashes


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("decision") != CONTRACT_DECISION
        or contract.get("route") != "C"
        or contract.get("failure_decision") != FAILURE_DECISION
    ):
        raise ValueError("history-capacity contract identity differs")
    hashes = contract.get("input_sha256")
    evidence = contract.get("historical_exact_opening_evidence")
    expected = contract.get("expected_audit_result")
    gate = contract.get("source_panel_capacity_gate")
    boundaries = contract.get("audit_boundaries")
    if not all(
        isinstance(value, Mapping)
        for value in (hashes, evidence, expected, gate, boundaries)
    ):
        raise ValueError("history-capacity contract sections are unavailable")
    if set(hashes) != set(INPUT_KEYS):
        raise ValueError("history-capacity contract input keys differ")
    if (
        gate.get("required_cases") != 8
        or gate.get("required_distinct_speakers") != 8
        or gate.get("required_splits")
        != ["surrogate_calibration", "surrogate_holdout"]
        or gate.get("required_strata_per_split")
        != ["female/cs", "female/sv", "male/cs", "male/sv"]
        or gate.get("one_case_per_speaker") is not True
        or gate.get("minimum_raw_mono_duration_seconds")
        != {"cs": 3.0, "sv": 1.0}
        or gate.get("thresholds_may_not_be_weakened_after_capacity_failure")
        is not True
        or gate.get("successor_selection_authorized_only_if_capacity_passes")
        is not True
    ):
        raise ValueError("history-capacity scientific gate differs")
    required_boundaries = {
        "historical_open_status_used_only_for_exclusion_and_audit": True,
        "historical_exact_scalar_values_used_for_new_selection": False,
        "new_target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
        "new_waveforms_materialized": False,
        "six_gradient_evaluation_submitted": False,
        "joint_panel_submitted": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    if boundaries != required_boundaries:
        raise ValueError("history-capacity authorization boundary differs")
    return dict(contract)


def validate_v9(
    panel: Mapping[str, Any],
    report: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[set[str], dict[str, Any]]:
    policy = contract["historical_exact_opening_evidence"]["ltas_svd_authority_v9"]
    selection = panel.get("selection")
    failures = report.get("exact_failures")
    if not isinstance(selection, Mapping) or not isinstance(failures, list):
        raise ValueError("v9 exact-opening evidence structure differs")
    speakers = canonical_speakers(selection.get("speakers", ()))
    prediction_speakers = row_speakers(predictions, "panel_speaker_id", "speaker_id")
    expected_attempts = int(policy["expected_variant_attempts"])
    successful = round(float(report.get("exact_coverage", -1.0)) * expected_attempts)
    if (
        panel.get("exact_scores_opened") is not False
        or int(selection.get("speaker_count", -1)) != len(speakers)
        or len(speakers) != int(policy["expected_speakers_attempted"])
        or len(predictions) != len(speakers) * len(VIEWS)
        or prediction_speakers != speakers
        or report.get("decision") != V9_DECISION
        or str(report.get("slurm_job_id")) != policy["expected_score_job_id"]
        or successful + len(failures) != expected_attempts
        or not math.isclose(
            float(report.get("exact_coverage", -1.0)),
            successful / expected_attempts,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or receipt.get("decision") != V9_DECISION
        or receipt.get("panel_seal_sha256") != hashes["v9_panel_seal"]
        or receipt.get("artifact_sha256", {}).get("diagnostic_report.json")
        != hashes["v9_diagnostic_report"]
        or receipt.get("artifact_sha256", {}).get("predictions.json")
        != hashes["v9_predictions"]
    ):
        raise ValueError("v9 exact-opening evidence differs")
    return speakers, {
        "decision": V9_DECISION,
        "score_job_id": policy["expected_score_job_id"],
        "opened_speakers": sorted(speakers),
        "attempted_exact_variants": expected_attempts,
        "successful_exact_variants": successful,
        "failed_exact_variants": len(failures),
    }


def validate_v10(
    panel: Mapping[str, Any],
    report: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    receipt: Mapping[str, Any],
    status_audit: Mapping[str, Any],
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
    v9_speakers: set[str],
) -> tuple[set[str], dict[str, Any]]:
    policy = contract["historical_exact_opening_evidence"]["ltas_svd_authority_v10"]
    selection = panel.get("selection")
    substitution = report.get("status_only_substitution")
    exact_runtime = report.get("exact_runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (selection, substitution, exact_runtime)
    ):
        raise ValueError("v10 exact-opening evidence structure differs")
    primary = canonical_speakers(selection.get("primary_speakers", ()))
    excluded_v9 = canonical_speakers(selection.get("excluded_v9_speakers", ()))
    reserve_attempts = substitution.get("reserve_attempts")
    if not isinstance(reserve_attempts, list):
        raise ValueError("v10 reserve attempts are unavailable")
    reserves = canonical_speakers(
        attempt.get("reserve_speaker") for attempt in reserve_attempts
    )
    failed_primary = canonical_speakers(
        substitution.get("failed_primary_speakers", ())
    )
    selected = canonical_speakers(substitution.get("selected_speakers", ()))
    prediction_speakers = row_speakers(predictions, "panel_speaker_id", "speaker_id")
    opened = primary | reserves
    expected_attempts = int(policy["expected_variant_attempts"])
    if (
        panel.get("exact_scores_opened") is not False
        or len(primary) != int(policy["expected_primary_speakers_attempted"])
        or len(reserves) != int(policy["expected_reserve_speakers_attempted"])
        or excluded_v9 != v9_speakers
        or selected != (primary - failed_primary) | reserves
        or prediction_speakers != selected
        or len(predictions) != len(selected) * len(VIEWS)
        or report.get("decision") != V10_DECISION
        or str(report.get("slurm_job_id")) != policy["expected_score_job_id"]
        or int(exact_runtime.get("attempted_rows", -1)) != expected_attempts
        or expected_attempts != len(opened) * len(VIEWS) * VARIANTS_PER_VIEW
        or status_audit.get("decision")
        != "REPRODUCED_STATUS_ONLY_PRIMARY_INCOMPLETENESS"
        or canonical_svd_speaker(status_audit.get("failed_primary_speaker"))
        not in failed_primary
        or status_audit.get("metric_values_recorded") is not False
        or receipt.get("decision") != V10_DECISION
        or receipt.get("panel_seal_sha256") != hashes["v10_panel_seal"]
        or receipt.get("artifact_sha256", {}).get("diagnostic_report.json")
        != hashes["v10_diagnostic_report"]
        or receipt.get("artifact_sha256", {}).get("predictions.json")
        != hashes["v10_predictions"]
    ):
        raise ValueError("v10 exact-opening evidence differs")
    return opened, {
        "decision": V10_DECISION,
        "score_job_id": policy["expected_score_job_id"],
        "opened_speakers": sorted(opened),
        "primary_speakers_attempted": len(primary),
        "failed_primary_speakers": sorted(failed_primary),
        "reserve_speakers_attempted": sorted(reserves),
        "attempted_exact_variants": expected_attempts,
    }


def validate_fresh(
    panel: Mapping[str, Any],
    alpha_selection: Mapping[str, Any],
    calibration_rows: list[dict[str, str]],
    final_seal: Mapping[str, Any],
    final_rows: list[dict[str, str]],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[set[str], dict[str, Any]]:
    policy = contract["historical_exact_opening_evidence"]["ltas_fresh_panel_v1"]
    panel_rows = panel.get("rows")
    sealed_final_rows = final_seal.get("rows")
    if not isinstance(panel_rows, list) or not isinstance(sealed_final_rows, list):
        raise ValueError("LTAS fresh panel rows are unavailable")
    calibration = row_speakers(
        (row for row in panel_rows if row.get("split") == "calibration"),
        "speaker_id",
    )
    final = row_speakers(
        (row for row in panel_rows if row.get("split") == "final"),
        "speaker_id",
    )
    calibration_csv_speakers = row_speakers(calibration_rows, "speaker_id")
    final_csv_speakers = row_speakers(final_rows, "speaker_id")
    final_seal_speakers = row_speakers(sealed_final_rows, "speaker_id")
    exact_columns = ("exact_target_slope", "exact_before_slope", "exact_after_slope")
    if (
        str(panel.get("slurm_job_id")) != policy["expected_job_id"]
        or len(calibration) != int(policy["expected_calibration_speakers"])
        or len(final) != int(policy["expected_final_speakers"])
        or calibration
        != canonical_speakers(alpha_selection.get("calibration_speakers", ()))
        or calibration_csv_speakers != calibration
        or final_csv_speakers != final
        or final_seal_speakers != final
        or len(calibration_rows) != 54
        or len(final_rows) != 6
        or any(not row.get(key) for row in calibration_rows for key in exact_columns)
        or any(not row.get(key) for row in final_rows for key in exact_columns)
        or final_seal.get("exact_final_scoring_started_after_this_seal") is not True
        or report.get("decision") != FRESH_DECISION
        or report.get("final_exact_panel_opened") is not True
        or report.get("generator_optimizer_steps") != 0
        or receipt.get("decision") != FRESH_DECISION
        or receipt.get("generator_optimizer_steps") != 0
        or receipt.get("artifact_sha256", {}).get("panel_contract.json")
        != hashes["ltas_fresh_panel_contract"]
        or receipt.get("artifact_sha256", {}).get("calibration_alpha_results.csv")
        != hashes["ltas_fresh_calibration_csv"]
        or receipt.get("artifact_sha256", {}).get("final_results.csv")
        != hashes["ltas_fresh_final_csv"]
        or receipt.get("artifact_sha256", {}).get("fresh_panel_report.json")
        != hashes["ltas_fresh_report"]
    ):
        raise ValueError("LTAS fresh exact-opening evidence differs")
    opened = calibration | final
    return opened, {
        "decision": FRESH_DECISION,
        "job_id": policy["expected_job_id"],
        "opened_speakers": sorted(opened),
        "calibration_speakers": sorted(calibration),
        "final_speakers": sorted(final),
        "calibration_exact_rows": len(calibration_rows),
        "final_exact_rows": len(final_rows),
    }


def validate_candidate_e(
    rows: list[dict[str, str]],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[set[str], dict[str, Any]]:
    policy = contract["historical_exact_opening_evidence"]["candidate_e_external_v32r8"]
    speakers = row_speakers(rows, "panel_speaker_id")
    if (
        len(rows) != int(policy["expected_rows"])
        or len(speakers) != int(policy["expected_speakers"])
        or any(
            row.get("candidate_exact_opened_only_after_selector_seal") != "True"
            for row in rows
        )
        or report.get("decision") != CANDIDATE_E_DECISION
        or str(report.get("slurm_job_id")) != policy["expected_job_id"]
        or receipt.get("decision") != CANDIDATE_E_DECISION
        or str(receipt.get("slurm_job_id")) != policy["expected_job_id"]
        or receipt.get("generator_optimizer_steps") != 0
        or receipt.get("artifact_sha256", {}).get(
            "external_svd_exact_results_v32r8.csv"
        )
        != hashes["candidate_e_exact_csv"]
        or receipt.get("artifact_sha256", {}).get("external_svd_report_v32r8.json")
        != hashes["candidate_e_exact_report"]
    ):
        raise ValueError("Candidate-E exact-opening evidence differs")
    return speakers, {
        "decision": CANDIDATE_E_DECISION,
        "job_id": policy["expected_job_id"],
        "opened_speakers": sorted(speakers),
        "exact_rows": len(rows),
        "exact_opened_only_after_selector_seal": True,
    }


def validate_invalid_v2(
    invalid_contract: Mapping[str, Any],
    seal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    updated_ledger: Mapping[str, Any],
    original_ledger: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[set[str], dict[str, Any]]:
    rows = seal.get("rows")
    if not isinstance(rows, list):
        raise ValueError("invalid v2 source rows are unavailable")
    selected = row_speakers(rows, "canonical_speaker_id")
    updated_speakers = validate_ledger(updated_ledger)
    original_speakers = validate_ledger(original_ledger)
    original_by_speaker = {
        entry["canonical_speaker_id"]: entry for entry in original_ledger["entries"]
    }
    updated_by_speaker = {
        entry["canonical_speaker_id"]: entry for entry in updated_ledger["entries"]
    }
    if (
        invalid_contract.get("decision")
        != "FROZEN_BEFORE_EXTERNAL_SVD_SOURCE_PANEL_SELECTION"
        or seal.get("decision") != INVALID_V2_DECISION
        or len(selected) != 8
        or seal.get("selection", {}).get("target_scalar_values_used") is not False
        or seal.get("selection", {}).get("base_or_candidate_exact_outcomes_used")
        is not False
        or receipt.get("decision") != INVALID_V2_DECISION
        or receipt.get("slurm_job_id") != "20083880"
        or receipt.get("target_scalar_values_opened") is not False
        or receipt.get("base_or_candidate_exact_outcomes_opened") is not False
        or receipt.get("generator_optimizer_steps") != 0
        or receipt.get("artifact_sha256", {}).get("svd_source_panel_seal_v2.json")
        != hashes["invalid_v2_source_seal"]
        or receipt.get("artifact_sha256", {}).get(
            "prior_speaker_ledger_after_svd_v2.json"
        )
        != hashes["invalid_v2_updated_ledger"]
        or updated_speakers != original_speakers | selected
        or any(
            updated_by_speaker[speaker] != entry
            for speaker, entry in original_by_speaker.items()
        )
    ):
        raise ValueError("invalid v2 source-panel evidence differs")
    return selected, {
        "decision_at_creation": INVALID_V2_DECISION,
        "job_id": receipt["slurm_job_id"],
        "selected_speakers": sorted(selected),
        "target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
        "waveforms_materialized": False,
    }


def history_sources_by_speaker(
    sources: Mapping[str, set[str]],
) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for source, speakers in sources.items():
        for speaker in speakers:
            indexed.setdefault(speaker, []).append(source)
    return {
        speaker: sorted(source_names)
        for speaker, source_names in sorted(indexed.items())
    }


def merged_history_ledger(
    original: Mapping[str, Any],
    original_sha256: str,
    sources_by_speaker: Mapping[str, list[str]],
    input_hashes: Mapping[str, str],
    source_commit: str,
) -> dict[str, Any]:
    original_speakers = validate_ledger(original)
    entries = [dict(entry) for entry in original["entries"]]
    for canonical, source_names in sources_by_speaker.items():
        if canonical in original_speakers:
            continue
        entries.append(
            {
                "dataset": "SVD",
                "speaker_id": canonical.removeprefix("SVD:"),
                "canonical_speaker_id": canonical,
                "panel_role": "historical_exact_opened_before_six_gradient_svd_v2",
                "historical_exact_opened_sources": list(source_names),
                "exact_opened_before_successor_selection": True,
            }
        )
    historical_speakers = set(sources_by_speaker)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "exact_outcomes_used_for_selection": False,
        "entries": sorted(entries, key=lambda entry: entry["canonical_speaker_id"]),
        "source_ledger_sha256": {
            SHIMMER_DB_LEDGER_SOURCE_KEY: original_sha256,
        },
        "historical_exact_opened_sources_by_speaker": dict(sources_by_speaker),
        "historical_exact_opened_svd_speaker_count": len(historical_speakers),
        "added_by": "svd_history_capacity_audit_v3",
        "added_speaker_count": len(historical_speakers - original_speakers),
        "audit_input_sha256": dict(input_hashes),
        "source_commit": source_commit,
        "new_source_selection_performed": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def existing_scorability_by_speaker(
    source_seal: Mapping[str, Any],
) -> dict[str, dict[str, bool]]:
    audit = source_seal.get("target_scorability_audit")
    rows = audit.get("rows") if isinstance(audit, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("invalid v2 scorability audit is unavailable")
    output: dict[str, dict[str, bool]] = {}
    for row in rows:
        identifier = str(row.get("id", ""))
        parts = identifier.split(":")
        if len(parts) != 3 or parts[0] != "SVD" or parts[2] not in VIEWS:
            raise ValueError("invalid v2 scorability row identity differs")
        output.setdefault(f"SVD:{parts[1]}", {})[parts[2]] = bool(
            row.get("all_six_components_scorable")
        )
    return output


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite SVD history-capacity audit: {args.output_dir}"
        )
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SVD history-capacity audit must run under Slurm")
    contract_path = args.contract.resolve()
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != args.contract_sha256
    ):
        raise ValueError("history-capacity contract SHA-256 differs")
    contract = validate_contract(read_json(contract_path, "history-capacity contract"))
    paths, input_hashes = verified_inputs(args, contract["input_sha256"])
    source = verify_source(
        args.source_root.resolve(), args.source_commit, args.accepted_base_commit
    )
    source["tree"] = repository_value(
        args.source_root.resolve(), "rev-parse", "HEAD^{tree}"
    )

    original_ledger = read_json(paths["candidate_e_prior_ledger"], "prior ledger")
    original_speakers = validate_ledger(original_ledger)
    expected = contract["expected_audit_result"]
    if len(original_speakers) != int(expected["candidate_e_prior_ledger_entries"]):
        raise ValueError("Candidate-E prior ledger entry count differs")

    v9_speakers, v9_evidence = validate_v9(
        read_json(paths["v9_panel_seal"], "v9 panel seal"),
        read_json(paths["v9_diagnostic_report"], "v9 diagnostic report"),
        read_json_rows(paths["v9_predictions"], "v9 predictions"),
        read_json(paths["v9_receipt"], "v9 receipt"),
        contract,
        input_hashes,
    )
    v10_speakers, v10_evidence = validate_v10(
        read_json(paths["v10_panel_seal"], "v10 panel seal"),
        read_json(paths["v10_diagnostic_report"], "v10 diagnostic report"),
        read_json_rows(paths["v10_predictions"], "v10 predictions"),
        read_json(paths["v10_receipt"], "v10 receipt"),
        read_json(paths["v10_status_audit"], "v10 status audit"),
        contract,
        input_hashes,
        v9_speakers,
    )
    fresh_speakers, fresh_evidence = validate_fresh(
        read_json(paths["ltas_fresh_panel_contract"], "fresh panel contract"),
        read_json(paths["ltas_fresh_alpha_selection"], "fresh alpha selection"),
        read_csv(paths["ltas_fresh_calibration_csv"]),
        read_json(paths["ltas_fresh_final_seal"], "fresh final seal"),
        read_csv(paths["ltas_fresh_final_csv"]),
        read_json(paths["ltas_fresh_report"], "fresh report"),
        read_json(paths["ltas_fresh_receipt"], "fresh receipt"),
        contract,
        input_hashes,
    )
    candidate_e_speakers, candidate_e_evidence = validate_candidate_e(
        read_csv(paths["candidate_e_exact_csv"]),
        read_json(paths["candidate_e_exact_report"], "Candidate-E report"),
        read_json(paths["candidate_e_exact_receipt"], "Candidate-E receipt"),
        contract,
        input_hashes,
    )
    invalid_v2_seal = read_json(paths["invalid_v2_source_seal"], "invalid v2 seal")
    invalid_v2_speakers, invalid_v2_evidence = validate_invalid_v2(
        read_json(paths["invalid_v2_contract"], "invalid v2 contract"),
        invalid_v2_seal,
        read_json(paths["invalid_v2_receipt"], "invalid v2 receipt"),
        read_json(paths["invalid_v2_updated_ledger"], "invalid v2 ledger"),
        original_ledger,
        input_hashes,
    )

    history_sources = {
        "ltas_svd_authority_v9_exact_attempt": v9_speakers,
        "ltas_svd_authority_v10_exact_attempt": v10_speakers,
        "ltas_fresh_panel_v1_exact": fresh_speakers,
        "candidate_e_external_v32r8_exact": candidate_e_speakers,
    }
    sources_by_speaker = history_sources_by_speaker(history_sources)
    historical_speakers = set(sources_by_speaker)
    if (
        len(historical_speakers)
        != int(expected["complete_historical_exact_opened_svd_speakers"])
        or not candidate_e_speakers.issubset(historical_speakers)
    ):
        raise ValueError("complete historical SVD exact-opened union differs")
    overlap = invalid_v2_speakers & historical_speakers
    if (
        len(invalid_v2_speakers) != int(expected["invalid_v2_selected_speakers"])
        or len(overlap) != int(expected["invalid_v2_historical_overlap"])
        or overlap != invalid_v2_speakers
    ):
        raise ValueError("invalid v2 historical-speaker overlap differs")

    sv_rows = read_csv(paths["sv_metadata"])
    cs_rows = read_csv(paths["cs_metadata"])
    eligible_before = eligible_svd_rows(
        sv_rows,
        cs_rows,
        args.sv_root.resolve(),
        args.cs_root.resolve(),
        set(),
    )
    complete_exclusions = original_speakers | historical_speakers
    eligible_after = eligible_svd_rows(
        sv_rows,
        cs_rows,
        args.sv_root.resolve(),
        args.cs_root.resolve(),
        complete_exclusions,
    )
    remaining_speakers = {
        canonical_svd_speaker(row["speaker_id"]) for row in eligible_after
    }
    expected_remaining = set(expected["remaining_eligible_svd_speakers"])
    female_count = sum(row["sex"] == "female" for row in eligible_after)
    male_count = sum(row["sex"] == "male" for row in eligible_after)
    if (
        len(eligible_before)
        != int(
            expected[
                "metadata_eligible_svd_speakers_before_complete_history_exclusion"
            ]
        )
        or remaining_speakers != expected_remaining
        or female_count != int(expected["remaining_female_speakers"])
        or male_count != int(expected["remaining_male_speakers"])
    ):
        raise ValueError("complete-history SVD source capacity differs")
    required_speakers = int(
        contract["source_panel_capacity_gate"]["required_distinct_speakers"]
    )
    capacity_pass = (
        len(eligible_after) >= required_speakers
        and female_count >= 4
        and male_count >= 4
    )
    if capacity_pass:
        raise ValueError("frozen failure decision is invalid because capacity passed")

    scorability = existing_scorability_by_speaker(invalid_v2_seal)
    remaining_rows = []
    for row in sorted(eligible_after, key=lambda item: int(item["speaker_id"])):
        canonical = canonical_svd_speaker(row["speaker_id"])
        if set(scorability.get(canonical, {})) != set(VIEWS):
            raise ValueError("remaining speaker lacks frozen v2 scorability coverage")
        remaining_rows.append(
            {
                "canonical_speaker_id": canonical,
                "speaker_id": row["speaker_id"],
                "session_id": row["session_id"],
                "sex": row["sex"],
                "diagnosis_record_only": row["diagnosis_record_only"],
                "existing_v2_all_six_component_scorability_boolean": scorability[
                    canonical
                ],
            }
        )

    merged_ledger = merged_history_ledger(
        original_ledger,
        input_hashes["candidate_e_prior_ledger"],
        sources_by_speaker,
        input_hashes,
        args.source_commit,
    )
    if len(validate_ledger(merged_ledger)) != len(
        original_speakers | historical_speakers
    ):
        raise ValueError("merged historical ledger coverage differs")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "decision": FAILURE_DECISION,
        "source": source,
        "slurm_job_id": job_id,
        "contract_sha256": args.contract_sha256,
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_sha256": input_hashes,
        "historical_exact_opening_evidence": {
            "ltas_svd_authority_v9": v9_evidence,
            "ltas_svd_authority_v10": v10_evidence,
            "ltas_fresh_panel_v1": fresh_evidence,
            "candidate_e_external_v32r8": candidate_e_evidence,
        },
        "complete_historical_exact_opened_svd_speakers": sorted(
            historical_speakers
        ),
        "complete_historical_exact_opened_svd_speaker_count": len(
            historical_speakers
        ),
        "candidate_e_prior_ledger": {
            "entries": len(original_speakers),
            "svd_entries": len(
                {speaker for speaker in original_speakers if speaker.startswith("SVD:")}
            ),
            "missing_historical_exact_opened_svd_speakers": sorted(
                historical_speakers - original_speakers
            ),
            "missing_count": len(historical_speakers - original_speakers),
        },
        "invalid_v2_source_panel": {
            **invalid_v2_evidence,
            "historical_overlap_speakers": sorted(overlap),
            "historical_overlap_count": len(overlap),
            "valid_for_materialization": False,
            "valid_for_six_gradient_evaluation": False,
        },
        "source_capacity_after_complete_history_exclusion": {
            "metadata_eligible_before_exclusion": len(eligible_before),
            "complete_exclusion_speaker_count": len(complete_exclusions),
            "remaining_eligible_speakers": remaining_rows,
            "remaining_count": len(eligible_after),
            "remaining_female_count": female_count,
            "remaining_male_count": male_count,
            "required_distinct_speakers": required_speakers,
            "required_female_speakers": 4,
            "required_male_speakers": 4,
            "capacity_pass": False,
        },
        "scientific_failure_preserved": True,
        "threshold_changed_after_failure": False,
        "successor_source_selection_performed": False,
        "new_target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
        "new_waveforms_materialized": False,
        "six_gradient_evaluation_submitted": False,
        "fusion_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "joint_panel_submitted": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "complete_historical_svd_speaker_ledger_v3.json"
    report_path = args.output_dir / "svd_history_capacity_report_v3.json"
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(ledger_path, merged_ledger)
    write_json(report_path, report)
    artifact_hashes = {
        ledger_path.name: sha256_file(ledger_path),
        report_path.name: sha256_file(report_path),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": FAILURE_DECISION,
        "slurm_job_id": job_id,
        "source_commit": source["head"],
        "source_tree": source["tree"],
        "contract_sha256": args.contract_sha256,
        "input_sha256": input_hashes,
        "artifact_sha256": artifact_hashes,
        "historical_exact_opened_svd_speaker_count": len(historical_speakers),
        "invalid_v2_historical_overlap_count": len(overlap),
        "remaining_eligible_svd_speaker_count": len(eligible_after),
        "source_panel_capacity_pass": False,
        "scientific_failure_preserved": True,
        "successor_source_selection_performed": False,
        "new_waveforms_materialized": False,
        "six_gradient_evaluation_submitted": False,
        "fusion_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
