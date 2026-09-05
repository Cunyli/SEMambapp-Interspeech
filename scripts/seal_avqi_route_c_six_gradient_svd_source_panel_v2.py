#!/usr/bin/env python3
"""Seal an unused-speaker SVD source panel before waveform materialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import soundfile as sf

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_gradient_fusion import (
    CAP_POLICY,
    CONFLICT_POLICY,
    FUSION_SCHEMA_VERSION,
    JOINT_NORMALIZATION,
    MAXIMUM_WEIGHTED_COMPONENT_NORM_SHARE,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.evaluate_avqi_route_c_multicomponent_gradients import verify_source
from scripts.evaluate_avqi_shimmer_fresh_panel import avqi_code_tree_sha256


CONTRACT_SCHEMA_VERSION = "avqi-route-c-six-gradient-fusion-contract-v2"
PANEL_SCHEMA_VERSION = "avqi-route-c-six-gradient-svd-source-panel-seal-v2"
RECEIPT_SCHEMA_VERSION = "avqi-route-c-six-gradient-svd-source-panel-receipt-v2"
LEDGER_SCHEMA_VERSION = "avqi-route-c-prior-panel-speaker-ledger-v1"
PANEL_DECISION = "SEALED_UNUSED_SPEAKER_SVD_SIX_GRADIENT_SOURCE_PANEL_V2"
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
AUDIT_SPLITS = ("surrogate_calibration", "surrogate_holdout")
STRATA = (("female", "cs"), ("female", "sv"), ("male", "cs"), ("male", "sv"))
RECIPE_INDICES = tuple(range(972, 980))
EXPECTED_CASES = len(AUDIT_SPLITS) * len(STRATA)
LOSS_TARGET = (
    "normalized bidirectional gap to same-speaker clean pathological CS/SV target"
)
BASE_WEIGHT_RULE = (
    "minimum calibration median gradient norm / component median gradient norm"
)
SCORABILITY_MARKER = "AVQI_SVD_TARGET_SCORABILITY_JSON="
SCORABILITY_PROGRAM = r"""
import json
import math
import sys

import parselmouth

sys.path.insert(0, sys.argv[1])
from avqi_code import run_avqi

step_versions = {
    "highpass": "praat",
    "read_and_resample": "praat",
    "sv_length_norm": "praat",
    "cs_voiced_segments": "praat",
    "concatenate": "praat",
    "cpps": "praat",
    "slope": "praat",
    "tilt": "praat",
    "shimmer": "praat",
    "hnr": "praat",
    "pitch": "praat",
}
component_names = ("cpps", "hnr", "shimmer_percent", "shimmer_db", "slope", "tilt")
request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    complete = False
    failure_class = "none"
    try:
        metrics = run_avqi(
            item["path"],
            item["path"],
            target_sr=16000,
            speaking_type=item["view"],
            step_versions=step_versions,
            remove_sv_silence_with_sox=False,
        )
        complete = all(math.isfinite(float(metrics[name])) for name in component_names)
        if not complete:
            failure_class = "nonfinite_six_component_target"
    except (
        OSError,
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        parselmouth.PraatError,
    ) as error:
        failure_class = type(error).__name__
    rows.append({"id": item["id"], "all_six_components_scorable": complete, "failure_class": failure_class})
print(
    "AVQI_SVD_TARGET_SCORABILITY_JSON="
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--failure-report-sha256", required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--failure-receipt-sha256", required=True)
    parser.add_argument("--sv-metadata", type=Path, required=True)
    parser.add_argument("--sv-metadata-sha256", required=True)
    parser.add_argument("--cs-metadata", type=Path, required=True)
    parser.add_argument("--cs-metadata-sha256", required=True)
    parser.add_argument("--prior-speaker-ledger", type=Path, required=True)
    parser.add_argument("--prior-speaker-ledger-sha256", required=True)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--cs-root", type=Path, required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accepted-base-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("decision")
        != "FROZEN_BEFORE_EXTERNAL_SVD_SOURCE_PANEL_SELECTION"
        or contract.get("route") != "C"
        or tuple(contract.get("component_order", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or contract.get("loss_target") != LOSS_TARGET
        or contract.get("base_weight_rule") != BASE_WEIGHT_RULE
    ):
        raise ValueError("SVD fusion contract identity differs")
    panel = contract.get("external_svd_source_panel")
    materialization = contract.get("waveform_materialization")
    scaled = contract.get("scaled_base_support")
    exact = contract.get("exact_authority")
    boundaries = contract.get("boundaries")
    if not all(
        isinstance(value, Mapping)
        for value in (panel, materialization, scaled, exact, boundaries)
    ):
        raise ValueError("SVD fusion contract sections are unavailable")
    fusion = contract.get("fusion_rule")
    expected_fusion = {
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
    }
    if fusion != expected_fusion:
        raise ValueError("frozen six-gradient fusion rule differs")
    if (
        panel.get("dataset") != "SVD"
        or panel.get("patient_only") is not True
        or panel.get("splits") != list(AUDIT_SPLITS)
        or panel.get("cases_per_split") != len(STRATA)
        or panel.get("strata_per_split")
        != [f"{sex}/{view}" for sex, view in STRATA]
        or panel.get("one_case_per_speaker") is not True
        or panel.get("speaker_disjoint_across_splits") is not True
        or panel.get("exclude_every_prior_ledger_speaker") is not True
        or panel.get("target_all_six_component_scorability_boolean_may_be_used")
        is not True
        or panel.get("target_scalar_values_used_for_selection") is not False
        or panel.get("base_or_candidate_exact_outcomes_used_for_selection")
        is not False
        or panel.get("diagnosis_used_for_selection") is not False
        or panel.get("source_split_sealed_before_simulation") is not True
    ):
        raise ValueError("SVD source-panel policy differs")
    if (
        materialization.get("target_waveform")
        != "raw_resampled_same_speaker_clean_pathological_source"
        or materialization.get("fixed_recipe_indices") != list(RECIPE_INDICES)
        or materialization.get("degradations") != ["reverb", "noise"]
        or materialization.get("snr_db_by_case_order")
        != [20, 10, 20, 10, 20, 10, 20, 10]
        or materialization.get("generator") != "S3_500"
        or materialization.get("generator_mode") != "frozen_inference_only"
        or materialization.get("emitted_waveform_highpass") is not False
        or materialization.get("metric_highpass_internal_only") is not True
        or materialization.get("base_exact_components_opened") is not False
        or materialization.get("candidate_exact_components_opened") is not False
        or materialization.get("generator_optimizer_created") is not False
        or materialization.get("generator_optimizer_steps") != 0
    ):
        raise ValueError("SVD source-panel recipe assignment differs")
    if (
        scaled.get("support_rule_frozen_before_new_panel_selection") is not True
        or scaled.get("candidate_exact_outcome_used") is not False
        or scaled.get("speaker_or_case_identity_used") is not False
    ):
        raise ValueError("Candidate-E scaled-base boundary differs")
    if (
        exact.get("preselection_role")
        != "clean_target_all_six_component_scorability_boolean_only"
        or exact.get("post_source_seal_role")
        != "clean_target_scalar_sealing_only"
        or exact.get("base_exact_components_opened") is not False
        or exact.get("candidate_exact_components_opened") is not False
    ):
        raise ValueError("exact AVQI authority boundary differs")
    if (
        boundaries.get("candidate_exact_outcomes_opened") is not False
        or boundaries.get("joint_panel_authorized") is not False
        or boundaries.get("generator_optimizer_steps") != 0
        or boundaries.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("SVD source-panel authorization boundary differs")
    expected_gates = {
        "all_component_gradients_finite_nonzero_bounded": True,
        "all_post_cap_joint_gradients_finite_nonzero_bounded": True,
        "all_post_cap_weighted_component_shares_le_0_80": True,
        "all_post_cap_component_to_joint_cosines_nonnegative": True,
        "all_candidate_e_peak_paths_pcm16_hash_bound": True,
        "no_component_amplified": True,
        "only_unique_dominant_component_may_be_attenuated": True,
        "calibration_and_holdout_speaker_disjoint": True,
        "all_prior_ledger_speakers_excluded": True,
    }
    if contract.get("promotion_gates") != expected_gates:
        raise ValueError("SVD fusion promotion gates differ")
    return dict(panel)


def validate_failure_evidence(
    contract: Mapping[str, Any],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    report_sha256: str,
) -> None:
    expected = contract.get("predecessor_evidence")
    if not isinstance(expected, Mapping):
        raise ValueError("predecessor evidence is unavailable")
    if (
        expected.get("failure_audit_report_sha256") != report_sha256
        or receipt.get("artifact_sha256", {}).get(
            "failure_audit_report.json"
        )
        != report_sha256
        or report.get("decision")
        != "NO_GO_ROUTE_C_SIX_GRADIENT_FUSION_VALIDATION_CANDIDATE_E_SCALED_BASE_UNSUPPORTED_V1"
        or receipt.get("decision") != report.get("decision")
        or report.get("fusion_scientific_promotion_granted") is not False
        or receipt.get("fusion_scientific_promotion_granted") is not False
        or report.get("joint_panel_authorized") is not False
        or receipt.get("joint_panel_authorized") is not False
        or report.get("generator_optimizer_steps") != 0
        or receipt.get("generator_optimizer_steps") != 0
    ):
        raise ValueError("immutable scaled-base failure evidence differs")


def validate_ledger(ledger: Mapping[str, Any]) -> set[str]:
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
        or ledger.get("exact_outcomes_used_for_selection") is not False
    ):
        raise ValueError("prior speaker ledger contract differs")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prior speaker ledger is empty")
    speakers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("prior speaker ledger entry differs")
        canonical = f"{str(entry.get('dataset', '')).upper()}:{entry.get('speaker_id', '')}"
        if entry.get("canonical_speaker_id") != canonical or canonical in speakers:
            raise ValueError("prior speaker ledger identity differs")
        speakers.add(canonical)
    return speakers


def eligible_svd_rows(
    sv_rows: list[dict[str, str]],
    cs_rows: list[dict[str, str]],
    sv_root: Path,
    cs_root: Path,
    excluded: set[str],
) -> list[dict[str, Any]]:
    sv_by_session = {row["session_id"]: row for row in sv_rows}
    cs_by_session = {row["session_id"]: row for row in cs_rows}
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for session_id in sorted(set(sv_by_session) & set(cs_by_session), key=int):
        sv_row = sv_by_session[session_id]
        cs_row = cs_by_session[session_id]
        speaker_id = str(sv_row.get("speaker id", ""))
        sex = str(sv_row.get("gender", ""))
        if (
            sv_row.get("health status") != "1"
            or cs_row.get("health status") != "1"
            or cs_row.get("speaker id") != speaker_id
            or cs_row.get("gender") != sex
            or sex not in {"female", "male"}
            or f"SVD:{speaker_id}" in excluded
        ):
            continue
        sv_path = (sv_root / sv_row["filename"]).resolve()
        cs_path = (cs_root / cs_row["filename"]).resolve()
        if not sv_path.is_file() or not cs_path.is_file():
            continue
        sv_info = sf.info(sv_path)
        cs_info = sf.info(cs_path)
        if (
            sv_info.channels != 1
            or cs_info.channels != 1
            or sv_info.frames / sv_info.samplerate < 1.0
            or cs_info.frames / cs_info.samplerate < 3.0
        ):
            continue
        by_speaker.setdefault(speaker_id, []).append(
            {
                "speaker_id": speaker_id,
                "session_id": session_id,
                "sex": sex,
                "diagnosis_record_only": str(sv_row.get("diagnosis", "")),
                "sv_path": str(sv_path),
                "cs_path": str(cs_path),
            }
        )
    return [
        min(rows, key=lambda row: int(row["session_id"]))
        for rows in by_speaker.values()
    ]


def scorability_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "id": f"SVD:{row['speaker_id']}:{view}",
            "path": str(row[f"{view}_path"]),
            "view": view,
        }
        for row in rows
        for view in ("cs", "sv")
    ]


def run_scorability(
    items: list[dict[str, str]],
    exact_python: Path,
    avqi_code_root: Path,
) -> tuple[dict[str, bool], dict[str, Any]]:
    completed = subprocess.run(
        [str(exact_python), "-c", SCORABILITY_PROGRAM, str(avqi_code_root)],
        input=json.dumps({"items": items}),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SVD target scorability subprocess failed: " + completed.stderr[-4000:]
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(SCORABILITY_MARKER)
    ]
    if len(lines) != 1:
        raise RuntimeError("SVD target scorability marker differs")
    payload = json.loads(lines[0][len(SCORABILITY_MARKER) :])
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != [
        item["id"] for item in items
    ]:
        raise ValueError("SVD target scorability coverage differs")
    allowed = {"id", "all_six_components_scorable", "failure_class"}
    for row in rows:
        if set(row) != allowed or not isinstance(
            row["all_six_components_scorable"], bool
        ):
            raise ValueError("SVD target scorability retained forbidden data")
    return (
        {row["id"]: row["all_six_components_scorable"] for row in rows},
        {
            "parselmouth_version": payload["parselmouth_version"],
            "praat_version": payload["praat_version"],
            "rows": rows,
            "target_scalar_values_retained": False,
            "base_or_candidate_exact_outcomes_opened": False,
        },
    )


def selection_digest(
    salt: str,
    split: str,
    sex: str,
    view: str,
    row: Mapping[str, Any],
) -> str:
    payload = ":".join(
        (salt, split, sex, view, str(row["speaker_id"]), str(row["session_id"]))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_cases(
    rows: list[dict[str, Any]],
    scorability: Mapping[str, bool],
    salt: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_speakers: set[str] = set()
    for split in AUDIT_SPLITS:
        for sex, view in STRATA:
            ranked = sorted(
                (
                    row
                    for row in rows
                    if row["sex"] == sex
                    and row["speaker_id"] not in used_speakers
                    and scorability[f"SVD:{row['speaker_id']}:{view}"]
                ),
                key=lambda row: selection_digest(salt, split, sex, view, row),
            )
            if not ranked:
                raise ValueError(f"no target-scorable SVD case for {split}/{sex}/{view}")
            row = ranked[0]
            used_speakers.add(str(row["speaker_id"]))
            selected.append(
                {
                    **row,
                    "split": split,
                    "view": view,
                    "sample_group": sex,
                    "condition": "aug16k_phone",
                    "selection_digest": selection_digest(
                        salt, split, sex, view, row
                    ),
                }
            )
    if len(selected) != EXPECTED_CASES or len(used_speakers) != EXPECTED_CASES:
        raise ValueError("SVD source-panel selection coverage differs")
    return selected


def extend_ledger(
    ledger: Mapping[str, Any],
    selected: list[dict[str, Any]],
    source_commit: str,
) -> dict[str, Any]:
    entries = [dict(entry) for entry in ledger["entries"]]
    existing = {entry["canonical_speaker_id"] for entry in entries}
    for row in selected:
        canonical = f"SVD:{row['speaker_id']}"
        if canonical in existing:
            raise ValueError("SVD source panel reused a ledger speaker")
        existing.add(canonical)
        entries.append(
            {
                "dataset": "SVD",
                "speaker_id": row["speaker_id"],
                "canonical_speaker_id": canonical,
                "panel_role": "six_gradient_fusion_svd_source_v2",
                "session_id": row["session_id"],
                "source_commit": source_commit,
                "target_all_six_component_scorability_boolean_used": True,
                "target_scalar_values_used": False,
                "base_or_candidate_exact_outcomes_used": False,
            }
        )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "exact_outcomes_used_for_selection": False,
        "entries": sorted(entries, key=lambda entry: entry["canonical_speaker_id"]),
        "added_by": "six_gradient_fusion_svd_source_panel_v2",
        "added_speaker_count": EXPECTED_CASES,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite SVD source seal: {args.output_dir}")
    paths = {
        "contract": _verified_file(args.contract, args.contract_sha256, "contract"),
        "failure_report": _verified_file(
            args.failure_report, args.failure_report_sha256, "failure report"
        ),
        "failure_receipt": _verified_file(
            args.failure_receipt, args.failure_receipt_sha256, "failure receipt"
        ),
        "sv_metadata": _verified_file(
            args.sv_metadata, args.sv_metadata_sha256, "SVD SV metadata"
        ),
        "cs_metadata": _verified_file(
            args.cs_metadata, args.cs_metadata_sha256, "SVD CS metadata"
        ),
        "prior_speaker_ledger": _verified_file(
            args.prior_speaker_ledger,
            args.prior_speaker_ledger_sha256,
            "prior speaker ledger",
        ),
    }
    source = verify_source(
        args.source_root.resolve(), args.source_commit, args.accepted_base_commit
    )
    contract = _read_json(paths["contract"], "fusion contract")
    panel_policy = validate_contract(contract)
    expected_input_hashes = {
        "sv_metadata": panel_policy["sv_metadata_sha256"],
        "cs_metadata": panel_policy["cs_metadata_sha256"],
        "prior_speaker_ledger": panel_policy["prior_speaker_ledger_sha256"],
    }
    observed_input_hashes = {
        "sv_metadata": args.sv_metadata_sha256,
        "cs_metadata": args.cs_metadata_sha256,
        "prior_speaker_ledger": args.prior_speaker_ledger_sha256,
    }
    if observed_input_hashes != expected_input_hashes:
        raise ValueError("SVD source inputs differ from frozen contract")
    failure_report = _read_json(paths["failure_report"], "failure report")
    failure_receipt = _read_json(paths["failure_receipt"], "failure receipt")
    validate_failure_evidence(
        contract, failure_report, failure_receipt, args.failure_report_sha256
    )
    if (
        contract["predecessor_evidence"]["failure_audit_receipt_sha256"]
        != args.failure_receipt_sha256
    ):
        raise ValueError("failure receipt differs from frozen contract")
    if not args.exact_python.resolve().is_file():
        raise FileNotFoundError("exact Python is unavailable")
    exact_policy = contract["exact_authority"]
    if (
        str(args.exact_python.resolve()) != exact_policy["python"]
        or str(args.avqi_code_root.resolve()) != exact_policy["avqi_code_root"]
        or args.avqi_code_tree_sha256 != exact_policy["avqi_code_tree_sha256"]
    ):
        raise ValueError("exact AVQI authority differs from frozen contract")
    if avqi_code_tree_sha256(args.avqi_code_root.resolve()) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code tree differs")
    ledger = _read_json(paths["prior_speaker_ledger"], "prior speaker ledger")
    excluded = validate_ledger(ledger)
    eligible = eligible_svd_rows(
        _read_csv(paths["sv_metadata"]),
        _read_csv(paths["cs_metadata"]),
        args.sv_root.resolve(),
        args.cs_root.resolve(),
        excluded,
    )
    if len(eligible) < EXPECTED_CASES:
        raise ValueError("insufficient ledger-disjoint SVD speakers")
    items = scorability_items(eligible)
    scorability, scorability_audit = run_scorability(
        items, args.exact_python.resolve(), args.avqi_code_root.resolve()
    )
    selected = select_cases(eligible, scorability, panel_policy["selection_salt"])
    rows = []
    for recipe_index, row in zip(RECIPE_INDICES, selected, strict=True):
        view = row["view"]
        target_path = Path(row[f"{view}_path"])
        sample_id = f"SVD:{row['speaker_id']}:{row['session_id']}"
        rows.append(
            {
                "case_id": ":".join(
                    (
                        "six-gradient-svd-v2",
                        row["split"],
                        str(row["speaker_id"]),
                        str(row["session_id"]),
                        row["sex"],
                        view,
                    )
                ),
                "dataset": "SVD",
                "speaker_id": str(row["speaker_id"]),
                "canonical_speaker_id": f"SVD:{row['speaker_id']}",
                "session_id": str(row["session_id"]),
                "sample_id": sample_id,
                "split": row["split"],
                "sample_group": row["sample_group"],
                "sex": row["sex"],
                "view": view,
                "condition": row["condition"],
                "diagnosis_record_only": row["diagnosis_record_only"],
                "target_source_path": str(target_path.resolve()),
                "target_source_sha256": sha256_file(target_path),
                "paired_cs_path": row["cs_path"],
                "paired_cs_sha256": sha256_file(Path(row["cs_path"])),
                "paired_sv_path": row["sv_path"],
                "paired_sv_sha256": sha256_file(Path(row["sv_path"])),
                "recipe_index": recipe_index,
                "selection_digest": row["selection_digest"],
                "target_all_six_components_scorable": True,
            }
        )
    updated_ledger = extend_ledger(ledger, selected, args.source_commit)
    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "prior_speaker_ledger_after_svd_v2.json"
    _write_json(ledger_path, updated_ledger)
    seal = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "source": source,
        "contract_sha256": args.contract_sha256,
        "input_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "exact_authority": {
            "python": str(args.exact_python.resolve()),
            "avqi_code_root": str(args.avqi_code_root.resolve()),
            "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
            "parselmouth_version": scorability_audit["parselmouth_version"],
            "praat_version": scorability_audit["praat_version"],
        },
        "selection": {
            "dataset": "SVD",
            "salt": panel_policy["selection_salt"],
            "eligible_ledger_disjoint_speakers": len(eligible),
            "strata_per_split": [f"{sex}/{view}" for sex, view in STRATA],
            "cases_by_split": {split: len(STRATA) for split in AUDIT_SPLITS},
            "selected_speakers_by_split": {
                split: sorted(row["canonical_speaker_id"] for row in rows if row["split"] == split)
                for split in AUDIT_SPLITS
            },
            "speaker_overlap": 0,
            "prior_ledger_overlap": 0,
            "target_scalar_values_used": False,
            "base_or_candidate_exact_outcomes_used": False,
        },
        "target_scorability_audit": scorability_audit,
        "rows": rows,
        "updated_speaker_ledger_sha256": sha256_file(ledger_path),
        "source_split_sealed_before_simulation": True,
        "waveform_generation_performed": False,
        "target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    seal_path = args.output_dir / "svd_source_panel_seal_v2.json"
    _write_json(seal_path, seal)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": PANEL_DECISION,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {
            seal_path.name: sha256_file(seal_path),
            ledger_path.name: sha256_file(ledger_path),
        },
        "input_sha256": seal["input_sha256"],
        "selected_cases": len(rows),
        "target_scalar_values_opened": False,
        "base_or_candidate_exact_outcomes_opened": False,
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
                "selected_cases": len(rows),
                "panel_seal_sha256": sha256_file(seal_path),
                "updated_ledger_sha256": sha256_file(ledger_path),
                "receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
