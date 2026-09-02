#!/usr/bin/env python3
"""Amend and seal the Candidate-E external SVD panel after target eligibility.

This is a versioned successor of scientific v24 and Candidate-E v30.  It keeps
the frozen salted speaker ranking, original recipe slots, simulation, and
S3_500 inference.  The only added selection input is an authoritative Praat
boolean stating whether paired clean CS/SV targets have finite local Shimmer
percent and local-dB values.  No target scalar and no base/candidate exact
outcome is emitted, retained, or used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from scripts import evaluate_avqi_shimmer_db_candidate_e_opened_v15_v29 as v29
from scripts import prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30 as v30
from scripts import prepare_avqi_shimmer_db_external_svd_panel_v24 as v24
from scripts import seal_avqi_shimmer_db_candidate_e_external_svd_target_v31 as v31
from scripts.evaluate_avqi_shimmer_fresh_panel import avqi_code_tree_sha256


PANEL_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-panel-v30r2"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-panel-receipt-v30r2"
)
SCORABILITY_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-target-scorability-boolean-v30r2"
)
EQUIVALENCE_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-v30-retained-waveform-equivalence-v30r2"
)
PANEL_DECISION = (
    "SEALED_CANDIDATE_E_EXTERNAL_SVD_PANEL_TARGET_SCORABILITY_ONLY_V30R2"
)
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
EXACT_MARKER = "AVQI_CANDIDATE_E_TARGET_SCORABILITY_JSON="
EXACT_SCORABILITY_SCORER = r"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from avqi_code.main import (
    get_voiced_segments,
    highpass_filter,
    length_normalize_sv,
    read_and_resample_signal,
)

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    percent_scorable = False
    db_scorable = False
    failure_class = "none"
    try:
        signal = read_and_resample_signal(item["path"], 16000)
        highpassed = highpass_filter("praat", signal, 16000)
        if item["view"] == "sv":
            metric = length_normalize_sv("praat", highpassed, 16000)
        elif item["view"] == "cs":
            metric = get_voiced_segments("praat", highpassed, 16000)
        else:
            raise ValueError(f"unsupported view: {item['view']}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            metric_path = handle.name
        try:
            sf.write(metric_path, metric, 16000)
            sound = parselmouth.Sound(metric_path)
            point_process = call(sound, "To PointProcess (periodic, cc)", 50, 400)
            shimmer_percent = call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            shimmer_db = call(
                [sound, point_process],
                "Get shimmer (local_dB)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            percent_scorable = math.isfinite(float(shimmer_percent))
            db_scorable = math.isfinite(float(shimmer_db))
            if not percent_scorable and not db_scorable:
                failure_class = "nonfinite_shimmer_percent_and_db"
            elif not percent_scorable:
                failure_class = "nonfinite_shimmer_percent"
            elif not db_scorable:
                failure_class = "nonfinite_shimmer_db"
        finally:
            os.unlink(metric_path)
    except Exception as error:
        # Praat eligibility errors are data status, not performance outcomes.
        failure_class = type(error).__name__
    rows.append(
        {
            "id": item["id"],
            "shimmer_percent_scorable": percent_scorable,
            "shimmer_db_scorable": db_scorable,
            "component_pair_scorable": percent_scorable and db_scorable,
            "failure_class": failure_class,
        }
    )
print(
    "AVQI_CANDIDATE_E_TARGET_SCORABILITY_JSON="
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
    for option in (
        "config",
        "v29-report",
        "v29-receipt",
        "prior-panel-speaker-ledger",
        "original-panel-seal",
        "original-panel-receipt",
        "original-updated-ledger",
        "sv-metadata",
        "cs-metadata",
        "fixed-recipes",
        "generator-config",
        "generator-checkpoint",
        "simulation-config",
    ):
        add_hashed_path(parser, option)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--cs-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--simulation-source-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260830)
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
        raise ValueError("repository root does not contain the v30r2 preparer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v30r2 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v30r2 preparation requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "preparer_sha256": v24.sha256_file(Path(__file__).resolve()),
        "inherited_v24_logic_sha256": v24.sha256_file(Path(v24.__file__).resolve()),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")


def validate_v30r2_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("v30r2 config schema drift")
    amendment = config.get("scorability_amendment", {})
    expected = {
        "ranking_salt_unchanged": v24.SELECTION_SALT,
        "ranking_algorithm_unchanged": True,
        "scan_order": "ascending_frozen_rank_within_sex",
        "selection_rule": (
            "first_three_per_sex_with_paired_cs_sv_target_"
            "shimmer_percent_and_db_scorable"
        ),
        "slot_assignment_rule": (
            "retain_each_original_v30_eligible_speaker_in_its_original_recipe_"
            "slot_and_fill_each_ineligible_slot_with_the_next_selected_frozen_rank"
        ),
        "raw_clean_preflight_required": True,
        "prepared_target_confirmation_required": True,
        "target_scalar_values_retained_or_used": False,
        "target_scorability_boolean_used": True,
        "target_shimmer_percent_scorability_boolean_used": True,
        "target_shimmer_db_scorability_boolean_used": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "speaker_or_case_identity_hardcoded": False,
        "diagnosis_or_severity_used": False,
    }
    for field, value in expected.items():
        if amendment.get(field) != value:
            raise ValueError(f"v30r2 scorability amendment drift: {field}")
    panel = config.get("panel_contract", {})
    expected_panel = {
        "dataset": "SVD",
        "patient_only": True,
        "speaker_split_before_simulation": True,
        "speakers_per_sex": v24.SPEAKERS_PER_SEX,
        "sex_coverage": ["female", "male"],
        "views_per_speaker": list(v24.VIEWS),
        "conditions": list(v24.CONDITIONS),
        "exclude_all_historical_tau_speakers": True,
    }
    for field, value in expected_panel.items():
        if panel.get(field) != value:
            raise ValueError(f"v30r2 panel contract drift: {field}")
    recipe = config.get("recipe_contract", {})
    if recipe.get("indices") != list(v24.RECIPE_ASSIGNMENT):
        raise ValueError("v30r2 recipe assignment drift")
    if recipe.get("assignment_unchanged") is not True:
        raise ValueError("v30r2 recipe inheritance drift")
    boundaries = config.get("immutable_boundaries", {})
    for field in (
        "old_v23_no_go_receipt_preserved",
        "old_v30_panel_and_v31_failure_artifacts_preserved",
        "candidate_e_source_config_alpha_selector_frozen",
        "no_final_waveform_highpass",
    ):
        if boundaries.get(field) is not True:
            raise ValueError(f"v30r2 immutable boundary drift: {field}")
    require_training_boundary(boundaries, "v30r2 config")


def validate_v29_authorization(
    config: dict[str, Any],
    report: dict[str, Any],
    receipt: dict[str, Any],
    *,
    report_sha256: str,
    receipt_sha256: str,
) -> None:
    validate_v30r2_config(config)
    authorization = config.get("authorization", {})
    expected = {
        "decision": v29.PASS_DECISION,
        "report_sha256": report_sha256,
        "receipt_sha256": receipt_sha256,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            raise ValueError(f"v30r2 authorization config drift: {field}")
    if report.get("schema_version") != v29.REPORT_SCHEMA:
        raise ValueError("v29 report schema drift")
    if receipt.get("schema_version") != v29.RECEIPT_SCHEMA:
        raise ValueError("v29 receipt schema drift")
    for label, value in (("report", report), ("receipt", receipt)):
        if value.get("decision") != v29.PASS_DECISION:
            raise ValueError(f"v29 {label} is not PASS")
        if value.get("candidate_e_frozen") is not True and value.get(
            "candidate_e_remains_frozen"
        ) is not True:
            raise ValueError(f"v29 {label} does not retain Candidate-E freeze")
        if value.get("retuning_authorized") is not False:
            raise ValueError(f"v29 {label} over-authorizes retuning")
        if value.get("external_panel_prepare_authorized") is not True:
            raise ValueError(f"v29 {label} did not authorize external prepare")
        if value.get("external_panel_authorized") is not False:
            raise ValueError(f"v29 {label} prematurely authorizes external panel")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"v29 {label} over-authorizes joint panel")
        require_training_boundary(value, f"v29 {label}")
    if not isinstance(report.get("gates"), dict) or not all(
        gate is True for gate in report["gates"].values()
    ):
        raise ValueError("v29 gates did not all pass")
    if receipt.get("report_sha256") != report_sha256:
        raise ValueError("v29 receipt/report binding drift")


def validate_original_v30(
    config: dict[str, Any],
    panel: dict[str, Any],
    receipt: dict[str, Any],
    ledger: dict[str, Any],
    *,
    panel_sha256: str,
    receipt_sha256: str,
    ledger_sha256: str,
) -> list[dict[str, Any]]:
    trigger = config.get("amendment_trigger", {})
    expected = {
        "original_panel_seal_sha256": panel_sha256,
        "original_panel_receipt_sha256": receipt_sha256,
        "original_updated_ledger_sha256": ledger_sha256,
        "failure_class": "same_speaker_target_shimmer_db_non_finite",
        "old_v30_artifacts_remain_immutable": True,
    }
    for field, value in expected.items():
        if trigger.get(field) != value:
            raise ValueError(f"v30r2 amendment trigger drift: {field}")
    rows = v31.validate_panel_binding(panel, receipt, panel_sha256=panel_sha256)
    if panel.get("prior_panel_speaker_ledger_after_v30_sha256") != ledger_sha256:
        raise ValueError("v30 panel/original updated ledger binding drift")
    if receipt.get("artifact_sha256", {}).get(
        "prior_panel_speaker_ledger_after_v30.json"
    ) != ledger_sha256:
        raise ValueError("v30 receipt/original updated ledger binding drift")
    v24.validate_prior_ledger(ledger)
    return rows


def ranked_svd_speakers(
    sv_rows: list[dict[str, str]],
    cs_rows: list[dict[str, str]],
    sv_root: Path,
    cs_root: Path,
    excluded_speakers: set[str],
) -> dict[str, list[dict[str, Any]]]:
    eligible = v24._eligible_svd_speakers(
        sv_rows,
        cs_rows,
        sv_root,
        cs_root,
        excluded_speakers,
    )
    one_session = [
        min(rows, key=lambda row: int(row["session_id"]))
        for rows in eligible.values()
    ]
    ranked: dict[str, list[dict[str, Any]]] = {}
    for sex in ("female", "male"):
        values = sorted(
            [row for row in one_session if row["sex"] == sex],
            key=lambda row: (
                v24.rank_digest(row["speaker_id"], row["session_id"]),
                row["speaker_id"],
                row["session_id"],
            ),
        )
        ranked[sex] = [
            {**row, "selection_rank_within_sex": index}
            for index, row in enumerate(values, start=1)
        ]
    return ranked


def scorability_items_for_speaker(
    row: dict[str, Any],
    *,
    prefix: str,
) -> list[dict[str, str]]:
    return [
        {
            "id": f"{prefix}:SVD:{row['speaker_id']}:{view}",
            "path": str(Path(row[f"{view}_path"]).resolve()),
            "view": view,
        }
        for view in v24.VIEWS
    ]


def run_target_scorability(
    items: list[dict[str, str]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORABILITY_SCORER, str(avqi_code_root)],
        input=json.dumps({"items": items}, ensure_ascii=False),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "target scorability-only Praat subprocess failed: "
            + completed.stderr[-4000:]
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(EXACT_MARKER)
    ]
    if len(lines) != 1:
        raise RuntimeError("target scorability-only marker drift")
    payload = json.loads(lines[0][len(EXACT_MARKER) :])
    observed_ids = [str(row.get("id", "")) for row in payload.get("rows", [])]
    if observed_ids != [item["id"] for item in items]:
        raise ValueError("target scorability-only coverage/order drift")
    allowed = {
        "id",
        "shimmer_percent_scorable",
        "shimmer_db_scorable",
        "component_pair_scorable",
        "failure_class",
    }
    for row in payload["rows"]:
        if set(row) != allowed:
            raise ValueError("target scorability payload retained forbidden fields")
        for field in (
            "shimmer_percent_scorable",
            "shimmer_db_scorable",
            "component_pair_scorable",
        ):
            if not isinstance(row[field], bool):
                raise ValueError(f"target scorability field is not boolean: {field}")
    return payload


def scan_ranked_speakers(
    ranked: dict[str, list[dict[str, Any]]],
    exact_python: Path,
    avqi_code_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    versions: dict[str, str] | None = None
    for sex in ("female", "male"):
        chosen: list[dict[str, Any]] = []
        for row in ranked[sex]:
            payload = run_target_scorability(
                scorability_items_for_speaker(row, prefix="raw-clean-preflight"),
                exact_python,
                avqi_code_root,
            )
            observed_versions = {
                "parselmouth": str(payload["parselmouth_version"]),
                "praat": str(payload["praat_version"]),
            }
            if versions is None:
                versions = observed_versions
            elif observed_versions != versions:
                raise ValueError("target scorability Praat version drift")
            result_by_view = {
                item["id"].rsplit(":", 1)[-1]: item
                for item in payload["rows"]
            }
            pair_scorable = all(
                result_by_view[view]["component_pair_scorable"]
                for view in v24.VIEWS
            )
            audit_rows.append(
                {
                    "panel_speaker_id": f"SVD:{row['speaker_id']}",
                    "speaker_id": row["speaker_id"],
                    "session_id": row["session_id"],
                    "sex": sex,
                    "selection_rank_within_sex": row[
                        "selection_rank_within_sex"
                    ],
                    "cs_shimmer_percent_scorable": result_by_view["cs"][
                        "shimmer_percent_scorable"
                    ],
                    "cs_shimmer_db_scorable": result_by_view["cs"][
                        "shimmer_db_scorable"
                    ],
                    "sv_shimmer_percent_scorable": result_by_view["sv"][
                        "shimmer_percent_scorable"
                    ],
                    "sv_shimmer_db_scorable": result_by_view["sv"][
                        "shimmer_db_scorable"
                    ],
                    "paired_component_scorable": pair_scorable,
                }
            )
            if pair_scorable:
                chosen.append(row)
            if len(chosen) == v24.SPEAKERS_PER_SEX:
                break
        if len(chosen) != v24.SPEAKERS_PER_SEX:
            raise ValueError(f"insufficient target-scorable SVD speakers: {sex}")
        selected[sex] = chosen
    if versions is None:
        raise ValueError("target scorability scan produced no versions")
    return selected, {
        "schema_version": SCORABILITY_SCHEMA,
        "stage": "raw_clean_frozen_rank_preflight",
        "scalar_values_retained_or_used": False,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "exact_scorer_versions": versions,
        "rows": audit_rows,
    }


def assign_selected_to_original_slots(
    selected: dict[str, list[dict[str, Any]]],
    original_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    original_by_sex: dict[str, list[str]] = {"female": [], "male": []}
    for row in original_rows[::2]:
        original_by_sex[str(row["sex"])].append(str(row["speaker_id"]))
    assigned: list[dict[str, Any]] = []
    for sex in ("female", "male"):
        chosen_by_id = {str(row["speaker_id"]): row for row in selected[sex]}
        replacements = iter(
            row
            for row in selected[sex]
            if str(row["speaker_id"]) not in original_by_sex[sex]
        )
        for slot, old_speaker_id in enumerate(original_by_sex[sex]):
            row = chosen_by_id.get(old_speaker_id)
            if row is None:
                row = next(replacements)
            assigned.append({**row, "recipe_slot_within_sex": slot})
    return assigned


def build_cases(assigned: list[dict[str, Any]]) -> list[v24.SVDCase]:
    cases: list[v24.SVDCase] = []
    condition_pairs = (
        ("rir_only", "snr20"),
        ("snr10", "rir_only"),
        ("snr20", "snr10"),
    )
    for speaker_slot, row in enumerate(assigned):
        slot_within_sex = int(row["recipe_slot_within_sex"])
        condition_pair = condition_pairs[slot_within_sex]
        for view_index, view in enumerate(v24.VIEWS):
            recipe_offset = speaker_slot * len(v24.VIEWS) + view_index
            cases.append(
                v24.SVDCase(
                    speaker_id=str(row["speaker_id"]),
                    session_id=str(row["session_id"]),
                    sex=str(row["sex"]),
                    diagnosis=str(row["diagnosis"]),
                    view=view,
                    condition=condition_pair[view_index],
                    recipe_index=v24.RECIPE_ASSIGNMENT[recipe_offset],
                    source_path=Path(row[f"{view}_path"]),
                    source_duration_seconds=float(
                        row[f"{view}_duration_seconds"]
                    ),
                    selection_rank_within_sex=int(
                        row["selection_rank_within_sex"]
                    ),
                    selection_digest=v24.rank_digest(
                        str(row["speaker_id"]),
                        str(row["session_id"]),
                    ),
                )
            )
    return cases


def build_selection_contract(
    cases: list[v24.SVDCase],
    preflight: dict[str, Any],
    original_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = {case.panel_speaker_id for case in cases}
    original = {str(row["panel_speaker_id"]) for row in original_rows}
    return {
        "dataset": "SVD",
        "selection_mode": "frozen_rank_then_target_scorability_boolean_only",
        "health_status_mapping": {"1": "patient"},
        "paired_cs_sv_same_session_required": True,
        "prior_ledger_excluded_before_hash_ranking": True,
        "speaker_selection_salt": v24.SELECTION_SALT,
        "ranking_digest": "SHA256(salt:speaker_id:session_id)",
        "selection_uses_diagnosis": False,
        "selection_uses_severity": False,
        "selection_uses_target_scalar_values": False,
        "selection_uses_target_scorability_boolean": True,
        "selection_uses_base_or_candidate_exact_outcomes": False,
        "slot_assignment_preserves_retained_v30_recipe_mapping": True,
        "selected_speakers": sorted(selected),
        "selected_sessions": sorted({case.session_id for case in cases}, key=int),
        "retained_v30_speakers": sorted(selected & original),
        "rejected_v30_speakers": sorted(original - selected),
        "replacement_speakers": sorted(selected - original),
        "sex_counts": dict(Counter(case.sex for case in cases[::2])),
        "prior_panel_speaker_overlap": 0,
        "scanned_speaker_count": len(preflight["rows"]),
    }


def final_target_scorability(
    rows: list[dict[str, Any]],
    exact_python: Path,
    avqi_code_root: Path,
    expected_versions: dict[str, str],
) -> dict[str, Any]:
    items = [
        {
            "id": f"prepared-target:{row['case_id']}",
            "path": str(Path(row["target_path"]).resolve()),
            "view": str(row["view"]),
        }
        for row in rows
    ]
    payload = run_target_scorability(items, exact_python, avqi_code_root)
    observed_versions = {
        "parselmouth": str(payload["parselmouth_version"]),
        "praat": str(payload["praat_version"]),
    }
    if observed_versions != expected_versions:
        raise ValueError("prepared target scorability Praat version drift")
    if not all(row["component_pair_scorable"] for row in payload["rows"]):
        raise ValueError("selected prepared target is not six-component scorable")
    return {
        "schema_version": SCORABILITY_SCHEMA,
        "stage": "prepared_target_confirmation",
        "scalar_values_retained_or_used": False,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "exact_scorer_versions": observed_versions,
        "rows": payload["rows"],
    }


def retained_waveform_equivalence(
    rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    original_by_case = {str(row["case_id"]): row for row in original_rows}
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        old = original_by_case.get(str(row["case_id"]))
        if old is None:
            continue
        fields = (
            "source_sha256",
            "target_sha256",
            "degraded_sha256",
            "base_sha256",
            "recipe_index",
            "recipe_uid",
            "simulation_seed",
            "noise_start_sample",
        )
        checks = {field: row[field] == old[field] for field in fields}
        comparisons.append(
            {
                "case_id": row["case_id"],
                "all_fields_identical": all(checks.values()),
                "field_identity": checks,
            }
        )
    if not comparisons or not all(row["all_fields_identical"] for row in comparisons):
        raise ValueError("retained v30 waveform equivalence failed")
    return {
        "schema_version": EQUIVALENCE_SCHEMA,
        "retained_case_count": len(comparisons),
        "all_retained_cases_byte_identical": True,
        "rows": comparisons,
    }


def extend_prior_ledger_v30r2(
    ledger: dict[str, Any],
    cases: list[v24.SVDCase],
    original_rows: list[dict[str, Any]],
    source_commit: str,
    original_ledger_sha256: str,
) -> dict[str, Any]:
    entries = [dict(entry) for entry in ledger["entries"]]
    selected = {case.panel_speaker_id for case in cases}
    original = {str(row["panel_speaker_id"]) for row in original_rows}
    case_by_speaker = {case.panel_speaker_id: case for case in cases[::2]}
    for entry in entries:
        canonical = str(entry["canonical_speaker_id"])
        if canonical in original:
            entry["candidate_e_v30r2_status"] = (
                "retained_in_original_recipe_slot"
                if canonical in selected
                else "target_component_unscorable_not_selected"
            )
    existing = {str(entry["canonical_speaker_id"]) for entry in entries}
    replacements = sorted(selected - original)
    for canonical in replacements:
        if canonical in existing:
            raise ValueError("v30r2 replacement already exists in v30 ledger")
        case = case_by_speaker[canonical]
        entries.append(
            {
                "dataset": "SVD",
                "speaker_id": case.speaker_id,
                "canonical_speaker_id": canonical,
                "panel_role": "shimmer_db_candidate_e_external_svd_v30r2",
                "session_id": case.session_id,
                "source_commit": source_commit,
                "exact_shimmer_outcomes_opened_at_ledger_update": False,
                "target_component_scorability_boolean_used": True,
                "target_scalar_values_used": False,
            }
        )
    output = {
        "schema_version": v24.PRIOR_LEDGER_SCHEMA,
        "exact_outcomes_used_for_selection": False,
        "target_component_scorability_boolean_used_for_selection": True,
        "target_scalar_values_used_for_selection": False,
        "entries": sorted(entries, key=lambda entry: entry["canonical_speaker_id"]),
        "added_by": "shimmer_db_candidate_e_external_svd_v30r2_panel_amendment",
        "added_speaker_count": len(replacements),
        "prior_v30_ledger_sha256": original_ledger_sha256,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    v24.validate_prior_ledger(output)
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not args.sv_root.is_dir() or not args.cs_root.is_dir():
        raise FileNotFoundError("SVD CS/SV root is missing")
    if not args.simulation_root.is_dir():
        raise FileNotFoundError(args.simulation_root)
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    source_provenance = validate_repository(args)
    input_paths = {
        "config": args.config,
        "v29_report": args.v29_report,
        "v29_receipt": args.v29_receipt,
        "prior_panel_speaker_ledger": args.prior_panel_speaker_ledger,
        "original_panel_seal": args.original_panel_seal,
        "original_panel_receipt": args.original_panel_receipt,
        "original_updated_ledger": args.original_updated_ledger,
        "sv_metadata": args.sv_metadata,
        "cs_metadata": args.cs_metadata,
        "fixed_recipes": args.fixed_recipes,
        "generator_config": args.generator_config,
        "generator_checkpoint": args.generator_checkpoint,
        "simulation_config": args.simulation_config,
    }
    source_hashes = {
        name: v24.validate_hash(path, getattr(args, f"{name}_sha256"), name)
        for name, path in input_paths.items()
    }
    simulation_source = args.simulation_root / "simulate_degradation.py"
    source_hashes["simulation_source"] = v24.validate_hash(
        simulation_source,
        args.simulation_source_sha256,
        "simulation source",
    )
    observed_avqi_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    config = v24.read_json(args.config)
    if config["scorability_amendment"].get("avqi_code_tree_sha256") != (
        observed_avqi_hash
    ):
        raise ValueError("v30r2 config/exact AVQI tree binding drift")
    v29_report = v24.read_json(args.v29_report)
    v29_receipt = v24.read_json(args.v29_receipt)
    validate_v29_authorization(
        config,
        v29_report,
        v29_receipt,
        report_sha256=source_hashes["v29_report"],
        receipt_sha256=source_hashes["v29_receipt"],
    )
    original_panel = v24.read_json(args.original_panel_seal)
    original_receipt = v24.read_json(args.original_panel_receipt)
    original_ledger = v24.read_json(args.original_updated_ledger)
    original_rows = validate_original_v30(
        config,
        original_panel,
        original_receipt,
        original_ledger,
        panel_sha256=source_hashes["original_panel_seal"],
        receipt_sha256=source_hashes["original_panel_receipt"],
        ledger_sha256=source_hashes["original_updated_ledger"],
    )
    prior_ledger = v24.read_json(args.prior_panel_speaker_ledger)
    excluded_speakers = v24.validate_prior_ledger(prior_ledger)
    ranked = ranked_svd_speakers(
        v24.read_csv(args.sv_metadata),
        v24.read_csv(args.cs_metadata),
        args.sv_root,
        args.cs_root,
        excluded_speakers,
    )
    selected, preflight = scan_ranked_speakers(
        ranked,
        args.exact_python,
        args.avqi_code_root,
    )
    assigned = assign_selected_to_original_slots(selected, original_rows)
    cases = build_cases(assigned)
    v24.validate_case_contract(cases, excluded_speakers)
    selection = build_selection_contract(cases, preflight, original_rows)
    recipes = v24.read_fixed_recipes(args.fixed_recipes)
    simulation_config = yaml.safe_load(
        args.simulation_config.read_text(encoding="utf-8")
    )
    if not isinstance(simulation_config, dict):
        raise ValueError("simulation config is not a mapping")
    simulation_config["stft_cfg"]["sampling_rate"] = v24.SAMPLE_RATE

    args.output_dir.mkdir(parents=True)
    preflight_path = args.output_dir / "target_scorability_preflight_v30r2.json"
    v24.write_json(preflight_path, preflight)
    prepared = v24.prepare_waveforms(args, cases, recipes, simulation_config)
    v24.run_frozen_generator(args, prepared)
    rows = v24.panel_rows(prepared)
    final_scorability = final_target_scorability(
        rows,
        args.exact_python,
        args.avqi_code_root,
        preflight["exact_scorer_versions"],
    )
    final_scorability_path = (
        args.output_dir / "target_scorability_confirmation_v30r2.json"
    )
    v24.write_json(final_scorability_path, final_scorability)
    equivalence = retained_waveform_equivalence(rows, original_rows)
    equivalence_path = args.output_dir / "retained_v30_equivalence_v30r2.json"
    v24.write_json(equivalence_path, equivalence)
    updated_ledger = extend_prior_ledger_v30r2(
        original_ledger,
        cases,
        original_rows,
        args.source_commit,
        source_hashes["original_updated_ledger"],
    )
    ledger_path = args.output_dir / "prior_panel_speaker_ledger_after_v30r2.json"
    v24.write_json(ledger_path, updated_ledger)
    seal = {
        "schema_version": PANEL_SCHEMA,
        "stage": "candidate_e_amended_prepare_and_seal_before_target_scalars",
        "scientific_stage_mapping": "v24_prepare_and_seal",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "authorization": {
            "candidate_e_v29_decision": v29_report["decision"],
            "v29_report_sha256": source_hashes["v29_report"],
            "v29_receipt_sha256": source_hashes["v29_receipt"],
            "original_v30_panel_sha256": source_hashes["original_panel_seal"],
            "original_v30_receipt_sha256": source_hashes[
                "original_panel_receipt"
            ],
            "external_panel_prepare_authorized": True,
            "old_v23_no_go_not_reinterpreted": True,
        },
        "selection": selection,
        "case_count": len(rows),
        "speaker_count": len({row["panel_speaker_id"] for row in rows}),
        "views": dict(Counter(row["view"] for row in rows)),
        "conditions": dict(Counter(row["condition"] for row in rows)),
        "sex": dict(Counter(row["sex"] for row in rows)),
        "severity_labels_created": False,
        "source_provenance": source_provenance,
        "source_sha256": {**source_hashes, "avqi_code_tree": observed_avqi_hash},
        "scorability_artifact_sha256": {
            preflight_path.name: v24.sha256_file(preflight_path),
            final_scorability_path.name: v24.sha256_file(final_scorability_path),
        },
        "retained_v30_equivalence_sha256": v24.sha256_file(equivalence_path),
        "prior_panel_speaker_ledger_input_sha256": source_hashes[
            "prior_panel_speaker_ledger"
        ],
        "prior_panel_speaker_ledger_after_v30r2_sha256": v24.sha256_file(
            ledger_path
        ),
        "recipe_assignment": {
            "indices": list(v24.RECIPE_ASSIGNMENT),
            "retained_v30_slots_preserved": True,
            "selection_uses_target_scalar_values": False,
        },
        "generator": {
            "candidate": "S3_500",
            "mode": "frozen_inference_only",
            "optimizer_created": False,
            "optimizer_steps": 0,
            "config_sha256": source_hashes["generator_config"],
            "checkpoint_sha256": source_hashes["generator_checkpoint"],
        },
        "waveform_contract": {
            "emitted_waveform_highpass": False,
            "exact_metric_highpass_branch_only": True,
            "target_is_same_speaker_same_view_clean_pathological": True,
            "full_band_pathology_guardrails_required_later": True,
            "denoising_nonregression_required_later": True,
        },
        "exact_contract": {
            "target_shimmer_scalar_values_opened": False,
            "target_scorability_boolean_opened": True,
            "base_exact_outcomes_opened": False,
            "candidate_exact_outcomes_opened": False,
            "target_scalar_stage_authorized": True,
            "selector_stage_authorized": False,
            "promotion_authorized": False,
        },
        "rows": rows,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    seal_path = args.output_dir / "panel_seal_v30r2.json"
    v24.write_json(seal_path, seal)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": PANEL_DECISION,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_shimmer_scalar_values_opened": False,
        "target_scorability_boolean_opened": True,
        "base_exact_outcomes_opened": False,
        "candidate_exact_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {
            seal_path.name: v24.sha256_file(seal_path),
            ledger_path.name: v24.sha256_file(ledger_path),
            preflight_path.name: v24.sha256_file(preflight_path),
            final_scorability_path.name: v24.sha256_file(final_scorability_path),
            equivalence_path.name: v24.sha256_file(equivalence_path),
        },
    }
    receipt_path = args.output_dir / "seal_receipt_v30r2.json"
    v24.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": PANEL_DECISION,
                "panel_seal_sha256": v24.sha256_file(seal_path),
                "updated_ledger_sha256": v24.sha256_file(ledger_path),
                "seal_receipt_sha256": v24.sha256_file(receipt_path),
                "retained_case_count": equivalence["retained_case_count"],
                "target_shimmer_scalar_values_opened": False,
                "target_scorability_boolean_opened": True,
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
