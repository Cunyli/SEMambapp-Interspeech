#!/usr/bin/env python3
"""Post-seal exact-Praat adjudication of the deterministic opened24 panel.

This opened-development adjudicator is downstream of a successful v22
deterministic capture and its bound repeat.  It re-scores the 24 durable
selected PCM24 waveforms with exact Praat, evaluates the already-frozen
Shimmer-dB effect and safety gates, and audits selector anti-shortcuts.

A PASS may authorize creation of a new speaker-disjoint external panel.  It
does not promote Shimmer dB, authorize the six-component joint panel, or run a
generator optimizer.  The immutable v18 comparison remains separate evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from scripts import evaluate_avqi_shimmer_db_runtime_v19_full_step_integration as v22
from scripts import evaluate_avqi_shimmer_fresh_panel as fresh
from scripts import evaluate_avqi_shimmer_hybrid_topology as hybrid
from scripts import evaluate_direct_avqi_waveform_optimization as direct
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    topology_stability,
)


REPORT_SCHEMA = "avqi-route-c-shimmer-db-opened24-exact-adjudication-v23"
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-opened24-exact-adjudication-receipt-v23"
)
PASS_DECISION = (
    "PASS_SHIMMER_DB_OPENED24_EXACT_ADJUDICATION_"
    "EXTERNAL_PANEL_AUTHORIZED_V23"
)
FAIL_DECISION = "NO_GO_SHIMMER_DB_OPENED24_EXACT_ADJUDICATION_V23"
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
V22_SOURCE_COMMIT = "5dc360d92f8045c9290a3504c03afcfbf06504ad"
V22_REPORT_SCHEMA = "avqi-route-c-shimmer-db-full-step-integration-v2"
V22_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-full-step-integration-receipt-v2"
)
TARGET_CONTRACT_SCHEMA = "avqi-route-c-shimmer-db-supervised-target-v1"
PANEL_SCHEMAS = {
    "v14": "avqi-route-c-shimmer-db-candidate-c-fresh-panel-v1",
    "v15": "avqi-route-c-shimmer-db-candidate-c-fresh-panel-runtime-v15-v1",
}
EXPECTED_CASE_COUNT = 24
EXPECTED_PANEL_CASE_COUNT = 12
EXPECTED_SPEAKER_COUNT = 12
EXPECTED_REPEATS = 3
SHIMMER_DB_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_db")
TARGET_REPRODUCTION_ABS_TOLERANCE = 1e-9


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "v22-capture-report",
        "v22-capture-receipt",
        "v22-deterministic-baseline-manifest",
        "v22-repeat-report",
        "v22-repeat-receipt",
        "v22-durable-selected-csv",
        "v14-panel-contract",
        "v14-target-contract",
        "v14-fresh-results",
        "v15-panel-contract",
        "v15-target-contract",
        "v15-fresh-results",
        "predictor-checkpoint",
    ):
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty exact-results CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid Boolean {label}: {value!r}")


def git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_repository_provenance(args: argparse.Namespace) -> dict[str, str]:
    root = args.repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain this v23 evaluator")
    observed_head = git_output(root, "rev-parse", "HEAD")
    if observed_head != args.source_commit:
        raise ValueError("v23 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v23 exact adjudication requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": observed_head,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }


def require_training_boundary(
    value: dict[str, Any],
    label: str,
    *,
    require_promotion_flag: bool = True,
) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} authoritative training decision drift")
    if require_promotion_flag and value.get("promotion_authorized") is not False:
        raise ValueError(f"{label} unexpectedly authorizes promotion")


def validate_v22_chain_payloads(
    capture_report: dict[str, Any],
    capture_receipt: dict[str, Any],
    baseline_manifest: dict[str, Any],
    repeat_report: dict[str, Any],
    repeat_receipt: dict[str, Any],
    *,
    capture_report_sha256: str,
    capture_receipt_sha256: str,
    baseline_manifest_sha256: str,
    repeat_report_sha256: str,
    repeat_receipt_sha256: str,
    durable_csv_sha256: str,
) -> dict[str, Any]:
    reports = {
        "capture": (capture_report, "deterministic_capture"),
        "repeat": (repeat_report, "deterministic_repeat"),
    }
    for label, (report, mode) in reports.items():
        if report.get("schema_version") != V22_REPORT_SCHEMA:
            raise ValueError(f"v22 {label} report schema drift")
        if report.get("source_commit") != V22_SOURCE_COMMIT:
            raise ValueError(f"v22 {label} source commit drift")
        if report.get("candidate_reference_mode") != mode:
            raise ValueError(f"v22 {label} reference mode drift")
        if report.get("candidate_exact_avqi_components_opened") is not False:
            raise ValueError(f"v22 {label} opened exact candidate outcomes")
        if report.get("exact_component_scoring_requested") is not False:
            raise ValueError(f"v22 {label} requested exact component scoring")
        if report.get("v18_artifacts_mutated") is not False:
            raise ValueError(f"v22 {label} mutated immutable v18 artifacts")
        if report.get("candidate_input_contract", {}).get(
            "immutable_v18_comparison_disclosed_separately"
        ) is not True:
            raise ValueError(f"v22 {label} collapsed the v18 comparison")
        if report.get("migration_review", {}).get(
            "immutable_v18_kept_separate"
        ) is not True:
            raise ValueError(f"v22 {label} migration boundary drift")
        if not v22.integration_authorized(report["gates"], mode):
            raise ValueError(f"v22 {label} gates did not pass")
        require_training_boundary(report, f"v22 {label} report")

    if capture_report.get("decision") != v22.CAPTURE_PASS_DECISION:
        raise ValueError("v22 deterministic capture decision drift")
    if capture_report.get("deterministic_repeat_authorized") is not True:
        raise ValueError("v22 capture did not authorize a bound repeat")
    if capture_report.get("new_sealed_panel_authorized") is not False:
        raise ValueError("v22 capture over-authorized a sealed panel")
    if repeat_report.get("decision") != v22.REPEAT_PASS_DECISION:
        raise ValueError("v22 deterministic repeat decision drift")
    if repeat_report.get("deterministic_repeat_authorized") is not False:
        raise ValueError("v22 repeat recursively authorized another repeat")
    if repeat_report.get("new_sealed_panel_authorized") is not True:
        raise ValueError("v22 repeat did not authorize post-seal adjudication")
    if repeat_report.get("deterministic_baseline_binding") is None:
        raise ValueError("v22 repeat lacks its capture binding")

    if baseline_manifest.get("schema_version") != v22.DETERMINISTIC_MANIFEST_SCHEMA:
        raise ValueError("v22 deterministic manifest schema drift")
    if baseline_manifest.get("source_commit") != V22_SOURCE_COMMIT:
        raise ValueError("v22 deterministic manifest source drift")
    if baseline_manifest.get("candidate_reference_mode") != "deterministic_capture":
        raise ValueError("v22 deterministic manifest is not a capture")
    if baseline_manifest.get("deterministic_repeat_authorized") is not True:
        raise ValueError("v22 deterministic manifest did not authorize repeat")
    if baseline_manifest.get("historical_v18_kept_separate") is not True:
        raise ValueError("v22 deterministic manifest collapsed v18 evidence")
    if baseline_manifest.get("candidate_exact_avqi_components_opened") is not False:
        raise ValueError("v22 deterministic manifest opened exact outcomes")
    if len(baseline_manifest.get("attempt_references", [])) != (
        v22.EXPECTED_REFERENCE_ATTEMPT_COUNT
    ):
        raise ValueError("v22 deterministic manifest attempt coverage drift")
    require_training_boundary(
        baseline_manifest,
        "v22 deterministic manifest",
        require_promotion_flag=False,
    )
    if baseline_manifest.get("new_sealed_panel_authorized") is not False:
        raise ValueError("v22 deterministic manifest over-authorized sealed work")

    receipt_inputs = {
        "capture": (
            capture_receipt,
            v22.CAPTURE_PASS_DECISION,
            capture_report_sha256,
        ),
        "repeat": (
            repeat_receipt,
            v22.REPEAT_PASS_DECISION,
            repeat_report_sha256,
        ),
    }
    for label, (receipt, decision, report_hash) in receipt_inputs.items():
        if receipt.get("schema_version") != V22_RECEIPT_SCHEMA:
            raise ValueError(f"v22 {label} receipt schema drift")
        if receipt.get("decision") != decision:
            raise ValueError(f"v22 {label} receipt decision drift")
        if receipt.get("source_commit") != V22_SOURCE_COMMIT:
            raise ValueError(f"v22 {label} receipt source drift")
        if receipt.get("artifact_sha256", {}).get("diagnostic_report.json") != (
            report_hash
        ):
            raise ValueError(f"v22 {label} receipt report binding drift")
        if receipt.get("candidate_exact_avqi_components_opened") is not False:
            raise ValueError(f"v22 {label} receipt opened exact outcomes")
        require_training_boundary(receipt, f"v22 {label} receipt")

    if capture_receipt.get("deterministic_repeat_authorized") is not True:
        raise ValueError("v22 capture receipt did not authorize repeat")
    if capture_receipt.get("new_sealed_panel_authorized") is not False:
        raise ValueError("v22 capture receipt over-authorized sealed work")
    if capture_receipt.get("artifact_sha256", {}).get(
        "deterministic_baseline_manifest.json"
    ) != baseline_manifest_sha256:
        raise ValueError("v22 capture receipt manifest binding drift")
    if repeat_receipt.get("new_sealed_panel_authorized") is not True:
        raise ValueError("v22 repeat receipt did not authorize adjudication")
    if repeat_receipt.get("artifact_sha256", {}).get(
        "durable_selected_equivalence.csv"
    ) != durable_csv_sha256:
        raise ValueError("v22 repeat receipt durable CSV binding drift")

    capture_output = capture_report.get("deterministic_baseline_output", {})
    if capture_output.get("sha256") != baseline_manifest_sha256:
        raise ValueError("v22 capture report manifest binding drift")
    if capture_output.get("deterministic_repeat_authorized") is not True:
        raise ValueError("v22 capture report manifest was not authorized")
    capture_migration = capture_report.get("migration_review", {}).get("sha256")
    repeat_migration = repeat_report.get("migration_review", {}).get("sha256")
    manifest_migration = baseline_manifest.get("migration_review_sha256")
    if not capture_migration or not (
        capture_migration == repeat_migration == manifest_migration
    ):
        raise ValueError("v22 migration review binding drift")

    return {
        "capture_report_sha256": capture_report_sha256,
        "capture_receipt_sha256": capture_receipt_sha256,
        "deterministic_baseline_manifest_sha256": baseline_manifest_sha256,
        "repeat_report_sha256": repeat_report_sha256,
        "repeat_receipt_sha256": repeat_receipt_sha256,
        "durable_selected_csv_sha256": durable_csv_sha256,
        "migration_review_sha256": capture_migration,
        "v18_evidence_kept_separate": True,
        "candidate_exact_closed_during_selection": True,
        "new_sealed_panel_authorized_by_repeat": True,
    }


def load_v22_chain(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    path_names = {
        "capture_report": "v22_capture_report",
        "capture_receipt": "v22_capture_receipt",
        "baseline_manifest": "v22_deterministic_baseline_manifest",
        "repeat_report": "v22_repeat_report",
        "repeat_receipt": "v22_repeat_receipt",
        "durable_csv": "v22_durable_selected_csv",
    }
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for key, attribute in path_names.items():
        path = getattr(args, attribute)
        paths[key] = path
        hashes[key] = validate_hash(
            path,
            getattr(args, f"{attribute}_sha256"),
            attribute,
        )
    capture_report = read_json(paths["capture_report"])
    capture_receipt = read_json(paths["capture_receipt"])
    baseline_manifest = read_json(paths["baseline_manifest"])
    repeat_report = read_json(paths["repeat_report"])
    repeat_receipt = read_json(paths["repeat_receipt"])
    evidence = validate_v22_chain_payloads(
        capture_report,
        capture_receipt,
        baseline_manifest,
        repeat_report,
        repeat_receipt,
        capture_report_sha256=hashes["capture_report"],
        capture_receipt_sha256=hashes["capture_receipt"],
        baseline_manifest_sha256=hashes["baseline_manifest"],
        repeat_report_sha256=hashes["repeat_report"],
        repeat_receipt_sha256=hashes["repeat_receipt"],
        durable_csv_sha256=hashes["durable_csv"],
    )
    return {
        "capture_report": capture_report,
        "capture_receipt": capture_receipt,
        "baseline_manifest": baseline_manifest,
        "repeat_report": repeat_report,
        "repeat_receipt": repeat_receipt,
        "durable_rows": read_csv(paths["durable_csv"]),
    }, evidence


def validate_waveform_hash(row: dict[str, Any], role: str) -> None:
    path = Path(row[f"{role}_path"])
    validate_hash(path, str(row[f"{role}_sha256"]), f"{role} waveform")
    info = sf.info(path)
    if info.samplerate != hybrid.SAMPLE_RATE or info.channels != 1:
        raise ValueError(f"{role} waveform format drift: {row['case_id']}")


def validate_panel_payloads(
    label: str,
    panel: dict[str, Any],
    target_contract: dict[str, Any],
    result_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if panel.get("schema_version") != PANEL_SCHEMAS[label]:
        raise ValueError(f"{label} panel schema drift")
    if panel.get("speaker_split_before_simulation") is not True:
        raise ValueError(f"{label} panel was not speaker-split before simulation")
    if panel.get("panel_status") != "sealed_new_speaker_panel_before_exact_outcomes":
        raise ValueError(f"{label} panel was not sealed before exact outcomes")
    rows = [dict(row) for row in panel.get("rows", [])]
    case_ids = [str(row.get("case_id")) for row in rows]
    speakers = {str(row.get("speaker_id")) for row in rows}
    if len(rows) != EXPECTED_PANEL_CASE_COUNT or len(set(case_ids)) != len(rows):
        raise ValueError(f"{label} panel case coverage drift")
    if len(speakers) != EXPECTED_PANEL_CASE_COUNT // 2:
        raise ValueError(f"{label} panel speaker coverage drift")
    if Counter(str(row["view"]) for row in rows) != Counter({"cs": 6, "sv": 6}):
        raise ValueError(f"{label} panel view balance drift")
    if Counter(str(row["sample_group"]) for row in rows) != Counter(
        {"pathological_mild": 6, "pathological_severe": 6}
    ):
        raise ValueError(f"{label} panel severity balance drift")
    if Counter(str(row["condition"]) for row in rows) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError(f"{label} panel condition balance drift")
    for speaker in speakers:
        selected = [row for row in rows if str(row["speaker_id"]) == speaker]
        if len(selected) != 2 or {row["view"] for row in selected} != {"cs", "sv"}:
            raise ValueError(f"{label} speaker view pairing drift: {speaker}")
        if len({row["sample_group"] for row in selected}) != 1:
            raise ValueError(f"{label} speaker severity drift: {speaker}")
        if len({row["condition"] for row in selected}) != 1:
            raise ValueError(f"{label} speaker condition drift: {speaker}")
    for row in rows:
        for role in ("base", "target"):
            validate_waveform_hash(row, role)

    if target_contract.get("schema_version") != TARGET_CONTRACT_SCHEMA:
        raise ValueError(f"{label} target contract schema drift")
    target_boundary = {
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
    }
    for key, expected in target_boundary.items():
        if target_contract.get(key) is not expected:
            raise ValueError(f"{label} target anti-shortcut drift: {key}")
    target_rows = {
        str(row["case_id"]): dict(row)
        for row in target_contract.get("rows", [])
    }
    if set(target_rows) != set(case_ids):
        raise ValueError(f"{label} target contract coverage drift")
    panel_by_case = {str(row["case_id"]): row for row in rows}
    for case_id, target_row in target_rows.items():
        panel_row = panel_by_case[case_id]
        if target_row.get("target_sha256") != panel_row.get("target_sha256"):
            raise ValueError(f"{label} target waveform binding drift: {case_id}")
        if target_row.get("speaker_id") != panel_row.get("speaker_id"):
            raise ValueError(f"{label} target speaker binding drift: {case_id}")
        if target_row.get("view") != panel_row.get("view"):
            raise ValueError(f"{label} target view binding drift: {case_id}")
        if not math.isfinite(float(target_row["exact_target_shimmer_db"])):
            raise ValueError(f"{label} non-finite target scalar: {case_id}")

    results = {str(row.get("case_id")): dict(row) for row in result_rows}
    if len(result_rows) != EXPECTED_PANEL_CASE_COUNT or set(results) != set(case_ids):
        raise ValueError(f"{label} frozen gradient result coverage drift")
    for case_id, row in results.items():
        if not parse_bool(row.get("gradient_finite"), f"{label}/{case_id} gradient"):
            raise ValueError(f"{label} frozen gradient is non-finite: {case_id}")
        norm = float(row["gradient_l2_norm"])
        if not hybrid.GRADIENT_NORM_RANGE[0] <= norm <= hybrid.GRADIENT_NORM_RANGE[1]:
            raise ValueError(f"{label} frozen gradient norm drift: {case_id}")
        if float(row["fixed_alpha"]) != hybrid.FIXED_ALPHA:
            raise ValueError(f"{label} frozen alpha drift: {case_id}")
        if row.get("optimized_component") != "shimmer_db":
            raise ValueError(f"{label} optimized component drift: {case_id}")
        validate_hash(
            Path(row["candidate_path"]),
            str(row["candidate_sha256"]),
            f"{label} frozen candidate evidence",
        )
    return rows, target_rows, results


def load_opened_panels(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    combined_rows: list[dict[str, Any]] = []
    combined_targets: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    speaker_sets: dict[str, set[str]] = {}
    for label in ("v14", "v15"):
        artifacts: dict[str, Path] = {}
        artifact_hashes: dict[str, str] = {}
        for artifact in ("panel_contract", "target_contract", "fresh_results"):
            attribute = f"{label}_{artifact}"
            path = getattr(args, attribute)
            artifacts[artifact] = path
            artifact_hashes[artifact] = validate_hash(
                path,
                getattr(args, f"{attribute}_sha256"),
                attribute,
            )
        panel_rows, target_rows, gradient_rows = validate_panel_payloads(
            label,
            read_json(artifacts["panel_contract"]),
            read_json(artifacts["target_contract"]),
            read_csv(artifacts["fresh_results"]),
        )
        speaker_sets[label] = {str(row["speaker_id"]) for row in panel_rows}
        for row in panel_rows:
            case_id = str(row["case_id"])
            row["opened_panel"] = label
            row["opened_role"] = (
                "development_calibration" if label == "v14" else "opened_validation"
            )
            row["frozen_gradient_l2_norm"] = float(
                gradient_rows[case_id]["gradient_l2_norm"]
            )
            row["frozen_gradient_finite"] = True
            combined_rows.append(row)
        overlap = set(combined_targets) & set(target_rows)
        if overlap:
            raise ValueError(
                f"duplicate case IDs across opened panels: {sorted(overlap)}"
            )
        combined_targets.update(target_rows)
        evidence[label] = {
            **{f"{name}_sha256": value for name, value in artifact_hashes.items()},
            "case_count": len(panel_rows),
            "speaker_count": len(speaker_sets[label]),
            "role": (
                "development_calibration" if label == "v14" else "opened_validation"
            ),
        }
    if speaker_sets["v14"] & speaker_sets["v15"]:
        raise ValueError("v14/v15 opened speaker sets overlap")
    if len(combined_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("combined opened24 case coverage drift")
    if len(speaker_sets["v14"] | speaker_sets["v15"]) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("combined opened24 speaker coverage drift")
    return combined_rows, combined_targets, evidence


def validate_durable_rows(
    rows: list[dict[str, str]],
    expected_case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if len(rows) != EXPECTED_CASE_COUNT * EXPECTED_REPEATS:
        raise ValueError("v22 durable selected row coverage drift")
    selected: dict[str, dict[str, Any]] = {}
    observed_paths: set[Path] = set()
    for repeat_index in range(1, EXPECTED_REPEATS + 1):
        repeat_rows = [
            row for row in rows if int(row["repeat_index"]) == repeat_index
        ]
        if len(repeat_rows) != EXPECTED_CASE_COUNT:
            raise ValueError(f"v22 durable repeat {repeat_index} coverage drift")
        if {str(row["case_id"]) for row in repeat_rows} != expected_case_ids:
            raise ValueError(f"v22 durable repeat {repeat_index} case-set drift")
        for row in repeat_rows:
            case_id = str(row["case_id"])
            if not parse_bool(
                row.get("selected_candidate_present"),
                f"{case_id} selected candidate",
            ):
                raise ValueError(f"v22 durable candidate missing: {case_id}")
            for field in (
                "durable_byte_equivalence_pass",
                "selected_path_updated_to_durable_before_future_seal",
                "durable_copy_after_timed_step",
            ):
                if not parse_bool(row.get(field), f"{case_id} {field}"):
                    raise ValueError(f"v22 durable boundary failed: {case_id}/{field}")
            path = Path(row["durable_selected_path"])
            resolved_path = path.resolve()
            if resolved_path in observed_paths:
                raise ValueError(f"duplicate v22 durable path: {resolved_path}")
            observed_paths.add(resolved_path)
            observed_hash = validate_hash(
                path,
                str(row["durable_selected_sha256"]),
                f"v22 durable selected PCM24 {case_id}/repeat {repeat_index}",
            )
            info = sf.info(path)
            if (
                info.samplerate != hybrid.SAMPLE_RATE
                or info.channels != 1
                or info.subtype != "PCM_24"
            ):
                raise ValueError(f"v22 durable PCM24 format drift: {case_id}")
            identity = {
                "selected_family": str(row["selected_family"]),
                "selected_alpha": float(row["selected_alpha"]),
                "durable_selected_sha256": observed_hash,
            }
            if repeat_index == 1:
                selected[case_id] = {
                    **identity,
                    "candidate_path": str(resolved_path),
                }
            elif any(
                selected[case_id][key] != value
                for key, value in identity.items()
            ):
                raise ValueError(f"v22 durable repeat identity drift: {case_id}")
    return selected


def selected_proxy_evidence(
    baseline_manifest: dict[str, Any],
    expected_case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in baseline_manifest["attempt_references"]:
        if row.get("selected_attempt") is not True:
            continue
        case_id = str(row["case_id"])
        if case_id in selected:
            raise ValueError(f"duplicate selected deterministic attempt: {case_id}")
        selected[case_id] = dict(row)
    if set(selected) != expected_case_ids:
        raise ValueError("deterministic selected-attempt coverage drift")
    return selected


def validate_selected_bindings(
    durable: dict[str, dict[str, Any]],
    proxy: dict[str, dict[str, Any]],
) -> None:
    for case_id, durable_row in durable.items():
        proxy_row = proxy[case_id]
        if proxy_row.get("candidate_sha256") != durable_row["durable_selected_sha256"]:
            raise ValueError(f"selected deterministic hash binding drift: {case_id}")
        if str(proxy_row.get("selected_family")) != durable_row["selected_family"]:
            raise ValueError(f"selected deterministic family binding drift: {case_id}")
        if float(proxy_row.get("selected_alpha")) != durable_row["selected_alpha"]:
            raise ValueError(f"selected deterministic alpha binding drift: {case_id}")
        if proxy_row.get("pcm24_effective_step_pass") is not True:
            raise ValueError(f"selected PCM24 step was ineffective: {case_id}")
        if proxy_row.get("finite_safety_pass") is not True:
            raise ValueError(f"selected deterministic safety failed: {case_id}")


def build_exact_items(
    panel_rows: list[dict[str, Any]],
    durable: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in panel_rows:
        case_id = str(row["case_id"])
        common = {"case_id": case_id, "view": row["view"], "score_components": True}
        items.extend(
            [
                {
                    **common,
                    "id": f"target:{case_id}",
                    "role": "same_speaker_clean_pathological_target",
                    "path": str(Path(row["target_path"]).resolve()),
                    "exact_metric_topology": False,
                },
                {
                    **common,
                    "id": f"base:{case_id}",
                    "role": "current_output_before_step",
                    "path": str(Path(row["base_path"]).resolve()),
                    "exact_metric_topology": True,
                },
                {
                    **common,
                    "id": f"candidate:{case_id}",
                    "role": "durable_selected_after_step",
                    "path": durable[case_id]["candidate_path"],
                    "exact_metric_topology": True,
                },
            ]
        )
    return items


def validate_exact_payload(
    payload: dict[str, Any],
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows", [])
    if [row.get("id") for row in rows] != expected_ids:
        raise ValueError("exact-Praat row order or coverage drift")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        if row.get("scoring_status") != "ok":
            raise ValueError(
                f"exact-Praat scoring failed for {row_id}: "
                f"{row.get('error_type')} {row.get('error_message')}"
            )
        components = row.get("components", {})
        values = np.asarray(
            [components.get(name, math.nan) for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite exact components: {row_id}")
        if row_id.startswith(("base:", "candidate:")):
            if int(row.get("metric_reconstruction_max_pcm16_error", -1)) != 0:
                raise ValueError(f"exact metric reconstruction error: {row_id}")
            if int(row.get("metric_reconstruction_differing_samples", -1)) != 0:
                raise ValueError(f"exact metric reconstruction drift: {row_id}")
            if int(row.get("pulse_count", 0)) < 3:
                raise ValueError(f"insufficient exact pulse topology: {row_id}")
        indexed[row_id] = dict(row)
    return indexed


def read_waveform(path: Path) -> torch.Tensor:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != hybrid.SAMPLE_RATE or values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected mono 16 kHz waveform: {path}")
    waveform = torch.from_numpy(values.copy())
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"non-finite waveform: {path}")
    return waveform


def build_result_rows(
    panel_rows: list[dict[str, Any]],
    target_contract: dict[str, dict[str, Any]],
    durable: dict[str, dict[str, Any]],
    proxy: dict[str, dict[str, Any]],
    exact: dict[str, dict[str, Any]],
    target_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        target_exact = exact[f"target:{case_id}"]
        base_exact = exact[f"base:{case_id}"]
        candidate_exact = exact[f"candidate:{case_id}"]
        target = np.asarray(
            [target_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        base = np.asarray(
            [base_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [candidate_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        frozen_target = float(target_contract[case_id]["exact_target_shimmer_db"])
        target_error = abs(float(target[SHIMMER_DB_INDEX]) - frozen_target)
        row: dict[str, Any] = {
            "case_id": case_id,
            "opened_panel": panel_row["opened_panel"],
            "opened_role": panel_row["opened_role"],
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": "v22_deterministic_selector_selected_pcm24",
            "optimized_component": "shimmer_db",
            "selected_family": durable[case_id]["selected_family"],
            "selected_alpha": durable[case_id]["selected_alpha"],
            "candidate_path": durable[case_id]["candidate_path"],
            "candidate_sha256": durable[case_id]["durable_selected_sha256"],
            "frozen_gradient_l2_norm": panel_row["frozen_gradient_l2_norm"],
            "frozen_gradient_finite": panel_row["frozen_gradient_finite"],
            "frozen_target_shimmer_db": frozen_target,
            "rescored_target_shimmer_db": float(target[SHIMMER_DB_INDEX]),
            "target_reproduction_abs_error_shimmer_db": target_error,
            "target_reproduction_pass": (
                target_error <= TARGET_REPRODUCTION_ABS_TOLERANCE
            ),
            "selector_proxy_before_shimmer_db": float(proxy[case_id]["proxy_before"]),
            "selector_proxy_after_shimmer_db": float(
                proxy[case_id]["proxy_after_frozen_topology"]
            ),
            "candidate_exact_pulse_count": int(candidate_exact["pulse_count"]),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                candidate_exact["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                candidate_exact["metric_reconstruction_differing_samples"]
            ),
            "base_metric_reconstruction_max_pcm16_error": int(
                base_exact["metric_reconstruction_max_pcm16_error"]
            ),
            "base_metric_reconstruction_differing_samples": int(
                base_exact["metric_reconstruction_differing_samples"]
            ),
            "clean_target_topology_drives_output": False,
            "candidate_exact_opened_only_after_durable_seal": True,
        }
        fresh.component_fields(row, target, base, candidate, target_scale)
        row["material_shimmer_db_gap"] = (
            row["exact_absolute_gap_before_shimmer_db"]
            / max(float(target_scale[SHIMMER_DB_INDEX]), 1e-8)
            > hybrid.MATERIAL_GAP_THRESHOLD
        )
        row["forward_normalized_abs_error_shimmer_db"] = (
            abs(
                row["selector_proxy_after_shimmer_db"]
                - row["exact_after_shimmer_db"]
            )
            / max(float(target_scale[SHIMMER_DB_INDEX]), 1e-8)
        )
        row.update(topology_stability(base_exact, candidate_exact))
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        candidate_waveform = read_waveform(Path(durable[case_id]["candidate_path"]))
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        row.update(
            hybrid.waveform_safety(
                base_waveform.numpy(),
                candidate_waveform.numpy(),
            )
        )
        row.update(
            direct.full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)
    return rows


def summarize_effect(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    material = [row for row in rows if row["material_shimmer_db_gap"]]
    reductions = [
        float(row["exact_normalized_gap_reduction_shimmer_db"])
        for row in material
    ]
    improvement_fraction = (
        sum(value > 0.0 for value in reductions) / len(reductions)
        if reductions
        else 0.0
    )
    median_reduction = median(reductions) if reductions else None
    nonselected_components = [
        name for name in AVQI_COMPONENT_NAMES if name != "shimmer_db"
    ]
    nonselected_medians = {
        name: median(
            -float(row[f"exact_normalized_gap_reduction_{name}"])
            for row in rows
        )
        for name in nonselected_components
    }
    predicates = {
        "view=cs": lambda row: row["view"] == "cs",
        "view=sv": lambda row: row["view"] == "sv",
        "severity=pathological_mild": (
            lambda row: row["sample_group"] == "pathological_mild"
        ),
        "severity=pathological_severe": (
            lambda row: row["sample_group"] == "pathological_severe"
        ),
        "condition=rir_only": lambda row: row["condition"] == "rir_only",
        "condition=snr20": lambda row: row["condition"] == "snr20",
        "condition=snr10": lambda row: row["condition"] == "snr10",
    }
    slices = {
        name: hybrid.summarize_effect_slice(
            [row for row in rows if predicate(row)]
        )
        for name, predicate in predicates.items()
    }
    pathology = direct.aggregate_pathology_guardrails(rows)
    denoising = direct.aggregate_denoising(rows)
    gates = {
        "complete_case_coverage": len(rows) == expected_rows,
        "material_cases_ge_5": len(material) >= 5,
        "exact_db_effect": (
            bool(reductions)
            and median_reduction is not None
            and median_reduction >= hybrid.MEDIAN_REDUCTION_GATE
            and improvement_fraction >= hybrid.IMPROVEMENT_FRACTION_GATE
        ),
        "required_effect_slices": all(
            slices[name]["decision"] == "PASS"
            for name in hybrid.REQUIRED_EFFECT_SLICES
        ),
        "gradient": all(
            row["frozen_gradient_finite"]
            and hybrid.GRADIENT_NORM_RANGE[0]
            <= float(row["frozen_gradient_l2_norm"])
            <= hybrid.GRADIENT_NORM_RANGE[1]
            for row in rows
        ),
        "nonselected": all(
            value <= hybrid.NONSELECTED_MEDIAN_INCREASE_GATE
            for value in nonselected_medians.values()
        ),
        "waveform_safety": all(
            float(row["residual_rms_db"]) <= hybrid.RESIDUAL_CEILING_DB
            and float(row["cosine_similarity"]) >= hybrid.MINIMUM_COSINE
            and float(row["clip_fraction"]) <= hybrid.MAXIMUM_CLIP_FRACTION
            for row in rows
        ),
        "exact_topology_stability": all(
            bool(row["topology_stability_pass"]) for row in rows
        ),
        "exact_metric_mapping_parity": all(
            int(row["base_metric_reconstruction_max_pcm16_error"]) == 0
            and int(row["base_metric_reconstruction_differing_samples"]) == 0
            and int(row["candidate_metric_reconstruction_max_pcm16_error"]) == 0
            and int(row["candidate_metric_reconstruction_differing_samples"]) == 0
            for row in rows
        ),
        "exact_target_reproduction": all(
            bool(row["target_reproduction_pass"]) for row in rows
        ),
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
    }
    return {
        "rows": len(rows),
        "material_rows": len(material),
        "median_exact_db_normalized_gap_reduction": median_reduction,
        "exact_db_improvement_fraction": improvement_fraction,
        "nonselected_median_normalized_gap_increase": nonselected_medians,
        "slices": slices,
        "pathology_guardrails": pathology,
        "denoising": denoising,
        "selector_forward_error_diagnostic_only": {
            "threshold_applied": False,
            "median": median(
                float(row["forward_normalized_abs_error_shimmer_db"])
                for row in rows
            ),
            "maximum": max(
                float(row["forward_normalized_abs_error_shimmer_db"])
                for row in rows
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def combined_exact_effect_pass(rows: list[dict[str, Any]]) -> bool:
    material = [row for row in rows if row["material_shimmer_db_gap"]]
    reductions = [
        float(row["exact_normalized_gap_reduction_shimmer_db"])
        for row in material
    ]
    return (
        bool(reductions)
        and median(reductions) >= hybrid.MEDIAN_REDUCTION_GATE
        and sum(value > 0.0 for value in reductions) / len(reductions)
        >= hybrid.IMPROVEMENT_FRACTION_GATE
    )


def anti_shortcut_contract() -> dict[str, bool]:
    return {
        "candidate_exact_closed_during_v22_selection": True,
        "candidate_exact_scored_only_after_durable_pcm24_seal": True,
        "target_scalar_frozen_before_candidate_exact_open": True,
        "clean_target_topology_not_used_by_selector": True,
        "selector_uses_proxy_and_frozen_thresholds_only": True,
        "old_v18_comparison_preserved_as_separate_evidence": True,
        "opened24_not_reused_as_external_promotion_panel": True,
    }


def completion_summary(report: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "external_speaker_panel_authorized",
        "scientific_promotion_granted",
        "joint_panel_authorized",
        "generator_optimizer_steps",
        "authoritative_training_decision",
    }
    if not required.issubset(report):
        missing = sorted(required - set(report))
        raise KeyError(f"v23 completion summary field missing: {missing}")
    summary = {key: report[key] for key in sorted(required)}
    if summary["scientific_promotion_granted"] is not False:
        raise ValueError("v23 report over-authorizes scientific promotion")
    if summary["joint_panel_authorized"] is not False:
        raise ValueError("v23 report over-authorizes joint panel execution")
    if summary["generator_optimizer_steps"] != 0:
        raise ValueError("v23 report optimizer-step boundary drift")
    if summary["authoritative_training_decision"] != TRAINING_DECISION:
        raise ValueError("v23 report training decision drift")
    expected_external = summary["decision"] == PASS_DECISION
    if summary["external_speaker_panel_authorized"] is not expected_external:
        raise ValueError("v23 external-panel authorization/decision mismatch")
    return summary


def write_completion_receipt(
    args: argparse.Namespace,
    report: dict[str, Any],
    report_path: Path,
    exact_csv_path: Path | None,
    input_sha256: dict[str, Any],
) -> Path:
    summary = completion_summary(report)
    artifacts = {report_path.name: sha256_file(report_path)}
    if exact_csv_path is not None:
        artifacts[exact_csv_path.name] = sha256_file(exact_csv_path)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        **summary,
        "input_sha256": input_sha256,
        "artifact_sha256": artifacts,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path


def exact_failure_report(
    args: argparse.Namespace,
    source_provenance: dict[str, str],
    input_sha256: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "decision": FAIL_DECISION,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "phase": "opened24_post_seal_exact_adjudication",
        "exact_scoring_complete": False,
        "exact_error": {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        },
        "source_provenance": source_provenance,
        "input_sha256": input_sha256,
        "external_speaker_panel_authorized": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    observed_tree_hash = direct.avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_provenance = validate_repository_provenance(args)
    v22_payloads, v22_evidence = load_v22_chain(args)
    panel_rows, target_contract, panel_evidence = load_opened_panels(args)
    expected_case_ids = {str(row["case_id"]) for row in panel_rows}
    durable = validate_durable_rows(
        v22_payloads["durable_rows"],
        expected_case_ids,
    )
    proxy = selected_proxy_evidence(
        v22_payloads["baseline_manifest"],
        expected_case_ids,
    )
    validate_selected_bindings(durable, proxy)
    predictor_hash = validate_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "frozen Shimmer predictor checkpoint",
    )
    _, _, _, target_scale_tensor = hybrid.load_predictor(
        args.predictor_checkpoint,
        torch.device("cpu"),
    )
    target_scale = target_scale_tensor.detach().cpu().numpy().astype(np.float64)
    input_sha256 = {
        "v22": v22_evidence,
        "opened_panels": panel_evidence,
        "predictor_checkpoint_sha256": predictor_hash,
        "avqi_code_tree_sha256": observed_tree_hash,
    }

    args.output_dir.mkdir(parents=True)
    exact_items = build_exact_items(panel_rows, durable)
    try:
        exact_payload = hybrid.run_exact(
            exact_items,
            args.exact_python,
            args.avqi_code_root,
        )
        exact = validate_exact_payload(
            exact_payload,
            [str(item["id"]) for item in exact_items],
        )
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        report = exact_failure_report(
            args,
            source_provenance,
            input_sha256,
            error,
        )
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        write_completion_receipt(
            args,
            report,
            report_path,
            None,
            input_sha256,
        )
        print(json.dumps(completion_summary(report), sort_keys=True), flush=True)
        return

    result_rows = build_result_rows(
        panel_rows,
        target_contract,
        durable,
        proxy,
        exact,
        target_scale,
    )
    panel_summaries = {
        label: summarize_effect(
            [row for row in result_rows if row["opened_panel"] == label],
            EXPECTED_PANEL_CASE_COUNT,
        )
        for label in ("v14", "v15")
    }
    anti_shortcut = anti_shortcut_contract()
    gates = {
        "v22_deterministic_chain_bound_and_passed": True,
        "opened24_contract_complete_and_speaker_disjoint": True,
        "v14_frozen_scientific_gates": panel_summaries["v14"]["all_gates_pass"],
        "v15_frozen_scientific_gates": panel_summaries["v15"]["all_gates_pass"],
        "combined_global_exact_effect": combined_exact_effect_pass(result_rows),
        "anti_shortcut_contract": all(anti_shortcut.values()),
        "old_v18_evidence_kept_separate": True,
    }
    passed = all(gates.values())
    decision = PASS_DECISION if passed else FAIL_DECISION
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "phase": "opened24_post_seal_exact_adjudication",
        "opened_development_evidence_only": True,
        "exact_scoring_complete": True,
        "exact_scorer_versions": {
            "parselmouth": exact_payload["parselmouth_version"],
            "praat": exact_payload["praat_version"],
        },
        "source_provenance": source_provenance,
        "input_sha256": input_sha256,
        "case_count": len(result_rows),
        "speaker_count": len({row["speaker_id"] for row in result_rows}),
        "candidate_reference": "v22_repeat_1_byte_identical_across_three_repeats",
        "fixed_scientific_thresholds": {
            "candidate_d_fixed_alpha": hybrid.FIXED_ALPHA,
            "opened_gradient_evidence_alpha": hybrid.FIXED_ALPHA,
            "material_normalized_gap": hybrid.MATERIAL_GAP_THRESHOLD,
            "median_normalized_reduction": hybrid.MEDIAN_REDUCTION_GATE,
            "improvement_fraction": hybrid.IMPROVEMENT_FRACTION_GATE,
            "nonselected_median_increase": hybrid.NONSELECTED_MEDIAN_INCREASE_GATE,
            "gradient_l2_range": list(hybrid.GRADIENT_NORM_RANGE),
            "residual_ceiling_db": hybrid.RESIDUAL_CEILING_DB,
            "minimum_cosine": hybrid.MINIMUM_COSINE,
            "maximum_clip_fraction": hybrid.MAXIMUM_CLIP_FRACTION,
            "target_reproduction_abs_tolerance": TARGET_REPRODUCTION_ABS_TOLERANCE,
        },
        "panel_summaries": panel_summaries,
        "combined_exact_effect": {
            "gate_pass": gates["combined_global_exact_effect"],
            "calibration_and_validation_both_required": True,
        },
        "selector_forward_error_use": (
            "diagnostic_only_no_new_numerical_threshold"
        ),
        "frozen_gradient_evidence": {
            "source": "v14_v15_opened_fresh_results_same_24_cases",
            "new_gradient_hash_repeat_claimed": False,
            "finite_and_norm_range_revalidated": True,
        },
        "anti_shortcut": anti_shortcut,
        "gates": gates,
        "external_speaker_panel_authorized": passed,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    exact_csv_path = args.output_dir / "opened24_exact_results.csv"
    report_path = args.output_dir / "diagnostic_report.json"
    write_csv(exact_csv_path, result_rows)
    write_json(report_path, report)
    receipt_path = write_completion_receipt(
        args,
        report,
        report_path,
        exact_csv_path,
        input_sha256,
    )
    print(
        json.dumps(
            {
                **completion_summary(report),
                "report_sha256": sha256_file(report_path),
                "exact_csv_sha256": sha256_file(exact_csv_path),
                "completion_receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
