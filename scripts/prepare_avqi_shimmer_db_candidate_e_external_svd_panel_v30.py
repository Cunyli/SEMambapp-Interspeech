#!/usr/bin/env python3
"""Prepare and seal the Candidate-E external SVD panel (v24 successor).

The SVD selection, prior-ledger exclusion, recipe assignment, simulation, and
frozen S3_500 inference logic are inherited unchanged from v24.  Authorization
comes only from the passing frozen Candidate-E v29 confirmation.  No Shimmer
target, base, or candidate exact value is opened in this stage.
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
from scripts import prepare_avqi_shimmer_db_external_svd_panel_v24 as v24


PANEL_SCHEMA = "avqi-route-c-shimmer-db-candidate-e-external-svd-panel-v30"
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-panel-receipt-v30"
)
PANEL_DECISION = "SEALED_CANDIDATE_E_EXTERNAL_SVD_PANEL_EXACT_UNOPENED_V30"
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"


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
        raise ValueError("repository root does not contain v30 preparer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v30 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v30 preparation requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "preparer_sha256": v24.sha256_file(Path(__file__).resolve()),
        "inherited_v24_logic_sha256": v24.sha256_file(
            Path(v24.__file__).resolve()
        ),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")


def validate_v30_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("v30 config schema drift")
    selection = config.get("panel_selection", {})
    expected_selection = {
        "dataset": "SVD",
        "patient_only": True,
        "speaker_split_before_simulation": True,
        "speakers_per_sex": v24.SPEAKERS_PER_SEX,
        "sex_coverage": ["female", "male"],
        "views_per_speaker": list(v24.VIEWS),
        "conditions": list(v24.CONDITIONS),
        "selection_salt": v24.SELECTION_SALT,
        "selection_algorithm_inherited_unchanged_from_v24": True,
        "exclude_all_historical_tau_speakers": True,
        "exact_outcomes_used_for_selection": False,
    }
    for field, value in expected_selection.items():
        if selection.get(field) != value:
            raise ValueError(f"v30 panel-selection config drift: {field}")
    recipe = config.get("recipe_contract", {})
    if recipe.get("indices") != list(v24.RECIPE_ASSIGNMENT):
        raise ValueError("v30 recipe assignment drift")
    if recipe.get("assignment_inherited_unchanged_from_v24") is not True:
        raise ValueError("v30 recipe inheritance drift")
    if recipe.get("speaker_selection_before_simulation_required") is not True:
        raise ValueError("v30 speaker-selection ordering drift")
    exact = config.get("exact_contract", {})
    for field in (
        "target_shimmer_values_opened",
        "base_exact_outcomes_opened",
        "candidate_exact_outcomes_opened",
    ):
        if exact.get(field) is not False:
            raise ValueError(f"v30 exact-unopened boundary drift: {field}")
    boundaries = config.get("immutable_boundaries", {})
    if boundaries.get("old_v23_no_go_receipt_preserved") is not True:
        raise ValueError("v30 reinterprets the immutable v23 NO_GO")
    if (
        boundaries.get("old_v24_v25_v26_scripts_not_used_as_authorization")
        is not True
    ):
        raise ValueError("v30 reuses the Candidate-D authorization chain")
    if boundaries.get("candidate_e_source_config_alpha_selector_frozen") is not True:
        raise ValueError("v30 does not retain the Candidate-E freeze")
    if boundaries.get("no_final_waveform_highpass") is not True:
        raise ValueError("v30 final-waveform high-pass boundary drift")
    require_training_boundary(boundaries, "v30 config")


def validate_v29_authorization(
    config: dict[str, Any],
    report: dict[str, Any],
    receipt: dict[str, Any],
    *,
    report_sha256: str,
    receipt_sha256: str,
) -> None:
    validate_v30_config(config)
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
            raise ValueError(f"v30 authorization config drift: {field}")
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


def extend_prior_ledger_v30(
    ledger: dict[str, Any],
    cases: list[v24.SVDCase],
    source_commit: str,
) -> dict[str, Any]:
    """Reuse v24 speaker registration while versioning Candidate-E evidence."""
    output = v24.extend_prior_ledger(ledger, cases, source_commit)
    selected_speakers = {case.panel_speaker_id for case in cases}
    for entry in output["entries"]:
        if entry["canonical_speaker_id"] in selected_speakers:
            entry["panel_role"] = "shimmer_db_candidate_e_external_svd_v30"
    output["added_by"] = "shimmer_db_candidate_e_external_svd_v30_panel_seal"
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
    source_provenance = validate_repository(args)
    input_paths = {
        "config": args.config,
        "v29_report": args.v29_report,
        "v29_receipt": args.v29_receipt,
        "prior_panel_speaker_ledger": args.prior_panel_speaker_ledger,
        "sv_metadata": args.sv_metadata,
        "cs_metadata": args.cs_metadata,
        "fixed_recipes": args.fixed_recipes,
        "generator_config": args.generator_config,
        "generator_checkpoint": args.generator_checkpoint,
        "simulation_config": args.simulation_config,
    }
    source_hashes = {
        name: v24.validate_hash(
            path,
            getattr(args, f"{name}_sha256"),
            name,
        )
        for name, path in input_paths.items()
    }
    simulation_source = args.simulation_root / "simulate_degradation.py"
    source_hashes["simulation_source"] = v24.validate_hash(
        simulation_source,
        args.simulation_source_sha256,
        "simulation source",
    )
    config = v24.read_json(args.config)
    v29_report = v24.read_json(args.v29_report)
    v29_receipt = v24.read_json(args.v29_receipt)
    validate_v29_authorization(
        config,
        v29_report,
        v29_receipt,
        report_sha256=source_hashes["v29_report"],
        receipt_sha256=source_hashes["v29_receipt"],
    )
    if (
        config["panel_selection"].get("prior_speaker_ledger_sha256")
        != source_hashes["prior_panel_speaker_ledger"]
    ):
        raise ValueError("v30 prior-speaker-ledger config binding drift")
    prior_ledger = v24.read_json(args.prior_panel_speaker_ledger)
    excluded_speakers = v24.validate_prior_ledger(prior_ledger)
    cases, selection = v24.select_svd_cases(
        v24.read_csv(args.sv_metadata),
        v24.read_csv(args.cs_metadata),
        args.sv_root,
        args.cs_root,
        excluded_speakers,
    )
    recipes = v24.read_fixed_recipes(args.fixed_recipes)
    simulation_config = yaml.safe_load(
        args.simulation_config.read_text(encoding="utf-8")
    )
    if not isinstance(simulation_config, dict):
        raise ValueError("simulation config is not a mapping")
    simulation_config["stft_cfg"]["sampling_rate"] = v24.SAMPLE_RATE

    args.output_dir.mkdir(parents=True)
    prepared = v24.prepare_waveforms(args, cases, recipes, simulation_config)
    v24.run_frozen_generator(args, prepared)
    rows = v24.panel_rows(prepared)
    updated_ledger = extend_prior_ledger_v30(
        prior_ledger,
        cases,
        args.source_commit,
    )
    ledger_path = args.output_dir / "prior_panel_speaker_ledger_after_v30.json"
    v24.write_json(ledger_path, updated_ledger)
    seal = {
        "schema_version": PANEL_SCHEMA,
        "stage": "candidate_e_prepare_and_seal_before_external_exact",
        "scientific_stage_mapping": "v24_prepare_and_seal",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "authorization": {
            "candidate_e_v29_decision": v29_report["decision"],
            "v29_report_sha256": source_hashes["v29_report"],
            "v29_receipt_sha256": source_hashes["v29_receipt"],
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
        "severity_gate_source": (
            "passed Candidate-E v14 and opened-v15 evidence only"
        ),
        "source_provenance": source_provenance,
        "source_sha256": source_hashes,
        "prior_panel_speaker_ledger_input_sha256": source_hashes[
            "prior_panel_speaker_ledger"
        ],
        "prior_panel_speaker_ledger_after_v30_sha256": v24.sha256_file(
            ledger_path
        ),
        "recipe_assignment": {
            "indices": list(v24.RECIPE_ASSIGNMENT),
            "selection_uses_exact_outcomes": False,
            "inherited_unchanged_from_v24": True,
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
            "target_shimmer_values_opened": False,
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
    seal_path = args.output_dir / "panel_seal_v30.json"
    v24.write_json(seal_path, seal)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": PANEL_DECISION,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "exact_shimmer_outcomes_opened": False,
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
        },
    }
    receipt_path = args.output_dir / "seal_receipt_v30.json"
    v24.write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": PANEL_DECISION,
                "panel_seal_sha256": v24.sha256_file(seal_path),
                "updated_ledger_sha256": v24.sha256_file(ledger_path),
                "seal_receipt_sha256": v24.sha256_file(receipt_path),
                "exact_shimmer_outcomes_opened": False,
                "generator_optimizer_steps": 0,
                "authoritative_training_decision": TRAINING_DECISION,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
