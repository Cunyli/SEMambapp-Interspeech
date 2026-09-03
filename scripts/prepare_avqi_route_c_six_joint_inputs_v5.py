#!/usr/bin/env python3
"""Select and seal the fresh SVD inputs for Route C six-joint v5.

This stage is metadata-only and result-blind.  It excludes the complete prior
speaker ledger before salted ranking, freezes calibration/final speakers, and
only then assigns unique GAP recipes.  It performs no simulation, inference,
exact scoring, candidate generation, or optimizer step.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import soundfile as sf

from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    CLEAN_PATHOLOGICAL_ROLE,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    FROZEN_SPLIT_SEAL_SCHEMA_VERSION,
    GAP_NOISE_MANIFEST_SHA256,
    GAP_RECIPE_ASSIGNMENT_SALT,
    GAP_RIR_MANIFEST_SHA256,
    GAP_SIMULATION_INVENTORY_SHA256,
    GLOBAL_ALPHA_GRID,
    HEALTHY_ROLE,
    NORMALIZATION_SOURCE,
    NORMALIZATION_TARGET_MEAN_FIELD,
    NORMALIZATION_TARGET_SCALE_FIELD,
    PANEL_ROW_FIELDS,
    PATHOLOGICAL_ROLE,
    PRIOR_PANEL_LEDGER_SCHEMA,
    REQUIRED_CONDITIONS,
    REQUIRED_SPLITS,
    REQUIRED_VIEWS,
    SHIMMER_DB_LEDGER_SOURCE_KEY,
    SIX_GRADIENT_PASS_DECISION,
    SOURCE_DATASET,
    SOURCE_GENDER_ALLOCATION,
    SVD_CS_METADATA_SHA256,
    SVD_HEALTH_STATUS_MAPPING,
    SVD_MINIMUM_RAW_MONO_SECONDS,
    SVD_SPEAKER_SELECTION_SALT,
    SVD_SV_METADATA_SHA256,
    TRAINING_NO_GO,
    _validate_panel_rows,
    _validate_prior_panel_ledger_merge,
    _validate_split_seal,
)
from scripts.evaluate_avqi_shimmer_fresh_panel import read_fixed_recipes


SOURCE_MANIFEST_SCHEMA = "avqi-route-c-six-joint-speaker-source-manifest-v1"
RECIPE_MANIFEST_SCHEMA = "avqi-route-c-six-joint-recipe-manifest-v1"
SELECTION_RECEIPT_SCHEMA = "avqi-route-c-six-joint-input-seal-receipt-v5"
SELECTION_DECISION = "SEALED_SIX_JOINT_INPUTS_RESULT_BLIND_V5"
HEALTH_STATUS = dict(SVD_HEALTH_STATUS_MAPPING)
MINIMUM_SECONDS = {
    key: float(value) for key, value in SVD_MINIMUM_RAW_MONO_SECONDS
}
GENDER_QUOTAS = {
    (split, label): {"female": female, "male": male}
    for split, label, female, male in SOURCE_GENDER_ALLOCATION
}
CONDITION_RECIPE_SEMANTICS = {
    "clean": "no_simulation",
    "rir_only": "rir_only",
    "snr20": "rir_plus_noise_fixed_target_snr_20db",
    "snr10": "rir_plus_noise_fixed_target_snr_10db",
}


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json_payload(value), encoding="utf-8")


def json_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json_payload(value).encode("utf-8")).hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(f"{label} hash differs: {observed} != {expected}")
    return observed


def repository_source(root: Path, expected_commit: str) -> dict[str, str]:
    resolved = root.resolve()

    def git_value(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = git_value("rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("six-joint v5 source commit differs")
    if git_value("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("six-joint v5 selection requires a clean worktree")
    return {
        "root": str(resolved),
        "head": head,
        "branch": git_value("branch", "--show-current"),
        "tree": git_value("rev-parse", "HEAD^{tree}"),
    }


def canonical_speaker_id(speaker_id: str) -> str:
    normalized = speaker_id.strip()
    if not normalized or ":" in normalized:
        raise ValueError("invalid SVD speaker identity")
    return f"SVD:{normalized}"


def selection_digest(speaker_id: str, session_id: str) -> str:
    try:
        int(session_id)
    except ValueError as error:
        raise ValueError("SVD session ID must be numeric") from error
    payload = f"{SVD_SPEAKER_SELECTION_SALT}:{speaker_id}:{session_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prior_speakers(ledger: Mapping[str, Any]) -> set[str]:
    if (
        ledger.get("schema_version") != PRIOR_PANEL_LEDGER_SCHEMA
        or ledger.get("exact_outcomes_used_for_selection") is not False
    ):
        raise ValueError("prior speaker ledger boundary differs")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prior speaker ledger is empty")
    observed = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("prior speaker ledger entry differs")
        dataset = str(entry.get("dataset", "")).strip().upper()
        speaker_id = str(entry.get("speaker_id", "")).strip()
        canonical = str(entry.get("canonical_speaker_id", ""))
        if canonical != f"{dataset}:{speaker_id}" or canonical in observed:
            raise ValueError("prior speaker ledger canonical identity differs")
        observed.add(canonical)
    return observed


def merged_prior_ledger(
    ledger: Mapping[str, Any],
    ledger_sha256: str,
) -> dict[str, Any]:
    output = dict(ledger)
    output["source_ledger_sha256"] = {
        SHIMMER_DB_LEDGER_SOURCE_KEY: ledger_sha256
    }
    output["merged_by"] = "six_joint_v5_result_blind_selection"
    output["added_speaker_count"] = 0
    output["generator_optimizer_steps"] = 0
    output["authoritative_training_decision"] = TRAINING_NO_GO
    _validate_prior_panel_ledger_merge(
        output,
        ledger,
        shimmer_ledger_sha256=ledger_sha256,
    )
    return output


def eligible_speakers(
    sv_rows: list[dict[str, str]],
    cs_rows: list[dict[str, str]],
    sv_root: Path,
    cs_root: Path,
    excluded: set[str],
) -> list[dict[str, Any]]:
    sv_by_session = {row["session_id"]: row for row in sv_rows}
    cs_by_session = {row["session_id"]: row for row in cs_rows}
    per_speaker: dict[str, list[dict[str, Any]]] = {}
    shared_sessions = sorted(set(sv_by_session) & set(cs_by_session), key=int)
    for session_id in shared_sessions:
        sv_row = sv_by_session[session_id]
        cs_row = cs_by_session[session_id]
        speaker_id = str(sv_row.get("speaker id", "")).strip()
        health = str(sv_row.get("health status", "")).strip()
        gender = str(sv_row.get("gender", "")).strip().lower()
        if (
            not speaker_id
            or speaker_id != str(cs_row.get("speaker id", "")).strip()
            or health != str(cs_row.get("health status", "")).strip()
            or health not in HEALTH_STATUS
            or gender != str(cs_row.get("gender", "")).strip().lower()
            or gender not in {"female", "male"}
        ):
            continue
        paths = {
            "sv": (sv_root / sv_row["filename"]).resolve(),
            "cs": (cs_root / cs_row["filename"]).resolve(),
        }
        if any(not path.is_file() for path in paths.values()):
            continue
        info = {view: sf.info(path) for view, path in paths.items()}
        if any(value.channels != 1 for value in info.values()):
            continue
        durations = {
            view: value.frames / value.samplerate for view, value in info.items()
        }
        if any(durations[view] < MINIMUM_SECONDS[view] for view in REQUIRED_VIEWS):
            continue
        diagnosis = str(sv_row.get("diagnosis", ""))
        if diagnosis != str(cs_row.get("diagnosis", "")):
            raise ValueError(f"SVD diagnosis record differs: {session_id}")
        per_speaker.setdefault(speaker_id, []).append(
            {
                "speaker_id": speaker_id,
                "session_id": session_id,
                "label": HEALTH_STATUS[health],
                "health_status": health,
                "gender": gender,
                "diagnosis_record_only": diagnosis,
                "paths": paths,
                "durations": durations,
                "sample_rates": {
                    view: info[view].samplerate for view in REQUIRED_VIEWS
                },
                "source_frames": {
                    view: info[view].frames for view in REQUIRED_VIEWS
                },
            }
        )
    retained_before_exclusion = [
        min(rows, key=lambda row: int(row["session_id"]))
        for rows in per_speaker.values()
    ]
    retained = [
        row
        for row in retained_before_exclusion
        if canonical_speaker_id(row["speaker_id"]) not in excluded
    ]
    for row in retained:
        row["selection_digest"] = selection_digest(
            row["speaker_id"], row["session_id"]
        )
    return retained


def select_speakers(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {
        (label, gender): sorted(
            [
                row
                for row in eligible
                if row["label"] == label and row["gender"] == gender
            ],
            key=lambda row: (
                row["selection_digest"],
                row["speaker_id"],
                int(row["session_id"]),
            ),
        )
        for label in ("patient", "healthy")
        for gender in ("female", "male")
    }
    cursors = {key: 0 for key in buckets}
    selected = []
    for split in REQUIRED_SPLITS:
        for label in ("patient", "healthy"):
            for gender in ("female", "male"):
                count = GENDER_QUOTAS[(split, label)][gender]
                key = (label, gender)
                start = cursors[key]
                stop = start + count
                if len(buckets[key]) < stop:
                    raise ValueError(f"insufficient SVD speakers for {key}")
                for rank_index, source in enumerate(
                    buckets[key][start:stop],
                    start=start + 1,
                ):
                    selected.append(
                        {
                            **source,
                            "split": split,
                            "rank_within_label_gender": rank_index,
                        }
                    )
                cursors[key] = stop
    if len(selected) != 12 or len({row["speaker_id"] for row in selected}) != 12:
        raise ValueError("six-joint speaker coverage differs")
    return selected


def panel_rows(selected: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for speaker in selected:
        for condition in REQUIRED_CONDITIONS:
            for view in REQUIRED_VIEWS:
                if speaker["label"] == "healthy":
                    role = HEALTHY_ROLE
                elif condition == "clean":
                    role = CLEAN_PATHOLOGICAL_ROLE
                else:
                    role = PATHOLOGICAL_ROLE
                rows.append(
                    {
                        "case_id": (
                            f"six_joint_v5__{speaker['split']}__"
                            f"SVD_{speaker['speaker_id']}__{view}__{condition}"
                        ),
                        "dataset": SOURCE_DATASET,
                        "speaker_id": speaker["speaker_id"],
                        "split": speaker["split"],
                        "view": view,
                        "condition": condition,
                        "label": speaker["label"],
                        "optimization_role": role,
                    }
                )
    _validate_panel_rows(rows, "six-joint v5 selection")
    return rows


def source_manifest(
    selected: list[dict[str, Any]],
    *,
    prior_ledger_sha256: str,
    source_prior_ledger_sha256: str,
) -> dict[str, Any]:
    rows = []
    for item in selected:
        waveforms = {
            view: {
                "path": str(item["paths"][view]),
                "sha256": sha256_file(item["paths"][view]),
                "duration_seconds": item["durations"][view],
                "source_sample_rate": item["sample_rates"][view],
                "source_frames": item["source_frames"][view],
                "channels": 1,
            }
            for view in REQUIRED_VIEWS
        }
        rows.append(
            {
                "dataset": SOURCE_DATASET,
                "canonical_speaker_id": canonical_speaker_id(item["speaker_id"]),
                "speaker_id": item["speaker_id"],
                "session_id": item["session_id"],
                "split": item["split"],
                "label": item["label"],
                "health_status": item["health_status"],
                "gender": item["gender"],
                "diagnosis_record_only": item["diagnosis_record_only"],
                "rank_within_label_gender": item["rank_within_label_gender"],
                "selection_digest": item["selection_digest"],
                "waveforms": waveforms,
            }
        )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "source_dataset": SOURCE_DATASET,
        "sv_metadata_sha256": SVD_SV_METADATA_SHA256,
        "cs_metadata_sha256": SVD_CS_METADATA_SHA256,
        "prior_panel_speaker_ledger_sha256": prior_ledger_sha256,
        "source_prior_panel_speaker_ledger_sha256": (
            source_prior_ledger_sha256
        ),
        "selection_salt": SVD_SPEAKER_SELECTION_SALT,
        "selection_mode": "metadata_only_result_blind",
        "selection_operation_order": [
            "map_health_status",
            "pair_cs_sv_by_same_session",
            "filter_raw_mono_minimum_duration",
            "retain_minimum_numeric_eligible_session_per_speaker",
            "exclude_prior_ledger_speakers",
            "bucket_by_health_status_and_gender",
            "rank_by_salted_sha256",
            "allocate_calibration_then_final_by_frozen_gender_quota",
        ],
        "diagnosis_used_for_selection": False,
        "exact_scores_opened": False,
        "candidate_outcomes_opened": False,
        "mild_severe_labels_created": False,
        "speaker_count": len(rows),
        "counts": {
            f"{split}:{label}:{gender}": sum(
                row["split"] == split
                and row["label"] == label
                and row["gender"] == gender
                for row in rows
            )
            for split in REQUIRED_SPLITS
            for label in ("patient", "healthy")
            for gender in ("female", "male")
        },
        "rows": rows,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def recipe_manifest(
    rows: list[dict[str, str]],
    recipes: list[dict[str, Any]],
    *,
    fixed_recipes_sha256: str,
) -> dict[str, Any]:
    usable = [
        (index, recipe)
        for index, recipe in enumerate(recipes)
        if recipe.get("split") == "test"
        and int(recipe.get("target_sample_rate", 0)) == 16_000
        and str(recipe.get("uid", ""))
    ]
    if len(usable) < 72:
        raise ValueError("insufficient frozen test recipes")
    used_indices: set[int] = set()
    assignments = []
    for row in sorted(rows, key=lambda value: value["case_id"]):
        case_id = row["case_id"]
        condition = row["condition"]
        assignment_digest = hashlib.sha256(
            f"{GAP_RECIPE_ASSIGNMENT_SALT}:{case_id}".encode("utf-8")
        ).hexdigest()
        if condition == "clean":
            recipe_index = None
            fixed_uid = None
            recipe_sha256 = None
            recipe_uid = f"clean_no_simulation_{assignment_digest[:24]}"
        else:
            candidates = sorted(
                usable,
                key=lambda value: hashlib.sha256(
                    (
                        f"{GAP_RECIPE_ASSIGNMENT_SALT}:{case_id}:"
                        f"{value[1]['uid']}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            recipe_index, recipe = next(
                value for value in candidates if value[0] not in used_indices
            )
            used_indices.add(recipe_index)
            fixed_uid = str(recipe["uid"])
            recipe_uid = fixed_uid
            recipe_sha256 = hashlib.sha256(
                json.dumps(
                    recipe,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        assignments.append(
            {
                "case_id": case_id,
                "split": row["split"],
                "condition": condition,
                "recipe_uid": recipe_uid,
                "recipe_index": recipe_index,
                "fixed_recipe_uid": fixed_uid,
                "fixed_recipe_sha256": recipe_sha256,
                "assignment_digest": assignment_digest,
                "simulation_applied": condition != "clean",
                "rir_applied": condition != "clean",
                "noise_applied": condition in {"snr20", "snr10"},
                "target_snr_db": (
                    20.0
                    if condition == "snr20"
                    else 10.0 if condition == "snr10" else None
                ),
            }
        )
    recipe_uids = [row["recipe_uid"] for row in assignments]
    if len(assignments) != 96 or len(set(recipe_uids)) != 96:
        raise ValueError("six-joint recipe UID coverage differs")
    if len(used_indices) != 72:
        raise ValueError("six-joint degraded recipe assignment is not unique")
    return {
        "schema_version": RECIPE_MANIFEST_SCHEMA,
        "fixed_recipes_sha256": fixed_recipes_sha256,
        "assignment_salt": GAP_RECIPE_ASSIGNMENT_SALT,
        "speaker_split_completed_before_assignment": True,
        "condition_recipe_semantics": CONDITION_RECIPE_SEMANTICS,
        "row_count": len(assignments),
        "recipe_uid_unique_per_row": True,
        "recipe_uid_reused_across_splits": False,
        "exact_scores_opened": False,
        "candidate_outcomes_opened": False,
        "rows": assignments,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def split_seal(
    rows: list[dict[str, str]],
    *,
    gate_sha256: str,
    target_sha256: str,
    ledger_sha256: str,
    source_sha256: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": FROZEN_SPLIT_SEAL_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "source_dataset": SOURCE_DATASET,
        "sv_metadata_sha256": SVD_SV_METADATA_SHA256,
        "cs_metadata_sha256": SVD_CS_METADATA_SHA256,
        "health_status_mapping": HEALTH_STATUS,
        "paired_cs_sv_same_session_required": True,
        "minimum_raw_mono_duration_seconds": MINIMUM_SECONDS,
        "eligible_session_per_speaker": "minimum_numeric_session_id",
        "speaker_selection_salt": SVD_SPEAKER_SELECTION_SALT,
        "speaker_rank_digest": "SHA256(salt:speaker_id:session_id)",
        "prior_ledger_excluded_before_hash_ranking": True,
        "diagnosis_used_for_selection": False,
        "exact_scores_opened": False,
        "speaker_split_before_simulation": True,
        "gap_simulation_inventory_sha256": GAP_SIMULATION_INVENTORY_SHA256,
        "gap_rir_manifest_sha256": GAP_RIR_MANIFEST_SHA256,
        "gap_noise_manifest_sha256": GAP_NOISE_MANIFEST_SHA256,
        "recipe_assignment_salt": GAP_RECIPE_ASSIGNMENT_SALT,
        "recipe_uid_unique_per_row": True,
        "recipe_uid_reused_across_splits": False,
        "condition_recipe_semantics": CONDITION_RECIPE_SEMANTICS,
        "metadata_only_result_blind_selection": True,
        "mild_severe_labels_created": False,
        "prior_panel_speaker_overlap": 0,
        "waveform_steps": 1,
        "one_global_alpha": True,
        "gradient_normalization": "waveform_rms_normalized",
        "alpha_grid": list(GLOBAL_ALPHA_GRID),
        "zero_alpha_selectable": False,
        "alpha_required_gate_families": [
            "all_six_components",
            "equal_weight_joint",
            "all_required_efficacy_slices",
            "waveform_safety",
            "full_band_pathology",
            "denoising",
        ],
        "alpha_required_gate_split": "calibration",
        "alpha_selection_objective": (
            "maximize_equal_weight_joint_exact_median_normalized_gap_reduction"
        ),
        "alpha_selection_tie_break": "smaller_alpha",
        "alpha_selection_split": "calibration",
        "final_tuning_permitted": False,
        "optimization_weight_source_decision": SIX_GRADIENT_PASS_DECISION,
        "optimization_weights_calibration_only": True,
        "optimization_weights_used_for_exact_joint_decision": False,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization_target_mean_field": NORMALIZATION_TARGET_MEAN_FIELD,
        "normalization_target_scale_field": NORMALIZATION_TARGET_SCALE_FIELD,
        "normalization_refit_permitted": False,
        "healthy_no_step_does_not_establish_optimized_healthy_safety": True,
        "joint_gate_contract_sha256": gate_sha256,
        "target_value_protocol_sha256": target_sha256,
        "prior_panel_speaker_ledger_sha256": ledger_sha256,
        "fresh_speaker_source_manifest_sha256": source_sha256,
        "joint_recipe_assignment_manifest_sha256": recipe_sha256,
        "rows": [
            {field: row[field] for field in PANEL_ROW_FIELDS} for row in rows
        ],
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "sv-metadata",
        "cs-metadata",
        "prior-panel-speaker-ledger",
        "fixed-recipes",
        "gap-simulation-inventory",
        "gap-rir-manifest",
        "gap-noise-manifest",
        "joint-gate-contract",
        "target-value-protocol",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--cs-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite input seal: {args.output_dir}")
    if not args.sv_root.is_dir() or not args.cs_root.is_dir():
        raise FileNotFoundError("SVD CS/SV root is unavailable")
    source = repository_source(args.source_root, args.source_commit)
    expected_hashes = {
        "sv_metadata": SVD_SV_METADATA_SHA256,
        "cs_metadata": SVD_CS_METADATA_SHA256,
        "gap_simulation_inventory": GAP_SIMULATION_INVENTORY_SHA256,
        "gap_rir_manifest": GAP_RIR_MANIFEST_SHA256,
        "gap_noise_manifest": GAP_NOISE_MANIFEST_SHA256,
    }
    input_paths = {
        "sv_metadata": args.sv_metadata,
        "cs_metadata": args.cs_metadata,
        "prior_panel_speaker_ledger": args.prior_panel_speaker_ledger,
        "fixed_recipes": args.fixed_recipes,
        "gap_simulation_inventory": args.gap_simulation_inventory,
        "gap_rir_manifest": args.gap_rir_manifest,
        "gap_noise_manifest": args.gap_noise_manifest,
        "joint_gate_contract": args.joint_gate_contract,
        "target_value_protocol": args.target_value_protocol,
    }
    hashes = {}
    for key, path in input_paths.items():
        expected = getattr(args, f"{key}_sha256")
        hashes[key] = validate_hash(path, expected, key)
        if key in expected_hashes and hashes[key] != expected_hashes[key]:
            raise ValueError(f"{key} is not the frozen source")
    prior = read_json(args.prior_panel_speaker_ledger, "prior ledger")
    excluded = prior_speakers(prior)
    merged = merged_prior_ledger(
        prior,
        hashes["prior_panel_speaker_ledger"],
    )
    eligible = eligible_speakers(
        read_csv(args.sv_metadata),
        read_csv(args.cs_metadata),
        args.sv_root,
        args.cs_root,
        excluded,
    )
    selected = select_speakers(eligible)
    rows = panel_rows(selected)
    merged_ledger_sha256 = json_sha256(merged)
    source_value = source_manifest(
        selected,
        prior_ledger_sha256=merged_ledger_sha256,
        source_prior_ledger_sha256=hashes["prior_panel_speaker_ledger"],
    )
    recipe_value = recipe_manifest(
        rows,
        read_fixed_recipes(args.fixed_recipes),
        fixed_recipes_sha256=hashes["fixed_recipes"],
    )

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "prior_panel_speaker_ledger_joint_v5.json"
    source_path = args.output_dir / "fresh_speaker_source_manifest.json"
    recipe_path = args.output_dir / "joint_recipe_assignment_manifest.json"
    write_json(ledger_path, merged)
    write_json(source_path, source_value)
    write_json(recipe_path, recipe_value)
    output_hashes = {
        "prior_panel_speaker_ledger": sha256_file(ledger_path),
        "fresh_speaker_source_manifest": sha256_file(source_path),
        "joint_recipe_assignment_manifest": sha256_file(recipe_path),
    }
    if output_hashes["prior_panel_speaker_ledger"] != merged_ledger_sha256:
        raise ValueError("six-joint prior ledger serialization drifted")
    split_value = split_seal(
        rows,
        gate_sha256=hashes["joint_gate_contract"],
        target_sha256=hashes["target_value_protocol"],
        ledger_sha256=output_hashes["prior_panel_speaker_ledger"],
        source_sha256=output_hashes["fresh_speaker_source_manifest"],
        recipe_sha256=output_hashes["joint_recipe_assignment_manifest"],
    )
    split_path = args.output_dir / "fresh_panel_split_seal.json"
    write_json(split_path, split_value)
    output_hashes["fresh_panel_split_seal"] = sha256_file(split_path)
    _validate_split_seal(
        split_value,
        gate_sha256=hashes["joint_gate_contract"],
        target_sha256=hashes["target_value_protocol"],
        ledger_sha256=output_hashes["prior_panel_speaker_ledger"],
        source_sha256=output_hashes["fresh_speaker_source_manifest"],
    )
    receipt = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "decision": SELECTION_DECISION,
        "source": source,
        "input_binding": {
            key: {"path": str(path.resolve()), "sha256": hashes[key]}
            for key, path in input_paths.items()
        },
        "input_sha256": hashes,
        "runtime_binding": {
            "sv_root": str(args.sv_root.resolve()),
            "cs_root": str(args.cs_root.resolve()),
        },
        "artifact_sha256": output_hashes,
        "speaker_count": len(selected),
        "row_count": len(rows),
        "eligible_speaker_count": len(eligible),
        "selected_counts": dict(
            Counter(f"{row['split']}:{row['label']}" for row in selected)
        ),
        "prior_panel_speaker_overlap": 0,
        "metadata_only_result_blind_selection": True,
        "speaker_split_before_recipe_assignment": True,
        "exact_scores_opened": False,
        "candidate_outcomes_opened": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "selection_completion_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
