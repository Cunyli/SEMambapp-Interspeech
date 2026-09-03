#!/usr/bin/env python3
"""Materialize sealed six-joint targets, bases, topology, and gradients.

The fresh split and recipe assignment are already sealed.  This stage runs the
frozen S3_500 generator in inference mode, opens exact Praat only for the clean
same-speaker target bank, obtains result-blind base-current topology, and emits
one full-length six-component joint gradient per patient row.  Candidate exact
outcomes stay closed and generator optimizer steps remain zero.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch
import yaml

from model.avqi_route_c import (
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    six_active_bidirectional_gap_losses,
)
from model.avqi_route_c_candidate_e import (
    CANDIDATE_E_RUNTIME_CLIENT_SHA256,
    CANDIDATE_E_RUNTIME_CONFIG_SHA256,
    CANDIDATE_E_SELECTOR_SHA256,
    CANDIDATE_E_SOURCE_COMMIT,
    CANDIDATE_E_REFERENCE_SHA256,
    CANDIDATE_E_WORKER_SHA256,
    build_cycle_gain_plan,
    candidate_e_proxy,
    project_cycle_gain_gradient_fixed_order,
    validate_candidate_e_base_peak_certificate,
)
from model.avqi_route_c_candidate_e_scorer import (
    load_route_c_candidate_e_six_scorer,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v4 import (
    PROMOTION_PASS,
    PROMOTION_RECEIPT_SHA256,
    PROMOTION_REPORT_SHA256,
    UPDATED_LEDGER_SHA256,
)
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    EXPECTED_TOTAL_ROWS,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    HEALTHY_ROLE,
    NORMALIZATION_SOURCE,
    PRIOR_PANEL_LEDGER_SCHEMA,
    SIX_GRADIENT_PASS_DECISION,
    SVD_CS_METADATA_SHA256,
    SVD_SPEAKER_SELECTION_SALT,
    SVD_SV_METADATA_SHA256,
    TRAINING_NO_GO,
    _finite_mapping,
    _require_optimizer_zero,
    _validate_panel_rows,
    _validate_six_gradient,
    _validate_split_seal,
)
from scripts.evaluate_avqi_component_backprop import (
    enhance_waveform,
    load_generator,
    set_model_seed,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    run_exact,
    validate_exact_authority,
)
from scripts.evaluate_avqi_shimmer_fresh_panel import (
    read_fixed_recipes,
    recipe_wds_row,
)
from scripts.prepare_avqi_component_expanded_data import (
    WdsReader,
    crop_or_tile,
    match_length,
    read_clean,
    stable_seed,
)
from scripts.prepare_avqi_route_c_six_joint_inputs_v5 import (
    GAP_RECIPE_ASSIGNMENT_SALT,
    GENDER_QUOTAS,
    HEALTH_STATUS,
    MINIMUM_SECONDS,
    RECIPE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    canonical_speaker_id,
    repository_source,
    selection_digest,
    validate_hash,
)
from scripts.prepare_avqi_route_c_six_joint_waveforms import (
    GRADIENT_MANIFEST_SCHEMA_VERSION,
    TARGET_BANK_SCHEMA_VERSION,
)
from scripts.seal_avqi_route_c_exact_authority_v1 import (
    RECEIPT_SCHEMA as EXACT_AUTHORITY_RECEIPT_SCHEMA,
    SEAL_DECISION as EXACT_AUTHORITY_SEAL_DECISION,
)
from utils import load_config


SAMPLE_RATE = 16_000
OUTPUT_SUBTYPE = "FLOAT"
MATERIALIZATION_RECEIPT_SCHEMA = (
    "avqi-route-c-six-joint-materialization-receipt-v5"
)
MATERIALIZATION_DECISION = "SEALED_SIX_JOINT_TARGETS_AND_GRADIENTS_V5"
RUNTIME_BINDING_SCHEMA = "avqi-route-c-candidate-e-joint-runtime-binding-v1"
GRADIENT_SOURCE = (
    "six_active_bidirectional_gap_losses_candidate_e_v32r8_exact_path_"
    "fixed_order_projection"
)
EXPECTED_CANDIDATE_E_COMMIT = CANDIDATE_E_SOURCE_COMMIT
EXPECTED_CANDIDATE_E_WORKER_SHA256 = CANDIDATE_E_WORKER_SHA256
EXPECTED_CANDIDATE_E_RUNTIME_CLIENT_SHA256 = CANDIDATE_E_RUNTIME_CLIENT_SHA256
EXPECTED_CANDIDATE_E_SELECTOR_SHA256 = CANDIDATE_E_SELECTOR_SHA256
EXPECTED_CANDIDATE_E_RUNTIME_CONFIG_SHA256 = CANDIDATE_E_RUNTIME_CONFIG_SHA256
EXPECTED_SIMULATION_CONFIG_SHA256 = (
    "0e665b5f3d97ad617cd1dde22a84b1ec5a8089e31b7657c7cb9989363115e276"
)
EXPECTED_SIMULATION_SOURCE_SHA256 = (
    "7f74a5727122bf3f8a6dbee297d9f3dd10165cba3bf2312bf2bd8704abc273bb"
)
EXPECTED_GENERATOR_CONFIG_SHA256 = (
    "5c3f75ecd2b2a9fa9c938509e9ac8917bb19b22fdbdbf07b275c868685360407"
)
EXPECTED_GENERATOR_CHECKPOINT_SHA256 = (
    "d1ef31ec180b2378fce5a36b5a29ae5a601ed2fa1a26b1b18a81de7941bc6dd9"
)
MATERIALIZATION_INPUT_NAMES = (
    "split_seal",
    "fresh_speaker_source_manifest",
    "prior_panel_speaker_ledger",
    "joint_recipe_assignment_manifest",
    "fixed_recipes",
    "simulation_config",
    "simulation_source",
    "generator_config",
    "generator_checkpoint",
    "six_gradient_raw_report",
    "six_gradient_report",
    "six_gradient_receipt",
    "candidate_e_promotion_report",
    "candidate_e_promotion_receipt",
    "candidate_e_reference_source",
    "candidate_e_runtime_client",
    "candidate_e_worker",
    "candidate_e_selector_source",
    "candidate_e_runtime_config",
    "exact_code_tree_manifest",
    "exact_runtime_manifest",
    "exact_authority_receipt",
    "cpps_checkpoint",
    "hnr_checkpoint",
    "shimmer_checkpoint",
    "slope_checkpoint",
    "tilt_checkpoint",
)


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_case_id(case_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in case_id
    )
    if not safe:
        raise ValueError("empty safe case ID")
    return safe


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def load_external_module(path: Path, expected_sha256: str) -> ModuleType:
    validate_hash(path, expected_sha256, "Candidate-E runtime client")
    spec = importlib.util.spec_from_file_location(
        "candidate_e_exact_topology_runtime_v32r8",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load Candidate-E exact topology runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_repository(root: Path, expected_commit: str) -> dict[str, str]:
    source = repository_source(root, expected_commit)
    if expected_commit != EXPECTED_CANDIDATE_E_COMMIT:
        raise ValueError("Candidate-E promoted commit differs")
    return source


def write_audio(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed audio: {path}")
    audio = np.asarray(values, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"invalid materialized audio: {path}")
    sf.write(path, audio, SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
    stored, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if (
        sample_rate != SAMPLE_RATE
        or stored.ndim != 1
        or stored.shape != audio.shape
        or info.subtype != OUTPUT_SUBTYPE
        or not np.array_equal(stored, audio)
    ):
        raise ValueError(f"sealed FLOAT waveform readback differs: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "samples": int(stored.size),
        "sample_rate": sample_rate,
        "subtype": info.subtype,
        "float32_sha256": hashlib.sha256(
            np.ascontiguousarray(stored, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def read_bound_audio(binding: Mapping[str, Any], label: str) -> np.ndarray:
    path = Path(str(binding.get("path", ""))).resolve()
    validate_hash(path, str(binding.get("sha256", "")), label)
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} is not mono 16 kHz")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.asarray(values, dtype=np.float32)


def validate_source_manifest(
    manifest: Mapping[str, Any],
    panel_rows: Mapping[str, Mapping[str, Any]],
    *,
    prior_ledger: Mapping[str, Any],
    prior_ledger_sha256: str,
    source_prior_ledger_sha256: str,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    expected_operation_order = [
        "map_health_status",
        "pair_cs_sv_by_same_session",
        "filter_raw_mono_minimum_duration",
        "retain_minimum_numeric_eligible_session_per_speaker",
        "exclude_prior_ledger_speakers",
        "bucket_by_health_status_and_gender",
        "rank_by_salted_sha256",
        "allocate_calibration_then_final_by_frozen_gender_quota",
    ]
    expected_manifest_counts = {
        f"{split}:{label}:{gender}": count
        for (split, label), by_gender in GENDER_QUOTAS.items()
        for gender, count in by_gender.items()
    }
    if (
        manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("source_dataset") != "SVD"
        or manifest.get("sv_metadata_sha256") != SVD_SV_METADATA_SHA256
        or manifest.get("cs_metadata_sha256") != SVD_CS_METADATA_SHA256
        or manifest.get("selection_salt") != SVD_SPEAKER_SELECTION_SALT
        or manifest.get("prior_panel_speaker_ledger_sha256")
        != prior_ledger_sha256
        or manifest.get("source_prior_panel_speaker_ledger_sha256")
        != source_prior_ledger_sha256
        or manifest.get("selection_operation_order")
        != expected_operation_order
        or manifest.get("selection_mode") != "metadata_only_result_blind"
        or manifest.get("diagnosis_used_for_selection") is not False
        or manifest.get("exact_scores_opened") is not False
        or manifest.get("candidate_outcomes_opened") is not False
        or manifest.get("mild_severe_labels_created") is not False
        or manifest.get("speaker_count") != 12
        or manifest.get("counts") != expected_manifest_counts
    ):
        raise ValueError("fresh source manifest boundary differs")
    _require_optimizer_zero(manifest, "fresh source manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("fresh source manifest speaker count differs")
    expected_row_fields = {
        "dataset",
        "canonical_speaker_id",
        "speaker_id",
        "session_id",
        "split",
        "label",
        "health_status",
        "gender",
        "diagnosis_record_only",
        "rank_within_label_gender",
        "selection_digest",
        "waveforms",
    }
    panel_speakers: dict[tuple[str, str], str] = {}
    for panel_row in panel_rows.values():
        key = (str(panel_row["speaker_id"]), str(panel_row["split"]))
        label = str(panel_row["label"])
        if key in panel_speakers and panel_speakers[key] != label:
            raise ValueError("fresh split changes a speaker label")
        panel_speakers[key] = label
    excluded = set()
    if (
        prior_ledger.get("schema_version") != PRIOR_PANEL_LEDGER_SCHEMA
        or prior_ledger.get("exact_outcomes_used_for_selection") is not False
    ):
        raise ValueError("prior speaker ledger boundary differs")
    prior_entries = prior_ledger.get("entries")
    if not isinstance(prior_entries, list) or not prior_entries:
        raise ValueError("prior speaker ledger entries are unavailable")
    for entry in prior_entries:
        if not isinstance(entry, dict):
            raise ValueError("prior speaker ledger entry differs")
        dataset = str(entry.get("dataset", "")).strip().upper()
        speaker_id = str(entry.get("speaker_id", "")).strip()
        canonical = str(entry.get("canonical_speaker_id", ""))
        if not dataset or not speaker_id or canonical != f"{dataset}:{speaker_id}":
            raise ValueError("prior speaker ledger identities differ")
        if canonical in excluded:
            raise ValueError("prior speaker ledger identities differ")
        excluded.add(canonical)

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    observed_speakers: set[tuple[str, str]] = set()
    quota_counts: dict[tuple[str, str, str], int] = {}
    ranks: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("fresh source row differs")
        speaker_id = str(row.get("speaker_id", ""))
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        health_status = str(row.get("health_status", ""))
        gender = str(row.get("gender", ""))
        session_id = str(row.get("session_id", ""))
        key_base = (speaker_id, split)
        if key_base in observed_speakers:
            raise ValueError("fresh source speaker is duplicated")
        observed_speakers.add(key_base)
        rank = row.get("rank_within_label_gender")
        if (
            row.get("dataset") != "SVD"
            or row.get("canonical_speaker_id") != canonical_speaker_id(speaker_id)
            or key_base not in panel_speakers
            or panel_speakers[key_base] != label
            or HEALTH_STATUS.get(health_status) != label
            or gender not in {"female", "male"}
            or not isinstance(rank, int)
            or rank <= 0
            or row.get("selection_digest")
            != selection_digest(speaker_id, session_id)
        ):
            raise ValueError("fresh source speaker metadata differs")
        if row["canonical_speaker_id"] in excluded:
            raise ValueError("fresh source overlaps the prior speaker ledger")
        quota_key = (split, label, gender)
        quota_counts[quota_key] = quota_counts.get(quota_key, 0) + 1
        ranks.setdefault((label, gender), []).append(rank)
        waveforms = row.get("waveforms")
        if not isinstance(waveforms, dict) or set(waveforms) != {"cs", "sv"}:
            raise ValueError("fresh source view coverage differs")
        for view, binding in waveforms.items():
            if not isinstance(binding, dict):
                raise ValueError("fresh source waveform binding differs")
            source_path = Path(str(binding.get("path", ""))).resolve()
            validate_hash(
                source_path,
                str(binding.get("sha256", "")),
                f"fresh source {speaker_id}/{view}",
            )
            info = sf.info(source_path)
            if (
                info.channels != 1
                or info.frames != binding.get("source_frames")
                or info.samplerate != binding.get("source_sample_rate")
                or abs(
                    float(binding.get("duration_seconds", -1.0))
                    - info.frames / info.samplerate
                )
                > 1e-12
                or float(binding.get("duration_seconds", -1.0))
                < MINIMUM_SECONDS[view]
            ):
                raise ValueError("fresh source audio header differs")
            indexed[(speaker_id, split, view)] = binding
    expected = {
        (str(row["speaker_id"]), str(row["split"]), str(row["view"]))
        for row in panel_rows.values()
    }
    if set(indexed) != expected:
        raise ValueError("fresh source manifest does not cover the split")
    expected_quota_counts = {
        (split, label, gender): count
        for (split, label), by_gender in GENDER_QUOTAS.items()
        for gender, count in by_gender.items()
    }
    if quota_counts != expected_quota_counts:
        raise ValueError("fresh source gender quotas differ")
    for bucket_ranks in ranks.values():
        if sorted(bucket_ranks) != list(range(1, len(bucket_ranks) + 1)):
            raise ValueError("fresh source salted ranks differ")
    return indexed


def validate_recipe_manifest(
    manifest: Mapping[str, Any],
    panel_rows: Mapping[str, Mapping[str, Any]],
    recipes: list[dict[str, Any]],
    fixed_recipes_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    if (
        manifest.get("schema_version") != RECIPE_MANIFEST_SCHEMA
        or manifest.get("fixed_recipes_sha256") != fixed_recipes_sha256
        or manifest.get("assignment_salt") != GAP_RECIPE_ASSIGNMENT_SALT
        or manifest.get("speaker_split_completed_before_assignment") is not True
        or manifest.get("condition_recipe_semantics")
        != {
            "clean": "no_simulation",
            "rir_only": "rir_only",
            "snr20": "rir_plus_noise_fixed_target_snr_20db",
            "snr10": "rir_plus_noise_fixed_target_snr_10db",
        }
        or manifest.get("row_count") != EXPECTED_TOTAL_ROWS
        or manifest.get("recipe_uid_unique_per_row") is not True
        or manifest.get("recipe_uid_reused_across_splits") is not False
        or manifest.get("exact_scores_opened") is not False
        or manifest.get("candidate_outcomes_opened") is not False
    ):
        raise ValueError("joint recipe manifest boundary differs")
    _require_optimizer_zero(manifest, "joint recipe manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError("joint recipe row count differs")
    indexed = {str(row.get("case_id", "")): row for row in rows}
    if set(indexed) != set(panel_rows):
        raise ValueError("joint recipe cases differ from split")
    recipe_uids = [str(row.get("recipe_uid", "")) for row in rows]
    if "" in recipe_uids or len(set(recipe_uids)) != len(recipe_uids):
        raise ValueError("joint recipe UIDs are not unique")
    used_indices = set()
    for case_id, assignment in indexed.items():
        condition = panel_rows[case_id]["condition"]
        digest = hashlib.sha256(
            f"{GAP_RECIPE_ASSIGNMENT_SALT}:{case_id}".encode("utf-8")
        ).hexdigest()
        expected_simulation = condition != "clean"
        expected_noise = condition in {"snr20", "snr10"}
        expected_snr = (
            20.0
            if condition == "snr20"
            else 10.0 if condition == "snr10" else None
        )
        if (
            assignment.get("split") != panel_rows[case_id]["split"]
            or assignment.get("condition") != condition
            or assignment.get("assignment_digest") != digest
            or assignment.get("simulation_applied") is not expected_simulation
            or assignment.get("rir_applied") is not expected_simulation
            or assignment.get("noise_applied") is not expected_noise
            or assignment.get("target_snr_db") != expected_snr
        ):
            raise ValueError("joint recipe condition differs")
        if condition == "clean":
            if (
                assignment.get("recipe_index") is not None
                or assignment.get("simulation_applied") is not False
                or assignment.get("fixed_recipe_uid") is not None
                or assignment.get("fixed_recipe_sha256") is not None
                or assignment.get("recipe_uid")
                != f"clean_no_simulation_{digest[:24]}"
            ):
                raise ValueError("clean case unexpectedly has a simulation recipe")
            continue
        recipe_index = assignment.get("recipe_index")
        if not isinstance(recipe_index, int) or not 0 <= recipe_index < len(recipes):
            raise ValueError("joint recipe index differs")
        if recipe_index in used_indices:
            raise ValueError("joint degraded recipe was reused")
        used_indices.add(recipe_index)
        recipe = recipes[recipe_index]
        if (
            str(recipe.get("uid")) != assignment.get("fixed_recipe_uid")
            or assignment.get("recipe_uid") != assignment.get("fixed_recipe_uid")
            or canonical_json_sha256(recipe)
            != assignment.get("fixed_recipe_sha256")
        ):
            raise ValueError("joint fixed recipe content differs")
    if len(used_indices) != 72:
        raise ValueError("joint degraded recipe coverage differs")
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for input_name in MATERIALIZATION_INPUT_NAMES:
        option = input_name.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--candidate-e-root", type=Path, required=True)
    parser.add_argument("--candidate-e-commit", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite materialization: {args.output_dir}"
        )
    source = repository_source(args.source_root, args.source_commit)
    candidate_source = candidate_repository(
        args.candidate_e_root,
        args.candidate_e_commit,
    )
    paths = {
        name: getattr(args, name).resolve()
        for name in MATERIALIZATION_INPUT_NAMES
    }
    hashes = {
        name: validate_hash(
            paths[name],
            getattr(args, f"{name}_sha256"),
            name,
        )
        for name in MATERIALIZATION_INPUT_NAMES
    }
    expected_candidate_hashes = {
        "candidate_e_promotion_report": PROMOTION_REPORT_SHA256,
        "candidate_e_promotion_receipt": PROMOTION_RECEIPT_SHA256,
        "candidate_e_reference_source": CANDIDATE_E_REFERENCE_SHA256,
        "candidate_e_runtime_client": EXPECTED_CANDIDATE_E_RUNTIME_CLIENT_SHA256,
        "candidate_e_worker": EXPECTED_CANDIDATE_E_WORKER_SHA256,
        "candidate_e_selector_source": EXPECTED_CANDIDATE_E_SELECTOR_SHA256,
        "candidate_e_runtime_config": EXPECTED_CANDIDATE_E_RUNTIME_CONFIG_SHA256,
    }
    if any(hashes[key] != value for key, value in expected_candidate_hashes.items()):
        raise ValueError("Candidate-E promoted runtime binding differs")
    expected_generation_hashes = {
        "simulation_config": EXPECTED_SIMULATION_CONFIG_SHA256,
        "simulation_source": EXPECTED_SIMULATION_SOURCE_SHA256,
        "generator_config": EXPECTED_GENERATOR_CONFIG_SHA256,
        "generator_checkpoint": EXPECTED_GENERATOR_CHECKPOINT_SHA256,
    }
    if any(
        hashes[key] != value
        for key, value in expected_generation_hashes.items()
    ):
        raise ValueError("frozen simulation or generator binding differs")
    promotion_report = read_json(
        paths["candidate_e_promotion_report"],
        "Candidate-E promotion report",
    )
    promotion_receipt = read_json(
        paths["candidate_e_promotion_receipt"],
        "Candidate-E promotion receipt",
    )
    if (
        promotion_report.get("decision") != PROMOTION_PASS
        or promotion_receipt.get("decision") != PROMOTION_PASS
        or promotion_report.get("scientific_promotion_granted") is not True
        or promotion_receipt.get("scientific_promotion_granted") is not True
    ):
        raise ValueError("Candidate-E promotion evidence is not PASS")

    split = read_json(paths["split_seal"], "fresh split seal")
    panel_rows, _ = _validate_panel_rows(split.get("rows"), "materialization")
    prior_ledger = read_json(
        paths["prior_panel_speaker_ledger"],
        "prior speaker ledger",
    )
    if split.get("prior_panel_speaker_ledger_sha256") != hashes[
        "prior_panel_speaker_ledger"
    ]:
        raise ValueError("fresh split prior-ledger binding differs")
    _validate_split_seal(
        split,
        gate_sha256=str(split.get("joint_gate_contract_sha256", "")),
        target_sha256=str(split.get("target_value_protocol_sha256", "")),
        ledger_sha256=hashes["prior_panel_speaker_ledger"],
        source_sha256=hashes["fresh_speaker_source_manifest"],
    )
    if split.get("joint_recipe_assignment_manifest_sha256") != hashes[
        "joint_recipe_assignment_manifest"
    ]:
        raise ValueError("fresh split recipe-manifest binding differs")
    source_rows = validate_source_manifest(
        read_json(
            paths["fresh_speaker_source_manifest"],
            "fresh source manifest",
        ),
        panel_rows,
        prior_ledger=prior_ledger,
        prior_ledger_sha256=hashes["prior_panel_speaker_ledger"],
        source_prior_ledger_sha256=UPDATED_LEDGER_SHA256,
    )
    recipes = read_fixed_recipes(paths["fixed_recipes"])
    recipe_rows = validate_recipe_manifest(
        read_json(
            paths["joint_recipe_assignment_manifest"],
            "joint recipe manifest",
        ),
        panel_rows,
        recipes,
        hashes["fixed_recipes"],
    )

    six_report = read_json(paths["six_gradient_report"], "six-gradient report")
    six_receipt = read_json(paths["six_gradient_receipt"], "six-gradient receipt")
    source_evidence = six_report.get("source_evidence_sha256")
    if not isinstance(source_evidence, dict):
        raise ValueError("six-gradient source evidence is unavailable")
    weights = _validate_six_gradient(
        six_report,
        six_receipt,
        hashes["six_gradient_report"],
        source_evidence,
    )
    raw_binding = six_report.get("raw_measurement_evidence")
    if (
        not isinstance(raw_binding, dict)
        or raw_binding.get("report_sha256")
        != hashes["six_gradient_raw_report"]
    ):
        raise ValueError("six-gradient decision does not bind the raw report")
    raw_report = read_json(
        paths["six_gradient_raw_report"],
        "six-gradient raw report",
    )
    normalization = raw_report.get("contract", {}).get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("six-gradient normalization is unavailable")
    normalization = {
        "target_mean": _finite_mapping(
            normalization.get("target_mean"),
            ROUTE_C_SIX_ACTIVE_COMPONENTS,
            "six-joint target means",
        ),
        "target_scale": _finite_mapping(
            normalization.get("target_scale"),
            ROUTE_C_SIX_ACTIVE_COMPONENTS,
            "six-joint target scales",
            positive=True,
        ),
    }
    exact_authority = validate_exact_authority(
        exact_python=args.exact_python,
        avqi_code_root=args.avqi_code_root,
        code_manifest=read_json(
            paths["exact_code_tree_manifest"],
            "exact code-tree manifest",
        ),
        code_manifest_sha256=hashes["exact_code_tree_manifest"],
        runtime_manifest=read_json(
            paths["exact_runtime_manifest"],
            "exact runtime manifest",
        ),
    )
    exact_receipt = read_json(
        paths["exact_authority_receipt"],
        "exact authority receipt",
    )
    if (
        exact_receipt.get("schema_version") != EXACT_AUTHORITY_RECEIPT_SCHEMA
        or exact_receipt.get("decision") != EXACT_AUTHORITY_SEAL_DECISION
        or exact_receipt.get("generator_optimizer_steps") != 0
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_avqi_code_tree_manifest.json"
        )
        != hashes["exact_code_tree_manifest"]
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_runtime_manifest.json"
        )
        != hashes["exact_runtime_manifest"]
    ):
        raise ValueError("exact authority receipt binding differs")
    checkpoint_paths = {
        "cpps": paths["cpps_checkpoint"],
        "hnr": paths["hnr_checkpoint"],
        "shimmer_percent": paths["shimmer_checkpoint"],
        "slope": paths["slope_checkpoint"],
        "tilt": paths["tilt_checkpoint"],
    }
    checkpoint_hashes = {
        "cpps": hashes["cpps_checkpoint"],
        "hnr": hashes["hnr_checkpoint"],
        "shimmer_percent": hashes["shimmer_checkpoint"],
        "slope": hashes["slope_checkpoint"],
        "tilt": hashes["tilt_checkpoint"],
    }

    simulation_config = yaml.safe_load(
        paths["simulation_config"].read_text(encoding="utf-8")
    )
    if not isinstance(simulation_config, dict):
        raise ValueError("simulation config is not a mapping")
    simulation_config["stft_cfg"]["sampling_rate"] = SAMPLE_RATE
    if paths["simulation_source"] != (
        args.simulation_root / "simulate_degradation.py"
    ).resolve():
        raise ValueError("simulation source is outside the bound root")
    # This external simulator is available only in the frozen Triton runtime.
    if str(args.simulation_root) not in sys.path:
        sys.path.insert(0, str(args.simulation_root))
    from simulate_degradation import apply_degradation_with_wind

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA materialization requested without a GPU")
    set_model_seed(args.seed)
    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    target_root = waveform_root / "target_clean_pathological"
    degraded_root = waveform_root / "degraded"
    base_root = waveform_root / "s3_500_base"
    topology_root = args.output_dir / "topologies"
    gradient_root = args.output_dir / "joint_gradients"
    for directory in (
        waveform_root,
        target_root,
        degraded_root,
        base_root,
        topology_root,
        gradient_root,
    ):
        directory.mkdir(exist_ok=True)

    generator_config = load_config(paths["generator_config"])
    generator = load_generator(
        generator_config,
        paths["generator_checkpoint"],
        device,
    )
    reader = WdsReader()
    prepared: dict[str, dict[str, Any]] = {}
    target_bindings: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        with torch.inference_mode():
            for index, case_id in enumerate(sorted(panel_rows), start=1):
                row = panel_rows[case_id]
                key = (str(row["speaker_id"]), str(row["split"]), str(row["view"]))
                clean = read_clean(Path(str(source_rows[key]["path"])))
                if key not in target_bindings:
                    target_path = target_root / (
                        f"{row['split']}__SVD_{row['speaker_id']}__"
                        f"{row['view']}__target.wav"
                    )
                    target_bindings[key] = write_audio(target_path, clean[0])
                assignment = recipe_rows[case_id]
                if row["condition"] == "clean":
                    degraded = clean.copy()
                    noise_start = None
                    simulation_seed = None
                else:
                    recipe = recipes[int(assignment["recipe_index"])]
                    simulation_seed = stable_seed(
                        args.seed,
                        "avqi-route-c-six-joint-v5",
                        case_id,
                        assignment["recipe_uid"],
                    )
                    rng = random.Random(simulation_seed)
                    noise_row = recipe_wds_row(recipe, "noise")
                    rir_row = recipe_wds_row(recipe, "rir")
                    noise, noise_start = crop_or_tile(
                        reader.read(noise_row),
                        clean.shape[1],
                        rng,
                    )
                    rir = reader.read(rir_row)
                    selected_degradations = ["reverb"]
                    if row["condition"] in {"snr20", "snr10"}:
                        selected_degradations.append("noise")
                    snr = assignment["target_snr_db"]
                    _, degraded = apply_degradation_with_wind(
                        copy.deepcopy(simulation_config),
                        clean,
                        noise,
                        rir,
                        None,
                        {"snr": 20 if snr is None else int(snr)},
                        selected_degradations,
                        seed=simulation_seed,
                    )
                    degraded = match_length(degraded, clean.shape[1]).astype(
                        np.float32
                    )
                safe = safe_case_id(case_id)
                degraded_binding = write_audio(
                    degraded_root / f"{safe}__degraded.wav",
                    degraded[0],
                )
                enhanced = enhance_waveform(
                    generator,
                    torch.from_numpy(degraded[0].copy()).to(device),
                    generator_config,
                ).detach().cpu().reshape(-1)
                if not bool(torch.isfinite(enhanced).all()) or float(
                    enhanced.abs().max()
                ) >= 1.0:
                    raise ValueError(f"invalid S3_500 output: {case_id}")
                base_binding = write_audio(
                    base_root / f"{safe}__s3_500.wav",
                    enhanced.numpy(),
                )
                prepared[case_id] = {
                    "panel": row,
                    "target": target_bindings[key],
                    "degraded": degraded_binding,
                    "base": base_binding,
                    "simulation_seed": simulation_seed,
                    "noise_start_sample": noise_start,
                    "recipe_uid": assignment["recipe_uid"],
                }
                print(
                    f"six_joint_base={index}/{EXPECTED_TOTAL_ROWS} case={case_id}",
                    flush=True,
                )
    finally:
        reader.close()
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    patient_target_keys = sorted(
        {
            (str(row["speaker_id"]), str(row["split"]), str(row["view"]))
            for row in panel_rows.values()
            if row["label"] == "patient"
        }
    )
    target_items = [
        {
            "id": f"target:{split}:{speaker_id}:{view}",
            "path": target_bindings[(speaker_id, split, view)]["path"],
            "view": view,
        }
        for speaker_id, split, view in patient_target_keys
    ]
    target_exact = run_exact(
        target_items,
        exact_python=args.exact_python.resolve(),
        avqi_code_root=args.avqi_code_root.resolve(),
        expected_runtime=exact_authority,
    )
    target_rows = []
    target_vectors = {}
    for speaker_id, split, view in patient_target_keys:
        key = (speaker_id, split, view)
        item_id = f"target:{split}:{speaker_id}:{view}"
        values = target_exact[item_id]
        target_vectors[key] = values
        target_rows.append(
            {
                "speaker_id": speaker_id,
                "split": split,
                "view": view,
                "target_waveform_path": target_bindings[key]["path"],
                "target_waveform_sha256": target_bindings[key]["sha256"],
                "exact_components": {
                    name: float(values[index])
                    for index, name in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS)
                },
            }
        )
    target_bank = {
        "schema_version": TARGET_BANK_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "split_seal_sha256": hashes["split_seal"],
        "fresh_speaker_source_manifest_sha256": hashes[
            "fresh_speaker_source_manifest"
        ],
        "target_value_protocol_sha256": split[
            "target_value_protocol_sha256"
        ],
        "exact_code_tree_manifest_sha256": hashes[
            "exact_code_tree_manifest"
        ],
        "exact_runtime_manifest_sha256": hashes["exact_runtime_manifest"],
        "target_protocol": (
            "same_speaker_same_session_same_view_clean_pathological_raw_svd"
        ),
        "target_exact_values_opened": True,
        "target_values_sealed_before_candidate_generation": True,
        "candidate_exact_outcomes_opened": False,
        "candidate_waveforms_scored": False,
        "rows": target_rows,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    target_bank_path = args.output_dir / "clean_target_label_bank.json"
    write_json(target_bank_path, target_bank)
    target_bank_sha256 = sha256_file(target_bank_path)

    runtime_module = load_external_module(
        paths["candidate_e_runtime_client"],
        hashes["candidate_e_runtime_client"],
    )
    topology_items = []
    topology_waveforms = []
    patient_case_ids = sorted(
        case_id
        for case_id, row in panel_rows.items()
        if row["label"] == "patient"
    )
    for case_id in patient_case_ids:
        row = panel_rows[case_id]
        topology_items.append(
            {
                "id": f"joint-v5-topology:{case_id}",
                "case_id": case_id,
                "role": "current_s3_500_output_topology",
                "path": prepared[case_id]["base"]["path"],
                "view": row["view"],
                "score_components": False,
                "exact_metric_topology": True,
                "highpass_mode": runtime_module.NUMPY_HIGHPASS_MODE,
            }
        )
        topology_waveforms.append(
            read_bound_audio(prepared[case_id]["base"], f"base {case_id}")
        )
    with runtime_module.ExactShimmerTopologyWorker(
        args.exact_python.resolve(),
        paths["candidate_e_worker"],
        args.avqi_code_root.resolve(),
        exact_authority["avqi_code_tree_sha256"],
    ) as worker:
        warmup, warmup_ms = worker.warmup()
        topology_rows, topology_runtime_ms, staging_rows = (
            worker.refresh_current_waveforms(
                topology_items,
                topology_waveforms,
                highpass_mode=runtime_module.NUMPY_HIGHPASS_MODE,
            )
        )
    topologies = {}
    topology_artifacts = {}
    for case_id, topology in zip(patient_case_ids, topology_rows, strict=True):
        topology_sha = runtime_module.topology_sha256(topology)
        topology_path = topology_root / f"{safe_case_id(case_id)}.json"
        write_json(topology_path, topology)
        topologies[case_id] = (topology, topology_sha)
        topology_artifacts[topology_path.name] = sha256_file(topology_path)

    bundle = load_route_c_candidate_e_six_scorer(
        checkpoint_paths,
        checkpoint_hashes,
    )
    scorer = bundle.scorer.to(device).eval()
    if sum(parameter.numel() for parameter in scorer.parameters()) != 0:
        raise ValueError("six-joint scorer unexpectedly has parameters")
    expected_mean = torch.tensor(
        [
            normalization["target_mean"][name]
            for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
        ],
        dtype=scorer.target_mean.dtype,
    )
    expected_scale = torch.tensor(
        [
            normalization["target_scale"][name]
            for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
        ],
        dtype=scorer.target_scale.dtype,
    )
    if not torch.equal(scorer.target_mean.detach().cpu(), expected_mean):
        raise ValueError("six-joint target means differ from six-gradient report")
    if not torch.equal(scorer.target_scale.detach().cpu(), expected_scale):
        raise ValueError("six-joint target scales differ from six-gradient report")

    gradient_rows = []
    projection_receipts = {}
    for index, case_id in enumerate(sorted(panel_rows), start=1):
        row = panel_rows[case_id]
        base = read_bound_audio(prepared[case_id]["base"], f"base {case_id}")
        if row["optimization_role"] == HEALTHY_ROLE:
            gradient_rows.append(
                {
                    "case_id": case_id,
                    "base_waveform_path": prepared[case_id]["base"]["path"],
                    "base_waveform_sha256": prepared[case_id]["base"]["sha256"],
                    "joint_gradient_path": None,
                    "joint_gradient_sha256": None,
                    "topology_sha256": None,
                }
            )
            continue
        topology, topology_sha = topologies[case_id]
        waveform = torch.from_numpy(base.copy()).to(device).requires_grad_(True)
        target_key = (
            str(row["speaker_id"]),
            str(row["split"]),
            str(row["view"]),
        )
        raw_target = torch.from_numpy(target_vectors[target_key].copy()).to(
            device=device,
            dtype=scorer.target_mean.dtype,
        ).unsqueeze(0)
        prediction = scorer(
            waveform,
            str(row["view"]),
            topology=topology,
            case_id=case_id,
            view=str(row["view"]),
            topology_sha256=topology_sha,
        )
        plan = build_cycle_gain_plan(base, topology)
        proxy = candidate_e_proxy(
            torch.from_numpy(base.copy()).to(dtype=torch.float64),
            torch.as_tensor(
                topology["pulse_positions_samples"],
                dtype=torch.float64,
            ),
            torch.from_numpy(plan["source_indices"]),
            int(topology["metric_constant_prefix_samples"]),
        )
        peak_certificate = validate_candidate_e_base_peak_certificate(
            topology,
            proxy,
        )
        proxy_value = float(proxy.shimmer_db.detach().cpu())
        scorer_value = float(
            scorer.denormalized_prediction(prediction)[0, 3].detach().cpu()
        )
        if not math.isclose(
            proxy_value,
            scorer_value,
            rel_tol=1e-6,
            abs_tol=1e-5,
        ):
            raise ValueError(f"Candidate-E scorer/proxy value differs: {case_id}")
        losses = six_active_bidirectional_gap_losses(
            prediction,
            raw_target,
            scorer.target_mean,
            scorer.target_scale,
        )[0]
        component_gradients = {}
        case_projection = None
        for component_index, component in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS):
            gradient = torch.autograd.grad(
                losses[component_index],
                waveform,
                retain_graph=component_index
                < len(ROUTE_C_SIX_ACTIVE_COMPONENTS) - 1,
                create_graph=False,
            )[0].detach().cpu().to(dtype=torch.float64)
            if component == "shimmer_db":
                gradient, case_projection = (
                    project_cycle_gain_gradient_fixed_order(
                        torch.from_numpy(base.copy()).to(dtype=torch.float64),
                        gradient,
                        plan,
                    )
                )
                if case_projection.get("projected_gradient_valid") is not True:
                    raise ValueError(
                        f"Candidate-E projection failed for {case_id}"
                    )
                case_projection = {
                    **case_projection,
                    "candidate_e_proxy_shimmer_db": proxy_value,
                    "candidate_e_sinc70_peak_upper_bound": (
                        peak_certificate["base_peak_upper_bound"]
                    ),
                    "candidate_e_local_sinc70_peak_upper_bound": (
                        peak_certificate[
                            "base_local_sinc70_peak_upper_bound"
                        ]
                    ),
                    "candidate_e_exact_sinc70_peak": peak_certificate[
                        "base_exact_sinc70_peak"
                    ],
                    "candidate_e_peak_check_mode": peak_certificate[
                        "base_peak_check_mode"
                    ],
                    "candidate_e_peak_scale_abstention_pass": True,
                }
            if not bool(torch.isfinite(gradient).all()) or float(
                torch.linalg.vector_norm(gradient)
            ) <= 0.0:
                raise ValueError(f"invalid {component} gradient for {case_id}")
            component_gradients[component] = gradient
        weighted = {
            component: component_gradients[component] * weights[component]
            for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
        }
        joint = sum(weighted.values()) / sum(weights.values())
        if not bool(torch.isfinite(joint).all()) or float(
            torch.linalg.vector_norm(joint)
        ) <= 0.0:
            raise ValueError(f"invalid joint gradient for {case_id}")
        gradient_path = gradient_root / f"{safe_case_id(case_id)}.npy"
        np.save(
            gradient_path,
            joint.numpy().astype(np.float32, copy=False),
            allow_pickle=False,
        )
        projection_receipts[case_id] = case_projection
        gradient_rows.append(
            {
                "case_id": case_id,
                "base_waveform_path": prepared[case_id]["base"]["path"],
                "base_waveform_sha256": prepared[case_id]["base"]["sha256"],
                "joint_gradient_path": str(gradient_path.resolve()),
                "joint_gradient_sha256": sha256_file(gradient_path),
                "topology_sha256": topology_sha,
            }
        )
        print(
            f"six_joint_gradient={index}/{EXPECTED_TOTAL_ROWS} case={case_id}",
            flush=True,
        )

    gradient_manifest = {
        "schema_version": GRADIENT_MANIFEST_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "split_seal_sha256": hashes["split_seal"],
        "clean_target_label_bank_sha256": target_bank_sha256,
        "six_gradient_report_sha256": hashes["six_gradient_report"],
        "six_gradient_receipt_sha256": hashes["six_gradient_receipt"],
        "six_gradient_raw_report_sha256": hashes["six_gradient_raw_report"],
        "six_gradient_decision": SIX_GRADIENT_PASS_DECISION,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "calibration_inverse_gradient_weights": weights,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization": normalization,
        "gradient_source": GRADIENT_SOURCE,
        "current_output_topology_bound": True,
        "candidate_e_projection": (
            "numpy_float64_fixed_cycle_order_before_six_component_combination"
        ),
        "candidate_e_projection_receipts": projection_receipts,
        "waveform_steps": 1,
        "gradient_normalization": "waveform_rms_normalized",
        "candidate_exact_outcomes_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
        "rows": gradient_rows,
    }
    gradient_manifest_path = args.output_dir / "joint_gradient_manifest.json"
    write_json(gradient_manifest_path, gradient_manifest)

    runtime_binding = {
        "schema_version": RUNTIME_BINDING_SCHEMA,
        "decision": "BOUND_CANDIDATE_E_JOINT_RUNTIME_V1",
        "source": source,
        "candidate_e_source": candidate_source,
        "promotion_decision": PROMOTION_PASS,
        "input_sha256": {
            key: hashes[key]
            for key in (
                "candidate_e_promotion_report",
                "candidate_e_promotion_receipt",
                "candidate_e_reference_source",
                "candidate_e_runtime_client",
                "candidate_e_worker",
                "candidate_e_selector_source",
                "candidate_e_runtime_config",
                "exact_code_tree_manifest",
                "exact_runtime_manifest",
                "exact_authority_receipt",
            )
        },
        "integrated_implementation_sha256": {
            "avqi_route_c_candidate_e.py": sha256_file(
                Path(__file__).resolve().parents[1]
                / "model"
                / "avqi_route_c_candidate_e.py"
            ),
            "avqi_route_c_candidate_e_scorer.py": sha256_file(
                Path(__file__).resolve().parents[1]
                / "model"
                / "avqi_route_c_candidate_e_scorer.py"
            ),
        },
        "exact_authority": exact_authority,
        "topology_worker_startup": worker.startup,
        "topology_warmup": warmup,
        "topology_warmup_ms": warmup_ms,
        "topology_batch_runtime_ms": topology_runtime_ms,
        "topology_case_count": len(topology_rows),
        "topology_artifact_sha256": topology_artifacts,
        "topology_staging": staging_rows,
        "topology_role": "base_current_output",
        "topology_candidate_exact_outcomes_opened": False,
        "candidate_exact_outcomes_used_for_runtime_selection": False,
        "speaker_or_case_identity_used_for_runtime_selection": False,
        "metric_highpass_only": True,
        "emitted_waveform_highpass": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    runtime_binding_path = args.output_dir / "candidate_e_joint_runtime_binding.json"
    write_json(runtime_binding_path, runtime_binding)
    receipt = {
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA,
        "decision": MATERIALIZATION_DECISION,
        "source": source,
        "input_binding": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in MATERIALIZATION_INPUT_NAMES
        },
        "input_sha256": hashes,
        "runtime_binding": {
            "simulation_root": str(args.simulation_root.resolve()),
            "candidate_e_source": candidate_source,
            "exact_python": str(args.exact_python.resolve()),
            "avqi_code_root": str(args.avqi_code_root.resolve()),
            "device": str(device),
            "seed": args.seed,
        },
        "artifact_sha256": {
            target_bank_path.name: target_bank_sha256,
            gradient_manifest_path.name: sha256_file(gradient_manifest_path),
            runtime_binding_path.name: sha256_file(runtime_binding_path),
        },
        "speaker_count": 12,
        "row_count": len(gradient_rows),
        "patient_gradient_count": len(patient_case_ids),
        "target_exact_count": len(target_rows),
        "topology_count": len(topology_rows),
        "candidate_exact_outcomes_opened": False,
        "target_exact_values_opened": True,
        "candidate_waveforms_generated": False,
        "generator_loaded_inference_only": True,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = args.output_dir / "materialization_completion_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
