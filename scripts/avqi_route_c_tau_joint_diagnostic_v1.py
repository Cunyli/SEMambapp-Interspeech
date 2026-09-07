#!/usr/bin/env python3
"""Run the explicitly authorized historically opened TAU diagnostic successor.

Selection and gradient computation reuse frozen research implementations. This
entry point has no fresh-promotion or generator-training path. All stage inputs
and outputs are bound to hashes; scientific failure leaves the joint reserve
unexecuted and never triggers speaker replacement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch
import yaml

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients
from model.avqi_route_c_candidate_e_scorer import load_route_c_candidate_e_six_scorer
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_tau_history_capacity_v1 import (
    FAILURE as HISTORICAL_FRESH_NO_GO,
    canonical_speaker,
    repository_source,
)
from scripts.decide_avqi_route_c_six_gradient_fusion_v1 import evaluate_fusion
from scripts.evaluate_avqi_component_backprop import (
    enhance_waveform,
    load_generator,
    set_model_seed,
)
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    AuditCase,
    SEGMENT_SAMPLES,
    load_fixed_segment,
)
from scripts.evaluate_avqi_route_c_six_component_gradients import (
    CANDIDATE_E_EVIDENCE_KEYS,
    REQUIRED_FIVE_SOURCE_EVIDENCE,
    TopologyAuditInput,
    aggregate_measurements,
    calibration_inverse_gradient_weights,
    extract_case_measurement,
    finalize_case_measurement,
    validate_candidate_e_evidence,
    validate_five_source_evidence,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import run_exact, validate_exact_authority
from scripts.evaluate_avqi_shimmer_fresh_panel import read_fixed_recipes, recipe_wds_row
from scripts.prepare_avqi_component_expanded_data import (
    WdsReader,
    crop_or_tile,
    match_length,
    read_clean,
    stable_seed,
)
from scripts.prepare_avqi_route_c_six_component_topologies_v2 import (
    load_runtime_module,
    waveform_float32_sha256,
)
from utils import load_config


SCHEMA = "avqi-route-c-tau-joint-diagnostic-v1"
CONTRACT_SCHEMA = "avqi-route-c-tau-joint-diagnostic-contract-v1"
SCOPE = "historically_opened_TAU_joint_diagnostic_only"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
SOURCE_DECISION = "SEALED_TAU_DIAGNOSTIC_SOURCES_V1"
MATERIALIZED_DECISION = "SEALED_TAU_DIAGNOSTIC_GRADIENT_TARGETS_AND_BASES_V1"
GRADIENT_PASS = "PASS_TAU_DIAGNOSTIC_SIX_GRADIENT_PREREQUISITES_V1"
GRADIENT_NO_GO = "NO_GO_TAU_DIAGNOSTIC_SIX_GRADIENT_PREREQUISITES_V1"
BOUNDARIES = {
    "historical_fresh_no_go_preserved": True,
    "independent_fresh_validation": False,
    "scientific_promotion_granted": False,
    "svd_used_for_new_testing": False,
    "generator_optimizer_created": False,
    "generator_optimizer_steps": 0,
    "formal_generator_training_submitted": False,
    "authoritative_training_decision": TRAINING_NO_GO,
}
GRADIENT_GATES = {
    "minimum_norm_exclusive": 1e-10,
    "maximum_norm_inclusive": 1e4,
    "maximum_weighted_share": 0.8,
    "minimum_component_to_joint_cosine": 0.0,
    "maximum_calibration_weighted_median_ratio": 1.000001,
}
GRADIENT_SPLITS = ("surrogate_calibration", "surrogate_holdout")
GRADIENT_STRATA = ("female/cs", "female/sv", "male/cs", "male/sv")
JOINT_ALLOCATION = (
    ("calibration", "patient", "female", 2),
    ("calibration", "patient", "male", 1),
    ("calibration", "healthy", "female", 1),
    ("calibration", "healthy", "male", 2),
    ("final", "patient", "female", 1),
    ("final", "patient", "male", 2),
    ("final", "healthy", "female", 2),
    ("final", "healthy", "male", 1),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def verified(binding_value: Mapping[str, Any]) -> Path:
    path = Path(binding_value["path"])
    digest = binding_value["sha256"]
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"missing absolute bound file: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"file SHA-256 differs: {path}")
    return path


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA or contract.get("scope") != SCOPE:
        raise ValueError("TAU diagnostic contract scope differs")
    if contract.get("boundaries") != BOUNDARIES:
        raise ValueError("TAU diagnostic authorization boundary differs")
    if contract.get("gradient_gates") != GRADIENT_GATES:
        raise ValueError("frozen gradient thresholds differ")
    selection = contract["selection"]
    if (
        selection.get("gradient_splits") != list(GRADIENT_SPLITS)
        or selection.get("gradient_strata") != list(GRADIENT_STRATA)
        or selection.get("joint_allocation") != [list(row) for row in JOINT_ALLOCATION]
        or selection.get("minimum_seconds") != {"cs": 3.0, "sv": 1.0}
        or selection.get("salt") != "route-c-tau-joint-diagnostic-v1-predeclared-20260908"
        or selection.get("seed") != 20260908
    ):
        raise ValueError("frozen TAU selection differs")
    for name, expected in {
        "speaker_disjoint_across_all_current_roles": True,
        "exclude_every_historically_opened_speaker": False,
        "historical_overlap_must_be_reported": True,
        "unknown_sex_ineligible": True,
        "diagnosis_or_severity_used": False,
        "exact_values_used": False,
        "target_scorability_used_for_selection": False,
        "failed_speaker_replacement_allowed": False,
    }.items():
        if selection.get(name) is not expected:
            raise ValueError(f"TAU selection boundary differs: {name}")
    materialization = contract["materialization"]
    for name, expected in {
        "sample_rate": 16000, "subtype": "FLOAT", "gradient_segment_samples": SEGMENT_SAMPLES,
        "gradient_recipe_indices": list(range(972, 980)),
        "gradient_snr_db": [20, 10, 20, 10, 20, 10, 20, 10],
        "joint_conditions": ["clean", "rir_only", "snr20", "snr10"],
        "joint_recipe_indices": list(range(900, 972)),
        "generator": "S3_500", "generator_mode": "frozen_inference_only",
        "target_exact_sealed_before_any_candidate": True,
        "healthy_loss_enabled": False, "healthy_waveform_step_enabled": False,
        "final_waveform_highpass": False, "metric_branch_highpass_only": True,
    }.items():
        if materialization.get(name) != expected:
            raise ValueError(f"TAU materialization contract differs: {name}")
    gate = read_json(SOURCE_ROOT / "configs/avqi_route_c_six_joint_gate_contract_v1.json")
    precedent = read_json(SOURCE_ROOT / "configs/avqi_route_c_six_gradient_fusion_contract_v2.json")
    if contract.get("joint_gates") != gate or contract.get("fusion_rule") != precedent["fusion_rule"]:
        raise ValueError("frozen joint or fusion rules differ")


def validate_source_record(row: Mapping[str, Any], roots: Mapping[str, str]) -> None:
    speaker = str(row["speaker_id"])
    if (
        row.get("dataset") != "TAU"
        or row.get("canonical_speaker_id") != canonical_speaker(speaker)
        or row.get("label") not in {"patient", "healthy"}
        or row.get("sex") not in {"female", "male", "unknown"}
        or row.get("same_speaker_cs_sv_verified") is not True
        or row.get("source") not in roots
        or set(row.get("sources", {})) != {"cs", "sv"}
    ):
        raise ValueError("TAU source identity, metadata or pairing differs")
    root = Path(roots[row["source"]]).resolve()
    duration_eligible = True
    for view, audio in row["sources"].items():
        path = verified(audio)
        if path.resolve().parent != root / speaker or path.name != f"{speaker}_{view}.wav":
            raise ValueError("TAU source is outside its same-speaker CS/SV directory")
        info = sf.info(path)
        if (info.channels, info.frames, info.samplerate) != (
            audio["channels"], audio["frames"], audio["sample_rate"]
        ):
            raise ValueError("TAU audio header differs from the audit")
        eligible = info.channels == 1 and info.duration >= {"cs": 3.0, "sv": 1.0}[view]
        if audio.get("mono_duration_eligible") is not eligible:
            raise ValueError("TAU source duration eligibility differs")
        duration_eligible = duration_eligible and eligible
    if row.get("source_metadata_eligible") is not (row["sex"] != "unknown" and duration_eligible):
        raise ValueError("TAU metadata eligibility differs")


def select_sources(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    """Choose only from audited metadata; no metric values enter the ordering."""
    if len({row["canonical_speaker_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate TAU speaker in source universe")
    eligible = [row for row in rows if row["source_metadata_eligible"]]
    used: set[str] = set()
    selected: list[dict[str, Any]] = []

    def allocate(role: str, split: str, label: str, sex: str, view: str, count: int) -> None:
        prefix = f"{salt}:{role}:{split}:{label}:{sex}:{view}:"
        candidates = [
            row for row in eligible
            if row["label"] == label and row["sex"] == sex
            and row["canonical_speaker_id"] not in used
        ]
        candidates.sort(key=lambda row: hashlib.sha256(
            (prefix + row["canonical_speaker_id"]).encode("utf-8")
        ).hexdigest())
        if len(candidates) < count:
            raise ValueError(f"insufficient TAU metadata stratum: {role}/{split}/{label}/{sex}/{view}")
        for row in candidates[:count]:
            used.add(row["canonical_speaker_id"])
            selected.append({
                **copy.deepcopy(row), "role": role, "split": split, "view": view,
                "selection_digest": hashlib.sha256(
                    (prefix + row["canonical_speaker_id"]).encode("utf-8")
                ).hexdigest(),
            })

    for split in GRADIENT_SPLITS:
        for stratum in GRADIENT_STRATA:
            sex, view = stratum.split("/")
            allocate("gradient", split, "patient", sex, view, 1)
    for split, label, sex, count in JOINT_ALLOCATION:
        allocate("joint_reserve", split, label, sex, "cs_sv", count)
    if len(selected) != 20 or len(used) != 20:
        raise ValueError("TAU gradient/joint split coverage differs")
    return selected


def receipt(
    output: Path, stage: str, decision: str, contract_binding: Mapping[str, str],
    source: Mapping[str, str], *, dependencies: list[dict[str, str]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    value = {
        "schema_version": SCHEMA + "-receipt", "scope": SCOPE,
        "stage": stage, "decision": decision, "source": dict(source),
        "contract": dict(contract_binding), "dependencies": dependencies,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "artifacts": artifacts, **BOUNDARIES, **dict(extra or {}),
    }
    write_json(output / "completion_receipt.json", value)
    return value


def load_stage(value: Mapping[str, str], contract_sha: str, stage: str, decision: str) -> tuple[Path, dict[str, Any]]:
    path = verified(value)
    result = read_json(path)
    if (
        result.get("schema_version") != SCHEMA + "-receipt"
        or result.get("scope") != SCOPE or result.get("stage") != stage
        or result.get("decision") != decision
        or result.get("contract", {}).get("sha256") != contract_sha
        or any(result.get(key) != expected for key, expected in BOUNDARIES.items())
    ):
        raise ValueError("TAU stage receipt scope, decision or contract differs")
    for relative, digest in result["artifacts"].items():
        artifact = path.parent / relative
        if not artifact.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("stage artifact escapes output directory")
        verified({"path": str(artifact), "sha256": digest})
    return path.parent, result


def seal_sources(contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path) -> str:
    audit = read_json(paths["tau_history_report"])
    old_receipt = read_json(paths["tau_history_receipt"])
    ledger = read_json(paths["tau_history_ledger"])
    readiness = read_json(paths["readiness_report"])
    if (readiness.get("all_six_scientific_components_ready") is not True
            or readiness.get("generator_optimizer_steps") != 0
            or readiness.get("joint_panel_authorized") is not False):
        raise ValueError("immutable component readiness or training boundary differs")
    if (
        audit.get("decision") != HISTORICAL_FRESH_NO_GO
        or old_receipt.get("decision") != HISTORICAL_FRESH_NO_GO
        or old_receipt["artifact_sha256"][paths["tau_history_report"].name]
        != contract["inputs"]["tau_history_report"]["sha256"]
        or old_receipt["artifact_sha256"][paths["tau_history_ledger"].name]
        != contract["inputs"]["tau_history_ledger"]["sha256"]
    ):
        raise ValueError("historical fresh NO_GO evidence differs")
    source_rows = audit["source_snapshot"]
    if len(source_rows) != 178 or audit["capacity"]["historically_opened_current_speakers"] != 178:
        raise ValueError("authoritative TAU audit coverage differs")
    for row in source_rows:
        validate_source_record(row, contract["source_roots"])
    for source_name, root in contract["source_roots"].items():
        live = {p.name for p in Path(root).iterdir() if p.is_dir()}
        expected = {row["speaker_id"] for row in source_rows if row["source"] == source_name}
        if live != expected:
            raise ValueError("live TAU source universe differs from the bound audit")
    selected = select_sources(source_rows, contract["selection"]["salt"])
    opened = {row["canonical_speaker_id"] for row in ledger["entries"]}
    recipes = read_fixed_recipes(paths["fixed_recipes"])
    gradient_rows = []
    joint_rows = []
    next_joint_recipe = iter(contract["materialization"]["joint_recipe_indices"])
    for index, row in enumerate(selected):
        row["historically_exact_opened"] = row["canonical_speaker_id"] in opened
        if row["role"] == "gradient":
            recipe_index = contract["materialization"]["gradient_recipe_indices"][index]
            recipe = recipes[recipe_index]
            gradient_rows.append({
                **row, "case_id": f"tau-diagnostic-gradient-{index + 1:02d}",
                "sample_id": f"tau-diagnostic-gradient-{index + 1:02d}",
                "condition": "rir_plus_noise", "recipe_index": recipe_index,
                "recipe_uid": recipe["uid"],
                "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                "snr_db": contract["materialization"]["gradient_snr_db"][index],
            })
        else:
            for view in ("cs", "sv"):
                for condition in contract["materialization"]["joint_conditions"]:
                    recipe_index = None if condition == "clean" else next(next_joint_recipe)
                    joint_rows.append({
                        **row, "view": view, "condition": condition,
                        "case_id": f"tau-diagnostic-joint-{index - 7:02d}-{view}-{condition}",
                        "recipe_index": recipe_index,
                        "recipe_uid": None if recipe_index is None else recipes[recipe_index]["uid"],
                        "snr_db": {"snr20": 20, "snr10": 10}.get(condition),
                    })
    if len(gradient_rows) != 8 or len(joint_rows) != 96:
        raise ValueError("TAU sealed row coverage differs")
    manifest = {
        "schema_version": SCHEMA + "-source-manifest", "scope": SCOPE,
        "rows": selected, "historical_overlap_count": sum(row["historically_exact_opened"] for row in selected),
        "within_run_speaker_overlap": 0, "historical_audit": contract["inputs"]["tau_history_report"],
        "metadata_only_selection": True, "metric_values_used_for_selection": False,
        **BOUNDARIES,
    }
    write_json(output / "tau_speaker_source_manifest.json", manifest)
    write_json(output / "tau_diagnostic_split_seal.json", {
        "schema_version": SCHEMA + "-split-seal", "scope": SCOPE,
        "decision": SOURCE_DECISION, "source_manifest": binding(output / "tau_speaker_source_manifest.json"),
        "gradient_rows": gradient_rows, "joint_reserved_rows": joint_rows,
        "selected_counts": dict(Counter(f"{r['role']}/{r['split']}/{r['label']}/{r['sex']}" for r in selected)),
        "source_split_sealed_before_simulation": True,
        "target_scalars_opened": False, "candidate_exact_outcomes_opened": False,
        "joint_panel_executed": False, **BOUNDARIES,
    })
    return SOURCE_DECISION


def write_audio(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audio: {path}")
    audio = np.asarray(values, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("nonfinite or empty TAU waveform")
    sf.write(path, audio, 16000, subtype="FLOAT")
    stored, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or not np.array_equal(stored, audio):
        raise ValueError("TAU FLOAT waveform roundtrip differs")
    return {**binding(path), "samples": int(audio.size), "sample_rate": sample_rate,
            "float32_sha256": waveform_float32_sha256(audio), "subtype": "FLOAT"}


def authority(contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, str]:
    return validate_exact_authority(
        exact_python=Path(contract["exact"]["python"]), avqi_code_root=Path(contract["exact"]["root"]),
        code_manifest=read_json(paths["exact_code_tree_manifest"]),
        code_manifest_sha256=contract["inputs"]["exact_code_tree_manifest"]["sha256"],
        runtime_manifest=read_json(paths["exact_runtime_manifest"]),
    )


def target_component_rows(
    results: Mapping[str, np.ndarray], case_ids: list[str],
) -> dict[str, dict[str, float]]:
    if len(set(case_ids)) != len(case_ids) or set(results) != set(case_ids):
        raise ValueError("TAU target Exact coverage differs")
    output = {}
    for case_id in case_ids:
        values = np.asarray(results[case_id], dtype=np.float64)
        if values.shape != (len(AVQI_COMPONENT_NAMES),) or not np.isfinite(values).all():
            raise ValueError("TAU target Exact vector is invalid")
        output[case_id] = dict(zip(AVQI_COMPONENT_NAMES, values.tolist(), strict=True))
    return output


def materialize(
    contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
    source_dir: Path, device_name: str,
) -> str:
    seal = read_json(source_dir / "tau_diagnostic_split_seal.json")
    rows = seal["gradient_rows"]
    exact_authority = authority(contract, paths)
    for row in rows:
        validate_source_record(row, contract["source_roots"])
    target_root = output / "target_clean_pathological"
    degraded_root = output / "degraded"
    base_root = output / "s3_500_base"
    for directory in (target_root, degraded_root, base_root):
        directory.mkdir()
    prepared = []
    for row in rows:
        clean = read_clean(Path(row["sources"][row["view"]]["path"]))[0]
        prepared.append({**row, "target": write_audio(target_root / f"{row['case_id']}.wav", clean)})
    exact_result = run_exact(
        [{"id": row["case_id"], "path": row["target"]["path"], "view": row["view"]} for row in prepared],
        exact_python=Path(contract["exact"]["python"]), avqi_code_root=Path(contract["exact"]["root"]),
        expected_runtime=exact_authority,
    )
    target_results = target_component_rows(exact_result, [row["case_id"] for row in prepared])
    write_json(output / "target_exact_result.json", {"components_by_case": target_results, "authority": exact_authority})
    for row in prepared:
        row["target_components"] = target_results[row["case_id"]]
    write_json(output / "clean_target_label_bank.json", {
        "schema_version": SCHEMA + "-clean-target-bank", "scope": SCOPE,
        "rows": [{key: row[key] for key in ("case_id", "canonical_speaker_id", "split", "view", "target", "target_components")} for row in prepared],
        "target_values_sealed_before_candidate_generation": True,
        "target_only_exact_scoring": True, "candidate_exact_outcomes_opened": False,
        **BOUNDARIES,
    })

    # This hash-verified simulator is external and exists only on Triton.
    simulation_root = paths["simulation_source"].parent
    sys.path.insert(0, str(simulation_root))
    from simulate_degradation import apply_degradation_with_wind

    simulation_config = yaml.safe_load(paths["simulation_config"].read_text())
    simulation_config["stft_cfg"]["sampling_rate"] = 16000
    recipes = read_fixed_recipes(paths["fixed_recipes"])
    reader = WdsReader()
    try:
        for row in prepared:
            recipe = recipes[row["recipe_index"]]
            if recipe["uid"] != row["recipe_uid"] or recipe["split"] != "test" or recipe["target_sample_rate"] != 16000:
                raise ValueError("TAU frozen recipe differs")
            clean = read_clean(verified(row["target"]))
            simulation_seed = stable_seed(contract["selection"]["seed"], contract["selection"]["salt"], row["case_id"], recipe["uid"])
            noise, start = crop_or_tile(reader.read(recipe_wds_row(recipe, "noise")), clean.shape[1], random.Random(simulation_seed))
            rir = reader.read(recipe_wds_row(recipe, "rir"))
            _, degraded = apply_degradation_with_wind(
                copy.deepcopy(simulation_config), clean, noise, rir, None,
                {"snr": row["snr_db"]}, ["reverb", "noise"], seed=simulation_seed,
            )
            row["degraded"] = write_audio(degraded_root / f"{row['case_id']}.wav", match_length(degraded, clean.shape[1])[0])
            row["simulation_seed"] = simulation_seed
            row["noise_start_sample"] = start
    finally:
        reader.close()
    set_model_seed(contract["selection"]["seed"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TAU materialization requested CUDA without an allocated GPU")
    generator_config = load_config(paths["generator_config"])
    generator = load_generator(generator_config, paths["generator_checkpoint"], device)
    generator.requires_grad_(False)
    with torch.inference_mode():
        for index, row in enumerate(prepared, start=1):
            degraded, _ = sf.read(verified(row["degraded"]), dtype="float32")
            enhanced = enhance_waveform(generator, torch.from_numpy(degraded.copy()).to(device), generator_config).detach().cpu().reshape(-1)
            if not bool(torch.isfinite(enhanced).all()) or float(enhanced.abs().max()) >= 1.0:
                raise ValueError("frozen S3_500 baseline is invalid or clipped")
            if enhanced.numel() != degraded.size:
                raise ValueError("frozen S3_500 baseline length differs")
            row["base"] = write_audio(base_root / f"{row['case_id']}.wav", enhanced.numpy())
            print(f"tau_gradient_base={index}/{len(prepared)}", flush=True)
    write_json(output / "tau_gradient_materialized_manifest.json", {
        "schema_version": SCHEMA + "-materialized-gradient-manifest", "scope": SCOPE,
        "rows": prepared, "source_split_seal": binding(source_dir / "tau_diagnostic_split_seal.json"),
        "clean_target_bank": binding(output / "clean_target_label_bank.json"),
        "base_or_candidate_exact_outcomes_opened": False, "joint_panel_executed": False,
        "generator_mode": "frozen_inference_only", "final_waveform_highpass": False, **BOUNDARIES,
    })
    return MATERIALIZED_DECISION


def gradient_cases(rows: list[dict[str, Any]]) -> list[AuditCase]:
    return [AuditCase(
        split=row["split"], speaker_id=row["canonical_speaker_id"], sample_id=row["sample_id"],
        sample_group="patient_" + row["sex"], view=row["view"], condition=row["condition"],
        waveform_path=verified(row["base"]), waveform_sha256=row["base"]["sha256"],
        clean_target=torch.tensor([row["target_components"][name] for name in AVQI_COMPONENT_NAMES], dtype=torch.float32),
    ) for row in rows]


def gradients(
    contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
    materialized_dir: Path, device_name: str,
) -> str:
    evidence = contract["inputs"]
    for names, validator in (
        (REQUIRED_FIVE_SOURCE_EVIDENCE, validate_five_source_evidence),
        (CANDIDATE_E_EVIDENCE_KEYS, validate_candidate_e_evidence),
    ):
        validator([[name, evidence[name]["path"], evidence[name]["sha256"]] for name in names])
    exact_authority = authority(contract, paths)
    candidate_source = repository_source(Path(contract["candidate_e_source"]["root"]), contract["candidate_e_source"]["commit"])
    manifest = read_json(materialized_dir / "tau_gradient_materialized_manifest.json")
    rows = manifest["rows"]
    cases = gradient_cases(rows)
    if len(cases) != 8 or [case.split for case in cases] != [GRADIENT_SPLITS[0]] * 4 + [GRADIENT_SPLITS[1]] * 4:
        raise ValueError("TAU gradient split order differs")
    runtime = load_runtime_module(paths["candidate_e_runtime_client"])
    waveforms = [load_fixed_segment(case).cpu().numpy() for case in cases]
    items = [{
        "id": "topology:" + row["case_id"], "case_id": row["case_id"],
        "role": "current_output_topology", "path": str(case.waveform_path),
        "view": case.view, "score_components": False, "exact_metric_topology": True,
        "highpass_mode": runtime.NUMPY_HIGHPASS_MODE,
    } for row, case in zip(rows, cases, strict=True)]
    with runtime.ExactShimmerTopologyWorker(
        Path(contract["exact"]["python"]).resolve(), paths["candidate_e_worker"],
        Path(contract["exact"]["root"]), exact_authority["avqi_code_tree_sha256"],
    ) as worker:
        warmup, warmup_ms = worker.warmup()
        topologies, runtime_ms, staging = worker.refresh_current_waveforms(items, waveforms, highpass_mode=runtime.NUMPY_HIGHPASS_MODE)
    if len(topologies) != len(rows):
        raise ValueError("TAU topology coverage differs")
    topology_inputs = []
    topology_rows = []
    for row, waveform, topology in zip(rows, waveforms, topologies, strict=True):
        waveform_sha = waveform_float32_sha256(waveform)
        if topology["source_waveform_float32_sha256"] != waveform_sha:
            raise ValueError("TAU topology current waveform hash differs")
        topology_sha = runtime.topology_sha256(topology)
        topology_inputs.append(TopologyAuditInput(row["case_id"], topology, topology_sha, waveform_sha))
        topology_rows.append({"case_id": row["case_id"], "topology_sha256": topology_sha, "topology": topology})
    write_json(output / "candidate_e_topology_manifest.json", {
        "schema_version": SCHEMA + "-topology-manifest", "scope": SCOPE,
        "rows": topology_rows, "runtime_ms": runtime_ms, "staging": staging,
        "warmup": warmup, "warmup_ms": warmup_ms,
        "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    })
    checkpoint_names = ("cpps", "hnr", "shimmer_percent", "slope", "tilt")
    bundle = load_route_c_candidate_e_six_scorer(
        {name: paths[name + "_checkpoint"] for name in checkpoint_names},
        {name: evidence[name + "_checkpoint"]["sha256"] for name in checkpoint_names},
    )
    device = torch.device(device_name)
    scorer = bundle.scorer.to(device).eval()
    if any(True for _ in scorer.parameters()):
        raise ValueError("TAU six-component scorer unexpectedly has trainable parameters")
    for key in ("target_mean", "target_scale"):
        expected = torch.tensor([contract["normalization"][key][name] for name in AVQI_COMPONENT_NAMES], dtype=torch.float32)
        if not torch.equal(getattr(scorer, key).detach().cpu(), expected):
            raise ValueError("TAU normalization differs from frozen source checkpoints")
    extracted = []
    weights = None
    medians = None
    for index, (case, topology) in enumerate(zip(cases, topology_inputs, strict=True)):
        if index == 4:
            medians, weights = calibration_inverse_gradient_weights(extracted)
            write_json(output / "calibration_frozen_weights.json", {
                "median_component_gradient_norms": medians, "frozen_inverse_gradient_weights": weights,
                "fit_case_ids": [record["case_id"] for record in extracted],
                "holdout_gradients_measured": False, **BOUNDARIES,
            })
        print(f"tau_six_gradient_case={index + 1}/8 split={case.split} view={case.view}", flush=True)
        extracted.append(extract_case_measurement(scorer, case, topology, device))
    if weights is None or medians is None:
        raise ValueError("TAU calibration weights were not frozen")
    gradient_root = output / "gradient_tensors"
    gradient_root.mkdir()
    tensor_fusions = {}
    for record in extracted:
        joint, fusion = fuse_tensor_gradients(AVQI_COMPONENT_NAMES, record["_gradients"], weights)
        tensor_fusions[record["case_id"]] = fusion
        tensor_path = gradient_root / (record["case_id"] + ".npz")
        np.savez_compressed(tensor_path, **{key: value.numpy() for key, value in record["_gradients"].items()}, joint=joint.numpy())
    finalized = [finalize_case_measurement(record, weights) for record in extracted]
    calibration = [row for row in finalized if row["split"] == GRADIENT_SPLITS[0]]
    holdout = [row for row in finalized if row["split"] == GRADIENT_SPLITS[1]]
    raw = {
        "schema_version": SCHEMA + "-six-gradient-measurement", "scope": SCOPE,
        "calibration": {**aggregate_measurements(calibration), "median_component_gradient_norms": medians,
                        "frozen_inverse_gradient_weights": weights,
                        "weighted_median_gradient_norms": {name: medians[name] * weights[name] for name in weights},
                        "weights_selected_on_holdout": False},
        "holdout": aggregate_measurements(holdout), "case_results": finalized,
        "normalization": contract["normalization"], "candidate_source": candidate_source,
        "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    }
    write_json(output / "six_gradient_raw_measurement.json", raw)
    gates, metrics, fusion_rows = evaluate_fusion(finalized, raw)
    gates["all_candidate_e_peak_paths_pcm16_hash_bound"] = all(
        row["components"]["shimmer_db"]["candidate_e_projection"]["candidate_e_peak_handling_pass"]
        and row["components"]["shimmer_db"]["candidate_e_projection"]["candidate_e_exact_highpass_pcm16_sha256"]
        == row["topology"]["highpass_pcm16_sha256"] for row in finalized
    )
    gates["tensor_and_gram_fusion_agree"] = all(
        math.isclose(row["fusion"]["joint_gradient_norm"], tensor_fusions[row["case_id"]]["joint_gradient_norm"], rel_tol=1e-9, abs_tol=1e-12)
        and all(math.isclose(row["fusion"]["component_to_joint_cosines"][name], tensor_fusions[row["case_id"]]["component_to_joint_cosines"][name], rel_tol=1e-9, abs_tol=1e-12) for name in AVQI_COMPONENT_NAMES)
        for row in fusion_rows
    )
    passed = all(gates.values())
    decision = GRADIENT_PASS if passed else GRADIENT_NO_GO
    write_json(output / "six_gradient_fusion_report.json", {
        "schema_version": SCHEMA + "-six-gradient-decision", "scope": SCOPE,
        "decision": decision, "gates": gates, "metrics": metrics,
        "case_results": fusion_rows, "tensor_fusions": tensor_fusions,
        "raw_report": binding(output / "six_gradient_raw_measurement.json"),
        "joint_diagnostic_preparation_eligible": passed,
        "joint_panel_authorized": False, "joint_panel_executed": False,
        "candidate_exact_outcomes_opened": False, "reserved_joint_speakers_replaced": False,
        **BOUNDARIES,
    })
    print(json.dumps({"decision": decision, "gates": gates, "metrics": metrics}, sort_keys=True), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seal", "materialize", "gradients"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dependency-receipt", type=Path)
    parser.add_argument("--dependency-receipt-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run TAU diagnostics in a Slurm compute allocation")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite TAU diagnostic output")
    contract_binding = {"path": str(args.contract.resolve()), "sha256": args.contract_sha256}
    contract = read_json(verified(contract_binding))
    validate_contract(contract)
    source = repository_source(SOURCE_ROOT, args.source_commit)
    paths = {name: verified(value) for name, value in contract["inputs"].items()}
    dependencies = []
    dependency_dir = None
    if args.stage != "seal":
        if args.dependency_receipt is None or args.dependency_receipt_sha256 is None:
            raise ValueError("TAU downstream stage requires a hash-bound dependency receipt")
        dependency = {"path": str(args.dependency_receipt.resolve()), "sha256": args.dependency_receipt_sha256}
        prior_stage, prior_decision = {"materialize": ("seal", SOURCE_DECISION), "gradients": ("materialize", MATERIALIZED_DECISION)}[args.stage]
        dependency_dir, _ = load_stage(dependency, args.contract_sha256, prior_stage, prior_decision)
        dependencies.append(dependency)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "frozen_contract.json", contract)
    if args.stage == "seal":
        decision = seal_sources(contract, paths, args.output_dir)
    elif args.stage == "materialize":
        decision = materialize(contract, paths, args.output_dir, dependency_dir, args.device)
    else:
        decision = gradients(contract, paths, args.output_dir, dependency_dir, args.device)
    for value in contract["inputs"].values():
        verified(value)
    repository_source(SOURCE_ROOT, args.source_commit)
    receipt(args.output_dir, args.stage, decision, contract_binding, source, dependencies=dependencies,
            extra={"joint_panel_executed": False, "candidate_exact_outcomes_opened": False})
    print(json.dumps({"stage": args.stage, "decision": decision, "receipt": binding(args.output_dir / "completion_receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
