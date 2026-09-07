#!/usr/bin/env python3
"""Sealed full-waveform joint diagnostics on the authorized historical TAU cohort.

The frozen selection, normalization, fusion and joint gates are inherited from
the preceding diagnostic seal. Exact candidate scoring occurs only after the
complete waveform grid is sealed. No stage authorizes fresh promotion or
generator training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
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
from model.avqi_route_c_candidate_e_scorer import load_route_c_candidate_e_six_scorer
from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients
from scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v4 import UNBOUND_JOINT_INPUTS
from scripts.audit_avqi_route_c_tau_history_capacity_v1 import repository_source
from scripts.avqi_route_c_tau_joint_diagnostic_v1 import (
    BOUNDARIES, GRADIENT_PASS, MATERIALIZED_DECISION, SCHEMA, SCOPE,
    SOURCE_DECISION, SOURCE_ROOT, authority, baseline_length_certificate,
    binding, load_stage, read_json, receipt, target_component_rows,
    validate_contract, validate_source_record, verified, write_audio, write_json,
)
from scripts.evaluate_avqi_component_backprop import enhance_waveform, load_generator, set_model_seed
from scripts.evaluate_avqi_route_c_multicomponent_gradients import AuditCase
from scripts.evaluate_avqi_route_c_six_component_gradients import TopologyAuditInput, extract_waveform_measurement
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    _verified_audio, choose_alpha, exact_items_for_stage, run_exact, stage_rows,
)
from scripts.evaluate_avqi_shimmer_fresh_panel import read_fixed_recipes, recipe_wds_row
from scripts.prepare_avqi_component_expanded_data import WdsReader, crop_or_tile, match_length, read_clean, stable_seed
from scripts.prepare_avqi_route_c_six_component_topologies_v2 import load_runtime_module, waveform_float32_sha256
from scripts.prepare_avqi_route_c_six_joint_waveforms import _write_pcm24, candidate_from_gradient
from utils import load_config


PREPARED = "SEALED_TAU_DIAGNOSTIC_JOINT_WAVEFORM_GRID_V1"
PREPARATION_NO_GO = "NO_GO_TAU_DIAGNOSTIC_FULL_WAVEFORM_GRADIENT_PREREQUISITES_V1"
CALIBRATED = "PASS_TAU_DIAGNOSTIC_JOINT_CALIBRATION_V1"
CALIBRATION_NO_GO = "NO_GO_TAU_DIAGNOSTIC_JOINT_CALIBRATION_V1"
FINAL_PASS = "PASS_TAU_DIAGNOSTIC_JOINT_PANEL_V1"
FINAL_NO_GO = "NO_GO_TAU_DIAGNOSTIC_JOINT_PANEL_V1"
ROW_KEYS = ("case_id", "canonical_speaker_id", "speaker_id", "split", "label", "sex", "view", "condition", "recipe_index", "recipe_uid")


def validate_reserved_rows(rows: list[Mapping[str, Any]], gradient_rows: list[Mapping[str, Any]]) -> None:
    if len(rows) != 96 or len({r["case_id"] for r in rows}) != 96:
        raise ValueError("joint reserve must contain 96 unique cases")
    speakers = {r["canonical_speaker_id"] for r in rows}
    if len(speakers) != 12 or speakers & {r["canonical_speaker_id"] for r in gradient_rows}:
        raise ValueError("joint speakers overlap the gradient cohort or coverage differs")
    for speaker in speakers:
        group = [r for r in rows if r["canonical_speaker_id"] == speaker]
        if len({(r["split"], r["label"], r["sex"]) for r in group}) != 1:
            raise ValueError("speaker crosses joint roles")
        expected = {(view, condition) for view in ("cs", "sv") for condition in ("clean", "rir_only", "snr20", "snr10")}
        if len(group) != 8 or {(r["view"], r["condition"]) for r in group} != expected:
            raise ValueError("joint per-speaker view/condition coverage differs")
        if any(r.get("dataset") != "TAU" or r.get("historically_exact_opened") is not True for r in group):
            raise ValueError("joint source must disclose historical TAU overlap")
    if Counter((r["split"], r["label"]) for r in rows) != Counter({
        ("calibration", "patient"): 24, ("calibration", "healthy"): 24,
        ("final", "patient"): 24, ("final", "healthy"): 24,
    }):
        raise ValueError("joint split and label coverage differs")


def load_joint_inputs(dependency: Mapping[str, str], contract_sha: str) -> tuple[Path, Path]:
    gradient_dir, gradient_receipt = load_stage(dependency, contract_sha, "gradients", GRADIENT_PASS)
    materialized_dir, materialized_receipt = load_stage(
        gradient_receipt["dependencies"][0], contract_sha, "materialize", MATERIALIZED_DECISION
    )
    source_dir, _ = load_stage(materialized_receipt["dependencies"][0], contract_sha, "seal", SOURCE_DECISION)
    materialized = read_json(materialized_dir / "tau_gradient_materialized_manifest.json")
    if materialized["source_split_seal"] != binding(source_dir / "tau_diagnostic_split_seal.json"):
        raise ValueError("gradient and joint source seals differ")
    gradient_report = read_json(gradient_dir / "six_gradient_fusion_report.json")
    if gradient_report["decision"] != GRADIENT_PASS or not all(gradient_report["gates"].values()):
        raise ValueError("joint preparation requires the real eight-case gradient PASS")
    return source_dir, gradient_dir


def materialize_joint(contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
                      reserved: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    exact_authority = authority(contract, paths)
    roots = {name: output / name for name in ("clean_reference", "degraded", "s3_500_base")}
    for root in roots.values():
        root.mkdir()
    references = {}
    target_items = []
    for row in reserved:
        validate_source_record(row, contract["source_roots"])
        key = (row["speaker_id"], row["view"])
        if key in references:
            continue
        clean = read_clean(verified(row["sources"][row["view"]]))[0]
        reference = write_audio(roots["clean_reference"] / f"{row['case_id']}.wav", clean)
        references[key] = reference
        if row["label"] == "patient":
            target_items.append({"id": f"target:{row['split']}:{key[0]}:{key[1]}", "path": reference["path"], "view": row["view"]})
    if len(target_items) != 12 or len(references) != 24:
        raise ValueError("joint clean reference coverage differs")
    exact = run_exact(target_items, exact_python=Path(contract["exact"]["python"]),
                      avqi_code_root=Path(contract["exact"]["root"]), expected_runtime=exact_authority)
    values = target_component_rows(exact, [item["id"] for item in target_items])
    targets = {}
    for row in reserved:
        if row["label"] == "patient":
            key = (row["speaker_id"], row["view"])
            targets[key] = {
                **references[key], "speaker_id": key[0], "view": key[1], "split": row["split"],
                "exact_components": values[f"target:{row['split']}:{key[0]}:{key[1]}"],
            }
    write_json(output / "clean_target_label_bank.json", {
        "schema_version": SCHEMA + "-joint-clean-target-bank", "scope": SCOPE,
        "rows": list(targets.values()), "authority": exact_authority,
        "target_values_sealed_before_candidate_generation": True,
        "target_only_exact_scoring": True, "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    })

    # The hash-verified simulation implementation is installed only on Triton.
    sys.path.insert(0, str(paths["simulation_source"].parent))
    from simulate_degradation import apply_degradation_with_wind

    simulation_config = yaml.safe_load(paths["simulation_config"].read_text())
    simulation_config["stft_cfg"]["sampling_rate"] = 16000
    recipes = read_fixed_recipes(paths["fixed_recipes"])
    rows = []
    reader = WdsReader()
    try:
        for original in reserved:
            row = copy.deepcopy(original)
            key = (row["speaker_id"], row["view"])
            row["clean_reference"] = references[key]
            row["target"] = targets.get(key)
            clean = read_clean(verified(references[key]))
            if row["condition"] == "clean":
                row["degraded"] = references[key]
            else:
                recipe = recipes[row["recipe_index"]]
                if recipe["uid"] != row["recipe_uid"] or recipe["split"] != "test" or recipe["target_sample_rate"] != 16000:
                    raise ValueError("joint sealed degradation recipe differs")
                seed = stable_seed(contract["selection"]["seed"], contract["selection"]["salt"], row["case_id"], recipe["uid"])
                noise, start = crop_or_tile(reader.read(recipe_wds_row(recipe, "noise")), clean.shape[1], random.Random(seed))
                rir = reader.read(recipe_wds_row(recipe, "rir"))
                stages = ["reverb"] if row["condition"] == "rir_only" else ["reverb", "noise"]
                _, degraded = apply_degradation_with_wind(
                    copy.deepcopy(simulation_config), clean, noise, rir, None,
                    {"snr": row["snr_db"] if row["snr_db"] is not None else 20}, stages, seed=seed,
                )
                row["degraded"] = write_audio(roots["degraded"] / f"{row['case_id']}.wav", match_length(degraded, clean.shape[1])[0])
                row["simulation_seed"] = seed
                row["noise_start_sample"] = start
            rows.append(row)
    finally:
        reader.close()

    set_model_seed(contract["selection"]["seed"])
    config = load_config(paths["generator_config"])
    generator = load_generator(config, paths["generator_checkpoint"], device)
    generator.requires_grad_(False)
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            degraded = _verified_audio(row["degraded"], row["case_id"])
            enhanced = enhance_waveform(generator, degraded.to(device), config).detach().cpu().reshape(-1)
            if not bool(torch.isfinite(enhanced).all()) or float(enhanced.abs().max()) >= 1.0:
                raise ValueError("joint frozen baseline is nonfinite or clipped")
            row["baseline_length_certificate"] = baseline_length_certificate(degraded.numel(), enhanced.numel(), int(config["stft_cfg"]["hop_size"]))
            row["base"] = write_audio(roots["s3_500_base"] / f"{row['case_id']}.wav", enhanced.numpy())
            print(f"tau_joint_base={index}/96", flush=True)
    write_json(output / "joint_materialized_manifest.json", {
        "schema_version": SCHEMA + "-joint-materialized-manifest", "scope": SCOPE, "rows": rows,
        "target_bank": binding(output / "clean_target_label_bank.json"),
        "base_or_candidate_exact_outcomes_opened": False, **BOUNDARIES,
    })
    return rows


def full_gradient_gates(record: Mapping[str, Any], fusion: Mapping[str, Any]) -> dict[str, bool]:
    norms = [record["components"][name]["gradient_norm"] for name in AVQI_COMPONENT_NAMES]
    projection = record["components"]["shimmer_db"]["candidate_e_projection"]
    return {
        "all_component_norms_bounded": all(1e-10 < value <= 1e4 for value in norms),
        "joint_norm_bounded": 1e-10 < fusion["joint_gradient_norm"] <= 1e4,
        "post_cap_share_le_0_80": fusion["post_cap_maximum_share"] <= 0.8,
        "all_component_to_joint_cosines_nonnegative": fusion["fusion_authorized"] is True,
        "no_component_amplified": fusion["no_component_amplified"] is True,
        "only_unique_dominant_component_attenuated": fusion["only_dominant_component_attenuated"] is True,
        "candidate_e_peak_pcm16_bound": (
            projection["candidate_e_peak_handling_pass"] is True
            and bool(record["topology"]["highpass_pcm16_sha256"])
            and projection["candidate_e_exact_highpass_pcm16_sha256"] == record["topology"]["highpass_pcm16_sha256"]
        ),
    }


def measure_full_gradients(contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
                           rows: list[dict[str, Any]], gradient_dir: Path,
                           device: torch.device) -> dict[str, Any]:
    candidate_source = repository_source(Path(contract["candidate_e_source"]["root"]), contract["candidate_e_source"]["commit"])
    patient_rows = [row for row in rows if row["label"] == "patient"]
    waveforms = [_verified_audio(row["base"], row["case_id"]).numpy() for row in patient_rows]
    runtime = load_runtime_module(paths["candidate_e_runtime_client"])
    exact_authority = authority(contract, paths)
    items = [{
        "id": "topology:" + row["case_id"], "case_id": row["case_id"], "role": "current_output_topology",
        "path": row["base"]["path"], "view": row["view"], "score_components": False,
        "exact_metric_topology": True, "highpass_mode": runtime.NUMPY_HIGHPASS_MODE,
    } for row in patient_rows]
    with runtime.ExactShimmerTopologyWorker(
        Path(contract["exact"]["python"]).resolve(), paths["candidate_e_worker"],
        Path(contract["exact"]["root"]), exact_authority["avqi_code_tree_sha256"],
    ) as worker:
        warmup, warmup_ms = worker.warmup()
        topologies, runtime_ms, staging = worker.refresh_current_waveforms(items, waveforms, highpass_mode=runtime.NUMPY_HIGHPASS_MODE)
    if len(topologies) != 48:
        raise ValueError("joint patient topology coverage differs")
    topology_rows = []
    for row, waveform, topology in zip(patient_rows, waveforms, topologies, strict=True):
        digest = waveform_float32_sha256(waveform)
        if topology["source_waveform_float32_sha256"] != digest:
            raise ValueError("joint topology is not bound to the full baseline")
        topology_rows.append({
            "case_id": row["case_id"], "samples": int(waveform.size),
            "source_waveform_float32_sha256": digest, "topology": topology,
            "topology_sha256": runtime.topology_sha256(topology),
        })
    write_json(output / "joint_topology_manifest.json", {
        "schema_version": SCHEMA + "-joint-topology", "scope": SCOPE, "rows": topology_rows,
        "runtime_ms": runtime_ms, "staging": staging, "warmup": warmup, "warmup_ms": warmup_ms,
        "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    })
    checkpoint_names = ("cpps", "hnr", "shimmer_percent", "slope", "tilt")
    scorer = load_route_c_candidate_e_six_scorer(
        {name: paths[name + "_checkpoint"] for name in checkpoint_names},
        {name: contract["inputs"][name + "_checkpoint"]["sha256"] for name in checkpoint_names},
    ).scorer.to(device).eval()
    if any(True for _ in scorer.parameters()):
        raise ValueError("joint scorer unexpectedly has trainable parameters")
    for key in ("target_mean", "target_scale"):
        expected = torch.tensor([contract["normalization"][key][name] for name in AVQI_COMPONENT_NAMES], dtype=torch.float32)
        if not torch.equal(getattr(scorer, key).detach().cpu(), expected):
            raise ValueError("joint normalization drifted")
    weights_binding = binding(gradient_dir / "calibration_frozen_weights.json")
    weights = read_json(verified(weights_binding))["frozen_inverse_gradient_weights"]
    tensor_root = output / "joint_gradient_tensors"
    tensor_root.mkdir()
    results = []
    for index, (row, waveform, topology_row) in enumerate(zip(patient_rows, waveforms, topology_rows, strict=True), 1):
        case = AuditCase(
            split=row["split"], speaker_id=row["canonical_speaker_id"], sample_id=row["case_id"],
            sample_group="patient_" + row["sex"], view=row["view"], condition=row["condition"],
            waveform_path=verified(row["base"]), waveform_sha256=row["base"]["sha256"],
            clean_target=torch.tensor([row["target"]["exact_components"][name] for name in AVQI_COMPONENT_NAMES], dtype=torch.float32),
        )
        topology_input = TopologyAuditInput(row["case_id"], topology_row["topology"],
                                           topology_row["topology_sha256"], topology_row["source_waveform_float32_sha256"])
        record = extract_waveform_measurement(scorer, case, topology_input, torch.from_numpy(waveform.copy()), device)
        gradients = record.pop("_gradients")
        joint, fusion = fuse_tensor_gradients(AVQI_COMPONENT_NAMES, gradients, weights)
        gates = full_gradient_gates(record, fusion)
        tensor_path = tensor_root / f"{row['case_id']}.npz"
        np.savez_compressed(tensor_path, **{name: value.numpy() for name, value in gradients.items()}, joint=joint.numpy())
        row["joint_gradient"] = {**binding(tensor_path), "samples": joint.numel(), "array_name": "joint"}
        row["gradient_gates"] = gates
        results.append({"case_id": row["case_id"], "gates": gates, "measurement": record, "fusion": fusion})
        print(f"tau_joint_full_gradient={index}/48 passed={all(gates.values())}", flush=True)
    passed = all(all(result["gates"].values()) for result in results)
    report = {
        "schema_version": SCHEMA + "-full-waveform-gradient-report", "scope": SCOPE,
        "full_waveform_gradient_prerequisites_pass": passed, "rows": results,
        "expected_patient_rows": 48, "measured_patient_rows": len(results),
        "healthy_gradient_rows": 0, "weights_refit_on_joint_rows": False, "frozen_weights": weights_binding,
        "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    }
    write_json(output / "full_waveform_gradient_report.json", report)
    write_json(output / "joint_gradient_manifest.json", {
        "schema_version": SCHEMA + "-joint-gradient-manifest", "scope": SCOPE,
        "rows": rows, "frozen_weights": weights_binding, "normalization": contract["normalization"],
        "report": binding(output / "full_waveform_gradient_report.json"),
        "full_waveform_gradient_prerequisites_pass": passed, **BOUNDARIES,
    })
    write_json(output / "candidate_e_joint_runtime_binding.json", {
        "schema_version": SCHEMA + "-joint-runtime-binding", "scope": SCOPE,
        "candidate_source": candidate_source, "exact_authority": exact_authority,
        "runtime_inputs": {name: value for name, value in contract["inputs"].items() if name.startswith("candidate_e_")},
        "topology_manifest": binding(output / "joint_topology_manifest.json"),
        "full_waveform_gradient_report": binding(output / "full_waveform_gradient_report.json"),
        "base_topology_only": True, "exact_candidate_outcomes_available_to_selector": False, **BOUNDARIES,
    })
    return report


def validate_execution_package(package: Mapping[str, Any], contract_sha: str) -> None:
    if (package.get("scope") != SCOPE or package.get("contract", {}).get("sha256") != contract_sha
            or any(package.get(key) != value for key, value in BOUNDARIES.items())):
        raise ValueError("joint diagnostic execution scope differs")
    if set(package["inputs"]) != set(UNBOUND_JOINT_INPUTS):
        raise ValueError("joint execution package must bind all seven authoritative input names")
    if package.get("legacy_fresh_input_names_are_diagnostic_aliases") is not True:
        raise ValueError("joint package must disclose historically opened source semantics")
    for value in package["inputs"].values():
        verified(value)
    inputs = package["inputs"]
    source_dir, gradient_dir = load_joint_inputs(inputs["six_gradient_receipt"], contract_sha)
    expected = {
        "six_gradient_report": binding(gradient_dir / "six_gradient_fusion_report.json"),
        "fresh_panel_split_seal": binding(source_dir / "tau_diagnostic_split_seal.json"),
        "fresh_speaker_source_manifest": binding(source_dir / "tau_speaker_source_manifest.json"),
    }
    if any(inputs[name] != value for name, value in expected.items()):
        raise ValueError("joint package does not descend from the successful gradient seal")
    report = read_json(verified(package["full_waveform_gradient_report"]))
    manifest = read_json(verified(inputs["joint_gradient_manifest"]))
    runtime = read_json(verified(inputs["candidate_e_joint_runtime_binding"]))
    target_bank = read_json(verified(inputs["clean_target_label_bank"]))
    for document in (report, manifest, runtime, target_bank):
        if document.get("scope") != SCOPE or any(document.get(key) != value for key, value in BOUNDARIES.items()):
            raise ValueError("joint input document changes diagnostic scope")
    if (manifest["report"] != package["full_waveform_gradient_report"]
            or runtime["full_waveform_gradient_report"] != package["full_waveform_gradient_report"]
            or manifest["frozen_weights"] != binding(gradient_dir / "calibration_frozen_weights.json")
            or report["frozen_weights"] != manifest["frozen_weights"]
            or report["weights_refit_on_joint_rows"] is not False
            or report["measured_patient_rows"] != 48 or len(report["rows"]) != 48
            or report["healthy_gradient_rows"] != 0):
        raise ValueError("joint gradient provenance or full-row coverage differs")
    source = read_json(verified(inputs["fresh_panel_split_seal"]))
    reserved = source["joint_reserved_rows"]
    validate_reserved_rows(reserved, source["gradient_rows"])
    if [{key: r[key] for key in ROW_KEYS} for r in manifest["rows"]] != [{key: r[key] for key in ROW_KEYS} for r in reserved]:
        raise ValueError("joint gradient rows differ from the sealed source reserve")
    targets = {(r["speaker_id"], r["view"]): r for r in target_bank["rows"]}
    if len(targets) != 12 or target_bank["target_values_sealed_before_candidate_generation"] is not True:
        raise ValueError("joint clean pathological target bank coverage differs")
    topology_manifest = read_json(verified(runtime["topology_manifest"]))
    topologies = {r["case_id"]: r for r in topology_manifest["rows"]}
    results = {r["case_id"]: r for r in report["rows"]}
    patients = {r["case_id"] for r in reserved if r["label"] == "patient"}
    if set(topologies) != patients or set(results) != patients:
        raise ValueError("joint topologies or measurements omit patient rows")
    for row in manifest["rows"]:
        if row["label"] == "healthy":
            if row.get("joint_gradient") is not None or row["target"] is not None:
                raise ValueError("healthy row contains optimization inputs")
            continue
        if row["target"] != targets[(row["speaker_id"], row["view"])]:
            raise ValueError("joint target speaker or view differs")
        verified(row["base"])
        verified(row["target"])
        topology = topologies[row["case_id"]]
        if (topology["source_waveform_float32_sha256"] != row["base"]["float32_sha256"]
                or topology["samples"] != row["base"]["samples"]
                or results[row["case_id"]]["measurement"]["segment_samples"] != row["base"]["samples"]):
            raise ValueError("joint gradient or topology does not cover the full baseline")
        with np.load(verified(row["joint_gradient"]), allow_pickle=False) as tensors:
            if set(tensors.files) != set(AVQI_COMPONENT_NAMES) | {"joint"}:
                raise ValueError("joint tensor archive does not contain all six gradients")
            if any(tensors[name].shape != (row["base"]["samples"],) or not np.isfinite(tensors[name]).all() for name in tensors.files):
                raise ValueError("joint tensor archive has invalid full-waveform gradients")
    if package["execution_authorized"] is not report["full_waveform_gradient_prerequisites_pass"]:
        raise ValueError("joint execution must follow all full-waveform gradient gates")
    if package["execution_authorized"] and not all(all(r["gates"].values()) for r in report["rows"]):
        raise ValueError("joint execution has an unresolved gradient gate")


def generate_candidates(row: Mapping[str, Any], base: np.ndarray, gradient: np.ndarray | None,
                        sealed_base: Mapping[str, Any], alphas: tuple[float, ...], root: Path) -> list[dict[str, Any]]:
    if row["label"] == "healthy":
        if gradient is not None or row.get("target") is not None:
            raise ValueError("healthy controls cannot carry targets or gradients")
        return [{"alpha": alpha, "available": True, "unavailable_reason": None, **sealed_base} for alpha in alphas]
    if gradient is None or gradient.shape != base.shape or not np.isfinite(gradient).all():
        raise ValueError("patient gradient must match the complete baseline")
    result = []
    for index, alpha in enumerate(alphas):
        candidate, reason = candidate_from_gradient(base, gradient, alpha)
        if candidate is None:
            result.append({"alpha": alpha, "available": False, "unavailable_reason": reason,
                           "path": None, "sha256": None, "samples": int(base.size), "sample_rate": 16000,
                           "subtype": "PCM_24", "float32_sha256": None})
        else:
            if alpha == 0.0:
                audio_binding = dict(sealed_base)
            else:
                alpha_root = root / f"alpha_{index:02d}"
                alpha_root.mkdir(exist_ok=True)
                audio_binding = _write_pcm24(alpha_root / f"{row['case_id']}.wav", candidate)
            result.append({"alpha": alpha, "available": True, "unavailable_reason": None, **audio_binding})
    return result


def prepare(contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
            dependency: Mapping[str, str], contract_binding: Mapping[str, str], device: torch.device) -> str:
    source_dir, gradient_dir = load_joint_inputs(dependency, contract_binding["sha256"])
    source_seal = read_json(source_dir / "tau_diagnostic_split_seal.json")
    reserved = source_seal["joint_reserved_rows"]
    validate_reserved_rows(reserved, source_seal["gradient_rows"])
    rows = materialize_joint(contract, paths, output, reserved, device)
    gradient_report = measure_full_gradients(contract, paths, output, rows, gradient_dir, device)
    package = {
        "schema_version": SCHEMA + "-joint-execution-package", "scope": SCOPE, "contract": dict(contract_binding),
        "inputs": {
            "candidate_e_joint_runtime_binding": binding(output / "candidate_e_joint_runtime_binding.json"),
            "six_gradient_report": binding(gradient_dir / "six_gradient_fusion_report.json"),
            "six_gradient_receipt": dict(dependency),
            "fresh_panel_split_seal": binding(source_dir / "tau_diagnostic_split_seal.json"),
            "fresh_speaker_source_manifest": binding(source_dir / "tau_speaker_source_manifest.json"),
            "clean_target_label_bank": binding(output / "clean_target_label_bank.json"),
            "joint_gradient_manifest": binding(output / "joint_gradient_manifest.json"),
        },
        "legacy_fresh_input_names_are_diagnostic_aliases": True,
        "historical_overlap_count": 12, "within_current_run_speaker_overlap": 0,
        "full_waveform_gradient_report": binding(output / "full_waveform_gradient_report.json"),
        "execution_authorized": gradient_report["full_waveform_gradient_prerequisites_pass"],
        "candidate_exact_outcomes_opened": False, **BOUNDARIES,
    }
    validate_execution_package(package, contract_binding["sha256"])
    write_json(output / "joint_execution_package.json", package)
    if not package["execution_authorized"]:
        return PREPARATION_NO_GO

    base_root = output / "sealed_base_pcm24"
    candidate_root = output / "sealed_candidates_pcm24"
    base_root.mkdir()
    candidate_root.mkdir()
    alphas = tuple(contract["joint_gates"]["global_alpha_grid"])
    sealed_rows = []
    for row in rows:
        base = _verified_audio(row["base"], row["case_id"]).numpy()
        sealed_base = _write_pcm24(base_root / f"{row['case_id']}.wav", base)
        gradient = None
        if row["label"] == "patient":
            with np.load(verified(row["joint_gradient"]), allow_pickle=False) as tensors:
                gradient = tensors["joint"].copy()
        candidates = generate_candidates(row, base, gradient, sealed_base, alphas, candidate_root)
        sealed_rows.append({
            **{key: row[key] for key in ROW_KEYS}, "base": sealed_base, "target": row["target"],
            "base_float": row["base"], "candidates": candidates,
        })
    write_json(output / "joint_waveform_seal.json", {
        "schema_version": SCHEMA + "-joint-waveform-seal", "scope": SCOPE, "rows": sealed_rows,
        "decision": PREPARED, "execution_package": binding(output / "joint_execution_package.json"),
        "global_alpha_grid": list(alphas), "candidate_exact_outcomes_opened": False,
        "final_waveform_highpass": False, "normalization": contract["normalization"], **BOUNDARIES,
    })
    return PREPARED


def validate_waveforms(output: Path, contract: Mapping[str, Any], contract_sha: str) -> list[dict[str, Any]]:
    package = read_json(output / "joint_execution_package.json")
    validate_execution_package(package, contract_sha)
    if package["execution_authorized"] is not True:
        raise ValueError("joint exact evaluation is not authorized")
    seal = read_json(output / "joint_waveform_seal.json")
    if (seal["scope"] != SCOPE or seal["decision"] != PREPARED
            or seal["global_alpha_grid"] != contract["joint_gates"]["global_alpha_grid"]
            or seal["execution_package"] != binding(output / "joint_execution_package.json")
            or seal["candidate_exact_outcomes_opened"] is not False
            or seal["final_waveform_highpass"] is not False
            or seal["normalization"] != contract["normalization"]
            or any(seal.get(key) != value for key, value in BOUNDARIES.items())):
        raise ValueError("joint waveform seal differs")
    source = read_json(verified(package["inputs"]["fresh_panel_split_seal"]))
    reserved = source["joint_reserved_rows"]
    validate_reserved_rows(reserved, source["gradient_rows"])
    if [{key: r[key] for key in ROW_KEYS} for r in seal["rows"]] != [{key: r[key] for key in ROW_KEYS} for r in reserved]:
        raise ValueError("waveform rows differ from the predeclared TAU reserve")
    targets = read_json(verified(package["inputs"]["clean_target_label_bank"]))["rows"]
    target_map = {(r["speaker_id"], r["view"]): r for r in targets}
    for row in seal["rows"]:
        _verified_audio(row["base"], row["case_id"], expected_subtype="PCM_24")
        if row["label"] == "healthy":
            if row["target"] is not None or any(c["sha256"] != row["base"]["sha256"] for c in row["candidates"]):
                raise ValueError("healthy no-step identity differs")
        else:
            if row["target"] != target_map[(row["speaker_id"], row["view"])]:
                raise ValueError("same-speaker same-view target binding differs")
            _verified_audio(row["target"], row["case_id"])
        if [c["alpha"] for c in row["candidates"]] != seal["global_alpha_grid"]:
            raise ValueError("joint candidate alpha grid differs")
        for candidate in row["candidates"]:
            if candidate["available"]:
                _verified_audio(candidate, row["case_id"], expected_subtype="PCM_24")
    return seal["rows"]


def final_exact_items(rows: list[Mapping[str, Any]], selected_alpha: float,
                      alpha_grid: tuple[float, ...]) -> list[dict[str, str]]:
    if selected_alpha <= 0.0 or selected_alpha not in alpha_grid:
        raise ValueError("final requires one frozen nonzero calibration alpha")
    return exact_items_for_stage(rows, split="final", alphas=(selected_alpha,))


def evaluate(contract: Mapping[str, Any], paths: Mapping[str, Path], output: Path,
             dependency: Mapping[str, str], contract_sha: str, stage: str) -> str:
    alphas = tuple(contract["joint_gates"]["global_alpha_grid"])
    if stage == "joint_calibration":
        prepared_dir, _ = load_stage(dependency, contract_sha, "joint_prepare", PREPARED)
        selected_alpha = None
    else:
        calibration_dir, calibration_receipt = load_stage(dependency, contract_sha, "joint_calibration", CALIBRATED)
        calibration = read_json(calibration_dir / "joint_calibration_report.json")
        selected_alpha = calibration["selected_alpha"]
        prepared_dir, _ = load_stage(calibration_receipt["dependencies"][0], contract_sha, "joint_prepare", PREPARED)
        if selected_alpha != choose_alpha({float(k): v for k, v in calibration["alpha_summaries"].items()}):
            raise ValueError("frozen calibration selection is inconsistent")
    rows = validate_waveforms(prepared_dir, contract, contract_sha)
    split = "calibration" if stage == "joint_calibration" else "final"
    items = (exact_items_for_stage(rows, split=split, alphas=alphas) if split == "calibration"
             else final_exact_items(rows, selected_alpha, alphas))
    write_json(output / "sealed_exact_judge_request.json", {
        "scope": SCOPE, "split": split, "items": items,
        "waveform_seal": binding(prepared_dir / "joint_waveform_seal.json"),
        "selected_alpha": selected_alpha, "selector_waveforms_already_sealed": True, **BOUNDARIES,
    })
    exact = run_exact(items, exact_python=Path(contract["exact"]["python"]),
                      avqi_code_root=Path(contract["exact"]["root"]), expected_runtime=authority(contract, paths))
    if split == "final":
        alpha_index = alphas.index(selected_alpha)
        exact = {key.replace("candidate:0:", f"candidate:{alpha_index}:"): value for key, value in exact.items()}
    write_json(output / "exact_judge_result.json", {"split": split, "components_by_id": {key: value.tolist() for key, value in exact.items()}})
    if split == "calibration":
        summaries = {alpha: stage_rows(rows, exact, contract["normalization"]["target_scale"],
                                      split=split, alpha=alpha, alpha_index=index) for index, alpha in enumerate(alphas)}
        selected_alpha = choose_alpha(summaries)
        decision = CALIBRATED if selected_alpha is not None else CALIBRATION_NO_GO
        write_json(output / "joint_calibration_report.json", {
            "schema_version": SCHEMA + "-joint-calibration", "scope": SCOPE, "decision": decision,
            "alpha_summaries": {str(key): value for key, value in summaries.items()},
            "selected_alpha": selected_alpha, "final_exact_outcomes_opened": False,
            "final_diagnostic_evaluation_authorized": selected_alpha is not None, **BOUNDARIES,
        })
    else:
        summary = stage_rows(rows, exact, contract["normalization"]["target_scale"], split=split,
                             alpha=selected_alpha, alpha_index=alphas.index(selected_alpha))
        decision = FINAL_PASS if summary["decision"] == "PASS" else FINAL_NO_GO
        write_json(output / "joint_final_report.json", {
            "schema_version": SCHEMA + "-joint-final", "scope": SCOPE, "decision": decision,
            "selected_alpha": selected_alpha, "final_alpha_count": 1, "summary": summary,
            "final_tuning_performed": False, **BOUNDARIES,
        })
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("joint_prepare", "joint_calibration", "joint_final"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dependency-receipt", type=Path, required=True)
    parser.add_argument("--dependency-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run joint diagnostics in a Slurm compute allocation")
    source = repository_source(SOURCE_ROOT, args.source_commit)
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite joint diagnostic outputs")
    contract_binding = {"path": str(args.contract.resolve()), "sha256": args.contract_sha256}
    contract = read_json(verified(contract_binding))
    validate_contract(contract)
    paths = {name: verified(value) for name, value in contract["inputs"].items()}
    dependency = {"path": str(args.dependency_receipt.resolve()), "sha256": args.dependency_receipt_sha256}
    verified(dependency)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("joint preparation requires an allocated CUDA device")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "frozen_contract.json", contract)
    if args.stage == "joint_prepare":
        decision = prepare(contract, paths, args.output_dir, dependency, contract_binding, device)
    else:
        decision = evaluate(contract, paths, args.output_dir, dependency, args.contract_sha256, args.stage)
    for value in contract["inputs"].values():
        verified(value)
    repository_source(SOURCE_ROOT, args.source_commit)
    receipt(args.output_dir, args.stage, decision, contract_binding, source, dependencies=[dependency], extra={
        "joint_panel_executed": args.stage != "joint_prepare",
        "candidate_exact_outcomes_opened": args.stage != "joint_prepare",
        "full_joint_panel_completed": args.stage == "joint_final",
    })
    print(f"decision={decision} receipt={binding(args.output_dir / 'completion_receipt.json')}", flush=True)


if __name__ == "__main__":
    main()
