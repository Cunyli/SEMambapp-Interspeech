#!/usr/bin/env python3
"""Materialize the sealed SVD fusion panel and seal clean target scalars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch
import yaml

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.evaluate_avqi_component_backprop import enhance_waveform, load_generator
from scripts.evaluate_avqi_route_c_multicomponent_gradients import verify_source
from scripts.evaluate_avqi_shimmer_fresh_panel import (
    avqi_code_tree_sha256,
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
from scripts.seal_avqi_route_c_six_gradient_svd_source_panel_v2 import (
    AUDIT_SPLITS,
    CONTRACT_SCHEMA_VERSION,
    EXPECTED_CASES,
    PANEL_DECISION as SOURCE_PANEL_DECISION,
    PANEL_SCHEMA_VERSION as SOURCE_PANEL_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION as SOURCE_RECEIPT_SCHEMA_VERSION,
    STRATA,
    TRAINING_NO_GO,
    validate_contract,
)
from utils import load_config


SAMPLE_RATE = 16_000
MATERIALIZED_SCHEMA_VERSION = "avqi-route-c-six-gradient-svd-materialized-panel-v2"
RECEIPT_SCHEMA_VERSION = "avqi-route-c-six-gradient-svd-materialized-receipt-v2"
MATERIALIZED_DECISION = "SEALED_SVD_SIX_GRADIENT_TARGETS_AND_BASE_WAVEFORMS_V2"
TARGET_MARKER = "AVQI_SVD_TARGET_COMPONENTS_JSON="
TARGET_METRICS = (
    "avqi",
    "cpps",
    "hnr",
    "jitter_local",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
TARGET_PROGRAM = r"""
import json
import math
import sys

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
metric_names = (
    "avqi",
    "cpps",
    "hnr",
    "jitter_local",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    metrics = run_avqi(
        item["path"],
        item["path"],
        target_sr=16000,
        speaking_type=item["view"],
        step_versions=step_versions,
        remove_sv_silence_with_sox=False,
    )
    values = {name: float(metrics[name]) for name in metric_names}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite sealed SVD target component")
    rows.append({"case_id": item["case_id"], "components": values})
print("AVQI_SVD_TARGET_COMPONENTS_JSON=" + json.dumps({"rows": rows}, sort_keys=True))
"""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "contract",
        "source-panel-seal",
        "source-panel-receipt",
        "updated-speaker-ledger",
        "base-label-bank",
        "fixed-recipes",
        "generator-config",
        "generator-checkpoint",
        "simulation-config",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--simulation-source-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accepted-base-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260906)
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return value


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


def validate_source_panel(
    seal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    seal_sha256: str,
    ledger_sha256: str,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    if (
        seal.get("schema_version") != SOURCE_PANEL_SCHEMA_VERSION
        or seal.get("decision") != SOURCE_PANEL_DECISION
        or receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("decision") != SOURCE_PANEL_DECISION
        or seal.get("contract_sha256") != contract_sha256
        or seal.get("updated_speaker_ledger_sha256") != ledger_sha256
        or receipt.get("artifact_sha256", {}).get("svd_source_panel_seal_v2.json")
        != seal_sha256
        or receipt.get("artifact_sha256", {}).get(
            "prior_speaker_ledger_after_svd_v2.json"
        )
        != ledger_sha256
        or receipt.get("source_commit") != seal.get("source", {}).get("head")
        or receipt.get("source_branch") != seal.get("source", {}).get("branch")
    ):
        raise ValueError("SVD source-panel binding differs")
    for value, label in ((seal, "source seal"), (receipt, "source receipt")):
        if (
            value.get("base_or_candidate_exact_outcomes_opened") is not False
            or value.get("joint_panel_authorized") is not False
            or value.get("generator_optimizer_steps") != 0
            or value.get("authoritative_training_decision") != TRAINING_NO_GO
        ):
            raise ValueError(f"{label} authorization boundary differs")
    if (
        seal.get("source_split_sealed_before_simulation") is not True
        or seal.get("waveform_generation_performed") is not False
        or seal.get("target_scalar_values_opened") is not False
    ):
        raise ValueError("SVD source panel was not sealed before materialization")
    rows = seal.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise ValueError("SVD source-panel row coverage differs")
    speakers: set[str] = set()
    case_ids: set[str] = set()
    strata = {split: set() for split in AUDIT_SPLITS}
    recipe_indices: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SVD source-panel row differs")
        split = row.get("split")
        speaker = row.get("canonical_speaker_id")
        case_id = row.get("case_id")
        view = row.get("view")
        target_path = Path(str(row.get("target_source_path", "")))
        paired_cs_path = Path(str(row.get("paired_cs_path", "")))
        paired_sv_path = Path(str(row.get("paired_sv_path", "")))
        if (
            split not in AUDIT_SPLITS
            or not isinstance(speaker, str)
            or speaker in speakers
            or not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or view not in {"cs", "sv"}
            or row.get("target_all_six_components_scorable") is not True
            or not target_path.is_absolute()
            or sha256_file(target_path) != row.get("target_source_sha256")
            or not paired_cs_path.is_absolute()
            or sha256_file(paired_cs_path) != row.get("paired_cs_sha256")
            or not paired_sv_path.is_absolute()
            or sha256_file(paired_sv_path) != row.get("paired_sv_sha256")
            or target_path != (paired_cs_path if view == "cs" else paired_sv_path)
        ):
            raise ValueError("SVD source-panel row contract differs")
        speakers.add(speaker)
        case_ids.add(case_id)
        recipe_indices.append(int(row["recipe_index"]))
        strata[split].add(f"{row['sex']}/{view}")
    expected_strata = {f"{sex}/{view}" for sex, view in STRATA}
    if any(value != expected_strata for value in strata.values()):
        raise ValueError("SVD source-panel strata differ")
    if recipe_indices != list(range(972, 980)):
        raise ValueError("SVD source-panel recipe assignment differs")
    return [dict(row) for row in rows]


def validate_updated_ledger(
    ledger: Mapping[str, Any],
    rows: list[dict[str, Any]],
    source_commit: str,
) -> None:
    if (
        ledger.get("schema_version")
        != "avqi-route-c-prior-panel-speaker-ledger-v1"
        or ledger.get("exact_outcomes_used_for_selection") is not False
        or ledger.get("added_by")
        != "six_gradient_fusion_svd_source_panel_v2"
        or ledger.get("added_speaker_count") != EXPECTED_CASES
        or ledger.get("generator_optimizer_steps") != 0
        or ledger.get("formal_generator_training_authorized") is not False
        or ledger.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("updated SVD speaker ledger boundary differs")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("updated SVD speaker ledger entries differ")
    by_speaker: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("updated SVD speaker ledger entry differs")
        canonical = entry.get("canonical_speaker_id")
        if not isinstance(canonical, str) or canonical in by_speaker:
            raise ValueError("updated SVD speaker ledger identity differs")
        by_speaker[canonical] = entry
    selected = {row["canonical_speaker_id"]: row for row in rows}
    if len(selected) != EXPECTED_CASES:
        raise ValueError("updated SVD speaker ledger selection differs")
    role_entries = {
        canonical: entry
        for canonical, entry in by_speaker.items()
        if entry.get("panel_role") == "six_gradient_fusion_svd_source_v2"
    }
    if set(role_entries) != set(selected):
        raise ValueError("updated SVD speaker ledger panel membership differs")
    for canonical, row in selected.items():
        entry = role_entries[canonical]
        if (
            entry.get("dataset") != "SVD"
            or str(entry.get("speaker_id")) != str(row["speaker_id"])
            or str(entry.get("session_id")) != str(row["session_id"])
            or entry.get("source_commit") != source_commit
            or entry.get("target_all_six_component_scorability_boolean_used")
            is not True
            or entry.get("target_scalar_values_used") is not False
            or entry.get("base_or_candidate_exact_outcomes_used") is not False
        ):
            raise ValueError("updated SVD speaker ledger selected entry differs")


def prepare_waveforms(
    rows: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
    simulation_config: dict[str, Any],
    simulation_root: Path,
    output_dir: Path,
    seed: int,
    selection_salt: str,
    snr_schedule: list[int],
) -> list[dict[str, Any]]:
    # The simulator lives outside this repository and is hash-bound by the caller.
    sys.path.insert(0, str(simulation_root))
    from simulate_degradation import apply_degradation_with_wind

    target_root = output_dir / "waveforms" / "target_clean_pathological"
    degraded_root = output_dir / "waveforms" / "degraded"
    base_root = output_dir / "waveforms" / "s3_500_base"
    for path in (target_root, degraded_root, base_root):
        path.mkdir(parents=True)
    reader = WdsReader()
    prepared = []
    try:
        for row, snr in zip(rows, snr_schedule, strict=True):
            recipe = recipes[int(row["recipe_index"])]
            if recipe.get("split") != "test" or recipe.get("target_sample_rate") != SAMPLE_RATE:
                raise ValueError("fixed degradation recipe differs")
            source = read_clean(Path(row["target_source_path"]))
            simulation_seed = stable_seed(
                seed,
                selection_salt,
                row["speaker_id"],
                row["session_id"],
                row["view"],
                recipe["uid"],
            )
            rng = random.Random(simulation_seed)
            noise = reader.read(recipe_wds_row(recipe, "noise"))
            noise, noise_start = crop_or_tile(noise, source.shape[1], rng)
            rir = reader.read(recipe_wds_row(recipe, "rir"))
            _, degraded = apply_degradation_with_wind(
                simulation_config,
                source,
                noise,
                rir,
                None,
                {"snr": snr},
                ["reverb", "noise"],
                seed=simulation_seed,
            )
            degraded = match_length(degraded, source.shape[1]).astype(np.float32)
            safe_id = hashlib.sha256(row["case_id"].encode("utf-8")).hexdigest()[:16]
            target_path = target_root / f"{safe_id}__target.wav"
            degraded_path = degraded_root / f"{safe_id}__degraded.wav"
            base_path = base_root / f"{safe_id}__s3_500.wav"
            sf.write(target_path, source[0], SAMPLE_RATE, subtype="FLOAT")
            sf.write(degraded_path, degraded[0], SAMPLE_RATE, subtype="FLOAT")
            prepared.append(
                {
                    **row,
                    "target_path": str(target_path.resolve()),
                    "degraded_path": str(degraded_path.resolve()),
                    "base_path": str(base_path.resolve()),
                    "recipe_uid": recipe["uid"],
                    "simulation_seed": simulation_seed,
                    "noise_start_sample": noise_start,
                    "snr_db": snr,
                }
            )
    finally:
        reader.close()
    return prepared


def run_generator(
    prepared: list[dict[str, Any]],
    generator_config: Path,
    generator_checkpoint: Path,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    config = load_config(generator_config)
    generator = load_generator(config, generator_checkpoint, device)
    with torch.inference_mode():
        for index, row in enumerate(prepared, start=1):
            degraded, sample_rate = sf.read(row["degraded_path"], dtype="float32")
            if sample_rate != SAMPLE_RATE or degraded.ndim != 1:
                raise ValueError("materialized degraded waveform differs")
            enhanced = enhance_waveform(
                generator,
                torch.from_numpy(degraded.copy()).to(device),
                config,
            ).detach().cpu().reshape(-1)
            if not bool(torch.isfinite(enhanced).all()) or float(enhanced.abs().max()) >= 1.0:
                raise ValueError("frozen S3_500 output is invalid")
            sf.write(row["base_path"], enhanced.numpy(), SAMPLE_RATE, subtype="FLOAT")
            print(f"materialized_svd_fusion_base={index}/{len(prepared)}", flush=True)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_targets(
    prepared: list[dict[str, Any]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, dict[str, float]]:
    request = {
        "items": [
            {"case_id": row["case_id"], "path": row["target_path"], "view": row["view"]}
            for row in prepared
        ]
    }
    completed = subprocess.run(
        [str(exact_python), "-c", TARGET_PROGRAM, str(avqi_code_root)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("sealed SVD target scoring failed: " + completed.stderr[-4000:])
    lines = [line for line in completed.stdout.splitlines() if line.startswith(TARGET_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("sealed SVD target scorer marker differs")
    payload = json.loads(lines[0][len(TARGET_MARKER) :])
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != [
        row["case_id"] for row in prepared
    ]:
        raise ValueError("sealed SVD target scorer coverage differs")
    result = {}
    for row in rows:
        components = row.get("components")
        if not isinstance(components, dict) or set(components) != set(TARGET_METRICS):
            raise ValueError("sealed SVD target component fields differ")
        parsed = {name: float(components[name]) for name in TARGET_METRICS}
        if any(not math.isfinite(value) for value in parsed.values()):
            raise ValueError("sealed SVD target component is nonfinite")
        result[row["case_id"]] = parsed
    return result


def audio_metadata(path: Path) -> tuple[str, int]:
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"invalid materialized audio: {path}")
    return sha256_file(path), int(info.frames)


def label_bank_rows(
    base_rows: list[dict[str, str]],
    fieldnames: list[str],
    prepared: list[dict[str, Any]],
    targets: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row) for row in base_rows]
    for item in prepared:
        target_path = Path(item["target_path"])
        base_path = Path(item["base_path"])
        target_sha256, target_frames = audio_metadata(target_path)
        base_sha256, base_frames = audio_metadata(base_path)
        paired_cs_info = sf.info(item["paired_cs_path"])
        paired_sv_info = sf.info(item["paired_sv_path"])
        if paired_cs_info.channels != 1 or paired_sv_info.channels != 1:
            raise ValueError("paired SVD source audio is not mono")
        common = {field: "" for field in fieldnames}
        common.update(
            {
                "schema_version": "svd_six_gradient_fusion_v2",
                "speaker_id": item["canonical_speaker_id"],
                "pair_id": item["sample_id"],
                "sample_id": item["sample_id"],
                "view": item["view"],
                "sample_group": item["sample_group"],
                "label": "patient",
                "source": "SVD",
                "sex": item["sex"],
                "split_version": "route-c-six-gradient-fusion-svd-v2",
                "split": item["split"],
                "target_sr": SAMPLE_RATE,
                "speaking_type": item["view"],
                "same_noise_rir_seed_across_cs_sv": 0,
            }
        )
        clean_row = dict(common)
        clean_row.update(
            {
                "condition_id": "clean",
                "cs_uid": item["sample_id"],
                "sv_uid": item["sample_id"],
                "cs_path": str(target_path) if item["view"] == "cs" else item["paired_cs_path"],
                "cs_sha256": target_sha256 if item["view"] == "cs" else item["paired_cs_sha256"],
                "cs_sample_rate": (
                    SAMPLE_RATE
                    if item["view"] == "cs"
                    else paired_cs_info.samplerate
                ),
                "cs_frames": (
                    target_frames
                    if item["view"] == "cs"
                    else paired_cs_info.frames
                ),
                "sv_path": str(target_path) if item["view"] == "sv" else item["paired_sv_path"],
                "sv_sha256": target_sha256 if item["view"] == "sv" else item["paired_sv_sha256"],
                "sv_sample_rate": (
                    SAMPLE_RATE
                    if item["view"] == "sv"
                    else paired_sv_info.samplerate
                ),
                "sv_frames": (
                    target_frames
                    if item["view"] == "sv"
                    else paired_sv_info.frames
                ),
                "all_praat": 1,
                "remove_sv_silence_with_sox": 0,
                "scoring_status": "ok",
                "error_type": "",
                "error_message": "",
                **targets[item["case_id"]],
            }
        )
        source_row = dict(common)
        source_row.update(
            {
                "condition_id": "aug16k_phone",
                "cs_uid": item["sample_id"],
                "sv_uid": item["sample_id"],
                "cs_path": str(base_path) if item["view"] == "cs" else item["paired_cs_path"],
                "cs_sha256": base_sha256 if item["view"] == "cs" else item["paired_cs_sha256"],
                "cs_sample_rate": (
                    SAMPLE_RATE
                    if item["view"] == "cs"
                    else paired_cs_info.samplerate
                ),
                "cs_frames": (
                    base_frames
                    if item["view"] == "cs"
                    else paired_cs_info.frames
                ),
                "sv_path": str(base_path) if item["view"] == "sv" else item["paired_sv_path"],
                "sv_sha256": base_sha256 if item["view"] == "sv" else item["paired_sv_sha256"],
                "sv_sample_rate": (
                    SAMPLE_RATE
                    if item["view"] == "sv"
                    else paired_sv_info.samplerate
                ),
                "sv_frames": (
                    base_frames
                    if item["view"] == "sv"
                    else paired_sv_info.frames
                ),
                "all_praat": 0,
                "remove_sv_silence_with_sox": 0,
                "scoring_status": "source_unscored_result_blind",
                "error_type": "",
                "error_message": "",
            }
        )
        rows.extend((clean_row, source_row))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite SVD materialization: {args.output_dir}")
    source = verify_source(
        args.source_root.resolve(), args.source_commit, args.accepted_base_commit
    )
    input_args = {
        "contract": (args.contract, args.contract_sha256),
        "source_panel_seal": (args.source_panel_seal, args.source_panel_seal_sha256),
        "source_panel_receipt": (args.source_panel_receipt, args.source_panel_receipt_sha256),
        "updated_speaker_ledger": (args.updated_speaker_ledger, args.updated_speaker_ledger_sha256),
        "base_label_bank": (args.base_label_bank, args.base_label_bank_sha256),
        "fixed_recipes": (args.fixed_recipes, args.fixed_recipes_sha256),
        "generator_config": (args.generator_config, args.generator_config_sha256),
        "generator_checkpoint": (args.generator_checkpoint, args.generator_checkpoint_sha256),
        "simulation_config": (args.simulation_config, args.simulation_config_sha256),
    }
    paths = {
        name: _verified_file(path, digest, name)
        for name, (path, digest) in input_args.items()
    }
    contract = _read_json(paths["contract"], "fusion contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("fusion contract schema differs")
    validate_contract(contract)
    materialization = contract["waveform_materialization"]
    expected_hashes = {
        "base_label_bank": contract["external_svd_source_panel"][
            "normalization_label_bank_sha256"
        ],
        "fixed_recipes": materialization["fixed_recipes_sha256"],
        "generator_config": materialization["generator_config_sha256"],
        "generator_checkpoint": materialization["generator_checkpoint_sha256"],
        "simulation_config": materialization["simulation_config_sha256"],
    }
    observed_hashes = {
        name: input_args[name][1] for name in expected_hashes
    }
    if observed_hashes != expected_hashes:
        raise ValueError("SVD materialization inputs differ from frozen contract")
    simulation_source = (args.simulation_root / "simulate_degradation.py").resolve()
    _verified_file(simulation_source, args.simulation_source_sha256, "simulation source")
    if args.simulation_source_sha256 != materialization["simulation_source_sha256"]:
        raise ValueError("simulation source differs from frozen contract")
    if avqi_code_tree_sha256(args.avqi_code_root.resolve()) != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code tree differs")
    exact_policy = contract["exact_authority"]
    if (
        str(args.exact_python.resolve()) != exact_policy["python"]
        or str(args.avqi_code_root.resolve()) != exact_policy["avqi_code_root"]
        or args.avqi_code_tree_sha256 != exact_policy["avqi_code_tree_sha256"]
    ):
        raise ValueError("exact AVQI authority differs from frozen contract")
    source_seal = _read_json(paths["source_panel_seal"], "source panel seal")
    source_receipt = _read_json(paths["source_panel_receipt"], "source panel receipt")
    rows = validate_source_panel(
        source_seal,
        source_receipt,
        seal_sha256=args.source_panel_seal_sha256,
        ledger_sha256=args.updated_speaker_ledger_sha256,
        contract_sha256=args.contract_sha256,
    )
    updated_ledger = _read_json(
        paths["updated_speaker_ledger"], "updated speaker ledger"
    )
    validate_updated_ledger(updated_ledger, rows, source["head"])
    recipes = read_fixed_recipes(paths["fixed_recipes"])
    simulation_config = yaml.safe_load(paths["simulation_config"].read_text(encoding="utf-8"))
    if not isinstance(simulation_config, dict):
        raise ValueError("simulation config must be a mapping")
    simulation_config["stft_cfg"]["sampling_rate"] = SAMPLE_RATE
    args.output_dir.mkdir(parents=True)
    prepared = prepare_waveforms(
        rows,
        recipes,
        simulation_config,
        args.simulation_root.resolve(),
        args.output_dir,
        args.seed,
        contract["external_svd_source_panel"]["selection_salt"],
        list(materialization["snr_db_by_case_order"]),
    )
    run_generator(
        prepared,
        paths["generator_config"],
        paths["generator_checkpoint"],
        args.device,
    )
    target_components = score_targets(
        prepared, args.exact_python.resolve(), args.avqi_code_root.resolve()
    )
    with paths["base_label_bank"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        base_rows = [dict(row) for row in reader]
    if not fieldnames or not base_rows:
        raise ValueError("normalization label bank is empty")
    merged_rows = label_bank_rows(base_rows, fieldnames, prepared, target_components)
    label_bank_path = args.output_dir / "exact_target_result_blind_source_label_bank_v2.csv"
    write_csv(label_bank_path, merged_rows, fieldnames)
    case_rows = []
    waveform_hashes = {}
    for row in prepared:
        target_path = Path(row["target_path"])
        degraded_path = Path(row["degraded_path"])
        base_path = Path(row["base_path"])
        hashes = {
            "target": sha256_file(target_path),
            "degraded": sha256_file(degraded_path),
            "base": sha256_file(base_path),
        }
        waveform_hashes[row["case_id"]] = hashes
        exact_values = {
            name: float(target_components[row["case_id"]][name])
            for name in TARGET_METRICS
        }
        values = torch.tensor(
            [exact_values[name] for name in AVQI_COMPONENT_NAMES],
            dtype=torch.float32,
        ).tolist()
        target_payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
        exact_target_payload = json.dumps(
            exact_values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        base_values, base_sr = sf.read(base_path, dtype="float32")
        if base_sr != SAMPLE_RATE or base_values.ndim != 1 or not np.isfinite(base_values).all():
            raise ValueError("materialized base waveform integrity differs")
        case_rows.append(
            {
                **{key: row[key] for key in (
                    "case_id", "speaker_id", "canonical_speaker_id", "session_id",
                    "sample_id", "split", "sample_group", "sex", "view", "condition",
                    "recipe_index", "recipe_uid", "simulation_seed", "snr_db",
                )},
                "target_path": str(target_path),
                "target_sha256": hashes["target"],
                "degraded_path": str(degraded_path),
                "degraded_sha256": hashes["degraded"],
                "base_path": str(base_path),
                "base_sha256": hashes["base"],
                "exact_target_components": exact_values,
                "exact_target_components_sha256": hashlib.sha256(
                    exact_target_payload
                ).hexdigest(),
                "target_components": values,
                "target_vector_sha256": hashlib.sha256(target_payload).hexdigest(),
                "base_abs_max": float(np.max(np.abs(base_values))),
                "base_clipping_fraction": float(np.mean(np.abs(base_values) >= 1.0)),
            }
        )
    seal = {
        "schema_version": MATERIALIZED_SCHEMA_VERSION,
        "decision": MATERIALIZED_DECISION,
        "source": source,
        "contract_sha256": args.contract_sha256,
        "source_panel_sha256": {
            "seal": args.source_panel_seal_sha256,
            "receipt": args.source_panel_receipt_sha256,
            "updated_speaker_ledger": args.updated_speaker_ledger_sha256,
        },
        "input_sha256": {
            **{name: digest for name, (_, digest) in input_args.items()},
            "simulation_source": args.simulation_source_sha256,
            "exact_avqi_code_tree": args.avqi_code_tree_sha256,
        },
        "label_bank_sha256": sha256_file(label_bank_path),
        "waveform_sha256": waveform_hashes,
        "rows": case_rows,
        "clean_target_scalars_sealed_before_gradient_measurement": True,
        "target_scalar_values_opened": True,
        "base_or_candidate_exact_outcomes_opened": False,
        "emitted_waveform_highpass": False,
        "generator_mode": "frozen_inference_only",
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "joint_panel_authorized": False,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    seal_path = args.output_dir / "svd_materialized_panel_seal_v2.json"
    _write_json(seal_path, seal)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "decision": MATERIALIZED_DECISION,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {
            label_bank_path.name: sha256_file(label_bank_path),
            seal_path.name: sha256_file(seal_path),
        },
        "waveform_sha256": waveform_hashes,
        "input_sha256": seal["input_sha256"],
        "target_scalar_values_opened": True,
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
                "decision": MATERIALIZED_DECISION,
                "label_bank_sha256": sha256_file(label_bank_path),
                "panel_seal_sha256": sha256_file(seal_path),
                "receipt_sha256": sha256_file(receipt_path),
                "base_or_candidate_exact_outcomes_opened": False,
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
