#!/usr/bin/env python3
"""Prepare and seal a result-blind external SVD Shimmer-dB panel.

The stage is authorized only by a passing v23 opened24 adjudication.  It uses
SVD metadata, a prior-speaker ledger, and a frozen salted rank to select six
patient speakers (three per sex).  Each speaker contributes paired CS/SV
waveforms, receives a preregistered degradation recipe, and is passed through
the frozen S3_500 generator in inference mode.

The emitted audio remains full-band.  No Shimmer target, base, or candidate
exact value is requested here, no waveform gradient step is taken, and no
generator optimizer is created.  The seal only authorizes the later target-
scalar and deterministic-selector stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from scripts.adjudicate_avqi_shimmer_db_deterministic_opened24_v23 import (
    PASS_DECISION as OPENED24_PASS_DECISION,
    RECEIPT_SCHEMA as OPENED24_RECEIPT_SCHEMA,
    REPORT_SCHEMA as OPENED24_REPORT_SCHEMA,
    TRAINING_DECISION,
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


SAMPLE_RATE = 16_000
PANEL_SCHEMA = "avqi-route-c-shimmer-db-external-svd-panel-seal-v24"
SEAL_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-panel-seal-receipt-v24"
)
PRIOR_LEDGER_SCHEMA = "avqi-route-c-prior-panel-speaker-ledger-v1"
SELECTION_SALT = "avqi-route-c-shimmer-db-external-svd-v24-20260830"
RECIPE_ASSIGNMENT = tuple(range(936, 948))
SPEAKERS_PER_SEX = 3
EXPECTED_SPEAKERS = 2 * SPEAKERS_PER_SEX
VIEWS = ("cs", "sv")
CONDITIONS = ("rir_only", "snr20", "snr10")
EXPECTED_CASES = EXPECTED_SPEAKERS * len(VIEWS)
SV_DURATION_MIN_SECONDS = 1.0
CS_DURATION_MIN_SECONDS = 3.0


@dataclass(frozen=True)
class SVDCase:
    speaker_id: str
    session_id: str
    sex: str
    diagnosis: str
    view: str
    condition: str
    recipe_index: int
    source_path: Path
    source_duration_seconds: float
    selection_rank_within_sex: int
    selection_digest: str

    @property
    def panel_speaker_id(self) -> str:
        return f"SVD:{self.speaker_id}"

    @property
    def case_id(self) -> str:
        return (
            f"sealed_external_svd__SVD_{self.speaker_id}__"
            f"{self.view}__{self.condition}"
        )


@dataclass
class PreparedCase:
    spec: SVDCase
    target_path: Path
    degraded_path: Path
    base_path: Path
    recipe: dict[str, Any]
    simulation_seed: int
    noise_start_sample: int


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "opened24-report",
        "opened24-receipt",
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
        raise ValueError("repository root does not contain the v24 preparer")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v24 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v24 preparation requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_opened24_authorization(
    report: dict[str, Any],
    receipt: dict[str, Any],
    *,
    report_sha256: str,
) -> None:
    if report.get("schema_version") != OPENED24_REPORT_SCHEMA:
        raise ValueError("opened24 v23 report schema drift")
    if receipt.get("schema_version") != OPENED24_RECEIPT_SCHEMA:
        raise ValueError("opened24 v23 receipt schema drift")
    for label, value in (("report", report), ("receipt", receipt)):
        if value.get("decision") != OPENED24_PASS_DECISION:
            raise ValueError(f"opened24 v23 {label} is not PASS")
        if value.get("external_speaker_panel_authorized") is not True:
            raise ValueError(f"opened24 v23 {label} did not authorize external panel")
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"opened24 v23 {label} over-authorized promotion")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"opened24 v23 {label} over-authorized joint panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"opened24 v23 {label} optimizer boundary drift")
        if value.get("authoritative_training_decision") != TRAINING_DECISION:
            raise ValueError(f"opened24 v23 {label} training decision drift")
    if report.get("exact_scoring_complete") is not True:
        raise ValueError("opened24 v23 exact scoring is incomplete")
    if not isinstance(report.get("gates"), dict) or not all(
        value is True for value in report["gates"].values()
    ):
        raise ValueError("opened24 v23 gates did not all pass")
    if receipt.get("artifact_sha256", {}).get("diagnostic_report.json") != (
        report_sha256
    ):
        raise ValueError("opened24 v23 receipt/report binding drift")


def canonical_speaker_id(dataset: str, speaker_id: str) -> str:
    dataset = dataset.strip().upper()
    speaker_id = speaker_id.strip()
    if not dataset or not speaker_id or ":" in dataset or ":" in speaker_id:
        raise ValueError("invalid prior-ledger speaker identity")
    return f"{dataset}:{speaker_id}"


def validate_prior_ledger(ledger: dict[str, Any]) -> set[str]:
    if ledger.get("schema_version") != PRIOR_LEDGER_SCHEMA:
        raise ValueError("prior-panel speaker ledger schema drift")
    if ledger.get("exact_outcomes_used_for_selection") is not False:
        raise ValueError("prior-panel ledger was selected using exact outcomes")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prior-panel speaker ledger is empty")
    speakers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("prior-panel ledger entry is not an object")
        canonical = canonical_speaker_id(
            str(entry.get("dataset", "")),
            str(entry.get("speaker_id", "")),
        )
        if entry.get("canonical_speaker_id") != canonical:
            raise ValueError("prior-panel ledger canonical identity drift")
        if not str(entry.get("panel_role", "")).strip():
            raise ValueError("prior-panel ledger entry lacks panel role")
        if canonical in speakers:
            raise ValueError(f"duplicate prior-ledger speaker: {canonical}")
        speakers.add(canonical)
    return speakers


def rank_digest(speaker_id: str, session_id: str) -> str:
    payload = f"{SELECTION_SALT}:{speaker_id}:{session_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _eligible_svd_speakers(
    sv_rows: list[dict[str, str]],
    cs_rows: list[dict[str, str]],
    sv_root: Path,
    cs_root: Path,
    excluded_speakers: set[str],
) -> dict[str, list[dict[str, Any]]]:
    sv_by_session = {row["session_id"]: row for row in sv_rows}
    cs_by_session = {row["session_id"]: row for row in cs_rows}
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for session_id in sorted(set(sv_by_session) & set(cs_by_session), key=int):
        sv_row = sv_by_session[session_id]
        cs_row = cs_by_session[session_id]
        if sv_row.get("health status") != "1" or cs_row.get("health status") != "1":
            continue
        speaker_id = str(sv_row.get("speaker id", ""))
        if speaker_id != cs_row.get("speaker id"):
            raise ValueError(f"SVD speaker mismatch: {session_id}")
        sex = str(sv_row.get("gender", ""))
        if sex != cs_row.get("gender") or sex not in {"female", "male"}:
            raise ValueError(f"SVD sex mismatch: {session_id}")
        if canonical_speaker_id("SVD", speaker_id) in excluded_speakers:
            continue
        sv_path = sv_root / sv_row["filename"]
        cs_path = cs_root / cs_row["filename"]
        if not sv_path.is_file() or not cs_path.is_file():
            continue
        sv_info = sf.info(sv_path)
        cs_info = sf.info(cs_path)
        if sv_info.channels != 1 or cs_info.channels != 1:
            continue
        sv_seconds = sv_info.frames / sv_info.samplerate
        cs_seconds = cs_info.frames / cs_info.samplerate
        if (
            sv_seconds < SV_DURATION_MIN_SECONDS
            or cs_seconds < CS_DURATION_MIN_SECONDS
        ):
            continue
        by_speaker.setdefault(speaker_id, []).append(
            {
                "speaker_id": speaker_id,
                "session_id": session_id,
                "sex": sex,
                "diagnosis": str(sv_row.get("diagnosis", "")),
                "sv_path": sv_path,
                "cs_path": cs_path,
                "sv_duration_seconds": sv_seconds,
                "cs_duration_seconds": cs_seconds,
            }
        )
    return by_speaker


def select_svd_cases(
    sv_rows: list[dict[str, str]],
    cs_rows: list[dict[str, str]],
    sv_root: Path,
    cs_root: Path,
    excluded_speakers: set[str],
) -> tuple[list[SVDCase], dict[str, Any]]:
    eligible = _eligible_svd_speakers(
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
    ranked = {
        sex: sorted(
            [row for row in one_session if row["sex"] == sex],
            key=lambda row: (
                rank_digest(row["speaker_id"], row["session_id"]),
                row["speaker_id"],
                row["session_id"],
            ),
        )
        for sex in ("female", "male")
    }
    if any(len(rows) < SPEAKERS_PER_SEX for rows in ranked.values()):
        raise ValueError("insufficient prior-ledger-disjoint SVD patient speakers")
    selected_rows = [
        row
        for sex in ("female", "male")
        for row in ranked[sex][:SPEAKERS_PER_SEX]
    ]
    cases: list[SVDCase] = []
    for speaker_index, row in enumerate(selected_rows):
        rank = (
            ranked[row["sex"]].index(row) + 1
        )
        digest = rank_digest(row["speaker_id"], row["session_id"])
        condition_pair = (
            ("rir_only", "snr20"),
            ("snr10", "rir_only"),
            ("snr20", "snr10"),
        )[speaker_index % SPEAKERS_PER_SEX]
        for view_index, view in enumerate(VIEWS):
            recipe_offset = speaker_index * len(VIEWS) + view_index
            cases.append(
                SVDCase(
                    speaker_id=row["speaker_id"],
                    session_id=row["session_id"],
                    sex=row["sex"],
                    diagnosis=row["diagnosis"],
                    view=view,
                    condition=condition_pair[view_index],
                    recipe_index=RECIPE_ASSIGNMENT[recipe_offset],
                    source_path=Path(row[f"{view}_path"]),
                    source_duration_seconds=float(
                        row[f"{view}_duration_seconds"]
                    ),
                    selection_rank_within_sex=rank,
                    selection_digest=digest,
                )
            )
    validate_case_contract(cases, excluded_speakers)
    selection = {
        "dataset": "SVD",
        "selection_mode": "metadata_only_result_blind",
        "health_status_mapping": {"1": "patient"},
        "paired_cs_sv_same_session_required": True,
        "minimum_raw_mono_duration_seconds": {
            "sv": SV_DURATION_MIN_SECONDS,
            "cs": CS_DURATION_MIN_SECONDS,
        },
        "eligible_session_per_speaker": "minimum_numeric_session_id",
        "prior_ledger_excluded_before_hash_ranking": True,
        "speaker_selection_salt": SELECTION_SALT,
        "ranking_digest": "SHA256(salt:speaker_id:session_id)",
        "selection_uses_diagnosis": False,
        "selection_uses_shimmer_or_avqi": False,
        "selected_speakers": sorted({case.panel_speaker_id for case in cases}),
        "selected_sessions": sorted({case.session_id for case in cases}, key=int),
        "sex_counts": dict(Counter(case.sex for case in cases[::2])),
        "prior_panel_speaker_overlap": 0,
    }
    return cases, selection


def validate_case_contract(
    cases: list[SVDCase],
    excluded_speakers: set[str],
) -> None:
    if len(cases) != EXPECTED_CASES or len({case.case_id for case in cases}) != (
        EXPECTED_CASES
    ):
        raise ValueError("external SVD panel case coverage drift")
    speakers = {case.panel_speaker_id for case in cases}
    if len(speakers) != EXPECTED_SPEAKERS or speakers & excluded_speakers:
        raise ValueError("external SVD panel speaker disjointness drift")
    if Counter(case.view for case in cases) != Counter({"cs": 6, "sv": 6}):
        raise ValueError("external SVD panel view balance drift")
    if Counter(case.condition for case in cases) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError("external SVD panel condition balance drift")
    if Counter(case.sex for case in cases[::2]) != Counter(
        {"female": 3, "male": 3}
    ):
        raise ValueError("external SVD panel sex balance drift")
    if tuple(case.recipe_index for case in cases) != RECIPE_ASSIGNMENT:
        raise ValueError("external SVD recipe assignment drift")
    for speaker in speakers:
        selected = [case for case in cases if case.panel_speaker_id == speaker]
        if len(selected) != 2 or {case.view for case in selected} != set(VIEWS):
            raise ValueError(f"external SVD speaker view drift: {speaker}")
        if len({case.session_id for case in selected}) != 1:
            raise ValueError(f"external SVD speaker session drift: {speaker}")


def safe_case_name(case_id: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]", "_", case_id)
    if not value:
        raise ValueError("external SVD case ID has no safe filename")
    return value


def extend_prior_ledger(
    ledger: dict[str, Any],
    cases: list[SVDCase],
    source_commit: str,
) -> dict[str, Any]:
    entries = [dict(entry) for entry in ledger["entries"]]
    existing = {
        str(entry["canonical_speaker_id"])
        for entry in entries
    }
    for case in cases[::2]:
        if case.panel_speaker_id in existing:
            raise ValueError("selected external SVD speaker is already in prior ledger")
        entries.append(
            {
                "dataset": "SVD",
                "speaker_id": case.speaker_id,
                "canonical_speaker_id": case.panel_speaker_id,
                "panel_role": "shimmer_db_external_svd_v24",
                "session_id": case.session_id,
                "source_commit": source_commit,
                "exact_shimmer_outcomes_opened_at_ledger_update": False,
            }
        )
    output = {
        "schema_version": PRIOR_LEDGER_SCHEMA,
        "exact_outcomes_used_for_selection": False,
        "entries": sorted(entries, key=lambda entry: entry["canonical_speaker_id"]),
        "added_by": "shimmer_db_external_svd_v24_panel_seal",
        "added_speaker_count": EXPECTED_SPEAKERS,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    validate_prior_ledger(output)
    return output


def prepare_waveforms(
    args: argparse.Namespace,
    cases: list[SVDCase],
    recipes: list[dict[str, Any]],
    simulation_config: dict[str, Any],
) -> list[PreparedCase]:
    # These dependencies belong to the remote simulation/GPU execution path.
    if str(args.simulation_root) not in sys.path:
        sys.path.insert(0, str(args.simulation_root))
    from simulate_degradation import apply_degradation_with_wind

    target_root = args.output_dir / "waveforms" / "target_clean_pathological"
    degraded_root = args.output_dir / "waveforms" / "degraded"
    base_root = args.output_dir / "waveforms" / "s3_500_base"
    for path in (target_root, degraded_root, base_root):
        path.mkdir(parents=True)
    reader = WdsReader()
    prepared: list[PreparedCase] = []
    try:
        for case in cases:
            source = read_clean(case.source_path)
            recipe = recipes[case.recipe_index]
            if recipe.get("split") != "test" or recipe.get("target_sample_rate") != (
                SAMPLE_RATE
            ):
                raise ValueError(f"fixed recipe drift: {case.recipe_index}")
            simulation_seed = stable_seed(
                args.seed,
                SELECTION_SALT,
                case.speaker_id,
                case.session_id,
                case.view,
                case.condition,
                recipe["uid"],
            )
            rng = random.Random(simulation_seed)
            noise_row = recipe_wds_row(recipe, "noise")
            rir_row = recipe_wds_row(recipe, "rir")
            noise, noise_start = crop_or_tile(
                reader.read(noise_row),
                source.shape[1],
                rng,
            )
            rir = reader.read(rir_row)
            selected_degradations = ["reverb"]
            snr = None
            if case.condition.startswith("snr"):
                selected_degradations.append("noise")
                snr = int(case.condition.removeprefix("snr"))
            clean_output, degraded = apply_degradation_with_wind(
                simulation_config,
                source,
                noise,
                rir,
                None,
                {"snr": 20 if snr is None else snr},
                selected_degradations,
                seed=simulation_seed,
            )
            clean_output = match_length(
                clean_output,
                source.shape[1],
            ).astype(np.float32)
            degraded = match_length(degraded, source.shape[1]).astype(np.float32)
            name = safe_case_name(case.case_id)
            target_path = target_root / f"{name}__target.wav"
            degraded_path = degraded_root / f"{name}__degraded.wav"
            base_path = base_root / f"{name}__s3_500.wav"
            sf.write(target_path, clean_output[0], SAMPLE_RATE, subtype="FLOAT")
            sf.write(degraded_path, degraded[0], SAMPLE_RATE, subtype="FLOAT")
            prepared.append(
                PreparedCase(
                    spec=case,
                    target_path=target_path,
                    degraded_path=degraded_path,
                    base_path=base_path,
                    recipe=recipe,
                    simulation_seed=simulation_seed,
                    noise_start_sample=noise_start,
                )
            )
    finally:
        reader.close()
    return prepared


def run_frozen_generator(
    args: argparse.Namespace,
    prepared: list[PreparedCase],
) -> None:
    # Generator helpers are loaded only where the frozen remote checkpoint exists.
    from scripts.evaluate_avqi_component_backprop import (
        enhance_waveform,
        load_generator,
    )
    from utils import load_config

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = load_config(args.generator_config)
    generator = load_generator(config, args.generator_checkpoint, device)
    with torch.inference_mode():
        for index, case in enumerate(prepared, start=1):
            values, sample_rate = sf.read(
                case.degraded_path,
                dtype="float32",
                always_2d=False,
            )
            if sample_rate != SAMPLE_RATE or values.ndim != 1:
                raise ValueError(f"invalid degraded waveform: {case.spec.case_id}")
            enhanced = enhance_waveform(
                generator,
                torch.from_numpy(values.copy()).to(device),
                config,
            ).detach().cpu().reshape(-1)
            if not bool(torch.isfinite(enhanced).all()) or float(
                enhanced.abs().max()
            ) >= 1.0:
                raise ValueError(
                    f"invalid frozen generator output: {case.spec.case_id}"
                )
            sf.write(case.base_path, enhanced.numpy(), SAMPLE_RATE, subtype="FLOAT")
            print(f"prepared_external_svd_base={index}/{len(prepared)}", flush=True)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def panel_rows(prepared: list[PreparedCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prepared:
        case = item.spec
        recipe = item.recipe
        rows.append(
            {
                "case_id": case.case_id,
                "dataset": "SVD",
                "panel_speaker_id": case.panel_speaker_id,
                "speaker_id": case.speaker_id,
                "session_id": case.session_id,
                "sex": case.sex,
                "diagnosis_record_only": case.diagnosis,
                "label": "patient",
                "view": case.view,
                "condition": case.condition,
                "selection_rank_within_sex": case.selection_rank_within_sex,
                "selection_digest": case.selection_digest,
                "recipe_index": case.recipe_index,
                "recipe_uid": recipe["uid"],
                "recipe_seed": recipe["seed"],
                "simulation_seed": item.simulation_seed,
                "source_path": str(case.source_path.resolve()),
                "source_sha256": sha256_file(case.source_path),
                "source_duration_seconds": case.source_duration_seconds,
                "target_path": str(item.target_path.resolve()),
                "target_sha256": sha256_file(item.target_path),
                "degraded_path": str(item.degraded_path.resolve()),
                "degraded_sha256": sha256_file(item.degraded_path),
                "base_path": str(item.base_path.resolve()),
                "base_sha256": sha256_file(item.base_path),
                "noise_shard_dir": recipe["noise"]["_shard_dir"],
                "noise_shard": recipe["noise"]["shard"],
                "noise_audio_member": recipe["noise"]["audio_member"],
                "noise_start_sample": item.noise_start_sample,
                "rir_shard_dir": recipe["rir"]["_shard_dir"],
                "rir_shard": recipe["rir"]["shard"],
                "rir_audio_member": recipe["rir"]["audio_member"],
            }
        )
    return rows


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
        "opened24_report": args.opened24_report,
        "opened24_receipt": args.opened24_receipt,
        "prior_panel_speaker_ledger": args.prior_panel_speaker_ledger,
        "sv_metadata": args.sv_metadata,
        "cs_metadata": args.cs_metadata,
        "fixed_recipes": args.fixed_recipes,
        "generator_config": args.generator_config,
        "generator_checkpoint": args.generator_checkpoint,
        "simulation_config": args.simulation_config,
    }
    source_hashes = {
        name: validate_hash(
            path,
            getattr(args, f"{name}_sha256"),
            name,
        )
        for name, path in input_paths.items()
    }
    simulation_source = args.simulation_root / "simulate_degradation.py"
    source_hashes["simulation_source"] = validate_hash(
        simulation_source,
        args.simulation_source_sha256,
        "simulation source",
    )
    opened24_report = read_json(args.opened24_report)
    opened24_receipt = read_json(args.opened24_receipt)
    validate_opened24_authorization(
        opened24_report,
        opened24_receipt,
        report_sha256=source_hashes["opened24_report"],
    )
    prior_ledger = read_json(args.prior_panel_speaker_ledger)
    excluded_speakers = validate_prior_ledger(prior_ledger)
    cases, selection = select_svd_cases(
        read_csv(args.sv_metadata),
        read_csv(args.cs_metadata),
        args.sv_root,
        args.cs_root,
        excluded_speakers,
    )
    recipes = read_fixed_recipes(args.fixed_recipes)
    simulation_config = yaml.safe_load(
        args.simulation_config.read_text(encoding="utf-8")
    )
    if not isinstance(simulation_config, dict):
        raise ValueError("simulation config is not a mapping")
    simulation_config["stft_cfg"]["sampling_rate"] = SAMPLE_RATE

    args.output_dir.mkdir(parents=True)
    prepared = prepare_waveforms(args, cases, recipes, simulation_config)
    run_frozen_generator(args, prepared)
    rows = panel_rows(prepared)
    updated_ledger = extend_prior_ledger(prior_ledger, cases, args.source_commit)
    ledger_path = args.output_dir / "prior_panel_speaker_ledger_after_v24.json"
    write_json(ledger_path, updated_ledger)
    seal = {
        "schema_version": PANEL_SCHEMA,
        "stage": "prepare_and_seal_before_external_shimmer_exact",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "authorization": {
            "opened24_v23_decision": opened24_report["decision"],
            "opened24_report_sha256": source_hashes["opened24_report"],
            "opened24_receipt_sha256": source_hashes["opened24_receipt"],
            "external_speaker_panel_authorized": True,
        },
        "selection": selection,
        "case_count": len(rows),
        "speaker_count": len({row["panel_speaker_id"] for row in rows}),
        "views": dict(Counter(row["view"] for row in rows)),
        "conditions": dict(Counter(row["condition"] for row in rows)),
        "severity_labels_created": False,
        "severity_gate_source": (
            "passed opened24 v23 calibration/validation evidence only"
        ),
        "source_provenance": source_provenance,
        "source_sha256": source_hashes,
        "prior_panel_speaker_ledger_input_sha256": source_hashes[
            "prior_panel_speaker_ledger"
        ],
        "prior_panel_speaker_ledger_after_v24_sha256": sha256_file(ledger_path),
        "recipe_assignment": {
            "indices": list(RECIPE_ASSIGNMENT),
            "selection_uses_exact_outcomes": False,
            "unused_by_shimmer_v14_v15": True,
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
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    seal_path = args.output_dir / "panel_seal.json"
    write_json(seal_path, seal)
    receipt = {
        "schema_version": SEAL_RECEIPT_SCHEMA,
        "decision": "SEALED_SHIMMER_DB_EXTERNAL_SVD_PANEL_EXACT_UNOPENED_V24",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "exact_shimmer_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {
            "panel_seal.json": sha256_file(seal_path),
            ledger_path.name: sha256_file(ledger_path),
        },
    }
    receipt_path = args.output_dir / "seal_receipt.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "panel_seal_sha256": sha256_file(seal_path),
                "updated_ledger_sha256": sha256_file(ledger_path),
                "seal_receipt_sha256": sha256_file(receipt_path),
                "exact_shimmer_outcomes_opened": False,
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
