#!/usr/bin/env python3
"""Audit existing TAU evidence and current source capacity without scoring audio.

This is a saturation proof for the pinned current TAU universe, not a panel
selector. Historical metric numbers are checked only for recorded availability;
neither metric values nor result-derived severity labels are emitted or ranked.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping
import unicodedata

import soundfile as sf


SCHEMA = "avqi-route-c-tau-history-capacity-v1"
FAILURE = "NO_GO_ROUTE_C_TAU_SOURCE_PANEL_ALL_CURRENT_SPEAKERS_EXACT_OPENED_V1"
TRAINING_NO_GO = "NO_GO_AVQI_T2_TRAINING"
COMPONENTS = ("cpps", "hnr", "shimmer_percent", "shimmer_db", "slope", "tilt")
SPEAKER_PATTERN = re.compile(r"(?:FD|SD|PD|V|ÄHH)\d+|(?:PD|HC)_\d+")
REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:FD|SD|PD|V|ÄHH)\d+|(?:PD|HC)_\d+)(?![A-Za-z0-9])"
)
INPUT_NAMES = {
    "tau_manifest", "tau_inventory", "tau_paired_scores", "tau_directory_policy",
    "tau_elina_metadata", "expanded_labels", "expanded_label_receipt",
    "external_exact", "external_exact_receipt", "sd13_exact", "sd13_exact_report",
    "prior_ledger", "readiness_report", "readiness_receipt", "exact_main",
}
BOUNDARIES = {
    "new_target_scalar_values_opened": False,
    "base_or_candidate_exact_outcomes_opened": False,
    "new_waveforms_materialized": False,
    "successor_source_selection_performed": False,
    "six_gradient_evaluation_submitted": False,
    "joint_panel_authorized": False,
    "joint_panel_submitted": False,
    "svd_used_for_new_testing": False,
    "formal_generator_training_submitted": False,
    "generator_optimizer_created": False,
    "generator_optimizer_steps": 0,
    "authoritative_training_decision": TRAINING_NO_GO,
}
MINIMUM_SECONDS = {"cs": 3.0, "sv": 1.0}
FUSION_QUOTAS = {"patient/female": 4, "patient/male": 4}
JOINT_QUOTAS = {
    "patient/female": 3, "patient/male": 3,
    "healthy/female": 3, "healthy/male": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def canonical_speaker(value: str) -> str:
    speaker = unicodedata.normalize("NFC", value.strip())
    if speaker.startswith("TAU:"):
        speaker = speaker.removeprefix("TAU:")
    if SPEAKER_PATTERN.fullmatch(speaker) is None:
        raise ValueError(f"invalid TAU speaker identity: {value!r}")
    return f"TAU:{speaker}"


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != SCHEMA or contract.get("failure_decision") != FAILURE:
        raise ValueError("TAU audit contract identity differs")
    if contract.get("boundaries") != BOUNDARIES:
        raise ValueError("TAU audit authorization boundary differs")
    if set(contract.get("inputs", {})) != INPUT_NAMES:
        raise ValueError("TAU audit input inventory differs")
    if contract.get("source_policy") != {
        "dataset": "TAU", "active_collections": ["Elina", "Nelly"],
        "origin_directories": "provenance_only_not_successor_source",
        "minimum_raw_mono_seconds": MINIMUM_SECONDS,
        "sex_inference_permitted": False, "metric_or_severity_ranking_permitted": False,
        "all_historical_opened_or_development_speakers_excluded": True,
    }:
        raise ValueError("TAU source policy differs")
    if contract.get("capacity_gates") != {
        "six_gradient_distinct_speaker_quotas": FUSION_QUOTAS,
        "joint_distinct_speaker_quotas": JOINT_QUOTAS,
        "gradient_and_joint_speakers_disjoint": True,
        "all_speakers_require_same_speaker_cs_sv": True,
        "thresholds_may_not_be_weakened": True,
    }:
        raise ValueError("TAU capacity thresholds differ")
    for item in contract["inputs"].values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("invalid input binding")
        if not Path(item["path"]).is_absolute() or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
            raise ValueError("input needs absolute path and SHA-256")


def verify_bindings(bindings: Mapping[str, Mapping[str, str]]) -> dict[str, Path]:
    paths = {}
    for key, item in bindings.items():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"input SHA-256 drift: {key}")
        paths[key] = path
    return paths


def exact_rows(
    rows: Iterable[Mapping[str, str]], *, metric_prefix: str = "", dataset_filter: bool = False,
) -> dict[str, dict[str, Any]]:
    """Keep attempted opening even when historical scoring failed."""
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        if dataset_filter and row.get("dataset", "").upper() != "TAU":
            continue
        speaker = canonical_speaker(row["speaker_id"])
        status = row.get("scoring_status")
        if status not in {"ok", "error"}:
            raise ValueError("historical exact row has no attempted scoring status")
        fields = [metric_prefix + name for name in COMPONENTS]
        if any(key not in row for key in fields):
            raise ValueError("historical exact schema misses one of six components")
        complete = status == "ok" and all(
            row[key].strip() and math.isfinite(float(row[key])) for key in fields
        )
        evidence = result.setdefault(speaker, {
            "row_numbers": [], "status_counts": {}, "successful_six_component_rows": 0,
        })
        evidence["row_numbers"].append(row_number)
        evidence["status_counts"][status] = evidence["status_counts"].get(status, 0) + 1
        evidence["successful_six_component_rows"] += int(complete)
    return result


def prove_current_universe_opened(
    speakers: set[str], evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, list[str]]:
    combined = set(evidence["expanded_labels"]) | set(evidence["external_exact"]) | set(evidence["sd13_exact"])
    if not speakers <= combined:
        raise ValueError("independent historical exact evidence does not cover current TAU")
    paired = evidence["tau_paired_scores"]
    if set(paired) != speakers or any(row["successful_six_component_rows"] < 1 for row in paired.values()):
        raise ValueError("paired-score opening proof does not cover every current TAU speaker")
    return {
        speaker: sorted(name for name, rows in evidence.items() if speaker in rows)
        for speaker in sorted(speakers)
    }


def audit_sources(
    manifest: list[dict[str, str]], inventory: list[dict[str, str]], roots: Mapping[str, str],
) -> list[dict[str, Any]]:
    indexed = {str(Path(row["audio_path"]).resolve()): row for row in inventory if row["dataset"] == "TAU"}
    if len(indexed) != 2 * len(manifest):
        raise ValueError("TAU source inventory is not one CS/SV pair per manifest speaker")
    directory_speakers = set()
    for source, root_text in roots.items():
        root = Path(root_text).resolve()
        directory_speakers.update((source, canonical_speaker(p.name)) for p in root.iterdir() if p.is_dir())
    manifest_speakers = {(row["source"], canonical_speaker(row["speaker_id"])) for row in manifest}
    if directory_speakers != manifest_speakers or len(manifest_speakers) != len(manifest):
        raise ValueError("active TAU directories and full manifest identities differ")
    if len({speaker for _, speaker in manifest_speakers}) != len(manifest):
        raise ValueError("TAU canonical identity collides across collections")
    output = []
    for row in sorted(manifest, key=lambda value: canonical_speaker(value["speaker_id"])):
        canonical = canonical_speaker(row["speaker_id"])
        speaker = canonical.removeprefix("TAU:")
        source = row["source"]
        if row["label"] not in {"patient", "healthy"}:
            raise ValueError("TAU health label is unavailable")
        sex = row["sex"].strip().lower() or "unknown"
        if sex not in {"female", "male", "unknown"}:
            raise ValueError("TAU sex metadata is unsupported")
        sources = {}
        for view, task in (("cs", "reading"), ("sv", "sustained_vowel")):
            path = Path(row[f"{view}_audio_path"]).resolve()
            if not path.is_relative_to(Path(roots[source]).resolve()):
                raise ValueError("TAU source leaves its active collection")
            if canonical_speaker(path.parent.name) != canonical or unicodedata.normalize("NFC", path.stem) != f"{speaker}_{view}":
                raise ValueError("TAU source is not the stated same-speaker view")
            meta = indexed[str(path)]
            if (canonical_speaker(meta["speaker_id"]) != canonical or meta["task"] != task
                    or meta["label"] != row["label"] or meta["source"] != source
                    or (meta["sex"].strip().lower() or "unknown") != sex):
                raise ValueError("TAU source metadata disagrees across manifests")
            info = sf.info(path)
            if info.channels != int(meta["channels"]) or info.samplerate != int(meta["sample_rate"]):
                raise ValueError("TAU live audio header disagrees with inventory")
            if not math.isclose(info.duration, float(meta["duration_sec"]), abs_tol=1e-5):
                raise ValueError("TAU live duration differs from inventory")
            sources[view] = {
                "path": str(path), "sha256": sha256_file(path),
                "sample_rate": info.samplerate, "frames": info.frames, "channels": info.channels,
                "duration_seconds": info.duration,
                "mono_duration_eligible": info.channels == 1 and info.duration >= MINIMUM_SECONDS[view],
            }
        output.append({
            "canonical_speaker_id": canonical, "speaker_id": speaker, "dataset": "TAU",
            "source": source, "label": row["label"], "sex": sex, "sources": sources,
            "same_speaker_cs_sv_verified": True,
            "source_metadata_eligible": sex != "unknown" and all(v["mono_duration_eligible"] for v in sources.values()),
        })
    return output


def history_paths(roots: list[Path], exclude_root: Path | None = None) -> list[Path]:
    args = ["rg", "--files", "--hidden", "--no-ignore", "-g", "*.json", "-g", "*.csv", "-g", "*.jsonl", "-g", "*.log"]
    result = subprocess.run(args + [str(root) for root in roots], check=True, capture_output=True, text=True)
    paths = {Path(line).resolve() for line in result.stdout.splitlines()}
    if exclude_root is not None:
        paths = {path for path in paths if not path.is_relative_to(exclude_root.resolve())}
    return sorted(paths)


def scan_history(
    roots: list[Path], exclude_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Inventory all metadata; text mentions supplement but never prove opening."""
    paths = history_paths(roots, exclude_root)
    inventory = []
    exact_evidence: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        before = path.stat()
        payload = path.read_bytes()
        content = unicodedata.normalize("NFC", payload.decode("utf-8-sig"))
        references = sorted({canonical_speaker(m.group()) for m in REFERENCE_PATTERN.finditer(content)})
        digest = hashlib.sha256(payload).hexdigest()
        record = {"path": str(path), "sha256": digest, "bytes": len(payload), "tau_identity_references": references}
        inventory.append(record)
        if path.suffix == ".csv" and references:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if {"speaker_id", "scoring_status"} <= fields and set(COMPONENTS) <= fields:
                    for row_number, row in enumerate(reader, start=2):
                        identity = unicodedata.normalize("NFC", row["speaker_id"].strip())
                        if identity.startswith("TAU:"):
                            identity = identity.removeprefix("TAU:")
                        if SPEAKER_PATTERN.fullmatch(identity) is None:
                            continue
                        if row.get("dataset", "TAU").upper() != "TAU":
                            continue
                        if row["scoring_status"] not in {"ok", "error"}:
                            continue
                        canonical = canonical_speaker(identity)
                        exact_evidence.setdefault(canonical, []).append({
                            "path": str(path), "sha256": digest, "row_number": row_number,
                            "scoring_status": row["scoring_status"],
                        })
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"history changed during audit: {path}")
    if paths != history_paths(roots, exclude_root):
        raise ValueError("history file inventory changed during audit")
    return inventory, exact_evidence


def capacity(sources: list[dict[str, Any]], excluded: set[str]) -> dict[str, Any]:
    remaining = [row for row in sources if row["canonical_speaker_id"] not in excluded]
    eligible = [row for row in remaining if row["source_metadata_eligible"]]
    counts = Counter(f"{row['label']}/{row['sex']}" for row in eligible)
    combined = Counter(FUSION_QUOTAS) + Counter(JOINT_QUOTAS)
    return {
        "current_speakers": len(sources), "historically_opened_current_speakers": len(sources) - len(remaining),
        "remaining_unopened_speakers": [row["canonical_speaker_id"] for row in remaining],
        "remaining_metadata_eligible_speakers": [row["canonical_speaker_id"] for row in eligible],
        "current_health_sex_counts": dict(Counter(f"{row['label']}/{row['sex']}" for row in sources)),
        "remaining_eligible_health_sex_counts": dict(counts),
        "six_gradient_capacity_pass": all(counts[k] >= v for k, v in FUSION_QUOTAS.items()),
        "joint_capacity_pass": all(counts[k] >= v for k, v in JOINT_QUOTAS.items()),
        "disjoint_gradient_and_joint_capacity_pass": all(counts[k] >= v for k, v in combined.items()),
    }


def repository_source(root: Path, commit: str) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()

    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("TAU audit requires the bound clean source commit")
    return {"root": str(root.resolve()), "commit": commit, "tree": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output_dir}")
    if sha256_file(args.contract) != args.contract_sha256:
        raise ValueError("TAU audit contract SHA-256 differs")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run the source/header/history audit on a Slurm compute node")
    contract = read_json(args.contract)
    validate_contract(contract)
    source = repository_source(args.source_root, args.source_commit)
    inputs = verify_bindings(contract["inputs"])
    expanded_receipt = read_json(inputs["expanded_label_receipt"])
    external_receipt = read_json(inputs["external_exact_receipt"])
    for name, receipt in (("expanded_labels", expanded_receipt), ("external_exact", external_receipt)):
        if receipt["output_csv_sha256"] != contract["inputs"][name]["sha256"]:
            raise ValueError(f"historical receipt does not bind exact CSV: {name}")
    readiness = read_json(inputs["readiness_report"])
    readiness_receipt = read_json(inputs["readiness_receipt"])
    if (readiness["all_six_scientific_components_ready"] is not True
            or readiness["joint_panel_authorized"] is not False
            or readiness["generator_optimizer_steps"] != 0
            or readiness_receipt["artifact_sha256"][inputs["readiness_report"].name] != contract["inputs"]["readiness_report"]["sha256"]):
        raise ValueError("historical v4 readiness boundary or receipt differs")
    evidence = {
        "expanded_labels": exact_rows(read_csv(inputs["expanded_labels"])),
        "external_exact": exact_rows(read_csv(inputs["external_exact"]), metric_prefix="audio_"),
        "sd13_exact": exact_rows(read_csv(inputs["sd13_exact"])),
        "tau_paired_scores": exact_rows(read_csv(inputs["tau_paired_scores"]), dataset_filter=True),
    }
    sources = audit_sources(read_csv(inputs["tau_manifest"]), read_csv(inputs["tau_inventory"]), contract["active_source_roots"])
    current = {row["canonical_speaker_id"] for row in sources}
    proof = prove_current_universe_opened(current, evidence)
    scan_roots = [Path(path) for path in contract["history_roots"]]
    history, discovered_exact = scan_history(scan_roots, args.output_dir.parent)
    prior = read_json(inputs["prior_ledger"])
    prior_tau = {
        canonical_speaker(row["speaker_id"])
        for row in prior["entries"] if row["dataset"].upper() == "TAU"
    }
    excluded = current | prior_tau | set(discovered_exact)
    source_by_speaker = {row["canonical_speaker_id"]: row for row in sources}
    ledger = {
        "schema_version": "avqi-route-c-prior-panel-speaker-ledger-v1",
        "scope": "complete current TAU source universe plus historical TAU speakers observed in the inventoried roots",
        "current_typed_exact_evidence_coverage_complete": True,
        "historical_json_mentions_are_reference_only": True,
        "exact_outcomes_used_for_selection": False,
        "source_commit": args.source_commit,
        "entries": [{
            "dataset": "TAU", "speaker_id": speaker.removeprefix("TAU:"), "canonical_speaker_id": speaker,
            "current_source": source_by_speaker.get(speaker),
            "current_exact_proof_input_keys": proof.get(speaker, []),
            "prior_ledger_present": speaker in prior_tau,
            "typed_historical_exact_rows": discovered_exact.get(speaker, []),
            "role": "historical_opened_or_prior_excluded_diagnostic_only",
        } for speaker in sorted(excluded)],
        "generator_optimizer_steps": 0, "authoritative_training_decision": TRAINING_NO_GO,
    }
    result = capacity(sources, excluded)
    if result["remaining_unopened_speakers"]:
        raise ValueError("pinned TAU saturation proof unexpectedly leaves unopened speakers")
    verify_bindings(contract["inputs"])
    repository_source(args.source_root, args.source_commit)
    report = {
        "schema_version": SCHEMA, "decision": FAILURE, "scientific_failure_preserved": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": source, "contract_sha256": args.contract_sha256,
        "input_bindings": contract["inputs"], "capacity": result,
        "historical_exact_evidence": evidence, "history_roots": [str(root) for root in scan_roots],
        "history_file_count": len(history), "historical_tau_exclusion_count": len(excluded),
        "previous_ledger_tau_count": len(prior_tau), "current_speakers_missing_from_previous_ledger": sorted(current - prior_tau),
        "source_snapshot": sources,
        "scope_limit": "current authoritative Elina/Nelly source; origin variants remain provenance only",
        "known_metadata_limit": "Nelly sex is unknown and must not be inferred from identities or audio",
        "component_readiness": "all_six_ready_from_immutable_v4",
        "joint_execution_inputs_bound": False, "thresholds_changed": False, **BOUNDARIES,
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "tau_history_capacity_report_v1.json", report)
    write_json(args.output_dir / "complete_current_tau_speaker_ledger_v1.json", ledger)
    write_json(args.output_dir / "historical_artifact_inventory_v1.json", {"roots": contract["history_roots"], "files": history})
    artifacts = {path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir())}
    receipt = {
        "schema_version": SCHEMA + "-receipt", "decision": FAILURE,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "source": source,
        "contract_sha256": args.contract_sha256, "input_bindings": contract["inputs"],
        "artifact_sha256": artifacts, "current_typed_exact_evidence_coverage_complete": True,
        "current_tau_speakers": len(current), "historical_tau_exclusion_count": len(excluded),
        "remaining_unopened_current_tau_speakers": 0,
        "scientific_failure_preserved": True, "thresholds_changed": False, **BOUNDARIES,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps({"decision": FAILURE, "current_speakers": len(current), "remaining_unopened": 0, "slurm_job_id": os.environ["SLURM_JOB_ID"]}))


if __name__ == "__main__":
    main()
