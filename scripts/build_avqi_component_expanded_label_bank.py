#!/usr/bin/env python3
"""Score expanded TAU waveforms and append them to a locked AVQI label bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import parselmouth
import soundfile as sf

from avqi_code import run_avqi


CONDITIONS = ("clean", "aug16k_phone")
VIEWS = ("both", "cs", "sv")
METRICS = (
    "avqi",
    "cpps",
    "hnr",
    "jitter_local",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
STEP_VERSIONS = {
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


@dataclass(frozen=True)
class ScoringTask:
    speaker_id: str
    condition: str
    view: str
    cs_path: str
    sv_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-label-bank", type=Path, required=True)
    parser.add_argument("--base-label-bank-sha256", required=True)
    parser.add_argument("--expansion-metadata", type=Path, required=True)
    parser.add_argument("--expansion-metadata-sha256", required=True)
    parser.add_argument("--external-exact-csv", type=Path, required=True)
    parser.add_argument("--external-exact-csv-sha256", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-base-speakers", type=int, default=98)
    parser.add_argument("--expected-expansion-speakers", type=int, default=55)
    parser.add_argument("--expected-train-speakers", type=int, default=125)
    parser.add_argument("--expected-calibration-speakers", type=int, default=14)
    parser.add_argument("--expected-holdout-speakers", type=int, default=14)
    parser.add_argument("--exact-runner", type=Path, required=True)
    parser.add_argument("--expected-exact-runner-sha256", required=True)
    parser.add_argument("--avqi-main", type=Path, required=True)
    parser.add_argument("--expected-avqi-main-sha256", required=True)
    parser.add_argument("--avqi-praat", type=Path, required=True)
    parser.add_argument("--expected-avqi-praat-sha256", required=True)
    parser.add_argument("--avqi-praat-script", type=Path, required=True)
    parser.add_argument("--expected-avqi-praat-script-sha256", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows, list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: expected {expected}, got {actual}")
    return actual


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    if info.samplerate != 16_000 or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"expected non-empty 16 kHz mono audio: {path}: {info}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
    }


def group_expansion_metadata(
    path: Path,
    expected_speakers: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("expansion metadata must be a list")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in payload:
        speaker = str(row["speaker_id"])
        task = str(row["task"])
        if task not in {"cs", "sv"}:
            raise ValueError(f"unexpected task: {speaker}/{task}")
        speaker_tasks = grouped.setdefault(speaker, {})
        if task in speaker_tasks:
            raise ValueError(f"duplicate expansion task: {speaker}/{task}")
        speaker_tasks[task] = row
    incomplete = [speaker for speaker, tasks in grouped.items() if set(tasks) != {"cs", "sv"}]
    if len(grouped) != expected_speakers or incomplete:
        raise ValueError(
            f"invalid expansion pairs: speakers={len(grouped)}, incomplete={incomplete[:10]}"
        )
    return grouped


def score_task(task: ScoringTask) -> dict[str, Any]:
    row: dict[str, Any] = asdict(task)
    if task.view == "both":
        sv_path = task.sv_path
        cs_path = task.cs_path
        speaking_type = "both"
    elif task.view == "cs":
        sv_path = task.cs_path
        cs_path = task.cs_path
        speaking_type = "cs"
    elif task.view == "sv":
        sv_path = task.sv_path
        cs_path = task.sv_path
        speaking_type = "sv"
    else:
        raise ValueError(f"unsupported view: {task.view}")
    try:
        metrics = run_avqi(
            sv_path,
            cs_path,
            target_sr=16_000,
            speaking_type=speaking_type,
            step_versions=STEP_VERSIONS,
            remove_sv_silence_with_sox=False,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        subprocess.CalledProcessError,
        parselmouth.PraatError,
    ) as error:
        row.update(
            {
                "scoring_status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        )
        for metric in METRICS:
            row[metric] = ""
        return row
    row.update(
        {
            "scoring_status": "ok",
            "error_type": "",
            "error_message": "",
        }
    )
    for metric in METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric}: {task}")
        row[metric] = value
    return row


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.output_csv.exists() or args.output_csv.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_csv}")
    source_hashes = {
        "base_label_bank": validate_hash(
            args.base_label_bank,
            args.base_label_bank_sha256,
            "base label bank",
        ),
        "expansion_metadata": validate_hash(
            args.expansion_metadata,
            args.expansion_metadata_sha256,
            "expansion metadata",
        ),
        "external_exact_csv": validate_hash(
            args.external_exact_csv,
            args.external_exact_csv_sha256,
            "external exact CSV",
        ),
        "exact_runner": validate_hash(
            args.exact_runner,
            args.expected_exact_runner_sha256,
            "exact runner",
        ),
        "avqi_main": validate_hash(
            args.avqi_main,
            args.expected_avqi_main_sha256,
            "AVQI main",
        ),
        "avqi_praat": validate_hash(
            args.avqi_praat,
            args.expected_avqi_praat_sha256,
            "AVQI Praat bridge",
        ),
        "avqi_praat_script": validate_hash(
            args.avqi_praat_script,
            args.expected_avqi_praat_script_sha256,
            "AVQI Praat script",
        ),
    }
    base_rows, fieldnames = read_csv(args.base_label_bank)
    required_fields = {
        "schema_version",
        "speaker_id",
        "split",
        "condition_id",
        "view",
        "scoring_status",
        *METRICS,
    }
    missing_fields = required_fields - set(fieldnames)
    if missing_fields:
        raise ValueError(f"base label bank is missing fields: {sorted(missing_fields)}")
    base_speakers = {row["speaker_id"] for row in base_rows}
    if len(base_speakers) != args.expected_base_speakers:
        raise ValueError(
            f"expected {args.expected_base_speakers} base speakers, found {len(base_speakers)}"
        )
    expansion = group_expansion_metadata(
        args.expansion_metadata,
        args.expected_expansion_speakers,
    )
    expansion_speakers = set(expansion)
    external_rows, _ = read_csv(args.external_exact_csv)
    external_speakers = {row["speaker_id"] for row in external_rows}
    if expansion_speakers & base_speakers:
        raise ValueError("expansion speakers overlap the base label bank")
    if expansion_speakers & external_speakers:
        raise ValueError("expansion speakers overlap the external test panel")

    metadata_by_path: dict[str, dict[str, Any]] = {}
    tasks: list[ScoringTask] = []
    for speaker in sorted(expansion):
        cs_metadata = expansion[speaker]["cs"]
        sv_metadata = expansion[speaker]["sv"]
        for condition in CONDITIONS:
            source_key = "clean_filepath" if condition == "clean" else "noisy_filepath"
            cs_path = Path(str(cs_metadata[source_key]))
            sv_path = Path(str(sv_metadata[source_key]))
            for path in (cs_path, sv_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                resolved = str(path.resolve())
                if resolved not in metadata_by_path:
                    metadata_by_path[resolved] = audio_metadata(path)
            for view in VIEWS:
                tasks.append(
                    ScoringTask(
                        speaker_id=speaker,
                        condition=condition,
                        view=view,
                        cs_path=str(cs_path.resolve()),
                        sv_path=str(sv_path.resolve()),
                    )
                )
    expected_scored_rows = args.expected_expansion_speakers * len(CONDITIONS) * len(VIEWS)
    if len(tasks) != expected_scored_rows:
        raise ValueError(f"expected {expected_scored_rows} scoring tasks, found {len(tasks)}")
    if args.workers == 1:
        scored_rows = [score_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            scored_rows = list(executor.map(score_task, tasks))

    expansion_rows: list[dict[str, Any]] = []
    for scored in scored_rows:
        speaker = str(scored["speaker_id"])
        cs_metadata = expansion[speaker]["cs"]
        sv_metadata = expansion[speaker]["sv"]
        cs_audio = metadata_by_path[str(Path(str(scored["cs_path"])).resolve())]
        sv_audio = metadata_by_path[str(Path(str(scored["sv_path"])).resolve())]
        same_nuisance = (
            cs_metadata.get("noise_shard") == sv_metadata.get("noise_shard")
            and cs_metadata.get("noise_audio_member") == sv_metadata.get("noise_audio_member")
            and cs_metadata.get("noise_start_sample") == sv_metadata.get("noise_start_sample")
            and cs_metadata.get("rir_shard") == sv_metadata.get("rir_shard")
            and cs_metadata.get("rir_audio_member") == sv_metadata.get("rir_audio_member")
            and cs_metadata.get("seed") == sv_metadata.get("seed")
        )
        row: dict[str, Any] = {
            "schema_version": "tau_pathology_component_bank_v2_expanded_train",
            "speaker_id": speaker,
            "pair_id": cs_metadata.get("pair_id", ""),
            "condition_id": scored["condition"],
            "view": scored["view"],
            "sample_group": cs_metadata.get("sample_group", ""),
            "label": cs_metadata.get("label", ""),
            "source": cs_metadata.get("source", ""),
            "sex": cs_metadata.get("sex", ""),
            "age": cs_metadata.get("age", ""),
            "language": cs_metadata.get("language", ""),
            "split_version": "tau-avqi-surrogate-split-v2-expanded-train",
            "split": "surrogate_train",
            "cs_uid": cs_metadata["uid"],
            "cs_path": cs_audio["path"],
            "cs_sha256": cs_audio["sha256"],
            "cs_sample_rate": cs_audio["sample_rate"],
            "cs_frames": cs_audio["frames"],
            "sv_uid": sv_metadata["uid"],
            "sv_path": sv_audio["path"],
            "sv_sha256": sv_audio["sha256"],
            "sv_sample_rate": sv_audio["sample_rate"],
            "sv_frames": sv_audio["frames"],
            "cs_native_snr_db": cs_metadata["degradation_config"]["snr"],
            "sv_native_snr_db": sv_metadata["degradation_config"]["snr"],
            "same_noise_rir_seed_across_cs_sv": int(same_nuisance),
            "target_sr": 16_000,
            "speaking_type": scored["view"],
            "all_praat": 1,
            "remove_sv_silence_with_sox": 0,
            "scoring_status": scored["scoring_status"],
            "error_type": scored["error_type"],
            "error_message": scored["error_message"],
        }
        for metric in METRICS:
            row[metric] = scored[metric]
        if set(row) != set(fieldnames):
            raise ValueError(
                "expanded row schema differs from base label bank: "
                f"missing={sorted(set(fieldnames) - set(row))}, "
                f"extra={sorted(set(row) - set(fieldnames))}"
            )
        expansion_rows.append(row)

    combined_rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        updated = dict(base_row)
        updated["schema_version"] = "tau_pathology_component_bank_v2_expanded_train"
        updated["split_version"] = "tau-avqi-surrogate-split-v2-expanded-train"
        combined_rows.append(updated)
    combined_rows.extend(expansion_rows)
    write_csv(args.output_csv, combined_rows, fieldnames)

    task_rows = [row for row in combined_rows if row["view"] in {"cs", "sv"}]
    usable_task_rows = [row for row in task_rows if row["scoring_status"] == "ok"]
    split_speakers = {
        split: len(
            {
                row["speaker_id"]
                for row in usable_task_rows
                if row["split"] == split
            }
        )
        for split in (
            "surrogate_train",
            "surrogate_calibration",
            "surrogate_holdout",
        )
    }
    expected_split_speakers = {
        "surrogate_train": args.expected_train_speakers,
        "surrogate_calibration": args.expected_calibration_speakers,
        "surrogate_holdout": args.expected_holdout_speakers,
    }
    coverage = len(usable_task_rows) / len(task_rows)
    decision = (
        "PASS_AVQI_COMPONENT_EXPANDED_LABEL_BANK_V2"
        if split_speakers == expected_split_speakers and coverage >= 0.95
        else "FAIL_AVQI_COMPONENT_EXPANDED_LABEL_BANK_V2"
    )
    summary = {
        "decision": decision,
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": sha256_file(args.output_csv),
        "row_count": len(combined_rows),
        "base_row_count": len(base_rows),
        "expanded_row_count": len(expansion_rows),
        "speaker_count": len(base_speakers | expansion_speakers),
        "base_speaker_count": len(base_speakers),
        "expanded_speaker_count": len(expansion_speakers),
        "split_speakers": split_speakers,
        "task_row_count": len(task_rows),
        "usable_task_row_count": len(usable_task_rows),
        "task_coverage": coverage,
        "expanded_error_rows": sum(
            row["scoring_status"] != "ok" for row in expansion_rows
        ),
        "expanded_labels": dict(
            Counter(expansion[speaker]["cs"]["label"] for speaker in expansion)
        ),
        "base_overlap": [],
        "external_overlap": [],
        "source_sha256": source_hashes,
        "workers": args.workers,
    }
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if decision.startswith("FAIL"):
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
