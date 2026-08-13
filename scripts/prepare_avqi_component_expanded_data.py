#!/usr/bin/env python3
"""Prepare a bounded TAU CS/SV expansion for AVQI-component prediction.

The expansion is deliberately training-only. It selects every speaker from the
full TAU sampling manifest that is absent from the locked 123-speaker panel,
then creates one deterministic 16 kHz phone-room simulation per CS/SV source.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import yaml


TARGET_SAMPLE_RATE = 16_000
OUTPUT_SUBTYPE = "FLOAT"
TASKS = ("cs", "sv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--full-manifest-sha256", required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest-sha256", required=True)
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--simulation-config-sha256", required=True)
    parser.add_argument("--noise-manifest", type=Path, required=True)
    parser.add_argument("--noise-manifest-sha256", required=True)
    parser.add_argument("--rir-manifest", type=Path, required=True)
    parser.add_argument("--rir-manifest-sha256", required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-new-speakers", type=int, default=55)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {path}: {actual} != {expected}")
    return actual


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def read_clean(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio[:, 0]
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = librosa.resample(
            mono,
            orig_sr=int(sample_rate),
            target_sr=TARGET_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"invalid clean audio: {path}")
    return mono[np.newaxis, :]


def load_wds_rows(manifest: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("status") != "done":
                continue
            if not str(row.get("audio_member", "")).endswith(".wav"):
                continue
            row["_root"] = str(root)
            row["_manifest_index"] = manifest_index
            rows.append(row)
    if not rows:
        raise ValueError(f"no usable WDS rows in {manifest}")
    return rows


class WdsReader:
    def __init__(self) -> None:
        self._handles: dict[tuple[str, str], tarfile.TarFile] = {}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def read(self, row: dict[str, Any]) -> np.ndarray:
        key = (str(row["_root"]), str(row["shard"]))
        handle = self._handles.get(key)
        if handle is None:
            handle = tarfile.open(Path(key[0]) / key[1])
            self._handles[key] = handle
        file_object = handle.extractfile(str(row["audio_member"]))
        if file_object is None:
            raise FileNotFoundError(f"missing tar member: {key}/{row['audio_member']}")
        audio, sample_rate = sf.read(
            io.BytesIO(file_object.read()),
            dtype="float32",
            always_2d=True,
        )
        mono = audio[:, 0]
        if sample_rate != TARGET_SAMPLE_RATE:
            mono = librosa.resample(
                mono,
                orig_sr=int(sample_rate),
                target_sr=TARGET_SAMPLE_RATE,
                res_type="soxr_hq",
            ).astype(np.float32, copy=False)
        if mono.size == 0 or not np.isfinite(mono).all():
            raise ValueError(f"invalid WDS audio: {key}/{row['audio_member']}")
        return mono[np.newaxis, :]


def crop_or_tile(audio: np.ndarray, length: int, rng: random.Random) -> tuple[np.ndarray, int]:
    if audio.shape[1] >= length:
        maximum_start = audio.shape[1] - length
        start = rng.randrange(maximum_start + 1) if maximum_start else 0
        return audio[:, start : start + length], start
    repeats = math.ceil(length / audio.shape[1])
    return np.tile(audio, (1, repeats))[:, :length], 0


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[1] >= length:
        return audio[:, :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[1])))


def sample_group(row: dict[str, str]) -> str:
    if row["label"] == "healthy":
        return "healthy_unselected"
    if row["label"] == "patient":
        return "pathological_unselected"
    raise ValueError(f"unsupported label: {row['label']}")


def main() -> None:
    args = parse_args()
    # The project simulator imports PyTorch. Load it only for real generation so
    # schema/help checks remain lightweight and do not initialize an ML runtime.
    from simulate_degradation import (
        apply_degradation_with_wind,
        random_select_and_order,
    )

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.expected_new_speakers <= 0:
        raise ValueError("expected-new-speakers must be positive")
    source_hashes = {
        "full_manifest": validate_hash(
            args.full_manifest,
            args.full_manifest_sha256,
        ),
        "selected_manifest": validate_hash(
            args.selected_manifest,
            args.selected_manifest_sha256,
        ),
        "simulation_config": validate_hash(
            args.simulation_config,
            args.simulation_config_sha256,
        ),
        "noise_manifest": validate_hash(
            args.noise_manifest,
            args.noise_manifest_sha256,
        ),
        "rir_manifest": validate_hash(
            args.rir_manifest,
            args.rir_manifest_sha256,
        ),
    }
    full_rows = read_csv(args.full_manifest)
    selected_rows = read_csv(args.selected_manifest)
    selected_speakers = {row["speaker_id"] for row in selected_rows}
    expansion_rows = [
        row for row in full_rows if row["speaker_id"] not in selected_speakers
    ]
    expansion_speakers = {row["speaker_id"] for row in expansion_rows}
    if len(expansion_rows) != len(expansion_speakers):
        raise ValueError("full manifest contains duplicate expansion speakers")
    if len(expansion_speakers) != args.expected_new_speakers:
        raise ValueError(
            f"expected {args.expected_new_speakers} new speakers, "
            f"found {len(expansion_speakers)}"
        )
    if expansion_speakers & selected_speakers:
        raise ValueError("expansion speakers overlap the locked selected panel")
    for row in expansion_rows:
        for key in ("cs_audio_path", "sv_audio_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])

    config = yaml.safe_load(args.simulation_config.read_text(encoding="utf-8"))
    config["stft_cfg"]["sampling_rate"] = TARGET_SAMPLE_RATE
    noise_rows = load_wds_rows(args.noise_manifest, args.noise_root)
    rir_rows = load_wds_rows(args.rir_manifest, args.rir_root)

    args.output_dir.mkdir(parents=True)
    clean_dir = args.output_dir / "clean"
    noisy_dir = args.output_dir / "noisy"
    rir_dir = args.output_dir / "rir"
    clean_dir.mkdir()
    noisy_dir.mkdir()
    rir_dir.mkdir()

    metadata: list[dict[str, Any]] = []
    reader = WdsReader()
    try:
        for speaker_index, row in enumerate(
            sorted(expansion_rows, key=lambda item: item["speaker_id"])
        ):
            for task_index, task in enumerate(TASKS):
                source_path = Path(row[f"{task}_audio_path"])
                clean = read_clean(source_path)
                seed = stable_seed(args.seed, "avqi_component_expand_v1", row["speaker_id"], task)
                rng = random.Random(seed)
                noise_row = noise_rows[rng.randrange(len(noise_rows))]
                rir_row = rir_rows[rng.randrange(len(rir_rows))]
                noise, noise_start = crop_or_tile(reader.read(noise_row), clean.shape[1], rng)
                rir = reader.read(rir_row)
                item_config = copy.deepcopy(config)
                degradation_config, selected_degradations = random_select_and_order(
                    item_config,
                    seed=seed,
                )
                if tuple(selected_degradations) != ("reverb", "noise"):
                    raise ValueError(
                        f"unexpected degradations for {row['speaker_id']}/{task}: "
                        f"{selected_degradations}"
                    )
                clean_output, noisy = apply_degradation_with_wind(
                    item_config,
                    clean,
                    noise,
                    rir,
                    None,
                    degradation_config,
                    selected_degradations,
                    seed=seed,
                )
                clean_output = match_length(clean_output, clean.shape[1]).astype(np.float32)
                noisy = match_length(noisy, clean.shape[1]).astype(np.float32)
                item_index = speaker_index * len(TASKS) + task_index
                uid = (
                    f"tau_avqi_expand_v1_{item_index:05d}_"
                    f"{row['speaker_id']}_{task}"
                )
                clean_path = clean_dir / f"{uid}_clean.wav"
                noisy_path = noisy_dir / f"{uid}_noisy.wav"
                rir_path = rir_dir / f"{uid}_rir.wav"
                sf.write(clean_path, clean_output[0], TARGET_SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
                sf.write(noisy_path, noisy[0], TARGET_SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
                sf.write(rir_path, rir[0], TARGET_SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
                metadata.append(
                    {
                        "uid": uid,
                        "speaker_id": row["speaker_id"],
                        "pair_id": row["pair_id"],
                        "task": task,
                        "sample_group": sample_group(row),
                        "label": row["label"],
                        "source": row["source"],
                        "sex": row.get("sex", ""),
                        "age": row.get("age", ""),
                        "language": row.get("language", ""),
                        "source_clean_path": str(source_path.resolve()),
                        "clean_filepath": str(clean_path.resolve()),
                        "noisy_filepath": str(noisy_path.resolve()),
                        "rir_filepath": str(rir_path.resolve()),
                        "target_sample_rate": TARGET_SAMPLE_RATE,
                        "base_seed": args.seed,
                        "seed": seed,
                        "selected_degradations": selected_degradations,
                        "degradation_config": degradation_config,
                        "noise_manifest_index": noise_row["_manifest_index"],
                        "noise_shard": noise_row["shard"],
                        "noise_audio_member": noise_row["audio_member"],
                        "noise_source_path": noise_row.get("source_path", ""),
                        "noise_start_sample": noise_start,
                        "rir_manifest_index": rir_row["_manifest_index"],
                        "rir_shard": rir_row["shard"],
                        "rir_audio_member": rir_row["audio_member"],
                        "rir_source_path": rir_row.get("source_path", ""),
                    }
                )
                print(
                    f"prepared={len(metadata)}/{len(expansion_rows) * len(TASKS)} "
                    f"speaker={row['speaker_id']} task={task}",
                    flush=True,
                )
    finally:
        reader.close()

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pair_rows = [
        {
            "uid": row["uid"],
            "speaker_id": row["speaker_id"],
            "task": row["task"],
            "clean_filepath": row["clean_filepath"],
            "noisy_filepath": row["noisy_filepath"],
            "sample_rate": TARGET_SAMPLE_RATE,
        }
        for row in metadata
    ]
    pairs_path = args.output_dir / "pairs.csv"
    write_csv(
        pairs_path,
        pair_rows,
        [
            "uid",
            "speaker_id",
            "task",
            "clean_filepath",
            "noisy_filepath",
            "sample_rate",
        ],
    )
    summary = {
        "decision": "PASS_AVQI_COMPONENT_EXPANSION_DATA_V1",
        "speaker_count": len(expansion_speakers),
        "task_count": len(metadata),
        "condition_count": 2,
        "expected_component_rows": len(expansion_speakers) * 2 * 3,
        "labels": dict(Counter(row["label"] for row in expansion_rows)),
        "sources": dict(Counter(row["source"] for row in expansion_rows)),
        "selected_panel_overlap": [],
        "conditions": ["clean", "aug16k_phone"],
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "seed": args.seed,
        "metadata_json": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "pairs_csv": str(pairs_path.resolve()),
        "pairs_sha256": sha256_file(pairs_path),
        "source_sha256": source_hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
