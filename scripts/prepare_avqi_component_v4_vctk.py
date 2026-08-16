#!/usr/bin/env python3
"""Prepare a speaker-disjoint VCTK expansion for AVQI-component scoring.

The split is frozen before any simulation.  Each selected utterance yields a
clean, reverberant, 20 dB SNR, and 10 dB SNR waveform.  This script does not
apply the AVQI 34 Hz metric-branch high-pass to generated audio.
"""

from __future__ import annotations

import argparse
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
from scipy.signal import fftconvolve


SAMPLE_RATE = 16_000
OUTPUT_SUBTYPE = "PCM_16"
SPLIT_COUNTS = {
    "surrogate_train": 72,
    "surrogate_calibration": 12,
    "surrogate_holdout": 12,
    "vctk_external": 12,
}
CONDITIONS = ("clean", "rir_only", "snr20", "snr10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vctk-manifest", type=Path, required=True)
    parser.add_argument("--vctk-manifest-sha256", required=True)
    parser.add_argument("--vctk-root", type=Path, required=True)
    parser.add_argument("--noise-manifest", type=Path, required=True)
    parser.add_argument("--noise-manifest-sha256", required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--rir-manifest", type=Path, required=True)
    parser.add_argument("--rir-manifest-sha256", required=True)
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--utterances-per-speaker", type=int, default=4)
    parser.add_argument("--minimum-duration-seconds", type=float, default=3.0)
    parser.add_argument("--maximum-duration-seconds", type=float, default=12.0)
    parser.add_argument("--expected-vctk-items", type=int, default=43_873)
    parser.add_argument("--expected-vctk-speakers", type=int, default=108)
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


def stable_rank(seed: int, *parts: object) -> str:
    value = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    return int(stable_rank(seed, *parts)[:16], 16) % (2**32)


def read_manifest(
    path: Path,
    root: Path,
    allowed_suffixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            member = str(row.get("audio_member", ""))
            if row.get("status") != "done" or not member.endswith(allowed_suffixes):
                continue
            row["_manifest_index"] = index
            row["_root"] = str(row.get("_shard_dir") or root)
            rows.append(row)
    if not rows:
        raise ValueError(f"no usable rows in {path}")
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
        mono = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            mono = librosa.resample(
                mono,
                orig_sr=int(sample_rate),
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            ).astype(np.float32, copy=False)
        if mono.size == 0 or not np.isfinite(mono).all():
            raise ValueError(f"invalid audio: {key}/{row['audio_member']}")
        return mono.astype(np.float32, copy=False)


def speaker_id(row: dict[str, Any]) -> str:
    speaker = Path(str(row["source_path"])).parent.name
    if not speaker or not speaker.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"cannot parse VCTK speaker from {row['source_path']}")
    return speaker


def crop_or_tile(audio: np.ndarray, length: int, rng: random.Random) -> tuple[np.ndarray, int]:
    if audio.size >= length:
        maximum_start = audio.size - length
        start = rng.randrange(maximum_start + 1) if maximum_start else 0
        return audio[start : start + length], start
    repeats = math.ceil(length / audio.size)
    return np.tile(audio, repeats)[:length], 0


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def peak_safe(audio: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio)))
    scale = min(1.0, 0.98 / max(peak, 1e-8))
    return (audio * scale).astype(np.float32), scale


def reverberate(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = rir - float(np.mean(rir[-min(rir.size, 1_000) :]))
    energy = float(np.sqrt(np.sum(np.square(rir, dtype=np.float64))))
    if energy <= 1e-8:
        raise ValueError("RIR has zero energy")
    convolved = fftconvolve(clean, rir / energy, mode="full")[: clean.size]
    clean_rms = rms(clean)
    convolved_rms = rms(convolved)
    if convolved_rms <= 1e-8:
        raise ValueError("reverberated signal has zero energy")
    return (convolved * (clean_rms / convolved_rms)).astype(np.float32)


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        raise ValueError("cannot mix zero-energy signal or noise")
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return (signal + noise * (target_noise_rms / noise_rms)).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty metadata")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.utterances_per_speaker <= 0:
        raise ValueError("utterances per speaker must be positive")
    if not 0.0 < args.minimum_duration_seconds <= args.maximum_duration_seconds:
        raise ValueError("invalid duration interval")
    source_hashes = {
        "vctk_manifest": validate_hash(
            args.vctk_manifest, args.vctk_manifest_sha256
        ),
        "noise_manifest": validate_hash(
            args.noise_manifest, args.noise_manifest_sha256
        ),
        "rir_manifest": validate_hash(args.rir_manifest, args.rir_manifest_sha256),
    }
    vctk_rows = read_manifest(args.vctk_manifest, args.vctk_root, (".flac", ".wav"))
    if len(vctk_rows) != args.expected_vctk_items:
        raise ValueError(
            f"expected {args.expected_vctk_items} VCTK rows, found {len(vctk_rows)}"
        )
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in vctk_rows:
        by_speaker.setdefault(speaker_id(row), []).append(row)
    if len(by_speaker) != args.expected_vctk_speakers:
        raise ValueError(
            f"expected {args.expected_vctk_speakers} speakers, found {len(by_speaker)}"
        )
    if sum(SPLIT_COUNTS.values()) != len(by_speaker):
        raise ValueError("frozen split counts do not cover every VCTK speaker")
    ranked_speakers = sorted(
        by_speaker,
        key=lambda value: stable_rank(args.seed, "speaker_split", value),
    )
    split_by_speaker: dict[str, str] = {}
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        for speaker in ranked_speakers[offset : offset + count]:
            split_by_speaker[speaker] = split
        offset += count

    noise_rows = read_manifest(args.noise_manifest, args.noise_root, (".wav", ".flac"))
    rir_rows = read_manifest(args.rir_manifest, args.rir_root, (".wav", ".flac"))
    args.output_dir.mkdir(parents=True)
    metadata: list[dict[str, Any]] = []
    rejected_durations = Counter()
    reader = WdsReader()
    try:
        for speaker_index, speaker in enumerate(ranked_speakers, start=1):
            candidates = sorted(
                by_speaker[speaker],
                key=lambda row: stable_rank(
                    args.seed,
                    "utterance_rank",
                    speaker,
                    row["key"],
                ),
            )
            selected: list[tuple[dict[str, Any], np.ndarray]] = []
            for candidate in candidates:
                clean = reader.read(candidate)
                duration = clean.size / SAMPLE_RATE
                if duration < args.minimum_duration_seconds:
                    rejected_durations["too_short"] += 1
                    continue
                if duration > args.maximum_duration_seconds:
                    rejected_durations["too_long"] += 1
                    continue
                selected.append((candidate, clean))
                if len(selected) == args.utterances_per_speaker:
                    break
            if len(selected) != args.utterances_per_speaker:
                raise ValueError(
                    f"speaker {speaker} has only {len(selected)} eligible utterances"
                )
            split = split_by_speaker[speaker]
            for candidate, clean in selected:
                sample_id = str(candidate["key"])
                seed = stable_seed(args.seed, "simulation", speaker, sample_id)
                rng = random.Random(seed)
                noise_row = noise_rows[rng.randrange(len(noise_rows))]
                rir_row = rir_rows[rng.randrange(len(rir_rows))]
                noise, noise_start = crop_or_tile(reader.read(noise_row), clean.size, rng)
                rir = reader.read(rir_row)
                reverberant = reverberate(clean, rir)
                variants = {
                    "clean": clean,
                    "rir_only": reverberant,
                    "snr20": add_noise(reverberant, noise, 20.0),
                    "snr10": add_noise(reverberant, noise, 10.0),
                }
                for condition in CONDITIONS:
                    audio, peak_scale = peak_safe(variants[condition])
                    output_dir = args.output_dir / "audio" / split / condition
                    output_dir.mkdir(parents=True, exist_ok=True)
                    path = output_dir / f"{sample_id}__{condition}.wav"
                    sf.write(path, audio, SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
                    metadata.append(
                        {
                            "schema_version": "avqi-component-vctk-v4",
                            "speaker_id": speaker,
                            "sample_id": sample_id,
                            "split": split,
                            "condition_id": condition,
                            "view": "cs",
                            "label": "healthy",
                            "sample_group": "healthy_vctk",
                            "source": "VCTK",
                            "audio_path": str(path.resolve()),
                            "audio_sha256": sha256_file(path),
                            "sample_rate": SAMPLE_RATE,
                            "frames": audio.size,
                            "duration_seconds": audio.size / SAMPLE_RATE,
                            "peak_scale": peak_scale,
                            "simulation_seed": seed,
                            "vctk_manifest_index": candidate["_manifest_index"],
                            "vctk_shard": candidate["shard"],
                            "vctk_audio_member": candidate["audio_member"],
                            "noise_manifest_index": noise_row["_manifest_index"],
                            "noise_shard": noise_row["shard"],
                            "noise_audio_member": noise_row["audio_member"],
                            "noise_start_sample": noise_start,
                            "rir_manifest_index": rir_row["_manifest_index"],
                            "rir_shard": rir_row["shard"],
                            "rir_audio_member": rir_row["audio_member"],
                            "metric_branch_highpass_applied": 0,
                        }
                    )
            print(
                f"prepared_speakers={speaker_index}/{len(ranked_speakers)} "
                f"rows={len(metadata)}",
                flush=True,
            )
    finally:
        reader.close()

    expected_rows = len(by_speaker) * args.utterances_per_speaker * len(CONDITIONS)
    if len(metadata) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, generated {len(metadata)}")
    write_csv(args.output_dir / "metadata.csv", metadata)
    split_receipt = {
        split: sorted(
            speaker for speaker, speaker_split in split_by_speaker.items()
            if speaker_split == split
        )
        for split in SPLIT_COUNTS
    }
    receipt = {
        "schema_version": "avqi-component-vctk-v4",
        "seed": args.seed,
        "source_hashes": source_hashes,
        "speaker_splits": split_receipt,
        "speaker_counts": {key: len(value) for key, value in split_receipt.items()},
        "speaker_overlap": 0,
        "utterances_per_speaker": args.utterances_per_speaker,
        "conditions": list(CONDITIONS),
        "row_count": len(metadata),
        "row_counts_by_split": dict(Counter(row["split"] for row in metadata)),
        "row_counts_by_condition": dict(
            Counter(row["condition_id"] for row in metadata)
        ),
        "rejected_duration_candidates": dict(rejected_durations),
        "full_band_audio_preserved": True,
        "avqi_metric_branch_highpass_applied": False,
        "metadata_csv": str((args.output_dir / "metadata.csv").resolve()),
        "metadata_sha256": sha256_file(args.output_dir / "metadata.csv"),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
