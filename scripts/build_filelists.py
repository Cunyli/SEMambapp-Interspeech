#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def collect_audio_files(root: Path):
    exts = {".wav", ".flac"}
    return sorted(
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    )


def write_json(paths, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build JSON file lists for SEMamba++ training/validation."
    )
    parser.add_argument("--train-speech", type=Path, required=True)
    parser.add_argument("--train-noise", type=Path, required=True)
    parser.add_argument("--train-rir", type=Path, required=True)
    parser.add_argument("--val-clean", type=Path, required=True)
    parser.add_argument("--val-degraded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    datasets = {
        "train_speech.json": collect_audio_files(args.train_speech),
        "train_noise.json": collect_audio_files(args.train_noise),
        "train_rir.json": collect_audio_files(args.train_rir),
        "val_clean.json": collect_audio_files(args.val_clean),
        "val_degraded.json": collect_audio_files(args.val_degraded),
    }

    for name, items in datasets.items():
        target = args.output_dir / name
        write_json(items, target)
        print(f"{name}: {len(items)} files -> {target}")


if __name__ == "__main__":
    main()
