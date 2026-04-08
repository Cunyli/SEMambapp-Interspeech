import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulate_degradation import apply_degradation, random_select_and_order


def load_json_list(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return data


def load_audio(path: str, target_sr: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.reshape(1, -1)


def make_output_path(clean_path: str, clean_root: Path, output_root: Path, index: int) -> Path:
    clean_file = Path(clean_path)
    try:
        rel = clean_file.relative_to(clean_root)
        out_path = output_root / rel
    except ValueError:
        out_path = output_root / f"{index:04d}_{clean_file.stem}{clean_file.suffix}"
    return out_path.with_suffix(".wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline degraded validation audio from val_clean.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--clean-json",
        type=Path,
        default=Path("data/val_clean.json"),
        help="Validation clean JSON file.",
    )
    parser.add_argument(
        "--noise-json",
        type=Path,
        default=Path("data/train_noise.json"),
        help="Noise JSON file.",
    )
    parser.add_argument(
        "--rir-json",
        type=Path,
        default=Path("data/train_rir.json"),
        help="RIR JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/val_degraded"),
        help="Directory to store degraded validation audio.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/val_degraded.json"),
        help="Output degraded JSON file.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("data/val_degraded_manifest.json"),
        help="Output manifest for reproducibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for partial generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing degraded files.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    clean_paths = load_json_list(args.clean_json)
    noise_paths = load_json_list(args.noise_json)
    rir_paths = load_json_list(args.rir_json)

    if not clean_paths:
        raise ValueError(f"No clean validation files found in {args.clean_json}.")
    if not noise_paths:
        raise ValueError(f"No noise files found in {args.noise_json}.")
    if not rir_paths:
        raise ValueError(f"No RIR files found in {args.rir_json}.")

    if args.limit is not None:
        clean_paths = clean_paths[: args.limit]

    sr = cfg["stft_cfg"]["sampling_rate"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    clean_root = Path(os.path.commonpath(clean_paths))
    degraded_paths: list[str] = []
    manifest_entries: list[dict] = []

    for index, clean_path in enumerate(clean_paths):
        item_seed = args.seed + index
        rng = random.Random(item_seed)
        np.random.seed(item_seed)

        noise_path = noise_paths[rng.randrange(len(noise_paths))]
        rir_path = rir_paths[rng.randrange(len(rir_paths))]

        clean_audio = load_audio(clean_path, sr)
        noise_audio = load_audio(noise_path, sr)
        rir_audio = load_audio(rir_path, sr)

        degrad_cfgs, selected_degrads = random_select_and_order(cfg, seed=item_seed)
        _, degraded_audio = apply_degradation(
            cfg,
            clean_audio,
            noise_audio,
            rir_audio,
            degrad_cfgs,
            selected_degrads,
            seed=item_seed,
        )

        out_path = make_output_path(clean_path, clean_root, args.output_root, index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")

        sf.write(out_path, degraded_audio.squeeze(), sr, subtype="FLOAT")
        degraded_paths.append(str(out_path.resolve()))

        manifest_entries.append(
            {
                "index": index,
                "seed": item_seed,
                "clean_path": clean_path,
                "degraded_path": str(out_path.resolve()),
                "noise_path": noise_path,
                "rir_path": rir_path,
                "selected_degradations": selected_degrads,
                "degradation_config": degrad_cfgs,
            }
        )

        if (index + 1) % 25 == 0 or index == len(clean_paths) - 1:
            print(f"Generated {index + 1}/{len(clean_paths)}")

    args.output_json.write_text(json.dumps(degraded_paths, indent=2) + "\n")
    args.manifest_json.write_text(json.dumps(manifest_entries, indent=2) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.manifest_json}")


if __name__ == "__main__":
    main()
