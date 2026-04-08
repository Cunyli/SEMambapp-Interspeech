import argparse
import json
import random
from pathlib import Path


def load_json_list(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return data


def dump_json_list(path: Path, items: list[str]) -> None:
    path.write_text(json.dumps(items, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample validation clean files from the training speech list."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/train_speech.json"),
        help="Input training speech JSON file.",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("data/val_clean.json"),
        help="Output validation clean JSON file.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("data/val_split_manifest.json"),
        help="Output manifest with split metadata.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of validation samples to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed split without writing files.",
    )
    args = parser.parse_args()

    train_items = load_json_list(args.train_json)
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.count >= len(train_items):
        raise ValueError("--count must be smaller than the number of training items.")

    rng = random.Random(args.seed)
    sampled_indices = set(rng.sample(range(len(train_items)), args.count))

    val_items = [item for idx, item in enumerate(train_items) if idx in sampled_indices]
    remaining_train_items = [
        item for idx, item in enumerate(train_items) if idx not in sampled_indices
    ]

    manifest = {
        "source_train_json": str(args.train_json),
        "output_val_json": str(args.val_json),
        "seed": args.seed,
        "val_count": len(val_items),
        "remaining_train_count": len(remaining_train_items),
        "sampled_indices": sorted(sampled_indices),
    }

    print(f"Selected {len(val_items)} validation files from {args.train_json}.")
    print(f"Remaining training files: {len(remaining_train_items)}")
    print("First 5 validation files:")
    for item in val_items[:5]:
        print(item)

    if args.dry_run:
        return

    dump_json_list(args.train_json, remaining_train_items)
    dump_json_list(args.val_json, val_items)
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.train_json}")
    print(f"Wrote {args.val_json}")
    print(f"Wrote {args.manifest_json}")


if __name__ == "__main__":
    main()
