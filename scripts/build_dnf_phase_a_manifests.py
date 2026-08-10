"""Build immutable clean-strict tuple manifests for the Phase-A causal pilot.

Only EARS and VCTK audio members are eligible.  The produced recipes reference
parameterized indoor HVAC, fan, and vehicle-cabin noise; they never consume a
Phase-B clean candidate, noise allowlist, or RIR.
"""

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

from dataloaders.dnf_controlled_phase_a import (
    ARTIFICIAL_NOISE_ENERGY_POLICY,
    DEFAULT_NOISE_PAIRING_POLICY,
    DEPLOYMENT_INPUT_DEFINITION,
    NOISE_FAMILIES,
    NOISE_PAIRING_CROSS_FAMILY_CYCLE,
    NOISE_PAIRING_POLICIES,
    NOISE_PAIRING_SAME_FAMILY_IID,
    ROUTE_CLEAN_REGULAR,
    ROUTE_CLEAN_WEAK,
    ROUTE_NOISY,
    SNR_DEFINITION,
    TRAINING_INPUT_DEFINITION,
    build_phase_a_manifest_rows,
    manifest_rows_sha256,
    write_jsonl,
)


AUDIO_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
ALLOWED_CLEAN_SOURCES = {"EARS", "VCTK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen EARS/VCTK Phase-A train and validation manifests."
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-rows", type=int, default=40000)
    parser.add_argument("--valid-rows", type=int, default=200)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--cut-duration", type=float, default=1.0)
    parser.add_argument(
        "--noise-pairing-policy",
        choices=NOISE_PAIRING_POLICIES,
        default=DEFAULT_NOISE_PAIRING_POLICY,
        help=(
            "same_family_iid is the paper-mechanism gate; cross_family_cycle "
            "is retained only to reproduce the earlier robustness variant"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source(source: str) -> str:
    return re.sub(r"_chunk\d+(?:_patch)?$", "", str(source))


def shard_sources(row: dict) -> set[str]:
    counts = row.get("dataset_counts")
    if isinstance(counts, dict) and counts:
        return {normalize_source(source) for source in counts}
    return {
        normalize_source(
            str(row.get("dataset") or row.get("source") or "<missing>")
        )
    }


def read_shards(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty shard manifest: {path}")
    return rows


def enumerate_clean_strict_items(path: Path) -> tuple[list[dict], Counter]:
    items = []
    counts: Counter = Counter()
    for shard in read_shards(path):
        sources = shard_sources(shard)
        if not sources.issubset(ALLOWED_CLEAN_SOURCES):
            continue
        if len(sources) != 1:
            raise ValueError(f"mixed clean-strict sources in shard: {shard}")
        source = next(iter(sources))
        shard_dir = Path(str(shard["_shard_dir"]))
        shard_name = str(shard["shard"])
        tar_path = shard_dir / shard_name
        with tarfile.open(tar_path, "r:") as archive:
            for member in archive:
                if (
                    not member.isfile()
                    or member.size <= 0
                    or PurePosixPath(member.name).suffix.lower()
                    not in AUDIO_EXTENSIONS
                ):
                    continue
                items.append(
                    {
                        "_shard_dir": str(shard_dir),
                        "shard": shard_name,
                        "audio_member": member.name,
                        "dataset": source,
                        "key": PurePosixPath(member.name).stem,
                    }
                )
                counts[source] += 1
    items.sort(
        key=lambda item: (
            item["dataset"],
            item["_shard_dir"],
            item["shard"],
            item["audio_member"],
        )
    )
    if not items:
        raise ValueError(f"no EARS/VCTK items in {path}")
    return items, counts


def route_counts(rows: list[dict]) -> dict[str, int]:
    return {
        route: sum(row["route"] == route for row in rows)
        for route in (ROUTE_NOISY, ROUTE_CLEAN_REGULAR, ROUTE_CLEAN_WEAK)
    }


def validate_rows(
    rows: list[dict],
    expected_count: int,
    *,
    noise_pairing_policy: str,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(rows)}")
    if expected_count % 20:
        raise ValueError("Phase-A manifest length must be a multiple of 20")
    expected = {
        ROUTE_NOISY: expected_count * 15 // 20,
        ROUTE_CLEAN_REGULAR: expected_count * 4 // 20,
        ROUTE_CLEAN_WEAK: expected_count // 20,
    }
    actual = route_counts(rows)
    if actual != expected:
        raise ValueError(f"route counts {actual} != frozen contract {expected}")
    clean_speech = {
        json.dumps(row["speech"], sort_keys=True)
        for row in rows
        if row["route"] != ROUTE_NOISY
    }
    noisy_speech = {
        json.dumps(row["speech"], sort_keys=True)
        for row in rows
        if row["route"] == ROUTE_NOISY
    }
    if clean_speech & noisy_speech:
        raise ValueError("clean and noisy routes must use disjoint speech items")
    if len(clean_speech) != expected[ROUTE_CLEAN_REGULAR] + expected[ROUTE_CLEAN_WEAK]:
        raise ValueError("clean-route speech items must be unique")
    if len(noisy_speech) != expected[ROUTE_NOISY]:
        raise ValueError("noisy-route speech items must be unique")
    for row in rows:
        source = str(row["speech"].get("dataset"))
        if source not in ALLOWED_CLEAN_SOURCES:
            raise ValueError(f"non-strict source leaked into Phase A: {source}")
        expected_partition = (
            "noisy_pool" if row["route"] == ROUTE_NOISY else "clean_pool"
        )
        if row.get("speech_partition") != expected_partition:
            raise ValueError("speech partition label disagrees with route")
        for noise_key in ("noise1", "noise2"):
            if row[noise_key]["family"] not in NOISE_FAMILIES:
                raise ValueError(
                    f"unexpected {noise_key} family: {row[noise_key]['family']}"
                )
        if row.get("noise_pairing_policy") != noise_pairing_policy:
            raise ValueError("row noise-pairing policy differs from the receipt")
        same_family = row["noise1"]["family"] == row["noise2"]["family"]
        if (
            noise_pairing_policy == NOISE_PAIRING_SAME_FAMILY_IID
            and not same_family
        ):
            raise ValueError("paper-mechanism rows require same-family n1/n2")
        if (
            noise_pairing_policy == NOISE_PAIRING_CROSS_FAMILY_CYCLE
            and same_family
        ):
            raise ValueError("cross-family rows require distinct n1/n2 families")
        if row["noise1"]["seed"] == row["noise2"]["seed"]:
            raise ValueError("n1 and n2 must be independent realizations")


def main() -> None:
    args = parse_args()
    if args.train_rows <= 0 or args.valid_rows <= 0:
        raise ValueError("train and valid row counts must be positive")
    if args.train_rows % 20 or args.valid_rows % 20:
        raise ValueError("train and valid row counts must be multiples of 20")
    sample_count = int(round(args.sample_rate * args.cut_duration))
    if sample_count <= 0:
        raise ValueError("cut duration must produce a positive sample count")

    train_input = args.split_root / "train" / "clean_shards.jsonl"
    valid_input = args.split_root / "valid" / "clean_shards.jsonl"
    train_items, train_source_counts = enumerate_clean_strict_items(train_input)
    valid_items, valid_source_counts = enumerate_clean_strict_items(valid_input)
    train_rows = build_phase_a_manifest_rows(
        train_items,
        row_count=args.train_rows,
        seed=args.seed,
        split="train",
        sample_rate=args.sample_rate,
        sample_count=sample_count,
        noise_pairing_policy=args.noise_pairing_policy,
    )
    valid_rows = build_phase_a_manifest_rows(
        valid_items,
        row_count=args.valid_rows,
        seed=args.seed,
        split="valid",
        sample_rate=args.sample_rate,
        sample_count=sample_count,
        noise_pairing_policy=args.noise_pairing_policy,
    )
    validate_rows(
        train_rows,
        args.train_rows,
        noise_pairing_policy=args.noise_pairing_policy,
    )
    validate_rows(
        valid_rows,
        args.valid_rows,
        noise_pairing_policy=args.noise_pairing_policy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.output_dir / "train_manifest.jsonl"
    valid_output = args.output_dir / "valid_manifest.jsonl"
    write_jsonl(train_output, train_rows)
    write_jsonl(valid_output, valid_rows)
    receipt = {
        "schema_version": "dnf-phase-a-manifest-receipt-v2",
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "sample_count": sample_count,
        "allowed_clean_sources": sorted(ALLOWED_CLEAN_SOURCES),
        "forbidden_phase_b_inputs": True,
        "noise_families": list(NOISE_FAMILIES),
        "noise_pairing_policy": args.noise_pairing_policy,
        "paper_mechanism_gate": (
            args.noise_pairing_policy == NOISE_PAIRING_SAME_FAMILY_IID
        ),
        "speech_partition_policy": "disjoint_item_pools",
        "speech_route_overlap_count": 0,
        "snr_definition": SNR_DEFINITION,
        "training_input": TRAINING_INPUT_DEFINITION,
        "deployment_validation_input": DEPLOYMENT_INPUT_DEFINITION,
        "artificial_noise_energy_policy": ARTIFICIAL_NOISE_ENERGY_POLICY,
        "splits": {
            "train": {
                "input_shards": str(train_input),
                "input_shards_sha256": sha256_file(train_input),
                "available_clean_strict_items": len(train_items),
                "available_by_source": dict(sorted(train_source_counts.items())),
                "output": str(train_output),
                "row_count": len(train_rows),
                "route_counts": route_counts(train_rows),
                "manifest_rows_sha256": manifest_rows_sha256(train_rows),
                "file_sha256": sha256_file(train_output),
            },
            "valid": {
                "input_shards": str(valid_input),
                "input_shards_sha256": sha256_file(valid_input),
                "available_clean_strict_items": len(valid_items),
                "available_by_source": dict(sorted(valid_source_counts.items())),
                "output": str(valid_output),
                "row_count": len(valid_rows),
                "route_counts": route_counts(valid_rows),
                "manifest_rows_sha256": manifest_rows_sha256(valid_rows),
                "file_sha256": sha256_file(valid_output),
            },
        },
    }
    receipt_path = args.output_dir / "manifest_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
