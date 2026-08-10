"""Build an immutable blind listening pack for one controlled Phase-A pair."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


OUTPUTS = ("standard", "eq14", "speech_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-dir", type=Path, required=True)
    parser.add_argument("--dnf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_rank(seed: int, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def index_manifest(path: Path) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in read_jsonl(path):
        key = (
            str(row["sample_uid"]),
            str(row["evaluation_input_view"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate listening row: {key}")
        indexed[key] = row
    return indexed


def build_pack(
    standard_dir: Path,
    dnf_dir: Path,
    output_dir: Path,
    *,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    standard = index_manifest(standard_dir / "listening" / "manifest.jsonl")
    dnf = index_manifest(dnf_dir / "listening" / "manifest.jsonl")
    if set(standard) != set(dnf):
        raise ValueError("Standard and DNF listening selections differ")
    output_dir.mkdir(parents=True, exist_ok=False)
    blind_rows = []
    private_rows = []
    for clip_index, key in enumerate(sorted(standard), start=1):
        uid, view = key
        left = standard[key]
        right = dnf[key]
        contract_fields = ("route", "noise_family", "target_snr_db")
        if any(left[field] != right[field] for field in contract_fields):
            raise ValueError(f"listening metadata differs for {key}")
        clip_id = f"clip_{clip_index:04d}_{view}"
        source_paths = {
            "input": standard_dir
            / "listening"
            / view
            / "input"
            / f"{uid}.wav",
            "clean": standard_dir
            / "listening"
            / view
            / "clean"
            / f"{uid}.wav",
            "standard": standard_dir
            / "listening"
            / view
            / "standard"
            / f"{uid}.wav",
            "eq14": dnf_dir
            / "listening"
            / view
            / "eq14"
            / f"{uid}.wav",
            "speech_head": dnf_dir
            / "listening"
            / view
            / "speech_head"
            / f"{uid}.wav",
        }
        missing = [str(path) for path in source_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"listening sources are missing: {missing}")
        clip_dir = output_dir / clip_id
        link_or_copy(source_paths["input"], clip_dir / "reference_input.wav")
        link_or_copy(source_paths["clean"], clip_dir / "reference_clean.wav")
        randomized = sorted(
            OUTPUTS,
            key=lambda output: stable_rank(seed, uid, view, output),
        )
        labels = ("A", "B", "C")
        mapping = {}
        for label, output in zip(labels, randomized, strict=True):
            link_or_copy(
                source_paths[output],
                clip_dir / f"candidate_{label}.wav",
            )
            mapping[label] = output
        blind_rows.append(
            {
                "clip_id": clip_id,
                "evaluation_input_view": view,
                "reference_input": str(clip_dir / "reference_input.wav"),
                "reference_clean": str(clip_dir / "reference_clean.wav"),
                "candidate_A": str(clip_dir / "candidate_A.wav"),
                "candidate_B": str(clip_dir / "candidate_B.wav"),
                "candidate_C": str(clip_dir / "candidate_C.wav"),
                "preferred_candidate": "",
                "speech_preserved": "",
                "over_smoothed_or_flattened": "",
                "notes": "",
            }
        )
        private_rows.append(
            {
                "clip_id": clip_id,
                "sample_uid": uid,
                "evaluation_input_view": view,
                "route": left["route"],
                "noise_family": left["noise_family"],
                "target_snr_db": left["target_snr_db"],
                "candidate_mapping": mapping,
            }
        )
    return blind_rows, private_rows


def main() -> None:
    args = parse_args()
    blind_rows, private_rows = build_pack(
        args.standard_dir,
        args.dnf_dir,
        args.output_dir,
        seed=args.seed,
    )
    with (args.output_dir / "listening_sheet_blind.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(blind_rows)
    with (args.output_dir / "mapping_private.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in private_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "dnf-phase-a-blind-listening-v1",
        "seed": args.seed,
        "clip_count": len(blind_rows),
        "candidate_outputs": list(OUTPUTS),
        "manual_verdict_required": True,
        "unblind_after_verdicts_are_frozen": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
