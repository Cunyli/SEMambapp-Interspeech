"""Build a stratified listening pack from scored Phase-B speech probes."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high, boundary, and low DNSMOS-BAK speech probes."
    )
    parser.add_argument("--scored-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_name(row: dict) -> str:
    return str(
        row.get("probe_family")
        or row.get("provenance")
        or row.get("dataset")
        or "<missing>"
    )


def select_strata(rows: list[dict], per_stratum: int) -> list[dict]:
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("dnsmos", {}).get("status") != "ok":
            raise ValueError("all review rows must have successful DNSMOS scores")
        if not bool(row.get("technical_gate", {}).get("hard_pass", False)):
            continue
        if row.get("probe_decision", {}).get("route") == "exclude_invalid":
            continue
        groups[group_name(row)].append(row)
    if not groups:
        raise ValueError("no technically valid speech probes for listening review")
    selected = []
    for group in sorted(groups):
        bucket = groups[group]
        bak = np.asarray(
            [float(row["dnsmos"]["bak"]) for row in bucket],
            dtype=np.float64,
        )
        p25 = float(np.quantile(bak, 0.25))
        rankings = {
            "low_bak": sorted(
                bucket,
                key=lambda row: (
                    float(row["dnsmos"]["bak"]),
                    str(row.get("key")),
                ),
            ),
            "p25_boundary": sorted(
                bucket,
                key=lambda row: (
                    abs(float(row["dnsmos"]["bak"]) - p25),
                    str(row.get("key")),
                ),
            ),
            "high_bak": sorted(
                bucket,
                key=lambda row: (
                    -float(row["dnsmos"]["bak"]),
                    str(row.get("key")),
                ),
            ),
        }
        used = set()
        for stratum in ("low_bak", "p25_boundary", "high_bak"):
            count = 0
            for row in rankings[stratum]:
                key = str(row.get("key") or row.get("audio_member"))
                if key in used:
                    continue
                used.add(key)
                selected.append(
                    {
                        **row,
                        "review_group": group,
                        "review_stratum": stratum,
                        "review_group_bak_p25": p25,
                    }
                )
                count += 1
                if count >= per_stratum:
                    break
    return selected


def load_audio(row: dict) -> tuple[np.ndarray, int]:
    tar_path = Path(row["_shard_dir"]) / str(row["shard"])
    with tarfile.open(tar_path, "r:") as archive:
        extracted = archive.extractfile(str(row["audio_member"]))
        if extracted is None:
            raise ValueError(f"missing audio member in {tar_path}")
        payload = extracted.read()
    audio, sample_rate = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError("empty/non-finite speech probe")
    return mono, int(sample_rate)


def safe_name(row: dict) -> str:
    raw = str(row["blind_id"])
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw
    )[:220]


def save_audio_pair(
    output_dir: Path,
    row: dict,
    audio: np.ndarray,
    sample_rate: int,
    max_seconds: float,
) -> dict:
    audio = audio[: int(round(max_seconds * sample_rate))]
    name = safe_name(row) + ".wav"
    original = output_dir / "original_level" / name
    normalized = output_dir / "rms_normalized_listening_only" / name
    original.parent.mkdir(parents=True, exist_ok=True)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    sf.write(original, audio, sample_rate, subtype="FLOAT")
    current_rms = float(
        np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    )
    target_rms = 10.0 ** (-23.0 / 20.0)
    scaled = audio * (target_rms / max(current_rms, 1e-12))
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.95:
        scaled *= 0.95 / peak
    sf.write(normalized, scaled, sample_rate, subtype="FLOAT")
    return {
        "original_level": str(original),
        "rms_normalized_listening_only": str(normalized),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def assign_blind_ids(rows: list[dict], seed: int) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (
                f"{seed}\0{row.get('key') or row.get('audio_member')}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    return [
        {
            **row,
            "blind_id": f"clip_{index:04d}",
        }
        for index, row in enumerate(ranked, start=1)
    ]


def main() -> None:
    args = parse_args()
    all_rows = []
    for path in args.scored_jsonl:
        all_rows.extend(read_jsonl(path))
    selected = assign_blind_ids(
        select_strata(all_rows, args.per_stratum),
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    review_rows = []
    for row in selected:
        audio, sample_rate = load_audio(row)
        review_rows.append(
            {
                **row,
                "review_audio": save_audio_pair(
                    args.output_dir,
                    row,
                    audio,
                    sample_rate,
                    args.max_seconds,
                ),
                "manual_verdict": "",
                "manual_notes": "",
                "training_ready": False,
            }
        )
    review_rows.sort(key=lambda row: row["blind_id"])
    write_jsonl(
        args.output_dir / "speech_review_manifest_private.jsonl",
        review_rows,
    )
    sheet_fields = [
        "blind_id",
        "original_level",
        "rms_normalized_listening_only",
        "manual_verdict",
        "manual_notes",
    ]
    with (args.output_dir / "manual_review_sheet_blind.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sheet_fields)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "blind_id": row["blind_id"],
                    **row["review_audio"],
                    "manual_verdict": "",
                    "manual_notes": "",
                }
            )
    mapping_fields = [
        "blind_id",
        "review_group",
        "review_stratum",
        "key",
        "dataset",
        "dnsmos_bak",
        "dnsmos_sig",
        "dnsmos_ovrl",
        "technical_hard_pass",
    ]
    with (args.output_dir / "blind_mapping_private.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "blind_id": row["blind_id"],
                    "review_group": row["review_group"],
                    "review_stratum": row["review_stratum"],
                    "key": row["key"],
                    "dataset": row["dataset"],
                    "dnsmos_bak": row["dnsmos"]["bak"],
                    "dnsmos_sig": row["dnsmos"]["sig"],
                    "dnsmos_ovrl": row["dnsmos"]["ovrl"],
                    "technical_hard_pass": row["technical_gate"][
                        "hard_pass"
                    ],
                }
            )
    summary = {
        "schema_version": "dnf-phase-b-speech-review-pack-v2",
        "selection_seed": args.seed,
        "selected_count": len(review_rows),
        "by_group_and_stratum": {
            f"{group}|{stratum}": sum(
                row["review_group"] == group
                and row["review_stratum"] == stratum
                for row in review_rows
            )
            for group, stratum in sorted(
                {
                    (row["review_group"], row["review_stratum"])
                    for row in review_rows
                }
            )
        },
        "training_ready": False,
        "review_blinding": (
            "The reviewer sheet and filenames hide source, score, and stratum. "
            "Unblind only after manual verdicts are frozen."
        ),
        "promotion_rule": (
            "DNSMOS BAK is a source-relative ranking signal only. Manual "
            "listening plus provenance review must pass before a row is "
            "reclassified as clean_strict for full-weight Eq.15. Otherwise it "
            "retains noisy_speech_target status or is excluded."
        ),
        "manual_verdict_values": [
            "promote_clean_strict",
            "retain_noisy_speech_target",
            "exclude_invalid_or_ambiguous",
        ],
    }
    (args.output_dir / "speech_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
