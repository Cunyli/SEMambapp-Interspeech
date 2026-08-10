"""Build review-only source proposals from immutable Phase-B probe scores."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MLS shard-level and Libri item-level clean proposals. "
            "Every output remains training_ready=false."
        )
    )
    parser.add_argument("--mls-scored", type=Path, required=True)
    parser.add_argument("--libri-scored", type=Path, required=True)
    parser.add_argument("--train-shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-mls-items-per-shard", type=int, default=16)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_json(row: dict) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (canonical_json(row) + "\n").encode("utf-8")
            digest.update(encoded)
            handle.write(encoded.decode("utf-8"))
    return digest.hexdigest()


def shard_key(row: dict) -> tuple[str, str]:
    return str(row["_shard_dir"]), str(row["shard"])


def valid_bak_thresholds(rows: list[dict]) -> tuple[float, float]:
    values = np.asarray(
        [
            float(row["dnsmos"]["bak"])
            for row in rows
            if bool(row["technical_gate"]["hard_pass"])
        ],
        dtype=np.float64,
    )
    if not values.size:
        raise ValueError("source has no technically valid probe scores")
    return float(np.quantile(values, 0.25)), float(np.quantile(values, 0.50))


def mls_shard_proposals(
    rows: list[dict],
    shard_rows: list[dict],
    *,
    expected_items_per_shard: int,
) -> tuple[list[dict], list[dict], dict]:
    p25, p50 = valid_bak_thresholds(rows)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[shard_key(row)].append(row)
    shard_lookup = {shard_key(row): row for row in shard_rows}
    strict = []
    relaxed = []
    for key, probes in sorted(grouped.items()):
        if len(probes) != expected_items_per_shard:
            raise ValueError(
                f"MLS shard {key} has {len(probes)} probes, "
                f"expected {expected_items_per_shard}"
            )
        if key not in shard_lookup:
            raise ValueError(f"missing MLS shard metadata for {key}")
        bak = np.asarray(
            [float(row["dnsmos"]["bak"]) for row in probes],
            dtype=np.float64,
        )
        technical_pass = sum(
            bool(row["technical_gate"]["hard_pass"]) for row in probes
        )
        above_p25 = int(np.sum(bak >= p25))
        metrics = {
            "probe_count": len(probes),
            "technical_pass_count": technical_pass,
            "bak_p10": float(np.quantile(bak, 0.10)),
            "bak_p50": float(np.quantile(bak, 0.50)),
            "bak_source_valid_p25": p25,
            "bak_source_valid_p50": p50,
            "bak_at_or_above_source_p25_count": above_p25,
        }
        base = {
            **shard_lookup[key],
            "schema_version": "dnf-phase-b-candidate-proposal-v2",
            "source": "MLS_HQ_en",
            "route": "clean_candidate",
            "audit_status": "independent_holdout_and_manual_review_pending",
            "training_ready": False,
            "probe_metrics": metrics,
            "promotion_target": "clean_strict_full_weight_eq15",
        }
        all_technical_pass = technical_pass == len(probes)
        if (
            all_technical_pass
            and metrics["bak_p50"] >= p50
            and metrics["bak_p10"] >= p25
        ):
            strict.append(
                {
                    **base,
                    "proposal_tier": "strict_shard_candidate",
                }
            )
        if all_technical_pass and above_p25 >= int(0.75 * len(probes)):
            relaxed.append(
                {
                    **base,
                    "proposal_tier": "relaxed_review_only",
                }
            )
    summary = {
        "source_valid_bak_p25": p25,
        "source_valid_bak_p50": p50,
        "probed_shards": len(grouped),
        "strict_shards": len(strict),
        "strict_samples": sum(int(row["sample_count"]) for row in strict),
        "relaxed_shards": len(relaxed),
        "relaxed_samples": sum(int(row["sample_count"]) for row in relaxed),
    }
    return strict, relaxed, summary


def libri_item_proposals(rows: list[dict]) -> tuple[list[dict], dict]:
    p25, p50 = valid_bak_thresholds(rows)
    selected = []
    for row in rows:
        if not bool(row["technical_gate"]["hard_pass"]):
            continue
        if float(row["dnsmos"]["bak"]) < p25:
            continue
        selected.append(
            {
                **row,
                "schema_version": "dnf-phase-b-candidate-proposal-v2",
                "route": "clean_candidate",
                "proposal_tier": "item_level_review_only",
                "audit_status": "manual_review_pending",
                "training_ready": False,
                "promotion_target": "clean_strict_full_weight_eq15",
            }
        )
    selected.sort(key=lambda row: str(row.get("key")))
    summary = {
        "source_valid_bak_p25": p25,
        "source_valid_bak_p50": p50,
        "probe_count": len(rows),
        "technical_pass_count": sum(
            bool(row["technical_gate"]["hard_pass"]) for row in rows
        ),
        "item_level_review_candidates": len(selected),
        "shard_level_promotion": False,
        "reason": (
            "The global Libri probe shows heterogeneous technical failures; "
            "probe evidence is not generalized to unscored shard members."
        ),
    }
    return selected, summary


def main() -> None:
    args = parse_args()
    mls_rows = read_jsonl(args.mls_scored)
    libri_rows = read_jsonl(args.libri_scored)
    shard_rows = read_jsonl(args.train_shard_manifest)
    mls_keys = {shard_key(row) for row in mls_rows}
    relevant_shards = [row for row in shard_rows if shard_key(row) in mls_keys]
    strict, relaxed, mls_summary = mls_shard_proposals(
        mls_rows,
        relevant_shards,
        expected_items_per_shard=args.expected_mls_items_per_shard,
    )
    libri_items, libri_summary = libri_item_proposals(libri_rows)
    outputs = {
        "mls_strict_shards": args.output_dir
        / "mls_strict_shard_candidates.jsonl",
        "mls_relaxed_shards": args.output_dir
        / "mls_relaxed_review_shards.jsonl",
        "libri_items": args.output_dir / "libri_item_review_candidates.jsonl",
    }
    hashes = {
        "mls_strict_shards": write_jsonl(outputs["mls_strict_shards"], strict),
        "mls_relaxed_shards": write_jsonl(
            outputs["mls_relaxed_shards"],
            relaxed,
        ),
        "libri_items": write_jsonl(outputs["libri_items"], libri_items),
    }
    summary = {
        "schema_version": "dnf-phase-b-candidate-proposal-v2",
        "training_ready": False,
        "mls": mls_summary,
        "libri": libri_summary,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "output_sha256": hashes,
        "claim_limit": (
            "These are review proposals. No row or shard enters training until "
            "an independent holdout and manual speech-quality review pass."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_proposal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
