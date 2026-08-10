"""Recalculate Phase-B routing decisions from immutable DNSMOS score outputs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_dnf_phase_b_probe import (
    probe_decision,
    quantile_summary,
    sha256_file,
    write_jsonl,
)


DECISION_SCHEMA_VERSION = "dnf-phase-b-probe-decision-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate review-only Phase-B decisions without re-running "
            "DNSMOS inference."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--probe-jsonl",
        type=Path,
        required=True,
        help="New audit probe whose immutable row identities must match scores.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_group(row: dict) -> str:
    return str(
        row.get("probe_family")
        or row.get("provenance_class")
        or row.get("provenance")
        or row.get("dataset")
        or "<missing>"
    )


def verify_probe_identity(probe_rows: list[dict], scored_rows: list[dict]) -> None:
    fields = (
        "_shard_dir",
        "shard",
        "audio_member",
        "source_path",
        "content_group_id",
        "probe_family",
    )
    probes = {
        str(row["key"]): tuple(row.get(field) for field in fields)
        for row in probe_rows
    }
    scored = {
        str(row["key"]): tuple(row.get(field) for field in fields)
        for row in scored_rows
    }
    if probes != scored:
        raise ValueError(
            "new audit probe identities differ from immutable scored rows"
        )


def recalibrate(rows: list[dict]) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("cannot recalibrate an empty scored manifest")
    valid_bak_by_group: dict[str, list[float]] = defaultdict(list)
    scores_by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("dnsmos", {}).get("status") != "ok":
            raise ValueError("recalibration requires successful immutable scores")
        group = score_group(row)
        for metric in ("bak", "sig", "ovrl"):
            scores_by_group[group][metric].append(
                float(row["dnsmos"][metric])
            )
        if bool(row["technical_gate"]["hard_pass"]):
            valid_bak_by_group[group].append(float(row["dnsmos"]["bak"]))
    missing = [
        group for group in scores_by_group if not valid_bak_by_group[group]
    ]
    if missing:
        raise ValueError(f"no technically valid rows in groups: {missing}")
    groups = {
        group: {
            "all_scores": {
                metric: quantile_summary(values)
                for metric, values in sorted(metric_values.items())
            },
            "technical_hard_pass_bak": quantile_summary(
                valid_bak_by_group[group]
            ),
            "promotion_metric": "bak",
            "promotion_threshold_population": "technical_hard_pass_only",
        }
        for group, metric_values in sorted(scores_by_group.items())
    }
    output = []
    for row in rows:
        group = score_group(row)
        threshold = groups[group]["technical_hard_pass_bak"]["p25"]
        updated = dict(row)
        updated["decision_schema_version"] = DECISION_SCHEMA_VERSION
        updated["probe_decision"] = probe_decision(
            technical_hard_pass=bool(row["technical_gate"]["hard_pass"]),
            technical_hard_reasons=list(
                row["technical_gate"]["hard_reasons"]
            ),
            dnsmos_bak=float(row["dnsmos"]["bak"]),
            source_bak_p25=float(threshold),
        )
        updated["training_ready"] = False
        output.append(updated)
    output.sort(key=lambda row: str(row.get("sample_uid") or row.get("uid")))
    summary = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "sample_count": len(output),
        "groups": groups,
        "decision_counts": {
            status: sum(
                row["probe_decision"]["status"] == status for row in output
            )
            for status in sorted(
                {row["probe_decision"]["status"] for row in output}
            )
        },
        "training_ready": False,
        "promotion_contract": (
            "Scores are audit strata only. Technical failures are excluded; "
            "a reviewed row may be promoted only by reclassification to "
            "clean_strict with full-weight Eq.15."
        ),
    }
    return output, summary


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    probe_rows = read_jsonl(args.probe_jsonl)
    verify_probe_identity(probe_rows, rows)
    output, summary = recalibrate(rows)
    summary["input_jsonl"] = str(args.input_jsonl)
    summary["input_sha256"] = sha256_file(args.input_jsonl)
    summary["probe_jsonl"] = str(args.probe_jsonl)
    summary["probe_sha256"] = sha256_file(args.probe_jsonl)
    summary["output_jsonl"] = str(args.output_jsonl)
    summary["output_sha256"] = write_jsonl(args.output_jsonl, output)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
