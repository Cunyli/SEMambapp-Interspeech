#!/usr/bin/env python3
"""Re-audit preserved AVQI waveform-pilot outputs under the v2 guardrails."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_direct_avqi_waveform_optimization import (
    AVQI_COMPONENT_NAMES,
    full_band_pathology_guardrails,
    load_waveform,
    repository_head,
    sha256_file,
    summarize,
    write_csv,
    write_json,
)


STRING_FIELDS = {
    "case_id",
    "speaker_id",
    "view",
    "sample_group",
    "condition",
    "candidate",
    "source_path",
    "source_sha256",
    "optimized_path",
    "optimized_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-csv", type=Path, required=True)
    parser.add_argument("--source-results-csv-sha256", required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-report-sha256", required=True)
    parser.add_argument("--external-exact-csv", type=Path, required=True)
    parser.add_argument("--external-exact-csv-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row: dict[str, Any] = {}
        for key, value in source.items():
            if key in STRING_FIELDS or key.endswith("_path") or key.endswith(
                "_sha256"
            ):
                row[key] = value
            else:
                row[key] = float(value)
        rows.append(row)
    if not rows:
        raise ValueError("source waveform result table is empty")
    return rows


def clean_reference_paths(path: Path) -> dict[tuple[str, str], Path]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[str, str], Path] = {}
    for row in rows:
        if (
            row["source_type"] != "clean_reference"
            or row["label"] != "patient"
            or row["scoring_status"] != "ok"
            or row["view"] not in {"cs", "sv"}
        ):
            continue
        view = row["view"]
        key = (row["speaker_id"], view)
        if key in output:
            raise ValueError(f"duplicate clean reference: {key}")
        output[key] = Path(row[f"{view}_path"])
    return output


def load_target_scale(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != "direct_praat_hard_v2":
        raise ValueError("guardrail audit requires direct_praat_hard_v2 scale")
    if tuple(checkpoint["components"]) != AVQI_COMPONENT_NAMES:
        raise ValueError("predictor checkpoint component order differs")
    scale = checkpoint["target_scale"].detach().cpu().to(torch.float32)
    if scale.shape != (len(AVQI_COMPONENT_NAMES),) or not torch.isfinite(
        scale
    ).all():
        raise ValueError("predictor checkpoint target scale is invalid")
    return scale


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    failed_slices = [
        name
        for name, item in summary["required_slices"].items()
        if item["decision"] != "PASS"
    ]
    return "\n".join(
        [
            "# AVQI waveform pilot v2 guardrail re-audit",
            "",
            f"**Decision:** `{report['decision']}`",
            "",
            (
                "Full-band pathology guardrails: "
                f"`{summary['full_band_pathology_guardrails']['decision']}`"
            ),
            f"Denoising non-regression: `{summary['denoising']['decision']}`",
            (
                "Required slices: all passed"
                if not failed_slices
                else "Failed required slices: " + ", ".join(failed_slices)
            ),
            "",
            "This audit reused hash-locked exact component results and preserved "
            "WAV files. It ran zero waveform or generator optimizer steps.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("declared audit source commit differs from repository HEAD")
    source_hashes = {
        args.source_results_csv: args.source_results_csv_sha256,
        args.source_report: args.source_report_sha256,
        args.external_exact_csv: args.external_exact_csv_sha256,
        args.predictor_checkpoint: args.predictor_checkpoint_sha256,
    }
    for path, expected_hash in source_hashes.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source hash drift: {path}")
    source_report = load_json(args.source_report)
    if source_report.get("generator_optimizer_steps") != 0:
        raise ValueError("source waveform report contains generator updates")
    if source_report.get("formal_pathology_training_submitted") is not False:
        raise ValueError("source formal training state is ambiguous")
    rows = load_result_rows(args.source_results_csv)
    references = clean_reference_paths(args.external_exact_csv)
    for row in rows:
        source_path = Path(row["source_path"])
        optimized_path = Path(row["optimized_path"])
        if sha256_file(source_path) != row["source_sha256"]:
            raise ValueError(f"source waveform hash drift: {source_path}")
        if sha256_file(optimized_path) != row["optimized_sha256"]:
            raise ValueError(f"optimized waveform hash drift: {optimized_path}")
        reference_path = references[(row["speaker_id"], row["view"])]
        reference, _ = load_waveform(reference_path)
        base, _ = load_waveform(source_path)
        candidate, _ = load_waveform(optimized_path)
        guardrails = full_band_pathology_guardrails(
            reference,
            base,
            candidate,
        )
        row.update(guardrails)
        row["clean_pathological_reference_path"] = str(
            reference_path.resolve()
        )
        row["clean_pathological_reference_sha256"] = sha256_file(
            reference_path
        )
    target_scale = load_target_scale(args.predictor_checkpoint)
    summary = summarize(
        rows,
        target_scale,
        float(source_report["contract"]["residual_ceiling_db"]),
    )
    report = {
        "schema_version": "direct-avqi-waveform-guardrail-reaudit-v1",
        "decision": summary["decision"],
        "source_decision": source_report["decision"],
        "source_commit": args.source_commit,
        "source_waveform_optimizer_steps": source_report[
            "waveform_optimizer_steps"
        ],
        "audit_waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "summary": summary,
        "source_sha256": {
            path.name: expected_hash
            for path, expected_hash in source_hashes.items()
        },
    }
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "audited_results.csv"
    report_path = args.output_dir / "guardrail_audit_report.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_csv(result_path, rows)
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    receipt = {
        "decision": report["decision"],
        "case_count": len(rows),
        "audit_waveform_optimizer_steps": 0,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (result_path, report_path, summary_path)
        },
        "source_sha256": report["source_sha256"],
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
