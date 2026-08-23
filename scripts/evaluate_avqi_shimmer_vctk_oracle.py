#!/usr/bin/env python3
"""Narrow VCTK exact-pulse oracle for Route C Shimmer dB diagnosis.

This diagnostic selects one frozen clean utterance per speaker from the
speaker-disjoint VCTK external bank, pairs it with the same sample under
``rir_only`` and ``snr20``, freezes the exact Praat pulse topology, and tests
the current differentiable Hann-RMS amplitude tier.  It is an oracle
diagnostic only: it does not train a generator or authorize a waveform panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator
from scripts.evaluate_avqi_shimmer_pulse_oracle_pilot import EXACT_SCORER


SAMPLE_RATE = 16_000
METRIC_SAMPLE_COUNT = 3 * SAMPLE_RATE
DEFAULT_CONDITIONS = ("rir_only", "snr20")
ALPHA_GRID = (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="comma-separated non-clean conditions to diagnose",
    )
    parser.add_argument("--max-speakers", type=int, default=12)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_conditions(raw: str) -> tuple[str, ...]:
    conditions = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be nonempty and unique")
    if "clean" in conditions:
        raise ValueError("clean is the fixed target and cannot be a candidate")
    return conditions


def read_label_rows(path: Path, expected_hash: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_hash:
        raise ValueError("VCTK label-bank SHA-256 mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "speaker_id",
        "sample_id",
        "condition_id",
        "view",
        "split",
        "scoring_status",
        "cs_path",
        "cs_sha256",
        "cs_sample_rate",
    }
    if not rows or not required <= rows[0].keys():
        raise ValueError("unexpected VCTK label-bank schema")
    return rows


def select_records(
    rows: list[dict[str, str]],
    conditions: tuple[str, ...],
    max_speakers: int,
) -> list[dict[str, Any]]:
    external = [
        row
        for row in rows
        if row["split"] == "vctk_external"
        and row["view"] == "cs"
        and row["scoring_status"] == "ok"
    ]
    speakers = sorted({row["speaker_id"] for row in external})
    if max_speakers < 1 or max_speakers > len(speakers):
        raise ValueError(f"invalid max-speakers={max_speakers}")
    selected_speakers = speakers[:max_speakers]
    by_key = {(row["speaker_id"], row["sample_id"], row["condition_id"]): row for row in external}
    records: list[dict[str, Any]] = []
    for speaker in selected_speakers:
        clean_rows = sorted(
            (row for row in external if row["speaker_id"] == speaker and row["condition_id"] == "clean"),
            key=lambda row: row["sample_id"],
        )
        if len(clean_rows) != 4:
            raise ValueError(f"expected four clean rows for {speaker}")
        target = clean_rows[0]
        for condition in conditions:
            candidate = by_key.get((speaker, target["sample_id"], condition))
            if candidate is None:
                raise ValueError(f"missing paired {speaker}/{target['sample_id']}/{condition}")
            records.append(
                {
                    "speaker_id": speaker,
                    "sample_id": target["sample_id"],
                    "condition": condition,
                    "target_row": target,
                    "candidate_row": candidate,
                }
            )
    if len(records) != max_speakers * len(conditions):
        raise ValueError("VCTK oracle record-count drift")
    return records


def validate_audio(row: dict[str, str]) -> None:
    path = Path(row["cs_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != row["cs_sha256"]:
        raise ValueError(f"VCTK audio hash mismatch: {path}")
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError(f"invalid VCTK audio geometry: {path}")
    if int(row["cs_sample_rate"]) != SAMPLE_RATE:
        raise ValueError(f"label-bank sample-rate drift: {path}")


def read_audio(row: dict[str, str]) -> np.ndarray:
    audio, sample_rate = sf.read(row["cs_path"], dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"invalid VCTK waveform: {row['cs_path']}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite VCTK waveform: {row['cs_path']}")
    return audio


def run_exact_batch(
    items: list[dict[str, str]],
    exact_python: Path,
    include_pulses: bool,
) -> dict[str, Any]:
    result = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, ""],
        input=json.dumps(
            {"items": items, "include_pulses": include_pulses},
            sort_keys=True,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_SHIMMER_EXACT_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"exact Shimmer scorer emitted {len(lines)} JSON records")
    payload = json.loads(lines[0][len(marker) :])
    if len(payload["rows"]) != len(items):
        raise ValueError("exact Shimmer row count drift")
    return payload


def proxy(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    audio: torch.Tensor,
    pulses: list[float],
) -> torch.Tensor:
    return estimator.raw_shimmer_from_pulse_positions(
        audio,
        audio.new_tensor(pulses),
        metric_sample_count=METRIC_SAMPLE_COUNT,
    )


def db_gap(value: float, target: float) -> float:
    return abs(value - target)


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(value, dtype=np.float64))))


def main() -> None:
    args = parse_args()
    conditions = parse_conditions(args.conditions)
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_label_rows(args.label_bank, args.label_bank_sha256)
    records = select_records(rows, conditions, args.max_speakers)
    for record in records:
        validate_audio(record["target_row"])
        validate_audio(record["candidate_row"])

    output_root = args.output_dir / "outputs"
    waveform_root = output_root / "waveforms"
    waveform_root.mkdir(parents=True)

    exact_items = []
    for record in records:
        for role, row in (("target", record["target_row"]), ("candidate", record["candidate_row"])):
            exact_items.append(
                {
                    "id": f"{record['speaker_id']}:{record['sample_id']}:{record['condition']}:{role}",
                    "path": row["cs_path"],
                }
            )
    exact_before = run_exact_batch(exact_items, args.exact_python, include_pulses=True)
    exact_index = {row["id"]: row for row in exact_before["rows"]}
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
    ).eval()
    output_items: list[dict[str, str]] = []
    result_rows: list[dict[str, Any]] = []
    for record in records:
        target_id = f"{record['speaker_id']}:{record['sample_id']}:{record['condition']}:target"
        candidate_id = f"{record['speaker_id']}:{record['sample_id']}:{record['condition']}:candidate"
        target_exact = exact_index[target_id]
        candidate_exact = exact_index[candidate_id]
        target_audio = read_audio(record["target_row"])
        candidate_audio = read_audio(record["candidate_row"])
        target_proxy = proxy(
            estimator,
            torch.from_numpy(target_audio),
            target_exact["pulse_positions_samples"],
        ).detach()
        waveform = torch.from_numpy(candidate_audio.copy()).requires_grad_(True)
        current_proxy = proxy(
            estimator,
            waveform,
            candidate_exact["pulse_positions_samples"],
        )
        loss = (current_proxy[0] - target_proxy[0]).square()
        gradient = torch.autograd.grad(loss, waveform)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite oracle gradient: {record['candidate_row']['cs_path']}")
        gradient_norm = float(gradient.norm())
        gradient_rms = gradient.square().mean().sqrt()
        base_rms = waveform.detach().square().mean().sqrt()
        base_db_gap = db_gap(float(current_proxy[1]), float(target_proxy[1]))
        candidates: list[tuple[float, float, torch.Tensor, torch.Tensor]] = []
        for alpha in ALPHA_GRID:
            if alpha == 0.0 or float(gradient_rms) <= 1e-15:
                candidate_tensor = waveform.detach()
            else:
                candidate_tensor = waveform.detach() - alpha * base_rms * gradient / gradient_rms
            if not torch.isfinite(candidate_tensor).all() or float(candidate_tensor.abs().max()) >= 1.0:
                continue
            with torch.inference_mode():
                candidate_proxy = proxy(
                    estimator,
                    candidate_tensor,
                    candidate_exact["pulse_positions_samples"],
                )
            if db_gap(float(candidate_proxy[1]), float(target_proxy[1])) > base_db_gap + 1e-5:
                continue
            candidates.append(
                (
                    abs(float(candidate_proxy[0] - target_proxy[0])),
                    alpha,
                    candidate_tensor,
                    candidate_proxy,
                )
            )
        if not candidates:
            raise RuntimeError(f"no safe oracle candidate: {record['candidate_row']['cs_path']}")
        _, selected_alpha, selected_tensor, selected_proxy = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        output_name = (
            f"{record['speaker_id']}__{record['sample_id']}__"
            f"{record['condition']}__oracle.wav"
        ).replace("/", "_")
        output_path = waveform_root / output_name
        sf.write(output_path, selected_tensor.numpy(), SAMPLE_RATE, subtype="PCM_24")
        output_items.append(
            {
                "id": f"{record['speaker_id']}:{record['sample_id']}:{record['condition']}:after",
                "path": str(output_path.resolve()),
            }
        )
        result_rows.append(
            {
                "speaker_id": record["speaker_id"],
                "sample_id": record["sample_id"],
                "condition": record["condition"],
                "target_path": record["target_row"]["cs_path"],
                "candidate_path": record["candidate_row"]["cs_path"],
                "target_exact_shimmer_db": float(target_exact["shimmer_db"]),
                "candidate_exact_shimmer_db_before": float(candidate_exact["shimmer_db"]),
                "target_proxy_shimmer_db": float(target_proxy[1]),
                "candidate_proxy_shimmer_db_before": float(current_proxy[1].detach()),
                "candidate_proxy_shimmer_db_after": float(selected_proxy[1]),
                "exact_pulse_count_target": int(target_exact["pulse_count"]),
                "exact_pulse_count_candidate": int(candidate_exact["pulse_count"]),
                "gradient_norm": gradient_norm,
                "selected_alpha": selected_alpha,
                "proxy_db_gap_before": base_db_gap,
                "proxy_db_gap_after": db_gap(float(selected_proxy[1]), float(target_proxy[1])),
                "candidate_rms": rms(candidate_audio),
                "output_path": str(output_path.resolve()),
            }
        )

    exact_after = run_exact_batch(output_items, args.exact_python, include_pulses=False)
    if (
        exact_after["parselmouth_version"] != exact_before["parselmouth_version"]
        or exact_after["praat_version"] != exact_before["praat_version"]
    ):
        raise ValueError("exact scorer version drift within oracle")
    exact_after_index = {row["id"]: row for row in exact_after["rows"]}
    for row in result_rows:
        after_id = f"{row['speaker_id']}:{row['sample_id']}:{row['condition']}:after"
        exact_after_row = exact_after_index[after_id]
        target_db = row["target_exact_shimmer_db"]
        before_gap = abs(row["candidate_exact_shimmer_db_before"] - target_db)
        after_gap = abs(float(exact_after_row["shimmer_db"]) - target_db)
        row["candidate_exact_shimmer_db_after"] = float(exact_after_row["shimmer_db"])
        row["exact_db_gap_before"] = before_gap
        row["exact_db_gap_after"] = after_gap
        row["exact_db_gap_reduction"] = before_gap - after_gap

    aggregates: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        selected = [row for row in result_rows if row["condition"] == condition]
        aggregates[condition] = {
            "rows": len(selected),
            "speakers": sorted({row["speaker_id"] for row in selected}),
            "median_exact_db_gap_reduction": float(
                np.median([row["exact_db_gap_reduction"] for row in selected])
            ),
            "exact_db_improvement_rate": float(
                np.mean([row["exact_db_gap_reduction"] > 0.0 for row in selected])
            ),
            "median_proxy_db_gap_before": float(
                np.median([row["proxy_db_gap_before"] for row in selected])
            ),
            "median_proxy_db_gap_after": float(
                np.median([row["proxy_db_gap_after"] for row in selected])
            ),
            "median_gradient_norm": float(
                np.median([row["gradient_norm"] for row in selected])
            ),
        }

    predictions_path = output_root / "predictions.json"
    write_json(predictions_path, result_rows)
    report = {
        "schema_version": "avqi-route-c-shimmer-db-vctk-oracle-v1",
        "decision": "COMPLETED_ORACLE_DIAGNOSTIC_NO_PROMOTION",
        "source_commit": args.source_commit,
        "label_bank": str(args.label_bank.resolve()),
        "label_bank_sha256": args.label_bank_sha256,
        "exact_python": str(args.exact_python),
        "parselmouth_version": exact_before["parselmouth_version"],
        "praat_version": exact_before["praat_version"],
        "conditions": list(conditions),
        "speaker_count": args.max_speakers,
        "samples_per_speaker": 1,
        "records": len(result_rows),
        "aggregates": aggregates,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "source_scope": "exact pulse topology oracle; not deployable pulse locator",
    }
    report_path = output_root / "diagnostic_report.json"
    write_json(report_path, report)
    receipt = {
        "decision": report["decision"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "predictions.json": sha256_file(predictions_path),
        },
    }
    write_json(output_root / "completion_receipt.json", receipt)
    (output_root / "SUMMARY.md").write_text(
        "# VCTK Shimmer dB exact-pulse oracle\n\n"
        f"Decision: `{report['decision']}`\n\n"
        "This is an exact-pulse topology diagnostic only. It does not authorize a bounded waveform panel or generator updates.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
