#!/usr/bin/env python3
"""Audit the frozen LTAS-slope low-pass anti-shortcut gate with exact Praat.

The differentiable anti-shortcut gate applies a 3 kHz hard low-pass to the
full waveform and expects the spectral-shape component to move.  This
diagnostic applies the same perturbation to fresh speaker-disjoint
surrogate-holdout pathological SV rows, then measures the exact Praat slope
after the AVQI metric branch's 34 Hz high-pass and last-three-seconds rule.
It is an authority audit only: it cannot change gates or authorize training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
EXACT_SLOPE_SCORER = r"""
import json
import sys

import parselmouth
from parselmouth.praat import call

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    sound = parselmouth.Sound(item["path"])
    sound = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
    duration = float(call(sound, "Get total duration"))
    if duration > 3.0:
        sound = call(
            sound,
            "Extract part",
            duration - 3.0,
            duration,
            "rectangular",
            1.0,
            "no",
        )
    ltas = call(sound, "To Ltas", 1)
    slope = float(call(ltas, "Get slope", 0, 1000, 1000, 10000, "energy"))
    rows.append({"id": item["id"], "slope": slope})
print(
    "AVQI_LTAS_SLOPE_EXACT_JSON="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=4)
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


def read_rows(path: Path, expected_hash: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_hash:
        raise ValueError("exact component label-bank SHA-256 mismatch")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "speaker_id",
        "sample_id",
        "condition_id",
        "view",
        "label",
        "split",
        "scoring_status",
        "sv_path",
        "sv_sha256",
        "sv_sample_rate",
        "slope",
    }
    if not rows or not required <= rows[0].keys():
        raise ValueError("unexpected exact component label-bank schema")
    return rows


def select_cases(rows: list[dict[str, str]], max_cases: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    speakers: set[str] = set()
    for row in rows:
        if (
            row["split"] != "surrogate_holdout"
            or row["view"] != "sv"
            or row["label"] != "patient"
            or row["condition_id"] != "clean"
            or row["scoring_status"] != "ok"
            or row["speaker_id"] in speakers
        ):
            continue
        selected.append(row)
        speakers.add(row["speaker_id"])
        if len(selected) == max_cases:
            break
    if len(selected) != max_cases:
        raise ValueError(f"expected {max_cases} disjoint holdout SV cases")
    return selected


def load_audio(row: dict[str, str]) -> np.ndarray:
    path = Path(row["sv_path"])
    if not path.is_file() or sha256_file(path) != row["sv_sha256"]:
        raise ValueError(f"SV audio hash mismatch: {path}")
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"invalid mono 16 kHz SV waveform: {path}")
    if int(row["sv_sample_rate"]) != SAMPLE_RATE or not np.isfinite(audio).all():
        raise ValueError(f"invalid SV waveform metadata: {path}")
    return audio


def lowpass_3khz(audio: np.ndarray) -> np.ndarray:
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / SAMPLE_RATE)
    spectrum = np.fft.rfft(audio)
    return np.fft.irfft(spectrum * (frequencies <= 3_000.0), n=audio.size).astype(
        np.float32
    )


def run_exact(items: list[dict[str, str]], exact_python: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(exact_python), "-c", EXACT_SLOPE_SCORER],
        input=json.dumps({"items": items}, sort_keys=True),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_LTAS_SLOPE_EXACT_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"exact LTAS scorer emitted {len(lines)} JSON records")
    return json.loads(lines[0][len(marker) :])


def main() -> None:
    args = parse_args()
    if args.max_cases < 2:
        raise ValueError("at least two disjoint cases are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_rows(args.label_bank, args.label_bank_sha256)
    cases = select_cases(rows, args.max_cases)
    train_slopes = np.asarray(
        [
            float(row["slope"])
            for row in rows
            if row["split"] == "surrogate_train"
            and row["view"] in {"cs", "sv"}
            and row["scoring_status"] == "ok"
        ],
        dtype=np.float64,
    )
    if train_slopes.size < 2:
        raise ValueError("insufficient surrogate-train slope labels")
    train_scale = float(train_slopes.std())
    if train_scale <= 0.0:
        raise ValueError("non-positive surrogate-train slope scale")

    waveform_root = args.output_dir / "outputs" / "waveforms"
    waveform_root.mkdir(parents=True)
    items: list[dict[str, str]] = []
    case_rows: list[dict[str, Any]] = []
    for row in cases:
        audio = load_audio(row)
        output_path = waveform_root / f"{row['speaker_id']}__{row['sample_id']}__lowpass3k.wav"
        sf.write(output_path, lowpass_3khz(audio), SAMPLE_RATE, subtype="PCM_24")
        items.extend(
            [
                {"id": f"{row['speaker_id']}:clean", "path": row["sv_path"]},
                {"id": f"{row['speaker_id']}:lowpass_3khz", "path": str(output_path)},
            ]
        )
        case_rows.append(
            {
                "speaker_id": row["speaker_id"],
                "sample_id": row["sample_id"],
                "clean_path": row["sv_path"],
                "clean_audio_sha256": row["sv_sha256"],
                "lowpass_path": str(output_path.resolve()),
                "lowpass_audio_sha256": sha256_file(output_path),
                "label_bank_clean_slope": float(row["slope"]),
            }
        )

    exact = run_exact(items, args.exact_python)
    exact_index = {row["id"]: row for row in exact["rows"]}
    for case in case_rows:
        clean = float(exact_index[f"{case['speaker_id']}:clean"]["slope"])
        lowpass = float(exact_index[f"{case['speaker_id']}:lowpass_3khz"]["slope"])
        case["exact_slope_clean"] = clean
        case["exact_slope_lowpass_3khz"] = lowpass
        case["exact_absolute_delta"] = abs(lowpass - clean)
        case["exact_standardized_distance"] = abs(lowpass - clean) / train_scale

    distances = np.asarray(
        [case["exact_standardized_distance"] for case in case_rows], dtype=np.float64
    )
    report = {
        "schema_version": "avqi-route-c-ltas-slope-lowpass-authority-v1",
        "decision": "COMPLETED_EXACT_AUTHORITY_AUDIT_NO_GATE_CHANGE",
        "source_commit": args.source_commit,
        "label_bank": str(args.label_bank.resolve()),
        "label_bank_sha256": args.label_bank_sha256,
        "exact_python": str(args.exact_python),
        "parselmouth_version": exact["parselmouth_version"],
        "praat_version": exact["praat_version"],
        "selection": {
            "split": "surrogate_holdout",
            "view": "sv",
            "label": "patient",
            "condition": "clean",
            "speaker_disjoint": True,
            "cases": len(case_rows),
        },
        "train_slope_scale_std_surrogate_train": train_scale,
        "exact_standardized_distance": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "min": float(distances.min()),
            "max": float(distances.max()),
            "threshold_010": 0.10,
            "passes_current_gate": bool(float(distances.mean()) >= 0.10),
        },
        "cases": case_rows,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "scope": "exact authority only; no gate or model change",
    }
    output_root = args.output_dir / "outputs"
    report_path = output_root / "diagnostic_report.json"
    write_json(output_root / "predictions.json", case_rows)
    write_json(report_path, report)
    receipt = {
        "decision": report["decision"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "predictions.json": sha256_file(output_root / "predictions.json"),
        },
    }
    write_json(output_root / "completion_receipt.json", receipt)
    (output_root / "SUMMARY.md").write_text(
        "# Exact LTAS slope low-pass authority audit\n\n"
        f"Decision: `{report['decision']}`\n\n"
        "The frozen 0.10 anti-shortcut gate was not changed by this audit.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
