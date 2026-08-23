#!/usr/bin/env python3
"""Characterize an exact-relative LTAS-slope low-pass gate.

This diagnostic compares the frozen Route C LTAS-slope response with exact
Praat on the same hard 3 kHz low-pass, gain, and circular-shift variants.  It
does not modify the production anti-shortcut gate.  Calibration must pass the
pre-registered authority-relative contract before a separate holdout run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    PraatDifferentiableAVQIComponentEstimator,
)
from scripts.evaluate_avqi_ltas_slope_lowpass_authority import (
    SAMPLE_RATE,
    load_audio,
    lowpass_3khz,
    read_rows,
    run_exact,
    sha256_file,
    write_json,
)


SLOPE_INDEX = AVQI_COMPONENT_NAMES.index("slope")
METRIC_SAMPLE_COUNT = 3 * SAMPLE_RATE
EXACT_MATERIAL_DISTANCE_MIN = 0.02
AUTHORITY_RATIO_RANGE = (0.75, 1.25)
DIRECTION_AGREEMENT_MIN = 0.80
INVARIANCE_DISTANCE_MAX = 0.10
CURRENT_ABSOLUTE_LOWPASS_MIN = 0.10
SUPPORTED_ARCHITECTURE = "direct_praat_hard_shimmer_pulse_path_v6"
VARIANT_NAMES = (
    "clean",
    "gain_minus12db",
    "circular_shift_100ms",
    "lowpass_3khz",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("surrogate_calibration", "surrogate_holdout"),
        required=True,
    )
    parser.add_argument("--max-cases", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", default="")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def select_cases(
    rows: list[dict[str, str]], split: str, max_cases: int
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    speakers: set[str] = set()
    for row in rows:
        if (
            row["split"] != split
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
        raise ValueError(f"expected {max_cases} disjoint {split} patient SV cases")
    return selected


def waveform_variants(audio: np.ndarray) -> dict[str, np.ndarray]:
    shift = int(0.1 * SAMPLE_RATE)
    return {
        "clean": audio,
        "gain_minus12db": (audio * 0.25).astype(np.float32),
        "circular_shift_100ms": np.roll(audio, shift).astype(np.float32),
        "lowpass_3khz": lowpass_3khz(audio),
    }


def load_predictor(
    checkpoint_path: Path,
    expected_hash: str,
    device: torch.device,
) -> tuple[PraatDifferentiableAVQIComponentEstimator, dict[str, torch.Tensor]]:
    if sha256_file(checkpoint_path) != expected_hash:
        raise ValueError("LTAS predictor checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != SUPPORTED_ARCHITECTURE:
        raise ValueError("unexpected LTAS predictor architecture")
    if tuple(checkpoint.get("components", ())) != AVQI_COMPONENT_NAMES:
        raise ValueError("unexpected LTAS predictor component order")
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
    ).to(device)
    estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    estimator.eval()
    for parameter in estimator.parameters():
        parameter.requires_grad_(False)
    tensors = {
        name: checkpoint[name].to(device)
        for name in (
            "target_mean",
            "target_scale",
            "calibration_scale",
            "calibration_bias",
        )
    }
    return estimator, tensors


def predict_slope(
    waveform: np.ndarray,
    estimator: PraatDifferentiableAVQIComponentEstimator,
    checkpoint: dict[str, torch.Tensor],
    *,
    exact_window: bool,
    device: torch.device,
) -> float:
    with torch.inference_mode():
        prepared = estimator._prepare(torch.from_numpy(waveform).to(device))
        if exact_window and prepared.numel() > METRIC_SAMPLE_COUNT:
            prepared = prepared[-METRIC_SAMPLE_COUNT:]
        ltas_input = estimator._soft_voiced_ltas_input(prepared)
        raw_slope, _ = estimator._global_ltas(ltas_input)
        normalized = (
            raw_slope * estimator.alignment_scale[SLOPE_INDEX]
            + estimator.alignment_bias[SLOPE_INDEX]
        )
        raw_value = (
            normalized * checkpoint["target_scale"][SLOPE_INDEX]
            + checkpoint["target_mean"][SLOPE_INDEX]
        )
        calibrated = (
            raw_value * checkpoint["calibration_scale"][SLOPE_INDEX]
            + checkpoint["calibration_bias"][SLOPE_INDEX]
        )
    return float(calibrated)


def direction_agreement(exact: np.ndarray, candidate: np.ndarray) -> float:
    if exact.shape != candidate.shape or exact.size == 0:
        raise ValueError("direction agreement expects aligned non-empty deltas")
    return float(np.mean(np.sign(exact) == np.sign(candidate)))


def summarize_mode(
    rows: list[dict[str, Any]], mode: str, train_scale: float
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in VARIANT_NAMES[1:]:
        exact_delta = np.asarray(
            [row["exact"][variant] - row["exact"]["clean"] for row in rows],
            dtype=np.float64,
        )
        candidate_delta = np.asarray(
            [
                row[mode][variant] - row[mode]["clean"]
                for row in rows
            ],
            dtype=np.float64,
        )
        exact_distance = np.abs(exact_delta) / train_scale
        candidate_distance = np.abs(candidate_delta) / train_scale
        exact_mean = float(exact_distance.mean())
        candidate_mean = float(candidate_distance.mean())
        summaries[variant] = {
            "exact_mean_standardized_distance": exact_mean,
            "candidate_mean_standardized_distance": candidate_mean,
            "candidate_to_exact_distance_ratio": (
                candidate_mean / exact_mean if exact_mean > 0.0 else math.inf
            ),
            "signed_direction_agreement": direction_agreement(
                exact_delta, candidate_delta
            ),
            "exact_signed_delta_mean": float(exact_delta.mean()),
            "candidate_signed_delta_mean": float(candidate_delta.mean()),
            "exact_standardized_distances": exact_distance.tolist(),
            "candidate_standardized_distances": candidate_distance.tolist(),
        }
    return summaries


def exact_relative_gate(summaries: dict[str, Any]) -> dict[str, Any]:
    lowpass = summaries["lowpass_3khz"]
    gain = summaries["gain_minus12db"]
    shift = summaries["circular_shift_100ms"]
    ratio = lowpass["candidate_to_exact_distance_ratio"]
    gates = {
        "exact_lowpass_is_material": (
            lowpass["exact_mean_standardized_distance"]
            >= EXACT_MATERIAL_DISTANCE_MIN
        ),
        "candidate_matches_exact_response_ratio": (
            AUTHORITY_RATIO_RANGE[0] <= ratio <= AUTHORITY_RATIO_RANGE[1]
        ),
        "signed_direction_agreement": (
            lowpass["signed_direction_agreement"] >= DIRECTION_AGREEMENT_MIN
        ),
        "gain_nearly_invariant": (
            gain["candidate_mean_standardized_distance"]
            <= INVARIANCE_DISTANCE_MAX
        ),
        "circular_shift_nearly_invariant": (
            shift["candidate_mean_standardized_distance"]
            <= INVARIANCE_DISTANCE_MAX
        ),
        "candidate_lowpass_exceeds_controls": (
            lowpass["candidate_mean_standardized_distance"]
            > max(
                gain["candidate_mean_standardized_distance"],
                shift["candidate_mean_standardized_distance"],
            )
        ),
        "exact_lowpass_exceeds_controls": (
            lowpass["exact_mean_standardized_distance"]
            > max(
                gain["exact_mean_standardized_distance"],
                shift["exact_mean_standardized_distance"],
            )
        ),
    }
    return {
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "current_absolute_gate_passes": (
            lowpass["candidate_mean_standardized_distance"]
            >= CURRENT_ABSOLUTE_LOWPASS_MIN
        ),
    }


def safe_stem(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_cases < 2:
        raise ValueError("at least two speaker-disjoint cases are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = read_rows(args.label_bank, args.label_bank_sha256)
    cases = select_cases(rows, args.split, args.max_cases)
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
    train_scale = float(train_slopes.std())
    if train_scale <= 0.0:
        raise ValueError("non-positive surrogate-train slope scale")
    estimator, checkpoint = load_predictor(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        device,
    )

    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir(parents=True)
    exact_items: list[dict[str, str]] = []
    result_rows: list[dict[str, Any]] = []
    for row in cases:
        audio = load_audio(row)
        variants = waveform_variants(audio)
        identifier = f"{row['speaker_id']}:{row['sample_id']}"
        paths: dict[str, str] = {"clean": row["sv_path"]}
        for variant in VARIANT_NAMES[1:]:
            path = waveform_root / (
                f"{safe_stem(row['speaker_id'])}__{safe_stem(row['sample_id'])}"
                f"__{variant}.wav"
            )
            sf.write(path, variants[variant], SAMPLE_RATE, subtype="PCM_24")
            paths[variant] = str(path.resolve())
        exact_items.extend(
            {
                "id": f"{identifier}:{variant}",
                "path": paths[variant],
            }
            for variant in VARIANT_NAMES
        )
        predictions = {
            mode: {
                variant: predict_slope(
                    variants[variant],
                    estimator,
                    checkpoint,
                    exact_window=(mode == "candidate_exact_window"),
                    device=device,
                )
                for variant in VARIANT_NAMES
            }
            for mode in ("candidate_frozen_full", "candidate_exact_window")
        }
        result_rows.append(
            {
                "speaker_id": row["speaker_id"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "clean_path": row["sv_path"],
                "clean_audio_sha256": row["sv_sha256"],
                "label_bank_clean_slope": float(row["slope"]),
                "variant_paths": paths,
                **predictions,
            }
        )

    exact = run_exact(exact_items, args.exact_python)
    exact_index = {row["id"]: row for row in exact["rows"]}
    for row in result_rows:
        identifier = f"{row['speaker_id']}:{row['sample_id']}"
        row["exact"] = {
            variant: float(exact_index[f"{identifier}:{variant}"]["slope"])
            for variant in VARIANT_NAMES
        }
        row["exact_clean_vs_label_bank_abs_error"] = abs(
            row["exact"]["clean"] - row["label_bank_clean_slope"]
        )

    modes = {
        mode: summarize_mode(result_rows, mode, train_scale)
        for mode in ("candidate_frozen_full", "candidate_exact_window")
    }
    gates = {mode: exact_relative_gate(summary) for mode, summary in modes.items()}
    primary_pass = gates["candidate_frozen_full"]["decision"] == "PASS"
    if args.split == "surrogate_calibration":
        decision = (
            "SUPPORTED_FREEZE_EXACT_RELATIVE_GATE_FOR_HOLDOUT"
            if primary_pass
            else "FALSIFIED_KEEP_FROZEN_PRODUCTION_GATE"
        )
    else:
        decision = (
            "PASS_EXACT_RELATIVE_LTAS_GATE_EXPERIMENT_NO_PRODUCTION_CHANGE"
            if primary_pass
            else "FAIL_EXACT_RELATIVE_LTAS_GATE_EXPERIMENT"
        )
    report = {
        "schema_version": "avqi-route-c-ltas-slope-gate-alignment-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id or None,
        "source_files_sha256": {
            "model/avqi_components.py": sha256_file(
                REPO_ROOT / "model/avqi_components.py"
            ),
            "scripts/evaluate_avqi_ltas_slope_gate_alignment.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "label_bank": str(args.label_bank.resolve()),
        "label_bank_sha256": args.label_bank_sha256,
        "predictor_checkpoint": str(args.predictor_checkpoint.resolve()),
        "predictor_checkpoint_sha256": args.predictor_checkpoint_sha256,
        "exact_python": str(args.exact_python),
        "parselmouth_version": exact["parselmouth_version"],
        "praat_version": exact["praat_version"],
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "selection": {
            "split": args.split,
            "view": "sv",
            "label": "patient",
            "condition": "clean",
            "speaker_disjoint": True,
            "cases": len(result_rows),
            "speakers": [row["speaker_id"] for row in result_rows],
        },
        "preregistered_contract": {
            "primary_candidate": "candidate_frozen_full",
            "secondary_diagnostic_only": "candidate_exact_window",
            "exact_material_distance_min": EXACT_MATERIAL_DISTANCE_MIN,
            "candidate_to_exact_distance_ratio": list(AUTHORITY_RATIO_RANGE),
            "signed_direction_agreement_min": DIRECTION_AGREEMENT_MIN,
            "gain_and_shift_distance_max": INVARIANCE_DISTANCE_MAX,
            "current_absolute_lowpass_min_unchanged": CURRENT_ABSOLUTE_LOWPASS_MIN,
            "ratio_rationale": (
                "reuse the frozen 0.75-to-1.25 calibration-slope tolerance"
            ),
            "holdout_authorized_only_by_calibration_primary_pass": True,
        },
        "train_slope_scale_std_surrogate_train": train_scale,
        "exact_clean_vs_label_bank_max_abs_error": max(
            row["exact_clean_vs_label_bank_abs_error"] for row in result_rows
        ),
        "modes": modes,
        "gate_experiment": gates,
        "rows": result_rows,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "scope": "LTAS slope gate alignment only; CPPS/HNR untouched",
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    write_json(args.output_dir / "predictions.json", result_rows)
    receipt = {
        "decision": decision,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "predictions.json": sha256_file(args.output_dir / "predictions.json"),
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    (args.output_dir / "SUMMARY.md").write_text(
        "# LTAS slope exact-relative gate alignment\n\n"
        f"Decision: `{decision}`\n\n"
        "The production anti-shortcut gate was not changed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
