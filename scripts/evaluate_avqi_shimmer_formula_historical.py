#!/usr/bin/env python3
"""Non-final historical diagnostic for Route C Shimmer formulas.

The script compares the frozen v2 analytic-envelope shimmer with the v5
AVQI-aligned pulse-chain shimmer against already-scored exact Praat labels.
It never loads or updates the enhancement generator and cannot promote a
formula to training. A fresh speaker-disjoint exact panel remains mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy import stats

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator


COMPONENTS = ("shimmer_percent", "shimmer_db")
ESTIMATOR_MODES = {
    "analytic_envelope_v2": "analytic_envelope_v2",
    "pulse_chain_v5": "praat_pulse_chain_v5",
    "pulse_path_v6": "praat_pulse_path_v6",
}
REALISTIC_PATTERN = re.compile(
    r"^(?P<case>[0-9]+)__(?P<speaker>.+?)__"
    r"(?P<condition>rir_only|snr30|snr20)__"
    r"(?P<view>cs|sv)__(?P<suffix>input|target_clean|B0_250|S3_500|S3_2000)\.wav$"
)
IDENTITY_PATTERN = re.compile(
    r"^(?P<case>[0-9]+)__(?P<speaker>.+?)__"
    r"(?P<condition>clean|snr10)__(?P<view>cs|sv)__"
    r"(?P<suffix>input|target_clean|B_pair_500|B_sv_match_500)\.wav$"
)
REALISTIC_CALIBRATION_SPEAKERS = frozenset({"ÄHH10", "PD08"})
REALISTIC_HOLDOUT_SPEAKERS = frozenset({"PD_51", "V55"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realistic-root", type=Path, required=True)
    parser.add_argument("--realistic-exact-csv", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--identity-exact-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_value(
    exact_rows: list[dict[str, str]],
    speaker: str,
    condition: str,
    view: str,
    suffix: str,
    component: str,
) -> float:
    if suffix == "target_clean":
        matches = [
            row
            for row in exact_rows
            if row["source_type"] == "clean_reference"
            and row["speaker_id"] == speaker
            and row["view"] == view
        ]
    elif suffix == "input":
        matches = [
            row
            for row in exact_rows
            if row["source_type"] == "input"
            and row["speaker_id"] == speaker
            and row["condition"] == condition
            and row["view"] == view
        ]
    else:
        matches = [
            row
            for row in exact_rows
            if row["source_type"] == "enhanced"
            and row["candidate"] == suffix
            and row["speaker_id"] == speaker
            and row["condition"] == condition
            and row["view"] == view
        ]
    if len(matches) != 1:
        raise ValueError(
            "exact label lookup is not unique: "
            f"{speaker=}, {condition=}, {view=}, {suffix=}, matches={len(matches)}"
        )
    return float(matches[0]["audio_" + component])


def analytic_envelope_shimmer(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    prepared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames, period, voicing_weight, _ = estimator._linear_ac_v2_pitch_features(
        prepared
    )
    envelope_spectrum = torch.fft.fft(prepared)
    hilbert_response = torch.zeros_like(envelope_spectrum)
    hilbert_response[0] = 1.0
    if prepared.numel() % 2 == 0:
        hilbert_response[1 : prepared.numel() // 2] = 2.0
        hilbert_response[prepared.numel() // 2] = 1.0
    else:
        hilbert_response[1 : (prepared.numel() + 1) // 2] = 2.0
    envelope = torch.fft.ifft(envelope_spectrum * hilbert_response).abs()
    centers = (
        torch.arange(
            frames.shape[0],
            device=prepared.device,
            dtype=prepared.dtype,
        )
        * estimator.hop_length
        + estimator.frame_length / 2.0
    )
    current_amplitude = estimator._sample_linear(envelope, centers)
    previous_positions = centers - period
    previous_amplitude = estimator._sample_linear(envelope, previous_positions)
    valid_weight = torch.sigmoid(previous_positions / 4.0)
    shimmer_weight = (voicing_weight * valid_weight).clamp_min(1e-5)
    difference = estimator._smooth_absolute(
        current_amplitude - previous_amplitude
    )
    percent = estimator._weighted_mean(
        200.0
        * difference
        / (current_amplitude + previous_amplitude).clamp_min(1e-8),
        shimmer_weight,
    )
    db = estimator._weighted_mean(
        estimator._smooth_absolute(
            20.0
            * torch.log10(
                current_amplitude.clamp_min(1e-8)
                / previous_amplitude.clamp_min(1e-8)
            )
        ),
        shimmer_weight,
    )
    return percent, db


def raw_shimmer(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    waveform: torch.Tensor,
    *,
    count_pulses: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    prepared = estimator._prepare(waveform)
    if estimator.shimmer_mode in {
        "praat_pulse_chain_v5",
        "praat_pulse_path_v6",
    }:
        pulse_count = None
        if count_pulses:
            pulses = estimator._praat_shimmer_pulse_chain(prepared)
            pulse_count = int(pulses.numel())
        percent, db = estimator._praat_pulse_chain_shimmer(prepared)
        return percent, db, pulse_count
    percent, db = analytic_envelope_shimmer(estimator, prepared)
    return percent, db, None


def split_name(dataset: str, case_number: int, speaker: str) -> str:
    if dataset == "realistic":
        if speaker in REALISTIC_CALIBRATION_SPEAKERS:
            return "calibration"
        if speaker in REALISTIC_HOLDOUT_SPEAKERS:
            return "holdout"
        raise ValueError(f"unexpected realistic-pack speaker: {speaker}")
    return "calibration" if case_number % 2 == 1 else "holdout"


def load_dataset(
    dataset: str,
    root: Path,
    exact_csv: Path,
    pattern: re.Pattern[str],
    expected_rows: int,
    estimators: dict[str, PraatDifferentiableAVQIComponentEstimator],
) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing waveform root: {root}")
    if not exact_csv.is_file():
        raise FileNotFoundError(f"missing exact label file: {exact_csv}")
    with exact_csv.open(encoding="utf-8", newline="") as handle:
        exact_rows = list(csv.DictReader(handle))

    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.wav")):
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(f"unexpected waveform filename: {path.name}")
        fields = match.groupdict()
        case_number = int(fields["case"])
        speaker = fields["speaker"]
        condition = fields["condition"]
        view = fields["view"]
        suffix = fields["suffix"]
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != 16_000 or audio.ndim != 1:
            raise ValueError(f"expected mono 16 kHz waveform: {path}")
        waveform = torch.from_numpy(audio)
        record: dict[str, Any] = {
            "dataset": dataset,
            "case_number": case_number,
            "split": split_name(dataset, case_number, speaker),
            "speaker": speaker,
            "condition": condition,
            "view": view,
            "suffix": suffix,
            "path": str(path.resolve()),
        }
        for component in COMPONENTS:
            record["exact_" + component] = exact_value(
                exact_rows,
                speaker,
                condition,
                view,
                suffix,
                component,
            )
        with torch.inference_mode():
            for estimator_name, estimator in estimators.items():
                percent, db, pulse_count = raw_shimmer(estimator, waveform)
                record[estimator_name + "_shimmer_percent"] = float(percent)
                record[estimator_name + "_shimmer_db"] = float(db)
                if pulse_count is not None:
                    record[estimator_name + "_pulse_count"] = pulse_count
        records.append(record)
        if len(records) % 12 == 0:
            print(f"{dataset}: evaluated {len(records)}/{expected_rows}", flush=True)

    if len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} {dataset} rows, found {len(records)}"
        )
    calibration_speakers = {
        row["speaker"] for row in records if row["split"] == "calibration"
    }
    holdout_speakers = {
        row["speaker"] for row in records if row["split"] == "holdout"
    }
    if calibration_speakers & holdout_speakers:
        raise ValueError(f"speaker leakage in {dataset} diagnostic split")
    return records


def positive_affine(raw: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    centered_raw = raw - raw.mean()
    variance = max(float(np.mean(centered_raw**2)), 1e-12)
    scale = max(
        float(np.mean(centered_raw * (target - target.mean())) / variance),
        1e-4,
    )
    return scale, float(target.mean() - scale * raw.mean())


def origin_scale(raw_delta: np.ndarray, exact_delta: np.ndarray) -> float:
    return max(
        float(
            np.dot(raw_delta, exact_delta)
            / max(float(np.dot(raw_delta, raw_delta)), 1e-12)
        ),
        0.0,
    )


def finite_spearman(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    value = float(stats.spearmanr(truth, prediction).statistic)
    return value if math.isfinite(value) else None


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    train_scale: float,
    include_direction: bool,
) -> dict[str, float | int | None]:
    truth_centered = truth - truth.mean()
    prediction_centered = prediction - prediction.mean()
    report: dict[str, float | int | None] = {
        "rows": int(truth.size),
        "spearman": finite_spearman(truth, prediction),
        "normalized_mae": float(
            np.mean(np.abs(prediction - truth)) / max(train_scale, 1e-8)
        ),
        "prediction_on_truth_slope": float(
            np.dot(truth_centered, prediction_centered)
            / max(float(np.dot(truth_centered, truth_centered)), 1e-12)
        ),
    }
    if include_direction:
        nonzero = truth != 0.0
        report["signed_direction_accuracy"] = (
            float(np.mean(np.sign(truth[nonzero]) == np.sign(prediction[nonzero])))
            if bool(nonzero.any())
            else None
        )
    return report


def slices(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {"all": rows}
    for field in ("condition", "view"):
        for value in sorted({str(row[field]) for row in rows}):
            output[f"{field}={value}"] = [
                row for row in rows if row[field] == value
            ]
    return output


def absolute_report(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in records}):
        dataset_rows = [row for row in records if row["dataset"] == dataset]
        calibration = [row for row in dataset_rows if row["split"] == "calibration"]
        holdout = [row for row in dataset_rows if row["split"] == "holdout"]
        output[dataset] = {}
        for component in COMPONENTS:
            output[dataset][component] = {}
            exact_calibration = np.array(
                [float(row["exact_" + component]) for row in calibration]
            )
            train_scale = max(float(exact_calibration.std()), 1e-8)
            for estimator_name in ESTIMATOR_MODES:
                raw_calibration = np.array(
                    [
                        float(row[estimator_name + "_" + component])
                        for row in calibration
                    ]
                )
                scale, bias = positive_affine(raw_calibration, exact_calibration)
                estimator_report: dict[str, Any] = {
                    "positive_affine_scale": scale,
                    "positive_affine_bias": bias,
                    "holdout": {},
                }
                for slice_name, slice_rows in slices(holdout).items():
                    truth = np.array(
                        [float(row["exact_" + component]) for row in slice_rows]
                    )
                    raw = np.array(
                        [
                            float(row[estimator_name + "_" + component])
                            for row in slice_rows
                        ]
                    )
                    estimator_report["holdout"][slice_name] = metrics(
                        truth,
                        scale * raw + bias,
                        train_scale,
                        include_direction=False,
                    )
                output[dataset][component][estimator_name] = estimator_report
    return output


def paired_delta_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["dataset"], row["speaker"], row["condition"], row["view"])
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for group in grouped.values():
        targets = [row for row in group if row["suffix"] == "target_clean"]
        if len(targets) != 1:
            raise ValueError(f"expected one paired target, found {len(targets)}")
        target = targets[0]
        for row in group:
            if row is target:
                continue
            delta: dict[str, Any] = {
                key: row[key]
                for key in (
                    "dataset",
                    "case_number",
                    "split",
                    "speaker",
                    "condition",
                    "view",
                    "suffix",
                    "path",
                )
            }
            for component in COMPONENTS:
                delta["exact_" + component] = float(
                    row["exact_" + component]
                ) - float(target["exact_" + component])
                for estimator_name in ESTIMATOR_MODES:
                    delta[estimator_name + "_" + component] = float(
                        row[estimator_name + "_" + component]
                    ) - float(target[estimator_name + "_" + component])
            output.append(delta)
    return output


def delta_report(delta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in delta_rows}):
        dataset_rows = [row for row in delta_rows if row["dataset"] == dataset]
        calibration = [row for row in dataset_rows if row["split"] == "calibration"]
        holdout = [row for row in dataset_rows if row["split"] == "holdout"]
        output[dataset] = {}
        for component in COMPONENTS:
            output[dataset][component] = {}
            exact_calibration = np.array(
                [float(row["exact_" + component]) for row in calibration]
            )
            train_scale = max(float(exact_calibration.std()), 1e-8)
            for estimator_name in ESTIMATOR_MODES:
                raw_calibration = np.array(
                    [
                        float(row[estimator_name + "_" + component])
                        for row in calibration
                    ]
                )
                scale = origin_scale(raw_calibration, exact_calibration)
                estimator_report: dict[str, Any] = {
                    "origin_scale": scale,
                    "holdout": {},
                }
                for slice_name, slice_rows in slices(holdout).items():
                    truth = np.array(
                        [float(row["exact_" + component]) for row in slice_rows]
                    )
                    raw = np.array(
                        [
                            float(row[estimator_name + "_" + component])
                            for row in slice_rows
                        ]
                    )
                    estimator_report["holdout"][slice_name] = metrics(
                        truth,
                        scale * raw,
                        train_scale,
                        include_direction=True,
                    )
                output[dataset][component][estimator_name] = estimator_report
    return output


def gradient_direction_metrics(
    rows: list[dict[str, float | str]],
    material_threshold: float,
) -> dict[str, Any]:
    truth = np.array([float(row["exact_change"]) for row in rows])
    prediction = np.array([float(row["local_gradient_change"]) for row in rows])
    cosine = np.array([float(row["gradient_direction_cosine"]) for row in rows])
    nonzero = truth != 0.0
    material = np.abs(truth) >= material_threshold
    return {
        "rows": len(rows),
        "material_exact_change_threshold": material_threshold,
        "material_rows": int(material.sum()),
        "spearman": finite_spearman(truth, prediction),
        "signed_direction_accuracy": (
            float(np.mean(np.sign(truth[nonzero]) == np.sign(prediction[nonzero])))
            if bool(nonzero.any())
            else None
        ),
        "material_signed_direction_accuracy": (
            float(
                np.mean(
                    np.sign(truth[material]) == np.sign(prediction[material])
                )
            )
            if bool(material.any())
            else None
        ),
        "median_gradient_direction_cosine": float(np.median(cosine)),
    }


def gradient_report(
    records: list[dict[str, Any]],
    estimators: dict[str, PraatDifferentiableAVQIComponentEstimator],
    absolute: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    target_by_group: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in records:
        if row["suffix"] != "target_clean":
            continue
        key = (row["dataset"], row["speaker"], row["condition"], row["view"])
        if key in target_by_group:
            raise ValueError(f"duplicate target waveform for {key}")
        target_by_group[key] = row

    target_waveforms: dict[str, torch.Tensor] = {}
    output: dict[str, Any] = {}
    direction_rows: list[dict[str, float | str]] = []
    for estimator_name, estimator in estimators.items():
        gradient_norms: dict[str, list[float]] = {
            component: [] for component in COMPONENTS
        }
        estimator_direction_rows: dict[str, list[dict[str, float | str]]] = {
            component: [] for component in COMPONENTS
        }
        for index, row in enumerate(records, start=1):
            audio, sample_rate = sf.read(row["path"], dtype="float32")
            if sample_rate != 16_000 or audio.ndim != 1:
                raise ValueError(f"expected mono 16 kHz waveform: {row['path']}")
            waveform = torch.from_numpy(audio).requires_grad_()
            percent, db, _ = raw_shimmer(
                estimator,
                waveform,
                count_pulses=False,
            )
            values = (percent, db)
            gradients = [
                torch.autograd.grad(
                    value,
                    waveform,
                    retain_graph=component_index == 0,
                )[0]
                for component_index, value in enumerate(values)
            ]
            for component, gradient in zip(COMPONENTS, gradients, strict=True):
                if not torch.isfinite(gradient).all():
                    raise ValueError(
                        f"non-finite {estimator_name} {component} gradient: "
                        f"{row['path']}"
                    )
                gradient_norms[component].append(float(gradient.norm()))

            if row["split"] == "holdout" and row["suffix"] != "target_clean":
                key = (
                    row["dataset"],
                    row["speaker"],
                    row["condition"],
                    row["view"],
                )
                target = target_by_group[key]
                target_path = str(target["path"])
                if target_path not in target_waveforms:
                    target_audio, target_sample_rate = sf.read(
                        target_path,
                        dtype="float32",
                    )
                    if target_sample_rate != 16_000 or target_audio.ndim != 1:
                        raise ValueError(
                            f"expected mono 16 kHz target: {target_path}"
                        )
                    target_waveforms[target_path] = torch.from_numpy(target_audio)
                target_waveform = target_waveforms[target_path]
                if target_waveform.shape != waveform.shape:
                    raise ValueError(
                        "paired waveform lengths differ: "
                        f"{row['path']} vs {target_path}"
                    )
                direction = target_waveform - waveform.detach()
                direction_norm = float(direction.norm())
                for component, gradient in zip(
                    COMPONENTS,
                    gradients,
                    strict=True,
                ):
                    scale = float(
                        absolute[row["dataset"]][component][estimator_name][
                            "positive_affine_scale"
                        ]
                    )
                    local_change = scale * float(torch.dot(gradient, direction))
                    gradient_norm = float(gradient.norm())
                    cosine = float(torch.dot(gradient, direction)) / max(
                        gradient_norm * direction_norm,
                        1e-12,
                    )
                    item: dict[str, float | str] = {
                        "dataset": str(row["dataset"]),
                        "speaker": str(row["speaker"]),
                        "condition": str(row["condition"]),
                        "view": str(row["view"]),
                        "suffix": str(row["suffix"]),
                        "estimator": estimator_name,
                        "component": component,
                        "exact_change": float(target["exact_" + component])
                        - float(row["exact_" + component]),
                        "local_gradient_change": local_change,
                        "gradient_direction_cosine": cosine,
                    }
                    direction_rows.append(item)
                    estimator_direction_rows[component].append(item)
            if index % 12 == 0:
                print(
                    f"gradient {estimator_name}: {index}/{len(records)}",
                    flush=True,
                )

        output[estimator_name] = {
            "component_input_gradients": {},
            "paired_target_direction": {},
        }
        for component in COMPONENTS:
            norms = np.array(gradient_norms[component])
            output[estimator_name]["component_input_gradients"][component] = {
                "waveforms": len(norms),
                "minimum_norm": float(norms.min()),
                "median_norm": float(np.median(norms)),
                "maximum_norm": float(norms.max()),
                "zero_count": int(np.sum(norms == 0.0)),
                "nonfinite_count": int(np.sum(~np.isfinite(norms))),
            }
            material_threshold = 0.1 if component == "shimmer_percent" else 0.01
            output[estimator_name]["paired_target_direction"][component] = (
                gradient_direction_metrics(
                    estimator_direction_rows[component],
                    material_threshold,
                )
            )
    return output, direction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    estimators = {
        name: PraatDifferentiableAVQIComponentEstimator(
            peak_mode="hard",
            shimmer_mode=mode,
        ).eval()
        for name, mode in ESTIMATOR_MODES.items()
    }
    records = load_dataset(
        "realistic",
        args.realistic_root,
        args.realistic_exact_csv,
        REALISTIC_PATTERN,
        40,
        estimators,
    )
    records.extend(
        load_dataset(
            "identity",
            args.identity_root,
            args.identity_exact_csv,
            IDENTITY_PATTERN,
            32,
            estimators,
        )
    )
    deltas = paired_delta_rows(records)
    pulse_counts = np.array(
        [float(row["pulse_chain_v5_pulse_count"]) for row in records]
    )
    absolute = absolute_report(records)
    gradient, gradient_rows = gradient_report(records, estimators, absolute)
    report = {
        "schema_version": "avqi-shimmer-formula-historical-v1",
        "decision": "NONFINAL_DIAGNOSTIC_ONLY",
        "scientific_boundary": {
            "historical_waveforms_only": True,
            "exact_labels_reused_without_rescoring": True,
            "fresh_speaker_disjoint_panel_required": True,
            "generator_loaded": False,
            "generator_optimizer_steps": 0,
            "promotion_authorized": False,
        },
        "source_sha256": {
            "realistic_exact_csv": sha256_file(args.realistic_exact_csv),
            "identity_exact_csv": sha256_file(args.identity_exact_csv),
        },
        "counts": {
            "waveforms": len(records),
            "paired_deltas": len(deltas),
            "pulse_chain_v5_zero_pulse_waveforms": int(
                np.sum(pulse_counts == 0.0)
            ),
            "pulse_chain_v5_minimum_pulses": int(pulse_counts.min()),
            "pulse_chain_v5_median_pulses": float(np.median(pulse_counts)),
        },
        "split": {
            "realistic": (
                "historical fixed calibration speakers ÄHH10, PD08; "
                "holdout speakers PD_51, V55"
            ),
            "identity": (
                "historical odd case numbers calibration; even case numbers "
                "holdout; speaker-disjoint"
            ),
        },
        "absolute": absolute,
        "paired_delta": delta_report(deltas),
        "gradient": gradient,
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "predictions.csv", records)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    write_csv(args.output_dir / "gradient_directions.csv", gradient_rows)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
