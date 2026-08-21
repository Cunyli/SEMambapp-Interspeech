#!/usr/bin/env python3
"""Historical exact-rescored pilot for the deployable Route C Shimmer path.

The v6 estimator locates its own detached pulse topology and backpropagates
through the live asymmetric-Hann RMS amplitude tier.  Exact Praat labels are
used only before and after a bounded waveform step.  This SV-only historical
diagnostic cannot authorize AVQI-T2 generator training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator
from scripts.evaluate_avqi_shimmer_pulse_oracle_pilot import (
    ALPHA_GRID,
    CALIBRATION_SPEAKERS,
    COMPONENTS,
    HOLDOUT_SPEAKERS,
    IDENTITY_PATTERN,
    MATERIAL_GAP_THRESHOLD,
    PROMOTION_REDUCTION_THRESHOLD,
    REALISTIC_PATTERN,
    SAMPLE_RATE,
    SV_METRIC_SAMPLES,
    aggregate,
    load_records,
    read_waveform,
    run_exact_batch,
    safety_metrics,
    sha256_file,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realistic-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument(
        "--parselmouth-root",
        type=Path,
        help="Optional isolated Parselmouth site-packages root for exact Python.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def internal_metric_waveform(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    waveform: torch.Tensor,
) -> torch.Tensor:
    prepared = estimator._prepare(waveform)
    if prepared.numel() > SV_METRIC_SAMPLES:
        prepared = prepared[-SV_METRIC_SAMPLES:]
    return prepared


def internal_proxy(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    waveform: torch.Tensor,
) -> torch.Tensor:
    prepared = internal_metric_waveform(estimator, waveform)
    return torch.stack(estimator._praat_pulse_chain_shimmer(prepared))


@torch.inference_mode()
def internal_pulses(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    waveform: torch.Tensor,
) -> np.ndarray:
    prepared = internal_metric_waveform(estimator, waveform)
    return estimator._praat_shimmer_pulse_chain(prepared).cpu().numpy()


def pulse_alignment(
    predicted: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float | int | None]:
    if predicted.size == 0 or exact.size == 0:
        return {
            "internal_pulse_count": int(predicted.size),
            "exact_pulse_count": int(exact.size),
            "internal_pulses_within_5_samples_fraction": None,
            "internal_to_exact_median_nearest_samples": None,
        }
    nearest = np.min(np.abs(predicted[:, None] - exact[None, :]), axis=1)
    return {
        "internal_pulse_count": int(predicted.size),
        "exact_pulse_count": int(exact.size),
        "internal_pulses_within_5_samples_fraction": float(np.mean(nearest <= 5.0)),
        "internal_to_exact_median_nearest_samples": float(np.median(nearest)),
    }


def safe_output_name(record: dict[str, Any]) -> str:
    raw = (
        f"{record['dataset']}__{record['case_number']:02d}__"
        f"{record['speaker']}__{record['suffix']}__internal_v6_step.wav"
    )
    return re.sub(r"[^0-9A-Za-z._ÄÖÅäöåÜüÉé_-]", "_", raw)


def split_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["material_rows"] >= 5
        and summary["percent_median_normalized_gap_reduction_material"]
        >= PROMOTION_REDUCTION_THRESHOLD
        and summary["percent_improvement_rate_material"] >= 0.8
        and summary["db_median_normalized_gap_reduction_material"] >= 0.0
        and summary["identity_zero_gap_rows"]
        == summary["identity_zero_gap_rows_unchanged"]
        and summary["minimum_cosine_similarity"] >= 0.99999
        and summary["maximum_residual_rms_db"] <= -50.0
        and summary["maximum_abs_low_0_80hz_energy_change_db"] <= 0.1
        and summary["clipping_samples"] == 0
    )


def locator_aggregate(
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    fractions = [
        float(row["internal_pulses_within_5_samples_fraction"])
        for row in selected
        if row["internal_pulses_within_5_samples_fraction"] is not None
    ]
    distances = [
        float(row["internal_to_exact_median_nearest_samples"])
        for row in selected
        if row["internal_to_exact_median_nearest_samples"] is not None
    ]
    count_ratio = [
        float(row["internal_pulse_count"]) / max(float(row["exact_pulse_count"]), 1.0)
        for row in selected
    ]
    return {
        "waveforms": len(selected),
        "zero_internal_pulse_waveforms": sum(
            int(row["internal_pulse_count"]) == 0 for row in selected
        ),
        "median_internal_pulses_within_5_samples_fraction": float(
            np.median(fractions)
        ),
        "median_internal_to_exact_nearest_samples": float(np.median(distances)),
        "median_internal_to_exact_pulse_count_ratio": float(
            np.median(count_ratio)
        ),
    }


def forward_error_aggregate(
    records: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    selected = [record for record in records if record["split"] == split]
    percent_error = [
        abs(float(record["proxy"][0] - record["exact"][0]))
        for record in selected
    ]
    db_error = [
        abs(float(record["proxy"][1] - record["exact"][1]))
        for record in selected
    ]
    return {
        "waveforms": len(selected),
        "shimmer_percent_median_absolute_error": float(np.median(percent_error)),
        "shimmer_percent_maximum_absolute_error": max(percent_error),
        "shimmer_db_median_absolute_error": float(np.median(db_error)),
        "shimmer_db_maximum_absolute_error": max(db_error),
    }


def gradient_aggregate(
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    norms = np.asarray([float(row["gradient_rms"]) for row in selected])
    return {
        "waveforms": int(norms.size),
        "finite_waveforms": int(np.isfinite(norms).sum()),
        "zero_waveforms": int(np.sum(norms == 0.0)),
        "minimum_rms": float(norms.min()),
        "median_rms": float(np.median(norms)),
        "maximum_rms": float(norms.max()),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Route C Shimmer internal-v6 historical pilot",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "The pulse locator is deployable and detached; exact Praat is used only "
        "for before/after scoring. This SV-only historical pilot does not "
        "authorize generator training.",
        "",
        "| Split | Material cases | Exact Shimmer % median normalized reduction | Improvement rate |",
        "|---|---:|---:|---:|",
    ]
    for split in ("calibration", "holdout"):
        summary = report["aggregates"][split]
        lines.append(
            f"| {split} | {summary['material_rows']} | "
            f"{summary['percent_median_normalized_gap_reduction_material']:.6f} | "
            f"{summary['percent_improvement_rate_material']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Both historical splits must pass the frozen waveform and safety "
            "gates before this locator can move to a fresh speaker-disjoint "
            "CS/SV panel.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    if args.parselmouth_root is not None and not args.parselmouth_root.is_dir():
        raise FileNotFoundError(f"missing Parselmouth root: {args.parselmouth_root}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    output_root = args.output_dir / "outputs"
    waveform_root = output_root / "waveforms"
    waveform_root.mkdir(parents=True)

    records = [
        *load_records("realistic", args.realistic_root, REALISTIC_PATTERN),
        *load_records("identity", args.identity_root, IDENTITY_PATTERN),
    ]
    if len(records) != 36:
        raise ValueError(f"expected 36 frozen SV waveforms, found {len(records)}")
    calibration_speakers = {
        row["speaker"] for row in records if row["split"] == "calibration"
    }
    holdout_speakers = {
        row["speaker"] for row in records if row["split"] == "holdout"
    }
    if calibration_speakers != CALIBRATION_SPEAKERS:
        raise ValueError("calibration speaker contract drift")
    if holdout_speakers != HOLDOUT_SPEAKERS:
        raise ValueError("holdout speaker contract drift")
    if calibration_speakers & holdout_speakers:
        raise ValueError("speaker leakage in internal-v6 diagnostic")

    exact_before = run_exact_batch(
        [{"id": row["id"], "path": str(row["path"])} for row in records],
        args.exact_python,
        args.parselmouth_root,
        include_pulses=True,
    )
    exact_index = {row["id"]: row for row in exact_before["rows"]}
    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_path_v6",
    ).eval()
    for record in records:
        exact = exact_index[record["id"]]
        if exact["metric_sample_count"] != min(record["audio"].size, SV_METRIC_SAMPLES):
            raise ValueError(f"SV metric crop drift: {record['path']}")
        record["exact"] = np.array(
            [exact["shimmer_percent"], exact["shimmer_db"]],
            dtype=np.float64,
        )
        waveform = torch.from_numpy(record["audio"])
        with torch.inference_mode():
            record["proxy"] = internal_proxy(estimator, waveform).numpy()
        predicted_pulses = internal_pulses(estimator, waveform)
        exact_pulses = np.asarray(exact["pulse_positions_samples"], dtype=np.float64)
        record["locator"] = pulse_alignment(predicted_pulses, exact_pulses)

    unique_calibration = {
        record["sha256"]: record
        for record in records
        if record["split"] == "calibration"
    }
    calibration_exact = np.stack(
        [record["exact"] for record in unique_calibration.values()]
    )
    target_scale = np.maximum(calibration_exact.std(axis=0), 1e-8)
    target_index = {
        (record["dataset"], record["case_number"], record["speaker"]): record
        for record in records
        if record["suffix"] == "target_clean"
    }
    cases = [record for record in records if record["suffix"] != "target_clean"]
    if len(cases) != 28:
        raise ValueError(f"expected 28 non-target cases, found {len(cases)}")

    output_items: list[dict[str, str]] = []
    for record in cases:
        target = target_index[
            (record["dataset"], record["case_number"], record["speaker"])
        ]
        waveform = torch.from_numpy(record["audio"].copy()).requires_grad_(True)
        current_proxy = internal_proxy(estimator, waveform)
        target_proxy = torch.from_numpy(target["proxy"]).to(waveform)
        loss = (
            (current_proxy[0] - target_proxy[0]) / float(target_scale[0])
        ).square()
        gradient = torch.autograd.grad(loss, waveform)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite internal-v6 gradient: {record['path']}")
        gradient_rms = gradient.square().mean().sqrt()
        base_rms = waveform.detach().square().mean().sqrt()
        base_db_gap = abs(float(current_proxy[1] - target_proxy[1]))
        candidates: list[tuple[float, float, np.ndarray]] = []
        for alpha in ALPHA_GRID:
            if alpha == 0.0 or float(gradient_rms) <= 1e-15:
                candidate_tensor = waveform.detach()
            else:
                candidate_tensor = waveform.detach() - (
                    alpha * base_rms * gradient / gradient_rms
                )
            if not torch.isfinite(candidate_tensor).all():
                continue
            if float(candidate_tensor.abs().max()) >= 1.0:
                continue
            with torch.inference_mode():
                candidate_proxy = internal_proxy(estimator, candidate_tensor)
            candidate_db_gap = abs(float(candidate_proxy[1] - target_proxy[1]))
            if candidate_db_gap > base_db_gap + 1e-5:
                continue
            candidates.append(
                (
                    abs(float(candidate_proxy[0] - target_proxy[0])),
                    alpha,
                    candidate_tensor.numpy(),
                )
            )
        if not candidates:
            raise RuntimeError(f"no safe line-search candidate: {record['path']}")
        _, selected_alpha, selected_audio = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        output_path = waveform_root / safe_output_name(record)
        sf.write(output_path, selected_audio, SAMPLE_RATE, subtype="PCM_24")
        stored_audio = read_waveform(output_path)
        with torch.inference_mode():
            stored_proxy = internal_proxy(
                estimator,
                torch.from_numpy(stored_audio),
            ).numpy()
        record["target"] = target
        record["loss"] = float(loss)
        record["gradient_rms"] = float(gradient_rms)
        record["selected_alpha"] = selected_alpha
        record["selected_audio"] = stored_audio
        record["selected_proxy"] = stored_proxy
        record["output_path"] = output_path.resolve()
        record["safety"] = safety_metrics(record["audio"], stored_audio)
        output_items.append({"id": record["id"], "path": str(output_path.resolve())})

    exact_after = run_exact_batch(
        output_items,
        args.exact_python,
        args.parselmouth_root,
        include_pulses=False,
    )
    if (
        exact_after["parselmouth_version"] != exact_before["parselmouth_version"]
        or exact_after["praat_version"] != exact_before["praat_version"]
    ):
        raise ValueError("exact scorer version drift within pilot")
    exact_after_index = {row["id"]: row for row in exact_after["rows"]}

    csv_rows: list[dict[str, Any]] = []
    for record in cases:
        target = record["target"]
        after_item = exact_after_index[record["id"]]
        exact_after_components = np.array(
            [after_item["shimmer_percent"], after_item["shimmer_db"]],
            dtype=np.float64,
        )
        percent_gap_before = abs(float(record["exact"][0] - target["exact"][0]))
        percent_gap_after = abs(float(exact_after_components[0] - target["exact"][0]))
        db_gap_before = abs(float(record["exact"][1] - target["exact"][1]))
        db_gap_after = abs(float(exact_after_components[1] - target["exact"][1]))
        material = percent_gap_before / target_scale[0] > MATERIAL_GAP_THRESHOLD
        csv_rows.append(
            {
                "dataset": record["dataset"],
                "case_number": record["case_number"],
                "split": record["split"],
                "speaker": record["speaker"],
                "label": record["label"],
                "sample_group": record["sample_group"],
                "condition": record["condition"],
                "view": record["view"],
                "suffix": record["suffix"],
                "input_path": str(record["path"]),
                "output_path": str(record["output_path"]),
                "input_sha256": record["sha256"],
                "output_sha256": sha256_file(record["output_path"]),
                **record["locator"],
                "selected_alpha": record["selected_alpha"],
                "gradient_rms": record["gradient_rms"],
                "proxy_loss_before": record["loss"],
                "proxy_percent_before": float(record["proxy"][0]),
                "proxy_percent_target": float(target["proxy"][0]),
                "proxy_percent_after": float(record["selected_proxy"][0]),
                "proxy_db_before": float(record["proxy"][1]),
                "proxy_db_target": float(target["proxy"][1]),
                "proxy_db_after": float(record["selected_proxy"][1]),
                "proxy_forward_percent_abs_error_before": abs(
                    float(record["proxy"][0] - record["exact"][0])
                ),
                "proxy_forward_db_abs_error_before": abs(
                    float(record["proxy"][1] - record["exact"][1])
                ),
                "exact_percent_before": float(record["exact"][0]),
                "exact_percent_target": float(target["exact"][0]),
                "exact_percent_after": float(exact_after_components[0]),
                "exact_percent_gap_before": percent_gap_before,
                "exact_percent_gap_after": percent_gap_after,
                "exact_percent_normalized_gap_reduction": (
                    percent_gap_before - percent_gap_after
                )
                / target_scale[0],
                "material_percent_gap": int(material),
                "exact_db_before": float(record["exact"][1]),
                "exact_db_target": float(target["exact"][1]),
                "exact_db_after": float(exact_after_components[1]),
                "exact_db_gap_before": db_gap_before,
                "exact_db_gap_after": db_gap_after,
                "exact_db_normalized_gap_reduction": (db_gap_before - db_gap_after)
                / target_scale[1],
                **record["safety"],
            }
        )

    aggregates = {split: aggregate(csv_rows, split) for split in ("calibration", "holdout")}
    gates = {split: split_gate(aggregates[split]) for split in aggregates}
    historical_pass = all(gates.values())
    report = {
        "decision": (
            "PASS_DEPLOYABLE_PATH_HISTORICAL_PILOT_ONLY"
            if historical_pass
            else "FAIL_DEPLOYABLE_PATH_HISTORICAL_PILOT"
        ),
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "route": "C_shimmer_internal_praat_candidate_path_v6",
        "panel_status": "historical_sv_only_nonfinal",
        "sample_rate": SAMPLE_RATE,
        "sv_metric_order": "whole-waveform high-pass, then final 3 seconds",
        "calibration_speakers": sorted(CALIBRATION_SPEAKERS),
        "holdout_speakers": sorted(HOLDOUT_SPEAKERS),
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selection": (
            "per-waveform internal-surrogate line search with pulse topology "
            "recomputed for each candidate and proxy Shimmer-dB non-worsening"
        ),
        "material_gap_threshold_normalized": MATERIAL_GAP_THRESHOLD,
        "promotion_reduction_threshold": PROMOTION_REDUCTION_THRESHOLD,
        "target_scale_from_unique_calibration_waveforms": {
            component: float(target_scale[index])
            for index, component in enumerate(COMPONENTS)
        },
        "exact_scorer": {
            "python": str(args.exact_python.resolve()),
            "parselmouth_root": (
                None
                if args.parselmouth_root is None
                else str(args.parselmouth_root.resolve())
            ),
            "parselmouth_version": exact_before["parselmouth_version"],
            "praat_version": exact_before["praat_version"],
            "highpass": "Filter (stop Hann band), 0, 34, 0.1",
            "point_process": "To PointProcess (periodic, cc), 50, 400",
            "shimmer_arguments": [0, 0, 0.0001, 0.02, 1.3, 1.6],
        },
        "anti_shortcut": {
            "exact_pulses_supplied_to_estimator": False,
            "exact_after_used_during_alpha_selection": False,
            "alpha_selection_uses_only_internal_proxy": True,
            "target_is_same_speaker_clean_pathological_proxy": True,
        },
        "split_gates": gates,
        "aggregates": aggregates,
        "forward_accuracy": {
            split: forward_error_aggregate(records, split)
            for split in ("calibration", "holdout")
        },
        "target_gap_gradient": {
            split: gradient_aggregate(csv_rows, split)
            for split in ("calibration", "holdout")
        },
        "pulse_locator": {
            split: locator_aggregate(csv_rows, split)
            for split in ("calibration", "holdout")
        },
        "artifacts": {
            "case_metrics_csv": "case_metrics.csv",
            "waveform_directory": "waveforms",
        },
    }
    write_csv(output_root / "case_metrics.csv", csv_rows)
    with (output_root / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    (output_root / "README.md").write_text(
        build_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
