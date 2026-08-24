#!/usr/bin/env python3
"""Adjudicate Route C LTAS-slope scorer promotion.

This audit binds the frozen Route C screen, calibration/holdout exact-relative
gate runs, and the sealed SVD authority panel.  Exact-Praat repeatability is
measured only on the calibration speakers used to freeze the gate.  Holdout
and SVD remain validation-only.  A pass authorizes a new bounded-waveform
pilot for LTAS slope; it never authorizes generator training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from scripts.evaluate_avqi_ltas_slope_gate_alignment import (
    AUTHORITY_RATIO_RANGE,
    CURRENT_ABSOLUTE_LOWPASS_MIN,
    DIRECTION_AGREEMENT_MIN,
    INVARIANCE_DISTANCE_MAX,
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


SLOPE_COMPONENT = "slope"
EXPECTED_ARCHITECTURE = "direct_praat_hard_shimmer_pulse_path_v6"
BASE_EXACT_MATERIAL_DISTANCE_MIN = 0.02
REPEATABILITY_RUNS = 3
REPEATABILITY_NOISE_MULTIPLIER = 10.0
REPEATABILITY_NORMALIZED_RANGE_MAX = 0.002
EXACT_REPLAY_ABS_TOLERANCE = 1e-5
PROMOTION_PASS = "GO_BOUNDED_LTAS_SLOPE_WAVEFORM_PILOT"
PROMOTION_FAIL = "NO_GO_LTAS_SLOPE_KEEP_CANDIDATE"
SCREEN_DECISION = "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
CALIBRATION_DECISION = "SUPPORTED_FREEZE_EXACT_RELATIVE_GATE_FOR_HOLDOUT"
HOLDOUT_DECISION = "PASS_EXACT_RELATIVE_LTAS_GATE_EXPERIMENT_NO_PRODUCTION_CHANGE"
SVD_DECISION = "PASS_EXTERNAL_SVD_LTAS_AUTHORITY_PANEL_NO_PRODUCTION_CHANGE"
INTERNAL_REQUIRED_SLICES = (
    "clean",
    "phone",
    "cs",
    "sv",
    "healthy",
    "patient",
)
PATHOLOGY_EXTERNAL_REQUIRED_SLICES = (
    "condition=clean",
    "condition=rir_only",
    "condition=snr30",
    "condition=snr20",
    "condition=snr15",
    "condition=snr10",
    "view=cs",
    "view=sv",
    "label=healthy",
    "label=patient",
    "view=sv&sample_group=pathological_severe",
)
VCTK_REQUIRED_SLICES = (
    "condition=clean",
    "condition=rir_only",
    "condition=snr20",
    "condition=snr10",
)
EXACT_SOURCE_FILES = {
    "python_version.py": "python_version.py",
    "praat_version.py": "praat_version.py",
    "praat_scripts/highpass_filter.praat": "praat_scripts/highpass_filter.praat",
    "praat_scripts/length_normalize_sv.praat": (
        "praat_scripts/length_normalize_sv.praat"
    ),
    "praat_scripts/voiced_segment_extraction.praat": (
        "praat_scripts/voiced_segment_extraction.praat"
    ),
    "praat_scripts/slope.praat": "praat_scripts/slope.praat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "calibration", "holdout", "svd"):
        parser.add_argument(f"--{name}-report", type=Path, required=True)
        parser.add_argument(f"--{name}-report-sha256", required=True)
        parser.add_argument(f"--{name}-receipt", type=Path, required=True)
        parser.add_argument(f"--{name}-receipt-sha256", required=True)
    parser.add_argument("--svd-panel-seal", type=Path, required=True)
    parser.add_argument("--svd-panel-seal-sha256", required=True)
    parser.add_argument("--svd-seal-receipt", type=Path, required=True)
    parser.add_argument("--svd-seal-receipt-sha256", required=True)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--python-version-sha256", required=True)
    parser.add_argument("--praat-version-sha256", required=True)
    parser.add_argument("--highpass-praat-sha256", required=True)
    parser.add_argument("--sv-length-praat-sha256", required=True)
    parser.add_argument("--cs-voiced-praat-sha256", required=True)
    parser.add_argument("--slope-praat-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def load_bound_report(
    report_path: Path,
    report_hash: str,
    receipt_path: Path,
    receipt_hash: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_hash(report_path, report_hash, f"{label} report")
    validate_hash(receipt_path, receipt_hash, f"{label} receipt")
    report = load_json(report_path)
    receipt = load_json(receipt_path)
    bound_hash = receipt.get("artifact_sha256", {}).get(report_path.name)
    if bound_hash != report_hash:
        raise ValueError(f"{label} receipt does not bind its report")
    if receipt.get("decision") != report.get("decision"):
        raise ValueError(f"{label} receipt decision differs from its report")
    return report, receipt


def require_zero_training(payload: dict[str, Any], label: str) -> None:
    if payload.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} contains generator optimizer steps")
    if payload.get("formal_pathology_training_submitted") is not False:
        raise ValueError(f"{label} formal-training boundary differs")


def component_pass(report: dict[str, Any], component: str = SLOPE_COMPONENT) -> bool:
    return report.get(component, {}).get("decision") == "PASS"


def covered_component_pass(
    report: dict[str, Any],
    required_slices: tuple[str, ...],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "primary_coverage": report.get("primary_coverage", {}).get("decision")
        == "PASS",
        "primary": component_pass(report.get("primary", {})),
    }
    for name in required_slices:
        checks[f"coverage:{name}"] = (
            report.get("slice_coverage", {}).get(name, {}).get("decision")
            == "PASS"
        )
        checks[f"metric:{name}"] = component_pass(
            report.get("slices", {}).get(name, {})
        )
    return all(checks.values()), checks


def screen_gate_audit(screen: dict[str, Any]) -> dict[str, Any]:
    if screen.get("decision") != SCREEN_DECISION:
        raise ValueError("unexpected Route C screen decision")
    require_zero_training(screen, "Route C screen")
    route = screen.get("routes", {}).get("direct_differentiable_estimator", {})
    if route.get("selected_architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Route C screen architecture differs")

    metrics = route.get("metrics", {})
    internal_checks = {
        "primary": component_pass(metrics.get("primary", {})),
        **{
            f"slice:{name}": component_pass(
                metrics.get("slices", {}).get(name, {})
            )
            for name in INTERNAL_REQUIRED_SLICES
        },
    }
    primary = metrics.get("primary", {}).get(SLOPE_COMPONENT, {})
    if "delta_spearman_ge_0_60" not in primary.get("gates", {}):
        raise ValueError("Route C slope primary gate lacks paired delta")

    legacy = route.get("anti_shortcut", {}).get("components", {}).get(
        SLOPE_COMPONENT,
        {},
    )
    legacy_false_gates = sorted(
        name for name, passed in legacy.get("gates", {}).items() if not passed
    )
    legacy_expected_failure = legacy_false_gates == ["lowpass_moves_away"]
    legacy_lowpass = legacy.get("mean_standardized_distance", {}).get(
        "lowpass_3khz"
    )
    if not isinstance(legacy_lowpass, (float, int)):
        raise ValueError("Route C screen lacks legacy LTAS low-pass distance")

    gradient = route.get("gradient", {})
    gradient_component = gradient.get("component_input_gradients", {}).get(
        SLOPE_COMPONENT,
        {},
    )
    transfer_component = route.get("training_segment_transfer", {}).get(
        "components",
        {},
    ).get(SLOPE_COMPONENT, {})
    pathology_pass, pathology_checks = covered_component_pass(
        route.get("external_enhancement_stress", {}),
        PATHOLOGY_EXTERNAL_REQUIRED_SLICES,
    )
    vctk_pass, vctk_checks = covered_component_pass(
        route.get("vctk_external_own_target_stress", {}),
        VCTK_REQUIRED_SLICES,
    )
    gates = {
        "internal_accuracy_calibration_delta_and_slices": all(
            internal_checks.values()
        ),
        "legacy_only_failure_is_absolute_lowpass": legacy_expected_failure,
        "legacy_lowpass_below_absolute_0_10": (
            float(legacy_lowpass) < CURRENT_ABSOLUTE_LOWPASS_MIN
        ),
        "component_input_gradient": (
            gradient.get("decision") == "PASS"
            and gradient_component.get("decision") == "PASS"
        ),
        "training_segment_transfer": transfer_component.get("decision")
        == "PASS",
        "pathology_external_all_required_slices": pathology_pass,
        "vctk_external_all_required_slices": vctk_pass,
    }
    return {
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "internal_checks": internal_checks,
        "pathology_external_checks": pathology_checks,
        "vctk_external_checks": vctk_checks,
        "internal_primary": primary,
        "legacy_anti_shortcut": legacy,
        "legacy_false_gates": legacy_false_gates,
        "component_input_gradient": gradient_component,
        "training_segment_transfer": transfer_component,
        "pathology_external": route.get("external_enhancement_stress", {}),
        "vctk_external": route.get("vctk_external_own_target_stress", {}),
        "screen_eligible_components": route.get("eligible_components", []),
    }


def exact_relative_gate_with_floor(
    summaries: dict[str, Any],
    exact_material_floor: float,
) -> dict[str, Any]:
    lowpass = summaries["lowpass_3khz"]
    gain = summaries["gain_minus12db"]
    shift = summaries["circular_shift_100ms"]
    ratio = float(lowpass["candidate_to_exact_distance_ratio"])
    gates = {
        "exact_lowpass_is_material": (
            float(lowpass["exact_mean_standardized_distance"])
            >= exact_material_floor
        ),
        "candidate_matches_exact_response_ratio": (
            AUTHORITY_RATIO_RANGE[0] <= ratio <= AUTHORITY_RATIO_RANGE[1]
        ),
        "signed_direction_agreement": (
            float(lowpass["signed_direction_agreement"])
            >= DIRECTION_AGREEMENT_MIN
        ),
        "gain_nearly_invariant": (
            float(gain["candidate_mean_standardized_distance"])
            <= INVARIANCE_DISTANCE_MAX
        ),
        "circular_shift_nearly_invariant": (
            float(shift["candidate_mean_standardized_distance"])
            <= INVARIANCE_DISTANCE_MAX
        ),
        "candidate_lowpass_exceeds_controls": (
            float(lowpass["candidate_mean_standardized_distance"])
            > max(
                float(gain["candidate_mean_standardized_distance"]),
                float(shift["candidate_mean_standardized_distance"]),
            )
        ),
        "exact_lowpass_exceeds_controls": (
            float(lowpass["exact_mean_standardized_distance"])
            > max(
                float(gain["exact_mean_standardized_distance"]),
                float(shift["exact_mean_standardized_distance"]),
            )
        ),
    }
    return {
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "exact_material_floor": exact_material_floor,
        "current_absolute_gate_passes": (
            float(lowpass["candidate_mean_standardized_distance"])
            >= CURRENT_ABSOLUTE_LOWPASS_MIN
        ),
        "summary": summaries,
    }


def repeatability_material_floor(normalized_p99_range: float) -> float:
    if not math.isfinite(normalized_p99_range) or normalized_p99_range < 0.0:
        raise ValueError("invalid exact repeatability range")
    return max(
        BASE_EXACT_MATERIAL_DISTANCE_MIN,
        REPEATABILITY_NOISE_MULTIPLIER * normalized_p99_range,
    )


def calibration_repeatability(
    calibration: dict[str, Any],
    label_rows: list[dict[str, str]],
    exact_python: Path,
    output_dir: Path,
    train_scale: float,
) -> dict[str, Any]:
    selected_keys = {
        (str(row["speaker_id"]), str(row["sample_id"]))
        for row in calibration.get("rows", [])
    }
    eligible = {
        (row["speaker_id"], row["sample_id"]): row
        for row in label_rows
        if row["split"] == "surrogate_calibration"
        and row["view"] == "sv"
        and row["label"] == "patient"
        and row["condition_id"] == "clean"
        and row["scoring_status"] == "ok"
    }
    if not selected_keys or not selected_keys <= eligible.keys():
        raise ValueError("calibration repeatability selection differs from label bank")

    waveform_dir = output_dir / "repeatability_waveforms"
    waveform_dir.mkdir()
    items: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []
    report_by_key = {
        (str(row["speaker_id"]), str(row["sample_id"])): row
        for row in calibration["rows"]
    }
    for speaker_id, sample_id in sorted(selected_keys):
        label_row = eligible[(speaker_id, sample_id)]
        clean = load_audio(label_row)
        lowpass = lowpass_3khz(clean)
        safe_name = f"{speaker_id}_{sample_id}".replace("/", "_").replace(":", "_")
        lowpass_path = waveform_dir / f"{safe_name}_lowpass_3khz.wav"
        sf.write(lowpass_path, lowpass, SAMPLE_RATE, subtype="PCM_24")
        base_id = f"{speaker_id}:{sample_id}"
        items.extend(
            (
                {"id": f"{base_id}:clean", "path": label_row["sv_path"]},
                {
                    "id": f"{base_id}:lowpass_3khz",
                    "path": str(lowpass_path.resolve()),
                },
            )
        )
        cases.append(
            {
                "speaker_id": speaker_id,
                "sample_id": sample_id,
                "clean_path": label_row["sv_path"],
                "clean_sha256": label_row["sv_sha256"],
                "lowpass_path": str(lowpass_path.resolve()),
                "lowpass_sha256": sha256_file(lowpass_path),
                "reference_exact": report_by_key[(speaker_id, sample_id)][
                    "exact"
                ],
            }
        )

    payloads = [run_exact(items, exact_python) for _ in range(REPEATABILITY_RUNS)]
    runtimes = [
        {
            "parselmouth_version": payload["parselmouth_version"],
            "praat_version": payload["praat_version"],
        }
        for payload in payloads
    ]
    if any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise ValueError("exact runtime identity drifted across repeats")
    expected_ids = [item["id"] for item in items]
    run_indices = []
    for payload in payloads:
        if [row["id"] for row in payload["rows"]] != expected_ids:
            raise ValueError("exact repeatability row order or coverage drift")
        run_indices.append({row["id"]: float(row["slope"]) for row in payload["rows"]})

    raw_normalized_ranges: list[float] = []
    delta_normalized_ranges: list[float] = []
    replay_errors: list[float] = []
    for case in cases:
        base_id = f"{case['speaker_id']}:{case['sample_id']}"
        clean_values = np.asarray(
            [run[f"{base_id}:clean"] for run in run_indices],
            dtype=np.float64,
        )
        lowpass_values = np.asarray(
            [run[f"{base_id}:lowpass_3khz"] for run in run_indices],
            dtype=np.float64,
        )
        delta_values = lowpass_values - clean_values
        clean_range = float(np.ptp(clean_values)) / train_scale
        lowpass_range = float(np.ptp(lowpass_values)) / train_scale
        delta_range = float(np.ptp(delta_values)) / train_scale
        raw_normalized_ranges.extend((clean_range, lowpass_range))
        delta_normalized_ranges.append(delta_range)
        reference = case.pop("reference_exact")
        replay_error = max(
            abs(float(clean_values[0]) - float(reference["clean"])),
            abs(float(lowpass_values[0]) - float(reference["lowpass_3khz"])),
        )
        replay_errors.append(replay_error)
        case.update(
            {
                "clean_values": clean_values.tolist(),
                "lowpass_values": lowpass_values.tolist(),
                "signed_delta_values": delta_values.tolist(),
                "standardized_distance_values": (
                    np.abs(delta_values) / train_scale
                ).tolist(),
                "normalized_raw_range_max": max(clean_range, lowpass_range),
                "normalized_delta_range": delta_range,
                "replay_max_abs_error": replay_error,
            }
        )

    all_ranges = np.asarray(
        raw_normalized_ranges + delta_normalized_ranges,
        dtype=np.float64,
    )
    normalized_p99_range = float(np.quantile(all_ranges, 0.99))
    normalized_max_range = float(all_ranges.max())
    material_floor = repeatability_material_floor(normalized_p99_range)
    gates = {
        "finite": bool(np.isfinite(all_ranges).all()),
        "three_repeats": len(payloads) == REPEATABILITY_RUNS,
        "normalized_p99_range_le_0_002": (
            normalized_p99_range <= REPEATABILITY_NORMALIZED_RANGE_MAX
        ),
        "replays_frozen_calibration_exact": (
            max(replay_errors) <= EXACT_REPLAY_ABS_TOLERANCE
        ),
    }
    return {
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "selection_role": "gate-setting calibration speakers only",
        "validation_rows_used_to_set_floor": 0,
        "runs": REPEATABILITY_RUNS,
        "speaker_count": len(cases),
        "item_count_per_run": len(items),
        "runtime": runtimes[0],
        "train_slope_scale": train_scale,
        "normalized_p99_range": normalized_p99_range,
        "normalized_max_range": normalized_max_range,
        "normalized_repeatability_limit": REPEATABILITY_NORMALIZED_RANGE_MAX,
        "material_floor_formula": "max(0.02, 10 * normalized_p99_range)",
        "frozen_exact_material_floor": material_floor,
        "max_replay_abs_error": max(replay_errors),
        "cases": cases,
    }


def validate_exact_sources(args: argparse.Namespace) -> dict[str, str]:
    code_root = args.avqi_code_root / "avqi_code"
    expected = {
        "python_version.py": args.python_version_sha256,
        "praat_version.py": args.praat_version_sha256,
        "praat_scripts/highpass_filter.praat": args.highpass_praat_sha256,
        "praat_scripts/length_normalize_sv.praat": args.sv_length_praat_sha256,
        "praat_scripts/voiced_segment_extraction.praat": (
            args.cs_voiced_praat_sha256
        ),
        "praat_scripts/slope.praat": args.slope_praat_sha256,
    }
    return {
        name: validate_hash(code_root / relative, expected[name], f"exact {name}")
        for name, relative in EXACT_SOURCE_FILES.items()
    }


def alignment_gate_audit(
    report: dict[str, Any],
    expected_decision: str,
    expected_split: str,
    exact_material_floor: float,
) -> dict[str, Any]:
    if report.get("decision") != expected_decision:
        raise ValueError(f"unexpected {expected_split} gate decision")
    require_zero_training(report, f"{expected_split} gate report")
    if report.get("selection", {}).get("split") != expected_split:
        raise ValueError(f"{expected_split} gate split differs")
    if report.get("production_gate_changed") is not False:
        raise ValueError(f"{expected_split} report changed the production gate")
    summaries = report.get("modes", {}).get("candidate_frozen_full", {})
    gate = exact_relative_gate_with_floor(summaries, exact_material_floor)
    return {
        "decision": gate["decision"],
        "selection": report["selection"],
        "train_slope_scale": report["train_slope_scale_std_surrogate_train"],
        "gate": gate,
        "legacy_gate_reported_only": True,
    }


def svd_gate_audit(
    report: dict[str, Any],
    seal: dict[str, Any],
    exact_material_floor: float,
) -> dict[str, Any]:
    if report.get("decision") != SVD_DECISION:
        raise ValueError("unexpected SVD authority-panel decision")
    require_zero_training(report, "SVD authority report")
    require_zero_training(seal, "SVD panel seal")
    if report.get("panel_seal_sha256") != sha256_file(Path(seal["_path"])):
        raise ValueError("SVD report does not bind the supplied panel seal")
    scopes = ("overall", "cs", "sv")
    scope_checks: dict[str, dict[str, bool]] = {}
    exact_relative: dict[str, Any] = {}
    for scope in scopes:
        exact_relative[scope] = exact_relative_gate_with_floor(
            report["anti_shortcut_exact_relative"][scope]["summary"],
            exact_material_floor,
        )
        scope_checks[scope] = {
            "level_alignment": report["level_alignment"][scope]["decision"]
            == "PASS",
            "paired_lowpass_delta": report["paired_lowpass_delta"][scope][
                "decision"
            ]
            == "PASS",
            "exact_relative_anti_shortcut": exact_relative[scope]["decision"]
            == "PASS",
            "component_input_gradient": report["component_input_gradient"][
                scope
            ]["decision"]
            == "PASS",
            "training_segment_transfer": report["training_segment_transfer"][
                scope
            ]["decision"]
            == "PASS",
        }
    exact_runtime = report.get("exact_runtime", {})
    coverage_checks = {
        "attempted_coverage": float(exact_runtime.get("attempted_coverage", 0.0))
        >= float(exact_runtime.get("coverage_gate", 1.0)),
        "selected_coverage": float(exact_runtime.get("selected_coverage", 0.0))
        == 1.0,
        "selected_speakers": len(
            report.get("status_only_substitution", {}).get(
                "selected_speakers",
                [],
            )
        )
        == 24,
        "selection_status_only": report.get("status_only_substitution", {}).get(
            "selection_field"
        )
        == "scoring_status",
    }
    passed = all(coverage_checks.values()) and all(
        all(checks.values()) for checks in scope_checks.values()
    )
    return {
        "decision": "PASS" if passed else "FAIL",
        "coverage_checks": coverage_checks,
        "scope_checks": scope_checks,
        "level_alignment": report["level_alignment"],
        "paired_lowpass_delta": report["paired_lowpass_delta"],
        "exact_relative_anti_shortcut": exact_relative,
        "component_input_gradient": report["component_input_gradient"],
        "training_segment_transfer": report["training_segment_transfer"],
        "status_only_substitution": report["status_only_substitution"],
    }


def speaker_audit(
    label_rows: list[dict[str, str]],
    calibration: dict[str, Any],
    holdout: dict[str, Any],
    svd: dict[str, Any],
    seal: dict[str, Any],
) -> dict[str, Any]:
    split_speakers = {
        split: {row["speaker_id"] for row in label_rows if row["split"] == split}
        for split in (
            "surrogate_train",
            "surrogate_calibration",
            "surrogate_holdout",
        )
    }
    calibration_speakers = set(calibration["selection"]["speakers"])
    holdout_speakers = set(holdout["selection"]["speakers"])
    svd_speakers = set(
        svd["status_only_substitution"]["selected_speakers"]
    )
    sealed_reserves = {
        row["panel_speaker_id"]
        for row in seal["rows"]
        if row["selection_role"] == "reserve"
    }
    unused_reserves = set(
        svd["status_only_substitution"]["unused_reserve_speakers"]
    )
    label_speakers = set().union(*split_speakers.values())
    checks = {
        "label_bank_splits_pairwise_disjoint": not (
            split_speakers["surrogate_train"]
            & split_speakers["surrogate_calibration"]
            or split_speakers["surrogate_train"]
            & split_speakers["surrogate_holdout"]
            or split_speakers["surrogate_calibration"]
            & split_speakers["surrogate_holdout"]
        ),
        "calibration_selection_in_calibration_split": calibration_speakers
        <= split_speakers["surrogate_calibration"],
        "holdout_selection_in_holdout_split": holdout_speakers
        <= split_speakers["surrogate_holdout"],
        "calibration_holdout_disjoint": not calibration_speakers
        & holdout_speakers,
        "svd_disjoint_from_label_bank": not svd_speakers & label_speakers,
        "unused_reserves_are_sealed_reserves": bool(unused_reserves)
        and unused_reserves <= sealed_reserves,
        "unused_reserves_not_in_scored_selection": not unused_reserves
        & svd_speakers,
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "label_bank_speaker_counts": {
            split: len(speakers) for split, speakers in split_speakers.items()
        },
        "calibration_speakers": sorted(calibration_speakers),
        "holdout_speakers": sorted(holdout_speakers),
        "svd_selected_speakers": sorted(svd_speakers),
        "unopened_svd_reserve_speakers": sorted(unused_reserves),
    }


def source_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "screen_report": args.screen_report_sha256,
        "screen_receipt": args.screen_receipt_sha256,
        "calibration_report": args.calibration_report_sha256,
        "calibration_receipt": args.calibration_receipt_sha256,
        "holdout_report": args.holdout_report_sha256,
        "holdout_receipt": args.holdout_receipt_sha256,
        "svd_report": args.svd_report_sha256,
        "svd_receipt": args.svd_receipt_sha256,
        "svd_panel_seal": args.svd_panel_seal_sha256,
        "svd_seal_receipt": args.svd_seal_receipt_sha256,
        "label_bank": args.label_bank_sha256,
        "predictor_checkpoint": args.predictor_checkpoint_sha256,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite promotion audit: {args.output_dir}")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    args.output_dir.mkdir(parents=True)

    screen, screen_receipt = load_bound_report(
        args.screen_report,
        args.screen_report_sha256,
        args.screen_receipt,
        args.screen_receipt_sha256,
        "Route C screen",
    )
    calibration, calibration_receipt = load_bound_report(
        args.calibration_report,
        args.calibration_report_sha256,
        args.calibration_receipt,
        args.calibration_receipt_sha256,
        "LTAS calibration gate",
    )
    holdout, holdout_receipt = load_bound_report(
        args.holdout_report,
        args.holdout_report_sha256,
        args.holdout_receipt,
        args.holdout_receipt_sha256,
        "LTAS holdout gate",
    )
    svd, svd_receipt = load_bound_report(
        args.svd_report,
        args.svd_report_sha256,
        args.svd_receipt,
        args.svd_receipt_sha256,
        "LTAS SVD authority",
    )
    validate_hash(
        args.svd_panel_seal,
        args.svd_panel_seal_sha256,
        "LTAS SVD panel seal",
    )
    validate_hash(
        args.svd_seal_receipt,
        args.svd_seal_receipt_sha256,
        "LTAS SVD seal receipt",
    )
    seal = load_json(args.svd_panel_seal)
    seal["_path"] = str(args.svd_panel_seal.resolve())
    seal_receipt = load_json(args.svd_seal_receipt)
    if seal_receipt.get("panel_seal_sha256") != args.svd_panel_seal_sha256:
        raise ValueError("SVD seal receipt does not bind its panel seal")
    if svd.get("panel_seal_sha256") != args.svd_panel_seal_sha256:
        raise ValueError("SVD report panel-seal hash differs")

    validate_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "LTAS predictor checkpoint",
    )
    label_rows = read_rows(args.label_bank, args.label_bank_sha256)
    exact_hashes = validate_exact_sources(args)
    for payload, label in (
        (screen_receipt, "screen receipt"),
        (calibration_receipt, "calibration receipt"),
        (holdout_receipt, "holdout receipt"),
        (svd_receipt, "SVD receipt"),
    ):
        require_zero_training(payload, label)

    train_slopes = np.asarray(
        [
            float(row["slope"])
            for row in label_rows
            if row["split"] == "surrogate_train"
            and row["view"] in {"cs", "sv"}
            and row["scoring_status"] == "ok"
        ],
        dtype=np.float64,
    )
    train_scale = float(train_slopes.std())
    if train_slopes.size < 2 or train_scale <= 0.0:
        raise ValueError("invalid surrogate-train LTAS slope scale")
    reported_scales = (
        float(calibration["train_slope_scale_std_surrogate_train"]),
        float(holdout["train_slope_scale_std_surrogate_train"]),
        float(svd["train_slope_scale_std_surrogate_train"]),
    )
    if not all(math.isclose(train_scale, value, abs_tol=1e-12) for value in reported_scales):
        raise ValueError("LTAS slope scale differs across frozen reports")

    repeatability = calibration_repeatability(
        calibration,
        label_rows,
        args.exact_python,
        args.output_dir,
        train_scale,
    )
    exact_material_floor = float(repeatability["frozen_exact_material_floor"])
    screen_audit = screen_gate_audit(screen)
    calibration_audit = alignment_gate_audit(
        calibration,
        CALIBRATION_DECISION,
        "surrogate_calibration",
        exact_material_floor,
    )
    holdout_audit = alignment_gate_audit(
        holdout,
        HOLDOUT_DECISION,
        "surrogate_holdout",
        exact_material_floor,
    )
    svd_audit = svd_gate_audit(svd, seal, exact_material_floor)
    speakers = speaker_audit(label_rows, calibration, holdout, svd, seal)

    calibration_exact = calibration_audit["gate"]["summary"]["lowpass_3khz"]
    holdout_exact = holdout_audit["gate"]["summary"]["lowpass_3khz"]
    svd_sv_exact = svd_audit["exact_relative_anti_shortcut"]["sv"]["summary"][
        "lowpass_3khz"
    ]
    legacy_authority = {
        "old_absolute_candidate_min": CURRENT_ABSOLUTE_LOWPASS_MIN,
        "calibration_exact_mean_standardized_distance": float(
            calibration_exact["exact_mean_standardized_distance"]
        ),
        "holdout_exact_mean_standardized_distance": float(
            holdout_exact["exact_mean_standardized_distance"]
        ),
        "svd_sv_exact_mean_standardized_distance": float(
            svd_sv_exact["exact_mean_standardized_distance"]
        ),
    }
    legacy_authority["authority_inconsistent"] = all(
        value < CURRENT_ABSOLUTE_LOWPASS_MIN
        for key, value in legacy_authority.items()
        if key.endswith("mean_standardized_distance")
    )

    gates = {
        "frozen_screen_accuracy_calibration_coverage_gradient_external": (
            screen_audit["decision"] == "PASS"
        ),
        "exact_repeatability": repeatability["decision"] == "PASS",
        "calibration_freezes_exact_relative_gate": (
            calibration_audit["decision"] == "PASS"
        ),
        "speaker_disjoint_holdout_validation": holdout_audit["decision"]
        == "PASS",
        "speaker_disjoint_svd_external_validation": svd_audit["decision"]
        == "PASS",
        "speaker_metadata_and_unopened_reserves": speakers["decision"]
        == "PASS",
        "old_absolute_gate_is_exact_authority_inconsistent": bool(
            legacy_authority["authority_inconsistent"]
        ),
    }
    passed = all(gates.values())
    decision = PROMOTION_PASS if passed else PROMOTION_FAIL
    gate_contract = {
        "schema_version": "avqi-ltas-slope-exact-relative-gate-v1",
        "candidate": "frozen full-waveform differentiable LTAS slope",
        "gate_setting_data": "surrogate_calibration pathological SV only",
        "validation_data": ["surrogate_holdout", "SVD v10 speaker-disjoint"],
        "exact_repeatability_runs": REPEATABILITY_RUNS,
        "exact_material_distance_min": exact_material_floor,
        "exact_material_distance_formula": (
            "max(0.02, 10 * calibration exact normalized p99 repeatability range)"
        ),
        "candidate_to_exact_distance_ratio": list(AUTHORITY_RATIO_RANGE),
        "signed_direction_agreement_min": DIRECTION_AGREEMENT_MIN,
        "gain_and_shift_distance_max": INVARIANCE_DISTANCE_MAX,
        "candidate_and_exact_lowpass_must_exceed_controls": True,
        "legacy_absolute_candidate_distance_min": CURRENT_ABSOLUTE_LOWPASS_MIN,
        "legacy_absolute_gate_role": "report_only_authority_inconsistent",
        "gate_parameters_tuned_on_holdout_or_svd": False,
    }
    repeatability_path = args.output_dir / "repeatability_predictions.json"
    write_json(repeatability_path, repeatability)
    report = {
        "schema_version": "avqi-route-c-ltas-slope-promotion-v1",
        "decision": decision,
        "scientific_status": (
            "SCORER_PROMOTED_PENDING_BOUNDED_WAVEFORM_PILOT"
            if passed
            else "SCORER_NOT_PROMOTED"
        ),
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_input_sha256": source_inputs(args),
        "exact_source_sha256": exact_hashes,
        "train_slope_scale_std_surrogate_train": train_scale,
        "frozen_gate_contract": gate_contract,
        "gates": gates,
        "screen_gate_audit": screen_audit,
        "exact_repeatability": repeatability,
        "calibration_gate_audit": calibration_audit,
        "holdout_gate_audit": holdout_audit,
        "svd_external_gate_audit": svd_audit,
        "speaker_audit": speakers,
        "legacy_absolute_gate_authority_audit": legacy_authority,
        "generic_route_c_screen_code_changed": False,
        "ltas_slope_promotion_contract_frozen": passed,
        "bounded_waveform_pilot_authorized": passed,
        "bounded_waveform_pilot_submitted": False,
        "production_generator_training_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "final_audio_highpass_applied": False,
        "scope": "LTAS slope only; Shimmer dB remains unresolved; CPPS/HNR untouched",
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    summary_path = args.output_dir / "SUMMARY.md"
    summary_path.write_text(
        "# Route C LTAS slope promotion audit\n\n"
        f"Decision: `{decision}`\n\n"
        "This decision can authorize only a new bounded-waveform LTAS-slope "
        "pilot. Generator optimizer steps remain zero.\n",
        encoding="utf-8",
    )
    receipt = {
        "decision": decision,
        "scientific_status": report["scientific_status"],
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_input_sha256": source_inputs(args),
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "repeatability_predictions.json": sha256_file(repeatability_path),
            "SUMMARY.md": sha256_file(summary_path),
        },
        "bounded_waveform_pilot_authorized": passed,
        "bounded_waveform_pilot_submitted": False,
        "production_generator_training_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
