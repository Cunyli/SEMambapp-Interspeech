#!/usr/bin/env python3
"""Apply the frozen phase-aware-to-full-TFGrid AVQI scorer promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
BASELINE_ARCHITECTURE = "frequency_aware"
CANDIDATE_ARCHITECTURE = "phase_compact_tfgrid"
EXPECTED_ARCHITECTURES = (
    BASELINE_ARCHITECTURE,
    "phase_frequency_aware",
    CANDIDATE_ARCHITECTURE,
)
CALIBRATION_RELATIVE_IMPROVEMENT_MIN = 0.15
PRIMARY_MEDIAN_NMAE_RELATIVE_IMPROVEMENT_MIN = 0.10
ADDITIONAL_COMPLETE_COMPONENTS_MIN = 1
REGRESSION_TOLERANCE = 1e-12
PROMOTE_DECISION = "PROMOTE_PRETRAINED_FULL_TFGRID_SCREEN"
KEEP_DECISION = "KEEP_COMPACT_NO_FULL_TFGRID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"expected positive finite {label}, found {value!r}")
    return number


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def median_component_nmae(component_report: dict[str, Any]) -> float:
    values = []
    for component in COMPONENTS:
        value = float(component_report[component]["normalized_mae"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"invalid normalized MAE for {component}: {value!r}"
            )
        values.append(value)
    return float(statistics.median(values))


def validate_screen(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("decision") != "COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE":
        raise ValueError("phase screen is not a completed scorer-only screen")
    if report.get("generator_optimizer_steps") != 0:
        raise ValueError("phase screen contains generator optimizer steps")
    if report.get("formal_pathology_training_submitted") is not False:
        raise ValueError("formal pathology training state is ambiguous")
    contract = report["contract"]
    if tuple(contract["components"]) != COMPONENTS:
        raise ValueError("AVQI component contract differs")
    architectures = tuple(
        contract["routes"]["frozen_independent_predictor"]["architectures"]
    )
    if architectures != EXPECTED_ARCHITECTURES:
        raise ValueError(
            f"unexpected phase screen architectures: {architectures}"
        )
    route = report["routes"]["frozen_independent_predictor"]
    if set(route["training"]) != set(EXPECTED_ARCHITECTURES):
        raise ValueError("training results do not match the frozen architecture set")
    calibration_winner = min(
        route["training"],
        key=lambda name: route["training"][name]["best_calibration_loss"],
    )
    if route["selected_architecture"] != calibration_winner:
        raise ValueError("selected architecture is not the calibration-only winner")
    for architecture in EXPECTED_ARCHITECTURES:
        route["all_architecture_metrics"][architecture]["calibrated"]["primary"]
        route["external_evaluation_by_architecture"][architecture]["pathology"]
        route["external_evaluation_by_architecture"][architecture]["vctk"]
        route["qualification_by_architecture"][architecture]
    return route


def architecture_complete_components(
    route: dict[str, Any], architecture: str
) -> list[str]:
    values = list(
        route["qualification_by_architecture"][architecture][
            "eligible_components"
        ]
    )
    if len(values) != len(set(values)) or any(
        component not in COMPONENTS for component in values
    ):
        raise ValueError(
            f"invalid complete component set for {architecture}: {values}"
        )
    return [component for component in COMPONENTS if component in values]


def required_slice_reports(
    route: dict[str, Any], architecture: str
) -> dict[str, tuple[dict[str, Any], bool]]:
    internal = route["all_architecture_metrics"][architecture]["calibrated"]
    external = route["external_evaluation_by_architecture"][architecture]
    pathology = external["pathology"]
    vctk = external["vctk"]
    if vctk is None:
        raise ValueError(f"missing VCTK external report for {architecture}")
    reports: dict[str, tuple[dict[str, Any], bool]] = {
        "internal_cs": (internal["slices"]["cs"], True),
        "internal_sv": (internal["slices"]["sv"], True),
        "internal_healthy": (internal["slices"]["healthy"], True),
        "internal_patient": (internal["slices"]["patient"], True),
        "pathology_cs": (
            pathology["slices"]["view=cs"],
            pathology["slice_coverage"]["view=cs"]["decision"] == "PASS",
        ),
        "pathology_sv": (
            pathology["slices"]["view=sv"],
            pathology["slice_coverage"]["view=sv"]["decision"] == "PASS",
        ),
        "pathology_healthy": (
            pathology["slices"]["label=healthy"],
            pathology["slice_coverage"]["label=healthy"]["decision"] == "PASS",
        ),
        "pathology_patient": (
            pathology["slices"]["label=patient"],
            pathology["slice_coverage"]["label=patient"]["decision"] == "PASS",
        ),
        "pathology_severe_sv": (
            pathology["slices"][
                "view=sv&sample_group=pathological_severe"
            ],
            pathology["slice_coverage"][
                "view=sv&sample_group=pathological_severe"
            ]["decision"]
            == "PASS",
        ),
        "pathology_snr10": (
            pathology["slices"]["condition=snr10"],
            pathology["slice_coverage"]["condition=snr10"]["decision"]
            == "PASS",
        ),
        "vctk_external": (
            vctk["primary"],
            vctk["primary_coverage"]["decision"] == "PASS",
        ),
    }
    for condition in ("clean", "rir_only", "snr20", "snr10"):
        key = f"condition={condition}"
        reports[f"vctk_{condition}"] = (
            vctk["slices"][key],
            vctk["slice_coverage"][key]["decision"] == "PASS",
        )
    return reports


def slice_regression_report(route: dict[str, Any]) -> dict[str, Any]:
    baseline_reports = required_slice_reports(route, BASELINE_ARCHITECTURE)
    candidate_reports = required_slice_reports(route, CANDIDATE_ARCHITECTURE)
    if tuple(baseline_reports) != tuple(candidate_reports):
        raise ValueError("baseline and candidate required slice sets differ")
    output: dict[str, Any] = {}
    for slice_name in baseline_reports:
        baseline_report, baseline_coverage = baseline_reports[slice_name]
        candidate_report, candidate_coverage = candidate_reports[slice_name]
        baseline_nmae = median_component_nmae(baseline_report)
        candidate_nmae = median_component_nmae(candidate_report)
        coverage_passed = baseline_coverage and candidate_coverage
        non_regressed = candidate_nmae <= baseline_nmae + REGRESSION_TOLERANCE
        output[slice_name] = {
            "baseline_median_component_nmae": baseline_nmae,
            "candidate_median_component_nmae": candidate_nmae,
            "candidate_minus_baseline": candidate_nmae - baseline_nmae,
            "relative_improvement": relative_improvement(
                positive_finite(baseline_nmae, f"{slice_name} baseline NMAE"),
                candidate_nmae,
            ),
            "coverage_passed": coverage_passed,
            "non_regressed": non_regressed,
            "decision": (
                "PASS" if coverage_passed and non_regressed else "FAIL"
            ),
        }
    return output


def evaluate_promotion(screen: dict[str, Any]) -> dict[str, Any]:
    route = validate_screen(screen)
    training = route["training"]
    baseline_calibration = positive_finite(
        training[BASELINE_ARCHITECTURE]["best_calibration_loss"],
        "baseline calibration loss",
    )
    candidate_calibration = positive_finite(
        training[CANDIDATE_ARCHITECTURE]["best_calibration_loss"],
        "candidate calibration loss",
    )
    calibration_improvement = relative_improvement(
        baseline_calibration,
        candidate_calibration,
    )
    architecture_metrics = route["all_architecture_metrics"]
    baseline_primary_nmae = median_component_nmae(
        architecture_metrics[BASELINE_ARCHITECTURE]["calibrated"]["primary"]
    )
    candidate_primary_nmae = median_component_nmae(
        architecture_metrics[CANDIDATE_ARCHITECTURE]["calibrated"]["primary"]
    )
    primary_nmae_improvement = relative_improvement(
        positive_finite(baseline_primary_nmae, "baseline primary median NMAE"),
        candidate_primary_nmae,
    )
    baseline_complete = architecture_complete_components(
        route, BASELINE_ARCHITECTURE
    )
    candidate_complete = architecture_complete_components(
        route, CANDIDATE_ARCHITECTURE
    )
    new_complete = [
        component for component in candidate_complete
        if component not in baseline_complete
    ]
    lost_complete = [
        component for component in baseline_complete
        if component not in candidate_complete
    ]
    slice_report = slice_regression_report(route)
    gates = {
        "candidate_selected_by_calibration_only": (
            route["selected_architecture"] == CANDIDATE_ARCHITECTURE
        ),
        "calibration_loss_relative_improvement_ge_0_15": (
            calibration_improvement >= CALIBRATION_RELATIVE_IMPROVEMENT_MIN
        ),
        "primary_median_component_nmae_relative_improvement_ge_0_10": (
            primary_nmae_improvement
            >= PRIMARY_MEDIAN_NMAE_RELATIVE_IMPROVEMENT_MIN
        ),
        "additional_complete_components_ge_1": (
            len(new_complete) >= ADDITIONAL_COMPLETE_COMPONENTS_MIN
        ),
        "baseline_complete_components_preserved": not lost_complete,
        "all_required_slice_medians_non_regressed": all(
            item["decision"] == "PASS" for item in slice_report.values()
        ),
    }
    promoted = all(gates.values())
    return {
        "schema_version": "avqi-component-phaseaware-v4-promotion-v1",
        "comparison": (
            f"{CANDIDATE_ARCHITECTURE}_vs_{BASELINE_ARCHITECTURE}"
        ),
        "thresholds": {
            "calibration_loss_relative_improvement_min": (
                CALIBRATION_RELATIVE_IMPROVEMENT_MIN
            ),
            "primary_median_component_nmae_relative_improvement_min": (
                PRIMARY_MEDIAN_NMAE_RELATIVE_IMPROVEMENT_MIN
            ),
            "additional_complete_components_min": (
                ADDITIONAL_COMPLETE_COMPONENTS_MIN
            ),
            "required_slice_regression_allowed": False,
        },
        "calibration_loss": {
            "baseline": baseline_calibration,
            "candidate": candidate_calibration,
            "relative_improvement": calibration_improvement,
        },
        "primary_median_component_nmae": {
            "baseline": baseline_primary_nmae,
            "candidate": candidate_primary_nmae,
            "relative_improvement": primary_nmae_improvement,
        },
        "complete_components": {
            "baseline": baseline_complete,
            "candidate": candidate_complete,
            "new_in_candidate": new_complete,
            "lost_from_baseline": lost_complete,
        },
        "required_slice_regression": slice_report,
        "gates": gates,
        "decision": PROMOTE_DECISION if promoted else KEEP_DECISION,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "interpretation_limit": (
            "This gate may authorize only a pretrained full-TFGrid scorer screen; "
            "it never authorizes generator training."
        ),
    }


def markdown_summary(report: dict[str, Any]) -> str:
    calibration = report["calibration_loss"]
    nmae = report["primary_median_component_nmae"]
    complete = report["complete_components"]
    failed_slices = [
        name
        for name, item in report["required_slice_regression"].items()
        if item["decision"] != "PASS"
    ]
    lines = [
        "# AVQI v4 phase-aware predictor 晋级结论",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "| 判据 | Frequency-aware CNN | Phase Compact TF-GridNet | 结果 |",
        "|---|---:|---:|---|",
        (
            "| Calibration loss | "
            f"{calibration['baseline']:.6f} | {calibration['candidate']:.6f} | "
            f"相对改善 {calibration['relative_improvement']:.1%} |"
        ),
        (
            "| Holdout 六项中位 NMAE | "
            f"{nmae['baseline']:.6f} | {nmae['candidate']:.6f} | "
            f"相对改善 {nmae['relative_improvement']:.1%} |"
        ),
        (
            "| 完整通过的 component 数 | "
            f"{len(complete['baseline'])} | {len(complete['candidate'])} | "
            f"新增 {len(complete['new_in_candidate'])} |"
        ),
        "",
        (
            "未回退的必需切片：全部通过"
            if not failed_slices
            else "发生回退或 coverage 失败的切片：" + ", ".join(failed_slices)
        ),
        "",
        "该结论最多授权运行冻结的 pretrained full-TFGrid scorer screen，"
        "不授权任何 generator optimizer step 或正式病理 AVQI-T2 训练。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    screen = load_json(args.screen_report)
    report = evaluate_promotion(screen)
    report["screen_report"] = str(args.screen_report.resolve())
    report["screen_report_sha256"] = sha256_file(args.screen_report)
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "promotion_report.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    receipt = {
        "decision": report["decision"],
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "screen_report_sha256": report["screen_report_sha256"],
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
