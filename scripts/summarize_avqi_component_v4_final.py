#!/usr/bin/env python3
"""Build the fail-closed final AVQI v4 route comparison and receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-completion", type=Path, required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--phase-screen", type=Path, required=True)
    parser.add_argument("--phase-consensus", type=Path, required=True)
    parser.add_argument("--direct-screen", type=Path, required=True)
    parser.add_argument("--direct-consensus", type=Path, required=True)
    parser.add_argument("--phase-promotion", type=Path, required=True)
    parser.add_argument("--waveform-guardrail", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
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


def repository_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_no_generator_update(name: str, report: dict[str, Any]) -> None:
    if report.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{name} contains generator optimizer steps")
    if report.get("formal_pathology_training_submitted") is not False:
        raise ValueError(f"{name} formal training state is ambiguous")


def metric_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(metrics) != set(COMPONENTS):
        raise ValueError(f"component set mismatch: {sorted(metrics)}")
    return {
        "median_normalized_mae": statistics.median(
            float(metrics[component]["normalized_mae"])
            for component in COMPONENTS
        ),
        "passing_components": [
            component
            for component in COMPONENTS
            if metrics[component]["decision"] == "PASS"
        ],
        "components": metrics,
    }


def pass_components(report: dict[str, Any]) -> list[str]:
    return [
        component
        for component in COMPONENTS
        if report["components"][component]["decision"] == "PASS"
    ]


def shared_route_summary(screen: dict[str, Any]) -> dict[str, Any]:
    route = screen["routes"]["shared_dual_head"]
    selected = route["selected_candidate"]
    gradient = route["gradient"]
    return {
        "route": "output_conditioned_dual_head",
        "form": selected,
        "best_calibration_loss": route["training"][selected][
            "best_calibration_loss"
        ],
        "internal": metric_summary(route["metrics"]["primary"]),
        "pathology_external": metric_summary(
            route["external_clean_target_stress"]["primary"]
        ),
        "pathology_external_coverage": route[
            "external_clean_target_stress"
        ]["primary_coverage"]["fraction"],
        "vctk_external": metric_summary(
            route["vctk_external_own_target_stress"]["primary"]
        ),
        "vctk_external_coverage": route["vctk_external_own_target_stress"][
            "primary_coverage"
        ]["fraction"],
        "anti_shortcut_pass_components": pass_components(
            route["anti_shortcut"]
        ),
        "segment_transfer_pass_components": pass_components(
            route["training_segment_transfer"]
        ),
        "gradient_decision": gradient["decision"],
        "component_gradient_pass": pass_components(
            {"components": gradient["component_input_gradients"]}
        ),
        "decoder_gradient_norm": gradient["decoder_gradient_norm"],
        "backbone_gradient_norm": gradient["backbone_gradient_norm"],
        "eligible_components": route["eligible_components"],
        "decision": route["decision"],
    }


def independent_architecture_summary(
    screen: dict[str, Any], architecture: str
) -> dict[str, Any]:
    route = screen["routes"]["frozen_independent_predictor"]
    calibrated = route["all_architecture_metrics"][architecture]["calibrated"]
    external = route["external_evaluation_by_architecture"][architecture]
    qualification = route["qualification_by_architecture"][architecture]
    gradient = qualification["gradient"]
    return {
        "route": "independent_predictor",
        "form": architecture,
        "selected_by_calibration": route["selected_architecture"] == architecture,
        "best_calibration_loss": route["training"][architecture][
            "best_calibration_loss"
        ],
        "internal": metric_summary(calibrated["primary"]),
        "pathology_external": metric_summary(external["pathology"]["primary"]),
        "pathology_external_coverage": external["pathology"][
            "primary_coverage"
        ]["fraction"],
        "vctk_external": metric_summary(external["vctk"]["primary"]),
        "vctk_external_coverage": external["vctk"]["primary_coverage"][
            "fraction"
        ],
        "anti_shortcut_pass_components": pass_components(
            qualification["anti_shortcut"]
        ),
        "segment_transfer_pass_components": pass_components(
            qualification["training_segment_transfer"]
        ),
        "gradient_decision": gradient["decision"],
        "component_gradient_pass": pass_components(
            {"components": gradient["component_input_gradients"]}
        ),
        "eligible_components": qualification["eligible_components"],
        "decision": qualification["decision"],
    }


def format_number(value: float) -> str:
    return f"{value:.4f}"


def component_text(components: list[str]) -> str:
    return ", ".join(components) if components else "none"


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for route in report["route_comparison"]:
        gradient = (
            f"{route['gradient_decision']} "
            f"({len(route['component_gradient_pass'])}/6 components)"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    route["display_name"],
                    format_number(route["best_calibration_loss"]),
                    format_number(route["internal"]["median_normalized_mae"]),
                    format_number(
                        route["pathology_external"]["median_normalized_mae"]
                    ),
                    format_number(
                        route["vctk_external"]["median_normalized_mae"]
                    ),
                    gradient,
                    component_text(route["eligible_components"]),
                    route["decision"],
                ]
            )
            + " |"
        )
    waveform = report["preserved_waveform_pilot"]
    failed_slices = ", ".join(waveform["failed_required_slices"])
    return "\n".join(
        [
            "# AVQI component 可微回传 v4 最终对照",
            "",
            f"**最终决策：`{report['decision']}`**",
            "",
            (
                "数值越低越好；NMAE 是同一冻结尺度下的六项中位标准化误差。"
                "所有路线都使用同一 speaker-disjoint 数据与 2000-step screen 预算。"
            ),
            "",
            "| 路线 | Calibration loss | Internal NMAE | 病理 external NMAE | VCTK NMAE | 梯度 | Eligible | 结论 |",
            "|---|---:|---:|---:|---:|---|---|---|",
            *rows,
            "",
            "## 已验证结论",
            "",
            (
                f"- 最准确 scorer 是 `{report['best_prediction_form']}`：病理 external "
                f"中位 NMAE={report['best_prediction_pathology_nmae']:.4f}。"
            ),
            (
                f"- VCTK external exact coverage={report['common_vctk_coverage']:.4%}，"
                f"低于 scorer promotion gate={report['scorer_external_coverage_gate']:.0%}；"
                "缺失样本没有插值或补标签。"
            ),
            (
                f"- Compact→Full gate：`{report['full_tfgrid_decision']}`；"
                "因此未提交 Full TF-Grid screen。"
            ),
            (
                f"- 三种子共识：phase=`{report['phase_consensus_decision']}`，"
                f"direct=`{report['direct_consensus_decision']}`，两者均为 0 eligible components。"
            ),
            "",
            "## 保存的 12-case waveform pilot",
            "",
            (
                f"- HNR：{waveform['hnr_improved_cases']}/12 方向改善，"
                f"中位标准化 reduction={waveform['hnr_median_reduction']:.5f}，FAIL。"
            ),
            (
                f"- LTAS tilt：{waveform['tilt_improved_cases']}/12 改善，"
                f"中位标准化 reduction={waveform['tilt_median_reduction']:.5f}，PASS。"
            ),
            (
                f"- full-band 病理 guardrails=`{waveform['full_band_decision']}`，"
                f"denoising non-regression=`{waveform['denoising_decision']}`，"
                f"基础 safety=`{waveform['safety_decision']}`。"
            ),
            f"- 未通过 slice：{failed_slices}。",
            "",
            "## Promotion boundary",
            "",
            "没有 scorer 路线通过多种子 gate，所以不运行新的 bounded waveform pilot，",
            "不提交正式 AVQI-T2 generator training。generator optimizer steps 始终为 0。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head() != args.source_commit:
        raise ValueError("declared finalizer source commit differs from repository HEAD")
    paths = {
        "data_completion": args.data_completion,
        "test_receipt": args.test_receipt,
        "phase_screen": args.phase_screen,
        "phase_consensus": args.phase_consensus,
        "direct_screen": args.direct_screen,
        "direct_consensus": args.direct_consensus,
        "phase_promotion": args.phase_promotion,
        "waveform_guardrail": args.waveform_guardrail,
    }
    inputs = {name: load_json(path) for name, path in paths.items()}
    for name, value in inputs.items():
        require_no_generator_update(name, value)
    if inputs["data_completion"]["decision"] != "DATA_READY_FOR_SCORER_SCREENS":
        raise ValueError("data completion did not authorize scorer screens")
    if inputs["test_receipt"]["decision"] != "PASS_AVQI_V4_REPOSITORY_TESTS":
        raise ValueError("repository test gate did not pass")
    if inputs["test_receipt"]["source_commit"] != args.source_commit:
        raise ValueError("test receipt source commit differs from finalizer source")
    for name in ("phase_screen", "direct_screen"):
        if (
            inputs[name]["decision"]
            != "COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
        ):
            raise ValueError(f"{name} did not complete")
    for name in ("phase_consensus", "direct_consensus"):
        if inputs[name]["promotion"]["decision"] != "NO_GO_AVQI_BACKPROP":
            raise ValueError(f"{name} unexpectedly promoted a route")
    if inputs["phase_promotion"]["decision"] != "KEEP_COMPACT_NO_FULL_TFGRID":
        raise ValueError("full TF-Grid promotion state differs")
    waveform = inputs["waveform_guardrail"]
    if waveform["decision"] != "FAIL_WAVEFORM_OPTIMIZATION":
        raise ValueError("preserved waveform pilot decision differs")
    if waveform["summary"]["full_band_pathology_guardrails"]["decision"] != "PASS":
        raise ValueError("preserved waveform pilot failed full-band guardrails")

    phase_screen = inputs["phase_screen"]
    direct_screen = inputs["direct_screen"]
    route_comparison = [
        {
            "display_name": "A1 output-conditioned dual-head",
            **shared_route_summary(phase_screen),
        },
        {
            "display_name": "B0 magnitude Frequency-aware CNN",
            **independent_architecture_summary(phase_screen, "frequency_aware"),
        },
        {
            "display_name": "B1 phase-aware Frequency-aware CNN",
            **independent_architecture_summary(
                phase_screen, "phase_frequency_aware"
            ),
        },
        {
            "display_name": "B2 phase-aware Compact TF-GridNet",
            **independent_architecture_summary(
                phase_screen, "phase_compact_tfgrid"
            ),
        },
        {
            "display_name": "C direct Praat-aligned hard-v2",
            **independent_architecture_summary(
                direct_screen, "direct_praat_hard_v2"
            ),
        },
    ]
    best_route = min(
        route_comparison,
        key=lambda route: route["pathology_external"]["median_normalized_mae"],
    )
    waveform_exact = waveform["summary"]["aggregates"]["exact"]
    failed_slices = [
        name
        for name, value in waveform["summary"]["required_slices"].items()
        if value["decision"] != "PASS"
    ]
    report = {
        "schema_version": "avqi-component-v4-final-comparison-v1",
        "decision": "NO_GO_AVQI_T2_TRAINING",
        "source_commit": args.source_commit,
        "data": {
            "prepared_rows": inputs["data_completion"]["prepared_rows"],
            "exact_valid_rows": inputs["data_completion"]["exact_valid_rows"],
            "exact_scored_rows": inputs["data_completion"]["exact_scored_rows"],
            "exact_coverage": inputs["data_completion"]["exact_coverage"],
            "merged_internal_rows": inputs["data_completion"][
                "merged_internal_rows"
            ],
            "external_rows": inputs["data_completion"]["external_rows"],
            "speaker_counts": inputs["data_completion"]["speaker_counts"],
            "speaker_overlap": inputs["data_completion"]["speaker_overlap"],
            "full_band_audio_preserved": inputs["data_completion"][
                "full_band_audio_preserved"
            ],
            "waveform_highpass_applied": inputs["data_completion"][
                "waveform_highpass_applied"
            ],
        },
        "route_comparison": route_comparison,
        "best_prediction_form": best_route["form"],
        "best_prediction_pathology_nmae": best_route["pathology_external"][
            "median_normalized_mae"
        ],
        "common_vctk_coverage": best_route["vctk_external_coverage"],
        "scorer_external_coverage_gate": 0.99,
        "phase_consensus_decision": inputs["phase_consensus"]["promotion"][
            "decision"
        ],
        "direct_consensus_decision": inputs["direct_consensus"]["promotion"][
            "decision"
        ],
        "phase_consensus": inputs["phase_consensus"]["routes"],
        "direct_consensus": inputs["direct_consensus"]["routes"],
        "full_tfgrid_decision": inputs["phase_promotion"]["decision"],
        "full_tfgrid_screen_submitted": False,
        "preserved_waveform_pilot": {
            "decision": waveform["decision"],
            "hnr_improved_cases": round(
                waveform_exact["hnr"]["improvement_fraction"]
                * waveform_exact["hnr"]["rows"]
            ),
            "hnr_median_reduction": waveform_exact["hnr"][
                "median_normalized_gap_reduction"
            ],
            "tilt_improved_cases": round(
                waveform_exact["tilt"]["improvement_fraction"]
                * waveform_exact["tilt"]["rows"]
            ),
            "tilt_median_reduction": waveform_exact["tilt"][
                "median_normalized_gap_reduction"
            ],
            "full_band_decision": waveform["summary"][
                "full_band_pathology_guardrails"
            ]["decision"],
            "denoising_decision": waveform["summary"]["denoising"]["decision"],
            "safety_decision": waveform["summary"]["safety"]["decision"],
            "failed_required_slices": failed_slices,
        },
        "new_bounded_waveform_pilot_submitted": False,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "comparison_report.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_json(report_path, report)
    summary_path.write_text(render_markdown(report), encoding="utf-8")
    receipt = {
        "decision": report["decision"],
        "best_prediction_form": report["best_prediction_form"],
        "phase_consensus_decision": report["phase_consensus_decision"],
        "direct_consensus_decision": report["direct_consensus_decision"],
        "full_tfgrid_screen_submitted": False,
        "new_bounded_waveform_pilot_submitted": False,
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
        },
        "input_sha256": report["input_sha256"],
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
