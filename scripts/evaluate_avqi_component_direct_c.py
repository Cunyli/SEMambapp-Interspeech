#!/usr/bin/env python3
"""Route C-only AVQI component screen with no learned A/B scorer.

The direct estimator is fitted only by a positive per-component affine alignment
on the surrogate training split. It has zero trainable parameters and zero
optimizer steps. Exact Praat labels remain the judge on speaker-disjoint
internal, pathology-external, and VCTK-external panels. The script stops before
any generator update.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_avqi_component_backprop import (
    AVQI_COMPONENT_LOSS_WEIGHTS,
    AVQI_COMPONENT_NAMES,
    CALIBRATION_SLOPE_RANGE,
    COMPONENT_INPUT_GRADIENT_MAX,
    DEFAULT_EXPECTED_SPLIT_SPEAKERS,
    DELTA_SPEARMAN_GATE,
    EXTERNAL_CANDIDATES,
    EXTERNAL_COVERAGE_GATE,
    EXTERNAL_PRIMARY_CANDIDATE,
    EXTERNAL_REQUIRED_SLICES,
    LEVEL_SPEARMAN_GATE,
    NORMALIZED_MAE_GATE,
    PRIMARY_GATE_COMPONENTS,
    SEGMENT_TRANSFER_NMAE_GATE,
    TRAINING_SEGMENT_SAMPLES,
    anti_shortcut_report,
    apply_component_calibrator,
    eligible_components,
    external_stress_test,
    fit_component_calibrator,
    freeze_module,
    independent_gradient_smoke,
    load_config,
    load_examples,
    load_generator,
    predict_waveforms,
    prediction_rows,
    route_has_minimum_component_coverage,
    route_metrics,
    sha256_file,
    train_waveform_predictor,
    training_segment_transfer_report,
    vctk_external_test,
    write_csv,
    write_json,
)


DIRECT_ARCHITECTURES = (
    "direct_exact_inspired",
    "direct_praat_soft_v2",
    "direct_praat_hard_v2",
    "direct_praat_hard_shimmer_rms_v3",
)
ROUTE_KEY = "direct_differentiable_estimator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--external-exact-csv", type=Path, required=True)
    parser.add_argument("--external-exact-csv-sha256", required=True)
    parser.add_argument("--vctk-external-label-bank", type=Path, required=True)
    parser.add_argument("--vctk-external-label-bank-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--route-scope",
        choices=("direct_only",),
        default="direct_only",
    )
    parser.add_argument(
        "--waveform-architectures",
        default="direct_praat_hard_v2",
        help="comma-separated zero-parameter direct estimators",
    )
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument(
        "--expected-train-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_train"],
    )
    parser.add_argument(
        "--expected-calibration-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_calibration"],
    )
    parser.add_argument(
        "--expected-holdout-speakers",
        type=int,
        default=DEFAULT_EXPECTED_SPLIT_SPEAKERS["surrogate_holdout"],
    )
    return parser.parse_args()


def comma_separated_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one direct estimator is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate direct estimators are not allowed: {values}")
    return values


def direct_contract(
    args: argparse.Namespace,
    architectures: tuple[str, ...],
    expected_split_speakers: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "avqi-component-direct-c-screen-v1",
        "purpose": "route_c_single_seed_screen_no_generator_update",
        "route_scope": "direct_only",
        "components": list(AVQI_COMPONENT_NAMES),
        "component_loss_weights": dict(
            zip(AVQI_COMPONENT_NAMES, AVQI_COMPONENT_LOSS_WEIGHTS, strict=True)
        ),
        "periodicity_anchor_components": list(PRIMARY_GATE_COMPONENTS),
        "minimum_route_coverage": (
            "at least one CPPS/HNR component and one component from shimmer or LTAS"
        ),
        "jitter_in_primary_task": False,
        "speaker_split": expected_split_speakers,
        "routes": {
            "shared_dual_head": {
                "status": "SKIPPED_USER_SCOPE",
                "candidates": [],
            },
            "frozen_independent_predictor": {
                "status": "SKIPPED_USER_SCOPE",
                "architectures": [],
            },
            ROUTE_KEY: {
                "route_family": "C",
                "architectures": list(architectures),
                "target": "input waveform own exact Praat components",
                "neural_predictor": False,
                "trainable_parameters": 0,
                "optimizer_steps": 0,
                "alignment": "positive per-component affine on train only",
                "exact_praat_role": "frozen final judge",
                "metric_branch_only_preprocessing": True,
                "full_band_enhancement_path_preserved": True,
            },
        },
        "calibration": {
            "method": "per-component positive-scale affine",
            "fit_split": "surrogate_calibration",
            "holdout_used_for_fit_or_selection": False,
        },
        "architecture_screen_seed": args.seed,
        "direct_formula_budget": {
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "maximum_optimizer_steps": args.max_optimizer_steps,
        },
        "multiseed_confirmation": {
            "seeds": [args.seed + 1, args.seed + 2, args.seed + 3],
            "architecture_locked_from": "screen calibration loss only",
            "component_rule": (
                "full component gate in at least two of three locked seeds; "
                "no post-hoc threshold changes"
            ),
            "generator_updates_allowed": False,
        },
        "matched_external_primary_candidate": EXTERNAL_PRIMARY_CANDIDATE,
        "additional_external_stress_candidates": [
            candidate
            for candidate in EXTERNAL_CANDIDATES
            if candidate != EXTERNAL_PRIMARY_CANDIDATE
        ],
        "anti_shortcut": {
            "common_invariance": ["gain_minus12db", "circular_shift_100ms"],
            "common_ood": ["silence", "rms_matched_150hz_tone"],
            "periodicity_noise": "noise_10db",
            "amplitude_modulation": "rms_matched_am_5hz",
            "spectral_shape": "lowpass_3khz",
        },
        "gates": {
            "level_spearman": LEVEL_SPEARMAN_GATE,
            "paired_delta_spearman": DELTA_SPEARMAN_GATE,
            "normalized_mae": NORMALIZED_MAE_GATE,
            "calibration_slope": list(CALIBRATION_SLOPE_RANGE),
            "component_input_gradient_norm": [
                1e-10,
                COMPONENT_INPUT_GRADIENT_MAX,
            ],
            "external_coverage": EXTERNAL_COVERAGE_GATE,
            "required_external_slices": list(EXTERNAL_REQUIRED_SLICES),
            "training_segment_samples": TRAINING_SEGMENT_SAMPLES,
            "training_segment_transfer_normalized_mae": (
                SEGMENT_TRANSFER_NMAE_GATE
            ),
        },
        "source_sha256": {
            "label_bank": args.label_bank_sha256,
            "config": args.config_sha256,
            "generator_checkpoint": args.checkpoint_sha256,
            "external_exact_csv": args.external_exact_csv_sha256,
            "vctk_external_label_bank": args.vctk_external_label_bank_sha256,
        },
        "source_commit": args.source_commit,
        "artifact_layout": {
            "run_output_dir": str(args.output_dir.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        },
    }


def human_summary(report: dict[str, Any]) -> str:
    route = report["routes"][ROUTE_KEY]
    return "\n".join(
        [
            "# AVQI Route C direct differentiable estimator",
            "",
            "## One-line result",
            "",
            report["plain_language_conclusion"],
            "",
            "| Route | Chosen form | Eligible components | Gradient | Decision |",
            "|---|---|---|---:|---|",
            (
                f"| C: direct formulas | {route['selected_architecture']} | "
                f"{', '.join(route['eligible_components']) or 'none'} | "
                f"{route['gradient']['decision']} | {route['decision']} |"
            ),
            "",
            "Routes A and B were skipped by contract. Exact Praat output metrics "
            "remain the promotion evidence.",
            "",
            "No generator optimizer step or formal pathology training was run.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    architectures = comma_separated_values(args.waveform_architectures)
    unknown = sorted(set(architectures) - set(DIRECT_ARCHITECTURES))
    if unknown:
        raise ValueError(f"Route C accepts only direct estimators, got: {unknown}")
    if args.max_optimizer_steps != 0:
        raise ValueError("Route C direct formulas require max optimizer steps = 0")
    expected_split_speakers = {
        "surrogate_train": args.expected_train_speakers,
        "surrogate_calibration": args.expected_calibration_speakers,
        "surrogate_holdout": args.expected_holdout_speakers,
    }
    if any(count <= 0 for count in expected_split_speakers.values()):
        raise ValueError(
            f"expected split speaker counts must be positive: {expected_split_speakers}"
        )
    if args.checkpoint.parent.name != EXTERNAL_PRIMARY_CANDIDATE:
        raise ValueError(
            "external comparison is locked to checkpoint directory "
            f"{EXTERNAL_PRIMARY_CANDIDATE}, got {args.checkpoint.parent.name}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.checkpoint_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoints: {args.checkpoint_dir}"
        )
    for path, expected_hash in (
        (args.label_bank, args.label_bank_sha256),
        (args.config, args.config_sha256),
        (args.checkpoint, args.checkpoint_sha256),
        (args.external_exact_csv, args.external_exact_csv_sha256),
        (
            args.vctk_external_label_bank,
            args.vctk_external_label_bank_sha256,
        ),
    ):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True)
    args.checkpoint_dir.mkdir(parents=True)
    contract = direct_contract(args, architectures, expected_split_speakers)
    write_json(args.output_dir / "experiment_contract.json", contract)

    examples, label_bank_coverage = load_examples(
        args.label_bank,
        expected_split_speakers,
    )
    config = load_config(args.config)
    generator = load_generator(config, args.checkpoint, device)

    models: dict[str, torch.nn.Module] = {}
    training_by_architecture: dict[str, Any] = {}
    stats_by_architecture: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    raw_predictions_by_architecture: dict[str, torch.Tensor] = {}
    predictions_by_architecture: dict[str, torch.Tensor] = {}
    calibrators: dict[str, Any] = {}
    for architecture in architectures:
        predictor, training, mean, scale, cached_inputs = train_waveform_predictor(
            examples,
            device,
            0,
            0,
            args.seed,
            architecture,
            None,
            None,
            0,
        )
        if training["optimizer_steps"] != 0:
            raise RuntimeError(f"direct estimator ran optimizer steps: {architecture}")
        if training["trainable_parameter_count"] != 0:
            raise RuntimeError(
                f"direct estimator has trainable parameters: {architecture}"
            )
        parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
        if parameter_count != 0:
            raise RuntimeError(
                f"direct estimator parameter count is nonzero: {architecture}"
            )
        raw_predictions = predict_waveforms(
            predictor,
            examples,
            mean,
            scale,
            device,
            cached_inputs,
        )
        calibrator = fit_component_calibrator(
            examples,
            raw_predictions,
            "own_target",
            device,
        )
        training["parameter_count"] = parameter_count
        models[architecture] = predictor
        training_by_architecture[architecture] = training
        stats_by_architecture[architecture] = (mean, scale)
        raw_predictions_by_architecture[architecture] = raw_predictions
        calibrators[architecture] = calibrator
        predictions_by_architecture[architecture] = apply_component_calibrator(
            raw_predictions,
            calibrator,
        )
        torch.save(
            {
                "state_dict": predictor.state_dict(),
                "target_mean": mean.cpu(),
                "target_scale": scale.cpu(),
                "calibration_scale": calibrator.scale.cpu(),
                "calibration_bias": calibrator.bias.cpu(),
                "components": AVQI_COMPONENT_NAMES,
                "architecture": architecture,
                "route_family": "C",
                "parameter_count": 0,
                "trainable_parameter_count": 0,
                "optimizer_steps": 0,
                "direct_alignment": training["direct_alignment"],
            },
            args.checkpoint_dir / f"direct_{architecture}_estimator.pt",
        )

    selected_architecture = min(
        training_by_architecture,
        key=lambda name: training_by_architecture[name]["best_calibration_loss"],
    )
    metrics_by_architecture: dict[str, Any] = {}
    anti_shortcut_by_architecture: dict[str, Any] = {}
    gradient_by_architecture: dict[str, Any] = {}
    segment_transfer_by_architecture: dict[str, Any] = {}
    pathology_external_by_architecture: dict[str, Any] = {}
    pathology_rows_by_architecture: dict[str, list[dict[str, Any]]] = {}
    vctk_external_by_architecture: dict[str, Any] = {}
    vctk_rows_by_architecture: dict[str, list[dict[str, Any]]] = {}
    eligible_by_architecture: dict[str, list[str]] = {}
    decision_by_architecture: dict[str, str] = {}
    surrogate_speaker_ids = {example.speaker_id for example in examples}

    for architecture, predictor in models.items():
        mean, scale = stats_by_architecture[architecture]
        calibrator = calibrators[architecture]
        freeze_module(predictor)

        def direct_predict(
            waveform: torch.Tensor,
            current_predictor: torch.nn.Module = predictor,
            current_mean: torch.Tensor = mean,
            current_scale: torch.Tensor = scale,
            current_calibrator: Any = calibrator,
        ) -> torch.Tensor:
            with torch.inference_mode():
                normalized = current_predictor(waveform.to(device))
                raw = normalized * current_scale + current_mean
                return current_calibrator(raw).cpu()[0]

        metrics = {
            "raw": route_metrics(
                examples,
                raw_predictions_by_architecture[architecture],
                "own_target",
                scale,
                primary_filter=lambda example: True,
                include_delta_gate=True,
            ),
            "calibrated": route_metrics(
                examples,
                predictions_by_architecture[architecture],
                "own_target",
                scale,
                primary_filter=lambda example: True,
                include_delta_gate=True,
            ),
        }
        anti_shortcut = anti_shortcut_report(
            examples,
            direct_predict,
            scale,
            expect_degradation_sensitivity=True,
        )
        segment_transfer = training_segment_transfer_report(
            examples,
            direct_predict,
            scale,
            "own_target",
        )
        gradient = independent_gradient_smoke(
            generator,
            config,
            predictor,
            mean,
            scale,
            calibrator,
            examples,
            device,
        )
        pathology_external, pathology_rows = external_stress_test(
            args.external_exact_csv,
            predictor,
            mean,
            scale,
            calibrator,
            surrogate_speaker_ids,
            device,
        )
        vctk_external, vctk_rows = vctk_external_test(
            args.vctk_external_label_bank,
            direct_predict,
            scale,
            surrogate_speaker_ids,
        )
        eligible = eligible_components(
            metrics["calibrated"],
            pathology_external,
            anti_shortcut,
            gradient,
            segment_transfer,
            vctk_external,
        )
        decision = (
            "ELIGIBLE_FOR_MULTISEED_CONFIRMATION"
            if route_has_minimum_component_coverage(eligible)
            else "NO_GO_GENERATOR_TRAINING"
        )
        metrics_by_architecture[architecture] = metrics
        anti_shortcut_by_architecture[architecture] = anti_shortcut
        gradient_by_architecture[architecture] = gradient
        segment_transfer_by_architecture[architecture] = segment_transfer
        pathology_external_by_architecture[architecture] = pathology_external
        pathology_rows_by_architecture[architecture] = pathology_rows
        vctk_external_by_architecture[architecture] = vctk_external
        vctk_rows_by_architecture[architecture] = vctk_rows
        eligible_by_architecture[architecture] = eligible
        decision_by_architecture[architecture] = decision

    selected_calibrator = calibrators[selected_architecture]
    selected_decision = decision_by_architecture[selected_architecture]
    selected_eligible = eligible_by_architecture[selected_architecture]
    if selected_decision == "ELIGIBLE_FOR_MULTISEED_CONFIRMATION":
        conclusion = (
            "Route C met the minimum component-family gate and advances only to "
            "locked multi-seed confirmation; no generator training starts yet."
        )
    else:
        conclusion = (
            "Route C did not qualify enough AVQI components across two concept "
            "families; generator training remains blocked."
        )

    route_report = {
        "selected_architecture": selected_architecture,
        "selection_rule": "lowest calibration loss before holdout evaluation",
        "route_family": "C",
        "neural_predictor": False,
        "training": training_by_architecture,
        "all_architecture_metrics": metrics_by_architecture,
        "metrics": metrics_by_architecture[selected_architecture]["calibrated"],
        "calibration": {
            "scale": selected_calibrator.scale.detach().cpu().tolist(),
            "bias": selected_calibrator.bias.detach().cpu().tolist(),
        },
        "anti_shortcut": anti_shortcut_by_architecture[selected_architecture],
        "training_segment_transfer": segment_transfer_by_architecture[
            selected_architecture
        ],
        "gradient": gradient_by_architecture[selected_architecture],
        "external_enhancement_stress": pathology_external_by_architecture[
            selected_architecture
        ],
        "vctk_external_own_target_stress": vctk_external_by_architecture[
            selected_architecture
        ],
        "external_evaluation_by_architecture": {
            architecture: {
                "pathology": pathology_external_by_architecture[architecture],
                "vctk": vctk_external_by_architecture[architecture],
            }
            for architecture in architectures
        },
        "qualification_by_architecture": {
            architecture: {
                "anti_shortcut": anti_shortcut_by_architecture[architecture],
                "training_segment_transfer": segment_transfer_by_architecture[
                    architecture
                ],
                "gradient": gradient_by_architecture[architecture],
                "eligible_components": eligible_by_architecture[architecture],
                "decision": decision_by_architecture[architecture],
            }
            for architecture in architectures
        },
        "eligible_components": selected_eligible,
        "decision": selected_decision,
    }
    report = {
        "decision": "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE",
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "plain_language_conclusion": conclusion,
        "contract": contract,
        "routes": {
            "shared_dual_head": {"status": "SKIPPED_USER_SCOPE"},
            "frozen_independent_predictor": {"status": "SKIPPED_USER_SCOPE"},
            ROUTE_KEY: route_report,
        },
        "coverage": label_bank_coverage,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    write_json(args.output_dir / "diagnostic_report.json", report)
    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows(
            examples,
            {
                f"direct_{name}": prediction
                for name, prediction in predictions_by_architecture.items()
            },
        ),
    )
    selected_pathology_rows = pathology_rows_by_architecture[selected_architecture]
    selected_vctk_rows = vctk_rows_by_architecture[selected_architecture]
    write_csv(
        args.output_dir / "external_direct_predictions.csv",
        selected_pathology_rows,
    )
    write_csv(
        args.output_dir / "vctk_external_direct_predictions.csv",
        selected_vctk_rows,
    )
    for architecture in architectures:
        write_csv(
            args.output_dir / f"external_direct_{architecture}_predictions.csv",
            pathology_rows_by_architecture[architecture],
        )
        write_csv(
            args.output_dir / f"vctk_external_direct_{architecture}_predictions.csv",
            vctk_rows_by_architecture[architecture],
        )
    (args.output_dir / "SUMMARY.md").write_text(
        human_summary(report),
        encoding="utf-8",
    )

    artifact_names = [
        "experiment_contract.json",
        "diagnostic_report.json",
        "predictions.csv",
        "external_direct_predictions.csv",
        "vctk_external_direct_predictions.csv",
        "SUMMARY.md",
        *(f"external_direct_{name}_predictions.csv" for name in architectures),
        *(
            f"vctk_external_direct_{name}_predictions.csv"
            for name in architectures
        ),
    ]
    receipt = {
        "decision": report["decision"],
        "route_scope": "direct_only",
        "route_c": selected_decision,
        "selected_direct_architecture": selected_architecture,
        "eligible_components": selected_eligible,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            name: sha256_file(args.output_dir / name) for name in artifact_names
        },
        "checkpoint_sha256": {
            path.name: sha256_file(path)
            for path in sorted(args.checkpoint_dir.glob("*.pt"))
        },
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
