from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_repository_documents_compact_artifact_contract() -> None:
    readme = read("README.md")
    checkpoint_manifest = read("checkpoints/manifest.csv")
    assert (
        "runs/              logs, manifests, metrics, and reports; no model weights"
        in readme
    )
    assert "pretrain_180k/ln_g_00180000.pth" in checkpoint_manifest
    assert "B0_250/ln_g_00000250.pth" in checkpoint_manifest
    assert "S6_500/ln_g_00000500.pth" in checkpoint_manifest
    assert "S3_2000/ln_g_00002000.pth" in checkpoint_manifest
    assert not (REPO_ROOT / "scripts" / "cluster").exists()


def test_avqi_diagnostic_separates_models_from_reports() -> None:
    source = read("scripts/evaluate_avqi_component_backprop.py")
    direct_source = read("scripts/evaluate_avqi_component_direct_c.py")
    launcher = read("scripts/slurm_avqi_component_backprop_diagnostic.sh")
    multiseed = read("scripts/summarize_avqi_component_multiseed.py")
    assert 'parser.add_argument("--checkpoint-dir"' in source
    assert 'parser.add_argument("--config-sha256"' in source
    assert 'parser.add_argument("--source-commit"' in source
    assert 'choices=("direct_only",)' in direct_source
    assert 'ROUTE_KEY = "direct_differentiable_estimator"' in direct_source
    assert '"direct_praat_hard_shimmer_rms_v3"' in source
    assert '"direct_praat_hard_shimmer_rms_v3"' in direct_source
    assert '"direct_praat_hard_shimmer_rms_v3"' in multiseed
    assert '"direct_praat_hard_shimmer_raw_cc_surrogate_v4"' in source
    assert '"direct_praat_hard_shimmer_raw_cc_surrogate_v4"' in direct_source
    assert '"direct_praat_hard_shimmer_raw_cc_surrogate_v4"' in multiseed
    assert '"direct_praat_hard_shimmer_pulse_chain_v5"' in source
    assert '"direct_praat_hard_shimmer_pulse_chain_v5"' in direct_source
    assert '"direct_praat_hard_shimmer_pulse_chain_v5"' in multiseed
    assert '"direct_praat_hard_shimmer_pulse_path_v6"' in source
    assert '"direct_praat_hard_shimmer_pulse_path_v6"' in direct_source
    assert '"direct_praat_hard_shimmer_pulse_path_v6"' in multiseed
    assert '"direct_praat_hard_cpps_relative_log1p_v10"' in source
    assert '"direct_praat_hard_cpps_relative_log1p_v10"' in direct_source
    assert '"direct_praat_hard_cpps_relative_log1p_v10"' in multiseed
    assert '"direct_praat_hard_cpps_view_input_v12"' in source
    assert '"direct_praat_hard_cpps_view_input_v12"' in direct_source
    assert '"shared_dual_head": {"status": "SKIPPED_USER_SCOPE"}' in direct_source
    assert (
        '"frozen_independent_predictor": {"status": "SKIPPED_USER_SCOPE"}'
        in direct_source
    )
    assert "train_shared_head" not in direct_source
    assert 'training["optimizer_steps"] != 0' in direct_source
    assert 'args.output_dir / "checkpoints"' not in source
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in launcher
    assert '--config-sha256 "$CONFIG_SHA256"' in launcher
    assert '--source-commit "$SOURCE_COMMIT"' in launcher
    assert 'ROUTE_SCOPE="${ROUTE_SCOPE:-all}"' in launcher
    assert "SHARED_CANDIDATE_ARGS=()" in launcher
    assert "export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT" in launcher
    assert "export RUN_ROOT LOG_DIR OUTPUT_DIR CHECKPOINT_DIR" in launcher
    assert '"late_tfgrid"' in source
    assert '"compact_tfgrid"' in source
    assert '"pretrained_full_tfgrid"' in source
    assert '"direct_exact_inspired"' in source
    assert '"direct_praat_soft_v2"' in source
    assert '"direct_praat_hard_v2"' in source
    assert '"neural_predictor": False' in source
    assert 'parser.add_argument("--full-tfgrid-checkpoint"' in source
    assert '--full-tfgrid-checkpoint "$FULL_TFGRID_CHECKPOINT"' in launcher
    assert "FULL_TFGRID_CHECKPOINT_SHA256" in launcher
    assert "circular_shift_100ms" in source
    assert "ELIGIBLE_FOR_MULTISEED_CONFIRMATION" in source
    assert ").clamp_min(1e-4)" in source
    assert "set_model_seed(seed)" in source
    assert 'EXTERNAL_PRIMARY_CANDIDATE = "S3_500"' in source
    assert '"view=sv&sample_group=pathological_severe"' in source
    assert "args.checkpoint.parent.name != EXTERNAL_PRIMARY_CANDIDATE" in source
    assert "component_input_gradient_report" in source
    assert 'gradient["component_input_gradients"]' in source
    assert "SCREEN_BATCH_SIZE = 16" in source
    assert "SCREEN_GRADIENT_CLIP_NORM = 5.0" in source
    assert '"gradient_clip_norm": SCREEN_GRADIENT_CLIP_NORM' in source
    assert "equal shared and waveform epochs" in source
    assert '"--expected-train-speakers"' in source
    assert '"--expected-calibration-speakers"' in source
    assert '"--expected-holdout-speakers"' in source
    assert "MIN_LABEL_BANK_COVERAGE = 0.95" in source
    assert "MIN_LABEL_BANK_SLICE_COVERAGE = 0.90" in source
    assert '--expected-train-speakers "$EXPECTED_TRAIN_SPEAKERS"' in launcher
    assert "external_coverage_report" in source
    assert "training_segment_transfer_report" in source
    assert "TRAINING_SEGMENT_SAMPLES = 48_000" in source
    assert 'segment_transfer["components"][component]["decision"]' in source
    assert "CONSENSUS_PASS_COUNT = 2" in multiseed
    assert '"generator_optimizer_steps": 0' in multiseed
    assert '"source_report_sha256"' in multiseed
    assert "refusing to overwrite output" in multiseed


def test_avqi_shimmer_internal_pilot_is_exact_rescored_and_fail_closed() -> None:
    source = read("scripts/evaluate_avqi_shimmer_internal_pilot.py")
    assert '"PASS_DEPLOYABLE_PATH_HISTORICAL_PILOT_ONLY"' in source
    assert '"authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING"' in source
    assert '"promotion_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"exact_pulses_supplied_to_estimator": False' in source
    assert '"exact_after_used_during_alpha_selection": False' in source
    assert '"alpha_selection_uses_only_internal_proxy": True' in source
    assert "gates = {split: split_gate" in source
    assert 'for split in ("calibration", "holdout")' in source


def test_route_c_hnr_formula_diagnostic_is_nonfinal_and_read_only() -> None:
    source = read("scripts/evaluate_direct_avqi_hnr_formula.py")
    launcher = read("scripts/run_avqi_direct_hnr_formula_diagnostic.sh")
    model = read("model/avqi_components.py")
    assert '"raw_cc_v3"' in model
    assert 'hnr_mode: str = "linear_ac_v2"' in model
    assert '"selection_split": "surrogate_calibration"' in source
    assert '"holdout_used_for_selection": False' in source
    assert '"final_pathological_waveform_pilot_panel_accessed": False' in source
    assert '"bounded_waveform_pilot_authorized": False' in source
    assert '"formal_generator_training_authorized": False' in source
    assert '"waveform_optimizer_steps": 0' in source
    assert '"generator_optimizer_steps": 0' in source
    assert "GO_INTEGRATE_RAW_CC_V3_INTO_FULL_ROUTE_C_SCREEN" in source
    assert "MINIMUM_RELATIVE_CALIBRATION_IMPROVEMENT" in launcher
    assert "CONFIRM_SLURM_SUBMIT" in launcher
    assert "Refusing to run from a dirty source tree" in launcher
    assert "--expected-internal-valid-rows 2134" in launcher
    assert "--expected-internal-usable-rows 2106" in launcher
    assert "missing_clean_pair_rows" in source


def test_avqi_expansion_is_training_only_and_hash_locked() -> None:
    prepare = read("scripts/prepare_avqi_component_expanded_data.py")
    score = read("scripts/build_avqi_component_expanded_label_bank.py")
    launcher = read("scripts/slurm_prepare_avqi_component_expanded_data.sh")
    assert '"surrogate_train"' in score
    assert "expansion speakers overlap the base label bank" in score
    assert "expansion speakers overlap the external test panel" in score
    assert 'if args.output_dir.exists()' in prepare
    assert "refusing to overwrite output" in prepare
    assert "--noise-manifest-sha256" in prepare
    assert "--rir-manifest-sha256" in prepare
    assert 'row.get("_shard_dir") or root' in prepare
    assert "EXPECTED_NEW_SPEAKERS=\"${EXPECTED_NEW_SPEAKERS:-55}\"" in launcher
    assert "--expected-train-speakers 125" in launcher
    assert "CONFIRM_SLURM_SUBMIT" in launcher


def test_avqi_phaseaware_v4_is_speaker_disjoint_and_no_train_highpass() -> None:
    prepare = read("scripts/prepare_avqi_component_v4_vctk.py")
    score = read("scripts/build_avqi_component_v4_label_bank.py")
    selection = read("scripts/avqi_vctk_selection.py")
    launcher = read("scripts/run_avqi_component_v4_data.sh")
    screen = read("scripts/run_avqi_component_v4_screen.sh")
    confirm = read("scripts/run_avqi_component_v4_confirm.sh")
    multiseed_runner = read("scripts/run_avqi_component_v4_multiseed.sh")
    promotion = read("scripts/evaluate_avqi_component_v4_phase_promotion.py")
    promotion_runner = read("scripts/run_avqi_component_v4_phase_promotion.sh")
    frozen_contract = read("configs/avqi_component_phaseaware_v4.yaml")
    diagnostic = read("scripts/evaluate_avqi_component_backprop.py")
    data_finalizer = read("scripts/finalize_avqi_component_v4_data.py")
    test_runner = read("scripts/run_avqi_component_v4_tests.sh")
    assert '"surrogate_train": 72' in prepare
    assert '"surrogate_calibration": 12' in prepare
    assert '"surrogate_holdout": 12' in prepare
    assert '"vctk_external": 12' in prepare
    assert '"metric_branch_highpass_applied": 0' in prepare
    assert 'default=2' in prepare
    assert '"external_reserve_utterances_per_speaker"' in prepare
    assert "popitem(last=False)" in prepare
    assert '"--max-open-shards"' in prepare
    assert '"speaker_overlap_with_base": 0' in score
    assert "select_exact_complete_external_rows" in score
    assert 'default=1_824' in score
    assert "minimum-split-condition-coverage" in score
    assert '"metric_values_used_for_selection": False' in selection
    assert '"selected_external_valid_rows"' in selection
    assert "select_nonzero_noise_crop" in prepare
    assert '"zero_energy_noise_crops_rejected"' in prepare
    assert '"maximum_noise_crop_attempts"' in data_finalizer
    assert 'EXTERNAL_RESERVE_UTTERANCES="${EXTERNAL_RESERVE_UTTERANCES:-2}"' in launcher
    assert '--expected-vctk-rows "$EXPECTED_VCTK_CANDIDATE_ROWS"' in launcher
    assert "CONFIRM_SLURM_SUBMIT" in launcher
    assert '"output_phase_tfgrid"' in diagnostic
    assert '"phase_frequency_aware"' in diagnostic
    assert '"phase_compact_tfgrid"' in diagnostic
    assert '"--max-optimizer-steps"' in diagnostic
    assert '"head_gradients_absent": head_gradients_absent' in diagnostic
    assert '"qualification_by_architecture"' in diagnostic
    assert "independent_gradient_smoke" in diagnostic
    assert 'SHARED_CANDIDATES="output_phase_tfgrid"' in screen
    assert "frequency_aware,phase_frequency_aware,phase_compact_tfgrid" in screen
    assert (
        'WAVEFORM_ARCHITECTURES="${WAVEFORM_ARCHITECTURES:-direct_praat_hard_v2}"'
        in screen
    )
    assert 'WAVEFORM_ARCHITECTURES="pretrained_full_tfgrid"' in screen
    assert 'SCREEN_KIND" == "full_tfgrid"' in screen
    assert 'EXPECTED_TRAIN_SPEAKERS="${EXPECTED_TRAIN_SPEAKERS:-197}"' in screen
    assert 'SEED="${SEED:-20260815}"' in screen
    assert 'DEPENDENCY_ARGS=(--dependency="afterok:$DEPENDENCY_JOB_ID")' in screen
    assert "exec bash \"$DIAGNOSTIC_LAUNCHER\"" in screen
    assert "CONFIRMATION_SEEDS=(20260816 20260817 20260818)" in confirm
    assert 'CONFIRM_KIND="${CONFIRM_KIND:-phase}"' in confirm
    assert 'ROUTE_SCOPE" != "direct_only"' in confirm
    assert '.routes.direct_differentiable_estimator.selected_architecture' in confirm
    assert "direct_praat_hard_shimmer_raw_cc_surrogate_v4" in confirm
    assert "direct_praat_hard_shimmer_pulse_chain_v5" in confirm
    assert "direct_praat_hard_shimmer_pulse_path_v6" in confirm
    assert "direct_praat_hard_cpps_relative_log1p_v10" in confirm
    assert 'full:pretrained_full_tfgrid' in confirm
    assert 'CONFIRM_RUN_STEM="avqi_component_direct_c_v6_confirm"' in confirm
    assert 'SHARED_CANDIDATES="$(jq -er' in confirm
    assert 'WAVEFORM_ARCHITECTURES="$(jq -er' in confirm
    assert 'contract.source_commit' in confirm
    assert "exec bash \"$DIAGNOSTIC_LAUNCHER\"" in confirm
    assert 'CONSENSUS_KIND="${CONSENSUS_KIND:-phase}"' in multiseed_runner
    assert 'DEPENDENCY_ARGS=(--dependency="afterok:$NORMALIZED_JOB_IDS")' in multiseed_runner
    assert 'ARGS+=(--confirmation-report "$path")' in multiseed_runner
    assert 'python3 "$SUMMARY_SCRIPT" "${ARGS[@]}"' in multiseed_runner
    assert 'CONFIRM_RUN_STEM="avqi_component_pretrained_full_tfgrid_v4_confirm"' in multiseed_runner
    assert 'PROMOTE_DECISION = "PROMOTE_PRETRAINED_FULL_TFGRID_SCREEN"' in promotion
    assert "all_required_slice_medians_non_regressed" in promotion
    assert '"generator_optimizer_steps": 0' in promotion
    assert "PROMOTE_PRETRAINED_FULL_TFGRID_SCREEN" in promotion_runner
    assert "env -u SLURM_JOB_ID" in promotion_runner
    assert 'SCREEN_KIND=full_tfgrid' in promotion_runner
    assert "full_tfgrid_submission.json" in promotion_runner
    assert '"DATA_READY_FOR_SCORER_SCREENS"' in data_finalizer
    assert '"merged_internal_rows": 2_454' in data_finalizer
    assert '"prepared_candidate_rows"' in data_finalizer
    assert '"selected_external_valid_rows"' in data_finalizer
    assert '"surrogate_train": 1_152' in data_finalizer
    assert '"vctk_external": 288' in data_finalizer
    assert 'condition: 456 for condition in EXPECTED_CONDITIONS' in data_finalizer
    assert "split-condition exact coverage failed" in data_finalizer
    assert '"waveform_highpass_applied": False' in data_finalizer
    assert "tests/test_shifted_anechoic_target.py" in test_runner
    assert 'python -m pytest -q "${TEST_FILES[@]}"' in test_runner
    assert '"PASS_AVQI_V4_REPOSITORY_TESTS"' in test_runner
    assert "pytest_log_sha256" in test_runner
    assert "metric_alignment: tail_crop_to_shortest_only_no_shift_filter_or_resample" in frozen_contract
    assert "pathology_db_median_gap_increase_max: 0.50" in frozen_contract
    assert "denoising_median_change_min_db: -0.10" in frozen_contract


def test_avqi_phaseaware_v4_full_tfgrid_promotion_is_frozen_and_conservative() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_component_v4_phase_promotion.py"
    )
    components = (
        "cpps",
        "hnr",
        "shimmer_percent",
        "shimmer_db",
        "slope",
        "tilt",
    )
    architectures = (
        "frequency_aware",
        "phase_frequency_aware",
        "phase_compact_tfgrid",
    )

    def metrics(nmae: float) -> dict[str, dict[str, float]]:
        return {
            component: {"normalized_mae": nmae}
            for component in components
        }

    def coverage() -> dict[str, str]:
        return {"decision": "PASS"}

    def pathology_report(nmae: float) -> dict[str, object]:
        slice_names = (
            "view=cs",
            "view=sv",
            "label=healthy",
            "label=patient",
            "view=sv&sample_group=pathological_severe",
            "condition=snr10",
        )
        return {
            "slices": {name: metrics(nmae) for name in slice_names},
            "slice_coverage": {name: coverage() for name in slice_names},
        }

    def vctk_report(nmae: float) -> dict[str, object]:
        slice_names = tuple(
            f"condition={condition}"
            for condition in ("clean", "rir_only", "snr20", "snr10")
        )
        return {
            "primary": metrics(nmae),
            "primary_coverage": coverage(),
            "slices": {name: metrics(nmae) for name in slice_names},
            "slice_coverage": {name: coverage() for name in slice_names},
        }

    nmae = {
        "frequency_aware": 0.40,
        "phase_frequency_aware": 0.35,
        "phase_compact_tfgrid": 0.25,
    }
    screen = {
        "decision": "COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE",
        "generator_optimizer_steps": 0,
        "formal_pathology_training_submitted": False,
        "contract": {
            "components": list(components),
            "routes": {
                "frozen_independent_predictor": {
                    "architectures": list(architectures)
                }
            },
        },
        "routes": {
            "frozen_independent_predictor": {
                "selected_architecture": "phase_compact_tfgrid",
                "training": {
                    "frequency_aware": {"best_calibration_loss": 0.10},
                    "phase_frequency_aware": {"best_calibration_loss": 0.09},
                    "phase_compact_tfgrid": {"best_calibration_loss": 0.08},
                },
                "all_architecture_metrics": {
                    architecture: {
                        "calibrated": {
                            "primary": metrics(nmae[architecture]),
                            "slices": {
                                name: metrics(nmae[architecture])
                                for name in ("cs", "sv", "healthy", "patient")
                            },
                        }
                    }
                    for architecture in architectures
                },
                "external_evaluation_by_architecture": {
                    architecture: {
                        "pathology": pathology_report(nmae[architecture]),
                        "vctk": vctk_report(nmae[architecture]),
                    }
                    for architecture in architectures
                },
                "qualification_by_architecture": {
                    "frequency_aware": {"eligible_components": ["hnr"]},
                    "phase_frequency_aware": {"eligible_components": ["hnr"]},
                    "phase_compact_tfgrid": {
                        "eligible_components": ["hnr", "tilt"]
                    },
                },
            }
        },
    }
    promoted = namespace["evaluate_promotion"](screen)
    assert promoted["decision"] == "PROMOTE_PRETRAINED_FULL_TFGRID_SCREEN"
    assert all(promoted["gates"].values())

    screen["routes"]["frozen_independent_predictor"][
        "external_evaluation_by_architecture"
    ]["phase_compact_tfgrid"]["vctk"]["slices"]["condition=snr10"] = metrics(
        0.45
    )
    rejected = namespace["evaluate_promotion"](screen)
    assert rejected["decision"] == "KEEP_COMPACT_NO_FULL_TFGRID"
    assert not rejected["gates"]["all_required_slice_medians_non_regressed"]


def test_direct_avqi_waveform_optimization_is_exact_scored_and_bounded() -> None:
    source = read("scripts/evaluate_direct_avqi_waveform_optimization.py")
    audit = read("scripts/audit_avqi_waveform_guardrails.py")
    audit_runner = read("scripts/run_avqi_waveform_guardrail_audit.sh")
    pilot_runner = read("scripts/run_avqi_component_direct_c_waveform_pilot.sh")
    assert 'CANDIDATE = "S3_500"' in source
    assert 'CONDITION = "snr10"' in source
    assert 'OPTIMIZED_COMPONENTS = ("hnr", "tilt")' in source
    assert "validate_route_c_authorization" in source
    assert 'parser.add_argument("--authorization-consensus"' in source
    assert 'parser.add_argument("--screen-report"' in source
    assert "screen_component_gradient_norms" in source
    assert "optimization_component_weights" in source
    assert "avqi_code_tree_sha256" in source
    assert 'parser.add_argument("--exact-python"' in source
    assert "AVQI_EXACT_JSON=" in source
    assert "declared source commit differs from repository HEAD" in source
    assert 'parser.add_argument("--speaker-offset"' in source
    assert "project_residual" in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"formal_pathology_training_submitted": False' in source
    assert "exact_absolute_gap_after" in source
    assert "full_band_pathology_guardrails" in source
    assert "same-speaker clean pathological CS or SV waveform" in source
    assert '"low_20_80hz"' in source
    assert "airflow_proxy_flatness_gap_increase" in source
    assert "pause_f1_change" in source
    assert "si_sdr_change_db" in source
    assert "required_slices" in source
    assert "audit_waveform_optimizer_steps" in audit
    assert "source waveform hash drift" in audit
    assert "optimized waveform hash drift" in audit
    assert "declared audit source commit differs from repository HEAD" in audit
    assert '"generator_optimizer_steps": 0' in audit
    assert 'SOURCE_RESULTS_CSV_SHA256="2d376668' in audit_runner
    assert 'PREDICTOR_CHECKPOINT_SHA256="11779b5' in audit_runner
    assert "CONFIRM_SLURM_SUBMIT" in audit_runner
    assert 'python "$AUDIT_SCRIPT"' in audit_runner
    assert 'SPEAKER_OFFSET="${SPEAKER_OFFSET:-4}"' in pilot_runner
    assert 'LEARNING_RATE_SCALE="${LEARNING_RATE_SCALE:-0.0002}"' in pilot_runner
    assert "waveform_pilot_offset4_20260817_02" in pilot_runner
    assert "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" in pilot_runner
    assert (
        'CONSENSUS_REPORT_SHA256="${CONSENSUS_REPORT_SHA256:-0a2b297'
        in pilot_runner
    )
    assert (
        'PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-07b69e'
        in pilot_runner
    )
    assert "verify_sha256" in pilot_runner
    assert '--authorization-consensus "$CONSENSUS_REPORT"' in pilot_runner
    assert '--slurm-job-id "$SLURM_JOB_ID"' in pilot_runner
    assert "CONFIRM_SLURM_SUBMIT" in pilot_runner
    assert "FAIL_WAVEFORM_OPTIMIZATION" in source


def test_avqi_diagnostic_entry_point_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_avqi_component_backprop.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Speaker-disjoint diagnostic" in result.stdout


def test_avqi_multiseed_entry_point_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_avqi_component_multiseed.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "three locked AVQI-component predictor confirmations" in result.stdout


def test_avqi_route_c_entry_point_runs_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_avqi_component_direct_c.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Route C-only AVQI component screen" in result.stdout


def test_avqi_multiseed_consensus_uses_two_of_three_complete_passes() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    confirmations = [
        {
            "routes": {
                "shared_dual_head": {
                    "selected_candidate": "late_tfgrid",
                    "eligible_components": components,
                },
                "frozen_independent_predictor": {
                    "selected_architecture": "compact_tfgrid",
                    "eligible_components": ["cpps", "tilt"],
                },
            }
        }
        for components in (
            ["cpps", "slope"],
            ["cpps", "slope"],
            ["hnr", "tilt"],
        )
    ]
    route = namespace["route_consensus"](confirmations, "shared_dual_head")
    assert route["consensus_components"] == ["cpps", "slope"]
    assert route["decision"] == "RELIABLE"


def test_avqi_multiseed_accepts_calibration_selected_direct_v2_screen() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    screen = {
        "contract": {
            "routes": {
                "shared_dual_head": {"candidates": ["late_tfgrid"]},
                "frozen_independent_predictor": {
                    "architectures": [
                        "direct_praat_soft_v2",
                        "direct_praat_hard_v2",
                    ]
                },
            },
            "matched_training_budget": {
                "shared_max_epochs": 60,
                "independent_max_epochs": 60,
            },
            "calibration": {"holdout_used_for_fit_or_selection": False},
        },
        "routes": {
            "shared_dual_head": {
                "selected_candidate": "late_tfgrid",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "late_tfgrid": {"best_calibration_loss": 0.12}
                },
            },
            "frozen_independent_predictor": {
                "selected_architecture": "direct_praat_hard_v2",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "direct_praat_soft_v2": {"best_calibration_loss": 0.13},
                    "direct_praat_hard_v2": {"best_calibration_loss": 0.06},
                },
            },
        },
    }
    namespace["validate_screen_contract"](
        screen,
        Path("direct-v2-screen.json"),
    )


def test_avqi_multiseed_accepts_phaseaware_v4_screen() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    screen = {
        "contract": {
            "routes": {
                "shared_dual_head": {"candidates": ["output_phase_tfgrid"]},
                "frozen_independent_predictor": {
                    "architectures": [
                        "frequency_aware",
                        "phase_frequency_aware",
                        "phase_compact_tfgrid",
                    ]
                },
            },
            "matched_training_budget": {
                "shared_max_epochs": 60,
                "independent_max_epochs": 60,
            },
            "calibration": {"holdout_used_for_fit_or_selection": False},
        },
        "routes": {
            "shared_dual_head": {
                "selected_candidate": "output_phase_tfgrid",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "output_phase_tfgrid": {"best_calibration_loss": 0.10}
                },
            },
            "frozen_independent_predictor": {
                "selected_architecture": "phase_compact_tfgrid",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "frequency_aware": {"best_calibration_loss": 0.12},
                    "phase_frequency_aware": {"best_calibration_loss": 0.09},
                    "phase_compact_tfgrid": {"best_calibration_loss": 0.08},
                },
            },
        },
    }
    namespace["validate_screen_contract"](
        screen,
        Path("phase-v4-screen.json"),
    )


def test_avqi_multiseed_accepts_pretrained_full_tfgrid_screen() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    screen = {
        "contract": {
            "routes": {
                "shared_dual_head": {"candidates": ["output_phase_tfgrid"]},
                "frozen_independent_predictor": {
                    "architectures": ["pretrained_full_tfgrid"]
                },
            },
            "matched_training_budget": {
                "shared_max_epochs": 60,
                "independent_max_epochs": 60,
            },
            "calibration": {"holdout_used_for_fit_or_selection": False},
        },
        "routes": {
            "shared_dual_head": {
                "selected_candidate": "output_phase_tfgrid",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "output_phase_tfgrid": {"best_calibration_loss": 0.10}
                },
            },
            "frozen_independent_predictor": {
                "selected_architecture": "pretrained_full_tfgrid",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "pretrained_full_tfgrid": {"best_calibration_loss": 0.08}
                },
            },
        },
    }
    namespace["validate_screen_contract"](
        screen,
        Path("full-tfgrid-v4-screen.json"),
    )


def test_avqi_multiseed_accepts_route_c_only_screen() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    screen = {
        "contract": {
            "route_scope": "direct_only",
            "routes": {
                "shared_dual_head": {
                    "status": "SKIPPED_USER_SCOPE",
                    "candidates": [],
                },
                "frozen_independent_predictor": {
                    "status": "SKIPPED_USER_SCOPE",
                    "architectures": [],
                },
                "direct_differentiable_estimator": {
                    "architectures": ["direct_praat_hard_v2"]
                },
            },
            "direct_formula_budget": {
                "trainable_parameters": 0,
                "optimizer_steps": 0,
                "maximum_optimizer_steps": 0,
            },
            "calibration": {"holdout_used_for_fit_or_selection": False},
        },
        "routes": {
            "shared_dual_head": {"status": "SKIPPED_USER_SCOPE"},
            "frozen_independent_predictor": {"status": "SKIPPED_USER_SCOPE"},
            "direct_differentiable_estimator": {
                "selected_architecture": "direct_praat_hard_v2",
                "selection_rule": "lowest calibration loss before holdout evaluation",
                "training": {
                    "direct_praat_hard_v2": {
                        "best_calibration_loss": 0.06,
                        "optimizer_steps": 0,
                        "trainable_parameter_count": 0,
                    }
                },
            },
        },
    }
    namespace["validate_screen_contract"](
        screen,
        Path("route-c-screen.json"),
    )


def test_avqi_multiseed_accepts_calibration_selected_shimmer_formula_screens() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    for shimmer_architecture in (
        "direct_praat_hard_shimmer_raw_cc_surrogate_v4",
        "direct_praat_hard_shimmer_pulse_chain_v5",
        "direct_praat_hard_shimmer_pulse_path_v6",
    ):
        architectures = [
            "direct_praat_hard_v2",
            shimmer_architecture,
        ]
        screen = {
            "contract": {
                "route_scope": "direct_only",
                "routes": {
                    "shared_dual_head": {
                        "status": "SKIPPED_USER_SCOPE",
                        "candidates": [],
                    },
                    "frozen_independent_predictor": {
                        "status": "SKIPPED_USER_SCOPE",
                        "architectures": [],
                    },
                    "direct_differentiable_estimator": {
                        "architectures": architectures,
                    },
                },
                "direct_formula_budget": {
                    "trainable_parameters": 0,
                    "optimizer_steps": 0,
                    "maximum_optimizer_steps": 0,
                },
                "calibration": {"holdout_used_for_fit_or_selection": False},
            },
            "routes": {
                "shared_dual_head": {"status": "SKIPPED_USER_SCOPE"},
                "frozen_independent_predictor": {"status": "SKIPPED_USER_SCOPE"},
                "direct_differentiable_estimator": {
                    "selected_architecture": shimmer_architecture,
                    "selection_rule": (
                        "lowest calibration loss before holdout evaluation"
                    ),
                    "training": {
                        "direct_praat_hard_v2": {
                            "best_calibration_loss": 0.06,
                            "optimizer_steps": 0,
                            "trainable_parameter_count": 0,
                        },
                        shimmer_architecture: {
                            "best_calibration_loss": 0.05,
                            "optimizer_steps": 0,
                            "trainable_parameter_count": 0,
                        },
                    },
                },
            },
        }
        namespace["validate_screen_contract"](
            screen,
            Path(f"route-c-{shimmer_architecture}-screen.json"),
        )


def test_avqi_multiseed_route_c_authorizes_only_bounded_waveform_pilot() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    direct_route = namespace["DIRECT_ROUTE"]
    routes = {
        direct_route: {
            "decision": "RELIABLE",
            "consensus_components": ["hnr", "tilt"],
        }
    }
    promotion = namespace["promotion_decision"](routes, (direct_route,))
    assert promotion["decision"] == "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT"
    assert promotion["components"] == ["hnr", "tilt"]


def test_avqi_multiseed_promotion_uses_common_components_for_two_routes() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "summarize_avqi_component_multiseed.py"
    )
    routes = {
        "shared_dual_head": {
            "decision": "RELIABLE",
            "consensus_components": ["cpps", "slope", "tilt"],
        },
        "frozen_independent_predictor": {
            "decision": "RELIABLE",
            "consensus_components": ["cpps", "shimmer_db", "tilt"],
        },
    }
    promotion = namespace["promotion_decision"](routes)
    assert promotion["decision"] == "GO_MATCHED_DUAL_ROUTE_BACKPROP"
    assert promotion["components"] == ["cpps", "tilt"]


def test_avqi_phaseaware_v4_final_summary_is_fail_closed() -> None:
    source = read("scripts/summarize_avqi_component_v4_final.py")
    assert '"NO_GO_AVQI_T2_TRAINING"' in source
    assert '"new_bounded_waveform_pilot_submitted": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"formal_pathology_training_submitted": False' in source
    assert '"full_tfgrid_screen_submitted": False' in source
    assert "require_no_generator_update" in source
    assert "refusing to overwrite output" in source
