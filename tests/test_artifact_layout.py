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
    launcher = read("scripts/slurm_avqi_component_backprop_diagnostic.sh")
    multiseed = read("scripts/summarize_avqi_component_multiseed.py")
    assert 'parser.add_argument("--checkpoint-dir"' in source
    assert 'parser.add_argument("--config-sha256"' in source
    assert 'parser.add_argument("--source-commit"' in source
    assert 'args.output_dir / "checkpoints"' not in source
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in launcher
    assert '--config-sha256 "$CONFIG_SHA256"' in launcher
    assert '--source-commit "$SOURCE_COMMIT"' in launcher
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
    assert '--expected-train-speakers "$EXPECTED_TRAIN_SPEAKERS"' in launcher
    assert "external_coverage_report" in source
    assert "training_segment_transfer_report" in source
    assert "TRAINING_SEGMENT_SAMPLES = 48_000" in source
    assert 'segment_transfer["components"][component]["decision"]' in source
    assert "CONSENSUS_PASS_COUNT = 2" in multiseed
    assert '"generator_optimizer_steps": 0' in multiseed
    assert '"source_report_sha256"' in multiseed
    assert "refusing to overwrite output" in multiseed


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


def test_direct_avqi_waveform_optimization_is_exact_scored_and_bounded() -> None:
    source = read("scripts/evaluate_direct_avqi_waveform_optimization.py")
    assert 'CANDIDATE = "S3_500"' in source
    assert 'CONDITION = "snr10"' in source
    assert 'OPTIMIZED_COMPONENTS = ("hnr", "tilt")' in source
    assert "SCREEN_COMPONENT_GRADIENT_NORMS" in source
    assert "OPTIMIZATION_COMPONENT_WEIGHTS" in source
    assert "avqi_code_tree_sha256" in source
    assert 'parser.add_argument("--exact-python"' in source
    assert "AVQI_EXACT_JSON=" in source
    assert "declared source commit differs from repository HEAD" in source
    assert 'parser.add_argument("--speaker-offset"' in source
    assert "project_residual" in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"formal_pathology_training_submitted": False' in source
    assert "exact_absolute_gap_after" in source
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
