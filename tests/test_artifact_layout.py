from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_training_entry_points_separate_checkpoints_from_runs() -> None:
    scripts = (
        "scripts/train_semambapp_dnf_nogan.py",
        "scripts/train_semambapp_dnf_paper_noisy.py",
        "scripts/train_semambapp_dnf_phase_a.py",
    )
    for relative_path in scripts:
        source = read(relative_path)
        assert "--checkpoint-root" in source
        assert 'output_dir / "checkpoints"' not in source
        assert "checkpoint_dir" in source


def test_cluster_helpers_pass_checkpoint_root_explicitly() -> None:
    helpers = (
        "scripts/cluster/slurm_semambapp_dnf_nogan.sh",
        "scripts/cluster/slurm_semambapp_dnf_paper_noisy_array.sh",
        "scripts/cluster/slurm_semambapp_dnf_phase_a_array.sh",
    )
    for relative_path in helpers:
        source = read(relative_path)
        assert "CHECKPOINT_ROOT=" in source
        assert '--checkpoint-root "$CHECKPOINT_ROOT"' in source


def test_avqi_diagnostic_separates_models_from_reports() -> None:
    source = read("scripts/evaluate_avqi_component_backprop.py")
    launcher = read(
        "scripts/cluster/slurm_avqi_component_backprop_diagnostic.sh"
    )
    assert 'parser.add_argument("--checkpoint-dir"' in source
    assert 'args.output_dir / "checkpoints"' not in source
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in launcher
