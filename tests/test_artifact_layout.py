from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_repository_documents_separate_checkpoint_and_run_roots() -> None:
    checkpoint_policy = read("checkpoints/README.md")
    run_policy = read("runs/README.md")
    assert "checkpoints/<experiment>/<run_id>/" in checkpoint_policy
    assert "New model weights do not belong here" in run_policy
    assert "checkpoints/<experiment>/<run_id>/" in run_policy


def test_avqi_diagnostic_separates_models_from_reports() -> None:
    source = read("scripts/evaluate_avqi_component_backprop.py")
    launcher = read(
        "scripts/cluster/slurm_avqi_component_backprop_diagnostic.sh"
    )
    assert 'parser.add_argument("--checkpoint-dir"' in source
    assert 'parser.add_argument("--config-sha256"' in source
    assert 'parser.add_argument("--source-commit"' in source
    assert 'args.output_dir / "checkpoints"' not in source
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in launcher
    assert '--config-sha256 "$CONFIG_SHA256"' in launcher
    assert '--source-commit "$SOURCE_COMMIT"' in launcher
