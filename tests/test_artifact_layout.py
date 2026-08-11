from __future__ import annotations

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
    assert 'parser.add_argument("--checkpoint-dir"' in source
    assert 'parser.add_argument("--config-sha256"' in source
    assert 'parser.add_argument("--source-commit"' in source
    assert 'args.output_dir / "checkpoints"' not in source
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in launcher
    assert '--config-sha256 "$CONFIG_SHA256"' in launcher
    assert '--source-commit "$SOURCE_COMMIT"' in launcher


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
