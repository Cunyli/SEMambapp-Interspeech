from __future__ import annotations

import csv
import runpy
from pathlib import Path

import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_namespace() -> dict[str, object]:
    return runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_direct_avqi_waveform_optimization.py"
    )


def synthetic_rows() -> list[dict[str, float]]:
    namespace = load_namespace()
    components = namespace["AVQI_COMPONENT_NAMES"]
    optimized = namespace["OPTIMIZED_COMPONENTS"]
    rows = []
    for _ in range(12):
        row = {
            "residual_rms_db": -30.1,
            "cosine_similarity": 0.999,
            "clip_fraction": 0.0,
        }
        for component in components:
            before = 1.0
            after = 0.5 if component in optimized else 1.0
            for domain in ("surrogate", "exact"):
                row[f"{domain}_absolute_gap_before_{component}"] = before
                row[f"{domain}_absolute_gap_after_{component}"] = after
        rows.append(row)
    return rows


def test_waveform_optimization_summary_passes_consistent_exact_moves() -> None:
    namespace = load_namespace()
    summary = namespace["summarize"](
        synthetic_rows(),
        torch.ones(6),
        -30.0,
    )
    assert summary["decision"] == "PASS_WAVEFORM_OPTIMIZATION"
    assert summary["component_gates"]["hnr"]["decision"] == "PASS"
    assert summary["component_gates"]["tilt"]["decision"] == "PASS"
    assert summary["safety"]["decision"] == "PASS"


def test_waveform_optimization_summary_rejects_surrogate_only_move() -> None:
    namespace = load_namespace()
    rows = synthetic_rows()
    for row in rows:
        row["exact_absolute_gap_after_hnr"] = 1.1
    summary = namespace["summarize"](rows, torch.ones(6), -30.0)
    assert summary["decision"] == "FAIL_WAVEFORM_OPTIMIZATION"
    assert summary["component_gates"]["hnr"]["decision"] == "FAIL"


def test_project_residual_enforces_rms_and_peak_limits() -> None:
    namespace = load_namespace()
    base = torch.full((1, 8_000), 0.5)
    residual = torch.linspace(-1.0, 1.0, base.numel()).reshape_as(base)
    maximum_rms = torch.tensor(0.01)
    namespace["project_residual"](base, residual, maximum_rms)
    assert float(residual.square().mean().sqrt()) <= 0.010001
    assert float((base + residual).abs().max()) <= 0.999001


def test_load_cases_honors_speaker_offset(tmp_path: Path) -> None:
    namespace = load_namespace()
    components = namespace["AVQI_COMPONENT_NAMES"]
    waveform_path = tmp_path / "source.wav"
    sf.write(waveform_path, torch.zeros(1_600).numpy(), 16_000)
    rows = []
    for group, prefix in (
        ("pathological_mild", "mild"),
        ("pathological_severe", "severe"),
    ):
        for speaker_index in range(5):
            for view in ("cs", "sv"):
                row = {
                    "candidate": "S3_500",
                    "condition": "snr10",
                    "view": view,
                    "sample_group": group,
                    "label": "patient",
                    "scoring_status": "ok",
                    "speaker_id": f"{prefix}_{speaker_index}",
                    "cs_path": str(waveform_path),
                    "sv_path": str(waveform_path),
                }
                for component_index, component in enumerate(components):
                    row[f"clean_{component}"] = str(component_index)
                    row[f"audio_{component}"] = str(component_index + 0.5)
                rows.append(row)
    csv_path = tmp_path / "exact.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases = namespace["load_cases"](csv_path, 2, 8, 1)
    assert {case.speaker_id for case in cases} == {
        "mild_1",
        "mild_2",
        "severe_1",
        "severe_2",
    }
