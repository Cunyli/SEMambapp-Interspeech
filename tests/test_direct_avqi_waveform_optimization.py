from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

import pytest
import soundfile as sf
import torch

from scripts.audit_avqi_waveform_guardrails import clean_reference_paths


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
    for index in range(12):
        row = {
            "view": "cs" if index % 2 == 0 else "sv",
            "sample_group": (
                "pathological_mild"
                if index < 6
                else "pathological_severe"
            ),
            "residual_rms_db": -30.1,
            "cosine_similarity": 0.999,
            "clip_fraction": 0.0,
            "low_20_80hz_gap_increase_db": 0.0,
            "low_80_300hz_gap_increase_db": 0.0,
            "pause_energy_gap_increase_db": 0.0,
            "airflow_proxy_energy_gap_increase_db": 0.0,
            "airflow_proxy_flatness_gap_increase": 0.0,
            "pause_f1_change": 0.0,
            "snr_change_db": 0.0,
            "si_sdr_change_db": 0.0,
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


def test_full_band_guardrail_uses_clean_pathological_reference() -> None:
    namespace = load_namespace()
    time = torch.arange(16_000) / 16_000
    reference = (
        0.1 * torch.sin(2.0 * torch.pi * 55.0 * time)
        + 0.03 * torch.sin(2.0 * torch.pi * 180.0 * time)
        + 0.005 * torch.sin(2.0 * torch.pi * 1_200.0 * time)
    )
    report = namespace["full_band_pathology_guardrails"](
        reference,
        reference.clone(),
        reference.clone(),
    )
    assert report["guardrail_tail_trim_samples"] == 0
    assert report["low_20_80hz_gap_increase_db"] == 0.0
    assert report["low_80_300hz_gap_increase_db"] == 0.0
    assert report["pause_energy_gap_increase_db"] == 0.0
    assert report["airflow_proxy_energy_gap_increase_db"] == 0.0
    assert report["airflow_proxy_flatness_gap_increase"] == 0.0
    assert report["pause_f1_change"] == 0.0


def test_guardrail_reference_map_uses_view_specific_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "exact.csv"
    fieldnames = [
        "source_type",
        "label",
        "scoring_status",
        "speaker_id",
        "view",
        "cs_path",
        "sv_path",
    ]
    rows = [
        {
            "source_type": "clean_reference",
            "label": "patient",
            "scoring_status": "ok",
            "speaker_id": "FD11",
            "view": "both",
            "cs_path": "/ignored/combined_cs.wav",
            "sv_path": "/ignored/combined_sv.wav",
        },
        {
            "source_type": "clean_reference",
            "label": "patient",
            "scoring_status": "ok",
            "speaker_id": "FD11",
            "view": "cs",
            "cs_path": "/clean/FD11_cs.wav",
            "sv_path": "/clean/FD11_sv.wav",
        },
        {
            "source_type": "clean_reference",
            "label": "patient",
            "scoring_status": "ok",
            "speaker_id": "FD11",
            "view": "sv",
            "cs_path": "/clean/FD11_cs.wav",
            "sv_path": "/clean/FD11_sv.wav",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    references = clean_reference_paths(csv_path)

    assert references == {
        ("FD11", "cs"): Path("/clean/FD11_cs.wav"),
        ("FD11", "sv"): Path("/clean/FD11_sv.wav"),
    }


def test_label_selection_excludes_valid_rows_without_clean_target() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_component_backprop.py"
    )
    rows = [
        {
            "speaker_id": "p317",
            "sample_id": "sample_missing_clean",
            "split": "surrogate_train",
            "condition_id": "clean",
            "view": "cs",
            "scoring_status": "error",
        },
        {
            "speaker_id": "p317",
            "sample_id": "sample_missing_clean",
            "split": "surrogate_train",
            "condition_id": "snr10",
            "view": "cs",
            "scoring_status": "ok",
        },
        {
            "speaker_id": "p317",
            "sample_id": "sample_complete",
            "split": "surrogate_train",
            "condition_id": "clean",
            "view": "cs",
            "scoring_status": "ok",
        },
        {
            "speaker_id": "p317",
            "sample_id": "sample_complete",
            "split": "surrogate_train",
            "condition_id": "snr10",
            "view": "cs",
            "scoring_status": "ok",
        },
    ]

    exact, usable, missing = namespace["select_usable_label_rows"](rows)

    assert len(exact) == 3
    assert {row["sample_id"] for row in usable} == {"sample_complete"}
    assert [row["condition_id"] for row in missing] == ["snr10"]


def test_shared_prediction_uses_bounded_batches() -> None:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_component_backprop.py"
    )

    class BatchLimitedHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maximum_batch = 0

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.maximum_batch = max(self.maximum_batch, len(features))
            if len(features) > 16:
                raise RuntimeError("unbounded shared prediction batch")
            return torch.zeros(len(features), 6, device=features.device)

    head = BatchLimitedHead()
    prediction = namespace["predict_shared"](
        head,
        torch.zeros(35, 3, 8, 8),
        torch.zeros(6),
        torch.ones(6),
        torch.device("cpu"),
        "output_phase_tfgrid",
    )

    assert prediction.shape == (35, 6)
    assert head.maximum_batch == 16


def test_load_cases_honors_speaker_offset(tmp_path: Path) -> None:
    namespace = load_namespace()
    components = namespace["AVQI_COMPONENT_NAMES"]
    waveform_path = tmp_path / "source.wav"
    cs_reference_path = tmp_path / "clean_cs.wav"
    sv_reference_path = tmp_path / "clean_sv.wav"
    sf.write(waveform_path, torch.zeros(1_600).numpy(), 16_000)
    sf.write(cs_reference_path, torch.zeros(1_600).numpy(), 16_000)
    sf.write(sv_reference_path, torch.zeros(1_600).numpy(), 16_000)
    rows = []
    for group, prefix in (
        ("pathological_mild", "mild"),
        ("pathological_severe", "severe"),
    ):
        for speaker_index in range(5):
            for view in ("cs", "sv"):
                row = {
                    "source_type": "enhanced",
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
            for clean_view in ("both", "cs", "sv"):
                clean_row = {
                    "source_type": "clean_reference",
                    "candidate": "",
                    "condition": "clean_reference",
                    "view": clean_view,
                    "sample_group": group,
                    "label": "patient",
                    "scoring_status": "ok",
                    "speaker_id": f"{prefix}_{speaker_index}",
                    "cs_path": str(
                        cs_reference_path
                        if clean_view in ("cs", "sv")
                        else waveform_path
                    ),
                    "sv_path": str(
                        sv_reference_path
                        if clean_view in ("cs", "sv")
                        else waveform_path
                    ),
                }
                for component_index, component in enumerate(components):
                    clean_row[f"clean_{component}"] = str(component_index)
                    clean_row[f"audio_{component}"] = str(component_index)
                rows.append(clean_row)
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
    assert all(
        case.reference_path
        == {"cs": cs_reference_path, "sv": sv_reference_path}[case.view]
        for case in cases
    )


def test_route_c_authorization_binds_screen_and_checkpoint(
    tmp_path: Path,
) -> None:
    namespace = load_namespace()
    sha256_file = namespace["sha256_file"]
    validate = namespace["validate_route_c_authorization"]
    predictor = tmp_path / "direct_direct_praat_hard_v2_estimator.pt"
    predictor.write_bytes(b"frozen direct estimator")
    predictor_sha256 = sha256_file(predictor)
    screen = {
        "decision": "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE",
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "contract": {
            "route_scope": "direct_only",
            "source_commit": "f" * 40,
        },
        "routes": {
            "direct_differentiable_estimator": {
                "selected_architecture": "direct_praat_hard_v2",
                "decision": "ELIGIBLE_FOR_MULTISEED_CONFIRMATION",
                "eligible_components": ["hnr", "tilt"],
                "gradient": {
                    "decision": "PASS",
                    "component_input_gradients": {
                        "hnr": {"decision": "PASS", "gradient_norm": 2.0},
                        "tilt": {"decision": "PASS", "gradient_norm": 100.0},
                    },
                },
            }
        },
    }
    screen_path = tmp_path / "diagnostic_report.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    screen_sha256 = sha256_file(screen_path)
    screen_receipt = {
        "decision": screen["decision"],
        "route_scope": "direct_only",
        "route_c": "ELIGIBLE_FOR_MULTISEED_CONFIRMATION",
        "eligible_components": ["hnr", "tilt"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {"diagnostic_report.json": screen_sha256},
        "checkpoint_sha256": {predictor.name: predictor_sha256},
        "checkpoint_dir": str(tmp_path),
    }
    receipt_path = tmp_path / "completion_receipt.json"
    receipt_path.write_text(json.dumps(screen_receipt), encoding="utf-8")
    receipt_sha256 = sha256_file(receipt_path)
    consensus = {
        "schema_version": "avqi-component-multiseed-consensus-v2",
        "route_scope": "direct_only",
        "active_routes": ["direct_differentiable_estimator"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "promotion": {
            "decision": "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT",
            "routes": ["direct_differentiable_estimator"],
            "components": ["hnr", "tilt"],
        },
        "routes": {
            "direct_differentiable_estimator": {
                "decision": "RELIABLE",
                "consensus_components": ["hnr", "tilt"],
                "component_pass_counts": {"hnr": 3, "tilt": 3},
            }
        },
        "source_report_sha256": {"screen": screen_sha256},
        "screen_report": str(screen_path),
    }
    consensus_path = tmp_path / "multiseed_consensus.json"
    consensus_path.write_text(json.dumps(consensus), encoding="utf-8")
    consensus_sha256 = sha256_file(consensus_path)

    gradient_norms, weights, authorization = validate(
        consensus_path,
        consensus_sha256,
        screen_path,
        screen_sha256,
        receipt_path,
        receipt_sha256,
        predictor,
        predictor_sha256,
    )

    assert gradient_norms == {"hnr": 2.0, "tilt": 100.0}
    assert weights == {"hnr": 1.0, "tilt": 0.02}
    assert authorization["decision"] == "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT"

    consensus["promotion"]["decision"] = "NO_GO_AVQI_BACKPROP"
    consensus_path.write_text(json.dumps(consensus), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        validate(
            consensus_path,
            sha256_file(consensus_path),
            screen_path,
            screen_sha256,
            receipt_path,
            receipt_sha256,
            predictor,
            predictor_sha256,
        )
