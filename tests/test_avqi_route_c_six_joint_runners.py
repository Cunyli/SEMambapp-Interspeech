from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from scripts import evaluate_avqi_route_c_six_joint_exact_panel as exact_runner
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    GLOBAL_ALPHA_GRID,
    REQUIRED_CONDITIONS,
    REQUIRED_VIEWS,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    EFFICACY_METRICS,
    JOINT_METRIC_NAME,
    choose_alpha,
    exact_gap_record,
    metric_summary,
    slice_metric_summary,
    summarize_exact_efficacy,
)
from scripts.prepare_avqi_route_c_six_joint_waveforms import (
    candidate_from_gradient,
)


def _metric_values(
    *,
    before: float = 1.0,
    after: float = 0.5,
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "target": 0.0,
            "before": before,
            "after": after,
            "normalized_gap_before": before,
            "normalized_gap_after": after,
            "normalized_gap_reduction": before - after,
        }
        for metric in EFFICACY_METRICS
    }


def _efficacy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in REQUIRED_CONDITIONS:
        if condition == "clean":
            continue
        for view in REQUIRED_VIEWS:
            for speaker_index in range(3):
                rows.append(
                    {
                        "case_id": f"{condition}-{view}-{speaker_index}",
                        "condition": condition,
                        "view": view,
                        "metrics": _metric_values(),
                    }
                )
    return rows


def test_waveform_rms_normalized_step_matches_frozen_formula() -> None:
    base = np.asarray([0.2, -0.1, 0.05, -0.15], dtype=np.float32)
    gradient = np.asarray([2.0, -1.0, 4.0, -3.0], dtype=np.float32)

    unchanged, reason = candidate_from_gradient(base, gradient, 0.0)
    assert reason is None
    np.testing.assert_array_equal(unchanged, base)

    alpha = 1e-3
    candidate, reason = candidate_from_gradient(base, gradient, alpha)
    assert reason is None
    residual_rms = np.sqrt(np.mean(np.square(candidate - base)))
    base_rms = np.sqrt(np.mean(np.square(base)))
    assert residual_rms == pytest.approx(alpha * base_rms, rel=2e-5)


def test_waveform_step_rejects_zero_gradient_and_peak_overflow() -> None:
    base = np.asarray([0.2, -0.2], dtype=np.float32)
    zero = np.zeros_like(base)
    candidate, reason = candidate_from_gradient(base, zero, 1e-3)
    assert candidate is None
    assert reason == "gradient_rms_invalid"

    candidate, reason = candidate_from_gradient(
        np.asarray([0.9988, -0.9988], dtype=np.float32),
        np.asarray([-1.0, 1.0], dtype=np.float32),
        3e-3,
    )
    assert candidate is None
    assert reason == "candidate_peak_outside_pcm24_contract"


def test_exact_equal_weight_joint_ignores_avqi_scalar_coefficients() -> None:
    target = np.zeros(6, dtype=np.float64)
    before = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    after = before / 2.0
    scales = {
        component: float(index + 1)
        for index, component in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    }

    record = exact_gap_record(target, before, after, scales)

    assert record[JOINT_METRIC_NAME]["normalized_gap_before"] == 1.0
    assert record[JOINT_METRIC_NAME]["normalized_gap_after"] == 0.5
    assert record[JOINT_METRIC_NAME]["normalized_gap_reduction"] == 0.5


def test_all_six_joint_and_each_condition_view_slice_must_pass() -> None:
    summary = summarize_exact_efficacy(_efficacy_rows())

    assert summary["rows"] == 18
    assert summary["decision"] == "PASS"
    assert all(summary["gates"].values())
    assert len(summary["required_slices"]) == 6
    assert all(
        item["decision"] == "PASS"
        for item in summary["required_slices"].values()
    )


def test_global_material_coverage_requires_fifteen_of_eighteen() -> None:
    rows = _efficacy_rows()
    for row in rows[:4]:
        row["metrics"]["cpps"]["normalized_gap_before"] = 0.01

    summary = metric_summary(rows, "cpps", expected_rows=18)

    assert summary["material_rows"] == 14
    assert summary["decision"] == "FAIL"
    assert summary["gates"]["material_cases_scaled_from_15_per_18"] is False


def test_slice_improvement_threshold_is_strictly_greater_than_half() -> None:
    values = [
        {
            "metrics": {
                "cpps": _metric_values(before=1.0, after=0.5)["cpps"]
            }
        },
        {
            "metrics": {
                "cpps": _metric_values(before=1.0, after=1.0)["cpps"]
            }
        },
    ]

    summary = slice_metric_summary(values, "cpps", expected_rows=2)

    assert summary["improvement_fraction_material"] == 0.5
    assert summary["gates"]["improvement_fraction_gt_0_50"] is False
    assert summary["decision"] == "FAIL"


def test_calibration_selects_only_passing_nonzero_alpha_and_ties_smaller() -> None:
    summaries = {}
    for alpha in GLOBAL_ALPHA_GRID:
        summaries[alpha] = {
            "decision": "FAIL",
            "efficacy": {
                "metrics": {
                    JOINT_METRIC_NAME: {
                        "median_normalized_gap_reduction_material": 0.0
                    }
                }
            },
        }
    for alpha in (1e-4, 3e-4):
        summaries[alpha] = {
            "decision": "PASS",
            "efficacy": {
                "metrics": {
                    JOINT_METRIC_NAME: {
                        "median_normalized_gap_reduction_material": 0.1
                    }
                }
            },
        }

    assert choose_alpha(summaries) == 1e-4


def test_runners_keep_exact_metric_branch_separate_from_emitted_audio() -> None:
    prepare_source = Path(
        "scripts/prepare_avqi_route_c_six_joint_waveforms.py"
    ).read_text(encoding="utf-8")
    exact_source = Path(
        "scripts/evaluate_avqi_route_c_six_joint_exact_panel.py"
    ).read_text(encoding="utf-8")

    assert '"emitted_waveform_highpass": False' in prepare_source
    assert "run_avqi(" in exact_source
    assert '"formal_generator_training_authorized": False' in exact_source
    assert "NO_GO_AVQI_T2_TRAINING" not in exact_source


def test_final_completion_receipt_serializes_success_without_training_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exact_runner, "run_exact", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        exact_runner,
        "stage_rows",
        lambda *args, **kwargs: {"decision": "PASS", "gates": {"frozen": True}},
    )
    output_dir = tmp_path / "final"
    output_dir.mkdir()
    source = {
        "root": str(tmp_path),
        "head": "a" * 40,
        "branch": "feat/avqi-route-c-six-joint-runners-v1",
    }

    result = exact_runner.run_final(
        rows=[],
        target_scale={name: 1.0 for name in ROUTE_C_SIX_ACTIVE_COMPONENTS},
        selected_alpha=1e-4,
        waveform_seal_sha256="b" * 64,
        waveform_seal_receipt_sha256="c" * 64,
        calibration_report_sha256="d" * 64,
        alpha_receipt_sha256="e" * 64,
        exact_python=tmp_path / "python",
        avqi_code_root=tmp_path / "avqi",
        exact_authority={"praat_version": "test"},
        exact_code_tree_manifest_sha256="f" * 64,
        exact_runtime_manifest_sha256="1" * 64,
        source=source,
        output_dir=output_dir,
    )

    stored = json.loads(
        (output_dir / "completion_receipt.json").read_text(encoding="utf-8")
    )
    assert stored == result["receipt"]
    assert stored["decision"] == exact_runner.FINAL_PASS_DECISION
    assert stored["joint_scientific_promotion_granted"] is True
    assert stored["one_batch_generator_gradient_check_authorized"] is True
    assert stored["formal_generator_training_authorized"] is False
    assert stored["generator_optimizer_steps"] == 0
