from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_namespace() -> dict[str, object]:
    return runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_fresh_panel.py"
    )


def test_fresh_panel_is_balanced_and_speaker_disjoint() -> None:
    namespace = load_namespace()
    specs = namespace["panel_specs"]()
    report = namespace["validate_panel_specs"](specs)

    assert report["case_count"] == 12
    assert report["previous_waveform_pilot_overlap"] == []
    calibration = {spec.speaker_id for spec in specs if spec.split == "calibration"}
    final = {spec.speaker_id for spec in specs if spec.split == "final"}
    previous = namespace["PREVIOUS_WAVEFORM_PILOT_SPEAKERS"]
    assert len(calibration) == len(final) == 3
    assert calibration.isdisjoint(final)
    assert (calibration | final).isdisjoint(previous)

    for split in ("calibration", "final"):
        selected = [spec for spec in specs if spec.split == split]
        assert Counter(spec.view for spec in selected) == {"cs": 3, "sv": 3}
        assert Counter(spec.condition for spec in selected) == {
            "rir_only": 2,
            "snr20": 2,
            "snr10": 2,
        }


def test_panel_validation_rejects_a_previous_pilot_speaker() -> None:
    namespace = load_namespace()
    specs = list(namespace["panel_specs"]())
    panel_spec = namespace["PanelSpec"]
    for index, original in enumerate(specs):
        if original.speaker_id != "FD26":
            continue
        specs[index] = panel_spec(
            original.split,
            "PD08",
            original.sample_group,
            original.view,
            original.condition,
            original.recipe_index,
        )

    with pytest.raises(ValueError, match="previous waveform pilot"):
        namespace["validate_panel_specs"](tuple(specs))


def test_alpha_selection_uses_only_passing_nonzero_candidates() -> None:
    namespace = load_namespace()
    summaries = {
        0.0: {
            "decision": "PASS",
            "exact_shimmer_percent": {
                "median_normalized_gap_reduction_material": 1.0
            },
        },
        1e-4: {
            "decision": "PASS",
            "exact_shimmer_percent": {
                "median_normalized_gap_reduction_material": 0.03
            },
        },
        3e-4: {
            "decision": "PASS",
            "exact_shimmer_percent": {
                "median_normalized_gap_reduction_material": 0.06
            },
        },
        1e-3: {
            "decision": "FAIL",
            "exact_shimmer_percent": {
                "median_normalized_gap_reduction_material": 0.50
            },
        },
    }

    assert namespace["choose_calibration_alpha"](summaries) == 3e-4


def test_alpha_selection_breaks_an_exact_tie_toward_smaller_step() -> None:
    namespace = load_namespace()
    summaries = {
        alpha: {
            "decision": "PASS",
            "exact_shimmer_percent": {
                "median_normalized_gap_reduction_material": 0.04
            },
        }
        for alpha in (1e-4, 3e-4)
    }

    assert namespace["choose_calibration_alpha"](summaries) == 1e-4


def test_normalized_gradient_step_obeys_frozen_rms_budget() -> None:
    namespace = load_namespace()
    base = torch.linspace(-0.2, 0.2, 16_000)
    gradient = torch.linspace(-1.0, 1.0, 16_000)
    alpha = 3e-4

    candidate = namespace["candidate_from_gradient"](base, gradient, alpha)

    assert candidate is not None
    residual = candidate - base
    ratio = residual.square().mean().sqrt() / base.square().mean().sqrt()
    assert float(ratio) == pytest.approx(alpha, rel=1e-4)
    assert float(torch.dot(residual, gradient)) < 0.0


def test_alpha_selection_fails_closed_when_no_candidate_passes() -> None:
    namespace = load_namespace()
    summaries = {
        0.0: {"decision": "PASS"},
        1e-4: {"decision": "FAIL"},
        3e-4: {"decision": "FAIL"},
    }

    assert namespace["choose_calibration_alpha"](summaries) is None


def synthetic_result_rows(namespace: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index in range(6):
        row: dict[str, object] = {
            "view": "cs" if index % 2 == 0 else "sv",
            "sample_group": (
                "pathological_mild" if index < 2 else "pathological_severe"
            ),
            "condition": ("rir_only", "snr20", "snr10")[index % 3],
            "material_shimmer_percent_gap": True,
            "proxy_absolute_gap_before_shimmer_percent": 1.0,
            "proxy_absolute_gap_after_shimmer_percent": 0.8,
            "proxy_normalized_gap_reduction_shimmer_percent": 0.03,
            "residual_rms_db": -50.5,
            "cosine_similarity": 0.999999,
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
        for component in namespace["AVQI_COMPONENT_NAMES"]:
            reduction = 0.03 if component == "shimmer_percent" else 0.0
            if component == "shimmer_db":
                reduction = 0.01
            row[f"exact_absolute_gap_before_{component}"] = 1.0
            row[f"exact_absolute_gap_after_{component}"] = 0.8
            row[f"exact_normalized_gap_reduction_{component}"] = reduction
        rows.append(row)
    return rows


def test_fresh_panel_summary_requires_exact_proxy_and_guardrail_agreement() -> None:
    namespace = load_namespace()
    summary = namespace["finalize_summary"](
        namespace["summarize_rows"](
            synthetic_result_rows(namespace),
            expected_rows=6,
        )
    )

    assert summary["decision"] == "PASS"
    assert all(summary["gates"].values())
    assert summary["required_slice_gate"]["decision"] == "PASS"


def test_fresh_panel_summary_rejects_nonselected_component_regression() -> None:
    namespace = load_namespace()
    rows = synthetic_result_rows(namespace)
    for row in rows:
        row["exact_normalized_gap_reduction_hnr"] = -0.06

    summary = namespace["summarize_rows"](rows, expected_rows=6)

    assert summary["decision"] == "FAIL"
    assert not summary["gates"][
        "all_nonselected_component_medians_within_0_05"
    ]


def test_launcher_keeps_generator_frozen_and_binds_exact_sources() -> None:
    source = (
        REPO_ROOT / "scripts" / "run_avqi_shimmer_fresh_panel.sh"
    ).read_text(encoding="utf-8")

    assert "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" in source
    assert "CONSENSUS_RECEIPT_SHA256" in source
    assert "AVQI_CODE_TREE_SHA256" in source
    assert "FIXED_RECIPES_SHA256" in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
    assert "evaluate_avqi_shimmer_fresh_panel.py" in source
    assert "train" not in "\n".join(
        line for line in source.splitlines() if line.lstrip().startswith("python ")
    )
