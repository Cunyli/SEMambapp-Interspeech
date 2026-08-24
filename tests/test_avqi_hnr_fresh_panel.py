from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_hnr_namespace() -> dict[str, object]:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_fresh_panel.py"
    )
    namespace["configure_pilot"](namespace["HNR_PILOT_PROFILE"])
    return namespace


def test_hnr_panel_is_balanced_and_final_speakers_are_fresh() -> None:
    namespace = load_hnr_namespace()
    specs = namespace["panel_specs"]()
    report = namespace["validate_panel_specs"](specs)

    assert report["case_count"] == 12
    assert report["previous_final_panel_overlap"] == []
    assert report["recipe_indices"] == list(range(940, 952))
    calibration = {spec.speaker_id for spec in specs if spec.split == "calibration"}
    final = {spec.speaker_id for spec in specs if spec.split == "final"}
    assert calibration == {"FD11", "FD26", "SD36"}
    assert final == {"PD_51", "SD13", "ÄHH28"}
    assert calibration.isdisjoint(final)

    for split in ("calibration", "final"):
        selected = [spec for spec in specs if spec.split == split]
        assert Counter(spec.view for spec in selected) == {"cs": 3, "sv": 3}
        assert Counter(spec.condition for spec in selected) == {
            "rir_only": 2,
            "snr20": 2,
            "snr10": 2,
        }


def synthetic_hnr_rows(namespace: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index in range(6):
        row: dict[str, object] = {
            "view": "cs" if index % 2 == 0 else "sv",
            "sample_group": (
                "pathological_mild" if index < 2 else "pathological_severe"
            ),
            "condition": ("rir_only", "snr20", "snr10")[index % 3],
            "material_hnr_gap": True,
            "proxy_absolute_gap_before_hnr": 1.0,
            "proxy_absolute_gap_after_hnr": 0.8,
            "proxy_normalized_gap_reduction_hnr": 0.03,
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
            row[f"exact_absolute_gap_before_{component}"] = 1.0
            row[f"exact_absolute_gap_after_{component}"] = 0.8
            row[f"exact_normalized_gap_reduction_{component}"] = (
                0.03 if component == "hnr" else 0.0
            )
        rows.append(row)
    return rows


def test_hnr_summary_requires_exact_proxy_and_guardrail_agreement() -> None:
    namespace = load_hnr_namespace()
    summary = namespace["finalize_summary"](
        namespace["summarize_rows"](
            synthetic_hnr_rows(namespace),
            expected_rows=6,
        )
    )

    assert summary["decision"] == "PASS"
    assert summary["exact_hnr"]["improvement_fraction_material"] == 1.0
    assert summary["companion_component"] is None
    assert summary["required_slice_gate"]["decision"] == "PASS"


def test_hnr_summary_rejects_nonselected_component_regression() -> None:
    namespace = load_hnr_namespace()
    rows = synthetic_hnr_rows(namespace)
    for row in rows:
        row["exact_normalized_gap_reduction_cpps"] = -0.06

    summary = namespace["summarize_rows"](rows, expected_rows=6)

    assert summary["decision"] == "FAIL"
    assert not summary["gates"][
        "all_nonselected_component_medians_within_0_05"
    ]


def test_hnr_launcher_is_authorization_bound_and_generator_frozen() -> None:
    source = (
        REPO_ROOT / "scripts" / "run_avqi_hnr_v7_fresh_panel.sh"
    ).read_text(encoding="utf-8")

    assert "direct_praat_hard_hnr_pitch_path_v7" in source
    assert "component_pass_counts.hnr" in source
    assert "--pilot-profile hnr_pitch_path_v7" in source
    assert "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
    assert "AVQI_CODE_TREE_SHA256" in source
    assert "generator_optimizer_steps" not in "\n".join(
        line for line in source.splitlines() if line.lstrip().startswith("python ")
    )
