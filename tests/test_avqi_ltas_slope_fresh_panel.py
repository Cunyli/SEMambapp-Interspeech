from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_ltas_namespace() -> dict[str, object]:
    namespace = runpy.run_path(
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_fresh_panel.py"
    )
    namespace["configure_pilot"](namespace["LTAS_SLOPE_PILOT_PROFILE"])
    return namespace


def test_ltas_panel_uses_only_unopened_svd_reserves() -> None:
    namespace = load_ltas_namespace()
    specs = namespace["panel_specs"]()
    report = namespace["validate_panel_specs"](specs)

    calibration = {spec.speaker_id for spec in specs if spec.split == "calibration"}
    final = {spec.speaker_id for spec in specs if spec.split == "final"}
    assert calibration == {"SVD:1438", "SVD:1516", "SVD:1849"}
    assert final == {"SVD:1322", "SVD:1872", "SVD:1923"}
    assert calibration.isdisjoint(final)
    assert report["panel_source"] == "svd_unused_reserve"
    assert report["unused_authority_reserves_after_selection"] == ["SVD:1301"]
    assert report["recipe_indices"] == list(range(960, 972))

    for split in ("calibration", "final"):
        selected = [spec for spec in specs if spec.split == split]
        assert Counter(spec.view for spec in selected) == {"cs": 3, "sv": 3}
        assert Counter(spec.condition for spec in selected) == {
            "rir_only": 2,
            "snr20": 2,
            "snr10": 2,
        }
        assert {spec.sample_group for spec in selected} == {
            "pathological_external"
        }


def test_ltas_svd_reader_hash_locks_paired_full_band_waveforms(
    tmp_path: Path,
) -> None:
    namespace = load_ltas_namespace()
    speakers = sorted({spec.speaker_id for spec in namespace["panel_specs"]()})
    rows = []
    for speaker_index, speaker_id in enumerate(speakers):
        for view in ("cs", "sv"):
            path = tmp_path / f"{speaker_id.replace(':', '_')}_{view}.wav"
            waveform = np.linspace(-0.1, 0.1, 16_000, dtype=np.float32)
            sf.write(path, waveform, 16_000, subtype="FLOAT")
            waveform_hash = namespace["sha256_file"](path)
            rows.append(
                {
                    "panel_speaker_id": speaker_id,
                    "session_id": str(1000 + speaker_index),
                    "view": view,
                    "selection_role": "reserve",
                    "label": "patient",
                    "variant_paths": {"clean": str(path)},
                    "variant_sha256": {"clean": waveform_hash},
                }
            )
    seal_path = tmp_path / "panel_seal.json"
    namespace["write_json"](seal_path, {"rows": rows})
    seal_hash = namespace["sha256_file"](seal_path)

    selected = namespace["read_svd_panel_seal"](
        seal_path,
        seal_hash,
        sorted(namespace["LTAS_SLOPE_UNUSED_RESERVE_SPEAKERS"]),
    )

    assert set(selected) == set(speakers)
    assert all(row["dataset"] == "SVD" for row in selected.values())
    assert all(row["label"] == "patient" for row in selected.values())


def synthetic_ltas_rows(namespace: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index in range(6):
        row: dict[str, object] = {
            "view": "cs" if index % 2 == 0 else "sv",
            "sample_group": "pathological_external",
            "condition": ("rir_only", "snr20", "snr10")[index % 3],
            "material_slope_gap": True,
            "proxy_absolute_gap_before_slope": 1.0,
            "proxy_absolute_gap_after_slope": 0.8,
            "proxy_normalized_gap_reduction_slope": 0.03,
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
                0.03 if component == "slope" else 0.0
            )
        rows.append(row)
    return rows


def test_ltas_summary_requires_views_conditions_and_all_safety_gates() -> None:
    namespace = load_ltas_namespace()
    summary = namespace["finalize_summary"](
        namespace["summarize_rows"](
            synthetic_ltas_rows(namespace),
            expected_rows=6,
        )
    )

    assert summary["decision"] == "PASS"
    assert summary["exact_slope"]["improvement_fraction_material"] == 1.0
    assert summary["companion_component"] is None
    assert summary["required_slice_gate"]["decision"] == "PASS"
    assert set(summary["required_slice_gate"]["slices"]) == {
        "view=cs",
        "view=sv",
        "condition=rir_only",
        "condition=snr20",
        "condition=snr10",
    }


def test_ltas_summary_rejects_nonselected_component_regression() -> None:
    namespace = load_ltas_namespace()
    rows = synthetic_ltas_rows(namespace)
    for row in rows:
        row["exact_normalized_gap_reduction_cpps"] = -0.06

    summary = namespace["summarize_rows"](rows, expected_rows=6)

    assert summary["decision"] == "FAIL"
    assert not summary["gates"][
        "all_nonselected_component_medians_within_0_05"
    ]


def test_ltas_launcher_binds_promotion_and_uses_absolute_python() -> None:
    source = (
        REPO_ROOT / "scripts" / "run_avqi_ltas_slope_fresh_panel.sh"
    ).read_text(encoding="utf-8")

    assert "GO_BOUNDED_LTAS_SLOPE_WAVEFORM_PILOT" not in source
    assert "PROMOTION_REPORT_SHA256" in source
    assert "PROMOTION_RECEIPT_SHA256" in source
    assert "SVD_PANEL_SEAL_SHA256" in source
    assert "--pilot-profile ltas_slope_authority_v1" in source
    assert '"$RUNTIME_PYTHON" "$PILOT_SCRIPT"' in source
    assert "python -c" not in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
