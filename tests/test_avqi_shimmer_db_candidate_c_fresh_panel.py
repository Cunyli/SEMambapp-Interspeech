from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_namespace() -> dict[str, object]:
    return runpy.run_path(
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_candidate_c_fresh_panel.py"
    )


def test_panel_is_new_balanced_and_recipe_locked() -> None:
    namespace = load_namespace()
    specs = namespace["panel_specs"]()
    report = namespace["validate_panel_specs"](specs)

    assert report["case_count"] == 12
    assert report["speaker_count"] == 6
    assert report["previous_waveform_speaker_overlap"] == []
    assert report["recipe_indices"] == list(range(912, 924))
    assert Counter(spec.view for spec in specs) == {"cs": 6, "sv": 6}
    assert Counter(spec.condition for spec in specs) == {
        "rir_only": 4,
        "snr20": 4,
        "snr10": 4,
    }
    assert Counter(spec.sample_group for spec in specs) == {
        "pathological_mild": 6,
        "pathological_severe": 6,
    }
    assert {spec.speaker_id for spec in specs} == {
        "SD05",
        "SD32",
        "ÄHH13",
        "ÄHH16",
        "SD17",
        "SD20",
    }


def test_panel_rejects_previously_opened_waveform_speaker() -> None:
    namespace = load_namespace()
    specs = list(namespace["panel_specs"]())
    panel_spec = namespace["PanelSpec"]
    original_speaker = specs[0].speaker_id
    for index, original in enumerate(specs):
        if original.speaker_id != original_speaker:
            continue
        specs[index] = panel_spec(
            "FD26",
            original.sample_group,
            original.view,
            original.condition,
            original.recipe_index,
        )
    with pytest.raises(ValueError, match="previous waveform"):
        namespace["validate_panel_specs"](tuple(specs))


def test_runtime_v15_panel_is_unseen_balanced_and_recipe_locked() -> None:
    namespace = load_namespace()
    specs = namespace["runtime_v15_panel_specs"]()
    report = namespace["validate_panel_specs"](
        specs,
        previous_speakers=namespace[
            "RUNTIME_V15_PREVIOUS_WAVEFORM_PILOT_SPEAKERS"
        ],
        expected_recipe_indices=range(924, 936),
    )

    assert report["case_count"] == 12
    assert report["speaker_count"] == 6
    assert report["previous_waveform_speaker_overlap"] == []
    assert report["recipe_indices"] == list(range(924, 936))
    assert {spec.speaker_id for spec in specs} == {
        "FD23",
        "SD25",
        "PD04",
        "FD09",
        "ÄHH32",
        "PD_37",
    }
    assert not (
        {spec.speaker_id for spec in specs}
        & namespace["RUNTIME_V15_PREVIOUS_WAVEFORM_PILOT_SPEAKERS"]
    )


def test_speaker_rank_binds_salt_group_and_id() -> None:
    namespace = load_namespace()
    rank = namespace["speaker_selection_rank"]
    salt = namespace["PANEL_SELECTION_SALT"]

    assert rank(salt, "pathological_mild", "SD05") == (
        "0141519975f8d9519c58182fbc12744c3ae4d8af97dee83cbe4f23e2e7f95ffe"
    )
    assert rank(salt, "pathological_mild", "SD05") != rank(
        salt,
        "pathological_severe",
        "SD05",
    )


def test_topology_item_builder_refreshes_each_base_once() -> None:
    namespace = load_namespace()
    panel_spec = namespace["PanelSpec"]
    prepared_case = namespace["PreparedCase"]
    cases = []
    for index in range(3):
        spec = panel_spec(
            f"S{index}",
            "pathological_mild",
            "cs" if index % 2 == 0 else "sv",
            "rir_only",
            912 + index,
        )
        cases.append(
            prepared_case(
                spec,
                Path(f"target-{index}.wav"),
                Path(f"noisy-{index}.wav"),
                Path(f"base-{index}.wav"),
                Path(f"source-{index}.wav"),
                {},
                index,
                0,
            )
        )

    items = namespace["build_base_topology_items"](cases)

    assert len(items) == 3
    assert len({item["case_id"] for item in items}) == 3
    assert all(item["score_components"] is False for item in items)
    assert all(item["exact_metric_topology"] is True for item in items)


def test_source_freezes_alpha_and_candidate_c_gates() -> None:
    namespace = load_namespace()

    assert namespace["FIXED_ALPHA"] == 0.001
    assert namespace["CANDIDATE_NAME"] == (
        "praat_current_output_topology_refresh_db_alpha_0p001"
    )
    assert namespace["CACHE_RUNTIME_MAX_MS"] == 500.0
    assert namespace["MATERIAL_GAP_THRESHOLD"] == 0.02
    assert namespace["MEDIAN_REDUCTION_GATE"] == 0.02
    assert namespace["IMPROVEMENT_FRACTION_GATE"] == 0.80
    assert namespace["REQUIRED_EFFECT_SLICES"] == (
        "view=cs",
        "view=sv",
        "severity=pathological_mild",
        "severity=pathological_severe",
        "condition=rir_only",
        "condition=snr20",
        "condition=snr10",
    )


def test_candidate_seal_precedes_exact_base_candidate_scoring() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    seal_write = source.index(
        'write_json(args.output_dir / "candidate_seal.json", candidate_seal)'
    )
    final_exact = source.index("final_exact = run_exact(", seal_write)
    assert seal_write < final_exact
    assert '"base_and_candidate_exact_outcomes_unopened_until_candidate_seal": True' in source
    assert '"clean_target_topology_drives_output": False' in source


def test_runner_binds_v13_and_never_trains_generator() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_candidate_c_fresh_panel.sh"
    ).read_text(encoding="utf-8")

    assert "547e1a3dd106f5a24e218440644ef1e88a9497e6fd3d4f873eb889b7e1c86bb6" in source
    assert "9caa69fa3cc967af6a8851c802cbf2c8d1baf52f8e50f131b81e65028b6c2d48" in source
    assert "PASS_CURRENT_OUTPUT_EXACT_TOPOLOGY_REFRESH_FREEZE_FOR_FRESH_PANEL" in source
    assert "CONFIRM_SLURM_SUBMIT=1" in source
    python_lines = [
        line for line in source.splitlines() if line.lstrip().startswith("python ")
    ]
    assert all("train" not in line.lower() for line in python_lines)


def test_runtime_v15_runner_binds_equivalence_receipts_and_gate() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_candidate_c_fresh_panel.sh"
    ).read_text(encoding="utf-8")

    assert "c2e7399eeb14a7e4f6d2c8b44402e4d4e8d0c460a24f122d903aa6c9d46b15d9" in source
    assert "ef56ff7066956967a8a22c977bbc92993689b295ee3bb9ec36d3de60ced3719a" in source
    assert "c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f" in source
    assert "PASS_SHIMMER_DB_RUNTIME_V15_EXACT_EQUIVALENCE_FREEZE_FOR_NEW_PANEL" in source
    assert ".runtime.formal_500ms_pass" in source
    assert '--panel-version "$PANEL_VERSION"' in source


def test_runtime_v15_refresh_uses_persistent_worker_and_end_to_end_gate() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    assert "with ExactShimmerTopologyWorker(" in source
    assert "worker.refresh_current_waveforms(" in source
    assert "highpass_mode=NUMPY_HIGHPASS_MODE" in source
    assert 'topology["end_to_end_refresh_ms"]' in source
    assert '"waveform_dependent_topology_cache": False' in source
    assert '"current_output_refresh_per_waveform_step": True' in source


def test_summarizer_extends_frozen_mechanism_with_fresh_guardrails() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_candidate_c_fresh_panel.py"
    ).read_text(encoding="utf-8")

    assert "mechanism = aggregate_candidate(CANDIDATE_NAME, rows)" in source
    assert "aggregate_pathology_guardrails(rows)" in source
    assert "aggregate_denoising(rows)" in source
    assert '"frozen_mechanism_gates": mechanism["all_gates_pass"]' in source
