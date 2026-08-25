from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.evaluate_avqi_shimmer_db_topology_family_selector_opened24 as opened24
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_opened24 import (
    EXPECTED_CASE_COUNT,
    FROZEN_SELECTOR_SOURCE_COMMIT,
    FROZEN_SELECTOR_SOURCE_SHA256,
    FROZEN_V16_SOURCE_SHA256,
    FROZEN_V17_SOURCE_SHA256,
    OPENED24_EXACT_NO_GO,
    opened24_gate_decision,
    summarize_scope,
    validate_combined_scope,
    validate_target_contract,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    ALPHA_LADDER,
    FIXED_ALPHA,
    selector_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_panel_rows(
    panel_label: str,
    recipe_start: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        speaker_index = index // 2
        rows.append(
            {
                "case_id": f"{panel_label}_case_{index:02d}",
                "opened_panel": panel_label,
                "speaker_id": f"{panel_label}_speaker_{speaker_index:02d}",
                "view": "cs" if index % 2 == 0 else "sv",
                "sample_group": (
                    "pathological_mild"
                    if index < 6
                    else "pathological_severe"
                ),
                "condition": ("rir_only", "snr10", "snr20")[index % 3],
                "recipe_index": recipe_start + index,
            }
        )
    return rows


def test_opened24_combined_scope_is_disjoint_and_balanced() -> None:
    v14_rows = synthetic_panel_rows("v14", 912)
    v15_rows = synthetic_panel_rows("v15", 924)
    combined = validate_combined_scope(v14_rows, v15_rows)
    assert len(combined) == EXPECTED_CASE_COUNT == 24
    assert len({row["speaker_id"] for row in combined}) == 12


def test_opened24_combined_scope_rejects_speaker_overlap() -> None:
    v14_rows = synthetic_panel_rows("v14", 912)
    v15_rows = synthetic_panel_rows("v15", 924)
    for row in v15_rows[:2]:
        row["speaker_id"] = v14_rows[0]["speaker_id"]
    with pytest.raises(ValueError, match="speaker overlap"):
        validate_combined_scope(v14_rows, v15_rows)


def test_target_contract_is_same_speaker_scalar_only() -> None:
    panel_rows = synthetic_panel_rows("v14", 912)
    for index, row in enumerate(panel_rows):
        row["target_sha256"] = f"target-{index}"
    input_by_case = {
        str(row["case_id"]): {"exact_target_shimmer_db": float(index)}
        for index, row in enumerate(panel_rows)
    }
    target_contract = {
        "schema_version": "avqi-route-c-shimmer-db-supervised-target-v1",
        "role": "same_speaker_target_scalar_required_by_candidate_loss",
        "selection_or_tuning_use": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "rows": [
            {
                "case_id": row["case_id"],
                "speaker_id": row["speaker_id"],
                "view": row["view"],
                "target_sha256": row["target_sha256"],
                "exact_target_shimmer_db": float(index),
            }
            for index, row in enumerate(panel_rows)
        ],
    }
    validate_target_contract(
        "v14",
        panel_rows,
        input_by_case,
        target_contract,
    )
    bad = deepcopy(target_contract)
    bad["clean_target_pulse_positions_exposed_to_output_branch"] = True
    with pytest.raises(ValueError, match="topology exposure"):
        validate_target_contract("v14", panel_rows, input_by_case, bad)


def test_scope_summary_rebinds_only_coverage_expectation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = {
        "mechanism_gates": {
            "complete_prototype_coverage": False,
            "exact_db_effect": True,
            "gradient": True,
        },
        "integration_gates": {
            "mechanism": False,
            "selector_coverage": True,
        },
        "all_gates_pass": False,
    }
    monkeypatch.setattr(opened24, "summarize_exact_rows", lambda rows: deepcopy(frozen))
    summary = summarize_scope([{} for _ in range(24)], 24, "complete_opened24_coverage")
    assert "complete_prototype_coverage" not in summary["mechanism_gates"]
    assert summary["mechanism_gates"]["complete_opened24_coverage"] is True
    assert summary["mechanism_gates"]["exact_db_effect"] is True
    assert summary["integration_gates"]["mechanism"] is True
    assert summary["all_gates_pass"] is True
    assert summary["scope_coverage"][
        "frozen_core_four_case_coverage_value_before_scope_rebind"
    ] is False


def test_panel_gate_prevents_combined_summary_from_masking_v15_failure() -> None:
    combined = {"all_gates_pass": True}
    panel_summaries = {
        "v14": {"all_gates_pass": True},
        "v15": {"all_gates_pass": False},
    }
    gates, decision, authorized = opened24_gate_decision(
        combined,
        panel_summaries,
    )
    assert gates == {
        "combined_24case_pass": True,
        "v14_panel_pass": True,
        "v15_panel_pass": False,
    }
    assert decision == OPENED24_EXACT_NO_GO
    assert authorized is False


def test_frozen_selector_and_family_sources_are_byte_unchanged() -> None:
    assert FROZEN_SELECTOR_SOURCE_COMMIT == (
        "c5c6e7612d6e7b641550b5706c4c3fe3a1a9927a"
    )
    assert sha256_file(
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_topology_family_selector_v18.py"
    ) == FROZEN_SELECTOR_SOURCE_SHA256
    assert sha256_file(
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_db_trust_region_v16.py"
    ) == FROZEN_V16_SOURCE_SHA256
    assert sha256_file(
        REPO_ROOT / "scripts" / "evaluate_avqi_shimmer_db_source_informed_v17.py"
    ) == FROZEN_V17_SOURCE_SHA256
    assert FIXED_ALPHA == 0.001
    assert ALPHA_LADDER == (0.001, 0.0005, 0.00025, 0.000125)
    assert selector_contract()["formal_total_metric_step_runtime_ms"] == 500.0


def test_opened24_seal_precedes_exact_and_no_new_routing_keys() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluate_avqi_shimmer_db_topology_family_selector_opened24.py"
    ).read_text(encoding="utf-8")
    run_start = source.index("def run_opened24(")
    run_source = source[run_start:]
    seal = "write_json(selector_seal_path, selector_seal)"
    exact = "rows, exact_versions = build_exact_rows("
    assert run_source.index(seal) < run_source.index(exact)
    assert "evaluate_selector_case(" in run_source
    assert '"candidate_exact_outcomes_present": False' in run_source
    assert '"selection_uses_candidate_exact_outcome": False' in run_source
    assert '"generator_optimizer_steps": 0' in run_source
    assert '"panel_gate_summaries": panel_gate_summaries' in run_source
    assert '"opened24_authorization_gates": opened24_gates' in run_source
    assert "FD23" not in source
    assert "PD_37" not in source


def test_opened24_runner_is_hash_bound_and_fail_closed() -> None:
    source = (
        REPO_ROOT
        / "scripts"
        / "run_avqi_shimmer_db_topology_family_selector_opened24.sh"
    ).read_text(encoding="utf-8")
    for token in (
        "CONFIRM_SLURM_SUBMIT",
        "V14_PANEL_CONTRACT_SHA256",
        "V14_TARGET_CONTRACT_SHA256",
        "V15_PANEL_CONTRACT_SHA256",
        "V15_TARGET_CONTRACT_SHA256",
        "SELECTOR4_REPORT_SHA256",
        "SELECTOR4_PRESELECTION_SHA256",
        "SELECTOR4_SEAL_SHA256",
        "SELECTOR4_RESULTS_SHA256",
        "SELECTOR4_RECEIPT_SHA256",
        "SELECTOR_CORE_SCRIPT_SHA256",
        "V16_FAMILY_SOURCE_SHA256",
        "V17_FAMILY_SOURCE_SHA256",
        "RUNTIME_WORKER_SCRIPT_SHA256",
        "Refusing to overwrite opened24 output",
    ):
        assert token in source
    assert "--job-name=\"avqi-shim-v18-opened24\"" in source
    assert "NO_GO_AVQI_T2_TRAINING" not in source
