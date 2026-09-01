from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.adjudicate_avqi_shimmer_db_candidate_e_v14_v28 import (
    PASS_DECISION,
    TRAINING_DECISION,
    sealed_selection,
)
from scripts.adjudicate_avqi_shimmer_db_deterministic_opened24_v23 import (
    sha256_file,
)


def synthetic_selection_payload(candidate_path: Path) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    case_id = "synthetic_case"
    candidate_sha256 = sha256_file(candidate_path)
    panel_rows = [{"case_id": case_id}]
    mechanism_report = {
        "case_diagnostics": [
            {
                "case_id": case_id,
                "candidate_e_raw_gradient_l2": 2.5,
                "candidate_e_projection": {
                    "projected_gradient_l2_norm": 1.25,
                },
            }
        ]
    }
    selector = {
        "rows": [
            {
                "case_id": case_id,
                "case_id_used_for_routing": False,
                "selected": {
                    "direction_family": (
                        "candidate_e_exact_path_raw_ablation"
                    ),
                    "alpha": 0.0005,
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": candidate_sha256,
                    "current_topology_sha256": "topology-sha256",
                    "exact_candidate_outcome_present": False,
                },
            }
        ]
    }
    candidate_grid = [
        {
            "case_id": case_id,
            "variant": "candidate_e_exact_path_raw_ablation",
            "alpha": "0.0",
            "candidate_sha256": "zero-step-not-selected",
            "current_topology_proxy_shimmer_db": "1.3",
        },
        {
            "case_id": case_id,
            "variant": "candidate_e_exact_path_raw_ablation",
            "alpha": "0.0005",
            "candidate_sha256": candidate_sha256,
            "current_topology_proxy_shimmer_db": "1.2",
        },
    ]
    selector_exact_rows = [
        {
            "case_id": case_id,
            "selected_candidate_present": "True",
            "selected_candidate_sha256": candidate_sha256,
            "selected_direction_family": (
                "candidate_e_exact_path_raw_ablation"
            ),
            "selected_alpha": "0.0005",
            "exact_improves": "True",
        }
    ]
    return (
        panel_rows,
        mechanism_report,
        selector,
        candidate_grid,
        selector_exact_rows,
    )


def test_sealed_selection_binds_hash_family_alpha_and_gradient(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.wav"
    candidate_path.write_bytes(b"sealed-candidate-pcm24-placeholder")
    payload = synthetic_selection_payload(candidate_path)
    selected = sealed_selection(*payload)
    row = selected["synthetic_case"]
    assert row["direction_family"] == "candidate_e_exact_path_raw_ablation"
    assert row["alpha"] == 0.0005
    assert row["candidate_sha256"] == sha256_file(candidate_path)
    assert row["gradient_l2_norm"] == 2.5
    assert row["gradient_finite"] is True


def test_sealed_selection_rejects_identity_routing(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.wav"
    candidate_path.write_bytes(b"sealed-candidate-pcm24-placeholder")
    payload = list(synthetic_selection_payload(candidate_path))
    selector = copy.deepcopy(payload[2])
    selector["rows"][0]["case_id_used_for_routing"] = True
    payload[2] = selector
    with pytest.raises(ValueError, match="identity routing drift"):
        sealed_selection(*payload)


def test_v28_authorization_boundary_is_component_only() -> None:
    assert PASS_DECISION == "PASS_CANDIDATE_E_V14_FULL_GATE_FROZEN_V28"
    assert TRAINING_DECISION == "NO_GO_AVQI_T2_TRAINING"
