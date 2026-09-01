from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts.evaluate_avqi_shimmer_db_candidate_e_opened_v15_v29 import (
    PASS_DECISION,
    TRAINING_DECISION,
    VARIANT_E_RAW,
    parse_args,
    selected_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "avqi_route_c_shimmer_db_candidate_e_opened_v15_confirmation_v29.json"
)


def test_v29_config_binds_freeze_and_forbids_retuning() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["candidate_e_freeze"]["decision"] == (
        "PASS_CANDIDATE_E_V14_FULL_GATE_FROZEN_V28"
    )
    assert config["dataset_contract"]["panel_role"] == "opened_validation"
    assert config["dataset_contract"]["external_panel_access_authorized"] is False
    assert config["frozen_directional_grid"]["alpha_upper_bound_unchanged"]
    assert config["immutable_boundaries"][
        "candidate_e_source_config_alpha_selector_frozen"
    ]
    assert config["immutable_boundaries"][
        "opened_v15_outcomes_may_not_trigger_retuning"
    ]
    assert config["immutable_boundaries"]["generator_optimizer_steps"] == 0
    assert config["immutable_boundaries"][
        "authoritative_training_decision"
    ] == TRAINING_DECISION


def test_v29_has_no_old_v15_result_table_argument() -> None:
    source = inspect.getsource(parse_args)
    assert "v15-fresh-results" not in source
    assert "fresh-results" not in source


def test_selected_candidates_binds_frozen_family_alpha_and_gradient() -> None:
    selector = {
        "rows": [
            {
                "case_id": "synthetic",
                "selected": {
                    "direction_family": VARIANT_E_RAW,
                    "alpha": 0.0005,
                },
            }
        ]
    }
    candidate_rows = [
        {
            "case_id": "synthetic",
            "variant": VARIANT_E_RAW,
            "alpha": 0.0,
            "current_topology_proxy_shimmer_db": 1.3,
        },
        {
            "case_id": "synthetic",
            "variant": VARIANT_E_RAW,
            "alpha": 0.0005,
            "candidate_path": "/tmp/synthetic.wav",
            "candidate_sha256": "synthetic-sha256",
            "current_topology_proxy_shimmer_db": 1.2,
            "current_topology_sha256": "topology-sha256",
        },
    ]
    diagnostics = [
        {
            "case_id": "synthetic",
            "candidate_e_raw_gradient_l2": 3.5,
            "candidate_e_projection": {
                "projected_gradient_l2_norm": 1.5,
            },
        }
    ]
    selected = selected_candidates(selector, candidate_rows, diagnostics)
    row = selected["synthetic"]
    assert row["direction_family"] == VARIANT_E_RAW
    assert row["alpha"] == 0.0005
    assert row["proxy_before"] == 1.3
    assert row["proxy_after"] == 1.2
    assert row["gradient_l2_norm"] == 3.5
    assert row["gradient_finite"] is True


def test_v29_pass_authorizes_only_external_preparation() -> None:
    assert PASS_DECISION == (
        "PASS_CANDIDATE_E_OPENED_V15_CONFIRMATION_"
        "EXTERNAL_PREP_AUTHORIZED_V29"
    )
    assert TRAINING_DECISION == "NO_GO_AVQI_T2_TRAINING"
