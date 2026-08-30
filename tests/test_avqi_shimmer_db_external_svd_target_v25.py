from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from model.avqi_components import AVQI_COMPONENT_NAMES
import scripts.seal_avqi_shimmer_db_external_svd_target_v25 as v25


def sealed_panel(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    rows = []
    conditions = ("rir_only", "snr20", "snr10")
    for speaker_index in range(6):
        speaker_id = str(100 + speaker_index)
        for view_index, view in enumerate(v25.VIEWS):
            case_id = f"case-{speaker_id}-{view}"
            target_path = tmp_path / f"{case_id}.wav"
            waveform = np.sin(np.arange(640, dtype=np.float32) / 19.0) * 0.05
            sf.write(target_path, waveform, 16_000, subtype="FLOAT")
            rows.append(
                {
                    "case_id": case_id,
                    "dataset": "SVD",
                    "panel_speaker_id": f"SVD:{speaker_id}",
                    "speaker_id": speaker_id,
                    "session_id": str(1000 + speaker_index),
                    "sex": "female" if speaker_index < 3 else "male",
                    "label": "patient",
                    "view": view,
                    "condition": conditions[(2 * speaker_index + view_index) % 3],
                    "target_path": str(target_path),
                    "target_sha256": v25.sha256_file(target_path),
                }
            )
    panel = {
        "schema_version": v25.PANEL_SCHEMA,
        "source_commit": "a" * 40,
        "case_count": 12,
        "speaker_count": 6,
        "severity_labels_created": False,
        "authorization": {
            "opened24_report_sha256": "b" * 64,
            "external_speaker_panel_authorized": True,
        },
        "waveform_contract": {
            "emitted_waveform_highpass": False,
            "exact_metric_highpass_branch_only": True,
            "target_is_same_speaker_same_view_clean_pathological": True,
        },
        "exact_contract": {
            "target_shimmer_values_opened": False,
            "base_exact_outcomes_opened": False,
            "candidate_exact_outcomes_opened": False,
            "target_scalar_stage_authorized": True,
            "selector_stage_authorized": False,
            "promotion_authorized": False,
        },
        "rows": rows,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v25.TRAINING_DECISION,
    }
    receipt = {
        "schema_version": v25.SEAL_RECEIPT_SCHEMA,
        "decision": v25.PANEL_DECISION,
        "source_commit": "a" * 40,
        "exact_shimmer_outcomes_opened": False,
        "target_scalar_stage_authorized": True,
        "selector_stage_authorized": False,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": v25.TRAINING_DECISION,
        "artifact_sha256": {"panel_seal.json": "c" * 64},
    }
    return panel, receipt


def test_target_stage_requires_bound_exact_unopened_panel(tmp_path: Path) -> None:
    panel, receipt = sealed_panel(tmp_path)
    rows = v25.validate_panel_binding(
        panel,
        receipt,
        panel_sha256="c" * 64,
    )
    assert len(rows) == 12

    opened = deepcopy(panel)
    opened["exact_contract"]["base_exact_outcomes_opened"] = True
    with pytest.raises(ValueError, match="exact-opening contract drift"):
        v25.validate_panel_binding(
            opened,
            receipt,
            panel_sha256="c" * 64,
        )

    unbound = deepcopy(receipt)
    unbound["artifact_sha256"]["panel_seal.json"] = "d" * 64
    with pytest.raises(ValueError, match="receipt/seal binding drift"):
        v25.validate_panel_binding(
            panel,
            unbound,
            panel_sha256="c" * 64,
        )


def test_target_stage_rejects_invented_svd_severity(tmp_path: Path) -> None:
    panel, receipt = sealed_panel(tmp_path)
    panel["rows"][0]["sample_group"] = "pathological_mild"
    with pytest.raises(ValueError, match="severity label leakage"):
        v25.validate_panel_binding(
            panel,
            receipt,
            panel_sha256="c" * 64,
        )


def test_target_contract_retains_only_supervised_shimmer_scalar(
    tmp_path: Path,
) -> None:
    panel, receipt = sealed_panel(tmp_path)
    rows = v25.validate_panel_binding(
        panel,
        receipt,
        panel_sha256="c" * 64,
    )
    exact_rows = []
    for index, row in enumerate(rows):
        components = {
            name: float(index + component)
            for component, name in enumerate(AVQI_COMPONENT_NAMES)
        }
        exact_rows.append(
            {
                "id": f"target:{row['case_id']}",
                "components": components,
            }
        )
    exact = {
        "parselmouth_version": "test-parselmouth",
        "praat_version": "test-praat",
        "rows": exact_rows,
    }
    contract = v25.build_target_contract(
        panel,
        rows,
        exact,
        panel_sha256="c" * 64,
        source_commit="a" * 40,
        slurm_job_id="123",
        avqi_tree_sha256="d" * 64,
    )

    assert contract["target_exact_components_retained"] == ["shimmer_db"]
    assert contract["selector_stage_authorized"] is True
    assert contract["scientific_promotion_granted"] is False
    assert contract["generator_optimizer_steps"] == 0
    assert all(
        set(row) == {
            "case_id",
            "panel_speaker_id",
            "speaker_id",
            "session_id",
            "sex",
            "view",
            "condition",
            "target_sha256",
            "exact_target_shimmer_db",
        }
        for row in contract["rows"]
    )


def test_target_stage_source_has_no_candidate_or_training_execution() -> None:
    source = Path(v25.__file__).read_text(encoding="utf-8")

    assert "evaluate_selector_case" not in source
    assert "torch.optim" not in source
    assert '"base_exact_outcomes_opened": False' in source
    assert '"candidate_exact_outcomes_opened": False' in source
    assert '"scientific_promotion_granted": False' in source
    assert '"joint_panel_authorized": False' in source
    assert '"generator_optimizer_steps": 0' in source
    assert '"emitted_waveform_highpass": False' in source
