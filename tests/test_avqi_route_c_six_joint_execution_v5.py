from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from scripts import materialize_avqi_route_c_six_joint_inputs_v5 as materializer
from scripts import prepare_avqi_route_c_six_joint_waveforms as waveform_seal
from scripts.audit_avqi_route_c_six_joint_execution_v5 import (
    EXECUTION_INPUT_NAMES,
    candidate_e_function_parity,
    validate_gate_contract,
    validate_target_protocol,
    parse_bindings,
)
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    PRIOR_PANEL_LEDGER_SCHEMA,
    SOURCE_GENDER_ALLOCATION,
)
from scripts.prepare_avqi_route_c_six_joint_inputs_v5 import (
    SOURCE_MANIFEST_SCHEMA,
    canonical_speaker_id,
    panel_rows as build_panel_rows,
    recipe_manifest,
    selection_digest,
)


ROOT = Path(__file__).resolve().parents[1]


def _selected_speakers() -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    bucket_rank: dict[tuple[str, str], int] = {}
    speaker_index = 0
    for split, label, female, male in SOURCE_GENDER_ALLOCATION:
        for gender, count in (("female", female), ("male", male)):
            for _ in range(count):
                speaker_index += 1
                key = (label, gender)
                bucket_rank[key] = bucket_rank.get(key, 0) + 1
                selected.append(
                    {
                        "speaker_id": str(9000 + speaker_index),
                        "session_id": str(8000 + speaker_index),
                        "split": split,
                        "label": label,
                        "health_status": "1" if label == "patient" else "0",
                        "gender": gender,
                        "rank_within_label_gender": bucket_rank[key],
                    }
                )
    return selected


def _source_manifest_fixture(
    selected: list[dict[str, object]],
) -> dict[str, object]:
    rows = []
    for speaker in selected:
        speaker_id = str(speaker["speaker_id"])
        session_id = str(speaker["session_id"])
        rows.append(
            {
                "dataset": "SVD",
                "canonical_speaker_id": canonical_speaker_id(speaker_id),
                "speaker_id": speaker_id,
                "session_id": session_id,
                "split": speaker["split"],
                "label": speaker["label"],
                "health_status": speaker["health_status"],
                "gender": speaker["gender"],
                "diagnosis_record_only": "record-only",
                "rank_within_label_gender": speaker[
                    "rank_within_label_gender"
                ],
                "selection_digest": selection_digest(speaker_id, session_id),
                "waveforms": {
                    "cs": {
                        "path": f"/tmp/{speaker_id}_cs.wav",
                        "sha256": "a" * 64,
                        "duration_seconds": 3.0,
                        "source_sample_rate": 16_000,
                        "source_frames": 48_000,
                        "channels": 1,
                    },
                    "sv": {
                        "path": f"/tmp/{speaker_id}_sv.wav",
                        "sha256": "b" * 64,
                        "duration_seconds": 1.0,
                        "source_sample_rate": 16_000,
                        "source_frames": 16_000,
                        "channels": 1,
                    },
                },
            }
        )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "source_dataset": "SVD",
        "sv_metadata_sha256": materializer.SVD_SV_METADATA_SHA256,
        "cs_metadata_sha256": materializer.SVD_CS_METADATA_SHA256,
        "selection_salt": materializer.SVD_SPEAKER_SELECTION_SALT,
        "prior_panel_speaker_ledger_sha256": "c" * 64,
        "source_prior_panel_speaker_ledger_sha256": "d" * 64,
        "selection_mode": "metadata_only_result_blind",
        "selection_operation_order": [
            "map_health_status",
            "pair_cs_sv_by_same_session",
            "filter_raw_mono_minimum_duration",
            "retain_minimum_numeric_eligible_session_per_speaker",
            "exclude_prior_ledger_speakers",
            "bucket_by_health_status_and_gender",
            "rank_by_salted_sha256",
            "allocate_calibration_then_final_by_frozen_gender_quota",
        ],
        "diagnosis_used_for_selection": False,
        "exact_scores_opened": False,
        "candidate_outcomes_opened": False,
        "mild_severe_labels_created": False,
        "speaker_count": 12,
        "counts": {
            f"{split}:{label}:{gender}": count
            for split, label, female, male in SOURCE_GENDER_ALLOCATION
            for gender, count in (("female", female), ("male", male))
        },
        "rows": rows,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": waveform_seal.TRAINING_NO_GO,
    }


def test_frozen_v5_contract_files_validate() -> None:
    gate = json.loads(
        (ROOT / "configs" / "avqi_route_c_six_joint_gate_contract_v1.json")
        .read_text(encoding="utf-8")
    )
    target = json.loads(
        (ROOT / "configs" / "avqi_route_c_six_joint_target_protocol_v1.json")
        .read_text(encoding="utf-8")
    )

    validate_gate_contract(gate)
    validate_target_protocol(target)


def test_binding_parser_requires_exact_seven_names(tmp_path: Path) -> None:
    rows = []
    for name in EXECUTION_INPUT_NAMES:
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append([name, str(path), digest])

    parsed = parse_bindings(rows, EXECUTION_INPUT_NAMES, "execution input")
    assert tuple(parsed) == EXECUTION_INPUT_NAMES


def test_binding_parser_rejects_missing_input(tmp_path: Path) -> None:
    path = tmp_path / "only.json"
    path.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="names differ"):
        parse_bindings(
            [[EXECUTION_INPUT_NAMES[0], str(path), digest]],
            EXECUTION_INPUT_NAMES,
            "execution input",
        )


def test_candidate_e_function_parity_detects_math_drift(
    tmp_path: Path,
) -> None:
    integrated = ROOT / "model" / "avqi_route_c_candidate_e.py"
    parity = candidate_e_function_parity(integrated, integrated)
    assert parity and all(parity.values())

    altered = tmp_path / "altered_candidate_e.py"
    altered.write_text(
        integrated.read_text(encoding="utf-8").replace(
            "bounded * 32768.0",
            "bounded * 32767.0",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="function parity differs"):
        candidate_e_function_parity(altered, integrated)


def test_source_manifest_revalidates_quota_and_prior_disjointness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _selected_speakers()
    panel = {
        row["case_id"]: row for row in build_panel_rows(selected)
    }
    manifest = _source_manifest_fixture(selected)
    monkeypatch.setattr(materializer, "validate_hash", lambda *args: "ok")

    def fake_info(path: Path) -> SimpleNamespace:
        is_cs = str(path).endswith("_cs.wav")
        return SimpleNamespace(
            channels=1,
            frames=48_000 if is_cs else 16_000,
            samplerate=16_000,
        )

    monkeypatch.setattr(materializer.sf, "info", fake_info)
    prior = {
        "schema_version": PRIOR_PANEL_LEDGER_SCHEMA,
        "exact_outcomes_used_for_selection": False,
        "entries": [
            {
                "dataset": "TAU",
                "speaker_id": "unused",
                "canonical_speaker_id": "TAU:unused",
            }
        ],
    }
    observed = materializer.validate_source_manifest(
        manifest,
        panel,
        prior_ledger=prior,
        prior_ledger_sha256="c" * 64,
        source_prior_ledger_sha256="d" * 64,
    )
    assert len(observed) == 24

    prior["entries"][0] = {
        "dataset": "SVD",
        "speaker_id": selected[0]["speaker_id"],
        "canonical_speaker_id": canonical_speaker_id(
            str(selected[0]["speaker_id"])
        ),
    }
    with pytest.raises(ValueError, match="overlaps"):
        materializer.validate_source_manifest(
            manifest,
            panel,
            prior_ledger=prior,
            prior_ledger_sha256="c" * 64,
            source_prior_ledger_sha256="d" * 64,
        )


def test_recipe_manifest_revalidates_every_frozen_assignment() -> None:
    panel_list = build_panel_rows(_selected_speakers())
    panel = {row["case_id"]: row for row in panel_list}
    recipes = [
        {"split": "test", "target_sample_rate": 16_000, "uid": f"r{index}"}
        for index in range(72)
    ]
    manifest = recipe_manifest(
        panel_list,
        recipes,
        fixed_recipes_sha256="f" * 64,
    )
    observed = materializer.validate_recipe_manifest(
        manifest,
        panel,
        recipes,
        "f" * 64,
    )
    assert len(observed) == 96

    altered = json.loads(json.dumps(manifest))
    degraded = next(
        row for row in altered["rows"] if row["condition"] == "snr20"
    )
    degraded["target_snr_db"] = 10.0
    with pytest.raises(ValueError, match="condition differs"):
        materializer.validate_recipe_manifest(
            altered,
            panel,
            recipes,
            "f" * 64,
        )


def test_gradient_manifest_requires_all_projection_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = {
        row["case_id"]: row
        for row in build_panel_rows(_selected_speakers())
    }
    weights = {component: 1.0 for component in ROUTE_C_SIX_ACTIVE_COMPONENTS}
    normalization = {
        "target_mean": {
            component: 0.0 for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
        },
        "target_scale": {
            component: 1.0 for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
        },
    }
    rows = []
    projections = {}
    for case_id, panel_row in panel.items():
        patient = panel_row["optimization_role"] != waveform_seal.HEALTHY_ROLE
        rows.append(
            {
                "case_id": case_id,
                "base_waveform_path": f"/tmp/{case_id}.wav",
                "base_waveform_sha256": "a" * 64,
                "joint_gradient_path": (
                    f"/tmp/{case_id}.npy" if patient else None
                ),
                "joint_gradient_sha256": "b" * 64 if patient else None,
                "topology_sha256": "c" * 64 if patient else None,
            }
        )
        if patient:
            projections[case_id] = {
                "projection_reduction": "numpy_float64_fixed_cycle_order",
                "projected_gradient_valid": True,
                "projected_gradient_finite": True,
                "complete_cycle_count": 16,
                "projected_gradient_l2_norm": 1.0,
                "candidate_e_sinc70_peak_upper_bound": 0.5,
                "candidate_e_peak_scale_abstention_pass": True,
            }
    manifest = {
        "schema_version": waveform_seal.GRADIENT_MANIFEST_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            waveform_seal.FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "split_seal_sha256": "1" * 64,
        "clean_target_label_bank_sha256": "2" * 64,
        "six_gradient_report_sha256": "3" * 64,
        "six_gradient_receipt_sha256": "4" * 64,
        "six_gradient_raw_report_sha256": "5" * 64,
        "six_gradient_decision": waveform_seal.SIX_GRADIENT_PASS_DECISION,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "calibration_inverse_gradient_weights": weights,
        "normalization_source": waveform_seal.NORMALIZATION_SOURCE,
        "normalization": normalization,
        "gradient_source": waveform_seal.GRADIENT_SOURCE,
        "current_output_topology_bound": True,
        "candidate_e_projection": (
            "numpy_float64_fixed_cycle_order_before_six_component_combination"
        ),
        "candidate_e_projection_receipts": projections,
        "waveform_steps": 1,
        "gradient_normalization": "waveform_rms_normalized",
        "candidate_exact_outcomes_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": waveform_seal.TRAINING_NO_GO,
        "rows": rows,
    }
    monkeypatch.setattr(
        waveform_seal,
        "_load_audio",
        lambda *args: np.ones(16, dtype=np.float32),
    )
    monkeypatch.setattr(
        waveform_seal,
        "_load_gradient",
        lambda *args: np.ones(16, dtype=np.float32),
    )
    observed = waveform_seal.validate_gradient_manifest(
        manifest,
        split_seal_sha256="1" * 64,
        target_bank_sha256="2" * 64,
        six_gradient_report_sha256="3" * 64,
        six_gradient_receipt_sha256="4" * 64,
        six_gradient_raw_report_sha256="5" * 64,
        weights=weights,
        normalization=normalization,
        panel_rows=panel,
    )
    assert len(observed) == 96

    del projections[next(iter(projections))]
    with pytest.raises(ValueError, match="projection receipt coverage"):
        waveform_seal.validate_gradient_manifest(
            manifest,
            split_seal_sha256="1" * 64,
            target_bank_sha256="2" * 64,
            six_gradient_report_sha256="3" * 64,
            six_gradient_receipt_sha256="4" * 64,
            six_gradient_raw_report_sha256="5" * 64,
            weights=weights,
            normalization=normalization,
            panel_rows=panel,
        )
