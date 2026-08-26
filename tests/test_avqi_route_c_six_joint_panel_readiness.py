from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from model.avqi_route_c import (
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_SCIENTIFIC_STATUS,
    route_c_six_registry_records,
)
from scripts import audit_avqi_route_c_six_joint_panel_readiness as readiness
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    DRAFT_SIX_GRADIENT_SCHEMA_VERSION,
    DRAFT_SPLIT_SEAL_SCHEMA_VERSION,
    DRAFT_PANEL_DATA_REQUIREMENTS,
    FIVE_COMPONENT_EVIDENCE_KEYS,
    HEALTHY_ROLE,
    MISSING_CODE_STAGES,
    PANEL_ROW_FIELDS,
    PATHOLOGICAL_ROLE,
    READINESS_SCHEMA_VERSION,
    REQUIRED_ARTIFACT_KEYS,
    REQUIRED_CONDITIONS,
    REQUIRED_SPLITS,
    REQUIRED_VIEWS,
    SHIMMER_DB_REQUIRED_STATUS,
    SIX_GRADIENT_PASS_DECISION,
    SIX_GRADIENT_SOURCE_EVIDENCE_KEYS,
    UNFROZEN_SCIENTIFIC_CONTRACTS,
    _validate_five_component_evidence,
    _validate_panel_rows,
    _validate_six_gradient,
    _validate_split_seal,
    current_blockers,
    readiness_requirements,
    sha256_file,
    validate_readiness_manifest,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _bind(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _promoted_registry() -> list[dict[str, object]]:
    records = route_c_six_registry_records()
    records[3]["scientific_status"] = SHIMMER_DB_REQUIRED_STATUS
    records[3]["scientific_promotion_granted"] = True
    return records


def _readiness_manifest(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "source_commit": "a" * 40,
        "generator_optimizer_steps": 0,
    }
    value.update(overrides)
    return value


def _panel_rows() -> list[dict[str, str]]:
    rows = []
    for split in REQUIRED_SPLITS:
        for label, role in (
            ("patient", PATHOLOGICAL_ROLE),
            ("healthy", HEALTHY_ROLE),
        ):
            speaker = f"{split}-{label}-speaker"
            for condition in REQUIRED_CONDITIONS:
                for view in REQUIRED_VIEWS:
                    rows.append(
                        {
                            "case_id": f"{speaker}-{condition}-{view}",
                            "speaker_id": speaker,
                            "split": split,
                            "view": view,
                            "condition": condition,
                            "label": label,
                            "optimization_role": role,
                            "source_waveform_sha256": "a" * 64,
                        }
                    )
    return rows


def _five_evidence_bundle(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    paths: dict[str, Path] = {}
    contracts = {
        "cpps": ("cpps_report", "cpps_receipt", "PASS_WAVEFORM_OPTIMIZATION"),
        "hnr": ("hnr_report", "hnr_receipt", "PASS_HNR_FRESH_SPEAKER_PANEL"),
        "shimmer_percent": (
            "shimmer_percent_report",
            "shimmer_percent_receipt",
            "PASS_SHIMMER_FRESH_SPEAKER_PANEL",
        ),
        "slope": (
            "slope_report",
            "slope_receipt",
            "PASS_LTAS_SLOPE_FRESH_SPEAKER_PANEL",
        ),
        "tilt": ("tilt_report", "tilt_receipt", "FAIL_WAVEFORM_OPTIMIZATION"),
    }
    paths["slope_final_panel_seal"] = _write_json(
        tmp_path / "final_panel_seal.json", {"sealed": True}
    )
    paths["slope_final_results"] = tmp_path / "final_results.csv"
    paths["slope_final_results"].write_text("case_id,decision\n", encoding="utf-8")
    for component, (report_key, receipt_key, decision) in contracts.items():
        if component in {"hnr", "shimmer_percent", "slope"}:
            report = {
                "decision": decision,
                "final_exact_panel_opened": True,
                "final": {"decision": "PASS", "gates": {"frozen": True}},
                "formal_pathology_training_submitted": False,
                "generator_optimizer_steps": 0,
            }
        else:
            report = {
                "schema_version": (
                    "direct-avqi-waveform-optimization-v3"
                    if component == "cpps"
                    else "direct-avqi-waveform-optimization-v1"
                ),
                "decision": decision,
                "summary": {
                    "component_gates": {
                        component: {
                            "decision": "PASS",
                            "gates": {"frozen": True},
                        }
                    },
                    "safety": {"decision": "PASS"},
                },
                "formal_pathology_training_submitted": False,
                "generator_optimizer_steps": 0,
            }
        paths[report_key] = _write_json(tmp_path / f"{report_key}.json", report)
        receipt_hashes = {
            paths[report_key].name: sha256_file(paths[report_key])
        }
        if component == "slope":
            receipt_hashes.update(
                {
                    paths["slope_final_panel_seal"].name: sha256_file(
                        paths["slope_final_panel_seal"]
                    ),
                    paths["slope_final_results"].name: sha256_file(
                        paths["slope_final_results"]
                    ),
                }
            )
        paths[receipt_key] = _write_json(
            tmp_path / f"{receipt_key}.json",
            {
                "decision": decision,
                "artifact_sha256": receipt_hashes,
                "generator_optimizer_steps": 0,
            },
        )

    artifacts = {key: _bind(paths[key]) for key in FIVE_COMPONENT_EVIDENCE_KEYS}
    five_report = {
        "schema_version": "avqi_route_c_five_component_gradient_audit_v1",
        "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
        "source_evidence": {
            key: artifacts[key] for key in FIVE_COMPONENT_EVIDENCE_KEYS
        },
        "gates": {"frozen": True},
        "generator_optimizer_steps": 0,
    }
    paths["five_gradient_report"] = _write_json(
        tmp_path / "gradient_interference_report.json", five_report
    )
    paths["five_gradient_receipt"] = _write_json(
        tmp_path / "five_completion_receipt.json",
        {
            "decision": "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT",
            "active_components": [
                "cpps",
                "hnr",
                "shimmer_percent",
                "slope",
                "tilt",
            ],
            "inactive_slots": ["shimmer_db"],
            "artifact_sha256": {
                paths["five_gradient_report"].name: sha256_file(
                    paths["five_gradient_report"]
                )
            },
            "generator_optimizer_steps": 0,
        },
    )
    artifacts.update(
        {
            key: _bind(paths[key])
            for key in ("five_gradient_report", "five_gradient_receipt")
        }
    )
    return artifacts, paths


def _pairwise_keys() -> tuple[str, ...]:
    return tuple(
        f"{left}__{right}"
        for index, left in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS)
        for right in ROUTE_C_SIX_ACTIVE_COMPONENTS[index + 1 :]
    )


def _six_gradient_evidence() -> tuple[dict[str, object], dict[str, object], str]:
    source_evidence = {
        key: "a" * 64 for key in SIX_GRADIENT_SOURCE_EVIDENCE_KEYS
    }
    share = 1.0 / len(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    report: dict[str, object] = {
        "schema_version": DRAFT_SIX_GRADIENT_SCHEMA_VERSION,
        "decision": SIX_GRADIENT_PASS_DECISION,
        "active_components": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "source_evidence_sha256": source_evidence,
        "shimmer_db_topology_role": "base_current_output",
        "slot3_checkpoint_affine_used": False,
        "slot2_checkpoint_unchanged": True,
        "selection": {
            "allowed_splits": ["surrogate_calibration", "surrogate_holdout"],
            "speaker_overlap": 0,
            "component_and_joint_share_split": True,
            "calibration_speaker_ids": ["cal-speaker"],
            "holdout_speaker_ids": ["hold-speaker"],
            "final_panel_opened": False,
        },
        "calibration": {
            "frozen_inverse_gradient_weights": {
                name: 1.0 for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
            }
        },
        "holdout": {
            "all_component_gradients_pass": True,
            "all_joint_gradients_pass": True,
            "all_pairwise_cosines_reported": True,
            "all_component_to_joint_cosines_reported": True,
            "bounded_gates_pass": True,
            "weighted_dominance_gate_pass": True,
            "component_gradient_norms": {
                name: 1.0 for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
            },
            "component_to_joint_cosines": {
                name: 0.5 for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
            },
            "pairwise_cosines": {key: 0.0 for key in _pairwise_keys()},
            "weighted_component_shares": {
                name: share for name in ROUTE_C_SIX_ACTIVE_COMPONENTS
            },
            "max_weighted_component_share": share,
            "joint_gradient_norm": 1.0,
        },
        "gates": {"draft_all_bounded_gates": True},
        "joint_scientific_promotion_granted": False,
        "combined_final_panel_opened": False,
        "generator_optimizer_steps": 0,
    }
    report_sha256 = "b" * 64
    receipt = {
        "decision": SIX_GRADIENT_PASS_DECISION,
        "artifact_sha256": {"gradient_report.json": report_sha256},
        "generator_optimizer_steps": 0,
    }
    return report, receipt, report_sha256


def test_current_requirements_are_explicitly_no_go() -> None:
    requirements = readiness_requirements()

    assert requirements["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert requirements["execution_authorized"] is False
    assert requirements["current_shimmer_db_scientific_status"] == (
        ROUTE_C_SIX_SCIENTIFIC_STATUS
    )
    assert requirements["required_conditions"] == list(REQUIRED_CONDITIONS)
    assert "clean" in requirements["required_conditions"]
    assert requirements["missing_code_stages"] == list(MISSING_CODE_STAGES)
    assert requirements["unfrozen_scientific_contracts"] == list(
        UNFROZEN_SCIENTIFIC_CONTRACTS
    )
    assert requirements["draft_panel_data_requirements"] == list(
        DRAFT_PANEL_DATA_REQUIREMENTS
    )
    assert set(FIVE_COMPONENT_EVIDENCE_KEYS) <= set(REQUIRED_ARTIFACT_KEYS)
    assert "thresholds" not in requirements
    assert current_blockers()[0] == "Shimmer dB scientific promotion remains pending"


def test_existing_cpps_evidence_is_unbound_not_missing() -> None:
    blockers = current_blockers()

    assert "not yet bound into a six-joint manifest: cpps_report" in blockers
    assert not any(
        "missing" in blocker and "cpps_report" in blocker for blocker in blockers
    )


def test_pending_registry_closes_manifest_before_artifacts() -> None:
    with pytest.raises(ValueError, match="scientific promotion is still pending"):
        validate_readiness_manifest(_readiness_manifest())


def test_promoted_registry_still_closes_unfrozen_scientific_schemas() -> None:
    with pytest.raises(ValueError, match="scientific schemas remain unfrozen"):
        validate_readiness_manifest(
            _readiness_manifest(),
            registry_records=_promoted_registry(),
        )


def test_opened_exact_outcomes_fail_before_registry_or_artifacts() -> None:
    with pytest.raises(ValueError, match="opened candidate outcomes"):
        validate_readiness_manifest(
            _readiness_manifest(candidate_exact_outcomes_opened=True)
        )


def test_complete_per_speaker_pathological_and_healthy_matrix_passes_draft() -> None:
    rows_by_case, speakers = _validate_panel_rows(_panel_rows(), "test panel")

    assert len(rows_by_case) == (
        len(REQUIRED_SPLITS)
        * 2
        * len(REQUIRED_CONDITIONS)
        * len(REQUIRED_VIEWS)
    )
    assert set(speakers) == set(REQUIRED_SPLITS)


def test_aggregate_cs_sv_coverage_cannot_hide_incomplete_speaker_matrix() -> None:
    rows = _panel_rows()
    rows.pop(0)

    with pytest.raises(ValueError, match="speaker matrix differs"):
        _validate_panel_rows(rows, "test panel")


def test_healthy_rows_cannot_become_optimization_targets() -> None:
    rows = _panel_rows()
    healthy = next(row for row in rows if row["label"] == "healthy")
    healthy["optimization_role"] = PATHOLOGICAL_ROLE

    with pytest.raises(ValueError, match="row semantics differ"):
        _validate_panel_rows(rows, "test panel")


def test_split_seal_requires_complete_stratified_speaker_matrix() -> None:
    rows = _panel_rows()
    seal = {
        "schema_version": DRAFT_SPLIT_SEAL_SCHEMA_VERSION,
        "exact_scores_opened": False,
        "speaker_split_before_simulation": True,
        "selection_or_tuning_on_this_panel": False,
        "joint_gate_contract_sha256": "a" * 64,
        "target_value_protocol_sha256": "b" * 64,
        "prior_panel_speaker_ledger_sha256": "c" * 64,
        "fresh_speaker_source_manifest_sha256": "d" * 64,
        "rows": [{field: row[field] for field in PANEL_ROW_FIELDS} for row in rows],
        "generator_optimizer_steps": 0,
    }

    speakers = _validate_split_seal(
        seal,
        gate_sha256="a" * 64,
        target_sha256="b" * 64,
        ledger_sha256="c" * 64,
        source_sha256="d" * 64,
    )
    assert set(speakers) == set(REQUIRED_SPLITS)

    seal["rows"][0]["speaker_id"] = "wrong-speaker"
    with pytest.raises(ValueError, match="speaker matrix differs|pairing differs"):
        _validate_split_seal(
            seal,
            gate_sha256="a" * 64,
            target_sha256="b" * 64,
            ledger_sha256="c" * 64,
            source_sha256="d" * 64,
        )


def test_five_component_frozen_evidence_set_accepts_tilt_component_pass(
    tmp_path: Path,
) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)

    evidence = _validate_five_component_evidence(artifacts, paths)

    assert set(evidence) == set(FIVE_COMPONENT_EVIDENCE_KEYS)
    tilt = json.loads(paths["tilt_report"].read_text(encoding="utf-8"))
    assert tilt["decision"] == "FAIL_WAVEFORM_OPTIMIZATION"
    assert tilt["summary"]["component_gates"]["tilt"]["decision"] == "PASS"


def test_tilt_container_fail_does_not_mask_component_gate_failure(
    tmp_path: Path,
) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)
    report = json.loads(paths["tilt_report"].read_text(encoding="utf-8"))
    report["summary"]["component_gates"]["tilt"]["gates"]["frozen"] = False
    _write_json(paths["tilt_report"], report)
    receipt = json.loads(paths["tilt_receipt"].read_text(encoding="utf-8"))
    receipt["artifact_sha256"][paths["tilt_report"].name] = sha256_file(
        paths["tilt_report"]
    )
    _write_json(paths["tilt_receipt"], receipt)
    artifacts["tilt_report"] = _bind(paths["tilt_report"])
    artifacts["tilt_receipt"] = _bind(paths["tilt_receipt"])

    with pytest.raises(ValueError, match="contains a failed gate"):
        _validate_five_component_evidence(artifacts, paths)


def test_slope_sealed_results_must_be_bound_by_receipt(tmp_path: Path) -> None:
    artifacts, paths = _five_evidence_bundle(tmp_path)
    receipt = json.loads(paths["slope_receipt"].read_text(encoding="utf-8"))
    receipt["artifact_sha256"].pop(paths["slope_final_results"].name)
    _write_json(paths["slope_receipt"], receipt)
    artifacts["slope_receipt"] = _bind(paths["slope_receipt"])

    with pytest.raises(ValueError, match="does not bind slope_final_results"):
        _validate_five_component_evidence(artifacts, paths)


def test_six_gradient_binds_five_evidence_and_reports_all_interference() -> None:
    report, receipt, report_sha256 = _six_gradient_evidence()
    expected = {
        key: "a" * 64 for key in SIX_GRADIENT_SOURCE_EVIDENCE_KEYS
    }

    weights = _validate_six_gradient(
        report,
        receipt,
        report_sha256,
        expected,
    )
    assert set(weights) == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)

    report["source_evidence_sha256"].pop("tilt_receipt")
    with pytest.raises(ValueError, match="source-evidence binding differs"):
        _validate_six_gradient(report, receipt, report_sha256, expected)


def test_preflight_source_contains_no_panel_execution_path() -> None:
    source = Path(readiness.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "sbatch",
        "parselmouth",
        "soundfile",
        "run_exact_batch",
        "torch.optim",
    ):
        assert forbidden not in source
    assert "raise SystemExit(2)" in source


def test_preflight_direct_cli_requirements_only_without_pythonpath() -> None:
    root = Path(readiness.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(readiness.__file__).resolve()),
            "--requirements-only",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert report["execution_authorized"] is False
    assert report["generator_optimizer_steps"] == 0


def test_preflight_direct_cli_execution_mode_fails_closed() -> None:
    root = Path(readiness.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(readiness.__file__).resolve())],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    assert report["decision"] == "NO_GO_SIX_JOINT_PANEL_EXECUTION"
    assert report["execution_authorized"] is False
    assert report["generator_optimizer_steps"] == 0
