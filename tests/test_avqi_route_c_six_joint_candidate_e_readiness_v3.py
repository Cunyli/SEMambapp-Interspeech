from __future__ import annotations

import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_avqi_route_c_six_joint_candidate_e_readiness_v3 as v3


def _training_boundary() -> dict[str, object]:
    return {
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": v3.TRAINING_NO_GO,
    }


def _promotion_report() -> dict[str, object]:
    return {
        "schema_version": v3.PROMOTION_REPORT_SCHEMA,
        "decision": v3.PROMOTION_PASS,
        "component": "shimmer_db",
        "component_status": v3.PROMOTION_PASS,
        "readiness_status": v3.SHIMMER_READINESS,
        "source_commit": v3.PROMOTION_SOURCE_COMMIT,
        "slurm_job_id": v3.PROMOTION_JOB_ID,
        "source_provenance": {"runner_sha256": v3.PROMOTION_RUNNER_SHA256},
        "source_sha256": {
            "runtime_successor": {
                "runtime_config": v3.RUNTIME_CONFIG_SHA256,
                "v32r2_report": v3.V32R2_REPORT_SHA256,
                "v32r2_receipt": v3.V32R2_RECEIPT_SHA256,
            },
            "panel_seal": v3.PANEL_SEAL_SHA256,
            "panel_receipt": v3.PANEL_RECEIPT_SHA256,
            "updated_speaker_ledger": v3.UPDATED_LEDGER_SHA256,
            "target_contract": v3.TARGET_SEAL_SHA256,
            "target_receipt": v3.TARGET_RECEIPT_SHA256,
        },
        "evidence_bindings": {
            "v32r2_preexact_no_go_preserved": True,
            "v32r2_preexact_no_go_report_sha256": v3.V32R2_REPORT_SHA256,
            "v32r2_preexact_no_go_receipt_sha256": v3.V32R2_RECEIPT_SHA256,
            "updated_speaker_ledger_sha256": v3.UPDATED_LEDGER_SHA256,
        },
        "preexact_gates": {key: True for key in v3.PREEXACT_GATES},
        "candidate_pool_equivalence": {
            "all_equal": True,
            "candidate_grid_waveform_byte_equal": True,
            "candidate_grid_topology_hash_equal": True,
            "selector_choice_equal": True,
            "maximum_current_topology_proxy_absolute_error": 0.0,
            "candidate_exact_outcomes_used": False,
        },
        "summary": {
            "all_gates_pass": True,
            "external_effect_slices": {"decision": "PASS"},
            "mechanism_gates": {"exact_effect": True, "runtime": True},
            "integration_gates": {"safety": True, "pathology": True},
        },
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "exact_scoring_complete": True,
        "result_blind_external_three_stage_chain_complete": True,
        "old_v23_no_go_preserved": True,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "external_speaker_gate_pass": True,
        "bounded_waveform_promotion_pass": True,
        "scientific_promotion_granted": True,
        "six_component_readiness_eligible": True,
        "joint_panel_authorized": False,
        **_training_boundary(),
    }


def _promotion_receipt(report_sha256: str) -> dict[str, object]:
    artifacts = {name: "a" * 64 for name in v3.RECEIPT_ARTIFACTS}
    artifacts["external_svd_report_v32r3.json"] = report_sha256
    return {
        "schema_version": v3.PROMOTION_RECEIPT_SCHEMA,
        "decision": v3.PROMOTION_PASS,
        "component": "shimmer_db",
        "source_commit": v3.PROMOTION_SOURCE_COMMIT,
        "slurm_job_id": v3.PROMOTION_JOB_ID,
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "exact_scoring_complete": True,
        "result_blind_external_three_stage_chain_complete": True,
        "old_v23_no_go_preserved": True,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "scientific_promotion_granted": True,
        "six_component_readiness_eligible": True,
        "joint_panel_authorized": False,
        "artifact_sha256": artifacts,
        **_training_boundary(),
    }


def _ledger() -> dict[str, object]:
    return {
        "schema_version": "avqi-route-c-prior-panel-speaker-ledger-v1",
        "exact_outcomes_used_for_selection": False,
        "target_component_scorability_boolean_used_for_selection": True,
        "target_scalar_values_used_for_selection": False,
        "entries": [
            {
                "dataset": "SVD",
                "speaker_id": str(index),
                "canonical_speaker_id": f"SVD:{index}",
            }
            for index in range(35)
        ],
    }


def test_v3_contract_was_frozen_result_blind_while_external_was_pending() -> None:
    path = (
        Path(v3.__file__).resolve().parents[1]
        / "configs"
        / "avqi_route_c_six_joint_candidate_e_readiness_contract_v3.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    v3.validate_contract(contract)
    assert contract["frozen_while_external_job_state"] == {
        "slurm_job_id": "20042036",
        "observed_state": "PENDING",
        "candidate_exact_outcomes_opened": False,
        "scientific_result_available": False,
    }
    assert contract["historical_boundary"][
        "candidate_d_v23_no_go_remains_immutable"
    ] is True
    assert contract["readiness_interpretation"]["joint_panel_authorized"] is False


def test_v3_accepts_only_complete_candidate_e_pass_evidence() -> None:
    report_hash = "b" * 64
    result = v3.validate_promotion(
        _promotion_report(),
        _promotion_receipt(report_hash),
        report_sha256=report_hash,
        ledger_sha256=v3.UPDATED_LEDGER_SHA256,
    )
    assert result == {
        "preexact_gate_count": len(v3.PREEXACT_GATES),
        "mechanism_gate_count": 2,
        "integration_gate_count": 2,
        "receipt_artifact_count": len(v3.RECEIPT_ARTIFACTS),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda report: report.update(
                {"decision": "NO_GO_CANDIDATE_E_EXTERNAL_SVD_EXACT_PROMOTION_V32R3"}
            ),
            "identity differs",
        ),
        (
            lambda report: report["preexact_gates"].update(
                {"total_metric_step_runtime_le_500ms": False}
            ),
            "pre-exact gates did not pass",
        ),
        (
            lambda report: report["candidate_pool_equivalence"].update(
                {"candidate_grid_waveform_byte_equal": False}
            ),
            "serial equivalence differs",
        ),
        (
            lambda report: report.update(
                {"formal_generator_training_authorized": True}
            ),
            "training boundary differs",
        ),
    ),
)
def test_v3_rejects_failed_or_overauthorized_external_evidence(
    mutation,
    message: str,
) -> None:
    report = _promotion_report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        v3.validate_promotion(
            report,
            _promotion_receipt("b" * 64),
            report_sha256="b" * 64,
            ledger_sha256=v3.UPDATED_LEDGER_SHA256,
        )


def test_v3_rejects_receipt_that_does_not_bind_report() -> None:
    receipt = _promotion_receipt("c" * 64)
    with pytest.raises(ValueError, match="receipt artifacts differ"):
        v3.validate_promotion(
            _promotion_report(),
            receipt,
            report_sha256="b" * 64,
            ledger_sha256=v3.UPDATED_LEDGER_SHA256,
        )


def test_v3_rejects_nonfinite_proxy_equivalence_error() -> None:
    report = _promotion_report()
    report["candidate_pool_equivalence"][
        "maximum_current_topology_proxy_absolute_error"
    ] = float("nan")
    with pytest.raises(ValueError, match="serial equivalence differs"):
        v3.validate_promotion(
            report,
            _promotion_receipt("b" * 64),
            report_sha256="b" * 64,
            ledger_sha256=v3.UPDATED_LEDGER_SHA256,
        )


def test_v3_recomputes_every_promotion_receipt_artifact(tmp_path: Path) -> None:
    artifacts = {}
    for name in v3.RECEIPT_ARTIFACTS:
        path = tmp_path / name
        path.write_text(f"sealed {name}\n", encoding="utf-8")
        artifacts[name] = v3.sha256_file(path)
    assert v3.validate_receipt_artifact_files(tmp_path, artifacts) == artifacts

    (tmp_path / v3.RECEIPT_ARTIFACTS[-1]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash differs"):
        v3.validate_receipt_artifact_files(tmp_path, artifacts)


def test_v3_rejects_non_sha256_receipt_artifact_value() -> None:
    receipt = _promotion_receipt("b" * 64)
    receipt["artifact_sha256"][v3.RECEIPT_ARTIFACTS[-1]] = "z" * 64
    with pytest.raises(ValueError, match="receipt artifacts differ"):
        v3.validate_promotion(
            _promotion_report(),
            receipt,
            report_sha256="b" * 64,
            ledger_sha256=v3.UPDATED_LEDGER_SHA256,
        )


def test_v3_validates_complete_result_blind_prior_ledger() -> None:
    summary = v3.validate_ledger(_ledger())
    assert summary == {"entry_count": 35, "unique_speakers": 35}

    ledger = _ledger()
    ledger["target_scalar_values_used_for_selection"] = True
    with pytest.raises(ValueError, match="ledger boundary differs"):
        v3.validate_ledger(ledger)


def test_v3_component_readiness_keeps_joint_panel_and_training_closed() -> None:
    report = v3.build_readiness_report(
        {
            "head": "c" * 40,
            "branch": "feat/avqi-route-c-six-joint-runners-v1",
        },
        "d" * 64,
        "e" * 64,
        v3.UPDATED_LEDGER_SHA256,
        "f" * 64,
        {
            "preexact_gate_count": len(v3.PREEXACT_GATES),
            "mechanism_gate_count": 9,
            "integration_gate_count": 16,
            "receipt_artifact_count": len(v3.RECEIPT_ARTIFACTS),
        },
        {"entry_count": 35, "unique_speakers": 35},
    )
    assert report["decision"] == v3.COMPONENT_PASS
    assert report["shimmer_db_six_component_readiness_eligible"] is True
    assert report["historical_candidate_d_v23"]["remains_immutable"] is True
    assert report["historical_candidate_d_v23"]["reinterpreted_as_pass"] is False
    assert report["joint_panel_decision"] == v3.JOINT_NO_GO
    assert report["joint_panel_authorized"] is False
    assert report["unbound_joint_inputs"] == list(v3.UNBOUND_JOINT_INPUTS)
    assert report["generator_optimizer_steps"] == 0
    assert report["formal_generator_training_authorized"] is False


def test_v3_source_contains_no_training_or_exact_scoring_path() -> None:
    source = inspect.getsource(v3)
    assert "optimizer.step" not in source
    assert "torch.optim" not in source
    assert "parselmouth" not in source
    assert "run_exact" not in source
    assert "joint_panel_authorized\": True" not in source


def test_v3_shell_runner_is_executable_and_requires_every_binding() -> None:
    path = (
        Path(v3.__file__).resolve().parent
        / "run_avqi_route_c_six_joint_candidate_e_readiness_v3.sh"
    )
    assert path.stat().st_mode & stat.S_IXUSR
    source = path.read_text(encoding="utf-8")
    for flag in (
        "--contract",
        "--contract-sha256",
        "--promotion-report",
        "--promotion-report-sha256",
        "--promotion-receipt",
        "--promotion-receipt-sha256",
        "--speaker-ledger",
        "--speaker-ledger-sha256",
        "--source-root",
        "--source-commit",
        "--output-dir",
    ):
        assert flag in source
    assert (
        "scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v3"
        in source
    )
    assert "sbatch" not in source
    assert "parselmouth" not in source
    assert "optimizer" not in source


def test_v3_shell_runner_fails_closed_before_python_without_bindings() -> None:
    path = (
        Path(v3.__file__).resolve().parent
        / "run_avqi_route_c_six_joint_candidate_e_readiness_v3.sh"
    )
    environment = dict(os.environ)
    environment["RUNTIME_PYTHON"] = sys.executable
    completed = subprocess.run(
        [str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert "Missing required fail-closed argument: --contract" in completed.stderr
