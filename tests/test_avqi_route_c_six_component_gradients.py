from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import soundfile as sf
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_v19_contracts import (
    ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
    ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
    ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
    ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
    ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
    ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE,
    ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
    ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS,
    sha256_file,
    validate_v19_exact_topology,
)
from scripts import evaluate_avqi_route_c_six_component_gradients as audit
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    readiness_requirements,
)
from scripts.evaluate_avqi_route_c_multicomponent_gradients import (
    SEGMENT_SAMPLES,
    AuditCase,
    load_fixed_segment,
)
from scripts.evaluate_avqi_route_c_six_component_gradients import (
    JOINT_PANEL_DECISION,
    MEASUREMENT_DECISION,
    PAIRWISE_COMPONENT_KEYS,
    REQUIRED_FIVE_SOURCE_EVIDENCE,
    TOPOLOGY_INPUT_SCHEMA_VERSION,
    TopologyAuditInput,
    aggregate_measurements,
    calibration_inverse_gradient_weights,
    extract_case_measurement,
    finalize_case_measurement,
    load_topology_inputs,
    load_v19_evidence_manifest,
    slot_separation_metadata,
    validate_five_source_evidence,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _waveform_sha256(waveform: torch.Tensor) -> str:
    values = waveform.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def _synthetic_topology(
    waveform: torch.Tensor,
    *,
    case_id: str,
    view: str = "cs",
) -> tuple[dict[str, object], str]:
    ranges = [[0, waveform.numel()]]
    pulses = [100.0, 200.0, 300.0, 400.0]
    topology: dict[str, object] = {
        "case_id": case_id,
        "view": view,
        "role": "current_output_topology",
        "scoring_status": "ok",
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "implementation": ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
        "metric_highpass": ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
        "topology_input_loader": ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
        "source_waveform_float32_sha256": _waveform_sha256(waveform),
        "source_sample_count": waveform.numel(),
        "metric_sample_count": waveform.numel(),
        "metric_constant_prefix_samples": 0,
        "metric_source_ranges": ranges,
        "metric_source_range_count": 1,
        "metric_mapped_sample_count": waveform.numel(),
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": pulses,
        "pulse_count": len(pulses),
        "source_ranges_sha256": hashlib.sha256(
            json.dumps(ranges, separators=(",", ":")).encode()
        ).hexdigest(),
        "pulse_positions_sha256": hashlib.sha256(
            struct.pack(f"<{len(pulses)}d", *pulses)
        ).hexdigest(),
    }
    scalar_payload = {
        field: int(topology[field]) for field in ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            scalar_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(json.dumps(ranges, separators=(",", ":")).encode())
    digest.update(struct.pack(f"<{len(pulses)}d", *pulses))
    return topology, digest.hexdigest()


def _case_and_topology(tmp_path: Path) -> tuple[AuditCase, TopologyAuditInput]:
    audio_path = tmp_path / "current.wav"
    source = torch.linspace(0.05, 0.25, SEGMENT_SAMPLES, dtype=torch.float32)
    sf.write(audio_path, source.numpy(), 16_000, subtype="FLOAT")
    case = AuditCase(
        split="surrogate_calibration",
        speaker_id="speaker-a",
        sample_id="sample-a",
        sample_group="pathological_mild",
        view="cs",
        condition="aug16k_phone",
        waveform_path=audio_path,
        waveform_sha256=sha256_file(audio_path),
        clean_target=torch.zeros(6),
    )
    waveform = load_fixed_segment(case)
    topology, topology_sha256 = _synthetic_topology(
        waveform,
        case_id="gradient-case-a",
    )
    return case, TopologyAuditInput(
        case_id="gradient-case-a",
        topology=topology,
        topology_sha256=topology_sha256,
        source_waveform_float32_sha256=_waveform_sha256(waveform),
    )


class ContractCheckingSixScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("target_mean", torch.zeros(6))
        self.register_buffer("target_scale", torch.ones(6))

    def normalized_target(self, raw_target: torch.Tensor) -> torch.Tensor:
        return raw_target

    def denormalized_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

    def forward(
        self,
        waveform: torch.Tensor,
        speaking_type: str,
        *,
        topology: dict[str, object],
        case_id: str,
        view: str,
        topology_sha256: str,
    ) -> torch.Tensor:
        assert speaking_type == view
        validate_v19_exact_topology(
            waveform,
            topology,
            case_id=case_id,
            view=view,
            expected_topology_sha256=topology_sha256,
            sample_rate=16_000,
        )
        ramp = torch.linspace(0.5, 1.5, waveform.numel(), device=waveform.device)
        features = (
            waveform.mean(),
            waveform.square().mean(),
            waveform.pow(3).mean(),
            (waveform * ramp).mean(),
            torch.sin(waveform).mean(),
            torch.sqrt(waveform.square().mean() + 1e-8),
        )
        return torch.stack(features).unsqueeze(0)


def test_raw_collector_reports_six_norms_and_all_cosines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = torch.linspace(0.05, 0.25, 2_000, dtype=torch.float32)
    case = AuditCase(
        split="surrogate_calibration",
        speaker_id="speaker-a",
        sample_id="sample-a",
        sample_group="pathological_mild",
        view="cs",
        condition="aug16k_phone",
        waveform_path=tmp_path / "not-read.wav",
        waveform_sha256="a" * 64,
        clean_target=torch.zeros(6),
    )
    topology, topology_sha256 = _synthetic_topology(
        waveform,
        case_id="gradient-case-a",
    )
    topology_input = TopologyAuditInput(
        case_id="gradient-case-a",
        topology=topology,
        topology_sha256=topology_sha256,
        source_waveform_float32_sha256=_waveform_sha256(waveform),
    )
    monkeypatch.setattr(audit, "load_fixed_segment", lambda _: waveform.clone())
    monkeypatch.setattr(audit, "SEGMENT_SAMPLES", waveform.numel())
    raw = extract_case_measurement(
        ContractCheckingSixScorer(),
        case,
        topology_input,
        torch.device("cpu"),
    )
    medians, weights = calibration_inverse_gradient_weights([raw])
    finalized = finalize_case_measurement(raw, weights)
    summary = aggregate_measurements([finalized])

    assert set(medians) == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    assert set(finalized["components"]) == set(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    assert set(finalized["joint"]["pairwise_component_cosines"]) == set(
        PAIRWISE_COMPONENT_KEYS
    )
    assert len(PAIRWISE_COMPONENT_KEYS) == math.comb(6, 2)
    assert set(finalized["joint"]["component_to_joint_cosines"]) == set(
        ROUTE_C_SIX_ACTIVE_COMPONENTS
    )
    assert math.isclose(
        sum(finalized["joint"]["weighted_component_norm_shares"].values()),
        1.0,
        abs_tol=1e-6,
    )
    assert summary["component_gradient_measurements"] == 6
    assert summary["pairwise_cosine_measurements"] == 15
    assert summary["component_to_joint_cosine_measurements"] == 6
    assert finalized["topology"]["slot2_shimmer_percent_uses_topology"] is False
    assert finalized["topology"]["slot3_shimmer_db_uses_topology"] is True
    assert finalized["joint"]["scientific_gate_applied"] is False


def test_inverse_weights_use_only_supplied_calibration_records() -> None:
    records = [
        {
            "components": {
                component: {"gradient_norm": float(index + multiplier)}
                for index, component in enumerate(
                    ROUTE_C_SIX_ACTIVE_COMPONENTS,
                    start=1,
                )
            }
        }
        for multiplier in (0.0, 2.0)
    ]
    medians, weights = calibration_inverse_gradient_weights(records)
    weighted = {
        component: medians[component] * weights[component]
        for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
    }
    assert max(weighted.values()) == min(weighted.values())


def _topology_manifest(
    case: AuditCase,
    topology_input: TopologyAuditInput,
) -> dict[str, object]:
    return {
        "schema_version": TOPOLOGY_INPUT_SCHEMA_VERSION,
        "v19_source_commit": "a" * 40,
        "v19_evidence_manifest_sha256": "b" * 64,
        "label_bank_sha256": "c" * 64,
        "selection_salt": "selection-salt",
        "sample_rate": 16_000,
        "segment_samples": SEGMENT_SAMPLES,
        "topology_role": "base_current_output",
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "final_panel_opened": False,
        "fresh_panel_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "rows": [
            {
                "case_id": topology_input.case_id,
                "split": case.split,
                "speaker_id": case.speaker_id,
                "sample_id": case.sample_id,
                "sample_group": case.sample_group,
                "view": case.view,
                "condition": case.condition,
                "source_waveform_path": str(case.waveform_path.resolve()),
                "source_audio_file_sha256": case.waveform_sha256,
                "source_waveform_float32_sha256": (
                    topology_input.source_waveform_float32_sha256
                ),
                "source_segment_samples": SEGMENT_SAMPLES,
                "topology_sha256": topology_input.topology_sha256,
                "topology": topology_input.topology,
            }
        ],
    }


def test_topology_manifest_exactly_binds_selected_current_outputs(
    tmp_path: Path,
) -> None:
    case, topology_input = _case_and_topology(tmp_path)
    manifest_path = tmp_path / "topologies.json"
    manifest = _topology_manifest(case, topology_input)
    _write_json(manifest_path, manifest)
    inputs, coverage = load_topology_inputs(
        manifest_path,
        sha256_file(manifest_path),
        [case],
        v19_source_commit="a" * 40,
        v19_evidence_manifest_sha256="b" * 64,
        label_bank_sha256="c" * 64,
        selection_salt="selection-salt",
    )
    assert len(inputs) == 1
    assert coverage["exact_selection_coverage"] is True
    assert coverage["cases_by_split"] == {
        "surrogate_calibration": 1,
        "surrogate_holdout": 0,
    }

    manifest["rows"][0]["topology"]["metric_highpass"] = (
        ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not the base high-pass path"):
        load_topology_inputs(
            manifest_path,
            sha256_file(manifest_path),
            [case],
            v19_source_commit="a" * 40,
            v19_evidence_manifest_sha256="b" * 64,
            label_bank_sha256="c" * 64,
            selection_salt="selection-salt",
        )


def test_v19_manifest_loader_requires_exact_bound_artifact_sets(
    tmp_path: Path,
) -> None:
    def bindings(keys: tuple[str, ...]) -> dict[str, dict[str, str]]:
        result = {}
        for key in keys:
            path = tmp_path / f"{key}.artifact"
            path.write_text(key, encoding="utf-8")
            result[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        return result

    manifest = {
        "schema_version": ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
        "source_commit": "a" * 40,
        "slurm_job_id": "1",
        "decision": "PASS_SHIMMER_DB_RUNTIME_V19_FULL_STEP_INTEGRATION",
        "implementation_artifacts": bindings(
            ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS
        ),
        "full_step_artifacts": bindings(ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS),
        "candidate_exact_avqi_components_opened": False,
        "exact_component_scoring_requested": False,
        "opened24_rerun_authorized": True,
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
    }
    path = tmp_path / "v19_manifest.json"
    _write_json(path, manifest)
    loaded = load_v19_evidence_manifest(path, sha256_file(path))
    assert loaded.promotion_authorized is False
    assert set(loaded.implementation_artifacts) == set(
        ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS
    )

    manifest["implementation_artifacts"].pop("v19_worker")
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="artifact keys differ"):
        load_v19_evidence_manifest(path, sha256_file(path))


def test_five_source_evidence_is_semantically_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = []
    for key in REQUIRED_FIVE_SOURCE_EVIDENCE:
        path = tmp_path / key
        path.write_text(key, encoding="utf-8")
        entries.append([key, str(path.resolve()), sha256_file(path)])
    observed: dict[str, object] = {}

    def fake_validator(artifacts: object, paths: object) -> dict[str, str]:
        observed["artifacts"] = artifacts
        observed["paths"] = paths
        return {}

    monkeypatch.setattr(audit, "_validate_five_component_evidence", fake_validator)
    records = validate_five_source_evidence(entries)
    assert set(records) == set(REQUIRED_FIVE_SOURCE_EVIDENCE)
    assert observed["artifacts"] == records


def test_slot2_and_slot3_sources_are_explicitly_separate() -> None:
    metadata = slot_separation_metadata(
        {
            "shimmer_percent": {"component_indices": [2]},
            "shimmer_db": {
                "component_indices": [3],
                "checkpoint_affine_used": False,
                "source": "v19_current_output_exact_topology",
            },
        }
    )
    assert metadata["slots_are_independent"] is True
    assert metadata["slot2_shimmer_percent"]["v19_topology_used"] is False
    assert metadata["slot3_shimmer_db"]["checkpoint_affine_used"] is False


def test_launcher_is_fail_closed_and_has_no_submission_or_pass_path() -> None:
    launcher = Path(
        "scripts/run_avqi_route_c_six_component_gradient_audit.sh"
    ).resolve()
    source = launcher.read_text(encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("RUNTIME_PYTHON", None)
    completed = subprocess.run(
        [str(launcher)],
        cwd=launcher.parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "RUNTIME_PYTHON" in completed.stderr
    assert "sbatch" not in source
    assert "PASS_ROUTE_C_SIX_ACTIVE_CODE_GRADIENT_AUDIT" not in source
    assert "scripts.evaluate_avqi_route_c_six_component_gradients" in source
    assert "--source-evidence" in source
    assert "--v19-evidence-manifest" in source
    assert "--v19-evidence-manifest-sha256" in source
    assert "--topology-manifest" in source
    assert "--topology-manifest-sha256" in source
    assert MEASUREMENT_DECISION.startswith("PENDING_")
    assert JOINT_PANEL_DECISION.startswith("NO_GO_")
    assert sys.executable not in source


def test_collector_source_contains_no_scientific_threshold_or_experiment_path() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "PASS_ROUTE_C_SIX_ACTIVE_CODE_GRADIENT_AUDIT",
        "MAX_WEIGHTED_COMPONENT_NORM_SHARE",
        "INPUT_GRADIENT_NORM_MAX",
        "torch.optim",
        "parselmouth",
        "sbatch",
    ):
        assert forbidden not in source
    assert tuple(ROUTE_C_SIX_ACTIVE_COMPONENTS) == AVQI_COMPONENT_NAMES


def test_readiness_distinguishes_raw_collector_from_scientific_decision() -> None:
    requirement = next(
        row
        for row in readiness_requirements()["source_requirement_matrix"]
        if row["requirement"] == "six-component gradient evaluator/runner"
    )
    assert requirement["current_evidence"] == (
        "scripts.evaluate_avqi_route_c_six_component_gradients"
    )
    assert requirement["status"] == (
        "present_dev_only_measurement_scientific_decision_pending"
    )
