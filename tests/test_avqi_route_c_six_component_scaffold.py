from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest
import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    AVQI_V0301_COEFFICIENTS,
    AVQI_V0301_EXPANDED_COEFFICIENTS,
    ComponentAffineCalibrator,
    PraatDifferentiableAVQIComponentEstimator,
)
from model.avqi_route_c import (
    ROUTE_C_ACTIVE_COMPONENTS,
    ROUTE_C_COMPONENT_REGISTRY,
    ROUTE_C_FIVE_ACTIVE_COMPONENTS,
    ROUTE_C_FIVE_SOURCE_ARCHITECTURES,
    ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS,
    ROUTE_C_FIVE_SOURCE_COMPONENT_INDICES,
    ROUTE_C_REGISTRY_SCHEMA_VERSION,
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_COMPONENT_REGISTRY,
    ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES,
    ROUTE_C_SIX_REGISTRY_SCHEMA_VERSION,
    ROUTE_C_SIX_SCIENTIFIC_STATUS,
    ROUTE_C_SIX_SOURCE_ARCHITECTURES,
    ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS,
    ROUTE_C_SIX_SOURCE_COMPONENT_INDICES,
    ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
    ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
    ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
    ROUTE_C_V19_FULL_STEP_GATE_KEYS,
    ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS,
    ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS,
    ROUTE_C_V19_PASS_DECISION,
    ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE,
    ROUTE_C_V19_PAIRED_CANDIDATE_INPUT_LOADER,
    ROUTE_C_V19_RECEIPT_SCHEMA_VERSION,
    ROUTE_C_V19_REPORT_SCHEMA_VERSION,
    ROUTE_C_V19_SAMPLE_RATE,
    ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
    ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS,
    RouteCArtifactBinding,
    RouteCSixActiveScorer,
    RouteCV19EvidenceManifest,
    active_bidirectional_gap_losses,
    five_active_bidirectional_gap_losses,
    load_route_c_five_active_scorer,
    load_route_c_six_active_scorer,
    route_c_six_registry_records,
    sha256_file,
    six_active_bidirectional_gap_losses,
    validate_v19_evidence_manifest,
    validate_v19_exact_topology,
)


TEST_JOB_ID = "1"
TEST_SOURCE_COMMIT = hashlib.sha1(b"synthetic-v19-evidence").hexdigest()


class LightweightRouteCEstimator(PraatDifferentiableAVQIComponentEstimator):
    """Keep the six-slot wrapper test focused on the external Shimmer dB path."""

    def forward(
        self,
        waveform: torch.Tensor,
        speaking_type: str | None = None,
    ) -> torch.Tensor:
        del speaking_type
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mean = waveform.mean(dim=-1)
        return torch.stack(tuple(mean * (index + 1) for index in range(6)), dim=-1)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_evidence_manifest(tmp_path: Path) -> RouteCV19EvidenceManifest:
    implementation_bindings: dict[str, RouteCArtifactBinding] = {}
    implementation_hashes: dict[str, str] = {}
    for key in ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS:
        path = tmp_path / f"{key}.py"
        path.write_text(f"# synthetic {key}\n", encoding="utf-8")
        digest = sha256_file(path)
        implementation_bindings[key] = RouteCArtifactBinding(path, digest)
        implementation_hashes[key] = digest

    csv_bindings: dict[str, RouteCArtifactBinding] = {}
    for key in ("attempts_csv", "runtime_csv", "durable_csv"):
        path = tmp_path / f"{key}.csv"
        path.write_text("case_id,value\ncase-a,1\n", encoding="utf-8")
        csv_bindings[key] = RouteCArtifactBinding(path, sha256_file(path))

    report_path = tmp_path / "diagnostic_report.json"
    core = {
        "decision": ROUTE_C_V19_PASS_DECISION,
        "source_commit": TEST_SOURCE_COMMIT,
        "slurm_job_id": TEST_JOB_ID,
        "candidate_exact_avqi_components_opened": False,
        "opened24_rerun_authorized": True,
        "promotion_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    write_json(
        report_path,
        {
            **core,
            "schema_version": ROUTE_C_V19_REPORT_SCHEMA_VERSION,
            "exact_component_scoring_requested": False,
            "generator_loaded": False,
            "generator_optimizer_created": False,
            "source_sha256": implementation_hashes,
            "source_provenance": {
                "repository_head": TEST_SOURCE_COMMIT,
                "repository_tree_clean": True,
                "implementation_sha256": implementation_hashes,
            },
            "phase": "opened24_full_selector_step_integration_only",
            "dev_only": True,
            "repeat_count_per_case": 3,
            "case_count": 24,
            "speaker_count": 12,
            "scientific_gates_changed": False,
            "v18_artifacts_mutated": False,
            "v19_topology_artifacts_mutated": False,
            "new_sealed_panel_authorized": False,
            "gates": {
                key: True for key in ROUTE_C_V19_FULL_STEP_GATE_KEYS
            },
        },
    )
    report_binding = RouteCArtifactBinding(
        report_path,
        sha256_file(report_path),
    )
    receipt_path = tmp_path / "completion_receipt.json"
    artifact_sha256 = {
        report_path.name: report_binding.sha256,
        **{
            binding.path.name: binding.sha256
            for binding in csv_bindings.values()
        },
    }
    write_json(
        receipt_path,
        {
            **core,
            "schema_version": ROUTE_C_V19_RECEIPT_SCHEMA_VERSION,
            "source_provenance": {
                "repository_head": TEST_SOURCE_COMMIT,
                "repository_tree_clean": True,
                "implementation_sha256": implementation_hashes,
            },
            "new_sealed_panel_authorized": False,
            "artifact_sha256": artifact_sha256,
        },
    )
    full_step_bindings = {
        "report": report_binding,
        **csv_bindings,
        "receipt": RouteCArtifactBinding(
            receipt_path,
            sha256_file(receipt_path),
        ),
    }
    return RouteCV19EvidenceManifest(
        schema_version=ROUTE_C_V19_EVIDENCE_SCHEMA_VERSION,
        source_commit=TEST_SOURCE_COMMIT,
        slurm_job_id=TEST_JOB_ID,
        decision=ROUTE_C_V19_PASS_DECISION,
        implementation_artifacts=implementation_bindings,
        full_step_artifacts=full_step_bindings,
        candidate_exact_avqi_components_opened=False,
        exact_component_scoring_requested=False,
        opened24_rerun_authorized=True,
        promotion_authorized=False,
        generator_optimizer_steps=0,
    )


def replace_bound_report(
    manifest: RouteCV19EvidenceManifest,
    report: dict[str, object],
    receipt: dict[str, object] | None = None,
) -> RouteCV19EvidenceManifest:
    full_step = dict(manifest.full_step_artifacts)
    report_path = full_step["report"].path
    write_json(report_path, report)
    report_binding = RouteCArtifactBinding(report_path, sha256_file(report_path))
    full_step["report"] = report_binding

    receipt_path = full_step["receipt"].path
    if receipt is None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"][report_path.name] = report_binding.sha256
    write_json(receipt_path, receipt)
    full_step["receipt"] = RouteCArtifactBinding(
        receipt_path,
        sha256_file(receipt_path),
    )
    return replace(manifest, full_step_artifacts=full_step)


def checkpoint(
    path: Path,
    key: str,
    offset: float,
) -> tuple[Path, str]:
    torch.save(
        {
            "state_dict": {
                "alignment_scale": torch.arange(1.0, 7.0) + offset,
                "alignment_bias": torch.arange(11.0, 17.0) + offset,
            },
            "target_mean": torch.tensor([1.0, 2.0, 3.0, 0.25, 5.0, 6.0]),
            "target_scale": torch.tensor([1.0, 2.0, 3.0, 0.5, 5.0, 6.0]),
            "calibration_scale": torch.arange(21.0, 27.0) + offset,
            "calibration_bias": torch.arange(31.0, 37.0) + offset,
            "components": AVQI_COMPONENT_NAMES,
            "architecture": ROUTE_C_SIX_SOURCE_ARCHITECTURES[key],
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "optimizer_steps": 0,
            "speaking_type_required": key == "cpps",
        },
        path,
    )
    return path, sha256_file(path)


def make_six_bundle(tmp_path: Path):
    manifest = make_evidence_manifest(tmp_path)
    created = {
        key: checkpoint(tmp_path / f"{key}.pt", key, 100.0 * (index + 1))
        for index, key in enumerate(ROUTE_C_SIX_SOURCE_CHECKPOINT_KEYS)
    }
    bundle = load_route_c_six_active_scorer(
        {key: value[0] for key, value in created.items()},
        {key: value[1] for key, value in created.items()},
        v19_evidence_manifest=manifest,
        max_frames=32,
        cpps_max_frames=64,
        hnr_max_frames=64,
    )
    return bundle, manifest


def waveform_sha256(waveform: torch.Tensor) -> str:
    values = waveform.detach().to(dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def source_ranges_sha256(ranges: list[list[int]]) -> str:
    payload = json.dumps(ranges, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pulse_positions_sha256(pulses: list[float]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(pulses)}d", *pulses)).hexdigest()


def synthetic_topology(
    waveform: torch.Tensor,
    *,
    case_id: str,
    view: str,
    ranges: list[list[int]],
    prefix: int,
    pulses: list[float],
) -> tuple[dict[str, object], str]:
    mapped_count = sum(length for _, length in ranges)
    topology: dict[str, object] = {
        "case_id": case_id,
        "view": view,
        "role": "current_output_topology",
        "scoring_status": "ok",
        "topology_preprocessing": "exact_avqi_view_metric_waveform",
        "implementation": ROUTE_C_V19_TOPOLOGY_IMPLEMENTATION,
        "metric_highpass": ROUTE_C_V19_BASE_TOPOLOGY_HIGHPASS_MODE,
        "topology_input_loader": ROUTE_C_V19_BASE_TOPOLOGY_INPUT_LOADER,
        "source_waveform_float32_sha256": waveform_sha256(waveform),
        "source_sample_count": waveform.numel(),
        "metric_sample_count": prefix + mapped_count,
        "metric_constant_prefix_samples": prefix,
        "metric_source_ranges": ranges,
        "metric_source_range_count": len(ranges),
        "metric_mapped_sample_count": mapped_count,
        "metric_reconstruction_max_pcm16_error": 0,
        "metric_reconstruction_differing_samples": 0,
        "pulse_positions_samples": pulses,
        "pulse_count": len(pulses),
        "source_ranges_sha256": source_ranges_sha256(ranges),
        "pulse_positions_sha256": pulse_positions_sha256(pulses),
    }
    scalar_payload = {
        field: int(topology[field])
        for field in ROUTE_C_V19_TOPOLOGY_SCALAR_FIELDS
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            scalar_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(json.dumps(ranges, separators=(",", ":")).encode("utf-8"))
    digest.update(struct.pack(f"<{len(pulses)}d", *pulses))
    return topology, digest.hexdigest()


def test_six_registry_preserves_four_five_and_keeps_promotion_pending() -> None:
    assert ROUTE_C_ACTIVE_COMPONENTS == (
        "cpps",
        "hnr",
        "shimmer_percent",
        "tilt",
    )
    assert ROUTE_C_FIVE_ACTIVE_COMPONENTS == (
        "cpps",
        "hnr",
        "shimmer_percent",
        "slope",
        "tilt",
    )
    assert ROUTE_C_SIX_ACTIVE_COMPONENTS == AVQI_COMPONENT_NAMES
    assert ROUTE_C_REGISTRY_SCHEMA_VERSION == "avqi-route-c-component-registry-v2"
    assert ROUTE_C_SIX_REGISTRY_SCHEMA_VERSION.endswith("scaffold-registry-v1")
    assert tuple(slot.name for slot in ROUTE_C_COMPONENT_REGISTRY) == (
        AVQI_COMPONENT_NAMES
    )
    assert tuple(slot.name for slot in ROUTE_C_SIX_COMPONENT_REGISTRY) == (
        AVQI_COMPONENT_NAMES
    )
    assert tuple(
        slot.avqi_coefficient for slot in ROUTE_C_SIX_COMPONENT_REGISTRY
    ) == AVQI_V0301_COEFFICIENTS
    assert tuple(
        slot.expanded_avqi_coefficient
        for slot in ROUTE_C_SIX_COMPONENT_REGISTRY
    ) == AVQI_V0301_EXPANDED_COEFFICIENTS
    shimmer_db = route_c_six_registry_records()[3]
    assert shimmer_db["active_in_six_component_scorer"] is True
    assert shimmer_db["scientific_status"] == ROUTE_C_SIX_SCIENTIFIC_STATUS
    assert shimmer_db["code_status"] == "fail_closed_scaffold"


def test_six_slot_mapping_separates_percent_from_external_db() -> None:
    assert ROUTE_C_SIX_SOURCE_COMPONENT_INDICES["shimmer_percent"] == (2,)
    assert ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES == {"shimmer_db": (3,)}
    checkpoint_slots = {
        index
        for indices in ROUTE_C_SIX_SOURCE_COMPONENT_INDICES.values()
        for index in indices
    }
    external_slots = set(ROUTE_C_SIX_EXTERNAL_COMPONENT_INDICES["shimmer_db"])
    assert checkpoint_slots.isdisjoint(external_slots)
    assert checkpoint_slots | external_slots == set(range(6))
    assert ROUTE_C_SIX_SOURCE_ARCHITECTURES is not (
        ROUTE_C_FIVE_SOURCE_ARCHITECTURES
    )
    assert ROUTE_C_SIX_SOURCE_ARCHITECTURES == ROUTE_C_FIVE_SOURCE_ARCHITECTURES


def test_v19_evidence_manifest_is_hash_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = make_evidence_manifest(tmp_path)
    metadata = validate_v19_evidence_manifest(manifest)
    assert metadata["opened24_rerun_authorized"] is True
    assert metadata["promotion_authorized"] is False
    assert metadata["scientific_promotion_granted"] is False
    assert metadata["generator_optimizer_steps"] == 0

    missing = dict(manifest.implementation_artifacts)
    missing.pop("v19_worker")
    with pytest.raises(ValueError, match="evidence keys"):
        validate_v19_evidence_manifest(
            replace(manifest, implementation_artifacts=missing)
        )
    fake = dict(manifest.full_step_artifacts)
    fake["report"] = replace(fake["report"], sha256="0" * 64)
    with pytest.raises(ValueError, match="bound SHA-256"):
        validate_v19_evidence_manifest(replace(manifest, full_step_artifacts=fake))
    with pytest.raises(ValueError, match="frozen PASS decision"):
        validate_v19_evidence_manifest(
            replace(manifest, decision="TEST_ONLY_DECISION")
        )
    with pytest.raises(ValueError, match="opened exact"):
        validate_v19_evidence_manifest(
            replace(manifest, candidate_exact_avqi_components_opened=True)
        )
    with pytest.raises(ValueError, match="optimizer steps"):
        validate_v19_evidence_manifest(
            replace(manifest, generator_optimizer_steps=1)
        )

    false_gate_dir = tmp_path / "false_gate"
    false_gate_dir.mkdir()
    false_gate_manifest = make_evidence_manifest(false_gate_dir)
    false_gate_report = json.loads(
        false_gate_manifest.full_step_artifacts["report"].path.read_text(
            encoding="utf-8"
        )
    )
    false_gate_report["gates"][ROUTE_C_V19_FULL_STEP_GATE_KEYS[0]] = False
    false_gate_manifest = replace_bound_report(
        false_gate_manifest,
        false_gate_report,
    )
    with pytest.raises(ValueError, match="gate did not pass"):
        validate_v19_evidence_manifest(false_gate_manifest)

    missing_gate_dir = tmp_path / "missing_gate"
    missing_gate_dir.mkdir()
    missing_gate_manifest = make_evidence_manifest(missing_gate_dir)
    missing_gate_report = json.loads(
        missing_gate_manifest.full_step_artifacts["report"].path.read_text(
            encoding="utf-8"
        )
    )
    missing_gate_report["gates"].pop(ROUTE_C_V19_FULL_STEP_GATE_KEYS[0])
    missing_gate_manifest = replace_bound_report(
        missing_gate_manifest,
        missing_gate_report,
    )
    with pytest.raises(ValueError, match="gate keys differ"):
        validate_v19_evidence_manifest(missing_gate_manifest)

    missing_source_dir = tmp_path / "missing_implementation_source"
    missing_source_dir.mkdir()
    missing_source_manifest = make_evidence_manifest(missing_source_dir)
    missing_source_report = json.loads(
        missing_source_manifest.full_step_artifacts["report"].path.read_text(
            encoding="utf-8"
        )
    )
    missing_source_receipt = json.loads(
        missing_source_manifest.full_step_artifacts["receipt"].path.read_text(
            encoding="utf-8"
        )
    )
    missing_key = "peak_certificate_helper"
    missing_source_report["source_provenance"]["implementation_sha256"].pop(
        missing_key
    )
    missing_source_receipt["source_provenance"]["implementation_sha256"].pop(
        missing_key
    )
    missing_source_manifest = replace_bound_report(
        missing_source_manifest,
        missing_source_report,
        missing_source_receipt,
    )
    with pytest.raises(ValueError, match="implementation keys differ"):
        validate_v19_evidence_manifest(missing_source_manifest)


def test_five_active_loader_keeps_default_composer_and_both_shimmer_slots(
    tmp_path: Path,
) -> None:
    created = {
        key: checkpoint(tmp_path / f"five_{key}.pt", key, 100.0 * (index + 1))
        for index, key in enumerate(ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS)
    }
    bundle = load_route_c_five_active_scorer(
        {key: value[0] for key, value in created.items()},
        {key: value[1] for key, value in created.items()},
        max_frames=32,
        cpps_max_frames=64,
        hnr_max_frames=64,
    )
    assert bundle.source_metadata["shimmer_percent"]["component_indices"] == [
        2,
        3,
    ]
    shimmer_offset = 100.0 * (
        ROUTE_C_FIVE_SOURCE_CHECKPOINT_KEYS.index("shimmer_percent") + 1
    )
    for index in ROUTE_C_FIVE_SOURCE_COMPONENT_INDICES["shimmer_percent"]:
        assert bundle.scorer.estimator.alignment_scale[index] == (
            index + 1 + shimmer_offset
        )
        assert bundle.scorer.calibrator.scale[index] == (
            index + 21 + shimmer_offset
        )
    assert "shimmer_db" not in bundle.source_metadata
    assert sum(parameter.numel() for parameter in bundle.scorer.parameters()) == 0


def test_six_bundle_neutralizes_slot_three_checkpoint_affine(
    tmp_path: Path,
) -> None:
    bundle, _ = make_six_bundle(tmp_path)
    scorer = bundle.scorer
    assert scorer.estimator.alignment_scale[3] == 1.0
    assert scorer.estimator.alignment_bias[3] == 0.0
    assert scorer.calibrator.scale[3] == 1.0
    assert scorer.calibrator.bias[3] == 0.0
    assert bundle.source_metadata["shimmer_percent"]["component_indices"] == [2]
    assert bundle.source_metadata["shimmer_db"] == {
        "source": "v19_current_output_exact_topology",
        "component_indices": [3],
        "checkpoint_affine_used": False,
        "scientific_status": ROUTE_C_SIX_SCIENTIFIC_STATUS,
        "scientific_promotion_granted": False,
        "optimizer_steps": 0,
    }
    assert bundle.scientific_status == ROUTE_C_SIX_SCIENTIFIC_STATUS
    assert bundle.generator_optimizer_steps == 0
    assert bundle.v19_evidence_metadata["scientific_promotion_granted"] is False
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 0


def test_topology_binding_covers_cs_indices_prefix_and_sv_final_crop() -> None:
    cs_time = torch.arange(800, dtype=torch.float32) / 16_000
    cs_waveform = torch.sin(2.0 * math.pi * 180.0 * cs_time) * (
        1.0 + 0.1 * torch.sin(2.0 * math.pi * 4.0 * cs_time)
    )
    cs_ranges = [[100, 300], [500, 200]]
    cs_pulses = [80.0, 170.0, 260.0, 350.0, 440.0]
    cs_topology, cs_hash = synthetic_topology(
        cs_waveform,
        case_id="case-cs",
        view="cs",
        ranges=cs_ranges,
        prefix=16,
        pulses=cs_pulses,
    )
    validated_cs = validate_v19_exact_topology(
        cs_waveform,
        cs_topology,
        case_id="case-cs",
        view="cs",
        expected_topology_sha256=cs_hash,
        sample_rate=ROUTE_C_V19_SAMPLE_RATE,
    )
    assert validated_cs.metric_constant_prefix_samples == 16
    assert validated_cs.metric_source_indices == tuple(
        [*range(100, 400), *range(500, 700)]
    )
    with pytest.raises(ValueError, match="requires a 16000 Hz sample rate"):
        validate_v19_exact_topology(
            cs_waveform,
            cs_topology,
            case_id="case-cs",
            view="cs",
            expected_topology_sha256=cs_hash,
            sample_rate=8_000,
        )
    estimator = PraatDifferentiableAVQIComponentEstimator(peak_mode="hard")
    source_indices = torch.tensor(validated_cs.metric_source_indices)
    prepared = estimator._prepare(cs_waveform).index_select(0, source_indices)
    prepared = torch.cat((prepared.new_zeros(16), prepared))
    pulse_tensor = cs_waveform.new_tensor(cs_pulses)
    expected = torch.stack(
        estimator._praat_fixed_pulse_shimmer(prepared, pulse_tensor)
    )
    actual = estimator.raw_shimmer_from_pulse_positions(
        cs_waveform,
        pulse_tensor,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=16,
    )
    assert torch.equal(actual, expected)

    sv_waveform = torch.linspace(-0.1, 0.1, 50_000, dtype=torch.float32)
    sv_topology, sv_hash = synthetic_topology(
        sv_waveform,
        case_id="case-sv",
        view="sv",
        ranges=[[2_000, 48_000]],
        prefix=0,
        pulses=[100.0, 200.0, 300.0],
    )
    validated_sv = validate_v19_exact_topology(
        sv_waveform,
        sv_topology,
        case_id="case-sv",
        view="sv",
        expected_topology_sha256=sv_hash,
        sample_rate=16_000,
    )
    assert validated_sv.metric_source_indices[0] == 2_000
    assert validated_sv.metric_source_indices[-1] == 49_999

    wrong_sv, wrong_sv_hash = synthetic_topology(
        sv_waveform,
        case_id="case-sv",
        view="sv",
        ranges=[[0, 48_000]],
        prefix=0,
        pulses=[100.0, 200.0, 300.0],
    )
    with pytest.raises(ValueError, match="final crop"):
        validate_v19_exact_topology(
            sv_waveform,
            wrong_sv,
            case_id="case-sv",
            view="sv",
            expected_topology_sha256=wrong_sv_hash,
            sample_rate=16_000,
        )


def test_six_scorer_rejects_missing_stale_or_cross_case_topology(
    tmp_path: Path,
) -> None:
    bundle, _ = make_six_bundle(tmp_path)
    waveform = torch.sin(torch.arange(4_000, dtype=torch.float32) / 8.0)
    topology, topology_hash = synthetic_topology(
        waveform,
        case_id="case-a",
        view="cs",
        ranges=[[0, waveform.numel()]],
        prefix=0,
        pulses=[100.0, 190.0, 280.0, 370.0],
    )
    with pytest.raises(ValueError, match="requires v19 topology"):
        bundle.scorer(waveform, "cs")
    for field in (
        "pulse_positions_samples",
        "metric_source_ranges",
        "metric_constant_prefix_samples",
        "implementation",
        "metric_highpass",
        "topology_input_loader",
    ):
        incomplete = dict(topology)
        incomplete.pop(field)
        with pytest.raises(ValueError, match="fields are missing"):
            bundle.scorer(
                waveform,
                "cs",
                topology=incomplete,
                case_id="case-a",
                view="cs",
                topology_sha256=topology_hash,
            )
    producer_contracts = (
        ("role", "clean_target_topology", "not from a current output"),
        ("implementation", "wrong-v19-producer", "implementation differs"),
        ("metric_highpass", "wrong-high-pass", "base topology high-pass"),
        ("topology_input_loader", "wrong-loader", "base topology input loader"),
    )
    for field, wrong_value, message in producer_contracts:
        wrong_producer = dict(topology)
        wrong_producer[field] = wrong_value
        with pytest.raises(ValueError, match=message):
            bundle.scorer(
                waveform,
                "cs",
                topology=wrong_producer,
                case_id="case-a",
                view="cs",
                topology_sha256=topology_hash,
            )
    paired_candidate_topology = dict(topology)
    paired_candidate_topology["metric_highpass"] = (
        ROUTE_C_V19_PAIRED_CANDIDATE_HIGHPASS_MODE
    )
    paired_candidate_topology["topology_input_loader"] = (
        ROUTE_C_V19_PAIRED_CANDIDATE_INPUT_LOADER
    )
    with pytest.raises(ValueError, match="base topology high-pass"):
        bundle.scorer(
            waveform,
            "cs",
            topology=paired_candidate_topology,
            case_id="case-a",
            view="cs",
            topology_sha256=topology_hash,
        )
    with pytest.raises(ValueError, match="case binding"):
        bundle.scorer(
            waveform,
            "cs",
            topology=topology,
            case_id="case-b",
            view="cs",
            topology_sha256=topology_hash,
        )
    with pytest.raises(ValueError, match="view binding"):
        bundle.scorer(
            waveform,
            "sv",
            topology=topology,
            case_id="case-a",
            view="sv",
            topology_sha256=topology_hash,
        )
    stale_waveform = waveform.clone()
    stale_waveform[0] += 0.01
    with pytest.raises(ValueError, match="current waveform"):
        bundle.scorer(
            stale_waveform,
            "cs",
            topology=topology,
            case_id="case-a",
            view="cs",
            topology_sha256=topology_hash,
        )
    with pytest.raises(ValueError, match="composite topology hash"):
        bundle.scorer(
            waveform,
            "cs",
            topology=topology,
            case_id="case-a",
            view="cs",
            topology_sha256=hashlib.sha256(b"wrong-topology").hexdigest(),
        )


def test_synthetic_v19_slot_three_gradient_is_finite_nonzero_and_bounded() -> None:
    sample_rate = ROUTE_C_V19_SAMPLE_RATE
    estimator = LightweightRouteCEstimator(
        peak_mode="hard",
        sample_rate=sample_rate,
    )
    scorer = RouteCSixActiveScorer(
        estimator,
        ComponentAffineCalibrator(
            torch.tensor([1.0, 1.0, 1.0, 123.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 456.0, 0.0, 0.0]),
        ),
        torch.tensor([1.0, 2.0, 3.0, 0.25, 5.0, 6.0]),
        torch.tensor([1.0, 2.0, 3.0, 0.5, 5.0, 6.0]),
    )
    time = torch.arange(2_000, dtype=torch.float32) / sample_rate
    waveform = (
        torch.sin(2.0 * math.pi * 180.0 * time)
        * (1.0 + 0.18 * torch.sin(2.0 * math.pi * 4.0 * time))
    ).requires_grad_()
    period = sample_rate / 180.0
    pulses = []
    position = 100.0
    while position < waveform.numel() - 100:
        pulses.append(position)
        position += period
    topology, topology_hash = synthetic_topology(
        waveform,
        case_id="gradient-case",
        view="cs",
        ranges=[[0, waveform.numel()]],
        prefix=0,
        pulses=pulses,
    )
    base_prediction = scorer.calibrator(
        scorer.estimator(waveform.unsqueeze(0), "cs")
    )
    prediction = scorer(
        waveform,
        "cs",
        topology=topology,
        case_id="gradient-case",
        view="cs",
        topology_sha256=topology_hash,
    )
    raw_shimmer_db = scorer.estimator.raw_shimmer_from_pulse_positions(
        waveform,
        waveform.new_tensor(pulses),
        metric_source_indices=torch.arange(waveform.numel()),
        metric_constant_prefix_samples=0,
    )[1]
    expected_slot_three = (
        raw_shimmer_db - scorer.target_mean[3]
    ) / scorer.target_scale[3]
    assert torch.equal(prediction[:, 2], base_prediction[:, 2])
    assert torch.equal(prediction[0, 3], expected_slot_three)
    assert not torch.equal(prediction[:, 3], base_prediction[:, 3])

    raw_target = scorer.denormalized_prediction(prediction.detach()).clone()
    raw_target[0, 3] += 0.25 * scorer.target_scale[3]
    slot_three_loss = six_active_bidirectional_gap_losses(
        prediction,
        raw_target,
        scorer.target_mean,
        scorer.target_scale,
    )[0, 3]
    gradient = torch.autograd.grad(slot_three_loss, waveform)[0]
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(gradient).all()
    assert 0.0 < gradient_norm <= 1e4


def test_six_loss_adds_db_without_changing_four_or_five_semantics() -> None:
    prediction = torch.tensor(
        [[2.0, -3.0, 4.0, -5.0, 6.0, -7.0]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    four = active_bidirectional_gap_losses(
        prediction,
        target,
        torch.zeros(6),
        torch.ones(6),
    )
    five = five_active_bidirectional_gap_losses(
        prediction,
        target,
        torch.zeros(6),
        torch.ones(6),
    )
    six = six_active_bidirectional_gap_losses(
        prediction,
        target,
        torch.zeros(6),
        torch.ones(6),
    )
    assert four.shape == (1, 4)
    assert five.shape == (1, 5)
    assert six.shape == (1, 6)
    five_indices = [
        AVQI_COMPONENT_NAMES.index(name)
        for name in ROUTE_C_FIVE_ACTIVE_COMPONENTS
    ]
    assert torch.equal(
        five,
        six[:, five_indices],
    )
    six.sum().backward()
    assert prediction.grad is not None
    assert torch.equal(
        prediction.grad.sign(),
        prediction.detach().sign(),
    )
    assert AVQI_V0301_COEFFICIENTS[0] < 0.0
    assert prediction.grad[0, 0] > 0.0


def test_manifest_contract_names_every_required_future_artifact() -> None:
    assert ROUTE_C_V19_IMPLEMENTATION_ARTIFACT_KEYS == (
        "peak_certificate_helper",
        "phase1_evaluator",
        "frozen_worker",
        "v19_worker",
        "v19_runtime_client",
        "integration_evaluator",
        "integration_runner",
    )
    assert ROUTE_C_V19_FULL_STEP_ARTIFACT_KEYS == (
        "report",
        "attempts_csv",
        "runtime_csv",
        "durable_csv",
        "receipt",
    )
