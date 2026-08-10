import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_dnf_phase_b_allowlist_registry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_dnf_phase_b_allowlist_registry",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def base_config(audit_dir: Path) -> dict:
    return {
        "schema_version": "dnf_phase_b_indoor_allowlist_registry_v1",
        "allowlist_id": "test-v1",
        "status": "review_allowlist_only_not_training_ready",
        "formal_submit_allowed": False,
        "source_audit": {
            "schema_version": "dnf_phase_b_source_audit_v2",
            "run_dir": str(audit_dir),
        },
        "noise_review_allowlist": {
            "candidate_file": "indoor_noise_candidates_pending.jsonl",
            "required_route": "audit_pending",
            "required_proposed_route": "noise_only",
            "required_training_ready": False,
            "allowed_datasets": ["DNS5_noise"],
            "allowed_selection_reasons": [
                "dns5_freesound_validated_fan"
            ],
            "manual_approval_required": True,
        },
        "variable_vehicle_exclusion": {
            "candidate_file": "variable_vehicle_event_candidates.jsonl",
            "required_route": "audit_pending",
            "required_proposed_route": "variable_vehicle_event_candidate",
            "required_training_ready": False,
            "must_be_disjoint_from_noise_review_allowlist": True,
            "prohibited_training_routes": [
                "noise_only",
                "primary_dnf_noise_component",
            ],
        },
        "rir_review_allowlist": {
            "candidate_file": "indoor_rir_candidates_pending.jsonl",
            "required_route": "audit_pending",
            "required_proposed_route": "rir",
            "required_training_ready": False,
            "allowed_datasets": ["Arni"],
            "operator_scope": "future_convolution_only_never_additive_eq13",
            "parameter_approval_required": True,
        },
        "review_evidence": {
            "noise_acoustic_review": (
                "indoor_asset_review/indoor_noise_acoustic_review.jsonl"
            ),
            "rir_parameter_review": (
                "indoor_asset_review/indoor_rir_decode_review.jsonl"
            ),
        },
        "approved_training_allowlists": {
            "noise_file": "approved_indoor_noise_v1.jsonl",
            "rir_file": "approved_indoor_rir_v1.jsonl",
            "initial_state": "empty_fail_closed",
            "training_ready": False,
            "promotion_requires_new_version": True,
        },
        "future_consumer_must_require": [
            "approved_status",
            "training_ready_true",
            "allowlist_version_match",
            "artifact_sha256_match",
            "route_specific_schema_match",
        ],
    }


def candidate(
    key: str,
    *,
    dataset: str,
    proposed_route: str,
    reason: str,
) -> dict:
    return {
        "key": key,
        "dataset": dataset,
        "selection_reason": reason,
        "route": "audit_pending",
        "proposed_route": proposed_route,
        "training_ready": False,
    }


def prepare_audit_dir(tmp_path: Path) -> tuple[Path, Path]:
    audit_dir = tmp_path / "audit"
    write_json(audit_dir / "audit_summary.json", {"training_ready": False})
    (audit_dir / "artifact_sha256.txt").write_text(
        "placeholder  audit_summary.json\n",
        encoding="utf-8",
    )
    write_jsonl(
        audit_dir / "indoor_noise_candidates_pending.jsonl",
        [
            candidate(
                "fan-1",
                dataset="DNS5_noise",
                proposed_route="noise_only",
                reason="dns5_freesound_validated_fan",
            )
        ],
    )
    write_jsonl(
        audit_dir / "variable_vehicle_event_candidates.jsonl",
        [
            candidate(
                "passby-1",
                dataset="FSD50K",
                proposed_route="variable_vehicle_event_candidate",
                reason="fsd50k_road_engine_fail_closed",
            )
        ],
    )
    write_jsonl(
        audit_dir / "indoor_rir_candidates_pending.jsonl",
        [
            candidate(
                "rir-1",
                dataset="Arni",
                proposed_route="rir",
                reason="arni_indoor_variable_acoustics_room",
            )
        ],
    )
    noise_review = candidate(
        "fan-1",
        dataset="DNS5_noise",
        proposed_route="noise_only",
        reason="dns5_freesound_validated_fan",
    )
    noise_review["automatic_gate"] = {"automatic_pass": True}
    rir_review = candidate(
        "rir-1",
        dataset="Arni",
        proposed_route="rir",
        reason="arni_indoor_variable_acoustics_room",
    )
    rir_review["automatic_gate"] = {"automatic_pass": True}
    write_jsonl(
        audit_dir
        / "indoor_asset_review"
        / "indoor_noise_acoustic_review.jsonl",
        [noise_review],
    )
    write_jsonl(
        audit_dir
        / "indoor_asset_review"
        / "indoor_rir_decode_review.jsonl",
        [rir_review],
    )
    config_path = tmp_path / "config.json"
    write_json(config_path, base_config(audit_dir))
    return audit_dir, config_path


def test_registry_is_versioned_and_fail_closed(tmp_path: Path) -> None:
    audit_dir, config_path = prepare_audit_dir(tmp_path)
    output_dir = tmp_path / "registry"
    registry = MODULE.build_registry(
        audit_dir=audit_dir,
        config_path=config_path,
        output_dir=output_dir,
    )
    assert registry["training_ready"] is False
    assert registry["review_allowlists"]["indoor_stable_noise"][
        "candidate_count"
    ] == 1
    assert registry["review_allowlists"]["indoor_rir"][
        "candidate_count"
    ] == 1
    assert registry["explicit_exclusions"]["variable_vehicle_events"][
        "candidate_count"
    ] == 1
    approved = registry["approved_training_allowlists"]
    assert approved["noise_count"] == 0
    assert approved["rir_count"] == 0
    assert (output_dir / approved["noise_file"]).read_text() == ""
    assert (output_dir / approved["rir_file"]).read_text() == ""
    assert (output_dir / "artifact_sha256.txt").is_file()


def test_registry_rejects_vehicle_overlap(tmp_path: Path) -> None:
    audit_dir, config_path = prepare_audit_dir(tmp_path)
    vehicle_path = audit_dir / "variable_vehicle_event_candidates.jsonl"
    rows = list(MODULE.iter_jsonl(vehicle_path))
    rows[0]["key"] = "fan-1"
    write_jsonl(vehicle_path, rows)
    with pytest.raises(ValueError, match="overlap"):
        MODULE.build_registry(
            audit_dir=audit_dir,
            config_path=config_path,
            output_dir=tmp_path / "registry",
        )


def test_registry_refuses_pending_training_ready_row(
    tmp_path: Path,
) -> None:
    audit_dir, config_path = prepare_audit_dir(tmp_path)
    noise_path = audit_dir / "indoor_noise_candidates_pending.jsonl"
    rows = list(MODULE.iter_jsonl(noise_path))
    rows[0]["training_ready"] = True
    write_jsonl(noise_path, rows)
    with pytest.raises(ValueError, match="training-ready pending row"):
        MODULE.build_registry(
            audit_dir=audit_dir,
            config_path=config_path,
            output_dir=tmp_path / "registry",
        )
