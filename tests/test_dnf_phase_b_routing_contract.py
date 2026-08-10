import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "train"
    / "dnf_source_routing_webdataset_v2_audit.json"
)


def test_route_definitions_are_fail_closed_and_indoor_scoped():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "audit_only_not_training_ready"
    assert not contract["formal_training_allowed"]
    assert contract["source_level_routes"]["clean_strict"] == [
        "EARS",
        "VCTK",
    ]
    assert "CommonVoice25" in contract["source_level_routes"][
        "noisy_speech_target"
    ]
    assert contract["indoor_noise_policy"][
        "source_level_noise_only_is_forbidden"
    ]
    assert "FMA" in contract["indoor_noise_policy"]["excluded_from_primary"]
    assert "car_passing_by" in contract["indoor_noise_policy"][
        "excluded_from_primary"
    ]
    assert "stable_vehicle_cabin_hum_if_later_found" in contract[
        "indoor_noise_policy"
    ]["primary_candidate_types"]
    assert not contract["candidate_overlays"]["MLS_HQ_en"]["training_ready"]
    assert not contract["candidate_overlays"]["LibriTTS_augmented"][
        "training_ready"
    ]


def test_low_confidence_speech_cannot_become_noise_only():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "low_confidence_speech_to_noise_only" in contract[
        "forbidden_transitions"
    ]
    assert "dnsmos_score_only_to_clean_strict" in contract[
        "forbidden_transitions"
    ]
    assert contract["consumer_contract"]["existing_v1_loader_compatible"] is False
