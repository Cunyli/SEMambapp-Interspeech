from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_dnf_phase_b_sources import (
    build_libri_candidate_rows,
    build_mls_probe_rows,
    candidate_pending_row,
    classify_libri_provenance,
    deterministic_select,
    dns_fan_allow_reason,
    fsd_allow_reason,
    fsd_proposed_route,
    libri_content_group_id,
    rir_allow_reason,
    write_jsonl,
)


def make_libri_item(
    *,
    split: str,
    source_path: str,
    shard: str,
    audio_member: str,
) -> dict[str, object]:
    return {
        "dataset": "LibriTTS_augmented_chunk0000",
        "_split": split,
        "_shard_dir": "/active/v1_libritts_augmented_chunk0000/clean",
        "shard": shard,
        "audio_member": audio_member,
        "json_member": str(Path(audio_member).with_suffix(".json")),
        "key": Path(audio_member).stem,
        "source_path": source_path,
    }


def make_mls_item(
    shard_dir: str,
    shard: str,
    index: int,
) -> dict[str, object]:
    key = f"clean_{shard}_{index:04d}"
    return {
        "dataset": "MLS_HQ_en_chunk0000",
        "_split": "train",
        "_shard_dir": shard_dir,
        "shard": shard,
        "audio_member": f"{key}.flac",
        "json_member": f"{key}.json",
        "key": key,
        "source_path": f"/mls/speaker/book/{key}.flac",
    }


def test_libri_provenance_and_ttv_group_normalization() -> None:
    data_wav = "/root/LibriTTS/data_wav_24k_10-folders/1/12_34_000001_000002.wav"
    g711 = "/root/LibriTTS/simulated-data/1/g711/12_34_000001_000002.wav"
    ttv = (
        "/root/LibriTTS/train-test-validation/validation/"
        "12_34_000001_000002_7.wav"
    )

    assert classify_libri_provenance(data_wav) == "data_wav_24k"
    assert classify_libri_provenance(g711) == "g711"
    assert classify_libri_provenance(ttv) == "train_test_validation"
    assert libri_content_group_id(data_wav) == "12_34_000001_000002"
    assert libri_content_group_id(ttv) == "12_34_000001_000002"


def test_libri_candidates_are_train_only_unique_and_pending() -> None:
    items = [
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/data_wav_24k_10-folders/1/"
                "a_1_000001_000001.wav"
            ),
            shard="clean-000001.tar",
            audio_member="a.wav",
        ),
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/simulated-data/1/original/"
                "a_1_000001_000001.wav"
            ),
            shard="clean-000002.tar",
            audio_member="a_original.wav",
        ),
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/data_wav_24k_10-folders/1/"
                "b_1_000001_000001.wav"
            ),
            shard="clean-000003.tar",
            audio_member="b.wav",
        ),
        make_libri_item(
            split="valid",
            source_path=(
                "/root/LibriTTS/simulated-data/1/g711/"
                "b_1_000001_000001.wav"
            ),
            shard="clean-000004.tar",
            audio_member="b_g711.wav",
        ),
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/data_wav_24k_10-folders/1/"
                "c_1_000001_000001.wav"
            ),
            shard="clean-000005.tar",
            audio_member="c.wav",
        ),
        make_libri_item(
            split="test",
            source_path=(
                "/root/LibriTTS/train-test-validation/test/"
                "c_1_000001_000001_9.wav"
            ),
            shard="clean-000006.tar",
            audio_member="c_ttv.wav",
        ),
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/data_wav_24k_10-folders/2/"
                "d_1_000001_000001.wav"
            ),
            shard="clean-000008.tar",
            audio_member="z_d.wav",
        ),
        make_libri_item(
            split="train",
            source_path=(
                "/root/LibriTTS/data_wav_24k_10-folders/1/"
                "d_1_000001_000001.wav"
            ),
            shard="clean-000007.tar",
            audio_member="a_d.wav",
        ),
    ]

    candidates, summary = build_libri_candidate_rows(items)

    assert [row["content_group_id"] for row in candidates] == [
        "a_1_000001_000001",
        "d_1_000001_000001",
    ]
    assert candidates[1]["source_path"].endswith(
        "/1/d_1_000001_000001.wav"
    )
    assert all(row["route"] == "clean_candidate" for row in candidates)
    assert all(row["audit_status"] == "audit_pending_scores" for row in candidates)
    assert all(row["training_ready"] is False for row in candidates)
    assert summary["cross_split_content_groups"] == 2
    assert summary["train_data_wav_duplicate_groups"] == 1
    assert summary["provenance_by_split"]["test"]["train_test_validation"] == 1


def test_deterministic_select_is_stable_and_fail_closed() -> None:
    rows = [
        {
            "dataset": "x",
            "_shard_dir": "/x",
            "shard": "s.tar",
            "audio_member": f"{index}.wav",
            "key": str(index),
        }
        for index in range(20)
    ]

    first = deterministic_select(rows, 5, 3407, "probe")
    second = deterministic_select(list(reversed(rows)), 5, 3407, "probe")

    assert [row["key"] for row in first] == [row["key"] for row in second]
    with pytest.raises(ValueError, match="Insufficient rows"):
        deterministic_select(rows, 21, 3407, "probe")


def test_mls_probe_is_exact_per_shard_and_pending() -> None:
    shard_dir = "/active/v1_mls_hq_en_clean_chunk0000/clean"
    selected_shards = [
        {
            "_shard_dir": shard_dir,
            "shard": "clean-000001.tar",
            "_split": "train",
        },
        {
            "_shard_dir": shard_dir,
            "shard": "clean-000002.tar",
            "_split": "train",
        },
    ]
    items = [
        make_mls_item(shard_dir, shard, index)
        for shard in ("clean-000001.tar", "clean-000002.tar")
        for index in range(5)
    ]

    probe = build_mls_probe_rows(selected_shards, items, 2, 3407)

    assert len(probe) == 4
    assert {
        shard: sum(row["shard"] == shard for row in probe)
        for shard in ("clean-000001.tar", "clean-000002.tar")
    } == {"clean-000001.tar": 2, "clean-000002.tar": 2}
    assert all(row["audit_status"] == "audit_pending_scores" for row in probe)
    assert all(row["training_ready"] is False for row in probe)


def test_fsd_rule_requires_target_context_and_rejects_contamination() -> None:
    assert fsd_allow_reason({"Mechanical_fan", "Mechanisms"}) is not None
    assert (
        fsd_allow_reason(
            {"Engine", "Idling", "Motor_vehicle_(road)", "Vehicle"}
        )
        == "fsd50k_road_engine_fail_closed"
    )
    assert (
        fsd_allow_reason(
            {
                "Traffic_noise_and_roadway_noise",
                "Motor_vehicle_(road)",
                "Vehicle",
            }
        )
        == "fsd50k_roadway_noise_fail_closed"
    )
    assert fsd_allow_reason({"Engine", "Vehicle"}) is None
    assert fsd_allow_reason({"Mechanical_fan", "Human_voice"}) is None
    assert fsd_allow_reason({"Engine", "Motor_vehicle_(road)", "Music"}) is None


def test_fsd_routes_vehicle_events_away_from_noise_only() -> None:
    fan_reason = fsd_allow_reason({"Mechanical_fan", "Mechanisms"})
    road_reason = fsd_allow_reason(
        {"Car_passing_by", "Car", "Motor_vehicle_(road)"}
    )
    engine_reason = fsd_allow_reason(
        {"Engine", "Idling", "Motor_vehicle_(road)"}
    )
    fan_and_road_reason = fsd_allow_reason(
        {"Mechanical_fan", "Car_passing_by", "Motor_vehicle_(road)"}
    )

    assert fan_reason == "fsd50k_mechanical_fan_fail_closed"
    assert fsd_proposed_route(fan_reason) == "noise_only"
    for reason in (road_reason, engine_reason, fan_and_road_reason):
        assert reason is not None
        assert fsd_proposed_route(reason) == "variable_vehicle_event_candidate"


def test_dns_fan_rule_is_path_and_dataset_limited() -> None:
    allowed = {
        "dataset": "DNS5_noise",
        "source_path": (
            "/root/datasets_fullband.noise_fullband.freesound_000/"
            "datasets_fullband/noise_fullband/"
            "fan_Freesound_validated_102686_0.wav"
        ),
    }
    wrong_label = {
        **allowed,
        "source_path": allowed["source_path"].replace("fan_", "door_"),
    }
    wrong_dataset = {**allowed, "dataset": "FSD50K"}

    assert dns_fan_allow_reason(allowed) == "dns5_freesound_validated_fan"
    assert dns_fan_allow_reason(wrong_label) is None
    assert dns_fan_allow_reason(wrong_dataset) is None


def test_rir_rule_requires_dataset_role_and_exact_path_contract() -> None:
    arni = {
        "dataset": "Arni",
        "role": "rir",
        "source_path": "/root/IR_numClosed_2_numComb_17_mic_3_sweep_4.wav",
    }
    slr26 = {
        "dataset": "DNS5_RIR_SLR26",
        "role": "rir",
        "source_member": (
            "datasets_fullband/impulse_responses/SLR26/"
            "simulated_rirs_48k/smallroom/Room149/Room149-00003.wav"
        ),
    }
    slr28 = {
        "dataset": "DNS5_RIR_SLR28",
        "role": "rir",
        "source_member": (
            "datasets_fullband/impulse_responses/SLR28/RIRS_NOISES/"
            "real_rirs_isotropic_noises/"
            "air_type1_air_binaural_office_0_1.wav"
        ),
    }

    assert rir_allow_reason(arni) == "arni_indoor_variable_acoustics_room"
    assert rir_allow_reason(slr26) == "slr26_simulated_indoor_room"
    assert rir_allow_reason(slr28) == "slr28_real_indoor_room_rir"
    assert rir_allow_reason({**slr26, "role": "noise"}) is None
    assert rir_allow_reason(
        {
            **slr28,
            "source_member": (
                "datasets_fullband/impulse_responses/SLR28/"
                "RIRS_NOISES/isotropic_noises/noise.wav"
            ),
        }
    ) is None


def test_jsonl_output_is_canonical_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"z": 1, "a": 2}, {"route": "clean_candidate"}]

    assert write_jsonl(path, rows) == 2
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw.splitlines()[0]) == {"a": 2, "z": 1}


def test_candidate_helper_cannot_mark_training_ready() -> None:
    row = candidate_pending_row(
        {
            "dataset": "x",
            "_split": "train",
            "_shard_dir": "/x",
            "shard": "s.tar",
            "audio_member": "a.wav",
            "key": "a",
            "source_path": "/x/a.wav",
        },
        content_group_id="a",
        selection_reason="test",
    )

    assert row["route"] == "clean_candidate"
    assert row["audit_status"] == "audit_pending_scores"
    assert row["training_ready"] is False
