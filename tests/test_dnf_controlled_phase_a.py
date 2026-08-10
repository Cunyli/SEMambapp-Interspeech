import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dataloaders"
    / "dnf_controlled_phase_a.py"
)
SPEC = importlib.util.spec_from_file_location("dnf_controlled_phase_a", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def speech_items() -> list[dict]:
    return [
        {
            "npy_path": f"unused_{index}.npy",
            "dataset": "EARS" if index % 2 else "VCTK",
            "key": f"speech-{index}",
        }
        for index in range(80)
    ]


def fake_speech(locator: dict) -> np.ndarray:
    seed = MODULE.stable_uint32(locator["key"])
    rng = np.random.default_rng(seed)
    time = np.arange(24000, dtype=np.float64) / 16000.0
    waveform = 0.1 * np.sin(2.0 * np.pi * 180.0 * time)
    waveform += 0.01 * rng.standard_normal(time.size)
    return waveform[None, :].astype(np.float32)


def test_route_schedule_is_exact_per_twenty_rows():
    schedule = MODULE.phase_a_route_schedule(60, seed=1234)
    for start in range(0, 60, 20):
        block = schedule[start : start + 20]
        assert block.count(MODULE.ROUTE_NOISY) == 15
        assert block.count(MODULE.ROUTE_CLEAN_REGULAR) == 4
        assert block.count(MODULE.ROUTE_CLEAN_WEAK) == 1


def test_manifest_rows_are_reproducible_and_strict_only():
    first = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=40,
        seed=1234,
        split="train",
    )
    second = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=40,
        seed=1234,
        split="train",
    )
    assert first == second
    assert MODULE.manifest_rows_sha256(first) == MODULE.manifest_rows_sha256(
        second
    )
    assert {row["speech_source_category"] for row in first} == {"clean_strict"}
    assert {
        row[key]["family"] for row in first for key in ("noise1", "noise2")
    } == set(MODULE.NOISE_FAMILIES)
    assert all(
        row["noise_pairing_policy"] == MODULE.NOISE_PAIRING_SAME_FAMILY_IID
        for row in first
    )
    assert all(
        row["noise1"]["family"] == row["noise2"]["family"] for row in first
    )
    assert all(
        row["noise1"]["seed"] != row["noise2"]["seed"] for row in first
    )
    clean_speech = {
        row["speech"]["key"] for row in first if row["route"] != MODULE.ROUTE_NOISY
    }
    noisy_speech = {
        row["speech"]["key"] for row in first if row["route"] == MODULE.ROUTE_NOISY
    }
    assert not clean_speech & noisy_speech
    assert len(clean_speech) == 10
    assert len(noisy_speech) == 30
    assert {
        row["speech_partition_policy"] for row in first
    } == {MODULE.SPEECH_PARTITION_DISJOINT}
    assert {row["schema_version"] for row in first} == {
        "dnf_controlled_phase_a_v2"
    }
    assert {row["snr_definition"] for row in first} == {
        MODULE.SNR_DEFINITION
    }
    assert {row["training_input"] for row in first} == {
        MODULE.TRAINING_INPUT_DEFINITION
    }
    assert {row["deployment_validation_input"] for row in first} == {
        MODULE.DEPLOYMENT_INPUT_DEFINITION
    }


def test_cross_family_cycle_is_explicit_robustness_variant():
    rows = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=40,
        seed=1234,
        split="train",
        noise_pairing_policy=MODULE.NOISE_PAIRING_CROSS_FAMILY_CYCLE,
    )
    assert all(
        row["noise_pairing_policy"] == MODULE.NOISE_PAIRING_CROSS_FAMILY_CYCLE
        for row in rows
    )
    assert all(
        row["noise1"]["family"] != row["noise2"]["family"] for row in rows
    )


def test_dataset_collate_enforces_route_indexed_supervision(tmp_path):
    rows = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=20,
        seed=1234,
        split="train",
    )
    manifest = tmp_path / "train.jsonl"
    MODULE.write_jsonl(manifest, rows)
    dataset = MODULE.PhaseAControlledStreamDataset(
        None,
        manifest,
        split="train",
        samples_per_epoch=20,
        target_sample_rate=16000,
        cut_duration=1.0,
        seed=1234,
        speech_reader=fake_speech,
    )
    items = [dataset[index] for index in range(20)]
    batch = MODULE.phase_a_collate(items)
    assert batch["model_input_wav"].shape == (20, 16000)
    assert batch["clean_indices"].numel() == 5
    assert batch["noisy_indices"].numel() == 15
    assert batch["clean_speech_wav"].shape == (5, 16000)
    assert batch["noisy_speech_target_wav"].shape == (15, 16000)
    assert "eval_clean_wav" not in batch
    assert torch.isfinite(batch["model_input_wav"]).all()
    for index in batch["clean_indices"].tolist():
        assert "noisy_speech_target" not in items[index]
    for index in batch["noisy_indices"].tolist():
        assert "clean_speech" not in items[index]


def test_eval_view_is_single_noise_and_identity_is_exposed(tmp_path):
    rows = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=20,
        seed=1234,
        split="valid",
    )
    manifest = tmp_path / "valid.jsonl"
    MODULE.write_jsonl(manifest, rows)
    dataset = MODULE.PhaseAControlledStreamDataset(
        None,
        manifest,
        split="valid",
        samples_per_epoch=20,
        target_sample_rate=16000,
        cut_duration=1.0,
        seed=1234,
        expose_clean_for_eval=True,
        speech_reader=fake_speech,
    )
    items = [dataset[index] for index in range(20)]
    batch = MODULE.phase_a_collate(items)
    assert batch["eval_model_input_wav"].shape == (20, 16000)
    assert batch["eval_clean_wav"].shape == (20, 16000)
    for item in items:
        clean = item["eval_clean_speech"].astype(np.float64)
        deployment = item["eval_model_input"].astype(np.float64)
        training = item["model_input"].astype(np.float64)
        n1 = deployment - clean
        n2 = training - deployment
        clean_energy = float(np.square(clean).sum())
        n1_energy = float(np.square(n1).sum())
        n2_energy = float(np.square(n2).sum())
        measured_snr = 10.0 * np.log10(clean_energy / n1_energy)
        assert measured_snr == pytest.approx(
            item["info"]["target_snr_db"],
            abs=1.0e-4,
        )
        assert n2_energy / n1_energy == pytest.approx(1.0, abs=1.0e-4)


def test_collate_handles_all_noisy_and_all_clean_microbatches(tmp_path):
    rows = MODULE.build_phase_a_manifest_rows(
        speech_items(),
        row_count=20,
        seed=1234,
        split="train",
    )
    manifest = tmp_path / "train.jsonl"
    MODULE.write_jsonl(manifest, rows)
    dataset = MODULE.PhaseAControlledStreamDataset(
        None,
        manifest,
        split="train",
        samples_per_epoch=20,
        target_sample_rate=16000,
        cut_duration=1.0,
        seed=1234,
        speech_reader=fake_speech,
    )
    items = [dataset[index] for index in range(20)]
    noisy = [item for item in items if item["route"] == MODULE.ROUTE_NOISY][:4]
    clean = [item for item in items if item["route"] != MODULE.ROUTE_NOISY][:4]
    noisy_batch = MODULE.phase_a_collate(noisy)
    clean_batch = MODULE.phase_a_collate(clean)
    assert noisy_batch["clean_indices"].numel() == 0
    assert noisy_batch["clean_speech_wav"].shape == (0, 16000)
    assert clean_batch["noisy_indices"].numel() == 0
    assert clean_batch["noisy_speech_target_wav"].shape == (0, 16000)


def test_mixture_is_additive_and_scale_consistent():
    time = np.arange(16000, dtype=np.float64) / 16000.0
    clean = np.sin(2.0 * np.pi * 211.0 * time)[None, :]
    noise1 = np.sin(2.0 * np.pi * 557.0 * time)[None, :]
    noise2 = np.sin(2.0 * np.pi * 883.0 * time)[None, :]
    mixture = MODULE.build_phase_a_mixture(
        clean,
        noise1,
        noise2,
        target_snr_db=5.0,
    )
    reconstructed = (
        mixture["clean_speech"] + mixture["noise1"] + mixture["noise2"]
    )
    np.testing.assert_allclose(mixture["model_input"], reconstructed, atol=1e-7)
    assert mixture["diagnostics"]["max_additive_error"] <= 1e-7
    clean_energy = float(np.square(mixture["clean_speech"]).sum())
    n1_energy = float(np.square(mixture["noise1"]).sum())
    n2_energy = float(np.square(mixture["noise2"]).sum())
    measured_snr = 10.0 * np.log10(clean_energy / n1_energy)
    assert measured_snr == pytest.approx(5.0, abs=1.0e-4)
    assert n2_energy / n1_energy == pytest.approx(1.0, abs=1.0e-4)


def test_partial_block_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        MODULE.phase_a_route_schedule(21, seed=1234)
