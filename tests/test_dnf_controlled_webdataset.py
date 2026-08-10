import numpy as np
import pytest
import torch

from dataloaders.dnf_controlled_webdataset import (
    ControlledSampleRejected,
    _select_source_shards,
    build_controlled_additive_mixture,
    controlled_dnf_collate,
)


def _orthogonal_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speech = np.array([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float32)
    noise1 = np.array([[1, 1, -1, -1, 1, 1, -1, -1]], dtype=np.float32)
    noise2 = np.array([[1, 1, 1, 1, -1, -1, -1, -1]], dtype=np.float32)
    return speech, noise1, noise2


def _mixture():
    speech, noise1, noise2 = _orthogonal_components()
    return build_controlled_additive_mixture(
        speech,
        noise1,
        noise2,
        seed=17,
        target_sample_rate=8,
        cut_duration=1.0,
        target_snr_db=0.0,
    )


def test_controlled_mixture_is_exact_addition_with_equal_noise_energy() -> None:
    mixture = _mixture()

    np.testing.assert_array_equal(
        mixture.degraded,
        mixture.noisy_speech_target + mixture.artificial_noise,
    )
    np.testing.assert_allclose(
        mixture.diagnostics["noise1_energy"],
        mixture.diagnostics["noise2_energy"],
        rtol=1e-6,
    )
    assert mixture.diagnostics["max_additive_error"] == 0.0
    assert abs(mixture.diagnostics["achieved_snr_db"]) < 1e-5
    assert mixture.diagnostics["peak_after_common_gain"] <= 0.95 + 1e-7
    assert mixture.degraded.dtype == np.float32


def test_common_gain_preserves_every_visible_additive_identity() -> None:
    mixture = _mixture()

    reconstructed_noise1 = mixture.noisy_speech_target - mixture.clean_speech
    np.testing.assert_allclose(reconstructed_noise1, mixture.noise1, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        mixture.degraded,
        mixture.clean_speech + mixture.noise1 + mixture.artificial_noise,
        atol=1e-7,
        rtol=0,
    )
    assert 0.0 < mixture.diagnostics["common_gain"] < 1.0


def test_correlated_components_are_rejected() -> None:
    speech, noise1, _ = _orthogonal_components()

    with pytest.raises(ControlledSampleRejected, match="correlation"):
        build_controlled_additive_mixture(
            speech,
            noise1,
            speech.copy(),
            seed=17,
            target_sample_rate=8,
            cut_duration=1.0,
            target_snr_db=0.0,
        )


def test_source_filter_never_accepts_mixed_or_unapproved_shards() -> None:
    records = [
        {"dataset_counts": {"EARS": 3}, "sample_count": 3, "shard": "ears.tar"},
        {"dataset_counts": {"VCTK": 4}, "sample_count": 4, "shard": "vctk.tar"},
        {
            "dataset_counts": {"EARS": 2, "MLS_HQ_en_chunk0001": 2},
            "sample_count": 4,
            "shard": "mixed.tar",
        },
        {
            "dataset_counts": {"CommonVoice25": 5},
            "sample_count": 5,
            "shard": "cv.tar",
        },
    ]

    selected = _select_source_shards(records, ("EARS", "VCTK"), "speech")

    assert [row["shard"] for row in selected] == ["ears.tar", "vctk.tar"]
    assert all(row["_route_category"] == "clean_strict" for row in selected)


def _item(include_eval_clean: bool) -> dict:
    mixture = _mixture()
    item = {
        "degraded": mixture.degraded,
        "s_noisy": mixture.noisy_speech_target,
        "added_noise": mixture.artificial_noise,
        "sample_rate": 8,
        "length": 8,
        "uid": "sample-1",
        "info": {"route_category": "noisy_eq13"},
    }
    if include_eval_clean:
        item["eval_clean"] = mixture.clean_speech
    return item


def test_training_collate_has_no_clean_tensor() -> None:
    batch = controlled_dnf_collate([_item(include_eval_clean=False)])

    assert set(batch) == {
        "mode",
        "degraded_wav",
        "s_noisy_wav",
        "added_noise_wav",
        "sample_rate",
        "length",
        "utterance_id",
        "info",
    }
    assert "clean_wav" not in batch
    assert "eval_clean_wav" not in batch
    assert batch["degraded_wav"].shape == (1, 8)
    assert batch["degraded_wav"].dtype == torch.float32


def test_eval_collate_exposes_clean_only_under_eval_name() -> None:
    batch = controlled_dnf_collate([_item(include_eval_clean=True)])

    assert "eval_clean_wav" in batch
    assert "clean_wav" not in batch
    assert batch["eval_clean_wav"].shape == (1, 8)


def test_collate_rejects_mixed_eval_visibility() -> None:
    with pytest.raises(ValueError, match="visibility"):
        controlled_dnf_collate(
            [_item(include_eval_clean=False), _item(include_eval_clean=True)]
        )
