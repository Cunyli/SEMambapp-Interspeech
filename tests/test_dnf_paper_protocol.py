import pytest
import torch

from dataloaders.dnf_paper_protocol import (
    build_clean_target_supervision,
    build_noisy_target_supervision,
)


DTYPE = torch.float64


def test_noisy_target_constructs_exact_additive_pair_and_visible_targets() -> None:
    speech = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=DTYPE)
    noise1 = torch.tensor([[0.0, 0.5, 0.0, -0.5]], dtype=DTYPE)
    noise2 = torch.tensor([[0.25, 0.0, -0.25, 0.0]], dtype=DTYPE)

    sample = build_noisy_target_supervision(
        speech,
        noise1,
        noise2,
        noise1_source_id="noise-a",
        noise2_source_id="noise-b",
    )

    expected_noisy = speech + noise1
    torch.testing.assert_close(sample.targets["noisy_speech"], expected_noisy)
    torch.testing.assert_close(sample.targets["artificial_noise"], noise2)
    torch.testing.assert_close(sample.model_input, expected_noisy + noise2)
    assert sample.supervision_type == "noisy_target"
    assert set(sample.targets) == {"noisy_speech", "artificial_noise"}
    assert sample.valid_mask.all()


def test_noisy_target_requires_distinct_noise_source_ids() -> None:
    waveform = torch.tensor([1.0, -1.0], dtype=DTYPE)

    with pytest.raises(ValueError, match="distinct source recordings"):
        build_noisy_target_supervision(
            waveform,
            waveform,
            waveform,
            noise1_source_id="same",
            noise2_source_id="same",
        )


def test_demean_is_applied_to_each_component_before_addition() -> None:
    speech = torch.tensor([[2.0, 4.0]], dtype=DTYPE)
    noise1 = torch.tensor([[10.0, 14.0]], dtype=DTYPE)
    noise2 = torch.tensor([[-3.0, 1.0]], dtype=DTYPE)

    sample = build_noisy_target_supervision(
        speech,
        noise1,
        noise2,
        noise1_source_id="noise-a",
        noise2_source_id="noise-b",
        demean=True,
    )

    centered_speech = speech - speech.mean(dim=-1, keepdim=True)
    centered_noise1 = noise1 - noise1.mean(dim=-1, keepdim=True)
    centered_noise2 = noise2 - noise2.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(
        sample.targets["noisy_speech"],
        centered_speech + centered_noise1,
    )
    torch.testing.assert_close(
        sample.model_input,
        centered_speech + centered_noise1 + centered_noise2,
    )
    torch.testing.assert_close(
        sample.diagnostics["model_input"].mean,
        torch.zeros(1, dtype=DTYPE),
        atol=1e-12,
        rtol=0,
    )


def test_demean_can_be_disabled_without_breaking_additive_identity() -> None:
    speech = torch.tensor([2.0, 4.0], dtype=DTYPE)
    noise = torch.tensor([1.0, 3.0], dtype=DTYPE)

    sample = build_clean_target_supervision(speech, noise, demean=False)

    torch.testing.assert_close(sample.targets["clean_speech"], speech)
    torch.testing.assert_close(sample.targets["mixture_noise"], noise)
    torch.testing.assert_close(sample.model_input, speech + noise)
    assert not sample.demeaned


def test_clean_target_returns_eq15_supervision_fields() -> None:
    speech = torch.tensor([[1.0, -1.0, 0.0]], dtype=DTYPE)
    noise = torch.tensor([[0.0, 0.5, -0.5]], dtype=DTYPE)

    sample = build_clean_target_supervision(speech, noise)

    assert sample.supervision_type == "clean_target"
    assert set(sample.targets) == {"clean_speech", "mixture_noise"}
    torch.testing.assert_close(
        sample.model_input,
        sample.targets["clean_speech"] + sample.targets["mixture_noise"],
    )
    assert sample.valid_mask.all()


def test_silent_component_is_reported_without_mutating_the_construction() -> None:
    speech = torch.tensor(
        [[1.0, -1.0], [1.0, -1.0]],
        dtype=DTYPE,
    )
    noise1 = torch.tensor(
        [[0.5, -0.5], [0.0, 0.0]],
        dtype=DTYPE,
    )
    noise2 = torch.tensor(
        [[0.25, -0.25], [0.25, -0.25]],
        dtype=DTYPE,
    )

    sample = build_noisy_target_supervision(
        speech,
        noise1,
        noise2,
        noise1_source_id="noise-a",
        noise2_source_id="noise-b",
        silence_rms_threshold=1e-6,
    )

    torch.testing.assert_close(
        sample.diagnostics["noise1"].energy,
        torch.tensor([0.5, 0.0], dtype=DTYPE),
    )
    assert not sample.diagnostics["noise1"].silent_mask[0]
    assert sample.diagnostics["noise1"].silent_mask[1]
    assert sample.valid_mask.tolist() == [True, False]
    torch.testing.assert_close(
        sample.model_input,
        sample.targets["noisy_speech"] + sample.targets["artificial_noise"],
    )


def test_shape_mismatch_is_rejected() -> None:
    speech = torch.ones(4, dtype=DTYPE)
    noise = torch.ones(5, dtype=DTYPE)

    with pytest.raises(ValueError, match="same shape"):
        build_clean_target_supervision(speech, noise)
