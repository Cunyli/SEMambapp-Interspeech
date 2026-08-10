import argparse
import math

import numpy as np
import pytest

from scripts.score_dnf_phase_b_probe import (
    DNSMOS_INPUT_SECONDS,
    DNSMOS_SAMPLE_RATE,
    DNSMOSP835,
    frame_signal,
    probe_decision,
    quantile_summary,
    technical_gate,
    technical_metrics,
)


class MockDNSMOSSession:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, ...]] = []

    def run(
        self,
        output_names: object,
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names is None
        batch = inputs["input_1"]
        self.batch_shapes.append(batch.shape)
        means = batch.mean(axis=1, dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(batch, dtype=np.float64), axis=1))
        peaks = np.max(np.abs(batch), axis=1)
        outputs = np.stack(
            (
                3.0 + means,
                2.5 + rms,
                2.0 + peaks,
            ),
            axis=1,
        )
        return [outputs.astype(np.float32)]


def _gate_args() -> argparse.Namespace:
    return argparse.Namespace(
        min_native_sample_rate=16000,
        min_duration=1.0,
        min_active_ratio=0.20,
        min_active_seconds=0.8,
        min_active_rms_dbfs=-55.0,
        max_clip_ratio=0.001,
        max_abs_dc=0.02,
    )


def test_frame_signal_covers_short_audio_without_empty_output() -> None:
    frames = frame_signal(np.ones(80, dtype=np.float32), 320, 160)

    assert frames.shape == (1, 320)
    np.testing.assert_array_equal(frames[0, :80], np.ones(80, dtype=np.float32))


def test_stationary_finite_audio_passes_technical_gate() -> None:
    sample_rate = 16000
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    audio = (0.1 * np.sin(2 * math.pi * 220.0 * time)).astype(np.float32)

    metrics, audit_audio = technical_metrics(
        audio,
        sample_rate,
        1,
        target_sample_rate=sample_rate,
        frame_length=320,
        hop_length=160,
        active_relative_db=-40.0,
    )
    hard_pass, reasons, _ = technical_gate(metrics, _gate_args())

    assert audit_audio.shape == audio.shape
    assert hard_pass
    assert reasons == []
    assert metrics["active_frame_ratio"] == 1.0
    assert metrics["clip_ratio"] == 0.0


def test_clipped_audio_fails_closed() -> None:
    audio = np.ones(32000, dtype=np.float32)

    metrics, _ = technical_metrics(
        audio,
        16000,
        1,
        target_sample_rate=16000,
        frame_length=320,
        hop_length=160,
        active_relative_db=-40.0,
    )
    hard_pass, reasons, _ = technical_gate(metrics, _gate_args())

    assert not hard_pass
    assert "clipping_above_maximum" in reasons
    assert "dc_offset_above_maximum" in reasons


def test_dnsmos_polynomial_mapping_matches_official_coefficients() -> None:
    sig, bak, ovrl = DNSMOSP835.polynomial_scores(3.0, 3.0, 3.0)

    expected_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])(3.0)
    expected_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])(3.0)
    expected_ovrl = np.poly1d([-0.06766283, 1.11546468, 0.04602535])(3.0)
    np.testing.assert_allclose(
        [sig, bak, ovrl],
        [expected_sig, expected_bak, expected_ovrl],
    )


def test_dnsmos_batch_matches_single_window_inference() -> None:
    duration_seconds = DNSMOS_INPUT_SECONDS + 4.0
    sample_count = int(duration_seconds * DNSMOS_SAMPLE_RATE)
    time = np.arange(sample_count, dtype=np.float64) / DNSMOS_SAMPLE_RATE
    audio = (
        0.08 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.02 * np.sin(2.0 * math.pi * 431.0 * time)
        + np.linspace(-0.01, 0.01, sample_count)
    ).astype(np.float32)
    single_session = MockDNSMOSSession()
    batch_session = MockDNSMOSSession()
    single = DNSMOSP835(
        None,
        batch_size=1,
        session=single_session,
    )(audio)
    batched = DNSMOSP835(
        None,
        batch_size=3,
        session=batch_session,
    )(audio)

    assert single["status"] == batched["status"] == "ok"
    assert single["segment_count"] == batched["segment_count"] == 5
    for field in ("sig_raw", "bak_raw", "ovrl_raw", "sig", "bak", "ovrl"):
        np.testing.assert_allclose(
            batched[field],
            single[field],
            rtol=0.0,
            atol=1.0e-12,
        )
    assert single_session.batch_shapes == [
        (1, int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLE_RATE))
    ] * 5
    assert batch_session.batch_shapes == [
        (3, int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLE_RATE)),
        (2, int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLE_RATE)),
    ]


def test_dnsmos_score_many_batches_across_utterances() -> None:
    required = int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLE_RATE)
    audios = [
        np.linspace(-0.1, 0.1, required, dtype=np.float32),
        np.linspace(-0.2, 0.2, required, dtype=np.float32),
        np.linspace(-0.3, 0.3, required, dtype=np.float32),
    ]
    separate_session = MockDNSMOSSession()
    separate_scorer = DNSMOSP835(
        None,
        batch_size=1,
        session=separate_session,
    )
    separate = [separate_scorer(audio) for audio in audios]

    combined_session = MockDNSMOSSession()
    combined_scorer = DNSMOSP835(
        None,
        batch_size=3,
        session=combined_session,
    )
    combined = combined_scorer.score_many(audios)

    assert len(combined) == len(separate) == 3
    for expected, actual in zip(separate, combined, strict=True):
        for field in ("sig_raw", "bak_raw", "ovrl_raw", "sig", "bak", "ovrl"):
            assert actual[field] == pytest.approx(expected[field], abs=1e-12)
    assert combined_session.batch_shapes == [(3, required)]


def test_technical_hard_fail_is_excluded_not_recast_as_noisy_target() -> None:
    decision = probe_decision(
        technical_hard_pass=False,
        technical_hard_reasons=["clipping_above_maximum"],
        dnsmos_bak=4.5,
        source_bak_p25=3.0,
    )

    assert decision == {
        "route": "exclude_invalid",
        "status": "technical_hard_fail",
        "training_ready": False,
        "reasons": ["clipping_above_maximum"],
    }


def test_low_bak_is_a_review_stratum_not_an_automatic_route_change() -> None:
    decision = probe_decision(
        technical_hard_pass=True,
        technical_hard_reasons=[],
        dnsmos_bak=2.0,
        source_bak_p25=3.0,
    )
    assert decision["route"] == "clean_candidate"
    assert decision["status"] == "low_bak_review_stratum"
    assert not decision["training_ready"]
    assert "score_is_ranking_not_route_definition" in decision["reasons"]


def test_quantile_summary_reports_expected_quartile() -> None:
    summary = quantile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p25"] == pytest.approx(1.75)
