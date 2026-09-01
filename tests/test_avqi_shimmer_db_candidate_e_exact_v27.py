from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

from scripts.avqi_shimmer_exact_topology_worker import ExactTopologyEngine
from scripts.score_avqi_shimmer_db_candidate_e_fixed_topology_v27 import (
    SAMPLE_RATE,
    score_item,
)


def test_exact_fixed_topology_scalar_reconstructs_from_amplitude_tier(
    tmp_path: Path,
) -> None:
    sample_count = SAMPLE_RATE
    samples = np.arange(sample_count, dtype=np.float64)
    envelope = 0.16 + 0.025 * np.sin(2.0 * np.pi * samples / 3_700.0)
    waveform = envelope * np.sin(
        2.0 * np.pi * 100.0 * samples / SAMPLE_RATE
    )
    waveform_path = tmp_path / "synthetic_am.wav"
    sf.write(waveform_path, waveform, SAMPLE_RATE, subtype="PCM_24")
    pulses = np.arange(320.0, sample_count - 320.0, 160.0)
    topology = {
        "source_sample_count": sample_count,
        "metric_source_ranges": [[0, sample_count]],
        "metric_mapped_sample_count": sample_count,
        "metric_constant_prefix_samples": 0,
        "metric_sample_count": sample_count,
        "pulse_positions_samples": pulses.tolist(),
    }
    engine = ExactTopologyEngine()
    try:
        row = score_item(
            engine,
            {
                "item_id": "synthetic:exact",
                "case_id": "synthetic",
                "variant": "synthetic",
                "alpha": 0.0,
                "waveform_path": str(waveform_path),
                "topology": topology,
                "include_pulse_evidence": True,
            },
        )
    finally:
        engine.close()
    assert row["amplitude_count"] > 2
    assert len(row["amplitudes"]) == row["amplitude_count"]
    assert len(row["valid_pair_mask"]) == row["amplitude_count"] - 1
    assert row["exact_shimmer_db"] > 0.0
    np.testing.assert_allclose(
        row["exact_shimmer_db"],
        row["reconstructed_shimmer_db"],
        rtol=0.0,
        atol=1e-12,
    )


def test_praat_wav_save_uses_round_to_nearest_pcm16() -> None:
    generator = np.random.default_rng(29)
    values = generator.uniform(-0.95, 0.95, 10_000).astype(np.float64)
    with tempfile.TemporaryDirectory(prefix="praat-pcm16-test-") as directory:
        path = Path(directory) / "values.wav"
        call(
            parselmouth.Sound(values, SAMPLE_RATE),
            "Save as WAV file",
            str(path),
        )
        observed, sample_rate = sf.read(path, dtype="float64")
    expected = np.rint(values * 32768.0) / 32768.0
    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(observed, expected)
