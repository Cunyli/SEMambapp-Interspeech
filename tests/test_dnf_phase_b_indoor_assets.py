import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_dnf_phase_b_indoor_assets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_dnf_phase_b_indoor_assets",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_deterministic_sample_is_balanced_and_reproducible():
    rows = [
        {"key": f"a-{index}", "group": "a"} for index in range(8)
    ] + [
        {"key": f"b-{index}", "group": "b"} for index in range(3)
    ]
    first = MODULE.deterministic_sample(
        rows,
        group_key="group",
        per_group=2,
        seed=3407,
    )
    second = MODULE.deterministic_sample(
        rows,
        group_key="group",
        per_group=2,
        seed=3407,
    )
    assert first == second
    assert sum(row["group"] == "a" for row in first) == 2
    assert sum(row["group"] == "b" for row in first) == 2


def test_stationary_noise_has_near_zero_frame_variation():
    sample_rate = 16000
    time = np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 100.0 * time)).astype(np.float32)
    metrics = MODULE.noise_metrics(audio, sample_rate, 1)
    assert metrics["frame_log_rms_std_db"] < 1e-3
    assert metrics["frame_rms_p95_to_p05_db"] < 1e-3
    assert metrics["spectral_centroid_std_hz"] < 1e-3
    assert metrics["spectral_flux_p95"] < 1e-3


def test_exponential_rir_rt60_is_finite():
    sample_rate = 16000
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    audio = np.exp(-time / 0.2).astype(np.float32)
    estimate = MODULE.estimate_rt60_seconds(audio, sample_rate)
    assert estimate is not None
    assert 1.0 < estimate < 2.0


def test_stationary_noise_passes_automatic_gate():
    sample_rate = 16000
    time = np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 100.0 * time)).astype(np.float32)
    gate = MODULE.noise_auto_gate(
        MODULE.noise_metrics(audio, sample_rate, 1)
    )
    assert gate["automatic_pass"]
    assert not gate["training_ready"]


def test_frequency_switching_noise_fails_spectral_stationarity_gate():
    sample_rate = 16000
    half_second = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    frames = []
    for index in range(8):
        frequency = 100.0 if index % 2 == 0 else 3000.0
        frames.append(
            0.1 * np.sin(2 * np.pi * frequency * half_second)
        )
    audio = np.concatenate(frames).astype(np.float32)
    metrics = MODULE.noise_metrics(audio, sample_rate, 1)
    gate = MODULE.noise_auto_gate(metrics)
    assert not gate["automatic_pass"]
    assert (
        "spectral_centroid_std" in gate["failures"]
        or "spectral_flux" in gate["failures"]
    )


def test_multichannel_rir_is_not_primary_gate_ready():
    metrics = {
        "sample_rate": 16000,
        "channels": 2,
        "peak_time_seconds": 0.01,
        "rt60_estimate_seconds": 0.5,
        "tail_rms_db_relative_peak": -60.0,
    }
    gate = MODULE.rir_auto_gate(metrics)
    assert not gate["automatic_pass"]
    assert "channels" in gate["failures"]


def test_config_thresholds_match_reviewer_constants():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "dnf_phase_b_v2.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    promotion = config["indoor_asset_promotion_gate"]
    assert promotion["noise_automatic_gate"] == MODULE.NOISE_GATE
    rir_config = dict(promotion["rir_automatic_gate"])
    rir_config.pop("multi_channel_policy")
    assert rir_config == MODULE.RIR_GATE
