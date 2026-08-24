from __future__ import annotations

from scripts.profile_avqi_shimmer_db_runtime_v15_outliers import (
    DEV_ENGINEERING_MARGIN_MS,
    DOMINANT_STAGE_FIELDS,
    FASTPATH_IMPLEMENTATION,
    FROZEN_OUTLIER_CASE_IDS,
    IMPLEMENTATION_CONFIGS,
    INPUT_LOADER,
    PULSE_REFRESH_GATE_MS,
    case_summary,
)
from scripts.profile_avqi_shimmer_exact_topology_runtime import EXACT_WORKER


def test_runtime_v15_outlier_scope_is_strictly_frozen() -> None:
    assert FROZEN_OUTLIER_CASE_IDS == (
        "sealed_final__SD05__cs__rir_only",
        "sealed_final__ÄHH16__cs__rir_only",
    )
    assert PULSE_REFRESH_GATE_MS == 500.0
    assert DEV_ENGINEERING_MARGIN_MS == 450.0
    assert INPUT_LOADER == "soundfile_float32_exact_16khz_mono"
    assert IMPLEMENTATION_CONFIGS[FASTPATH_IMPLEMENTATION] == {
        "frame_scan_mode": "numpy_exact_aligned_frames",
        "pulse_enumeration_mode": "praat_pointprocess_to_matrix",
        "wav_roundtrip_mode": "soundfile_in_memory_pcm16",
        "sounding_assembly_mode": "numpy_exact_interval_slices",
    }
    assert 'request.get(\n                    "input_loader"' in EXACT_WORKER
    assert 'elif input_loader == "soundfile_float32_exact_16khz_mono"' in (
        EXACT_WORKER
    )


def test_case_summary_uses_repeated_warm_maximum() -> None:
    rows = []
    for repeat_index, total_ms in enumerate((320.0, 410.0, 430.0, 440.0, 449.0), 1):
        row = {
            "case_id": FROZEN_OUTLIER_CASE_IDS[0],
            "speaker_id": "SD05",
            "implementation": FASTPATH_IMPLEMENTATION,
            "base_frame_count": 160_000,
            "base_duration_seconds": 10.0,
            "metric_sample_count": 64_000,
            "metric_source_range_count": 30,
            "pulse_count": 600,
            "phase": "warm",
            "repeat_index": repeat_index,
            "total_refresh_ms": total_ms,
            "request_wall_ms": total_ms + 1.0,
            "wall_minus_internal_ms": 1.0,
        }
        for field in DOMINANT_STAGE_FIELDS:
            row[f"{field}_ms"] = 20.0
        row["highpass_ms"] = 200.0
        rows.append(row)

    summary = case_summary(rows, FASTPATH_IMPLEMENTATION)
    assert summary["dominant_stage"] == "highpass"
    assert summary["warm_total_maximum_ms"] == 449.0
    assert summary["formal_500ms_pass"] is True
    assert summary["development_450ms_margin_pass"] is True

    rows[-1]["total_refresh_ms"] = 501.0
    summary = case_summary(rows, FASTPATH_IMPLEMENTATION)
    assert summary["formal_500ms_pass"] is False
    assert summary["development_450ms_margin_pass"] is False
