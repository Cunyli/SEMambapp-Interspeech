"""Targeted acoustic/decode review for Phase-B indoor noise and RIR allowlists."""

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


NOISE_GATE = {
    "min_sample_rate": 16000,
    "min_duration_seconds": 1.0,
    "max_clip_ratio": 0.001,
    "max_abs_dc_offset": 0.02,
    "max_frame_log_rms_std_db": 6.0,
    "max_frame_rms_p95_to_p05_db": 12.0,
    "max_crest_factor_db": 22.0,
    "max_spectral_centroid_std_hz": 400.0,
    "max_spectral_flux_p95": 0.35,
}
RIR_GATE = {
    "min_sample_rate": 16000,
    "allowed_channels": [1],
    "max_peak_time_seconds": 0.1,
    "min_rt60_seconds": 0.1,
    "max_rt60_seconds": 3.0,
    "max_tail_rms_db_relative_peak": -35.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a deterministic stratified indoor-noise/RIR review sample."
    )
    parser.add_argument("--noise-jsonl", type=Path, required=True)
    parser.add_argument("--rir-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-per-reason", type=int, default=16)
    parser.add_argument("--rir-per-dataset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def deterministic_sample(
    rows: list[dict],
    *,
    group_key: str,
    per_group: int,
    seed: int,
) -> list[dict]:
    if per_group <= 0:
        raise ValueError("per_group must be positive")
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key) or "<missing>")
        uid = str(row.get("key") or row.get("audio_member") or "")
        digest = hashlib.sha256(
            f"{seed}\0{group}\0{uid}".encode("utf-8")
        ).hexdigest()
        groups[group].append((digest, row))
    selected = []
    for group in sorted(groups):
        ranked = sorted(groups[group], key=lambda pair: pair[0])
        selected.extend(row for _, row in ranked[:per_group])
    return selected


def load_tar_audio(row: dict) -> tuple[np.ndarray, int, int]:
    tar_path = Path(row["_shard_dir"]) / str(row["shard"])
    member_name = str(row["audio_member"])
    with tarfile.open(tar_path, "r:") as archive:
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise ValueError(f"missing {member_name} in {tar_path}")
        payload = extracted.read()
    audio, sample_rate = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=True,
    )
    channels = int(audio.shape[1])
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"empty/non-finite audio: {tar_path}:{member_name}")
    return mono, int(sample_rate), channels


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(0.5 * sample_rate)))
    count = max(1, int(math.ceil(audio.size / frame)))
    padded = np.pad(audio, (0, count * frame - audio.size))
    return np.sqrt(
        np.mean(
            np.square(padded.reshape(count, frame), dtype=np.float64),
            axis=1,
        )
    )


def spectral_stationarity(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    frame = max(32, int(round(0.5 * sample_rate)))
    count = max(1, int(math.ceil(audio.size / frame)))
    padded = np.pad(audio, (0, count * frame - audio.size))
    frames = padded.reshape(count, frame)
    window = np.hanning(frame)
    power = np.square(np.abs(np.fft.rfft(frames * window, axis=1)))
    power_sum = power.sum(axis=1, keepdims=True)
    normalized = power / np.maximum(power_sum, 1e-12)
    frequencies = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    centroids = (normalized * frequencies[None, :]).sum(axis=1)
    if count > 1:
        flux = 0.5 * np.abs(np.diff(normalized, axis=0)).sum(axis=1)
    else:
        flux = np.zeros(1, dtype=np.float64)
    return {
        "spectral_centroid_std_hz": float(centroids.std()),
        "spectral_flux_p50": float(np.quantile(flux, 0.50)),
        "spectral_flux_p95": float(np.quantile(flux, 0.95)),
    }


def noise_metrics(audio: np.ndarray, sample_rate: int, channels: int) -> dict:
    frames = frame_rms(audio, sample_rate)
    active = frames[frames > 1e-8]
    overall_rms = rms(audio)
    if overall_rms <= 1e-8 or not active.size:
        raise ValueError("indoor-noise review sample is silent")
    log_frames = 20.0 * np.log10(np.maximum(active, 1e-12))
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": float(audio.size / sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms_dbfs": float(20.0 * math.log10(max(overall_rms, 1e-12))),
        "dc_offset": float(np.mean(audio, dtype=np.float64)),
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)),
        "crest_factor_db": float(
            20.0
            * math.log10(
                max(float(np.max(np.abs(audio))), 1e-12)
                / max(overall_rms, 1e-12)
            )
        ),
        "active_half_second_frames": int(active.size),
        "frame_log_rms_std_db": float(log_frames.std()),
        "frame_rms_p95_to_p05_db": (
            float(
                20.0
                * math.log10(
                    max(float(np.quantile(active, 0.95)), 1e-12)
                    / max(float(np.quantile(active, 0.05)), 1e-12)
                )
            )
        ),
        **spectral_stationarity(audio, sample_rate),
    }


def estimate_rt60_seconds(audio: np.ndarray, sample_rate: int) -> float | None:
    peak_index = int(np.argmax(np.abs(audio)))
    tail = np.asarray(audio[peak_index:], dtype=np.float64)
    energy = np.square(tail)
    if energy.size < 2 or float(energy.sum()) <= 0.0:
        return None
    decay = np.cumsum(energy[::-1])[::-1]
    decay_db = 10.0 * np.log10(np.maximum(decay / decay[0], 1e-12))
    mask = (decay_db <= -5.0) & (decay_db >= -35.0)
    if int(mask.sum()) < 10:
        return None
    time = np.arange(tail.size, dtype=np.float64) / sample_rate
    slope, _ = np.polyfit(time[mask], decay_db[mask], 1)
    if not math.isfinite(float(slope)) or slope >= 0.0:
        return None
    return float(-60.0 / slope)


def rir_metrics(audio: np.ndarray, sample_rate: int, channels: int) -> dict:
    peak_index = int(np.argmax(np.abs(audio)))
    early_end = min(audio.size, peak_index + int(round(0.05 * sample_rate)))
    early_energy = float(
        np.square(audio[peak_index:early_end], dtype=np.float64).sum()
    )
    late_energy = float(
        np.square(audio[early_end:], dtype=np.float64).sum()
    )
    total_energy = float(np.square(audio, dtype=np.float64).sum())
    if total_energy <= 1e-12:
        raise ValueError("RIR review sample is silent")
    tail_samples = max(
        1,
        min(audio.size, max(int(round(0.1 * sample_rate)), audio.size // 10)),
    )
    tail_rms = rms(audio[-tail_samples:])
    peak = float(np.max(np.abs(audio)))
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": float(audio.size / sample_rate),
        "peak": peak,
        "peak_index": peak_index,
        "peak_time_seconds": float(peak_index / sample_rate),
        "energy": total_energy,
        "early_to_late_db": float(
            10.0
            * math.log10(
                max(early_energy, 1e-12) / max(late_energy, 1e-12)
            )
        ),
        "rt60_estimate_seconds": estimate_rt60_seconds(audio, sample_rate),
        "tail_rms_db_relative_peak": float(
            20.0
            * math.log10(max(tail_rms, 1e-12) / max(peak, 1e-12))
        ),
    }


def noise_auto_gate(metrics: dict) -> dict:
    failures = []
    checks = {
        "sample_rate": metrics["sample_rate"] >= NOISE_GATE["min_sample_rate"],
        "duration": (
            metrics["duration_seconds"] >= NOISE_GATE["min_duration_seconds"]
        ),
        "clip_ratio": metrics["clip_ratio"] <= NOISE_GATE["max_clip_ratio"],
        "dc_offset": (
            abs(metrics["dc_offset"]) <= NOISE_GATE["max_abs_dc_offset"]
        ),
        "frame_log_rms_std": (
            metrics["frame_log_rms_std_db"]
            <= NOISE_GATE["max_frame_log_rms_std_db"]
        ),
        "frame_rms_range": (
            metrics["frame_rms_p95_to_p05_db"]
            <= NOISE_GATE["max_frame_rms_p95_to_p05_db"]
        ),
        "crest_factor": (
            metrics["crest_factor_db"] <= NOISE_GATE["max_crest_factor_db"]
        ),
        "spectral_centroid_std": (
            metrics["spectral_centroid_std_hz"]
            <= NOISE_GATE["max_spectral_centroid_std_hz"]
        ),
        "spectral_flux": (
            metrics["spectral_flux_p95"]
            <= NOISE_GATE["max_spectral_flux_p95"]
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "automatic_pass": not failures,
        "failures": failures,
        "thresholds": NOISE_GATE,
        "manual_no_speech_music_transient_review_required": True,
        "training_ready": False,
    }


def rir_auto_gate(metrics: dict) -> dict:
    rt60 = metrics["rt60_estimate_seconds"]
    checks = {
        "sample_rate": metrics["sample_rate"] >= RIR_GATE["min_sample_rate"],
        "channels": metrics["channels"] in RIR_GATE["allowed_channels"],
        "peak_time": (
            metrics["peak_time_seconds"] <= RIR_GATE["max_peak_time_seconds"]
        ),
        "rt60_available": rt60 is not None,
        "rt60_min": (
            rt60 is not None and rt60 >= RIR_GATE["min_rt60_seconds"]
        ),
        "rt60_max": (
            rt60 is not None and rt60 <= RIR_GATE["max_rt60_seconds"]
        ),
        "tail_noise": (
            metrics["tail_rms_db_relative_peak"]
            <= RIR_GATE["max_tail_rms_db_relative_peak"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "automatic_pass": not failures,
        "failures": failures,
        "thresholds": RIR_GATE,
        "channel_policy": (
            "primary pilot accepts mono RIR only; multi-channel and binaural "
            "RIR require a separate channel-handling ablation"
        ),
        "training_ready": False,
    }


def safe_name(row: dict) -> str:
    raw = str(row.get("key") or Path(str(row["audio_member"])).stem)
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw
    )[:160]


def save_noise_listening(
    output_dir: Path,
    row: dict,
    audio: np.ndarray,
    sample_rate: int,
) -> dict:
    duration = min(audio.size, 10 * sample_rate)
    excerpt = audio[:duration]
    name = safe_name(row) + ".wav"
    original = output_dir / "noise_original" / name
    normalized = output_dir / "noise_rms_normalized" / name
    original.parent.mkdir(parents=True, exist_ok=True)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    sf.write(original, excerpt, sample_rate, subtype="FLOAT")
    target_rms = 10.0 ** (-23.0 / 20.0)
    gain = target_rms / max(rms(excerpt), 1e-12)
    normalized_audio = excerpt * gain
    peak = float(np.max(np.abs(normalized_audio)))
    if peak > 0.95:
        normalized_audio *= 0.95 / peak
    sf.write(normalized, normalized_audio, sample_rate, subtype="FLOAT")
    return {
        "original": str(original),
        "rms_normalized_for_listening_only": str(normalized),
    }


def main() -> None:
    args = parse_args()
    noise_rows = deterministic_sample(
        read_jsonl(args.noise_jsonl),
        group_key="selection_reason",
        per_group=args.noise_per_reason,
        seed=args.seed,
    )
    rir_rows = deterministic_sample(
        read_jsonl(args.rir_jsonl),
        group_key="dataset",
        per_group=args.rir_per_dataset,
        seed=args.seed,
    )
    reviewed_noise = []
    for row in noise_rows:
        audio, sample_rate, channels = load_tar_audio(row)
        metrics = noise_metrics(audio, sample_rate, channels)
        reviewed_noise.append(
            {
                **row,
                "acoustic_metrics": metrics,
                "automatic_gate": noise_auto_gate(metrics),
                "listening": save_noise_listening(
                    args.output_dir,
                    row,
                    audio,
                    sample_rate,
                ),
                "training_ready": False,
                "review_status": "manual_listening_pending",
            }
        )
    reviewed_rir = []
    for row in rir_rows:
        audio, sample_rate, channels = load_tar_audio(row)
        metrics = rir_metrics(audio, sample_rate, channels)
        reviewed_rir.append(
            {
                **row,
                "decode_metrics": metrics,
                "automatic_gate": rir_auto_gate(metrics),
                "training_ready": False,
                "review_status": "decode_pass_parameter_review_pending",
            }
        )
    reviewed_noise.sort(
        key=lambda row: (row["selection_reason"], row["key"])
    )
    reviewed_rir.sort(key=lambda row: (row["dataset"], row["key"]))
    noise_output = args.output_dir / "indoor_noise_acoustic_review.jsonl"
    rir_output = args.output_dir / "indoor_rir_decode_review.jsonl"
    write_jsonl(noise_output, reviewed_noise)
    write_jsonl(rir_output, reviewed_rir)
    payload = {
        "schema_version": "dnf-phase-b-indoor-acoustic-review-v2",
        "noise_review_count": len(reviewed_noise),
        "noise_by_reason": {
            reason: sum(row["selection_reason"] == reason for row in reviewed_noise)
            for reason in sorted({row["selection_reason"] for row in reviewed_noise})
        },
        "noise_automatic_pass_count": sum(
            row["automatic_gate"]["automatic_pass"] for row in reviewed_noise
        ),
        "rir_review_count": len(reviewed_rir),
        "rir_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in reviewed_rir)
            for dataset in sorted({row["dataset"] for row in reviewed_rir})
        },
        "rir_automatic_pass_count": sum(
            row["automatic_gate"]["automatic_pass"] for row in reviewed_rir
        ),
        "noise_output": str(noise_output),
        "rir_output": str(rir_output),
        "training_ready": False,
        "next_gate": (
            "Manual listening must reject speech, music, transient, and "
            "non-stationary event contamination before promotion."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "indoor_asset_review_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
