"""Score a frozen Phase-B speech probe with technical gates and DNSMOS P.835.

This script never promotes samples directly into Phase-A training.  It appends
auditable technical measurements and source-relative DNSMOS BAK ranks to a
probe JSONL produced by ``audit_dnf_phase_b_sources.py``.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SCHEMA_VERSION = "dnf-phase-b-probe-score-v2"
DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_SECONDS = 9.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Technical and DNSMOS audit for a frozen DNF Phase-B probe."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--dnsmos-model", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional positive probe prefix for benchmark/smoke runs; zero uses all rows.",
    )
    parser.add_argument(
        "--dnsmos-batch-size",
        type=int,
        default=8,
        help="Number of 9.01-second DNSMOS windows per CPU ONNX call.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=8,
        help="Number of utterances prepared before cross-utterance DNSMOS batching.",
    )
    parser.add_argument(
        "--python-deps",
        type=Path,
        default=None,
        help="Optional audit-only dependency directory appended after the active environment.",
    )
    parser.add_argument("--target-sample-rate", type=int, default=DNSMOS_SAMPLE_RATE)
    parser.add_argument("--frame-length", type=int, default=320)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--active-relative-db", type=float, default=-40.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.20)
    parser.add_argument("--min-active-seconds", type=float, default=0.8)
    parser.add_argument("--min-native-sample-rate", type=int, default=16000)
    parser.add_argument("--max-clip-ratio", type=float, default=0.001)
    parser.add_argument("--max-abs-dc", type=float, default=0.02)
    parser.add_argument("--min-active-rms-dbfs", type=float, default=-55.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_json(row: dict) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (canonical_json(row) + "\n").encode("utf-8")
            digest.update(encoded)
            handle.write(encoded.decode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locator_fields(row: dict) -> tuple[Path, str]:
    locator = row.get("locator") or {}
    shard_dir = (
        locator.get("shard_dir")
        or row.get("shard_dir")
        or row.get("_shard_dir")
    )
    shard = locator.get("shard") or row.get("shard")
    audio_member = (
        locator.get("audio_member")
        or row.get("audio_member")
        or locator.get("member")
    )
    if not shard_dir or not shard or not audio_member:
        raise ValueError(f"Probe row lacks a complete tar locator: {row}")
    return Path(shard_dir) / shard, str(audio_member)


def load_tar_audio(row: dict) -> tuple[np.ndarray, int, int]:
    tar_path, audio_member = locator_fields(row)
    with tarfile.open(tar_path, "r:") as archive:
        member = archive.getmember(audio_member)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Could not extract {audio_member} from {tar_path}")
        payload = extracted.read()
    audio, sample_rate = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=True,
    )
    channels = int(audio.shape[1])
    mono = audio.mean(axis=1, dtype=np.float32)
    return mono, int(sample_rate), channels


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    resampled = librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=int(source_rate),
        target_sr=int(target_rate),
        res_type="kaiser_best",
    )
    return np.asarray(resampled, dtype=np.float32)


def frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + int(math.ceil((audio.size - frame_length) / hop_length))
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(audio, (0, max(0, padded_length - audio.size)))
    return np.lib.stride_tricks.sliding_window_view(
        padded,
        frame_length,
    )[::hop_length]


def maximum_zero_run_ms(audio: np.ndarray, sample_rate: int) -> float:
    zero_mask = np.abs(audio) <= 1e-8
    if not zero_mask.any():
        return 0.0
    changes = np.diff(np.pad(zero_mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return float((ends - starts).max() * 1000.0 / sample_rate)


def effective_bandwidth_hz(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window)) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    index = int(np.searchsorted(np.cumsum(spectrum), 0.99 * total))
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    return float(frequencies[min(index, frequencies.size - 1)])


def technical_metrics(
    audio: np.ndarray,
    native_sample_rate: int,
    channels: int,
    *,
    target_sample_rate: int,
    frame_length: int,
    hop_length: int,
    active_relative_db: float,
) -> tuple[dict, np.ndarray]:
    finite = bool(np.isfinite(audio).all())
    if not finite:
        raise ValueError("Audio contains non-finite samples")
    audit_audio = resample_audio(audio, native_sample_rate, target_sample_rate)
    frames = frame_signal(audit_audio, frame_length, hop_length)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak_frame_rms = float(frame_rms.max(initial=0.0))
    active_floor = peak_frame_rms * 10.0 ** (active_relative_db / 20.0)
    active_mask = frame_rms >= max(active_floor, 1e-8)
    active_values = frames[active_mask].reshape(-1) if active_mask.any() else np.array([])
    active_rms = (
        float(np.sqrt(np.mean(np.square(active_values, dtype=np.float64))))
        if active_values.size
        else 0.0
    )
    rms = float(np.sqrt(np.mean(np.square(audit_audio, dtype=np.float64))))
    metrics = {
        "finite": finite,
        "native_sample_rate": int(native_sample_rate),
        "native_channels": int(channels),
        "native_duration_seconds": float(audio.size / native_sample_rate),
        "audit_sample_rate": int(target_sample_rate),
        "audit_duration_seconds": float(audit_audio.size / target_sample_rate),
        "peak": float(np.max(np.abs(audit_audio), initial=0.0)),
        "clip_ratio": float(np.mean(np.abs(audit_audio) >= 0.999)),
        "dc_offset": float(np.mean(audit_audio, dtype=np.float64)),
        "rms_dbfs": float(20.0 * math.log10(max(rms, 1e-12))),
        "active_rms_dbfs": float(20.0 * math.log10(max(active_rms, 1e-12))),
        "active_frame_ratio": float(active_mask.mean()),
        "active_seconds": float(
            active_mask.sum() * hop_length / target_sample_rate
        ),
        "max_zero_run_ms": maximum_zero_run_ms(audit_audio, target_sample_rate),
        "effective_bandwidth_hz": effective_bandwidth_hz(
            audit_audio,
            target_sample_rate,
        ),
    }
    return metrics, audit_audio


def technical_gate(metrics: dict, args: argparse.Namespace) -> tuple[bool, list[str], list[str]]:
    hard_reasons = []
    soft_flags = []
    if metrics["native_sample_rate"] < args.min_native_sample_rate:
        hard_reasons.append("native_sample_rate_below_minimum")
    if metrics["native_duration_seconds"] < args.min_duration:
        hard_reasons.append("duration_below_minimum")
    if metrics["active_frame_ratio"] < args.min_active_ratio:
        hard_reasons.append("active_ratio_below_minimum")
    if metrics["active_seconds"] < args.min_active_seconds:
        hard_reasons.append("active_seconds_below_minimum")
    if metrics["active_rms_dbfs"] < args.min_active_rms_dbfs:
        hard_reasons.append("active_rms_below_minimum")
    if metrics["clip_ratio"] > args.max_clip_ratio:
        hard_reasons.append("clipping_above_maximum")
    if abs(metrics["dc_offset"]) > args.max_abs_dc:
        hard_reasons.append("dc_offset_above_maximum")
    if metrics["native_duration_seconds"] > 30.0:
        soft_flags.append("duration_above_review_threshold")
    if metrics["effective_bandwidth_hz"] < 6500.0:
        soft_flags.append("effective_bandwidth_below_review_threshold")
    if metrics["max_zero_run_ms"] > 100.0:
        soft_flags.append("long_zero_run")
    return not hard_reasons, hard_reasons, soft_flags


class DNSMOSP835:
    """Small wrapper around Microsoft's public DNSMOS P.835 ONNX model."""

    def __init__(
        self,
        model_path: Path | None,
        *,
        batch_size: int = 8,
        session: object | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("DNSMOS batch_size must be positive")
        self.batch_size = int(batch_size)
        if session is not None:
            self.session = session
            return
        if model_path is None:
            raise ValueError("model_path is required when session is not provided")
        # ONNX Runtime is an optional, expensive audit-only dependency.
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(
            1,
            int(os.environ.get("OMP_NUM_THREADS", "1")),
        )
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def polynomial_scores(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        sig_score = np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)
        bak_score = np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)
        ovr_score = np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)
        return float(sig_score), float(bak_score), float(ovr_score)

    @staticmethod
    def prepare_segments(audio: np.ndarray) -> np.ndarray:
        required = int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLE_RATE)
        if audio.size == 0:
            raise ValueError("DNSMOS received empty audio")
        repeats = int(math.ceil(required / audio.size))
        tiled = np.tile(audio, repeats) if repeats > 1 else audio
        hop = DNSMOS_SAMPLE_RATE
        segment_count = int(
            math.floor(tiled.size / DNSMOS_SAMPLE_RATE - DNSMOS_INPUT_SECONDS) + 1
        )
        segments = []
        for index in range(max(segment_count, 1)):
            segment = tiled[index * hop : index * hop + required]
            if segment.size < required:
                segment = np.pad(segment, (0, required - segment.size), mode="wrap")
            segments.append(np.asarray(segment, dtype=np.float32))
        return np.stack(segments, axis=0)

    def score_many(self, audios: list[np.ndarray]) -> list[dict]:
        if not audios:
            return []
        per_audio_segments = [self.prepare_segments(audio) for audio in audios]
        segment_counts = [segments.shape[0] for segments in per_audio_segments]
        segments = np.concatenate(per_audio_segments, axis=0)
        raw_batches = []
        for start in range(0, segments.shape[0], self.batch_size):
            batch = segments[start : start + self.batch_size]
            outputs = self.session.run(
                None,
                {"input_1": batch},
            )[0]
            output_array = np.asarray(outputs, dtype=np.float64)
            expected_shape = (batch.shape[0], 3)
            if output_array.shape != expected_shape:
                raise ValueError(
                    "Unexpected DNSMOS ONNX output shape: "
                    f"expected={expected_shape} actual={output_array.shape}"
                )
            raw_batches.append(output_array)

        all_raw_scores = np.concatenate(raw_batches, axis=0)
        results = []
        start = 0
        for segment_count in segment_counts:
            raw_scores = all_raw_scores[start : start + segment_count]
            start += segment_count
            mapped_scores = np.asarray(
                [
                    self.polynomial_scores(
                        float(sig_raw),
                        float(bak_raw),
                        float(ovr_raw),
                    )
                    for sig_raw, bak_raw, ovr_raw in raw_scores
                ],
                dtype=np.float64,
            )
            raw_mean = np.mean(raw_scores, axis=0)
            mapped_mean = np.mean(mapped_scores, axis=0)
            results.append(
                {
                    "status": "ok",
                    "sig_raw": float(raw_mean[0]),
                    "bak_raw": float(raw_mean[1]),
                    "ovrl_raw": float(raw_mean[2]),
                    "sig": float(mapped_mean[0]),
                    "bak": float(mapped_mean[1]),
                    "ovrl": float(mapped_mean[2]),
                    "segment_count": int(segment_count),
                }
            )
        if start != all_raw_scores.shape[0]:
            raise RuntimeError("DNSMOS segment regrouping left unused outputs")
        return results

    def __call__(self, audio: np.ndarray) -> dict:
        return self.score_many([audio])[0]


def quantile_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def probe_decision(
    *,
    technical_hard_pass: bool,
    technical_hard_reasons: list[str],
    dnsmos_bak: float,
    source_bak_p25: float,
) -> dict:
    if not technical_hard_pass:
        route = "exclude_invalid"
        status = "technical_hard_fail"
        reasons = list(technical_hard_reasons)
    elif dnsmos_bak < source_bak_p25:
        route = "clean_candidate"
        status = "low_bak_review_stratum"
        reasons = [
            "dnsmos_bak_below_source_p25",
            "score_is_ranking_not_route_definition",
            "manual_review_may_retain_noisy_speech_target",
        ]
    else:
        route = "clean_candidate"
        status = "review_not_training_ready"
        reasons = [
            "technical_gate_pass",
            "dnsmos_bak_at_or_above_source_p25",
            "manual_stratified_listening_pending",
        ]
    return {
        "route": route,
        "status": status,
        "training_ready": False,
        "reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    if args.python_deps is not None:
        sys.path.append(str(args.python_deps))
    rows = read_jsonl(args.input_jsonl)
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    if args.max_rows:
        rows = rows[: args.max_rows]
    scorer = DNSMOSP835(
        args.dnsmos_model,
        batch_size=args.dnsmos_batch_size,
    )
    scored = []
    scores_by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    valid_bak_by_group: dict[str, list[float]] = defaultdict(list)

    if args.row_chunk_size <= 0:
        raise ValueError("--row-chunk-size must be positive")
    for start in range(0, len(rows), args.row_chunk_size):
        chunk_rows = rows[start : start + args.row_chunk_size]
        prepared = []
        audit_audios = []
        for row in chunk_rows:
            audio, native_sample_rate, channels = load_tar_audio(row)
            metrics, audit_audio = technical_metrics(
                audio,
                native_sample_rate,
                channels,
                target_sample_rate=args.target_sample_rate,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                active_relative_db=args.active_relative_db,
            )
            hard_pass, hard_reasons, soft_flags = technical_gate(metrics, args)
            group = str(
                row.get("probe_family")
                or row.get("provenance_class")
                or row.get("provenance")
                or row.get("dataset")
                or "<missing>"
            )
            prepared.append(
                (
                    row,
                    metrics,
                    hard_pass,
                    hard_reasons,
                    soft_flags,
                    group,
                )
            )
            audit_audios.append(audit_audio)
        dnsmos_rows = scorer.score_many(audit_audios)
        for prepared_row, dnsmos in zip(prepared, dnsmos_rows, strict=True):
            (
                row,
                metrics,
                hard_pass,
                hard_reasons,
                soft_flags,
                group,
            ) = prepared_row
            for metric in ("bak", "sig", "ovrl"):
                scores_by_group[group][metric].append(float(dnsmos[metric]))
            if hard_pass:
                valid_bak_by_group[group].append(float(dnsmos["bak"]))
            updated = dict(row)
            updated["score_schema_version"] = SCHEMA_VERSION
            updated["technical_metrics"] = metrics
            updated["technical_gate"] = {
                "hard_pass": hard_pass,
                "hard_reasons": hard_reasons,
                "soft_flags": soft_flags,
                "speech_music_classifier_status": (
                    "not_available_review_required"
                ),
            }
            updated["dnsmos"] = dnsmos
            updated["_score_group"] = group
            scored.append(updated)
        completed = start + len(chunk_rows)
        if completed % 100 < len(chunk_rows) or completed == len(rows):
            print(f"scored {completed}/{len(rows)}", flush=True)

    missing_valid_groups = [
        group
        for group in scores_by_group
        if not valid_bak_by_group[group]
    ]
    if missing_valid_groups:
        raise ValueError(
            f"no technically valid rows in score groups: {missing_valid_groups}"
        )
    group_thresholds = {
        group: {
            "all_scores": {
                metric: quantile_summary(values)
                for metric, values in sorted(metric_values.items())
            },
            "technical_hard_pass_bak": quantile_summary(
                valid_bak_by_group[group]
            ),
            "promotion_metric": "bak",
            "promotion_threshold_population": "technical_hard_pass_only",
        }
        for group, metric_values in sorted(scores_by_group.items())
    }
    for row in scored:
        threshold = group_thresholds[row.pop("_score_group")][
            "technical_hard_pass_bak"
        ]["p25"]
        row["probe_decision"] = probe_decision(
            technical_hard_pass=bool(row["technical_gate"]["hard_pass"]),
            technical_hard_reasons=list(
                row["technical_gate"]["hard_reasons"]
            ),
            dnsmos_bak=float(row["dnsmos"]["bak"]),
            source_bak_p25=float(threshold),
        )

    scored.sort(key=lambda row: str(row.get("sample_uid") or row.get("uid")))
    output_sha256 = write_jsonl(args.output_jsonl, scored)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_sha256": output_sha256,
        "dnsmos_model": str(args.dnsmos_model),
        "dnsmos_model_sha256": sha256_file(args.dnsmos_model),
        "dnsmos_batch_size": int(args.dnsmos_batch_size),
        "row_chunk_size": int(args.row_chunk_size),
        "input_row_limit": int(args.max_rows),
        "sample_count": len(scored),
        "groups": group_thresholds,
        "decision_counts": {
            status: sum(
                row["probe_decision"]["status"] == status for row in scored
            )
            for status in sorted(
                {row["probe_decision"]["status"] for row in scored}
            )
        },
        "promotion_contract": (
            "Probe decisions are review-only and cannot be consumed by Phase A."
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
