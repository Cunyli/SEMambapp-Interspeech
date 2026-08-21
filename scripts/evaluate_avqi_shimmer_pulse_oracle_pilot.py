#!/usr/bin/env python3
"""Isolate Route C Shimmer gradients with detached exact Praat pulses.

This is a historical, SV-only waveform diagnostic.  Exact Praat supplies a
piecewise-constant pulse topology, while PyTorch measures the live asymmetric
Hann-RMS amplitude tier and supplies the waveform gradient.  The script never
loads or updates the enhancement generator and cannot promote AVQI-T2
training.  A fresh speaker-disjoint panel and a deployable pulse locator remain
mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator


SAMPLE_RATE = 16_000
SV_METRIC_SAMPLES = 3 * SAMPLE_RATE
COMPONENTS = ("shimmer_percent", "shimmer_db")
ALPHA_GRID = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
CALIBRATION_SPEAKERS = frozenset({"ÄHH10", "PD08"})
HOLDOUT_SPEAKERS = frozenset({"V55", "ÄHH28"})
MATERIAL_GAP_THRESHOLD = 0.02
PROMOTION_REDUCTION_THRESHOLD = 0.02
REALISTIC_PATTERN = re.compile(
    r"^(?P<case>[0-9]+)__(?P<speaker>.+?)__"
    r"(?P<condition>rir_only|snr30|snr20)__"
    r"(?P<view>cs|sv)__(?P<suffix>"
    r"input|target_clean|B0_250|S3_500|S3_2000)\.wav$"
)
IDENTITY_PATTERN = re.compile(
    r"^(?P<case>[0-9]+)__(?P<speaker>.+?)__"
    r"(?P<condition>clean|snr10)__(?P<view>cs|sv)__"
    r"(?P<suffix>input|target_clean|B_pair_500|B_sv_match_500)\.wav$"
)
EXACT_SCORER = r"""
import json
import sys

import numpy as np

parselmouth_root = sys.argv[1]
if parselmouth_root:
    sys.path.insert(0, parselmouth_root)
import parselmouth
from parselmouth.praat import call

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    sound = parselmouth.Sound(item["path"])
    sound = call(sound, "Filter (stop Hann band)", 0, 34, 0.1)
    duration = float(call(sound, "Get total duration"))
    if duration > 3.0:
        sound = call(
            sound,
            "Extract part",
            duration - 3.0,
            duration,
            "rectangular",
            1.0,
            "no",
        )
    point_process = call(sound, "To PointProcess (periodic, cc)", 50, 400)
    pulse_count = int(call(point_process, "Get number of points"))
    pulses = []
    if request["include_pulses"]:
        pulses = [
            (
                float(call(point_process, "Get time from index", index))
                - float(sound.x1)
            )
            / float(sound.dx)
            for index in range(1, pulse_count + 1)
        ]
    percent = 100.0 * float(
        call(
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
    )
    db = float(
        call(
            [sound, point_process],
            "Get shimmer (local_dB)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
    )
    rows.append(
        {
            "id": item["id"],
            "shimmer_percent": percent,
            "shimmer_db": db,
            "pulse_count": pulse_count,
            "pulse_positions_samples": pulses,
            "metric_sample_count": int(sound.n_samples),
        }
    )
print(
    "AVQI_SHIMMER_EXACT_JSON="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realistic-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument(
        "--parselmouth-root",
        type=Path,
        help="Optional isolated Parselmouth site-packages root for exact Python.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(root: Path) -> dict[int, dict[str, str]]:
    path = root / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing listening-pack metadata: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    output = {int(row["order"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate listening-pack order in {path}")
    return output


def read_waveform(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"expected nonempty mono 16 kHz waveform: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite waveform: {path}")
    return audio


def load_records(
    dataset: str,
    root: Path,
    pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing listening pack: {root}")
    metadata = read_metadata(root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.wav")):
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(f"unexpected waveform name: {path}")
        fields = match.groupdict()
        if fields["view"] != "sv":
            continue
        speaker = fields["speaker"]
        if speaker not in CALIBRATION_SPEAKERS | HOLDOUT_SPEAKERS:
            continue
        case_number = int(fields["case"])
        metadata_row = metadata[case_number]
        if metadata_row["speaker_id"] != speaker:
            raise ValueError(f"metadata speaker mismatch: {path}")
        split = (
            "calibration"
            if speaker in CALIBRATION_SPEAKERS
            else "holdout"
        )
        records.append(
            {
                "id": f"{dataset}:{path.stem}",
                "dataset": dataset,
                "case_number": case_number,
                "speaker": speaker,
                "condition": fields["condition"],
                "view": fields["view"],
                "suffix": fields["suffix"],
                "label": metadata_row["label"],
                "sample_group": metadata_row["sample_group"],
                "split": split,
                "path": path.resolve(),
                "sha256": sha256_file(path),
                "audio": read_waveform(path),
            }
        )
    return records


def run_exact_batch(
    items: list[dict[str, str]],
    exact_python: Path,
    parselmouth_root: Path | None,
    *,
    include_pulses: bool,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(exact_python),
            "-c",
            EXACT_SCORER,
            "" if parselmouth_root is None else str(parselmouth_root),
        ],
        input=json.dumps(
            {"items": items, "include_pulses": include_pulses},
            sort_keys=True,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "AVQI_SHIMMER_EXACT_JSON="
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(marker)
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"exact Shimmer scorer emitted {len(lines)} JSON records"
        )
    payload = json.loads(lines[0][len(marker) :])
    if len(payload["rows"]) != len(items):
        raise ValueError("exact Shimmer row count drift")
    return payload


def proxy_shimmer(
    estimator: PraatDifferentiableAVQIComponentEstimator,
    waveform: torch.Tensor,
    pulse_positions: list[float],
) -> torch.Tensor:
    pulses = waveform.new_tensor(pulse_positions)
    return estimator.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_sample_count=SV_METRIC_SAMPLES,
    )


def rms(audio: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))


def db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-15) / max(denominator, 1e-15))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right)) / max(denominator, 1e-15)


def band_energy(audio: np.ndarray, upper_hz: float) -> float:
    spectrum = np.fft.rfft(audio.astype(np.float64))
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / SAMPLE_RATE)
    return float(np.square(np.abs(spectrum[frequencies <= upper_hz])).sum())


def safety_metrics(base: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    residual = candidate.astype(np.float64) - base.astype(np.float64)
    base_rms = rms(base)
    candidate_rms = rms(candidate)
    low_before = band_energy(base, 80.0)
    low_after = band_energy(candidate, 80.0)
    return {
        "residual_rms_db": db_ratio(rms(residual), base_rms),
        "cosine_similarity": cosine_similarity(base, candidate),
        "rms_change_db": db_ratio(candidate_rms, base_rms),
        "low_0_80hz_energy_change_db": 10.0
        * math.log10(max(low_after, 1e-30) / max(low_before, 1e-30)),
        "peak_absolute": float(np.max(np.abs(candidate))),
        "clipping_samples": int(np.sum(np.abs(candidate) >= 1.0)),
    }


def safe_output_name(record: dict[str, Any]) -> str:
    raw = (
        f"{record['dataset']}__{record['case_number']:02d}__"
        f"{record['speaker']}__{record['suffix']}__oracle_step.wav"
    )
    return re.sub(r"[^0-9A-Za-z._ÄÖÅäöåÜüÉé_-]", "_", raw)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def aggregate(
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    material = [row for row in selected if row["material_percent_gap"]]
    return {
        "rows": len(selected),
        "speakers": sorted({row["speaker"] for row in selected}),
        "material_rows": len(material),
        "percent_median_normalized_gap_reduction_all": median(
            [row["exact_percent_normalized_gap_reduction"] for row in selected]
        ),
        "percent_improvement_rate_all": float(
            np.mean(
                [row["exact_percent_normalized_gap_reduction"] > 0.0 for row in selected]
            )
        ),
        "percent_median_normalized_gap_reduction_material": median(
            [row["exact_percent_normalized_gap_reduction"] for row in material]
        ),
        "percent_improvement_rate_material": (
            float(
                np.mean(
                    [
                        row["exact_percent_normalized_gap_reduction"] > 0.0
                        for row in material
                    ]
                )
            )
            if material
            else None
        ),
        "db_median_normalized_gap_reduction_material": median(
            [row["exact_db_normalized_gap_reduction"] for row in material]
        ),
        "identity_zero_gap_rows": sum(
            row["exact_percent_gap_before"] <= 1e-12 for row in selected
        ),
        "identity_zero_gap_rows_unchanged": sum(
            row["exact_percent_gap_before"] <= 1e-12
            and row["exact_percent_gap_after"] <= 1e-12
            for row in selected
        ),
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in selected
        ),
        "maximum_residual_rms_db": max(
            row["residual_rms_db"] for row in selected
        ),
        "maximum_abs_low_0_80hz_energy_change_db": max(
            abs(row["low_0_80hz_energy_change_db"]) for row in selected
        ),
        "clipping_samples": sum(row["clipping_samples"] for row in selected),
    }


def build_markdown(report: dict[str, Any]) -> str:
    holdout = report["aggregates"]["holdout"]
    return "\n".join(
        [
            "# Route C Shimmer exact-pulse isolation pilot",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "This historical SV-only diagnostic freezes exact Praat pulse "
            "positions and backpropagates only through the PyTorch amplitude "
            "tier. It does not authorize generator training.",
            "",
            "| Holdout result | Value |",
            "|---|---:|",
            f"| Speakers | {', '.join(holdout['speakers'])} |",
            f"| Material cases | {holdout['material_rows']} |",
            "| Median exact Shimmer % normalized gap reduction | "
            f"{holdout['percent_median_normalized_gap_reduction_material']:.6f} |",
            "| Exact Shimmer % material improvement rate | "
            f"{holdout['percent_improvement_rate_material']:.3f} |",
            "| Minimum waveform cosine | "
            f"{holdout['minimum_cosine_similarity']:.9f} |",
            "| Maximum residual RMS | "
            f"{holdout['maximum_residual_rms_db']:.3f} dB |",
            "",
            "The result isolates pulse localization as the remaining v5 "
            "bottleneck. A fresh CS/SV panel with all six exact components is "
            "still required before AVQI-T2 training.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    if args.parselmouth_root is not None and not args.parselmouth_root.is_dir():
        raise FileNotFoundError(
            f"missing Parselmouth root: {args.parselmouth_root}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    output_root = args.output_dir / "outputs"
    waveform_root = output_root / "waveforms"
    waveform_root.mkdir(parents=True)

    records = [
        *load_records(
            "realistic",
            args.realistic_root,
            REALISTIC_PATTERN,
        ),
        *load_records(
            "identity",
            args.identity_root,
            IDENTITY_PATTERN,
        ),
    ]
    if len(records) != 36:
        raise ValueError(f"expected 36 frozen SV waveforms, found {len(records)}")
    calibration_speakers = {
        row["speaker"] for row in records if row["split"] == "calibration"
    }
    holdout_speakers = {
        row["speaker"] for row in records if row["split"] == "holdout"
    }
    if calibration_speakers != CALIBRATION_SPEAKERS:
        raise ValueError("calibration speaker contract drift")
    if holdout_speakers != HOLDOUT_SPEAKERS:
        raise ValueError("holdout speaker contract drift")
    if calibration_speakers & holdout_speakers:
        raise ValueError("speaker leakage in oracle-topology diagnostic")

    exact_before = run_exact_batch(
        [
            {"id": row["id"], "path": str(row["path"])}
            for row in records
        ],
        args.exact_python,
        args.parselmouth_root,
        include_pulses=True,
    )
    exact_index = {row["id"]: row for row in exact_before["rows"]}
    for record in records:
        exact = exact_index[record["id"]]
        if exact["metric_sample_count"] != min(
            record["audio"].size,
            SV_METRIC_SAMPLES,
        ):
            raise ValueError(f"SV metric crop drift: {record['path']}")
        if exact["pulse_count"] < 3:
            raise ValueError(f"insufficient exact pulses: {record['path']}")
        record["exact"] = np.array(
            [exact["shimmer_percent"], exact["shimmer_db"]],
            dtype=np.float64,
        )
        record["pulse_positions"] = exact["pulse_positions_samples"]

    estimator = PraatDifferentiableAVQIComponentEstimator(
        peak_mode="hard",
        shimmer_mode="praat_pulse_chain_v5",
    ).eval()
    for record in records:
        with torch.inference_mode():
            record["proxy"] = proxy_shimmer(
                estimator,
                torch.from_numpy(record["audio"]),
                record["pulse_positions"],
            ).numpy()

    unique_calibration = {
        record["sha256"]: record
        for record in records
        if record["split"] == "calibration"
    }
    calibration_exact = np.stack(
        [record["exact"] for record in unique_calibration.values()]
    )
    target_scale = np.maximum(calibration_exact.std(axis=0), 1e-8)
    target_index = {
        (record["dataset"], record["case_number"], record["speaker"]): record
        for record in records
        if record["suffix"] == "target_clean"
    }
    cases = [record for record in records if record["suffix"] != "target_clean"]
    if len(cases) != 28:
        raise ValueError(f"expected 28 non-target cases, found {len(cases)}")

    output_items: list[dict[str, str]] = []
    for record in cases:
        target = target_index[
            (record["dataset"], record["case_number"], record["speaker"])
        ]
        waveform = torch.from_numpy(record["audio"].copy()).requires_grad_(True)
        current_proxy = proxy_shimmer(
            estimator,
            waveform,
            record["pulse_positions"],
        )
        target_proxy = torch.from_numpy(target["proxy"]).to(waveform)
        loss = (
            (current_proxy[0] - target_proxy[0]) / float(target_scale[0])
        ).square()
        gradient = torch.autograd.grad(loss, waveform)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite oracle gradient: {record['path']}")
        gradient_rms = gradient.square().mean().sqrt()
        base_rms = waveform.detach().square().mean().sqrt()
        base_db_gap = abs(float(current_proxy[1] - target_proxy[1]))
        candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for alpha in ALPHA_GRID:
            if alpha == 0.0 or float(gradient_rms) <= 1e-15:
                candidate_tensor = waveform.detach()
            else:
                candidate_tensor = waveform.detach() - (
                    alpha * base_rms * gradient / gradient_rms
                )
            if not torch.isfinite(candidate_tensor).all():
                continue
            if float(candidate_tensor.abs().max()) >= 1.0:
                continue
            with torch.inference_mode():
                candidate_proxy = proxy_shimmer(
                    estimator,
                    candidate_tensor,
                    record["pulse_positions"],
                )
            candidate_db_gap = abs(float(candidate_proxy[1] - target_proxy[1]))
            if candidate_db_gap > base_db_gap + 1e-5:
                continue
            candidates.append(
                (
                    abs(float(candidate_proxy[0] - target_proxy[0])),
                    alpha,
                    candidate_tensor.numpy(),
                    candidate_proxy.numpy(),
                )
            )
        if not candidates:
            raise RuntimeError(f"no safe line-search candidate: {record['path']}")
        _, selected_alpha, selected_audio, _ = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        output_path = waveform_root / safe_output_name(record)
        sf.write(
            output_path,
            selected_audio,
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        stored_audio = read_waveform(output_path)
        with torch.inference_mode():
            stored_proxy = proxy_shimmer(
                estimator,
                torch.from_numpy(stored_audio),
                record["pulse_positions"],
            ).numpy()
        record["target"] = target
        record["loss"] = float(loss)
        record["gradient_rms"] = float(gradient_rms)
        record["selected_alpha"] = selected_alpha
        record["selected_audio"] = stored_audio
        record["selected_proxy"] = stored_proxy
        record["output_path"] = output_path.resolve()
        record["safety"] = safety_metrics(record["audio"], stored_audio)
        output_items.append(
            {"id": record["id"], "path": str(output_path.resolve())}
        )

    exact_after = run_exact_batch(
        output_items,
        args.exact_python,
        args.parselmouth_root,
        include_pulses=False,
    )
    if (
        exact_after["parselmouth_version"]
        != exact_before["parselmouth_version"]
        or exact_after["praat_version"] != exact_before["praat_version"]
    ):
        raise ValueError("exact scorer version drift within pilot")
    exact_after_index = {row["id"]: row for row in exact_after["rows"]}

    csv_rows: list[dict[str, Any]] = []
    for record in cases:
        target = record["target"]
        after_item = exact_after_index[record["id"]]
        exact_after_components = np.array(
            [after_item["shimmer_percent"], after_item["shimmer_db"]],
            dtype=np.float64,
        )
        percent_gap_before = abs(float(record["exact"][0] - target["exact"][0]))
        percent_gap_after = abs(
            float(exact_after_components[0] - target["exact"][0])
        )
        db_gap_before = abs(float(record["exact"][1] - target["exact"][1]))
        db_gap_after = abs(float(exact_after_components[1] - target["exact"][1]))
        material = percent_gap_before / target_scale[0] > MATERIAL_GAP_THRESHOLD
        row: dict[str, Any] = {
            "dataset": record["dataset"],
            "case_number": record["case_number"],
            "split": record["split"],
            "speaker": record["speaker"],
            "label": record["label"],
            "sample_group": record["sample_group"],
            "condition": record["condition"],
            "view": record["view"],
            "suffix": record["suffix"],
            "input_path": str(record["path"]),
            "output_path": str(record["output_path"]),
            "input_sha256": record["sha256"],
            "output_sha256": sha256_file(record["output_path"]),
            "exact_pulse_count": len(record["pulse_positions"]),
            "selected_alpha": record["selected_alpha"],
            "gradient_rms": record["gradient_rms"],
            "proxy_loss_before": record["loss"],
            "proxy_percent_before": float(record["proxy"][0]),
            "proxy_percent_target": float(target["proxy"][0]),
            "proxy_percent_after": float(record["selected_proxy"][0]),
            "proxy_db_before": float(record["proxy"][1]),
            "proxy_db_target": float(target["proxy"][1]),
            "proxy_db_after": float(record["selected_proxy"][1]),
            "oracle_forward_percent_abs_error_before": abs(
                float(record["proxy"][0] - record["exact"][0])
            ),
            "oracle_forward_db_abs_error_before": abs(
                float(record["proxy"][1] - record["exact"][1])
            ),
            "exact_percent_before": float(record["exact"][0]),
            "exact_percent_target": float(target["exact"][0]),
            "exact_percent_after": float(exact_after_components[0]),
            "exact_percent_gap_before": percent_gap_before,
            "exact_percent_gap_after": percent_gap_after,
            "exact_percent_normalized_gap_reduction": (
                percent_gap_before - percent_gap_after
            )
            / target_scale[0],
            "material_percent_gap": int(material),
            "exact_db_before": float(record["exact"][1]),
            "exact_db_target": float(target["exact"][1]),
            "exact_db_after": float(exact_after_components[1]),
            "exact_db_gap_before": db_gap_before,
            "exact_db_gap_after": db_gap_after,
            "exact_db_normalized_gap_reduction": (
                db_gap_before - db_gap_after
            )
            / target_scale[1],
            **record["safety"],
        }
        csv_rows.append(row)

    aggregates = {
        split: aggregate(csv_rows, split)
        for split in ("calibration", "holdout")
    }
    holdout = aggregates["holdout"]
    isolation_pass = (
        holdout["material_rows"] >= 5
        and holdout["percent_median_normalized_gap_reduction_material"]
        >= PROMOTION_REDUCTION_THRESHOLD
        and holdout["percent_improvement_rate_material"] >= 0.8
        and holdout["db_median_normalized_gap_reduction_material"] >= 0.0
        and holdout["identity_zero_gap_rows"]
        == holdout["identity_zero_gap_rows_unchanged"]
        and holdout["minimum_cosine_similarity"] >= 0.99999
        and holdout["maximum_residual_rms_db"] <= -50.0
        and holdout["maximum_abs_low_0_80hz_energy_change_db"] <= 0.1
        and holdout["clipping_samples"] == 0
    )
    decision = (
        "PASS_ORACLE_TOPOLOGY_ISOLATION_ONLY"
        if isolation_pass
        else "FAIL_ORACLE_TOPOLOGY_ISOLATION"
    )
    report = {
        "decision": decision,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "promotion_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "route": "C_shimmer_detached_exact_pulse_topology",
        "panel_status": "historical_sv_only_nonfinal",
        "sample_rate": SAMPLE_RATE,
        "sv_metric_order": "whole-waveform high-pass, then final 3 seconds",
        "calibration_speakers": sorted(CALIBRATION_SPEAKERS),
        "holdout_speakers": sorted(HOLDOUT_SPEAKERS),
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selection": (
            "per-waveform surrogate-only line search with fixed base pulse "
            "topology and Shimmer-dB non-worsening constraint"
        ),
        "material_gap_threshold_normalized": MATERIAL_GAP_THRESHOLD,
        "promotion_reduction_threshold": PROMOTION_REDUCTION_THRESHOLD,
        "target_scale_from_unique_calibration_waveforms": {
            component: float(target_scale[index])
            for index, component in enumerate(COMPONENTS)
        },
        "exact_scorer": {
            "python": str(args.exact_python.resolve()),
            "parselmouth_root": (
                None
                if args.parselmouth_root is None
                else str(args.parselmouth_root.resolve())
            ),
            "parselmouth_version": exact_before["parselmouth_version"],
            "praat_version": exact_before["praat_version"],
            "highpass": "Filter (stop Hann band), 0, 34, 0.1",
            "point_process": "To PointProcess (periodic, cc), 50, 400",
            "shimmer_arguments": [0, 0, 0.0001, 0.02, 1.3, 1.6],
        },
        "oracle_forward_error": {
            "percent_median_absolute": median(
                [row["oracle_forward_percent_abs_error_before"] for row in csv_rows]
            ),
            "percent_maximum_absolute": max(
                row["oracle_forward_percent_abs_error_before"] for row in csv_rows
            ),
            "db_median_absolute": median(
                [row["oracle_forward_db_abs_error_before"] for row in csv_rows]
            ),
            "db_maximum_absolute": max(
                row["oracle_forward_db_abs_error_before"] for row in csv_rows
            ),
        },
        "aggregates": aggregates,
        "limitations": [
            "Exact Praat pulse positions are an oracle isolation tool, not a deployable GPU pulse locator.",
            "The panel is historical, SV-only, and has two holdout speakers.",
            "CS, all-six-component non-target gates, denoising metrics, and a fresh external panel are absent.",
            "The result cannot promote generator training even if the isolation gate passes.",
        ],
        "input_sha256": {
            record["id"]: record["sha256"] for record in records
        },
    }
    rows_path = output_root / "cases.csv"
    report_path = output_root / "report.json"
    markdown_path = output_root / "README.md"
    write_csv(rows_path, csv_rows)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
