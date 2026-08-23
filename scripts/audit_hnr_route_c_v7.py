#!/usr/bin/env python3
"""Freeze the pre-implementation Route C HNR definition audit.

This script is intentionally read-only with respect to source code and audio.
It records the authoritative Praat definition, the secondary NumPy reference,
the frozen Torch ``raw_cc_v3`` baseline, and the already-opened HNR waveform
pilot.  It does not score new waveforms or authorize generator training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PRAAT_SOURCE_VERSION = "6.1.38"
PRAAT_SOURCE_URLS = {
    "sound_to_pitch": (
        "https://sources.debian.org/data/main/p/praat/6.1.38-1/"
        "fon/Sound_to_Pitch.cpp"
    ),
    "pitch": (
        "https://sources.debian.org/data/main/p/praat/6.1.38-1/fon/Pitch.cpp"
    ),
    "voice_analysis": (
        "https://sources.debian.org/data/main/p/praat/6.1.38-1/"
        "fon/VoiceAnalysis.cpp"
    ),
}

OPERATION_MAPPING = (
    {
        "operation": "AVQI metric input",
        "exact_praat": (
            "Apply the 34 Hz stop-Hann high-pass, then use Praat's sounding "
            "CS blocks or the final 3 s of SV before HNR."
        ),
        "existing_numpy": (
            "get_hnr receives the already prepared AVQI input from the Python "
            "pipeline; its own function does no view preparation."
        ),
        "current_torch": (
            "_prepare applies a non-power-of-two FFT high-pass and RMS "
            "normalization, but raw_hnr has no CS/SV view topology."
        ),
        "mismatch": (
            "Torch filtering geometry and view selection differ from the exact "
            "metric branch."
        ),
        "differentiability_treatment": (
            "Freeze any CS/SV selection indices; retain live waveform samples "
            "and keep the 34 Hz filter inside the metric branch only."
        ),
    },
    {
        "operation": "Pitch-analysis timing",
        "exact_praat": (
            "To Pitch (cc) with floor 75 Hz, ceiling 600 Hz, one period per "
            "window, and default step 1/(4*75) s. At 16 kHz the even raw-CC "
            "reference window is 210 samples."
        ),
        "existing_numpy": "40 ms frames with a 10 ms hop and a 60 Hz floor.",
        "current_torch": (
            "Uses round(16000/75)=213 samples and a 53-sample hop, with frames "
            "starting at sample zero."
        ),
        "mismatch": (
            "NumPy has different timing and floor; Torch has the right nominal "
            "step but not Praat's even window or symmetric frame placement."
        ),
        "differentiability_treatment": (
            "Use exact integer geometry; frame placement is fixed and needs no "
            "surrogate gradient."
        ),
    },
    {
        "operation": "Local centering and intensity",
        "exact_praat": (
            "Subtract one local mean computed across two longest periods; frame "
            "intensity is the central one-period local peak divided by the "
            "global demeaned absolute peak."
        ),
        "existing_numpy": (
            "Subtract each 40 ms frame mean and call frames voiced when frame "
            "RMS exceeds 10% of the maximum frame RMS."
        ),
        "current_torch": (
            "Subtract the entire raw-CC segment mean and use a sigmoid around "
            "local/global peak 0.03."
        ),
        "mismatch": "Both references use different centering and voicing intensity.",
        "differentiability_treatment": (
            "Use Praat geometry for live correlations; detach only the hard "
            "voiced/unvoiced topology."
        ),
    },
    {
        "operation": "Forward cross-correlation",
        "exact_praat": (
            "For every lag, correlate the fixed reference window with a shifted "
            "window and normalize by both live energies."
        ),
        "existing_numpy": (
            "Use overlap-shrinking autocorrelation normalized only by lag-zero "
            "autocorrelation."
        ),
        "current_torch": (
            "Uses fixed-window forward cross-correlation with per-lag energy "
            "normalization."
        ),
        "mismatch": (
            "Torch has the correct correlation family, while its frame origin, "
            "window length, and local mean still differ."
        ),
        "differentiability_treatment": (
            "Keep numerator and both denominator energies live in autograd."
        ),
    },
    {
        "operation": "Candidate discovery",
        "exact_praat": (
            "Keep local correlation maxima above 0.5*0.45, with one unvoiced "
            "candidate and at most 14 voiced candidates after octave-cost "
            "retention."
        ),
        "existing_numpy": "Take one global argmax in the 60--600 Hz lag band.",
        "current_torch": "Take one global argmax in the 75--600 Hz lag band.",
        "mismatch": (
            "Both approximations discard the candidate topology required by "
            "Praat's later path finder."
        ),
        "differentiability_treatment": (
            "Detach local-maximum, top-k, and candidate-index decisions; gather "
            "the selected correlation strength from the live tensor."
        ),
    },
    {
        "operation": "Peak interpolation",
        "exact_praat": (
            "Use a parabolic frequency seed followed by sinc70 maximum "
            "interpolation; reflect strengths above one by reciprocal."
        ),
        "existing_numpy": "No sub-bin interpolation.",
        "current_torch": "Use three-bin parabolic interpolation only.",
        "mismatch": "Neither approximation implements Praat's sinc70 refinement.",
        "differentiability_treatment": (
            "Candidate location may be detached; evaluate a bounded live "
            "interpolated strength at that frozen location."
        ),
    },
    {
        "operation": "Global pitch path",
        "exact_praat": (
            "Viterbi path with silence 0.03, voicing 0.45, octave 0.01, octave "
            "jump 0.35, voiced/unvoiced 0.14, and 0.01/time-step correction."
        ),
        "existing_numpy": "No temporal path model.",
        "current_torch": "No temporal path model; every frame is selected independently.",
        "mismatch": (
            "The selected candidate and voiced-frame set can differ even when "
            "level correlation is high."
        ),
        "differentiability_treatment": (
            "Run path selection on detached scores and retain the chosen state "
            "indices for live strength gathering."
        ),
    },
    {
        "operation": "HNR transform",
        "exact_praat": (
            "For each selected voiced frame, compute 10*log10(r/(1-r)); return "
            "-150/+150 dB outside the 1e-15 endpoint guards."
        ),
        "existing_numpy": (
            "Clip r to [0, 0.999999] and apply the same log-ratio with extra "
            "1e-12 terms."
        ),
        "current_torch": "Clamp r to [1e-4, 0.9999] before the log-ratio.",
        "mismatch": (
            "The central formula agrees, but endpoint behavior and therefore "
            "gradient magnitude differ."
        ),
        "differentiability_treatment": (
            "Preserve the exact forward transform in the observed range and use "
            "a documented straight-through bounded backward near endpoints, "
            "never an arbitrary gradient multiplier."
        ),
    },
    {
        "operation": "Across-frame aggregation",
        "exact_praat": "Arithmetic mean of transformed strengths on selected voiced frames.",
        "existing_numpy": "Median of hard RMS-selected frame HNR values.",
        "current_torch": "Soft intensity/periodicity-weighted mean over all frames.",
        "mismatch": "Both references use a different frame set or reducer.",
        "differentiability_treatment": (
            "Freeze the voiced-frame mask and use an ordinary live arithmetic mean."
        ),
    },
    {
        "operation": "PointProcess in Voice report",
        "exact_praat": (
            "The script constructs a PointProcess for Voice report, but HNR is "
            "read directly from Pitch_getMeanStrength and does not use pulses."
        ),
        "existing_numpy": "No PointProcess.",
        "current_torch": "No PointProcess in HNR.",
        "mismatch": "No HNR-value mismatch; pulse reconstruction is unnecessary.",
        "differentiability_treatment": "Do not add pulse topology to HNR.",
    },
)

REQUIRED_SNIPPETS = {
    "exact_hnr_script": (
        "To Pitch (cc)... 0 75 15 no 0.03 0.45 0.01 0.35 0.14 600",
        "Mean harmonics-to-noise ratio: ",
    ),
    "numpy_source": (
        "def get_hnr(avqi_input, sampling_rate):",
        "frame_len = int(round(0.04 * sampling_rate))",
        "return float(np.median(hnr_values))",
    ),
    "torch_source": (
        'hnr_mode: str = "linear_ac_v2"',
        'if hnr_mode not in {"linear_ac_v2", "raw_cc_v3"}:',
        "def _raw_cc_v3_pitch_features(",
        "hnr = self._weighted_mean(frame_hnr, voicing_weight)",
    ),
    "sound_to_pitch_source": (
        "Pitch_pathFinder (thee.get(), silenceThreshold, voicingThreshold,",
        "r [i] > 0.5 * voicingThreshold",
        "NUMimproveMaximum",
    ),
    "pitch_source": (
        "Pitch_getMeanStrength",
        "Pitch_STRENGTH_UNIT_HARMONICS_NOISE_DB",
        "10.0 * log10 (value / (1.0 - value))",
    ),
    "voice_analysis_source": (
        "Mean harmonics-to-noise ratio:",
        "Pitch_getMeanStrength (pitch, tmin, tmax, Pitch_STRENGTH_UNIT_HARMONICS_NOISE_DB)",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-hnr-script", type=Path, required=True)
    parser.add_argument("--numpy-source", type=Path, required=True)
    parser.add_argument("--torch-source", type=Path, required=True)
    parser.add_argument("--sound-to-pitch-source", type=Path, required=True)
    parser.add_argument("--pitch-source", type=Path, required=True)
    parser.add_argument("--voice-analysis-source", type=Path, required=True)
    parser.add_argument("--old-waveform-report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(name: str, path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS[name] if snippet not in text]
    if missing:
        raise ValueError(f"{name} definition drift at {path}: missing {missing}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def old_waveform_evidence(report: dict[str, Any]) -> dict[str, Any]:
    exact = report["summary"]["aggregates"]["exact"]["hnr"]
    surrogate = report["summary"]["aggregates"]["surrogate"]["hnr"]
    return {
        "decision": report["decision"],
        "exact_rows": exact["rows"],
        "exact_improved_cases": round(exact["improvement_fraction"] * exact["rows"]),
        "exact_median_normalized_gap_reduction": exact[
            "median_normalized_gap_reduction"
        ],
        "surrogate_rows": surrogate["rows"],
        "surrogate_improved_cases": round(
            surrogate["improvement_fraction"] * surrogate["rows"]
        ),
        "surrogate_median_normalized_gap_reduction": surrogate[
            "median_normalized_gap_reduction"
        ],
        "source_report_sha256": sha256_file(Path(report["_source_path"])),
        "selection_or_promotion_reuse_allowed": False,
    }


def markdown_table(rows: tuple[dict[str, str], ...]) -> str:
    headings = (
        "Exact Praat operation",
        "Existing NumPy approximation",
        "Current Torch operation",
        "Mismatch",
        "Differentiability treatment",
    )
    lines = [
        "| " + " | ".join(headings) + " |",
        "|" + "|".join("---" for _ in headings) + "|",
    ]
    for row in rows:
        values = (
            f"{row['operation']}: {row['exact_praat']}",
            row["existing_numpy"],
            row["current_torch"],
            row["mismatch"],
            row["differentiability_treatment"],
        )
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_paths = {
        "exact_hnr_script": args.exact_hnr_script,
        "numpy_source": args.numpy_source,
        "torch_source": args.torch_source,
        "sound_to_pitch_source": args.sound_to_pitch_source,
        "pitch_source": args.pitch_source,
        "voice_analysis_source": args.voice_analysis_source,
    }
    sources = {
        name: validate_source(name, path) for name, path in source_paths.items()
    }
    old_report = json.loads(args.old_waveform_report.read_text(encoding="utf-8"))
    old_report["_source_path"] = str(args.old_waveform_report)

    report = {
        "schema_version": "avqi-route-c-hnr-definition-audit-v7",
        "source_commit": args.source_commit,
        "praat_source_version": PRAAT_SOURCE_VERSION,
        "praat_source_urls": PRAAT_SOURCE_URLS,
        "sources": sources,
        "operation_mapping": OPERATION_MAPPING,
        "old_waveform_evidence": old_waveform_evidence(old_report),
        "finding": (
            "raw_cc_v3 already matches the central HNR transform and correlation "
            "family, but it omits Praat's multi-candidate global pitch path and "
            "hard voiced-frame arithmetic mean; these are the primary v7 targets."
        ),
        "authorization": {
            "candidate_implementation_authorized": True,
            "fresh_bounded_waveform_panel_authorized": False,
            "formal_generator_training_authorized": False,
        },
        "generator_optimizer_steps": 0,
    }
    args.output_dir.mkdir(parents=True)
    json_path = args.output_dir / "hnr_definition_audit.json"
    markdown_path = args.output_dir / "hnr_operation_mapping.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Route C HNR v7 operation mapping\n\n"
        + markdown_table(OPERATION_MAPPING)
        + "\n\n"
        + "Decision: implement only the detached Praat pitch-candidate/path "
        + "topology with live, bounded HNR strengths. Fresh waveform and generator "
        + "training remain unauthorized.\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "avqi-route-c-hnr-definition-audit-receipt-v7",
        "source_commit": args.source_commit,
        "report": str(json_path.resolve()),
        "report_sha256": sha256_file(json_path),
        "operation_mapping": str(markdown_path.resolve()),
        "operation_mapping_sha256": sha256_file(markdown_path),
        "generator_optimizer_steps": 0,
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
