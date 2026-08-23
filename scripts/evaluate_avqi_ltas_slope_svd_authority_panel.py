#!/usr/bin/env python3
"""Run a sealed SVD external authority panel for Route C LTAS slope.

The panel is selected from patient Saarbruecken Voice Database sessions using
metadata only.  ``seal`` writes canonical 16 kHz CS/SV waveforms and freezes
their hashes before any exact LTAS value is requested.  ``score`` requires the
seal hash, applies the authoritative Praat CS/SV preprocessing, and evaluates
the already-frozen differentiable candidate and gate contract.  This script
never changes the production gate and never runs a generator optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import stats

from model.avqi_components import PraatDifferentiableAVQIComponentEstimator
from scripts.evaluate_avqi_ltas_slope_gate_alignment import (
    AUTHORITY_RATIO_RANGE,
    CURRENT_ABSOLUTE_LOWPASS_MIN,
    DIRECTION_AGREEMENT_MIN,
    EXACT_MATERIAL_DISTANCE_MIN,
    INVARIANCE_DISTANCE_MAX,
    SLOPE_INDEX,
    VARIANT_NAMES,
    exact_relative_gate,
    load_predictor,
    predict_slope,
    summarize_mode,
    waveform_variants,
)
from scripts.evaluate_avqi_ltas_slope_lowpass_authority import (
    SAMPLE_RATE,
    read_rows,
    sha256_file,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V9_PANEL_ROWS = (
    ("1347", "360", "female"),
    ("1662", "1275", "female"),
    ("1413", "721", "female"),
    ("1930", "1631", "female"),
    ("1739", "1388", "female"),
    ("1475", "1721", "female"),
    ("1359", "450", "female"),
    ("1652", "1265", "female"),
    ("1727", "1379", "female"),
    ("1383", "627", "female"),
    ("1947", "1648", "female"),
    ("1726", "1378", "female"),
    ("1669", "1283", "male"),
    ("1693", "1315", "male"),
    ("1485", "891", "male"),
    ("1411", "719", "male"),
    ("1991", "1692", "male"),
    ("1636", "1250", "male"),
    ("1416", "723", "male"),
    ("1749", "1398", "male"),
    ("1922", "1621", "male"),
    ("1867", "1560", "male"),
    ("1426", "814", "male"),
    ("1882", "1576", "male"),
)
V9_PANEL_SPEAKERS = frozenset(row[0] for row in V9_PANEL_ROWS)
PRIMARY_ROWS = (
    ("1877", "1784", "female"),
    ("1382", "565", "female"),
    ("1525", "933", "female"),
    ("1891", "1591", "female"),
    ("1302", "105", "female"),
    ("1459", "859", "female"),
    ("1377", "562", "female"),
    ("1545", "1049", "female"),
    ("1615", "1227", "female"),
    ("1894", "1594", "female"),
    ("1395", "668", "female"),
    ("2007", "1863", "female"),
    ("1941", "1647", "male"),
    ("1865", "1555", "male"),
    ("1594", "1197", "male"),
    ("1969", "1670", "male"),
    ("1449", "850", "male"),
    ("1448", "849", "male"),
    ("1446", "1606", "male"),
    ("1495", "918", "male"),
    ("1805", "2384", "male"),
    ("1741", "1389", "male"),
    ("2000", "1716", "male"),
    ("1486", "892", "male"),
)
RESERVE_ROWS = (
    ("1301", "101", "female"),
    ("1438", "826", "female"),
    ("1516", "924", "female"),
    ("1849", "1502", "female"),
    ("1923", "1624", "female"),
    ("1322", "143", "female"),
    ("1603", "2548", "male"),
    ("1872", "1565", "male"),
)
PANEL_ROWS = PRIMARY_ROWS + RESERVE_ROWS
PANEL_SELECTION_RULE = (
    "patient paired sessions only; exclude every v9 speaker; raw mono SV >= "
    "1.0 s and CS >= 3.0 s; one minimum numeric eligible session per SVD "
    "speaker; SHA256(speaker_id:session_id) rank; first 12 per sex are primary; "
    "next up to 6 per sex are ordered same-sex reserves"
)
PRIMARY_SPEAKERS = 24
RESERVE_SPEAKERS = len(RESERVE_ROWS)
SEALED_SPEAKERS = PRIMARY_SPEAKERS + RESERVE_SPEAKERS
PANEL_VIEWS = ("cs", "sv")
PRIMARY_CASES = PRIMARY_SPEAKERS * len(PANEL_VIEWS)
SEALED_CASES = SEALED_SPEAKERS * len(PANEL_VIEWS)
EXPECTED_SELECTED_EXACT_ROWS = PRIMARY_CASES * len(VARIANT_NAMES)
SV_DURATION_MIN_SECONDS = 1.0
CS_DURATION_MIN_SECONDS = 3.0
PRIMARY_PER_SEX = 12
RESERVE_PER_SEX_TARGET = 6
LEVEL_SPEARMAN_GATE = 0.70
DELTA_SPEARMAN_GATE = 0.60
NORMALIZED_MAE_GATE = 0.50
CALIBRATION_SLOPE_RANGE = (0.75, 1.25)
EXTERNAL_COVERAGE_GATE = 0.99
COMPONENT_INPUT_GRADIENT_MIN = 1e-10
COMPONENT_INPUT_GRADIENT_MAX = 1e4
TRAINING_SEGMENT_SAMPLES = 3 * SAMPLE_RATE
EXACT_MARKER = "AVQI_LTAS_SVD_EXACT_JSON="

EXACT_SCORER = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import parselmouth
from avqi_code.python_version import read_and_resample_signal
from avqi_code.praat_version import (
    get_slope,
    get_voiced_segments,
    highpass_filter,
    length_normalize_sv,
)

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    row = {"id": item["id"], "view": item["view"]}
    try:
        signal = read_and_resample_signal(item["path"], 16000)
        signal = highpass_filter(signal, 16000)
        if item["view"] == "sv":
            avqi_input = length_normalize_sv(signal, 16000)
        elif item["view"] == "cs":
            avqi_input = get_voiced_segments(signal, 16000)
        else:
            raise ValueError("unsupported view: " + item["view"])
        row.update(
            {
                "scoring_status": "ok",
                "slope": float(get_slope(avqi_input, 16000)),
                "preprocessed_samples": int(len(avqi_input)),
                "error_type": "",
                "error_message": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "scoring_status": "error",
                "slope": None,
                "preprocessed_samples": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
        )
    rows.append(row)
print(
    "AVQI_LTAS_SVD_EXACT_JSON="
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


@dataclass(frozen=True)
class PanelCase:
    speaker_id: str
    session_id: str
    sex: str
    diagnosis: str
    sv_path: Path
    cs_path: Path
    selection_role: str
    selection_rank_within_sex: int
    sv_duration_seconds: float
    cs_duration_seconds: float

    @property
    def panel_speaker_id(self) -> str:
        return f"SVD:{self.speaker_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seal", "score"), required=True)
    parser.add_argument("--sv-metadata", type=Path, required=True)
    parser.add_argument("--sv-metadata-sha256", required=True)
    parser.add_argument("--cs-metadata", type=Path, required=True)
    parser.add_argument("--cs-metadata-sha256", required=True)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--cs-root", type=Path, required=True)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--python-version-sha256", required=True)
    parser.add_argument("--praat-version-sha256", required=True)
    parser.add_argument("--highpass-praat-sha256", required=True)
    parser.add_argument("--sv-length-praat-sha256", required=True)
    parser.add_argument("--cs-voiced-praat-sha256", required=True)
    parser.add_argument("--slope-praat-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-seal-sha256", default="")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", default="")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def validate_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def exact_source_hashes(args: argparse.Namespace) -> dict[str, str]:
    code = args.avqi_code_root / "avqi_code"
    paths = {
        "python_version.py": (
            code / "python_version.py",
            args.python_version_sha256,
        ),
        "praat_version.py": (
            code / "praat_version.py",
            args.praat_version_sha256,
        ),
        "praat_scripts/highpass_filter.praat": (
            code / "praat_scripts" / "highpass_filter.praat",
            args.highpass_praat_sha256,
        ),
        "praat_scripts/length_normalize_sv.praat": (
            code / "praat_scripts" / "length_normalize_sv.praat",
            args.sv_length_praat_sha256,
        ),
        "praat_scripts/voiced_segment_extraction.praat": (
            code / "praat_scripts" / "voiced_segment_extraction.praat",
            args.cs_voiced_praat_sha256,
        ),
        "praat_scripts/slope.praat": (
            code / "praat_scripts" / "slope.praat",
            args.slope_praat_sha256,
        ),
    }
    return {
        name: validate_hash(path, expected, f"exact {name}")
        for name, (path, expected) in paths.items()
    }


def select_panel_cases(args: argparse.Namespace) -> list[PanelCase]:
    validate_hash(args.sv_metadata, args.sv_metadata_sha256, "SVD SV metadata")
    validate_hash(args.cs_metadata, args.cs_metadata_sha256, "SVD CS metadata")
    sv_rows = {row["session_id"]: row for row in read_csv(args.sv_metadata)}
    cs_rows = {row["session_id"]: row for row in read_csv(args.cs_metadata)}
    by_speaker: dict[str, list[tuple[str, str, float, float]]] = {}
    for session_id in set(sv_rows) & set(cs_rows):
        sv_row = sv_rows[session_id]
        cs_row = cs_rows[session_id]
        if sv_row["health status"] != "1" or cs_row["health status"] != "1":
            continue
        if sv_row["speaker id"] != cs_row["speaker id"]:
            raise ValueError(f"SVD metadata speaker mismatch: {session_id}")
        if sv_row["gender"] != cs_row["gender"]:
            raise ValueError(f"SVD metadata sex mismatch: {session_id}")
        if sv_row["speaker id"] in V9_PANEL_SPEAKERS:
            continue
        sv_path = args.sv_root / sv_row["filename"]
        cs_path = args.cs_root / cs_row["filename"]
        if not sv_path.is_file() or not cs_path.is_file():
            continue
        sv_info = sf.info(sv_path)
        cs_info = sf.info(cs_path)
        if (
            sv_info.frames <= 0
            or cs_info.frames <= 0
            or sv_info.channels != 1
            or cs_info.channels != 1
        ):
            continue
        sv_duration = sv_info.frames / sv_info.samplerate
        cs_duration = cs_info.frames / cs_info.samplerate
        if (
            sv_duration < SV_DURATION_MIN_SECONDS
            or cs_duration < CS_DURATION_MIN_SECONDS
        ):
            continue
        by_speaker.setdefault(sv_row["speaker id"], []).append(
            (session_id, sv_row["gender"], sv_duration, cs_duration)
        )
    one_session = [
        (speaker_id, *min(rows, key=lambda value: int(value[0])))
        for speaker_id, rows in by_speaker.items()
    ]
    one_session.sort(
        key=lambda value: hashlib.sha256(
            f"{value[0]}:{value[1]}".encode()
        ).hexdigest()
    )
    by_sex = {
        sex: [row for row in one_session if row[2] == sex]
        for sex in ("female", "male")
    }
    derived_primary = tuple(
        (row[0], row[1], row[2])
        for sex in ("female", "male")
        for row in by_sex[sex][:PRIMARY_PER_SEX]
    )
    derived_reserve = tuple(
        (row[0], row[1], row[2])
        for sex in ("female", "male")
        for row in by_sex[sex][
            PRIMARY_PER_SEX : PRIMARY_PER_SEX + RESERVE_PER_SEX_TARGET
        ]
    )
    if derived_primary != PRIMARY_ROWS:
        raise ValueError("SVD metadata-only primary derivation differs from frozen rows")
    if derived_reserve != RESERVE_ROWS:
        raise ValueError("SVD metadata-only reserve derivation differs from frozen rows")
    derived_by_key = {
        (row[0], row[1]): row for rows in by_sex.values() for row in rows
    }
    selected: list[PanelCase] = []
    observed_speakers: set[str] = set()
    role_rows = (("primary", PRIMARY_ROWS), ("reserve", RESERVE_ROWS))
    rank_by_role_and_sex: dict[tuple[str, str], int] = {}
    for role, frozen_rows in role_rows:
        for expected_speaker, session_id, expected_sex in frozen_rows:
            rank_key = (role, expected_sex)
            rank_by_role_and_sex[rank_key] = rank_by_role_and_sex.get(rank_key, 0) + 1
            rank = rank_by_role_and_sex[rank_key]
            if (expected_speaker, session_id) not in derived_by_key:
                raise ValueError(
                    f"SVD frozen {role} row is no longer eligible: "
                    f"{expected_speaker}:{session_id}"
                )
            _, _, _, sv_duration, cs_duration = derived_by_key[
                (expected_speaker, session_id)
            ]
            if session_id not in sv_rows or session_id not in cs_rows:
                raise ValueError(
                    f"SVD panel session missing from metadata: {session_id}"
                )
            sv_row = sv_rows[session_id]
            cs_row = cs_rows[session_id]
            if sv_row["speaker id"] != expected_speaker:
                raise ValueError(f"SVD SV speaker drift for session {session_id}")
            if cs_row["speaker id"] != expected_speaker:
                raise ValueError(f"SVD CS speaker drift for session {session_id}")
            if sv_row["health status"] != "1" or cs_row["health status"] != "1":
                raise ValueError(
                    f"SVD panel session is not patient-labelled: {session_id}"
                )
            if sv_row["gender"] != expected_sex or cs_row["gender"] != expected_sex:
                raise ValueError(f"SVD sex metadata drift for session {session_id}")
            if expected_speaker in observed_speakers:
                raise ValueError(f"duplicate SVD speaker in panel: {expected_speaker}")
            sv_path = args.sv_root / sv_row["filename"]
            cs_path = args.cs_root / cs_row["filename"]
            if not sv_path.is_file() or not cs_path.is_file():
                raise FileNotFoundError(
                    f"missing paired SVD audio for session {session_id}"
                )
            selected.append(
                PanelCase(
                    speaker_id=expected_speaker,
                    session_id=session_id,
                    sex=expected_sex,
                    diagnosis=sv_row["diagnosis"],
                    sv_path=sv_path,
                    cs_path=cs_path,
                    selection_role=role,
                    selection_rank_within_sex=rank,
                    sv_duration_seconds=float(sv_duration),
                    cs_duration_seconds=float(cs_duration),
                )
            )
            observed_speakers.add(expected_speaker)
    if len(selected) != SEALED_SPEAKERS:
        raise ValueError(f"expected {SEALED_SPEAKERS} sealed SVD speakers")
    primary = [case for case in selected if case.selection_role == "primary"]
    reserve = [case for case in selected if case.selection_role == "reserve"]
    if len(primary) != PRIMARY_SPEAKERS or len(reserve) != RESERVE_SPEAKERS:
        raise ValueError("SVD primary/reserve count drifted")
    if sum(case.sex == "female" for case in primary) != PRIMARY_PER_SEX:
        raise ValueError("SVD primary panel must contain 12 female speakers")
    if sum(case.sex == "male" for case in primary) != PRIMARY_PER_SEX:
        raise ValueError("SVD primary panel must contain 12 male speakers")
    return selected


def read_canonical_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"expected mono SVD audio: {path}")
    waveform = audio[:, 0]
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=int(sample_rate),
            target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid SVD waveform: {path}")
    return waveform.astype(np.float32, copy=False)


def load_canonical_audio(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"invalid sealed canonical waveform: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"non-finite sealed canonical waveform: {path}")
    return waveform


def train_slope_scale(args: argparse.Namespace) -> tuple[float, set[str]]:
    rows = read_rows(args.label_bank, args.label_bank_sha256)
    slopes = np.asarray(
        [
            float(row["slope"])
            for row in rows
            if row["split"] == "surrogate_train"
            and row["view"] in {"cs", "sv"}
            and row["scoring_status"] == "ok"
        ],
        dtype=np.float64,
    )
    scale = float(slopes.std())
    if slopes.size < 2 or scale <= 0.0:
        raise ValueError("invalid surrogate-train LTAS slope scale")
    return scale, {row["speaker_id"] for row in rows}


def source_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_commit": args.source_commit,
        "source_files_sha256": {
            "model/avqi_components.py": sha256_file(
                REPO_ROOT / "model/avqi_components.py"
            ),
            "scripts/evaluate_avqi_ltas_slope_gate_alignment.py": sha256_file(
                REPO_ROOT / "scripts/evaluate_avqi_ltas_slope_gate_alignment.py"
            ),
            "scripts/evaluate_avqi_ltas_slope_svd_authority_panel.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "data_sha256": {
            "sv_metadata.csv": args.sv_metadata_sha256,
            "cs_metadata.csv": args.cs_metadata_sha256,
            "exact_component_label_bank_v4.csv": args.label_bank_sha256,
            "predictor_checkpoint": args.predictor_checkpoint_sha256,
        },
        "exact_source_sha256": exact_source_hashes(args),
    }


def preregistered_contract() -> dict[str, Any]:
    return {
        "candidate": "frozen full-waveform PraatDifferentiable LTAS slope",
        "primary_views": list(PANEL_VIEWS),
        "require_overall_and_each_view": True,
        "level_spearman_min": LEVEL_SPEARMAN_GATE,
        "paired_lowpass_delta_spearman_min": DELTA_SPEARMAN_GATE,
        "normalized_mae_max": NORMALIZED_MAE_GATE,
        "calibration_slope_range": list(CALIBRATION_SLOPE_RANGE),
        "external_coverage_min": EXTERNAL_COVERAGE_GATE,
        "complete_selected_speakers_required": PRIMARY_SPEAKERS,
        "reserve_substitution": (
            "same-sex frozen rank using exact scoring_status only"
        ),
        "exact_values_used_for_panel_selection": False,
        "component_input_gradient_norm": [
            COMPONENT_INPUT_GRADIENT_MIN,
            COMPONENT_INPUT_GRADIENT_MAX,
        ],
        "training_segment_transfer_nmae_max": NORMALIZED_MAE_GATE,
        "exact_material_distance_min": EXACT_MATERIAL_DISTANCE_MIN,
        "candidate_to_exact_distance_ratio": list(AUTHORITY_RATIO_RANGE),
        "signed_direction_agreement_min": DIRECTION_AGREEMENT_MIN,
        "gain_and_shift_distance_max": INVARIANCE_DISTANCE_MAX,
        "current_absolute_lowpass_min_unchanged": CURRENT_ABSOLUTE_LOWPASS_MIN,
        "production_gate_changed": False,
    }


def seal_panel(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite panel seal: {args.output_dir}")
    validate_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "LTAS predictor checkpoint",
    )
    cases = select_panel_cases(args)
    scale, label_bank_speakers = train_slope_scale(args)
    overlap = {
        case.panel_speaker_id for case in cases
    } & label_bank_speakers
    if overlap:
        raise ValueError(f"SVD panel overlaps label-bank speakers: {sorted(overlap)}")

    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir(parents=True)
    sealed_rows: list[dict[str, Any]] = []
    for case in cases:
        for view, source_path in (("cs", case.cs_path), ("sv", case.sv_path)):
            waveform = read_canonical_audio(source_path)
            variants = waveform_variants(waveform)
            variant_paths: dict[str, str] = {}
            variant_hashes: dict[str, str] = {}
            for variant in VARIANT_NAMES:
                path = waveform_root / (
                    f"SVD_{case.speaker_id}_{case.session_id}_{view}_{variant}.wav"
                )
                sf.write(path, variants[variant], SAMPLE_RATE, subtype="FLOAT")
                variant_paths[variant] = str(path.resolve())
                variant_hashes[variant] = sha256_file(path)
            sealed_rows.append(
                {
                    "panel_speaker_id": case.panel_speaker_id,
                    "speaker_id": case.speaker_id,
                    "session_id": case.session_id,
                    "sex": case.sex,
                    "selection_role": case.selection_role,
                    "selection_rank_within_sex": case.selection_rank_within_sex,
                    "label": "patient",
                    "diagnosis": case.diagnosis,
                    "view": view,
                    "source_path": str(source_path.resolve()),
                    "source_audio_sha256": sha256_file(source_path),
                    "source_sample_rate": sf.info(source_path).samplerate,
                    "source_duration_seconds": (
                        case.cs_duration_seconds
                        if view == "cs"
                        else case.sv_duration_seconds
                    ),
                    "canonical_sample_rate": SAMPLE_RATE,
                    "canonical_samples": int(waveform.size),
                    "variant_paths": variant_paths,
                    "variant_sha256": variant_hashes,
                }
            )

    seal = {
        "schema_version": "avqi-route-c-ltas-svd-authority-panel-seal-v2",
        "stage": "seal",
        "exact_scores_opened": False,
        "selection": {
            "dataset": "SVD",
            "label_source": "SVD metadata health status 1=patient",
            "selection_rule": PANEL_SELECTION_RULE,
            "speaker_identity_key": "SVD speaker id",
            "session_identity_key": "SVD session_id",
            "speaker_split_before_transform": True,
            "speaker_disjoint_from_label_bank": True,
            "speaker_disjoint_from_v9_panel": True,
            "excluded_v9_speakers": sorted(
                f"SVD:{speaker_id}" for speaker_id in V9_PANEL_SPEAKERS
            ),
            "raw_duration_eligibility_seconds": {
                "sv_min": SV_DURATION_MIN_SECONDS,
                "cs_min": CS_DURATION_MIN_SECONDS,
            },
            "primary_speaker_count": PRIMARY_SPEAKERS,
            "primary_case_count": PRIMARY_CASES,
            "reserve_speaker_count": RESERVE_SPEAKERS,
            "sealed_speaker_count": SEALED_SPEAKERS,
            "sealed_case_count": SEALED_CASES,
            "views": list(PANEL_VIEWS),
            "primary_sex_counts": {"female": 12, "male": 12},
            "reserve_sex_counts": {"female": 6, "male": 2},
            "reserve_target_per_sex": RESERVE_PER_SEX_TARGET,
            "reserve_shortfall_disclosed": {"female": 0, "male": 4},
            "primary_speakers": [
                case.panel_speaker_id
                for case in cases
                if case.selection_role == "primary"
            ],
            "reserve_speakers": [
                case.panel_speaker_id
                for case in cases
                if case.selection_role == "reserve"
            ],
            "all_sealed_speakers": [case.panel_speaker_id for case in cases],
            "all_sealed_sessions": [case.session_id for case in cases],
            "substitution_policy": (
                "same-sex reserve rank; exact scoring_status only; exact LTAS "
                "values never used for selection"
            ),
        },
        "preprocessing": {
            "canonical_resample": "librosa soxr_hq to 16 kHz",
            "sv_exact": "Praat 34 Hz high-pass then length_normalize_sv last-3 rule",
            "cs_exact": "Praat 34 Hz high-pass then voiced_segment_extraction",
            "lowpass_transform": "hard FFT mask at 3 kHz before exact metric branch",
            "final_audio_highpass_applied": False,
        },
        "train_slope_scale_std_surrogate_train": scale,
        "preregistered_contract": preregistered_contract(),
        **source_identity(args),
        "rows": sealed_rows,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "scope": "LTAS slope only; Shimmer dB remains NO-GO; CPPS/HNR untouched",
    }
    seal_path = args.output_dir / "panel_seal.json"
    write_json(seal_path, seal)
    receipt = {
        "decision": "SEALED_SVD_PANEL_EXACT_SCORES_UNOPENED",
        "panel_seal_sha256": sha256_file(seal_path),
        "exact_scores_opened": False,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
    }
    write_json(args.output_dir / "seal_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def run_exact(
    items: list[dict[str, str]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    result = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, str(avqi_code_root)],
        input=json.dumps({"items": items}, sort_keys=True),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(EXACT_MARKER)
    ]
    if len(lines) != 1:
        raise RuntimeError(f"exact SVD scorer emitted {len(lines)} JSON records")
    payload = json.loads(lines[0][len(EXACT_MARKER) :])
    if [row["id"] for row in payload["rows"]] != [item["id"] for item in items]:
        raise ValueError("exact SVD scorer order or coverage drift")
    return payload


def safe_spearman(reference: np.ndarray, estimate: np.ndarray) -> float:
    value = float(stats.spearmanr(reference, estimate).statistic)
    return value if math.isfinite(value) else -1.0


def level_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([row["exact"]["clean"] for row in rows], dtype=np.float64)
    estimate = np.asarray(
        [row["candidate_frozen_full"]["clean"] for row in rows],
        dtype=np.float64,
    )
    train_scale = float(rows[0]["train_slope_scale"])
    variance = float(np.sum((truth - truth.mean()) ** 2))
    slope = float(
        np.sum((truth - truth.mean()) * (estimate - estimate.mean()))
        / max(variance, 1e-12)
    )
    metrics = {
        "rows": len(rows),
        "level_spearman": safe_spearman(truth, estimate),
        "normalized_mae": float(np.mean(np.abs(estimate - truth))) / train_scale,
        "calibration_slope": slope,
    }
    gates = {
        "level_spearman_ge_0_70": (
            metrics["level_spearman"] >= LEVEL_SPEARMAN_GATE
        ),
        "normalized_mae_le_0_50": (
            metrics["normalized_mae"] <= NORMALIZED_MAE_GATE
        ),
        "calibration_slope_0_75_to_1_25": (
            CALIBRATION_SLOPE_RANGE[0]
            <= metrics["calibration_slope"]
            <= CALIBRATION_SLOPE_RANGE[1]
        ),
    }
    return {**metrics, "gates": gates, "decision": "PASS" if all(gates.values()) else "FAIL"}


def paired_lowpass_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = np.asarray(
        [row["exact"]["lowpass_3khz"] - row["exact"]["clean"] for row in rows],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [
            row["candidate_frozen_full"]["lowpass_3khz"]
            - row["candidate_frozen_full"]["clean"]
            for row in rows
        ],
        dtype=np.float64,
    )
    rho = safe_spearman(exact, candidate)
    return {
        "rows": len(rows),
        "spearman": rho,
        "gate": DELTA_SPEARMAN_GATE,
        "decision": "PASS" if rho >= DELTA_SPEARMAN_GATE else "FAIL",
    }


def candidate_slope_tensor(
    waveform: torch.Tensor,
    estimator: PraatDifferentiableAVQIComponentEstimator,
    checkpoint: dict[str, torch.Tensor],
) -> torch.Tensor:
    prepared = estimator._prepare(waveform)
    ltas_input = estimator._soft_voiced_ltas_input(prepared)
    raw_slope, _ = estimator._global_ltas(ltas_input)
    normalized = (
        raw_slope * estimator.alignment_scale[SLOPE_INDEX]
        + estimator.alignment_bias[SLOPE_INDEX]
    )
    raw_value = (
        normalized * checkpoint["target_scale"][SLOPE_INDEX]
        + checkpoint["target_mean"][SLOPE_INDEX]
    )
    return (
        raw_value * checkpoint["calibration_scale"][SLOPE_INDEX]
        + checkpoint["calibration_bias"][SLOPE_INDEX]
    )


def gradient_report(
    rows: list[dict[str, Any]],
    estimator: PraatDifferentiableAVQIComponentEstimator,
    checkpoint: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    values = []
    for row in rows:
        waveform = torch.from_numpy(
            load_canonical_audio(Path(row["variant_paths"]["clean"]))
        ).to(device)
        waveform = waveform.requires_grad_(True)
        slope = candidate_slope_tensor(waveform, estimator, checkpoint)
        gradient = torch.autograd.grad(slope, waveform)[0]
        norm = float(torch.linalg.vector_norm(gradient.detach()).cpu())
        finite = bool(torch.isfinite(gradient).all()) and math.isfinite(norm)
        values.append(
            {
                "panel_speaker_id": row["panel_speaker_id"],
                "view": row["view"],
                "gradient_norm": norm,
                "finite": finite,
                "nonzero": norm > COMPONENT_INPUT_GRADIENT_MIN,
                "bounded": norm <= COMPONENT_INPUT_GRADIENT_MAX,
            }
        )
    norms = np.asarray([row["gradient_norm"] for row in values], dtype=np.float64)
    passed = all(
        row["finite"] and row["nonzero"] and row["bounded"] for row in values
    )
    return {
        "rows": len(values),
        "gradient_norm": {
            "min": float(norms.min()),
            "median": float(np.median(norms)),
            "max": float(norms.max()),
        },
        "gate": [COMPONENT_INPUT_GRADIENT_MIN, COMPONENT_INPUT_GRADIENT_MAX],
        "decision": "PASS" if passed else "FAIL",
        "cases": values,
    }


def deterministic_training_segments(waveform: np.ndarray) -> list[np.ndarray]:
    tensor = torch.from_numpy(waveform)
    if tensor.numel() <= TRAINING_SEGMENT_SAMPLES:
        return [
            F.pad(tensor, (0, TRAINING_SEGMENT_SAMPLES - tensor.numel())).numpy()
        ]
    last_start = tensor.numel() - TRAINING_SEGMENT_SAMPLES
    starts = sorted({0, last_start // 2, last_start})
    return [
        tensor[start : start + TRAINING_SEGMENT_SAMPLES].numpy() for start in starts
    ]


def transfer_report(
    rows: list[dict[str, Any]],
    estimator: PraatDifferentiableAVQIComponentEstimator,
    checkpoint: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    errors = []
    segment_count = 0
    train_scale = float(rows[0]["train_slope_scale"])
    for row in rows:
        clean = load_canonical_audio(Path(row["variant_paths"]["clean"]))
        target = float(row["exact"]["clean"])
        for segment in deterministic_training_segments(clean):
            prediction = predict_slope(
                segment,
                estimator,
                checkpoint,
                exact_window=False,
                device=device,
            )
            errors.append(abs(prediction - target) / train_scale)
            segment_count += 1
    nmae = float(np.mean(errors))
    return {
        "example_count": len(rows),
        "segment_count": segment_count,
        "training_segment_samples": TRAINING_SEGMENT_SAMPLES,
        "normalized_mae": nmae,
        "gate": NORMALIZED_MAE_GATE,
        "decision": "PASS" if nmae <= NORMALIZED_MAE_GATE else "FAIL",
    }


def by_scope(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "overall": metric(rows),
        "cs": metric([row for row in rows if row["view"] == "cs"]),
        "sv": metric([row for row in rows if row["view"] == "sv"]),
    }


def write_score_artifacts(
    args: argparse.Namespace,
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    report_path = args.output_dir / "diagnostic_report.json"
    predictions_path = args.output_dir / "predictions.json"
    write_json(report_path, report)
    write_json(predictions_path, predictions)
    receipt = {
        "decision": report["decision"],
        "panel_seal_sha256": args.panel_seal_sha256,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            "diagnostic_report.json": sha256_file(report_path),
            "predictions.json": sha256_file(predictions_path),
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    (args.output_dir / "SUMMARY.md").write_text(
        "# LTAS slope SVD external authority panel\n\n"
        f"Decision: `{report['decision']}`\n\n"
        "The production gate is unchanged; generator optimizer steps remain zero.\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def ordered_speaker_ids(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(row["panel_speaker_id"] for row in rows))


def exact_items_for_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = []
    for row in rows:
        for variant in VARIANT_NAMES:
            items.append(
                {
                    "id": (
                        f"{row['panel_speaker_id']}:{row['session_id']}:"
                        f"{row['view']}:{variant}"
                    ),
                    "path": row["variant_paths"][variant],
                    "view": row["view"],
                }
            )
    return items


def exact_rows_for_speaker(
    rows: list[dict[str, Any]],
    panel_speaker_id: str,
) -> list[dict[str, Any]]:
    prefix = f"{panel_speaker_id}:"
    return [row for row in rows if row["id"].startswith(prefix)]


def exact_speaker_complete(
    rows: list[dict[str, Any]],
    panel_speaker_id: str,
) -> bool:
    speaker_rows = exact_rows_for_speaker(rows, panel_speaker_id)
    expected = len(PANEL_VIEWS) * len(VARIANT_NAMES)
    return len(speaker_rows) == expected and all(
        row["scoring_status"] == "ok" for row in speaker_rows
    )


def exact_failure_receipts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "view": row["view"],
            "scoring_status": row["scoring_status"],
            "error_type": row["error_type"],
            "error_message": row["error_message"],
        }
        for row in rows
        if row["scoring_status"] != "ok"
    ]


def score_panel(args: argparse.Namespace) -> None:
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"sealed panel output does not exist: {args.output_dir}")
    if not args.panel_seal_sha256:
        raise ValueError("score stage requires --panel-seal-sha256")
    if (args.output_dir / "diagnostic_report.json").exists():
        raise FileExistsError("refusing to overwrite an opened SVD panel report")
    seal_path = args.output_dir / "panel_seal.json"
    validate_hash(seal_path, args.panel_seal_sha256, "SVD panel seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("exact_scores_opened") is not False:
        raise ValueError("SVD panel seal does not prove unopened exact scores")
    if seal.get("source_commit") != args.source_commit:
        raise ValueError("SVD panel source commit drifted after sealing")
    current_identity = source_identity(args)
    for key in ("source_files_sha256", "data_sha256", "exact_source_sha256"):
        if seal.get(key) != current_identity[key]:
            raise ValueError(f"SVD panel {key} drifted after sealing")
    if seal.get("preregistered_contract") != preregistered_contract():
        raise ValueError("SVD panel gate contract drifted after sealing")
    if len(seal.get("rows", [])) != SEALED_CASES:
        raise ValueError("SVD panel seal case count drifted")
    for row in seal["rows"]:
        for variant in VARIANT_NAMES:
            validate_hash(
                Path(row["variant_paths"][variant]),
                row["variant_sha256"][variant],
                f"sealed {row['panel_speaker_id']} {row['view']} {variant}",
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    estimator, checkpoint = load_predictor(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        device,
    )
    train_scale, label_bank_speakers = train_slope_scale(args)
    if {row["panel_speaker_id"] for row in seal["rows"]} & label_bank_speakers:
        raise ValueError("sealed SVD speakers overlap the label bank")

    rows_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in seal["rows"]:
        rows_by_speaker.setdefault(row["panel_speaker_id"], []).append(row)
    if any(len(rows) != len(PANEL_VIEWS) for rows in rows_by_speaker.values()):
        raise ValueError("sealed SVD speaker does not have exactly CS and SV rows")
    primary_rows = [
        row for row in seal["rows"] if row["selection_role"] == "primary"
    ]
    reserve_rows = [
        row for row in seal["rows"] if row["selection_role"] == "reserve"
    ]
    primary_speakers = ordered_speaker_ids(primary_rows)
    reserve_speakers = ordered_speaker_ids(reserve_rows)
    if len(primary_speakers) != PRIMARY_SPEAKERS:
        raise ValueError("sealed SVD primary speaker count drifted")
    if len(reserve_speakers) != RESERVE_SPEAKERS:
        raise ValueError("sealed SVD reserve speaker count drifted")
    if primary_speakers != seal["selection"]["primary_speakers"]:
        raise ValueError("sealed SVD primary order drifted")
    if reserve_speakers != seal["selection"]["reserve_speakers"]:
        raise ValueError("sealed SVD reserve order drifted")

    speaker_sex = {
        speaker_id: rows[0]["sex"] for speaker_id, rows in rows_by_speaker.items()
    }
    reserve_by_sex = {
        sex: [speaker for speaker in reserve_speakers if speaker_sex[speaker] == sex]
        for sex in ("female", "male")
    }
    attempted_exact_rows: list[dict[str, Any]] = []
    exact_runtime_identity: dict[str, str] | None = None

    def score_speakers(speaker_ids: list[str]) -> list[dict[str, Any]]:
        nonlocal exact_runtime_identity
        batch_rows = [
            row for speaker_id in speaker_ids for row in rows_by_speaker[speaker_id]
        ]
        payload = run_exact(
            exact_items_for_rows(batch_rows),
            args.exact_python,
            args.avqi_code_root,
        )
        expected_rows = (
            len(speaker_ids) * len(PANEL_VIEWS) * len(VARIANT_NAMES)
        )
        if len(payload["rows"]) != expected_rows:
            raise ValueError("unexpected SVD exact batch row count")
        runtime = {
            "parselmouth_version": payload["parselmouth_version"],
            "praat_version": payload["praat_version"],
        }
        if exact_runtime_identity is None:
            exact_runtime_identity = runtime
        elif runtime != exact_runtime_identity:
            raise ValueError("exact SVD runtime drifted between status-only batches")
        existing_ids = {row["id"] for row in attempted_exact_rows}
        if existing_ids & {row["id"] for row in payload["rows"]}:
            raise ValueError("exact SVD speaker was scored more than once")
        attempted_exact_rows.extend(payload["rows"])
        return payload["rows"]

    primary_exact_rows = score_speakers(primary_speakers)
    if len(primary_exact_rows) != EXPECTED_SELECTED_EXACT_ROWS:
        raise ValueError("unexpected SVD primary exact row count")
    failed_primary = [
        speaker
        for speaker in primary_speakers
        if not exact_speaker_complete(attempted_exact_rows, speaker)
    ]
    replacement_by_primary: dict[str, str] = {}
    reserve_attempts: list[dict[str, Any]] = []
    reserve_cursor = {"female": 0, "male": 0}
    unresolved_primary: list[str] = []
    for primary_speaker in failed_primary:
        sex = speaker_sex[primary_speaker]
        replacement = None
        while reserve_cursor[sex] < len(reserve_by_sex[sex]):
            reserve_speaker = reserve_by_sex[sex][reserve_cursor[sex]]
            reserve_cursor[sex] += 1
            batch = score_speakers([reserve_speaker])
            complete = exact_speaker_complete(batch, reserve_speaker)
            reserve_attempts.append(
                {
                    "replaces_primary_speaker": primary_speaker,
                    "reserve_speaker": reserve_speaker,
                    "sex": sex,
                    "selection_input": "exact scoring_status only",
                    "complete_cs_sv_variants": complete,
                    "failures": exact_failure_receipts(batch),
                }
            )
            if complete:
                replacement = reserve_speaker
                break
        if replacement is None:
            unresolved_primary.append(primary_speaker)
        else:
            replacement_by_primary[primary_speaker] = replacement

    selected_speakers = [
        replacement_by_primary.get(speaker, speaker)
        for speaker in primary_speakers
        if speaker not in unresolved_primary
    ]
    if len(set(selected_speakers)) != len(selected_speakers):
        raise ValueError("status-only SVD selection produced duplicate speakers")
    selected_exact_rows = [
        row
        for speaker in selected_speakers
        for row in exact_rows_for_speaker(attempted_exact_rows, speaker)
    ]
    selected_ok_rows = [
        row for row in selected_exact_rows if row["scoring_status"] == "ok"
    ]
    selected_coverage = len(selected_ok_rows) / EXPECTED_SELECTED_EXACT_ROWS
    attempted_ok_rows = [
        row for row in attempted_exact_rows if row["scoring_status"] == "ok"
    ]
    attempted_coverage = len(attempted_ok_rows) / len(attempted_exact_rows)
    substitution_audit = {
        "policy": (
            "score all primary speakers first; replace an incomplete primary "
            "with the first exact-complete same-sex reserve in frozen rank order"
        ),
        "exact_values_used_for_selection": False,
        "selection_field": "scoring_status",
        "primary_speakers": primary_speakers,
        "failed_primary_speakers": failed_primary,
        "reserve_speakers_by_sex": reserve_by_sex,
        "reserve_attempts": reserve_attempts,
        "substitutions": [
            {
                "primary_speaker": primary,
                "reserve_speaker": reserve,
                "sex": speaker_sex[primary],
            }
            for primary, reserve in replacement_by_primary.items()
        ],
        "unresolved_primary_speakers": unresolved_primary,
        "selected_speakers": selected_speakers,
        "unused_reserve_speakers": [
            speaker
            for speaker in reserve_speakers
            if not exact_rows_for_speaker(attempted_exact_rows, speaker)
        ],
    }
    if (
        unresolved_primary
        or len(selected_speakers) != PRIMARY_SPEAKERS
        or selected_coverage < 1.0
    ):
        report = {
            "schema_version": "avqi-route-c-ltas-svd-authority-panel-v2",
            "decision": "FAIL_EXTERNAL_SVD_LTAS_EXACT_COVERAGE",
            "panel_seal_sha256": args.panel_seal_sha256,
            "source_commit": args.source_commit,
            "slurm_job_id": args.slurm_job_id or None,
            "status_only_substitution": substitution_audit,
            "selected_exact_coverage": selected_coverage,
            "exact_coverage_gate": EXTERNAL_COVERAGE_GATE,
            "complete_selected_speakers_required": PRIMARY_SPEAKERS,
            "attempted_exact_rows": len(attempted_exact_rows),
            "attempted_exact_coverage": attempted_coverage,
            "exact_failures": exact_failure_receipts(attempted_exact_rows),
            "preregistered_contract": preregistered_contract(),
            "production_gate_changed": False,
            "generator_optimizer_steps": 0,
            "bounded_waveform_pilot_submitted": False,
            "formal_pathology_training_submitted": False,
        }
        write_score_artifacts(args, report, [])
        return

    if exact_runtime_identity is None:
        raise RuntimeError("exact SVD runtime identity was not recorded")
    if sum(speaker_sex[speaker] == "female" for speaker in selected_speakers) != 12:
        raise ValueError("selected SVD panel does not retain 12 female speakers")
    if sum(speaker_sex[speaker] == "male" for speaker in selected_speakers) != 12:
        raise ValueError("selected SVD panel does not retain 12 male speakers")

    result_rows: list[dict[str, Any]] = []
    for speaker in selected_speakers:
        for sealed in rows_by_speaker[speaker]:
            candidate = {}
            for variant in VARIANT_NAMES:
                waveform = load_canonical_audio(
                    Path(sealed["variant_paths"][variant])
                )
                candidate[variant] = predict_slope(
                    waveform,
                    estimator,
                    checkpoint,
                    exact_window=False,
                    device=device,
                )
            result_rows.append(
                {
                    **sealed,
                    "train_slope_scale": train_scale,
                    "candidate_frozen_full": candidate,
                }
            )

    exact_index = {row["id"]: row for row in selected_exact_rows}
    for row in result_rows:
        prefix = f"{row['panel_speaker_id']}:{row['session_id']}:{row['view']}"
        row["exact"] = {
            variant: float(exact_index[f"{prefix}:{variant}"]["slope"])
            for variant in VARIANT_NAMES
        }
        row["exact_preprocessed_samples"] = {
            variant: int(exact_index[f"{prefix}:{variant}"]["preprocessed_samples"])
            for variant in VARIANT_NAMES
        }

    level = by_scope(result_rows, level_metrics)
    paired_delta = by_scope(result_rows, paired_lowpass_delta)
    anti_shortcut = {}
    for scope, scoped_rows in {
        "overall": result_rows,
        "cs": [row for row in result_rows if row["view"] == "cs"],
        "sv": [row for row in result_rows if row["view"] == "sv"],
    }.items():
        summary = summarize_mode(scoped_rows, "candidate_frozen_full", train_scale)
        anti_shortcut[scope] = {
            "summary": summary,
            "gate": exact_relative_gate(summary),
        }
    gradient = by_scope(
        result_rows,
        lambda rows: gradient_report(rows, estimator, checkpoint, device),
    )
    transfer = by_scope(
        result_rows,
        lambda rows: transfer_report(rows, estimator, checkpoint, device),
    )
    complete_pass = all(
        result[scope]["decision"] == "PASS"
        for result in (level, paired_delta, gradient, transfer)
        for scope in ("overall", "cs", "sv")
    ) and all(
        anti_shortcut[scope]["gate"]["decision"] == "PASS"
        for scope in ("overall", "cs", "sv")
    )
    decision = (
        "PASS_EXTERNAL_SVD_LTAS_AUTHORITY_PANEL_NO_PRODUCTION_CHANGE"
        if complete_pass
        else "FAIL_EXTERNAL_SVD_LTAS_AUTHORITY_PANEL_KEEP_PRODUCTION_GATE"
    )
    report = {
        "schema_version": "avqi-route-c-ltas-svd-authority-panel-v2",
        "decision": decision,
        "promotion_boundary": (
            "ELIGIBLE_FOR_MINIMAL_PRODUCTION_GATE_REVISION_REVIEW_ONLY"
            if complete_pass
            else "NOT_ELIGIBLE_FOR_PRODUCTION_GATE_REVISION"
        ),
        "panel_seal_sha256": args.panel_seal_sha256,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id or None,
        "source_identity": current_identity,
        "selection": seal["selection"],
        "status_only_substitution": substitution_audit,
        "preprocessing": seal["preprocessing"],
        "preregistered_contract": preregistered_contract(),
        "train_slope_scale_std_surrogate_train": train_scale,
        "exact_runtime": {
            **exact_runtime_identity,
            "attempted_rows": len(attempted_exact_rows),
            "attempted_coverage": attempted_coverage,
            "selected_rows": len(selected_exact_rows),
            "selected_coverage": selected_coverage,
            "coverage_gate": EXTERNAL_COVERAGE_GATE,
        },
        "level_alignment": level,
        "paired_lowpass_delta": paired_delta,
        "anti_shortcut_exact_relative": anti_shortcut,
        "component_input_gradient": gradient,
        "training_segment_transfer": transfer,
        "rows": result_rows,
        "production_gate_changed": False,
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "shimmer_db_decision": "COMPONENT_LEVEL_NO_GO_SHIMMER_DB_V8",
        "scope": "LTAS slope only; CPPS/HNR untouched",
    }
    write_score_artifacts(args, report, result_rows)


def main() -> None:
    args = parse_args()
    if args.stage == "seal":
        if args.panel_seal_sha256:
            raise ValueError("seal stage must not receive an existing seal hash")
        seal_panel(args)
    else:
        score_panel(args)


if __name__ == "__main__":
    main()
