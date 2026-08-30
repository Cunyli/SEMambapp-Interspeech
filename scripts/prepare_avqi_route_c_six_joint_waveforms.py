#!/usr/bin/env python3
"""Prepare and hash-seal full-band Route C six-joint candidate waveforms.

This stage consumes an already hash-bound joint-gradient manifest. It applies
the frozen one-step waveform-RMS normalization to patient rows, preserves
healthy rows byte-identically after PCM24 canonicalization, and seals every
calibration and final candidate before any candidate exact-Praat outcome is
opened. It never loads or updates generator parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf


if __name__ == "__main__" and __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_avqi_route_c_six_joint_waveforms",
            *sys.argv[1:],
        ],
        cwd=project_root,
        check=False,
    )
    raise SystemExit(completed.returncode)

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    CLEAN_PATHOLOGICAL_ROLE,
    EXPECTED_TOTAL_ROWS,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    GLOBAL_ALPHA_GRID,
    HEALTHY_ROLE,
    NORMALIZATION_SOURCE,
    PANEL_ROW_FIELDS,
    PATHOLOGICAL_ROLE,
    SIX_GRADIENT_PASS_DECISION,
    TRAINING_NO_GO,
    _finite_mapping,
    _require_optimizer_zero,
    _validate_panel_rows,
    _validate_six_gradient,
    _validate_split_seal,
    validate_source,
)


SAMPLE_RATE = 16_000
OUTPUT_SUBTYPE = "PCM_24"
GRADIENT_MANIFEST_SCHEMA_VERSION = (
    "avqi-route-c-six-joint-gradient-manifest-v1"
)
TARGET_BANK_SCHEMA_VERSION = "avqi-route-c-six-joint-clean-target-bank-v1"
WAVEFORM_SEAL_SCHEMA_VERSION = "avqi-route-c-six-joint-waveform-seal-v1"
WAVEFORM_SEAL_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-joint-waveform-seal-receipt-v1"
)
WAVEFORM_SEAL_DECISION = "SEALED_SIX_JOINT_WAVEFORMS_EXACT_UNOPENED"
PATIENT_ROLES = frozenset({PATHOLOGICAL_ROLE, CLEAN_PATHOLOGICAL_ROLE})
GRADIENT_ROW_FIELDS = frozenset(
    {
        "case_id",
        "base_waveform_path",
        "base_waveform_sha256",
        "joint_gradient_path",
        "joint_gradient_sha256",
        "topology_sha256",
    }
)
TARGET_ROW_FIELDS = frozenset(
    {
        "speaker_id",
        "split",
        "view",
        "target_waveform_path",
        "target_waveform_sha256",
        "exact_components",
    }
)
FORBIDDEN_RESULT_FIELDS = frozenset(
    {
        "alpha_selected",
        "candidate_exact_components",
        "candidate_exact_outcomes",
        "exact_after",
        "exact_improvement",
        "final_decision",
    }
)


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is unavailable: {resolved}")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return resolved


def _load_audio(path: Path, expected_sha256: str, label: str) -> np.ndarray:
    resolved = _verified_file(path, expected_sha256, label)
    audio, sample_rate = sf.read(resolved, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{label} must be nonempty mono 16 kHz audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} contains non-finite samples")
    return np.asarray(audio, dtype=np.float32)


def _load_gradient(
    path: Path,
    expected_sha256: str,
    expected_samples: int,
    label: str,
) -> np.ndarray:
    resolved = _verified_file(path, expected_sha256, label)
    gradient = np.load(resolved, allow_pickle=False)
    if gradient.ndim != 1 or gradient.size != expected_samples:
        raise ValueError(f"{label} shape differs from its base waveform")
    gradient = np.asarray(gradient, dtype=np.float32)
    if not np.isfinite(gradient).all():
        raise ValueError(f"{label} contains non-finite values")
    gradient_rms = float(np.sqrt(np.mean(np.square(gradient, dtype=np.float64))))
    if not math.isfinite(gradient_rms) or gradient_rms <= 1e-15:
        raise ValueError(f"{label} has no usable RMS")
    return gradient


def candidate_from_gradient(
    base: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray | None, str | None]:
    """Apply the frozen waveform-RMS-normalized one-step update."""
    if alpha == 0.0:
        return np.asarray(base, dtype=np.float32).copy(), None
    base64 = np.asarray(base, dtype=np.float64)
    gradient64 = np.asarray(gradient, dtype=np.float64)
    base_rms = max(float(np.sqrt(np.mean(np.square(base64)))), 1e-12)
    gradient_rms = float(np.sqrt(np.mean(np.square(gradient64))))
    if not math.isfinite(gradient_rms) or gradient_rms <= 1e-15:
        return None, "gradient_rms_invalid"
    candidate = base64 - alpha * base_rms * gradient64 / gradient_rms
    if not np.isfinite(candidate).all():
        return None, "candidate_non_finite"
    if float(np.max(np.abs(candidate))) >= 0.999:
        return None, "candidate_peak_outside_pcm24_contract"
    return np.asarray(candidate, dtype=np.float32), None


def _write_pcm24(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed waveform: {path}")
    sf.write(path, audio, SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
    stored, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if (
        sample_rate != SAMPLE_RATE
        or stored.ndim != 1
        or stored.shape != audio.shape
        or not np.isfinite(stored).all()
        or info.subtype != OUTPUT_SUBTYPE
    ):
        raise ValueError(f"stored PCM24 waveform failed readback: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "samples": int(stored.size),
        "sample_rate": sample_rate,
        "subtype": info.subtype,
        "float32_sha256": hashlib.sha256(
            np.ascontiguousarray(stored, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def _existing_audio_binding(
    path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    audio = _load_audio(path, expected_sha256, label)
    resolved = path.resolve()
    info = sf.info(resolved)
    return {
        "path": str(resolved),
        "sha256": expected_sha256,
        "samples": int(audio.size),
        "sample_rate": info.samplerate,
        "subtype": info.subtype,
        "float32_sha256": hashlib.sha256(
            np.ascontiguousarray(audio, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def _safe_case_id(case_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", case_id)
    if not safe:
        raise ValueError("case ID cannot be converted to a safe filename")
    return safe


def _normalization_from_raw_report(
    report: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    contract = report.get("contract")
    normalization = (
        contract.get("normalization") if isinstance(contract, dict) else None
    )
    if not isinstance(normalization, dict):
        raise ValueError("six-gradient raw normalization is unavailable")
    target_mean = _finite_mapping(
        normalization.get("target_mean"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-gradient target means",
    )
    target_scale = _finite_mapping(
        normalization.get("target_scale"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-gradient target scales",
        positive=True,
    )
    return {"target_mean": target_mean, "target_scale": target_scale}


def validate_target_bank(
    bank: Mapping[str, Any],
    *,
    bank_sha256: str,
    split_seal_sha256: str,
    panel_rows: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if bank.get("schema_version") != TARGET_BANK_SCHEMA_VERSION:
        raise ValueError("clean target-bank schema differs")
    if bank.get("scientific_contract_schema_version") != (
        FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("clean target-bank scientific contract differs")
    if bank.get("split_seal_sha256") != split_seal_sha256:
        raise ValueError("clean target bank does not bind the split seal")
    if (
        bank.get("candidate_exact_outcomes_opened") is not False
        or bank.get("candidate_waveforms_scored") is not False
    ):
        raise ValueError("clean target bank opened candidate outcomes")
    _require_optimizer_zero(bank, "clean target bank")
    rows = bank.get("rows")
    if not isinstance(rows, list):
        raise ValueError("clean target bank rows are unavailable")
    expected_keys = {
        (str(row["speaker_id"]), str(row["view"]))
        for row in panel_rows.values()
        if row["label"] == "patient"
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != TARGET_ROW_FIELDS:
            raise ValueError("clean target-bank row fields differ")
        if FORBIDDEN_RESULT_FIELDS & set(row):
            raise ValueError("clean target bank contains candidate result fields")
        key = (str(row["speaker_id"]), str(row["view"]))
        if key in indexed:
            raise ValueError("clean target bank contains duplicate speaker/view")
        matching = [
            panel_row
            for panel_row in panel_rows.values()
            if panel_row["speaker_id"] == key[0]
            and panel_row["view"] == key[1]
            and panel_row["label"] == "patient"
        ]
        if not matching or {item["split"] for item in matching} != {row["split"]}:
            raise ValueError("clean target-bank row split differs")
        exact_components = _finite_mapping(
            row.get("exact_components"),
            ROUTE_C_SIX_ACTIVE_COMPONENTS,
            f"clean target components {key}",
        )
        target_path = Path(str(row["target_waveform_path"]))
        target_sha256 = str(row["target_waveform_sha256"])
        _load_audio(target_path, target_sha256, f"clean target waveform {key}")
        indexed[key] = {
            **row,
            "target_waveform_path": str(target_path.resolve()),
            "exact_components": exact_components,
            "target_bank_sha256": bank_sha256,
        }
    if set(indexed) != expected_keys:
        raise ValueError("clean target bank does not exactly cover patient speakers")
    return indexed


def validate_gradient_manifest(
    manifest: Mapping[str, Any],
    *,
    split_seal_sha256: str,
    target_bank_sha256: str,
    six_gradient_report_sha256: str,
    six_gradient_receipt_sha256: str,
    six_gradient_raw_report_sha256: str,
    weights: Mapping[str, float],
    normalization: Mapping[str, Mapping[str, float]],
    panel_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != GRADIENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("six-joint gradient manifest schema differs")
    expected_header = {
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "split_seal_sha256": split_seal_sha256,
        "clean_target_label_bank_sha256": target_bank_sha256,
        "six_gradient_report_sha256": six_gradient_report_sha256,
        "six_gradient_receipt_sha256": six_gradient_receipt_sha256,
        "six_gradient_raw_report_sha256": six_gradient_raw_report_sha256,
        "six_gradient_decision": SIX_GRADIENT_PASS_DECISION,
        "normalization_source": NORMALIZATION_SOURCE,
        "gradient_source": (
            "six_active_bidirectional_gap_losses_current_output_v19_topology"
        ),
        "current_output_topology_bound": True,
        "waveform_steps": 1,
        "gradient_normalization": "waveform_rms_normalized",
        "candidate_exact_outcomes_opened": False,
        "waveform_generation_performed": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise ValueError("six-joint gradient manifest header differs")
    if tuple(manifest.get("component_order", ())) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("six-joint gradient component order differs")
    observed_weights = _finite_mapping(
        manifest.get("calibration_inverse_gradient_weights"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-joint gradient weights",
        positive=True,
    )
    if observed_weights != dict(weights):
        raise ValueError("six-joint gradient weights differ from passed decision")
    if manifest.get("normalization") != normalization:
        raise ValueError("six-joint gradient normalization differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError("six-joint gradient manifest row count differs")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != GRADIENT_ROW_FIELDS:
            raise ValueError("six-joint gradient row fields differ")
        if FORBIDDEN_RESULT_FIELDS & set(row):
            raise ValueError("six-joint gradient manifest contains result fields")
        case_id = str(row["case_id"])
        if case_id in indexed or case_id not in panel_rows:
            raise ValueError("six-joint gradient case coverage differs")
        panel_row = panel_rows[case_id]
        base_path = Path(str(row["base_waveform_path"]))
        base_sha256 = str(row["base_waveform_sha256"])
        _load_audio(base_path, base_sha256, f"base waveform {case_id}")
        if panel_row["optimization_role"] == HEALTHY_ROLE:
            if any(
                row[key] is not None
                for key in (
                    "joint_gradient_path",
                    "joint_gradient_sha256",
                    "topology_sha256",
                )
            ):
                raise ValueError("healthy gradient row must contain no step inputs")
        else:
            if panel_row["optimization_role"] not in PATIENT_ROLES:
                raise ValueError("unknown six-joint optimization role")
            if any(
                not isinstance(row[key], str) or not row[key]
                for key in (
                    "joint_gradient_path",
                    "joint_gradient_sha256",
                    "topology_sha256",
                )
            ):
                raise ValueError("patient gradient row lacks step inputs")
        indexed[case_id] = {
            **row,
            "base_waveform_path": str(base_path.resolve()),
        }
    if set(indexed) != set(panel_rows):
        raise ValueError("six-joint gradient manifest does not exactly cover panel")
    return indexed


def prepare_and_seal(
    *,
    split_seal: Mapping[str, Any],
    split_seal_sha256: str,
    target_bank: Mapping[str, Any],
    target_bank_sha256: str,
    gradient_manifest: Mapping[str, Any],
    gradient_manifest_sha256: str,
    six_gradient_report: Mapping[str, Any],
    six_gradient_receipt: Mapping[str, Any],
    six_gradient_report_sha256: str,
    six_gradient_receipt_sha256: str,
    six_gradient_raw_report: Mapping[str, Any],
    six_gradient_raw_report_sha256: str,
    gate_sha256: str,
    target_protocol_sha256: str,
    ledger_sha256: str,
    source_manifest_sha256: str,
    source: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite waveform seal: {output_dir}")
    _validate_split_seal(
        split_seal,
        gate_sha256=gate_sha256,
        target_sha256=target_protocol_sha256,
        ledger_sha256=ledger_sha256,
        source_sha256=source_manifest_sha256,
    )
    panel_rows, speakers_by_split = _validate_panel_rows(
        split_seal.get("rows"), "six-joint waveform preparation"
    )
    expected_source_evidence = six_gradient_report.get("source_evidence_sha256")
    if not isinstance(expected_source_evidence, dict):
        raise ValueError("six-gradient source evidence is unavailable")
    weights = _validate_six_gradient(
        six_gradient_report,
        six_gradient_receipt,
        six_gradient_report_sha256,
        expected_source_evidence,
    )
    raw_binding = six_gradient_report.get("raw_measurement_evidence")
    if (
        not isinstance(raw_binding, dict)
        or raw_binding.get("report_sha256") != six_gradient_raw_report_sha256
    ):
        raise ValueError("six-gradient decision does not bind the raw report")
    normalization = _normalization_from_raw_report(six_gradient_raw_report)
    targets = validate_target_bank(
        target_bank,
        bank_sha256=target_bank_sha256,
        split_seal_sha256=split_seal_sha256,
        panel_rows=panel_rows,
    )
    gradients = validate_gradient_manifest(
        gradient_manifest,
        split_seal_sha256=split_seal_sha256,
        target_bank_sha256=target_bank_sha256,
        six_gradient_report_sha256=six_gradient_report_sha256,
        six_gradient_receipt_sha256=six_gradient_receipt_sha256,
        six_gradient_raw_report_sha256=six_gradient_raw_report_sha256,
        weights=weights,
        normalization=normalization,
        panel_rows=panel_rows,
    )

    safe_case_ids = {
        case_id: _safe_case_id(case_id) for case_id in sorted(panel_rows)
    }
    if len(set(safe_case_ids.values())) != len(safe_case_ids):
        raise ValueError("case IDs collide after safe-filename normalization")

    base_root = output_dir / "waveforms" / "base"
    candidate_root = output_dir / "waveforms" / "candidates"
    for path in (base_root, candidate_root):
        path.mkdir(parents=True, exist_ok=False)

    sealed_targets: dict[tuple[str, str], dict[str, Any]] = {}
    sealed_rows: list[dict[str, Any]] = []
    unavailable_candidates = 0
    for case_id in sorted(panel_rows):
        panel_row = panel_rows[case_id]
        gradient_row = gradients[case_id]
        safe_case = safe_case_ids[case_id]
        base = _load_audio(
            Path(gradient_row["base_waveform_path"]),
            str(gradient_row["base_waveform_sha256"]),
            f"base waveform {case_id}",
        )
        sealed_base = _write_pcm24(base_root / f"{safe_case}.wav", base)
        target_record: dict[str, Any] | None = None
        gradient: np.ndarray | None = None
        if panel_row["label"] == "patient":
            target_key = (str(panel_row["speaker_id"]), str(panel_row["view"]))
            target_source = targets[target_key]
            if target_key not in sealed_targets:
                sealed_targets[target_key] = {
                    **_existing_audio_binding(
                        Path(target_source["target_waveform_path"]),
                        str(target_source["target_waveform_sha256"]),
                        f"target waveform {target_key}",
                    ),
                    "speaker_id": target_key[0],
                    "view": target_key[1],
                    "split": target_source["split"],
                    "exact_components": target_source["exact_components"],
                }
            target_record = sealed_targets[target_key]
            gradient = _load_gradient(
                Path(str(gradient_row["joint_gradient_path"])),
                str(gradient_row["joint_gradient_sha256"]),
                base.size,
                f"joint gradient {case_id}",
            )

        candidates: list[dict[str, Any]] = []
        for alpha_index, alpha in enumerate(GLOBAL_ALPHA_GRID):
            if panel_row["optimization_role"] == HEALTHY_ROLE:
                candidates.append(
                    {
                        "alpha": alpha,
                        "available": True,
                        "unavailable_reason": None,
                        **sealed_base,
                    }
                )
                continue
            if gradient is None:
                raise ValueError("patient row reached waveform step without gradient")
            candidate, reason = candidate_from_gradient(base, gradient, alpha)
            if candidate is None:
                unavailable_candidates += 1
                candidates.append(
                    {
                        "alpha": alpha,
                        "available": False,
                        "unavailable_reason": reason,
                        "path": None,
                        "sha256": None,
                        "samples": int(base.size),
                        "sample_rate": SAMPLE_RATE,
                        "subtype": OUTPUT_SUBTYPE,
                        "float32_sha256": None,
                    }
                )
                continue
            if alpha == 0.0:
                candidate_binding = sealed_base
            else:
                alpha_dir = candidate_root / f"alpha_{alpha_index:02d}"
                alpha_dir.mkdir(exist_ok=True)
                candidate_binding = _write_pcm24(
                    alpha_dir / f"{safe_case}.wav", candidate
                )
            candidates.append(
                {
                    "alpha": alpha,
                    "available": True,
                    "unavailable_reason": None,
                    **candidate_binding,
                }
            )

        if panel_row["optimization_role"] == HEALTHY_ROLE and any(
            candidate["sha256"] != sealed_base["sha256"]
            for candidate in candidates
        ):
            raise ValueError("healthy candidate differs from sealed base")
        sealed_rows.append(
            {
                **{field: panel_row[field] for field in PANEL_ROW_FIELDS},
                "base": sealed_base,
                "target": target_record,
                "joint_gradient_sha256": gradient_row["joint_gradient_sha256"],
                "topology_sha256": gradient_row["topology_sha256"],
                "candidates": candidates,
            }
        )

    implementation_path = Path(__file__).resolve()
    seal = {
        "schema_version": WAVEFORM_SEAL_SCHEMA_VERSION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "decision": WAVEFORM_SEAL_DECISION,
        "source": dict(source),
        "implementation_sha256": {
            implementation_path.name: sha256_file(implementation_path)
        },
        "input_sha256": {
            "fresh_panel_split_seal": split_seal_sha256,
            "clean_target_label_bank": target_bank_sha256,
            "joint_gradient_manifest": gradient_manifest_sha256,
            "six_gradient_raw_report": six_gradient_raw_report_sha256,
            "six_gradient_report": six_gradient_report_sha256,
            "six_gradient_receipt": six_gradient_receipt_sha256,
            "joint_gate_contract": gate_sha256,
            "target_value_protocol_contract": target_protocol_sha256,
            "prior_panel_speaker_ledger": ledger_sha256,
            "fresh_speaker_source_manifest": source_manifest_sha256,
        },
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "calibration_inverse_gradient_weights": weights,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization": normalization,
        "alpha_grid": list(GLOBAL_ALPHA_GRID),
        "waveform_step": {
            "steps": 1,
            "formula": "base - alpha * rms(base) * gradient / rms(gradient)",
            "gradient_normalization": "waveform_rms_normalized",
            "output_sample_rate": SAMPLE_RATE,
            "output_subtype": OUTPUT_SUBTYPE,
            "emitted_waveform_highpass": False,
        },
        "speaker_count": sum(len(value) for value in speakers_by_split.values()),
        "row_count": len(sealed_rows),
        "unavailable_candidate_count": unavailable_candidates,
        "rows": sealed_rows,
        "candidate_exact_outcomes_opened": False,
        "calibration_exact_outcomes_opened": False,
        "final_exact_outcomes_opened": False,
        "final_waveforms_sealed_before_final_exact_open": True,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    seal_path = output_dir / "waveform_seal.json"
    _write_json(seal_path, seal)
    receipt = {
        "schema_version": WAVEFORM_SEAL_RECEIPT_SCHEMA_VERSION,
        "decision": WAVEFORM_SEAL_DECISION,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "artifact_sha256": {seal_path.name: sha256_file(seal_path)},
        "row_count": len(sealed_rows),
        "unavailable_candidate_count": unavailable_candidates,
        "candidate_exact_outcomes_opened": False,
        "final_exact_outcomes_opened": False,
        "new_sealed_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = output_dir / "completion_receipt.json"
    _write_json(receipt_path, receipt)
    return {"seal": seal, "receipt": receipt, "seal_path": str(seal_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seal", type=Path, required=True)
    parser.add_argument("--split-seal-sha256", required=True)
    parser.add_argument("--clean-target-label-bank", type=Path, required=True)
    parser.add_argument("--clean-target-label-bank-sha256", required=True)
    parser.add_argument("--gradient-manifest", type=Path, required=True)
    parser.add_argument("--gradient-manifest-sha256", required=True)
    parser.add_argument("--six-gradient-raw-report", type=Path, required=True)
    parser.add_argument("--six-gradient-raw-report-sha256", required=True)
    parser.add_argument("--six-gradient-report", type=Path, required=True)
    parser.add_argument("--six-gradient-report-sha256", required=True)
    parser.add_argument("--six-gradient-receipt", type=Path, required=True)
    parser.add_argument("--six-gradient-receipt-sha256", required=True)
    parser.add_argument("--joint-gate-contract", type=Path, required=True)
    parser.add_argument("--joint-gate-contract-sha256", required=True)
    parser.add_argument("--target-value-protocol", type=Path, required=True)
    parser.add_argument("--target-value-protocol-sha256", required=True)
    parser.add_argument("--prior-panel-speaker-ledger", type=Path, required=True)
    parser.add_argument("--prior-panel-speaker-ledger-sha256", required=True)
    parser.add_argument("--fresh-speaker-source-manifest", type=Path, required=True)
    parser.add_argument("--fresh-speaker-source-manifest-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = validate_source(args.source_root, args.source_commit)
    paths_and_hashes = {
        "split_seal": (args.split_seal, args.split_seal_sha256),
        "target_bank": (
            args.clean_target_label_bank,
            args.clean_target_label_bank_sha256,
        ),
        "gradient_manifest": (
            args.gradient_manifest,
            args.gradient_manifest_sha256,
        ),
        "six_gradient_raw_report": (
            args.six_gradient_raw_report,
            args.six_gradient_raw_report_sha256,
        ),
        "six_gradient_report": (
            args.six_gradient_report,
            args.six_gradient_report_sha256,
        ),
        "six_gradient_receipt": (
            args.six_gradient_receipt,
            args.six_gradient_receipt_sha256,
        ),
        "joint_gate_contract": (
            args.joint_gate_contract,
            args.joint_gate_contract_sha256,
        ),
        "target_value_protocol": (
            args.target_value_protocol,
            args.target_value_protocol_sha256,
        ),
        "prior_panel_speaker_ledger": (
            args.prior_panel_speaker_ledger,
            args.prior_panel_speaker_ledger_sha256,
        ),
        "fresh_speaker_source_manifest": (
            args.fresh_speaker_source_manifest,
            args.fresh_speaker_source_manifest_sha256,
        ),
    }
    verified = {
        key: _verified_file(path, expected_hash, key)
        for key, (path, expected_hash) in paths_and_hashes.items()
    }
    result = prepare_and_seal(
        split_seal=_read_json_mapping(verified["split_seal"], "split seal"),
        split_seal_sha256=args.split_seal_sha256,
        target_bank=_read_json_mapping(verified["target_bank"], "target bank"),
        target_bank_sha256=args.clean_target_label_bank_sha256,
        gradient_manifest=_read_json_mapping(
            verified["gradient_manifest"], "gradient manifest"
        ),
        gradient_manifest_sha256=args.gradient_manifest_sha256,
        six_gradient_report=_read_json_mapping(
            verified["six_gradient_report"], "six-gradient report"
        ),
        six_gradient_receipt=_read_json_mapping(
            verified["six_gradient_receipt"], "six-gradient receipt"
        ),
        six_gradient_report_sha256=args.six_gradient_report_sha256,
        six_gradient_receipt_sha256=args.six_gradient_receipt_sha256,
        six_gradient_raw_report=_read_json_mapping(
            verified["six_gradient_raw_report"], "six-gradient raw report"
        ),
        six_gradient_raw_report_sha256=args.six_gradient_raw_report_sha256,
        gate_sha256=args.joint_gate_contract_sha256,
        target_protocol_sha256=args.target_value_protocol_sha256,
        ledger_sha256=args.prior_panel_speaker_ledger_sha256,
        source_manifest_sha256=args.fresh_speaker_source_manifest_sha256,
        source=source,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result["receipt"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
