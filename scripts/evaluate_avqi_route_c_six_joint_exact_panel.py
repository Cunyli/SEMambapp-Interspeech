#!/usr/bin/env python3
"""Run the two-stage exact-Praat decision for a sealed Route C six-joint panel.

Calibration may choose one nonzero global alpha from the frozen grid. Final
scoring requires the hash-bound calibration report and alpha receipt, performs
no tuning, and keeps formal generator training closed even when the bounded
joint panel passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch


if __name__ == "__main__" and __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_avqi_route_c_six_joint_exact_panel",
            *sys.argv[1:],
        ],
        cwd=project_root,
        check=False,
    )
    raise SystemExit(completed.returncode)

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX,
    AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX,
    DEGRADED_EFFICACY_CONDITIONS,
    DENOISING_MEDIAN_CHANGE_MIN_DB,
    DENOISING_WORST_CHANGE_MIN_DB,
    EXACT_IMPROVEMENT_FRACTION_MIN,
    EXPECTED_TOTAL_ROWS,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    GLOBAL_ALPHA_GRID,
    GUARDRAIL_PASS_FRACTION_MIN,
    HEALTHY_ROLE,
    MATERIAL_CASES_ABSOLUTE_MIN,
    MATERIAL_CASES_PER_18_MIN,
    MATERIAL_COVERAGE_FRACTION_MIN,
    MATERIAL_NORMALIZED_BEFORE_GAP_THRESHOLD,
    MEDIAN_NORMALIZED_GAP_REDUCTION_MIN,
    NORMALIZATION_SOURCE,
    PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX,
    PATHOLOGY_DB_WORST_GAP_INCREASE_MAX,
    PAUSE_F1_MEDIAN_DECREASE_MAX,
    PAUSE_F1_WORST_DECREASE_MAX,
    REQUIRED_EFFICACY_SLICES,
    REQUIRED_SLICE_IMPROVEMENT_FRACTION_EXCLUSIVE_MIN,
    REQUIRED_SLICE_MEDIAN_NORMALIZED_REDUCTION_MIN,
    SAFETY_CLIP_FRACTION_MAX,
    SAFETY_COSINE_SIMILARITY_MIN,
    SAFETY_RESIDUAL_RMS_DB_MAX,
    TRAINING_NO_GO,
    _finite_mapping,
    _require_optimizer_zero,
    _validate_panel_rows,
    validate_source,
)
from scripts.evaluate_direct_avqi_waveform_optimization import (
    STEP_VERSIONS,
    aggregate_denoising,
    aggregate_pathology_guardrails,
    avqi_code_tree_sha256,
    full_band_pathology_guardrails,
    waveform_safety,
)
from scripts.prepare_avqi_route_c_six_joint_waveforms import (
    OUTPUT_SUBTYPE,
    SAMPLE_RATE,
    WAVEFORM_SEAL_DECISION,
    WAVEFORM_SEAL_RECEIPT_SCHEMA_VERSION,
    WAVEFORM_SEAL_SCHEMA_VERSION,
)


JOINT_METRIC_NAME = "joint_equal_weight"
EFFICACY_METRICS = (*ROUTE_C_SIX_ACTIVE_COMPONENTS, JOINT_METRIC_NAME)
EXACT_CODE_TREE_MANIFEST_SCHEMA_VERSION = (
    "avqi-route-c-exact-code-tree-manifest-v1"
)
EXACT_RUNTIME_MANIFEST_SCHEMA_VERSION = "avqi-route-c-exact-runtime-manifest-v1"
CALIBRATION_REPORT_SCHEMA_VERSION = (
    "avqi-route-c-six-joint-calibration-exact-report-v1"
)
ALPHA_RECEIPT_SCHEMA_VERSION = "avqi-route-c-six-joint-alpha-receipt-v1"
FINAL_REPORT_SCHEMA_VERSION = "avqi-route-c-six-joint-final-exact-report-v1"
FINAL_RECEIPT_SCHEMA_VERSION = "avqi-route-c-six-joint-final-receipt-v1"
CALIBRATION_PASS_DECISION = "PASS_SIX_JOINT_CALIBRATION_ALPHA_SELECTED"
CALIBRATION_NO_GO_DECISION = "NO_GO_SIX_JOINT_CALIBRATION_FINAL_UNOPENED"
FINAL_PASS_DECISION = "PASS_SIX_JOINT_EXACT_AND_SAFETY_PANEL"
FINAL_NO_GO_DECISION = "NO_GO_SIX_JOINT_EXACT_AND_SAFETY_PANEL"
EXACT_MARKER = "AVQI_SIX_JOINT_EXACT_JSON="

EXACT_SCORER = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import parselmouth
from avqi_code import run_avqi

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    row = {"id": item["id"]}
    try:
        metrics = run_avqi(
            item["path"],
            item["path"],
            target_sr=16000,
            speaking_type=item["view"],
            step_versions=request["step_versions"],
            remove_sv_silence_with_sox=False,
        )
        row.update(
            {
                "status": "ok",
                "components": {
                    name: float(metrics[name])
                    for name in request["components"]
                },
                "error_type": "",
                "error_message": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "components": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
        )
    rows.append(row)
print(
    "AVQI_SIX_JOINT_EXACT_JSON="
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


def _verified_audio(
    binding: Mapping[str, Any],
    label: str,
    *,
    expected_subtype: str | None = None,
) -> torch.Tensor:
    path = Path(str(binding.get("path", ""))).resolve()
    expected_sha256 = str(binding.get("sha256", ""))
    _verified_file(path, expected_sha256, label)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{label} must be nonempty mono 16 kHz audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} contains non-finite samples")
    float32_sha256 = hashlib.sha256(
        np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    ).hexdigest()
    if (
        binding.get("samples") != int(audio.size)
        or binding.get("sample_rate") != sample_rate
        or binding.get("subtype") != info.subtype
        or binding.get("float32_sha256") != float32_sha256
    ):
        raise ValueError(f"{label} readback binding differs")
    if expected_subtype is not None and info.subtype != expected_subtype:
        raise ValueError(f"{label} subtype differs from the sealed contract")
    return torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())


def validate_exact_authority(
    *,
    exact_python: Path,
    avqi_code_root: Path,
    code_manifest: Mapping[str, Any],
    code_manifest_sha256: str,
    runtime_manifest: Mapping[str, Any],
) -> dict[str, str]:
    if code_manifest.get("schema_version") != (
        EXACT_CODE_TREE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("exact AVQI code-tree manifest schema differs")
    resolved_root = avqi_code_root.resolve()
    if code_manifest.get("avqi_code_root") != str(resolved_root):
        raise ValueError("exact AVQI code root differs")
    if not resolved_root.is_dir():
        raise ValueError("exact AVQI code root is unavailable")
    expected_tree_sha256 = str(code_manifest.get("avqi_code_tree_sha256", ""))
    if avqi_code_tree_sha256(resolved_root) != expected_tree_sha256:
        raise ValueError("exact AVQI code-tree hash differs")
    main_path = resolved_root / "avqi_code" / "main.py"
    if sha256_file(main_path) != code_manifest.get("main_py_sha256"):
        raise ValueError("exact AVQI main.py hash differs")
    head = subprocess.run(
        ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != code_manifest.get("avqi_repo_commit"):
        raise ValueError("exact AVQI repository commit differs")
    if subprocess.run(
        ["git", "-C", str(resolved_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("exact AVQI code tree is dirty")

    if runtime_manifest.get("schema_version") != (
        EXACT_RUNTIME_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("exact runtime manifest schema differs")
    resolved_python = exact_python.resolve()
    if (
        not resolved_python.is_file()
        or runtime_manifest.get("exact_python") != str(resolved_python)
        or runtime_manifest.get("avqi_code_tree_manifest_sha256")
        != code_manifest_sha256
        or runtime_manifest.get("avqi_code_tree_sha256")
        != expected_tree_sha256
        or runtime_manifest.get("step_versions") != STEP_VERSIONS
    ):
        raise ValueError("exact runtime binding differs")
    _require_optimizer_zero(runtime_manifest, "exact runtime manifest")
    return {
        "exact_python": str(resolved_python),
        "avqi_code_root": str(resolved_root),
        "avqi_repo_commit": head,
        "avqi_code_tree_sha256": expected_tree_sha256,
        "parselmouth_version": str(runtime_manifest.get("parselmouth_version")),
        "praat_version": str(runtime_manifest.get("praat_version")),
    }


def validate_waveform_seal(
    seal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    seal_path: Path,
    seal_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if (
        seal.get("schema_version") != WAVEFORM_SEAL_SCHEMA_VERSION
        or seal.get("scientific_contract_schema_version")
        != FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        or seal.get("decision") != WAVEFORM_SEAL_DECISION
        or tuple(seal.get("component_order", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or seal.get("alpha_grid") != list(GLOBAL_ALPHA_GRID)
        or seal.get("normalization_source") != NORMALIZATION_SOURCE
    ):
        raise ValueError("six-joint waveform seal header differs")
    if (
        seal.get("candidate_exact_outcomes_opened") is not False
        or seal.get("calibration_exact_outcomes_opened") is not False
        or seal.get("final_exact_outcomes_opened") is not False
        or seal.get("final_waveforms_sealed_before_final_exact_open") is not True
        or seal.get("waveform_step", {}).get("emitted_waveform_highpass")
        is not False
    ):
        raise ValueError("six-joint waveform seal opened or altered candidates")
    _require_optimizer_zero(seal, "six-joint waveform seal")
    if (
        receipt.get("schema_version") != WAVEFORM_SEAL_RECEIPT_SCHEMA_VERSION
        or receipt.get("decision") != WAVEFORM_SEAL_DECISION
        or receipt.get("artifact_sha256", {}).get(seal_path.name)
        != seal_sha256
        or receipt.get("candidate_exact_outcomes_opened") is not False
        or receipt.get("final_exact_outcomes_opened") is not False
    ):
        raise ValueError("six-joint waveform seal receipt differs")
    _require_optimizer_zero(receipt, "six-joint waveform seal receipt")
    source = seal.get("source")
    if (
        not isinstance(source, dict)
        or receipt.get("source_commit") != source.get("head")
        or receipt.get("source_branch") != source.get("branch")
        or seal.get("authoritative_training_decision") != TRAINING_NO_GO
        or receipt.get("authoritative_training_decision") != TRAINING_NO_GO
        or seal.get("formal_generator_training_submitted") is not False
    ):
        raise ValueError("six-joint waveform seal source/training boundary differs")
    prepare_path = Path(__file__).with_name(
        "prepare_avqi_route_c_six_joint_waveforms.py"
    )
    if seal.get("implementation_sha256") != {
        prepare_path.name: sha256_file(prepare_path)
    }:
        raise ValueError("six-joint waveform seal implementation differs")
    rows = seal.get("rows")
    panel_rows, _ = _validate_panel_rows(rows, "six-joint waveform seal")
    if (
        len(panel_rows) != EXPECTED_TOTAL_ROWS
        or len(rows) != EXPECTED_TOTAL_ROWS
        or seal.get("row_count") != EXPECTED_TOTAL_ROWS
        or receipt.get("row_count") != EXPECTED_TOTAL_ROWS
    ):
        raise ValueError("six-joint waveform seal row count differs")
    target_scale = _finite_mapping(
        seal.get("normalization", {}).get("target_scale"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-joint target scales",
        positive=True,
    )
    sealed_root = seal_path.resolve().parent
    targets_by_speaker_view: dict[tuple[str, str], tuple[Any, ...]] = {}
    observed_unavailable = 0
    for row in rows:
        case_id = str(row["case_id"])
        base = row.get("base")
        if not isinstance(base, dict):
            raise ValueError("sealed row lacks a base waveform binding")
        _verified_audio(
            base,
            f"sealed base {case_id}",
            expected_subtype=OUTPUT_SUBTYPE,
        )
        if not Path(str(base["path"])).resolve().is_relative_to(sealed_root):
            raise ValueError("sealed base is outside the immutable seal root")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or [
            candidate.get("alpha") for candidate in candidates
        ] != list(GLOBAL_ALPHA_GRID):
            raise ValueError("sealed candidate alpha coverage differs")
        if row["label"] == "patient":
            target = row.get("target")
            if not isinstance(target, dict):
                raise ValueError("patient row lacks sealed target")
            _verified_audio(target, f"sealed target {case_id}")
            exact_components = _finite_mapping(
                target.get("exact_components"),
                ROUTE_C_SIX_ACTIVE_COMPONENTS,
                f"sealed target components {case_id}",
            )
            target_key = (str(row["speaker_id"]), str(row["view"]))
            if (
                target.get("speaker_id") != target_key[0]
                or target.get("view") != target_key[1]
                or target.get("split") != row["split"]
            ):
                raise ValueError("sealed target speaker/view/split binding differs")
            target_fingerprint = (
                str(Path(str(target["path"])).resolve()),
                target["sha256"],
                target["samples"],
                target["sample_rate"],
                target["subtype"],
                target["float32_sha256"],
                tuple(exact_components[name] for name in ROUTE_C_SIX_ACTIVE_COMPONENTS),
            )
            previous = targets_by_speaker_view.setdefault(
                target_key, target_fingerprint
            )
            if previous != target_fingerprint:
                raise ValueError("same-speaker same-view target binding drifted")
        elif row["optimization_role"] == HEALTHY_ROLE:
            if row.get("target") is not None:
                raise ValueError("healthy row contains a pathological target")
            base_sha256 = row["base"]["sha256"]
            if any(
                candidate.get("available") is not True
                or candidate.get("sha256") != base_sha256
                for candidate in candidates
            ):
                raise ValueError("healthy candidate is not identical to base")
        else:
            raise ValueError("waveform seal contains unknown row role")
        for alpha_index, candidate in enumerate(candidates):
            if candidate.get("available") is True:
                _verified_audio(
                    candidate,
                    f"sealed candidate {case_id}",
                    expected_subtype=OUTPUT_SUBTYPE,
                )
                candidate_path = Path(str(candidate["path"])).resolve()
                if not candidate_path.is_relative_to(sealed_root):
                    raise ValueError(
                        "sealed candidate is outside the immutable seal root"
                    )
                if candidate.get("samples") != base.get("samples"):
                    raise ValueError("sealed candidate sample count differs from base")
                if alpha_index == 0 and candidate.get("sha256") != base.get("sha256"):
                    raise ValueError("zero-alpha candidate differs from sealed base")
            elif candidate.get("available") is False:
                if (
                    any(
                        candidate.get(key) is not None
                        for key in ("path", "sha256", "float32_sha256")
                    )
                    or not isinstance(candidate.get("unavailable_reason"), str)
                    or not candidate["unavailable_reason"]
                ):
                    raise ValueError("unavailable candidate binding differs")
                observed_unavailable += 1
            else:
                raise ValueError("sealed candidate availability flag differs")
    if (
        seal.get("unavailable_candidate_count") != observed_unavailable
        or receipt.get("unavailable_candidate_count") != observed_unavailable
    ):
        raise ValueError("sealed unavailable-candidate count differs")
    return list(rows), target_scale


def run_exact(
    items: list[dict[str, str]],
    *,
    exact_python: Path,
    avqi_code_root: Path,
    expected_runtime: Mapping[str, str],
) -> dict[str, np.ndarray]:
    completed = subprocess.run(
        [str(exact_python), "-c", EXACT_SCORER, str(avqi_code_root)],
        input=json.dumps(
            {
                "items": items,
                "components": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
                "step_versions": STEP_VERSIONS,
            },
            sort_keys=True,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        line[len(EXACT_MARKER) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EXACT_MARKER)
    ]
    if len(records) != 1:
        raise ValueError("exact Praat emitted an unexpected result count")
    payload = json.loads(records[0])
    if (
        payload.get("parselmouth_version")
        != expected_runtime["parselmouth_version"]
        or payload.get("praat_version") != expected_runtime["praat_version"]
    ):
        raise ValueError("exact Praat runtime version drifted")
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != [
        item["id"] for item in items
    ]:
        raise ValueError("exact Praat result order or coverage differs")
    output: dict[str, np.ndarray] = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(
                f"exact Praat failed for {row.get('id')}: "
                f"{row.get('error_type')} {row.get('error_message')}"
            )
        components = row.get("components")
        if not isinstance(components, dict) or set(components) != set(
            ROUTE_C_SIX_ACTIVE_COMPONENTS
        ):
            raise ValueError("exact Praat component order differs")
        values = np.asarray(
            [components[name] for name in ROUTE_C_SIX_ACTIVE_COMPONENTS],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"exact Praat returned non-finite values: {row['id']}")
        output[str(row["id"])] = values
    return output


def exact_gap_record(
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    target_scale: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    scales = np.asarray(
        [target_scale[name] for name in ROUTE_C_SIX_ACTIVE_COMPONENTS],
        dtype=np.float64,
    )
    normalized_before = np.abs(before - target) / scales
    normalized_after = np.abs(after - target) / scales
    output = {
        component: {
            "target": float(target[index]),
            "before": float(before[index]),
            "after": float(after[index]),
            "normalized_gap_before": float(normalized_before[index]),
            "normalized_gap_after": float(normalized_after[index]),
            "normalized_gap_reduction": float(
                normalized_before[index] - normalized_after[index]
            ),
        }
        for index, component in enumerate(ROUTE_C_SIX_ACTIVE_COMPONENTS)
    }
    joint_before = float(np.mean(normalized_before))
    joint_after = float(np.mean(normalized_after))
    output[JOINT_METRIC_NAME] = {
        "target": 0.0,
        "before": joint_before,
        "after": joint_after,
        "normalized_gap_before": joint_before,
        "normalized_gap_after": joint_after,
        "normalized_gap_reduction": joint_before - joint_after,
    }
    return output


def _improvement_fraction(values: list[Mapping[str, float]]) -> float | None:
    if not values:
        return None
    return float(
        np.mean(
            [
                value["normalized_gap_after"]
                < value["normalized_gap_before"]
                for value in values
            ]
        )
    )


def _median_reduction(values: list[Mapping[str, float]]) -> float | None:
    if not values:
        return None
    return float(
        statistics.median(
            value["normalized_gap_reduction"] for value in values
        )
    )


def metric_summary(
    rows: list[Mapping[str, Any]],
    metric: str,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    material = [
        row["metrics"][metric]
        for row in rows
        if row["metrics"][metric]["normalized_gap_before"]
        > MATERIAL_NORMALIZED_BEFORE_GAP_THRESHOLD
    ]
    coverage = len(material) / expected_rows if expected_rows else 0.0
    improvement = _improvement_fraction(material)
    median_reduction = _median_reduction(material)
    scaled_minimum = math.ceil(MATERIAL_CASES_PER_18_MIN * expected_rows / 18)
    gates = {
        "complete_row_coverage": len(rows) == expected_rows,
        "material_coverage_fraction_ge_0_80": (
            coverage >= MATERIAL_COVERAGE_FRACTION_MIN
        ),
        "material_cases_ge_5": len(material) >= MATERIAL_CASES_ABSOLUTE_MIN,
        "material_cases_scaled_from_15_per_18": len(material) >= scaled_minimum,
        "exact_improvement_fraction_ge_0_80": (
            improvement is not None
            and improvement >= EXACT_IMPROVEMENT_FRACTION_MIN
        ),
        "median_normalized_gap_reduction_ge_0_02": (
            median_reduction is not None
            and median_reduction >= MEDIAN_NORMALIZED_GAP_REDUCTION_MIN
        ),
    }
    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "material_rows": len(material),
        "material_coverage_fraction": coverage,
        "improvement_fraction_material": improvement,
        "median_normalized_gap_reduction_material": median_reduction,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def slice_metric_summary(
    rows: list[Mapping[str, Any]],
    metric: str,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    material = [
        row["metrics"][metric]
        for row in rows
        if row["metrics"][metric]["normalized_gap_before"]
        > MATERIAL_NORMALIZED_BEFORE_GAP_THRESHOLD
    ]
    improvement = _improvement_fraction(material)
    median_reduction = _median_reduction(material)
    gates = {
        "complete_row_coverage": len(rows) == expected_rows,
        "material_case_present": bool(material),
        "improvement_fraction_gt_0_50": (
            improvement is not None
            and improvement
            > REQUIRED_SLICE_IMPROVEMENT_FRACTION_EXCLUSIVE_MIN
        ),
        "median_normalized_gap_reduction_ge_0": (
            median_reduction is not None
            and median_reduction
            >= REQUIRED_SLICE_MEDIAN_NORMALIZED_REDUCTION_MIN
        ),
    }
    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "material_rows": len(material),
        "improvement_fraction_material": improvement,
        "median_normalized_gap_reduction_material": median_reduction,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def summarize_exact_efficacy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected_rows = 18
    metrics = {
        metric: metric_summary(rows, metric, expected_rows=expected_rows)
        for metric in EFFICACY_METRICS
    }
    slices: dict[str, dict[str, Any]] = {}
    for slice_name in REQUIRED_EFFICACY_SLICES:
        parts = dict(part.split("=", 1) for part in slice_name.split("&"))
        selected = [
            row
            for row in rows
            if row["condition"] == parts["condition"]
            and row["view"] == parts["view"]
        ]
        slices[slice_name] = {
            metric: slice_metric_summary(selected, metric, expected_rows=3)
            for metric in EFFICACY_METRICS
        }
        slices[slice_name]["decision"] = (
            "PASS"
            if all(
                slices[slice_name][metric]["decision"] == "PASS"
                for metric in EFFICACY_METRICS
            )
            else "FAIL"
        )
    gates = {
        "all_six_components": all(
            metrics[component]["decision"] == "PASS"
            for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
        ),
        "equal_weight_joint": metrics[JOINT_METRIC_NAME]["decision"] == "PASS",
        "all_required_efficacy_slices": all(
            value["decision"] == "PASS" for value in slices.values()
        ),
    }
    return {
        "rows": len(rows),
        "metrics": metrics,
        "required_slices": slices,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def _failed_aggregate(reason: str) -> dict[str, Any]:
    return {"decision": "FAIL", "reason": reason}


def summarize_full_band(
    patient_rows: list[Mapping[str, Any]],
    degraded_rows: list[Mapping[str, Any]],
    *,
    expected_patient_rows: int = 24,
    expected_degraded_rows: int = 18,
) -> dict[str, Any]:
    complete_patient = len(patient_rows) == expected_patient_rows
    complete_degraded = len(degraded_rows) == expected_degraded_rows
    worst_residual = max(
        (float(row["residual_rms_db"]) for row in patient_rows),
        default=math.inf,
    )
    minimum_cosine = min(
        (float(row["cosine_similarity"]) for row in patient_rows),
        default=-math.inf,
    )
    maximum_clip = max(
        (float(row["clip_fraction"]) for row in patient_rows),
        default=math.inf,
    )
    safety_gates = {
        "complete_patient_control_coverage": complete_patient,
        "residual_rms_db_le_minus_50": (
            worst_residual <= SAFETY_RESIDUAL_RMS_DB_MAX
        ),
        "cosine_similarity_ge_0_99999": (
            minimum_cosine >= SAFETY_COSINE_SIMILARITY_MIN
        ),
        "clip_fraction_zero": maximum_clip <= SAFETY_CLIP_FRACTION_MAX,
    }
    pathology = (
        aggregate_pathology_guardrails(list(patient_rows))
        if complete_patient
        else _failed_aggregate("incomplete patient full-band coverage")
    )
    denoising = (
        aggregate_denoising(list(degraded_rows))
        if complete_degraded
        else _failed_aggregate("incomplete degraded denoising coverage")
    )
    gates = {
        "waveform_safety": all(safety_gates.values()),
        "full_band_pathology": pathology["decision"] == "PASS",
        "denoising": denoising["decision"] == "PASS",
    }
    return {
        "safety": {
            "rows": len(patient_rows),
            "expected_rows": expected_patient_rows,
            "worst_residual_rms_db": worst_residual,
            "minimum_cosine_similarity": minimum_cosine,
            "maximum_clip_fraction": maximum_clip,
            "gates": safety_gates,
            "decision": "PASS" if all(safety_gates.values()) else "FAIL",
        },
        "full_band_pathology": pathology,
        "denoising": denoising,
        "frozen_thresholds": {
            "pathology_db_median_gap_increase_max": (
                PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX
            ),
            "pathology_db_worst_gap_increase_max": (
                PATHOLOGY_DB_WORST_GAP_INCREASE_MAX
            ),
            "airflow_flatness_median_gap_increase_max": (
                AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX
            ),
            "airflow_flatness_worst_gap_increase_max": (
                AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX
            ),
            "pause_f1_median_decrease_max": PAUSE_F1_MEDIAN_DECREASE_MAX,
            "pause_f1_worst_decrease_max": PAUSE_F1_WORST_DECREASE_MAX,
            "guardrail_pass_fraction_min": GUARDRAIL_PASS_FRACTION_MIN,
            "denoising_median_change_min_db": DENOISING_MEDIAN_CHANGE_MIN_DB,
            "denoising_worst_change_min_db": DENOISING_WORST_CHANGE_MIN_DB,
        },
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def _candidate_for_alpha(
    row: Mapping[str, Any], alpha: float
) -> Mapping[str, Any] | None:
    matches = [
        candidate
        for candidate in row["candidates"]
        if candidate.get("alpha") == alpha
    ]
    if len(matches) != 1 or matches[0].get("available") is not True:
        return None
    return matches[0]


def exact_items_for_stage(
    rows: list[Mapping[str, Any]],
    *,
    split: str,
    alphas: tuple[float, ...],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    observed_ids: set[str] = set()

    def append(item_id: str, binding: Mapping[str, Any], view: str) -> None:
        if item_id in observed_ids:
            return
        observed_ids.add(item_id)
        items.append(
            {"id": item_id, "path": str(binding["path"]), "view": view}
        )

    for row in rows:
        if row["split"] != split or row["label"] != "patient":
            continue
        target_id = f"target:{split}:{row['speaker_id']}:{row['view']}"
        append(target_id, row["target"], str(row["view"]))
        append(f"base:{row['case_id']}", row["base"], str(row["view"]))
        for alpha_index, alpha in enumerate(alphas):
            candidate = _candidate_for_alpha(row, alpha)
            if candidate is not None:
                append(
                    f"candidate:{alpha_index}:{row['case_id']}",
                    candidate,
                    str(row["view"]),
                )
    return items


def stage_rows(
    rows: list[Mapping[str, Any]],
    exact: Mapping[str, np.ndarray],
    target_scale: Mapping[str, float],
    *,
    split: str,
    alpha: float,
    alpha_index: int,
) -> dict[str, Any]:
    efficacy_rows: list[dict[str, Any]] = []
    clean_control_rows: list[dict[str, Any]] = []
    patient_guardrail_rows: list[dict[str, Any]] = []
    degraded_guardrail_rows: list[dict[str, Any]] = []
    healthy_rows = [
        row for row in rows if row["split"] == split and row["label"] == "healthy"
    ]
    healthy_identity_pass = all(
        _candidate_for_alpha(row, alpha) is not None
        and _candidate_for_alpha(row, alpha)["sha256"] == row["base"]["sha256"]
        for row in healthy_rows
    )
    for row in rows:
        if row["split"] != split or row["label"] != "patient":
            continue
        candidate = _candidate_for_alpha(row, alpha)
        if candidate is None:
            continue
        case_id = str(row["case_id"])
        target_id = f"target:{split}:{row['speaker_id']}:{row['view']}"
        base_id = f"base:{case_id}"
        candidate_id = f"candidate:{alpha_index}:{case_id}"
        if not all(key in exact for key in (target_id, base_id, candidate_id)):
            continue
        target = exact[target_id]
        frozen_target = np.asarray(
            [
                row["target"]["exact_components"][component]
                for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
            ],
            dtype=np.float64,
        )
        if not np.allclose(target, frozen_target, rtol=0.0, atol=1e-9):
            raise ValueError(f"exact clean target drifted: {case_id}")
        metrics = exact_gap_record(
            target,
            exact[base_id],
            exact[candidate_id],
            target_scale,
        )
        result_row = {
            "case_id": case_id,
            "speaker_id": row["speaker_id"],
            "split": split,
            "view": row["view"],
            "condition": row["condition"],
            "alpha": alpha,
            "metrics": metrics,
        }
        if row["condition"] in DEGRADED_EFFICACY_CONDITIONS:
            efficacy_rows.append(result_row)
        else:
            clean_control_rows.append(result_row)

        target_waveform = _verified_audio(row["target"], f"target {case_id}")
        base_waveform = _verified_audio(row["base"], f"base {case_id}")
        candidate_waveform = _verified_audio(candidate, f"candidate {case_id}")
        guardrail_row = {
            "case_id": case_id,
            "condition": row["condition"],
            **waveform_safety(base_waveform, candidate_waveform),
            **full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            ),
        }
        patient_guardrail_rows.append(guardrail_row)
        if row["condition"] in DEGRADED_EFFICACY_CONDITIONS:
            degraded_guardrail_rows.append(guardrail_row)

    efficacy = summarize_exact_efficacy(efficacy_rows)
    full_band = summarize_full_band(
        patient_guardrail_rows,
        degraded_guardrail_rows,
    )
    gates = {
        **efficacy["gates"],
        **full_band["gates"],
        "healthy_no_step_identity": (
            len(healthy_rows) == 24 and healthy_identity_pass
        ),
    }
    return {
        "alpha": alpha,
        "efficacy": efficacy,
        "patient_clean_exact_controls": {
            "rows": len(clean_control_rows),
            "expected_rows": 6,
            "observational_only": True,
            "results": clean_control_rows,
        },
        "full_band": full_band,
        "healthy_controls": {
            "rows": len(healthy_rows),
            "expected_rows": 24,
            "candidate_sha256_equals_base_sha256": healthy_identity_pass,
            "optimized_healthy_safety_claimed": False,
        },
        "efficacy_rows": efficacy_rows,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def choose_alpha(summaries: Mapping[float, Mapping[str, Any]]) -> float | None:
    passing = [
        alpha
        for alpha, summary in summaries.items()
        if alpha > 0.0 and summary.get("decision") == "PASS"
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda alpha: (
            -float(
                summaries[alpha]["efficacy"]["metrics"][JOINT_METRIC_NAME][
                    "median_normalized_gap_reduction_material"
                ]
            ),
            alpha,
        ),
    )


def run_calibration(
    *,
    rows: list[dict[str, Any]],
    target_scale: Mapping[str, float],
    waveform_seal_sha256: str,
    waveform_seal_receipt_sha256: str,
    exact_python: Path,
    avqi_code_root: Path,
    exact_authority: Mapping[str, str],
    exact_code_tree_manifest_sha256: str,
    exact_runtime_manifest_sha256: str,
    source: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    alphas = tuple(GLOBAL_ALPHA_GRID)
    items = exact_items_for_stage(rows, split="calibration", alphas=alphas)
    exact = run_exact(
        items,
        exact_python=exact_python,
        avqi_code_root=avqi_code_root,
        expected_runtime=exact_authority,
    )
    summaries = {
        alpha: stage_rows(
            rows,
            exact,
            target_scale,
            split="calibration",
            alpha=alpha,
            alpha_index=index,
        )
        for index, alpha in enumerate(alphas)
    }
    selected_alpha = choose_alpha(summaries)
    decision = (
        CALIBRATION_PASS_DECISION
        if selected_alpha is not None
        else CALIBRATION_NO_GO_DECISION
    )
    report = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "decision": decision,
        "source": dict(source),
        "waveform_seal_sha256": waveform_seal_sha256,
        "waveform_seal_receipt_sha256": waveform_seal_receipt_sha256,
        "exact_authority": dict(exact_authority),
        "exact_code_tree_manifest_sha256": exact_code_tree_manifest_sha256,
        "exact_runtime_manifest_sha256": exact_runtime_manifest_sha256,
        "alpha_grid": list(alphas),
        "alpha_selection_split": "calibration",
        "alpha_selection_objective": (
            "maximize_equal_weight_joint_exact_median_normalized_gap_reduction"
        ),
        "alpha_selection_tie_break": "smaller_alpha",
        "selected_alpha": selected_alpha,
        "summaries": {str(alpha): summary for alpha, summary in summaries.items()},
        "candidate_exact_outcomes_opened": True,
        "calibration_exact_outcomes_opened": True,
        "final_exact_outcomes_opened": False,
        "final_tuning_permitted": False,
        "joint_scientific_promotion_granted": False,
        "one_batch_generator_gradient_check_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    report_path = output_dir / "calibration_exact_report.json"
    _write_json(report_path, report)
    receipt = {
        "schema_version": ALPHA_RECEIPT_SCHEMA_VERSION,
        "decision": decision,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "waveform_seal_sha256": waveform_seal_sha256,
        "waveform_seal_receipt_sha256": waveform_seal_receipt_sha256,
        "exact_code_tree_manifest_sha256": exact_code_tree_manifest_sha256,
        "exact_runtime_manifest_sha256": exact_runtime_manifest_sha256,
        "selected_alpha": selected_alpha,
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
        "calibration_exact_outcomes_opened": True,
        "final_exact_outcomes_opened": False,
        "final_waveforms_sealed_before_final_exact_open": True,
        "joint_scientific_promotion_granted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    receipt_path = output_dir / "alpha_selection_receipt.json"
    _write_json(receipt_path, receipt)
    return {"report": report, "receipt": receipt}


def validate_calibration_authorization(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    report_path: Path,
    report_sha256: str,
    waveform_seal_sha256: str,
    waveform_seal_receipt_sha256: str,
    exact_code_tree_manifest_sha256: str,
    exact_runtime_manifest_sha256: str,
    source_commit: str,
    exact_authority: Mapping[str, str],
) -> float:
    if (
        report.get("schema_version") != CALIBRATION_REPORT_SCHEMA_VERSION
        or report.get("decision") != CALIBRATION_PASS_DECISION
        or report.get("final_exact_outcomes_opened") is not False
        or report.get("joint_scientific_promotion_granted") is not False
    ):
        raise ValueError("calibration report does not authorize final exact open")
    if (
        receipt.get("schema_version") != ALPHA_RECEIPT_SCHEMA_VERSION
        or receipt.get("decision") != CALIBRATION_PASS_DECISION
        or receipt.get("artifact_sha256", {}).get(report_path.name)
        != report_sha256
        or receipt.get("final_exact_outcomes_opened") is not False
        or receipt.get("final_waveforms_sealed_before_final_exact_open")
        is not True
    ):
        raise ValueError("alpha-selection receipt does not bind calibration")
    expected = {
        "waveform_seal_sha256": waveform_seal_sha256,
        "waveform_seal_receipt_sha256": waveform_seal_receipt_sha256,
        "exact_code_tree_manifest_sha256": exact_code_tree_manifest_sha256,
        "exact_runtime_manifest_sha256": exact_runtime_manifest_sha256,
    }
    if any(
        report.get(key) != value or receipt.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("calibration/final evidence binding differs")
    report_source = report.get("source")
    if (
        not isinstance(report_source, dict)
        or report_source.get("head") != source_commit
        or receipt.get("source_commit") != source_commit
        or receipt.get("source_branch") != report_source.get("branch")
        or report.get("exact_authority") != dict(exact_authority)
        or report.get("candidate_exact_outcomes_opened") is not True
        or report.get("calibration_exact_outcomes_opened") is not True
        or report.get("final_tuning_permitted") is not False
        or report.get("one_batch_generator_gradient_check_authorized")
        is not False
        or report.get("formal_generator_training_submitted") is not False
        or report.get("authoritative_training_decision") != TRAINING_NO_GO
        or receipt.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("calibration source/runtime/training boundary differs")
    _require_optimizer_zero(report, "calibration exact report")
    _require_optimizer_zero(receipt, "alpha-selection receipt")
    selected_alpha = report.get("selected_alpha")
    summaries = report.get("summaries")
    expected_summary_keys = {str(alpha) for alpha in GLOBAL_ALPHA_GRID}
    if not isinstance(summaries, dict) or set(summaries) != expected_summary_keys:
        raise ValueError("calibration alpha summary coverage differs")
    parsed_summaries = {
        alpha: summaries[str(alpha)] for alpha in GLOBAL_ALPHA_GRID
    }
    if (
        selected_alpha != receipt.get("selected_alpha")
        or selected_alpha not in GLOBAL_ALPHA_GRID
        or float(selected_alpha) <= 0.0
        or not isinstance(parsed_summaries[selected_alpha], dict)
        or parsed_summaries[selected_alpha].get("decision") != "PASS"
        or choose_alpha(parsed_summaries) != selected_alpha
    ):
        raise ValueError("selected calibration alpha differs")
    return float(selected_alpha)


def run_final(
    *,
    rows: list[dict[str, Any]],
    target_scale: Mapping[str, float],
    selected_alpha: float,
    waveform_seal_sha256: str,
    waveform_seal_receipt_sha256: str,
    calibration_report_sha256: str,
    alpha_receipt_sha256: str,
    exact_python: Path,
    avqi_code_root: Path,
    exact_authority: Mapping[str, str],
    exact_code_tree_manifest_sha256: str,
    exact_runtime_manifest_sha256: str,
    source: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    alpha_index = GLOBAL_ALPHA_GRID.index(selected_alpha)
    items = exact_items_for_stage(
        rows,
        split="final",
        alphas=(selected_alpha,),
    )
    exact = run_exact(
        items,
        exact_python=exact_python,
        avqi_code_root=avqi_code_root,
        expected_runtime=exact_authority,
    )
    # The one-alpha item list uses local index zero. Normalize IDs to the frozen
    # grid index expected by stage_rows without reopening any other alpha.
    if alpha_index != 0:
        exact = {
            (
                key.replace("candidate:0:", f"candidate:{alpha_index}:", 1)
                if key.startswith("candidate:0:")
                else key
            ): value
            for key, value in exact.items()
        }
    summary = stage_rows(
        rows,
        exact,
        target_scale,
        split="final",
        alpha=selected_alpha,
        alpha_index=alpha_index,
    )
    passed = summary["decision"] == "PASS"
    decision = FINAL_PASS_DECISION if passed else FINAL_NO_GO_DECISION
    report = {
        "schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "decision": decision,
        "source": dict(source),
        "waveform_seal_sha256": waveform_seal_sha256,
        "waveform_seal_receipt_sha256": waveform_seal_receipt_sha256,
        "calibration_report_sha256": calibration_report_sha256,
        "alpha_selection_receipt_sha256": alpha_receipt_sha256,
        "exact_code_tree_manifest_sha256": exact_code_tree_manifest_sha256,
        "exact_runtime_manifest_sha256": exact_runtime_manifest_sha256,
        "exact_authority": dict(exact_authority),
        "selected_alpha": selected_alpha,
        "final": summary,
        "candidate_exact_outcomes_opened": True,
        "calibration_exact_outcomes_opened": True,
        "final_exact_outcomes_opened": True,
        "final_tuning_permitted": False,
        "joint_scientific_promotion_granted": passed,
        "one_batch_generator_gradient_check_authorized": passed,
        "formal_generator_training_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    report_path = output_dir / "final_exact_report.json"
    _write_json(report_path, report)
    receipt = {
        "schema_version": FINAL_RECEIPT_SCHEMA_VERSION,
        "decision": decision,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "selected_alpha": selected_alpha,
        "joint_scientific_promotion_granted": passed,
        "one_batch_generator_gradient_check_authorized": passed,
        "formal_generator_training_authorized": False,
        "artifact_sha256": {report_path.name: sha256_file(report_path)},
        "input_sha256": {
            "waveform_seal": waveform_seal_sha256,
            "waveform_seal_receipt": waveform_seal_receipt_sha256,
            "calibration_exact_report": calibration_report_sha256,
            "alpha_selection_receipt": alpha_receipt_sha256,
            "exact_code_tree_manifest": exact_code_tree_manifest_sha256,
            "exact_runtime_manifest": exact_runtime_manifest_sha256,
        },
        "final_exact_outcomes_opened": True,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    return {"report": report, "receipt": receipt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "final"), required=True)
    parser.add_argument("--waveform-seal", type=Path, required=True)
    parser.add_argument("--waveform-seal-sha256", required=True)
    parser.add_argument("--waveform-seal-receipt", type=Path, required=True)
    parser.add_argument("--waveform-seal-receipt-sha256", required=True)
    parser.add_argument("--exact-code-tree-manifest", type=Path, required=True)
    parser.add_argument("--exact-code-tree-manifest-sha256", required=True)
    parser.add_argument("--exact-runtime-manifest", type=Path, required=True)
    parser.add_argument("--exact-runtime-manifest-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--calibration-report-sha256")
    parser.add_argument("--alpha-selection-receipt", type=Path)
    parser.add_argument("--alpha-selection-receipt-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite exact output: {args.output_dir}")
    source = validate_source(args.source_root, args.source_commit)
    seal_path = _verified_file(
        args.waveform_seal, args.waveform_seal_sha256, "waveform seal"
    )
    seal_receipt_path = _verified_file(
        args.waveform_seal_receipt,
        args.waveform_seal_receipt_sha256,
        "waveform seal receipt",
    )
    code_manifest_path = _verified_file(
        args.exact_code_tree_manifest,
        args.exact_code_tree_manifest_sha256,
        "exact code-tree manifest",
    )
    runtime_manifest_path = _verified_file(
        args.exact_runtime_manifest,
        args.exact_runtime_manifest_sha256,
        "exact runtime manifest",
    )
    seal = _read_json_mapping(seal_path, "waveform seal")
    seal_receipt = _read_json_mapping(seal_receipt_path, "waveform seal receipt")
    rows, target_scale = validate_waveform_seal(
        seal,
        seal_receipt,
        seal_path,
        args.waveform_seal_sha256,
    )
    exact_authority = validate_exact_authority(
        exact_python=args.exact_python,
        avqi_code_root=args.avqi_code_root,
        code_manifest=_read_json_mapping(code_manifest_path, "exact code manifest"),
        code_manifest_sha256=args.exact_code_tree_manifest_sha256,
        runtime_manifest=_read_json_mapping(
            runtime_manifest_path, "exact runtime manifest"
        ),
    )
    if args.stage == "calibration":
        if any(
            value is not None
            for value in (
                args.calibration_report,
                args.calibration_report_sha256,
                args.alpha_selection_receipt,
                args.alpha_selection_receipt_sha256,
            )
        ):
            raise ValueError("calibration stage accepts no prior exact artifacts")
        args.output_dir.mkdir(parents=True)
        result = run_calibration(
            rows=rows,
            target_scale=target_scale,
            waveform_seal_sha256=args.waveform_seal_sha256,
            waveform_seal_receipt_sha256=args.waveform_seal_receipt_sha256,
            exact_python=args.exact_python.resolve(),
            avqi_code_root=args.avqi_code_root.resolve(),
            exact_authority=exact_authority,
            exact_code_tree_manifest_sha256=(
                args.exact_code_tree_manifest_sha256
            ),
            exact_runtime_manifest_sha256=args.exact_runtime_manifest_sha256,
            source=source,
            output_dir=args.output_dir,
        )
    else:
        required = (
            args.calibration_report,
            args.calibration_report_sha256,
            args.alpha_selection_receipt,
            args.alpha_selection_receipt_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("final stage requires bound calibration artifacts")
        calibration_path = _verified_file(
            args.calibration_report,
            args.calibration_report_sha256,
            "calibration exact report",
        )
        alpha_receipt_path = _verified_file(
            args.alpha_selection_receipt,
            args.alpha_selection_receipt_sha256,
            "alpha-selection receipt",
        )
        selected_alpha = validate_calibration_authorization(
            _read_json_mapping(calibration_path, "calibration exact report"),
            _read_json_mapping(alpha_receipt_path, "alpha-selection receipt"),
            report_path=calibration_path,
            report_sha256=args.calibration_report_sha256,
            waveform_seal_sha256=args.waveform_seal_sha256,
            waveform_seal_receipt_sha256=args.waveform_seal_receipt_sha256,
            exact_code_tree_manifest_sha256=(
                args.exact_code_tree_manifest_sha256
            ),
            exact_runtime_manifest_sha256=args.exact_runtime_manifest_sha256,
            source_commit=source["head"],
            exact_authority=exact_authority,
        )
        args.output_dir.mkdir(parents=True)
        result = run_final(
            rows=rows,
            target_scale=target_scale,
            selected_alpha=selected_alpha,
            waveform_seal_sha256=args.waveform_seal_sha256,
            waveform_seal_receipt_sha256=args.waveform_seal_receipt_sha256,
            calibration_report_sha256=args.calibration_report_sha256,
            alpha_receipt_sha256=args.alpha_selection_receipt_sha256,
            exact_python=args.exact_python.resolve(),
            avqi_code_root=args.avqi_code_root.resolve(),
            exact_authority=exact_authority,
            exact_code_tree_manifest_sha256=(
                args.exact_code_tree_manifest_sha256
            ),
            exact_runtime_manifest_sha256=args.exact_runtime_manifest_sha256,
            source=source,
            output_dir=args.output_dir,
        )
    print(json.dumps(result["receipt"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
