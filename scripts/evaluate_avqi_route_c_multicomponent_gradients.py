"""Audit four-active Route C waveform gradients on non-final dev splits."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any

import soundfile as sf
import torch

from model.avqi_components import (
    AVQI_COMPONENT_NAMES,
    AVQI_V0301_COEFFICIENTS,
    AVQI_V0301_EXPANDED_COEFFICIENTS,
    AVQI_V0301_INTERCEPT,
    AVQI_V0301_SCALE,
)
from model.avqi_route_c import (
    ROUTE_C_ACTIVE_COMPONENTS,
    ROUTE_C_FOUR_ACTIVE_ARCHITECTURE,
    active_bidirectional_gap_losses,
    load_route_c_four_active_scorer,
    route_c_registry_records,
    sha256_file,
)


SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 48_000
AUDIT_SPLITS = ("surrogate_calibration", "surrogate_holdout")
SELECTION_STRATA = (
    ("pathological_mild", "cs"),
    ("pathological_mild", "sv"),
    ("pathological_severe", "cs"),
    ("pathological_severe", "sv"),
)
INPUT_GRADIENT_NORM_MAX = 1e4
NONZERO_GRADIENT_NORM_MIN = 1e-10
MAX_WEIGHTED_COMPONENT_NORM_SHARE = 0.80
ACCEPTED_INTEGRATION_BASE = "2390ce0543d8c17c6e249160333855229c689434"
REQUIRED_EVIDENCE = {
    "cpps_report",
    "cpps_receipt",
    "hnr_report",
    "hnr_receipt",
    "shimmer_percent_report",
    "shimmer_percent_receipt",
    "tilt_report",
    "tilt_receipt",
}


@dataclass(frozen=True)
class AuditCase:
    split: str
    speaker_id: str
    sample_id: str
    sample_group: str
    view: str
    condition: str
    waveform_path: Path
    waveform_sha256: str
    clean_target: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--label-bank-sha256", required=True)
    parser.add_argument("--cpps-checkpoint", type=Path, required=True)
    parser.add_argument("--cpps-checkpoint-sha256", required=True)
    parser.add_argument("--hnr-checkpoint", type=Path, required=True)
    parser.add_argument("--hnr-checkpoint-sha256", required=True)
    parser.add_argument("--shimmer-checkpoint", type=Path, required=True)
    parser.add_argument("--shimmer-checkpoint-sha256", required=True)
    parser.add_argument("--tilt-checkpoint", type=Path, required=True)
    parser.add_argument("--tilt-checkpoint-sha256", required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accepted-base-commit", default=ACCEPTED_INTEGRATION_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def repository_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_source(root: Path, source_commit: str, accepted_base: str) -> dict[str, str]:
    head = repository_value(root, "rev-parse", "HEAD")
    if head != source_commit:
        raise ValueError(f"source HEAD drifted: {head} != {source_commit}")
    if repository_value(root, "status", "--porcelain"):
        raise ValueError("source worktree is dirty")
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", accepted_base, head],
        check=True,
    )
    return {
        "head": head,
        "branch": repository_value(root, "branch", "--show-current"),
        "accepted_base_commit": accepted_base,
    }


def row_sample_id(row: dict[str, str]) -> str:
    return (
        row.get("sample_id", "").strip()
        or row.get("pair_id", "").strip()
        or row["speaker_id"]
    )


def clean_target_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["speaker_id"],
        row_sample_id(row),
        row["split"],
        row["view"],
    )


def component_tensor(row: dict[str, str]) -> torch.Tensor:
    value = torch.tensor(
        [float(row[name]) for name in AVQI_COMPONENT_NAMES],
        dtype=torch.float32,
    )
    if not torch.isfinite(value).all():
        raise ValueError("non-finite exact AVQI target")
    return value


def selection_rank(salt: str, row: dict[str, str]) -> str:
    payload = "\0".join(
        (
            salt,
            row["split"],
            row["sample_group"],
            row["view"],
            row["speaker_id"],
            row_sample_id(row),
        )
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_label_bank(
    path: Path,
    expected_sha256: str,
    selection_salt: str,
) -> tuple[list[AuditCase], torch.Tensor, torch.Tensor, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("label-bank hash mismatch")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    task_rows = [row for row in rows if row["view"] in {"cs", "sv"}]
    observed_splits = {row["split"] for row in task_rows}
    forbidden = {split for split in observed_splits if "final" in split.lower()}
    if forbidden:
        raise ValueError(f"label bank exposes forbidden final splits: {forbidden}")
    exact_rows = [row for row in task_rows if row["scoring_status"] == "ok"]
    clean_rows = [row for row in exact_rows if row["condition_id"] == "clean"]
    clean_by_key = {clean_target_key(row): row for row in clean_rows}
    if len(clean_by_key) != len(clean_rows):
        raise ValueError("duplicate same-speaker clean targets")
    usable_rows = [row for row in exact_rows if clean_target_key(row) in clean_by_key]
    training_targets = torch.stack(
        [
            component_tensor(row)
            for row in usable_rows
            if row["split"] == "surrogate_train"
        ]
    )
    target_mean = training_targets.mean(dim=0)
    target_scale = training_targets.std(dim=0, unbiased=False).clamp_min(1e-6)

    candidates = [
        row
        for row in usable_rows
        if row["split"] in AUDIT_SPLITS
        and row["condition_id"] == "aug16k_phone"
        and row["label"] == "patient"
        and (row["sample_group"], row["view"]) in SELECTION_STRATA
    ]
    selected_rows: list[dict[str, str]] = []
    for split in AUDIT_SPLITS:
        used_speakers: set[str] = set()
        for sample_group, view in SELECTION_STRATA:
            eligible = sorted(
                (
                    row
                    for row in candidates
                    if row["split"] == split
                    and row["sample_group"] == sample_group
                    and row["view"] == view
                    and row["speaker_id"] not in used_speakers
                ),
                key=lambda row: selection_rank(selection_salt, row),
            )
            if not eligible:
                raise ValueError(
                    f"no unique dev case for {split}/{sample_group}/{view}"
                )
            selected_rows.append(eligible[0])
            used_speakers.add(eligible[0]["speaker_id"])

    split_speakers = {
        split: {
            row["speaker_id"] for row in selected_rows if row["split"] == split
        }
        for split in AUDIT_SPLITS
    }
    overlap = split_speakers[AUDIT_SPLITS[0]] & split_speakers[AUDIT_SPLITS[1]]
    if overlap:
        raise ValueError(f"calibration/holdout speaker overlap: {sorted(overlap)}")
    cases = []
    for row in selected_rows:
        view = row["view"]
        clean_row = clean_by_key[clean_target_key(row)]
        cases.append(
            AuditCase(
                split=row["split"],
                speaker_id=row["speaker_id"],
                sample_id=row_sample_id(row),
                sample_group=row["sample_group"],
                view=view,
                condition=row["condition_id"],
                waveform_path=Path(row[f"{view}_path"]),
                waveform_sha256=row[f"{view}_sha256"],
                clean_target=component_tensor(clean_row),
            )
        )
    selection = {
        "selection_salt": selection_salt,
        "allowed_splits": list(AUDIT_SPLITS),
        "final_panel_opened": False,
        "cases": len(cases),
        "cases_by_split": {
            split: sum(case.split == split for case in cases)
            for split in AUDIT_SPLITS
        },
        "speakers_by_split": {
            split: sorted(split_speakers[split]) for split in AUDIT_SPLITS
        },
        "speaker_overlap": 0,
        "strata": [f"{group}/{view}" for group, view in SELECTION_STRATA],
        "target_stat_rows": int(training_targets.shape[0]),
    }
    return cases, target_mean, target_scale, selection


def load_fixed_segment(case: AuditCase) -> torch.Tensor:
    if sha256_file(case.waveform_path) != case.waveform_sha256:
        raise ValueError(f"audio hash mismatch: {case.waveform_path}")
    audio, sample_rate = sf.read(
        case.waveform_path,
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0:
        raise ValueError(f"invalid 16 kHz mono audio: {case.waveform_path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite audio: {case.waveform_path}")
    if waveform.numel() >= SEGMENT_SAMPLES:
        return waveform[:SEGMENT_SAMPLES]
    return torch.nn.functional.pad(waveform, (0, SEGMENT_SAMPLES - waveform.numel()))


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 0.0:
        return math.nan
    return float(torch.dot(first.reshape(-1), second.reshape(-1)) / denominator)


def extract_case_gradients(
    scorer: torch.nn.Module,
    case: AuditCase,
    device: torch.device,
) -> dict[str, Any]:
    waveform = load_fixed_segment(case).to(device).requires_grad_(True)
    raw_target = case.clean_target.to(device).unsqueeze(0)
    prediction = scorer(waveform.unsqueeze(0), case.view)
    losses = active_bidirectional_gap_losses(
        prediction,
        raw_target,
        scorer.target_mean,
        scorer.target_scale,
    )[0]
    normalized_target = scorer.normalized_target(raw_target)[0]
    denormalized_prediction = scorer.denormalized_prediction(prediction)[0]
    gradients: dict[str, torch.Tensor] = {}
    components: dict[str, dict[str, Any]] = {}
    for offset, component in enumerate(ROUTE_C_ACTIVE_COMPONENTS):
        gradient = torch.autograd.grad(
            losses[offset],
            waveform,
            retain_graph=offset < len(ROUTE_C_ACTIVE_COMPONENTS) - 1,
            create_graph=False,
        )[0].detach().cpu()
        norm = float(torch.linalg.vector_norm(gradient))
        finite = bool(torch.isfinite(gradient).all()) and math.isfinite(norm)
        gates = {
            "finite": finite,
            "nonzero": norm > NONZERO_GRADIENT_NORM_MIN,
            "bounded": norm <= INPUT_GRADIENT_NORM_MAX,
        }
        index = AVQI_COMPONENT_NAMES.index(component)
        gradients[component] = gradient
        components[component] = {
            "prediction": float(denormalized_prediction[index].detach().cpu()),
            "clean_target": float(case.clean_target[index]),
            "normalized_signed_error": float(
                (prediction[0, index] - normalized_target[index]).detach().cpu()
            ),
            "normalized_bidirectional_gap": float(
                (prediction[0, index] - normalized_target[index]).abs().detach().cpu()
            ),
            "smooth_l1_loss": float(losses[offset].detach().cpu()),
            "gradient_norm": norm,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
    return {
        "split": case.split,
        "speaker_id": case.speaker_id,
        "sample_id": case.sample_id,
        "sample_group": case.sample_group,
        "view": case.view,
        "condition": case.condition,
        "audio_sha256": case.waveform_sha256,
        "segment_samples": SEGMENT_SAMPLES,
        "components": components,
        "_gradients": gradients,
    }


def frozen_inverse_gradient_weights(
    calibration_records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    if not calibration_records:
        raise ValueError("no calibration gradient records")
    median_norms = {
        component: statistics.median(
            record["components"][component]["gradient_norm"]
            for record in calibration_records
        )
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    if any(
        not math.isfinite(norm) or norm <= NONZERO_GRADIENT_NORM_MIN
        for norm in median_norms.values()
    ):
        raise ValueError(f"invalid calibration gradient medians: {median_norms}")
    minimum = min(median_norms.values())
    weights = {
        component: minimum / median_norms[component]
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    return median_norms, weights


def finalize_case(
    record: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    if set(weights) != set(ROUTE_C_ACTIVE_COMPONENTS):
        raise ValueError("joint gradient weights differ from active components")
    if any(
        not math.isfinite(weight) or weight <= 0.0 for weight in weights.values()
    ):
        raise ValueError("joint gradient weights must be finite and positive")
    gradients = record.pop("_gradients")
    weighted = {
        component: gradients[component] * weights[component]
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    weight_sum = sum(weights.values())
    joint = sum(weighted.values()) / weight_sum
    joint_norm = float(torch.linalg.vector_norm(joint))
    joint_finite = bool(torch.isfinite(joint).all()) and math.isfinite(joint_norm)
    weighted_norms = {
        component: float(torch.linalg.vector_norm(value))
        for component, value in weighted.items()
    }
    weighted_norm_sum = sum(weighted_norms.values())
    shares = {
        component: weighted_norms[component] / weighted_norm_sum
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    pairwise: dict[str, dict[str, Any]] = {}
    for first_index, first in enumerate(ROUTE_C_ACTIVE_COMPONENTS):
        for second in ROUTE_C_ACTIVE_COMPONENTS[first_index + 1 :]:
            value = cosine(gradients[first], gradients[second])
            pairwise[f"{first}__{second}"] = {
                "cosine": value,
                "direction_conflict": value < 0.0,
            }
    component_to_joint = {
        component: {
            "cosine": cosine(gradients[component], joint),
            "opposed_to_joint": cosine(gradients[component], joint) < 0.0,
        }
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    maximum_share = max(shares.values())
    joint_gates = {
        "finite": joint_finite,
        "nonzero": joint_norm > NONZERO_GRADIENT_NORM_MIN,
        "bounded": joint_norm <= INPUT_GRADIENT_NORM_MAX,
        "no_component_norm_share_above_0_80": (
            maximum_share <= MAX_WEIGHTED_COMPONENT_NORM_SHARE
        ),
    }
    record["joint"] = {
        "gradient_norm": joint_norm,
        "component_weights": weights,
        "weighted_component_gradient_norms": weighted_norms,
        "weighted_component_norm_shares": shares,
        "maximum_component_norm_share": maximum_share,
        "dominant_component": max(shares, key=shares.__getitem__),
        "pairwise_component_cosines": pairwise,
        "component_to_joint_cosines": component_to_joint,
        "gates": joint_gates,
        "decision": "PASS" if all(joint_gates.values()) else "FAIL",
    }
    return record


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty gradient audit")
    component_summary = {}
    for component in ROUTE_C_ACTIVE_COMPONENTS:
        norms = [row["components"][component]["gradient_norm"] for row in records]
        shares = [row["joint"]["weighted_component_norm_shares"][component] for row in records]
        joint_cosines = [
            row["joint"]["component_to_joint_cosines"][component]["cosine"]
            for row in records
        ]
        component_summary[component] = {
            "gradient_norm_min": min(norms),
            "gradient_norm_median": statistics.median(norms),
            "gradient_norm_max": max(norms),
            "weighted_norm_share_median": statistics.median(shares),
            "weighted_norm_share_max": max(shares),
            "joint_cosine_median": statistics.median(joint_cosines),
            "joint_cosine_min": min(joint_cosines),
            "opposed_to_joint_cases": sum(value < 0.0 for value in joint_cosines),
        }
    pairwise_summary = {}
    pair_names = records[0]["joint"]["pairwise_component_cosines"]
    for pair in pair_names:
        values = [
            row["joint"]["pairwise_component_cosines"][pair]["cosine"]
            for row in records
        ]
        pairwise_summary[pair] = {
            "cosine_min": min(values),
            "cosine_median": statistics.median(values),
            "cosine_max": max(values),
            "direction_conflict_cases": sum(value < 0.0 for value in values),
            "direction_conflict_fraction": sum(value < 0.0 for value in values)
            / len(values),
        }
    joint_norms = [row["joint"]["gradient_norm"] for row in records]
    maximum_shares = [row["joint"]["maximum_component_norm_share"] for row in records]
    return {
        "cases": len(records),
        "components": component_summary,
        "pairwise_component_cosines": pairwise_summary,
        "joint_gradient_norm_min": min(joint_norms),
        "joint_gradient_norm_median": statistics.median(joint_norms),
        "joint_gradient_norm_max": max(joint_norms),
        "maximum_component_norm_share": max(maximum_shares),
        "all_component_gradients_pass": all(
            row["components"][component]["decision"] == "PASS"
            for row in records
            for component in ROUTE_C_ACTIVE_COMPONENTS
        ),
        "all_joint_gradients_pass": all(
            row["joint"]["decision"] == "PASS" for row in records
        ),
    }


def validate_evidence(
    entries: list[list[str]],
) -> dict[str, dict[str, str]]:
    if len(entries) != len(REQUIRED_EVIDENCE):
        raise ValueError("source evidence count differs")
    evidence = {name: (Path(path), digest) for name, path, digest in entries}
    if set(evidence) != REQUIRED_EVIDENCE:
        raise ValueError(
            f"evidence keys differ: {sorted(evidence)} != {sorted(REQUIRED_EVIDENCE)}"
        )
    records = {}
    for name, (path, expected_hash) in evidence.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source evidence hash mismatch: {name}")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
        }
    return records


def summary_markdown(report: dict[str, Any]) -> str:
    holdout = report["holdout"]
    lines = [
        "# Route C four-active gradient interference audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Source commit: `{report['contract']['source']['head']}`",
        f"- Active components: `{', '.join(ROUTE_C_ACTIVE_COMPONENTS)}`",
        "- Objective: normalized bidirectional gap to the same-speaker clean pathological target",
        "- AVQI coefficient signs used for direction: `false`",
        "- Calibration / holdout cases: "
        f"`{report['selection']['cases_by_split']['surrogate_calibration']} / "
        f"{report['selection']['cases_by_split']['surrogate_holdout']}`",
        "- Holdout maximum weighted component norm share: "
        f"`{holdout['maximum_component_norm_share']:.6f}`",
        "- Holdout joint gradient norm median / max: "
        f"`{holdout['joint_gradient_norm_median']:.6f} / "
        f"{holdout['joint_gradient_norm_max']:.6f}`",
        "- Frozen final panel opened: `false`",
        "- Generator optimizer steps: `0`",
        "- Scientific promotion granted: `false`",
        "- Authoritative training decision: `NO_GO_AVQI_T2_TRAINING`",
        "",
        "| Component | Calibration median norm | Frozen weight | Holdout "
        "median norm | Holdout max weighted share | Holdout min cosine to joint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for component in ROUTE_C_ACTIVE_COMPONENTS:
        calibration_norm = report["calibration"]["median_component_gradient_norms"][component]
        weight = report["calibration"]["frozen_inverse_gradient_weights"][component]
        item = holdout["components"][component]
        lines.append(
            f"| {component} | {calibration_norm:.6f} | {weight:.8f} | "
            f"{item['gradient_norm_median']:.6f} | "
            f"{item['weighted_norm_share_max']:.6f} | "
            f"{item['joint_cosine_min']:.6f} |"
        )
    lines.extend(
        (
            "",
            "This is a code/scorer and dev-only gradient audit. It does not "
            "authorize a combined scientific panel or generator training.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA audit requested but no GPU is visible")
    source = verify_source(
        args.source_root.resolve(),
        args.source_commit,
        args.accepted_base_commit,
    )
    evidence = validate_evidence(args.evidence)
    checkpoint_paths = {
        "cpps": args.cpps_checkpoint,
        "hnr": args.hnr_checkpoint,
        "shimmer_percent": args.shimmer_checkpoint,
        "tilt": args.tilt_checkpoint,
    }
    checkpoint_hashes = {
        "cpps": args.cpps_checkpoint_sha256,
        "hnr": args.hnr_checkpoint_sha256,
        "shimmer_percent": args.shimmer_checkpoint_sha256,
        "tilt": args.tilt_checkpoint_sha256,
    }
    bundle = load_route_c_four_active_scorer(
        checkpoint_paths,
        checkpoint_hashes,
    )
    device = torch.device(args.device)
    scorer = bundle.scorer.to(device).eval()
    if sum(parameter.numel() for parameter in scorer.parameters()) != 0:
        raise ValueError("composed Route C scorer unexpectedly has parameters")
    cases, label_mean, label_scale, selection = load_label_bank(
        args.label_bank,
        args.label_bank_sha256,
        args.selection_salt,
    )
    if not torch.equal(bundle.scorer.target_mean.cpu(), label_mean):
        raise ValueError("checkpoint and live label-bank target means differ")
    if not torch.equal(bundle.scorer.target_scale.cpu(), label_scale):
        raise ValueError("checkpoint and live label-bank target scales differ")

    extracted = []
    for index, case in enumerate(cases, start=1):
        print(
            f"gradient_case={index}/{len(cases)} split={case.split} "
            f"view={case.view} group={case.sample_group}",
            flush=True,
        )
        extracted.append(extract_case_gradients(scorer, case, device))
    calibration_records = [
        row for row in extracted if row["split"] == "surrogate_calibration"
    ]
    median_norms, weights = frozen_inverse_gradient_weights(calibration_records)
    finalized = [finalize_case(row, weights) for row in extracted]
    calibration_rows = [
        row for row in finalized if row["split"] == "surrogate_calibration"
    ]
    holdout_rows = [
        row for row in finalized if row["split"] == "surrogate_holdout"
    ]
    calibration_summary = aggregate_records(calibration_rows)
    holdout_summary = aggregate_records(holdout_rows)
    weighted_calibration_medians = {
        component: median_norms[component] * weights[component]
        for component in ROUTE_C_ACTIVE_COMPONENTS
    }
    calibration_balance_ratio = max(weighted_calibration_medians.values()) / min(
        weighted_calibration_medians.values()
    )
    gates = {
        "six_slot_order_matches_avqi_v0301": tuple(
            slot["name"] for slot in route_c_registry_records()
        )
        == AVQI_COMPONENT_NAMES,
        "four_active_components_exact": tuple(
            slot["name"]
            for slot in route_c_registry_records()
            if slot["active_in_four_component_scorer"]
        )
        == ROUTE_C_ACTIVE_COMPONENTS,
        "calibration_holdout_speaker_disjoint": selection["speaker_overlap"] == 0,
        "dev_only_no_final_panel": selection["final_panel_opened"] is False,
        "calibration_component_gradients_pass": calibration_summary[
            "all_component_gradients_pass"
        ],
        "calibration_joint_gradients_pass": calibration_summary[
            "all_joint_gradients_pass"
        ],
        "holdout_component_gradients_pass": holdout_summary[
            "all_component_gradients_pass"
        ],
        "holdout_joint_gradients_pass": holdout_summary[
            "all_joint_gradients_pass"
        ],
        "calibration_weighted_median_norm_ratio_le_1_000001": (
            calibration_balance_ratio <= 1.000001
        ),
        "holdout_no_component_norm_share_above_0_80": (
            holdout_summary["maximum_component_norm_share"]
            <= MAX_WEIGHTED_COMPONENT_NORM_SHARE
        ),
        "zero_scorer_parameters": sum(
            parameter.numel() for parameter in scorer.parameters()
        )
        == 0,
        "generator_optimizer_steps_zero": True,
    }
    decision = (
        "PASS_ROUTE_C_FOUR_ACTIVE_CODE_GRADIENT_AUDIT"
        if all(gates.values())
        else "NO_GO_ROUTE_C_FOUR_ACTIVE_GRADIENT_INTERFERENCE"
    )
    report = {
        "schema_version": "avqi_route_c_multicomponent_gradient_audit_v1",
        "decision": decision,
        "contract": {
            "source": source,
            "architecture": ROUTE_C_FOUR_ACTIVE_ARCHITECTURE,
            "component_order": list(AVQI_COMPONENT_NAMES),
            "active_components": list(ROUTE_C_ACTIVE_COMPONENTS),
            "component_registry": route_c_registry_records(),
            "avqi_v0301": {
                "intercept": AVQI_V0301_INTERCEPT,
                "outer_scale": AVQI_V0301_SCALE,
                "coefficients": list(AVQI_V0301_COEFFICIENTS),
                "expanded_coefficients": list(
                    AVQI_V0301_EXPANDED_COEFFICIENTS
                ),
            },
            "loss_target": (
                "normalized bidirectional gap to same-speaker clean "
                "pathological CS/SV target"
            ),
            "avqi_scalar_coefficient_used_for_direction": False,
            "calibration_only_weight_selection": True,
            "weight_rule": "minimum calibration median norm / component median norm",
            "metric_branch_34hz_highpass_only": True,
            "final_output_highpass_applied": False,
            "waveform_mutation_performed": False,
            "full_band_pathology_guards_required_before_training": True,
        },
        "source_checkpoints": bundle.source_metadata,
        "source_evidence": evidence,
        "label_bank": {
            "path": str(args.label_bank.resolve()),
            "sha256": args.label_bank_sha256,
        },
        "selection": selection,
        "calibration": {
            **calibration_summary,
            "median_component_gradient_norms": median_norms,
            "frozen_inverse_gradient_weights": weights,
            "weighted_median_gradient_norms": weighted_calibration_medians,
            "weighted_median_norm_ratio": calibration_balance_ratio,
        },
        "holdout": holdout_summary,
        "case_results": finalized,
        "gates": gates,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "scientific_promotion_granted": False,
        "combined_final_panel_opened": False,
        "generator_loaded": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "gradient_interference_report.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_json(report_path, report)
    summary_path.write_text(summary_markdown(report), encoding="utf-8")
    receipt = {
        "decision": decision,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "accepted_base_commit": source["accepted_base_commit"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "active_components": list(ROUTE_C_ACTIVE_COMPONENTS),
        "inactive_slots": ["shimmer_db", "slope"],
        "calibration_cases": selection["cases_by_split"]["surrogate_calibration"],
        "holdout_cases": selection["cases_by_split"]["surrogate_holdout"],
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
        },
        "scientific_promotion_granted": False,
        "combined_final_panel_opened": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
