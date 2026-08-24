#!/usr/bin/env python3
"""Screen a topology-preserving Shimmer-dB backward on opened dev panels.

The exact-forward scalar and detached Praat pulse topology are unchanged.  The
raw waveform gradient is orthogonally projected onto one multiplicative basis
per exact pulse cycle before the frozen alpha=0.001 direct waveform step.  This
is a dev-only mechanism test; it cannot promote a component or authorize
generator training.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    CANDIDATE_NAME,
    FIXED_ALPHA,
    MATERIAL_GAP_THRESHOLD,
    SAMPLE_RATE,
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    component_fields,
    exact_components,
    load_predictor,
    metric_source_indices_from_topology,
    normalized_gradient_step,
    pulse_positions_sha256,
    read_waveform,
    run_exact,
    sha256_file,
    topology_stability,
    waveform_safety,
    write_csv,
    write_json,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import aggregate_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTED_CANDIDATE_NAME = (
    "praat_current_output_topology_cycle_projected_backward_db_alpha_0p001"
)
EXPECTED_CASE_COUNT = 12
EXPECTED_SLICE_COUNTS = {
    "view": {"cs": 6, "sv": 6},
    "sample_group": {
        "pathological_mild": 6,
        "pathological_severe": 6,
    },
    "condition": {"rir_only": 4, "snr20": 4, "snr10": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--fresh-results", type=Path, required=True)
    parser.add_argument("--fresh-results-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def repository_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def coerce_csv_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {key: coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")
    return observed


def validate_panel(
    panel: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    panel_rows = [dict(row) for row in panel.get("rows", [])]
    if len(panel_rows) != EXPECTED_CASE_COUNT or len(
        {row["case_id"] for row in panel_rows}
    ) != EXPECTED_CASE_COUNT:
        raise ValueError("cycle-projected screen requires twelve unique cases")
    if len({row["speaker_id"] for row in panel_rows}) != 6:
        raise ValueError("cycle-projected screen requires six speakers")
    for field, expected in EXPECTED_SLICE_COUNTS.items():
        if Counter(row[field] for row in panel_rows) != Counter(expected):
            raise ValueError(f"opened panel slice contract drift: {field}")
    if panel.get("panel_status") != "sealed_new_speaker_panel_before_exact_outcomes":
        raise ValueError("input was not sealed before exact outcomes")
    if panel.get("candidate_c", {}).get("fixed_alpha") != FIXED_ALPHA:
        raise ValueError("opened panel alpha drift")
    if panel.get("generator", {}).get("optimizer_steps") != 0:
        raise ValueError("opened panel contains generator optimizer steps")
    by_case = {row["case_id"]: row for row in result_rows}
    if set(by_case) != {row["case_id"] for row in panel_rows}:
        raise ValueError("opened panel/result coverage drift")
    for row in panel_rows:
        validate_hash(Path(row["base_path"]), row["base_sha256"], "base waveform")
        result = by_case[row["case_id"]]
        if result["candidate"] != CANDIDATE_NAME:
            raise ValueError("opened result candidate drift")
        if float(result["fixed_alpha"]) != FIXED_ALPHA:
            raise ValueError("opened result alpha drift")
    return panel_rows, by_case


def cycle_multiplicative_gradient_projection(
    waveform: torch.Tensor,
    gradient: torch.Tensor,
    topology: dict[str, Any],
) -> torch.Tensor:
    """Project a gradient onto detached per-pulse-cycle gain directions."""
    if waveform.ndim != 1 or gradient.shape != waveform.shape:
        raise ValueError("cycle projection expects matching 1-D tensors")
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            topology,
            source_sample_count=waveform.numel(),
        ),
        dtype=torch.long,
        device=waveform.device,
    )
    pulses = waveform.new_tensor(topology["pulse_positions_samples"]).detach()
    if pulses.numel() < 3:
        raise ValueError("cycle projection requires at least three pulses")
    metric_positions = torch.arange(
        source_indices.numel(),
        dtype=waveform.dtype,
        device=waveform.device,
    ) + int(topology["metric_constant_prefix_samples"])
    boundaries = 0.5 * (pulses[:-1] + pulses[1:])
    left_edge = pulses[0] - 0.5 * (pulses[1] - pulses[0])
    right_edge = pulses[-1] + 0.5 * (pulses[-1] - pulses[-2])
    supported = (metric_positions >= left_edge) & (metric_positions <= right_edge)
    supported_indices = source_indices[supported]
    if supported_indices.numel() == 0:
        raise ValueError("cycle projection has no mapped waveform support")
    cells = torch.bucketize(metric_positions[supported], boundaries)
    cycle_count = int(pulses.numel())
    reference = waveform.detach().index_select(0, supported_indices)
    raw = gradient.detach().index_select(0, supported_indices)
    numerators = waveform.new_zeros(cycle_count)
    denominators = waveform.new_zeros(cycle_count)
    numerators.scatter_add_(0, cells, raw * reference)
    denominators.scatter_add_(0, cells, reference.square())
    coefficients = torch.where(
        denominators > torch.finfo(waveform.dtype).tiny,
        numerators / denominators.clamp_min(torch.finfo(waveform.dtype).tiny),
        torch.zeros_like(numerators),
    )
    projected = torch.zeros_like(gradient)
    projected.index_copy_(
        0,
        supported_indices,
        coefficients.index_select(0, cells) * reference,
    )
    if not torch.isfinite(projected).all():
        raise ValueError("cycle-projected gradient is non-finite")
    return projected


def topology_items(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"base:{row['case_id']}",
            "case_id": row["case_id"],
            "role": "current_s3_500_output_topology",
            "path": row["base_path"],
            "view": row["view"],
            "score_components": False,
            "exact_metric_topology": True,
        }
        for row in panel_rows
    ]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def exact_vector(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray(
        [float(row[f"exact_{prefix}_{name}"]) for name in AVQI_COMPONENT_NAMES],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head() != args.source_commit:
        raise ValueError("diagnostic source commit differs from repository HEAD")
    source_hashes = {
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "opened panel contract",
        ),
        "fresh_results": validate_hash(
            args.fresh_results,
            args.fresh_results_sha256,
            "opened fresh-panel results",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen Shimmer checkpoint",
        ),
    }
    observed_avqi_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_avqi_tree_hash
    panel = load_json(args.panel_contract)
    input_results = read_csv(args.fresh_results)
    panel_rows, input_by_case = validate_panel(panel, input_results)

    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    exact_topology = run_exact(
        topology_items(panel_rows),
        args.exact_python,
        args.avqi_code_root,
    )
    if len(exact_topology["rows"]) != EXPECTED_CASE_COUNT:
        raise ValueError("base topology coverage drift")
    topology_by_case = {
        row["case_id"]: row for row in exact_topology["rows"]
    }

    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    candidate_records: dict[str, dict[str, Any]] = {}
    for index, panel_row in enumerate(panel_rows, start=1):
        case_id = panel_row["case_id"]
        input_row = input_by_case[case_id]
        topology = topology_by_case[case_id]
        waveform = read_waveform(Path(panel_row["base_path"])).to(device)
        waveform = waveform.requires_grad_(True)
        source_indices = torch.as_tensor(
            metric_source_indices_from_topology(
                topology,
                source_sample_count=waveform.numel(),
            ),
            dtype=torch.long,
            device=device,
        )
        pulses = waveform.new_tensor(topology["pulse_positions_samples"])
        synchronize(device)
        started = time.perf_counter()
        proxy_before = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=int(
                topology["metric_constant_prefix_samples"]
            ),
        )[1]
        target = float(input_row["exact_target_shimmer_db"])
        scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
        loss = ((proxy_before - target) / scale).square()
        raw_gradient = torch.autograd.grad(loss, waveform)[0]
        projected_gradient = cycle_multiplicative_gradient_projection(
            waveform,
            raw_gradient,
            topology,
        )
        candidate = normalized_gradient_step(
            waveform,
            projected_gradient,
            FIXED_ALPHA,
        )
        synchronize(device)
        step_runtime_ms = 1000.0 * (time.perf_counter() - started)
        if float(projected_gradient.norm()) <= 0.0:
            raise ValueError(f"zero cycle-projected gradient: {case_id}")
        if not torch.isfinite(candidate).all() or float(candidate.abs().max()) >= 0.999:
            raise ValueError(f"invalid cycle-projected candidate: {case_id}")
        output_path = waveform_root / f"{case_id}__cycle_projected.wav"
        sf.write(
            output_path,
            candidate.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        candidate_records[case_id] = {
            "path": output_path,
            "proxy_before": float(proxy_before.detach()),
            "loss": float(loss.detach()),
            "raw_gradient_l2_norm": float(raw_gradient.norm()),
            "projected_gradient_l2_norm": float(projected_gradient.norm()),
            "projected_gradient_rms": float(
                projected_gradient.square().mean().sqrt()
            ),
            "step_runtime_ms": step_runtime_ms,
            "base_topology_sha256": pulse_positions_sha256(
                topology["pulse_positions_samples"]
            ),
        }
        print(f"cycle_projected_step={index}/{len(panel_rows)}", flush=True)

    exact_items = [
        {
            "id": f"projected:{row['case_id']}",
            "case_id": row["case_id"],
            "role": "cycle_projected_candidate",
            "path": str(candidate_records[row["case_id"]]["path"].resolve()),
            "view": row["view"],
            "score_components": True,
            "exact_metric_topology": True,
        }
        for row in panel_rows
    ]
    exact_after = run_exact(
        exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    after_by_case = {row["case_id"]: row for row in exact_after["rows"]}

    rows: list[dict[str, Any]] = []
    for panel_row in panel_rows:
        case_id = panel_row["case_id"]
        input_row = input_by_case[case_id]
        topology = topology_by_case[case_id]
        after = after_by_case[case_id]
        if after.get("scoring_status") != "ok" or int(
            after.get("pulse_count", 0)
        ) < 3:
            raise ValueError(f"projected exact scoring failed: {case_id}")
        record = candidate_records[case_id]
        target_components = exact_vector(input_row, "target")
        base_components = exact_vector(input_row, "before")
        after_components = exact_components(after)
        base_waveform = read_waveform(Path(panel_row["base_path"])).numpy()
        candidate_waveform = read_waveform(record["path"]).numpy()
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": PROJECTED_CANDIDATE_NAME,
            "optimized_component": "shimmer_db",
            "fixed_alpha": FIXED_ALPHA,
            "candidate_path": str(record["path"].resolve()),
            "candidate_sha256": sha256_file(record["path"]),
            "proxy_before": record["proxy_before"],
            "proxy_target": float(input_row["exact_target_shimmer_db"]),
            "proxy_loss": record["loss"],
            "gradient_l2_norm": record["projected_gradient_l2_norm"],
            "raw_gradient_l2_norm": record["raw_gradient_l2_norm"],
            "gradient_rms": record["projected_gradient_rms"],
            "gradient_finite": True,
            "pulse_refresh_runtime_ms": float(
                input_row["pulse_refresh_runtime_ms"]
            ),
            "torch_step_runtime_ms": record["step_runtime_ms"],
            "total_metric_step_overhead_ms": (
                float(input_row["pulse_refresh_runtime_ms"])
                + record["step_runtime_ms"]
            ),
            "base_output_exact_metric_pulse_count": int(
                topology["pulse_count"]
            ),
            "candidate_exact_metric_pulse_count": int(after["pulse_count"]),
            "forward_normalized_abs_error_shimmer_db": float(
                input_row["forward_normalized_abs_error_shimmer_db"]
            ),
            "clean_target_topology_drives_output": False,
        }
        component_fields(
            row,
            target_components,
            base_components,
            after_components,
            target_scale_np,
        )
        row["material_shimmer_db_gap"] = (
            row["exact_absolute_gap_before_shimmer_db"]
            / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
            > MATERIAL_GAP_THRESHOLD
        )
        row.update(topology_stability(topology, after))
        row.update(waveform_safety(base_waveform, candidate_waveform))
        rows.append(row)

    write_csv(args.output_dir / "cycle_projected_results.csv", rows)
    original_summary = aggregate_candidate(CANDIDATE_NAME, input_results)
    projected_summary = aggregate_candidate(PROJECTED_CANDIDATE_NAME, rows)
    backward_gates = {
        name: passed
        for name, passed in projected_summary["gates"].items()
        if name != "pulse_refresh_runtime"
    }
    decision = (
        "PASS_SHIMMER_DB_CYCLE_PROJECTED_BACKWARD_DEV_SCREEN"
        if all(backward_gates.values())
        else "NO_GO_SHIMMER_DB_CYCLE_PROJECTED_BACKWARD_DEV_SCREEN"
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-db-cycle-projected-backward-v1",
        "decision": decision,
        "dev_only": True,
        "opened_panel_reused_for_mechanism_diagnosis": True,
        "promotion_authorized": False,
        "new_sealed_panel_authorized": False,
        "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
        "forward_scalar_changed": False,
        "detached_topology_changed": False,
        "backward_projection": "orthogonal_per_exact_pulse_cycle_multiplicative",
        "backward_projection_has_tunable_parameters": False,
        "fixed_alpha": FIXED_ALPHA,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_sha256": source_hashes,
        "input_panel": {
            "schema_version": panel.get("schema_version"),
            "source_commit": panel.get("source_commit"),
            "slurm_job_id": panel.get("slurm_job_id"),
            "speakers": sorted({row["speaker_id"] for row in panel_rows}),
        },
        "original_candidate_summary": original_summary,
        "cycle_projected_candidate_summary": projected_summary,
        "backward_screen_gates": backward_gates,
        "runtime_gate_excluded_from_backward_screen": True,
        "runtime_gate_changed": False,
        "comparison": {
            "original_topology_stability_fraction": original_summary[
                "topology_stability_fraction"
            ],
            "projected_topology_stability_fraction": projected_summary[
                "topology_stability_fraction"
            ],
            "original_exact_improvement_fraction": original_summary[
                "exact_db_improvement_fraction"
            ],
            "projected_exact_improvement_fraction": projected_summary[
                "exact_db_improvement_fraction"
            ],
            "original_median_normalized_reduction": original_summary[
                "median_exact_db_normalized_gap_reduction"
            ],
            "projected_median_normalized_reduction": projected_summary[
                "median_exact_db_normalized_gap_reduction"
            ],
        },
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifacts": {"results": "cycle_projected_results.csv"},
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-cycle-projected-receipt-v1",
        "decision": decision,
        "dev_only": True,
        "promotion_authorized": False,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "case_count": len(rows),
        "speaker_count": len({row["speaker_id"] for row in rows}),
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            "cycle_projected_results.csv": sha256_file(
                args.output_dir / "cycle_projected_results.csv"
            ),
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
