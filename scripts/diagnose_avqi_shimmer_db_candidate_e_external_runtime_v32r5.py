#!/usr/bin/env python3
"""Profile frozen Candidate-E GPU construction without opening exact outcomes.

The diagnostic is result-blind: exact Praat is invoked only to recover the
sealed base waveform pulse topology. Candidate waveforms remain in memory and
are never passed to the exact scorer. The first measured repeat is retained as
the cold observation; later repeats expose steady-state implementation cost.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from scripts.avqi_shimmer_db_candidate_e_proxy_v27 import (
    candidate_e_proxy,
    normalized_gradient_step,
    project_cycle_gain_gradient_fixed_order,
)
from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    topology_sha256,
)
from scripts.diagnose_avqi_shimmer_db_candidate_e_direction_v27 import (
    CANDIDATE_E_VARIANTS,
    VARIANT_E_PROJECTED,
    VARIANT_E_RAW,
    synchronize,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    SHIMMER_DB_INDEX,
    load_predictor,
    metric_source_indices_from_topology,
    read_waveform,
)
from scripts.evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r3 import (
    synthetic_runtime_warmup,
)
from scripts.evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r5 import (
    PREEXACT_NO_GO,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    build_zero_crossing_cycle_plan_vectorized,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import ALPHA_LADDER


SCHEMA_VERSION = (
    "avqi-route-c-shimmer-db-candidate-e-external-runtime-microdiagnostic-"
    "v32r5"
)
EXPECTED_CASE_COUNT = 12
TIMING_FIELDS = (
    "waveform_setup_ms",
    "proxy_forward_ms",
    "loss_backward_ms",
    "zero_crossing_plan_ms",
    "fixed_order_projection_ms",
    "candidate_batch_and_validation_ms",
    "candidate_gpu_to_cpu_transfer_ms",
    "instrumented_total_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-seal", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--v32r5-report", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {args.output}")
    if args.repeats < 3:
        raise ValueError("runtime diagnostic requires at least three repeats")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)

    panel = read_json(args.panel_seal)
    target = read_json(args.target_contract)
    report = read_json(args.v32r5_report)
    panel_rows = panel.get("rows")
    target_rows = target.get("rows")
    if not isinstance(panel_rows, list) or len(panel_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("external panel coverage drift")
    if not isinstance(target_rows, list) or len(target_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("external target coverage drift")
    if (
        panel.get("exact_contract", {}).get("candidate_exact_outcomes_opened")
        is not False
    ):
        raise ValueError("panel contract has opened candidate exact outcomes")
    if target.get("candidate_exact_outcomes_present") is not False:
        raise ValueError("target contract has candidate exact outcomes")
    if report.get("decision") != PREEXACT_NO_GO:
        raise ValueError("v32r5 terminal decision drift")
    if report.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("v32r5 has opened candidate exact outcomes")
    if report.get("exact_scoring_complete") is not False:
        raise ValueError("v32r5 exact scoring unexpectedly complete")
    target_by_case = {str(row["case_id"]): row for row in target_rows}
    case_ids = [str(row["case_id"]) for row in panel_rows]
    if len(set(case_ids)) != EXPECTED_CASE_COUNT or set(case_ids) != set(
        target_by_case
    ):
        raise ValueError("panel/target case binding drift")
    runtime_by_case = report.get("runtime_by_case")
    if not isinstance(runtime_by_case, dict) or set(runtime_by_case) != set(case_ids):
        raise ValueError("v32r5 runtime coverage drift")
    return panel_rows, target_by_case, runtime_by_case


def measured_component(
    device: torch.device,
    operation: Any,
) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    value = operation()
    synchronize(device)
    return value, 1000.0 * (time.perf_counter() - started)


def summarize_repeats(repeats: list[dict[str, float]]) -> dict[str, Any]:
    warm = repeats[1:]
    return {
        "cold_repeat": dict(repeats[0]),
        "warm_repeat_count": len(warm),
        "warm_median": {
            field: median(row[field] for row in warm) for field in TIMING_FIELDS
        },
        "warm_minimum": {
            field: min(row[field] for row in warm) for field in TIMING_FIELDS
        },
        "warm_maximum": {
            field: max(row[field] for row in warm) for field in TIMING_FIELDS
        },
    }


def profile_case(
    panel_row: dict[str, Any],
    target_row: dict[str, Any],
    v32r5_runtime: dict[str, Any],
    target_scale: float,
    worker: ExactShimmerTopologyWorker,
    device: torch.device,
    repeats: int,
) -> dict[str, Any]:
    case_id = str(panel_row["case_id"])
    base_float = read_waveform(Path(panel_row["base_path"]))
    base_values = np.asarray(base_float.numpy(), dtype=np.float32)
    topology_item = {
        "id": f"runtime_microdiagnostic_base:{case_id}",
        "case_id": case_id,
        "role": "current_output_topology",
        "path": str(Path(panel_row["base_path"]).resolve()),
        "view": str(panel_row["view"]),
        "score_components": False,
        "exact_metric_topology": True,
        "highpass_mode": NUMPY_HIGHPASS_MODE,
    }
    topology_rows, topology_wall_ms, topology_staging = (
        worker.refresh_current_waveforms(
            [topology_item],
            [base_values],
            NUMPY_HIGHPASS_MODE,
        )
    )
    topology = topology_rows[0]
    base_device = base_float.to(device)
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            topology,
            source_sample_count=base_float.numel(),
        ),
        dtype=torch.long,
        device=device,
    )
    pulses = torch.as_tensor(
        topology["pulse_positions_samples"],
        dtype=torch.float64,
        device=device,
    )
    target = float(target_row["exact_target_shimmer_db"])
    repeat_rows: list[dict[str, float]] = []
    plan_summary: dict[str, Any] | None = None
    fft_sample_count: int | None = None

    for _ in range(repeats):
        total_started = time.perf_counter()
        waveform, waveform_setup_ms = measured_component(
            device,
            lambda: base_device.detach().to(dtype=torch.float64).requires_grad_(True),
        )
        proxy, proxy_forward_ms = measured_component(
            device,
            lambda: candidate_e_proxy(
                waveform,
                pulses,
                source_indices,
                int(topology["metric_constant_prefix_samples"]),
            ),
        )
        raw, loss_backward_ms = measured_component(
            device,
            lambda: torch.autograd.grad(
                ((proxy.shimmer_db - target) / target_scale).square(),
                waveform,
            )[0],
        )

        synchronize(device)
        plan_started = time.perf_counter()
        plan = build_zero_crossing_cycle_plan_vectorized(base_values, topology)
        zero_crossing_plan_ms = 1000.0 * (time.perf_counter() - plan_started)
        projected, fixed_order_projection_ms = measured_component(
            device,
            lambda: project_cycle_gain_gradient_fixed_order(
                waveform,
                raw,
                plan,
            ),
        )
        projected_gradient, projection = projected
        if not projection["projected_gradient_valid"]:
            raise ValueError(f"Candidate-E projection invalid: {case_id}")

        def build_candidate_batch() -> torch.Tensor:
            directions = {
                VARIANT_E_PROJECTED: projected_gradient,
                VARIANT_E_RAW: raw,
            }
            if tuple(directions) != CANDIDATE_E_VARIANTS:
                raise ValueError("Candidate-E direction order drift")
            candidate_tensors = [
                normalized_gradient_step(waveform, direction, alpha)
                for direction in directions.values()
                for alpha in (0.0, *ALPHA_LADDER)
            ]
            candidate_batch = torch.stack(candidate_tensors)
            if not bool(torch.isfinite(candidate_batch).all().detach().cpu()):
                raise ValueError(f"non-finite Candidate-E batch: {case_id}")
            if float(candidate_batch.detach().abs().max().cpu()) >= 0.999:
                raise ValueError(f"Candidate-E batch clips: {case_id}")
            return candidate_batch

        candidate_batch, candidate_batch_ms = measured_component(
            device,
            build_candidate_batch,
        )
        candidate_values, transfer_ms = measured_component(
            device,
            lambda: candidate_batch.detach().cpu().numpy(),
        )
        if candidate_values.shape[0] != 2 * (1 + len(ALPHA_LADDER)):
            raise ValueError("Candidate-E candidate count drift")
        instrumented_total_ms = 1000.0 * (time.perf_counter() - total_started)
        repeat_rows.append(
            {
                "waveform_setup_ms": waveform_setup_ms,
                "proxy_forward_ms": proxy_forward_ms,
                "loss_backward_ms": loss_backward_ms,
                "zero_crossing_plan_ms": zero_crossing_plan_ms,
                "fixed_order_projection_ms": fixed_order_projection_ms,
                "candidate_batch_and_validation_ms": candidate_batch_ms,
                "candidate_gpu_to_cpu_transfer_ms": transfer_ms,
                "instrumented_total_ms": instrumented_total_ms,
            }
        )
        plan_summary = dict(plan["summary"])
        fft_sample_count = int(proxy.fft_sample_count)

    if plan_summary is None or fft_sample_count is None:
        raise RuntimeError("runtime diagnostic produced no repeats")
    return {
        "case_id": case_id,
        "speaker_id": str(panel_row["speaker_id"]),
        "view": str(panel_row["view"]),
        "condition": str(panel_row["condition"]),
        "sample_count": int(base_values.size),
        "duration_seconds": float(base_values.size / 16_000),
        "fft_sample_count": fft_sample_count,
        "pulse_count": int(topology["pulse_count"]),
        "complete_cycle_count": int(plan_summary["complete_cycle_count"]),
        "supported_sample_count": int(plan_summary["supported_sample_count"]),
        "base_topology_sha256": topology_sha256(topology),
        "base_topology_refresh_wall_ms": topology_wall_ms,
        "base_topology_staging_ms": sum(
            float(row["staging_ms"]) for row in topology_staging
        ),
        "v32r5_candidate_gradient_projection_and_batch_ms": float(
            v32r5_runtime["candidate_gradient_projection_and_batch_ms"]
        ),
        "v32r5_total_metric_step_runtime_ms": float(
            v32r5_runtime["total_metric_step_runtime_ms"]
        ),
        "repeats": repeat_rows,
        "summary": summarize_repeats(repeat_rows),
    }


def main() -> None:
    args = parse_args()
    panel_rows, target_by_case, runtime_by_case = validate_inputs(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime diagnostic requires an allocated CUDA device")
    _, synthetic_warmup = synthetic_runtime_warmup(device)
    _, _, _, target_scale_tensor = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale = float(target_scale_tensor[SHIMMER_DB_INDEX].detach().cpu())
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("invalid frozen Shimmer dB target scale")

    rows: list[dict[str, Any]] = []
    with ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    ) as worker:
        worker_warmup, worker_warmup_ms = worker.warmup()
        for index, panel_row in enumerate(panel_rows, start=1):
            case_id = str(panel_row["case_id"])
            rows.append(
                profile_case(
                    panel_row,
                    target_by_case[case_id],
                    runtime_by_case[case_id],
                    target_scale,
                    worker,
                    device,
                    args.repeats,
                )
            )
            print(
                f"candidate_e_external_runtime_diagnostic={index}/"
                f"{EXPECTED_CASE_COUNT}",
                flush=True,
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": "result_blind_preexact_runtime_microdiagnostic",
        "slurm_job_id": args.slurm_job_id,
        "case_count": len(rows),
        "repeat_count": args.repeats,
        "candidate_e_math_changed": False,
        "candidate_grid_changed": False,
        "candidate_waveforms_persisted": False,
        "candidate_exact_outcomes_present": False,
        "candidate_exact_outcomes_opened": False,
        "candidate_exact_outcomes_used": False,
        "base_exact_topology_only": True,
        "formal_generator_training_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "synthetic_candidate_e_warmup": synthetic_warmup,
        "exact_topology_worker_warmup": worker_warmup,
        "exact_topology_worker_warmup_ms": worker_warmup_ms,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"runtime_diagnostic={args.output}", flush=True)


if __name__ == "__main__":
    main()
