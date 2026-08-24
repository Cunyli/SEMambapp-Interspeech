#!/usr/bin/env python3
"""Dev-screen a topology-certified Shimmer-dB trust region.

The candidate keeps the sample-wise Candidate-C gradient and treats 0.001 as
the maximum step.  A fixed four-level half-step ladder is topology-certified
with exact Praat in parallel.  Selection may use only topology, frozen-forward
proxy, finite/safety, and PCM24 nonzero-step checks.  Candidate exact component
outcomes are opened only after the selector seal is written.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    require_exact_topology_equal,
    topology_sha256,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
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
from scripts.evaluate_avqi_shimmer_db_cycle_projected_backward import (
    exact_vector,
    read_csv,
    repository_head,
    validate_hash,
    validate_panel,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RUNTIME_MAX_MS,
    FIXED_ALPHA,
    GRADIENT_NORM_RANGE,
    IMPROVEMENT_FRACTION_GATE,
    MAXIMUM_CLIP_FRACTION,
    MEDIAN_REDUCTION_GATE,
    MINIMUM_COSINE,
    NONSELECTED_MEDIAN_INCREASE_GATE,
    RESIDUAL_CEILING_DB,
    aggregate_candidate,
)
from scripts.evaluate_direct_avqi_waveform_optimization import (
    aggregate_denoising,
    aggregate_pathology_guardrails,
    full_band_pathology_guardrails,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRUST_REGION_CANDIDATE_NAME = (
    "praat_current_output_topology_certified_backtracking_db_v16"
)
ALPHA_LADDER = (FIXED_ALPHA, FIXED_ALPHA / 2, FIXED_ALPHA / 4, FIXED_ALPHA / 8)
MAX_BACKTRACKS = len(ALPHA_LADDER) - 1
PROXY_GAP_TOLERANCE = 1e-7
PCM24_MIN_CHANGED_SAMPLES = round(0.001 * SAMPLE_RATE)
PCM24_MIN_RESIDUAL_RMS_LSB = 1.0
WORKER_COUNT = len(ALPHA_LADDER)
PROTOTYPE_CASE_IDS = (
    "sealed_final__FD23__cs__rir_only",
    "sealed_final__PD_37__cs__snr20",
    "sealed_final__FD23__sv__snr20",
    "sealed_final__PD_37__sv__snr10",
)
EXPECTED_CASE_COUNT = len(PROTOTYPE_CASE_IDS)
SELECTOR_KEYS = frozenset(
    {
        "alpha",
        "backtrack_index",
        "topology_stability_pass",
        "finite_safety_pass",
        "proxy_nonregression_pass",
        "pcm24_effective_step_pass",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-label", choices=("v14", "v15"), required=True)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--fresh-results", type=Path, required=True)
    parser.add_argument("--fresh-results-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--runtime-worker-script", type=Path, required=True)
    parser.add_argument("--runtime-worker-script-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def selector_view(attempt: dict[str, Any]) -> dict[str, Any]:
    return {key: attempt[key] for key in SELECTOR_KEYS}


def select_topology_certified_step(
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the largest fixed alpha passing only preregistered certificates."""
    if len(attempts) != len(ALPHA_LADDER):
        raise ValueError("selector requires the complete fixed alpha ladder")
    for index, (attempt, expected_alpha) in enumerate(
        zip(attempts, ALPHA_LADDER, strict=True)
    ):
        if set(attempt) != SELECTOR_KEYS:
            extra = sorted(set(attempt) - SELECTOR_KEYS)
            missing = sorted(SELECTOR_KEYS - set(attempt))
            raise ValueError(
                f"selector input contract drift: extra={extra}, missing={missing}"
            )
        if not math.isclose(
            float(attempt["alpha"]), expected_alpha, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("selector alpha order drift")
        if int(attempt["backtrack_index"]) != index:
            raise ValueError("selector backtrack index drift")
        if all(
            bool(attempt[key])
            for key in (
                "topology_stability_pass",
                "finite_safety_pass",
                "proxy_nonregression_pass",
                "pcm24_effective_step_pass",
            )
        ):
            return dict(attempt)
    return None


def pcm24_effective_step(
    base_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    base, base_rate = sf.read(base_path, dtype="int32", always_2d=False)
    candidate, candidate_rate = sf.read(
        candidate_path,
        dtype="int32",
        always_2d=False,
    )
    if (
        base_rate != SAMPLE_RATE
        or candidate_rate != SAMPLE_RATE
        or base.ndim != 1
        or candidate.ndim != 1
        or base.shape != candidate.shape
    ):
        raise ValueError("PCM24 effective-step comparison shape drift")
    difference_lsb = (
        candidate.astype(np.int64) - base.astype(np.int64)
    ).astype(np.float64) / 256.0
    changed = int(np.count_nonzero(difference_lsb))
    residual_rms_lsb = float(np.sqrt(np.mean(np.square(difference_lsb))))
    sha_differs = sha256_file(base_path) != sha256_file(candidate_path)
    passed = (
        sha_differs
        and changed >= PCM24_MIN_CHANGED_SAMPLES
        and residual_rms_lsb >= PCM24_MIN_RESIDUAL_RMS_LSB
    )
    return {
        "pcm24_sha_differs_from_base": sha_differs,
        "pcm24_changed_samples": changed,
        "pcm24_changed_fraction": changed / max(int(base.size), 1),
        "pcm24_residual_rms_lsb": residual_rms_lsb,
        "pcm24_effective_step_pass": passed,
    }


def finite_safety(
    base: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    metrics = waveform_safety(base, candidate)
    finite = bool(np.isfinite(candidate).all())
    bounded = bool(float(np.max(np.abs(candidate), initial=0.0)) < 0.999)
    passed = (
        finite
        and bounded
        and metrics["residual_rms_db"] <= RESIDUAL_CEILING_DB
        and metrics["cosine_similarity"] >= MINIMUM_COSINE
        and metrics["clip_fraction"] <= MAXIMUM_CLIP_FRACTION
    )
    return {
        **metrics,
        "waveform_finite": finite,
        "waveform_bound_pass": bounded,
        "finite_safety_pass": passed,
    }


def synthetic_torch_warmup(
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample_count = SAMPLE_RATE
    timeline = torch.arange(sample_count, device=device, dtype=torch.float32)
    waveform = (0.08 * torch.sin(2.0 * math.pi * 125.0 * timeline / SAMPLE_RATE))
    waveform = waveform.requires_grad_(True)
    pulses = torch.arange(320, sample_count - 320, 128, device=device).float()
    source_indices = torch.arange(sample_count, device=device, dtype=torch.long)
    proxy = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=0,
    )[1]
    scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    loss = (proxy / scale).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    candidate = normalized_gradient_step(waveform, gradient, FIXED_ALPHA)
    synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "synthetic_only": True,
        "panel_waveform_used": False,
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "candidate_finite": bool(torch.isfinite(candidate).all()),
        "runtime_ms": elapsed_ms,
    }


def base_topology_item(panel_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"base_topology:{panel_row['case_id']}",
        "case_id": panel_row["case_id"],
        "role": "current_s3_500_output_topology",
        "path": panel_row["base_path"],
        "view": panel_row["view"],
        "score_components": False,
        "exact_metric_topology": True,
    }


def candidate_topology_item(
    case_id: str,
    view: str,
    path: Path,
    backtrack_index: int,
) -> dict[str, Any]:
    return {
        "id": f"candidate_topology:{case_id}:{backtrack_index}",
        "case_id": case_id,
        "role": "current_output_topology",
        "path": str(path.resolve()),
        "view": view,
        "score_components": False,
        "exact_metric_topology": True,
        "highpass_mode": NUMPY_HIGHPASS_MODE,
    }


def refresh_one_candidate(
    worker: ExactShimmerTopologyWorker,
    item: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    rows, request_wall_ms = worker.refresh([item])
    return dict(rows[0]), request_wall_ms


def validate_dev_files(panel_rows: list[dict[str, Any]]) -> None:
    for row in panel_rows:
        validate_hash(Path(row["target_path"]), row["target_sha256"], "target")
        validate_hash(
            Path(row["degraded_path"]),
            row["degraded_sha256"],
            "degraded",
        )


def evaluate_case(
    panel_row: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
) -> dict[str, Any]:
    case_id = panel_row["case_id"]
    total_started = time.perf_counter()
    base_path = Path(panel_row["base_path"])
    base_waveform = read_waveform(base_path)
    base_values = base_waveform.numpy()
    base_rows, base_request_wall_ms, base_staging = workers[
        0
    ].refresh_current_waveforms(
        [base_topology_item(panel_row)],
        [base_values],
        highpass_mode=NUMPY_HIGHPASS_MODE,
    )
    base_topology = dict(base_rows[0])
    base_staging_ms = float(base_staging[0]["staging_ms"])
    base_refresh_ms = base_staging_ms + base_request_wall_ms
    base_topology["client_tmpfs_staging_ms"] = base_staging_ms
    base_topology["request_wall_ms"] = base_request_wall_ms
    base_topology["end_to_end_refresh_ms"] = base_refresh_ms

    waveform = base_waveform.to(device).requires_grad_(True)
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            base_topology,
            source_sample_count=waveform.numel(),
        ),
        dtype=torch.long,
        device=device,
    )
    pulses = waveform.new_tensor(base_topology["pulse_positions_samples"])
    synchronize(device)
    gradient_started = time.perf_counter()
    proxy_before = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            base_topology["metric_constant_prefix_samples"]
        ),
    )[1]
    target = float(target_shimmer_db)
    scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    scale_value = float(scale)
    loss = ((proxy_before - target) / scale).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    candidates = [
        normalized_gradient_step(waveform, gradient, alpha)
        for alpha in ALPHA_LADDER
    ]
    synchronize(device)
    gradient_runtime_ms = 1000.0 * (time.perf_counter() - gradient_started)
    if not torch.isfinite(gradient).all() or float(gradient.norm()) <= 0.0:
        raise ValueError(f"invalid trust-region gradient: {case_id}")

    attempt_records: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    stored_waveforms: list[np.ndarray] = []
    pcm24_write_ms: list[float] = []
    for backtrack_index, (alpha, candidate) in enumerate(
        zip(ALPHA_LADDER, candidates, strict=True)
    ):
        path = waveform_root / (
            f"{case_id}__trust_region_bt{backtrack_index}_alpha_{alpha:.7f}.wav"
        )
        write_started = time.perf_counter()
        sf.write(
            path,
            candidate.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        pcm24_write_ms.append(1000.0 * (time.perf_counter() - write_started))
        stored = read_waveform(path).numpy()
        stored_tensor = torch.from_numpy(stored).to(device)
        with torch.inference_mode():
            proxy_after = predictor.raw_shimmer_from_pulse_positions(
                stored_tensor,
                pulses,
                metric_source_indices=source_indices,
                metric_constant_prefix_samples=int(
                    base_topology["metric_constant_prefix_samples"]
                ),
            )[1]
        normalized_proxy_gap_before = abs(float(proxy_before.detach()) - target) / max(
            scale_value,
            1e-8,
        )
        normalized_proxy_gap_after = abs(float(proxy_after.detach()) - target) / max(
            scale_value,
            1e-8,
        )
        attempt_records.append(
            {
                "case_id": case_id,
                "alpha": alpha,
                "backtrack_index": backtrack_index,
                "candidate_path": str(path.resolve()),
                "candidate_sha256": sha256_file(path),
                "proxy_before": float(proxy_before.detach()),
                "proxy_after_frozen_topology": float(proxy_after.detach()),
                "proxy_target": target,
                "normalized_proxy_gap_before": normalized_proxy_gap_before,
                "normalized_proxy_gap_after": normalized_proxy_gap_after,
                "proxy_nonregression_pass": (
                    normalized_proxy_gap_after
                    <= normalized_proxy_gap_before + PROXY_GAP_TOLERANCE
                ),
                **finite_safety(base_values, stored),
                **pcm24_effective_step(base_path, path),
            }
        )
        candidate_paths.append(path)
        stored_waveforms.append(stored)
    synchronize(device)

    candidate_refresh_started = time.perf_counter()
    futures = [
        executor.submit(
            refresh_one_candidate,
            worker,
            candidate_topology_item(
                case_id,
                panel_row["view"],
                path,
                backtrack_index,
            ),
        )
        for backtrack_index, (worker, path) in enumerate(
            zip(workers, candidate_paths, strict=True)
        )
    ]
    candidate_refresh_results = [future.result() for future in futures]
    candidate_refresh_wall_ms = 1000.0 * (
        time.perf_counter() - candidate_refresh_started
    )
    candidate_topologies: list[dict[str, Any]] = []
    for attempt, (topology, request_wall_ms) in zip(
        attempt_records,
        candidate_refresh_results,
        strict=True,
    ):
        stability = topology_stability(base_topology, topology)
        attempt.update(stability)
        attempt["candidate_topology_sha256"] = topology_sha256(topology)
        attempt["candidate_pulse_positions_sha256"] = pulse_positions_sha256(
            topology["pulse_positions_samples"]
        )
        attempt["candidate_pulse_count"] = int(topology["pulse_count"])
        attempt["candidate_refresh_request_wall_ms"] = request_wall_ms
        attempt["candidate_refresh_internal_ms"] = float(
            topology["pulse_runtime_ms"]
        )
        candidate_topologies.append(topology)

    selector_inputs = [selector_view(attempt) for attempt in attempt_records]
    selected_input = select_topology_certified_step(selector_inputs)
    selected_index = (
        int(selected_input["backtrack_index"])
        if selected_input is not None
        else None
    )
    synchronize(device)
    total_metric_step_runtime_ms = 1000.0 * (
        time.perf_counter() - total_started
    )
    return {
        "case_id": case_id,
        "base_topology": base_topology,
        "base_topology_sha256": topology_sha256(base_topology),
        "base_pulse_positions_sha256": pulse_positions_sha256(
            base_topology["pulse_positions_samples"]
        ),
        "proxy_before": float(proxy_before.detach()),
        "proxy_target": target,
        "proxy_loss": float(loss.detach()),
        "gradient_l2_norm": float(gradient.norm()),
        "gradient_rms": float(gradient.square().mean().sqrt()),
        "gradient_finite": True,
        "gradient_runtime_ms": gradient_runtime_ms,
        "base_refresh_runtime_ms": base_refresh_ms,
        "base_refresh_internal_ms": float(base_topology["pulse_runtime_ms"]),
        "base_refresh_request_wall_ms": base_request_wall_ms,
        "base_refresh_client_staging_ms": base_staging_ms,
        "pcm24_write_total_ms": sum(pcm24_write_ms),
        "pcm24_write_ms": pcm24_write_ms,
        "candidate_refresh_concurrent_wall_ms": candidate_refresh_wall_ms,
        "candidate_refresh_request_wall_sum_ms": sum(
            float(attempt["candidate_refresh_request_wall_ms"])
            for attempt in attempt_records
        ),
        "candidate_refresh_internal_sum_ms": sum(
            float(attempt["candidate_refresh_internal_ms"])
            for attempt in attempt_records
        ),
        "total_metric_step_runtime_ms": total_metric_step_runtime_ms,
        "runtime_gate_pass": total_metric_step_runtime_ms <= CACHE_RUNTIME_MAX_MS,
        "attempts": attempt_records,
        "candidate_topologies": candidate_topologies,
        "selected_backtrack_index": selected_index,
        "selected_alpha": (
            float(selected_input["alpha"]) if selected_input is not None else None
        ),
        "selector_pass": selected_input is not None,
        "selected_path": (
            candidate_paths[selected_index] if selected_index is not None else None
        ),
        "selected_waveform": (
            stored_waveforms[selected_index] if selected_index is not None else None
        ),
        "selected_topology": (
            candidate_topologies[selected_index]
            if selected_index is not None
            else None
        ),
        "selected_attempt": (
            attempt_records[selected_index] if selected_index is not None else None
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism = aggregate_candidate(TRUST_REGION_CANDIDATE_NAME, rows)
    pathology = aggregate_pathology_guardrails(rows)
    denoising = aggregate_denoising(rows)
    prototype_mechanism_gates = {
        "complete_prototype_coverage": len(rows) == EXPECTED_CASE_COUNT,
        "exact_db_effect": (
            mechanism["exact_db_improvement_fraction"]
            >= IMPROVEMENT_FRACTION_GATE
            and mechanism["median_exact_db_normalized_gap_reduction"]
            >= MEDIAN_REDUCTION_GATE
        ),
        "gradient": all(
            row["gradient_finite"]
            and GRADIENT_NORM_RANGE[0]
            <= row["gradient_l2_norm"]
            <= GRADIENT_NORM_RANGE[1]
            for row in rows
        ),
        "total_metric_step_runtime": all(
            row["total_metric_step_runtime_ms"] <= CACHE_RUNTIME_MAX_MS
            for row in rows
        ),
        "nonselected": all(
            value <= NONSELECTED_MEDIAN_INCREASE_GATE
            for value in mechanism[
                "nonselected_median_normalized_gap_increase"
            ].values()
        ),
        "safety": all(
            row["residual_rms_db"] <= RESIDUAL_CEILING_DB
            and row["cosine_similarity"] >= MINIMUM_COSINE
            and row["clip_fraction"] <= MAXIMUM_CLIP_FRACTION
            for row in rows
        ),
        "topology_stability": all(
            row["topology_stability_pass"] for row in rows
        ),
    }
    integration_gates = {
        "prototype_mechanism": all(prototype_mechanism_gates.values()),
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
        "selector_coverage": all(row["selector_pass"] for row in rows),
        "selector_uses_no_candidate_exact_outcome": all(
            row["selector_uses_no_candidate_exact_outcome"] for row in rows
        ),
        "certified_topology_rebound": all(
            row["certified_topology_rebound"] for row in rows
        ),
        "base_topology_rebound": all(row["base_topology_rebound"] for row in rows),
        "pcm24_effective_step": all(row["pcm24_effective_step_pass"] for row in rows),
        "all_four_candidate_topologies_refreshed": all(
            row["candidate_topology_refresh_count"] == len(ALPHA_LADDER)
            for row in rows
        ),
        "target_topology_not_used": all(
            row["clean_target_topology_drives_output"] is False for row in rows
        ),
        "exact_metric_mapping_parity": all(
            row["metric_reconstruction_max_pcm16_error"] == 0
            and row["metric_reconstruction_differing_samples"] == 0
            and row["candidate_metric_reconstruction_max_pcm16_error"] == 0
            and row["candidate_metric_reconstruction_differing_samples"] == 0
            for row in rows
        ),
    }
    return {
        "candidate": TRUST_REGION_CANDIDATE_NAME,
        "alpha_max": FIXED_ALPHA,
        "alpha_ladder": list(ALPHA_LADDER),
        "mechanism": mechanism,
        "prototype_mechanism_gates": prototype_mechanism_gates,
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "integration_gates": integration_gates,
        "selected_alpha_counts": {
            str(alpha): sum(math.isclose(row["selected_alpha"], alpha) for row in rows)
            for alpha in ALPHA_LADDER
        },
        "selected_backtrack_median": median(
            row["selected_backtrack_index"] for row in rows
        ),
        "total_metric_step_runtime_ms": {
            "median": median(row["total_metric_step_runtime_ms"] for row in rows),
            "maximum": max(row["total_metric_step_runtime_ms"] for row in rows),
            "formal_gate_ms": CACHE_RUNTIME_MAX_MS,
        },
        "all_gates_pass": all(integration_gates.values()),
    }


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
        "runtime_worker": validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "runtime-v15 exact topology worker",
        ),
    }
    observed_avqi_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_avqi_tree_hash
    panel = read_json(args.panel_contract)
    input_results = read_csv(args.fresh_results)
    full_panel_rows, input_by_case = validate_panel(panel, input_results)
    full_by_case = {row["case_id"]: row for row in full_panel_rows}
    if not set(PROTOTYPE_CASE_IDS).issubset(full_by_case):
        missing = sorted(set(PROTOTYPE_CASE_IDS) - set(full_by_case))
        raise ValueError(f"prototype case coverage drift: {missing}")
    panel_rows = [full_by_case[case_id] for case_id in PROTOTYPE_CASE_IDS]
    validate_dev_files(panel_rows)

    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    torch_warmup = synthetic_torch_warmup(predictor, target_scale, device)

    workers: list[ExactShimmerTopologyWorker] = []
    worker_startups: list[dict[str, Any]] = []
    worker_warmups: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    try:
        for worker_index in range(WORKER_COUNT):
            worker = ExactShimmerTopologyWorker(
                args.exact_python,
                args.runtime_worker_script,
                args.avqi_code_root,
                args.avqi_code_tree_sha256,
            )
            workers.append(worker)
            warmup, warmup_ms = worker.warmup()
            worker_startups.append(
                {
                    "worker_index": worker_index,
                    "startup_ms": worker.startup_ms,
                    **worker.startup,
                }
            )
            worker_warmups.append(
                {
                    "worker_index": worker_index,
                    "warmup_ms": warmup_ms,
                    **warmup,
                }
            )
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            for index, panel_row in enumerate(panel_rows, start=1):
                case_records.append(
                    evaluate_case(
                        panel_row,
                        float(
                            input_by_case[panel_row["case_id"]][
                                "exact_target_shimmer_db"
                            ]
                        ),
                        predictor,
                        target_scale,
                        device,
                        workers,
                        executor,
                        waveform_root,
                    )
                )
                print(f"trust_region_step={index}/{len(panel_rows)}", flush=True)
    finally:
        for worker in workers:
            worker.close()

    attempt_rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, case_records, strict=True):
        for attempt in record["attempts"]:
            attempt_rows.append(
                {
                    "case_id": panel_row["case_id"],
                    "speaker_id": panel_row["speaker_id"],
                    "view": panel_row["view"],
                    "condition": panel_row["condition"],
                    "sample_group": panel_row["sample_group"],
                    **attempt,
                    "selected": (
                        record["selected_backtrack_index"]
                        == attempt["backtrack_index"]
                    ),
                    "total_metric_step_runtime_ms": record[
                        "total_metric_step_runtime_ms"
                    ],
                }
            )
    attempts_path = args.output_dir / "trust_region_attempts.csv"
    write_csv(attempts_path, attempt_rows)

    selector_failures = [
        record["case_id"] for record in case_records if not record["selector_pass"]
    ]
    if selector_failures:
        decision = (
            "NO_GO_SHIMMER_DB_TOPOLOGY_TRUST_REGION_SELECTOR_4CASE_PROTOTYPE"
        )
        report = {
            "schema_version": "avqi-route-c-shimmer-db-trust-region-v16-dev-v1",
            "decision": decision,
            "panel_label": args.panel_label,
            "dev_only": True,
            "candidate_exact_outcomes_opened": False,
            "selector_failures": selector_failures,
            "selector_contract": selector_contract(),
            "source_commit": args.source_commit,
            "slurm_job_id": args.slurm_job_id,
            "source_sha256": source_hashes,
            "torch_synthetic_warmup": torch_warmup,
            "worker_startups": worker_startups,
            "worker_synthetic_warmups": worker_warmups,
            "generator_loaded": False,
            "generator_optimizer_created": False,
            "generator_optimizer_steps": 0,
            "formal_generator_training_authorized": False,
            "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        }
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        write_receipt(args, decision, report_path, attempts_path, None)
        print(json.dumps({"decision": decision, "failures": selector_failures}))
        return

    selector_seal = {
        "schema_version": "avqi-route-c-shimmer-db-trust-region-v16-selector-seal-v1",
        "candidate": TRUST_REGION_CANDIDATE_NAME,
        "panel_label": args.panel_label,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "selector_contract": selector_contract(),
        "candidate_exact_outcomes_present": False,
        "selection_uses_candidate_exact_outcome": False,
        "attempts_sha256": sha256_file(attempts_path),
        "rows": [
            {
                "case_id": panel_row["case_id"],
                "selected_alpha": record["selected_alpha"],
                "selected_backtrack_index": record["selected_backtrack_index"],
                "candidate_path": str(record["selected_path"].resolve()),
                "candidate_sha256": sha256_file(record["selected_path"]),
                "certified_topology_sha256": topology_sha256(
                    record["selected_topology"]
                ),
                "total_metric_step_runtime_ms": record[
                    "total_metric_step_runtime_ms"
                ],
                "runtime_gate_pass": record["runtime_gate_pass"],
            }
            for panel_row, record in zip(panel_rows, case_records, strict=True)
        ],
    }
    selector_seal_path = args.output_dir / "selector_seal.json"
    write_json(selector_seal_path, selector_seal)

    exact_items = [
        {
            "id": f"selected:{panel_row['case_id']}",
            "case_id": panel_row["case_id"],
            "role": "trust_region_selected_candidate",
            "path": str(record["selected_path"].resolve()),
            "view": panel_row["view"],
            "score_components": True,
            "exact_metric_topology": True,
        }
        for panel_row, record in zip(panel_rows, case_records, strict=True)
    ]
    exact_after = run_exact(exact_items, args.exact_python, args.avqi_code_root)
    after_by_case = {row["case_id"]: row for row in exact_after["rows"]}
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, case_records, strict=True):
        case_id = panel_row["case_id"]
        input_row = input_by_case[case_id]
        after = after_by_case[case_id]
        certified_topology_rebound = bool(
            require_exact_topology_equal(
                record["selected_topology"],
                after,
                f"selected topology rebound {case_id}",
            )
        )
        selected_attempt = record["selected_attempt"]
        target_components = exact_vector(input_row, "target")
        base_components = exact_vector(input_row, "before")
        after_components = exact_components(after)
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        candidate_waveform = read_waveform(record["selected_path"])
        base_topology = record["base_topology"]
        base_topology_rebound = (
            record["base_topology_sha256"]
            == str(input_row["composite_topology_sha256"])
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": TRUST_REGION_CANDIDATE_NAME,
            "optimized_component": "shimmer_db",
            "alpha_max": FIXED_ALPHA,
            "selected_alpha": record["selected_alpha"],
            "selected_backtrack_index": record["selected_backtrack_index"],
            "candidate_path": str(record["selected_path"].resolve()),
            "candidate_sha256": sha256_file(record["selected_path"]),
            "proxy_before": record["proxy_before"],
            "proxy_after_frozen_topology": selected_attempt[
                "proxy_after_frozen_topology"
            ],
            "proxy_target": record["proxy_target"],
            "proxy_loss": record["proxy_loss"],
            "gradient_l2_norm": record["gradient_l2_norm"],
            "gradient_rms": record["gradient_rms"],
            "gradient_finite": record["gradient_finite"],
            "pulse_refresh_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "torch_step_runtime_ms": record["gradient_runtime_ms"],
            "total_metric_step_overhead_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "total_metric_step_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "runtime_gate_pass": record["runtime_gate_pass"],
            "base_refresh_runtime_ms": record["base_refresh_runtime_ms"],
            "base_refresh_internal_ms": record["base_refresh_internal_ms"],
            "base_refresh_request_wall_ms": record[
                "base_refresh_request_wall_ms"
            ],
            "base_refresh_client_staging_ms": record[
                "base_refresh_client_staging_ms"
            ],
            "gradient_runtime_ms": record["gradient_runtime_ms"],
            "pcm24_write_total_ms": record["pcm24_write_total_ms"],
            "candidate_refresh_concurrent_wall_ms": record[
                "candidate_refresh_concurrent_wall_ms"
            ],
            "candidate_refresh_request_wall_sum_ms": record[
                "candidate_refresh_request_wall_sum_ms"
            ],
            "candidate_refresh_internal_sum_ms": record[
                "candidate_refresh_internal_sum_ms"
            ],
            "candidate_topology_refresh_count": len(record["attempts"]),
            "selector_pass": record["selector_pass"],
            "selector_uses_no_candidate_exact_outcome": True,
            "pcm24_effective_step_pass": selected_attempt[
                "pcm24_effective_step_pass"
            ],
            "pcm24_changed_samples": selected_attempt[
                "pcm24_changed_samples"
            ],
            "pcm24_changed_fraction": selected_attempt[
                "pcm24_changed_fraction"
            ],
            "pcm24_residual_rms_lsb": selected_attempt[
                "pcm24_residual_rms_lsb"
            ],
            "base_output_exact_metric_pulse_count": int(
                base_topology["pulse_count"]
            ),
            "candidate_exact_metric_pulse_count": int(after["pulse_count"]),
            "metric_sample_count": int(base_topology["metric_sample_count"]),
            "metric_constant_prefix_samples": int(
                base_topology["metric_constant_prefix_samples"]
            ),
            "metric_source_range_count": int(
                base_topology["metric_source_range_count"]
            ),
            "metric_mapped_sample_count": int(
                base_topology["metric_mapped_sample_count"]
            ),
            "metric_reconstruction_max_pcm16_error": int(
                base_topology["metric_reconstruction_max_pcm16_error"]
            ),
            "metric_reconstruction_differing_samples": int(
                base_topology["metric_reconstruction_differing_samples"]
            ),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                after["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                after["metric_reconstruction_differing_samples"]
            ),
            "candidate_exact_pulse_runtime_ms": float(after["pulse_runtime_ms"]),
            "target_label_rebound": True,
            "base_topology_rebound": base_topology_rebound,
            "certified_topology_rebound": certified_topology_rebound,
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
        row["forward_normalized_abs_error_shimmer_db"] = abs(
            row["proxy_before"] - row["exact_before_shimmer_db"]
        ) / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        row.update(topology_stability(base_topology, after))
        row.update(waveform_safety(base_waveform.numpy(), candidate_waveform.numpy()))
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)

    results_path = args.output_dir / "trust_region_results.csv"
    write_csv(results_path, rows)
    summary = summarize(rows)
    decision = (
        "PASS_SHIMMER_DB_TOPOLOGY_TRUST_REGION_4CASE_PROTOTYPE"
        if summary["all_gates_pass"]
        else "NO_GO_SHIMMER_DB_TOPOLOGY_TRUST_REGION_4CASE_PROTOTYPE"
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-db-trust-region-v16-dev-v1",
        "decision": decision,
        "panel_label": args.panel_label,
        "dev_only": True,
        "opened_panel_reused_for_mechanism_diagnosis": True,
        "promotion_authorized": False,
        "new_sealed_panel_authorized": False,
        "candidate": TRUST_REGION_CANDIDATE_NAME,
        "route_type": "hybrid_praat_assisted_topology_certified_trust_region",
        "pure_torch_estimator": False,
        "selector_contract": selector_contract(),
        "selector_seal_sha256": sha256_file(selector_seal_path),
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_sha256": source_hashes,
        "torch_synthetic_warmup": torch_warmup,
        "worker_startups": worker_startups,
        "worker_synthetic_warmups": worker_warmups,
        "exact_scorer_versions": {
            "parselmouth": exact_after["parselmouth_version"],
            "praat": exact_after["praat_version"],
        },
        "summary": summary,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifacts": {
            "selector_seal": selector_seal_path.name,
            "attempts": attempts_path.name,
            "results": results_path.name,
        },
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    write_receipt(
        args,
        decision,
        report_path,
        attempts_path,
        selector_seal_path,
        results_path,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "panel_label": args.panel_label,
                "selected_alpha_counts": summary["selected_alpha_counts"],
                "runtime": summary["total_metric_step_runtime_ms"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def selector_contract() -> dict[str, Any]:
    return {
        "alpha_max": FIXED_ALPHA,
        "alpha_ladder": list(ALPHA_LADDER),
        "max_backtracks": MAX_BACKTRACKS,
        "selection": "largest_alpha_passing_all_certificates",
        "allowed_information": [
            "base_current_output_exact_topology",
            "candidate_exact_topology_stability",
            "finite_and_frozen_waveform_safety",
            "frozen_topology_proxy_gap_nonregression",
            "pcm24_effective_step",
        ],
        "forbidden_information": [
            "candidate_exact_shimmer_db",
            "candidate_exact_avqi_components",
            "candidate_other_exact_outcomes",
            "speaker_id",
            "severity",
            "condition",
            "panel_specific_exception",
        ],
        "proxy_gap_tolerance": PROXY_GAP_TOLERANCE,
        "pcm24_min_changed_samples": PCM24_MIN_CHANGED_SAMPLES,
        "pcm24_min_residual_rms_lsb": PCM24_MIN_RESIDUAL_RMS_LSB,
        "worker_count": WORKER_COUNT,
        "candidate_refresh_execution": "fixed_ladder_parallel_persistent_workers",
        "formal_total_metric_step_runtime_ms": CACHE_RUNTIME_MAX_MS,
        "runtime_includes": [
            "base_read_staging_and_exact_topology_refresh",
            "gradient_and_frozen_proxy",
            "all_four_pcm24_writes",
            "all_four_candidate_exact_topology_refreshes",
            "selector_and_synchronization",
        ],
    }


def write_receipt(
    args: argparse.Namespace,
    decision: str,
    report_path: Path,
    attempts_path: Path,
    selector_seal_path: Path | None,
    results_path: Path | None = None,
) -> None:
    artifacts = {
        report_path.name: sha256_file(report_path),
        attempts_path.name: sha256_file(attempts_path),
    }
    if selector_seal_path is not None:
        artifacts[selector_seal_path.name] = sha256_file(selector_seal_path)
    if results_path is not None:
        artifacts[results_path.name] = sha256_file(results_path)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-trust-region-v16-receipt-v1",
        "decision": decision,
        "panel_label": args.panel_label,
        "dev_only": True,
        "promotion_authorized": False,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "case_count": EXPECTED_CASE_COUNT,
        "prototype_case_ids": list(PROTOTYPE_CASE_IDS),
        "alpha_ladder": list(ALPHA_LADDER),
        "candidate_exact_outcome_used_by_selector": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": artifacts,
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)


if __name__ == "__main__":
    main()
