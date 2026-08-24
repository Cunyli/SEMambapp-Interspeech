#!/usr/bin/env python3
"""Finalize a hash-sealed Candidate-C panel without regenerating waveforms.

This recovery path exists only for the post-seal target-topology assertion in
job 19906678.  It validates every sealed artifact, scores targets with the
standard exact AVQI scorer, independently scores/relocates base and candidate
pulses, and reconstructs the frozen Torch step in memory.  It never runs the
simulator, generator, or a waveform update and never changes a frozen gate.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    AVQI_COMPONENT_NAMES,
    CACHE_RUNTIME_MAX_MS,
    CANDIDATE_NAME,
    FIXED_ALPHA,
    SAMPLE_RATE,
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    component_fields,
    exact_components,
    exact_index,
    full_band_pathology_guardrails,
    load_json,
    load_predictor,
    metric_source_indices_from_topology,
    normalized_gradient_step,
    pulse_positions_sha256,
    read_waveform,
    repository_head,
    run_exact,
    run_exact_batch,
    sha256_file,
    summarize_fresh_panel,
    topology_stability,
    validate_authorization,
    validate_file_hash,
    waveform_safety,
    write_completion,
    write_csv,
)


SEALED_SOURCE_COMMIT = "60dd0fe9dc748ebb793937e67aa0e38a7909876f"
SEALED_JOB_ID = "19906678"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--mechanism-report-sha256", required=True)
    parser.add_argument("--mechanism-receipt", type=Path, required=True)
    parser.add_argument("--mechanism-receipt-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--generator-config-sha256", required=True)
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--generator-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--sealed-output-dir", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--target-label-contract-sha256", required=True)
    parser.add_argument("--candidate-seal-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--sealed-source-commit", required=True)
    parser.add_argument("--sealed-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_sealed_artifacts(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if args.sealed_source_commit != SEALED_SOURCE_COMMIT:
        raise ValueError("sealed source commit argument drift")
    if args.sealed_job_id != SEALED_JOB_ID:
        raise ValueError("sealed job ID argument drift")
    if not args.sealed_output_dir.is_dir():
        raise FileNotFoundError(args.sealed_output_dir)
    for name in (
        "fresh_panel_results.csv",
        "fresh_panel_report.json",
        "completion_receipt.json",
    ):
        if (args.sealed_output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite finalized artifact: {name}")

    panel_path = args.sealed_output_dir / "panel_contract.json"
    target_path = args.sealed_output_dir / "target_label_contract.json"
    seal_path = args.sealed_output_dir / "candidate_seal.json"
    validate_file_hash(
        panel_path,
        args.panel_contract_sha256,
        "sealed panel contract",
    )
    validate_file_hash(
        target_path,
        args.target_label_contract_sha256,
        "sealed target-label contract",
    )
    validate_file_hash(
        seal_path,
        args.candidate_seal_sha256,
        "sealed Candidate-C contract",
    )
    panel = load_json(panel_path)
    target_contract = load_json(target_path)
    seal = load_json(seal_path)
    if panel.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-c-fresh-panel-v1"
    ):
        raise ValueError("sealed panel schema drift")
    if target_contract.get("schema_version") != (
        "avqi-route-c-shimmer-db-supervised-target-v1"
    ):
        raise ValueError("sealed target-label schema drift")
    if seal.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-c-seal-v1"
    ):
        raise ValueError("sealed candidate schema drift")
    if panel.get("source_commit") != SEALED_SOURCE_COMMIT:
        raise ValueError("sealed panel source commit drift")
    if panel.get("slurm_job_id") != SEALED_JOB_ID:
        raise ValueError("sealed panel job ID drift")
    if seal.get("source_commit") != SEALED_SOURCE_COMMIT:
        raise ValueError("sealed candidate source commit drift")
    if seal.get("slurm_job_id") != SEALED_JOB_ID:
        raise ValueError("sealed candidate job ID drift")
    if seal.get("candidate") != CANDIDATE_NAME:
        raise ValueError("sealed candidate name drift")
    if float(seal.get("fixed_alpha", math.nan)) != FIXED_ALPHA:
        raise ValueError("sealed candidate alpha drift")
    if seal.get("selection_or_tuning_on_this_panel") is not False:
        raise ValueError("sealed panel contains selection or tuning")
    if seal.get("exact_base_or_candidate_scoring_started_after_this_seal") is not True:
        raise ValueError("sealed exact ordering contract drift")
    if seal.get("panel_contract_sha256") != args.panel_contract_sha256:
        raise ValueError("candidate seal does not bind the panel contract")
    if seal.get("target_label_contract_sha256") != (
        args.target_label_contract_sha256
    ):
        raise ValueError("candidate seal does not bind target labels")
    if panel.get("candidate_c", {}).get(
        "selection_or_tuning_on_this_panel"
    ) is not False:
        raise ValueError("panel contract permits Candidate-C tuning")
    if panel.get("candidate_c", {}).get(
        "clean_target_topology_drives_output"
    ) is not False:
        raise ValueError("panel contract exposes target topology to output")
    if target_contract.get(
        "clean_target_pulse_positions_exposed_to_output_branch"
    ) is not False:
        raise ValueError("target-label contract exposes pulse positions")

    panel_by_case = {row["case_id"]: dict(row) for row in panel["rows"]}
    target_by_case = {
        row["case_id"]: dict(row) for row in target_contract["rows"]
    }
    seal_by_case = {row["case_id"]: dict(row) for row in seal["rows"]}
    case_ids = set(panel_by_case)
    if (
        len(case_ids) != 12
        or set(target_by_case) != case_ids
        or set(seal_by_case) != case_ids
    ):
        raise ValueError("sealed case coverage drift")
    if panel.get("panel_validation", {}).get("speaker_count") != 6:
        raise ValueError("sealed speaker count drift")
    if panel.get("panel_validation", {}).get(
        "previous_waveform_speaker_overlap"
    ) != []:
        raise ValueError("sealed panel speaker overlap drift")
    if panel.get("speaker_selection", {}).get(
        "selection_uses_exact_scores"
    ) is not False:
        raise ValueError("sealed speaker selection used exact scores")

    for case_id in sorted(case_ids):
        panel_row = panel_by_case[case_id]
        target_row = target_by_case[case_id]
        seal_row = seal_by_case[case_id]
        if panel_row["target_sha256"] != seal_row["target_sha256"]:
            raise ValueError(f"sealed target hash binding drift: {case_id}")
        if panel_row["base_sha256"] != seal_row["base_sha256"]:
            raise ValueError(f"sealed base hash binding drift: {case_id}")
        if target_row["target_sha256"] != seal_row["target_sha256"]:
            raise ValueError(f"target-label hash binding drift: {case_id}")
        validate_file_hash(
            Path(panel_row["target_path"]),
            seal_row["target_sha256"],
            f"sealed target waveform {case_id}",
        )
        validate_file_hash(
            Path(panel_row["base_path"]),
            seal_row["base_sha256"],
            f"sealed base waveform {case_id}",
        )
        validate_file_hash(
            Path(seal_row["candidate_path"]),
            seal_row["candidate_sha256"],
            f"sealed candidate waveform {case_id}",
        )
    return panel, panel_by_case, target_by_case, seal_by_case


def pcm24_roundtrip(waveform: torch.Tensor) -> torch.Tensor:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        waveform.detach().cpu().numpy(),
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_24",
    )
    buffer.seek(0)
    audio, sample_rate = sf.read(buffer, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("in-memory PCM24 reconstruction drift")
    return torch.from_numpy(audio.copy())


def reconstruct_step_diagnostics(
    panel_row: dict[str, Any],
    target_row: dict[str, Any],
    seal_row: dict[str, Any],
    base_exact: dict[str, Any],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    if base_exact.get("scoring_status") != "ok" or base_exact.get(
        "pulse_count",
        0,
    ) < 3:
        raise RuntimeError(f"sealed base topology unavailable: {panel_row['case_id']}")
    observed_topology_hash = pulse_positions_sha256(
        base_exact["pulse_positions_samples"]
    )
    if observed_topology_hash != seal_row["base_topology_sha256"]:
        raise ValueError(f"sealed base topology drift: {panel_row['case_id']}")
    waveform = read_waveform(Path(panel_row["base_path"])).to(device)
    waveform = waveform.requires_grad_(True)
    source_indices_np = metric_source_indices_from_topology(
        base_exact,
        source_sample_count=waveform.numel(),
    )
    source_indices = torch.as_tensor(
        source_indices_np,
        device=device,
        dtype=torch.long,
    )
    pulses = waveform.new_tensor(base_exact["pulse_positions_samples"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    proxy_before = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            base_exact["metric_constant_prefix_samples"]
        ),
    )[1]
    target_value = float(target_row["exact_target_shimmer_db"])
    loss = (
        (proxy_before - target_value)
        / target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    ).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    reconstructed = normalized_gradient_step(
        waveform,
        gradient,
        FIXED_ALPHA,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torch_runtime_ms = 1000.0 * (time.perf_counter() - started)
    reconstructed_stored = pcm24_roundtrip(reconstructed)
    sealed_candidate = read_waveform(Path(seal_row["candidate_path"]))
    reconstructed_equal = bool(
        torch.equal(reconstructed_stored, sealed_candidate)
    )
    sealed_device = sealed_candidate.to(device)
    with torch.inference_mode():
        proxy_after = predictor.raw_shimmer_from_pulse_positions(
            sealed_device,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=int(
                base_exact["metric_constant_prefix_samples"]
            ),
        )[1]
    return {
        "proxy_before": float(proxy_before.detach()),
        "proxy_after_frozen_topology": float(proxy_after.detach()),
        "proxy_target": target_value,
        "proxy_loss": float(loss.detach()),
        "gradient_l2_norm": float(gradient.norm()),
        "gradient_rms": float(gradient.square().mean().sqrt()),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "torch_step_runtime_ms": torch_runtime_ms,
        "pulse_refresh_runtime_ms": float(
            seal_row["pulse_refresh_runtime_ms"]
        ),
        "total_metric_step_overhead_ms": (
            float(seal_row["pulse_refresh_runtime_ms"]) + torch_runtime_ms
        ),
        "pulse_topology_sha256": observed_topology_hash,
        "pulse_count": int(base_exact["pulse_count"]),
        "metric_sample_count": int(base_exact["metric_sample_count"]),
        "metric_constant_prefix_samples": int(
            base_exact["metric_constant_prefix_samples"]
        ),
        "metric_source_range_count": int(
            base_exact["metric_source_range_count"]
        ),
        "metric_mapped_sample_count": int(
            base_exact["metric_mapped_sample_count"]
        ),
        "metric_reconstruction_max_pcm16_error": int(
            base_exact["metric_reconstruction_max_pcm16_error"]
        ),
        "metric_reconstruction_differing_samples": int(
            base_exact["metric_reconstruction_differing_samples"]
        ),
        "sealed_candidate_reconstructed_exactly": reconstructed_equal,
    }


def main() -> None:
    args = parse_args()
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("finalizer source commit differs from repository HEAD")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    if not args.avqi_code_root.is_dir():
        raise FileNotFoundError(args.avqi_code_root)
    authorization = validate_authorization(args)
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code tree drift")
    panel, panel_by_case, target_by_case, seal_by_case = (
        validate_sealed_artifacts(args)
    )
    case_ids = [row["case_id"] for row in panel["rows"]]

    target_items = [
        {
            "id": f"target:{case_id}",
            "path": panel_by_case[case_id]["target_path"],
            "view": panel_by_case[case_id]["view"],
        }
        for case_id in case_ids
    ]
    target_exact = run_exact_batch(
        target_items,
        args.exact_python,
        args.avqi_code_root,
    )
    target_exact_by_id = exact_index(target_exact)

    output_items: list[dict[str, Any]] = []
    for case_id in case_ids:
        panel_row = panel_by_case[case_id]
        seal_row = seal_by_case[case_id]
        for role, path in (
            ("base", panel_row["base_path"]),
            ("candidate", seal_row["candidate_path"]),
        ):
            output_items.append(
                {
                    "id": f"{role}:{case_id}",
                    "case_id": case_id,
                    "role": role,
                    "path": path,
                    "view": panel_row["view"],
                    "score_components": True,
                    "exact_metric_topology": True,
                }
            )
    exact_wall_started = time.perf_counter()
    output_exact = run_exact(
        output_items,
        args.exact_python,
        args.avqi_code_root,
    )
    exact_output_wall_ms = 1000.0 * (
        time.perf_counter() - exact_wall_started
    )
    if (
        output_exact["parselmouth_version"]
        != target_exact["parselmouth_version"]
        or output_exact["praat_version"] != target_exact["praat_version"]
    ):
        raise ValueError("exact scorer version drift during sealed finalization")
    output_by_id = {row["id"]: row for row in output_exact["rows"]}

    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        panel_row = panel_by_case[case_id]
        target_contract_row = target_by_case[case_id]
        seal_row = seal_by_case[case_id]
        base_row = output_by_id[f"base:{case_id}"]
        candidate_row = output_by_id[f"candidate:{case_id}"]
        for exact_row in (base_row, candidate_row):
            if (
                exact_row.get("scoring_status") != "ok"
                or exact_row.get("pulse_count", 0) < 3
            ):
                raise RuntimeError(
                    f"sealed output exact scoring failed: {exact_row.get('id')} "
                    f"{exact_row.get('error_type')} "
                    f"{exact_row.get('error_message')}"
                )
        diagnostics = reconstruct_step_diagnostics(
            panel_row,
            target_contract_row,
            seal_row,
            base_row,
            predictor,
            target_scale,
            device,
        )
        target_components = target_exact_by_id[f"target:{case_id}"]
        base_components = exact_components(base_row)
        candidate_components = exact_components(candidate_row)
        target_label_rebound = (
            float(target_components[SHIMMER_DB_INDEX])
            == float(target_contract_row["exact_target_shimmer_db"])
        )
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        candidate_waveform = read_waveform(Path(seal_row["candidate_path"]))
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": CANDIDATE_NAME,
            "optimized_component": "shimmer_db",
            "fixed_alpha": FIXED_ALPHA,
            "candidate_path": seal_row["candidate_path"],
            "candidate_sha256": seal_row["candidate_sha256"],
            "unique_topology_refresh_key": f"metric_base_output:{case_id}",
            "base_output_exact_metric_pulse_count": diagnostics["pulse_count"],
            "candidate_exact_metric_pulse_count": int(
                candidate_row["pulse_count"]
            ),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                candidate_row["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                candidate_row["metric_reconstruction_differing_samples"]
            ),
            "candidate_exact_pulse_runtime_ms": float(
                candidate_row["pulse_runtime_ms"]
            ),
            "target_label_rebound": target_label_rebound,
            "base_topology_rebound": (
                diagnostics["pulse_topology_sha256"]
                == seal_row["base_topology_sha256"]
            ),
            "clean_target_topology_drives_output": False,
            **diagnostics,
        }
        component_fields(
            row,
            target_components,
            base_components,
            candidate_components,
            target_scale_np,
        )
        row["material_shimmer_db_gap"] = (
            row["exact_absolute_gap_before_shimmer_db"]
            / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
            > 0.02
        )
        row["forward_normalized_abs_error_shimmer_db"] = (
            abs(row["proxy_before"] - row["exact_before_shimmer_db"])
            / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        )
        row.update(topology_stability(base_row, candidate_row))
        row.update(
            waveform_safety(
                base_waveform.numpy(),
                candidate_waveform.numpy(),
            )
        )
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)

    write_csv(args.sealed_output_dir / "fresh_panel_results.csv", rows)
    summary = summarize_fresh_panel(rows)
    summary["fresh_panel_gates"][
        "sealed_candidate_reconstructed_exactly"
    ] = all(row["sealed_candidate_reconstructed_exactly"] for row in rows)
    summary["fresh_panel_gates"]["sealed_artifact_hashes_valid"] = True
    summary["all_gates_pass"] = all(
        summary["fresh_panel_gates"].values()
    )
    mechanism_gates = summary["mechanism"]["gates"]
    effect_and_nonruntime_gates = {
        name: value
        for name, value in mechanism_gates.items()
        if name != "pulse_refresh_runtime"
    }
    effect_and_nonruntime_pass = (
        all(effect_and_nonruntime_gates.values())
        and summary["full_band_pathology_guardrails"]["decision"] == "PASS"
        and summary["denoising"]["decision"] == "PASS"
    )
    component_status = (
        "PASS_SHIMMER_DB_PRAAT_ASSISTED_ROUTE_C_COMPONENT_AND_BOUNDED_PILOT"
        if summary["all_gates_pass"]
        else "NO_GO_SHIMMER_DB_CANDIDATE_C_FRESH_PANEL"
    )
    sealed_runtimes = [
        float(seal_by_case[case_id]["pulse_refresh_runtime_ms"])
        for case_id in case_ids
    ]
    report = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-c-fresh-v1",
        "decision": component_status,
        "component_status": component_status,
        "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
        "pure_torch_estimator": False,
        "fresh_panel_status": "finalized_from_original_hash_seal",
        "fresh_waveform_speakers_not_scorer_domain_external": True,
        "fixed_alpha": FIXED_ALPHA,
        "alpha_selected_or_tuned_on_this_panel": False,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "sealed_candidate_source_commit": SEALED_SOURCE_COMMIT,
        "sealed_candidate_job_id": SEALED_JOB_ID,
        "device": str(device),
        "speaker_count": 6,
        "case_count": 12,
        "generator_loaded_for_frozen_inference_in_sealed_job": True,
        "generator_loaded_in_finalizer": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "formal_pathology_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "effect_and_nonruntime_gates_pass": effect_and_nonruntime_pass,
        "sealed_runtime_gate_pass": mechanism_gates[
            "pulse_refresh_runtime"
        ],
        "postseal_recovery": {
            "reason": "unused clean-target topology parity assertion",
            "simulation_rerun": False,
            "generator_inference_rerun": False,
            "candidate_step_rerun_or_written": False,
            "sealed_waveform_overwrite": False,
            "target_scoring": "standard exact AVQI scorer",
            "base_candidate_scoring": (
                "independent exact AVQI plus exact output topology relocation"
            ),
            "panel_contract_sha256": args.panel_contract_sha256,
            "target_label_contract_sha256": (
                args.target_label_contract_sha256
            ),
            "candidate_seal_sha256": args.candidate_seal_sha256,
        },
        "exact_scorer_versions": {
            "parselmouth": output_exact["parselmouth_version"],
            "praat": output_exact["praat_version"],
        },
        "topology_refresh": {
            "training_side_unique_refresh_calls": len(sealed_runtimes),
            "training_side_runtime_ms": {
                "median": median(sealed_runtimes),
                "maximum": max(sealed_runtimes),
                "frozen_gate_maximum": CACHE_RUNTIME_MAX_MS,
                "gate_pass": max(sealed_runtimes) <= CACHE_RUNTIME_MAX_MS,
            },
            "final_exact_output_scoring_wall_ms": exact_output_wall_ms,
            "final_exact_base_candidate_rows": len(output_items),
        },
        "authorization": authorization,
        "summary": summary,
        "artifacts": {
            "panel_contract": "panel_contract.json",
            "target_label_contract": "target_label_contract.json",
            "candidate_seal": "candidate_seal.json",
            "results": "fresh_panel_results.csv",
        },
    }
    write_completion(args.sealed_output_dir, report)


if __name__ == "__main__":
    main()
