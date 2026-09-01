#!/usr/bin/env python3
"""Run the frozen v14-only Candidate-E directional microdiagnostic.

The script compares Candidate-D and Candidate-E raw/projected directions on a
pre-registered symmetric alpha grid.  Exact Praat is used only after every
candidate waveform has been materialized.  Exact outcomes are written to an
adjudication report and are never available to runtime selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from scripts.avqi_shimmer_db_candidate_e_proxy_v27 import (
    CandidateEProxyResult,
    candidate_e_proxy,
    fixed_pulse_shimmer_db,
    normalized_gradient_step,
    project_cycle_gain_gradient_fixed_order,
)
from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    topology_sha256,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    MATERIAL_GAP_THRESHOLD,
    SAMPLE_RATE,
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    load_predictor,
    metric_source_indices_from_topology,
    read_waveform,
    sha256_file,
)
from scripts.evaluate_avqi_shimmer_db_cycle_projected_backward import (
    read_csv,
    validate_hash,
    validate_panel,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    build_zero_crossing_cycle_plan_vectorized,
    candidate_d_projection_vectorized,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    base_topology_item,
    validate_dev_files,
)
from scripts.evaluate_direct_avqi_waveform_optimization import waveform_safety


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT_D_PROJECTED = "candidate_d_frozen_projected"
VARIANT_E_PROJECTED = "candidate_e_exact_path_projected"
VARIANT_D_RAW = "candidate_d_frozen_raw_ablation"
VARIANT_E_RAW = "candidate_e_exact_path_raw_ablation"
PROJECTED_VARIANTS = (VARIANT_D_PROJECTED, VARIANT_E_PROJECTED)
ABLATION_VARIANTS = (VARIANT_D_RAW, VARIANT_E_RAW)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
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
    parser.add_argument("--exact-scorer-script", type=Path, required=True)
    parser.add_argument("--exact-scorer-script-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty Candidate-E CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def repository_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def alpha_label(alpha: float) -> str:
    sign = "minus" if alpha < 0.0 else "plus" if alpha > 0.0 else "zero"
    magnitude = f"{abs(alpha):.7f}".replace(".", "p")
    return f"{sign}_{magnitude}"


def validate_config(
    config: dict[str, Any],
    panel_rows: list[dict[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
    if config.get("scientific_role") != (
        "development_calibration_mechanism_diagnosis_only"
    ):
        raise ValueError("Candidate-E config is not development-only")
    dataset = config["dataset_contract"]
    speakers = sorted({str(row["speaker_id"]) for row in panel_rows})
    if speakers != sorted(dataset["expected_speakers"]):
        raise ValueError("Candidate-E v14 speaker contract drift")
    forbidden = set(dataset["forbidden_speakers"])
    if forbidden & set(speakers):
        raise ValueError("Candidate-E development overlaps opened-v15 speakers")
    if dataset.get("opened_v15_access_authorized") is not False:
        raise ValueError("Candidate-E config unexpectedly authorizes v15")
    if dataset.get("external_panel_access_authorized") is not False:
        raise ValueError("Candidate-E config unexpectedly authorizes external data")
    if Counter(row["view"] for row in panel_rows) != Counter(
        dataset["expected_views"]
    ):
        raise ValueError("Candidate-E v14 view contract drift")
    grid = tuple(float(value) for value in config["frozen_directional_grid"]["alphas"])
    if tuple(sorted(grid)) != grid or 0.0 not in grid:
        raise ValueError("Candidate-E alpha grid must be sorted and contain zero")
    if set(grid) != {-value for value in grid}:
        raise ValueError("Candidate-E alpha grid is not symmetric")
    ablation = tuple(float(value) for value in config["ablation_alphas"])
    if set(ablation) != {-value for value in ablation} or 0.0 not in ablation:
        raise ValueError("Candidate-E ablation grid is not symmetric")
    primary = float(config["frozen_directional_grid"]["primary_local_alpha"])
    legacy = float(config["frozen_directional_grid"]["legacy_operating_alpha"])
    if {-primary, primary, -legacy, legacy} - set(grid):
        raise ValueError("Candidate-E alpha grid omits a frozen decision point")
    boundaries = config["immutable_boundaries"]
    if boundaries.get("generator_optimizer_steps") != 0:
        raise ValueError("Candidate-E diagnostic may not contain optimizer steps")
    if boundaries.get("authoritative_training_decision") != (
        "NO_GO_AVQI_T2_TRAINING"
    ):
        raise ValueError("Candidate-E training boundary drift")
    if boundaries.get("no_final_waveform_highpass") is not True:
        raise ValueError("Candidate-E emitted-waveform highpass boundary drift")
    return grid, ablation, primary, legacy


def old_proxy_evidence(
    predictor: torch.nn.Module,
    waveform: torch.Tensor,
    pulse_positions: torch.Tensor,
    source_indices: torch.Tensor,
    prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared = predictor._prepare(waveform)
    prepared = prepared.index_select(0, source_indices)
    if prefix:
        prepared = torch.cat((prepared.new_zeros(prefix), prepared))
    return fixed_pulse_shimmer_db(prepared, pulse_positions)


def proxy_evidence(
    variant: str,
    predictor: torch.nn.Module,
    waveform: torch.Tensor,
    pulse_positions: torch.Tensor,
    source_indices: torch.Tensor,
    prefix: int,
) -> tuple[
    float,
    dict[str, Any],
]:
    if variant.startswith("candidate_d_"):
        direct = predictor.raw_shimmer_from_pulse_positions(
            waveform,
            pulse_positions,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=prefix,
        )[1]
        scalar, amplitudes, centers, valid_pair, contributions = old_proxy_evidence(
            predictor,
            waveform,
            pulse_positions,
            source_indices,
            prefix,
        )
        return float(direct.detach()), {
            "evidence_reconstruction_scalar": float(scalar.detach()),
            "evidence_reconstruction_absolute_error": abs(
                float(direct.detach()) - float(scalar.detach())
            ),
            "amplitude_positions_samples": centers.detach().cpu().tolist(),
            "amplitudes": amplitudes.detach().cpu().tolist(),
            "valid_pair_mask": valid_pair.detach().cpu().tolist(),
            "pair_contributions_db": contributions.detach().cpu().tolist(),
        }
    result: CandidateEProxyResult = candidate_e_proxy(
        waveform,
        pulse_positions,
        source_indices,
        prefix,
    )
    return float(result.shimmer_db.detach()), {
        "evidence_reconstruction_scalar": float(result.shimmer_db.detach()),
        "evidence_reconstruction_absolute_error": 0.0,
        "amplitude_positions_samples": (
            result.amplitude_centers.detach().cpu().tolist()
        ),
        "amplitudes": result.amplitudes.detach().cpu().tolist(),
        "valid_pair_mask": result.valid_pair_mask.detach().cpu().tolist(),
        "pair_contributions_db": (
            result.pair_contributions_db.detach().cpu().tolist()
        ),
        "fft_sample_count": result.fft_sample_count,
        "metric_sample_abs_max": result.metric_sample_abs_max,
        "sinc70_peak_upper_bound": result.sinc70_peak_upper_bound,
        "peak_scale_abstention_pass": result.peak_scale_abstention_pass,
    }


def exact_sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def array_max_error(left: list[Any], right: list[Any]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return math.inf
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def mask_equal(left: list[Any], right: list[Any]) -> bool:
    return np.array_equal(
        np.asarray(left, dtype=np.bool_),
        np.asarray(right, dtype=np.bool_),
    )


def materialize_direction(
    case_id: str,
    variant: str,
    alphas: tuple[float, ...],
    base: torch.Tensor,
    direction: torch.Tensor,
    predictor: torch.nn.Module,
    pulses: torch.Tensor,
    source_indices: torch.Tensor,
    prefix: int,
    topology: dict[str, Any],
    primary_alpha: float,
    waveform_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exact_items: list[dict[str, Any]] = []
    evidence_by_item: dict[str, dict[str, Any]] = {}
    variant_root = waveform_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    for alpha in alphas:
        candidate = normalized_gradient_step(base, direction, alpha)
        if not bool(torch.isfinite(candidate).all().detach().cpu()):
            raise ValueError(f"non-finite Candidate-E diagnostic step: {case_id}")
        if float(candidate.detach().abs().max().cpu()) >= 0.999:
            raise ValueError(f"Candidate-E diagnostic step clips: {case_id}")
        path = variant_root / (
            f"{safe_name(case_id)}__{variant}__{alpha_label(alpha)}.wav"
        )
        sf.write(
            path,
            candidate.detach().cpu().numpy(),
            SAMPLE_RATE,
            subtype="PCM_24",
        )
        stored = read_waveform(path).to(base.device)
        if base.dtype == torch.float64:
            stored = stored.to(dtype=torch.float64)
        proxy_value, evidence = proxy_evidence(
            variant,
            predictor,
            stored,
            pulses.to(dtype=stored.dtype),
            source_indices,
            prefix,
        )
        item_id = f"{case_id}:{variant}:{alpha_label(alpha)}"
        include_pulse = alpha in {-primary_alpha, 0.0, primary_alpha}
        if include_pulse:
            evidence_by_item[item_id] = evidence
        safety = waveform_safety(
            base.detach().to(dtype=torch.float32).cpu(),
            stored.detach().to(dtype=torch.float32).cpu(),
        )
        rows.append(
            {
                "item_id": item_id,
                "case_id": case_id,
                "variant": variant,
                "alpha": alpha,
                "candidate_path": str(path.resolve()),
                "candidate_sha256": sha256_file(path),
                "proxy_shimmer_db": proxy_value,
                **safety,
            }
        )
        exact_items.append(
            {
                "item_id": item_id,
                "case_id": case_id,
                "variant": variant,
                "alpha": alpha,
                "waveform_path": str(path.resolve()),
                "topology": topology,
                "include_pulse_evidence": include_pulse,
            }
        )
    return rows, exact_items, evidence_by_item


def directional_summary(
    variant: str,
    case_ids: list[str],
    merged_by_key: dict[tuple[str, str, float], dict[str, Any]],
    target_by_case: dict[str, float],
    scale_value: float,
    primary_alpha: float,
    legacy_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for case_id in case_ids:
        minus = merged_by_key[(case_id, variant, -primary_alpha)]
        zero = merged_by_key[(case_id, variant, 0.0)]
        plus = merged_by_key[(case_id, variant, primary_alpha)]
        target = target_by_case[case_id]
        proxy_gaps = [
            abs(float(row["proxy_shimmer_db"]) - target) / scale_value
            for row in (minus, zero, plus)
        ]
        exact_gaps = [
            abs(float(row["exact_shimmer_db"]) - target) / scale_value
            for row in (minus, zero, plus)
        ]
        proxy_slope = (proxy_gaps[2] - proxy_gaps[0]) / (
            2.0 * primary_alpha
        )
        exact_slope = (exact_gaps[2] - exact_gaps[0]) / (
            2.0 * primary_alpha
        )
        legacy = merged_by_key.get((case_id, variant, legacy_alpha))
        legacy_reduction = None
        if legacy is not None:
            legacy_reduction = (
                abs(float(zero["exact_shimmer_db"]) - target)
                - abs(float(legacy["exact_shimmer_db"]) - target)
            ) / scale_value
        rows.append(
            {
                "case_id": case_id,
                "variant": variant,
                "target_shimmer_db": target,
                "proxy_before": float(zero["proxy_shimmer_db"]),
                "exact_before_fixed_topology": float(zero["exact_shimmer_db"]),
                "proxy_normalized_gap_slope": proxy_slope,
                "exact_normalized_gap_slope": exact_slope,
                "proxy_slope_sign": exact_sign(proxy_slope),
                "exact_slope_sign": exact_sign(exact_slope),
                "direction_sign_match": (
                    exact_sign(proxy_slope) == exact_sign(exact_slope)
                    and exact_sign(proxy_slope) != 0
                ),
                "positive_primary_exact_improves": exact_gaps[2] < exact_gaps[1],
                "negative_primary_exact_improves": exact_gaps[0] < exact_gaps[1],
                "material_exact_gap": (
                    abs(float(zero["exact_shimmer_db"]) - target)
                    > MATERIAL_GAP_THRESHOLD
                ),
                "legacy_alpha_exact_normalized_gap_reduction": legacy_reduction,
            }
        )
    material = [row for row in rows if row["material_exact_gap"]]
    legacy_rows = [
        row
        for row in material
        if row["legacy_alpha_exact_normalized_gap_reduction"] is not None
    ]
    return rows, {
        "variant": variant,
        "case_count": len(rows),
        "direction_sign_match_count": sum(
            bool(row["direction_sign_match"]) for row in rows
        ),
        "direction_sign_match_fraction": sum(
            bool(row["direction_sign_match"]) for row in rows
        )
        / len(rows),
        "positive_primary_exact_improvement_count": sum(
            bool(row["positive_primary_exact_improves"]) for row in rows
        ),
        "material_case_count": len(material),
        "legacy_alpha_material_improvement_count": sum(
            float(row["legacy_alpha_exact_normalized_gap_reduction"]) > 0.0
            for row in legacy_rows
        ),
        "legacy_alpha_material_improvement_fraction": (
            sum(
                float(row["legacy_alpha_exact_normalized_gap_reduction"]) > 0.0
                for row in legacy_rows
            )
            / max(len(legacy_rows), 1)
        ),
        "legacy_alpha_material_median_normalized_gap_reduction": (
            median(
                float(row["legacy_alpha_exact_normalized_gap_reduction"])
                for row in legacy_rows
            )
            if legacy_rows
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    source_hashes = {
        "config": validate_hash(args.config, args.config_sha256, "v27 config"),
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "v14 panel contract",
        ),
        "fresh_results": validate_hash(
            args.fresh_results,
            args.fresh_results_sha256,
            "v14 fresh results",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen predictor",
        ),
        "runtime_worker": validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "exact topology worker",
        ),
        "exact_scorer": validate_hash(
            args.exact_scorer_script,
            args.exact_scorer_script_sha256,
            "v27 exact scorer",
        ),
    }
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_tree_hash
    if repository_head() != args.source_commit:
        raise ValueError("v27 source commit drift")

    config = read_object(args.config)
    panel = read_object(args.panel_contract)
    input_results = read_csv(args.fresh_results)
    panel_rows, input_by_case = validate_panel(panel, input_results)
    validate_dev_files(panel_rows)
    grid, ablation_grid, primary_alpha, legacy_alpha = validate_config(
        config,
        panel_rows,
    )
    if panel.get("speaker_split_before_simulation") is not True:
        raise ValueError("v14 speaker split was not sealed before simulation")

    output_dir = args.output_dir.resolve()
    waveform_root = output_dir / "waveforms"
    output_dir.mkdir(parents=True, exist_ok=False)
    waveform_root.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Candidate-E v27 requires an allocated CUDA device")
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    scale_value = float(target_scale[SHIMMER_DB_INDEX].detach().cpu())
    target_by_case = {
        case_id: float(row["exact_target_shimmer_db"])
        for case_id, row in input_by_case.items()
    }

    topology_worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []
    exact_items: list[dict[str, Any]] = []
    proxy_evidence_by_item: dict[str, dict[str, Any]] = {}
    case_diagnostics: list[dict[str, Any]] = []
    try:
        topology_rows, topology_wall_ms = topology_worker.refresh(
            [
                {
                    **base_topology_item(row),
                    "highpass_mode": NUMPY_HIGHPASS_MODE,
                }
                for row in panel_rows
            ]
        )
        for panel_row, topology in zip(panel_rows, topology_rows, strict=True):
            case_id = str(panel_row["case_id"])
            base_float = read_waveform(Path(panel_row["base_path"])).to(device)
            source_indices = torch.as_tensor(
                metric_source_indices_from_topology(
                    topology,
                    source_sample_count=base_float.numel(),
                ),
                dtype=torch.long,
                device=device,
            )
            pulses_float = base_float.new_tensor(
                topology["pulse_positions_samples"]
            )
            target = target_by_case[case_id]

            d_waveform = base_float.detach().clone().requires_grad_(True)
            d_proxy = predictor.raw_shimmer_from_pulse_positions(
                d_waveform,
                pulses_float,
                metric_source_indices=source_indices,
                metric_constant_prefix_samples=int(
                    topology["metric_constant_prefix_samples"]
                ),
            )[1]
            d_loss = ((d_proxy - target) / target_scale[SHIMMER_DB_INDEX]).square()
            d_raw = torch.autograd.grad(d_loss, d_waveform)[0]

            e_waveform = base_float.detach().to(dtype=torch.float64).requires_grad_(True)
            e_pulses = e_waveform.new_tensor(topology["pulse_positions_samples"])
            e_result = candidate_e_proxy(
                e_waveform,
                e_pulses,
                source_indices,
                int(topology["metric_constant_prefix_samples"]),
            )
            e_loss = ((e_result.shimmer_db - target) / scale_value).square()
            e_raw = torch.autograd.grad(e_loss, e_waveform)[0]
            if not e_result.peak_scale_abstention_pass:
                raise ValueError(f"Candidate-E peak-scale abstention failed: {case_id}")

            plan = build_zero_crossing_cycle_plan_vectorized(
                base_float.detach().cpu().numpy(),
                topology,
            )
            d_projected, d_projection = candidate_d_projection_vectorized(
                d_waveform,
                d_raw,
                plan,
            )
            e_projected, e_projection = project_cycle_gain_gradient_fixed_order(
                e_waveform,
                e_raw,
                plan,
            )
            if not d_projection["projected_gradient_valid"]:
                raise ValueError(f"Candidate-D projection invalid: {case_id}")
            if not e_projection["projected_gradient_valid"]:
                raise ValueError(f"Candidate-E projection invalid: {case_id}")
            synchronize(device)
            cosine_raw = float(
                torch.nn.functional.cosine_similarity(
                    d_raw.to(dtype=torch.float64).reshape(1, -1),
                    e_raw.reshape(1, -1),
                    dim=-1,
                )[0].detach().cpu()
            )
            cosine_projected = float(
                torch.nn.functional.cosine_similarity(
                    d_projected.to(dtype=torch.float64).reshape(1, -1),
                    e_projected.reshape(1, -1),
                    dim=-1,
                )[0].detach().cpu()
            )
            case_diagnostics.append(
                {
                    "case_id": case_id,
                    "speaker_id": panel_row["speaker_id"],
                    "view": panel_row["view"],
                    "condition": panel_row["condition"],
                    "target_shimmer_db": target,
                    "candidate_d_proxy_before": float(d_proxy.detach()),
                    "candidate_e_proxy_before": float(e_result.shimmer_db.detach()),
                    "candidate_d_raw_gradient_l2": float(d_raw.norm().detach()),
                    "candidate_e_raw_gradient_l2": float(e_raw.norm().detach()),
                    "candidate_d_e_raw_gradient_cosine": cosine_raw,
                    "candidate_d_e_projected_gradient_cosine": cosine_projected,
                    "candidate_e_peak_scale_abstention_pass": (
                        e_result.peak_scale_abstention_pass
                    ),
                    "candidate_e_fft_sample_count": e_result.fft_sample_count,
                    "base_topology_sha256": topology_sha256(topology),
                    "candidate_d_projection": d_projection,
                    "candidate_e_projection": e_projection,
                }
            )
            variant_directions = {
                VARIANT_D_PROJECTED: (d_waveform, d_projected, grid),
                VARIANT_E_PROJECTED: (e_waveform, e_projected, grid),
                VARIANT_D_RAW: (d_waveform, d_raw, ablation_grid),
                VARIANT_E_RAW: (e_waveform, e_raw, ablation_grid),
            }
            for variant, (base, direction, alphas) in variant_directions.items():
                rows, items, evidence = materialize_direction(
                    case_id,
                    variant,
                    alphas,
                    base,
                    direction,
                    predictor,
                    pulses_float if base.dtype == torch.float32 else e_pulses,
                    source_indices,
                    int(topology["metric_constant_prefix_samples"]),
                    topology,
                    primary_alpha,
                    waveform_root,
                )
                candidate_rows.extend(rows)
                exact_items.extend(items)
                proxy_evidence_by_item.update(evidence)
    finally:
        topology_worker.close()

    exact_request_path = output_dir / "exact_fixed_topology_request.json"
    exact_output_path = output_dir / "exact_fixed_topology_results.json"
    write_json(
        exact_request_path,
        {
            "scientific_role": "development_calibration_only",
            "avqi_code_root": str(args.avqi_code_root.resolve()),
            "avqi_code_tree_sha256": args.avqi_code_tree_sha256,
            "items": exact_items,
        },
    )
    subprocess.run(
        [
            str(args.exact_python),
            str(args.exact_scorer_script),
            "--request",
            str(exact_request_path),
            "--output",
            str(exact_output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        check=True,
    )
    exact_output = read_object(exact_output_path)
    exact_by_item = {
        str(row["item_id"]): dict(row) for row in exact_output["rows"]
    }
    if set(exact_by_item) != {str(row["item_id"]) for row in candidate_rows}:
        raise ValueError("Candidate-E exact adjudication coverage drift")

    merged_rows: list[dict[str, Any]] = []
    merged_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    parity_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        exact = exact_by_item[str(candidate["item_id"])]
        merged = {
            **candidate,
            "exact_shimmer_db": float(exact["exact_shimmer_db"]),
            "exact_metric_pcm16_sha256": exact["metric_pcm16_sha256"],
            "exact_wall_ms": float(exact["wall_ms"]),
        }
        merged_rows.append(merged)
        key = (
            str(candidate["case_id"]),
            str(candidate["variant"]),
            float(candidate["alpha"]),
        )
        merged_by_key[key] = merged
        evidence = proxy_evidence_by_item.get(str(candidate["item_id"]))
        if evidence is not None:
            parity_rows.append(
                {
                    "item_id": candidate["item_id"],
                    "case_id": candidate["case_id"],
                    "variant": candidate["variant"],
                    "alpha": float(candidate["alpha"]),
                    "scalar_absolute_error": abs(
                        float(candidate["proxy_shimmer_db"])
                        - float(exact["exact_shimmer_db"])
                    ),
                    "amplitude_position_max_absolute_error_samples": (
                        array_max_error(
                            evidence["amplitude_positions_samples"],
                            exact["amplitude_positions_samples"],
                        )
                    ),
                    "amplitude_max_absolute_error": array_max_error(
                        evidence["amplitudes"],
                        exact["amplitudes"],
                    ),
                    "valid_pair_mask_match": mask_equal(
                        evidence["valid_pair_mask"],
                        exact["valid_pair_mask"],
                    ),
                    "contribution_max_absolute_error_db": array_max_error(
                        evidence["pair_contributions_db"],
                        exact["pair_contributions_db"],
                    ),
                    "proxy_evidence_reconstruction_absolute_error": float(
                        evidence["evidence_reconstruction_absolute_error"]
                    ),
                }
            )

    case_ids = [str(row["case_id"]) for row in panel_rows]
    directional_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for variant in (*PROJECTED_VARIANTS, *ABLATION_VARIANTS):
        variant_rows, variant_summary = directional_summary(
            variant,
            case_ids,
            merged_by_key,
            target_by_case,
            scale_value,
            primary_alpha,
            legacy_alpha,
        )
        directional_rows.extend(variant_rows)
        summaries[variant] = variant_summary

    gates = config["development_gates"]
    e_parity = [
        row for row in parity_rows if str(row["variant"]).startswith("candidate_e_")
    ]
    e_projected_summary = summaries[VARIANT_E_PROJECTED]
    diagnostic_gates = {
        "candidate_e_scalar_forward_parity": max(
            float(row["scalar_absolute_error"]) for row in e_parity
        )
        <= float(gates["proxy_exact_forward_absolute_error_max"]),
        "candidate_e_amplitude_forward_parity": max(
            float(row["amplitude_max_absolute_error"]) for row in e_parity
        )
        <= float(gates["proxy_exact_pulse_amplitude_absolute_error_max"]),
        "candidate_e_pair_mask_parity": all(
            bool(row["valid_pair_mask_match"]) for row in e_parity
        ),
        "candidate_e_primary_direction_sign": (
            float(e_projected_summary["direction_sign_match_fraction"])
            >= float(gates["primary_local_direction_sign_match_fraction_min"])
        ),
        "candidate_e_legacy_improvement_fraction": (
            float(e_projected_summary["legacy_alpha_material_improvement_fraction"])
            >= float(gates["legacy_alpha_exact_improvement_fraction_min"])
        ),
        "candidate_e_legacy_median_reduction": (
            float(
                e_projected_summary[
                    "legacy_alpha_material_median_normalized_gap_reduction"
                ]
            )
            >= float(gates["legacy_alpha_median_exact_normalized_gap_reduction_min"])
        ),
        "generator_optimizer_steps_zero": True,
        "formal_generator_training_remains_no_go": True,
    }
    diagnostic_pass = all(diagnostic_gates.values())
    decision = (
        "PASS_CANDIDATE_E_EXACT_PATH_DIRECTIONAL_MECHANISM_V27"
        if diagnostic_pass
        else "NO_GO_CANDIDATE_E_EXACT_PATH_DIRECTIONAL_MECHANISM_V27"
    )

    write_csv(output_dir / "candidate_grid_results.csv", merged_rows)
    write_csv(output_dir / "directional_sign_audit.csv", directional_rows)
    write_csv(output_dir / "pulse_forward_parity.csv", parity_rows)
    report = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-e-diagnostic-v27",
        "decision": decision,
        "scientific_role": "development_calibration_mechanism_diagnosis_only",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_hashes": source_hashes,
        "dataset": {
            "panel_role": "development_calibration",
            "case_count": len(case_ids),
            "speakers": sorted({str(row["speaker_id"]) for row in panel_rows}),
            "opened_v15_accessed": False,
            "external_panel_accessed": False,
            "speaker_split_before_simulation": True,
        },
        "frozen_alpha_grid": list(grid),
        "ablation_alpha_grid": list(ablation_grid),
        "primary_local_alpha": primary_alpha,
        "legacy_operating_alpha": legacy_alpha,
        "case_diagnostics": case_diagnostics,
        "variant_summaries": summaries,
        "pulse_forward_parity": {
            "row_count": len(parity_rows),
            "candidate_e_scalar_error_max": max(
                float(row["scalar_absolute_error"]) for row in e_parity
            ),
            "candidate_e_amplitude_error_max": max(
                float(row["amplitude_max_absolute_error"]) for row in e_parity
            ),
            "candidate_e_contribution_error_max_db": max(
                float(row["contribution_max_absolute_error_db"]) for row in e_parity
            ),
            "candidate_e_pair_masks_all_match": all(
                bool(row["valid_pair_mask_match"]) for row in e_parity
            ),
        },
        "gates": diagnostic_gates,
        "candidate_e_runtime_selector_uses_exact_outcomes": False,
        "exact_outcomes_used_for_development_adjudication_only": True,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "wall_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "candidate_e_directional_diagnostic_report_v27.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-e-diagnostic-receipt-v27",
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "candidate_grid_results_sha256": sha256_file(
            output_dir / "candidate_grid_results.csv"
        ),
        "directional_sign_audit_sha256": sha256_file(
            output_dir / "directional_sign_audit.csv"
        ),
        "pulse_forward_parity_sha256": sha256_file(
            output_dir / "pulse_forward_parity.csv"
        ),
        "exact_results_sha256": sha256_file(exact_output_path),
        "config_sha256": args.config_sha256,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_e_frozen": False,
        "external_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }
    write_json(output_dir / "completion_receipt_v27.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
