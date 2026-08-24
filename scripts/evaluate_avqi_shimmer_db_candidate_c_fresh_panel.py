#!/usr/bin/env python3
"""Run the sealed fresh-panel pilot for Shimmer-dB Candidate C.

Candidate C is a Praat-assisted straight-through metric branch, not a pure
Torch estimator.  Exact Praat refreshes the current S3_500 output topology
once per waveform, Torch differentiates only the live amplitude/Shimmer-dB
path, and exact Praat independently relocates pulses after the candidate
waveforms have been hash-sealed.  The generator is frozen inference only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from model.avqi_components import AVQI_COMPONENT_NAMES
from evaluate_direct_avqi_waveform_optimization import (
    aggregate_denoising,
    aggregate_pathology_guardrails,
    avqi_code_tree_sha256,
    full_band_pathology_guardrails,
)
from evaluate_avqi_shimmer_fresh_panel import (
    component_fields,
    exact_index,
    git_file_sha256,
    read_fixed_recipes,
    recipe_wds_row,
    repository_head,
    run_exact_batch,
    sha256_file,
    validate_file_hash,
    write_csv,
    write_json,
)
from evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RUNTIME_MAX_MS,
    CURRENT_OUTPUT_REFRESH_CANDIDATES,
    FIXED_ALPHA,
    GRADIENT_NORM_RANGE,
    IMPROVEMENT_FRACTION_GATE,
    MATERIAL_GAP_THRESHOLD,
    MAXIMUM_CLIP_FRACTION,
    MEDIAN_REDUCTION_GATE,
    MINIMUM_COSINE,
    NONSELECTED_MEDIAN_INCREASE_GATE,
    REQUIRED_EFFECT_SLICES,
    RESIDUAL_CEILING_DB,
    TOPOLOGY_COUNT_RATIO_DRIFT_MAX,
    TOPOLOGY_MATCH_DROP_MAX,
    TOPOLOGY_MATCH_TOLERANCE_SAMPLES,
    aggregate_candidate,
    load_predictor,
    metric_source_indices_from_topology,
    nearest_match_rate,
    normalized_gradient_step,
    pulse_positions_sha256,
    run_exact,
    waveform_safety,
)
from prepare_avqi_component_expanded_data import (
    WdsReader,
    crop_or_tile,
    match_length,
    read_clean,
    stable_seed,
)


SAMPLE_RATE = 16_000
SHIMMER_DB_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_db")
CANDIDATE_NAME = CURRENT_OUTPUT_REFRESH_CANDIDATES[1]
PANEL_SELECTION_SALT = (
    "shimmer-db-candidate-c-fresh-panel-b1bcb76-20260824"
)
EXPECTED_ELIGIBLE_COUNTS = {
    "pathological_mild": 34,
    "pathological_severe": 33,
}
PREVIOUS_WAVEFORM_PILOT_SPEAKERS = frozenset(
    {
        "PD08",
        "PD_51",
        "SD37",
        "V55",
        "ÄHH05",
        "ÄHH10",
        "ÄHH22",
        "ÄHH25",
        "ÄHH28",
        "ÄHH29",
        "FD26",
        "SD36",
        "FD11",
        "ÄHH20",
        "FD20",
        "SD23",
    }
)

# Frozen before any exact score from these speakers was opened. Each speaker
# contributes both views; condition counts are balanced over severity and view.
PANEL_ROWS = (
    ("SD05", "pathological_mild", "cs", "rir_only", 912),
    ("SD05", "pathological_mild", "sv", "snr20", 913),
    ("SD32", "pathological_mild", "cs", "snr10", 914),
    ("SD32", "pathological_mild", "sv", "rir_only", 915),
    ("ÄHH13", "pathological_mild", "cs", "snr20", 916),
    ("ÄHH13", "pathological_mild", "sv", "snr10", 917),
    ("ÄHH16", "pathological_severe", "cs", "rir_only", 918),
    ("ÄHH16", "pathological_severe", "sv", "snr20", 919),
    ("SD17", "pathological_severe", "cs", "snr10", 920),
    ("SD17", "pathological_severe", "sv", "rir_only", 921),
    ("SD20", "pathological_severe", "cs", "snr20", 922),
    ("SD20", "pathological_severe", "sv", "snr10", 923),
)


@dataclass(frozen=True)
class PanelSpec:
    speaker_id: str
    sample_group: str
    view: str
    condition: str
    recipe_index: int

    @property
    def case_id(self) -> str:
        return (
            f"sealed_final__{self.speaker_id}__{self.view}__{self.condition}"
        )


@dataclass
class PreparedCase:
    spec: PanelSpec
    target_path: Path
    noisy_path: Path
    base_path: Path
    source_path: Path
    recipe: dict[str, Any]
    simulation_seed: int
    noise_start_sample: int


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
    parser.add_argument("--tau-manifest", type=Path, required=True)
    parser.add_argument("--tau-manifest-sha256", required=True)
    parser.add_argument("--fixed-recipes", type=Path, required=True)
    parser.add_argument("--fixed-recipes-sha256", required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--simulation-config-sha256", required=True)
    parser.add_argument("--simulation-source-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def panel_specs() -> tuple[PanelSpec, ...]:
    return tuple(PanelSpec(*row) for row in PANEL_ROWS)


def speaker_selection_rank(
    selection_salt: str,
    severity_group: str,
    speaker_id: str,
) -> str:
    payload = (
        f"{selection_salt}\0{severity_group}\0{speaker_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_panel_specs(specs: tuple[PanelSpec, ...]) -> dict[str, Any]:
    if len(specs) != 12 or len({spec.case_id for spec in specs}) != 12:
        raise ValueError("Candidate-C fresh panel requires twelve unique cases")
    speakers = {spec.speaker_id for spec in specs}
    if len(speakers) != 6:
        raise ValueError("Candidate-C fresh panel requires six speakers")
    if speakers & PREVIOUS_WAVEFORM_PILOT_SPEAKERS:
        raise ValueError("fresh panel overlaps a previous waveform speaker")
    if Counter(spec.speaker_id for spec in specs) != Counter(
        {speaker_id: 2 for speaker_id in speakers}
    ):
        raise ValueError("each fresh speaker must contribute CS and SV")
    if Counter(spec.view for spec in specs) != {"cs": 6, "sv": 6}:
        raise ValueError("fresh panel CS/SV balance drift")
    if Counter(spec.condition for spec in specs) != {
        "rir_only": 4,
        "snr20": 4,
        "snr10": 4,
    }:
        raise ValueError("fresh panel degradation balance drift")
    if Counter(spec.sample_group for spec in specs) != {
        "pathological_mild": 6,
        "pathological_severe": 6,
    }:
        raise ValueError("fresh panel severity balance drift")
    for speaker_id in speakers:
        selected = [spec for spec in specs if spec.speaker_id == speaker_id]
        if {spec.view for spec in selected} != {"cs", "sv"}:
            raise ValueError(f"speaker view coverage drift: {speaker_id}")
        if len({spec.sample_group for spec in selected}) != 1:
            raise ValueError(f"speaker severity drift: {speaker_id}")
    recipe_indices = [spec.recipe_index for spec in specs]
    if recipe_indices != list(range(912, 924)):
        raise ValueError("fresh panel must use frozen recipes 912-923")
    return {
        "case_count": len(specs),
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "previous_waveform_speaker_overlap": [],
        "views": dict(Counter(spec.view for spec in specs)),
        "conditions": dict(Counter(spec.condition for spec in specs)),
        "sample_groups": dict(Counter(spec.sample_group for spec in specs)),
        "recipe_indices": recipe_indices,
    }


def safe_case_name(spec: PanelSpec) -> str:
    return re.sub(r"[^0-9A-Za-z._ÄÖÅäöåÜüÉé_-]", "_", spec.case_id)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_authorization(args: argparse.Namespace) -> dict[str, Any]:
    report_hash = validate_file_hash(
        args.mechanism_report,
        args.mechanism_report_sha256,
        "Candidate-C v13 mechanism report",
    )
    receipt_hash = validate_file_hash(
        args.mechanism_receipt,
        args.mechanism_receipt_sha256,
        "Candidate-C v13 mechanism receipt",
    )
    predictor_hash = validate_file_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "frozen Shimmer v6 checkpoint",
    )
    generator_config_hash = validate_file_hash(
        args.generator_config,
        args.generator_config_sha256,
        "S3_500 generator config",
    )
    generator_checkpoint_hash = validate_file_hash(
        args.generator_checkpoint,
        args.generator_checkpoint_sha256,
        "S3_500 generator checkpoint",
    )
    report = load_json(args.mechanism_report)
    receipt = load_json(args.mechanism_receipt)
    if report.get("schema_version") != (
        "avqi-route-c-shimmer-hybrid-opened-diagnostic-v3"
    ):
        raise ValueError("unexpected Candidate-C mechanism schema")
    if report.get("candidate_c_decision") != (
        "PASS_CURRENT_OUTPUT_EXACT_TOPOLOGY_REFRESH_FREEZE_FOR_FRESH_PANEL"
    ):
        raise ValueError("Candidate C did not authorize a fresh panel")
    if report.get("fresh_panel_authorized") is not True:
        raise ValueError("Candidate-C fresh-panel authorization is absent")
    if report.get("promotion_authorized") is not False:
        raise ValueError("opened mechanism panel cannot authorize promotion")
    if report.get("formal_generator_training_authorized") is not False:
        raise ValueError("mechanism report unexpectedly authorizes training")
    if report.get("authoritative_training_decision") != (
        "NO_GO_AVQI_T2_TRAINING"
    ):
        raise ValueError("mechanism training boundary drift")
    if float(report.get("fixed_alpha", math.nan)) != FIXED_ALPHA:
        raise ValueError("Candidate-C fixed alpha drift")
    candidate = report.get("candidate_c", {})
    if candidate.get("selected_candidate") != CANDIDATE_NAME:
        raise ValueError("Candidate-C selected mechanism drift")
    if float(candidate.get("selected_alpha", math.nan)) != FIXED_ALPHA:
        raise ValueError("Candidate-C selected alpha drift")
    if candidate.get("route_type") != (
        "hybrid_praat_assisted_straight_through_metric_branch"
    ):
        raise ValueError("Candidate-C route disclosure drift")
    if candidate.get("pure_torch_estimator") is not False:
        raise ValueError("Candidate C must not be labeled pure Torch")
    if candidate.get("topology_detached") is not True:
        raise ValueError("Candidate-C topology is not detached")
    if candidate.get("pulse_extractor_called_once_per_waveform_step") is not True:
        raise ValueError("Candidate-C refresh-count contract drift")
    if candidate.get("oracle_alias_at_alpha_0p001", {}).get(
        "proved_equal"
    ) is not True:
        raise ValueError("Candidate-C oracle alias was not proved")
    observed_runtime = float(
        candidate.get("pulse_refresh_internal_runtime_ms", {}).get(
            "maximum",
            math.inf,
        )
    )
    if observed_runtime > CACHE_RUNTIME_MAX_MS:
        raise ValueError("Candidate-C mechanism runtime gate failed")

    frozen_gates = report.get("gates", {})
    expected_gates = {
        "material_gap_threshold": MATERIAL_GAP_THRESHOLD,
        "median_reduction_min": MEDIAN_REDUCTION_GATE,
        "improvement_fraction_min": IMPROVEMENT_FRACTION_GATE,
        "nonselected_median_increase_max": NONSELECTED_MEDIAN_INCREASE_GATE,
        "gradient_l2_range": list(GRADIENT_NORM_RANGE),
        "residual_ceiling_db": RESIDUAL_CEILING_DB,
        "minimum_cosine": MINIMUM_COSINE,
        "maximum_clip_fraction": MAXIMUM_CLIP_FRACTION,
        "topology_match_tolerance_samples": TOPOLOGY_MATCH_TOLERANCE_SAMPLES,
        "topology_match_drop_max": TOPOLOGY_MATCH_DROP_MAX,
        "topology_count_ratio_drift_max": TOPOLOGY_COUNT_RATIO_DRIFT_MAX,
        "pulse_refresh_runtime_max_ms": CACHE_RUNTIME_MAX_MS,
        "required_effect_slices": list(REQUIRED_EFFECT_SLICES),
    }
    if frozen_gates != expected_gates:
        raise ValueError("Candidate-C frozen gate contract drift")

    if receipt.get("report_sha256") != report_hash:
        raise ValueError("Candidate-C receipt does not bind its report")
    if receipt.get("candidate_c_decision") != report["candidate_c_decision"]:
        raise ValueError("Candidate-C receipt decision drift")
    if receipt.get("fresh_panel_authorized") is not True:
        raise ValueError("Candidate-C receipt does not authorize fresh panel")
    if receipt.get("generator_optimizer_steps") != 0:
        raise ValueError("Candidate-C receipt contains generator updates")
    if float(receipt.get("candidate_c_selected_alpha", math.nan)) != FIXED_ALPHA:
        raise ValueError("Candidate-C receipt alpha drift")
    if receipt.get("candidate_c_oracle_alias_proved_equal") is not True:
        raise ValueError("Candidate-C receipt lost oracle equivalence")
    mechanism_commit = str(report["source_commit"])
    if receipt.get("source_commit") != mechanism_commit:
        raise ValueError("Candidate-C report/receipt commit drift")
    for relative_path in (
        "model/avqi_components.py",
        "scripts/evaluate_avqi_shimmer_hybrid_topology.py",
    ):
        frozen_hash = git_file_sha256(mechanism_commit, relative_path)
        live_hash = sha256_file(REPO_ROOT / relative_path)
        if live_hash != frozen_hash:
            raise ValueError(
                f"Candidate-C implementation changed after v13: {relative_path}"
            )
    return {
        "candidate_c_decision": report["candidate_c_decision"],
        "mechanism_report_sha256": report_hash,
        "mechanism_receipt_sha256": receipt_hash,
        "mechanism_source_commit": mechanism_commit,
        "mechanism_job_id": str(report["slurm_job_id"]),
        "candidate": CANDIDATE_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "route_type": candidate["route_type"],
        "pure_torch_estimator": False,
        "predictor_checkpoint_sha256": predictor_hash,
        "generator_config_sha256": generator_config_hash,
        "generator_checkpoint_sha256": generator_checkpoint_hash,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }


def read_tau_manifest(
    path: Path,
    specs: tuple[PanelSpec, ...],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    by_speaker: dict[str, dict[str, str]] = {}
    for row in rows:
        speaker_id = row["speaker_id"]
        if speaker_id in by_speaker:
            raise ValueError(f"duplicate TAU manifest speaker: {speaker_id}")
        by_speaker[speaker_id] = row
    selected_ids = {spec.speaker_id for spec in specs}
    if not selected_ids <= set(by_speaker):
        raise ValueError(
            f"TAU manifest misses fresh speakers: "
            f"{sorted(selected_ids - set(by_speaker))}"
        )
    ranking: dict[str, list[dict[str, str]]] = {}
    for group, expected_count in EXPECTED_ELIGIBLE_COUNTS.items():
        eligible = [
            row["speaker_id"]
            for row in rows
            if row.get("sample_group") == group
            and row.get("label") == "patient"
            and row["speaker_id"] not in PREVIOUS_WAVEFORM_PILOT_SPEAKERS
        ]
        if len(eligible) != expected_count:
            raise ValueError(
                f"eligible speaker count drift for {group}: "
                f"{len(eligible)} != {expected_count}"
            )
        ranked = sorted(
            eligible,
            key=lambda speaker_id: (
                speaker_selection_rank(
                    PANEL_SELECTION_SALT,
                    group,
                    speaker_id,
                ),
                speaker_id,
            ),
        )
        expected_selected = ranked[:3]
        observed_selected = sorted(
            {
                spec.speaker_id
                for spec in specs
                if spec.sample_group == group
            },
            key=lambda speaker_id: ranked.index(speaker_id),
        )
        if observed_selected != expected_selected:
            raise ValueError(
                f"hash-ranked speaker selection drift for {group}: "
                f"{observed_selected} != {expected_selected}"
            )
        ranking[group] = [
            {
                "speaker_id": speaker_id,
                "rank_sha256": speaker_selection_rank(
                    PANEL_SELECTION_SALT,
                    group,
                    speaker_id,
                ),
            }
            for speaker_id in ranked[:10]
        ]
    selected = {speaker_id: by_speaker[speaker_id] for speaker_id in selected_ids}
    for speaker_id, row in selected.items():
        if row.get("label") != "patient":
            raise ValueError(f"fresh speaker is not pathological: {speaker_id}")
        expected_group = {
            spec.sample_group for spec in specs if spec.speaker_id == speaker_id
        }
        if expected_group != {row.get("sample_group")}:
            raise ValueError(f"fresh speaker severity drift: {speaker_id}")
        for view in ("cs", "sv"):
            source = Path(row[f"{view}_audio_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
    return selected, {
        "method": "sha256 rank by frozen salt, severity group, and speaker ID",
        "salt": PANEL_SELECTION_SALT,
        "excluded_previous_waveform_speakers": sorted(
            PREVIOUS_WAVEFORM_PILOT_SPEAKERS
        ),
        "eligible_counts": dict(EXPECTED_ELIGIBLE_COUNTS),
        "ranked_first_ten": ranking,
        "selected": {
            group: [row["speaker_id"] for row in values[:3]]
            for group, values in ranking.items()
        },
        "selection_uses_exact_scores": False,
    }


def read_waveform(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"expected mono 16 kHz waveform: {path}")
    waveform = torch.from_numpy(audio.copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite waveform: {path}")
    return waveform


def exact_components(row: dict[str, Any]) -> np.ndarray:
    values = np.asarray(
        [row["components"][name] for name in AVQI_COMPONENT_NAMES],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite exact components: {row.get('id')}")
    return values


def build_base_topology_items(
    cases: list[PreparedCase],
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"base_topology:{case.spec.case_id}",
            "case_id": case.spec.case_id,
            "role": "current_s3_500_output_topology",
            "path": str(case.base_path.resolve()),
            "view": case.spec.view,
            "score_components": False,
            "exact_metric_topology": True,
        }
        for case in cases
    ]


def topology_stability(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_positions = np.asarray(
        reference["pulse_positions_samples"],
        dtype=np.float64,
    )
    candidate_positions = np.asarray(
        candidate["pulse_positions_samples"],
        dtype=np.float64,
    )
    reference_to_candidate = nearest_match_rate(
        reference_positions,
        candidate_positions,
    )
    candidate_to_reference = nearest_match_rate(
        candidate_positions,
        reference_positions,
    )
    count_ratio = candidate_positions.size / max(reference_positions.size, 1)
    passed = (
        reference_to_candidate >= 1.0 - TOPOLOGY_MATCH_DROP_MAX
        and candidate_to_reference >= 1.0 - TOPOLOGY_MATCH_DROP_MAX
        and abs(count_ratio - 1.0) <= TOPOLOGY_COUNT_RATIO_DRIFT_MAX
    )
    return {
        "reference_to_candidate_match_rate_16_samples": reference_to_candidate,
        "candidate_to_reference_match_rate_16_samples": candidate_to_reference,
        "candidate_reference_pulse_count_ratio": count_ratio,
        "topology_stability_pass": passed,
    }


def candidate_step(
    case: PreparedCase,
    topology: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    candidate_root: Path,
) -> dict[str, Any]:
    if topology.get("scoring_status") != "ok" or topology.get("pulse_count", 0) < 3:
        raise RuntimeError(f"base pulse topology unavailable: {case.spec.case_id}")
    waveform = read_waveform(case.base_path).to(device).requires_grad_(True)
    source_indices_np = metric_source_indices_from_topology(
        topology,
        source_sample_count=waveform.numel(),
    )
    source_indices = torch.as_tensor(
        source_indices_np,
        device=device,
        dtype=torch.long,
    )
    pulses = waveform.new_tensor(topology["pulse_positions_samples"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    proxy_before = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            topology["metric_constant_prefix_samples"]
        ),
    )[1]
    scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    loss = ((proxy_before - target_shimmer_db) / scale).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    candidate = normalized_gradient_step(waveform, gradient, FIXED_ALPHA)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torch_runtime_ms = 1000.0 * (time.perf_counter() - started)
    if not torch.isfinite(gradient).all() or not torch.isfinite(candidate).all():
        raise RuntimeError(f"non-finite Candidate-C step: {case.spec.case_id}")
    if float(candidate.abs().max()) >= 0.999:
        raise RuntimeError(f"Candidate-C step exceeds waveform bound: {case.spec.case_id}")
    output_path = candidate_root / (
        f"{safe_case_name(case.spec)}__candidate_c_alpha_0p001.wav"
    )
    sf.write(
        output_path,
        candidate.detach().cpu().numpy(),
        SAMPLE_RATE,
        subtype="PCM_24",
    )
    stored = read_waveform(output_path).to(device)
    with torch.inference_mode():
        proxy_after = predictor.raw_shimmer_from_pulse_positions(
            stored,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=int(
                topology["metric_constant_prefix_samples"]
            ),
        )[1]
    return {
        "path": output_path,
        "proxy_before": float(proxy_before.detach()),
        "proxy_after_frozen_topology": float(proxy_after.detach()),
        "proxy_target": float(target_shimmer_db),
        "proxy_loss": float(loss.detach()),
        "gradient_l2_norm": float(gradient.norm()),
        "gradient_rms": float(gradient.square().mean().sqrt()),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "torch_step_runtime_ms": torch_runtime_ms,
        "pulse_refresh_runtime_ms": float(topology["pulse_runtime_ms"]),
        "total_metric_step_overhead_ms": (
            float(topology["pulse_runtime_ms"]) + torch_runtime_ms
        ),
        "pulse_topology_sha256": pulse_positions_sha256(
            topology["pulse_positions_samples"]
        ),
        "pulse_count": int(topology["pulse_count"]),
        "metric_sample_count": int(topology["metric_sample_count"]),
        "metric_constant_prefix_samples": int(
            topology["metric_constant_prefix_samples"]
        ),
        "metric_source_range_count": int(
            topology["metric_source_range_count"]
        ),
        "metric_mapped_sample_count": int(
            topology["metric_mapped_sample_count"]
        ),
        "metric_reconstruction_max_pcm16_error": int(
            topology["metric_reconstruction_max_pcm16_error"]
        ),
        "metric_reconstruction_differing_samples": int(
            topology["metric_reconstruction_differing_samples"]
        ),
    }


def summarize_fresh_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism = aggregate_candidate(CANDIDATE_NAME, rows)
    pathology = aggregate_pathology_guardrails(rows)
    denoising = aggregate_denoising(rows)
    fresh_gates = {
        "frozen_mechanism_gates": mechanism["all_gates_pass"],
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
        "target_label_rebound": all(row["target_label_rebound"] for row in rows),
        "base_topology_rebound": all(
            row["base_topology_rebound"] for row in rows
        ),
        "exact_metric_mapping_parity": all(
            row["metric_reconstruction_max_pcm16_error"] == 0
            and row["metric_reconstruction_differing_samples"] == 0
            and row["candidate_metric_reconstruction_max_pcm16_error"] == 0
            and row["candidate_metric_reconstruction_differing_samples"] == 0
            for row in rows
        ),
        "one_unique_topology_refresh_per_case": (
            len(rows) == 12
            and len({row["unique_topology_refresh_key"] for row in rows}) == 12
        ),
        "pulse_coverage": all(
            row["base_output_exact_metric_pulse_count"] >= 3
            and row["candidate_exact_metric_pulse_count"] >= 3
            for row in rows
        ),
        "clean_target_topology_not_used_for_output": all(
            row["clean_target_topology_drives_output"] is False for row in rows
        ),
    }
    return {
        "candidate": CANDIDATE_NAME,
        "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
        "pure_torch_estimator": False,
        "fixed_alpha": FIXED_ALPHA,
        "mechanism": mechanism,
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "fresh_panel_gates": fresh_gates,
        "all_gates_pass": all(fresh_gates.values()),
    }


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    mechanism = summary["mechanism"]
    return "\n".join(
        [
            "# Route C Shimmer dB Candidate-C fresh panel",
            "",
            f"**Decision:** `{report['decision']}`",
            "",
            (
                "Candidate C is a Praat-assisted straight-through metric "
                "branch. It is not a pure-Torch estimator."
            ),
            "",
            f"- Cases: `{mechanism['rows']}`",
            f"- Material cases: `{mechanism['material_rows']}`",
            (
                "- Exact Shimmer-dB improvement fraction: "
                f"`{mechanism['exact_db_improvement_fraction']}`"
            ),
            (
                "- Median normalized Shimmer-dB gap reduction: "
                f"`{mechanism['median_exact_db_normalized_gap_reduction']}`"
            ),
            (
                "- Pulse-refresh median/max ms: "
                f"`{mechanism['pulse_refresh_runtime_ms']['median']}` / "
                f"`{mechanism['pulse_refresh_runtime_ms']['maximum']}`"
            ),
            "",
            (
                "S3_500 was frozen inference only; generator optimizer steps "
                "remain 0. Formal AVQI-T2 training remains blocked."
            ),
            "",
        ]
    )


def write_completion(
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    report_path = output_dir / "fresh_panel_report.json"
    summary_path = output_dir / "SUMMARY.md"
    results_path = output_dir / "fresh_panel_results.csv"
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-c-fresh-receipt-v1",
        "decision": report["decision"],
        "component_status": report["component_status"],
        "route_type": report["route_type"],
        "pure_torch_estimator": False,
        "fixed_alpha": FIXED_ALPHA,
        "speaker_count": 6,
        "waveform_case_count": 12,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "final_exact_panel_opened": True,
        "source_commit": report["source_commit"],
        "slurm_job_id": report["slurm_job_id"],
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
            results_path.name: sha256_file(results_path),
            "panel_contract.json": sha256_file(
                output_dir / "panel_contract.json"
            ),
            "target_label_contract.json": sha256_file(
                output_dir / "target_label_contract.json"
            ),
            "candidate_seal.json": sha256_file(
                output_dir / "candidate_seal.json"
            ),
        },
    }
    write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("fresh-panel source commit differs from repository HEAD")
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    if not args.simulation_root.is_dir():
        raise FileNotFoundError(args.simulation_root)
    if not args.avqi_code_root.is_dir():
        raise FileNotFoundError(args.avqi_code_root)

    authorization = validate_authorization(args)
    source_hashes = {
        "tau_manifest": validate_file_hash(
            args.tau_manifest,
            args.tau_manifest_sha256,
            "TAU sampling manifest",
        ),
        "fixed_test_recipes": validate_file_hash(
            args.fixed_recipes,
            args.fixed_recipes_sha256,
            "fixed test recipes",
        ),
        "simulation_config": validate_file_hash(
            args.simulation_config,
            args.simulation_config_sha256,
            "phone-room simulation config",
        ),
        "simulation_source": validate_file_hash(
            args.simulation_root / "simulate_degradation.py",
            args.simulation_source_sha256,
            "simulation source",
        ),
    }
    observed_avqi_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError(
            "exact AVQI code tree drift: "
            f"{observed_avqi_tree_hash} != {args.avqi_code_tree_sha256}"
        )
    source_hashes["avqi_code_tree"] = observed_avqi_tree_hash

    specs = panel_specs()
    panel_validation = validate_panel_specs(specs)
    tau_rows, selection_contract = read_tau_manifest(args.tau_manifest, specs)
    recipes = read_fixed_recipes(args.fixed_recipes)
    simulation_config = yaml.safe_load(
        args.simulation_config.read_text(encoding="utf-8")
    )
    simulation_config["stft_cfg"]["sampling_rate"] = SAMPLE_RATE

    # These remote-only dependencies are loaded only in the GPU execution path.
    if str(args.simulation_root) not in sys.path:
        sys.path.insert(0, str(args.simulation_root))
    from simulate_degradation import apply_degradation_with_wind
    from evaluate_avqi_component_backprop import enhance_waveform, load_generator
    from utils import load_config

    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    target_root = waveform_root / "target_clean"
    noisy_root = waveform_root / "degraded"
    base_root = waveform_root / "s3_500_base"
    candidate_root = waveform_root / "candidate_c"
    for path in (target_root, noisy_root, base_root, candidate_root):
        path.mkdir(parents=True)

    reader = WdsReader()
    prepared: list[PreparedCase] = []
    try:
        for spec in specs:
            manifest_row = tau_rows[spec.speaker_id]
            source_path = Path(manifest_row[f"{spec.view}_audio_path"])
            clean = read_clean(source_path)
            recipe = recipes[spec.recipe_index]
            if (
                recipe.get("split") != "test"
                or recipe.get("target_sample_rate") != SAMPLE_RATE
            ):
                raise ValueError(
                    f"fixed recipe contract drift at {spec.recipe_index}"
                )
            simulation_seed = stable_seed(
                args.seed,
                "avqi_shimmer_db_candidate_c_fresh_panel_v1",
                PANEL_SELECTION_SALT,
                spec.speaker_id,
                spec.view,
                recipe["uid"],
            )
            rng = random.Random(simulation_seed)
            noise_row = recipe_wds_row(recipe, "noise")
            rir_row = recipe_wds_row(recipe, "rir")
            noise, noise_start = crop_or_tile(
                reader.read(noise_row),
                clean.shape[1],
                rng,
            )
            rir = reader.read(rir_row)
            selected_degradations = ["reverb"]
            snr = None
            if spec.condition.startswith("snr"):
                selected_degradations.append("noise")
                snr = int(spec.condition.removeprefix("snr"))
            degradation_config = {"snr": 20 if snr is None else snr}
            clean_output, degraded = apply_degradation_with_wind(
                simulation_config,
                clean,
                noise,
                rir,
                None,
                degradation_config,
                selected_degradations,
                seed=simulation_seed,
            )
            clean_output = match_length(
                clean_output,
                clean.shape[1],
            ).astype(np.float32)
            degraded = match_length(
                degraded,
                clean.shape[1],
            ).astype(np.float32)
            name = safe_case_name(spec)
            target_path = target_root / f"{name}__target_clean.wav"
            noisy_path = noisy_root / f"{name}__degraded.wav"
            base_path = base_root / f"{name}__s3_500.wav"
            sf.write(target_path, clean_output[0], SAMPLE_RATE, subtype="FLOAT")
            sf.write(noisy_path, degraded[0], SAMPLE_RATE, subtype="FLOAT")
            prepared.append(
                PreparedCase(
                    spec=spec,
                    target_path=target_path,
                    noisy_path=noisy_path,
                    base_path=base_path,
                    source_path=source_path,
                    recipe=recipe,
                    simulation_seed=simulation_seed,
                    noise_start_sample=noise_start,
                )
            )
    finally:
        reader.close()

    device = torch.device(args.device)
    generator_config = load_config(args.generator_config)
    generator = load_generator(
        generator_config,
        args.generator_checkpoint,
        device,
    )
    with torch.inference_mode():
        for index, case in enumerate(prepared, start=1):
            degraded = read_waveform(case.noisy_path).to(device)
            enhanced = enhance_waveform(generator, degraded, generator_config)
            enhanced = enhanced.detach().cpu().reshape(-1)
            if (
                not torch.isfinite(enhanced).all()
                or float(enhanced.abs().max()) >= 1.0
            ):
                raise ValueError(f"invalid S3_500 output: {case.spec.case_id}")
            sf.write(case.base_path, enhanced.numpy(), SAMPLE_RATE, subtype="FLOAT")
            print(f"prepared_base={index}/{len(prepared)}", flush=True)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    panel_rows: list[dict[str, Any]] = []
    for case in prepared:
        recipe = case.recipe
        panel_rows.append(
            {
                "case_id": case.spec.case_id,
                "speaker_id": case.spec.speaker_id,
                "sample_group": case.spec.sample_group,
                "label": "patient",
                "view": case.spec.view,
                "condition": case.spec.condition,
                "recipe_index": case.spec.recipe_index,
                "recipe_uid": recipe["uid"],
                "recipe_seed": recipe["seed"],
                "simulation_seed": case.simulation_seed,
                "source_path": str(case.source_path.resolve()),
                "source_sha256": sha256_file(case.source_path),
                "target_path": str(case.target_path.resolve()),
                "target_sha256": sha256_file(case.target_path),
                "degraded_path": str(case.noisy_path.resolve()),
                "degraded_sha256": sha256_file(case.noisy_path),
                "base_path": str(case.base_path.resolve()),
                "base_sha256": sha256_file(case.base_path),
                "noise_shard_dir": recipe["noise"]["_shard_dir"],
                "noise_shard": recipe["noise"]["shard"],
                "noise_audio_member": recipe["noise"]["audio_member"],
                "noise_start_sample": case.noise_start_sample,
                "rir_shard_dir": recipe["rir"]["_shard_dir"],
                "rir_shard": recipe["rir"]["shard"],
                "rir_audio_member": recipe["rir"]["audio_member"],
            }
        )
    panel_contract = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-c-fresh-panel-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": args.seed,
        "speaker_split_before_simulation": True,
        "panel_status": "sealed_new_speaker_panel_before_exact_outcomes",
        "panel_validation": panel_validation,
        "speaker_selection": selection_contract,
        "generator": {
            "candidate": "S3_500",
            "mode": "frozen_inference_only",
            "optimizer_created": False,
            "optimizer_steps": 0,
            "config": str(args.generator_config.resolve()),
            "config_sha256": args.generator_config_sha256,
            "checkpoint": str(args.generator_checkpoint.resolve()),
            "checkpoint_sha256": args.generator_checkpoint_sha256,
        },
        "candidate_c": {
            "candidate": CANDIDATE_NAME,
            "fixed_alpha": FIXED_ALPHA,
            "alpha_grid": None,
            "selection_or_tuning_on_this_panel": False,
            "route_type": (
                "hybrid_praat_assisted_straight_through_metric_branch"
            ),
            "pure_torch_estimator": False,
            "topology_detached": True,
            "current_output_topology_refreshes_per_case": 1,
            "clean_target_topology_drives_output": False,
            "same_speaker_target_shimmer_db_scalar_is_supervision": True,
        },
        "exact_contract": {
            "target_scalar_is_declared_supervised_input": True,
            "base_and_candidate_exact_outcomes_unopened_until_candidate_seal": True,
            "exact_output_scorer_independently_relocates_pulses": True,
            "exact_praat_recomputes_all_six_components": True,
            "remove_sv_silence_with_sox": False,
        },
        "waveform_contract": {
            "emitted_waveform_highpass": False,
            "avqi_compatible_highpass_metric_branch_only": True,
            "full_band_pathology_guardrails": True,
            "denoising_nonregression": True,
        },
        "authorization": authorization,
        "source_sha256": source_hashes,
        "rows": panel_rows,
    }
    write_json(args.output_dir / "panel_contract.json", panel_contract)

    target_items = [
        {
            "id": f"target:{case.spec.case_id}",
            "path": str(case.target_path.resolve()),
            "view": case.spec.view,
        }
        for case in prepared
    ]
    target_exact = run_exact_batch(
        target_items,
        args.exact_python,
        args.avqi_code_root,
    )
    target_exact_by_id = exact_index(target_exact)
    target_label_contract = {
        "schema_version": "avqi-route-c-shimmer-db-supervised-target-v1",
        "role": "same_speaker_target_scalar_required_by_candidate_loss",
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "exact_scorer_versions": {
            "parselmouth": target_exact["parselmouth_version"],
            "praat": target_exact["praat_version"],
        },
        "rows": [
            {
                "case_id": case.spec.case_id,
                "speaker_id": case.spec.speaker_id,
                "view": case.spec.view,
                "target_sha256": sha256_file(case.target_path),
                "exact_target_shimmer_db": float(
                    target_exact_by_id[f"target:{case.spec.case_id}"][
                        SHIMMER_DB_INDEX
                    ]
                ),
            }
            for case in prepared
        ],
    }
    write_json(
        args.output_dir / "target_label_contract.json",
        target_label_contract,
    )

    topology_items = build_base_topology_items(prepared)
    topology_wall_started = time.perf_counter()
    base_topology_exact = run_exact(
        topology_items,
        args.exact_python,
        args.avqi_code_root,
    )
    topology_batch_wall_ms = 1000.0 * (
        time.perf_counter() - topology_wall_started
    )
    if (
        base_topology_exact["parselmouth_version"]
        != target_exact["parselmouth_version"]
        or base_topology_exact["praat_version"]
        != target_exact["praat_version"]
    ):
        raise ValueError("exact runtime drift before Candidate-C step")
    topology_by_case = {
        row["case_id"]: row for row in base_topology_exact["rows"]
    }
    if set(topology_by_case) != {case.spec.case_id for case in prepared}:
        raise ValueError("base topology coverage drift")

    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    target_label_by_case = {
        row["case_id"]: float(row["exact_target_shimmer_db"])
        for row in target_label_contract["rows"]
    }
    candidate_records: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(prepared, start=1):
        candidate_records[case.spec.case_id] = candidate_step(
            case,
            topology_by_case[case.spec.case_id],
            target_label_by_case[case.spec.case_id],
            predictor,
            target_scale,
            device,
            candidate_root,
        )
        print(f"candidate_step={index}/{len(prepared)}", flush=True)

    candidate_seal = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-c-seal-v1",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate": CANDIDATE_NAME,
        "fixed_alpha": FIXED_ALPHA,
        "selection_or_tuning_on_this_panel": False,
        "target_label_contract_sha256": sha256_file(
            args.output_dir / "target_label_contract.json"
        ),
        "panel_contract_sha256": sha256_file(
            args.output_dir / "panel_contract.json"
        ),
        "exact_base_or_candidate_scoring_started_after_this_seal": True,
        "rows": [
            {
                "case_id": case.spec.case_id,
                "speaker_id": case.spec.speaker_id,
                "view": case.spec.view,
                "condition": case.spec.condition,
                "target_sha256": sha256_file(case.target_path),
                "base_sha256": sha256_file(case.base_path),
                "candidate_path": str(
                    candidate_records[case.spec.case_id]["path"].resolve()
                ),
                "candidate_sha256": sha256_file(
                    candidate_records[case.spec.case_id]["path"]
                ),
                "base_topology_sha256": candidate_records[
                    case.spec.case_id
                ]["pulse_topology_sha256"],
                "unique_topology_refresh_key": (
                    f"metric_base_output:{case.spec.case_id}"
                ),
                "pulse_refresh_runtime_ms": candidate_records[
                    case.spec.case_id
                ]["pulse_refresh_runtime_ms"],
            }
            for case in prepared
        ],
    }
    write_json(args.output_dir / "candidate_seal.json", candidate_seal)

    final_exact_items: list[dict[str, Any]] = []
    for case in prepared:
        candidate_path = candidate_records[case.spec.case_id]["path"]
        for role, path in (
            ("target", case.target_path),
            ("base", case.base_path),
            ("candidate", candidate_path),
        ):
            final_exact_items.append(
                {
                    "id": f"{role}:{case.spec.case_id}",
                    "case_id": case.spec.case_id,
                    "role": role,
                    "path": str(path.resolve()),
                    "view": case.spec.view,
                    "score_components": True,
                    "exact_metric_topology": True,
                }
            )
    final_exact = run_exact(
        final_exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    if (
        final_exact["parselmouth_version"] != target_exact["parselmouth_version"]
        or final_exact["praat_version"] != target_exact["praat_version"]
    ):
        raise ValueError("exact scorer version drift after candidate seal")
    final_by_id = {row["id"]: row for row in final_exact["rows"]}

    result_rows: list[dict[str, Any]] = []
    for case in prepared:
        case_id = case.spec.case_id
        target_row = final_by_id[f"target:{case_id}"]
        base_row = final_by_id[f"base:{case_id}"]
        candidate_row = final_by_id[f"candidate:{case_id}"]
        for row in (target_row, base_row, candidate_row):
            if row.get("scoring_status") != "ok" or row.get("pulse_count", 0) < 3:
                raise RuntimeError(
                    f"sealed exact scoring failed: {row.get('id')} "
                    f"{row.get('error_type')} {row.get('error_message')}"
                )
        target_components = exact_components(target_row)
        base_components = exact_components(base_row)
        candidate_components = exact_components(candidate_row)
        presealed_target = target_exact_by_id[f"target:{case_id}"]
        target_label_rebound = bool(
            np.array_equal(target_components, presealed_target)
        )
        pre_step_topology = topology_by_case[case_id]
        base_topology_rebound = (
            pulse_positions_sha256(base_row["pulse_positions_samples"])
            == pulse_positions_sha256(
                pre_step_topology["pulse_positions_samples"]
            )
        )
        candidate_record = candidate_records[case_id]
        target_waveform = read_waveform(case.target_path)
        base_waveform = read_waveform(case.base_path)
        candidate_waveform = read_waveform(candidate_record["path"])
        row: dict[str, Any] = {
            "case_id": case_id,
            "speaker_id": case.spec.speaker_id,
            "sample_group": case.spec.sample_group,
            "view": case.spec.view,
            "condition": case.spec.condition,
            "candidate": CANDIDATE_NAME,
            "optimized_component": "shimmer_db",
            "fixed_alpha": FIXED_ALPHA,
            "candidate_path": str(candidate_record["path"].resolve()),
            "candidate_sha256": sha256_file(candidate_record["path"]),
            "proxy_before": candidate_record["proxy_before"],
            "proxy_after_frozen_topology": candidate_record[
                "proxy_after_frozen_topology"
            ],
            "proxy_target": candidate_record["proxy_target"],
            "proxy_loss": candidate_record["proxy_loss"],
            "gradient_l2_norm": candidate_record["gradient_l2_norm"],
            "gradient_rms": candidate_record["gradient_rms"],
            "gradient_finite": candidate_record["gradient_finite"],
            "pulse_refresh_runtime_ms": candidate_record[
                "pulse_refresh_runtime_ms"
            ],
            "torch_step_runtime_ms": candidate_record[
                "torch_step_runtime_ms"
            ],
            "total_metric_step_overhead_ms": candidate_record[
                "total_metric_step_overhead_ms"
            ],
            "unique_topology_refresh_key": f"metric_base_output:{case_id}",
            "pulse_topology_sha256": candidate_record[
                "pulse_topology_sha256"
            ],
            "base_output_exact_metric_pulse_count": candidate_record[
                "pulse_count"
            ],
            "candidate_exact_metric_pulse_count": int(
                candidate_row["pulse_count"]
            ),
            "metric_sample_count": candidate_record["metric_sample_count"],
            "metric_constant_prefix_samples": candidate_record[
                "metric_constant_prefix_samples"
            ],
            "metric_source_range_count": candidate_record[
                "metric_source_range_count"
            ],
            "metric_mapped_sample_count": candidate_record[
                "metric_mapped_sample_count"
            ],
            "metric_reconstruction_max_pcm16_error": candidate_record[
                "metric_reconstruction_max_pcm16_error"
            ],
            "metric_reconstruction_differing_samples": candidate_record[
                "metric_reconstruction_differing_samples"
            ],
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
            "base_topology_rebound": base_topology_rebound,
            "clean_target_topology_drives_output": False,
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
            > MATERIAL_GAP_THRESHOLD
        )
        row["forward_normalized_abs_error_shimmer_db"] = (
            abs(
                row["proxy_before"]
                - row["exact_before_shimmer_db"]
            )
            / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        )
        row.update(topology_stability(pre_step_topology, candidate_row))
        row.update(waveform_safety(base_waveform.numpy(), candidate_waveform.numpy()))
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        result_rows.append(row)

    write_csv(args.output_dir / "fresh_panel_results.csv", result_rows)
    summary = summarize_fresh_panel(result_rows)
    passed = summary["all_gates_pass"]
    component_status = (
        "PASS_SHIMMER_DB_PRAAT_ASSISTED_ROUTE_C_COMPONENT_AND_BOUNDED_PILOT"
        if passed
        else "NO_GO_SHIMMER_DB_CANDIDATE_C_FRESH_PANEL"
    )
    report = {
        "schema_version": "avqi-route-c-shimmer-db-candidate-c-fresh-v1",
        "decision": component_status,
        "component_status": component_status,
        "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
        "pure_torch_estimator": False,
        "fresh_panel_status": "opened_after_candidate_seal",
        "fresh_waveform_speakers_not_scorer_domain_external": True,
        "fixed_alpha": FIXED_ALPHA,
        "alpha_selected_or_tuned_on_this_panel": False,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "device": str(device),
        "speaker_count": 6,
        "case_count": 12,
        "generator_loaded_for_frozen_inference": True,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "formal_pathology_training_submitted": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "training_boundary_reason": (
            "A passing isolated bounded-waveform panel only makes the disclosed "
            "Praat-assisted Shimmer-dB component eligible for integration and "
            "a later joint panel; it cannot authorize formal generator training."
        ),
        "exact_scorer_versions": {
            "parselmouth": final_exact["parselmouth_version"],
            "praat": final_exact["praat_version"],
        },
        "topology_refresh": {
            "unique_refresh_calls": len(topology_items),
            "alpha_candidate_count": 1,
            "topology_reused_across_alpha_candidates": False,
            "batch_wall_ms": topology_batch_wall_ms,
            "amortized_wall_ms_per_unique_waveform": (
                topology_batch_wall_ms / len(topology_items)
            ),
            "internal_runtime_ms": {
                "median": median(
                    row["pulse_runtime_ms"]
                    for row in base_topology_exact["rows"]
                ),
                "maximum": max(
                    row["pulse_runtime_ms"]
                    for row in base_topology_exact["rows"]
                ),
                "frozen_gate_maximum": CACHE_RUNTIME_MAX_MS,
            },
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
    write_completion(args.output_dir, report)


if __name__ == "__main__":
    main()
