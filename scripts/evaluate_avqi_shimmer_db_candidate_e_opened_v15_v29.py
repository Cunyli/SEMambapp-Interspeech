#!/usr/bin/env python3
"""Run the one-time frozen Candidate-E confirmation on opened v15.

Candidate-E source, directions, alpha ladder, selector, and thresholds are
bound to the passing v28 freeze receipt.  The old v15 candidate-result table is
not an input.  This runner materializes projected/raw candidates, refreshes
their own pulse topology, writes the proxy-only selector seal, and only then
opens exact Praat components for the selected PCM24 waveforms.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts import adjudicate_avqi_shimmer_db_candidate_e_v14_v28 as v28
from scripts import adjudicate_avqi_shimmer_db_deterministic_opened24_v23 as v23
from scripts import evaluate_avqi_shimmer_hybrid_topology as hybrid
from scripts.avqi_shimmer_db_candidate_e_proxy_v27 import (
    candidate_e_proxy,
    project_cycle_gain_gradient_fixed_order,
)
from scripts.avqi_shimmer_exact_topology_runtime import (
    NUMPY_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    topology_sha256,
)
from scripts.avqi_shimmer_peak_certificate_v19 import (
    pcm16_roundtrip_values_to_codes,
)
from scripts.diagnose_avqi_shimmer_db_candidate_e_direction_v27 import (
    CANDIDATE_E_VARIANTS,
    VARIANT_E_PROJECTED,
    VARIANT_E_RAW,
    dual_direction_selector_seal,
    impulse_certificate,
    materialize_direction,
    pcm16_roundtrip,
    proxy_evidence,
    pulse_position_drift,
    synchronize,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    load_predictor,
    metric_source_indices_from_topology,
    read_waveform,
    sha256_file,
    topology_stability,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    build_zero_crossing_cycle_plan_vectorized,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    ALPHA_LADDER,
    base_topology_item,
)


REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-opened-v15-confirmation-v29"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-opened-v15-confirmation-receipt-v29"
)
PASS_DECISION = (
    "PASS_CANDIDATE_E_OPENED_V15_CONFIRMATION_EXTERNAL_PREP_AUTHORIZED_V29"
)
FAIL_DECISION = "NO_GO_CANDIDATE_E_OPENED_V15_CONFIRMATION_V29"
V28_PASS_DECISION = "PASS_CANDIDATE_E_V14_FULL_GATE_FROZEN_V28"
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
PANEL_SCHEMA = "avqi-route-c-shimmer-db-candidate-c-fresh-panel-runtime-v15-v1"
TARGET_SCHEMA = "avqi-route-c-shimmer-db-supervised-target-v1"
EXPECTED_CASE_COUNT = 12
EXPECTED_SPEAKER_COUNT = 6
PRIMARY_ALPHA = 0.000125
FORWARD_PARITY_ABSOLUTE_TOLERANCE = 1e-9


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "config",
        "v28-report",
        "v28-receipt",
        "mechanism-config",
        "mechanism-selector",
        "v15-panel-contract",
        "v15-target-contract",
        "predictor-checkpoint",
        "runtime-worker-script",
    ):
        add_hashed_path(parser, option)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return v23.read_json(path)


def write_json(path: Path, value: Any) -> None:
    v23.write_json(path, value)


def repository_provenance(args: argparse.Namespace) -> dict[str, str]:
    root = args.repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain v29 runner")
    head = v23.git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v29 repository HEAD/source commit drift")
    status = v23.git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v29 confirmation requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")


def validate_freeze_chain(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed = {
        "config": v23.validate_hash(
            args.config,
            args.config_sha256,
            "v29 confirmation config",
        ),
        "v28_report": v23.validate_hash(
            args.v28_report,
            args.v28_report_sha256,
            "v28 freeze report",
        ),
        "v28_receipt": v23.validate_hash(
            args.v28_receipt,
            args.v28_receipt_sha256,
            "v28 freeze receipt",
        ),
        "mechanism_config": v23.validate_hash(
            args.mechanism_config,
            args.mechanism_config_sha256,
            "frozen mechanism config",
        ),
        "mechanism_selector": v23.validate_hash(
            args.mechanism_selector,
            args.mechanism_selector_sha256,
            "frozen mechanism selector",
        ),
        "predictor_checkpoint": v23.validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen predictor checkpoint",
        ),
        "runtime_worker": v23.validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "exact topology worker",
        ),
    }
    config = read_json(args.config)
    report = read_json(args.v28_report)
    receipt = read_json(args.v28_receipt)
    mechanism_config = read_json(args.mechanism_config)
    mechanism_selector = read_json(args.mechanism_selector)
    if config.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("v29 config schema drift")
    freeze = config.get("candidate_e_freeze", {})
    expected_freeze = {
        "decision": V28_PASS_DECISION,
        "report_sha256": observed["v28_report"],
        "receipt_sha256": observed["v28_receipt"],
        "mechanism_config_sha256": observed["mechanism_config"],
        "mechanism_selector_sha256": observed["mechanism_selector"],
    }
    for field, expected in expected_freeze.items():
        if freeze.get(field) != expected:
            raise ValueError(f"v29 frozen input drift: {field}")
    if report.get("schema_version") != v28.REPORT_SCHEMA:
        raise ValueError("v28 freeze report schema drift")
    if report.get("decision") != V28_PASS_DECISION:
        raise ValueError("v28 freeze report did not pass")
    if report.get("candidate_e_frozen") is not True:
        raise ValueError("v28 report did not freeze Candidate-E")
    require_training_boundary(report, "v28 report")
    if receipt.get("schema_version") != v28.RECEIPT_SCHEMA:
        raise ValueError("v28 freeze receipt schema drift")
    if receipt.get("decision") != V28_PASS_DECISION:
        raise ValueError("v28 freeze receipt did not pass")
    if receipt.get("report_sha256") != observed["v28_report"]:
        raise ValueError("v28 report/receipt binding drift")
    if receipt.get("candidate_e_frozen") is not True:
        raise ValueError("v28 receipt did not freeze Candidate-E")
    if receipt.get("opened_v15_confirmation_authorized") is not True:
        raise ValueError("v28 receipt did not authorize v15 confirmation")
    if receipt.get("external_panel_authorized") is not False:
        raise ValueError("v28 receipt prematurely authorized external panel")
    require_training_boundary(receipt, "v28 receipt")
    if mechanism_config.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-dual-direction-selector-v27r4"
    ):
        raise ValueError("frozen mechanism config schema drift")
    if mechanism_selector.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-dual-direction-selector-v27r4"
    ):
        raise ValueError("frozen mechanism selector schema drift")
    if mechanism_selector.get("candidate_exact_outcomes_present") is not False:
        raise ValueError("frozen selector contains exact outcomes")
    if mechanism_selector.get(
        "candidate_exact_outcomes_used_for_selection"
    ) is not False:
        raise ValueError("frozen selector used exact outcomes")
    if mechanism_selector.get("speaker_or_case_identity_used_for_routing"):
        raise ValueError("frozen selector used identity")
    require_training_boundary(mechanism_selector, "frozen selector")
    if config["frozen_direction_families"] != list(CANDIDATE_E_VARIANTS):
        raise ValueError("v29 direction-family drift")
    if tuple(config["frozen_directional_grid"]["alpha_ladder"]) != ALPHA_LADDER:
        raise ValueError("v29 alpha-ladder drift")
    if config["immutable_boundaries"].get(
        "opened_v15_outcomes_may_not_trigger_retuning"
    ) is not True:
        raise ValueError("v29 retuning boundary drift")
    return config, receipt, observed


def validate_opened_v15(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    observed = {
        "v15_panel_contract": v23.validate_hash(
            args.v15_panel_contract,
            args.v15_panel_contract_sha256,
            "v15 panel contract",
        ),
        "v15_target_contract": v23.validate_hash(
            args.v15_target_contract,
            args.v15_target_contract_sha256,
            "v15 target contract",
        ),
    }
    panel = read_json(args.v15_panel_contract)
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("v15 panel schema drift")
    if panel.get("speaker_split_before_simulation") is not True:
        raise ValueError("v15 panel was not speaker-split before simulation")
    if panel.get("panel_status") != "sealed_new_speaker_panel_before_exact_outcomes":
        raise ValueError("v15 panel was not sealed before exact outcomes")
    rows = [dict(row) for row in panel.get("rows", [])]
    case_ids = [str(row.get("case_id")) for row in rows]
    speakers = {str(row.get("speaker_id")) for row in rows}
    dataset = config["dataset_contract"]
    if len(rows) != EXPECTED_CASE_COUNT or len(set(case_ids)) != len(rows):
        raise ValueError("v15 case coverage drift")
    if len(speakers) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("v15 speaker coverage drift")
    if sorted(speakers) != sorted(dataset["expected_speakers"]):
        raise ValueError("v15 speaker identity contract drift")
    if speakers & set(dataset["forbidden_development_speakers"]):
        raise ValueError("v15 overlaps v14 development speakers")
    if Counter(str(row["view"]) for row in rows) != Counter({"cs": 6, "sv": 6}):
        raise ValueError("v15 view balance drift")
    if Counter(str(row["sample_group"]) for row in rows) != Counter(
        {"pathological_mild": 6, "pathological_severe": 6}
    ):
        raise ValueError("v15 severity balance drift")
    if Counter(str(row["condition"]) for row in rows) != Counter(
        {"rir_only": 4, "snr20": 4, "snr10": 4}
    ):
        raise ValueError("v15 condition balance drift")
    for speaker in speakers:
        paired = [row for row in rows if str(row["speaker_id"]) == speaker]
        if len(paired) != 2 or {row["view"] for row in paired} != {"cs", "sv"}:
            raise ValueError(f"v15 speaker view pairing drift: {speaker}")
        if len({row["sample_group"] for row in paired}) != 1:
            raise ValueError(f"v15 speaker severity drift: {speaker}")
    for row in rows:
        v23.validate_waveform_hash(row, "base")
        v23.validate_waveform_hash(row, "target")
    target = read_json(args.v15_target_contract)
    if target.get("schema_version") != TARGET_SCHEMA:
        raise ValueError("v15 target schema drift")
    target_boundary = {
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
    }
    for field, expected in target_boundary.items():
        if target.get(field) is not expected:
            raise ValueError(f"v15 target anti-shortcut drift: {field}")
    target_rows = {
        str(row["case_id"]): dict(row) for row in target.get("rows", [])
    }
    if set(target_rows) != set(case_ids):
        raise ValueError("v15 target coverage drift")
    panel_by_case = {str(row["case_id"]): row for row in rows}
    for case_id, target_row in target_rows.items():
        panel_row = panel_by_case[case_id]
        if target_row.get("target_sha256") != panel_row.get("target_sha256"):
            raise ValueError(f"v15 target hash binding drift: {case_id}")
        if target_row.get("speaker_id") != panel_row.get("speaker_id"):
            raise ValueError(f"v15 target speaker binding drift: {case_id}")
        if target_row.get("view") != panel_row.get("view"):
            raise ValueError(f"v15 target view binding drift: {case_id}")
        if not math.isfinite(float(target_row["exact_target_shimmer_db"])):
            raise ValueError(f"v15 target scalar is non-finite: {case_id}")
    return rows, target_rows, observed


def build_candidate_pool(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    target_by_case: dict[str, dict[str, Any]],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    waveform_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    float,
]:
    scale_value = float(target_scale[SHIMMER_DB_INDEX].detach().cpu())
    topology_worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    candidate_rows: list[dict[str, Any]] = []
    case_diagnostics: list[dict[str, Any]] = []
    base_topology_by_case: dict[str, dict[str, Any]] = {}
    try:
        topology_rows, base_topology_wall_ms = topology_worker.refresh(
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
            base_topology_by_case[case_id] = dict(topology)
            base_float = read_waveform(Path(panel_row["base_path"])).to(device)
            source_indices = torch.as_tensor(
                metric_source_indices_from_topology(
                    topology,
                    source_sample_count=base_float.numel(),
                ),
                dtype=torch.long,
                device=device,
            )
            e_waveform = (
                base_float.detach().to(dtype=torch.float64).requires_grad_(True)
            )
            e_pulses = e_waveform.new_tensor(topology["pulse_positions_samples"])
            e_result = candidate_e_proxy(
                e_waveform,
                e_pulses,
                source_indices,
                int(topology["metric_constant_prefix_samples"]),
            )
            target = float(target_by_case[case_id]["exact_target_shimmer_db"])
            e_loss = ((e_result.shimmer_db - target) / scale_value).square()
            e_raw = torch.autograd.grad(e_loss, e_waveform)[0]
            plan = build_zero_crossing_cycle_plan_vectorized(
                base_float.detach().cpu().numpy(),
                topology,
            )
            e_projected, e_projection = project_cycle_gain_gradient_fixed_order(
                e_waveform,
                e_raw,
                plan,
            )
            if not e_projection["projected_gradient_valid"]:
                raise ValueError(f"Candidate-E projection invalid: {case_id}")
            base_highpass_timing = dict(topology["timing_ms"])
            if base_highpass_timing.get("highpass_peak_scaled") is not False:
                raise ValueError(f"Candidate-E base exact highpass scaled: {case_id}")
            base_pcm16_codes = pcm16_roundtrip_values_to_codes(
                pcm16_roundtrip(base_float.detach().cpu().numpy())
            )
            stop_hann_certificate = impulse_certificate(base_float.numel())
            synchronize(device)
            case_diagnostics.append(
                {
                    "case_id": case_id,
                    "speaker_id": panel_row["speaker_id"],
                    "view": panel_row["view"],
                    "condition": panel_row["condition"],
                    "target_shimmer_db": target,
                    "candidate_e_proxy_before": float(
                        e_result.shimmer_db.detach()
                    ),
                    "candidate_e_raw_gradient_l2": float(e_raw.norm().detach()),
                    "candidate_e_projection": e_projection,
                    "candidate_e_peak_scale_abstention_pass": (
                        e_result.peak_scale_abstention_pass
                    ),
                    "candidate_e_fft_sample_count": e_result.fft_sample_count,
                    "base_topology_sha256": topology_sha256(topology),
                }
            )
            directions = {
                VARIANT_E_PROJECTED: e_projected,
                VARIANT_E_RAW: e_raw,
            }
            for variant, direction in directions.items():
                rows, _, _ = materialize_direction(
                    case_id,
                    str(panel_row["view"]),
                    Path(panel_row["base_path"]),
                    variant,
                    (0.0, *ALPHA_LADDER),
                    e_waveform,
                    direction,
                    predictor,
                    e_pulses,
                    source_indices,
                    int(topology["metric_constant_prefix_samples"]),
                    topology,
                    base_pcm16_codes,
                    base_highpass_timing,
                    stop_hann_certificate,
                    PRIMARY_ALPHA,
                    waveform_root,
                )
                candidate_rows.extend(rows)
    finally:
        topology_worker.close()

    current_items = [
        {
            "id": f"current_topology:{row['item_id']}",
            "case_id": row["case_id"],
            "role": "current_output_topology",
            "path": row["candidate_path"],
            "view": row["view"],
            "score_components": False,
            "exact_metric_topology": True,
            "highpass_mode": NUMPY_HIGHPASS_MODE,
        }
        for row in candidate_rows
    ]
    current_rows: list[dict[str, Any]] = []
    current_wall_ms = 0.0
    current_worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    try:
        for start in range(0, len(current_items), 24):
            batch, wall_ms = current_worker.refresh(
                current_items[start : start + 24]
            )
            current_rows.extend(dict(row) for row in batch)
            current_wall_ms += wall_ms
    finally:
        current_worker.close()
    if len(current_rows) != len(candidate_rows):
        raise ValueError("v29 current-topology coverage drift")
    for candidate, current_topology in zip(
        candidate_rows,
        current_rows,
        strict=True,
    ):
        case_id = str(candidate["case_id"])
        waveform = read_waveform(Path(candidate["candidate_path"]))
        waveform = waveform.to(device=device, dtype=torch.float64)
        source_indices = torch.as_tensor(
            metric_source_indices_from_topology(
                current_topology,
                source_sample_count=waveform.numel(),
            ),
            dtype=torch.long,
            device=device,
        )
        pulses = waveform.new_tensor(current_topology["pulse_positions_samples"])
        current_proxy, _ = proxy_evidence(
            str(candidate["variant"]),
            predictor,
            waveform,
            pulses,
            source_indices,
            int(current_topology["metric_constant_prefix_samples"]),
        )
        candidate.update(
            {
                "current_topology_proxy_shimmer_db": current_proxy,
                "current_topology_sha256": topology_sha256(current_topology),
                "current_topology_pulse_count": int(
                    current_topology["pulse_count"]
                ),
                **topology_stability(
                    base_topology_by_case[case_id],
                    current_topology,
                ),
                **pulse_position_drift(
                    base_topology_by_case[case_id],
                    current_topology,
                ),
            }
        )
    return (
        candidate_rows,
        case_diagnostics,
        base_topology_by_case,
        base_topology_wall_ms + current_wall_ms,
    )


def selected_candidates(
    selector: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    case_diagnostics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grid = {
        (
            str(row["case_id"]),
            str(row["variant"]),
            float(row["alpha"]),
        ): row
        for row in candidate_rows
    }
    diagnostics = {
        str(row["case_id"]): row for row in case_diagnostics
    }
    selected_by_case: dict[str, dict[str, Any]] = {}
    for selector_row in selector["rows"]:
        case_id = str(selector_row["case_id"])
        selected = selector_row.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"frozen Candidate-E selector abstained: {case_id}")
        family = str(selected["direction_family"])
        alpha = float(selected["alpha"])
        row = grid[(case_id, family, alpha)]
        zero = grid[(case_id, family, 0.0)]
        diagnostic = diagnostics[case_id]
        if family == VARIANT_E_PROJECTED:
            gradient_norm = float(
                diagnostic["candidate_e_projection"][
                    "projected_gradient_l2_norm"
                ]
            )
        elif family == VARIANT_E_RAW:
            gradient_norm = float(diagnostic["candidate_e_raw_gradient_l2"])
        else:
            raise ValueError(f"unknown frozen direction family: {family}")
        selected_by_case[case_id] = {
            "direction_family": family,
            "alpha": alpha,
            "candidate_path": row["candidate_path"],
            "candidate_sha256": row["candidate_sha256"],
            "proxy_before": float(
                zero["current_topology_proxy_shimmer_db"]
            ),
            "proxy_after": float(row["current_topology_proxy_shimmer_db"]),
            "current_topology_sha256": row["current_topology_sha256"],
            "gradient_l2_norm": gradient_norm,
            "gradient_finite": math.isfinite(gradient_norm)
            and gradient_norm > 0.0,
        }
    return selected_by_case


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    source_provenance = repository_provenance(args)
    config, _, freeze_evidence = validate_freeze_chain(args)
    panel_rows, target_contract, panel_evidence = validate_opened_v15(
        args,
        config,
    )
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    output_dir = args.output_dir.resolve()
    waveform_root = output_dir / "waveforms"
    output_dir.mkdir(parents=True)
    waveform_root.mkdir()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v29 confirmation requires an allocated CUDA device")
    predictor, _, _, target_scale_tensor = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale = target_scale_tensor.detach().cpu().numpy().astype(np.float64)
    started = time.perf_counter()
    candidate_rows, case_diagnostics, _, topology_wall_ms = build_candidate_pool(
        args,
        panel_rows,
        target_contract,
        predictor,
        target_scale_tensor,
        device,
        waveform_root,
    )
    target_scalar = {
        case_id: float(row["exact_target_shimmer_db"])
        for case_id, row in target_contract.items()
    }
    selector = dual_direction_selector_seal(
        candidate_rows,
        target_scalar,
        float(target_scale[SHIMMER_DB_INDEX]),
    )
    selector_path = output_dir / "candidate_e_selector_seal_pre_exact_v29.json"
    attempts_path = output_dir / "candidate_e_attempts_pre_exact_v29.csv"
    write_json(selector_path, selector)
    v23.write_csv(attempts_path, candidate_rows)
    selector_sha256 = sha256_file(selector_path)
    attempts_sha256 = sha256_file(attempts_path)
    selected = selected_candidates(selector, candidate_rows, case_diagnostics)
    if len(selected) != EXPECTED_CASE_COUNT:
        raise ValueError("v29 selected candidate coverage drift")

    exact_items = v28.build_exact_items(panel_rows, selected)
    exact_payload = hybrid.run_exact(
        exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    exact = v23.validate_exact_payload(
        exact_payload,
        [str(item["id"]) for item in exact_items],
    )
    result_rows = v28.build_result_rows(
        panel_rows,
        target_contract,
        selected,
        exact,
        target_scale,
    )
    for row in result_rows:
        row["opened_panel"] = "v15"
        row["opened_role"] = "opened_validation"
    scientific_summary = v23.summarize_effect(
        result_rows,
        EXPECTED_CASE_COUNT,
    )
    maximum_forward_error = max(
        abs(
            float(row["selector_proxy_after_shimmer_db"])
            - float(row["exact_after_shimmer_db"])
        )
        for row in result_rows
    )
    anti_shortcut = {
        "candidate_e_frozen_before_v15_access": True,
        "old_v15_candidate_result_table_not_read": True,
        "selector_written_before_candidate_exact": True,
        "candidate_exact_not_used_for_direction_or_alpha": True,
        "speaker_and_case_identity_not_used_for_routing": True,
        "external_panel_not_accessed": True,
        "opened_v15_outcomes_cannot_trigger_retuning": True,
    }
    gates = {
        "v28_freeze_chain_bound_and_passed": True,
        "v15_contract_complete_and_speaker_disjoint": True,
        "v15_full_scientific_gates": scientific_summary["all_gates_pass"],
        "candidate_e_current_topology_forward_parity": (
            maximum_forward_error <= FORWARD_PARITY_ABSOLUTE_TOLERANCE
        ),
        "anti_shortcut_contract": all(anti_shortcut.values()),
        "generator_optimizer_steps_zero": True,
        "formal_generator_training_remains_no_go": True,
    }
    passed = all(gates.values())
    decision = PASS_DECISION if passed else FAIL_DECISION
    input_sha256 = {
        "freeze": freeze_evidence,
        "v15": panel_evidence,
        "avqi_code_tree": observed_tree_hash,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": decision,
        "phase": "frozen_candidate_e_opened_v15_confirmation",
        "scientific_role": "opened_validation_not_new_final_holdout",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "input_sha256": input_sha256,
        "selector_seal_pre_exact_path": str(selector_path),
        "selector_seal_pre_exact_sha256": selector_sha256,
        "candidate_attempts_pre_exact_sha256": attempts_sha256,
        "case_count": len(result_rows),
        "speaker_count": len({row["speaker_id"] for row in result_rows}),
        "selected_direction_family_counts": {
            family: sum(
                row["selected_family"] == family for row in result_rows
            )
            for family in sorted({row["selected_family"] for row in result_rows})
        },
        "selected_alpha_counts": {
            str(alpha): sum(row["selected_alpha"] == alpha for row in result_rows)
            for alpha in sorted({row["selected_alpha"] for row in result_rows})
        },
        "exact_scorer_versions": {
            "parselmouth": exact_payload["parselmouth_version"],
            "praat": exact_payload["praat_version"],
        },
        "scientific_summary": scientific_summary,
        "maximum_forward_absolute_error_shimmer_db": maximum_forward_error,
        "topology_refresh_wall_ms": topology_wall_ms,
        "anti_shortcut": anti_shortcut,
        "gates": gates,
        "candidate_e_remains_frozen": True,
        "retuning_authorized": False,
        "external_panel_prepare_authorized": passed,
        "external_panel_authorized": False,
        "joint_panel_authorized": False,
        "scientific_promotion_granted": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "wall_seconds": time.perf_counter() - started,
    }
    exact_csv_path = output_dir / "candidate_e_opened_v15_exact_results_v29.csv"
    report_path = output_dir / "candidate_e_opened_v15_report_v29.json"
    v23.write_csv(exact_csv_path, result_rows)
    write_json(report_path, report)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": decision,
        "report_sha256": sha256_file(report_path),
        "exact_results_csv_sha256": sha256_file(exact_csv_path),
        "selector_seal_pre_exact_sha256": selector_sha256,
        "candidate_attempts_pre_exact_sha256": attempts_sha256,
        "input_sha256": input_sha256,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "external_panel_prepare_authorized": passed,
        "external_panel_authorized": False,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    receipt_path = output_dir / "completion_receipt_v29.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": decision,
                "candidate_e_frozen": True,
                "retuning_authorized": False,
                "external_panel_prepare_authorized": passed,
                "external_panel_authorized": False,
                "joint_panel_authorized": False,
                "generator_optimizer_steps": 0,
                "authoritative_training_decision": TRAINING_DECISION,
                "report_sha256": sha256_file(report_path),
                "exact_csv_sha256": sha256_file(exact_csv_path),
                "completion_receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
