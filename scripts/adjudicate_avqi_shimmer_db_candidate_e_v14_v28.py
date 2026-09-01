#!/usr/bin/env python3
"""Run the full v14 component and safety gate for sealed Candidate-E.

The script consumes a passing v27r4 mechanism receipt and its pre-exact
selector seal.  It never re-selects a candidate.  Exact Praat then scores the
same-speaker clean pathological target, frozen S3_500 output, and sealed
Candidate-E PCM24 waveform for all six AVQI components.  A PASS freezes the
Candidate-E source/config/selector and authorizes one opened-v15 confirmation;
it does not authorize an external panel, a joint panel, or generator training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from scripts import adjudicate_avqi_shimmer_db_deterministic_opened24_v23 as v23
from scripts import evaluate_avqi_shimmer_fresh_panel as fresh
from scripts import evaluate_avqi_shimmer_hybrid_topology as hybrid
from scripts import evaluate_direct_avqi_waveform_optimization as direct
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    SHIMMER_DB_INDEX,
    topology_stability,
)
from scripts.evaluate_avqi_shimmer_db_cycle_projected_backward import (
    validate_panel,
)


REPORT_SCHEMA = "avqi-route-c-shimmer-db-candidate-e-v14-full-gate-v28"
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-v14-full-gate-receipt-v28"
)
PASS_DECISION = "PASS_CANDIDATE_E_V14_FULL_GATE_FROZEN_V28"
FAIL_DECISION = "NO_GO_CANDIDATE_E_V14_FULL_GATE_V28"
MECHANISM_PASS_DECISION = (
    "PASS_CANDIDATE_E_DUAL_DIRECTION_MECHANISM_V27R4"
)
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
EXPECTED_CASE_COUNT = 12
EXPECTED_SPEAKER_COUNT = 6
TARGET_CONTRACT_SCHEMA = "avqi-route-c-shimmer-db-supervised-target-v1"
FORWARD_PARITY_ABSOLUTE_TOLERANCE = 1e-9


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "mechanism-config",
        "mechanism-report",
        "mechanism-receipt",
        "selector-seal",
        "candidate-grid-results",
        "selector-exact-adjudication",
        "v14-panel-contract",
        "v14-target-contract",
        "v14-fresh-results",
        "predictor-checkpoint",
    ):
        add_hashed_path(parser, option)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def repository_provenance(
    repository_root: Path,
    source_commit: str,
) -> dict[str, str]:
    root = repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain the v28 adjudicator")
    observed_head = v23.git_output(root, "rev-parse", "HEAD")
    if observed_head != source_commit:
        raise ValueError("v28 repository HEAD/source commit drift")
    status = v23.git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v28 full gate requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": observed_head,
        "adjudicator_sha256": v23.sha256_file(Path(__file__).resolve()),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")


def validate_mechanism_chain(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
]:
    mechanism_config = v23.read_json(args.mechanism_config)
    mechanism_report = v23.read_json(args.mechanism_report)
    mechanism_receipt = v23.read_json(args.mechanism_receipt)
    selector = v23.read_json(args.selector_seal)
    candidate_grid = v23.read_csv(args.candidate_grid_results)
    selector_exact = v23.read_csv(args.selector_exact_adjudication)
    observed = {
        "mechanism_config": v23.validate_hash(
            args.mechanism_config,
            args.mechanism_config_sha256,
            "v27r4 mechanism config",
        ),
        "mechanism_report": v23.validate_hash(
            args.mechanism_report,
            args.mechanism_report_sha256,
            "v27r4 mechanism report",
        ),
        "mechanism_receipt": v23.validate_hash(
            args.mechanism_receipt,
            args.mechanism_receipt_sha256,
            "v27r4 mechanism receipt",
        ),
        "selector_seal": v23.validate_hash(
            args.selector_seal,
            args.selector_seal_sha256,
            "v27r4 pre-exact selector seal",
        ),
        "candidate_grid_results": v23.validate_hash(
            args.candidate_grid_results,
            args.candidate_grid_results_sha256,
            "v27r4 candidate grid",
        ),
        "selector_exact_adjudication": v23.validate_hash(
            args.selector_exact_adjudication,
            args.selector_exact_adjudication_sha256,
            "v27r4 selector exact adjudication",
        ),
    }
    if mechanism_config.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-dual-direction-selector-v27r4"
    ):
        raise ValueError("v27r4 mechanism config schema drift")
    if mechanism_config["dataset_contract"].get(
        "opened_v15_access_authorized"
    ) is not False:
        raise ValueError("v27r4 mechanism config accessed opened v15")
    if mechanism_config["dataset_contract"].get(
        "external_panel_access_authorized"
    ) is not False:
        raise ValueError("v27r4 mechanism config accessed external data")
    if mechanism_report.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-diagnostic-v27r4"
    ):
        raise ValueError("v27r4 mechanism report schema drift")
    if mechanism_report.get("decision") != MECHANISM_PASS_DECISION:
        raise ValueError("v27r4 mechanism did not pass")
    if mechanism_report.get("source_commit") != mechanism_receipt.get(
        "source_commit"
    ):
        raise ValueError("v27r4 report/receipt source commit drift")
    if mechanism_report.get("candidate_e_runtime_selector_uses_exact_outcomes"):
        raise ValueError("v27r4 selector used exact outcomes")
    if mechanism_report.get(
        "candidate_e_selector_sealed_before_exact_scoring"
    ) is not True:
        raise ValueError("v27r4 selector was not sealed before exact scoring")
    dataset = mechanism_report.get("dataset", {})
    if dataset.get("opened_v15_accessed") is not False:
        raise ValueError("v27r4 mechanism report accessed v15")
    if dataset.get("external_panel_accessed") is not False:
        raise ValueError("v27r4 mechanism report accessed external data")
    require_training_boundary(mechanism_report, "v27r4 mechanism report")
    if mechanism_receipt.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-diagnostic-receipt-v27r4"
    ):
        raise ValueError("v27r4 mechanism receipt schema drift")
    if mechanism_receipt.get("decision") != MECHANISM_PASS_DECISION:
        raise ValueError("v27r4 mechanism receipt did not pass")
    if mechanism_receipt.get("report_sha256") != observed["mechanism_report"]:
        raise ValueError("v27r4 report/receipt binding drift")
    if mechanism_report.get("selector_seal_pre_exact_sha256") != observed[
        "selector_seal"
    ]:
        raise ValueError("v27r4 report/selector binding drift")
    receipt_bindings = {
        "config_sha256": observed["mechanism_config"],
        "selector_seal_pre_exact_sha256": observed["selector_seal"],
        "candidate_grid_results_sha256": observed["candidate_grid_results"],
        "selector_exact_adjudication_sha256": observed[
            "selector_exact_adjudication"
        ],
    }
    for field, expected in receipt_bindings.items():
        if mechanism_receipt.get(field) != expected:
            raise ValueError(f"v27r4 receipt binding drift: {field}")
    if mechanism_receipt.get("candidate_e_frozen") is not False:
        raise ValueError("v27r4 mechanism prematurely froze Candidate-E")
    require_training_boundary(mechanism_receipt, "v27r4 mechanism receipt")
    if selector.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-dual-direction-selector-v27r4"
    ):
        raise ValueError("v27r4 selector schema drift")
    if selector.get("candidate_exact_outcomes_present") is not False:
        raise ValueError("v27r4 selector contains exact outcomes")
    if selector.get("candidate_exact_outcomes_used_for_selection") is not False:
        raise ValueError("v27r4 selector routes on exact outcomes")
    if selector.get("speaker_or_case_identity_used_for_routing") is not False:
        raise ValueError("v27r4 selector routes on identity")
    require_training_boundary(selector, "v27r4 selector")
    return mechanism_report, selector, candidate_grid, {
        "selector_exact_rows": selector_exact,
        "input_sha256": observed,
    }


def validate_v14_contracts(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    observed = {
        "v14_panel_contract": v23.validate_hash(
            args.v14_panel_contract,
            args.v14_panel_contract_sha256,
            "v14 panel contract",
        ),
        "v14_target_contract": v23.validate_hash(
            args.v14_target_contract,
            args.v14_target_contract_sha256,
            "v14 target contract",
        ),
        "v14_fresh_results": v23.validate_hash(
            args.v14_fresh_results,
            args.v14_fresh_results_sha256,
            "v14 fresh results",
        ),
    }
    panel = v23.read_json(args.v14_panel_contract)
    fresh_rows = v23.read_csv(args.v14_fresh_results)
    panel_rows, fresh_by_case = validate_panel(panel, fresh_rows)
    if len(panel_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("v14 case count drift")
    if len({str(row["speaker_id"]) for row in panel_rows}) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("v14 speaker count drift")
    if panel.get("speaker_split_before_simulation") is not True:
        raise ValueError("v14 speaker split was not sealed before simulation")
    target = v23.read_json(args.v14_target_contract)
    if target.get("schema_version") != TARGET_CONTRACT_SCHEMA:
        raise ValueError("v14 target contract schema drift")
    target_boundary = {
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
    }
    for field, expected in target_boundary.items():
        if target.get(field) is not expected:
            raise ValueError(f"v14 target anti-shortcut drift: {field}")
    target_by_case = {
        str(row["case_id"]): dict(row) for row in target.get("rows", [])
    }
    case_ids = {str(row["case_id"]) for row in panel_rows}
    if set(target_by_case) != case_ids:
        raise ValueError("v14 target contract coverage drift")
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        target_row = target_by_case[case_id]
        if target_row.get("target_sha256") != panel_row["target_sha256"]:
            raise ValueError(f"v14 target waveform binding drift: {case_id}")
        if str(target_row.get("speaker_id")) != str(panel_row["speaker_id"]):
            raise ValueError(f"v14 target speaker binding drift: {case_id}")
        if str(target_row.get("view")) != str(panel_row["view"]):
            raise ValueError(f"v14 target view binding drift: {case_id}")
        if not math.isclose(
            float(target_row["exact_target_shimmer_db"]),
            float(fresh_by_case[case_id]["exact_target_shimmer_db"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"v14 target scalar binding drift: {case_id}")
    return panel_rows, fresh_by_case, target_by_case, observed


def sealed_selection(
    panel_rows: list[dict[str, Any]],
    mechanism_report: dict[str, Any],
    selector: dict[str, Any],
    candidate_grid: list[dict[str, str]],
    selector_exact_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    case_ids = {str(row["case_id"]) for row in panel_rows}
    selector_rows = {
        str(row["case_id"]): dict(row) for row in selector.get("rows", [])
    }
    if set(selector_rows) != case_ids:
        raise ValueError("v27r4 selector/v14 case coverage drift")
    grid = {
        (
            str(row["case_id"]),
            str(row["variant"]),
            float(row["alpha"]),
        ): dict(row)
        for row in candidate_grid
    }
    exact = {
        str(row["case_id"]): dict(row) for row in selector_exact_rows
    }
    if set(exact) != case_ids:
        raise ValueError("v27r4 selector exact coverage drift")
    diagnostic = {
        str(row["case_id"]): dict(row)
        for row in mechanism_report.get("case_diagnostics", [])
    }
    if set(diagnostic) != case_ids:
        raise ValueError("v27r4 diagnostic coverage drift")
    selected_by_case: dict[str, dict[str, Any]] = {}
    for case_id in sorted(case_ids):
        selector_row = selector_rows[case_id]
        if selector_row.get("case_id_used_for_routing") is not False:
            raise ValueError(f"selector identity routing drift: {case_id}")
        selected = selector_row.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"v27r4 selector abstained: {case_id}")
        if selected.get("exact_candidate_outcome_present") is not False:
            raise ValueError(f"selector attempt contains exact outcome: {case_id}")
        family = str(selected["direction_family"])
        alpha = float(selected["alpha"])
        grid_row = grid.get((case_id, family, alpha))
        zero_row = grid.get((case_id, family, 0.0))
        if grid_row is None or zero_row is None:
            raise ValueError(f"v27r4 selected grid binding drift: {case_id}")
        candidate_path = Path(str(selected["candidate_path"]))
        candidate_sha256 = v23.sha256_file(candidate_path)
        if candidate_sha256 != selected["candidate_sha256"]:
            raise ValueError(f"sealed candidate hash drift: {case_id}")
        if candidate_sha256 != grid_row["candidate_sha256"]:
            raise ValueError(f"selector/grid candidate hash drift: {case_id}")
        exact_row = exact[case_id]
        if exact_row.get("selected_candidate_present") != "True":
            raise ValueError(f"v27r4 exact row lacks selection: {case_id}")
        if exact_row.get("selected_candidate_sha256") != candidate_sha256:
            raise ValueError(f"selector/exact candidate hash drift: {case_id}")
        if exact_row.get("selected_direction_family") != family:
            raise ValueError(f"selector/exact family drift: {case_id}")
        if float(exact_row["selected_alpha"]) != alpha:
            raise ValueError(f"selector/exact alpha drift: {case_id}")
        if exact_row.get("exact_improves") != "True":
            raise ValueError(f"v27r4 selected exact outcome regressed: {case_id}")
        case_diagnostic = diagnostic[case_id]
        if family == "candidate_e_exact_path_projected":
            gradient_norm = float(
                case_diagnostic["candidate_e_projection"][
                    "projected_gradient_l2_norm"
                ]
            )
        elif family == "candidate_e_exact_path_raw_ablation":
            gradient_norm = float(
                case_diagnostic["candidate_e_raw_gradient_l2"]
            )
        else:
            raise ValueError(f"unknown Candidate-E direction family: {family}")
        selected_by_case[case_id] = {
            "direction_family": family,
            "alpha": alpha,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": candidate_sha256,
            "proxy_before": float(
                zero_row["current_topology_proxy_shimmer_db"]
            ),
            "proxy_after": float(
                grid_row["current_topology_proxy_shimmer_db"]
            ),
            "current_topology_sha256": str(
                selected["current_topology_sha256"]
            ),
            "gradient_l2_norm": gradient_norm,
            "gradient_finite": math.isfinite(gradient_norm)
            and gradient_norm > 0.0,
        }
    return selected_by_case


def build_exact_items(
    panel_rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in panel_rows:
        case_id = str(row["case_id"])
        common = {
            "case_id": case_id,
            "view": row["view"],
            "score_components": True,
        }
        items.extend(
            [
                {
                    **common,
                    "id": f"target:{case_id}",
                    "role": "same_speaker_clean_pathological_target",
                    "path": str(Path(row["target_path"]).resolve()),
                    "exact_metric_topology": False,
                },
                {
                    **common,
                    "id": f"base:{case_id}",
                    "role": "frozen_s3_500_output_before_step",
                    "path": str(Path(row["base_path"]).resolve()),
                    "exact_metric_topology": True,
                },
                {
                    **common,
                    "id": f"candidate:{case_id}",
                    "role": "sealed_candidate_e_after_step",
                    "path": selected[case_id]["candidate_path"],
                    "exact_metric_topology": True,
                },
            ]
        )
    return items


def build_result_rows(
    panel_rows: list[dict[str, Any]],
    target_contract: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    exact: dict[str, dict[str, Any]],
    target_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        target_exact = exact[f"target:{case_id}"]
        base_exact = exact[f"base:{case_id}"]
        candidate_exact = exact[f"candidate:{case_id}"]
        target = np.asarray(
            [target_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        base = np.asarray(
            [base_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [candidate_exact["components"][name] for name in AVQI_COMPONENT_NAMES],
            dtype=np.float64,
        )
        frozen_target = float(
            target_contract[case_id]["exact_target_shimmer_db"]
        )
        selection = selected[case_id]
        row: dict[str, Any] = {
            "case_id": case_id,
            "opened_panel": "v14",
            "opened_role": "development_calibration",
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": "candidate_e_v27r4_pre_exact_selector_selected_pcm24",
            "optimized_component": "shimmer_db",
            "selected_family": selection["direction_family"],
            "selected_alpha": selection["alpha"],
            "candidate_path": selection["candidate_path"],
            "candidate_sha256": selection["candidate_sha256"],
            "frozen_gradient_l2_norm": selection["gradient_l2_norm"],
            "frozen_gradient_finite": selection["gradient_finite"],
            "frozen_target_shimmer_db": frozen_target,
            "rescored_target_shimmer_db": float(target[SHIMMER_DB_INDEX]),
            "target_reproduction_abs_error_shimmer_db": abs(
                float(target[SHIMMER_DB_INDEX]) - frozen_target
            ),
            "target_reproduction_pass": math.isclose(
                float(target[SHIMMER_DB_INDEX]),
                frozen_target,
                rel_tol=0.0,
                abs_tol=v23.TARGET_REPRODUCTION_ABS_TOLERANCE,
            ),
            "selector_proxy_before_shimmer_db": selection["proxy_before"],
            "selector_proxy_after_shimmer_db": selection["proxy_after"],
            "selected_current_topology_sha256": selection[
                "current_topology_sha256"
            ],
            "candidate_exact_pulse_count": int(candidate_exact["pulse_count"]),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                candidate_exact["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                candidate_exact["metric_reconstruction_differing_samples"]
            ),
            "base_metric_reconstruction_max_pcm16_error": int(
                base_exact["metric_reconstruction_max_pcm16_error"]
            ),
            "base_metric_reconstruction_differing_samples": int(
                base_exact["metric_reconstruction_differing_samples"]
            ),
            "clean_target_topology_drives_output": False,
            "candidate_exact_opened_only_after_selector_seal": True,
        }
        fresh.component_fields(row, target, base, candidate, target_scale)
        row["material_shimmer_db_gap"] = (
            row["exact_absolute_gap_before_shimmer_db"]
            / max(float(target_scale[SHIMMER_DB_INDEX]), 1e-8)
            > hybrid.MATERIAL_GAP_THRESHOLD
        )
        row["forward_normalized_abs_error_shimmer_db"] = (
            abs(
                row["selector_proxy_after_shimmer_db"]
                - row["exact_after_shimmer_db"]
            )
            / max(float(target_scale[SHIMMER_DB_INDEX]), 1e-8)
        )
        row.update(topology_stability(base_exact, candidate_exact))
        base_waveform = v23.read_waveform(Path(panel_row["base_path"]))
        candidate_waveform = v23.read_waveform(Path(selection["candidate_path"]))
        target_waveform = v23.read_waveform(Path(panel_row["target_path"]))
        row.update(
            hybrid.waveform_safety(
                base_waveform.numpy(),
                candidate_waveform.numpy(),
            )
        )
        row.update(
            direct.full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)
    return rows


def write_receipt(
    args: argparse.Namespace,
    report: dict[str, Any],
    report_path: Path,
    exact_csv_path: Path,
    input_sha256: dict[str, Any],
) -> Path:
    passed = report["decision"] == PASS_DECISION
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": report["decision"],
        "report_sha256": v23.sha256_file(report_path),
        "exact_results_csv_sha256": v23.sha256_file(exact_csv_path),
        "input_sha256": input_sha256,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_e_frozen": passed,
        "opened_v15_confirmation_authorized": passed,
        "external_panel_authorized": False,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    path = args.output_dir / "completion_receipt_v28.json"
    v23.write_json(path, receipt)
    return path


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if not args.exact_python.is_file():
        raise FileNotFoundError(f"missing exact Python: {args.exact_python}")
    source_provenance = repository_provenance(
        args.repository_root,
        args.source_commit,
    )
    observed_tree_hash = direct.avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    mechanism_report, selector, candidate_grid, mechanism_evidence = (
        validate_mechanism_chain(args)
    )
    panel_rows, _, target_contract, v14_evidence = validate_v14_contracts(args)
    selected = sealed_selection(
        panel_rows,
        mechanism_report,
        selector,
        candidate_grid,
        mechanism_evidence["selector_exact_rows"],
    )
    predictor_hash = v23.validate_hash(
        args.predictor_checkpoint,
        args.predictor_checkpoint_sha256,
        "frozen AVQI predictor checkpoint",
    )
    _, _, _, target_scale_tensor = hybrid.load_predictor(
        args.predictor_checkpoint,
        torch.device("cpu"),
    )
    target_scale = target_scale_tensor.detach().cpu().numpy().astype(np.float64)
    input_sha256 = {
        "mechanism": mechanism_evidence["input_sha256"],
        "v14": v14_evidence,
        "predictor_checkpoint": predictor_hash,
        "avqi_code_tree": observed_tree_hash,
    }
    args.output_dir.mkdir(parents=True)
    exact_items = build_exact_items(panel_rows, selected)
    exact_payload = hybrid.run_exact(
        exact_items,
        args.exact_python,
        args.avqi_code_root,
    )
    exact = v23.validate_exact_payload(
        exact_payload,
        [str(item["id"]) for item in exact_items],
    )
    result_rows = build_result_rows(
        panel_rows,
        target_contract,
        selected,
        exact,
        target_scale,
    )
    scientific_summary = v23.summarize_effect(
        result_rows,
        EXPECTED_CASE_COUNT,
    )
    maximum_forward_error = max(
        float(row["forward_normalized_abs_error_shimmer_db"])
        for row in result_rows
    )
    anti_shortcut = {
        "selector_sealed_before_candidate_exact": True,
        "candidate_exact_not_used_for_direction_or_alpha": True,
        "same_speaker_clean_pathological_target": True,
        "clean_target_topology_not_used_by_output_branch": True,
        "opened_v15_not_accessed": True,
        "external_panel_not_accessed": True,
        "old_candidate_d_v23_no_go_preserved": True,
    }
    gates = {
        "v27r4_mechanism_chain_bound_and_passed": True,
        "v14_contract_complete": len(result_rows) == EXPECTED_CASE_COUNT,
        "v14_full_scientific_gates": scientific_summary["all_gates_pass"],
        "candidate_e_current_topology_forward_parity": (
            maximum_forward_error
            <= FORWARD_PARITY_ABSOLUTE_TOLERANCE
            / max(float(target_scale[SHIMMER_DB_INDEX]), 1e-8)
        ),
        "anti_shortcut_contract": all(anti_shortcut.values()),
        "generator_optimizer_steps_zero": True,
        "formal_generator_training_remains_no_go": True,
    }
    passed = all(gates.values())
    decision = PASS_DECISION if passed else FAIL_DECISION
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": decision,
        "phase": "candidate_e_v14_full_component_and_safety_gate",
        "scientific_role": "development_calibration_freeze_gate",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "input_sha256": input_sha256,
        "exact_scorer_versions": {
            "parselmouth": exact_payload["parselmouth_version"],
            "praat": exact_payload["praat_version"],
        },
        "case_count": len(result_rows),
        "speaker_count": len({row["speaker_id"] for row in result_rows}),
        "selected_direction_family_counts": {
            family: sum(
                row["selected_family"] == family for row in result_rows
            )
            for family in sorted({row["selected_family"] for row in result_rows})
        },
        "fixed_scientific_thresholds": {
            "material_normalized_gap": hybrid.MATERIAL_GAP_THRESHOLD,
            "median_normalized_reduction": hybrid.MEDIAN_REDUCTION_GATE,
            "improvement_fraction": hybrid.IMPROVEMENT_FRACTION_GATE,
            "nonselected_median_increase": hybrid.NONSELECTED_MEDIAN_INCREASE_GATE,
            "gradient_l2_range": list(hybrid.GRADIENT_NORM_RANGE),
            "residual_ceiling_db": hybrid.RESIDUAL_CEILING_DB,
            "minimum_cosine": hybrid.MINIMUM_COSINE,
            "maximum_clip_fraction": hybrid.MAXIMUM_CLIP_FRACTION,
            "target_reproduction_abs_tolerance": (
                v23.TARGET_REPRODUCTION_ABS_TOLERANCE
            ),
            "candidate_e_forward_absolute_tolerance": (
                FORWARD_PARITY_ABSOLUTE_TOLERANCE
            ),
        },
        "scientific_summary": scientific_summary,
        "maximum_forward_normalized_abs_error_shimmer_db": (
            maximum_forward_error
        ),
        "anti_shortcut": anti_shortcut,
        "gates": gates,
        "candidate_e_frozen": passed,
        "opened_v15_confirmation_authorized": passed,
        "external_panel_authorized": False,
        "joint_panel_authorized": False,
        "scientific_promotion_granted": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    exact_csv_path = args.output_dir / "candidate_e_v14_exact_results_v28.csv"
    report_path = args.output_dir / "candidate_e_v14_full_gate_report_v28.json"
    v23.write_csv(exact_csv_path, result_rows)
    v23.write_json(report_path, report)
    receipt_path = write_receipt(
        args,
        report,
        report_path,
        exact_csv_path,
        input_sha256,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "candidate_e_frozen": passed,
                "opened_v15_confirmation_authorized": passed,
                "external_panel_authorized": False,
                "joint_panel_authorized": False,
                "generator_optimizer_steps": 0,
                "authoritative_training_decision": TRAINING_DECISION,
                "report_sha256": v23.sha256_file(report_path),
                "exact_csv_sha256": v23.sha256_file(exact_csv_path),
                "completion_receipt_sha256": v23.sha256_file(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
