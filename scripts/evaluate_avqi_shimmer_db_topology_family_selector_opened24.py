#!/usr/bin/env python3
"""Run the frozen v18 D-then-C selector on opened v14+v15 cases.

This is a development-only integration audit.  Candidate selection uses the
unchanged v18 topology/proxy/safety/PCM24 certificates.  Candidate exact AVQI
outcomes remain closed until all 24 cases select inside the unchanged 500-ms
per-case metric-step contract and a hash-bound selector seal is written.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.avqi_shimmer_exact_topology_runtime import (
    ExactShimmerTopologyWorker,
    require_exact_topology_equal,
)
from scripts.evaluate_avqi_shimmer_db_candidate_c_fresh_panel import (
    MATERIAL_GAP_THRESHOLD,
    SHIMMER_DB_INDEX,
    avqi_code_tree_sha256,
    component_fields,
    exact_components,
    load_predictor,
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
from scripts.evaluate_avqi_shimmer_db_source_informed_v17 import (
    synthetic_candidate_d_warmup,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    FIXED_ALPHA,
    SELECTOR_NAME,
    WORKER_COUNT,
    evaluate_selector_case,
    preselection_rows,
    selector_contract,
    summarize_exact_rows,
    synthetic_v18_warmup,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    validate_dev_files,
    synthetic_torch_warmup,
)
from scripts.evaluate_direct_avqi_waveform_optimization import (
    full_band_pathology_guardrails,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SELECTOR_SOURCE_COMMIT = "c5c6e7612d6e7b641550b5706c4c3fe3a1a9927a"
FROZEN_SELECTOR_SOURCE_SHA256 = (
    "7401b4b80f6dbb546a4a88886c469bb4df6b4681bad9314f1244a046fbb2b69b"
)
FROZEN_V16_SOURCE_SHA256 = (
    "d8bfb0f31d9d98832d6c4409e5044b5d7cbe0b8b585e72f359fa3119d22aa662"
)
FROZEN_V17_SOURCE_SHA256 = (
    "324660709b2e6a4994d057c4d532cf89613f535ec96490f2cb038d7b33f55b22"
)
FROZEN_SELECTOR4_DECISION = (
    "PASS_SHIMMER_DB_V18_TOPOLOGY_FAMILY_SELECTOR_4CASE_MECHANISM"
)
OPENED24_PASS_DECISION = "PASS_SHIMMER_DB_V18_OPENED_V14_V15_24CASE"
OPENED24_PRESELECTION_NO_GO = "NO_GO_SHIMMER_DB_V18_OPENED24_PRESELECTION"
OPENED24_EXACT_NO_GO = "NO_GO_SHIMMER_DB_V18_OPENED_V14_V15_24CASE"
EXPECTED_CASE_COUNT = 24
EXPECTED_SPEAKER_COUNT = 12
EXPECTED_PANEL_CASE_COUNT = 12
EXPECTED_PANEL_SPEAKER_COUNT = 6
EXPECTED_COMBINED_SLICES = {
    "view": Counter({"cs": 12, "sv": 12}),
    "sample_group": Counter(
        {"pathological_mild": 12, "pathological_severe": 12}
    ),
    "condition": Counter({"rir_only": 8, "snr10": 8, "snr20": 8}),
}
PANEL_EXPECTATIONS = {
    "v14": {
        "source_commit": "60dd0fe9dc748ebb793937e67aa0e38a7909876f",
        "slurm_job_id": "19906678",
        "recipe_indices": tuple(range(912, 924)),
    },
    "v15": {
        "source_commit": "6ebc68c5a269b53122f10643c80e2f56007312ca",
        "slurm_job_id": "19907818",
        "recipe_indices": tuple(range(924, 936)),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for panel_label in ("v14", "v15"):
        parser.add_argument(
            f"--{panel_label}-panel-contract", type=Path, required=True
        )
        parser.add_argument(
            f"--{panel_label}-panel-contract-sha256", required=True
        )
        parser.add_argument(
            f"--{panel_label}-target-contract", type=Path, required=True
        )
        parser.add_argument(
            f"--{panel_label}-target-contract-sha256", required=True
        )
        parser.add_argument(
            f"--{panel_label}-fresh-results", type=Path, required=True
        )
        parser.add_argument(
            f"--{panel_label}-fresh-results-sha256", required=True
        )
    for artifact in ("report", "preselection", "seal", "results", "receipt"):
        parser.add_argument(
            f"--selector4-{artifact}", type=Path, required=True
        )
        parser.add_argument(f"--selector4-{artifact}-sha256", required=True)
    parser.add_argument("--selector-core-script", type=Path, required=True)
    parser.add_argument("--selector-core-script-sha256", required=True)
    parser.add_argument("--v16-family-source", type=Path, required=True)
    parser.add_argument("--v16-family-source-sha256", required=True)
    parser.add_argument("--v17-family-source", type=Path, required=True)
    parser.add_argument("--v17-family-source-sha256", required=True)
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_frozen_selector_ancestor(source_commit: str) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            FROZEN_SELECTOR_SOURCE_COMMIT,
            source_commit,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("opened24 source does not descend from frozen v18 source")


def validate_selector4_evidence(args: argparse.Namespace) -> dict[str, str]:
    artifact_names = ("report", "preselection", "seal", "results", "receipt")
    paths = {
        name: getattr(args, f"selector4_{name}") for name in artifact_names
    }
    hashes = {
        f"selector4_{name}": validate_hash(
            paths[name],
            getattr(args, f"selector4_{name}_sha256"),
            f"selector4 {name}",
        )
        for name in artifact_names
    }
    report = read_json(paths["report"])
    seal = read_json(paths["seal"])
    receipt = read_json(paths["receipt"])
    preselection = read_csv(paths["preselection"])
    results = read_csv(paths["results"])
    if report.get("decision") != FROZEN_SELECTOR4_DECISION:
        raise ValueError("selector4 mechanism decision is not PASS")
    if receipt.get("decision") != report.get("decision"):
        raise ValueError("selector4 report/receipt decision drift")
    if report.get("source_commit") != FROZEN_SELECTOR_SOURCE_COMMIT:
        raise ValueError("selector4 source commit drift")
    if seal.get("source_commit") != FROZEN_SELECTOR_SOURCE_COMMIT:
        raise ValueError("selector4 seal source commit drift")
    if receipt.get("source_commit") != FROZEN_SELECTOR_SOURCE_COMMIT:
        raise ValueError("selector4 receipt source commit drift")
    if report.get("opened_v14_v15_expansion_authorized") is not True:
        raise ValueError("selector4 did not authorize opened24 expansion")
    if report.get("promotion_authorized") is not False:
        raise ValueError("selector4 unexpectedly authorized promotion")
    if report.get("new_sealed_panel_authorized") is not False:
        raise ValueError("selector4 unexpectedly authorized a sealed panel")
    if report.get("candidate_exact_outcomes_opened_after_selector_seal") is not True:
        raise ValueError("selector4 exact-opening order is not certified")
    if report.get("summary", {}).get("all_gates_pass") is not True:
        raise ValueError("selector4 summary gates did not pass")
    if not all(report["summary"].get("mechanism_gates", {}).values()):
        raise ValueError("selector4 mechanism gates did not all pass")
    if seal.get("candidate_exact_outcomes_present") is not False:
        raise ValueError("selector4 seal contains candidate exact outcomes")
    if seal.get("selection_uses_candidate_exact_outcome") is not False:
        raise ValueError("selector4 selection used candidate exact outcomes")
    if seal.get("selector_contract") != selector_contract():
        raise ValueError("selector4 contract drift")
    if seal.get("preselection_sha256") != hashes["selector4_preselection"]:
        raise ValueError("selector4 seal/preselection hash drift")
    if report.get("selector_seal_sha256") != hashes["selector4_seal"]:
        raise ValueError("selector4 report/seal hash drift")
    if len(seal.get("rows", [])) != 4 or len(results) != 4:
        raise ValueError("selector4 evidence no longer contains four cases")
    if len({row["case_id"] for row in preselection}) != 4:
        raise ValueError("selector4 preselection case coverage drift")
    for row in seal["rows"]:
        validate_hash(
            Path(row["candidate_path"]),
            row["candidate_sha256"],
            "selector4 selected waveform",
        )
    expected_receipt_hashes = {
        "diagnostic_report.json": hashes["selector4_report"],
        "family_selector_preselection.csv": hashes["selector4_preselection"],
        "selector_seal.json": hashes["selector4_seal"],
        "family_selector_results.csv": hashes["selector4_results"],
    }
    if receipt.get("artifact_sha256") != expected_receipt_hashes:
        raise ValueError("selector4 receipt artifact binding drift")
    if any(
        (
            report.get("formal_generator_training_authorized") is not False,
            report.get("generator_loaded") is not False,
            report.get("generator_optimizer_created") is not False,
            report.get("generator_optimizer_steps") != 0,
            receipt.get("generator_optimizer_steps") != 0,
        )
    ):
        raise ValueError("selector4 training boundary drift")
    if report.get("authoritative_training_decision") != "NO_GO_AVQI_T2_TRAINING":
        raise ValueError("selector4 authoritative training decision drift")
    return hashes


def validate_target_contract(
    panel_label: str,
    panel_rows: list[dict[str, Any]],
    input_by_case: dict[str, dict[str, Any]],
    target_contract: dict[str, Any],
) -> None:
    if target_contract.get("schema_version") != (
        "avqi-route-c-shimmer-db-supervised-target-v1"
    ):
        raise ValueError(f"{panel_label} target schema drift")
    if target_contract.get("role") != (
        "same_speaker_target_scalar_required_by_candidate_loss"
    ):
        raise ValueError(f"{panel_label} target role drift")
    if target_contract.get("selection_or_tuning_use") is not False:
        raise ValueError(f"{panel_label} target contract was used for tuning")
    if target_contract.get("candidate_exact_outcomes_present") is not False:
        raise ValueError(f"{panel_label} target contract exposes candidate exact data")
    if (
        target_contract.get("clean_target_pulse_positions_exposed_to_output_branch")
        is not False
    ):
        raise ValueError(f"{panel_label} clean target topology exposure drift")
    target_rows = target_contract.get("rows", [])
    if len(target_rows) != EXPECTED_PANEL_CASE_COUNT:
        raise ValueError(f"{panel_label} target row count drift")
    target_by_case = {row["case_id"]: row for row in target_rows}
    if set(target_by_case) != {row["case_id"] for row in panel_rows}:
        raise ValueError(f"{panel_label} target case coverage drift")
    for panel_row in panel_rows:
        case_id = panel_row["case_id"]
        target_row = target_by_case[case_id]
        input_row = input_by_case[case_id]
        if target_row["speaker_id"] != panel_row["speaker_id"]:
            raise ValueError(f"{panel_label} target speaker drift: {case_id}")
        if target_row["view"] != panel_row["view"]:
            raise ValueError(f"{panel_label} target view drift: {case_id}")
        if target_row["target_sha256"] != panel_row["target_sha256"]:
            raise ValueError(f"{panel_label} target waveform hash drift: {case_id}")
        if not math.isclose(
            float(target_row["exact_target_shimmer_db"]),
            float(input_row["exact_target_shimmer_db"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"{panel_label} target scalar drift: {case_id}")


def validate_opened_panel(
    panel_label: str,
    panel_path: Path,
    panel_sha256: str,
    target_path: Path,
    target_sha256: str,
    results_path: Path,
    results_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    hashes = {
        f"{panel_label}_panel_contract": validate_hash(
            panel_path, panel_sha256, f"{panel_label} panel contract"
        ),
        f"{panel_label}_target_contract": validate_hash(
            target_path, target_sha256, f"{panel_label} target contract"
        ),
        f"{panel_label}_fresh_results": validate_hash(
            results_path, results_sha256, f"{panel_label} fresh results"
        ),
    }
    panel = read_json(panel_path)
    target_contract = read_json(target_path)
    result_rows = read_csv(results_path)
    panel_rows, input_by_case = validate_panel(panel, result_rows)
    validate_dev_files(panel_rows)
    expected = PANEL_EXPECTATIONS[panel_label]
    if panel.get("source_commit") != expected["source_commit"]:
        raise ValueError(f"{panel_label} source commit drift")
    if str(panel.get("slurm_job_id")) != expected["slurm_job_id"]:
        raise ValueError(f"{panel_label} Slurm job drift")
    if panel.get("speaker_split_before_simulation") is not True:
        raise ValueError(f"{panel_label} speaker split contract drift")
    if len(panel_rows) != EXPECTED_PANEL_CASE_COUNT:
        raise ValueError(f"{panel_label} case count drift")
    if len({row["speaker_id"] for row in panel_rows}) != EXPECTED_PANEL_SPEAKER_COUNT:
        raise ValueError(f"{panel_label} speaker count drift")
    recipes = tuple(sorted(int(row["recipe_index"]) for row in panel_rows))
    if recipes != expected["recipe_indices"]:
        raise ValueError(f"{panel_label} recipe coverage drift")
    for row in panel_rows:
        validate_hash(Path(row["source_path"]), row["source_sha256"], "source")
        validate_hash(Path(row["base_path"]), row["base_sha256"], "base")
        validate_hash(Path(row["degraded_path"]), row["degraded_sha256"], "degraded")
        validate_hash(Path(row["target_path"]), row["target_sha256"], "target")
        row["opened_panel"] = panel_label
    validate_target_contract(
        panel_label,
        panel_rows,
        input_by_case,
        target_contract,
    )
    metadata = {
        "panel_label": panel_label,
        "source_commit": panel["source_commit"],
        "slurm_job_id": str(panel["slurm_job_id"]),
        "case_count": len(panel_rows),
        "speaker_count": len({row["speaker_id"] for row in panel_rows}),
        "speakers": sorted({row["speaker_id"] for row in panel_rows}),
        "recipe_indices": list(recipes),
        "speaker_split_before_simulation": True,
        "source_sha256": panel.get("source_sha256", {}),
    }
    return panel_rows, input_by_case, hashes, metadata


def validate_combined_scope(
    v14_rows: list[dict[str, Any]],
    v15_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [*v14_rows, *v15_rows]
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError("opened24 requires exactly 24 cases")
    case_ids = [row["case_id"] for row in rows]
    if len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise ValueError("opened24 case overlap or duplicate")
    speakers = [row["speaker_id"] for row in rows]
    if len(set(speakers)) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("opened24 speaker overlap or cardinality drift")
    v14_speakers = {row["speaker_id"] for row in v14_rows}
    v15_speakers = {row["speaker_id"] for row in v15_rows}
    if v14_speakers & v15_speakers:
        raise ValueError("opened v14/v15 speaker overlap")
    v14_recipes = {int(row["recipe_index"]) for row in v14_rows}
    v15_recipes = {int(row["recipe_index"]) for row in v15_rows}
    if v14_recipes & v15_recipes:
        raise ValueError("opened v14/v15 recipe overlap")
    for field, expected in EXPECTED_COMBINED_SLICES.items():
        if Counter(row[field] for row in rows) != expected:
            raise ValueError(f"opened24 combined slice drift: {field}")
    return rows


def validate_sources_and_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    if repository_head() != args.source_commit:
        raise ValueError("opened24 source commit differs from repository HEAD")
    require_frozen_selector_ancestor(args.source_commit)
    source_hashes = {
        "selector_core": validate_hash(
            args.selector_core_script,
            args.selector_core_script_sha256,
            "frozen v18 selector core",
        ),
        "v16_family_source": validate_hash(
            args.v16_family_source,
            args.v16_family_source_sha256,
            "frozen v16 family source",
        ),
        "v17_family_source": validate_hash(
            args.v17_family_source,
            args.v17_family_source_sha256,
            "frozen v17 family source",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen Shimmer predictor",
        ),
        "runtime_worker": validate_hash(
            args.runtime_worker_script,
            args.runtime_worker_script_sha256,
            "exact topology worker",
        ),
    }
    if source_hashes["selector_core"] != FROZEN_SELECTOR_SOURCE_SHA256:
        raise ValueError("v18 selector core no longer matches c5c6e76")
    if source_hashes["v16_family_source"] != FROZEN_V16_SOURCE_SHA256:
        raise ValueError("v16 family source drift")
    if source_hashes["v17_family_source"] != FROZEN_V17_SOURCE_SHA256:
        raise ValueError("v17 family source drift")
    observed_avqi_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    source_hashes["avqi_code_tree"] = observed_avqi_hash
    source_hashes.update(validate_selector4_evidence(args))

    opened_rows: dict[str, list[dict[str, Any]]] = {}
    opened_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    panel_metadata: list[dict[str, Any]] = []
    for panel_label in ("v14", "v15"):
        rows, inputs, hashes, metadata = validate_opened_panel(
            panel_label,
            getattr(args, f"{panel_label}_panel_contract"),
            getattr(args, f"{panel_label}_panel_contract_sha256"),
            getattr(args, f"{panel_label}_target_contract"),
            getattr(args, f"{panel_label}_target_contract_sha256"),
            getattr(args, f"{panel_label}_fresh_results"),
            getattr(args, f"{panel_label}_fresh_results_sha256"),
        )
        opened_rows[panel_label] = rows
        opened_inputs[panel_label] = inputs
        source_hashes.update(hashes)
        panel_metadata.append(metadata)
    rows = validate_combined_scope(opened_rows["v14"], opened_rows["v15"])
    input_by_case = {
        **opened_inputs["v14"],
        **opened_inputs["v15"],
    }
    if set(input_by_case) != {row["case_id"] for row in rows}:
        raise ValueError("opened24 result coverage drift")
    return rows, input_by_case, source_hashes, panel_metadata


def summarize_scope(
    rows: list[dict[str, Any]],
    expected_case_count: int,
    coverage_gate_name: str,
) -> dict[str, Any]:
    summary = summarize_exact_rows(rows)
    mechanism_gates = dict(summary["mechanism_gates"])
    original_coverage = mechanism_gates.pop("complete_prototype_coverage")
    mechanism_gates[coverage_gate_name] = len(rows) == expected_case_count
    integration_gates = dict(summary["integration_gates"])
    integration_gates["mechanism"] = all(mechanism_gates.values())
    summary["mechanism_gates"] = mechanism_gates
    summary["integration_gates"] = integration_gates
    summary["all_gates_pass"] = all(integration_gates.values())
    summary["scope_coverage"] = {
        "expected_case_count": expected_case_count,
        "observed_case_count": len(rows),
        "gate_name": coverage_gate_name,
        "gate_pass": len(rows) == expected_case_count,
        "frozen_core_four_case_coverage_value_before_scope_rebind": (
            original_coverage
        ),
    }
    return summary


def opened24_gate_decision(
    combined_summary: dict[str, Any],
    panel_gate_summaries: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], str, bool]:
    if set(panel_gate_summaries) != {"v14", "v15"}:
        raise ValueError("opened24 panel gate summary coverage drift")
    gates = {
        "combined_24case_pass": bool(combined_summary["all_gates_pass"]),
        "v14_panel_pass": bool(
            panel_gate_summaries["v14"]["all_gates_pass"]
        ),
        "v15_panel_pass": bool(
            panel_gate_summaries["v15"]["all_gates_pass"]
        ),
    }
    authorized = all(gates.values())
    decision = OPENED24_PASS_DECISION if authorized else OPENED24_EXACT_NO_GO
    return gates, decision, authorized


def case_runtime_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "case_id",
        "selected_family",
        "selected_alpha",
        "attempted_family_count",
        "candidate_topology_refresh_count",
        "base_refresh_runtime_ms",
        "gradient_runtime_ms",
        "candidate_d_plan_runtime_ms",
        "candidate_d_projection_runtime_ms",
        "candidate_d_batch_runtime",
        "candidate_d_refresh_runtime",
        "candidate_c_batch_runtime",
        "candidate_c_refresh_runtime",
        "total_metric_step_runtime_ms",
        "runtime_gate_pass",
        "selector_pass",
    )
    return [{key: record[key] for key in keys} for record in records]


def write_completion_receipt(
    args: argparse.Namespace,
    decision: str,
    artifact_paths: list[Path],
    new_sealed_panel_authorized: bool,
) -> None:
    receipt = {
        "schema_version": (
            "avqi-route-c-shimmer-db-topology-family-selector-opened24-receipt-v1"
        ),
        "phase": "opened_v14_v15_24case",
        "decision": decision,
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "slurm_job_id": args.slurm_job_id,
        "dev_only": True,
        "opened_case_count": EXPECTED_CASE_COUNT,
        "promotion_authorized": False,
        "new_sealed_panel_authorized": new_sealed_panel_authorized,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)


def build_exact_rows(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    input_by_case: dict[str, dict[str, Any]],
    case_records: list[dict[str, Any]],
    target_scale: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    exact_items = [
        {
            "id": f"v18-opened24-selected:{panel_row['case_id']}",
            "case_id": panel_row["case_id"],
            "role": "topology_family_selected_candidate",
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
        selected = record["selected_record"]
        if selected is None:
            raise ValueError(f"missing selected record after opened24 seal: {case_id}")
        selected_topology_rebound = bool(
            require_exact_topology_equal(
                record["selected_topology"],
                after,
                f"v18 opened24 selected topology rebound {case_id}",
            )
        )
        target_components = exact_vector(input_row, "target")
        base_components = exact_vector(input_row, "before")
        after_components = exact_components(after)
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        candidate_waveform = read_waveform(record["selected_path"])
        base_topology = record["base_topology"]
        row: dict[str, Any] = {
            "case_id": case_id,
            "opened_panel": panel_row["opened_panel"],
            "speaker_id": panel_row["speaker_id"],
            "sample_group": panel_row["sample_group"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "candidate": SELECTOR_NAME,
            "selected_family": record["selected_family"],
            "optimized_component": "shimmer_db",
            "alpha_max": FIXED_ALPHA,
            "selected_alpha": record["selected_alpha"],
            "selected_backtrack_index": record["selected_backtrack_index"],
            "candidate_path": str(record["selected_path"].resolve()),
            "candidate_sha256": selected["candidate_sha256"],
            "proxy_before": record["proxy_before"],
            "proxy_after_frozen_topology": selected[
                "proxy_after_frozen_topology"
            ],
            "proxy_target": record["proxy_target"],
            "proxy_loss": record["proxy_loss"],
            "gradient_l2_norm": selected["gradient_l2_norm"],
            "gradient_rms": selected["gradient_rms"],
            "gradient_finite": selected["gradient_finite"],
            "pulse_refresh_runtime_ms": record["total_metric_step_runtime_ms"],
            "torch_step_runtime_ms": record["gradient_runtime_ms"],
            "total_metric_step_overhead_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "total_metric_step_runtime_ms": record[
                "total_metric_step_runtime_ms"
            ],
            "runtime_gate_pass": record["runtime_gate_pass"],
            "base_refresh_runtime_ms": record["base_refresh_runtime_ms"],
            "candidate_topology_refresh_count": record[
                "candidate_topology_refresh_count"
            ],
            "attempted_family_count": record["attempted_family_count"],
            "selector_pass": record["selector_pass"],
            "selector_uses_no_candidate_exact_outcome": True,
            "pcm24_effective_step_pass": selected["pcm24_effective_step_pass"],
            "pcm24_changed_samples": selected["pcm24_changed_samples"],
            "pcm24_changed_fraction": selected["pcm24_changed_fraction"],
            "pcm24_residual_rms_lsb": selected["pcm24_residual_rms_lsb"],
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
            "selected_topology_rebound": selected_topology_rebound,
            "base_topology_rebound": (
                record["base_topology_sha256"]
                == str(input_row["composite_topology_sha256"])
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
        row["forward_normalized_abs_error_shimmer_db"] = abs(
            row["proxy_before"] - row["exact_before_shimmer_db"]
        ) / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        row.update(topology_stability(base_topology, after))
        row.update(
            waveform_safety(base_waveform.numpy(), candidate_waveform.numpy())
        )
        row.update(
            full_band_pathology_guardrails(
                target_waveform,
                base_waveform,
                candidate_waveform,
            )
        )
        rows.append(row)
    versions = {
        "parselmouth": exact_after["parselmouth_version"],
        "praat": exact_after["praat_version"],
    }
    return rows, versions


def run_opened24(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    input_by_case: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
    panel_metadata: list[dict[str, Any]],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
    runtime_environment: dict[str, Any],
) -> None:
    case_records: list[dict[str, Any]] = []
    for index, panel_row in enumerate(panel_rows, start=1):
        case_records.append(
            evaluate_selector_case(
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
        print(f"v18_opened24_selector={index}/{EXPECTED_CASE_COUNT}", flush=True)

    preselection_path = args.output_dir / "family_selector_preselection.csv"
    write_csv(preselection_path, preselection_rows(panel_rows, case_records))
    selector_failures = [
        record["case_id"] for record in case_records if not record["selector_pass"]
    ]
    common_report = {
        "schema_version": (
            "avqi-route-c-shimmer-db-topology-family-selector-opened24-v1"
        ),
        "candidate": SELECTOR_NAME,
        "route_type": "hybrid_praat_assisted_topology_family_selector",
        "pure_torch_estimator": False,
        "phase": "opened_v14_v15_24case",
        "dev_only": True,
        "opened_cases_only": True,
        "selector_contract": selector_contract(),
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "source_sha256": source_hashes,
        "panel_bindings": panel_metadata,
        "runtime_environment": runtime_environment,
        "case_runtime": case_runtime_rows(case_records),
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "promotion_authorized": False,
    }
    if selector_failures:
        report = {
            **common_report,
            "decision": OPENED24_PRESELECTION_NO_GO,
            "candidate_exact_outcomes_opened": False,
            "exact_component_scoring_requested": False,
            "selector_failures": selector_failures,
            "selector_coverage": (
                EXPECTED_CASE_COUNT - len(selector_failures)
            )
            / EXPECTED_CASE_COUNT,
            "new_sealed_panel_authorized": False,
        }
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        write_completion_receipt(
            args,
            OPENED24_PRESELECTION_NO_GO,
            [report_path, preselection_path],
            False,
        )
        print(
            json.dumps(
                {
                    "decision": OPENED24_PRESELECTION_NO_GO,
                    "failures": selector_failures,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    selector_seal = {
        "schema_version": (
            "avqi-route-c-shimmer-db-topology-family-selector-opened24-seal-v1"
        ),
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_present": False,
        "selection_uses_candidate_exact_outcome": False,
        "selector_contract": selector_contract(),
        "source_sha256": source_hashes,
        "panel_bindings": panel_metadata,
        "preselection_sha256": sha256_file(preselection_path),
        "rows": [
            {
                "case_id": record["case_id"],
                "opened_panel": panel_row["opened_panel"],
                "selected_family": record["selected_family"],
                "selected_alpha": record["selected_alpha"],
                "selected_backtrack_index": record["selected_backtrack_index"],
                "candidate_path": str(record["selected_path"].resolve()),
                "candidate_sha256": record["selected_record"][
                    "candidate_sha256"
                ],
                "candidate_topology_sha256": record["selected_record"][
                    "candidate_topology_sha256"
                ],
                "attempted_family_count": record["attempted_family_count"],
                "candidate_topology_refresh_count": record[
                    "candidate_topology_refresh_count"
                ],
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

    rows, exact_versions = build_exact_rows(
        args,
        panel_rows,
        input_by_case,
        case_records,
        target_scale,
    )
    results_path = args.output_dir / "family_selector_results.csv"
    write_csv(results_path, rows)
    summary = summarize_scope(
        rows,
        EXPECTED_CASE_COUNT,
        "complete_opened24_coverage",
    )
    panel_gate_summaries = {
        panel_label: summarize_scope(
            [row for row in rows if row["opened_panel"] == panel_label],
            EXPECTED_PANEL_CASE_COUNT,
            f"complete_{panel_label}_coverage",
        )
        for panel_label in ("v14", "v15")
    }
    opened24_gates, decision, new_sealed_panel_authorized = (
        opened24_gate_decision(summary, panel_gate_summaries)
    )
    report = {
        **common_report,
        "decision": decision,
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "selector_seal_sha256": sha256_file(selector_seal_path),
        "summary": summary,
        "panel_gate_summaries": panel_gate_summaries,
        "opened24_authorization_gates": opened24_gates,
        "new_sealed_panel_authorized": new_sealed_panel_authorized,
        "exact_scorer_versions": exact_versions,
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    write_completion_receipt(
        args,
        decision,
        [report_path, preselection_path, selector_seal_path, results_path],
        new_sealed_panel_authorized,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "runtime": summary["total_metric_step_runtime_ms"],
                "exact_improvement_fraction": summary["mechanism"][
                    "exact_db_improvement_fraction"
                ],
                "median_normalized_reduction": summary["mechanism"][
                    "median_exact_db_normalized_gap_reduction"
                ],
                "new_sealed_panel_authorized": new_sealed_panel_authorized,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    panel_rows, input_by_case, source_hashes, panel_metadata = (
        validate_sources_and_inputs(args)
    )
    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    device = torch.device(args.device)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    torch_warmup = synthetic_torch_warmup(predictor, target_scale, device)
    candidate_d_warmup = synthetic_candidate_d_warmup(device)
    optimized_v18_warmup = synthetic_v18_warmup(device)
    workers: list[ExactShimmerTopologyWorker] = []
    worker_startups: list[dict[str, Any]] = []
    worker_warmups: list[dict[str, Any]] = []
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
                    "request_wall_ms": warmup_ms,
                    **warmup,
                }
            )
        runtime_environment = {
            "torch_synthetic_warmup": torch_warmup,
            "candidate_d_synthetic_warmup": candidate_d_warmup,
            "optimized_v18_synthetic_warmup": optimized_v18_warmup,
            "worker_startups": worker_startups,
            "worker_synthetic_warmups": worker_warmups,
            "worker_count": WORKER_COUNT,
            "warmups_outside_case_timer": True,
        }
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            run_opened24(
                args,
                panel_rows,
                input_by_case,
                source_hashes,
                panel_metadata,
                predictor,
                target_scale,
                device,
                workers,
                executor,
                waveform_root,
                runtime_environment,
            )
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
