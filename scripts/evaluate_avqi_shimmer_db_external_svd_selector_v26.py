#!/usr/bin/env python3
"""Run the frozen D-then-C selector and exact external SVD adjudication.

The v24 waveform panel and v25 target scalars must already be hash sealed.
All 12 cases must select using only frozen proxy/topology/safety/PCM24
certificates before the selected candidate seal is written.  Exact Praat then
re-scores target, base, and selected candidate.  SVD has no valid mild/severe
labels, so severity evidence remains bound to the passing v23 opened24 panel;
the external panel adds result-blind domain, sex, view, and condition coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
import torch

from scripts.adjudicate_avqi_shimmer_db_deterministic_opened24_v23 import (
    TARGET_REPRODUCTION_ABS_TOLERANCE,
    TRAINING_DECISION,
)
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
    topology_stability,
)
from scripts.evaluate_avqi_shimmer_db_source_informed_v17 import (
    synthetic_candidate_d_warmup,
)
from scripts.evaluate_avqi_shimmer_db_runtime_v19_full_step_integration import (
    validate_deterministic_process_contract,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_opened24 import (
    FROZEN_SELECTOR_SOURCE_COMMIT,
    FROZEN_SELECTOR_SOURCE_SHA256,
    FROZEN_V16_SOURCE_SHA256,
    FROZEN_V17_SOURCE_SHA256,
    require_frozen_selector_ancestor,
    validate_selector4_evidence,
)
from scripts.evaluate_avqi_shimmer_db_topology_family_selector_v18 import (
    FIXED_ALPHA,
    SELECTOR_NAME,
    WORKER_COUNT,
    evaluate_selector_case,
    selector_contract,
    synthetic_v18_warmup,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    synthetic_torch_warmup,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    CACHE_RUNTIME_MAX_MS,
    GRADIENT_NORM_RANGE,
    IMPROVEMENT_FRACTION_GATE,
    MAXIMUM_CLIP_FRACTION,
    MEDIAN_REDUCTION_GATE,
    MINIMUM_COSINE,
    NONSELECTED_MEDIAN_INCREASE_GATE,
    RESIDUAL_CEILING_DB,
    aggregate_candidate,
    summarize_effect_slice,
    waveform_safety,
)
from scripts.evaluate_direct_avqi_waveform_optimization import (
    aggregate_denoising,
    aggregate_pathology_guardrails,
    full_band_pathology_guardrails,
)
from scripts.prepare_avqi_shimmer_db_external_svd_panel_v24 import (
    EXPECTED_CASES,
    EXPECTED_SPEAKERS,
    PRIOR_LEDGER_SCHEMA,
    validate_opened24_authorization,
    validate_prior_ledger,
)
from scripts.seal_avqi_shimmer_db_external_svd_target_v25 import (
    TARGET_DECISION,
    TARGET_RECEIPT_SCHEMA,
    TARGET_SCHEMA,
    validate_panel_binding,
)


REPORT_SCHEMA = "avqi-route-c-shimmer-db-external-svd-exact-promotion-v26"
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-exact-promotion-receipt-v26"
)
SELECTOR_SEAL_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-selector-seal-v26"
)
PASS_DECISION = "PASS_SHIMMER_DB_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V26"
PRESELECTION_NO_GO = "NO_GO_SHIMMER_DB_EXTERNAL_SVD_PRESELECTION_V26"
EXACT_NO_GO = "NO_GO_SHIMMER_DB_EXTERNAL_SVD_EXACT_PROMOTION_V26"
READINESS_PASS = "READY_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
READINESS_NO_GO = "NO_GO_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
UPDATED_LEDGER_NAME = "prior_panel_speaker_ledger_after_v24.json"
EXTERNAL_REQUIRED_EFFECT_SLICES = (
    "view=cs",
    "view=sv",
    "condition=rir_only",
    "condition=snr20",
    "condition=snr10",
    "sex=female",
    "sex=male",
)


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "panel-seal",
        "seal-receipt",
        "updated-speaker-ledger",
        "target-contract",
        "target-receipt",
        "opened24-report",
        "opened24-receipt",
        "selector-core-script",
        "v16-family-source",
        "v17-family-source",
        "predictor-checkpoint",
        "runtime-worker-script",
    ):
        add_hashed_path(parser, option)
    for artifact in ("report", "preselection", "seal", "results", "receipt"):
        add_hashed_path(parser, f"selector4-{artifact}")
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_repository(args: argparse.Namespace) -> dict[str, str]:
    root = args.repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain the v26 adjudicator")
    head = git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v26 repository HEAD/source commit drift")
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v26 adjudication requires a clean repository")
    require_frozen_selector_ancestor(args.source_commit)
    return {
        "repository_root": str(root),
        "source_commit": head,
        "adjudicator_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_updated_ledger(
    ledger: dict[str, Any],
    panel_rows: list[dict[str, Any]],
    panel_receipt: dict[str, Any],
    *,
    ledger_sha256: str,
    source_commit: str,
) -> set[str]:
    if ledger.get("schema_version") != PRIOR_LEDGER_SCHEMA:
        raise ValueError("updated speaker ledger schema drift")
    if panel_receipt.get("artifact_sha256", {}).get(UPDATED_LEDGER_NAME) != (
        ledger_sha256
    ):
        raise ValueError("panel receipt/updated speaker ledger binding drift")
    speakers = validate_prior_ledger(ledger)
    selected = {str(row["panel_speaker_id"]) for row in panel_rows}
    if not selected.issubset(speakers):
        raise ValueError("external SVD speakers are absent from updated ledger")
    additions = [
        entry
        for entry in ledger["entries"]
        if entry.get("panel_role") == "shimmer_db_external_svd_v24"
    ]
    if (
        ledger.get("added_speaker_count") != EXPECTED_SPEAKERS
        or {entry.get("canonical_speaker_id") for entry in additions} != selected
        or any(entry.get("source_commit") != source_commit for entry in additions)
    ):
        raise ValueError("updated speaker ledger external additions drift")
    if any(
        entry.get("exact_shimmer_outcomes_opened_at_ledger_update") is not False
        for entry in additions
    ):
        raise ValueError("updated speaker ledger used external exact outcomes")
    return speakers


def validate_target_binding(
    panel_rows: list[dict[str, Any]],
    target_contract: dict[str, Any],
    target_receipt: dict[str, Any],
    *,
    panel_sha256: str,
    panel_receipt_sha256: str,
    target_sha256: str,
    source_commit: str,
) -> dict[str, dict[str, Any]]:
    if target_contract.get("schema_version") != TARGET_SCHEMA:
        raise ValueError("external SVD target contract schema drift")
    if target_receipt.get("schema_version") != TARGET_RECEIPT_SCHEMA:
        raise ValueError("external SVD target receipt schema drift")
    if target_receipt.get("decision") != TARGET_DECISION:
        raise ValueError("external SVD target stage is not sealed")
    if target_contract.get("source_commit") != source_commit or (
        target_receipt.get("source_commit") != source_commit
    ):
        raise ValueError("external SVD target source commit drift")
    if target_contract.get("panel_seal_sha256") != panel_sha256:
        raise ValueError("external SVD target/panel binding drift")
    if target_receipt.get("input_sha256", {}).get("panel_seal.json") != (
        panel_sha256
    ):
        raise ValueError("external SVD target receipt/panel binding drift")
    if target_receipt.get("input_sha256", {}).get("seal_receipt.json") != (
        panel_receipt_sha256
    ):
        raise ValueError("external SVD target receipt/panel receipt drift")
    if target_receipt.get("artifact_sha256", {}).get(
        "target_label_contract.json"
    ) != target_sha256:
        raise ValueError("external SVD target receipt/contract binding drift")
    if target_contract.get("role") != (
        "same_speaker_target_scalar_required_by_candidate_loss"
    ):
        raise ValueError("external SVD target supervision role drift")
    if (
        target_contract.get("selection_or_tuning_use") is not False
        or target_contract.get("base_exact_outcomes_present") is not False
        or target_contract.get("candidate_exact_outcomes_present") is not False
        or target_contract.get(
            "clean_target_pulse_positions_exposed_to_output_branch"
        )
        is not False
        or target_contract.get("target_exact_components_retained")
        != ["shimmer_db"]
        or target_contract.get("severity_labels_created") is not False
        or target_contract.get("emitted_waveform_highpass") is not False
        or target_contract.get("selector_stage_authorized") is not True
    ):
        raise ValueError("external SVD target information boundary drift")
    for label, value in (
        ("target contract", target_contract),
        ("target receipt", target_receipt),
    ):
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"external SVD {label} over-authorized promotion")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"external SVD {label} over-authorized joint panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"external SVD {label} optimizer boundary drift")
        if value.get("authoritative_training_decision") != TRAINING_DECISION:
            raise ValueError(f"external SVD {label} training decision drift")
    if (
        target_receipt.get("target_exact_shimmer_opened") is not True
        or target_receipt.get("base_exact_outcomes_opened") is not False
        or target_receipt.get("candidate_exact_outcomes_opened") is not False
        or target_receipt.get("selector_stage_authorized") is not True
    ):
        raise ValueError("external SVD target receipt opening-order drift")
    target_rows = target_contract.get("rows")
    if not isinstance(target_rows, list) or len(target_rows) != EXPECTED_CASES:
        raise ValueError("external SVD target row coverage drift")
    target_by_case = {str(row.get("case_id")): row for row in target_rows}
    if set(target_by_case) != {str(row["case_id"]) for row in panel_rows}:
        raise ValueError("external SVD target case coverage drift")
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        target_row = target_by_case[case_id]
        for field in (
            "panel_speaker_id",
            "speaker_id",
            "session_id",
            "sex",
            "view",
            "condition",
            "target_sha256",
        ):
            if target_row.get(field) != panel_row.get(field):
                raise ValueError(f"external SVD target {field} drift: {case_id}")
        if not math.isfinite(float(target_row["exact_target_shimmer_db"])):
            raise ValueError(f"non-finite external target scalar: {case_id}")
    versions = target_contract.get("exact_scorer_versions", {})
    if not versions.get("parselmouth") or not versions.get("praat"):
        raise ValueError("external SVD target exact scorer version drift")
    return target_by_case


def validate_sources_and_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    source_hashes = {
        name: validate_hash(
            getattr(args, name),
            getattr(args, f"{name}_sha256"),
            name.replace("_", " "),
        )
        for name in (
            "panel_seal",
            "seal_receipt",
            "updated_speaker_ledger",
            "target_contract",
            "target_receipt",
            "opened24_report",
            "opened24_receipt",
            "selector_core_script",
            "v16_family_source",
            "v17_family_source",
            "predictor_checkpoint",
            "runtime_worker_script",
        )
    }
    panel = read_json(args.panel_seal)
    panel_receipt = read_json(args.seal_receipt)
    panel_rows = validate_panel_binding(
        panel,
        panel_receipt,
        panel_sha256=source_hashes["panel_seal"],
    )
    if panel.get("source_commit") != args.source_commit:
        raise ValueError("v26 source commit differs from the v24 panel seal")
    ledger = read_json(args.updated_speaker_ledger)
    ledger_speakers = validate_updated_ledger(
        ledger,
        panel_rows,
        panel_receipt,
        ledger_sha256=source_hashes["updated_speaker_ledger"],
        source_commit=args.source_commit,
    )
    target_contract = read_json(args.target_contract)
    target_receipt = read_json(args.target_receipt)
    target_by_case = validate_target_binding(
        panel_rows,
        target_contract,
        target_receipt,
        panel_sha256=source_hashes["panel_seal"],
        panel_receipt_sha256=source_hashes["seal_receipt"],
        target_sha256=source_hashes["target_contract"],
        source_commit=args.source_commit,
    )
    opened24_report = read_json(args.opened24_report)
    opened24_receipt = read_json(args.opened24_receipt)
    validate_opened24_authorization(
        opened24_report,
        opened24_receipt,
        report_sha256=source_hashes["opened24_report"],
    )
    authorization = panel.get("authorization", {})
    if (
        authorization.get("opened24_report_sha256")
        != source_hashes["opened24_report"]
        or authorization.get("opened24_receipt_sha256")
        != source_hashes["opened24_receipt"]
    ):
        raise ValueError("v24 panel/opened24 v23 binding drift")
    if source_hashes["selector_core_script"] != FROZEN_SELECTOR_SOURCE_SHA256:
        raise ValueError("frozen v18 selector core source drift")
    if source_hashes["v16_family_source"] != FROZEN_V16_SOURCE_SHA256:
        raise ValueError("frozen v16 family source drift")
    if source_hashes["v17_family_source"] != FROZEN_V17_SOURCE_SHA256:
        raise ValueError("frozen v17 family source drift")
    observed_avqi_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_avqi_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    if target_receipt.get("input_sha256", {}).get("avqi_code_tree") != (
        observed_avqi_hash
    ):
        raise ValueError("v25 target/v26 exact AVQI code-tree drift")
    source_hashes["avqi_code_tree"] = observed_avqi_hash
    source_hashes.update(validate_selector4_evidence(args))
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        for role in ("source", "target", "degraded", "base"):
            validate_hash(
                Path(panel_row[f"{role}_path"]),
                str(panel_row[f"{role}_sha256"]),
                f"{role} waveform {case_id}",
            )
    bindings = {
        "opened24_v23_decision": opened24_report["decision"],
        "opened24_v23_all_gates_pass": all(opened24_report["gates"].values()),
        "opened24_v23_old_v18_evidence_kept_separate": opened24_report[
            "gates"
        ]["old_v18_evidence_kept_separate"],
        "opened24_v23_report_sha256": source_hashes["opened24_report"],
        "opened24_v23_receipt_sha256": source_hashes["opened24_receipt"],
        "panel_seal_sha256": source_hashes["panel_seal"],
        "panel_receipt_sha256": source_hashes["seal_receipt"],
        "target_contract_sha256": source_hashes["target_contract"],
        "target_receipt_sha256": source_hashes["target_receipt"],
        "updated_speaker_ledger_sha256": source_hashes[
            "updated_speaker_ledger"
        ],
        "updated_speaker_ledger_count": len(ledger_speakers),
    }
    return panel, panel_rows, target_by_case, source_hashes, bindings


def external_preselection_rows(
    panel_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, records, strict=True):
        for attempt_index, attempt in enumerate(record["attempts"]):
            rows.append(
                {
                    "case_id": panel_row["case_id"],
                    "panel_speaker_id": panel_row["panel_speaker_id"],
                    "sex": panel_row["sex"],
                    "view": panel_row["view"],
                    "condition": panel_row["condition"],
                    "severity_label_present": False,
                    "attempt_index": attempt_index,
                    "family": attempt["family"],
                    "alpha": attempt["alpha"],
                    "backtrack_index": attempt["backtrack_index"],
                    "candidate_path": str(attempt["candidate_path"].resolve()),
                    "candidate_sha256": attempt["candidate_sha256"],
                    "candidate_topology_sha256": attempt[
                        "candidate_topology_sha256"
                    ],
                    "candidate_pulse_count": int(
                        attempt["candidate_topology"]["pulse_count"]
                    ),
                    "proxy_before": attempt["proxy_before"],
                    "proxy_after_frozen_topology": attempt[
                        "proxy_after_frozen_topology"
                    ],
                    "normalized_proxy_gap_before": attempt[
                        "normalized_proxy_gap_before"
                    ],
                    "normalized_proxy_gap_after": attempt[
                        "normalized_proxy_gap_after"
                    ],
                    "proxy_nonregression_pass": attempt[
                        "proxy_nonregression_pass"
                    ],
                    "topology_stability_pass": attempt[
                        "topology_stability_pass"
                    ],
                    "finite_safety_pass": attempt["finite_safety_pass"],
                    "pcm24_effective_step_pass": attempt[
                        "pcm24_effective_step_pass"
                    ],
                    "selected_family": record["selected_family"],
                    "selected_alpha": record["selected_alpha"],
                    "selected_attempt": record["selected_record"] is attempt,
                    "runtime_gate_pass": record["runtime_gate_pass"],
                    "selector_pass": record["selector_pass"],
                    "total_metric_step_runtime_ms": record[
                        "total_metric_step_runtime_ms"
                    ],
                }
            )
    return rows


def build_exact_rows(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    target_by_case: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    target_scale: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    exact_items: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, records, strict=True):
        for role, path, topology_required in (
            ("target", Path(panel_row["target_path"]), False),
            ("base", Path(panel_row["base_path"]), True),
            ("candidate", record["selected_path"], True),
        ):
            exact_items.append(
                {
                    "id": f"{role}:{panel_row['case_id']}",
                    "case_id": panel_row["case_id"],
                    "role": role,
                    "path": str(path.resolve()),
                    "view": panel_row["view"],
                    "score_components": True,
                    "exact_metric_topology": topology_required,
                }
            )
    exact = run_exact(exact_items, args.exact_python, args.avqi_code_root)
    exact_by_id = {str(row["id"]): row for row in exact["rows"]}
    if set(exact_by_id) != {str(item["id"]) for item in exact_items}:
        raise ValueError("external SVD post-seal exact coverage drift")
    target_scale_np = target_scale.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for panel_row, record in zip(panel_rows, records, strict=True):
        case_id = str(panel_row["case_id"])
        target_exact = exact_by_id[f"target:{case_id}"]
        base_exact = exact_by_id[f"base:{case_id}"]
        candidate_exact = exact_by_id[f"candidate:{case_id}"]
        for exact_row in (target_exact, base_exact, candidate_exact):
            topology_required = exact_row.get("role") != "target"
            if exact_row.get("scoring_status") != "ok" or (
                topology_required and int(exact_row.get("pulse_count", 0)) < 3
            ):
                raise RuntimeError(
                    f"external SVD exact scoring failed: {exact_row.get('id')}"
                )
        target_components = exact_components(target_exact)
        base_components = exact_components(base_exact)
        candidate_components = exact_components(candidate_exact)
        target_scalar = float(target_by_case[case_id]["exact_target_shimmer_db"])
        target_reproduction_error = abs(
            float(target_components[SHIMMER_DB_INDEX]) - target_scalar
        )
        base_topology_hash = require_exact_topology_equal(
            record["base_topology"],
            base_exact,
            f"external SVD base topology rebound {case_id}",
        )
        selected_topology_hash = require_exact_topology_equal(
            record["selected_topology"],
            candidate_exact,
            f"external SVD selected topology rebound {case_id}",
        )
        selected = record["selected_record"]
        if selected is None:
            raise ValueError(f"external SVD selected record missing: {case_id}")
        target_waveform = read_waveform(Path(panel_row["target_path"]))
        base_waveform = read_waveform(Path(panel_row["base_path"]))
        candidate_waveform = read_waveform(record["selected_path"])
        row: dict[str, Any] = {
            "case_id": case_id,
            "dataset": "SVD",
            "panel_speaker_id": panel_row["panel_speaker_id"],
            "speaker_id": panel_row["speaker_id"],
            "session_id": panel_row["session_id"],
            "sex": panel_row["sex"],
            "view": panel_row["view"],
            "condition": panel_row["condition"],
            "severity_label_present": False,
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
            "pcm24_effective_step_pass": selected[
                "pcm24_effective_step_pass"
            ],
            "pcm24_changed_samples": selected["pcm24_changed_samples"],
            "pcm24_changed_fraction": selected["pcm24_changed_fraction"],
            "pcm24_residual_rms_lsb": selected["pcm24_residual_rms_lsb"],
            "base_output_exact_metric_pulse_count": int(
                record["base_topology"]["pulse_count"]
            ),
            "candidate_exact_metric_pulse_count": int(
                candidate_exact["pulse_count"]
            ),
            "metric_sample_count": int(
                record["base_topology"]["metric_sample_count"]
            ),
            "metric_constant_prefix_samples": int(
                record["base_topology"]["metric_constant_prefix_samples"]
            ),
            "metric_source_range_count": int(
                record["base_topology"]["metric_source_range_count"]
            ),
            "metric_mapped_sample_count": int(
                record["base_topology"]["metric_mapped_sample_count"]
            ),
            "metric_reconstruction_max_pcm16_error": int(
                record["base_topology"][
                    "metric_reconstruction_max_pcm16_error"
                ]
            ),
            "metric_reconstruction_differing_samples": int(
                record["base_topology"][
                    "metric_reconstruction_differing_samples"
                ]
            ),
            "candidate_metric_reconstruction_max_pcm16_error": int(
                candidate_exact["metric_reconstruction_max_pcm16_error"]
            ),
            "candidate_metric_reconstruction_differing_samples": int(
                candidate_exact["metric_reconstruction_differing_samples"]
            ),
            "base_topology_rebound": True,
            "base_topology_sha256": base_topology_hash,
            "selected_topology_rebound": True,
            "selected_topology_sha256": selected_topology_hash,
            "target_reproduction_abs_error": target_reproduction_error,
            "target_reproduction_pass": (
                target_reproduction_error
                <= TARGET_REPRODUCTION_ABS_TOLERANCE
            ),
            "clean_target_topology_drives_output": False,
            "emitted_waveform_highpass": False,
            "exact_metric_highpass_branch_only": True,
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
        row["forward_normalized_abs_error_shimmer_db"] = abs(
            row["proxy_before"] - row["exact_before_shimmer_db"]
        ) / max(float(target_scale_np[SHIMMER_DB_INDEX]), 1e-8)
        row.update(topology_stability(record["base_topology"], candidate_exact))
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
        "parselmouth": str(exact["parselmouth_version"]),
        "praat": str(exact["praat_version"]),
    }
    return rows, versions


def external_effect_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "view=cs": lambda row: row["view"] == "cs",
        "view=sv": lambda row: row["view"] == "sv",
        "condition=rir_only": lambda row: row["condition"] == "rir_only",
        "condition=snr20": lambda row: row["condition"] == "snr20",
        "condition=snr10": lambda row: row["condition"] == "snr10",
        "sex=female": lambda row: row["sex"] == "female",
        "sex=male": lambda row: row["sex"] == "male",
    }
    slices = {
        name: summarize_effect_slice([row for row in rows if predicate(row)])
        for name, predicate in predicates.items()
    }
    return {
        "required": list(EXTERNAL_REQUIRED_EFFECT_SLICES),
        "slices": slices,
        "decision": (
            "PASS"
            if all(
                slices[name]["decision"] == "PASS"
                for name in EXTERNAL_REQUIRED_EFFECT_SLICES
            )
            else "FAIL"
        ),
    }


def summarize_external(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compatibility_rows = [
        {**row, "sample_group": "external_svd_severity_not_available"}
        for row in rows
    ]
    mechanism = aggregate_candidate(SELECTOR_NAME, compatibility_rows)
    external_slices = external_effect_slices(rows)
    pathology = aggregate_pathology_guardrails(rows)
    denoising = aggregate_denoising(rows)
    mechanism_gates = {
        "complete_external_12case_coverage": len(rows) == EXPECTED_CASES,
        "overall_exact_db_effect": (
            mechanism["exact_db_improvement_fraction"]
            >= IMPROVEMENT_FRACTION_GATE
            and mechanism["median_exact_db_normalized_gap_reduction"]
            is not None
            and mechanism["median_exact_db_normalized_gap_reduction"]
            >= MEDIAN_REDUCTION_GATE
        ),
        "external_required_effect_slices": (
            external_slices["decision"] == "PASS"
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
        "mechanism": all(mechanism_gates.values()),
        "full_band_pathology_guardrails": pathology["decision"] == "PASS",
        "denoising_nonregression": denoising["decision"] == "PASS",
        "selector_coverage": all(row["selector_pass"] for row in rows),
        "selector_uses_no_candidate_exact_outcome": all(
            row["selector_uses_no_candidate_exact_outcome"] for row in rows
        ),
        "selected_topology_rebound": all(
            row["selected_topology_rebound"] for row in rows
        ),
        "base_topology_rebound": all(
            row["base_topology_rebound"] for row in rows
        ),
        "pcm24_effective_step": all(
            row["pcm24_effective_step_pass"] for row in rows
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
        "target_exact_reproduction": all(
            row["target_reproduction_pass"] for row in rows
        ),
        "full_band_emission": all(
            row["emitted_waveform_highpass"] is False for row in rows
        ),
        "external_speaker_sex_view_condition_coverage": (
            len({row["panel_speaker_id"] for row in rows})
            == EXPECTED_SPEAKERS
            and Counter(row["sex"] for row in rows)
            == Counter({"female": 6, "male": 6})
            and Counter(row["view"] for row in rows)
            == Counter({"cs": 6, "sv": 6})
            and Counter(row["condition"] for row in rows)
            == Counter({"rir_only": 4, "snr20": 4, "snr10": 4})
        ),
        "severity_gate_bound_to_v23_not_invented_on_svd": all(
            row["severity_label_present"] is False for row in rows
        ),
    }
    return {
        "candidate": SELECTOR_NAME,
        "mechanism": mechanism,
        "mechanism_gates": mechanism_gates,
        "external_effect_slices": external_slices,
        "severity_gate_source": "bound passing v23 opened24 mild/severe evidence",
        "svd_severity_labels_available": False,
        "frozen_core_severity_slice_gate_applied_to_svd": False,
        "full_band_pathology_guardrails": pathology,
        "denoising": denoising,
        "integration_gates": integration_gates,
        "selected_family_counts": dict(
            Counter(str(row["selected_family"]) for row in rows)
        ),
        "total_metric_step_runtime_ms": {
            "median": median(row["total_metric_step_runtime_ms"] for row in rows),
            "maximum": max(row["total_metric_step_runtime_ms"] for row in rows),
            "formal_gate_ms": CACHE_RUNTIME_MAX_MS,
        },
        "all_gates_pass": all(integration_gates.values()),
    }


def write_completion_receipt(
    args: argparse.Namespace,
    decision: str,
    artifact_paths: list[Path],
    *,
    exact_opened: bool,
    promotion_granted: bool,
) -> Path:
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": decision,
        "component": "shimmer_db",
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_opened_after_selector_seal": exact_opened,
        "old_v18_evidence_kept_separate": True,
        "scientific_promotion_granted": promotion_granted,
        "six_component_readiness_eligible": promotion_granted,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    receipt_path = args.output_dir / "completion_receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path


def run_external_selector(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    target_by_case: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
    bindings: dict[str, Any],
    source_provenance: dict[str, str],
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    waveform_root: Path,
    runtime_environment: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    for index, panel_row in enumerate(panel_rows, start=1):
        records.append(
            evaluate_selector_case(
                panel_row,
                float(
                    target_by_case[panel_row["case_id"]][
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
        print(f"external_svd_selector={index}/{EXPECTED_CASES}", flush=True)
    preselection_path = args.output_dir / "family_selector_preselection.csv"
    write_csv(
        preselection_path,
        external_preselection_rows(panel_rows, records),
    )
    selector_failures = [
        record["case_id"] for record in records if not record["selector_pass"]
    ]
    common_report = {
        "schema_version": REPORT_SCHEMA,
        "component": "shimmer_db",
        "candidate": SELECTOR_NAME,
        "route_type": "hybrid_praat_assisted_topology_family_selector",
        "pure_torch_estimator": False,
        "phase": "external_svd_12case_post_target_seal",
        "selector_contract": selector_contract(),
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "source_sha256": source_hashes,
        "evidence_bindings": bindings,
        "runtime_environment": runtime_environment,
        "severity_labels_created": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "formal_generator_training_submitted": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "joint_panel_authorized": False,
    }
    if selector_failures:
        report = {
            **common_report,
            "decision": PRESELECTION_NO_GO,
            "component_status": PRESELECTION_NO_GO,
            "readiness_status": READINESS_NO_GO,
            "candidate_exact_outcomes_opened": False,
            "selector_failures": selector_failures,
            "selector_coverage": (
                EXPECTED_CASES - len(selector_failures)
            )
            / EXPECTED_CASES,
            "scientific_promotion_granted": False,
            "six_component_readiness_eligible": False,
        }
        report_path = args.output_dir / "diagnostic_report.json"
        write_json(report_path, report)
        receipt_path = write_completion_receipt(
            args,
            PRESELECTION_NO_GO,
            [report_path, preselection_path],
            exact_opened=False,
            promotion_granted=False,
        )
        print(
            json.dumps(
                {
                    "decision": PRESELECTION_NO_GO,
                    "failures": selector_failures,
                    "completion_receipt_sha256": sha256_file(receipt_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    selector_seal = {
        "schema_version": SELECTOR_SEAL_SCHEMA,
        "candidate": SELECTOR_NAME,
        "source_commit": args.source_commit,
        "frozen_selector_source_commit": FROZEN_SELECTOR_SOURCE_COMMIT,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_present": False,
        "base_exact_component_outcomes_present": False,
        "selection_uses_candidate_exact_outcome": False,
        "target_scalar_is_declared_supervised_input": True,
        "severity_labels_created": False,
        "selector_contract": selector_contract(),
        "source_sha256": source_hashes,
        "evidence_bindings": bindings,
        "preselection_sha256": sha256_file(preselection_path),
        "rows": [
            {
                "case_id": record["case_id"],
                "panel_speaker_id": panel_row["panel_speaker_id"],
                "sex": panel_row["sex"],
                "view": panel_row["view"],
                "condition": panel_row["condition"],
                "selected_family": record["selected_family"],
                "selected_alpha": record["selected_alpha"],
                "selected_backtrack_index": record[
                    "selected_backtrack_index"
                ],
                "base_sha256": panel_row["base_sha256"],
                "target_sha256": panel_row["target_sha256"],
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
            for panel_row, record in zip(panel_rows, records, strict=True)
        ],
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    selector_seal_path = args.output_dir / "selector_seal.json"
    write_json(selector_seal_path, selector_seal)

    rows, exact_versions = build_exact_rows(
        args,
        panel_rows,
        target_by_case,
        records,
        target_scale,
    )
    target_versions = read_json(args.target_contract)["exact_scorer_versions"]
    if exact_versions != target_versions:
        raise ValueError("v25 target/v26 post-seal exact scorer version drift")
    results_path = args.output_dir / "external_svd_exact_results.csv"
    write_csv(results_path, rows)
    summary = summarize_external(rows)
    passed = summary["all_gates_pass"]
    decision = PASS_DECISION if passed else EXACT_NO_GO
    report = {
        **common_report,
        "decision": decision,
        "component_status": decision,
        "readiness_status": READINESS_PASS if passed else READINESS_NO_GO,
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "selector_seal_sha256": sha256_file(selector_seal_path),
        "exact_scorer_versions": exact_versions,
        "fixed_scientific_thresholds": {
            "candidate_d_fixed_alpha": FIXED_ALPHA,
            "material_normalized_gap": MATERIAL_GAP_THRESHOLD,
            "median_normalized_reduction": MEDIAN_REDUCTION_GATE,
            "improvement_fraction": IMPROVEMENT_FRACTION_GATE,
            "nonselected_median_increase": NONSELECTED_MEDIAN_INCREASE_GATE,
            "gradient_l2_range": list(GRADIENT_NORM_RANGE),
            "residual_ceiling_db": RESIDUAL_CEILING_DB,
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_clip_fraction": MAXIMUM_CLIP_FRACTION,
            "target_reproduction_abs_tolerance": (
                TARGET_REPRODUCTION_ABS_TOLERANCE
            ),
            "external_slice_threshold_source": (
                "frozen generic effect-slice rule: material present, "
                "improvement fraction >= 0.5, median reduction >= 0"
            ),
        },
        "summary": summary,
        "old_v18_evidence_kept_separate": True,
        "opened24_v23_severity_accuracy_calibration_anti_shortcut_bound": True,
        "external_speaker_gate_pass": passed,
        "bounded_waveform_promotion_pass": passed,
        "scientific_promotion_granted": passed,
        "six_component_readiness_eligible": passed,
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    receipt_path = write_completion_receipt(
        args,
        decision,
        [report_path, preselection_path, selector_seal_path, results_path],
        exact_opened=True,
        promotion_granted=passed,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "scientific_promotion_granted": passed,
                "six_component_readiness_eligible": passed,
                "exact_improvement_fraction": summary["mechanism"][
                    "exact_db_improvement_fraction"
                ],
                "median_normalized_reduction": summary["mechanism"][
                    "median_exact_db_normalized_gap_reduction"
                ],
                "completion_receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_provenance = validate_repository(args)
    panel, panel_rows, target_by_case, source_hashes, bindings = (
        validate_sources_and_inputs(args)
    )
    if panel.get("severity_labels_created") is not False:
        raise ValueError("external SVD panel severity boundary drift")
    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms"
    waveform_root.mkdir()
    device = torch.device(args.device)
    deterministic_process_contract = validate_deterministic_process_contract(
        "deterministic_repeat"
    )
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
            "deterministic_process_contract": deterministic_process_contract,
            "external_panel_does_not_claim_a_new_repeat": True,
            "torch_synthetic_warmup": torch_warmup,
            "candidate_d_synthetic_warmup": candidate_d_warmup,
            "optimized_v18_synthetic_warmup": optimized_v18_warmup,
            "worker_startups": worker_startups,
            "worker_synthetic_warmups": worker_warmups,
            "worker_count": WORKER_COUNT,
            "warmups_outside_case_timer": True,
        }
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            run_external_selector(
                args,
                panel_rows,
                target_by_case,
                source_hashes,
                bindings,
                source_provenance,
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
