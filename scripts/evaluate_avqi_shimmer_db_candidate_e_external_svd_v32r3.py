#!/usr/bin/env python3
"""Run frozen Candidate-E selector and exact external SVD adjudication.

This is the Candidate-E successor of scientific v26. The v30r2 panel and v31r2
target scalar seal are immutable inputs. Candidate selection uses only the
frozen dual-direction proxy/topology/safety/PCM24 contract. A pre-exact seal is
written before authoritative Praat components are opened. Formal generator
training remains prohibited. V32r3 changes only the pre-exact implementation
runtime and must reproduce the immutable v32r2 candidate pool and selector.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from scripts import adjudicate_avqi_shimmer_db_candidate_e_v14_v28 as v28
from scripts import adjudicate_avqi_shimmer_db_deterministic_opened24_v23 as v23
from scripts import evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r2 as v32r2
from scripts import evaluate_avqi_shimmer_db_candidate_e_opened_v15_v29 as v29
from scripts import evaluate_avqi_shimmer_hybrid_topology as hybrid
from scripts import evaluate_direct_avqi_waveform_optimization as direct
from scripts import prepare_avqi_shimmer_db_candidate_e_external_svd_panel_v30r2 as v30
from scripts import prepare_avqi_shimmer_db_external_svd_panel_v24 as v24
from scripts import seal_avqi_shimmer_db_candidate_e_external_svd_target_v31r2 as v31
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
from scripts.avqi_shimmer_peak_certificate_v19 import (
    paired_candidate_peak_certificate,
    pcm16_roundtrip_values_to_codes,
)
from scripts.diagnose_avqi_shimmer_db_candidate_e_direction_v27 import (
    CANDIDATE_E_VARIANTS,
    VARIANT_E_PROJECTED,
    VARIANT_E_RAW,
    alpha_label,
    dual_direction_selector_seal,
    impulse_certificate,
    pcm16_roundtrip,
    pulse_position_drift,
    safe_name,
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
    materialize_candidate_pcm24,
    pcm24_codes,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    ALPHA_LADDER,
    base_topology_item,
)


REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-exact-promotion-v32r3"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-receipt-v32r3"
)
SELECTOR_SEAL_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-selector-seal-v32r3"
)
PASS_DECISION = "PASS_CANDIDATE_E_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V32R3"
PREEXACT_NO_GO = "NO_GO_CANDIDATE_E_EXTERNAL_SVD_PREEXACT_V32R3"
EXACT_NO_GO = "NO_GO_CANDIDATE_E_EXTERNAL_SVD_EXACT_PROMOTION_V32R3"
READINESS_PASS = "READY_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
READINESS_NO_GO = "NO_GO_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
RUNTIME_CONFIG_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-runtime-successor-v32r3"
)
RUNTIME_WORKER_COUNT = 8
SYNTHETIC_WARMUP_FFT_LENGTHS = (
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
)
EXPECTED_CASES = 12
EXPECTED_SPEAKERS = 6
FORWARD_PARITY_ABSOLUTE_TOLERANCE = 1e-9
EXTERNAL_REQUIRED_EFFECT_SLICES = (
    "view=cs",
    "view=sv",
    "condition=rir_only",
    "condition=snr20",
    "condition=snr10",
    "sex=female",
    "sex=male",
)
EXACT_MARKER = "AVQI_CANDIDATE_E_SIX_COMPONENT_JSON="
EXACT_COMPONENT_SCORER = r"""
import json
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from avqi_code.main import (
    get_cpps,
    get_hnr,
    get_slope,
    get_tilt,
    get_voiced_segments,
    highpass_filter,
    length_normalize_sv,
    read_and_resample_signal,
)

request = json.load(sys.stdin)
rows = []
for item in request["items"]:
    signal = read_and_resample_signal(item["path"], 16000)
    highpassed = highpass_filter("praat", signal, 16000)
    if item["view"] == "sv":
        metric = length_normalize_sv("praat", highpassed, 16000)
    elif item["view"] == "cs":
        metric = get_voiced_segments("praat", highpassed, 16000)
    else:
        raise ValueError(f"unsupported view: {item['view']}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        metric_path = handle.name
    try:
        sf.write(metric_path, metric, 16000)
        sound = parselmouth.Sound(metric_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 50, 400)
        shimmer_percent = 100.0 * call(
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
        shimmer_db = call(
            [sound, point_process],
            "Get shimmer (local_dB)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
    finally:
        os.unlink(metric_path)
    rows.append(
        {
            "id": item["id"],
            "components": {
                "cpps": float(get_cpps("praat", metric, 16000)),
                "hnr": float(get_hnr("praat", metric, 16000)),
                "shimmer_percent": float(shimmer_percent),
                "shimmer_db": float(shimmer_db),
                "slope": float(get_slope("praat", metric, 16000)),
                "tilt": float(get_tilt("praat", metric, 16000)),
            },
        }
    )
print(
    "AVQI_CANDIDATE_E_SIX_COMPONENT_JSON="
    + json.dumps(
        {
            "parselmouth_version": parselmouth.__version__,
            "praat_version": parselmouth.PRAAT_VERSION,
            "rows": rows,
        },
        sort_keys=True,
    )
)
"""


def add_hashed_path(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(f"--{option}-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "config",
        "runtime-config",
        "v32r2-report",
        "v32r2-receipt",
        "v28-report",
        "v28-receipt",
        "mechanism-config",
        "mechanism-selector",
        "v29-report",
        "v29-receipt",
        "panel-seal",
        "panel-receipt",
        "updated-speaker-ledger",
        "target-contract",
        "target-receipt",
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


def repository_provenance(args: argparse.Namespace) -> dict[str, str]:
    root = args.repository_root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repository root does not contain v32r3 runner")
    head = v23.git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v32r3 repository HEAD/source commit drift")
    status = v23.git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v32r3 adjudication requires a clean repository")
    return {
        "repository_root": str(root),
        "source_commit": head,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def require_training_boundary(value: dict[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} optimizer-step boundary drift")
    if value.get("formal_generator_training_authorized") is not False:
        raise ValueError(f"{label} over-authorizes generator training")
    if value.get("authoritative_training_decision") != TRAINING_DECISION:
        raise ValueError(f"{label} training decision drift")


def validate_runtime_successor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    observed = {
        "runtime_config": v23.validate_hash(
            args.runtime_config,
            args.runtime_config_sha256,
            "v32r3 runtime successor config",
        ),
        "v32r2_report": v23.validate_hash(
            args.v32r2_report,
            args.v32r2_report_sha256,
            "v32r2 pre-exact NO_GO report",
        ),
        "v32r2_receipt": v23.validate_hash(
            args.v32r2_receipt,
            args.v32r2_receipt_sha256,
            "v32r2 pre-exact NO_GO receipt",
        ),
    }
    config = v23.read_json(args.runtime_config)
    report = v23.read_json(args.v32r2_report)
    receipt = v23.read_json(args.v32r2_receipt)
    if config.get("schema_version") != RUNTIME_CONFIG_SCHEMA:
        raise ValueError("v32r3 runtime config schema drift")
    predecessor = config.get("predecessor_no_go", {})
    expected_predecessor = {
        "job_id": "20041915",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "scientific_decision": v32r2.PREEXACT_NO_GO,
        "source_commit": "defc4deb329fc3272cc65c1ddaf9a21af670195b",
        "report_sha256": observed["v32r2_report"],
        "receipt_sha256": observed["v32r2_receipt"],
        "selector_seal_sha256": (
            "224d659dab548f8a3d7ff4cb8a1cee57b569539f2bdafd7224c8aaa2c7c9fec9"
        ),
        "candidate_attempts_sha256": (
            "348b775e224d727c8e241f3e4f135a39529c7615415cf8666146573bf1f70f3c"
        ),
        "candidate_pool_equivalence_sha256": (
            "a2d6cfc5dbff0e1d0eed74b2daa119b247fbd59fb583abcf2989918991f2a4a6"
        ),
        "stdout_sha256": (
            "b87fedadb77c352ee85f68ea6e023a15292df6c4507bfae72aabd979739dc4c4"
        ),
        "stderr_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "old_receipt_remains_immutable": True,
        "no_go_not_reinterpreted_as_pass": True,
    }
    if predecessor != expected_predecessor:
        raise ValueError("v32r3 predecessor NO_GO binding drift")
    if report.get("schema_version") != v32r2.REPORT_SCHEMA:
        raise ValueError("v32r2 predecessor report schema drift")
    if report.get("decision") != v32r2.PREEXACT_NO_GO:
        raise ValueError("v32r2 predecessor is not pre-exact NO_GO")
    if report.get("source_commit") != predecessor["source_commit"]:
        raise ValueError("v32r2 predecessor source binding drift")
    if report.get("slurm_job_id") != predecessor["job_id"]:
        raise ValueError("v32r2 predecessor job binding drift")
    expected_gates = {
        "complete_selector_coverage": True,
        "candidate_pool_frozen_serial_equivalence": True,
        "total_metric_step_runtime_le_500ms": False,
        "selector_uses_no_candidate_exact_outcome": True,
        "selector_uses_no_identity": True,
        "candidate_e_remains_frozen": True,
        "generator_optimizer_steps_zero": True,
    }
    if report.get("preexact_gates") != expected_gates:
        raise ValueError("v32r2 predecessor failure-class drift")
    if (
        report.get("candidate_exact_outcomes_opened") is not False
        or report.get("exact_scoring_complete") is not False
        or report.get("scientific_promotion_granted") is not False
    ):
        raise ValueError("v32r2 predecessor opened or promoted exact outcomes")
    if receipt.get("schema_version") != v32r2.RECEIPT_SCHEMA:
        raise ValueError("v32r2 predecessor receipt schema drift")
    if receipt.get("decision") != v32r2.PREEXACT_NO_GO:
        raise ValueError("v32r2 predecessor receipt decision drift")
    if receipt.get("artifact_sha256", {}).get(
        "external_svd_report_v32r2.json"
    ) != observed["v32r2_report"]:
        raise ValueError("v32r2 predecessor report/receipt binding drift")
    if (
        receipt.get("candidate_exact_outcomes_opened_after_selector_seal")
        is not False
        or receipt.get("exact_scoring_complete") is not False
        or receipt.get("scientific_promotion_granted") is not False
    ):
        raise ValueError("v32r2 predecessor receipt exact-opening drift")
    for label, value in (("report", report), ("receipt", receipt)):
        require_training_boundary(value, f"v32r2 predecessor {label}")

    runtime = config.get("runtime_successor", {})
    expected_runtime = {
        "failure_class": "implementation_runtime_only_before_candidate_exact",
        "worker_count": RUNTIME_WORKER_COUNT,
        "synthetic_only_warmup": True,
        "warmup_fft_lengths": list(SYNTHETIC_WARMUP_FFT_LENGTHS),
        "batched_candidate_gpu_to_cpu_transfer": True,
        "concurrent_pcm24_write_read_safety": True,
        "in_memory_float32_topology_staging": True,
        "per_case_selector_wall_time": True,
        "serial_oracle_required": True,
        "candidate_waveform_byte_equivalence_required": True,
        "current_topology_hash_equivalence_required": True,
        "current_proxy_absolute_tolerance": 1e-12,
        "selector_choice_equivalence_required": True,
        "uses_candidate_exact_outcomes": False,
        "uses_speaker_or_case_identity_for_routing": False,
    }
    if runtime != expected_runtime:
        raise ValueError("v32r3 runtime optimization contract drift")
    scientific = config.get("frozen_scientific_contract", {})
    if (
        scientific.get("direction_families") != list(CANDIDATE_E_VARIANTS)
        or scientific.get("alpha_ladder") != list(ALPHA_LADDER)
        or scientific.get("selector")
        != (
            "maximum_current_topology_proxy_gap_reduction_across_"
            "projected_and_raw_exact_path_directions"
        )
        or scientific.get("deterministic_tie_break")
        != [
            "projected_before_raw",
            "larger_alpha_before_smaller_alpha",
        ]
        or scientific.get("formal_total_metric_step_runtime_ms")
        != hybrid.CACHE_RUNTIME_MAX_MS
        or scientific.get("thresholds_unchanged") is not True
        or scientific.get("candidate_e_math_unchanged") is not True
        or scientific.get("target_scalar_contract_unchanged") is not True
        or scientific.get("exact_praat_remains_final_judge") is not True
        or scientific.get("no_final_waveform_highpass") is not True
    ):
        raise ValueError("v32r3 frozen scientific contract drift")
    boundaries = config.get("immutable_boundaries", {})
    for field in (
        "v30r2_panel_seal_unchanged",
        "v31r2_target_scalar_seal_unchanged",
        "v32r2_preexact_no_go_preserved",
        "candidate_exact_must_remain_unopened_until_new_selector_seal",
    ):
        if boundaries.get(field) is not True:
            raise ValueError(f"v32r3 immutable boundary drift: {field}")
    require_training_boundary(boundaries, "v32r3 runtime config")
    return config, observed


def validate_v29_pass(
    report: dict[str, Any],
    receipt: dict[str, Any],
    panel: dict[str, Any],
    *,
    report_sha256: str,
    receipt_sha256: str,
) -> None:
    if report.get("schema_version") != v29.REPORT_SCHEMA:
        raise ValueError("v29 report schema drift")
    if receipt.get("schema_version") != v29.RECEIPT_SCHEMA:
        raise ValueError("v29 receipt schema drift")
    for label, value in (("report", report), ("receipt", receipt)):
        if value.get("decision") != v29.PASS_DECISION:
            raise ValueError(f"v29 {label} is not PASS")
        if value.get("candidate_e_frozen") is not True and value.get(
            "candidate_e_remains_frozen"
        ) is not True:
            raise ValueError(f"v29 {label} does not retain Candidate-E freeze")
        if value.get("retuning_authorized") is not False:
            raise ValueError(f"v29 {label} over-authorizes retuning")
        if value.get("external_panel_prepare_authorized") is not True:
            raise ValueError(f"v29 {label} did not authorize external prepare")
        if value.get("external_panel_authorized") is not False:
            raise ValueError(f"v29 {label} prematurely authorizes promotion")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"v29 {label} over-authorizes joint panel")
        require_training_boundary(value, f"v29 {label}")
    if not all(report.get("gates", {}).values()):
        raise ValueError("v29 report gates did not all pass")
    if receipt.get("report_sha256") != report_sha256:
        raise ValueError("v29 receipt/report binding drift")
    authorization = panel.get("authorization", {})
    if authorization.get("v29_report_sha256") != report_sha256:
        raise ValueError("v30r2 panel/v29 report binding drift")
    if authorization.get("v29_receipt_sha256") != receipt_sha256:
        raise ValueError("v30r2 panel/v29 receipt binding drift")


def validate_updated_ledger(
    ledger: dict[str, Any],
    panel_rows: list[dict[str, Any]],
    panel_selection: dict[str, Any],
    panel_receipt: dict[str, Any],
    *,
    ledger_sha256: str,
    panel_source_commit: str,
) -> set[str]:
    if panel_receipt.get("artifact_sha256", {}).get(
        "prior_panel_speaker_ledger_after_v30r2.json"
    ) != ledger_sha256:
        raise ValueError("v30r2 receipt/updated ledger binding drift")
    speakers = v24.validate_prior_ledger(ledger)
    selected = {str(row["panel_speaker_id"]) for row in panel_rows}
    if not selected.issubset(speakers):
        raise ValueError("external SVD speakers are absent from updated ledger")
    retained = set(panel_selection.get("retained_v30_speakers", []))
    rejected = set(panel_selection.get("rejected_v30_speakers", []))
    replacements = set(panel_selection.get("replacement_speakers", []))
    if retained | replacements != selected or retained & replacements:
        raise ValueError("Candidate-E v30r2 ledger selection partition drift")
    entries = {
        str(entry.get("canonical_speaker_id")): entry
        for entry in ledger["entries"]
    }
    for speaker in retained:
        entry = entries.get(speaker, {})
        if (
            entry.get("panel_role")
            != "shimmer_db_candidate_e_external_svd_v30"
            or entry.get("candidate_e_v30r2_status")
            != "retained_in_original_recipe_slot"
        ):
            raise ValueError("Candidate-E v30r2 retained ledger entry drift")
    for speaker in rejected:
        entry = entries.get(speaker, {})
        if entry.get("candidate_e_v30r2_status") != (
            "target_component_unscorable_not_selected"
        ):
            raise ValueError("Candidate-E v30r2 rejected ledger entry drift")
    for speaker in replacements:
        entry = entries.get(speaker, {})
        if (
            entry.get("panel_role")
            != "shimmer_db_candidate_e_external_svd_v30r2"
            or entry.get("source_commit") != panel_source_commit
            or entry.get("target_component_scorability_boolean_used") is not True
            or entry.get("target_scalar_values_used") is not False
        ):
            raise ValueError("Candidate-E v30r2 replacement ledger entry drift")
    if ledger.get("added_speaker_count") != len(replacements):
        raise ValueError("Candidate-E v30r2 ledger addition count drift")
    if (
        ledger.get("target_component_scorability_boolean_used_for_selection")
        is not True
        or ledger.get("target_scalar_values_used_for_selection") is not False
    ):
        raise ValueError("Candidate-E v30r2 ledger information boundary drift")
    additions = [entries[speaker] for speaker in selected]
    if any(
        entry.get("exact_shimmer_outcomes_opened_at_ledger_update") is not False
        for entry in additions
    ):
        raise ValueError("updated ledger used external performance outcomes")
    return speakers


def validate_target_binding(
    panel_rows: list[dict[str, Any]],
    target: dict[str, Any],
    receipt: dict[str, Any],
    *,
    panel_sha256: str,
    panel_receipt_sha256: str,
    target_sha256: str,
    avqi_tree_sha256: str,
) -> dict[str, dict[str, Any]]:
    if target.get("schema_version") != v31.TARGET_SCHEMA:
        raise ValueError("v31 target schema drift")
    if receipt.get("schema_version") != v31.RECEIPT_SCHEMA:
        raise ValueError("v31 target receipt schema drift")
    if receipt.get("decision") != v31.TARGET_DECISION:
        raise ValueError("v31 target stage is not sealed")
    if target.get("source_commit") != receipt.get("source_commit"):
        raise ValueError("v31 target source binding drift")
    if target.get("panel_seal_sha256") != panel_sha256:
        raise ValueError("v31 target/panel seal binding drift")
    if target.get("panel_receipt_sha256") != panel_receipt_sha256:
        raise ValueError("v31 target/panel receipt binding drift")
    inputs = receipt.get("input_sha256", {})
    if inputs.get("panel_seal_v30r2.json") != panel_sha256:
        raise ValueError("v31r2 receipt/panel seal binding drift")
    if inputs.get("seal_receipt_v30r2.json") != panel_receipt_sha256:
        raise ValueError("v31r2 receipt/panel receipt binding drift")
    if inputs.get("avqi_code_tree") != avqi_tree_sha256:
        raise ValueError("v31 receipt/exact AVQI tree binding drift")
    if receipt.get("artifact_sha256", {}).get(
        "target_scalar_seal_v31r2.json"
    ) != target_sha256:
        raise ValueError("v31 receipt/target seal binding drift")
    boundary = {
        "role": "same_speaker_clean_pathological_target_scalar",
        "selection_or_tuning_use": False,
        "base_exact_outcomes_present": False,
        "candidate_exact_outcomes_present": False,
        "clean_target_pulse_positions_exposed_to_output_branch": False,
        "target_exact_components_retained": ["shimmer_db"],
        "severity_labels_created": False,
        "emitted_waveform_highpass": False,
        "selector_stage_authorized": True,
    }
    for field, value in boundary.items():
        if target.get(field) != value:
            raise ValueError(f"v31 target information boundary drift: {field}")
    for label, value in (("target", target), ("receipt", receipt)):
        if value.get("scientific_promotion_granted") is not False:
            raise ValueError(f"v31 {label} promotes early")
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"v31 {label} authorizes joint panel early")
        require_training_boundary(value, f"v31 {label}")
    if (
        receipt.get("target_exact_shimmer_opened") is not True
        or receipt.get("base_exact_outcomes_opened") is not False
        or receipt.get("candidate_exact_outcomes_opened") is not False
        or receipt.get("selector_stage_authorized") is not True
    ):
        raise ValueError("v31 target opening-order drift")
    target_rows = target.get("rows")
    if not isinstance(target_rows, list) or len(target_rows) != EXPECTED_CASES:
        raise ValueError("v31 target row coverage drift")
    target_by_case = {str(row.get("case_id")): row for row in target_rows}
    if set(target_by_case) != {str(row["case_id"]) for row in panel_rows}:
        raise ValueError("v31 target case coverage drift")
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
                raise ValueError(f"v31 target {field} drift: {case_id}")
        if not math.isfinite(float(target_row["exact_target_shimmer_db"])):
            raise ValueError(f"v31 target scalar is non-finite: {case_id}")
    return target_by_case


def validate_sources_and_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    _, runtime_hashes = validate_runtime_successor(args)
    _, _, freeze_hashes = v29.validate_freeze_chain(args)
    names = (
        "v29_report",
        "v29_receipt",
        "panel_seal",
        "panel_receipt",
        "updated_speaker_ledger",
        "target_contract",
        "target_receipt",
    )
    observed = {
        name: v23.validate_hash(
            getattr(args, name),
            getattr(args, f"{name}_sha256"),
            name.replace("_", " "),
        )
        for name in names
    }
    panel = v23.read_json(args.panel_seal)
    panel_receipt = v23.read_json(args.panel_receipt)
    panel_rows = v31.validate_panel_binding(
        panel,
        panel_receipt,
        panel_sha256=observed["panel_seal"],
    )
    v29_report = v23.read_json(args.v29_report)
    v29_receipt = v23.read_json(args.v29_receipt)
    validate_v29_pass(
        v29_report,
        v29_receipt,
        panel,
        report_sha256=observed["v29_report"],
        receipt_sha256=observed["v29_receipt"],
    )
    ledger = v23.read_json(args.updated_speaker_ledger)
    ledger_speakers = validate_updated_ledger(
        ledger,
        panel_rows,
        panel["selection"],
        panel_receipt,
        ledger_sha256=observed["updated_speaker_ledger"],
        panel_source_commit=str(panel["source_commit"]),
    )
    observed_tree_hash = avqi_code_tree_sha256(args.avqi_code_root)
    if observed_tree_hash != args.avqi_code_tree_sha256:
        raise ValueError("exact AVQI code-tree hash drift")
    target = v23.read_json(args.target_contract)
    target_receipt = v23.read_json(args.target_receipt)
    target_by_case = validate_target_binding(
        panel_rows,
        target,
        target_receipt,
        panel_sha256=observed["panel_seal"],
        panel_receipt_sha256=observed["panel_receipt"],
        target_sha256=observed["target_contract"],
        avqi_tree_sha256=observed_tree_hash,
    )
    for row in panel_rows:
        case_id = str(row["case_id"])
        for role in ("source", "target", "degraded", "base"):
            v23.validate_hash(
                Path(row[f"{role}_path"]),
                str(row[f"{role}_sha256"]),
                f"{role} waveform {case_id}",
            )
    source_hashes = {
        "freeze": freeze_hashes,
        "runtime_successor": runtime_hashes,
        **observed,
        "avqi_code_tree": observed_tree_hash,
    }
    bindings = {
        "v32r2_preexact_no_go_decision": v32r2.PREEXACT_NO_GO,
        "v32r2_preexact_no_go_job_id": "20041915",
        "v32r2_preexact_no_go_report_sha256": runtime_hashes[
            "v32r2_report"
        ],
        "v32r2_preexact_no_go_receipt_sha256": runtime_hashes[
            "v32r2_receipt"
        ],
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_change_class": "implementation_runtime_only_before_candidate_exact",
        "v29_decision": v29.PASS_DECISION,
        "panel_source_commit": panel["source_commit"],
        "target_source_commit": target["source_commit"],
        "panel_seal_sha256": observed["panel_seal"],
        "panel_receipt_sha256": observed["panel_receipt"],
        "target_contract_sha256": observed["target_contract"],
        "target_receipt_sha256": observed["target_receipt"],
        "updated_speaker_ledger_sha256": observed[
            "updated_speaker_ledger"
        ],
        "updated_speaker_ledger_count": len(ledger_speakers),
        "old_v23_no_go_not_reinterpreted": panel["authorization"][
            "old_v23_no_go_not_reinterpreted"
        ],
        "scientific_stage_mapping": {
            "v30r2": "v24_prepare_and_seal",
            "v31r2": "v25_target_scalar_seal",
            "v32r3": "v26_selector_and_exact_adjudication",
        },
    }
    return panel_rows, target_by_case, source_hashes, bindings


def synthetic_runtime_warmup(
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Warm only generic FFT sizes and build waveform-independent certificates."""

    certificates: dict[int, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for fft_length in SYNTHETIC_WARMUP_FFT_LENGTHS:
        started = time.perf_counter()
        sample_index = torch.arange(
            fft_length,
            dtype=torch.float64,
            device=device,
        )
        carrier = torch.sin(2.0 * math.pi * sample_index / 160.0)
        envelope = 1.0 + 0.08 * torch.sin(
            2.0 * math.pi * sample_index / 1_600.0
        )
        waveform = (0.04 * envelope * carrier).requires_grad_(True)
        pulses = torch.arange(
            80.0,
            float(fft_length - 80),
            160.0,
            dtype=torch.float64,
            device=device,
        )
        source_indices = torch.arange(
            fft_length,
            dtype=torch.long,
            device=device,
        )
        result = candidate_e_proxy(
            waveform,
            pulses,
            source_indices,
            0,
        )
        loss = (result.shimmer_db - 1.0).square()
        gradient = torch.autograd.grad(loss, waveform)[0]
        normalized_gradient_step(waveform, gradient, ALPHA_LADDER[0])
        certificate = impulse_certificate(fft_length)
        synchronize(device)
        if result.fft_sample_count != fft_length:
            raise ValueError("synthetic warmup FFT length drift")
        if int(certificate["fft_length"]) != fft_length:
            raise ValueError("synthetic certificate FFT length drift")
        certificates[fft_length] = certificate
        evidence.append(
            {
                "fft_length": fft_length,
                "synthetic_only": True,
                "panel_waveform_used": False,
                "training_waveform_used": False,
                "candidate_e_forward_finite": math.isfinite(
                    float(result.shimmer_db.detach().cpu())
                ),
                "candidate_e_gradient_finite": bool(
                    torch.isfinite(gradient).all().detach().cpu()
                ),
                "certificate_cache_key": certificate["response_cache_key"],
                "wall_ms": 1000.0 * (time.perf_counter() - started),
            }
        )
    return certificates, evidence


def refresh_waveform_chunks(
    workers: list[ExactShimmerTopologyWorker],
    executor: ThreadPoolExecutor,
    items: list[dict[str, Any]],
    waveforms: list[np.ndarray],
) -> tuple[list[dict[str, Any]], float, list[float], float]:
    if len(items) != len(waveforms):
        raise ValueError("parallel topology waveform/item coverage drift")
    item_chunks = [items[index:: len(workers)] for index in range(len(workers))]
    waveform_chunks = [
        waveforms[index:: len(workers)] for index in range(len(workers))
    ]
    started = time.perf_counter()
    futures = [
        executor.submit(
            worker.refresh_current_waveforms,
            item_chunk,
            waveform_chunk,
            NUMPY_HIGHPASS_MODE,
        )
        for worker, item_chunk, waveform_chunk in zip(
            workers,
            item_chunks,
            waveform_chunks,
            strict=True,
        )
        if item_chunk
    ]
    results = [future.result() for future in futures]
    wall_ms = 1000.0 * (time.perf_counter() - started)
    rows_by_id = {
        str(row["id"]): dict(row)
        for rows, _, _ in results
        for row in rows
    }
    expected_ids = [str(item["id"]) for item in items]
    if set(rows_by_id) != set(expected_ids):
        raise ValueError("parallel topology refresh coverage drift")
    staging_ms = sum(
        float(staging["staging_ms"])
        for _, _, staging_rows in results
        for staging in staging_rows
    )
    return (
        [rows_by_id[item_id] for item_id in expected_ids],
        wall_ms,
        [float(request_ms) for _, request_ms, _ in results],
        staging_ms,
    )


def materialize_runtime_candidate(
    context: dict[str, Any],
    entry: dict[str, Any],
    values: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    record, write_ms, read_safety_pcm_ms = materialize_candidate_pcm24(
        context,
        values,
        Path(entry["candidate_path"]),
        str(entry["item_id"]),
    )
    stored = np.asarray(record["stored_waveform"], dtype=np.float32)
    candidate_pcm16_codes = pcm16_roundtrip_values_to_codes(
        pcm16_roundtrip(stored)
    )
    peak_certificate = paired_candidate_peak_certificate(
        context["base_pcm16_codes"],
        candidate_pcm16_codes,
        context["base_highpass_timing"],
        context["stop_hann_impulse_certificate"],
    )
    safety = direct.waveform_safety(
        torch.from_numpy(context["base_values"]),
        torch.from_numpy(stored),
    )
    row = {
        "item_id": entry["item_id"],
        "case_id": entry["case_id"],
        "view": entry["view"],
        "variant": entry["variant"],
        "alpha": entry["alpha"],
        "base_path": entry["base_path"],
        "candidate_path": str(Path(entry["candidate_path"]).resolve()),
        "candidate_sha256": record["candidate_sha256"],
        "base_peak_check_mode": peak_certificate["base_peak_check_mode"],
        "base_highpass_peak_scaled": peak_certificate[
            "base_highpass_peak_scaled"
        ],
        "paired_candidate_sinc70_peak_upper_bound": peak_certificate[
            "candidate_sinc70_peak_upper_bound"
        ],
        "paired_candidate_sinc70_search_may_be_skipped": peak_certificate[
            "candidate_sinc70_search_may_be_skipped"
        ],
        "paired_peak_certificate_failure_mode": peak_certificate["failure_mode"],
        **safety,
        "_stored_waveform": stored,
    }
    return row, {
        "pcm24_write_ms": write_ms,
        "pcm24_read_safety_ms": read_safety_pcm_ms,
    }


def candidate_proxy_value(
    waveform: np.ndarray,
    topology: dict[str, Any],
    device: torch.device,
) -> float:
    values = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(
        device=device,
        dtype=torch.float64,
    )
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            topology,
            source_sample_count=values.numel(),
        ),
        dtype=torch.long,
        device=device,
    )
    pulses = values.new_tensor(topology["pulse_positions_samples"])
    with torch.inference_mode():
        result = candidate_e_proxy(
            values,
            pulses,
            source_indices,
            int(topology["metric_constant_prefix_samples"]),
        )
    return float(result.shimmer_db.detach().cpu())


def build_candidate_pool_parallel(
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
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    scale_value = float(target_scale[SHIMMER_DB_INDEX].detach().cpu())
    workers: list[ExactShimmerTopologyWorker] = []
    worker_evidence: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    base_by_case: dict[str, dict[str, Any]] = {}
    runtime_by_case: dict[str, dict[str, float]] = {}
    certificate_cache, synthetic_warmup = synthetic_runtime_warmup(device)
    for variant in CANDIDATE_E_VARIANTS:
        (waveform_root / variant).mkdir(parents=True, exist_ok=True)
    try:
        for worker_index in range(RUNTIME_WORKER_COUNT):
            worker = ExactShimmerTopologyWorker(
                args.exact_python,
                args.runtime_worker_script,
                args.avqi_code_root,
                args.avqi_code_tree_sha256,
            )
            workers.append(worker)
            warmup, warmup_ms = worker.warmup()
            worker_evidence.append(
                {
                    "worker_index": worker_index,
                    "startup_ms": worker.startup_ms,
                    "warmup_request_wall_ms": warmup_ms,
                    "startup": worker.startup,
                    "warmup": warmup,
                }
            )
        with ThreadPoolExecutor(max_workers=RUNTIME_WORKER_COUNT) as executor:
            for panel_index, panel_row in enumerate(panel_rows, start=1):
                case_id = str(panel_row["case_id"])
                case_started = time.perf_counter()
                base_float = read_waveform(Path(panel_row["base_path"]))
                base_values = np.asarray(base_float.numpy(), dtype=np.float32)
                base_items = [
                    {
                        **base_topology_item(panel_row),
                        "highpass_mode": NUMPY_HIGHPASS_MODE,
                    }
                ]
                base_rows, base_wall_ms, base_requests, base_staging_ms = (
                    refresh_waveform_chunks(
                        workers,
                        executor,
                        base_items,
                        [base_values],
                    )
                )
                topology = base_rows[0]
                base_by_case[case_id] = dict(topology)
                compute_started = time.perf_counter()
                base_float = base_float.to(device)
                source_indices = torch.as_tensor(
                    metric_source_indices_from_topology(
                        topology,
                        source_sample_count=base_float.numel(),
                    ),
                    dtype=torch.long,
                    device=device,
                )
                waveform = (
                    base_float.detach().to(dtype=torch.float64).requires_grad_(True)
                )
                pulses = waveform.new_tensor(topology["pulse_positions_samples"])
                proxy = candidate_e_proxy(
                    waveform,
                    pulses,
                    source_indices,
                    int(topology["metric_constant_prefix_samples"]),
                )
                target = float(target_by_case[case_id]["exact_target_shimmer_db"])
                loss = ((proxy.shimmer_db - target) / scale_value).square()
                raw = torch.autograd.grad(loss, waveform)[0]
                plan = build_zero_crossing_cycle_plan_vectorized(
                    base_float.detach().cpu().numpy(),
                    topology,
                )
                projected, projection = project_cycle_gain_gradient_fixed_order(
                    waveform,
                    raw,
                    plan,
                )
                if not projection["projected_gradient_valid"]:
                    raise ValueError(f"Candidate-E projection invalid: {case_id}")
                highpass_timing = dict(topology["timing_ms"])
                if highpass_timing.get("highpass_peak_scaled") is not False:
                    raise ValueError(f"Candidate-E exact highpass scaled: {case_id}")
                base_codes = pcm16_roundtrip_values_to_codes(
                    pcm16_roundtrip(base_values)
                )
                stop_hann = certificate_cache.get(proxy.fft_sample_count)
                if stop_hann is None:
                    raise ValueError(
                        f"unregistered Candidate-E FFT length: {proxy.fft_sample_count}"
                    )
                case_rows: list[dict[str, Any]] = []
                directions = {
                    VARIANT_E_PROJECTED: projected,
                    VARIANT_E_RAW: raw,
                }
                candidate_tensors: list[torch.Tensor] = []
                entries: list[dict[str, Any]] = []
                for variant, direction in directions.items():
                    for alpha in (0.0, *ALPHA_LADDER):
                        candidate = normalized_gradient_step(
                            waveform,
                            direction,
                            alpha,
                        )
                        candidate_tensors.append(candidate)
                        item_id = f"{case_id}:{variant}:{alpha_label(alpha)}"
                        path = waveform_root / variant / (
                            f"{safe_name(case_id)}__{variant}__"
                            f"{alpha_label(alpha)}.wav"
                        )
                        entries.append(
                            {
                                "item_id": item_id,
                                "case_id": case_id,
                                "view": str(panel_row["view"]),
                                "variant": variant,
                                "alpha": alpha,
                                "base_path": str(
                                    Path(panel_row["base_path"]).resolve()
                                ),
                                "candidate_path": str(path.resolve()),
                            }
                        )
                candidate_batch = torch.stack(candidate_tensors)
                if not bool(torch.isfinite(candidate_batch).all().detach().cpu()):
                    raise ValueError(f"non-finite Candidate-E step: {case_id}")
                if float(candidate_batch.detach().abs().max().cpu()) >= 0.999:
                    raise ValueError(f"Candidate-E step clips: {case_id}")
                synchronize(device)
                candidate_compute_ms = 1000.0 * (
                    time.perf_counter() - compute_started
                )
                transfer_started = time.perf_counter()
                candidate_values = candidate_batch.detach().cpu().numpy()
                candidate_transfer_ms = 1000.0 * (
                    time.perf_counter() - transfer_started
                )
                materialize_context = {
                    "base_values": base_values,
                    "base_codes": pcm24_codes(base_values),
                    "base_sha256": str(panel_row["base_sha256"]),
                    "base_pcm16_codes": base_codes,
                    "base_highpass_timing": highpass_timing,
                    "stop_hann_impulse_certificate": stop_hann,
                }
                io_started = time.perf_counter()
                materialize_futures = [
                    executor.submit(
                        materialize_runtime_candidate,
                        materialize_context,
                        entry,
                        values,
                    )
                    for entry, values in zip(
                        entries,
                        candidate_values,
                        strict=True,
                    )
                ]
                materialized = [future.result() for future in materialize_futures]
                candidate_io_wall_ms = 1000.0 * (
                    time.perf_counter() - io_started
                )
                case_rows = [row for row, _ in materialized]
                frozen_proxy_started = time.perf_counter()
                for row in case_rows:
                    row["proxy_shimmer_db"] = candidate_proxy_value(
                        row["_stored_waveform"],
                        topology,
                        device,
                    )
                synchronize(device)
                frozen_proxy_ms = 1000.0 * (
                    time.perf_counter() - frozen_proxy_started
                )
                current_items = [
                    {
                        "id": f"current_topology:{row['item_id']}",
                        "case_id": case_id,
                        "role": "current_output_topology",
                        "path": row["candidate_path"],
                        "view": row["view"],
                        "score_components": False,
                        "exact_metric_topology": True,
                        "highpass_mode": NUMPY_HIGHPASS_MODE,
                    }
                    for row in case_rows
                ]
                (
                    current_rows,
                    current_wall_ms,
                    current_requests,
                    current_staging_ms,
                ) = refresh_waveform_chunks(
                    workers,
                    executor,
                    current_items,
                    [row["_stored_waveform"] for row in case_rows],
                )
                proxy_started = time.perf_counter()
                for candidate, current in zip(
                    case_rows,
                    current_rows,
                    strict=True,
                ):
                    current_proxy = candidate_proxy_value(
                        candidate["_stored_waveform"],
                        current,
                        device,
                    )
                    candidate.update(
                        {
                            "current_topology_proxy_shimmer_db": current_proxy,
                            "current_topology_sha256": topology_sha256(current),
                            "current_topology_pulse_count": int(
                                current["pulse_count"]
                            ),
                            **topology_stability(topology, current),
                            **pulse_position_drift(topology, current),
                        }
                    )
                    candidate.pop("_stored_waveform")
                synchronize(device)
                current_proxy_ms = 1000.0 * (
                    time.perf_counter() - proxy_started
                )
                total_ms = 1000.0 * (time.perf_counter() - case_started)
                runtime_by_case[case_id] = {
                    "base_topology_refresh_ms": base_wall_ms,
                    "base_topology_client_staging_ms": base_staging_ms,
                    "candidate_gradient_projection_and_batch_ms": (
                        candidate_compute_ms
                    ),
                    "candidate_batched_gpu_to_cpu_transfer_ms": (
                        candidate_transfer_ms
                    ),
                    "candidate_concurrent_pcm24_io_safety_ms": (
                        candidate_io_wall_ms
                    ),
                    "candidate_frozen_topology_proxy_ms": frozen_proxy_ms,
                    "candidate_materialize_and_proxy_ms": (
                        candidate_compute_ms
                        + candidate_transfer_ms
                        + candidate_io_wall_ms
                        + frozen_proxy_ms
                    ),
                    "candidate_current_topology_refresh_ms": current_wall_ms,
                    "candidate_current_topology_client_staging_ms": (
                        current_staging_ms
                    ),
                    "candidate_current_proxy_ms": current_proxy_ms,
                    "selector_runtime_per_case_ms": 0.0,
                    "total_metric_step_runtime_ms": total_ms,
                    "worker_request_sum_ms_diagnostic": sum(
                        base_requests + current_requests
                    ),
                    "pcm24_write_sum_ms_diagnostic": sum(
                        timing["pcm24_write_ms"]
                        for _, timing in materialized
                    ),
                    "pcm24_read_safety_sum_ms_diagnostic": sum(
                        timing["pcm24_read_safety_ms"]
                        for _, timing in materialized
                    ),
                }
                diagnostics.append(
                    {
                        "case_id": case_id,
                        "speaker_id": panel_row["speaker_id"],
                        "view": panel_row["view"],
                        "condition": panel_row["condition"],
                        "target_shimmer_db": target,
                        "candidate_e_proxy_before": float(proxy.shimmer_db.detach()),
                        "candidate_e_raw_gradient_l2": float(raw.norm().detach()),
                        "candidate_e_projection": projection,
                        "candidate_e_peak_scale_abstention_pass": (
                            proxy.peak_scale_abstention_pass
                        ),
                        "candidate_e_fft_sample_count": proxy.fft_sample_count,
                        "base_topology_sha256": topology_sha256(topology),
                    }
                )
                candidate_rows.extend(case_rows)
                print(
                    f"candidate_e_external_preselection={panel_index}/{EXPECTED_CASES}",
                    flush=True,
                )
    finally:
        for worker in workers:
            worker.close()
    environment = {
        "worker_count": RUNTIME_WORKER_COUNT,
        "worker_startups_and_warmups": worker_evidence,
        "synthetic_candidate_e_warmup": synthetic_warmup,
        "synthetic_only_warmup": True,
        "warmups_outside_case_timer": True,
        "batched_candidate_gpu_to_cpu_transfer": True,
        "concurrent_pcm24_write_read_safety": True,
        "in_memory_float32_topology_staging": True,
        "per_case_candidate_refresh_split_across_workers": True,
        "per_case_selector_wall_time": True,
        "serial_oracle_excluded_from_runtime_gate": True,
        "runtime_gate_ms": hybrid.CACHE_RUNTIME_MAX_MS,
    }
    return candidate_rows, diagnostics, base_by_case, runtime_by_case, environment


def seal_selector_per_case(
    candidate_rows: list[dict[str, Any]],
    target_by_case: dict[str, float],
    scale_value: float,
    runtime_by_case: dict[str, dict[str, float]],
) -> dict[str, Any]:
    selector_metadata: dict[str, Any] | None = None
    selected_rows: list[dict[str, Any]] = []
    for case_id in sorted(target_by_case):
        case_rows = [
            row for row in candidate_rows if str(row["case_id"]) == case_id
        ]
        started = time.perf_counter()
        case_selector = dual_direction_selector_seal(
            case_rows,
            {case_id: target_by_case[case_id]},
            scale_value,
        )
        selector_ms = 1000.0 * (time.perf_counter() - started)
        if len(case_selector["rows"]) != 1:
            raise ValueError(f"per-case selector coverage drift: {case_id}")
        metadata = {
            key: value
            for key, value in case_selector.items()
            if key != "rows"
        }
        if selector_metadata is None:
            selector_metadata = metadata
        elif metadata != selector_metadata:
            raise ValueError("per-case selector metadata drift")
        selected_rows.extend(case_selector["rows"])
        runtime_by_case[case_id]["selector_runtime_per_case_ms"] = selector_ms
        runtime_by_case[case_id]["total_metric_step_runtime_ms"] += selector_ms
    if selector_metadata is None:
        raise ValueError("per-case selector received no cases")
    return {**selector_metadata, "rows": selected_rows}


def compare_candidate_pools(
    optimized: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    optimized_selector: dict[str, Any],
    reference_selector: dict[str, Any],
) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[str, str, float]:
        return str(row["case_id"]), str(row["variant"]), float(row["alpha"])

    optimized_by_key = {key(row): row for row in optimized}
    reference_by_key = {key(row): row for row in reference}
    if set(optimized_by_key) != set(reference_by_key):
        raise ValueError("optimized/reference Candidate-E grid coverage drift")
    mismatches = []
    maximum_proxy_error = 0.0
    for item_key in sorted(optimized_by_key):
        candidate = optimized_by_key[item_key]
        oracle = reference_by_key[item_key]
        proxy_error = abs(
            float(candidate["current_topology_proxy_shimmer_db"])
            - float(oracle["current_topology_proxy_shimmer_db"])
        )
        maximum_proxy_error = max(maximum_proxy_error, proxy_error)
        checks = {
            "candidate_sha256": (
                candidate["candidate_sha256"] == oracle["candidate_sha256"]
            ),
            "current_topology_sha256": (
                candidate["current_topology_sha256"]
                == oracle["current_topology_sha256"]
            ),
            "topology_stability_pass": (
                candidate["topology_stability_pass"]
                == oracle["topology_stability_pass"]
            ),
            "proxy_absolute_error": proxy_error <= 1e-12,
        }
        if not all(checks.values()):
            mismatches.append({"key": item_key, "checks": checks})

    def selected(selector: dict[str, Any]) -> dict[str, tuple[str, float, str]]:
        output = {}
        for row in selector["rows"]:
            choice = row.get("selected")
            if isinstance(choice, dict):
                output[str(row["case_id"])] = (
                    str(choice["direction_family"]),
                    float(choice["alpha"]),
                    str(choice["candidate_sha256"]),
                )
        return output

    selector_equal = selected(optimized_selector) == selected(reference_selector)
    return {
        "candidate_grid_row_count": len(optimized_by_key),
        "candidate_grid_waveform_byte_equal": not any(
            not row["checks"]["candidate_sha256"] for row in mismatches
        ),
        "candidate_grid_topology_hash_equal": not any(
            not row["checks"]["current_topology_sha256"]
            for row in mismatches
        ),
        "maximum_current_topology_proxy_absolute_error": maximum_proxy_error,
        "proxy_tolerance": 1e-12,
        "selector_choice_equal": selector_equal,
        "mismatches": mismatches,
        "all_equal": not mismatches and selector_equal,
        "candidate_exact_outcomes_used": False,
    }


def run_exact_components(
    items: list[dict[str, Any]],
    exact_python: Path,
    avqi_code_root: Path,
) -> dict[str, Any]:
    request_items = [
        {
            "id": item["id"],
            "path": item["path"],
            "view": item["view"],
        }
        for item in items
    ]
    completed = subprocess.run(
        [str(exact_python), "-c", EXACT_COMPONENT_SCORER, str(avqi_code_root)],
        input=json.dumps({"items": request_items}, ensure_ascii=False),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "six-component exact scorer failed: " + completed.stderr[-4000:]
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(EXACT_MARKER)
    ]
    if len(lines) != 1:
        raise RuntimeError("six-component exact scorer marker drift")
    payload = json.loads(lines[0][len(EXACT_MARKER) :])
    expected_ids = [str(item["id"]) for item in request_items]
    observed_ids = [str(row["id"]) for row in payload.get("rows", [])]
    if observed_ids != expected_ids:
        raise ValueError("six-component exact scorer coverage/order drift")
    for row in payload["rows"]:
        components = row.get("components", {})
        if set(components) != set(AVQI_COMPONENT_NAMES):
            raise ValueError(f"six-component exact field drift: {row['id']}")
        if not all(math.isfinite(float(value)) for value in components.values()):
            raise ValueError(f"non-finite exact component: {row['id']}")
    return payload


def postseal_topologies(
    args: argparse.Namespace,
    panel_rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in panel_rows:
        case_id = str(row["case_id"])
        for role, path in (
            ("base", row["base_path"]),
            ("candidate", selected[case_id]["candidate_path"]),
        ):
            items.append(
                {
                    "id": f"{role}:{case_id}",
                    "case_id": case_id,
                    "role": "current_output_topology",
                    "path": str(Path(path).resolve()),
                    "view": row["view"],
                    "score_components": False,
                    "exact_metric_topology": True,
                    "highpass_mode": NUMPY_HIGHPASS_MODE,
                }
            )
    worker = ExactShimmerTopologyWorker(
        args.exact_python,
        args.runtime_worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    )
    try:
        rows, _ = worker.refresh(items)
    finally:
        worker.close()
    output = {str(row["id"]): dict(row) for row in rows}
    if set(output) != {str(item["id"]) for item in items}:
        raise ValueError("post-seal exact topology coverage drift")
    return output


def build_external_result_rows(
    panel_rows: list[dict[str, Any]],
    target_by_case: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    exact_payload: dict[str, Any],
    topology_by_id: dict[str, dict[str, Any]],
    base_preselection: dict[str, dict[str, Any]],
    runtime_by_case: dict[str, dict[str, float]],
    target_scale: np.ndarray,
) -> list[dict[str, Any]]:
    exact = {str(row["id"]): dict(row) for row in exact_payload["rows"]}
    for panel_row in panel_rows:
        case_id = str(panel_row["case_id"])
        exact[f"base:{case_id}"].update(topology_by_id[f"base:{case_id}"])
        exact[f"candidate:{case_id}"].update(
            topology_by_id[f"candidate:{case_id}"]
        )
    compatibility_rows = [
        {**row, "sample_group": "external_svd_severity_not_available"}
        for row in panel_rows
    ]
    rows = v28.build_result_rows(
        compatibility_rows,
        target_by_case,
        selected,
        exact,
        target_scale,
    )
    panel_by_case = {str(row["case_id"]): row for row in panel_rows}
    for row in rows:
        case_id = str(row["case_id"])
        panel = panel_by_case[case_id]
        row.pop("sample_group", None)
        row["opened_panel"] = "external_svd_v30r2_v31r2_v32r3"
        row["opened_role"] = "result_blind_external_validation"
        row["dataset"] = "SVD"
        row["panel_speaker_id"] = panel["panel_speaker_id"]
        row["sex"] = panel["sex"]
        row["severity_label_present"] = False
        row["selector_uses_no_candidate_exact_outcome"] = True
        row["selector_pass"] = True
        row["emitted_waveform_highpass"] = False
        row["exact_metric_highpass_branch_only"] = True
        row["pcm24_effective_step_pass"] = selected[case_id][
            "pcm24_effective_step_pass"
        ]
        row["base_topology_rebound"] = (
            topology_sha256(topology_by_id[f"base:{case_id}"])
            == topology_sha256(base_preselection[case_id])
        )
        row["selected_topology_rebound"] = (
            topology_sha256(topology_by_id[f"candidate:{case_id}"])
            == selected[case_id]["current_topology_sha256"]
        )
        row["total_metric_step_runtime_ms"] = runtime_by_case[case_id][
            "total_metric_step_runtime_ms"
        ]
        row["runtime_gate_pass"] = (
            row["total_metric_step_runtime_ms"]
            <= hybrid.CACHE_RUNTIME_MAX_MS
        )
        row["candidate_pool_equivalence_pass"] = True
        if "severity" in row or "sample_group" in row:
            raise ValueError(f"external SVD severity leakage: {case_id}")
    return rows


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
        name: hybrid.summarize_effect_slice(
            [row for row in rows if predicate(row)]
        )
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
    material = [row for row in rows if row["material_shimmer_db_gap"]]
    reductions = [
        float(row["exact_normalized_gap_reduction_shimmer_db"])
        for row in material
    ]
    improvement_fraction = (
        sum(value > 0.0 for value in reductions) / len(reductions)
        if reductions
        else 0.0
    )
    median_reduction = median(reductions) if reductions else None
    nonselected = {
        name: median(
            -float(row[f"exact_normalized_gap_reduction_{name}"])
            for row in rows
        )
        for name in AVQI_COMPONENT_NAMES
        if name != "shimmer_db"
    }
    slices = external_effect_slices(rows)
    pathology = direct.aggregate_pathology_guardrails(rows)
    denoising = direct.aggregate_denoising(rows)
    maximum_forward_error = max(
        abs(
            float(row["selector_proxy_after_shimmer_db"])
            - float(row["exact_after_shimmer_db"])
        )
        for row in rows
    )
    mechanism_gates = {
        "complete_external_12case_coverage": len(rows) == EXPECTED_CASES,
        "material_cases_ge_5": len(material) >= 5,
        "overall_exact_db_effect": (
            bool(reductions)
            and median_reduction is not None
            and median_reduction >= hybrid.MEDIAN_REDUCTION_GATE
            and improvement_fraction >= hybrid.IMPROVEMENT_FRACTION_GATE
        ),
        "external_required_effect_slices": slices["decision"] == "PASS",
        "gradient": all(
            row["frozen_gradient_finite"]
            and hybrid.GRADIENT_NORM_RANGE[0]
            <= float(row["frozen_gradient_l2_norm"])
            <= hybrid.GRADIENT_NORM_RANGE[1]
            for row in rows
        ),
        "total_metric_step_runtime": all(
            row["runtime_gate_pass"] for row in rows
        ),
        "nonselected": all(
            value <= hybrid.NONSELECTED_MEDIAN_INCREASE_GATE
            for value in nonselected.values()
        ),
        "waveform_safety": all(
            float(row["residual_rms_db"]) <= hybrid.RESIDUAL_CEILING_DB
            and float(row["cosine_similarity"]) >= hybrid.MINIMUM_COSINE
            and float(row["clip_fraction"]) <= hybrid.MAXIMUM_CLIP_FRACTION
            for row in rows
        ),
        "exact_topology_stability": all(
            bool(row["topology_stability_pass"]) for row in rows
        ),
        "candidate_e_forward_parity": (
            maximum_forward_error <= FORWARD_PARITY_ABSOLUTE_TOLERANCE
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
        "candidate_pool_frozen_serial_equivalence": all(
            row["candidate_pool_equivalence_pass"] for row in rows
        ),
        "selected_topology_rebound": all(
            row["selected_topology_rebound"] for row in rows
        ),
        "base_topology_rebound": all(row["base_topology_rebound"] for row in rows),
        "pcm24_effective_step": all(
            row["pcm24_effective_step_pass"] for row in rows
        ),
        "target_topology_not_used": all(
            row["clean_target_topology_drives_output"] is False for row in rows
        ),
        "exact_metric_mapping_parity": all(
            int(row["base_metric_reconstruction_max_pcm16_error"]) == 0
            and int(row["base_metric_reconstruction_differing_samples"]) == 0
            and int(row["candidate_metric_reconstruction_max_pcm16_error"]) == 0
            and int(row["candidate_metric_reconstruction_differing_samples"]) == 0
            for row in rows
        ),
        "target_exact_reproduction": all(
            row["target_reproduction_pass"] for row in rows
        ),
        "full_band_emission": all(
            row["emitted_waveform_highpass"] is False for row in rows
        ),
        "external_speaker_sex_view_condition_coverage": (
            len({row["panel_speaker_id"] for row in rows}) == EXPECTED_SPEAKERS
            and Counter(row["sex"] for row in rows)
            == Counter({"female": 6, "male": 6})
            and Counter(row["view"] for row in rows)
            == Counter({"cs": 6, "sv": 6})
            and Counter(row["condition"] for row in rows)
            == Counter({"rir_only": 4, "snr20": 4, "snr10": 4})
        ),
        "severity_not_invented_on_svd": all(
            row["severity_label_present"] is False
            and "severity" not in row
            and "sample_group" not in row
            for row in rows
        ),
    }
    return {
        "rows": len(rows),
        "material_rows": len(material),
        "median_exact_db_normalized_gap_reduction": median_reduction,
        "exact_db_improvement_fraction": improvement_fraction,
        "nonselected_median_normalized_gap_increase": nonselected,
        "external_effect_slices": slices,
        "pathology_guardrails": pathology,
        "denoising": denoising,
        "mechanism_gates": mechanism_gates,
        "integration_gates": integration_gates,
        "maximum_forward_absolute_error_shimmer_db": maximum_forward_error,
        "total_metric_step_runtime_ms": {
            "median": median(
                float(row["total_metric_step_runtime_ms"]) for row in rows
            ),
            "maximum": max(
                float(row["total_metric_step_runtime_ms"]) for row in rows
            ),
            "formal_gate_ms": hybrid.CACHE_RUNTIME_MAX_MS,
        },
        "all_gates_pass": all(integration_gates.values()),
    }


def write_receipt(
    args: argparse.Namespace,
    decision: str,
    artifacts: list[Path],
    *,
    exact_opened: bool,
    exact_scoring_complete: bool,
    promotion_granted: bool,
) -> Path:
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": decision,
        "component": "shimmer_db",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate_exact_outcomes_opened_after_selector_seal": exact_opened,
        "exact_scoring_complete": exact_scoring_complete,
        "result_blind_external_three_stage_chain_complete": promotion_granted,
        "old_v23_no_go_preserved": True,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "scientific_promotion_granted": promotion_granted,
        "six_component_readiness_eligible": promotion_granted,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "artifact_sha256": {path.name: sha256_file(path) for path in artifacts},
    }
    path = args.output_dir / "completion_receipt_v32r3.json"
    v23.write_json(path, receipt)
    return path


def preexact_no_go(
    args: argparse.Namespace,
    source_provenance: dict[str, str],
    source_hashes: dict[str, Any],
    bindings: dict[str, Any],
    selector_path: Path,
    attempts_path: Path,
    equivalence_path: Path,
    preexact_gates: dict[str, bool],
    runtime_by_case: dict[str, dict[str, float]],
) -> None:
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": PREEXACT_NO_GO,
        "component": "shimmer_db",
        "component_status": PREEXACT_NO_GO,
        "readiness_status": READINESS_NO_GO,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "source_sha256": source_hashes,
        "evidence_bindings": bindings,
        "preexact_gates": preexact_gates,
        "runtime_by_case": runtime_by_case,
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "result_blind_external_three_stage_chain_complete": False,
        "old_v23_no_go_preserved": True,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "scientific_promotion_granted": False,
        "six_component_readiness_eligible": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    report_path = args.output_dir / "external_svd_report_v32r3.json"
    v23.write_json(report_path, report)
    receipt = write_receipt(
        args,
        PREEXACT_NO_GO,
        [report_path, selector_path, attempts_path, equivalence_path],
        exact_opened=False,
        exact_scoring_complete=False,
        promotion_granted=False,
    )
    print(
        json.dumps(
            {
                "decision": PREEXACT_NO_GO,
                "candidate_exact_outcomes_opened": False,
                "preexact_gates": preexact_gates,
                "completion_receipt_sha256": sha256_file(receipt),
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
    if not args.exact_python.is_file():
        raise FileNotFoundError(args.exact_python)
    source_provenance = repository_provenance(args)
    panel_rows, target_by_case, source_hashes, bindings = (
        validate_sources_and_inputs(args)
    )
    args.output_dir.mkdir(parents=True)
    waveform_root = args.output_dir / "waveforms_optimized"
    oracle_root = args.output_dir / "waveforms_frozen_serial_oracle"
    waveform_root.mkdir()
    oracle_root.mkdir()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v32r3 requires an allocated CUDA device")
    predictor, _, _, target_scale_tensor = load_predictor(
        args.predictor_checkpoint,
        device,
    )
    target_scale = target_scale_tensor.detach().cpu().numpy().astype(np.float64)
    started = time.perf_counter()
    (
        candidate_rows,
        diagnostics,
        base_by_case,
        runtime_by_case,
        runtime_environment,
    ) = build_candidate_pool_parallel(
        args,
        panel_rows,
        target_by_case,
        predictor,
        target_scale_tensor,
        device,
        waveform_root,
    )
    target_scalar = {
        case_id: float(row["exact_target_shimmer_db"])
        for case_id, row in target_by_case.items()
    }
    selector = seal_selector_per_case(
        candidate_rows,
        target_scalar,
        float(target_scale[SHIMMER_DB_INDEX]),
        runtime_by_case,
    )

    (
        oracle_rows,
        _,
        _,
        _,
    ) = v29.build_candidate_pool(
        args,
        panel_rows,
        target_by_case,
        predictor,
        target_scale_tensor,
        device,
        oracle_root,
    )
    oracle_selector = dual_direction_selector_seal(
        oracle_rows,
        target_scalar,
        float(target_scale[SHIMMER_DB_INDEX]),
    )
    equivalence = compare_candidate_pools(
        candidate_rows,
        oracle_rows,
        selector,
        oracle_selector,
    )
    equivalence_path = args.output_dir / "candidate_pool_equivalence_v32r3.json"
    v23.write_json(equivalence_path, equivalence)
    attempts_path = args.output_dir / "candidate_e_attempts_pre_exact_v32r3.csv"
    v23.write_csv(attempts_path, candidate_rows)
    selector_envelope = {
        "schema_version": SELECTOR_SEAL_SCHEMA,
        "scientific_stage_mapping": "v26_selector_and_exact_adjudication",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "frozen_selector": selector,
        "candidate_exact_outcomes_present": False,
        "candidate_exact_outcomes_used_for_selection": False,
        "speaker_or_case_identity_used_for_routing": False,
        "target_scalar_is_declared_supervised_input": True,
        "severity_labels_created": False,
        "candidate_pool_equivalence_sha256": sha256_file(equivalence_path),
        "candidate_attempts_sha256": sha256_file(attempts_path),
        "runtime_by_case": runtime_by_case,
        "runtime_environment": runtime_environment,
        "source_sha256": source_hashes,
        "evidence_bindings": bindings,
        "scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    selector_path = args.output_dir / "selector_seal_pre_exact_v32r3.json"
    v23.write_json(selector_path, selector_envelope)
    selector_coverage = all(
        isinstance(row.get("selected"), dict) for row in selector["rows"]
    ) and len(selector["rows"]) == EXPECTED_CASES
    preexact_gates = {
        "complete_selector_coverage": selector_coverage,
        "candidate_pool_frozen_serial_equivalence": equivalence["all_equal"],
        "total_metric_step_runtime_le_500ms": all(
            timing["total_metric_step_runtime_ms"]
            <= hybrid.CACHE_RUNTIME_MAX_MS
            for timing in runtime_by_case.values()
        ),
        "selector_uses_no_candidate_exact_outcome": (
            selector["candidate_exact_outcomes_present"] is False
            and selector["candidate_exact_outcomes_used_for_selection"] is False
        ),
        "selector_uses_no_identity": (
            selector["speaker_or_case_identity_used_for_routing"] is False
        ),
        "candidate_e_remains_frozen": True,
        "generator_optimizer_steps_zero": True,
    }
    if not all(preexact_gates.values()):
        preexact_no_go(
            args,
            source_provenance,
            source_hashes,
            bindings,
            selector_path,
            attempts_path,
            equivalence_path,
            preexact_gates,
            runtime_by_case,
        )
        return

    selected = v29.selected_candidates(selector, candidate_rows, diagnostics)
    if len(selected) != EXPECTED_CASES:
        raise ValueError("v32r3 selected candidate coverage drift")
    selector_choice = {
        str(row["case_id"]): row["selected"] for row in selector["rows"]
    }
    for case_id, choice in selected.items():
        routing = selector_choice[case_id]
        if not isinstance(routing, dict):
            raise ValueError(f"v32r3 selector unexpectedly abstained: {case_id}")
        choice["pcm24_effective_step_pass"] = bool(
            routing["pcm24_effective_step_pass"]
        )
        v23.validate_hash(
            Path(choice["candidate_path"]),
            str(choice["candidate_sha256"]),
            f"sealed Candidate-E waveform {case_id}",
        )
    exact_items = v28.build_exact_items(panel_rows, selected)
    try:
        exact_payload = run_exact_components(
            exact_items,
            args.exact_python,
            args.avqi_code_root,
        )
    except (RuntimeError, ValueError) as error:
        report = {
            "schema_version": REPORT_SCHEMA,
            "decision": EXACT_NO_GO,
            "component": "shimmer_db",
            "component_status": EXACT_NO_GO,
            "readiness_status": READINESS_NO_GO,
            "phase": "result_blind_external_svd_post_selector_exact_failure",
            "source_commit": args.source_commit,
            "slurm_job_id": args.slurm_job_id,
            "source_provenance": source_provenance,
            "source_sha256": source_hashes,
            "evidence_bindings": bindings,
            "selector_seal_pre_exact_sha256": sha256_file(selector_path),
            "candidate_attempts_pre_exact_sha256": sha256_file(attempts_path),
            "candidate_pool_equivalence_sha256": sha256_file(equivalence_path),
            "candidate_exact_outcomes_opened_after_selector_seal": True,
            "exact_scoring_complete": False,
            "result_blind_external_three_stage_chain_complete": False,
            "old_v23_no_go_preserved": True,
            "exact_failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "preexact_gates": preexact_gates,
            "runtime_by_case": runtime_by_case,
            "candidate_e_frozen": True,
            "retuning_authorized": False,
            "scientific_promotion_granted": False,
            "six_component_readiness_eligible": False,
            "joint_panel_authorized": False,
            "generator_optimizer_steps": 0,
            "formal_generator_training_submitted": False,
            "formal_generator_training_authorized": False,
            "authoritative_training_decision": TRAINING_DECISION,
        }
        report_path = args.output_dir / "external_svd_report_v32r3.json"
        v23.write_json(report_path, report)
        receipt = write_receipt(
            args,
            EXACT_NO_GO,
            [report_path, selector_path, attempts_path, equivalence_path],
            exact_opened=True,
            exact_scoring_complete=False,
            promotion_granted=False,
        )
        print(
            json.dumps(
                {
                    "decision": EXACT_NO_GO,
                    "exact_scoring_complete": False,
                    "exact_failure_type": type(error).__name__,
                    "completion_receipt_sha256": sha256_file(receipt),
                    "generator_optimizer_steps": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    target_versions = v23.read_json(args.target_contract)["exact_scorer_versions"]
    exact_versions = {
        "parselmouth": exact_payload["parselmouth_version"],
        "praat": exact_payload["praat_version"],
    }
    if exact_versions != target_versions:
        raise ValueError("v31r2 target/v32r3 exact scorer version drift")
    topology_by_id = postseal_topologies(args, panel_rows, selected)
    result_rows = build_external_result_rows(
        panel_rows,
        target_by_case,
        selected,
        exact_payload,
        topology_by_id,
        base_by_case,
        runtime_by_case,
        target_scale,
    )
    results_path = args.output_dir / "external_svd_exact_results_v32r3.csv"
    v23.write_csv(results_path, result_rows)
    summary = summarize_external(result_rows)
    passed = summary["all_gates_pass"]
    decision = PASS_DECISION if passed else EXACT_NO_GO
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": decision,
        "component": "shimmer_db",
        "component_status": decision,
        "readiness_status": READINESS_PASS if passed else READINESS_NO_GO,
        "phase": "result_blind_external_svd_post_target_seal",
        "scientific_stage_mapping": "v26_selector_and_exact_adjudication",
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "source_provenance": source_provenance,
        "source_sha256": source_hashes,
        "evidence_bindings": bindings,
        "selector_seal_pre_exact_sha256": sha256_file(selector_path),
        "candidate_attempts_pre_exact_sha256": sha256_file(attempts_path),
        "candidate_pool_equivalence_sha256": sha256_file(equivalence_path),
        "candidate_exact_outcomes_opened_after_selector_seal": True,
        "exact_scoring_complete": True,
        "result_blind_external_three_stage_chain_complete": passed,
        "old_v23_no_go_preserved": True,
        "exact_scorer_versions": exact_versions,
        "fixed_scientific_thresholds": {
            "alpha_ladder": list(ALPHA_LADDER),
            "material_normalized_gap": hybrid.MATERIAL_GAP_THRESHOLD,
            "median_normalized_reduction": hybrid.MEDIAN_REDUCTION_GATE,
            "improvement_fraction": hybrid.IMPROVEMENT_FRACTION_GATE,
            "nonselected_median_increase": (
                hybrid.NONSELECTED_MEDIAN_INCREASE_GATE
            ),
            "gradient_l2_range": list(hybrid.GRADIENT_NORM_RANGE),
            "residual_ceiling_db": hybrid.RESIDUAL_CEILING_DB,
            "minimum_cosine": hybrid.MINIMUM_COSINE,
            "maximum_clip_fraction": hybrid.MAXIMUM_CLIP_FRACTION,
            "runtime_gate_ms": hybrid.CACHE_RUNTIME_MAX_MS,
            "target_reproduction_abs_tolerance": (
                v23.TARGET_REPRODUCTION_ABS_TOLERANCE
            ),
            "external_slice_threshold_source": (
                "frozen generic effect-slice rule: material present, "
                "improvement fraction >= 0.5, median reduction >= 0"
            ),
        },
        "preexact_gates": preexact_gates,
        "runtime_by_case": runtime_by_case,
        "runtime_environment": runtime_environment,
        "candidate_pool_equivalence": equivalence,
        "summary": summary,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "external_speaker_gate_pass": passed,
        "bounded_waveform_promotion_pass": passed,
        "scientific_promotion_granted": passed,
        "six_component_readiness_eligible": passed,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
        "wall_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "external_svd_report_v32r3.json"
    v23.write_json(report_path, report)
    receipt = write_receipt(
        args,
        decision,
        [
            report_path,
            results_path,
            selector_path,
            attempts_path,
            equivalence_path,
        ],
        exact_opened=True,
        exact_scoring_complete=True,
        promotion_granted=passed,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "scientific_promotion_granted": passed,
                "six_component_readiness_eligible": passed,
                "exact_improvement_fraction": summary[
                    "exact_db_improvement_fraction"
                ],
                "median_normalized_reduction": summary[
                    "median_exact_db_normalized_gap_reduction"
                ],
                "maximum_runtime_ms": summary[
                    "total_metric_step_runtime_ms"
                ]["maximum"],
                "completion_receipt_sha256": sha256_file(receipt),
                "generator_optimizer_steps": 0,
                "authoritative_training_decision": TRAINING_DECISION,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
