#!/usr/bin/env python3
"""Run frozen Candidate-E external SVD with a result-blind runtime pipeline.

V32r6 is a pure pre-exact implementation successor of v32r5. It preserves
the Candidate-E math, candidate grid, selector, thresholds, panel, target
seal, and exact-Praat adjudicator. Candidate exact outcomes remain closed
unless byte/topology/proxy/selector serial equivalence and the frozen 500-ms
per-case runtime gate all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from scripts import adjudicate_avqi_shimmer_db_deterministic_opened24_v23 as v23
from scripts import evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r2 as v32r2
from scripts import evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r3 as v32r3
from scripts import evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r4 as v32r4
from scripts import evaluate_avqi_shimmer_db_candidate_e_external_svd_v32r5 as v32r5
from scripts import evaluate_avqi_shimmer_db_candidate_e_opened_v15_v29 as v29
from scripts import evaluate_avqi_shimmer_hybrid_topology as hybrid
from scripts import evaluate_direct_avqi_waveform_optimization as direct
from scripts.avqi_shimmer_db_candidate_e_proxy_v27 import (
    SAMPLE_RATE,
    STOP_HANN_HIGH_HZ,
    STOP_HANN_LOW_HZ,
    candidate_e_proxy,
    fixed_pulse_shimmer_db,
    next_power_of_two,
    normalized_gradient_step,
    pcm16_ste,
    praat_pcm16_ste,
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
    pcm24_metrics_from_loaded,
)
from scripts.evaluate_avqi_shimmer_db_trust_region_v16 import (
    ALPHA_LADDER,
    base_topology_item,
    finite_safety,
)


REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-exact-promotion-v32r6"
)
RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-receipt-v32r6"
)
SELECTOR_SEAL_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-selector-seal-v32r6"
)
PASS_DECISION = "PASS_CANDIDATE_E_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V32R6"
PREEXACT_NO_GO = "NO_GO_CANDIDATE_E_EXTERNAL_SVD_PREEXACT_V32R6"
EXACT_NO_GO = "NO_GO_CANDIDATE_E_EXTERNAL_SVD_EXACT_PROMOTION_V32R6"
READINESS_PASS = "READY_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
READINESS_NO_GO = "NO_GO_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
TRAINING_DECISION = "NO_GO_AVQI_T2_TRAINING"
RUNTIME_CONFIG_SCHEMA = (
    "avqi-route-c-shimmer-db-candidate-e-external-svd-runtime-successor-v32r6"
)
TOPOLOGY_WORKER_COUNT = 8
EXECUTOR_WORKER_COUNT = 8
EXPECTED_UNIQUE_PCM24_COUNT = 9
EXPECTED_CASES = v32r3.EXPECTED_CASES
EXPECTED_SPEAKERS = v32r3.EXPECTED_SPEAKERS
SYNTHETIC_WARMUP_FFT_LENGTHS = v32r3.SYNTHETIC_WARMUP_FFT_LENGTHS


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
        "v29-report",
        "v29-receipt",
        "panel-seal",
        "panel-receipt",
        "updated-speaker-ledger",
        "target-contract",
        "target-receipt",
        "predictor-checkpoint",
        "runtime-worker-script",
        "runtime-config",
        "v32r5-report",
        "v32r5-receipt",
        "runtime-diagnostic",
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
        raise ValueError("repository root does not contain v32r6 runner")
    head = v23.git_output(root, "rev-parse", "HEAD")
    if head != args.source_commit:
        raise ValueError("v32r6 repository HEAD/source commit drift")
    status = v23.git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("v32r6 adjudication requires a clean repository")
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
            "v32r6 runtime successor config",
        ),
        "v32r5_report": v23.validate_hash(
            args.v32r5_report,
            args.v32r5_report_sha256,
            "v32r5 pre-exact NO_GO report",
        ),
        "v32r5_receipt": v23.validate_hash(
            args.v32r5_receipt,
            args.v32r5_receipt_sha256,
            "v32r5 pre-exact NO_GO receipt",
        ),
        "runtime_diagnostic": v23.validate_hash(
            args.runtime_diagnostic,
            args.runtime_diagnostic_sha256,
            "v32r5 result-blind runtime microdiagnostic",
        ),
    }
    config = v23.read_json(args.runtime_config)
    report = v23.read_json(args.v32r5_report)
    receipt = v23.read_json(args.v32r5_receipt)
    diagnostic = v23.read_json(args.runtime_diagnostic)
    if config.get("schema_version") != RUNTIME_CONFIG_SCHEMA:
        raise ValueError("v32r6 runtime config schema drift")
    predecessor = config.get("predecessor_no_go", {})
    expected_predecessor = {
        "job_id": "20043463",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "scientific_decision": v32r5.PREEXACT_NO_GO,
        "source_commit": "f394dc751bf5788cf11eb6180c788d529efe93ce",
        "report_sha256": observed["v32r5_report"],
        "receipt_sha256": observed["v32r5_receipt"],
        "selector_seal_sha256": (
            "e58dfe58a86903052d67e8f952112d3eca2255d6e7519bb0fccd1154748a92ec"
        ),
        "candidate_attempts_sha256": (
            "a2456db5e4a5236b941d01d7046e9d4df0603fc4fbf1af81681560c134029b36"
        ),
        "candidate_pool_equivalence_sha256": (
            "a2d6cfc5dbff0e1d0eed74b2daa119b247fbd59fb583abcf2989918991f2a4a6"
        ),
        "stdout_sha256": (
            "d2c6ac779c8ba505ae5c28d9f334402eccb730b1d829cd1e0323abfdfe2e473f"
        ),
        "stderr_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "failed_gate": "total_metric_step_runtime_le_500ms",
        "failed_case_count": 3,
        "maximum_total_metric_step_runtime_ms": 675.8026629686356,
        "old_receipt_remains_immutable": True,
        "no_go_not_reinterpreted_as_pass": True,
    }
    if predecessor != expected_predecessor:
        raise ValueError("v32r6 predecessor NO_GO binding drift")
    if report.get("schema_version") != v32r5.REPORT_SCHEMA:
        raise ValueError("v32r5 predecessor report schema drift")
    if report.get("decision") != v32r5.PREEXACT_NO_GO:
        raise ValueError("v32r5 predecessor is not pre-exact NO_GO")
    if report.get("source_commit") != predecessor["source_commit"]:
        raise ValueError("v32r5 predecessor source binding drift")
    if report.get("slurm_job_id") != predecessor["job_id"]:
        raise ValueError("v32r5 predecessor job binding drift")
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
        raise ValueError("v32r5 predecessor failure-class drift")
    runtimes = report.get("runtime_by_case", {})
    if (
        not isinstance(runtimes, dict)
        or len(runtimes) != EXPECTED_CASES
        or sum(
            float(row["total_metric_step_runtime_ms"])
            > hybrid.CACHE_RUNTIME_MAX_MS
            for row in runtimes.values()
        )
        != 3
        or not math.isclose(
            max(
                float(row["total_metric_step_runtime_ms"])
                for row in runtimes.values()
            ),
            675.8026629686356,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("v32r5 predecessor runtime evidence drift")
    if (
        report.get("candidate_exact_outcomes_opened") is not False
        or report.get("exact_scoring_complete") is not False
        or report.get("scientific_promotion_granted") is not False
    ):
        raise ValueError("v32r5 predecessor opened or promoted exact outcomes")
    if receipt.get("schema_version") != v32r5.RECEIPT_SCHEMA:
        raise ValueError("v32r5 predecessor receipt schema drift")
    if receipt.get("decision") != v32r5.PREEXACT_NO_GO:
        raise ValueError("v32r5 predecessor receipt decision drift")
    expected_artifacts = {
        "external_svd_report_v32r5.json": observed["v32r5_report"],
        "selector_seal_pre_exact_v32r5.json": predecessor[
            "selector_seal_sha256"
        ],
        "candidate_e_attempts_pre_exact_v32r5.csv": predecessor[
            "candidate_attempts_sha256"
        ],
        "candidate_pool_equivalence_v32r5.json": predecessor[
            "candidate_pool_equivalence_sha256"
        ],
    }
    if receipt.get("artifact_sha256") != expected_artifacts:
        raise ValueError("v32r5 predecessor receipt artifact binding drift")
    if (
        receipt.get("candidate_exact_outcomes_opened_after_selector_seal")
        is not False
        or receipt.get("exact_scoring_complete") is not False
        or receipt.get("scientific_promotion_granted") is not False
    ):
        raise ValueError("v32r5 predecessor receipt exact-opening drift")
    for label, value in (("report", report), ("receipt", receipt)):
        require_training_boundary(value, f"v32r5 predecessor {label}")

    diagnostic_binding = config.get("result_blind_runtime_microdiagnostic", {})
    expected_diagnostic_binding = {
        "job_id": "20043682",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "source_commit": "3cb7eded97c5d1ffb8954154f6dd44a9ec0859ee",
        "diagnostic_sha256": observed["runtime_diagnostic"],
        "diagnostic_source_sha256": (
            "d07a1b6f2a36aba00e994edf006e1bd798c3c906e717e9e9524c14e3b948250f"
        ),
        "launcher_sha256": (
            "edeebccadc56dad9e97430b6a8889528e292ceb78ed356a4b62c18562d62f86d"
        ),
        "stdout_sha256": (
            "dbd4841a9b768986ea9967dd6223a207ee2e8a79daaf7d17dcbcb6fe56527c13"
        ),
        "stderr_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "case_count": EXPECTED_CASES,
        "repeat_count": 5,
        "candidate_waveforms_persisted": False,
        "candidate_exact_outcomes_present": False,
        "candidate_exact_outcomes_opened": False,
        "candidate_exact_outcomes_used": False,
        "base_exact_topology_only": True,
        "finding": (
            "first_case_cold_path_and_serial_pipeline_cost_not_duration_or_"
            "case_identity"
        ),
        "uses_speaker_or_case_identity_for_routing": False,
    }
    if diagnostic_binding != expected_diagnostic_binding:
        raise ValueError("v32r6 runtime diagnostic binding drift")
    if diagnostic.get("schema_version") != (
        "avqi-route-c-shimmer-db-candidate-e-external-runtime-"
        "microdiagnostic-v32r5"
    ):
        raise ValueError("v32r5 runtime diagnostic schema drift")
    expected_diagnostic_fields = {
        "slurm_job_id": diagnostic_binding["job_id"],
        "case_count": EXPECTED_CASES,
        "repeat_count": 5,
        "candidate_e_math_changed": False,
        "candidate_grid_changed": False,
        "candidate_waveforms_persisted": False,
        "candidate_exact_outcomes_present": False,
        "candidate_exact_outcomes_opened": False,
        "candidate_exact_outcomes_used": False,
        "base_exact_topology_only": True,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    for field, expected in expected_diagnostic_fields.items():
        if diagnostic.get(field) != expected:
            raise ValueError(f"v32r5 runtime diagnostic field drift: {field}")
    diagnostic_rows = diagnostic.get("rows")
    if (
        not isinstance(diagnostic_rows, list)
        or len(diagnostic_rows) != EXPECTED_CASES
        or len({str(row.get("case_id")) for row in diagnostic_rows})
        != EXPECTED_CASES
    ):
        raise ValueError("v32r5 runtime diagnostic case coverage drift")

    runtime = config.get("runtime_successor", {})
    expected_runtime = {
        "failure_class": "implementation_runtime_only_before_candidate_exact",
        "topology_worker_count": TOPOLOGY_WORKER_COUNT,
        "executor_worker_count": EXECUTOR_WORKER_COUNT,
        "synthetic_only_warmup": True,
        "warmup_fft_lengths": list(SYNTHETIC_WARMUP_FFT_LENGTHS),
        "synthetic_full_candidate_pipeline_warmup": True,
        "pcm24_in_memory_encoding": True,
        "pcm24_in_memory_decode_for_certificates": True,
        "parallel_pcm24_in_memory_preparation": True,
        "serial_durable_single_writes_without_disk_readback": True,
        "candidate_topology_and_gpu_metric_overlap": True,
        "batched_metric_reused_for_frozen_and_current_proxy": True,
        "zero_step_pcm24_identity_asserted": True,
        "zero_step_exact_input_pcm16_identity_asserted": True,
        "zero_step_current_topology_reused_from_base": True,
        "nonzero_unique_pcm24_topology_refresh": True,
        "candidate_topology_refresh_count": TOPOLOGY_WORKER_COUNT,
        "precomputed_selector_certificates": True,
        "serial_oracle_required": True,
        "candidate_waveform_byte_equivalence_required": True,
        "current_topology_hash_equivalence_required": True,
        "current_proxy_absolute_tolerance": 1e-12,
        "selector_choice_equivalence_required": True,
        "uses_candidate_exact_outcomes": False,
        "uses_speaker_or_case_identity_for_routing": False,
    }
    if runtime != expected_runtime:
        raise ValueError("v32r6 runtime implementation contract drift")
    scientific = config.get("frozen_scientific_contract", {})
    if scientific != {
        "direction_families": list(CANDIDATE_E_VARIANTS),
        "alpha_ladder": list(ALPHA_LADDER),
        "selector": (
            "maximum_current_topology_proxy_gap_reduction_across_"
            "projected_and_raw_exact_path_directions"
        ),
        "deterministic_tie_break": [
            "projected_before_raw",
            "larger_alpha_before_smaller_alpha",
        ],
        "formal_total_metric_step_runtime_ms": 500.0,
        "thresholds_unchanged": True,
        "candidate_e_math_unchanged": True,
        "target_scalar_contract_unchanged": True,
        "exact_praat_remains_final_judge": True,
        "no_final_waveform_highpass": True,
    }:
        raise ValueError("v32r6 scientific contract drift")
    immutable = config.get("immutable_boundaries", {})
    expected_immutable = {
        "v30r2_panel_seal_unchanged": True,
        "v31r2_target_scalar_seal_unchanged": True,
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_preexact_no_go_preserved": True,
        "v32r4_preexact_no_go_preserved": True,
        "v32r5_preexact_no_go_preserved": True,
        "candidate_exact_must_remain_unopened_until_new_selector_seal": True,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    if immutable != expected_immutable:
        raise ValueError("v32r6 immutable boundary drift")
    return config, observed


def prepare_runtime_candidate(
    context: dict[str, Any],
    entry: dict[str, Any],
    values: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    encode_started = time.perf_counter()
    buffer = io.BytesIO()
    sf.write(
        buffer,
        values,
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_24",
    )
    encoded = buffer.getvalue()
    encode_ms = 1000.0 * (time.perf_counter() - encode_started)
    candidate_path = Path(entry["candidate_path"])

    certificate_started = time.perf_counter()
    stored, sample_rate = sf.read(
        io.BytesIO(encoded),
        dtype="float32",
        always_2d=False,
    )
    if (
        sample_rate != SAMPLE_RATE
        or stored.ndim != 1
        or stored.shape != context["base_values"].shape
    ):
        raise ValueError("v32r6 in-memory PCM24 waveform shape drift")
    stored = np.asarray(stored, dtype=np.float32)
    candidate_sha256 = hashlib.sha256(encoded).hexdigest()
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
    finite = finite_safety(context["base_values"], stored)
    pcm24 = pcm24_metrics_from_loaded(
        context["base_codes"],
        context["base_sha256"],
        stored,
        candidate_sha256,
    )
    certificate_ms = 1000.0 * (time.perf_counter() - certificate_started)
    row = {
        "item_id": entry["item_id"],
        "case_id": entry["case_id"],
        "view": entry["view"],
        "variant": entry["variant"],
        "alpha": entry["alpha"],
        "base_path": entry["base_path"],
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": candidate_sha256,
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
        "paired_peak_certificate_failure_mode": peak_certificate[
            "failure_mode"
        ],
        **safety,
        "waveform_finite": bool(finite["waveform_finite"]),
        "waveform_bound_pass": bool(finite["waveform_bound_pass"]),
        "finite_safety_pass": bool(finite["finite_safety_pass"]),
        **pcm24,
        "_encoded_wav": encoded,
        "_stored_waveform": stored,
    }
    return row, {
        "pcm24_in_memory_encode_ms": encode_ms,
        "pcm24_memory_decode_and_certificates_ms": certificate_ms,
    }


def write_prepared_runtime_candidate(row: dict[str, Any]) -> float:
    encoded = row.get("_encoded_wav")
    if not isinstance(encoded, bytes):
        raise ValueError("v32r6 prepared candidate WAV payload is missing")
    candidate_path = Path(str(row["candidate_path"]))
    started = time.perf_counter()
    written = candidate_path.write_bytes(encoded)
    if written != len(encoded):
        raise OSError(f"incomplete PCM24 write: {candidate_path}")
    return 1000.0 * (time.perf_counter() - started)


def current_topology_item(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"current_topology:{entry['item_id']}",
        "case_id": entry["case_id"],
        "role": "current_output_topology",
        "path": entry["candidate_path"],
        "view": entry["view"],
        "score_components": False,
        "exact_metric_topology": True,
        "highpass_mode": NUMPY_HIGHPASS_MODE,
    }


def candidate_layout() -> tuple[list[int], int, int]:
    candidates_per_direction = 1 + len(ALPHA_LADDER)
    projected_zero = 0
    raw_zero = candidates_per_direction
    unique_indices = [
        index
        for index in range(2 * candidates_per_direction)
        if index not in (projected_zero, raw_zero)
    ]
    if len(unique_indices) != TOPOLOGY_WORKER_COUNT:
        raise ValueError("v32r6 topology worker/candidate layout drift")
    return unique_indices, projected_zero, raw_zero


def batched_metric_pcm16(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        raise ValueError("v32r6 proxy batch must be nonempty and two-dimensional")
    input_pcm16 = pcm16_ste(waveforms)
    fft_sample_count = next_power_of_two(int(input_pcm16.shape[-1]))
    spectrum = torch.fft.rfft(input_pcm16, n=fft_sample_count, dim=-1)
    frequencies = torch.fft.rfftfreq(
        fft_sample_count,
        d=1.0 / SAMPLE_RATE,
        device=input_pcm16.device,
        dtype=input_pcm16.dtype,
    )
    response = torch.ones_like(frequencies)
    response = torch.where(
        frequencies <= STOP_HANN_LOW_HZ,
        torch.zeros_like(response),
        response,
    )
    transition = (
        (frequencies > STOP_HANN_LOW_HZ)
        & (frequencies <= STOP_HANN_HIGH_HZ)
    )
    transition_response = 0.5 - 0.5 * torch.cos(
        math.pi
        / (STOP_HANN_HIGH_HZ - STOP_HANN_LOW_HZ)
        * (frequencies - STOP_HANN_LOW_HZ)
    )
    response = torch.where(transition, transition_response, response)
    filtered = torch.fft.irfft(
        spectrum * response.unsqueeze(0),
        n=fft_sample_count,
        dim=-1,
    )[..., : input_pcm16.shape[-1]]
    return praat_pcm16_ste(filtered)


def proxy_from_metric_pcm16(
    metric_pcm16_full: torch.Tensor,
    topology: dict[str, Any],
) -> float:
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            topology,
            source_sample_count=metric_pcm16_full.numel(),
        ),
        dtype=torch.long,
        device=metric_pcm16_full.device,
    )
    metric = metric_pcm16_full.index_select(0, source_indices)
    prefix = int(topology["metric_constant_prefix_samples"])
    if prefix:
        metric = torch.cat((metric.new_zeros(prefix), metric))
    pulses = metric.new_tensor(topology["pulse_positions_samples"])
    shimmer_db, _, _, _, _ = fixed_pulse_shimmer_db(metric, pulses)
    return float(shimmer_db.detach().cpu())


def batched_metric_pcm16_from_waveforms(
    waveforms: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    if not waveforms:
        raise ValueError("v32r6 proxy batch must be nonempty")
    lengths = {int(np.asarray(waveform).size) for waveform in waveforms}
    if len(lengths) != 1:
        raise ValueError("v32r6 proxy batch waveform length drift")
    values = torch.from_numpy(
        np.stack(
            [np.asarray(waveform, dtype=np.float32) for waveform in waveforms]
        )
    ).to(device=device, dtype=torch.float64)
    with torch.inference_mode():
        return batched_metric_pcm16(values)


def topology_proxy_values_from_metric_batch(
    metric_batch: torch.Tensor,
    frozen_topology: dict[str, Any],
    current_topologies: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    if metric_batch.ndim != 2 or metric_batch.shape[0] != len(
        current_topologies
    ):
        raise ValueError("v32r6 metric/topology batch coverage drift")
    with torch.inference_mode():
        frozen = [
            proxy_from_metric_pcm16(metric, frozen_topology)
            for metric in metric_batch
        ]
        current = [
            proxy_from_metric_pcm16(metric, topology)
            for metric, topology in zip(
                metric_batch,
                current_topologies,
                strict=True,
            )
        ]
    return frozen, current


def batched_topology_proxy_values(
    waveforms: list[np.ndarray],
    frozen_topology: dict[str, Any],
    current_topologies: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[float], list[float]]:
    if len(waveforms) != len(current_topologies):
        raise ValueError("v32r6 proxy/topology batch coverage drift")
    metric_batch = batched_metric_pcm16_from_waveforms(waveforms, device)
    return topology_proxy_values_from_metric_batch(
        metric_batch,
        frozen_topology,
        current_topologies,
    )


def dual_direction_selector_from_certificates(
    candidate_rows: list[dict[str, Any]],
    target_by_case: dict[str, float],
    scale_value: float,
) -> dict[str, Any]:
    selected_rows = []
    case_ids = sorted(
        {
            str(row["case_id"])
            for row in candidate_rows
            if row["variant"] in CANDIDATE_E_VARIANTS
        }
    )
    for case_id in case_ids:
        by_key = {
            (str(row["variant"]), float(row["alpha"])): row
            for row in candidate_rows
            if row["case_id"] == case_id
            and row["variant"] in CANDIDATE_E_VARIANTS
        }
        expected = {
            (variant, alpha)
            for variant in CANDIDATE_E_VARIANTS
            for alpha in (0.0, *ALPHA_LADDER)
        }
        if expected - set(by_key):
            raise ValueError(
                f"Candidate-E dual-direction coverage drift: {case_id}"
            )
        target = target_by_case[case_id]
        base_proxy_values = {
            float(
                by_key[(variant, 0.0)][
                    "current_topology_proxy_shimmer_db"
                ]
            )
            for variant in CANDIDATE_E_VARIANTS
        }
        if len(base_proxy_values) != 1:
            raise ValueError(
                f"Candidate-E zero-step proxy differs by direction: {case_id}"
            )
        base_proxy_gap = abs(base_proxy_values.pop() - target) / scale_value
        attempts = []
        for family_rank, variant in enumerate(CANDIDATE_E_VARIANTS):
            for alpha_rank, alpha in enumerate(ALPHA_LADDER):
                row = by_key[(variant, float(alpha))]
                proxy_gap = abs(
                    float(row["current_topology_proxy_shimmer_db"])
                    - target
                ) / scale_value
                routing = {
                    "proxy_strict_improvement_pass": proxy_gap < base_proxy_gap,
                    "finite_safety_pass": bool(row["finite_safety_pass"]),
                    "pcm24_effective_step_pass": bool(
                        row["pcm24_effective_step_pass"]
                    ),
                    "topology_stability_pass": bool(
                        row["topology_stability_pass"]
                    ),
                    "paired_peak_certificate_pass": bool(
                        row[
                            "paired_candidate_sinc70_search_may_be_skipped"
                        ]
                    ),
                }
                eligible = all(routing.values())
                attempts.append(
                    {
                        "direction_family": variant,
                        "direction_family_rank": family_rank,
                        "alpha_rank": alpha_rank,
                        "alpha": float(alpha),
                        "candidate_path": row["candidate_path"],
                        "candidate_sha256": row["candidate_sha256"],
                        "current_topology_sha256": row[
                            "current_topology_sha256"
                        ],
                        "base_normalized_proxy_gap": base_proxy_gap,
                        "candidate_normalized_proxy_gap": proxy_gap,
                        "normalized_proxy_gap_reduction": (
                            base_proxy_gap - proxy_gap
                        ),
                        **routing,
                        "eligible": eligible,
                        "exact_candidate_outcome_present": False,
                    }
                )
        eligible_attempts = [
            attempt for attempt in attempts if attempt["eligible"]
        ]
        selected = (
            min(
                eligible_attempts,
                key=lambda attempt: (
                    -float(attempt["normalized_proxy_gap_reduction"]),
                    int(attempt["direction_family_rank"]),
                    int(attempt["alpha_rank"]),
                ),
            )
            if eligible_attempts
            else None
        )
        selected_rows.append(
            {
                "case_id": case_id,
                "case_id_used_for_routing": False,
                "attempts": attempts,
                "selected": selected,
                "selected_candidate_present": selected is not None,
            }
        )
    return {
        "schema_version": (
            "avqi-route-c-shimmer-db-candidate-e-dual-direction-selector-v27r4"
        ),
        "selector": (
            "maximum_current_topology_proxy_gap_reduction_across_"
            "projected_and_raw_exact_path_directions"
        ),
        "direction_families": list(CANDIDATE_E_VARIANTS),
        "alpha_ladder": list(ALPHA_LADDER),
        "deterministic_tie_break": [
            "projected_before_raw",
            "larger_alpha_before_smaller_alpha",
        ],
        "selector_inputs": [
            "waveform",
            "current_output_pulse_topology",
            "candidate_e_proxy",
            "direction_family",
            "paired_peak_certificate",
            "pcm24_effective_step_certificate",
            "waveform_safety_certificate",
        ],
        "candidate_exact_outcomes_present": False,
        "candidate_exact_outcomes_used_for_selection": False,
        "speaker_or_case_identity_used_for_routing": False,
        "rows": selected_rows,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }


def synthetic_full_candidate_pipeline_warmup(
    device: torch.device,
) -> list[dict[str, Any]]:
    """Warm only synthetic plan, projection, full grid, and batched proxy."""

    evidence: list[dict[str, Any]] = []
    for sample_count in SYNTHETIC_WARMUP_FFT_LENGTHS:
        started = time.perf_counter()
        sample_index = np.arange(sample_count, dtype=np.float32)
        carrier = np.sin(2.0 * np.pi * sample_index / 160.0)
        envelope = 1.0 + 0.08 * np.sin(
            2.0 * np.pi * sample_index / 1_600.0
        )
        values = np.asarray(0.04 * envelope * carrier, dtype=np.float32)
        pulse_positions = np.arange(
            80.0,
            float(sample_count - 80),
            160.0,
            dtype=np.float64,
        )
        topology = {
            "topology_preprocessing": "exact_avqi_view_metric_waveform",
            "source_sample_count": sample_count,
            "metric_sample_count": sample_count,
            "metric_constant_prefix_samples": 0,
            "metric_source_ranges": [[0, sample_count]],
            "metric_source_range_count": 1,
            "metric_mapped_sample_count": sample_count,
            "metric_reconstruction_max_pcm16_error": 0,
            "metric_reconstruction_differing_samples": 0,
            "pulse_positions_samples": pulse_positions.tolist(),
            "pulse_count": int(pulse_positions.size),
        }
        waveform = (
            torch.from_numpy(values)
            .to(device=device, dtype=torch.float64)
            .requires_grad_(True)
        )
        pulses = waveform.new_tensor(pulse_positions)
        source_indices = torch.arange(
            sample_count,
            dtype=torch.long,
            device=device,
        )
        proxy = candidate_e_proxy(waveform, pulses, source_indices, 0)
        raw = torch.autograd.grad((proxy.shimmer_db - 1.0).square(), waveform)[0]
        plan = build_zero_crossing_cycle_plan_vectorized(values, topology)
        projected, projection = project_cycle_gain_gradient_fixed_order(
            waveform,
            raw,
            plan,
        )
        if not projection["projected_gradient_valid"]:
            raise ValueError("synthetic Candidate-E projection warmup failed")
        directions = {
            VARIANT_E_PROJECTED: projected,
            VARIANT_E_RAW: raw,
        }
        if tuple(directions) != CANDIDATE_E_VARIANTS:
            raise ValueError("synthetic Candidate-E direction order drift")
        candidate_batch = torch.stack(
            [
                normalized_gradient_step(waveform, direction, alpha)
                for direction in directions.values()
                for alpha in (0.0, *ALPHA_LADDER)
            ]
        )
        if not bool(torch.isfinite(candidate_batch).all().detach().cpu()):
            raise ValueError("synthetic Candidate-E candidate batch is non-finite")
        candidate_values = candidate_batch.detach().cpu().numpy().astype(
            np.float32
        )
        frozen, current = batched_topology_proxy_values(
            [row for row in candidate_values],
            topology,
            [topology for _ in candidate_values],
            device,
        )
        synchronize(device)
        if len(frozen) != candidate_batch.shape[0] or frozen != current:
            raise ValueError("synthetic Candidate-E proxy batch warmup drift")
        evidence.append(
            {
                "fft_length": sample_count,
                "synthetic_only": True,
                "panel_or_training_waveform_used": False,
                "candidate_count": int(candidate_batch.shape[0]),
                "complete_cycle_count": int(
                    projection["complete_cycle_count"]
                ),
                "projected_gradient_valid": True,
                "candidate_batch_finite": True,
                "wall_ms": 1000.0 * (time.perf_counter() - started),
            }
        )
    return evidence


def build_candidate_pool_pipeline(
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
    del predictor
    scale_value = float(target_scale[SHIMMER_DB_INDEX].detach().cpu())
    workers: list[ExactShimmerTopologyWorker] = []
    worker_evidence: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    base_by_case: dict[str, dict[str, Any]] = {}
    runtime_by_case: dict[str, dict[str, float]] = {}
    certificate_cache, synthetic_warmup = v32r3.synthetic_runtime_warmup(
        device
    )
    full_pipeline_warmup = synthetic_full_candidate_pipeline_warmup(device)
    unique_indices, projected_zero, raw_zero = candidate_layout()
    for variant in CANDIDATE_E_VARIANTS:
        (waveform_root / variant).mkdir(parents=True, exist_ok=True)
    try:
        for worker_index in range(TOPOLOGY_WORKER_COUNT):
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
        with ThreadPoolExecutor(max_workers=EXECUTOR_WORKER_COUNT) as executor:
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
                    v32r3.refresh_waveform_chunks(
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
                    base_float.detach()
                    .to(dtype=torch.float64)
                    .requires_grad_(True)
                )
                pulses = waveform.new_tensor(topology["pulse_positions_samples"])
                proxy = candidate_e_proxy(
                    waveform,
                    pulses,
                    source_indices,
                    int(topology["metric_constant_prefix_samples"]),
                )
                target = float(
                    target_by_case[case_id]["exact_target_shimmer_db"]
                )
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
                    raise ValueError(
                        f"Candidate-E exact highpass scaled: {case_id}"
                    )
                base_codes = pcm16_roundtrip_values_to_codes(
                    pcm16_roundtrip(base_values)
                )
                stop_hann = certificate_cache.get(proxy.fft_sample_count)
                if stop_hann is None:
                    raise ValueError(
                        "unregistered Candidate-E FFT length: "
                        f"{proxy.fft_sample_count}"
                    )
                directions = {
                    VARIANT_E_PROJECTED: projected,
                    VARIANT_E_RAW: raw,
                }
                candidate_tensors: list[torch.Tensor] = []
                entries: list[dict[str, Any]] = []
                for variant, direction in directions.items():
                    for alpha in (0.0, *ALPHA_LADDER):
                        candidate_tensors.append(
                            normalized_gradient_step(
                                waveform,
                                direction,
                                alpha,
                            )
                        )
                        item_id = (
                            f"{case_id}:{variant}:{alpha_label(alpha)}"
                        )
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
                if not bool(
                    torch.isfinite(candidate_batch).all().detach().cpu()
                ):
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
                if not np.array_equal(
                    candidate_values[projected_zero],
                    candidate_values[raw_zero],
                ):
                    raise ValueError(
                        f"Candidate-E zero-step identity drift: {case_id}"
                    )
                materialize_context = {
                    "base_values": base_values,
                    "base_codes": v32r3.pcm24_codes(base_values),
                    "base_sha256": str(panel_row["base_sha256"]),
                    "base_pcm16_codes": base_codes,
                    "base_highpass_timing": highpass_timing,
                    "stop_hann_impulse_certificate": stop_hann,
                }

                prepare_started = time.perf_counter()
                prepare_futures = [
                    executor.submit(
                        prepare_runtime_candidate,
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
                materialized = [future.result() for future in prepare_futures]
                prepare_wall_ms = 1000.0 * (
                    time.perf_counter() - prepare_started
                )
                case_rows = [row for row, _ in materialized]
                timings_by_index = {
                    index: timing
                    for index, (_, timing) in enumerate(materialized)
                }
                write_started = time.perf_counter()
                durable_write_ms_by_index = {
                    index: write_prepared_runtime_candidate(row)
                    for index, row in enumerate(case_rows)
                }
                durable_write_wall_ms = 1000.0 * (
                    time.perf_counter() - write_started
                )
                if (
                    case_rows[projected_zero]["candidate_sha256"]
                    != case_rows[raw_zero]["candidate_sha256"]
                ):
                    raise ValueError(
                        f"Candidate-E zero-step PCM24 drift: {case_id}"
                    )
                base_pcm16 = pcm16_roundtrip(base_values)
                for zero_index in (projected_zero, raw_zero):
                    zero_pcm16 = pcm16_roundtrip(
                        case_rows[zero_index]["_stored_waveform"]
                    )
                    if not np.array_equal(base_pcm16, zero_pcm16):
                        raise ValueError(
                            "Candidate-E zero-step exact-input PCM16 drift: "
                            f"{case_id}"
                        )

                topology_items = [
                    current_topology_item(entries[index])
                    for index in unique_indices
                ]
                topology_waveforms = [
                    case_rows[index]["_stored_waveform"]
                    for index in unique_indices
                ]
                overlap_started = time.perf_counter()
                topology_futures = [
                    executor.submit(
                        worker.refresh_current_waveforms,
                        [item],
                        [values],
                        NUMPY_HIGHPASS_MODE,
                    )
                    for worker, item, values in zip(
                        workers,
                        topology_items,
                        topology_waveforms,
                        strict=True,
                    )
                ]
                metric_started = time.perf_counter()
                metric_batch = batched_metric_pcm16_from_waveforms(
                    [row["_stored_waveform"] for row in case_rows],
                    device,
                )
                synchronize(device)
                metric_ms = 1000.0 * (
                    time.perf_counter() - metric_started
                )
                topology_results = [
                    future.result() for future in topology_futures
                ]
                topology_metric_overlap_ms = 1000.0 * (
                    time.perf_counter() - overlap_started
                )
                refreshed_topologies = [
                    rows[0] for rows, _, _ in topology_results
                ]
                topology_requests = [
                    float(request_ms)
                    for _, request_ms, _ in topology_results
                ]
                topology_staging_ms = sum(
                    float(staging["staging_ms"])
                    for _, _, staging_rows in topology_results
                    for staging in staging_rows
                )
                current_by_index = {
                    index: current
                    for index, current in zip(
                        unique_indices,
                        refreshed_topologies,
                        strict=True,
                    )
                }
                for zero_index in (projected_zero, raw_zero):
                    current = dict(topology)
                    current["id"] = (
                        f"current_topology:{entries[zero_index]['item_id']}"
                    )
                    current_by_index[zero_index] = current
                current_rows = [
                    current_by_index[index] for index in range(len(entries))
                ]
                unique_sha_count = len(
                    {str(row["candidate_sha256"]) for row in case_rows}
                )
                if unique_sha_count != EXPECTED_UNIQUE_PCM24_COUNT:
                    raise ValueError(
                        f"Candidate-E unique PCM24 coverage drift: {case_id}"
                    )

                proxy_started = time.perf_counter()
                frozen_values, current_values = (
                    topology_proxy_values_from_metric_batch(
                        metric_batch,
                        topology,
                        current_rows,
                    )
                )
                synchronize(device)
                proxy_reduction_ms = 1000.0 * (
                    time.perf_counter() - proxy_started
                )
                for candidate, current, frozen_value, current_value in zip(
                    case_rows,
                    current_rows,
                    frozen_values,
                    current_values,
                    strict=True,
                ):
                    candidate.update(
                        {
                            "proxy_shimmer_db": frozen_value,
                            "current_topology_proxy_shimmer_db": current_value,
                            "current_topology_sha256": topology_sha256(current),
                            "current_topology_pulse_count": int(
                                current["pulse_count"]
                            ),
                            **topology_stability(topology, current),
                            **pulse_position_drift(topology, current),
                        }
                    )
                    candidate.pop("_encoded_wav")
                    candidate.pop("_stored_waveform")

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
                    "candidate_parallel_pcm24_preparation_ms": prepare_wall_ms,
                    "candidate_serial_durable_write_ms": durable_write_wall_ms,
                    "candidate_unique_pcm24_count": float(unique_sha_count),
                    "candidate_nonzero_topology_refresh_count": float(
                        len(unique_indices)
                    ),
                    "candidate_topology_and_metric_overlap_ms": (
                        topology_metric_overlap_ms
                    ),
                    "candidate_batched_metric_pcm16_ms_diagnostic": metric_ms,
                    "candidate_proxy_reduction_after_topology_ms": (
                        proxy_reduction_ms
                    ),
                    "selector_runtime_per_case_ms": 0.0,
                    "total_metric_step_runtime_ms": total_ms,
                    "worker_request_sum_ms_diagnostic": sum(
                        base_requests + topology_requests
                    ),
                    "candidate_topology_max_request_ms_diagnostic": max(
                        topology_requests
                    ),
                    "candidate_topology_staging_ms_diagnostic": (
                        topology_staging_ms
                    ),
                    "pcm24_in_memory_encode_sum_ms_diagnostic": sum(
                        timing["pcm24_in_memory_encode_ms"]
                        for timing in timings_by_index.values()
                    ),
                    "pcm24_durable_single_write_sum_ms_diagnostic": sum(
                        durable_write_ms_by_index.values()
                    ),
                    "pcm24_memory_decode_and_certificates_sum_ms_diagnostic": sum(
                        timing[
                            "pcm24_memory_decode_and_certificates_ms"
                        ]
                        for timing in timings_by_index.values()
                    ),
                }
                diagnostics.append(
                    {
                        "case_id": case_id,
                        "speaker_id": panel_row["speaker_id"],
                        "view": panel_row["view"],
                        "condition": panel_row["condition"],
                        "target_shimmer_db": target,
                        "candidate_e_proxy_before": float(
                            proxy.shimmer_db.detach()
                        ),
                        "candidate_e_raw_gradient_l2": float(
                            raw.norm().detach()
                        ),
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
                    f"candidate_e_external_preselection={panel_index}/"
                    f"{EXPECTED_CASES}",
                    flush=True,
                )
    finally:
        for worker in workers:
            worker.close()
    environment = {
        "topology_worker_count": TOPOLOGY_WORKER_COUNT,
        "executor_worker_count": EXECUTOR_WORKER_COUNT,
        "worker_startups_and_warmups": worker_evidence,
        "synthetic_candidate_e_warmup": synthetic_warmup,
        "synthetic_full_candidate_pipeline_warmup": full_pipeline_warmup,
        "synthetic_only_warmup": True,
        "warmups_outside_case_timer": True,
        "pcm24_in_memory_encoding": True,
        "pcm24_in_memory_decode_for_certificates": True,
        "parallel_pcm24_in_memory_preparation": True,
        "serial_durable_single_writes_without_disk_readback": True,
        "candidate_topology_and_gpu_metric_overlap": True,
        "batched_metric_reused_for_frozen_and_current_proxy": True,
        "zero_step_pcm24_identity_asserted": True,
        "zero_step_exact_input_pcm16_identity_asserted": True,
        "zero_step_current_topology_reused_from_base": True,
        "nonzero_unique_pcm24_topology_refresh": True,
        "candidate_topology_refresh_count": TOPOLOGY_WORKER_COUNT,
        "precomputed_selector_certificates": True,
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
            row
            for row in candidate_rows
            if str(row["case_id"]) == case_id
        ]
        started = time.perf_counter()
        case_selector = dual_direction_selector_from_certificates(
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
        runtime_by_case[case_id][
            "total_metric_step_runtime_ms"
        ] += selector_ms
    if selector_metadata is None:
        raise ValueError("per-case selector received no cases")
    return {**selector_metadata, "rows": selected_rows}


def validate_sources_and_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
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
    panel_rows = v32r3.v31.validate_panel_binding(
        panel,
        panel_receipt,
        panel_sha256=observed["panel_seal"],
    )
    v29_report = v23.read_json(args.v29_report)
    v29_receipt = v23.read_json(args.v29_receipt)
    v32r3.validate_v29_pass(
        v29_report,
        v29_receipt,
        panel,
        report_sha256=observed["v29_report"],
        receipt_sha256=observed["v29_receipt"],
    )
    ledger = v23.read_json(args.updated_speaker_ledger)
    ledger_speakers = v32r3.validate_updated_ledger(
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
    target_by_case = v32r3.validate_target_binding(
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
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_preexact_no_go_decision": v32r3.PREEXACT_NO_GO,
        "v32r3_preexact_no_go_job_id": "20042036",
        "v32r3_preexact_no_go_preserved": True,
        "v32r4_preexact_no_go_decision": v32r4.PREEXACT_NO_GO,
        "v32r4_preexact_no_go_job_id": "20043338",
        "v32r4_preexact_no_go_preserved": True,
        "v32r5_preexact_no_go_decision": v32r5.PREEXACT_NO_GO,
        "v32r5_preexact_no_go_job_id": "20043463",
        "v32r5_preexact_no_go_report_sha256": runtime_hashes[
            "v32r5_report"
        ],
        "v32r5_preexact_no_go_receipt_sha256": runtime_hashes[
            "v32r5_receipt"
        ],
        "v32r5_preexact_no_go_preserved": True,
        "runtime_microdiagnostic_job_id": "20043682",
        "runtime_microdiagnostic_sha256": runtime_hashes[
            "runtime_diagnostic"
        ],
        "runtime_microdiagnostic_candidate_exact_outcomes_used": False,
        "v32r6_change_class": (
            "implementation_runtime_only_before_candidate_exact"
        ),
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
            "v32r6": "v26_selector_and_exact_adjudication",
        },
    }
    return panel_rows, target_by_case, source_hashes, bindings


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
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_preexact_no_go_preserved": True,
        "v32r4_preexact_no_go_preserved": True,
        "v32r5_preexact_no_go_preserved": True,
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
    path = args.output_dir / "completion_receipt_v32r6.json"
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
    runtime_environment: dict[str, Any],
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
        "runtime_environment": runtime_environment,
        "candidate_exact_outcomes_opened": False,
        "exact_scoring_complete": False,
        "result_blind_external_three_stage_chain_complete": False,
        "old_v23_no_go_preserved": True,
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_preexact_no_go_preserved": True,
        "v32r4_preexact_no_go_preserved": True,
        "v32r5_preexact_no_go_preserved": True,
        "candidate_e_frozen": True,
        "retuning_authorized": False,
        "scientific_promotion_granted": False,
        "six_component_readiness_eligible": False,
        "joint_panel_authorized": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_DECISION,
    }
    report_path = args.output_dir / "external_svd_report_v32r6.json"
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
    rows = v32r3.build_external_result_rows(
        panel_rows,
        target_by_case,
        selected,
        exact_payload,
        topology_by_id,
        base_preselection,
        runtime_by_case,
        target_scale,
    )
    for row in rows:
        row["opened_panel"] = "external_svd_v30r2_v31r2_v32r6"
    return rows


def external_effect_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v32r3.external_effect_slices(rows)


def summarize_external(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v32r3.summarize_external(rows)


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
        raise RuntimeError("v32r6 requires an allocated CUDA device")
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
    ) = build_candidate_pool_pipeline(
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

    oracle_rows, _, _, _ = v29.build_candidate_pool(
        args,
        panel_rows,
        target_by_case,
        predictor,
        target_scale_tensor,
        device,
        oracle_root,
    )
    oracle_selector = v32r3.dual_direction_selector_seal(
        oracle_rows,
        target_scalar,
        float(target_scale[SHIMMER_DB_INDEX]),
    )
    equivalence = v32r3.compare_candidate_pools(
        candidate_rows,
        oracle_rows,
        selector,
        oracle_selector,
    )
    equivalence_path = args.output_dir / "candidate_pool_equivalence_v32r6.json"
    v23.write_json(equivalence_path, equivalence)
    attempts_path = args.output_dir / "candidate_e_attempts_pre_exact_v32r6.csv"
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
    selector_path = args.output_dir / "selector_seal_pre_exact_v32r6.json"
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
            runtime_environment,
        )
        return

    selected = v29.selected_candidates(selector, candidate_rows, diagnostics)
    if len(selected) != EXPECTED_CASES:
        raise ValueError("v32r6 selected candidate coverage drift")
    selector_choice = {
        str(row["case_id"]): row["selected"] for row in selector["rows"]
    }
    for case_id, choice in selected.items():
        routing = selector_choice[case_id]
        if not isinstance(routing, dict):
            raise ValueError(f"v32r6 selector unexpectedly abstained: {case_id}")
        choice["pcm24_effective_step_pass"] = bool(
            routing["pcm24_effective_step_pass"]
        )
        v23.validate_hash(
            Path(choice["candidate_path"]),
            str(choice["candidate_sha256"]),
            f"sealed Candidate-E waveform {case_id}",
        )
    exact_items = v32r3.v28.build_exact_items(panel_rows, selected)
    try:
        exact_payload = v32r3.run_exact_components(
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
            "candidate_pool_equivalence_sha256": sha256_file(
                equivalence_path
            ),
            "candidate_exact_outcomes_opened_after_selector_seal": True,
            "exact_scoring_complete": False,
            "result_blind_external_three_stage_chain_complete": False,
            "old_v23_no_go_preserved": True,
            "v32r2_preexact_no_go_preserved": True,
            "v32r3_preexact_no_go_preserved": True,
            "v32r4_preexact_no_go_preserved": True,
            "v32r5_preexact_no_go_preserved": True,
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
        report_path = args.output_dir / "external_svd_report_v32r6.json"
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

    target_versions = v23.read_json(args.target_contract)[
        "exact_scorer_versions"
    ]
    exact_versions = {
        "parselmouth": exact_payload["parselmouth_version"],
        "praat": exact_payload["praat_version"],
    }
    if exact_versions != target_versions:
        raise ValueError("v31r2 target/v32r6 exact scorer version drift")
    topology_by_id = v32r3.postseal_topologies(args, panel_rows, selected)
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
    results_path = args.output_dir / "external_svd_exact_results_v32r6.csv"
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
        "v32r2_preexact_no_go_preserved": True,
        "v32r3_preexact_no_go_preserved": True,
        "v32r4_preexact_no_go_preserved": True,
        "v32r5_preexact_no_go_preserved": True,
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
    report_path = args.output_dir / "external_svd_report_v32r6.json"
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
