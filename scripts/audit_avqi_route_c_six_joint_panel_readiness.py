#!/usr/bin/env python3
"""Fail-closed preflight for a future Route C six-component joint panel.

This script only audits immutable metadata and file hashes.  It cannot create
candidate waveforms, open a fresh panel, invoke Praat, or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


# Direct-path execution cannot import the project package until it re-enters
# through the supported module path. This narrow re-entry precedes local imports.
if __name__ == "__main__" and __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_avqi_route_c_six_joint_panel_readiness",
            *sys.argv[1:],
        ],
        cwd=project_root,
        check=False,
    )
    raise SystemExit(completed.returncode)

from model.avqi_route_c import (
    ROUTE_C_FIVE_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_ACTIVE_COMPONENTS,
    ROUTE_C_SIX_SCIENTIFIC_STATUS,
    route_c_six_registry_records,
)
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.decide_avqi_route_c_six_component_gradients import (
    ACTIVE_COMPONENTS as FROZEN_SIX_GRADIENT_COMPONENTS,
    DECISION_RECEIPT_SCHEMA_VERSION as SIX_GRADIENT_RECEIPT_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION as SIX_GRADIENT_SCHEMA_VERSION,
    DECISION_IMPLEMENTATION_KEYS as SIX_GRADIENT_DECISION_IMPLEMENTATION_KEYS,
    FROZEN_FIVE_JOB_ID,
    FROZEN_FIVE_RECEIPT_SHA256,
    FROZEN_FIVE_REPORT_SHA256,
    FROZEN_GATE_KEYS as SIX_GRADIENT_FROZEN_GATE_KEYS,
    JOINT_PANEL_NO_GO as SIX_GRADIENT_JOINT_PANEL_NO_GO,
    MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO,
    MAXIMUM_WEIGHTED_COMPONENT_SHARE,
    PASS_DECISION as SIX_GRADIENT_PASS_DECISION,
    RAW_PENDING_DECISION as SIX_GRADIENT_RAW_PENDING_DECISION,
    READINESS_SOURCE_EVIDENCE_KEYS,
    TRAINING_NO_GO,
    decision_requirements as six_gradient_decision_requirements,
)


READINESS_SCHEMA_VERSION = "avqi-route-c-six-joint-panel-readiness-v2"
READINESS_RECEIPT_SCHEMA_VERSION = (
    "avqi-route-c-six-joint-panel-readiness-receipt-v1"
)
READINESS_PASS_DECISION = "PASS_SIX_JOINT_PANEL_READINESS"
FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION = (
    "avqi-route-c-six-joint-scientific-contract-v1"
)
FROZEN_SPLIT_SEAL_SCHEMA_VERSION = "avqi-route-c-six-joint-split-seal-v1"
SHIMMER_DB_REQUIRED_STATUS = "fresh_speaker_panel_pass"
SHIMMER_DB_PROMOTION_REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-exact-promotion-v26"
)
SHIMMER_DB_PROMOTION_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-external-svd-exact-promotion-receipt-v26"
)
SHIMMER_DB_PROMOTION_PASS_DECISION = (
    "PASS_SHIMMER_DB_EXACT_PRAAT_EXTERNAL_SVD_PROMOTION_V26"
)
SHIMMER_DB_READINESS_PASS = (
    "READY_SHIMMER_DB_FOR_SIX_COMPONENT_JOINT_READINESS"
)
SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_SCHEMA = (
    "avqi-route-c-shimmer-db-component-no-go-closure-v23"
)
SHIMMER_DB_COMPONENT_NO_GO_DECISION = (
    "COMPONENT_LEVEL_NO_GO_SHIMMER_DB_CANDIDATE_D_V23"
)
SHIMMER_DB_COMPONENT_NO_GO_STATUS = "component_level_no_go_candidate_d_v23"
SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "avqi_route_c_shimmer_db_component_no_go_v23.json"
)
SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_SHA256 = (
    "56a314b65b7a1272f34ad300253ed86e5618c33d5bd9c958b4515d4e12719492"
)
SHIMMER_DB_V23_REPORT_SCHEMA = (
    "avqi-route-c-shimmer-db-opened24-exact-adjudication-v23"
)
SHIMMER_DB_V23_RECEIPT_SCHEMA = (
    "avqi-route-c-shimmer-db-opened24-exact-adjudication-receipt-v23"
)
SHIMMER_DB_V23_NO_GO_DECISION = (
    "NO_GO_SHIMMER_DB_OPENED24_EXACT_ADJUDICATION_V23"
)
SHIMMER_DB_V23_JOB_ID = "20006447"
SHIMMER_DB_V23_SOURCE_COMMIT = "a23a3684e6c6d4a7dd667c71ddc7343dd683875f"
SHIMMER_DB_V23_REPORT_SHA256 = (
    "b38392d7ce47d982bc6e69e7b5a6289f1c163d70e27f832b43e050ed80a657e4"
)
SHIMMER_DB_V23_RECEIPT_SHA256 = (
    "1436769f7d7e406cdf39d5c98d90d1532991602cdc906591873db71daaeebd59"
)
SHIMMER_DB_V23_EXACT_CSV_SHA256 = (
    "0d20f918f089d34594690e62fb8c0c25ba801e73956c9e1954a632c77bee53bb"
)
PRIOR_PANEL_LEDGER_SCHEMA = "avqi-route-c-prior-panel-speaker-ledger-v1"
SHIMMER_DB_LEDGER_SOURCE_KEY = "shimmer_db_external_svd_v24"
SOURCE_DATASET = "SVD"
SVD_SV_METADATA_SHA256 = (
    "36d8a725a209578744a862e63b5990d348e3d17d066a0247cdcd2e657c7ffc03"
)
SVD_CS_METADATA_SHA256 = (
    "465c15e46c9c9e325c14e5672abead050bbfd9a4bba75d0ace46bf5d58884966"
)
SVD_SPEAKER_SELECTION_SALT = "avqi-route-c-six-joint-svd-v1-20260826"
SVD_HEALTH_STATUS_MAPPING = (("1", "patient"), ("0", "healthy"))
SVD_MINIMUM_RAW_MONO_SECONDS = (("sv", 1.0), ("cs", 3.0))
GAP_SIMULATION_INVENTORY_SHA256 = (
    "859a9e058f4f44c8e15d4b37d992cefa4d1501d1127a374d7e8cb1403c020384"
)
GAP_RIR_SOURCE_NAME = "v1_arni_rir"
GAP_RIR_MANIFEST_SHA256 = (
    "2bac3a563292a5a0a1377e3e98d29b6cfb8808d81f2e53ec1cbbafb08642d9da"
)
GAP_NOISE_SOURCE_NAME = "v1_dns5_noise"
GAP_NOISE_MANIFEST_SHA256 = (
    "c6f9441cdd76f50b4eb7f4fa5b83b994a509d3a925d7ae9b887059af31794d65"
)
GAP_RECIPE_ASSIGNMENT_SALT = "avqi-route-c-six-joint-recipes-v1-20260826"
REQUIRED_SPLITS = ("calibration", "final")
REQUIRED_VIEWS = ("cs", "sv")
REQUIRED_CONDITIONS = ("clean", "rir_only", "snr20", "snr10")
DEGRADED_EFFICACY_CONDITIONS = ("rir_only", "snr20", "snr10")
REQUIRED_LABELS = ("patient", "healthy")
PATIENT_SPEAKERS_PER_SPLIT = 3
HEALTHY_SPEAKERS_PER_SPLIT = 3
SPEAKERS_PER_SPLIT = PATIENT_SPEAKERS_PER_SPLIT + HEALTHY_SPEAKERS_PER_SPLIT
ROWS_PER_SPEAKER = len(REQUIRED_CONDITIONS) * len(REQUIRED_VIEWS)
ROWS_PER_SPLIT = SPEAKERS_PER_SPLIT * ROWS_PER_SPEAKER
EXPECTED_TOTAL_SPEAKERS = len(REQUIRED_SPLITS) * SPEAKERS_PER_SPLIT
EXPECTED_TOTAL_ROWS = len(REQUIRED_SPLITS) * ROWS_PER_SPLIT
PATIENT_DEGRADED_EFFICACY_ROWS_PER_SPLIT = (
    PATIENT_SPEAKERS_PER_SPLIT
    * len(DEGRADED_EFFICACY_CONDITIONS)
    * len(REQUIRED_VIEWS)
)
PATIENT_CLEAN_CONTROL_ROWS_PER_SPLIT = (
    PATIENT_SPEAKERS_PER_SPLIT * len(REQUIRED_VIEWS)
)
HEALTHY_GUARDRAIL_ROWS_PER_SPLIT = HEALTHY_SPEAKERS_PER_SPLIT * ROWS_PER_SPEAKER
SOURCE_GENDER_ALLOCATION = (
    ("calibration", "patient", 2, 1),
    ("calibration", "healthy", 1, 2),
    ("final", "patient", 1, 2),
    ("final", "healthy", 2, 1),
)
PATHOLOGICAL_ROLE = (
    "degraded_efficacy_same_speaker_same_view_clean_pathological_target"
)
CLEAN_PATHOLOGICAL_ROLE = (
    "clean_no_overprocessing_same_speaker_same_view_clean_pathological_target"
)
HEALTHY_ROLE = "guardrail_only_no_target_no_loss_no_step"
FORBIDDEN_PANEL_ROW_FIELDS = frozenset(
    {
        "avqi",
        "clinical_severity",
        "component_values",
        "exact_avqi",
        "exact_components",
        "sample_group",
        "severity",
    }
)
GLOBAL_ALPHA_GRID = (
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
)
ALPHA_REQUIRED_GATE_FAMILIES = (
    "all_six_components",
    "equal_weight_joint",
    "all_required_efficacy_slices",
    "waveform_safety",
    "full_band_pathology",
    "denoising",
)
ALPHA_SELECTION_OBJECTIVE = (
    "maximize_equal_weight_joint_exact_median_normalized_gap_reduction"
)
ALPHA_SELECTION_TIE_BREAK = "smaller_alpha"
NORMALIZATION_SOURCE = "passed_six_gradient_raw_measurement_report"
NORMALIZATION_TARGET_MEAN_FIELD = "normalization.target_mean"
NORMALIZATION_TARGET_SCALE_FIELD = "normalization.target_scale"
MATERIAL_NORMALIZED_BEFORE_GAP_THRESHOLD = 0.02
MATERIAL_COVERAGE_FRACTION_MIN = 0.80
MATERIAL_CASES_ABSOLUTE_MIN = 5
MATERIAL_CASES_PER_18_MIN = 15
EXACT_IMPROVEMENT_FRACTION_MIN = 0.80
MEDIAN_NORMALIZED_GAP_REDUCTION_MIN = 0.02
REQUIRED_SLICE_IMPROVEMENT_FRACTION_EXCLUSIVE_MIN = 0.50
REQUIRED_SLICE_MEDIAN_NORMALIZED_REDUCTION_MIN = 0.0
REQUIRED_EFFICACY_SLICES = tuple(
    f"condition={condition}&view={view}"
    for condition in DEGRADED_EFFICACY_CONDITIONS
    for view in REQUIRED_VIEWS
)
SAFETY_RESIDUAL_RMS_DB_MAX = -50.0
SAFETY_COSINE_SIMILARITY_MIN = 0.99999
SAFETY_CLIP_FRACTION_MAX = 0.0
LOW_FREQUENCY_BANDS_HZ = ((20.0, 80.0), (80.0, 300.0))
AIRFLOW_PROXY_FREQUENCY_RANGE_HZ = (500.0, 4_000.0)
LOW_ENERGY_QUANTILE = 0.25
PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX = 0.50
PATHOLOGY_DB_WORST_GAP_INCREASE_MAX = 1.50
AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX = 0.05
AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX = 0.10
PAUSE_F1_MEDIAN_DECREASE_MAX = 0.05
PAUSE_F1_WORST_DECREASE_MAX = 0.15
GUARDRAIL_PASS_FRACTION_MIN = 2.0 / 3.0
DENOISING_MEDIAN_CHANGE_MIN_DB = -0.10
DENOISING_WORST_CHANGE_MIN_DB = -0.50
REQUIRED_GUARDRAILS = (
    "full_band_low_frequency",
    "pause",
    "airflow",
    "cs_sv_pathology",
    "residual",
    "cosine",
    "clipping",
    "snr",
    "si_sdr",
)
FIVE_COMPONENT_EVIDENCE_KEYS = (
    "cpps_report",
    "cpps_receipt",
    "hnr_report",
    "hnr_receipt",
    "shimmer_percent_report",
    "shimmer_percent_receipt",
    "slope_report",
    "slope_receipt",
    "slope_final_panel_seal",
    "slope_final_results",
    "tilt_report",
    "tilt_receipt",
)
FIVE_COMPONENT_REPORT_CONTRACTS = {
    "cpps": ("cpps_report", "cpps_receipt", "PASS_WAVEFORM_OPTIMIZATION"),
    "hnr": ("hnr_report", "hnr_receipt", "PASS_HNR_FRESH_SPEAKER_PANEL"),
    "shimmer_percent": (
        "shimmer_percent_report",
        "shimmer_percent_receipt",
        "PASS_SHIMMER_FRESH_SPEAKER_PANEL",
    ),
    "slope": (
        "slope_report",
        "slope_receipt",
        "PASS_LTAS_SLOPE_FRESH_SPEAKER_PANEL",
    ),
    "tilt": ("tilt_report", "tilt_receipt", "FAIL_WAVEFORM_OPTIMIZATION"),
}
REQUIRED_ARTIFACT_KEYS = (
    *FIVE_COMPONENT_EVIDENCE_KEYS,
    "five_gradient_report",
    "five_gradient_receipt",
    "shimmer_db_promotion_report",
    "shimmer_db_promotion_receipt",
    "shimmer_db_prior_panel_speaker_ledger",
    "six_gradient_raw_report",
    "six_gradient_raw_receipt",
    "six_gradient_report",
    "six_gradient_receipt",
    "fresh_panel_split_seal",
    "fresh_speaker_source_manifest",
    "svd_sv_metadata",
    "svd_cs_metadata",
    "gap_simulation_inventory",
    "gap_v1_arni_rir_manifest",
    "gap_v1_dns5_noise_manifest",
    "joint_recipe_assignment_manifest",
    "prior_panel_speaker_ledger",
    "joint_gate_contract",
    "target_value_protocol_contract",
    "clean_target_label_bank",
    "cpps_checkpoint",
    "hnr_checkpoint",
    "shimmer_percent_checkpoint",
    "slope_checkpoint",
    "tilt_checkpoint",
    "v19_runtime_evidence_manifest",
    "v19_worker",
    "v19_runtime_client",
    "generator_config",
    "generator_checkpoint",
    "fixed_recipes",
    "simulation_config",
    "simulation_source",
    "exact_avqi_code_tree_manifest",
    "exact_runtime_manifest",
    "joint_gradient_manifest",
)
SIX_GRADIENT_SOURCE_EVIDENCE_KEYS = (
    *READINESS_SOURCE_EVIDENCE_KEYS,
)
MISSING_CODE_STAGES: tuple[str, ...] = ()
UNFROZEN_SCIENTIFIC_CONTRACTS: tuple[str, ...] = ()
UNBOUND_EXECUTION_INPUTS = (
    "reviewed Shimmer dB fresh-promotion evidence",
    "reviewed six-component gradient PASS evidence",
    "SVD metadata-only prior-ledger-disjoint speaker selection",
    "reviewed GAP simulation inventory, RIR manifest, and noise manifest",
    "post-split unique recipe assignment manifest",
    "fresh SVD speaker source manifest",
    "fresh panel split seal",
    "same-speaker same-view clean pathological six-component target bank",
    (
        "96-row hash-bound joint-gradient manifest with current-output "
        "Shimmer dB topology"
    ),
)
FROZEN_PANEL_DATA_REQUIREMENTS = (
    "source dataset is SVD with frozen SV/CS metadata hashes",
    "selection is metadata-only, result-blind, and prior-ledger-disjoint",
    "prior-ledger exclusion precedes salted hash ranking",
    "one minimum numeric eligible paired CS/SV session is retained per speaker",
    "selection does not read exact AVQI or component values",
    "diagnosis may be recorded but is not a selection input",
    "selection does not create or infer mild/severe labels",
    "each speaker has clean/RIR/SNR20/SNR10 x CS/SV rows",
    "calibration and final speakers are disjoint",
    "each split has exactly three patient and three healthy speakers",
    "patient degraded rows alone enter the efficacy denominator",
    "patient clean rows are no-overprocessing controls",
    "healthy rows have no target, loss, or waveform step",
    "GAP simulation source inventory, RIR, and noise manifests are hash-frozen",
    "speaker splitting precedes salted unique recipe assignment",
    "source manifest binds every selected case and waveform hash",
    "target bank covers every patient case and all six exact columns",
    (
        "joint-gradient manifest exactly covers all rows and carries no "
        "candidate exact outcomes"
    ),
    "selected speakers do not overlap the prior-panel ledger",
)
SOURCE_REQUIREMENT_MATRIX = (
    {
        "requirement": "six-slot composed scorer",
        "current_evidence": "model.avqi_route_c.load_route_c_six_active_scorer",
        "status": "present_fail_closed_scaffold",
    },
    {
        "requirement": "current-output v19 topology binding for slot 3",
        "current_evidence": (
            "model.avqi_route_c_v19_contracts.validate_v19_exact_topology"
        ),
        "status": "present_fail_closed_scaffold",
    },
    {
        "requirement": "same-speaker normalized bidirectional six-slot loss",
        "current_evidence": "model.avqi_route_c.six_active_bidirectional_gap_losses",
        "status": "present_contract_frozen_actual_target_bank_required",
    },
    {
        "requirement": "six-component gradient evaluator/runner",
        "current_evidence": (
            "scripts.evaluate_avqi_route_c_six_component_gradients + "
            "scripts.decide_avqi_route_c_six_component_gradients"
        ),
        "status": "present_dev_only_raw_measurement_plus_frozen_code_decision",
    },
    {
        "requirement": "two-stage sealed joint waveform evaluator/runner",
        "current_evidence": (
            "scripts.prepare_avqi_route_c_six_joint_waveforms + "
            "scripts.evaluate_avqi_route_c_six_joint_exact_panel"
        ),
        "status": "present_fail_closed_hash_bound_runners",
    },
    {
        "requirement": "five promoted-component scientific evidence bundle",
        "current_evidence": (
            "frozen source_evidence set in five-component gradient audit"
        ),
        "status": "external_immutable_bindings_required",
    },
    {
        "requirement": "Shimmer dB Candidate-D scientific outcome",
        "current_evidence": (
            "configs/avqi_route_c_shimmer_db_component_no_go_v23.json "
            "binding v22 deterministic and v23 exact-Praat artifacts"
        ),
        "status": "component_level_no_go_v23_joint_closed",
    },
    {
        "requirement": "fresh-panel source/split/target schemas",
        "current_evidence": (
            "frozen salted SVD 12-speaker/96-row contract and structural "
            "validators"
        ),
        "status": "frozen_contract_actual_manifests_required",
    },
    {
        "requirement": "GAP simulation source and recipe assignment contract",
        "current_evidence": (
            "frozen inventory/RIR/noise hashes, post-split recipe salt, and "
            "condition semantics"
        ),
        "status": "frozen_contract_actual_manifests_and_assignment_required",
    },
    {
        "requirement": "joint waveform/exact gate thresholds",
        "current_evidence": (
            "frozen result-independent six-component exact, slice, safety, "
            "pathology, and denoising gates with tested post-seal evaluator"
        ),
        "status": "frozen_contract_evaluator_runner_present",
    },
)


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping_values(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(
        value.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise ValueError(f"{label} differs")
    return value


def _validate_shimmer_db_component_no_go_closure(
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable Candidate-D terminal scientific boundary."""
    _require_mapping_values(
        closure,
        {
            "schema_version": SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_SCHEMA,
            "decision": SHIMMER_DB_COMPONENT_NO_GO_DECISION,
            "component": "shimmer_db",
            "route": "C",
        },
        "Shimmer dB component NO-GO closure identity",
    )
    if not isinstance(closure.get("recorded_at"), str) or not closure[
        "recorded_at"
    ]:
        raise ValueError("Shimmer dB component NO-GO closure time differs")

    _require_mapping_values(
        closure.get("scope"),
        {
            "candidate": "Candidate-D",
            "candidate_reference": (
                "v22_repeat_1_byte_identical_across_three_repeats"
            ),
            "candidate_d_mathematics_changed": False,
            "alpha_changed": False,
            "selector_changed": False,
            "thresholds_changed": False,
            "pcm24_gate_changed": False,
            "fixed_alpha": 0.001,
            "future_pre_registered_candidate_contracts_adjudicated": False,
        },
        "Shimmer dB component NO-GO frozen scope",
    )
    _require_mapping_values(
        closure.get("v21_receipt_fix"),
        {
            "schema_version": (
                "avqi-route-c-shimmer-db-v21-receipt-field-fix-audit-v1"
            ),
            "audit_receipt_sha256": (
                "3e12214a57aa2ac9b634f546731118138fa87c13979899c994a17cde53b32ab3"
            ),
            "source_commit": "21e871a4a4f8f023b8b7cab8d88ea79a6c307fef",
            "wrapper_post_patch_sha256": (
                "80393bc55105e1bdb1632e7f43f98b66e9bb431cc5498c437d6b26adf3a923d6"
            ),
            "launcher_post_patch_sha256": (
                "39fb9e3aae50cfae494a47001d0883c8ebb87acdada905beed275a0d58736c74"
            ),
            "regression_test_sha256": (
                "19569304704bc032e4b91fe1bb23a0fabcf3a2fbb44f849cc007bad2982bcd85"
            ),
            "summary_source_field": (
                "output_wav_pcm24_byte_repeat_observed"
            ),
            "legacy_wrong_field_rejected": "byte_equivalence_observed",
            "live_pytest_tests_passed": 3,
            "live_pytest_subtests_passed": 2,
            "real_v21_report_summary_smoke_passed": True,
            "posthoc_audit_receipt_sufficient": True,
            "clean_bounded_rerun_scientifically_required": False,
            "scientific_gates_changed": False,
            "immutable_v21_artifacts_mutated": False,
            "old_v18_evidence_kept_separate": True,
            "generator_optimizer_steps": 0,
            "authoritative_training_decision": TRAINING_NO_GO,
        },
        "Shimmer dB v21 receipt fix closure",
    )
    deterministic = _require_mapping_values(
        closure.get("deterministic_contract"),
        {
            "capture_job_id": "19997287",
            "capture_decision": (
                "CAPTURED_SHIMMER_DB_DETERMINISTIC_FULL_STEP_BASELINE_"
                "NO_PROMOTION"
            ),
            "capture_manifest_sha256": (
                "57a284084621f6b57d3f35ef1834eafd4ef67c82bf210542f723d68d4f29310d"
            ),
            "capture_report_sha256": (
                "85444856b69972ce1bdad09c855352dc17dece4e38883335b336e3d8c754b99d"
            ),
            "capture_receipt_sha256": (
                "04f5da2f212998ef994fa62f4464ed9009bd65231ea42bbbaafdaea87d5502d2"
            ),
            "repeat_job_id": "20005708",
            "repeat_decision": (
                "PASS_SHIMMER_DB_DETERMINISTIC_FULL_STEP_REPEAT_V22"
            ),
            "repeat_report_sha256": (
                "ea6415eaa3b554a01a234eaf1ec4fe163f8735c182459970313bef2a7bbc0842"
            ),
            "repeat_receipt_sha256": (
                "075df8f5a24f6453429dd8ad70c2b49a0ce9900f06f7bb4ec3622e385062c160"
            ),
            "durable_selected_csv_sha256": (
                "af0561535f841348397ac82e0675d8fe9475c4a3c0b259874636890a39fad5e6"
            ),
            "durable_wav_count": 72,
            "durable_wav_byte_equal_count": 72,
            "attempt_row_count": 108,
            "runtime_row_count": 72,
            "old_v18_evidence_kept_separate": True,
            "old_v18_attempt_count": 36,
            "old_v18_mismatch_attempt_count": 22,
            "old_v18_full_equivalence_count": 14,
            "old_v18_gate_used_for_new_authorization": False,
        },
        "Shimmer dB deterministic closure",
    )
    for key, value in deterministic.items():
        if key.endswith("_sha256") and not _is_sha256(value):
            raise ValueError("Shimmer dB deterministic closure hash differs")

    exact = _require_mapping_values(
        closure.get("exact_adjudication"),
        {
            "schema_version": SHIMMER_DB_V23_REPORT_SCHEMA,
            "receipt_schema_version": SHIMMER_DB_V23_RECEIPT_SCHEMA,
            "slurm_job_id": SHIMMER_DB_V23_JOB_ID,
            "slurm_state": "COMPLETED",
            "slurm_exit_code": "0:0",
            "slurm_elapsed": "00:01:58",
            "slurm_node": "skl6",
            "decision": SHIMMER_DB_V23_NO_GO_DECISION,
            "case_count": 24,
            "speaker_count": 12,
            "parselmouth_version": "0.4.6",
            "praat_version": "6.1.38",
            "exact_scoring_complete": True,
            "opened_development_evidence_only": True,
            "target_reproduction_max_abs_error": 0.0,
            "candidate_metric_reconstruction_max_pcm16_error": 0,
            "full_band_pathology_guardrails_passed": True,
            "waveform_safety_passed": True,
            "denoising_nonregression_passed": True,
            "anti_shortcut_contract_passed": True,
        },
        "Shimmer dB exact adjudication closure",
    )
    expected_top_level_gates = {
        "anti_shortcut_contract": True,
        "combined_global_exact_effect": True,
        "old_v18_evidence_kept_separate": True,
        "opened24_contract_complete_and_speaker_disjoint": True,
        "v14_frozen_scientific_gates": False,
        "v15_frozen_scientific_gates": True,
        "v22_deterministic_chain_bound_and_passed": True,
    }
    if exact.get("top_level_gates") != expected_top_level_gates:
        raise ValueError("Shimmer dB v23 top-level gates differ")
    v14 = _require_mapping_values(
        exact.get("v14"),
        {
            "role": "development_calibration",
            "case_count": 12,
            "speaker_count": 6,
            "material_rows": 11,
            "exact_db_improvement_fraction": 0.9090909090909091,
            "median_exact_db_normalized_gap_reduction": (
                0.01911191379993479
            ),
            "median_threshold": 0.02,
            "median_margin": -0.0008880862000652107,
            "all_gates_pass": False,
            "only_failed_gate": "exact_db_effect",
        },
        "Shimmer dB v14 exact failure boundary",
    )
    if not float(v14["median_exact_db_normalized_gap_reduction"]) < float(
        v14["median_threshold"]
    ):
        raise ValueError("Shimmer dB v14 NO-GO margin is not negative")
    _require_mapping_values(
        exact.get("v15"),
        {
            "role": "opened_validation",
            "case_count": 12,
            "speaker_count": 6,
            "material_rows": 12,
            "exact_db_improvement_fraction": 1.0,
            "median_exact_db_normalized_gap_reduction": (
                0.021922072292837234
            ),
            "median_threshold": 0.02,
            "median_margin": 0.001922072292837234,
            "all_gates_pass": True,
        },
        "Shimmer dB v15 exact boundary",
    )
    _require_mapping_values(
        exact.get("failed_material_case"),
        {
            "case_id": "sealed_final__SD20__cs__snr20",
            "opened_panel": "v14",
            "exact_absolute_gap_before_shimmer_db": 0.15829437779222322,
            "exact_absolute_gap_after_shimmer_db": 0.16602488503782875,
            "exact_normalized_gap_reduction_shimmer_db": (
                -0.01748682752871204
            ),
        },
        "Shimmer dB v23 failed material case",
    )

    _require_mapping_values(
        closure.get("source"),
        {
            "branch": "feat/avqi-route-c-shimmer-db-exact-promotion-v23",
            "commit": SHIMMER_DB_V23_SOURCE_COMMIT,
            "tree_clean_at_terminal_audit": True,
            "evaluator_sha256": (
                "9772f6d2d29767897d2caef3fc380038aad6a31e58bbd6146d64ef07c86ee554"
            ),
            "launcher_sha256": (
                "cd7b6ac4dcdd58a65ac3ddaad6e3316ce16aeaf99c38bbe582b299ea6e50072c"
            ),
            "submission_receipt_sha256": (
                "e0b9584d09b3e169d6affb528167e4a1c5e1b40e3e888baa43df5d87789428b0"
            ),
            "predictor_checkpoint_sha256": (
                "40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc"
            ),
            "avqi_code_tree_sha256": (
                "46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2"
            ),
        },
        "Shimmer dB v23 source closure",
    )
    artifacts = _require_mapping_values(
        closure.get("artifact_sha256"),
        {
            "diagnostic_report.json": SHIMMER_DB_V23_REPORT_SHA256,
            "completion_receipt.json": SHIMMER_DB_V23_RECEIPT_SHA256,
            "opened24_exact_results.csv": SHIMMER_DB_V23_EXACT_CSV_SHA256,
            "slurm_stdout": (
                "14603d1b363c4afe31e6df0eb4bca5320e35294bb3ea61debd1a6ca7b415fac7"
            ),
            "slurm_stderr": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
        "Shimmer dB v23 artifact closure",
    )
    if any(not _is_sha256(value) for value in artifacts.values()):
        raise ValueError("Shimmer dB v23 artifact hash differs")
    _require_mapping_values(
        closure.get("authorization"),
        {
            "scientific_promotion_granted": False,
            "external_speaker_panel_authorized": False,
            "v24_prepare_authorized": False,
            "v25_target_seal_authorized": False,
            "v26_external_exact_authorized": False,
            "six_component_readiness_eligible": False,
            "joint_panel_authorized": False,
            "generator_loaded": False,
            "generator_optimizer_created": False,
            "generator_optimizer_steps": 0,
            "formal_generator_training_submitted": False,
            "authoritative_training_decision": TRAINING_NO_GO,
            "component_research_closed_for_frozen_candidate_d_contract": True,
        },
        "Shimmer dB v23 authorization boundary",
    )
    _require_mapping_values(
        closure.get("immutability"),
        {
            "raw_artifacts_rewritten": False,
            "old_v18_evidence_rewritten": False,
            "closure_is_post_hoc_summary_only": True,
        },
        "Shimmer dB v23 immutability boundary",
    )
    return {
        "scientific_status": SHIMMER_DB_COMPONENT_NO_GO_STATUS,
        "decision": SHIMMER_DB_COMPONENT_NO_GO_DECISION,
        "v23_job_id": SHIMMER_DB_V23_JOB_ID,
        "v23_report_sha256": SHIMMER_DB_V23_REPORT_SHA256,
        "v23_receipt_sha256": SHIMMER_DB_V23_RECEIPT_SHA256,
        "v23_exact_csv_sha256": SHIMMER_DB_V23_EXACT_CSV_SHA256,
        "failed_gate": "v14.exact_db_effect",
        "v14_median_margin": v14["median_margin"],
        "old_v18_evidence_kept_separate": True,
        "six_component_readiness_eligible": False,
        "joint_panel_authorized": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }


def load_shimmer_db_component_no_go_closure() -> dict[str, Any]:
    closure_sha256 = sha256_file(SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_PATH)
    if closure_sha256 != SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_SHA256:
        raise ValueError("Shimmer dB component NO-GO closure hash differs")
    closure = _read_json_mapping(
        SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_PATH,
        "Shimmer dB component NO-GO closure",
    )
    validated = _validate_shimmer_db_component_no_go_closure(closure)
    return {
        **validated,
        "path": str(SHIMMER_DB_COMPONENT_NO_GO_CLOSURE_PATH),
        "sha256": closure_sha256,
    }


def frozen_svd_speaker_rank(speaker_id: str, session_id: str) -> str:
    """Return the preregistered result-blind rank for one eligible SVD speaker."""
    if not speaker_id or not session_id.isdecimal():
        raise ValueError("SVD speaker rank requires a numeric session ID")
    payload = f"{SVD_SPEAKER_SELECTION_SALT}:{speaker_id}:{session_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_optimizer_zero(value: Mapping[str, Any], label: str) -> None:
    if value.get("generator_optimizer_steps") != 0:
        raise ValueError(f"{label} contains generator optimizer steps")


def _require_all_true_gates(value: Mapping[str, Any], label: str) -> None:
    gates = value.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"{label} has no frozen gates")
    if any(gate is not True for gate in gates.values()):
        raise ValueError(f"{label} contains a failed gate")


def _finite_mapping(
    value: Any,
    expected_keys: tuple[str, ...],
    label: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError(f"{label} keys differ")
    parsed = {key: float(value[key]) for key in expected_keys}
    if any(
        not math.isfinite(number)
        or (positive and number <= 0.0)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
        for number in parsed.values()
    ):
        raise ValueError(f"{label} values are invalid")
    return parsed


def frozen_scientific_contract() -> dict[str, Any]:
    """Return the result-independent six-joint contract frozen before execution."""
    gender_allocation: dict[str, dict[str, dict[str, int]]] = {
        split: {} for split in REQUIRED_SPLITS
    }
    for split, label, female, male in SOURCE_GENDER_ALLOCATION:
        gender_allocation[split][label] = {"female": female, "male": male}
    return {
        "schema_version": FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
        "frozen_before_source_selection": True,
        "source": {
            "dataset": SOURCE_DATASET,
            "metadata_sha256": {
                "sv": SVD_SV_METADATA_SHA256,
                "cs": SVD_CS_METADATA_SHA256,
            },
            "selection_mode": "metadata_only_result_blind",
            "health_status_mapping": dict(SVD_HEALTH_STATUS_MAPPING),
            "allowed_selection_fields": [
                "health_status",
                "speaker_id",
                "session_id",
                "gender",
                "paired_cs_sv",
                "audio_integrity",
                "prior_panel_ledger",
            ],
            "record_only_fields": ["diagnosis"],
            "forbidden_selection_inputs": [
                "diagnosis",
                "avqi",
                "exact_avqi",
                "component_values",
                "exact_component_values",
                "surrogate_component_values",
                "clinical_severity",
                "mild_severe",
            ],
            "paired_cs_sv_same_session_required": True,
            "minimum_raw_mono_duration_seconds": dict(
                SVD_MINIMUM_RAW_MONO_SECONDS
            ),
            "eligible_session_per_speaker": "minimum_numeric_session_id",
            "speaker_selection_salt": SVD_SPEAKER_SELECTION_SALT,
            "selection_operation_order": [
                "map_health_status",
                "pair_cs_sv_by_same_session",
                "filter_raw_mono_minimum_duration",
                "retain_minimum_numeric_eligible_session_per_speaker",
                "exclude_prior_ledger_speakers",
                "bucket_by_health_status_and_gender",
                "rank_by_salted_sha256",
                "allocate_calibration_then_final_by_frozen_gender_quota",
            ],
            "ranking": {
                "bucket_fields": ["health_status", "gender"],
                "digest": "SHA256(salt:speaker_id:session_id)",
                "order": "ascending_hex_digest",
                "collision_tie_break": ["speaker_id", "session_id"],
                "split_allocation_order": list(REQUIRED_SPLITS),
                "quota_source": "source.gender_allocation",
            },
            "prior_ledger_exclusion_stage": "before_hash_ranking",
            "prior_panel_speaker_overlap": 0,
            "mild_severe_labels_created": False,
            "selection_performed_at_contract_freeze": False,
            "gender_allocation": gender_allocation,
        },
        "panel": {
            "splits": list(REQUIRED_SPLITS),
            "conditions": list(REQUIRED_CONDITIONS),
            "views": list(REQUIRED_VIEWS),
            "patient_speakers_per_split": PATIENT_SPEAKERS_PER_SPLIT,
            "healthy_speakers_per_split": HEALTHY_SPEAKERS_PER_SPLIT,
            "speakers_per_split": SPEAKERS_PER_SPLIT,
            "rows_per_speaker": ROWS_PER_SPEAKER,
            "rows_per_split": ROWS_PER_SPLIT,
            "total_speakers": EXPECTED_TOTAL_SPEAKERS,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "patient_degraded_efficacy_rows_per_split": (
                PATIENT_DEGRADED_EFFICACY_ROWS_PER_SPLIT
            ),
            "patient_clean_control_rows_per_split": (
                PATIENT_CLEAN_CONTROL_ROWS_PER_SPLIT
            ),
            "healthy_guardrail_rows_per_split": (
                HEALTHY_GUARDRAIL_ROWS_PER_SPLIT
            ),
        },
        "simulation": {
            "source_inventory_sha256": GAP_SIMULATION_INVENTORY_SHA256,
            "rir_source": {
                "name": GAP_RIR_SOURCE_NAME,
                "manifest_sha256": GAP_RIR_MANIFEST_SHA256,
            },
            "noise_source": {
                "name": GAP_NOISE_SOURCE_NAME,
                "manifest_sha256": GAP_NOISE_MANIFEST_SHA256,
            },
            "speaker_split_before_recipe_assignment": True,
            "recipe_assignment_salt": GAP_RECIPE_ASSIGNMENT_SALT,
            "recipe_uid_required_for_every_row": True,
            "recipe_uid_unique_per_row": True,
            "recipe_uid_reused_across_splits": False,
            "condition_recipes": {
                "clean": {
                    "rir": False,
                    "noise": False,
                    "target_snr_db": None,
                },
                "rir_only": {
                    "rir": True,
                    "noise": False,
                    "target_snr_db": None,
                },
                "snr20": {
                    "rir": True,
                    "noise": True,
                    "target_snr_db": 20.0,
                },
                "snr10": {
                    "rir": True,
                    "noise": True,
                    "target_snr_db": 10.0,
                },
            },
            "actual_recipes_selected_at_contract_freeze": False,
        },
        "two_stage_opening": {
            "calibration_selection_sealed_before_final_open": True,
            "calibration_and_final_speakers_disjoint": True,
            "calibration_may_select": ["one_global_alpha"],
            "final_may_select_or_tune": [],
            "alpha_selection_receipt_sealed_before_final_exact_open": True,
            "final_waveforms_sealed_before_final_exact_open": True,
            "final_exact_outcomes_opened_at_contract_freeze": False,
        },
        "waveform_step": {
            "steps": 1,
            "global_alpha": True,
            "gradient_normalization": "waveform_rms_normalized",
            "alpha_grid": list(GLOBAL_ALPHA_GRID),
            "zero_alpha_role": "negative_control_only",
            "zero_alpha_selectable": False,
            "nonzero_alpha_required_gate_families": list(
                ALPHA_REQUIRED_GATE_FAMILIES
            ),
            "nonzero_alpha_gate_split": "calibration",
            "selection_objective": ALPHA_SELECTION_OBJECTIVE,
            "selection_tie_break": ALPHA_SELECTION_TIE_BREAK,
            "selection_split": "calibration",
            "final_tuning_permitted": False,
        },
        "optimization_weights": {
            "only_allowed_source": "passed_six_gradient_report",
            "source_report_decision": SIX_GRADIENT_PASS_DECISION,
            "source_report_and_receipt_hash_bound": True,
            "source_field": (
                "measurement_summary.calibration_inverse_gradient_weights"
            ),
            "calibration_only": True,
            "used_for": "waveform_gradient_generation_only",
            "used_for_exact_joint_decision": False,
            "held_fixed_for_final": True,
            "final_reestimation_permitted": False,
        },
        "normalization": {
            "only_allowed_source": NORMALIZATION_SOURCE,
            "source_bound_by_passed_six_gradient_decision": True,
            "target_mean_field": NORMALIZATION_TARGET_MEAN_FIELD,
            "target_scale_field": NORMALIZATION_TARGET_SCALE_FIELD,
            "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
            "target_scales_finite_and_positive": True,
            "joint_panel_refit_permitted": False,
            "joint_panel_rows_used_to_fit_mean_or_scale": False,
        },
        "target_and_aggregate": {
            "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
            "patient_target": (
                "exact same-speaker same-view clean pathological target"
            ),
            "patient_degraded_enters_efficacy_denominator": True,
            "patient_clean_role": "no-overprocessing control",
            "patient_clean_enters_degraded_efficacy_denominator": False,
            "healthy_target": None,
            "healthy_loss_enabled": False,
            "healthy_waveform_step_enabled": False,
            "healthy_enters_degraded_efficacy_denominator": False,
            "healthy_control_role": (
                "routing/source/topology/coverage control only"
            ),
            "optimized_healthy_safety_claimed": False,
            "joint_aggregate": (
                "equal-weight mean of six normalized exact component gaps"
            ),
            "avqi_scalar_coefficient_used_for_direction": False,
            "avqi_scalar_coefficient_used_for_aggregate": False,
            "exact_praat_is_final_judge": True,
            "avqi_preprocessing_is_metric_only": True,
            "emitted_waveform_highpass": False,
        },
        "efficacy_gates": {
            "scope": "patient degraded rows only",
            "material_normalized_before_gap": {
                "comparison": ">",
                "value": MATERIAL_NORMALIZED_BEFORE_GAP_THRESHOLD,
            },
            "material_coverage_fraction": {
                "comparison": ">=",
                "value": MATERIAL_COVERAGE_FRACTION_MIN,
            },
            "material_cases_absolute": {
                "comparison": ">=",
                "value": MATERIAL_CASES_ABSOLUTE_MIN,
            },
            "material_cases_per_18": {
                "comparison": ">=",
                "value": MATERIAL_CASES_PER_18_MIN,
            },
            "exact_improvement_fraction": {
                "comparison": ">=",
                "value": EXACT_IMPROVEMENT_FRACTION_MIN,
            },
            "median_normalized_gap_reduction": {
                "comparison": ">=",
                "value": MEDIAN_NORMALIZED_GAP_REDUCTION_MIN,
            },
            "applies_to_each_component_and_joint": True,
        },
        "required_efficacy_slices": {
            "keys": list(REQUIRED_EFFICACY_SLICES),
            "expected_rows_per_slice": PATIENT_SPEAKERS_PER_SPLIT,
            "zero_coverage_decision": "FAIL",
            "material_case_present": True,
            "applies_to_each_component_and_joint": True,
            "improvement_fraction": {
                "comparison": ">",
                "value": REQUIRED_SLICE_IMPROVEMENT_FRACTION_EXCLUSIVE_MIN,
            },
            "median_normalized_gap_reduction": {
                "comparison": ">=",
                "value": REQUIRED_SLICE_MEDIAN_NORMALIZED_REDUCTION_MIN,
            },
            "additional_reports": ["view", "condition", "patient_vs_healthy"],
        },
        "safety_gates": {
            "residual_rms_db": {
                "comparison": "<=",
                "value": SAFETY_RESIDUAL_RMS_DB_MAX,
            },
            "cosine_similarity": {
                "comparison": ">=",
                "value": SAFETY_COSINE_SIMILARITY_MIN,
            },
            "clip_fraction": {
                "comparison": "=",
                "value": SAFETY_CLIP_FRACTION_MAX,
            },
        },
        "full_band_pathology_denoising_gates": {
            "scope": {
                "patient_degraded": "efficacy and guardrails",
                "patient_clean": "no-overprocessing control only",
                "healthy": "routing/source/topology/coverage control only",
            },
            "reference_by_role": {
                "patient_degraded": (
                    "same-speaker clean pathological CS or SV waveform"
                ),
                "patient_clean": (
                    "same-speaker clean pathological CS or SV waveform"
                ),
                "healthy": None,
            },
            "healthy_candidate_contract": "candidate_sha256_equals_base_sha256",
            "healthy_pathological_reference_applied": False,
            "alignment": (
                "tail crop to shortest waveform only; no shift, filter, "
                "resample, or metric-branch high-pass"
            ),
            "low_frequency_bands_hz": [
                list(band) for band in LOW_FREQUENCY_BANDS_HZ
            ],
            "airflow_proxy_frequency_range_hz": list(
                AIRFLOW_PROXY_FREQUENCY_RANGE_HZ
            ),
            "low_energy_quantile": LOW_ENERGY_QUANTILE,
            "pathology_db_median_gap_increase_max": (
                PATHOLOGY_DB_MEDIAN_GAP_INCREASE_MAX
            ),
            "pathology_db_worst_gap_increase_max": (
                PATHOLOGY_DB_WORST_GAP_INCREASE_MAX
            ),
            "airflow_flatness_median_gap_increase_max": (
                AIRFLOW_FLATNESS_MEDIAN_GAP_INCREASE_MAX
            ),
            "airflow_flatness_worst_gap_increase_max": (
                AIRFLOW_FLATNESS_WORST_GAP_INCREASE_MAX
            ),
            "pause_f1_median_decrease_max": PAUSE_F1_MEDIAN_DECREASE_MAX,
            "pause_f1_worst_decrease_max": PAUSE_F1_WORST_DECREASE_MAX,
            "guardrail_pass_fraction_min": GUARDRAIL_PASS_FRACTION_MIN,
            "denoising_median_change_min_db": DENOISING_MEDIAN_CHANGE_MIN_DB,
            "denoising_worst_change_min_db": DENOISING_WORST_CHANGE_MIN_DB,
            "denoising_metrics": ["snr", "si_sdr"],
            "airflow_proxy_limit": (
                "low-energy band energy and spectral flatness are signal "
                "proxies, not clinical airflow labels"
            ),
        },
        "execution_prerequisites": {
            "shimmer_db_fresh_scientific_status": SHIMMER_DB_REQUIRED_STATUS,
            "six_gradient_decision": SIX_GRADIENT_PASS_DECISION,
        },
        "boundaries": {
            "generator_optimizer_steps": 0,
            "joint_scientific_promotion_granted": False,
            "joint_panel_authorized": False,
            "healthy_no_step_does_not_establish_optimized_healthy_safety": True,
            "authoritative_training_decision": TRAINING_NO_GO,
        },
    }


def readiness_requirements() -> dict[str, Any]:
    """Describe current blockers without reading any future panel artifact."""
    registry = route_c_six_registry_records()
    shimmer = next(row for row in registry if row["name"] == "shimmer_db")
    component_no_go = load_shimmer_db_component_no_go_closure()
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "decision": "NO_GO_SIX_JOINT_PANEL_EXECUTION",
        "source_requirement_matrix": list(SOURCE_REQUIREMENT_MATRIX),
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "required_artifacts": list(REQUIRED_ARTIFACT_KEYS),
        "required_splits": list(REQUIRED_SPLITS),
        "required_views": list(REQUIRED_VIEWS),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "required_guardrails": list(REQUIRED_GUARDRAILS),
        "scientific_contract_frozen": True,
        "frozen_scientific_contract": frozen_scientific_contract(),
        "current_shimmer_db_scientific_status": component_no_go[
            "scientific_status"
        ],
        "registry_shimmer_db_scientific_status": shimmer["scientific_status"],
        "required_shimmer_db_scientific_status": SHIMMER_DB_REQUIRED_STATUS,
        "shimmer_db_component_no_go_evidence": component_no_go,
        "missing_code_stages": list(MISSING_CODE_STAGES),
        "unfrozen_scientific_contracts": list(UNFROZEN_SCIENTIFIC_CONTRACTS),
        "unbound_execution_inputs": list(UNBOUND_EXECUTION_INPUTS),
        "frozen_panel_data_requirements": list(FROZEN_PANEL_DATA_REQUIREMENTS),
        "actual_manifests_bound": False,
        "execution_authorized": False,
        "joint_scientific_promotion_granted": False,
        "joint_panel_authorized": False,
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
    }


def current_blockers() -> list[str]:
    requirements = readiness_requirements()
    blockers = list(requirements["missing_code_stages"])
    blockers.extend(requirements["unfrozen_scientific_contracts"])
    blockers.extend(
        f"actual six-joint input remains unbound: {value}"
        for value in requirements["unbound_execution_inputs"]
    )
    if requirements["current_shimmer_db_scientific_status"] == (
        SHIMMER_DB_COMPONENT_NO_GO_STATUS
    ):
        blockers.insert(
            0,
            (
                "Shimmer dB Candidate-D component-level NO-GO is bound to "
                "v23 exact-Praat evidence; six-joint panel remains closed"
            ),
        )
    elif requirements["registry_shimmer_db_scientific_status"] == (
        ROUTE_C_SIX_SCIENTIFIC_STATUS
    ):
        blockers.insert(0, "Shimmer dB scientific promotion remains pending")
    blockers.extend(
        f"not yet bound into a six-joint manifest: {key}"
        for key in REQUIRED_ARTIFACT_KEYS
    )
    return blockers


def _receipt_binds_report(
    receipt: Mapping[str, Any],
    report_path: Path,
    report_sha256: str,
    label: str,
) -> None:
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or receipt_hashes.get(report_path.name) != report_sha256
    ):
        raise ValueError(f"{label} receipt does not bind its report")


def _validate_shimmer_db_promotion(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    report_sha256: str,
    receipt_sha256: str,
    shimmer_ledger_sha256: str,
) -> dict[str, str]:
    """Validate the external exact-Praat Shimmer promotion boundary."""
    if report.get("schema_version") != SHIMMER_DB_PROMOTION_REPORT_SCHEMA:
        raise ValueError("Shimmer dB promotion report schema differs")
    if receipt.get("schema_version") != SHIMMER_DB_PROMOTION_RECEIPT_SCHEMA:
        raise ValueError("Shimmer dB promotion receipt schema differs")
    if (
        report.get("decision") != SHIMMER_DB_PROMOTION_PASS_DECISION
        or receipt.get("decision") != SHIMMER_DB_PROMOTION_PASS_DECISION
        or report.get("component_status")
        != SHIMMER_DB_PROMOTION_PASS_DECISION
        or report.get("readiness_status") != SHIMMER_DB_READINESS_PASS
    ):
        raise ValueError("Shimmer dB external exact promotion did not pass")
    if report.get("component") != "shimmer_db" or receipt.get(
        "component"
    ) != "shimmer_db":
        raise ValueError("Shimmer dB promotion component identity differs")
    summary = report.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("all_gates_pass") is not True
        or not isinstance(summary.get("mechanism_gates"), dict)
        or any(value is not True for value in summary["mechanism_gates"].values())
        or not isinstance(summary.get("integration_gates"), dict)
        or any(
            value is not True
            for value in summary["integration_gates"].values()
        )
        or summary.get("external_effect_slices", {}).get("decision") != "PASS"
        or summary.get("svd_severity_labels_available") is not False
        or summary.get("frozen_core_severity_slice_gate_applied_to_svd")
        is not False
    ):
        raise ValueError("Shimmer dB promotion gates differ")
    thresholds = report.get("fixed_scientific_thresholds")
    if not isinstance(thresholds, dict) or thresholds.get(
        "candidate_d_fixed_alpha"
    ) != 0.001:
        raise ValueError("Shimmer dB frozen alpha differs")
    required_true = (
        "candidate_exact_outcomes_opened_after_selector_seal",
        "old_v18_evidence_kept_separate",
        "opened24_v23_severity_accuracy_calibration_anti_shortcut_bound",
        "external_speaker_gate_pass",
        "bounded_waveform_promotion_pass",
        "scientific_promotion_granted",
        "six_component_readiness_eligible",
    )
    if any(report.get(key) is not True for key in required_true):
        raise ValueError("Shimmer dB promotion evidence boundary differs")
    if (
        receipt.get("candidate_exact_outcomes_opened_after_selector_seal")
        is not True
        or receipt.get("old_v18_evidence_kept_separate") is not True
        or receipt.get("scientific_promotion_granted") is not True
        or receipt.get("six_component_readiness_eligible") is not True
    ):
        raise ValueError("Shimmer dB promotion receipt boundary differs")
    for label, value in (("report", report), ("receipt", receipt)):
        if value.get("joint_panel_authorized") is not False:
            raise ValueError(f"Shimmer dB {label} over-authorized joint panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"Shimmer dB {label} optimizer boundary differs")
        if value.get("formal_generator_training_submitted") is not False:
            raise ValueError(f"Shimmer dB {label} overclaims formal training")
        if value.get("authoritative_training_decision") != TRAINING_NO_GO:
            raise ValueError(f"Shimmer dB {label} training decision differs")
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or receipt_hashes.get("diagnostic_report.json") != report_sha256
        or receipt_hashes.get("selector_seal.json")
        != report.get("selector_seal_sha256")
        or not _is_sha256(receipt_hashes.get("external_svd_exact_results.csv"))
        or not _is_sha256(
            receipt_hashes.get("family_selector_preselection.csv")
        )
    ):
        raise ValueError("Shimmer dB promotion receipt artifact binding differs")
    evidence = report.get("evidence_bindings")
    source_sha256 = report.get("source_sha256")
    if (
        not isinstance(evidence, dict)
        or evidence.get("updated_speaker_ledger_sha256")
        != shimmer_ledger_sha256
        or not isinstance(source_sha256, dict)
        or source_sha256.get("updated_speaker_ledger")
        != shimmer_ledger_sha256
    ):
        raise ValueError("Shimmer dB promotion speaker-ledger binding differs")
    return {
        "report": report_sha256,
        "receipt": receipt_sha256,
        "shimmer_db_prior_panel_speaker_ledger": shimmer_ledger_sha256,
    }


def _ledger_entries(
    ledger: Mapping[str, Any],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if ledger.get("schema_version") != PRIOR_PANEL_LEDGER_SCHEMA:
        raise ValueError(f"{label} schema differs")
    if ledger.get("exact_outcomes_used_for_selection") is not False:
        raise ValueError(f"{label} was selected using exact outcomes")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} entries are unavailable")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} entry is not a mapping")
        dataset = str(entry.get("dataset", "")).strip().upper()
        speaker_id = str(entry.get("speaker_id", "")).strip()
        canonical = str(entry.get("canonical_speaker_id", ""))
        if (
            not dataset
            or not speaker_id
            or canonical != f"{dataset}:{speaker_id}"
            or canonical in indexed
        ):
            raise ValueError(f"{label} canonical speaker identity differs")
        indexed[canonical] = entry
    return indexed


def _validate_prior_panel_ledger_merge(
    merged_ledger: Mapping[str, Any],
    shimmer_ledger: Mapping[str, Any],
    *,
    shimmer_ledger_sha256: str,
) -> dict[str, int]:
    """Require the six-joint ledger to preserve the full Shimmer ledger."""
    shimmer_entries = _ledger_entries(shimmer_ledger, "Shimmer dB prior ledger")
    merged_entries = _ledger_entries(merged_ledger, "six-joint prior ledger")
    source_hashes = merged_ledger.get("source_ledger_sha256")
    if (
        not isinstance(source_hashes, dict)
        or source_hashes.get(SHIMMER_DB_LEDGER_SOURCE_KEY)
        != shimmer_ledger_sha256
    ):
        raise ValueError("six-joint prior ledger does not bind the Shimmer ledger")
    if not set(shimmer_entries).issubset(merged_entries):
        raise ValueError("six-joint prior ledger omits Shimmer speakers")
    for speaker, shimmer_entry in shimmer_entries.items():
        merged_entry = merged_entries[speaker]
        if any(
            merged_entry.get(key) != value
            for key, value in shimmer_entry.items()
        ):
            raise ValueError(
                f"six-joint prior ledger rewrites Shimmer history: {speaker}"
            )
    return {
        "shimmer_speaker_count": len(shimmer_entries),
        "merged_speaker_count": len(merged_entries),
    }


def validate_readiness_authorization(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    report_name: str,
    report_sha256: str,
    expected_source_commit: str,
    expected_input_sha256: Mapping[str, str],
) -> None:
    """Require an independent PASS receipt before waveform preparation."""
    if report.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError("six-joint readiness authorization schema differs")
    if receipt.get("schema_version") != READINESS_RECEIPT_SCHEMA_VERSION:
        raise ValueError("six-joint readiness receipt schema differs")
    if (
        report.get("decision") != READINESS_PASS_DECISION
        or receipt.get("decision") != READINESS_PASS_DECISION
    ):
        raise ValueError("six-joint readiness authorization did not pass")
    if (
        report.get("source_commit") != expected_source_commit
        or receipt.get("source_commit") != expected_source_commit
    ):
        raise ValueError("six-joint readiness source binding differs")
    for label, value in (("report", report), ("receipt", receipt)):
        if value.get("execution_authorized") is not True:
            raise ValueError(f"six-joint readiness {label} did not authorize execution")
        if value.get("joint_panel_authorized") is not True:
            raise ValueError(f"six-joint readiness {label} did not authorize panel")
        if value.get("joint_scientific_promotion_granted") is not False:
            raise ValueError(f"six-joint readiness {label} overclaims science")
        if value.get("candidate_exact_outcomes_opened") is not False:
            raise ValueError(f"six-joint readiness {label} opened exact outcomes")
        if value.get("fresh_panel_opened") is not False:
            raise ValueError(f"six-joint readiness {label} opened fresh panel")
        if value.get("generator_optimizer_steps") != 0:
            raise ValueError(f"six-joint readiness {label} optimizer boundary differs")
        if value.get("authoritative_training_decision") != TRAINING_NO_GO:
            raise ValueError(f"six-joint readiness {label} training decision differs")
    inputs = report.get("input_sha256")
    if not isinstance(inputs, dict) or any(
        inputs.get(key) != value for key, value in expected_input_sha256.items()
    ):
        raise ValueError("six-joint readiness/preparation input binding differs")
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or receipt_hashes.get(report_name) != report_sha256
    ):
        raise ValueError("six-joint readiness receipt/report binding differs")


def _validate_five_component_evidence(
    artifacts: Mapping[str, Mapping[str, str]],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    """Validate the evidence set frozen by the accepted five-active audit."""
    required = {
        *FIVE_COMPONENT_EVIDENCE_KEYS,
        "five_gradient_report",
        "five_gradient_receipt",
    }
    if set(artifacts) != required or set(paths) != required:
        raise ValueError("five-component frozen evidence keys differ")
    for key in required:
        path = paths[key]
        binding = artifacts[key]
        if (
            not path.is_absolute()
            or not path.is_file()
            or set(binding) != {"path", "sha256"}
            or binding["path"] != str(path)
            or not _is_sha256(binding["sha256"])
            or sha256_file(path) != binding["sha256"]
        ):
            raise ValueError(f"five-component frozen evidence hash differs: {key}")
    for component, (report_key, receipt_key, decision) in (
        FIVE_COMPONENT_REPORT_CONTRACTS.items()
    ):
        report = _read_json_mapping(paths[report_key], f"{component} report")
        receipt = _read_json_mapping(paths[receipt_key], f"{component} receipt")
        if report.get("decision") != decision or receipt.get("decision") != decision:
            raise ValueError(f"{component} frozen evidence decision differs")
        waveform_schema = {
            "cpps": "direct-avqi-waveform-optimization-v3",
            "tilt": "direct-avqi-waveform-optimization-v1",
        }.get(component)
        if waveform_schema is not None and report.get("schema_version") != (
            waveform_schema
        ):
            raise ValueError(f"{component} frozen evidence schema differs")
        _require_optimizer_zero(report, f"{component} report")
        _require_optimizer_zero(receipt, f"{component} receipt")
        if report.get("formal_pathology_training_submitted") is not False:
            raise ValueError(f"{component} report overclaims training")
        _receipt_binds_report(
            receipt,
            paths[report_key],
            artifacts[report_key]["sha256"],
            component,
        )
        if component in {"hnr", "shimmer_percent", "slope"}:
            final = report.get("final")
            if (
                report.get("final_exact_panel_opened") is not True
                or not isinstance(final, dict)
                or final.get("decision") != "PASS"
            ):
                raise ValueError(f"{component} fresh-panel evidence differs")
            _require_all_true_gates(final, f"{component} final panel")
        elif component in {"cpps", "tilt"}:
            summary = report.get("summary")
            component_gates = (
                summary.get("component_gates")
                if isinstance(summary, dict)
                else None
            )
            component_gate = (
                component_gates.get(component)
                if isinstance(component_gates, dict)
                else None
            )
            safety = summary.get("safety") if isinstance(summary, dict) else None
            if (
                not isinstance(component_gate, dict)
                or component_gate.get("decision") != "PASS"
                or not isinstance(safety, dict)
                or safety.get("decision") != "PASS"
            ):
                raise ValueError(f"{component} component-level evidence differs")
            _require_all_true_gates(component_gate, f"{component} component")

    slope_receipt = _read_json_mapping(paths["slope_receipt"], "slope receipt")
    slope_hashes = slope_receipt.get("artifact_sha256")
    if not isinstance(slope_hashes, dict):
        raise ValueError("slope receipt artifact bindings differ")
    for key in ("slope_final_panel_seal", "slope_final_results"):
        if slope_hashes.get(paths[key].name) != artifacts[key]["sha256"]:
            raise ValueError(f"slope receipt does not bind {key}")

    report = _read_json_mapping(paths["five_gradient_report"], "five-gradient report")
    receipt = _read_json_mapping(
        paths["five_gradient_receipt"], "five-gradient receipt"
    )
    if (
        report.get("schema_version")
        != "avqi_route_c_five_component_gradient_audit_v1"
        or report.get("decision") != "PASS_ROUTE_C_FIVE_ACTIVE_CODE_GRADIENT_AUDIT"
        or receipt.get("decision") != report.get("decision")
    ):
        raise ValueError("five-gradient frozen decision differs")
    _require_all_true_gates(report, "five-gradient audit")
    _require_optimizer_zero(report, "five-gradient report")
    _require_optimizer_zero(receipt, "five-gradient receipt")
    if tuple(receipt.get("active_components", ())) != ROUTE_C_FIVE_ACTIVE_COMPONENTS:
        raise ValueError("five-gradient active components differ")
    if receipt.get("inactive_slots") != ["shimmer_db"]:
        raise ValueError("five-gradient inactive slot differs")
    _receipt_binds_report(
        receipt,
        paths["five_gradient_report"],
        artifacts["five_gradient_report"]["sha256"],
        "five-gradient",
    )
    source_evidence = report.get("source_evidence")
    if not isinstance(source_evidence, dict) or set(source_evidence) != set(
        FIVE_COMPONENT_EVIDENCE_KEYS
    ):
        raise ValueError("five-gradient source-evidence keys differ")
    for key in FIVE_COMPONENT_EVIDENCE_KEYS:
        if source_evidence[key] != artifacts[key]:
            raise ValueError(f"five-gradient source evidence differs: {key}")
    return {key: artifacts[key]["sha256"] for key in FIVE_COMPONENT_EVIDENCE_KEYS}


def _validate_six_gradient(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    report_sha256: str,
    expected_source_evidence: Mapping[str, str],
) -> dict[str, float]:
    if report.get("schema_version") != SIX_GRADIENT_SCHEMA_VERSION:
        raise ValueError("six-component gradient schema differs")
    if report.get("decision") != SIX_GRADIENT_PASS_DECISION:
        raise ValueError("six-component gradient audit did not pass")
    if receipt.get("schema_version") != SIX_GRADIENT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("six-component gradient receipt schema differs")
    if receipt.get("decision") != report["decision"]:
        raise ValueError("six-component gradient receipt decision differs")
    if (
        report.get("joint_panel_decision") != SIX_GRADIENT_JOINT_PANEL_NO_GO
        or receipt.get("joint_panel_decision")
        != SIX_GRADIENT_JOINT_PANEL_NO_GO
    ):
        raise ValueError("six-component gradient joint-panel decision differs")
    if (
        tuple(report.get("active_components", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or tuple(receipt.get("active_components", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or tuple(report.get("active_components", ()))
        != FROZEN_SIX_GRADIENT_COMPONENTS
    ):
        raise ValueError("six-component gradient active order differs")
    if report.get("source_evidence_sha256") != expected_source_evidence:
        raise ValueError("six-component gradient source-evidence binding differs")
    if report.get("frozen_contract") != six_gradient_decision_requirements().get(
        "frozen_contract"
    ):
        raise ValueError("six-component gradient frozen contract differs")
    precedent = report.get("accepted_numeric_precedent")
    expected_precedent = {
        "slurm_job_id": FROZEN_FIVE_JOB_ID,
        "report_sha256": FROZEN_FIVE_REPORT_SHA256,
        "receipt_sha256": FROZEN_FIVE_RECEIPT_SHA256,
    }
    if precedent != expected_precedent or receipt.get(
        "accepted_numeric_precedent"
    ) != expected_precedent:
        raise ValueError("six-component gradient numeric precedent differs")
    raw = report.get("raw_measurement_evidence")
    raw_receipt_hashes = receipt.get("raw_measurement_sha256")
    if not isinstance(raw, dict) or not isinstance(raw_receipt_hashes, dict):
        raise ValueError("six-component raw measurement binding is unavailable")
    if (
        raw.get("raw_decision") != SIX_GRADIENT_RAW_PENDING_DECISION
        or raw.get("raw_artifacts_rewritten") is not False
        or receipt.get("raw_artifacts_rewritten") is not False
        or raw_receipt_hashes
        != {"report": raw.get("report_sha256"), "receipt": raw.get("receipt_sha256")}
        or not _is_sha256(raw.get("report_sha256"))
        or not _is_sha256(raw.get("receipt_sha256"))
    ):
        raise ValueError("six-component raw measurement binding differs")
    decision_source = report.get("decision_source")
    if (
        not isinstance(decision_source, dict)
        or receipt.get("source_commit") != decision_source.get("head")
        or receipt.get("source_branch") != decision_source.get("branch")
    ):
        raise ValueError("six-component gradient decision source differs")
    implementation = report.get("implementation_sha256")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(SIX_GRADIENT_DECISION_IMPLEMENTATION_KEYS)
        or implementation != receipt.get("implementation_sha256")
        or any(not _is_sha256(value) for value in implementation.values())
    ):
        raise ValueError("six-component gradient decision implementation differs")
    immutability = report.get("post_evaluation_immutability")
    expected_immutability_hashes = {
        "raw_report": raw["report_sha256"],
        "raw_receipt": raw["receipt_sha256"],
        "five_precedent_report": FROZEN_FIVE_REPORT_SHA256,
        "five_precedent_receipt": FROZEN_FIVE_RECEIPT_SHA256,
    }
    if (
        not isinstance(immutability, dict)
        or immutability.get("verified") is not True
        or immutability.get("artifact_sha256") != expected_immutability_hashes
        or receipt.get("post_evaluation_immutability") != immutability
    ):
        raise ValueError("six-component gradient input immutability differs")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != set(SIX_GRADIENT_FROZEN_GATE_KEYS)
        or any(value is not True for value in gates.values())
    ):
        raise ValueError("six-component gradient frozen gates failed")
    summary = report.get("measurement_summary")
    if not isinstance(summary, dict):
        raise ValueError("six-component gradient measurement summary is unavailable")
    weights = _finite_mapping(
        summary.get("calibration_inverse_gradient_weights"),
        ROUTE_C_SIX_ACTIVE_COMPONENTS,
        "six-component frozen weights",
        positive=True,
    )
    ratio = summary.get("calibration_weighted_median_norm_ratio")
    maximum_share = summary.get("maximum_weighted_component_norm_share")
    minimum_joint_cosine = summary.get("minimum_component_to_joint_cosine")
    if (
        not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio > MAXIMUM_CALIBRATION_WEIGHTED_MEDIAN_RATIO
        or not isinstance(maximum_share, (int, float))
        or not math.isfinite(maximum_share)
        or maximum_share > MAXIMUM_WEIGHTED_COMPONENT_SHARE
        or not isinstance(minimum_joint_cosine, (int, float))
        or not math.isfinite(minimum_joint_cosine)
        or minimum_joint_cosine < 0.0
        or summary.get("calibration_cases") != 4
        or summary.get("holdout_cases") != 4
        or summary.get("pairwise_negative_values_are_diagnostic_only") is not True
    ):
        raise ValueError("six-component gradient frozen summary differs")
    required_false = (
        "scientific_promotion_granted",
        "joint_scientific_promotion_granted",
        "joint_panel_authorized",
        "combined_final_panel_opened",
        "fresh_panel_opened",
        "exact_candidate_scoring_requested",
        "waveform_generation_performed",
        "formal_generator_training_submitted",
    )
    if any(report.get(key) is not False for key in required_false):
        raise ValueError("six-component gradient report overclaims science")
    if any(receipt.get(key) is not False for key in required_false):
        raise ValueError("six-component gradient receipt overclaims science")
    if (
        report.get("scientific_contract_frozen_before_six_holdout_open") is not True
        or report.get("raw_measurement_recomputed") is not False
        or report.get("authoritative_training_decision") != TRAINING_NO_GO
        or receipt.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("six-component gradient decision boundaries differ")
    _require_optimizer_zero(report, "six-component gradient report")
    _require_optimizer_zero(receipt, "six-component gradient receipt")
    receipt_hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(receipt_hashes, dict)
        or len(receipt_hashes) != 1
        or report_sha256 not in receipt_hashes.values()
    ):
        raise ValueError("six-component gradient receipt does not bind its report")
    return weights


PANEL_ROW_FIELDS = {
    "case_id",
    "dataset",
    "speaker_id",
    "split",
    "view",
    "condition",
    "label",
    "optimization_role",
}


def _validate_panel_rows(
    rows: Any,
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} has no rows")
    if any(
        not isinstance(row, dict) or not PANEL_ROW_FIELDS <= set(row)
        for row in rows
    ):
        raise ValueError(f"{label} row fields differ")
    if any(FORBIDDEN_PANEL_ROW_FIELDS & set(row) for row in rows):
        raise ValueError(f"{label} contains forbidden result-derived row fields")
    rows_by_case = {str(row["case_id"]): row for row in rows}
    if len(rows_by_case) != len(rows) or "" in rows_by_case:
        raise ValueError(f"{label} case IDs are not unique")
    if len(rows_by_case) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"{label} row count differs")
    for row in rows:
        expected_role = None
        if row["label"] == "patient":
            expected_role = (
                CLEAN_PATHOLOGICAL_ROLE
                if row["condition"] == "clean"
                else PATHOLOGICAL_ROLE
            )
        elif row["label"] == "healthy":
            expected_role = HEALTHY_ROLE
        if (
            row["dataset"] != SOURCE_DATASET
            or row["split"] not in REQUIRED_SPLITS
            or row["view"] not in REQUIRED_VIEWS
            or row["condition"] not in REQUIRED_CONDITIONS
            or expected_role != row["optimization_role"]
        ):
            raise ValueError(f"{label} row semantics differ")

    expected_matrix = {
        (condition, view)
        for condition in REQUIRED_CONDITIONS
        for view in REQUIRED_VIEWS
    }
    speakers = {str(row["speaker_id"]) for row in rows}
    if "" in speakers:
        raise ValueError(f"{label} has an empty speaker ID")
    if len(speakers) != EXPECTED_TOTAL_SPEAKERS:
        raise ValueError(f"{label} speaker count differs")
    for speaker in speakers:
        speaker_rows = [row for row in rows if row["speaker_id"] == speaker]
        if (
            len({row["split"] for row in speaker_rows}) != 1
            or len({row["label"] for row in speaker_rows}) != 1
            or len(speaker_rows) != len(expected_matrix)
            or {(row["condition"], row["view"]) for row in speaker_rows}
            != expected_matrix
        ):
            raise ValueError(f"{label} speaker matrix differs: {speaker}")
    speakers_by_split = {
        split: {str(row["speaker_id"]) for row in rows if row["split"] == split}
        for split in REQUIRED_SPLITS
    }
    if speakers_by_split["calibration"] & speakers_by_split["final"]:
        raise ValueError(f"{label} calibration/final speakers overlap")
    for split in REQUIRED_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_speakers = speakers_by_split[split]
        if (
            len(split_rows) != ROWS_PER_SPLIT
            or len(split_speakers) != SPEAKERS_PER_SPLIT
        ):
            raise ValueError(f"{label} {split} split counts differ")
        speakers_by_label = {
            value: {
                str(row["speaker_id"])
                for row in split_rows
                if row["label"] == value
            }
            for value in REQUIRED_LABELS
        }
        if (
            len(speakers_by_label["patient"]) != PATIENT_SPEAKERS_PER_SPLIT
            or len(speakers_by_label["healthy"]) != HEALTHY_SPEAKERS_PER_SPLIT
            or set().union(*speakers_by_label.values()) != split_speakers
        ):
            raise ValueError(f"{label} {split} label counts differ")
        patient_rows = [row for row in split_rows if row["label"] == "patient"]
        healthy_rows = [row for row in split_rows if row["label"] == "healthy"]
        patient_clean = [
            row for row in patient_rows if row["condition"] == "clean"
        ]
        patient_degraded = [
            row
            for row in patient_rows
            if row["condition"] in DEGRADED_EFFICACY_CONDITIONS
        ]
        if (
            len(patient_degraded)
            != PATIENT_DEGRADED_EFFICACY_ROWS_PER_SPLIT
            or len(patient_clean) != PATIENT_CLEAN_CONTROL_ROWS_PER_SPLIT
            or len(healthy_rows) != HEALTHY_GUARDRAIL_ROWS_PER_SPLIT
        ):
            raise ValueError(f"{label} {split} row-role counts differ")
    return rows_by_case, {
        split: sorted(values) for split, values in speakers_by_split.items()
    }


def _validate_split_seal(
    seal: Mapping[str, Any],
    *,
    gate_sha256: str,
    target_sha256: str,
    ledger_sha256: str,
    source_sha256: str,
) -> dict[str, list[str]]:
    if seal.get("schema_version") != FROZEN_SPLIT_SEAL_SCHEMA_VERSION:
        raise ValueError("six-joint frozen split-seal schema differs")
    required_values = {
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "source_dataset": SOURCE_DATASET,
        "sv_metadata_sha256": SVD_SV_METADATA_SHA256,
        "cs_metadata_sha256": SVD_CS_METADATA_SHA256,
        "health_status_mapping": dict(SVD_HEALTH_STATUS_MAPPING),
        "paired_cs_sv_same_session_required": True,
        "minimum_raw_mono_duration_seconds": dict(
            SVD_MINIMUM_RAW_MONO_SECONDS
        ),
        "eligible_session_per_speaker": "minimum_numeric_session_id",
        "speaker_selection_salt": SVD_SPEAKER_SELECTION_SALT,
        "speaker_rank_digest": "SHA256(salt:speaker_id:session_id)",
        "prior_ledger_excluded_before_hash_ranking": True,
        "diagnosis_used_for_selection": False,
        "exact_scores_opened": False,
        "speaker_split_before_simulation": True,
        "gap_simulation_inventory_sha256": GAP_SIMULATION_INVENTORY_SHA256,
        "gap_rir_manifest_sha256": GAP_RIR_MANIFEST_SHA256,
        "gap_noise_manifest_sha256": GAP_NOISE_MANIFEST_SHA256,
        "recipe_assignment_salt": GAP_RECIPE_ASSIGNMENT_SALT,
        "recipe_uid_unique_per_row": True,
        "recipe_uid_reused_across_splits": False,
        "condition_recipe_semantics": {
            "clean": "no_simulation",
            "rir_only": "rir_only",
            "snr20": "rir_plus_noise_fixed_target_snr_20db",
            "snr10": "rir_plus_noise_fixed_target_snr_10db",
        },
        "metadata_only_result_blind_selection": True,
        "mild_severe_labels_created": False,
        "prior_panel_speaker_overlap": 0,
        "waveform_steps": 1,
        "one_global_alpha": True,
        "gradient_normalization": "waveform_rms_normalized",
        "alpha_grid": list(GLOBAL_ALPHA_GRID),
        "zero_alpha_selectable": False,
        "alpha_required_gate_families": list(ALPHA_REQUIRED_GATE_FAMILIES),
        "alpha_required_gate_split": "calibration",
        "alpha_selection_objective": ALPHA_SELECTION_OBJECTIVE,
        "alpha_selection_tie_break": ALPHA_SELECTION_TIE_BREAK,
        "alpha_selection_split": "calibration",
        "final_tuning_permitted": False,
        "optimization_weight_source_decision": SIX_GRADIENT_PASS_DECISION,
        "optimization_weights_calibration_only": True,
        "optimization_weights_used_for_exact_joint_decision": False,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization_target_mean_field": NORMALIZATION_TARGET_MEAN_FIELD,
        "normalization_target_scale_field": NORMALIZATION_TARGET_SCALE_FIELD,
        "normalization_refit_permitted": False,
        "healthy_no_step_does_not_establish_optimized_healthy_safety": True,
    }
    if any(seal.get(key) != value for key, value in required_values.items()):
        raise ValueError("six-joint split seal semantics differ")
    bindings = {
        "joint_gate_contract_sha256": gate_sha256,
        "target_value_protocol_sha256": target_sha256,
        "prior_panel_speaker_ledger_sha256": ledger_sha256,
        "fresh_speaker_source_manifest_sha256": source_sha256,
    }
    if any(seal.get(key) != value for key, value in bindings.items()):
        raise ValueError("six-joint split seal hash binding differs")
    _, speakers_by_split = _validate_panel_rows(
        seal.get("rows"), "six-joint split seal"
    )
    _require_optimizer_zero(seal, "six-joint split seal")
    return speakers_by_split


def validate_readiness_manifest(
    manifest: Mapping[str, Any],
    *,
    registry_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate fail-closed headers before any execution authorization."""
    if manifest.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError("six-joint readiness schema differs")
    if (
        manifest.get("scientific_contract_schema_version")
        != FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("six-joint frozen scientific contract binding differs")
    if manifest.get("candidate_exact_outcomes_opened") is not False:
        raise ValueError("six-joint readiness opened candidate outcomes")
    if manifest.get("fresh_panel_opened") is not False:
        raise ValueError("six-joint readiness opened a fresh panel")
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("six-joint source commit binding differs")
    _require_optimizer_zero(manifest, "six-joint readiness manifest")

    component_no_go = load_shimmer_db_component_no_go_closure()
    if component_no_go["scientific_status"] == (
        SHIMMER_DB_COMPONENT_NO_GO_STATUS
    ):
        raise ValueError(
            "Shimmer dB Candidate-D component-level NO-GO is bound to v23 "
            "exact-Praat evidence; joint panel closed"
        )

    registry = (
        route_c_six_registry_records()
        if registry_records is None
        else registry_records
    )
    if tuple(row.get("name") for row in registry) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("six-joint live registry component order differs")
    shimmer = registry[3]
    if shimmer.get("scientific_status") != SHIMMER_DB_REQUIRED_STATUS:
        raise ValueError(
            "Shimmer dB scientific promotion is still pending; joint panel closed"
        )
    if UNFROZEN_SCIENTIFIC_CONTRACTS:
        raise ValueError(
            "six-joint scientific schemas remain unfrozen: "
            + "; ".join(UNFROZEN_SCIENTIFIC_CONTRACTS)
        )
    raise ValueError(
        "six-joint execution remains closed: actual manifests, evidence, "
        "speaker selection, target bank, and joint gradients remain unbound"
    )


def repository_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(root: Path, expected_commit: str) -> dict[str, str]:
    resolved = root.resolve()
    head = repository_value(resolved, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("six-joint source HEAD differs")
    if repository_value(resolved, "status", "--porcelain"):
        raise ValueError("six-joint preflight requires a clean worktree")
    return {
        "root": str(resolved),
        "head": head,
        "branch": repository_value(resolved, "branch", "--show-current"),
    }


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("six-joint readiness manifest hash differs")
    return _read_json_mapping(path, "six-joint readiness manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execution_inputs = (
        args.manifest,
        args.manifest_sha256,
        args.source_root,
        args.source_commit,
    )
    if args.requirements_only:
        if any(value is not None for value in execution_inputs):
            raise ValueError("requirements-only mode accepts no execution inputs")
        report = readiness_requirements()
        report["blockers"] = current_blockers()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    if any(value is None for value in execution_inputs):
        raise ValueError("six-joint preflight inputs are incomplete")
    source = validate_source(args.source_root, args.source_commit)
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    if manifest.get("source_commit") != source["head"]:
        raise ValueError("six-joint manifest/source commit binding differs")
    validate_readiness_manifest(manifest)
    raise ValueError("six-joint preflight returned without authorization")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(
            json.dumps(
                {
                    "decision": "NO_GO_SIX_JOINT_PANEL_EXECUTION",
                    "execution_authorized": False,
                    "reason": str(error),
                    "generator_optimizer_steps": 0,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2) from None
