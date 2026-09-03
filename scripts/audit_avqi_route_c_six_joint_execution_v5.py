#!/usr/bin/env python3
"""Bind and authorize the sealed Candidate-E six-joint execution package.

This audit consumes already sealed artifacts.  It neither selects speakers nor
creates waveforms, opens candidate exact outcomes, or trains the generator.  A
PASS authorizes only the one-step six-component joint panel described by the
frozen v1 scientific contract.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from model.avqi_route_c import ROUTE_C_SIX_ACTIVE_COMPONENTS
from model.avqi_route_c_candidate_e import CANDIDATE_E_REFERENCE_SHA256
from model.avqi_route_c_v19_contracts import sha256_file
from scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v4 import (
    COMPONENT_PASS,
    JOINT_NO_GO,
    PROMOTION_PASS,
    PROMOTION_RECEIPT_SHA256,
    PROMOTION_REPORT_SHA256,
    TRAINING_NO_GO,
    UPDATED_LEDGER_SHA256,
    validate_ledger,
    validate_promotion,
    validate_receipt_artifact_files,
)
from scripts.audit_avqi_route_c_six_joint_panel_readiness import (
    FIVE_COMPONENT_EVIDENCE_KEYS,
    FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
    READINESS_PASS_DECISION,
    READINESS_RECEIPT_SCHEMA_VERSION,
    READINESS_SCHEMA_VERSION,
    READINESS_SOURCE_EVIDENCE_KEYS,
    SIX_GRADIENT_PASS_DECISION,
    _require_optimizer_zero,
    _validate_five_component_evidence,
    _validate_panel_rows,
    _validate_prior_panel_ledger_merge,
    _validate_six_gradient,
    _validate_split_seal,
    frozen_scientific_contract,
)
from scripts.decide_avqi_route_c_six_component_gradients import (
    CANDIDATE_E_EVIDENCE_KEYS,
    RAW_SOURCE_EVIDENCE_KEYS,
)
from scripts.evaluate_avqi_route_c_six_component_gradients import (
    TOPOLOGY_RECEIPT_SCHEMA_VERSION,
    TOPOLOGY_SEAL_DECISION,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import (
    run_exact,
    validate_exact_authority,
)
from scripts.evaluate_avqi_shimmer_fresh_panel import read_fixed_recipes
from scripts.materialize_avqi_route_c_six_joint_inputs_v5 import (
    EXPECTED_GENERATOR_CHECKPOINT_SHA256,
    EXPECTED_GENERATOR_CONFIG_SHA256,
    EXPECTED_SIMULATION_CONFIG_SHA256,
    EXPECTED_SIMULATION_SOURCE_SHA256,
    MATERIALIZATION_INPUT_NAMES,
    MATERIALIZATION_DECISION,
    MATERIALIZATION_RECEIPT_SCHEMA,
    RUNTIME_BINDING_SCHEMA,
    validate_recipe_manifest,
    validate_source_manifest,
)
from scripts.prepare_avqi_route_c_six_joint_inputs_v5 import (
    RECIPE_MANIFEST_SCHEMA,
    SELECTION_DECISION,
    SELECTION_RECEIPT_SCHEMA,
    eligible_speakers,
    prior_speakers,
    read_csv,
    recipe_manifest as build_recipe_manifest,
    select_speakers,
    source_manifest as build_source_manifest,
)
from scripts.prepare_avqi_route_c_six_joint_waveforms import (
    validate_gradient_manifest,
    validate_target_bank,
)
from scripts.seal_avqi_route_c_exact_authority_v1 import (
    RECEIPT_SCHEMA as EXACT_AUTHORITY_RECEIPT_SCHEMA,
    SEAL_DECISION as EXACT_AUTHORITY_SEAL_DECISION,
)


PACKAGE_SCHEMA = "avqi-route-c-six-joint-execution-package-v5"
PACKAGE_RECEIPT_SCHEMA = "avqi-route-c-six-joint-execution-package-receipt-v5"
EXPECTED_V4_REPORT_SHA256 = (
    "7357bb0fb6b7fd655cfc4ea693473fa76e48b714e87a50eb8ecb95af045821b0"
)
EXPECTED_V4_RECEIPT_SHA256 = (
    "18705aba74601b6344013ce207eacf472846071b0c55cdac5c030f43663d105f"
)
EXPECTED_CANDIDATE_E_COMMIT = "109f398d607bce936a9576c826ff74ce0ea9f636"
EXPECTED_CANDIDATE_E_WORKER_SHA256 = (
    "c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f"
)
EXPECTED_CANDIDATE_E_RUNTIME_CLIENT_SHA256 = (
    "28e48fc3de99bb2c7258559f4f58be2760c7804f53a08bab162fff670b36153b"
)
EXPECTED_CANDIDATE_E_SELECTOR_SHA256 = (
    "7401b4b80f6dbb546a4a88886c469bb4df6b4681bad9314f1244a046fbb2b69b"
)
EXPECTED_CANDIDATE_E_RUNTIME_CONFIG_SHA256 = (
    "4dec4b018b6cd9f7a5a7f87966cc7f2dde057f152df256f65fc397faefb53b98"
)
EXECUTION_INPUT_NAMES = (
    "candidate_e_joint_runtime_binding",
    "six_gradient_report",
    "six_gradient_receipt",
    "fresh_panel_split_seal",
    "fresh_speaker_source_manifest",
    "clean_target_label_bank",
    "joint_gradient_manifest",
)
SUPPORTING_ARTIFACT_NAMES = (
    "candidate_e_promotion_report",
    "candidate_e_promotion_receipt",
    "candidate_e_prior_panel_speaker_ledger",
    "candidate_e_reference_source",
    "candidate_e_runtime_client",
    "candidate_e_worker",
    "candidate_e_selector_source",
    "candidate_e_runtime_config",
    "readiness_v4_report",
    "readiness_v4_receipt",
    "selection_receipt",
    "joint_recipe_assignment_manifest",
    "prior_panel_speaker_ledger",
    "joint_gate_contract",
    "target_value_protocol_contract",
    "fixed_recipes",
    "six_gradient_raw_report",
    "six_gradient_raw_receipt",
    "materialization_receipt",
    "exact_code_tree_manifest",
    "exact_runtime_manifest",
    "exact_authority_receipt",
    "topology_receipt",
)
CANDIDATE_E_PARITY_FUNCTIONS = (
    "pcm16_ste",
    "praat_pcm16_ste",
    "next_power_of_two",
    "official_stop_hann",
    "exact_metric_branch_ste",
    "asymmetric_hann_rms",
    "fixed_pulse_shimmer_db",
    "candidate_e_proxy",
    "project_cycle_gain_gradient_fixed_order",
)


class _CandidateEParityNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        node.type_comment = None
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        for argument in arguments:
            argument.annotation = None
            argument.type_comment = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "ValueError"
        ):
            node.exc.args = [ast.Constant("normalized-error-message")]
        return node


def candidate_e_function_parity(
    reference_path: Path,
    integrated_path: Path,
) -> dict[str, bool]:
    def functions(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    reference = functions(reference_path)
    integrated = functions(integrated_path)
    if any(
        name not in reference or name not in integrated
        for name in CANDIDATE_E_PARITY_FUNCTIONS
    ):
        raise ValueError("Candidate-E parity function coverage differs")
    output = {}
    for name in CANDIDATE_E_PARITY_FUNCTIONS:
        reference_node = _CandidateEParityNormalizer().visit(reference[name])
        integrated_node = _CandidateEParityNormalizer().visit(integrated[name])
        output[name] = ast.dump(
            reference_node,
            include_attributes=False,
        ) == ast.dump(integrated_node, include_attributes=False)
    if any(value is not True for value in output.values()):
        raise ValueError("Candidate-E integrated function parity differs")
    return output


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON mapping")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_bindings(
    rows: list[list[str]] | None,
    expected_names: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, str]]:
    if rows is None:
        raise ValueError(f"{label} bindings are unavailable")
    bindings: dict[str, dict[str, str]] = {}
    for name, raw_path, expected_sha256 in rows:
        if name in bindings:
            raise ValueError(f"duplicate {label} binding: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(f"{label} hash differs: {name}")
        bindings[name] = {"path": str(path), "sha256": observed}
    if set(bindings) != set(expected_names):
        missing = sorted(set(expected_names) - set(bindings))
        extra = sorted(set(bindings) - set(expected_names))
        raise ValueError(f"{label} names differ: missing={missing}, extra={extra}")
    return bindings


def validate_receipt_input_bindings(
    receipt: Mapping[str, Any],
    expected_names: set[str],
    label: str,
) -> dict[str, dict[str, str]]:
    bindings = receipt.get("input_binding")
    hashes = receipt.get("input_sha256")
    if (
        not isinstance(bindings, dict)
        or not isinstance(hashes, dict)
        or set(bindings) != expected_names
        or set(hashes) != expected_names
    ):
        raise ValueError(f"{label} input names differ")
    parsed: dict[str, dict[str, str]] = {}
    for name in sorted(expected_names):
        binding = bindings[name]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"{label} input binding differs: {name}")
        path = Path(str(binding["path"]))
        digest = str(binding["sha256"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or sha256_file(path) != digest
            or hashes.get(name) != digest
        ):
            raise ValueError(f"{label} input artifact differs: {name}")
        parsed[name] = {"path": str(path), "sha256": digest}
    return parsed


def clean_source(root: Path, expected_commit: str) -> dict[str, str]:
    resolved = root.resolve()

    def git_value(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = git_value("rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError("six-joint v5 source commit differs")
    if git_value("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("six-joint v5 execution audit requires a clean worktree")
    return {
        "root": str(resolved),
        "head": head,
        "branch": git_value("branch", "--show-current"),
        "tree": git_value("rev-parse", "HEAD^{tree}"),
    }


def validate_v4_component_readiness(
    report: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    expected = {
        "decision": COMPONENT_PASS,
        "component": "shimmer_db",
        "shimmer_db_six_component_readiness_eligible": True,
        "all_six_scientific_components_ready": True,
        "joint_panel_decision": JOINT_NO_GO,
        "execution_authorized": False,
        "joint_panel_authorized": False,
        "joint_scientific_promotion_granted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    for label, value in (("report", report), ("receipt", receipt)):
        if any(
            value.get(key) != expected_value
            for key, expected_value in expected.items()
        ):
            raise ValueError(f"Candidate-E readiness v4 {label} boundary differs")
    if tuple(report.get("component_order", ())) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("Candidate-E readiness v4 component order differs")
    if tuple(receipt.get("component_order", ())) != ROUTE_C_SIX_ACTIVE_COMPONENTS:
        raise ValueError("Candidate-E readiness v4 receipt order differs")
    if receipt.get("artifact_sha256", {}).get(
        "candidate_e_component_readiness_report_v4.json"
    ) != EXPECTED_V4_REPORT_SHA256:
        raise ValueError("Candidate-E readiness v4 receipt/report binding differs")


def validate_gate_contract(contract: Mapping[str, Any]) -> None:
    frozen = frozen_scientific_contract()
    expected = {
        "schema_version": "avqi-route-c-six-joint-gate-contract-v1",
        "frozen_before_panel_selection": True,
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "global_alpha_grid": frozen["waveform_step"]["alpha_grid"],
        "zero_alpha_selectable": False,
        "selection_split": frozen["waveform_step"]["selection_split"],
        "selection_objective": frozen["waveform_step"]["selection_objective"],
        "selection_tie_break": frozen["waveform_step"]["selection_tie_break"],
        "final_tuning_permitted": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("six-joint gate contract header differs")
    efficacy = contract.get("efficacy")
    slices = contract.get("required_slices")
    safety = contract.get("waveform_safety")
    pathology = contract.get("full_band_pathology")
    denoising = contract.get("denoising")
    if not all(isinstance(value, dict) for value in (
        efficacy, slices, safety, pathology, denoising
    )):
        raise ValueError("six-joint gate contract sections are unavailable")
    frozen_efficacy = frozen["efficacy_gates"]
    expected_efficacy = {
        "scope": "patient degraded rows only",
        "material_normalized_before_gap_strictly_greater_than": (
            frozen_efficacy["material_normalized_before_gap"]["value"]
        ),
        "material_coverage_fraction_minimum": (
            frozen_efficacy["material_coverage_fraction"]["value"]
        ),
        "material_cases_absolute_minimum": (
            frozen_efficacy["material_cases_absolute"]["value"]
        ),
        "material_cases_per_18_minimum": (
            frozen_efficacy["material_cases_per_18"]["value"]
        ),
        "exact_improvement_fraction_minimum": (
            frozen_efficacy["exact_improvement_fraction"]["value"]
        ),
        "median_normalized_gap_reduction_minimum": (
            frozen_efficacy["median_normalized_gap_reduction"]["value"]
        ),
        "applies_to_each_component_and_equal_weight_joint": True,
    }
    if efficacy != expected_efficacy:
        raise ValueError("six-joint efficacy gates differ")
    frozen_slices = frozen["required_efficacy_slices"]
    if slices != {
        "keys": frozen_slices["keys"],
        "expected_rows_per_slice": frozen_slices["expected_rows_per_slice"],
        "improvement_fraction_strictly_greater_than": (
            frozen_slices["improvement_fraction"]["value"]
        ),
        "median_normalized_reduction_minimum": (
            frozen_slices["median_normalized_gap_reduction"]["value"]
        ),
        "zero_coverage_decision": "FAIL",
    }:
        raise ValueError("six-joint slice gates differ")
    frozen_safety = frozen["safety_gates"]
    if safety != {
        "residual_rms_db_maximum": frozen_safety["residual_rms_db"]["value"],
        "cosine_similarity_minimum": frozen_safety["cosine_similarity"]["value"],
        "clip_fraction_maximum": frozen_safety["clip_fraction"]["value"],
    }:
        raise ValueError("six-joint safety gates differ")
    frozen_guardrails = frozen["full_band_pathology_denoising_gates"]
    expected_pathology = {
        "low_frequency_bands_hz": frozen_guardrails["low_frequency_bands_hz"],
        "airflow_proxy_frequency_range_hz": (
            frozen_guardrails["airflow_proxy_frequency_range_hz"]
        ),
        "low_energy_quantile": frozen_guardrails["low_energy_quantile"],
        "pathology_db_median_gap_increase_maximum": (
            frozen_guardrails["pathology_db_median_gap_increase_max"]
        ),
        "pathology_db_worst_gap_increase_maximum": (
            frozen_guardrails["pathology_db_worst_gap_increase_max"]
        ),
        "airflow_flatness_median_gap_increase_maximum": (
            frozen_guardrails["airflow_flatness_median_gap_increase_max"]
        ),
        "airflow_flatness_worst_gap_increase_maximum": (
            frozen_guardrails["airflow_flatness_worst_gap_increase_max"]
        ),
        "pause_f1_median_decrease_maximum": (
            frozen_guardrails["pause_f1_median_decrease_max"]
        ),
        "pause_f1_worst_decrease_maximum": (
            frozen_guardrails["pause_f1_worst_decrease_max"]
        ),
        "guardrail_pass_fraction_minimum": (
            frozen_guardrails["guardrail_pass_fraction_min"]
        ),
        "emitted_waveform_highpass": False,
    }
    if pathology != expected_pathology:
        raise ValueError("six-joint pathology guardrails differ")
    if denoising != {
        "metrics": frozen_guardrails["denoising_metrics"],
        "median_change_minimum_db": frozen_guardrails["denoising_median_change_min_db"],
        "worst_change_minimum_db": frozen_guardrails["denoising_worst_change_min_db"],
    }:
        raise ValueError("six-joint denoising gates differ")


def validate_target_protocol(contract: Mapping[str, Any]) -> None:
    expected_header = {
        "schema_version": "avqi-route-c-six-joint-target-protocol-v1",
        "frozen_before_panel_selection": True,
        "source_dataset": "SVD",
        "patient_target": (
            "exact same-speaker same-view clean pathological CS/SV target"
        ),
        "patient_degraded_target_enabled": True,
        "patient_clean_role": "no-overprocessing control",
        "healthy_target": None,
        "healthy_loss_enabled": False,
        "healthy_waveform_step_enabled": False,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    if any(contract.get(key) != value for key, value in expected_header.items()):
        raise ValueError("six-joint target protocol header differs")
    waveform = contract.get("target_waveform")
    components = contract.get("target_components")
    if waveform != {
        "source": "selected same-session raw SVD CS or SV",
        "resample_hz": 16000,
        "channel": "mono first channel after mono eligibility check",
        "simulation_applied": False,
        "generator_applied": False,
        "emitted_highpass_applied": False,
    }:
        raise ValueError("six-joint target waveform protocol differs")
    if (
        not isinstance(components, dict)
        or components.get("authority") != "exact Praat AVQI implementation"
        or tuple(components.get("component_order", ()))
        != ROUTE_C_SIX_ACTIVE_COMPONENTS
        or components.get("sealed_before_candidate_generation") is not True
        or components.get("candidate_exact_outcomes_opened") is not False
        or components.get("candidate_exact_outcomes_used_for_target_construction")
        is not False
    ):
        raise ValueError("six-joint target component protocol differs")
    forbidden = set(contract.get("selection_forbidden_inputs", ()))
    if not {
        "diagnosis",
        "clinical_severity",
        "mild_severe",
        "exact_avqi",
        "exact_component_values",
        "surrogate_component_values",
        "candidate_exact_outcomes",
    } <= forbidden:
        raise ValueError("six-joint target selection exclusions differ")
    allowed_runtime = set(contract.get("runtime_selector_may_use", ()))
    forbidden_runtime = set(contract.get("runtime_selector_may_not_use", ()))
    if allowed_runtime != {
        "proxy_values",
        "waveform_certificates",
        "topology_certificates",
        "safety_certificates",
        "pcm_certificates",
    } or forbidden_runtime != {
        "candidate_exact_outcomes",
        "speaker_identity",
        "case_identity",
    }:
        raise ValueError("Candidate-E runtime selection boundary differs")


def validate_raw_source_evidence(
    raw_report: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, str]], dict[str, str]]:
    evidence = raw_report.get("source_evidence")
    hashes = raw_report.get("source_evidence_sha256")
    if not isinstance(evidence, dict) or not isinstance(hashes, dict):
        raise ValueError("six-gradient raw source evidence is unavailable")
    if (
        set(evidence) != set(RAW_SOURCE_EVIDENCE_KEYS)
        or set(hashes) != set(RAW_SOURCE_EVIDENCE_KEYS)
    ):
        raise ValueError("six-gradient raw source evidence keys differ")
    parsed: dict[str, Mapping[str, str]] = {}
    for name, raw_binding in evidence.items():
        if not isinstance(raw_binding, dict) or set(raw_binding) != {"path", "sha256"}:
            raise ValueError(f"six-gradient source binding differs: {name}")
        path = Path(str(raw_binding["path"]))
        digest = str(raw_binding["sha256"])
        if not path.is_absolute() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"six-gradient source artifact differs: {name}")
        if hashes.get(name) != digest:
            raise ValueError(f"six-gradient source hash index differs: {name}")
        parsed[name] = {"path": str(path), "sha256": digest}
    return parsed, {key: str(value) for key, value in hashes.items()}


def validate_runtime_binding(
    runtime: Mapping[str, Any],
    *,
    bindings: Mapping[str, Mapping[str, str]],
    support: Mapping[str, Mapping[str, str]],
    source: Mapping[str, str],
    exact_authority: Mapping[str, str],
) -> None:
    if (
        runtime.get("schema_version") != RUNTIME_BINDING_SCHEMA
        or runtime.get("decision") != "BOUND_CANDIDATE_E_JOINT_RUNTIME_V1"
        or runtime.get("promotion_decision") != PROMOTION_PASS
        or runtime.get("source") != source
        or runtime.get("candidate_e_source", {}).get("head")
        != EXPECTED_CANDIDATE_E_COMMIT
        or runtime.get("exact_authority") != exact_authority
        or runtime.get("topology_case_count") != 48
        or runtime.get("topology_role") != "base_current_output"
        or runtime.get("topology_candidate_exact_outcomes_opened") is not False
        or runtime.get("candidate_exact_outcomes_used_for_runtime_selection")
        is not False
        or runtime.get("speaker_or_case_identity_used_for_runtime_selection")
        is not False
        or runtime.get("metric_highpass_only") is not True
        or runtime.get("emitted_waveform_highpass") is not False
        or runtime.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("Candidate-E joint runtime binding differs")
    _require_optimizer_zero(runtime, "Candidate-E joint runtime binding")
    expected_inputs = {
        name: support[name]["sha256"]
        for name in (
            "candidate_e_promotion_report",
            "candidate_e_promotion_receipt",
            "candidate_e_reference_source",
            "candidate_e_runtime_client",
            "candidate_e_worker",
            "candidate_e_selector_source",
            "candidate_e_runtime_config",
            "exact_code_tree_manifest",
            "exact_runtime_manifest",
        )
    }
    if any(
        runtime.get("input_sha256", {}).get(key) != value
        for key, value in expected_inputs.items()
    ):
        raise ValueError("Candidate-E joint runtime input binding differs")
    expected_hashes = {
        "candidate_e_reference_source": CANDIDATE_E_REFERENCE_SHA256,
        "candidate_e_runtime_client": EXPECTED_CANDIDATE_E_RUNTIME_CLIENT_SHA256,
        "candidate_e_worker": EXPECTED_CANDIDATE_E_WORKER_SHA256,
        "candidate_e_selector_source": EXPECTED_CANDIDATE_E_SELECTOR_SHA256,
        "candidate_e_runtime_config": EXPECTED_CANDIDATE_E_RUNTIME_CONFIG_SHA256,
    }
    if any(
        support[key]["sha256"] != value
        for key, value in expected_hashes.items()
    ):
        raise ValueError("Candidate-E promoted runtime artifact differs")
    project_root = Path(__file__).resolve().parents[1]
    expected_integrated = {
        "avqi_route_c_candidate_e.py": sha256_file(
            project_root / "model" / "avqi_route_c_candidate_e.py"
        ),
        "avqi_route_c_candidate_e_scorer.py": sha256_file(
            project_root / "model" / "avqi_route_c_candidate_e_scorer.py"
        ),
    }
    if runtime.get("integrated_implementation_sha256") != expected_integrated:
        raise ValueError("Candidate-E integrated implementation binding differs")
    topology_hashes = runtime.get("topology_artifact_sha256")
    topology_root = (
        Path(bindings["candidate_e_joint_runtime_binding"]["path"]).parent
        / "topologies"
    )
    if (
        not isinstance(topology_hashes, dict)
        or len(topology_hashes) != 48
        or any(
            not (topology_root / name).is_file()
            or sha256_file(topology_root / name) != digest
            for name, digest in topology_hashes.items()
        )
    ):
        raise ValueError("Candidate-E joint topology artifact binding differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-input",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument(
        "--supporting-artifact",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite package: {args.output_dir}")
    inputs = parse_bindings(
        args.execution_input, EXECUTION_INPUT_NAMES, "execution input"
    )
    support = parse_bindings(
        args.supporting_artifact,
        SUPPORTING_ARTIFACT_NAMES,
        "supporting artifact",
    )
    source = clean_source(args.source_root, args.source_commit)

    if support["readiness_v4_report"]["sha256"] != EXPECTED_V4_REPORT_SHA256:
        raise ValueError("Candidate-E readiness v4 report hash differs")
    if support["readiness_v4_receipt"]["sha256"] != EXPECTED_V4_RECEIPT_SHA256:
        raise ValueError("Candidate-E readiness v4 receipt hash differs")
    validate_v4_component_readiness(
        read_json(Path(support["readiness_v4_report"]["path"]), "v4 report"),
        read_json(Path(support["readiness_v4_receipt"]["path"]), "v4 receipt"),
    )

    if (
        support["candidate_e_promotion_report"]["sha256"]
        != PROMOTION_REPORT_SHA256
    ):
        raise ValueError("Candidate-E promotion report hash differs")
    if (
        support["candidate_e_promotion_receipt"]["sha256"]
        != PROMOTION_RECEIPT_SHA256
    ):
        raise ValueError("Candidate-E promotion receipt hash differs")
    if (
        support["candidate_e_prior_panel_speaker_ledger"]["sha256"]
        != UPDATED_LEDGER_SHA256
    ):
        raise ValueError("Candidate-E prior ledger hash differs")
    prior_candidate_e = read_json(
        Path(support["candidate_e_prior_panel_speaker_ledger"]["path"]),
        "Candidate-E prior ledger",
    )
    validate_ledger(prior_candidate_e)
    promotion_report = read_json(
        Path(support["candidate_e_promotion_report"]["path"]),
        "Candidate-E promotion report",
    )
    promotion_receipt = read_json(
        Path(support["candidate_e_promotion_receipt"]["path"]),
        "Candidate-E promotion receipt",
    )
    validate_promotion(
        promotion_report,
        promotion_receipt,
        report_sha256=PROMOTION_REPORT_SHA256,
        ledger_sha256=UPDATED_LEDGER_SHA256,
    )
    validate_receipt_artifact_files(
        Path(support["candidate_e_promotion_report"]["path"]).parent,
        promotion_receipt["artifact_sha256"],
    )
    integrated_candidate_e_path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "avqi_route_c_candidate_e.py"
    )
    candidate_e_parity = candidate_e_function_parity(
        Path(support["candidate_e_reference_source"]["path"]),
        integrated_candidate_e_path,
    )

    gate = read_json(Path(support["joint_gate_contract"]["path"]), "gate contract")
    target_protocol = read_json(
        Path(support["target_value_protocol_contract"]["path"]),
        "target protocol",
    )
    validate_gate_contract(gate)
    validate_target_protocol(target_protocol)

    merged_ledger = read_json(
        Path(support["prior_panel_speaker_ledger"]["path"]),
        "six-joint prior ledger",
    )
    _validate_prior_panel_ledger_merge(
        merged_ledger,
        prior_candidate_e,
        shimmer_ledger_sha256=UPDATED_LEDGER_SHA256,
    )
    split = read_json(
        Path(inputs["fresh_panel_split_seal"]["path"]), "fresh split"
    )
    panel_rows, speakers_by_split = _validate_panel_rows(
        split.get("rows"), "six-joint v5"
    )
    source_manifest = read_json(
        Path(inputs["fresh_speaker_source_manifest"]["path"]),
        "fresh source manifest",
    )
    validate_source_manifest(
        source_manifest,
        panel_rows,
        prior_ledger=merged_ledger,
        prior_ledger_sha256=support[
            "prior_panel_speaker_ledger"
        ]["sha256"],
        source_prior_ledger_sha256=support[
            "candidate_e_prior_panel_speaker_ledger"
        ]["sha256"],
    )
    recipe_manifest = read_json(
        Path(support["joint_recipe_assignment_manifest"]["path"]),
        "joint recipe manifest",
    )
    if recipe_manifest.get("schema_version") != RECIPE_MANIFEST_SCHEMA:
        raise ValueError("joint recipe manifest schema differs")
    fixed_recipes = read_fixed_recipes(
        Path(support["fixed_recipes"]["path"])
    )
    validate_recipe_manifest(
        recipe_manifest,
        panel_rows,
        fixed_recipes,
        support["fixed_recipes"]["sha256"],
    )
    _validate_split_seal(
        split,
        gate_sha256=support["joint_gate_contract"]["sha256"],
        target_sha256=support["target_value_protocol_contract"]["sha256"],
        ledger_sha256=support["prior_panel_speaker_ledger"]["sha256"],
        source_sha256=inputs["fresh_speaker_source_manifest"]["sha256"],
    )
    if split.get("joint_recipe_assignment_manifest_sha256") != support[
        "joint_recipe_assignment_manifest"
    ]["sha256"]:
        raise ValueError("fresh split does not bind recipe assignment")

    selection_receipt = read_json(
        Path(support["selection_receipt"]["path"]), "selection receipt"
    )
    expected_selection_outputs = {
        "prior_panel_speaker_ledger": support[
            "prior_panel_speaker_ledger"
        ]["sha256"],
        "fresh_speaker_source_manifest": inputs[
            "fresh_speaker_source_manifest"
        ]["sha256"],
        "joint_recipe_assignment_manifest": support[
            "joint_recipe_assignment_manifest"
        ]["sha256"],
        "fresh_panel_split_seal": inputs["fresh_panel_split_seal"]["sha256"],
    }
    if (
        selection_receipt.get("schema_version") != SELECTION_RECEIPT_SCHEMA
        or selection_receipt.get("decision") != SELECTION_DECISION
        or selection_receipt.get("source") != source
        or selection_receipt.get("artifact_sha256") != expected_selection_outputs
        or selection_receipt.get("speaker_count") != 12
        or selection_receipt.get("row_count") != 96
        or selection_receipt.get("prior_panel_speaker_overlap") != 0
        or selection_receipt.get("metadata_only_result_blind_selection") is not True
        or selection_receipt.get("exact_scores_opened") is not False
        or selection_receipt.get("candidate_outcomes_opened") is not False
        or selection_receipt.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("six-joint selection receipt differs")
    _require_optimizer_zero(selection_receipt, "six-joint selection receipt")
    selection_inputs = validate_receipt_input_bindings(
        selection_receipt,
        {
            "sv_metadata",
            "cs_metadata",
            "prior_panel_speaker_ledger",
            "fixed_recipes",
            "gap_simulation_inventory",
            "gap_rir_manifest",
            "gap_noise_manifest",
            "joint_gate_contract",
            "target_value_protocol",
        },
        "six-joint selection",
    )
    expected_selection_inputs = {
        "sv_metadata": split["sv_metadata_sha256"],
        "cs_metadata": split["cs_metadata_sha256"],
        "prior_panel_speaker_ledger": support[
            "candidate_e_prior_panel_speaker_ledger"
        ]["sha256"],
        "fixed_recipes": support["fixed_recipes"]["sha256"],
        "gap_simulation_inventory": split[
            "gap_simulation_inventory_sha256"
        ],
        "gap_rir_manifest": split["gap_rir_manifest_sha256"],
        "gap_noise_manifest": split["gap_noise_manifest_sha256"],
        "joint_gate_contract": support["joint_gate_contract"]["sha256"],
        "target_value_protocol": support[
            "target_value_protocol_contract"
        ]["sha256"],
    }
    if any(
        selection_inputs[name]["sha256"] != digest
        for name, digest in expected_selection_inputs.items()
    ):
        raise ValueError("six-joint selection input hash chain differs")
    selection_runtime = selection_receipt.get("runtime_binding")
    if (
        not isinstance(selection_runtime, dict)
        or set(selection_runtime) != {"sv_root", "cs_root"}
        or any(
            not Path(str(selection_runtime[key])).is_absolute()
            or not Path(str(selection_runtime[key])).is_dir()
            for key in ("sv_root", "cs_root")
        )
    ):
        raise ValueError("six-joint selection source roots differ")
    reproduced_eligible = eligible_speakers(
        read_csv(Path(selection_inputs["sv_metadata"]["path"])),
        read_csv(Path(selection_inputs["cs_metadata"]["path"])),
        Path(str(selection_runtime["sv_root"])),
        Path(str(selection_runtime["cs_root"])),
        prior_speakers(prior_candidate_e),
    )
    reproduced_selected = select_speakers(reproduced_eligible)
    reproduced_source = build_source_manifest(
        reproduced_selected,
        prior_ledger_sha256=support[
            "prior_panel_speaker_ledger"
        ]["sha256"],
        source_prior_ledger_sha256=support[
            "candidate_e_prior_panel_speaker_ledger"
        ]["sha256"],
    )
    if source_manifest != reproduced_source:
        raise ValueError("fresh source manifest is not reproducible")
    if selection_receipt.get("eligible_speaker_count") != len(
        reproduced_eligible
    ):
        raise ValueError("six-joint eligible speaker count differs")
    reproduced_recipe = build_recipe_manifest(
        list(panel_rows.values()),
        fixed_recipes,
        fixed_recipes_sha256=support["fixed_recipes"]["sha256"],
    )
    if recipe_manifest != reproduced_recipe:
        raise ValueError("joint recipe manifest is not reproducible")

    six_report = read_json(
        Path(inputs["six_gradient_report"]["path"]), "six-gradient report"
    )
    six_receipt = read_json(
        Path(inputs["six_gradient_receipt"]["path"]), "six-gradient receipt"
    )
    raw_report = read_json(
        Path(support["six_gradient_raw_report"]["path"]),
        "six-gradient raw report",
    )
    raw_receipt = read_json(
        Path(support["six_gradient_raw_receipt"]["path"]),
        "six-gradient raw receipt",
    )
    raw_evidence, raw_evidence_hashes = validate_raw_source_evidence(raw_report)
    five_paths = {
        key: Path(raw_evidence[key]["path"])
        for key in (
            *FIVE_COMPONENT_EVIDENCE_KEYS,
            "five_gradient_report",
            "five_gradient_receipt",
        )
    }
    five_bindings = {
        key: dict(raw_evidence[key])
        for key in (
            *FIVE_COMPONENT_EVIDENCE_KEYS,
            "five_gradient_report",
            "five_gradient_receipt",
        )
    }
    _validate_five_component_evidence(five_bindings, five_paths)
    readiness_evidence = {
        key: raw_evidence_hashes[key] for key in READINESS_SOURCE_EVIDENCE_KEYS
    }
    weights = _validate_six_gradient(
        six_report,
        six_receipt,
        inputs["six_gradient_report"]["sha256"],
        readiness_evidence,
    )
    raw_binding = six_report.get("raw_measurement_evidence", {})
    if (
        raw_binding.get("report_sha256")
        != support["six_gradient_raw_report"]["sha256"]
        or raw_binding.get("receipt_sha256")
        != support["six_gradient_raw_receipt"]["sha256"]
        or raw_receipt.get("artifact_sha256", {}).get(
            Path(support["six_gradient_raw_report"]["path"]).name
        ) != support["six_gradient_raw_report"]["sha256"]
    ):
        raise ValueError("six-gradient raw evidence binding differs")
    if (
        raw_evidence_hashes.get("candidate_e_reference_source")
        != CANDIDATE_E_REFERENCE_SHA256
    ):
        raise ValueError("six-gradient Candidate-E reference differs")
    for key in CANDIDATE_E_EVIDENCE_KEYS:
        if raw_evidence[key]["sha256"] != support[key]["sha256"]:
            raise ValueError(f"six-gradient Candidate-E evidence differs: {key}")
    if raw_evidence["topology_receipt"]["sha256"] != support[
        "topology_receipt"
    ]["sha256"]:
        raise ValueError("six-gradient topology receipt binding differs")
    topology_receipt = read_json(
        Path(support["topology_receipt"]["path"]),
        "six-gradient topology receipt",
    )
    if (
        topology_receipt.get("schema_version")
        != TOPOLOGY_RECEIPT_SCHEMA_VERSION
        or topology_receipt.get("decision") != TOPOLOGY_SEAL_DECISION
        or topology_receipt.get("topology_count") != 8
        or topology_receipt.get("candidate_exact_outcomes_opened") is not False
        or topology_receipt.get("generator_optimizer_steps") != 0
        or topology_receipt.get("artifact_sha256", {}).get(
            Path(raw_evidence["topology_manifest"]["path"]).name
        )
        != raw_evidence["topology_manifest"]["sha256"]
    ):
        raise ValueError("six-gradient topology receipt differs")

    exact_authority = validate_exact_authority(
        exact_python=args.exact_python,
        avqi_code_root=args.avqi_code_root,
        code_manifest=read_json(
            Path(support["exact_code_tree_manifest"]["path"]),
            "exact code manifest",
        ),
        code_manifest_sha256=support["exact_code_tree_manifest"]["sha256"],
        runtime_manifest=read_json(
            Path(support["exact_runtime_manifest"]["path"]),
            "exact runtime manifest",
        ),
    )
    exact_receipt = read_json(
        Path(support["exact_authority_receipt"]["path"]),
        "exact authority receipt",
    )
    if (
        exact_receipt.get("schema_version") != EXACT_AUTHORITY_RECEIPT_SCHEMA
        or exact_receipt.get("decision") != EXACT_AUTHORITY_SEAL_DECISION
        or exact_receipt.get("generator_optimizer_steps") != 0
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_avqi_code_tree_manifest.json"
        )
        != support["exact_code_tree_manifest"]["sha256"]
        or exact_receipt.get("artifact_sha256", {}).get(
            "exact_runtime_manifest.json"
        )
        != support["exact_runtime_manifest"]["sha256"]
    ):
        raise ValueError("exact authority receipt binding differs")

    target_bank = read_json(
        Path(inputs["clean_target_label_bank"]["path"]), "clean target bank"
    )
    target_rows = validate_target_bank(
        target_bank,
        bank_sha256=inputs["clean_target_label_bank"]["sha256"],
        split_seal_sha256=inputs["fresh_panel_split_seal"]["sha256"],
        source_manifest_sha256=inputs[
            "fresh_speaker_source_manifest"
        ]["sha256"],
        target_protocol_sha256=support[
            "target_value_protocol_contract"
        ]["sha256"],
        panel_rows=panel_rows,
    )
    if (
        target_bank.get("exact_code_tree_manifest_sha256")
        != support["exact_code_tree_manifest"]["sha256"]
        or target_bank.get("exact_runtime_manifest_sha256")
        != support["exact_runtime_manifest"]["sha256"]
    ):
        raise ValueError("clean target bank exact-authority binding differs")
    target_exact_items = [
        {
            "id": f"target:{target['split']}:{speaker_id}:{view}",
            "path": str(target["target_waveform_path"]),
            "view": view,
        }
        for (speaker_id, view), target in sorted(
            target_rows.items(),
            key=lambda item: (
                str(item[1]["split"]),
                item[0][0],
                item[0][1],
            ),
        )
    ]
    reproduced_target_exact = run_exact(
        target_exact_items,
        exact_python=args.exact_python.resolve(),
        avqi_code_root=args.avqi_code_root.resolve(),
        expected_runtime=exact_authority,
    )
    for (speaker_id, view), target in target_rows.items():
        item_id = f"target:{target['split']}:{speaker_id}:{view}"
        expected_values = np.asarray(
            [
                target["exact_components"][component]
                for component in ROUTE_C_SIX_ACTIVE_COMPONENTS
            ],
            dtype=np.float64,
        )
        if not np.array_equal(reproduced_target_exact[item_id], expected_values):
            raise ValueError("clean target exact-Praat values are not reproducible")
    normalization = raw_report.get("contract", {}).get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("six-gradient normalization is unavailable")
    gradient_manifest = read_json(
        Path(inputs["joint_gradient_manifest"]["path"]),
        "joint gradient manifest",
    )
    gradient_rows = validate_gradient_manifest(
        gradient_manifest,
        split_seal_sha256=inputs["fresh_panel_split_seal"]["sha256"],
        target_bank_sha256=inputs["clean_target_label_bank"]["sha256"],
        six_gradient_report_sha256=inputs["six_gradient_report"]["sha256"],
        six_gradient_receipt_sha256=inputs["six_gradient_receipt"]["sha256"],
        six_gradient_raw_report_sha256=support["six_gradient_raw_report"]["sha256"],
        weights=weights,
        normalization=normalization,
        panel_rows=panel_rows,
    )

    materialization = read_json(
        Path(support["materialization_receipt"]["path"]),
        "materialization receipt",
    )
    materialization_inputs = validate_receipt_input_bindings(
        materialization,
        set(MATERIALIZATION_INPUT_NAMES),
        "six-joint materialization",
    )
    expected_materialization_inputs = {
        "split_seal": inputs["fresh_panel_split_seal"]["sha256"],
        "fresh_speaker_source_manifest": inputs[
            "fresh_speaker_source_manifest"
        ]["sha256"],
        "prior_panel_speaker_ledger": support[
            "prior_panel_speaker_ledger"
        ]["sha256"],
        "joint_recipe_assignment_manifest": support[
            "joint_recipe_assignment_manifest"
        ]["sha256"],
        "fixed_recipes": support["fixed_recipes"]["sha256"],
        "simulation_config": EXPECTED_SIMULATION_CONFIG_SHA256,
        "simulation_source": EXPECTED_SIMULATION_SOURCE_SHA256,
        "generator_config": EXPECTED_GENERATOR_CONFIG_SHA256,
        "generator_checkpoint": EXPECTED_GENERATOR_CHECKPOINT_SHA256,
        "six_gradient_raw_report": support[
            "six_gradient_raw_report"
        ]["sha256"],
        "six_gradient_report": inputs["six_gradient_report"]["sha256"],
        "six_gradient_receipt": inputs["six_gradient_receipt"]["sha256"],
        **{
            key: raw_evidence[key]["sha256"]
            for key in CANDIDATE_E_EVIDENCE_KEYS
        },
        "topology_receipt": raw_evidence["topology_receipt"]["sha256"],
        "candidate_e_promotion_report": support[
            "candidate_e_promotion_report"
        ]["sha256"],
        "candidate_e_promotion_receipt": support[
            "candidate_e_promotion_receipt"
        ]["sha256"],
        "candidate_e_reference_source": support[
            "candidate_e_reference_source"
        ]["sha256"],
        "candidate_e_runtime_client": support[
            "candidate_e_runtime_client"
        ]["sha256"],
        "candidate_e_worker": support["candidate_e_worker"]["sha256"],
        "candidate_e_selector_source": support[
            "candidate_e_selector_source"
        ]["sha256"],
        "candidate_e_runtime_config": support[
            "candidate_e_runtime_config"
        ]["sha256"],
        "exact_code_tree_manifest": support[
            "exact_code_tree_manifest"
        ]["sha256"],
        "exact_runtime_manifest": support[
            "exact_runtime_manifest"
        ]["sha256"],
        "cpps_checkpoint": raw_evidence["cpps_checkpoint"]["sha256"],
        "hnr_checkpoint": raw_evidence["hnr_checkpoint"]["sha256"],
        "shimmer_checkpoint": raw_evidence[
            "shimmer_percent_checkpoint"
        ]["sha256"],
        "slope_checkpoint": raw_evidence["slope_checkpoint"]["sha256"],
        "tilt_checkpoint": raw_evidence["tilt_checkpoint"]["sha256"],
    }
    if any(
        materialization_inputs[name]["sha256"] != digest
        for name, digest in expected_materialization_inputs.items()
    ):
        raise ValueError("six-joint materialization input hash chain differs")
    materialization_runtime = materialization.get("runtime_binding")
    if (
        not isinstance(materialization_runtime, dict)
        or materialization_runtime.get("exact_python")
        != str(args.exact_python.resolve())
        or materialization_runtime.get("avqi_code_root")
        != str(args.avqi_code_root.resolve())
        or materialization_runtime.get("device") != "cuda"
        or materialization_runtime.get("seed") != 20260903
        or not Path(
            str(materialization_runtime.get("simulation_root", ""))
        ).is_dir()
        or materialization_runtime.get("candidate_e_source", {}).get("head")
        != EXPECTED_CANDIDATE_E_COMMIT
    ):
        raise ValueError("six-joint materialization runtime binding differs")
    expected_materialized = {
        "clean_target_label_bank.json": inputs["clean_target_label_bank"]["sha256"],
        "joint_gradient_manifest.json": inputs["joint_gradient_manifest"]["sha256"],
        "candidate_e_joint_runtime_binding.json": inputs[
            "candidate_e_joint_runtime_binding"
        ]["sha256"],
    }
    if (
        materialization.get("schema_version") != MATERIALIZATION_RECEIPT_SCHEMA
        or materialization.get("decision") != MATERIALIZATION_DECISION
        or materialization.get("source") != source
        or materialization.get("artifact_sha256") != expected_materialized
        or materialization.get("row_count") != 96
        or materialization.get("patient_gradient_count") != 48
        or materialization.get("target_exact_count") != 12
        or materialization.get("topology_count") != 48
        or materialization.get("candidate_exact_outcomes_opened") is not False
        or materialization.get("candidate_waveforms_generated") is not False
        or materialization.get("generator_optimizer_created") is not False
        or materialization.get("authoritative_training_decision") != TRAINING_NO_GO
    ):
        raise ValueError("six-joint materialization receipt differs")
    _require_optimizer_zero(materialization, "six-joint materialization")
    runtime_binding = read_json(
        Path(inputs["candidate_e_joint_runtime_binding"]["path"]),
        "Candidate-E joint runtime binding",
    )
    validate_runtime_binding(
        runtime_binding,
        bindings=inputs,
        support=support,
        source=source,
        exact_authority=exact_authority,
    )

    input_sha256 = {
        "candidate_e_joint_runtime_binding": inputs[
            "candidate_e_joint_runtime_binding"
        ]["sha256"],
        "six_gradient_report": inputs["six_gradient_report"]["sha256"],
        "six_gradient_receipt": inputs["six_gradient_receipt"]["sha256"],
        "fresh_panel_split_seal": inputs["fresh_panel_split_seal"]["sha256"],
        "fresh_speaker_source_manifest": inputs[
            "fresh_speaker_source_manifest"
        ]["sha256"],
        "clean_target_label_bank": inputs["clean_target_label_bank"]["sha256"],
        "joint_gradient_manifest": inputs["joint_gradient_manifest"]["sha256"],
        "six_gradient_raw_report": support["six_gradient_raw_report"]["sha256"],
        "joint_gate_contract": support["joint_gate_contract"]["sha256"],
        "target_value_protocol_contract": support[
            "target_value_protocol_contract"
        ]["sha256"],
        "prior_panel_speaker_ledger": support[
            "prior_panel_speaker_ledger"
        ]["sha256"],
    }
    package = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "package_schema_version": PACKAGE_SCHEMA,
        "decision": READINESS_PASS_DECISION,
        "scientific_contract_schema_version": (
            FROZEN_SCIENTIFIC_CONTRACT_SCHEMA_VERSION
        ),
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "source_tree": source["tree"],
        "component_order": list(ROUTE_C_SIX_ACTIVE_COMPONENTS),
        "all_six_scientific_components_ready": True,
        "candidate_e_component_readiness": COMPONENT_PASS,
        "candidate_e_external_promotion": PROMOTION_PASS,
        "candidate_e_reference_function_parity": candidate_e_parity,
        "six_gradient_decision": SIX_GRADIENT_PASS_DECISION,
        "bound_execution_inputs": list(EXECUTION_INPUT_NAMES),
        "execution_input_binding": inputs,
        "supporting_artifact_binding": support,
        "six_gradient_source_evidence": raw_evidence,
        "input_sha256": input_sha256,
        "speaker_count": 12,
        "row_count": 96,
        "patient_target_count": len(target_rows),
        "patient_gradient_count": sum(
            row["joint_gradient_path"] is not None for row in gradient_rows.values()
        ),
        "speakers_by_split": speakers_by_split,
        "metadata_only_result_blind_selection": True,
        "same_speaker_same_view_clean_pathological_targets": True,
        "exact_target_values_sealed_before_candidate_generation": True,
        "exact_target_values_revalidated": True,
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "actual_manifests_bound": True,
        "execution_authorized": True,
        "joint_panel_authorized": True,
        "joint_scientific_promotion_granted": False,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_submitted": False,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
    }
    args.output_dir.mkdir(parents=True)
    package_path = args.output_dir / "six_joint_execution_package_v5.json"
    write_json(package_path, package)
    package_sha256 = sha256_file(package_path)
    receipt = {
        "schema_version": READINESS_RECEIPT_SCHEMA_VERSION,
        "package_receipt_schema_version": PACKAGE_RECEIPT_SCHEMA,
        "decision": READINESS_PASS_DECISION,
        "source_commit": source["head"],
        "source_branch": source["branch"],
        "execution_authorized": True,
        "joint_panel_authorized": True,
        "joint_scientific_promotion_granted": False,
        "candidate_exact_outcomes_opened": False,
        "fresh_panel_opened": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": TRAINING_NO_GO,
        "artifact_sha256": {package_path.name: package_sha256},
    }
    receipt_path = args.output_dir / "completion_receipt_v5.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "decision": READINESS_PASS_DECISION,
                "execution_authorized": True,
                "joint_panel_authorized": True,
                "package_sha256": package_sha256,
                "completion_receipt_sha256": sha256_file(receipt_path),
                "generator_optimizer_steps": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
