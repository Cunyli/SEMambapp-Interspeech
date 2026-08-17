#!/usr/bin/env python3
"""Summarize three locked AVQI-component predictor confirmations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPONENTS = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)
COMPONENT_FAMILIES = {
    "cpps": "periodicity_noise",
    "hnr": "periodicity_noise",
    "shimmer_percent": "amplitude_modulation",
    "shimmer_db": "amplitude_modulation",
    "slope": "spectral_shape",
    "tilt": "spectral_shape",
}
ROUTES = ("shared_dual_head", "frozen_independent_predictor")
DIRECT_ROUTE = "direct_differentiable_estimator"
DIRECT_ONLY_ROUTES = (DIRECT_ROUTE,)
EXPECTED_CONFIRMATION_REPORTS = 3
CONSENSUS_PASS_COUNT = 2
EXPECTED_SCREEN_FORMS = {
    "shared_dual_head": ["late_global", "late_frequency", "late_tfgrid"],
    "frozen_independent_predictor": [
        "global_stats",
        "frequency_aware",
        "compact_tfgrid",
    ],
}
EXPECTED_DIRECT_V2_SCREEN_FORMS = {
    "shared_dual_head": ["late_tfgrid"],
    "frozen_independent_predictor": [
        "direct_praat_soft_v2",
        "direct_praat_hard_v2",
    ],
}
EXPECTED_PHASE_V4_SCREEN_FORMS = {
    "shared_dual_head": ["output_phase_tfgrid"],
    "frozen_independent_predictor": [
        "frequency_aware",
        "phase_frequency_aware",
        "phase_compact_tfgrid",
    ],
}
EXPECTED_DIRECT_V4_SCREEN_FORMS = {
    "shared_dual_head": ["output_phase_tfgrid"],
    "frozen_independent_predictor": ["direct_praat_hard_v2"],
}
EXPECTED_FULL_V4_SCREEN_FORMS = {
    "shared_dual_head": ["output_phase_tfgrid"],
    "frozen_independent_predictor": ["pretrained_full_tfgrid"],
}
EXPECTED_DIRECT_C_SCREEN_FORMS = (
    {DIRECT_ROUTE: ["direct_praat_hard_v2"]},
    {
        DIRECT_ROUTE: [
            "direct_praat_hard_v2",
            "direct_praat_hard_shimmer_rms_v3",
        ]
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument(
        "--confirmation-report",
        type=Path,
        action="append",
        required=True,
        dest="confirmation_reports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_form(report: dict[str, Any], route: str) -> str:
    route_report = report["routes"][route]
    if route == "shared_dual_head":
        return str(route_report["selected_candidate"])
    return str(route_report["selected_architecture"])


def active_routes(report: dict[str, Any]) -> tuple[str, ...]:
    route_scope = report["contract"].get("route_scope", "all")
    if route_scope == "direct_only":
        return DIRECT_ONLY_ROUTES
    if route_scope == "all":
        return ROUTES
    raise ValueError(f"unknown route scope: {route_scope}")


def validate_report_shape(report: dict[str, Any], path: Path) -> None:
    route_scope = report.get("contract", {}).get("route_scope", "all")
    expected_decision = (
        "COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
        if route_scope == "direct_only"
        else "COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE"
    )
    if report.get("decision") != expected_decision:
        raise ValueError(f"report is not a completed predictor screen: {path}")
    if report.get("generator_optimizer_steps") != 0:
        raise ValueError(f"generator update found in predictor report: {path}")
    if report.get("formal_pathology_training_submitted") is not False:
        raise ValueError(f"formal training state is ambiguous: {path}")
    if tuple(report["contract"]["components"]) != COMPONENTS:
        raise ValueError(f"AVQI component contract differs: {path}")
    for route in active_routes(report):
        if route not in report["routes"]:
            raise ValueError(f"missing route {route}: {path}")
    if route_scope == "direct_only":
        for skipped_route in ROUTES:
            if report["routes"].get(skipped_route, {}).get("status") != (
                "SKIPPED_USER_SCOPE"
            ):
                raise ValueError(
                    f"Route C report did not skip {skipped_route}: {path}"
                )


def validate_screen_contract(screen: dict[str, Any], path: Path) -> None:
    contract = screen["contract"]
    if contract.get("route_scope", "all") == "direct_only":
        for skipped_route in ROUTES:
            if contract["routes"][skipped_route]["status"] != "SKIPPED_USER_SCOPE":
                raise ValueError(
                    f"Route C contract did not skip {skipped_route}: {path}"
                )
        observed_forms = {
            DIRECT_ROUTE: contract["routes"][DIRECT_ROUTE]["architectures"]
        }
        if observed_forms not in EXPECTED_DIRECT_C_SCREEN_FORMS:
            raise ValueError(f"Route C screen forms are incomplete: {path}")
        route_report = screen["routes"][DIRECT_ROUTE]
        if route_report["selection_rule"] != (
            "lowest calibration loss before holdout evaluation"
        ):
            raise ValueError(f"Route C selection rule differs: {path}")
        selected = selected_form(screen, DIRECT_ROUTE)
        training = route_report["training"]
        calibration_winner = min(
            training,
            key=lambda name: training[name]["best_calibration_loss"],
        )
        if selected != calibration_winner:
            raise ValueError(
                f"Route C selection did not follow calibration loss: {path}"
            )
        if any(
            item["optimizer_steps"] != 0
            or item["trainable_parameter_count"] != 0
            for item in training.values()
        ):
            raise ValueError(f"Route C screen contains learned optimization: {path}")
        if contract["direct_formula_budget"] != {
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "maximum_optimizer_steps": 0,
        }:
            raise ValueError(f"Route C formula budget differs: {path}")
        if contract["calibration"]["holdout_used_for_fit_or_selection"] is not False:
            raise ValueError(f"holdout was used during Route C selection: {path}")
        return
    observed_forms = {
        "shared_dual_head": contract["routes"]["shared_dual_head"][
            "candidates"
        ],
        "frozen_independent_predictor": contract["routes"][
            "frozen_independent_predictor"
        ]["architectures"],
    }
    if observed_forms not in (
        EXPECTED_SCREEN_FORMS,
        EXPECTED_DIRECT_V2_SCREEN_FORMS,
        EXPECTED_PHASE_V4_SCREEN_FORMS,
        EXPECTED_DIRECT_V4_SCREEN_FORMS,
        EXPECTED_FULL_V4_SCREEN_FORMS,
    ):
        raise ValueError(f"architecture screen forms are incomplete: {path}")
    for route in ROUTES:
        route_report = screen["routes"][route]
        if route_report["selection_rule"] != (
            "lowest calibration loss before holdout evaluation"
        ):
            raise ValueError(f"selection rule differs for {route}: {path}")
        selected = selected_form(screen, route)
        training = route_report["training"]
        calibration_winner = min(
            training,
            key=lambda name: training[name]["best_calibration_loss"],
        )
        if selected != calibration_winner:
            raise ValueError(
                f"{route} selection did not follow calibration loss: {path}"
            )
    budget = contract["matched_training_budget"]
    if budget["shared_max_epochs"] != budget["independent_max_epochs"]:
        raise ValueError(f"architecture screen budget is not matched: {path}")
    if contract["calibration"]["holdout_used_for_fit_or_selection"] is not False:
        raise ValueError(f"holdout was used during architecture selection: {path}")


def validate_confirmation_set(
    screen: dict[str, Any],
    confirmations: list[dict[str, Any]],
    paths: list[Path],
) -> list[int]:
    if len(confirmations) != EXPECTED_CONFIRMATION_REPORTS:
        raise ValueError(
            f"expected {EXPECTED_CONFIRMATION_REPORTS} confirmation reports, "
            f"found {len(confirmations)}"
        )
    expected_seeds = screen["contract"]["multiseed_confirmation"]["seeds"]
    if len(expected_seeds) != EXPECTED_CONFIRMATION_REPORTS:
        raise ValueError("screen report does not declare exactly three seeds")
    source_hashes = screen["contract"]["source_sha256"]
    source_commit = screen["contract"]["source_commit"]
    locked_contract_keys = [
        "anti_shortcut",
        "gates",
        "matched_external_primary_candidate",
    ]
    if screen["contract"].get("route_scope", "all") == "direct_only":
        locked_contract_keys.append("direct_formula_budget")
    else:
        locked_contract_keys.append("matched_training_budget")
    screen_routes = active_routes(screen)
    observed_seeds = []
    for report, path in zip(confirmations, paths, strict=True):
        validate_report_shape(report, path)
        contract = report["contract"]
        if active_routes(report) != screen_routes:
            raise ValueError(f"route scope differs: {path}")
        if contract["source_sha256"] != source_hashes:
            raise ValueError(f"source hashes differ: {path}")
        if contract["source_commit"] != source_commit:
            raise ValueError(f"source commit differs: {path}")
        for key in locked_contract_keys:
            if contract[key] != screen["contract"][key]:
                raise ValueError(f"locked contract key {key} differs: {path}")
        seed = int(contract["architecture_screen_seed"])
        observed_seeds.append(seed)
        for route in screen_routes:
            if selected_form(report, route) != selected_form(screen, route):
                raise ValueError(f"{route} architecture was not locked: {path}")
        if screen_routes == DIRECT_ONLY_ROUTES:
            if contract["routes"][DIRECT_ROUTE]["architectures"] != [
                selected_form(screen, DIRECT_ROUTE)
            ]:
                raise ValueError(f"Route C estimator was not locked: {path}")
        else:
            if contract["routes"]["shared_dual_head"]["candidates"] != [
                selected_form(screen, "shared_dual_head")
            ]:
                raise ValueError(f"shared candidate list was not locked: {path}")
            if contract["routes"]["frozen_independent_predictor"][
                "architectures"
            ] != [selected_form(screen, "frozen_independent_predictor")]:
                raise ValueError(f"independent architecture was not locked: {path}")
    if sorted(observed_seeds) != sorted(expected_seeds):
        raise ValueError(
            f"confirmation seeds differ: expected {expected_seeds}, "
            f"found {observed_seeds}"
        )
    return observed_seeds


def has_minimum_coverage(components: list[str]) -> bool:
    families = {COMPONENT_FAMILIES[component] for component in components}
    return "periodicity_noise" in families and len(families) >= 2


def route_consensus(
    confirmations: list[dict[str, Any]],
    route: str,
) -> dict[str, Any]:
    pass_counts = {
        component: sum(
            component in report["routes"][route]["eligible_components"]
            for report in confirmations
        )
        for component in COMPONENTS
    }
    components = [
        component
        for component in COMPONENTS
        if pass_counts[component] >= CONSENSUS_PASS_COUNT
    ]
    return {
        "selected_form": selected_form(confirmations[0], route),
        "component_pass_counts": pass_counts,
        "consensus_components": components,
        "minimum_coverage_passed": has_minimum_coverage(components),
        "decision": (
            "RELIABLE"
            if has_minimum_coverage(components)
            else "NOT_RELIABLE"
        ),
    }


def promotion_decision(
    routes: dict[str, dict[str, Any]],
    route_names: tuple[str, ...] = ROUTES,
) -> dict[str, Any]:
    reliable = [
        route for route in route_names if routes[route]["decision"] == "RELIABLE"
    ]
    if not reliable:
        return {
            "decision": "NO_GO_AVQI_BACKPROP",
            "routes": [],
            "components": [],
            "reason": (
                "no in-scope route has stable coverage across two concept families"
            ),
        }
    if route_names == DIRECT_ONLY_ROUTES:
        return {
            "decision": "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT",
            "routes": [DIRECT_ROUTE],
            "components": routes[DIRECT_ROUTE]["consensus_components"],
            "reason": (
                "Route C passed the three-seed scorer gate; this authorizes only "
                "a bounded waveform pilot"
            ),
        }
    if len(reliable) == 1:
        route = reliable[0]
        return {
            "decision": "GO_BOUNDED_SINGLE_ROUTE_BACKPROP",
            "routes": [route],
            "components": routes[route]["consensus_components"],
            "reason": "one route passed the three-seed predictor gate",
        }
    shared = set(routes["shared_dual_head"]["consensus_components"])
    independent = set(
        routes["frozen_independent_predictor"]["consensus_components"]
    )
    common = [component for component in COMPONENTS if component in shared & independent]
    if not has_minimum_coverage(common):
        return {
            "decision": "NO_GO_MATCHED_ROUTE_BACKPROP",
            "routes": reliable,
            "components": common,
            "reason": "both routes passed separately but lack a fair common component set",
        }
    return {
        "decision": "GO_MATCHED_DUAL_ROUTE_BACKPROP",
        "routes": reliable,
        "components": common,
        "reason": "both routes and their common component set passed three seeds",
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# AVQI component predictor multi-seed conclusion",
        "",
        f"**Decision:** `{report['promotion']['decision']}`",
        "",
        "| Route | Locked form | Stable components | Decision |",
        "|---|---|---|---|",
    ]
    for route in report["active_routes"]:
        route_report = report["routes"][route]
        components = ", ".join(route_report["consensus_components"]) or "none"
        lines.append(
            f"| {route} | {route_report['selected_form']} | "
            f"{components} | {route_report['decision']} |"
        )
    promoted = ", ".join(report["promotion"]["components"]) or "none"
    lines.extend(
        [
            "",
            f"Components allowed in the bounded backprop pilot: **{promoted}**.",
            "",
            "This conclusion covers predictor reliability only. It does not prove "
            "that an enhanced waveform improves under exact Praat evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    screen = load_json(args.screen_report)
    validate_report_shape(screen, args.screen_report)
    validate_screen_contract(screen, args.screen_report)
    confirmations = [load_json(path) for path in args.confirmation_reports]
    seeds = validate_confirmation_set(
        screen,
        confirmations,
        args.confirmation_reports,
    )
    route_names = active_routes(screen)
    routes = {
        route: route_consensus(confirmations, route)
        for route in route_names
    }
    report = {
        "schema_version": (
            "avqi-component-multiseed-consensus-v2"
            if route_names == DIRECT_ONLY_ROUTES
            else "avqi-component-multiseed-consensus-v1"
        ),
        "screen_report": str(args.screen_report.resolve()),
        "confirmation_reports": [
            str(path.resolve()) for path in args.confirmation_reports
        ],
        "source_report_sha256": {
            "screen": sha256_file(args.screen_report),
            "confirmations": [
                sha256_file(path) for path in args.confirmation_reports
            ],
        },
        "seeds": seeds,
        "consensus_rule": (
            "component passes its complete gate in at least two of three locked seeds"
        ),
        "route_scope": screen["contract"].get("route_scope", "all"),
        "active_routes": list(route_names),
        "routes": routes,
        "promotion": promotion_decision(routes, route_names),
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "multiseed_consensus.json"
    summary_path = args.output_dir / "SUMMARY.md"
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    receipt = {
        "decision": report["promotion"]["decision"],
        "routes": report["promotion"]["routes"],
        "components": report["promotion"]["components"],
        "generator_optimizer_steps": 0,
        "bounded_waveform_pilot_submitted": False,
        "formal_pathology_training_submitted": False,
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            summary_path.name: sha256_file(summary_path),
        },
        "source_report_sha256": report["source_report_sha256"],
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
