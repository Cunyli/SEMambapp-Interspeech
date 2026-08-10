"""Compare matched Standard and DNF outputs on the stratified TAU safety gate."""

import argparse
import csv
import json
from pathlib import Path


PAIR_FIELDS = (
    "checkpoint_step",
    "loss_variant",
    "train_manifest_sha256",
    "valid_manifest_sha256",
    "canonical_speech_init_sha256",
    "contract_sha256",
    "model_config_sha256",
    "code_surface_sha256",
    "controlled_comparison",
    "controlled_comparison_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired, severity-stratified Phase-A TAU gates."
    )
    parser.add_argument("--standard-dir", type=Path, required=True)
    parser.add_argument("--dnf-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def selected_uids(path: Path) -> set[str]:
    return {row["uid"] for row in read_csv(path)}


def index_signal_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    indexed = {}
    for row in read_csv(path):
        key = (row["uid"], row["output_name"])
        if key in indexed:
            raise ValueError(f"duplicate signal row: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"empty signal guard: {path}")
    return indexed


def select_signal_rows(
    indexed: dict[tuple[str, str], dict[str, str]],
    *,
    output_name: str,
    groups: set[str],
    tasks: set[str],
) -> list[dict[str, str]]:
    rows = [
        row
        for (_, name), row in indexed.items()
        if name == output_name
        and row["sample_group"] in groups
        and row["task"] in tasks
    ]
    if not rows:
        raise ValueError(
            f"missing signal slice output={output_name}, "
            f"groups={groups}, tasks={tasks}"
        )
    return rows


def signal_summary(rows: list[dict[str, str]]) -> dict:
    metric_names = (
        "gain_db_to_input",
        "gain_db_to_clean",
        "clean_active_gain_db_to_input",
        "clean_active_gain_db_to_clean",
        "active_ratio_delta_to_clean",
        "si_sdri_db",
        "sdri_db",
        "si_sdr_db",
        "sdr_db",
    )
    summary = {"n": len(rows)}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def avqi_metrics(directory: Path, output_name: str) -> dict:
    payload = read_json(directory / "metrics.json")
    metrics = payload["metrics"][output_name]
    required = {
        "abs_avqi_gap_to_clean",
        "pathology_avqi_gap_to_clean",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(
            f"{directory} output {output_name!r} misses AVQI fields {missing}"
        )
    return metrics


def validate_checkpoint_pair(standard_dir: Path, dnf_dir: Path) -> dict:
    standard = read_json(standard_dir / "checkpoint_receipt.json")
    dnf = read_json(dnf_dir / "checkpoint_receipt.json")
    mismatches = {
        field: {
            "standard": standard.get(field),
            "dnf": dnf.get(field),
        }
        for field in PAIR_FIELDS
        if standard.get(field) != dnf.get(field)
    }
    if standard.get("mode") != "standard":
        mismatches["standard_mode"] = standard.get("mode")
    if dnf.get("mode") != "dnf":
        mismatches["dnf_mode"] = dnf.get("mode")
    if mismatches:
        raise ValueError(f"TAU checkpoint pair mismatch: {mismatches}")
    controlled = read_json(Path(str(standard["controlled_comparison"])))
    if not controlled.get("controlled_gate_pass", False):
        raise ValueError("TAU checkpoint pair lacks a passing controlled gate")
    return {
        "standard": standard,
        "dnf": dnf,
        "controlled_gate": controlled,
    }


def main() -> None:
    args = parse_args()
    standard_selection = read_json(
        args.standard_dir / "selection_receipt.json"
    )
    dnf_selection = read_json(args.dnf_dir / "selection_receipt.json")
    if standard_selection != dnf_selection:
        raise ValueError("Standard and DNF TAU selections differ")
    standard_uids = selected_uids(args.standard_dir / "pair_manifest.csv")
    dnf_uids = selected_uids(args.dnf_dir / "pair_manifest.csv")
    if standard_uids != dnf_uids:
        raise ValueError("Standard and DNF TAU UIDs differ")
    checkpoint_pair = validate_checkpoint_pair(
        args.standard_dir,
        args.dnf_dir,
    )

    standard_signal = index_signal_rows(
        args.standard_dir / "signal_guard.csv"
    )
    dnf_signal = index_signal_rows(args.dnf_dir / "signal_guard.csv")
    severe_sv = {
        "standard": signal_summary(
            select_signal_rows(
                standard_signal,
                output_name="standard",
                groups={"pathological_severe"},
                tasks={"sv"},
            )
        ),
        "eq14": signal_summary(
            select_signal_rows(
                dnf_signal,
                output_name="eq14",
                groups={"pathological_severe"},
                tasks={"sv"},
            )
        ),
        "speech_head": signal_summary(
            select_signal_rows(
                dnf_signal,
                output_name="speech_head",
                groups={"pathological_severe"},
                tasks={"sv"},
            )
        ),
    }
    pathology_eq14 = signal_summary(
        select_signal_rows(
            dnf_signal,
            output_name="eq14",
            groups={"pathological_mild", "pathological_severe"},
            tasks={"cs", "sv"},
        )
    )
    all_signal = {
        "standard": signal_summary(
            select_signal_rows(
                standard_signal,
                output_name="standard",
                groups={
                    "healthy_low",
                    "pathological_mild",
                    "pathological_severe",
                },
                tasks={"cs", "sv"},
            )
        ),
        "eq14": signal_summary(
            select_signal_rows(
                dnf_signal,
                output_name="eq14",
                groups={
                    "healthy_low",
                    "pathological_mild",
                    "pathological_severe",
                },
                tasks={"cs", "sv"},
            )
        ),
        "speech_head": signal_summary(
            select_signal_rows(
                dnf_signal,
                output_name="speech_head",
                groups={
                    "healthy_low",
                    "pathological_mild",
                    "pathological_severe",
                },
                tasks={"cs", "sv"},
            )
        ),
    }

    standard_avqi = avqi_metrics(args.standard_dir, "standard")
    standard_input_avqi = avqi_metrics(args.standard_dir, "input")
    dnf_input_avqi = avqi_metrics(args.dnf_dir, "input")
    numeric_input_keys = {
        key
        for key, value in standard_input_avqi.items()
        if isinstance(value, (int, float))
        and isinstance(dnf_input_avqi.get(key), (int, float))
    }
    input_mismatches = {
        key: {
            "standard": standard_input_avqi[key],
            "dnf": dnf_input_avqi[key],
        }
        for key in numeric_input_keys
        if abs(
            float(standard_input_avqi[key])
            - float(dnf_input_avqi[key])
        )
        > 1.0e-9
    }
    if input_mismatches:
        raise ValueError(
            f"Standard and DNF input AVQI baselines differ: {input_mismatches}"
        )
    eq14_avqi = avqi_metrics(args.dnf_dir, "eq14")
    speech_avqi = avqi_metrics(args.dnf_dir, "speech_head")
    severe_eq14_gain = severe_sv["eq14"][
        "clean_active_gain_db_to_clean"
    ]
    severe_standard_gain = severe_sv["standard"][
        "clean_active_gain_db_to_clean"
    ]
    severe_speech_gain = severe_sv["speech_head"][
        "clean_active_gain_db_to_clean"
    ]
    gates = {
        "all_required_tau_strata_present": (
            set(standard_selection["selected_speakers"])
            == {
                "healthy_low",
                "pathological_mild",
                "pathological_severe",
            }
            and set(standard_selection["required_tasks"]) == {"cs", "sv"}
        ),
        "severe_sv_eq14_worst_clean_active_gain_at_least_minus_3db": (
            severe_eq14_gain["min"] >= -3.0
        ),
        "severe_sv_eq14_mean_clean_active_gain_not_over_1db_below_standard": (
            severe_eq14_gain["mean"] - severe_standard_gain["mean"] >= -1.0
        ),
        "severe_sv_eq14_mean_clean_active_gain_not_over_1db_below_speech_head": (
            severe_eq14_gain["mean"] - severe_speech_gain["mean"] >= -1.0
        ),
        "pathology_eq14_worst_active_ratio_delta_at_least_minus_0_10": (
            pathology_eq14["active_ratio_delta_to_clean"]["min"] >= -0.10
        ),
        "eq14_mean_si_sdri_nonnegative": (
            all_signal["eq14"]["si_sdri_db"]["mean"] >= 0.0
        ),
        "eq14_mean_sdri_nonnegative": (
            all_signal["eq14"]["sdri_db"]["mean"] >= 0.0
        ),
        "eq14_mean_si_sdri_not_over_0_5db_below_standard": (
            all_signal["eq14"]["si_sdri_db"]["mean"]
            - all_signal["standard"]["si_sdri_db"]["mean"]
            >= -0.5
        ),
        "eq14_overall_abs_avqi_gap_not_worse_than_input": (
            float(eq14_avqi["abs_avqi_gap_to_clean"])
            <= float(standard_input_avqi["abs_avqi_gap_to_clean"])
        ),
        "eq14_overall_abs_avqi_gap_not_over_0_25_worse_than_standard": (
            float(eq14_avqi["abs_avqi_gap_to_clean"])
            - float(standard_avqi["abs_avqi_gap_to_clean"])
            <= 0.25
        ),
        "eq14_pathology_avqi_gap_not_over_0_25_worse_than_standard": (
            abs(float(eq14_avqi["pathology_avqi_gap_to_clean"]))
            - abs(float(standard_avqi["pathology_avqi_gap_to_clean"]))
            <= 0.25
        ),
        "eq14_overall_abs_avqi_gap_not_over_0_25_worse_than_speech_head": (
            float(eq14_avqi["abs_avqi_gap_to_clean"])
            - float(speech_avqi["abs_avqi_gap_to_clean"])
            <= 0.25
        ),
    }
    payload = {
        "schema_version": "dnf-phase-a-tau-pair-comparison-v2",
        "standard_dir": str(args.standard_dir.resolve()),
        "dnf_dir": str(args.dnf_dir.resolve()),
        "uid_count": len(standard_uids),
        "selection": standard_selection,
        "checkpoint_pair": checkpoint_pair,
        "signal_slices": {
            "all": all_signal,
            "pathological_severe_sv": severe_sv,
            "pathology_eq14_all_tasks": pathology_eq14,
        },
        "avqi": {
            "input": standard_input_avqi,
            "standard": standard_avqi,
            "eq14": eq14_avqi,
            "speech_head": speech_avqi,
        },
        "gates": gates,
        "tau_safety_gate_pass": all(gates.values()),
        "manual_blind_listening_required": True,
        "claim_limit": (
            "This is a small stratified safety screen with raw-scale inference "
            "and clean-reference signal metrics, not evidence that TAU "
            "pathology is preserved in general."
        ),
    }
    if args.output_json.exists():
        raise FileExistsError(
            f"refusing to overwrite TAU comparison: {args.output_json}"
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if args.require_pass and not payload["tau_safety_gate_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
