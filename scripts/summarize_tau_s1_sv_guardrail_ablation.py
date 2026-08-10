#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("clean_identity", "phone_room_mid")
SIGNAL_SLICES = (
    ("pathological_severe", "sv"),
    ("pathological_mild", "sv"),
    ("healthy_low", "sv"),
    ("pathological_severe", "cs"),
    ("pathological_mild", "cs"),
    ("healthy_low", "cs"),
)
EXPECTED_INPUTS = {
    "clean_identity": {
        "avqi": 2.707136085355239,
        "health_avqi_gap_to_clean": 0.0,
        "pathology_avqi_gap_to_clean": 0.0,
        "n_speakers": 25.0,
    },
    "phone_room_mid": {
        "avqi": 6.269037423272441,
        "health_avqi_gap_to_clean": 4.507063697127421,
        "pathology_avqi_gap_to_clean": 3.030247510861455,
        "n_speakers": 25.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize fixed-panel TAU S1 SV-guardrail ablation probes."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Probe run directory containing clean_identity/ and phone_room_mid/. Repeat per checkpoint.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"Invalid --run value {spec!r}; expected LABEL=PATH.")
    return label, Path(raw_path).expanduser()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def finite_number(mapping: dict[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise KeyError(f"Missing {context}.{key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"{context}.{key} is not finite: {value}")
    return value


def validate_input_metrics(condition: str, metrics: dict[str, Any], tolerance: float) -> None:
    expected = EXPECTED_INPUTS[condition]
    for key, expected_value in expected.items():
        actual = finite_number(metrics, key, f"{condition}.input_metrics")
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"{condition} input baseline changed for {key}: "
                f"expected {expected_value:.12g}, got {actual:.12g}"
            )


def read_signal_summary(path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    selected: dict[tuple[str, str], dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_group", "task", "metric", "n", "mean", "min", "max"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")
        for raw_row in reader:
            if raw_row["metric"] != "gain_db_vs_input":
                continue
            key = (raw_row["sample_group"], raw_row["task"])
            if key in selected:
                raise ValueError(f"Duplicate gain_db_vs_input row for {key} in {path}")
            values = {
                "n": int(raw_row["n"]),
                "mean": float(raw_row["mean"]),
                "min": float(raw_row["min"]),
                "max": float(raw_row["max"]),
            }
            if not all(math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"Non-finite signal summary for {key} in {path}")
            selected[key] = values

    missing = [key for key in SIGNAL_SLICES if key not in selected]
    if missing:
        raise ValueError(f"{path} is missing gain_db_vs_input slices: {missing}")
    return selected


def signal_prefix(sample_group: str, task: str) -> str:
    return f"{sample_group}_{task}_gain"


def load_condition(label: str, run_dir: Path, condition: str, tolerance: float) -> dict[str, Any]:
    condition_dir = run_dir / condition
    metrics_path = condition_dir / "metrics.json"
    signal_path = condition_dir / "signal_guard_summary.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not signal_path.is_file():
        raise FileNotFoundError(signal_path)

    payload = read_json_object(metrics_path)
    input_metrics = payload.get("input_metrics")
    enhanced_metrics = payload.get("enhanced_metrics")
    metric_delta = payload.get("metric_delta")
    if not isinstance(input_metrics, dict) or not isinstance(enhanced_metrics, dict) or not isinstance(metric_delta, dict):
        raise ValueError(f"{metrics_path} is missing metric objects.")
    validate_input_metrics(condition, input_metrics, tolerance)

    row: dict[str, Any] = {
        "run": label,
        "condition": condition,
        "run_dir": str(run_dir),
        "checkpoint": str(payload.get("checkpoint", "")),
        "config": str(payload.get("config", "")),
        "input_avqi": finite_number(input_metrics, "avqi", f"{condition}.input_metrics"),
        "enhanced_avqi": finite_number(enhanced_metrics, "avqi", f"{condition}.enhanced_metrics"),
        "delta_avqi": finite_number(
            metric_delta,
            "delta_enhanced_minus_input_avqi",
            f"{condition}.metric_delta",
        ),
        "input_pathology_avqi_gap": finite_number(
            input_metrics,
            "pathology_avqi_gap_to_clean",
            f"{condition}.input_metrics",
        ),
        "enhanced_pathology_avqi_gap": finite_number(
            enhanced_metrics,
            "pathology_avqi_gap_to_clean",
            f"{condition}.enhanced_metrics",
        ),
        "input_health_avqi_gap": finite_number(
            input_metrics,
            "health_avqi_gap_to_clean",
            f"{condition}.input_metrics",
        ),
        "enhanced_health_avqi_gap": finite_number(
            enhanced_metrics,
            "health_avqi_gap_to_clean",
            f"{condition}.enhanced_metrics",
        ),
        "enhanced_abs_avqi_gap": finite_number(
            enhanced_metrics,
            "abs_avqi_gap_to_clean",
            f"{condition}.enhanced_metrics",
        ),
        "enhanced_overclean_rate": finite_number(
            enhanced_metrics,
            "overclean_rate",
            f"{condition}.enhanced_metrics",
        ),
        "n_speakers": finite_number(enhanced_metrics, "n_speakers", f"{condition}.enhanced_metrics"),
    }

    signal_rows = read_signal_summary(signal_path)
    for sample_group, task in SIGNAL_SLICES:
        prefix = signal_prefix(sample_group, task)
        values = signal_rows[(sample_group, task)]
        row[f"{prefix}_mean_db"] = values["mean"]
        row[f"{prefix}_min_db"] = values["min"]
        row[f"{prefix}_max_db"] = values["max"]
        row[f"{prefix}_n"] = values["n"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_avqi(row: dict[str, Any]) -> str:
    return f"{row['input_avqi']:.3f} -> {row['enhanced_avqi']:.3f} ({row['delta_avqi']:+.3f})"


def format_gap(row: dict[str, Any], prefix: str) -> str:
    return f"{row[f'input_{prefix}_avqi_gap']:.3f} -> {row[f'enhanced_{prefix}_avqi_gap']:.3f}"


def format_gain(row: dict[str, Any], sample_group: str, task: str) -> str:
    prefix = signal_prefix(sample_group, task)
    return f"{row[f'{prefix}_mean_db']:.2f} / {row[f'{prefix}_min_db']:.2f} / {row[f'{prefix}_n']}"


def markdown_table(condition: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {condition}",
        "",
        "| Run | AVQI input -> enhanced (delta) | Pathology gap input -> enhanced | Health gap input -> enhanced | Severe SV mean/min/n dB | Mild SV mean/min/n dB | Healthy SV mean/min/n dB | Severe CS mean/min/n dB | Mild CS mean/min/n dB | Healthy CS mean/min/n dB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    format_avqi(row),
                    format_gap(row, "pathology"),
                    format_gap(row, "health"),
                    format_gain(row, "pathological_severe", "sv"),
                    format_gain(row, "pathological_mild", "sv"),
                    format_gain(row, "healthy_low", "sv"),
                    format_gain(row, "pathological_severe", "cs"),
                    format_gain(row, "pathological_mild", "cs"),
                    format_gain(row, "healthy_low", "cs"),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TAU S1 SV-guardrail ablation metrics",
        "",
        "Gain cells are `mean / min / n` for `gain_db_vs_input`. Input baselines are validated before writing.",
        "",
    ]
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        lines.extend(markdown_table(condition, condition_rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative.")
    if not args.validate_only and args.output_root is None:
        raise ValueError("--output-root is required unless --validate-only is set.")

    run_specs = [parse_run_spec(spec) for spec in args.run]
    labels = [label for label, _ in run_specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate run labels: {labels}")

    rows = [
        load_condition(label, run_dir, condition, args.tolerance)
        for label, run_dir in run_specs
        for condition in CONDITIONS
    ]
    if args.validate_only:
        print(f"Validated {len(run_specs)} probe runs across {len(rows)} condition outputs.")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "ablation_metrics.csv", rows)
    (args.output_root / "ablation_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_root / "ablation_metrics.md", rows)
    print(f"Wrote {len(rows)} rows under {args.output_root}")


if __name__ == "__main__":
    main()
