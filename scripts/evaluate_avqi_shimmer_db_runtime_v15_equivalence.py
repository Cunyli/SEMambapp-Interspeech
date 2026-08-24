#!/usr/bin/env python3
"""Audit Candidate-C v15 latency and exact equivalence on dev-only cases.

The opened v14 panel is permanently dev-only here.  Frozen v13 topology is
compared with a persistent exact-Praat worker before any new speaker panel is
authorized.  This script never loads or updates the generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from scripts.avqi_shimmer_exact_topology_runtime import (
    EXPECTED_IMPLEMENTATION,
    NUMPY_HIGHPASS_MODE,
    PRAAT_HIGHPASS_MODE,
    ExactShimmerTopologyWorker,
    require_exact_topology_equal,
    topology_sha256,
)
from scripts.evaluate_avqi_shimmer_hybrid_topology import (
    FIXED_ALPHA,
    load_predictor,
    metric_source_indices_from_topology,
    normalized_gradient_step,
    read_waveform,
    run_exact,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
SHIMMER_DB_INDEX = AVQI_COMPONENT_NAMES.index("shimmer_db")
FORMAL_REFRESH_GATE_MS = 500.0
DEV_ENGINEERING_MARGIN_MS = 450.0
DEFAULT_WARM_REPEATS = 3
EXPECTED_CASE_COUNT = 12
EXPECTED_SPEAKER_COUNT = 6
EXPECTED_SLICE_COUNTS = {
    "view": {"cs": 6, "sv": 6},
    "sample_group": {
        "pathological_mild": 6,
        "pathological_severe": 6,
    },
    "condition": {"rir_only": 4, "snr20": 4, "snr10": 4},
}
PASS_DECISION = (
    "PASS_SHIMMER_DB_RUNTIME_V15_EXACT_EQUIVALENCE_FREEZE_FOR_NEW_PANEL"
)
FAIL_DECISION = "NO_GO_SHIMMER_DB_RUNTIME_V15_DEV_EQUIVALENCE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--panel-contract-sha256", required=True)
    parser.add_argument("--target-label-contract", type=Path, required=True)
    parser.add_argument("--target-label-contract-sha256", required=True)
    parser.add_argument("--candidate-seal", type=Path, required=True)
    parser.add_argument("--candidate-seal-sha256", required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint-sha256", required=True)
    parser.add_argument("--exact-python", type=Path, required=True)
    parser.add_argument("--avqi-code-root", type=Path, required=True)
    parser.add_argument("--avqi-code-tree-sha256", required=True)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "avqi_shimmer_exact_topology_worker.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warm-repeats", type=int, default=DEFAULT_WARM_REPEATS)
    parser.add_argument(
        "--highpass-mode",
        choices=(PRAAT_HIGHPASS_MODE, NUMPY_HIGHPASS_MODE),
        default=PRAAT_HIGHPASS_MODE,
        help="Optimized worker high-pass; frozen reference remains exact Praat.",
    )
    return parser.parse_args()


def repository_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty equivalence CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")
    return observed


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_dev_contract(
    panel: dict[str, Any],
    targets: dict[str, Any],
    seal: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, Any]]]:
    rows = [dict(row) for row in panel.get("rows", [])]
    if len(rows) != EXPECTED_CASE_COUNT or len(
        {row["case_id"] for row in rows}
    ) != EXPECTED_CASE_COUNT:
        raise ValueError("v15 equivalence requires twelve unique v14 dev cases")
    if len({row["speaker_id"] for row in rows}) != EXPECTED_SPEAKER_COUNT:
        raise ValueError("v15 equivalence requires six v14 dev speakers")
    for field, expected in EXPECTED_SLICE_COUNTS.items():
        if Counter(row[field] for row in rows) != Counter(expected):
            raise ValueError(f"v15 dev slice contract drift: {field}")
    if panel.get("panel_status") != "sealed_new_speaker_panel_before_exact_outcomes":
        raise ValueError("v14 panel contract status drift")
    if panel.get("candidate_c", {}).get("fixed_alpha") != FIXED_ALPHA:
        raise ValueError("v14 fixed alpha drift")
    if panel.get("generator", {}).get("optimizer_steps") != 0:
        raise ValueError("v14 panel contains generator optimizer steps")

    target_rows = {
        row["case_id"]: float(row["exact_target_shimmer_db"])
        for row in targets.get("rows", [])
    }
    seal_rows = {
        row["case_id"]: dict(row) for row in seal.get("rows", [])
    }
    case_ids = {row["case_id"] for row in rows}
    if set(target_rows) != case_ids or set(seal_rows) != case_ids:
        raise ValueError("v14 target/seal case coverage drift")
    if seal.get("fixed_alpha") != FIXED_ALPHA:
        raise ValueError("v14 candidate seal alpha drift")
    if seal.get("selection_or_tuning_on_this_panel") is not False:
        raise ValueError("v14 candidate seal selection contract drift")

    for row in rows:
        base_path = Path(row["base_path"])
        validate_hash(base_path, row["base_sha256"], "v14 base waveform")
        candidate = seal_rows[row["case_id"]]
        validate_hash(
            Path(candidate["candidate_path"]),
            candidate["candidate_sha256"],
            "v14 sealed candidate waveform",
        )
        if candidate["view"] != row["view"]:
            raise ValueError("v14 candidate seal view drift")
    return rows, target_rows, seal_rows


def topology_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"dev_base:{row['case_id']}",
        "case_id": row["case_id"],
        "role": "current_s3_500_output_topology",
        "path": str(Path(row["base_path"]).resolve()),
        "view": row["view"],
        "score_components": False,
        "exact_metric_topology": True,
    }


def step_signature(
    row: dict[str, Any],
    topology: dict[str, Any],
    target_shimmer_db: float,
    predictor: torch.nn.Module,
    target_scale: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    waveform = read_waveform(Path(row["base_path"])).to(device)
    waveform = waveform.requires_grad_(True)
    source_indices = torch.as_tensor(
        metric_source_indices_from_topology(
            topology,
            source_sample_count=waveform.numel(),
        ),
        dtype=torch.long,
        device=device,
    )
    pulses = waveform.new_tensor(topology["pulse_positions_samples"])
    synchronize(device)
    started = time.perf_counter()
    proxy_before = predictor.raw_shimmer_from_pulse_positions(
        waveform,
        pulses,
        metric_source_indices=source_indices,
        metric_constant_prefix_samples=int(
            topology["metric_constant_prefix_samples"]
        ),
    )[1]
    scale = target_scale[SHIMMER_DB_INDEX].clamp_min(1e-8)
    loss = ((proxy_before - target_shimmer_db) / scale).square()
    gradient = torch.autograd.grad(loss, waveform)[0]
    candidate = normalized_gradient_step(waveform, gradient, FIXED_ALPHA)
    synchronize(device)
    torch_step_ms = 1000.0 * (time.perf_counter() - started)
    if not bool(torch.isfinite(gradient).all()) or float(gradient.norm()) <= 0.0:
        raise ValueError(f"invalid Shimmer-dB gradient: {row['case_id']}")
    if not bool(torch.isfinite(candidate).all()) or float(candidate.abs().max()) >= 0.999:
        raise ValueError(f"invalid Shimmer-dB candidate: {row['case_id']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        candidate.detach().cpu().numpy(),
        SAMPLE_RATE,
        subtype="PCM_24",
    )
    stored = read_waveform(output_path).to(device)
    with torch.inference_mode():
        proxy_after = predictor.raw_shimmer_from_pulse_positions(
            stored,
            pulses,
            metric_source_indices=source_indices,
            metric_constant_prefix_samples=int(
                topology["metric_constant_prefix_samples"]
            ),
        )[1]
    return {
        "proxy_before": float(proxy_before.detach()),
        "proxy_before_hex": float(proxy_before.detach()).hex(),
        "proxy_after": float(proxy_after.detach()),
        "proxy_after_hex": float(proxy_after.detach()).hex(),
        "loss": float(loss.detach()),
        "loss_hex": float(loss.detach()).hex(),
        "gradient_sha256": tensor_sha256(gradient),
        "gradient_l2_norm": float(gradient.norm()),
        "gradient_l2_norm_hex": float(gradient.norm()).hex(),
        "candidate_tensor_sha256": tensor_sha256(candidate),
        "candidate_pcm24_sha256": sha256_file(output_path),
        "torch_step_ms": torch_step_ms,
        "path": str(output_path.resolve()),
    }


def require_step_equal(
    frozen: dict[str, Any],
    optimized: dict[str, Any],
    case_id: str,
) -> None:
    fields = (
        "proxy_before_hex",
        "proxy_after_hex",
        "loss_hex",
        "gradient_sha256",
        "gradient_l2_norm_hex",
        "candidate_tensor_sha256",
        "candidate_pcm24_sha256",
    )
    differences = {
        field: {"frozen": frozen[field], "optimized": optimized[field]}
        for field in fields
        if frozen[field] != optimized[field]
    }
    if differences:
        raise ValueError(
            f"{case_id}: frozen/optimized forward-backward drift: "
            + json.dumps(differences, sort_keys=True)
        )


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    internal = [float(row["internal_refresh_ms"]) for row in rows]
    wall = [float(row["request_wall_ms"]) for row in rows]
    end_to_end = [float(row["end_to_end_refresh_ms"]) for row in rows]
    formal_pass = (
        max(internal) <= FORMAL_REFRESH_GATE_MS
        and max(end_to_end) <= FORMAL_REFRESH_GATE_MS
    )
    development_pass = (
        max(internal) <= DEV_ENGINEERING_MARGIN_MS
        and max(end_to_end) <= DEV_ENGINEERING_MARGIN_MS
    )
    return {
        "measurement_count": len(rows),
        "internal_refresh_ms": {
            "minimum": min(internal),
            "median": median(internal),
            "maximum": max(internal),
            "formal_gate_maximum": FORMAL_REFRESH_GATE_MS,
            "development_margin_maximum": DEV_ENGINEERING_MARGIN_MS,
        },
        "request_wall_ms": {
            "minimum": min(wall),
            "median": median(wall),
            "maximum": max(wall),
        },
        "end_to_end_refresh_ms": {
            "minimum": min(end_to_end),
            "median": median(end_to_end),
            "maximum": max(end_to_end),
            "includes_client_tmpfs_staging": True,
        },
        "formal_500ms_pass": formal_pass,
        "development_450ms_margin_pass": development_pass,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if repository_head(REPO_ROOT) != args.source_commit:
        raise ValueError("v15 equivalence source commit differs from HEAD")
    if args.warm_repeats < 3:
        raise ValueError("v15 equivalence requires at least three warm repeats")
    if FIXED_ALPHA != 0.001:
        raise ValueError("Candidate-C fixed alpha drift")

    source_hashes = {
        "panel_contract": validate_hash(
            args.panel_contract,
            args.panel_contract_sha256,
            "v14 panel contract",
        ),
        "target_label_contract": validate_hash(
            args.target_label_contract,
            args.target_label_contract_sha256,
            "v14 target-label contract",
        ),
        "candidate_seal": validate_hash(
            args.candidate_seal,
            args.candidate_seal_sha256,
            "v14 candidate seal",
        ),
        "predictor_checkpoint": validate_hash(
            args.predictor_checkpoint,
            args.predictor_checkpoint_sha256,
            "frozen Shimmer v6 predictor",
        ),
        "worker_script": sha256_file(args.worker_script),
        "avqi_code_tree": args.avqi_code_tree_sha256,
    }
    panel = load_json(args.panel_contract)
    target_contract = load_json(args.target_label_contract)
    seal = load_json(args.candidate_seal)
    panel_rows, target_by_case, seal_by_case = validate_dev_contract(
        panel,
        target_contract,
        seal,
    )

    args.output_dir.mkdir(parents=True)
    frozen_root = args.output_dir / "waveforms" / "frozen_v13"
    optimized_root = args.output_dir / "waveforms" / "optimized_v15"
    frozen_root.mkdir(parents=True)
    optimized_root.mkdir(parents=True)

    items = [topology_item(row) for row in panel_rows]
    base_waveforms = {
        row["case_id"]: read_waveform(Path(row["base_path"])).numpy()
        for row in panel_rows
    }
    frozen_started = time.perf_counter()
    frozen_report = run_exact(items, args.exact_python, args.avqi_code_root)
    frozen_batch_wall_ms = 1000.0 * (time.perf_counter() - frozen_started)
    frozen_by_case = {row["case_id"]: row for row in frozen_report["rows"]}

    runtime_rows: list[dict[str, Any]] = []
    path_runtime_rows: list[dict[str, Any]] = []
    optimized_by_case: dict[str, dict[str, Any]] = {}
    with ExactShimmerTopologyWorker(
        args.exact_python,
        args.worker_script,
        args.avqi_code_root,
        args.avqi_code_tree_sha256,
    ) as worker:
        worker_startup = dict(worker.startup)
        worker_startup_ms = worker.startup_ms
        synthetic_warmup, synthetic_warmup_wall_ms = worker.warmup()
        if (
            worker_startup["parselmouth_version"]
            != frozen_report["parselmouth_version"]
            or worker_startup["praat_version"] != frozen_report["praat_version"]
        ):
            raise ValueError("frozen/optimized exact runtime version drift")

        for case_index, (row, item) in enumerate(
            zip(panel_rows, items, strict=True),
            start=1,
        ):
            frozen_topology = frozen_by_case[row["case_id"]]
            path_item = {**item, "highpass_mode": PRAAT_HIGHPASS_MODE}
            path_refreshed, path_wall_ms = worker.refresh([path_item])
            path_topology = path_refreshed[0]
            require_exact_topology_equal(
                frozen_topology,
                path_topology,
                f"{row['case_id']}:persistent-path-reference",
            )
            path_timing = path_topology["timing_ms"]
            path_runtime_rows.append(
                {
                    "case_id": row["case_id"],
                    "speaker_id": row["speaker_id"],
                    "view": row["view"],
                    "condition": row["condition"],
                    "input_read_ms": path_timing["input_read"],
                    "highpass_ms": path_timing["highpass"],
                    "highpass_mode": path_topology["metric_highpass"],
                    "highpass_pcm16_sha256": path_topology[
                        "highpass_pcm16_sha256"
                    ],
                    "metric_pcm16_sha256": path_topology[
                        "metric_pcm16_sha256"
                    ],
                    "highpass_input_pcm16_roundtrip_ms": path_timing[
                        "highpass_input_pcm16_roundtrip"
                    ],
                    "highpass_sound_construct_ms": path_timing[
                        "highpass_sound_construct"
                    ],
                    "highpass_stop_hann_filter_ms": path_timing[
                        "highpass_stop_hann_filter"
                    ],
                    "highpass_peak_extremum_ms": path_timing[
                        "highpass_peak_extremum"
                    ],
                    "highpass_scale_peak_ms": path_timing[
                        "highpass_scale_peak"
                    ],
                    "highpass_filter_compute_ms": path_timing[
                        "highpass_filter_compute"
                    ],
                    "highpass_quantize_ms": path_timing["highpass_quantize"],
                    "internal_refresh_ms": path_topology["pulse_runtime_ms"],
                    "request_wall_ms": path_wall_ms,
                }
            )
            for repeat_index in range(1, args.warm_repeats + 1):
                refreshed, wall_ms, staging_rows = (
                    worker.refresh_current_waveforms(
                        [item],
                        [base_waveforms[row["case_id"]]],
                        highpass_mode=args.highpass_mode,
                    )
                )
                optimized_topology = refreshed[0]
                staging = staging_rows[0]
                identity_hash = require_exact_topology_equal(
                    frozen_topology,
                    optimized_topology,
                    f"{row['case_id']}:repeat={repeat_index}",
                )
                highpass_pcm16_equal = (
                    optimized_topology["highpass_pcm16_sha256"]
                    == path_topology["highpass_pcm16_sha256"]
                )
                metric_pcm16_equal = (
                    optimized_topology["metric_pcm16_sha256"]
                    == path_topology["metric_pcm16_sha256"]
                )
                if not highpass_pcm16_equal or not metric_pcm16_equal:
                    raise ValueError(
                        f"{row['case_id']}: optimized high-pass PCM16 drift"
                    )
                if repeat_index == 1:
                    optimized_by_case[row["case_id"]] = optimized_topology
                else:
                    require_exact_topology_equal(
                        optimized_by_case[row["case_id"]],
                        optimized_topology,
                        f"{row['case_id']}:persistent-repeat={repeat_index}",
                    )
                timing = optimized_topology["timing_ms"]
                runtime_rows.append(
                    {
                        "case_id": row["case_id"],
                        "speaker_id": row["speaker_id"],
                        "view": row["view"],
                        "sample_group": row["sample_group"],
                        "condition": row["condition"],
                        "repeat_index": repeat_index,
                        "topology_sha256": identity_hash,
                        "pulse_count": optimized_topology["pulse_count"],
                        "source_range_count": optimized_topology[
                            "metric_source_range_count"
                        ],
                        "input_read_ms": timing["input_read"],
                        "client_tmpfs_staging_ms": staging["staging_ms"],
                        "highpass_ms": timing["highpass"],
                        "highpass_mode": optimized_topology[
                            "metric_highpass"
                        ],
                        "highpass_pcm16_sha256": optimized_topology[
                            "highpass_pcm16_sha256"
                        ],
                        "metric_pcm16_sha256": optimized_topology[
                            "metric_pcm16_sha256"
                        ],
                        "highpass_pcm16_equal_to_praat": highpass_pcm16_equal,
                        "metric_pcm16_equal_to_praat": metric_pcm16_equal,
                        "highpass_input_pcm16_roundtrip_ms": timing[
                            "highpass_input_pcm16_roundtrip"
                        ],
                        "highpass_sound_construct_ms": timing[
                            "highpass_sound_construct"
                        ],
                        "highpass_stop_hann_filter_ms": timing[
                            "highpass_stop_hann_filter"
                        ],
                        "highpass_peak_extremum_ms": timing[
                            "highpass_peak_extremum"
                        ],
                        "highpass_scale_peak_ms": timing[
                            "highpass_scale_peak"
                        ],
                        "highpass_filter_compute_ms": timing[
                            "highpass_filter_compute"
                        ],
                        "highpass_quantize_ms": timing["highpass_quantize"],
                        "textgrid_ms": timing["textgrid"],
                        "source_selection_ms": timing["source_selection"],
                        "metric_gather_ms": timing["metric_gather"],
                        "pointprocess_construct_ms": timing[
                            "pointprocess_construct"
                        ],
                        "pulse_enumeration_ms": timing["pulse_enumeration"],
                        "internal_refresh_ms": optimized_topology[
                            "pulse_runtime_ms"
                        ],
                        "request_wall_ms": wall_ms,
                        "end_to_end_refresh_ms": (
                            float(staging["staging_ms"]) + wall_ms
                        ),
                    }
                )
            print(
                f"topology_equivalence={case_index}/{len(panel_rows)}",
                flush=True,
            )

    device = torch.device(args.device)
    torch.manual_seed(20260824)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260824)
    torch.use_deterministic_algorithms(True)
    predictor, _, _, target_scale = load_predictor(
        args.predictor_checkpoint,
        device,
    )

    equivalence_rows: list[dict[str, Any]] = []
    for index, row in enumerate(panel_rows, start=1):
        case_id = row["case_id"]
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in case_id
        )
        frozen_signature = step_signature(
            row,
            frozen_by_case[case_id],
            target_by_case[case_id],
            predictor,
            target_scale,
            device,
            frozen_root / f"{safe_name}__frozen_v13.wav",
        )
        optimized_signature = step_signature(
            row,
            optimized_by_case[case_id],
            target_by_case[case_id],
            predictor,
            target_scale,
            device,
            optimized_root / f"{safe_name}__optimized_v15.wav",
        )
        require_step_equal(frozen_signature, optimized_signature, case_id)
        sealed_hash = seal_by_case[case_id]["candidate_sha256"]
        sealed_pcm24_equal = (
            optimized_signature["candidate_pcm24_sha256"] == sealed_hash
        )
        if not sealed_pcm24_equal:
            raise ValueError(f"{case_id}: v15 candidate differs from v14 PCM24 seal")
        runtime_for_case = [
            runtime
            for runtime in runtime_rows
            if runtime["case_id"] == case_id
        ]
        equivalence_rows.append(
            {
                "case_id": case_id,
                "speaker_id": row["speaker_id"],
                "view": row["view"],
                "sample_group": row["sample_group"],
                "condition": row["condition"],
                "fixed_alpha": FIXED_ALPHA,
                "topology_sha256": topology_sha256(
                    optimized_by_case[case_id]
                ),
                "highpass_mode": optimized_by_case[case_id][
                    "metric_highpass"
                ],
                "highpass_pcm16_sha256": optimized_by_case[case_id][
                    "highpass_pcm16_sha256"
                ],
                "metric_pcm16_sha256": optimized_by_case[case_id][
                    "metric_pcm16_sha256"
                ],
                "highpass_pcm16_equal_to_praat": True,
                "metric_pcm16_equal_to_praat": True,
                "source_mapping_equal": True,
                "pulse_positions_equal": True,
                "forward_proxy_hex": optimized_signature["proxy_before_hex"],
                "forward_proxy_equal": True,
                "loss_equal": True,
                "gradient_sha256": optimized_signature["gradient_sha256"],
                "gradient_equal": True,
                "candidate_tensor_sha256": optimized_signature[
                    "candidate_tensor_sha256"
                ],
                "candidate_tensor_equal": True,
                "candidate_pcm24_sha256": optimized_signature[
                    "candidate_pcm24_sha256"
                ],
                "frozen_v13_pcm24_equal": True,
                "v14_sealed_pcm24_equal": sealed_pcm24_equal,
                "gradient_l2_norm": optimized_signature["gradient_l2_norm"],
                "optimized_torch_step_ms": optimized_signature[
                    "torch_step_ms"
                ],
                "refresh_internal_max_ms": max(
                    value["internal_refresh_ms"] for value in runtime_for_case
                ),
                "refresh_wall_max_ms": max(
                    value["request_wall_ms"] for value in runtime_for_case
                ),
                "refresh_end_to_end_max_ms": max(
                    value["end_to_end_refresh_ms"]
                    for value in runtime_for_case
                ),
                "total_metric_step_overhead_max_ms": max(
                    value["end_to_end_refresh_ms"]
                    for value in runtime_for_case
                )
                + optimized_signature["torch_step_ms"],
            }
        )
        print(f"step_equivalence={index}/{len(panel_rows)}", flush=True)

    runtime_summary = summarize_runtime(runtime_rows)
    all_equivalent = len(equivalence_rows) == EXPECTED_CASE_COUNT and all(
        row["source_mapping_equal"]
        and row["pulse_positions_equal"]
        and row["forward_proxy_equal"]
        and row["gradient_equal"]
        and row["candidate_tensor_equal"]
        and row["frozen_v13_pcm24_equal"]
        and row["v14_sealed_pcm24_equal"]
        for row in equivalence_rows
    )
    passed = (
        all_equivalent
        and runtime_summary["formal_500ms_pass"]
        and runtime_summary["development_450ms_margin_pass"]
    )
    decision = PASS_DECISION if passed else FAIL_DECISION

    runtime_path = args.output_dir / "runtime_samples.csv"
    path_runtime_path = args.output_dir / "path_reference_runtime_samples.csv"
    results_path = args.output_dir / "equivalence_results.csv"
    write_csv(runtime_path, runtime_rows)
    write_csv(path_runtime_path, path_runtime_rows)
    write_csv(results_path, equivalence_rows)
    report = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v15-equivalence-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "candidate": "praat_current_output_topology_refresh_db_alpha_0p001",
        "route_type": "hybrid_praat_assisted_straight_through_metric_branch",
        "pure_torch_estimator": False,
        "implementation": EXPECTED_IMPLEMENTATION,
        "optimized_highpass_mode": args.highpass_mode,
        "reference_highpass_mode": PRAAT_HIGHPASS_MODE,
        "optimized_highpass_contract": (
            "official_praat_6_1_38_power_of_two_fft_stop_hann_"
            "33p9_to_34p1_inverse_truncate"
        ),
        "fixed_alpha": FIXED_ALPHA,
        "scientific_gates_changed": False,
        "formal_refresh_gate_ms": FORMAL_REFRESH_GATE_MS,
        "development_engineering_margin_ms": DEV_ENGINEERING_MARGIN_MS,
        "metric_highpass_only": True,
        "emitted_waveform_full_band": True,
        "waveform_dependent_topology_cache": False,
        "current_output_handoff": (
            "hash_bound_client_tmpfs_raw_float32_per_waveform_step"
        ),
        "shared_or_dataset_filesystem_read_inside_timed_refresh": False,
        "current_output_topology_refresh_per_waveform_step": True,
        "clean_target_topology_drives_output": False,
        "dev_cases_are_ineligible_for_future_promotion_panel": True,
        "dev_case_count": len(panel_rows),
        "dev_speaker_count": len({row["speaker_id"] for row in panel_rows}),
        "frozen_v13_batch_wall_ms": frozen_batch_wall_ms,
        "worker_startup": worker_startup,
        "worker_startup_ms": worker_startup_ms,
        "synthetic_warmup": synthetic_warmup,
        "synthetic_warmup_wall_ms": synthetic_warmup_wall_ms,
        "warm_repeats_per_case": args.warm_repeats,
        "runtime": runtime_summary,
        "equivalence": {
            "all_12_topologies_equal": all_equivalent,
            "all_12_source_mappings_equal": all(
                row["source_mapping_equal"] for row in equivalence_rows
            ),
            "all_12_pulse_arrays_equal": all(
                row["pulse_positions_equal"] for row in equivalence_rows
            ),
            "all_12_forward_values_equal": all(
                row["forward_proxy_equal"] for row in equivalence_rows
            ),
            "all_12_gradients_equal": all(
                row["gradient_equal"] for row in equivalence_rows
            ),
            "all_12_step_tensors_equal": all(
                row["candidate_tensor_equal"] for row in equivalence_rows
            ),
            "all_12_pcm24_equal_to_frozen_v13": all(
                row["frozen_v13_pcm24_equal"] for row in equivalence_rows
            ),
            "all_12_pcm24_equal_to_v14_seal": all(
                row["v14_sealed_pcm24_equal"] for row in equivalence_rows
            ),
        },
        "source_sha256": source_hashes,
        "generator_loaded": False,
        "generator_optimizer_created": False,
        "generator_optimizer_steps": 0,
        "formal_generator_training_authorized": False,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "new_sealed_panel_authorized": passed,
        "artifacts": {
            "runtime_samples": runtime_path.name,
            "path_reference_runtime_samples": path_runtime_path.name,
            "equivalence_results": results_path.name,
        },
    }
    report_path = args.output_dir / "diagnostic_report.json"
    write_json(report_path, report)
    receipt = {
        "schema_version": "avqi-route-c-shimmer-db-runtime-v15-receipt-v1",
        "decision": decision,
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
        "fixed_alpha": FIXED_ALPHA,
        "formal_refresh_gate_ms": FORMAL_REFRESH_GATE_MS,
        "generator_optimizer_steps": 0,
        "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
        "new_sealed_panel_authorized": passed,
        "artifact_sha256": {
            report_path.name: sha256_file(report_path),
            runtime_path.name: sha256_file(runtime_path),
            path_runtime_path.name: sha256_file(path_runtime_path),
            results_path.name: sha256_file(results_path),
        },
    }
    write_json(args.output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
