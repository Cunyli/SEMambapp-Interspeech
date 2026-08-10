"""Compare a frozen scratch Standard/DNF Phase-A pair.

The primary efficacy view is the unmodified single-noise deployment input
``y = s + n1``.  A clean identity view is evaluated separately so aggregate
denoising gains cannot hide destructive over-processing.  The comparator
fails closed on pair receipts, code/config hashes, route coverage, Eq.14
validity, scale tails, and per-route behavior.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEPLOYMENT_VIEW = "single_noise_s_plus_n1"
IDENTITY_VIEW = "identity_clean_s"
REQUIRED_VIEWS = {DEPLOYMENT_VIEW, IDENTITY_VIEW}
REQUIRED_ROUTES = {"noisy", "clean_regular", "clean_weak"}

PAIR_EQUAL_FIELDS = (
    "train_manifest_sha256",
    "valid_manifest_sha256",
    "train_manifest_length",
    "valid_manifest_length",
    "canonical_speech_init_sha256",
    "seed",
    "max_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "learning_rate",
    "cut_duration_seconds",
    "validation_samples",
    "checkpoint_steps",
    "geometry_eps",
    "loss_variant",
    "active_log_rms_weight",
    "contract_sha256",
    "model_config_sha256",
    "training_script_sha256",
    "code_surface_sha256",
    "noise_pairing_policy",
    "speech_partition_policy",
    "deployment_validation_input",
    "evaluation_input_views",
    "paper_mechanism_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a Phase-A A/B pair.")
    parser.add_argument("--standard-dir", type=Path, required=True)
    parser.add_argument("--dnf-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/train/dnf_phase_ab_v2_contract.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Write the complete result, then exit non-zero when a gate fails.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    indexed = {}
    for row in read_jsonl(path):
        key = (
            str(row["sample_uid"]),
            str(row["evaluation_input_view"]),
            str(row["output_name"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate validation row: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"empty validation result: {path}")
    return indexed


def output_rows(
    indexed: dict[tuple[str, str, str], dict],
    *,
    view: str,
    output_name: str,
) -> dict[str, dict]:
    selected = {
        uid: row
        for (uid, row_view, name), row in indexed.items()
        if row_view == view and name == output_name
    }
    if not selected:
        raise ValueError(f"missing output {view}/{output_name}")
    return selected


def bootstrap_summary(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired values must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("paired values contain non-finite entries")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "degraded_fraction": float(np.mean(values < 0.0)),
        "bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": int(samples),
    }


def paired_difference(
    left: dict[str, dict],
    right: dict[str, dict],
    metric: str,
    *,
    seed: int,
    samples: int,
) -> dict:
    if set(left) != set(right):
        raise ValueError(
            f"paired UID mismatch: left_only={len(set(left) - set(right))}, "
            f"right_only={len(set(right) - set(left))}"
        )
    uids = sorted(left)
    values = np.asarray(
        [float(left[uid][metric]) - float(right[uid][metric]) for uid in uids],
        dtype=np.float64,
    )
    summary = bootstrap_summary(values, seed=seed, samples=samples)
    summary["metric"] = metric
    summary["direction"] = "left_minus_right"
    return summary


def summarize_output(rows: dict[str, dict]) -> dict:
    metrics = {}
    for metric in (
        "si_sdri_db",
        "sdri_db",
        "si_sdr_db",
        "sdr_db",
        "gain_db_to_clean",
        "gain_db_to_input",
        "active_gain_db_to_clean",
        "active_gain_db_to_input",
    ):
        values = np.asarray(
            [float(rows[uid][metric]) for uid in sorted(rows)],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"output metric {metric} contains non-finite values")
        metrics[metric] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return metrics


def split_rows(rows: dict[str, dict], field: str) -> dict[str, dict[str, dict]]:
    buckets: dict[str, dict[str, dict]] = defaultdict(dict)
    for uid, row in rows.items():
        value = row[field]
        key = f"{float(value):g}" if field == "target_snr_db" else str(value)
        buckets[key][uid] = row
    return dict(sorted(buckets.items()))


def summarize_slices(rows: dict[str, dict]) -> dict:
    return {
        field: {
            value: summarize_output(bucket)
            for value, bucket in split_rows(rows, field).items()
        }
        for field in ("route", "noise_family", "target_snr_db")
    }


def paired_slices(
    left: dict[str, dict],
    right: dict[str, dict],
    *,
    field: str,
    metric: str,
    seed: int,
    samples: int,
) -> dict:
    left_buckets = split_rows(left, field)
    right_buckets = split_rows(right, field)
    if set(left_buckets) != set(right_buckets):
        raise ValueError(f"paired {field} slices differ")
    return {
        value: paired_difference(
            left_buckets[value],
            right_buckets[value],
            metric,
            seed=seed + index,
            samples=samples,
        )
        for index, value in enumerate(sorted(left_buckets))
    }


def validate_row_contract(
    standard: dict[str, dict],
    eq14: dict[str, dict],
    speech_head: dict[str, dict],
) -> None:
    if set(standard) != set(eq14) or set(eq14) != set(speech_head):
        raise ValueError("Standard, Eq.14, and speech-head validation UIDs differ")
    fields = (
        "evaluation_input_view",
        "route",
        "noise_family",
        "target_snr_db",
        "weak_degradation",
    )
    mismatches = []
    for uid in sorted(standard):
        for field in fields:
            values = {
                "standard": standard[uid].get(field),
                "eq14": eq14[uid].get(field),
                "speech_head": speech_head[uid].get(field),
            }
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
                mismatches.append((uid, field, values))
                if len(mismatches) == 5:
                    break
        if len(mismatches) == 5:
            break
    if mismatches:
        raise ValueError(f"validation row contract mismatch: {mismatches}")
    if not all(bool(row.get("eq14_valid", False)) for row in eq14.values()):
        raise ValueError("deployment/identity results contain invalid Eq.14 rows")


def expected_route_counts(contract: dict) -> dict[str, int]:
    total = int(contract["training"]["validation_samples"])
    if total % 20:
        raise ValueError("validation sample count must be a multiple of 20")
    blocks = total // 20
    return {
        "noisy": 15 * blocks,
        "clean_regular": 4 * blocks,
        "clean_weak": blocks,
    }


def validate_route_coverage(rows: dict[str, dict], contract: dict) -> None:
    counts = Counter(str(row["route"]) for row in rows.values())
    expected = expected_route_counts(contract)
    if dict(counts) != expected:
        raise ValueError(f"validation route counts {dict(counts)} != {expected}")


def validate_arm(
    metadata: dict,
    summary: dict,
    *,
    mode: str,
    contract: dict,
    contract_sha256: str,
) -> None:
    required = {
        "phase": "A",
        "mode": mode,
        "scratch_init": True,
        "init_checkpoint": None,
        "resume": None,
        "gan": False,
        "max_steps": int(contract["training"]["max_steps"]),
        "validation_samples": int(contract["training"]["validation_samples"]),
        "contract_sha256": contract_sha256,
        "paper_mechanism_gate": True,
        "deployment_validation_input": contract["data"][
            "deployment_validation_input"
        ],
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": expected}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if summary.get("step") != int(contract["training"]["max_steps"]):
        mismatches["completed_step"] = {
            "observed": summary.get("step"),
            "expected": int(contract["training"]["max_steps"]),
        }
    if set(metadata.get("evaluation_input_views", [])) != REQUIRED_VIEWS:
        mismatches["evaluation_input_views"] = {
            "observed": metadata.get("evaluation_input_views"),
            "expected": sorted(REQUIRED_VIEWS),
        }
    if mismatches:
        raise ValueError(f"{mode} arm violates the frozen contract: {mismatches}")
    final_checkpoint = Path(str(summary["final_checkpoint"]))
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {final_checkpoint}")


def validate_pair_integrity(
    standard_dir: Path,
    dnf_dir: Path,
    standard_metadata: dict,
    dnf_metadata: dict,
    standard_summary: dict,
    dnf_summary: dict,
) -> dict:
    mismatches = {
        field: {
            "standard": standard_metadata.get(field),
            "dnf": dnf_metadata.get(field),
        }
        for field in PAIR_EQUAL_FIELDS
        if standard_metadata.get(field) != dnf_metadata.get(field)
    }
    if mismatches:
        raise ValueError(f"Standard and DNF pair metadata differ: {mismatches}")
    pair_dirs = {
        Path(str(standard_metadata["pair_contract_dir"])).resolve(),
        Path(str(dnf_metadata["pair_contract_dir"])).resolve(),
    }
    if len(pair_dirs) != 1:
        raise ValueError("Standard and DNF use different pair-contract directories")
    pair_dir = next(iter(pair_dirs))
    receipts = {
        mode: read_json(pair_dir / "receipts" / f"{mode}.json")
        for mode in ("standard", "dnf")
    }
    verification = read_json(pair_dir / "pair_verification.json")
    if verification.get("status") != "matched":
        raise ValueError("pair verification is not matched")
    receipt_mismatches = {
        field: {
            "standard": receipts["standard"].get(field),
            "dnf": receipts["dnf"].get(field),
            "verification": verification.get(field),
        }
        for field in PAIR_EQUAL_FIELDS
        if (
            receipts["standard"].get(field)
            != receipts["dnf"].get(field)
            or receipts["standard"].get(field) != verification.get(field)
        )
    }
    for field in ("uid_sequence_sha256", "uid_sequence_count"):
        values = {
            "standard": receipts["standard"].get(field),
            "dnf": receipts["dnf"].get(field),
            "verification": verification.get(field),
            "standard_summary": standard_summary.get(field),
            "dnf_summary": dnf_summary.get(field),
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            receipt_mismatches[field] = values
    expected_outputs = {
        "standard": str(standard_dir.resolve()),
        "dnf": str(dnf_dir.resolve()),
    }
    for mode, expected in expected_outputs.items():
        if receipts[mode].get("mode") != mode:
            receipt_mismatches[f"{mode}_mode"] = receipts[mode].get("mode")
        if str(Path(str(receipts[mode].get("output_dir"))).resolve()) != expected:
            receipt_mismatches[f"{mode}_output_dir"] = {
                "observed": receipts[mode].get("output_dir"),
                "expected": expected,
            }
    if receipt_mismatches:
        raise ValueError(f"pair receipt mismatch: {receipt_mismatches}")
    return {
        "pair_contract_dir": str(pair_dir),
        "pair_verification": verification,
        "standard_final_checkpoint_sha256": sha256_file(
            Path(str(standard_summary["final_checkpoint"]))
        ),
        "dnf_final_checkpoint_sha256": sha256_file(
            Path(str(dnf_summary["final_checkpoint"]))
        ),
    }


def geometry_summary(eq14_rows: dict[str, dict]) -> dict:
    keys = sorted(
        {
            key
            for row in eq14_rows.values()
            for key in row.get("dnf_geometry", {})
        }
    )
    output = {}
    for key in keys:
        values = np.asarray(
            [
                float(row["dnf_geometry"][key])
                for row in eq14_rows.values()
                if key in row.get("dnf_geometry", {})
            ],
            dtype=np.float64,
        )
        if values.size != len(eq14_rows) or not np.isfinite(values).all():
            raise ValueError(f"incomplete DNF geometry metric {key}")
        output[key] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    if not output:
        raise ValueError("deployment Eq.14 rows contain no DNF geometry")
    return output


def build_comparison(
    *,
    standard_dir: Path,
    dnf_dir: Path,
    contract: dict,
    contract_sha256: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    standard_metadata = read_json(standard_dir / "metadata.json")
    dnf_metadata = read_json(dnf_dir / "metadata.json")
    standard_summary = read_json(standard_dir / "train_summary.json")
    dnf_summary = read_json(dnf_dir / "train_summary.json")
    validate_arm(
        standard_metadata,
        standard_summary,
        mode="standard",
        contract=contract,
        contract_sha256=contract_sha256,
    )
    validate_arm(
        dnf_metadata,
        dnf_summary,
        mode="dnf",
        contract=contract,
        contract_sha256=contract_sha256,
    )
    pair_integrity = validate_pair_integrity(
        standard_dir,
        dnf_dir,
        standard_metadata,
        dnf_metadata,
        standard_summary,
        dnf_summary,
    )

    standard_index = index_rows(
        standard_dir / "validation_per_sample.jsonl"
    )
    dnf_index = index_rows(dnf_dir / "validation_per_sample.jsonl")
    rows_by_view = {}
    for view in sorted(REQUIRED_VIEWS):
        standard = output_rows(
            standard_index,
            view=view,
            output_name="standard",
        )
        eq14 = output_rows(dnf_index, view=view, output_name="eq14")
        speech_head = output_rows(
            dnf_index,
            view=view,
            output_name="speech_head",
        )
        validate_row_contract(standard, eq14, speech_head)
        validate_route_coverage(standard, contract)
        rows_by_view[view] = {
            "standard": standard,
            "eq14": eq14,
            "speech_head": speech_head,
        }

    deployment = rows_by_view[DEPLOYMENT_VIEW]
    identity = rows_by_view[IDENTITY_VIEW]
    comparisons = {
        "deployment": {
            "eq14_minus_speech_head_si_sdri_db": paired_difference(
                deployment["eq14"],
                deployment["speech_head"],
                "si_sdri_db",
                seed=seed,
                samples=bootstrap_samples,
            ),
            "eq14_minus_standard_si_sdri_db": paired_difference(
                deployment["eq14"],
                deployment["standard"],
                "si_sdri_db",
                seed=seed + 1,
                samples=bootstrap_samples,
            ),
            "eq14_minus_standard_sdri_db": paired_difference(
                deployment["eq14"],
                deployment["standard"],
                "sdri_db",
                seed=seed + 2,
                samples=bootstrap_samples,
            ),
            "route_eq14_minus_speech_head_si_sdri_db": paired_slices(
                deployment["eq14"],
                deployment["speech_head"],
                field="route",
                metric="si_sdri_db",
                seed=seed + 10,
                samples=bootstrap_samples,
            ),
            "route_eq14_minus_standard_si_sdri_db": paired_slices(
                deployment["eq14"],
                deployment["standard"],
                field="route",
                metric="si_sdri_db",
                seed=seed + 20,
                samples=bootstrap_samples,
            ),
            "family_eq14_minus_standard_si_sdri_db": paired_slices(
                deployment["eq14"],
                deployment["standard"],
                field="noise_family",
                metric="si_sdri_db",
                seed=seed + 30,
                samples=bootstrap_samples,
            ),
            "snr_eq14_minus_standard_si_sdri_db": paired_slices(
                deployment["eq14"],
                deployment["standard"],
                field="target_snr_db",
                metric="si_sdri_db",
                seed=seed + 40,
                samples=bootstrap_samples,
            ),
        },
        "identity": {
            "eq14_minus_standard_si_sdr_db": paired_difference(
                identity["eq14"],
                identity["standard"],
                "si_sdr_db",
                seed=seed + 50,
                samples=bootstrap_samples,
            ),
            "eq14_minus_speech_head_si_sdr_db": paired_difference(
                identity["eq14"],
                identity["speech_head"],
                "si_sdr_db",
                seed=seed + 51,
                samples=bootstrap_samples,
            ),
        },
    }
    outputs = {
        view: {
            output: summarize_output(rows)
            for output, rows in view_rows.items()
        }
        for view, view_rows in rows_by_view.items()
    }
    slices = {
        view: {
            output: summarize_slices(rows)
            for output, rows in view_rows.items()
        }
        for view, view_rows in rows_by_view.items()
    }
    evaluation_gates = contract["evaluation_gates"]
    mechanism = comparisons["deployment"][
        "eq14_minus_speech_head_si_sdri_db"
    ]
    ab = comparisons["deployment"]["eq14_minus_standard_si_sdri_db"]
    scale = comparisons["deployment"]["eq14_minus_standard_sdri_db"]
    eq14 = outputs[DEPLOYMENT_VIEW]["eq14"]
    eq14_gain = eq14["active_gain_db_to_clean"]
    weak_gain = slices[DEPLOYMENT_VIEW]["eq14"]["route"]["clean_weak"][
        "active_gain_db_to_input"
    ]
    identity_eq14 = outputs[IDENTITY_VIEW]["eq14"]
    identity_gain = identity_eq14["active_gain_db_to_input"]
    route_mechanism = comparisons["deployment"][
        "route_eq14_minus_speech_head_si_sdri_db"
    ]
    route_ab = comparisons["deployment"][
        "route_eq14_minus_standard_si_sdri_db"
    ]
    gates = {
        "eq14_beats_speech_head": (
            mechanism["mean"]
            >= float(
                evaluation_gates[
                    "eq14_minus_speech_head_mean_si_sdri_db"
                ]
            )
            and mechanism["bootstrap_ci95_low"]
            > float(
                evaluation_gates[
                    "paired_bootstrap_ci95_low_must_exceed_db"
                ]
            )
        ),
        "dnf_beats_standard": (
            ab["mean"]
            >= float(
                evaluation_gates[
                    "eq14_minus_standard_mean_si_sdri_db"
                ]
            )
            and ab["bootstrap_ci95_low"]
            > float(
                evaluation_gates[
                    "paired_bootstrap_ci95_low_must_exceed_db"
                ]
            )
        ),
        "scale_dependent_sdr_not_worse": scale["mean"] >= 0.0,
        "eq14_mean_scale_dependent_sdri_nonnegative": (
            eq14["sdri_db"]["mean"] >= 0.0
        ),
        "eq14_active_gain_median_within_1db": (
            abs(eq14_gain["p50"])
            <= float(
                evaluation_gates[
                    "eq14_active_gain_median_abs_max_db"
                ]
            )
        ),
        "eq14_active_gain_central_90pct_within_3db": (
            eq14_gain["p05"]
            >= float(
                evaluation_gates["eq14_active_gain_p05_min_db"]
            )
            and eq14_gain["p95"]
            <= float(
                evaluation_gates["eq14_active_gain_p95_max_db"]
            )
        ),
        "weak_clean_eq14_input_gain_preserved": (
            abs(weak_gain["p50"])
            <= float(
                evaluation_gates[
                    "weak_clean_eq14_active_gain_to_input_median_abs_max_db"
                ]
            )
            and weak_gain["p05"]
            >= float(
                evaluation_gates[
                    "weak_clean_eq14_active_gain_to_input_p05_min_db"
                ]
            )
            and weak_gain["p95"]
            <= float(
                evaluation_gates[
                    "weak_clean_eq14_active_gain_to_input_p95_max_db"
                ]
            )
        ),
        "eq14_not_worse_than_speech_head_on_any_route": all(
            summary["mean"] >= 0.0 for summary in route_mechanism.values()
        ),
        "eq14_noninferior_to_standard_on_every_route": all(
            summary["mean"]
            >= float(
                evaluation_gates[
                    "per_route_eq14_minus_standard_mean_si_sdri_min_db"
                ]
            )
            for summary in route_ab.values()
        ),
        "eq14_no_route_has_negative_mean_si_sdri": all(
            metrics["si_sdri_db"]["mean"] >= 0.0
            for metrics in slices[DEPLOYMENT_VIEW]["eq14"]["route"].values()
        ),
        "identity_eq14_absolute_si_sdr": (
            identity_eq14["si_sdr_db"]["mean"]
            >= float(
                evaluation_gates[
                    "identity_eq14_mean_si_sdr_min_db"
                ]
            )
        ),
        "identity_eq14_noninferior_to_standard": (
            comparisons["identity"]["eq14_minus_standard_si_sdr_db"][
                "mean"
            ]
            >= float(
                evaluation_gates[
                    "identity_eq14_minus_standard_mean_si_sdr_min_db"
                ]
            )
        ),
        "identity_eq14_gain_preserved": (
            abs(identity_gain["p50"])
            <= float(
                evaluation_gates[
                    "identity_eq14_active_gain_to_input_median_abs_max_db"
                ]
            )
            and identity_gain["p05"]
            >= float(
                evaluation_gates[
                    "identity_eq14_active_gain_to_input_p05_min_db"
                ]
            )
            and identity_gain["p95"]
            <= float(
                evaluation_gates[
                    "identity_eq14_active_gain_to_input_p95_max_db"
                ]
            )
        ),
        "all_deployment_and_identity_eq14_rows_valid": True,
    }
    controlled_gate_pass = all(gates.values())
    loss_variant = str(standard_metadata["loss_variant"])
    return {
        "schema_version": "dnf-phase-a-pair-comparison-v3",
        "standard_dir": str(standard_dir.resolve()),
        "dnf_dir": str(dnf_dir.resolve()),
        "loss_variant": loss_variant,
        "active_log_rms_weight": standard_metadata[
            "active_log_rms_weight"
        ],
        "uid_count_per_view": len(deployment["standard"]),
        "pair_integrity": pair_integrity,
        "model_parameter_counts": {
            "standard": standard_metadata["model_parameter_count"],
            "dnf": dnf_metadata["model_parameter_count"],
            "dnf_minus_standard": (
                int(dnf_metadata["model_parameter_count"])
                - int(standard_metadata["model_parameter_count"])
            ),
        },
        "outputs": outputs,
        "slices": slices,
        "comparisons": comparisons,
        "dnf_deployment_geometry": geometry_summary(
            deployment["eq14"]
        ),
        "gates": gates,
        "controlled_gate_pass": controlled_gate_pass,
        "paper_exact_dnf_efficacy_screen_pass": (
            loss_variant == "paper_exact" and controlled_gate_pass
        ),
        "matched_scale_gain_repair_screen_pass": (
            loss_variant == "matched_scale" and controlled_gate_pass
        ),
        "manual_blind_listening_required": True,
        "replication_required_after_first_seed_pass": True,
        "claim_limit": (
            "A passing paper_exact pair is a one-seed, 2000-step screen of "
            "the complete DNF package, not a general efficacy claim. DNF has "
            "a larger two-head model; Eq.14-versus-speech-head isolates the "
            "projection output but not the training losses. A matched_scale "
            "pair is a separate gain-repair ablation."
        ),
    }


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    payload = build_comparison(
        standard_dir=args.standard_dir,
        dnf_dir=args.dnf_dir,
        contract=contract,
        contract_sha256=sha256_file(args.contract),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if args.require_pass and not payload["controlled_gate_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
