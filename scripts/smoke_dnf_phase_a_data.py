"""Decode and validate frozen Phase-A tuples without constructing a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dataloaders.dnf_controlled_phase_a import (
    PhaseAControlledStreamDataset,
    phase_a_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Phase-A manifests.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=40)
    parser.add_argument("--valid-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def inspect(
    manifest: Path,
    *,
    split: str,
    sample_limit: int,
    seed: int,
) -> dict:
    dataset = PhaseAControlledStreamDataset(
        None,
        manifest,
        split=split,
        samples_per_epoch=sample_limit,
        target_sample_rate=16000,
        cut_duration=1.0,
        seed=seed,
        expose_clean_for_eval=True,
    )
    routes: Counter = Counter()
    sources: Counter = Counter()
    correlation_maxima: Counter = Counter()
    peak_max = 0.0
    additive_error_max = 0.0
    uids = set()
    clean_speech = set()
    noisy_speech = set()
    noise_pairing_policies = set()
    speech_partition_policies = set()
    same_family_pairs = 0
    approximate_orthogonality_warnings = 0
    deployment_snr_error_max_db = 0.0
    noise_energy_ratio_error_max = 0.0
    training_reconstruction_error_max = 0.0
    training_snr_delta_db = []
    block = []
    for index in range(len(dataset)):
        item = dataset[index]
        if item["uid"] in uids:
            raise ValueError(f"duplicate tuple UID: {item['uid']}")
        uids.add(item["uid"])
        routes[item["route"]] += 1
        sources[str(item["info"]["speech"]["dataset"])] += 1
        info = item["info"]
        noise_pairing_policies.add(info["noise_pairing_policy"])
        speech_partition_policies.add(info["speech_partition_policy"])
        if info["noise1"]["family"] == info["noise2"]["family"]:
            same_family_pairs += 1
        speech_id = json.dumps(info["speech"], sort_keys=True)
        if item["route"] == "noisy":
            noisy_speech.add(speech_id)
        else:
            clean_speech.add(speech_id)
        diagnostics = item["info"]["mixture_diagnostics"]
        clean = item["eval_clean_speech"].astype(np.float64)
        deployment = item["eval_model_input"].astype(np.float64)
        training_input = item["model_input"].astype(np.float64)
        noise1 = deployment - clean
        noise2 = training_input - deployment
        clean_energy = float(np.square(clean).sum())
        noise1_energy = float(np.square(noise1).sum())
        noise2_energy = float(np.square(noise2).sum())
        measured_deployment_snr = 10.0 * np.log10(
            clean_energy / noise1_energy
        )
        measured_training_snr = 10.0 * np.log10(
            clean_energy / float(np.square(noise1 + noise2).sum())
        )
        deployment_snr_error_max_db = max(
            deployment_snr_error_max_db,
            abs(measured_deployment_snr - float(info["target_snr_db"])),
        )
        noise_energy_ratio_error_max = max(
            noise_energy_ratio_error_max,
            abs(noise2_energy / noise1_energy - 1.0),
        )
        training_reconstruction_error_max = max(
            training_reconstruction_error_max,
            float(np.max(np.abs(training_input - deployment - noise2))),
        )
        training_snr_delta_db.append(
            measured_training_snr - measured_deployment_snr
        )
        peak_max = max(peak_max, float(diagnostics["peak_after_common_gain"]))
        additive_error_max = max(
            additive_error_max,
            float(diagnostics["max_additive_error"]),
        )
        for name, value in diagnostics["absolute_correlations"].items():
            correlation_maxima[name] = max(
                float(correlation_maxima[name]),
                float(value),
            )
        if diagnostics["absolute_correlations"]["noise1_noise2"] > 0.2:
            approximate_orthogonality_warnings += 1
        block.append(item)
        if len(block) == 20:
            batch = phase_a_collate(block)
            if batch["clean_indices"].numel() != 5:
                raise ValueError("20-row block does not contain five clean routes")
            if batch["noisy_indices"].numel() != 15:
                raise ValueError("20-row block does not contain 15 noisy routes")
            if not np.isfinite(batch["model_input_wav"].numpy()).all():
                raise ValueError("non-finite model input")
            if not np.isfinite(batch["eval_model_input_wav"].numpy()).all():
                raise ValueError("non-finite deployment model input")
            block = []
    if block:
        raise ValueError("sample limit must contain complete 20-row blocks")
    if set(sources) - {"EARS", "VCTK"}:
        raise ValueError(f"non-strict source leaked into smoke: {sources}")
    if clean_speech & noisy_speech:
        raise ValueError("clean/noisy speech pools overlap in the smoke slice")
    if noise_pairing_policies != {"same_family_iid"}:
        raise ValueError(
            f"smoke is not paper-mechanism pairing: {noise_pairing_policies}"
        )
    if speech_partition_policies != {"disjoint_item_pools"}:
        raise ValueError(
            f"smoke is not disjoint speech partitioning: "
            f"{speech_partition_policies}"
        )
    if same_family_pairs != len(dataset):
        raise ValueError("paper-mechanism smoke contains cross-family n1/n2")
    if deployment_snr_error_max_db > 1.0e-4:
        raise ValueError(
            "single-noise deployment SNR differs from the frozen target"
        )
    if noise_energy_ratio_error_max > 1.0e-4:
        raise ValueError("n1 and n2 do not have equal energy")
    if training_reconstruction_error_max > 1.0e-7:
        raise ValueError("x != (s+n1)+n2 in the smoke slice")
    return {
        "split": split,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": dataset.manifest_sha256,
        "sample_count": len(dataset),
        "uid_count": len(uids),
        "routes": dict(sorted(routes.items())),
        "sources": dict(sorted(sources.items())),
        "noise_pairing_policies": sorted(noise_pairing_policies),
        "speech_partition_policies": sorted(speech_partition_policies),
        "clean_noisy_speech_overlap_count": len(clean_speech & noisy_speech),
        "same_family_pair_count": same_family_pairs,
        "correlation_maxima": dict(sorted(correlation_maxima.items())),
        "approximate_orthogonality_threshold": 0.2,
        "approximate_orthogonality_warning_count": (
            approximate_orthogonality_warnings
        ),
        "peak_max": peak_max,
        "additive_error_max": additive_error_max,
        "deployment_snr_error_max_db": deployment_snr_error_max_db,
        "noise_energy_ratio_error_max": noise_energy_ratio_error_max,
        "training_reconstruction_error_max": (
            training_reconstruction_error_max
        ),
        "training_minus_deployment_snr_db": {
            "mean": float(np.mean(training_snr_delta_db)),
            "min": float(np.min(training_snr_delta_db)),
            "max": float(np.max(training_snr_delta_db)),
            "independent_equal_energy_expectation_db": (
                -10.0 * np.log10(2.0)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.train_samples % 20 or args.valid_samples % 20:
        raise ValueError("smoke sample limits must be multiples of 20")
    payload = {
        "schema_version": "dnf-phase-a-data-smoke-v2",
        "train": inspect(
            args.train_manifest,
            split="train",
            sample_limit=args.train_samples,
            seed=args.seed,
        ),
        "valid": inspect(
            args.valid_manifest,
            split="valid",
            sample_limit=args.valid_samples,
            seed=args.seed,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
