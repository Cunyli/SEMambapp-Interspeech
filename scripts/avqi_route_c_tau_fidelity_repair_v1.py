"""Controlled, authorized TAU fidelity repairs; old results stay immutable."""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import copy
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataloaders.legacy_online_degradation import _rir_direct_path_delay
from model.avqi_training_reference import (
    five_component_clean_prediction, training_row_with_same_formula_reference,
)
from model.fidelity_gradient_budget import bound_auxiliary_parameter_gradients
from model.sv_guardrail_v2 import align_waveform_pair, best_normalized_cross_correlation_lag
from scripts import avqi_route_c_tau_amplitude_training_v1 as prior
from scripts.evaluate_avqi_shimmer_fresh_panel import read_fixed_recipes, recipe_wds_row
from scripts.prepare_avqi_component_expanded_data import WdsReader, crop_or_tile, match_length, stable_seed


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "avqi-route-c-tau-fidelity-repair-v1"
RIR_REFERENCE = "rir_argmax_abs_shifted_anechoic_full_reference"


def fixed_input_lag(target: torch.Tensor, degraded: torch.Tensor, maximum_lag: int = 1600) -> int:
    """Freeze timing from input and target, never from the model's output."""
    if torch.equal(target, degraded):
        return 0
    lag = best_normalized_cross_correlation_lag(
        target, degraded, max_lag_samples=maximum_lag)
    if abs(lag) == maximum_lag:
        raise ValueError("input alignment reached the frozen search boundary")
    return lag


def shifted_dry_reference(target: torch.Tensor, lag: int) -> torch.Tensor:
    """Match the original shifted-anechoic target, including its zero prefix."""
    if target.ndim != 1 or not 0 <= lag < target.numel():
        raise ValueError("RIR delay must preserve a nonempty mono reference")
    return F.pad(target, (lag, 0))[:target.numel()] if lag else target


def reference_pair(target: torch.Tensor, estimate: torch.Tensor, lag: int,
                   method: str) -> tuple[torch.Tensor, torch.Tensor]:
    if method == RIR_REFERENCE:
        reference = shifted_dry_reference(target, lag)
        n = min(reference.numel(), estimate.numel())
        return reference[:n], estimate[:n]
    return align_waveform_pair(target, estimate, lag)


def fidelity_for_arm(y: torch.Tensor, target: torch.Tensor, lag: int, cfg: dict,
                     protocol: dict, arm: str):
    effective_lag = lag if protocol["arms"][arm]["alignment"] else 0
    aligned_target, aligned_y = reference_pair(target, y, effective_lag, protocol["alignment"]["method"])
    loss_protocol = {"loss": {
        "time_l1_weight": protocol["training"]["time_l1_weight"],
        "compressed_magnitude_mse_weight": protocol["training"]["compressed_magnitude_mse_weight"],
    }}
    loss, terms = prior.fidelity_loss(aligned_y, aligned_target, cfg, loss_protocol)
    return loss, dict(terms, alignment_lag_samples=effective_lag,
                      fidelity_samples=aligned_y.numel())


def paired_metrics(target, before, after, lag, method="input_correlation"):
    n = min(target.numel(), before.numel(), after.numel())
    target, before, after = target[:n], before[:n], after[:n]
    aligned_target, aligned_before = reference_pair(target, before, lag, method)
    _, aligned_after = reference_pair(target, after, lag, method)
    relative_gain = float(torch.dot(before.double(), after.double()) / before.double().square().sum())
    denominator = aligned_target.double().square().sum().clamp_min(1e-12)
    coherent_before = float(torch.dot(aligned_target.double(), aligned_before.double()) / denominator)
    coherent_after = float(torch.dot(aligned_target.double(), aligned_after.double()) / denominator)
    raw_before, raw_after = prior.snr_db(target, before), prior.snr_db(target, after)
    snr_before = prior.snr_db(aligned_target, aligned_before)
    snr_after = prior.snr_db(aligned_target, aligned_after)
    return dict(
        raw_snr_before_db=raw_before, raw_snr_after_db=raw_after,
        raw_snr_change_db=raw_after - raw_before,
        aligned_snr_before_db=snr_before, aligned_snr_after_db=snr_after,
        aligned_snr_change_db=snr_after - snr_before,
        aligned_si_sdr_before_db=prior.si_sdr_db(aligned_target, aligned_before),
        aligned_si_sdr_after_db=prior.si_sdr_db(aligned_target, aligned_after),
        aligned_l1_before=float((aligned_before - aligned_target).abs().mean()),
        aligned_l1_after=float((aligned_after - aligned_target).abs().mean()),
        relative_gain=relative_gain,
        relative_gain_db=float(20 * np.log10(max(abs(relative_gain), 1e-12))),
        target_coherent_gain_before=coherent_before, target_coherent_gain_after=coherent_after,
        aligned_samples=aligned_after.numel(), fixed_input_lag_samples=lag,
        output_peak=float(after.abs().max()), clip_fraction=float((after.abs() >= 1).float().mean()),
    )


def summaries(metrics):
    result = {}
    for role in sorted({r["role"] for r in metrics}):
        group = [r["metrics"] for r in metrics if r["role"] == role]
        result[role] = dict(n=len(group), **{
            key + "_median": statistics.median(r[key] for r in group)
            for key in ("raw_snr_change_db", "aligned_snr_change_db", "relative_gain_db",
                        "aligned_l1_before", "aligned_l1_after",
                        "target_coherent_gain_before", "target_coherent_gain_after")},
            aligned_snr_worst_change_db=min(r["aligned_snr_change_db"] for r in group),
            clipped_rows=sum(r["clip_fraction"] > 0 for r in group))
    return result


def receipt(run, stage, **extra):
    artifacts = [prior.binding(p) for directory in ("outputs", "checkpoints")
                 for p in sorted((run / directory).rglob("*")) if p.is_file()]
    result = dict(schema_version=SCHEMA, stage=stage, status="COMPLETED",
                  slurm_job_id=os.environ["SLURM_JOB_ID"],
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  protocol_sha256=os.environ["PROTOCOL_SHA256"], artifacts=artifacts,
                  scientific_promotion=False, independent_validation=False,
                  svd_used_for_new_testing=False, **extra)
    prior.write_json(run / "outputs/completion_receipt.json", result)


def load_frozen_simulator(contract):
    source = prior.verified(contract["inputs"]["simulation_source"])
    sys.path.insert(0, str(source.parent))
    # This optional simulator and its audio dependencies exist only on Triton;
    # keep local unit tests independent from that external installation.
    from simulate_degradation import apply_degradation_with_wind

    if Path(apply_degradation_with_wind.__code__.co_filename).resolve() != source.resolve():
        raise ValueError("simulator resolved outside the verified source")
    config = yaml.safe_load(prior.verified(contract["inputs"]["simulation_config"]).read_text())
    config["stft_cfg"]["sampling_rate"] = 16000
    return apply_degradation_with_wind, config


def freeze_alignment(run, protocol, rows, baseline):
    entries, measurements = [], []
    method = protocol["alignment"]["method"]
    reader_context = nullcontext(None)
    recipes = None
    simulator, simulation_config = None, None
    if method == RIR_REFERENCE:
        old_protocol = prior.read_json(ROOT / protocol["previous_protocol"])
        contract = prior.read_json(prior.verified(old_protocol["historical_contract"]))
        recipes = read_fixed_recipes(prior.verified(contract["inputs"]["fixed_recipes"]))
        simulator, simulation_config = load_frozen_simulator(contract)
        reader_context = closing(WdsReader())
    with reader_context as reader:
        for row in rows:
            target, degraded = prior.audio(row["target"]), prior.audio(row["degraded"])
            rir_evidence = None
            if method == RIR_REFERENCE:
                lag = 0
                if row["condition"] != "clean":
                    recipe = recipes[row["recipe_index"]]
                    if recipe["uid"] != row["recipe_uid"] or recipe["target_sample_rate"] != 16000:
                        raise ValueError("RIR recipe binding differs")
                    rir = reader.read(recipe_wds_row(recipe, "rir"))
                    lag = _rir_direct_path_delay(rir, mode="argmax_abs")
                    rir_evidence = dict(recipe_uid=recipe["uid"], source=recipe["rir"],
                        float32_sha256=prior.waveform_float32_sha256(rir.reshape(-1)),
                        samples=int(rir.size), peak_value=float(rir.reshape(-1)[lag]))
                    if row["training_role"] != "evaluation":
                        seed = stable_seed(contract["selection"]["seed"], contract["selection"]["salt"],
                                           row["case_id"], recipe["uid"])
                        noise, start = crop_or_tile(reader.read(recipe_wds_row(recipe, "noise")),
                                                   target.numel(), random.Random(seed))
                        returned_clean, rebuilt = simulator(copy.deepcopy(simulation_config),
                            target.numpy()[None], noise, rir, None, {"snr": row["snr_db"]},
                            ["reverb", "noise"], seed=seed)
                        rebuilt = match_length(rebuilt, target.numel())[0]
                        np.testing.assert_array_equal(returned_clean[0], target.numpy())
                        np.testing.assert_allclose(rebuilt, degraded.numpy(), rtol=1e-5, atol=1e-7)
                        rir_evidence["degraded_replay_max_sample_error"] = float(np.max(np.abs(rebuilt - degraded.numpy())))
                        rir_evidence["noise_start_sample"] = start
                shifted_dry_reference(target, lag)  # Validate physical reference support.
            else:
                lag = fixed_input_lag(target, degraded, protocol["alignment"]["maximum_lag_samples"])
            entries.append(dict(case_id=row["case_id"], role=row["training_role"],
                                target=row["target"], degraded=row["degraded"],
                                lag_samples=lag, lag_ms=lag / 16, candidate_used_for_lag=False,
                                rir_evidence=rir_evidence))
            if row["training_role"] != "evaluation":
                before = prior.audio(baseline[row["case_id"]])
                measurements.append(dict(case_id=row["case_id"], role=row["training_role"],
                                         metrics=paired_metrics(target, before, before, lag, method)))
    prior.write_json(run / "outputs/fixed_alignment.json", dict(
        schema_version=SCHEMA, alignment_rule=protocol["alignment"], rows=entries,
        source_safety_report_sha256=protocol["safety_report_sha256"],
        lag_sealed_before_training=True, candidate_outcomes_used=False))
    prior.write_json(run / "outputs/alignment_audit.json", dict(rows=measurements,
                     summary=summaries(measurements), generator_optimizer_steps=0))
    receipt(run, "alignment", generator_optimizer_steps=0)
    print(json.dumps(dict(stage="alignment", development=measurements), allow_nan=False), flush=True)


def load_alignment(path, expected_sha256, protocol, rows):
    if prior.binding(path)["sha256"] != expected_sha256:
        raise ValueError("fixed alignment hash differs")
    document = prior.read_json(path)
    if (document["alignment_rule"] != protocol["alignment"]
            or document["source_safety_report_sha256"] != protocol["safety_report_sha256"]
            or not document["lag_sealed_before_training"] or document["candidate_outcomes_used"]):
        raise ValueError("fixed alignment definition differs")
    entries = {r["case_id"]: r for r in document["rows"]}
    for row in rows:
        entry = entries[row["case_id"]]
        if entry["target"] != row["target"] or entry["degraded"] != row["degraded"]:
            raise ValueError("alignment bound to a different waveform")
    return {key: value["lag_samples"] for key, value in entries.items()}


def evaluate_development(run, model, rows, baseline, lags, cfg, device, protocol):
    model.eval()
    measured = []
    (run / "outputs/after").mkdir()
    for row in rows:
        if row["training_role"] == "evaluation":
            continue
        with torch.inference_mode():
            y = prior.enhance_waveform(model, prior.audio(row["degraded"]).to(device), cfg)[0].cpu()
        if not torch.isfinite(y).all() or y.abs().max() > 0.95:
            raise ValueError("invalid development waveform")
        after = prior.write_audio(run / "outputs/after" / (row["case_id"] + ".wav"), y.numpy())
        measured.append(dict(case_id=row["case_id"], role=row["training_role"],
                             before=baseline[row["case_id"]], after=after,
                             metrics=paired_metrics(prior.audio(row["target"]),
                                      prior.audio(baseline[row["case_id"]]), y, lags[row["case_id"]],
                                      protocol["alignment"]["method"])))
    result = dict(rows=measured, summaries=summaries(measured), evaluation_reserve_opened=False,
                  gain_matching_used=False, lag_sealed_before_training=True)
    prior.write_json(run / "outputs/development_report.json", result)
    return result


def train(run, protocol, old_protocol, rows, baseline, lags, arm, device):
    old, paths, exact, cfg = prior.context(old_protocol)
    train_rows = [r for r in rows if r["training_role"] == "train"]
    prior.set_model_seed(protocol["training"]["seed"])
    model = prior.load_generator(cfg, paths["generator_checkpoint"], device).train()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t = protocol["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=0)
    use_six = protocol["arms"][arm]["six_component_fusion"]
    scorer, runtime, weights = None, None, None
    proxy_rows = {}
    worker_context = nullcontext(None)
    if use_six:
        scorer = prior.scorer_bundle(old, paths)
        runtime = prior.load_runtime_module(paths["candidate_e_runtime_client"])
        preflight = ROOT / protocol["original_preflight"]
        weights = prior.read_json(preflight.parent / "frozen_weights.json")["weights"]
        worker_context = runtime.ExactShimmerTopologyWorker(
            Path(old["exact"]["python"]).resolve(), prior.EXACT_PCM_WORKER,
            Path(old["exact"]["root"]), exact["avqi_code_tree_sha256"])
        if protocol["arms"][arm].get("same_formula_clean_reference", False):
            references = []
            for row in train_rows:
                clean = five_component_clean_prediction(scorer, prior.audio(row["target"]), row["view"])
                loss_row = training_row_with_same_formula_reference(row, clean)
                proxy_rows[row["case_id"]] = loss_row
                references.append(dict(case_id=row["case_id"], role="train", waveform=row["target"],
                    clean_proxy=clean, loss_target_components=loss_row["target_components"],
                    exact_target_components=row["target_components"]))
            prior.write_json(run / "outputs/frozen_proxy_references.json", dict(rows=references,
                sealed_before_optimizer_step=True, candidate_outcomes_used_for_references=False,
                exact_evaluation_targets_changed=False, calibration_weights_changed=False))
    checkpoints = [prior.save_checkpoint(run, model, optimizer, 0, protocol, cfg, initial)]
    with worker_context as worker:
        if use_six:
            worker.warmup()
        # Initialize stochastic training identically across the controlled arms.
        prior.set_model_seed(t["seed"])
        for step in range(1, t["maximum_optimizer_steps"] + 1):
            started = time.monotonic()
            row = train_rows[(step - 1) % len(train_rows)]
            optimizer.zero_grad(set_to_none=True)
            y = prior.enhance_waveform(model, prior.audio(row["degraded"]).to(device), cfg)[0]
            if not torch.isfinite(y).all() or y.abs().max() > 0.95:
                raise ValueError("invalid training output")
            fidelity, terms = fidelity_for_arm(y, prior.audio(row["target"]).to(device),
                                               lags[row["case_id"]], cfg, protocol, arm)
            fidelity_gradient = torch.autograd.grad(fidelity, y, retain_graph=True)[0]
            fusion = None
            record = None
            auxiliary_balance = None
            if use_six:
                loss_row = proxy_rows.get(row["case_id"], row)
                record, gradients = prior.measure(loss_row, y, scorer, runtime, worker,
                                                  run / "outputs", f"step{step:06d}")
                joint, fusion, gates = prior.protocol_fusion(old_protocol, record, gradients, weights)
                if not all(gates.values()):
                    prior.write_json(run / "outputs/failed_gradient_gate.json", dict(step=step, gates=gates, fusion=fusion))
                    raise ValueError("unchanged six-component gradient gate failed")
                ratio = protocol["arms"][arm].get("parameter_auxiliary_norm_ratio_cap")
                if ratio is None:
                    torch.autograd.backward((y, fidelity), (joint.to(y), None))
                else:
                    parameters = tuple(model.parameters())
                    primary = torch.autograd.grad(fidelity, parameters, retain_graph=True)
                    auxiliary = torch.autograd.grad(y, parameters, grad_outputs=joint.to(y))
                    combined, auxiliary_balance = bound_auxiliary_parameter_gradients(primary, auxiliary, ratio)
                    for parameter, gradient in zip(parameters, combined):
                        parameter.grad = gradient
                    del primary, auxiliary, combined
            else:
                fidelity.backward()
            parameter_gradient = prior.finite_parameter_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            log = dict(step=step, case_id=row["case_id"], role="train", arm=arm,
                       fidelity=terms, fidelity_waveform_gradient_norm=float(fidelity_gradient.double().norm()),
                       component_measurements=(record["components"] if record is not None else None),
                       gradient_target_source=(proxy_rows.get(row["case_id"], {}).get("gradient_target_source", "exact_clean_target")),
                       fusion=fusion, auxiliary_balance=auxiliary_balance, parameter_gradient=parameter_gradient,
                       output_peak=float(y.detach().abs().max()), seconds=time.monotonic() - started)
            with (run / "outputs/training_steps.jsonl").open("a") as handle:
                handle.write(json.dumps(log, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if step % t["checkpoint_interval"] == 0:
                checkpoints.append(prior.save_checkpoint(run, model, optimizer, step, protocol, cfg, initial))
            print(f"{arm}: optimizer_step={step}/128 loss={float(fidelity.detach()):.7f}", flush=True)
    restored = prior.load_generator(cfg, Path(checkpoints[-1]["path"]), device)
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise ValueError("saved model did not reload exactly")
    delta = prior.parameter_delta(model, initial)
    if delta["changed_tensors"] == 0:
        raise ValueError("optimizer did not change the generator")
    prior.write_json(run / "outputs/training_report.json", dict(
        arm=arm, optimizer_steps=t["maximum_optimizer_steps"], heldout_optimizer_steps=0,
        checkpoints=checkpoints, delta_from_initial=delta, checkpoint_reload_exact=True))
    result = evaluate_development(run, restored, rows, baseline, lags, cfg, device, protocol)
    receipt(run, "train", arm=arm, generator_optimizer_steps=t["maximum_optimizer_steps"],
            heldout_optimizer_steps=0, evaluation_reserve_opened=False)
    print(json.dumps(result["summaries"], allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("alignment", "train"), required=True)
    parser.add_argument("--arm", choices=("unaligned_fidelity", "aligned_fidelity", "aligned_joint", "aligned_joint_bounded", "aligned_joint_proxy_reference"))
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--alignment-sha256")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Slurm compute node required")
    protocol = prior.read_json(args.protocol)
    if protocol["schema_version"] not in tuple(SCHEMA.replace("v1", v) for v in ("v1", "v2", "v3", "v4")) or protocol["scientific_promotion"] is not False:
        raise ValueError("repair protocol differs")
    if prior.binding(args.protocol)["sha256"] != os.environ["PROTOCOL_SHA256"]:
        raise ValueError("repair protocol hash differs")
    if protocol["training"]["maximum_optimizer_steps"] != 128:
        raise ValueError("bounded training budget differs")
    old_protocol = prior.read_json(ROOT / protocol["previous_protocol"])
    prior.validate_protocol(old_protocol)
    safety_path = ROOT / protocol["safety_report"]
    if prior.binding(safety_path)["sha256"] != protocol["safety_report_sha256"]:
        raise ValueError("baseline safety binding differs")
    safety = prior.read_json(safety_path)
    rows = safety["source_rows"]
    prior.validate_roles(rows)
    baseline = {r["case_id"]: r["output"] for r in safety["rows"]}
    run = args.run_root
    (run / "outputs").mkdir()
    (run / "checkpoints").mkdir()
    if args.stage == "alignment":
        freeze_alignment(run, protocol, rows, baseline)
    else:
        if (not args.arm or not args.alignment or not args.alignment_sha256
                or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise ValueError("training requires one allocated GPU, a frozen alignment and an arm")
        lags = load_alignment(args.alignment, args.alignment_sha256, protocol, rows)
        train(run, protocol, old_protocol, rows, baseline, lags, args.arm, torch.device("cuda"))


if __name__ == "__main__":
    main()
