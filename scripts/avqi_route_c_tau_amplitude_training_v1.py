#!/usr/bin/env python3
"""Authorized bounded TAU training, with explicit amplitude and backward gates.

Historical diagnostic NO_GO receipts stay immutable. Completion here proves an
engineering training run, not independent validation or scientific promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c_candidate_e_scorer import load_route_c_candidate_e_six_scorer
from model.avqi_route_c_candidate_e import exact_numpy_highpass_pcm16
from model.avqi_route_c_gradient_fusion import fuse_tensor_gradients
from model.avqi_route_c_training_fusion import fuse_training_gradients
from model.stfts import mag_phase_stft
from model.waveform_output_safety import attenuate_output_peak
from scripts.avqi_route_c_tau_joint_diagnostic_v1 import (
    authority, baseline_length_certificate, binding, read_json, verified,
    write_audio, write_json,
)
from scripts.avqi_route_c_tau_joint_panel_v1 import full_gradient_gates, validate_reserved_rows
from scripts.evaluate_avqi_component_backprop import enhance_waveform, load_generator, set_model_seed
from scripts.evaluate_avqi_route_c_multicomponent_gradients import AuditCase
from scripts.evaluate_avqi_route_c_six_component_gradients import (
    TopologyAuditInput, calibration_inverse_gradient_weights, extract_waveform_measurement,
)
from scripts.evaluate_avqi_route_c_six_joint_exact_panel import run_exact
from scripts.evaluate_direct_avqi_waveform_optimization import (
    full_band_pathology_guardrails, si_sdr_db, snr_db, waveform_safety,
)
from scripts.prepare_avqi_route_c_six_component_topologies_v2 import (
    load_runtime_module, waveform_float32_sha256,
)
from utils import load_config

SCHEMA = "avqi-route-c-tau-amplitude-training-v1"
NAMES = tuple(AVQI_COMPONENT_NAMES)
EXACT_PCM_WORKER = Path(__file__).with_name("avqi_shimmer_exact_pcm_worker_v1.py")


def audio(value):
    path = verified(value)
    x, rate = sf.read(path, dtype="float32")
    if rate != 16000 or x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("invalid sealed mono 16 kHz audio: " + str(path))
    if value.get("float32_sha256") and waveform_float32_sha256(x) != value["float32_sha256"]:
        raise ValueError("sealed sample stream differs")
    return torch.from_numpy(x)


def validate_protocol(p):
    if p["schema_version"] not in (SCHEMA, SCHEMA.replace("v1", "v2")) or p["authorization"]["training_authorized"] is not True:
        raise ValueError("explicit bounded training authorization missing")
    t = p["training"]
    if (t["maximum_optimizer_steps"] != 128 or t["checkpoint_interval"] != 32
            or t["learning_rate"] != 1e-6 or t["batch_size"] != 1
            or t["seed"] != 20260908 or t["weight_decay"] != 0
            or p["signal_safety_cfg"]["output_peak_limit"] != 0.95
            or p["acceptance"]["scientific_promotion"] is not False):
        raise ValueError("frozen bounded training settings differ")


def dataset(p):
    docs = {k: read_json(verified(v)) for k, v in p["historical_inputs"].items()}
    sealed = docs["source_seal"]
    materialized = docs["gradient_materialized"]["rows"]
    reserved = sealed["joint_reserved_rows"]
    validate_reserved_rows(reserved, sealed["gradient_rows"])
    if [r["case_id"] for r in materialized] != [r["case_id"] for r in sealed["gradient_rows"]]:
        raise ValueError("gradient source order differs")
    rows = []
    for index, original in enumerate(materialized):
        row = copy.deepcopy(original)
        expected = "surrogate_calibration" if index < 4 else "surrogate_holdout"
        if row["split"] != expected or row["dataset"] != "TAU" or row["label"] != "patient":
            raise ValueError("training or validation role differs")
        row["training_role"] = "train" if index < 4 else "validation"
        rows.append(row)
    audited = {r["case_id"]: r for r in docs["joint_audit"]["rows"]}
    targets = {(r["speaker_id"], r["view"]): r for r in docs["joint_targets"]["rows"]}
    clean = {(r["speaker_id"], r["view"]): r["degraded"]
             for r in audited.values() if r["condition"] == "clean"}
    for original in reserved:
        row = copy.deepcopy(original)
        audit = audited[row["case_id"]]
        for key in ("speaker_id", "view", "condition", "split", "label"):
            if row[key] != audit[key]:
                raise ValueError("joint source identity differs")
        row.update(training_role="evaluation", degraded=audit["degraded"],
                   historical_base=audit["native_baseline"],
                   target=clean[(row["speaker_id"], row["view"])])
        row["target_components"] = (targets[(row["speaker_id"], row["view"])]["exact_components"]
                                    if row["label"] == "patient" else None)
        rows.append(row)
    validate_roles(rows)
    for row in rows:
        audio(row["degraded"])
        audio(row["target"])
    return rows


def validate_roles(rows):
    if len(rows) != 104 or len({r["case_id"] for r in rows}) != 104:
        raise ValueError("expected all 104 sealed cases")
    if Counter(r["training_role"] for r in rows) != {"train": 4, "validation": 4, "evaluation": 96}:
        raise ValueError("bounded training split differs")
    roles = {}
    for row in rows:
        if row["dataset"] != "TAU" or not row["historically_exact_opened"]:
            raise ValueError("TAU historical overlap disclosure missing")
        roles.setdefault(row["canonical_speaker_id"], set()).add(row["training_role"])
    if len(roles) != 20 or any(len(v) != 1 for v in roles.values()):
        raise ValueError("speaker overlap across training and evaluation")


def context(p):
    old = read_json(verified(p["historical_contract"]))
    paths = {k: verified(v) for k, v in old["inputs"].items()}
    exact = authority(old, paths)
    cfg = load_config(paths["generator_config"])
    cfg["signal_safety_cfg"] = p["signal_safety_cfg"]
    return old, paths, exact, cfg


def load_dependency(path, stage, protocol):
    r = read_json(path)
    if r["stage"] != stage or r["status"] != "COMPLETED":
        raise ValueError("successful prerequisite receipt missing: " + stage)
    if r["protocol_sha256"] != os.environ["PROTOCOL_SHA256"]:
        previous = protocol.get("previous_protocol")
        if stage != "safety" or not previous or r["protocol_sha256"] != previous["sha256"]:
            raise ValueError("prerequisite protocol differs")
        old = read_json(verified(previous))
        for key in ("historical_inputs", "historical_contract", "signal_safety_cfg", "training"):
            if old[key] != protocol[key]:
                raise ValueError("previous amplitude validation is incompatible")
    for b in r["artifacts"]:
        verified(b)
    return Path(r["run_root"]) / "outputs"


def save_stage(run, stage, p, **result):
    files = [binding(path) for folder in ("outputs", "checkpoints")
             for path in sorted((run / folder).rglob("*"))
             if path.is_file() and path.name != "completion_receipt.json"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True,
                          capture_output=True).stdout.strip()
    receipt = dict(schema_version=SCHEMA, stage=stage, status="COMPLETED",
                   run_root=str(run), slurm_job_id=os.environ["SLURM_JOB_ID"],
                   source_commit=head, protocol_sha256=os.environ["PROTOCOL_SHA256"],
                   scientific_promotion=False, independent_validation=False,
                   svd_used_for_new_testing=False, artifacts=files, **result)
    write_json(run / "outputs/completion_receipt.json", receipt)


def safety(run, p, rows, cfg, paths):
    model = load_generator(cfg, paths["generator_checkpoint"], torch.device("cuda"))
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg.pop("signal_safety_cfg")
    results = []
    (run / "outputs/baseline").mkdir()
    for index, row in enumerate(rows, 1):
        x = audio(row["degraded"]).cuda()
        with torch.no_grad():
            raw = enhance_waveform(model, x, raw_cfg)[0]
            safe, gain = attenuate_output_peak(raw, 0.95)
            live = enhance_waveform(model, x, cfg)[0]
        torch.testing.assert_close(live, safe, rtol=1e-4, atol=1e-5)
        length = baseline_length_certificate(x.numel(), safe.numel(), cfg["stft_cfg"]["hop_size"])
        if safe.abs().max() > 0.95 or not torch.isfinite(safe).all():
            raise ValueError("output safety failed: " + row["case_id"])
        out = write_audio(run / "outputs/baseline" / (row["case_id"] + ".wav"), safe.cpu().numpy())
        historical = audio(row.get("historical_base", row.get("base")))
        raw_difference = float((raw.cpu() - historical).abs().max())
        results.append(dict(case_id=row["case_id"], original_peak=float(raw.abs().max()),
                            corrected_peak=float(safe.abs().max()), gain=float(gain),
                            historical_raw_max_sample_difference=raw_difference,
                            length_certificate=length, output=out))
        print(f"amplitude_validation={index}/104 peak={float(safe.abs().max()):.7f}", flush=True)
    write_json(run / "outputs/safety_report.json", dict(
        schema_version=SCHEMA, rows=results, all_finite=True, all_peak_safe=True,
        changed_gain_rows=sum(r["gain"] < 1 for r in results),
        final_waveform_highpass=False, source_rows=rows))
    save_stage(run, "safety", p, generator_optimizer_steps=0, validated_waveforms=104)


def scorer_bundle(old, paths):
    names = ("cpps", "hnr", "shimmer_percent", "slope", "tilt")
    scorer = load_route_c_candidate_e_six_scorer(
        {n: paths[n + "_checkpoint"] for n in names},
        {n: old["inputs"][n + "_checkpoint"]["sha256"] for n in names},
    ).scorer.cuda().eval()
    if list(scorer.parameters()):
        raise ValueError("scorer must be frozen and parameter-free")
    for key in ("target_mean", "target_scale"):
        expected = torch.tensor([old["normalization"][key][n] for n in NAMES])
        if not torch.equal(getattr(scorer, key).cpu(), expected):
            raise ValueError("six-component normalization drift")
    return scorer


def protocol_fusion(p, record, gradients, weights):
    v2 = p["schema_version"].endswith("v2")
    joint, fusion = (fuse_training_gradients if v2 else fuse_tensor_gradients)(NAMES, gradients, weights)
    gates = full_gradient_gates(record, fusion)
    if v2:
        gates.pop("only_unique_dominant_component_attenuated")
        gates["all_six_contributions_positive_and_attenuation_only"] = all(
            0 < fusion["effective_weights"][n] <= weights[n] for n in NAMES)
    return joint, fusion, gates


def measure(row, waveform, scorer, runtime, worker, output, tag):
    values = waveform.detach().cpu().contiguous().numpy()
    case_id = row["case_id"] + ":" + tag
    topologies, runtime_ms, staging = worker.refresh_current_waveforms(
        [dict(id="topology:" + case_id, case_id=case_id, role="current_output_topology",
              path=row["degraded"]["path"], view=row["view"], score_components=False,
              exact_metric_topology=True, highpass_mode=runtime.NUMPY_HIGHPASS_MODE)],
        [values], highpass_mode=runtime.NUMPY_HIGHPASS_MODE)
    topology = topologies[0]
    if topology.get("highpass_pcm_transport_schema") != "exact-worker-current-pcm16-v1":
        raise ValueError("current Exact PCM transport is required")
    waveform_binding = write_audio(output / (tag + "_" + row["case_id"] + "_current.wav"), values)
    _, legacy_digest = exact_numpy_highpass_pcm16(
        waveform.detach().to(dtype=torch.float64),
        peak_scale_required=topology["timing_ms"]["highpass_peak_scaled"])
    write_json(output / (tag + "_" + row["case_id"] + "_topology.json"),
               dict(topology=topology, topology_sha256=runtime.topology_sha256(topology),
                    waveform=waveform_binding, runtime_ms=runtime_ms, staging=staging,
                    local_recomputation_pcm16_sha256=legacy_digest,
                    local_recomputation_matches_exact=legacy_digest == topology["highpass_pcm16_sha256"]))
    topo = TopologyAuditInput(case_id, topology, runtime.topology_sha256(topology),
                              waveform_float32_sha256(values))
    case = AuditCase(split=row["split"], speaker_id=row["canonical_speaker_id"],
                     sample_id=case_id, sample_group="patient_" + row["sex"],
                     view=row["view"], condition=row["condition"],
                     waveform_path=Path(row["degraded"]["path"]),
                     waveform_sha256=row["degraded"]["sha256"],
                     clean_target=torch.tensor([row["target_components"][n] for n in NAMES]))
    record = extract_waveform_measurement(scorer, case, topo, waveform.detach(), torch.device("cuda"))
    gradients = record.pop("_gradients")
    return record, gradients


def finite_parameter_gradients(model):
    total = 0.0
    tensors = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("nonfinite parameter gradient: " + name)
        total += float(parameter.grad.double().square().sum())
        tensors += 1
    if not np.isfinite(total) or total <= 0:
        raise ValueError("generator gradients are zero or nonfinite")
    return dict(norm=total ** 0.5, tensors=tensors)


def fidelity_loss(y, target, cfg, p):
    target = target[:y.numel()]
    stft = cfg["stft_cfg"]
    args = (stft["n_fft"], stft["hop_size"], stft["win_size"], cfg["model_cfg"]["compress_factor"])
    # The existing energy epsilon gives finite fractional-power derivatives
    # at silent spectral bins; it does not alter the emitted waveform.
    ymag = mag_phase_stft(y[None], *args, addeps=True)[0]
    tmag = mag_phase_stft(target[None], *args, addeps=True)[0]
    time_loss = F.l1_loss(y, target)
    mag_loss = F.mse_loss(ymag, tmag)
    loss = p["loss"]["time_l1_weight"] * time_loss + p["loss"]["compressed_magnitude_mse_weight"] * mag_loss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite waveform or magnitude fidelity loss")
    return loss, dict(time_l1=float(time_loss.detach()), magnitude_mse=float(mag_loss.detach()))


def preflight(run, p, rows, cfg, paths, old, exact, safety_dir):
    safety_report = read_json(safety_dir / "safety_report.json")
    if not safety_report["all_peak_safe"] or len(safety_report["rows"]) != 104:
        raise ValueError("real amplitude validation missing")
    model = load_generator(cfg, paths["generator_checkpoint"], torch.device("cuda")).train()
    scorer = scorer_bundle(old, paths)
    runtime = load_runtime_module(paths["candidate_e_runtime_client"])
    measured, saved_gradients = [], []
    with runtime.ExactShimmerTopologyWorker(
        Path(old["exact"]["python"]).resolve(), EXACT_PCM_WORKER,
        Path(old["exact"]["root"]), exact["avqi_code_tree_sha256"],
    ) as worker:
        worker.warmup()
        for index, row in enumerate(rows[:8]):
            y = enhance_waveform(model, audio(row["degraded"]).cuda(), cfg)[0]
            record, gradients = measure(row, y, scorer, runtime, worker, run / "outputs", "preflight")
            record["source_case_id"] = row["case_id"]
            measured.append(record)
            saved_gradients.append(gradients)
            if index == 3:
                medians, weights = calibration_inverse_gradient_weights(measured)
                write_json(run / "outputs/frozen_weights.json", dict(
                    weights=weights, medians=medians, fit_case_ids=[r["case_id"] for r in rows[:4]],
                    holdout_used_for_fit=False))
            # Prove that every component really reaches generator parameters.
            if index < 4:
                component_norms = {}
                for name in NAMES:
                    model.zero_grad(set_to_none=True)
                    y.backward(gradients[name].to(y), retain_graph=True)
                    component_norms[name] = finite_parameter_gradients(model)
                record["generator_component_gradient_norms"] = component_norms
            del y
            print(f"real_backward_preflight={index + 1}/8", flush=True)
    fusions = []
    for record, gradients in zip(measured, saved_gradients):
        joint, fusion, gates = protocol_fusion(p, record, gradients, weights)
        fusions.append(dict(case_id=record["source_case_id"], measurement=record, fusion=fusion, gates=gates))
    write_json(run / "outputs/preflight_report.json", dict(rows=fusions, generator_optimizer_steps=0))
    if not all(all(r["gates"].values()) for r in fusions):
        raise ValueError("six-component preflight requires general gradient repair; see report")
    save_stage(run, "preflight", p, generator_optimizer_steps=0, all_six_generator_backward_verified=True)


def parameter_delta(model, initial):
    sq = 0.0
    changed = 0
    for name, value in model.state_dict().items():
        before = initial[name].to(value)
        if torch.is_floating_point(value):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite generator state: " + name)
            sq += float((value.double() - before.double()).square().sum())
            changed += int(not torch.equal(value, before))
    return dict(l2=sq ** 0.5, changed_tensors=changed)


def save_checkpoint(run, model, optimizer, step, p, cfg, initial):
    path = run / "checkpoints" / f"generator_step_{step:06d}.pt"
    delta = parameter_delta(model, initial)
    payload = dict(generator=model.state_dict(), optimizer=optimizer.state_dict(),
                   generator_optimizer_steps=step, protocol_sha256=os.environ["PROTOCOL_SHA256"],
                   protocol=p, generator_config=cfg, delta_from_initial=delta,
                   torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all())
    torch.save(payload, path)
    return dict(**binding(path), optimizer_steps=step, delta_from_initial=delta)


def train(run, p, rows, cfg, paths, old, exact, preflight_dir, safety_dir):
    weights = read_json(preflight_dir / "frozen_weights.json")["weights"]
    train_rows = [r for r in rows if r["training_role"] == "train"]
    if len(train_rows) != 4:
        raise ValueError("training cohort differs")
    model = load_generator(cfg, paths["generator_checkpoint"], torch.device("cuda")).train()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t = p["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=0)
    scorer = scorer_bundle(old, paths)
    runtime = load_runtime_module(paths["candidate_e_runtime_client"])
    checkpoints = [save_checkpoint(run, model, optimizer, 0, p, cfg, initial)]
    steps = []
    with runtime.ExactShimmerTopologyWorker(
        Path(old["exact"]["python"]).resolve(), EXACT_PCM_WORKER,
        Path(old["exact"]["root"]), exact["avqi_code_tree_sha256"],
    ) as worker:
        worker.warmup()
        for step in range(1, t["maximum_optimizer_steps"] + 1):
            started = time.monotonic()
            row = train_rows[(step - 1) % len(train_rows)]
            optimizer.zero_grad(set_to_none=True)
            y = enhance_waveform(model, audio(row["degraded"]).cuda(), cfg)[0]
            if not torch.isfinite(y).all() or y.abs().max() > 0.95:
                raise ValueError("invalid training output before update")
            record, gradients = measure(row, y, scorer, runtime, worker, run / "outputs", f"step{step:06d}")
            joint, fusion, gates = protocol_fusion(p, record, gradients, weights)
            if not all(gates.values()):
                write_json(run / "outputs" / f"failed_step_{step:06d}.json",
                           dict(measurement=record, fusion=fusion, gates=gates))
                raise ValueError("gradient prerequisite failed before optimizer step")
            fidelity, fidelity_values = fidelity_loss(y, audio(row["target"]).cuda(), cfg, p)
            torch.autograd.backward((y, fidelity), (joint.to(y), None))
            grad = finite_parameter_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            delta = parameter_delta(model, initial)
            if delta["l2"] <= 0:
                raise ValueError("optimizer made no parameter update")
            log = dict(step=step, case_id=row["case_id"], role=row["training_role"],
                       components=record["components"], fusion=fusion, fidelity=fidelity_values,
                       parameter_gradient=grad, delta_from_initial=delta,
                       output_peak=float(y.detach().abs().max()), seconds=time.monotonic() - started)
            steps.append(log)
            with (run / "outputs/training_steps.jsonl").open("a") as handle:
                handle.write(json.dumps(log, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if step % t["checkpoint_interval"] == 0:
                checkpoints.append(save_checkpoint(run, model, optimizer, step, p, cfg, initial))
            print(f"optimizer_step={step}/128 parameter_delta={delta['l2']:.8f} peak={log['output_peak']:.7f}", flush=True)
    # Reload the actual saved final model before reporting completion.
    restored = load_generator(cfg, Path(checkpoints[-1]["path"]), torch.device("cuda"))
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise ValueError("checkpoint reload differs: " + name)
    write_json(run / "outputs/training_report.json", dict(
        optimizer_steps=len(steps), train_case_ids=[r["case_id"] for r in train_rows],
        heldout_optimizer_steps=0, checkpoints=checkpoints, checkpoint_reload_exact=True,
        delta_from_initial=parameter_delta(model, initial),
        before_safety_report=binding(safety_dir / "safety_report.json"),
        preflight_report=binding(preflight_dir / "preflight_report.json")))
    save_stage(run, "train", p, generator_optimizer_steps=len(steps), checkpoints=checkpoints)


def evaluate(run, p, rows, cfg, paths, old, exact, train_dir, safety_dir):
    trained = read_json(train_dir / "training_report.json")
    if trained["optimizer_steps"] != 128:
        raise ValueError("complete actual training required")
    checkpoint = verified(trained["checkpoints"][-1])
    model = load_generator(cfg, checkpoint, torch.device("cuda"))
    baseline = {r["case_id"]: r["output"] for r in read_json(safety_dir / "safety_report.json")["rows"]}
    (run / "outputs/after").mkdir()
    metrics, exact_items = [], []
    for index, row in enumerate(rows, 1):
        with torch.no_grad():
            y = enhance_waveform(model, audio(row["degraded"]).cuda(), cfg)[0].cpu()
        after = write_audio(run / "outputs/after" / (row["case_id"] + ".wav"), y.numpy())
        base, target = audio(baseline[row["case_id"]]), audio(row["target"])
        n = min(base.numel(), target.numel(), y.numel())
        guard = full_band_pathology_guardrails(target, base, y)
        safe = waveform_safety(base, y)
        denoise = dict(snr_before=snr_db(target[:n], base[:n]), snr_after=snr_db(target[:n], y[:n]),
                       si_sdr_before=si_sdr_db(target[:n], base[:n]), si_sdr_after=si_sdr_db(target[:n], y[:n]))
        if not torch.isfinite(y).all() or y.abs().max() > 0.95:
            raise ValueError("post-training amplitude invalid")
        metrics.append(dict(case_id=row["case_id"], role=row["training_role"], split=row["split"],
                            label=row["label"], view=row["view"], condition=row["condition"],
                            before=baseline[row["case_id"]], after=after,
                            full_band=guard, safety=safe, denoising=denoise))
        if row["label"] == "patient":
            for phase, value in (("before", baseline[row["case_id"]]), ("after", after)):
                exact_items.append(dict(id=row["case_id"] + ":" + phase, path=value["path"], view=row["view"]))
        print(f"post_training_waveform={index}/104", flush=True)
    # Seal every before/after waveform before opening any Exact outcome.
    write_json(run / "outputs/evaluation_waveform_seal.json", dict(rows=metrics, exact_items=exact_items))
    scores = run_exact(exact_items, exact_python=Path(old["exact"]["python"]),
                       avqi_code_root=Path(old["exact"]["root"]), expected_runtime=exact)
    by_id = {r["case_id"]: r for r in rows}
    scale = np.array([old["normalization"]["target_scale"][n] for n in NAMES])
    for metric in metrics:
        row = by_id[metric["case_id"]]
        if row["label"] != "patient":
            continue
        target = np.array([row["target_components"][n] for n in NAMES])
        before = scores[row["case_id"] + ":before"]
        after = scores[row["case_id"] + ":after"]
        gap_before, gap_after = np.abs(before - target) / scale, np.abs(after - target) / scale
        metric["exact"] = dict(target=dict(zip(NAMES, target.tolist())),
                               before=dict(zip(NAMES, before.tolist())), after=dict(zip(NAMES, after.tolist())),
                               normalized_gap_before=dict(zip(NAMES, gap_before.tolist())),
                               normalized_gap_after=dict(zip(NAMES, gap_after.tolist())),
                               gap_reduction=dict(zip(NAMES, (gap_before - gap_after).tolist())))
    summaries = {}
    for role in ("train", "validation", "evaluation"):
        group = [r for r in metrics if r["role"] == role and "exact" in r]
        summaries[role] = dict(rows=len(group), components={n: dict(
            median_normalized_gap_reduction=float(np.median([r["exact"]["gap_reduction"][n] for r in group])),
            fraction_improved=float(np.mean([r["exact"]["gap_reduction"][n] > 0 for r in group])),
        ) for n in NAMES})
    write_json(run / "outputs/before_after_report.json", dict(
        optimizer_steps=128, checkpoint=binding(checkpoint), rows=metrics, summaries=summaries,
        scientific_promotion=False, independent_validation=False))
    save_stage(run, "evaluate", p, generator_optimizer_steps=128, evaluated_waveforms=104,
               exact_patient_rows=56)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("safety", "preflight", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--safety-receipt", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--train-receipt", type=Path)
    args = parser.parse_args()
    p = read_json(args.protocol)
    validate_protocol(p)
    if binding(args.protocol)["sha256"] != os.environ["PROTOCOL_SHA256"]:
        raise ValueError("protocol hash differs")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one Slurm GPU is required")
    torch.set_num_threads(4)
    set_model_seed(p["training"]["seed"])
    np.random.seed(p["training"]["seed"])
    random.seed(p["training"]["seed"])
    torch.backends.cudnn.benchmark = False
    run = args.run_root.resolve()
    (run / "outputs").mkdir(exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    old, paths, exact, cfg = context(p)
    rows = dataset(p)
    safety_dir = load_dependency(args.safety_receipt, "safety", p) if args.safety_receipt else None
    preflight_dir = load_dependency(args.preflight_receipt, "preflight", p) if args.preflight_receipt else None
    train_dir = load_dependency(args.train_receipt, "train", p) if args.train_receipt else None
    if args.stage == "safety":
        safety(run, p, rows, cfg, paths)
    elif args.stage == "preflight":
        preflight(run, p, rows, cfg, paths, old, exact, safety_dir)
    elif args.stage == "train":
        train(run, p, rows, cfg, paths, old, exact, preflight_dir, safety_dir)
    else:
        evaluate(run, p, rows, cfg, paths, old, exact, train_dir, safety_dir)


if __name__ == "__main__":
    main()
