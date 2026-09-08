"""Diagnose frozen TAU waveform geometry and proxy/Exact disagreement.

No generator inference, optimizer, waveform correction, or reserve scoring.
The Exact component scores are inherited from the bound completed audit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np
import torch

from model.avqi_training_reference import five_component_clean_prediction
from scripts import avqi_route_c_tau_amplitude_training_v1 as prior
from scripts.avqi_route_c_tau_fidelity_repair_v1 import load_alignment, reference_pair


ROOT = Path(__file__).resolve().parents[1]


def signal_geometry(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    """Keep projection sign visible alongside the sign-invariant SI-SDR."""
    if reference.ndim != 1 or estimate.shape != reference.shape:
        raise ValueError("equal-length mono signals required")
    r, y = reference.double(), estimate.double()
    r, y = r - r.mean(), y - y.mean()
    reference_energy, output_energy = r.square().sum(), y.square().sum()
    if min(float(reference_energy), float(output_energy)) <= 1e-12:
        raise ValueError("non-silent signals required for correlation diagnosis")
    alpha = torch.dot(y, r) / reference_energy
    projected = alpha * r
    residual = y - projected
    projection_energy = projected.square().sum().clamp_min(1e-12)
    residual_energy = residual.square().sum().clamp_min(1e-12)
    return dict(
        signed_projection_gain=float(alpha),
        signed_correlation=float(torch.dot(y, r) / (reference_energy * output_energy).sqrt()),
        output_energy=float(output_energy),
        projection_energy=float(projection_energy),
        residual_energy=float(residual_energy),
        centered_reference_error_energy=float((y - r).square().sum()),
        si_sdr_db=float(10 * torch.log10(projection_energy / residual_energy)),
    )


def geometry_delta(before: dict, after: dict) -> dict[str, float]:
    result = {key + "_change_db": float(10 * np.log10(after[key] / before[key]))
              for key in ("output_energy", "projection_energy", "residual_energy")}
    result["si_sdr_change_db"] = after["si_sdr_db"] - before["si_sdr_db"]
    np.testing.assert_allclose(result["si_sdr_change_db"],
        result["projection_energy_change_db"] - result["residual_energy_change_db"], atol=1e-10)
    return result


def clean_proxy(row, waveform, scorer):
    """Measure five-slot identity bias through the existing non-topology API.

    Candidate-E accepts current model outputs only. Its clean-target identity
    check stays unmeasured; the legacy Shimmer dB slot cannot substitute for it.
    """
    return five_component_clean_prediction(scorer, waveform, row["view"])


def summaries(rows):
    result = {}
    for role in ("train", "validation"):
        group = [row for row in rows if row["role"] == role]
        result[role] = dict(n=len(group),
            si_sdr_delta_median_db=statistics.median(row["geometry_delta"]["si_sdr_change_db"] for row in group),
            negative_baseline_projection=sum(row["geometry_before"]["signed_projection_gain"] < 0 for row in group),
            components={name: dict(
                clean_identity_normalized_gap_median=(None if name == "shimmer_db" else statistics.median(
                    row["components"][name]["clean_identity_normalized_gap"] for row in group)),
                proxy_reduction_median=statistics.median(row["components"][name]["proxy_normalized_gap_reduction"] for row in group),
                exact_reduction_median=statistics.median(row["components"][name]["exact_normalized_gap_reduction"] for row in group),
                baseline_proxy_exact_opposite_target_sides=sum(row["components"][name]["before_opposite_target_sides"] for row in group),
                after_proxy_exact_opposite_target_sides=sum(row["components"][name]["after_opposite_target_sides"] for row in group),
                proxy_exact_improvement_sign_disagreements=sum(row["components"][name]["improvement_sign_disagrees"] for row in group),
                net_displacement_opposes_baseline_proxy_descent=sum(row["components"][name]["gradient_dot_net_displacement"] > 0 for row in group),
            ) for name in prior.NAMES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("allocated Slurm GPU required")
    torch.set_num_threads(1)
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or properties.total_memory < 30 * 1024 ** 3:
        raise ValueError("exactly one 32 GB class GPU required for CPPS")
    spec = prior.read_json(args.spec)
    protocol = prior.read_json(prior.verified(spec["protocol"]))
    old_protocol = prior.read_json(ROOT / protocol["previous_protocol"])
    prior.validate_protocol(old_protocol)
    old, paths, exact_runtime, _ = prior.context(old_protocol)
    safety = prior.read_json(prior.verified(dict(path=str(ROOT / protocol["safety_report"]),
                                                sha256=protocol["safety_report_sha256"])))
    prior.validate_roles(safety["source_rows"])
    rows = [row for row in safety["source_rows"] if row["training_role"] != "evaluation"]
    if len(rows) != 8:
        raise ValueError("exactly eight frozen development rows required")
    lags = load_alignment(prior.verified(spec["alignment"]), spec["alignment"]["sha256"], protocol, safety["source_rows"])
    development = prior.read_json(prior.verified(spec["development_report"]))
    previous_audit = prior.read_json(prior.verified(spec["exact_audit"]))
    if (development["evaluation_reserve_opened"] or previous_audit["evaluation_reserve_opened"]
            or previous_audit["inputs"]["candidates"][0]["development_report"] != spec["development_report"]):
        raise ValueError("Exact outcomes must bind the same completed development candidate")
    measured = {row["case_id"]: row for row in development["rows"]}
    if set(measured) != {row["case_id"] for row in rows}:
        raise ValueError("development case coverage differs")
    baseline = {row["case_id"]: row["output"] for row in safety["rows"] if row["case_id"] in measured}
    scores = previous_audit["exact_scores"]
    scales = old["normalization"]["target_scale"]
    output = args.run_root / "outputs"
    output.mkdir()
    bindings = []
    for row in rows:
        case = measured[row["case_id"]]
        if case["before"] != baseline[row["case_id"]] or case["role"] != row["training_role"]:
            raise ValueError("baseline or role differs")
        for phase, value in (("target", row["target"]), ("before", case["before"]), ("after", case["after"])):
            prior.audio(value)
            bindings.append(dict(case_id=row["case_id"], phase=phase, waveform=value))
    prior.write_json(output / "input_seal.json", dict(spec=spec, waveforms=bindings,
        generator_optimizer_steps=0, input_waveforms_modified=False))
    scorer = prior.scorer_bundle(old, paths)
    runtime = prior.load_runtime_module(paths["candidate_e_runtime_client"])
    results = []
    with runtime.ExactShimmerTopologyWorker(Path(old["exact"]["python"]), prior.EXACT_PCM_WORKER,
            Path(old["exact"]["root"]), exact_runtime["avqi_code_tree_sha256"]) as worker:
        worker.warmup()
        for row in rows:
            case = measured[row["case_id"]]
            target, before, after = prior.audio(row["target"]), prior.audio(case["before"]), prior.audio(case["after"])
            reference, before_view = reference_pair(target, before, lags[row["case_id"]], protocol["alignment"]["method"])
            _, after_view = reference_pair(target, after, lags[row["case_id"]], protocol["alignment"]["method"])
            gb, ga = signal_geometry(reference, before_view), signal_geometry(reference, after_view)
            np.testing.assert_allclose([gb["si_sdr_db"], ga["si_sdr_db"]],
                [case["metrics"]["aligned_si_sdr_before_db"], case["metrics"]["aligned_si_sdr_after_db"]], atol=1e-3)
            clean = clean_proxy(row, target, scorer)
            rb, gradients = prior.measure(row, before, scorer, runtime, worker, output, "before")
            ra, unused = prior.measure(row, after, scorer, runtime, worker, output, "after")
            del unused
            prior.write_json(output / (row["case_id"] + "_measurements.json"), dict(before=rb, after=ra, clean_proxy=clean))
            components = {}
            delta = (after.double() - before.double())
            for name in prior.NAMES:
                pb, pa = rb["components"][name]["prediction"], ra["components"][name]["prediction"]
                eb = scores[row["case_id"] + ":before"][name]
                ea = scores[row["case_id"] + ":aligned_joint_bounded"][name]
                t, scale = row["target_components"][name], scales[name]
                proxy_reduction, exact_reduction = (abs(pb-t)-abs(pa-t))/scale, (abs(eb-t)-abs(ea-t))/scale
                gradient = gradients[name]
                components[name] = dict(target=t, proxy_before=pb, proxy_after=pa, exact_before=eb, exact_after=ea,
                    clean_proxy=clean[name], clean_identity_normalized_gap=(None if clean[name] is None else abs(clean[name]-t)/scale),
                    before_proxy_exact_normalized_bias=(pb-eb)/scale, after_proxy_exact_normalized_bias=(pa-ea)/scale,
                    before_opposite_target_sides=(pb-t)*(eb-t) < 0,
                    after_opposite_target_sides=(pa-t)*(ea-t) < 0,
                    proxy_normalized_gap_reduction=proxy_reduction, exact_normalized_gap_reduction=exact_reduction,
                    improvement_sign_disagrees=proxy_reduction*exact_reduction < 0,
                    gradient_dot_net_displacement=float(torch.dot(gradient, delta)),
                    gradient_net_displacement_cosine=float(torch.dot(gradient, delta)/(gradient.norm()*delta.norm())))
            results.append(dict(case_id=row["case_id"], role=row["training_role"], view=row["view"],
                geometry_before=gb, geometry_after=ga, geometry_delta=geometry_delta(gb, ga), components=components))
            print(json.dumps(dict(case_id=row["case_id"], complete=True)), flush=True)
            del gradients, rb, ra
    report = dict(schema_version="avqi-route-c-tau-remaining-error-audit-v2", inputs=spec,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        audit_source=prior.binding(Path(__file__).resolve()), slurm_job_id=os.environ["SLURM_JOB_ID"],
        gpu=dict(name=properties.name, memory_bytes=properties.total_memory), rows=results, summaries=summaries(results),
        generator_optimizer_steps=0, model_inference=False, candidate_exact_scoring_performed=False,
        evaluation_reserve_opened=False, scientific_promotion=False,
        limits=["Eight repeatedly opened development cases; no independent validation.",
                "Clean identity bias covers five components only. Candidate-E clean topology is forbidden; its identity result is null, never replaced by the legacy Shimmer dB slot.",
                "The baseline-gradient dot full trajectory is a local diagnostic, not a per-step or AdamW proof.",
                "No output polarity, gain, time shift, target, training weight, or model parameter was changed."])
    prior.write_json(output / "remaining_error_audit.json", report)
    print(json.dumps(dict(report=prior.binding(output / "remaining_error_audit.json"), summaries=report["summaries"])), flush=True)


if __name__ == "__main__":
    main()
