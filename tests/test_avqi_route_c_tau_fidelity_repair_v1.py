import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from dataloaders.legacy_online_degradation import _target_audio_for_selected_degradations
from scripts.avqi_route_c_tau_amplitude_training_v1 import binding, fidelity_loss
from scripts.avqi_route_c_tau_fidelity_repair_v1 import (
    configure_numerical_policy, configure_training_scope, fidelity_for_arm, fixed_input_lag,
    load_alignment, maximum_circular_phase_difference, paired_metrics,
    RIR_REFERENCE, shifted_dry_reference,
)


CFG = {"stft_cfg": {"n_fft": 400, "hop_size": 100, "win_size": 400},
       "model_cfg": {"compress_factor": 0.3}}


def protocol():
    return json.loads((Path(__file__).parents[1] / "configs/avqi_route_c_tau_fidelity_repair_v1.json").read_text())


def waveform():
    generator = torch.Generator().manual_seed(52)
    return torch.randn(8192, generator=generator) * 0.08


@pytest.mark.parametrize("lag", [0, 73, -113])
def test_frozen_input_lag_and_aligned_loss_recover_identical_delayed_audio(lag):
    target = waveform()
    degraded = (F.pad(target, (lag, 0))[:target.numel()] if lag >= 0
                else F.pad(target[-lag:], (0, -lag)))
    frozen = fixed_input_lag(target, degraded)
    assert frozen == lag
    y = degraded.clone().requires_grad_()
    loss, terms = fidelity_for_arm(y, target, frozen, CFG, protocol(), "aligned_fidelity")
    assert float(loss) == 0
    assert terms["fidelity_samples"] == target.numel() - abs(lag)
    loss.backward()
    assert torch.isfinite(y.grad).all()


def test_alignment_does_not_forgive_output_gain_error():
    target = waveform()
    delayed = F.pad(target, (83, 0))[:target.numel()]
    y = (delayed * 1.2).requires_grad_()
    loss, _ = fidelity_for_arm(y, target, 83, CFG, protocol(), "aligned_fidelity")
    loss.backward()
    assert float(loss) > 0
    assert torch.dot(y.grad, y.detach()) > 0  # Descent reduces the excessive gain.
    metrics = paired_metrics(target, delayed, y.detach(), 83)
    assert metrics["aligned_snr_change_db"] < -20
    assert metrics["relative_gain"] == pytest.approx(1.2, rel=1e-6)


def test_unaligned_control_preserves_original_fidelity_exactly():
    target = waveform()
    y = F.pad(target, (63, 0))[:target.numel()]
    p = protocol()
    actual, _ = fidelity_for_arm(y, target, 63, CFG, p, "unaligned_fidelity")
    expected, _ = fidelity_loss(y, target, CFG, {"loss": p["training"]})
    assert torch.equal(actual, expected)
    aligned, _ = fidelity_for_arm(y, target, 63, CFG, p, "aligned_fidelity")
    assert float(aligned) == 0 and float(actual) > 0


def test_input_alignment_boundary_abstains():
    target = waveform()
    delayed = F.pad(target, (100, 0))[:target.numel()]
    with pytest.raises(ValueError, match="boundary"):
        fixed_input_lag(target, delayed, maximum_lag=100)


def test_fixed_alignment_rejects_tampering_and_input_swap(tmp_path):
    p = protocol()
    rows = [dict(case_id="train", target={"sha256": "a"}, degraded={"sha256": "b"})]
    doc = dict(alignment_rule=p["alignment"], source_safety_report_sha256=p["safety_report_sha256"],
               lag_sealed_before_training=True, candidate_outcomes_used=False,
               rows=[dict(rows[0], lag_samples=17)])
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(doc))
    sha = binding(path)["sha256"]
    assert load_alignment(path, sha, p, rows) == {"train": 17}
    path.write_text(json.dumps(dict(doc, candidate_outcomes_used=True)))
    with pytest.raises(ValueError, match="hash differs"):
        load_alignment(path, sha, p, rows)
    with pytest.raises(ValueError, match="definition differs"):
        load_alignment(path, binding(path)["sha256"], p, rows)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="different waveform"):
        load_alignment(path, binding(path)["sha256"], p,
                       [dict(rows[0], degraded={"sha256": "c"})])


@pytest.mark.parametrize("lag", [0, 79, 715])
def test_physical_rir_reference_matches_canonical_pretraining_target(lag):
    target = waveform()
    rir = torch.zeros(900)
    rir[lag] = -1.0  # Delay follows absolute RIR peak; target polarity stays intact.
    expected = _target_audio_for_selected_degradations(
        target.numpy(), rir.numpy(), ["reverb", "noise"], target_type="shifted_anechoic")
    actual = shifted_dry_reference(target, lag)
    assert torch.equal(actual, torch.from_numpy(expected))
    p = protocol()
    p["alignment"]["method"] = RIR_REFERENCE
    y = actual.clone().requires_grad_()
    loss, terms = fidelity_for_arm(y, target, lag, CFG, p, "aligned_fidelity")
    assert float(loss) == 0 and terms["fidelity_samples"] == y.numel()
    loss.backward()
    assert torch.isfinite(y.grad).all()


def test_physical_reference_penalizes_noise_before_speech_arrival():
    target = waveform()
    y = shifted_dry_reference(target, 200).clone()
    y[:200] = 0.04
    y.requires_grad_()
    p = protocol()
    p["alignment"]["method"] = RIR_REFERENCE
    loss, _ = fidelity_for_arm(y, target, 200, CFG, p, "aligned_fidelity")
    loss.backward()
    assert float(loss) > 0 and y.grad[:200].abs().sum() > 0


def test_physical_reference_rejects_entirely_truncated_target():
    with pytest.raises(ValueError, match="nonempty"):
        shifted_dry_reference(waveform(), 8192)


def test_mask_only_training_keeps_shared_state_and_phase_fixed():
    torch.manual_seed(11)
    model = torch.nn.Module()
    model.shared = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    model.phase_decoder = torch.nn.Linear(4, 4)
    model.mask_decoder = torch.nn.Linear(4, 4)
    configure_training_scope(model, "mask_decoder_only")
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(3, 4)
    phase_before = model.phase_decoder(model.shared(x)).detach().clone()
    magnitude_before = model.mask_decoder(model.shared(x)).detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.mask_decoder(model.shared(x)).square().sum().backward()
    optimizer.step()
    assert torch.equal(model.phase_decoder(model.shared(x)), phase_before)
    assert not torch.equal(model.mask_decoder(model.shared(x)), magnitude_before)
    for name, value in model.state_dict().items():
        if not name.startswith("mask_decoder."):
            assert torch.equal(value, initial[name])
    configure_training_scope(model, "all_parameters")
    assert model.training and all(p.requires_grad for p in model.parameters())


def test_unknown_training_scope_abstains():
    with pytest.raises(ValueError, match="unknown"):
        configure_training_scope(torch.nn.Linear(2, 2), "phase_sometimes")


def test_phase_invariance_uses_circular_distance_at_pi_boundary():
    before = torch.tensor([torch.pi - 1e-8], dtype=torch.float64)
    after = torch.tensor([-torch.pi + 1e-8], dtype=torch.float64)
    assert maximum_circular_phase_difference(before, after) == pytest.approx(2e-8, abs=1e-14)
    assert maximum_circular_phase_difference(before, before) == 0


def test_stable_policy_is_explicit_and_does_not_relax_phase_tolerance(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    assert configure_numerical_policy("deterministic_float32") == dict(
        policy="deterministic_float32", cudnn_deterministic=True,
        cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False)
    with pytest.raises(ValueError, match="unknown numerical"):
        configure_numerical_policy("automatic_relaxation")
