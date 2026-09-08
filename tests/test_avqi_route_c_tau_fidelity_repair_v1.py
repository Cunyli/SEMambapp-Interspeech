import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.avqi_route_c_tau_amplitude_training_v1 import binding, fidelity_loss
from scripts.avqi_route_c_tau_fidelity_repair_v1 import (
    fidelity_for_arm, fixed_input_lag, load_alignment, paired_metrics,
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
