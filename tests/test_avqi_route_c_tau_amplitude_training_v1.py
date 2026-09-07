import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.avqi_route_c_tau_amplitude_training_v1 import (
    fidelity_loss, finite_parameter_gradients, parameter_delta, validate_protocol,
    validate_roles,
)


def protocol():
    return json.loads((Path(__file__).parents[1] / "configs/avqi_route_c_tau_amplitude_training_v1.json").read_text())


def test_real_training_authorization_is_required():
    p = protocol()
    validate_protocol(p)
    p["authorization"]["training_authorized"] = False
    with pytest.raises(ValueError, match="authorization"):
        validate_protocol(p)


def test_no_silent_step_budget_or_promotion_expansion():
    for key, value in (("maximum_optimizer_steps", 1024), ("learning_rate", 0.01)):
        p = protocol()
        p["training"][key] = value
        with pytest.raises(ValueError, match="settings"):
            validate_protocol(p)


def role_rows():
    rows = []
    for index in range(20):
        role = "train" if index < 4 else "validation" if index < 8 else "evaluation"
        for view in range(1 if index < 8 else 8):
            rows.append(dict(case_id=f"{index}:{view}", canonical_speaker_id=f"TAU:{index}",
                             dataset="TAU", historically_exact_opened=True, training_role=role))
    return rows


def test_speaker_leakage_rejected_even_when_row_counts_match():
    rows = role_rows()
    validate_roles(rows)
    rows[4]["canonical_speaker_id"] = rows[0]["canonical_speaker_id"]
    with pytest.raises(ValueError, match="speaker overlap"):
        validate_roles(rows)


def test_nonfinite_and_detached_parameter_gradients_rejected():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="zero or nonfinite"):
        finite_parameter_gradients(model)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(ValueError, match="nonfinite parameter"):
        finite_parameter_gradients(model)


def test_waveform_vjp_changes_actual_generator_parameters():
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 4)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    y = model(torch.tensor([1.0, -0.5, 0.2]))
    y.backward(torch.tensor([0.2, -0.4, 0.1, 0.3]))
    assert finite_parameter_gradients(model)["norm"] > 0
    optimizer.step()
    delta = parameter_delta(model, initial)
    assert delta["l2"] > 0 and delta["changed_tensors"] == 2


def test_fidelity_silent_bins_have_finite_gradients():
    cfg = {"stft_cfg": {"n_fft": 400, "hop_size": 100, "win_size": 400},
           "model_cfg": {"compress_factor": 0.3}}
    y = torch.zeros(16000, requires_grad=True)
    loss, values = fidelity_loss(y, torch.ones_like(y) * 0.1, cfg, protocol())
    loss.backward()
    assert torch.isfinite(y.grad).all()
    assert values["magnitude_mse"] > 0
