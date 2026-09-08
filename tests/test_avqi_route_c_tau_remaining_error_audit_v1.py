import pytest
import torch

from model.avqi_route_c import RouteCFiveActiveScorer
from scripts.avqi_route_c_tau_remaining_error_audit_v1 import clean_proxy, geometry_delta, signal_geometry


def test_signed_projection_exposes_sisdr_polarity_ambiguity():
    r = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    noise = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
    positive = signal_geometry(r, r + noise)
    negative = signal_geometry(r, -r + noise)
    assert positive["si_sdr_db"] == pytest.approx(negative["si_sdr_db"])
    assert positive["signed_correlation"] == pytest.approx(-negative["signed_correlation"])
    assert negative["centered_reference_error_energy"] > positive["centered_reference_error_energy"]


def test_pure_gain_change_cannot_explain_sisdr_regression():
    r = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    y = r + torch.tensor([.5, .5, -.5, -.5], dtype=torch.float64)
    delta = geometry_delta(signal_geometry(r, y), signal_geometry(r, .4 * y))
    assert delta["si_sdr_change_db"] == pytest.approx(0, abs=1e-10)
    assert delta["projection_energy_change_db"] == pytest.approx(delta["residual_energy_change_db"])


def test_weaker_negative_projection_can_reduce_error_but_lower_sisdr():
    r = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    n = torch.tensor([.5, .5, -.5, -.5], dtype=torch.float64)
    before = signal_geometry(r, -.5 * r + n)
    after = signal_geometry(r, -.2 * r + n)
    assert after["centered_reference_error_energy"] < before["centered_reference_error_energy"]
    assert geometry_delta(before, after)["si_sdr_change_db"] < 0


def test_silent_signal_abstains_from_correlation_diagnosis():
    with pytest.raises(ValueError, match="non-silent"):
        signal_geometry(torch.zeros(8), torch.ones(8))


def test_clean_identity_never_substitutes_legacy_shimmer_db(monkeypatch):
    class Scorer:
        target_mean = torch.zeros(6)

        def denormalized_prediction(self, values):
            return values

    monkeypatch.setattr(RouteCFiveActiveScorer, "forward",
                        lambda self, waveform, view: torch.tensor([[1., 2., 3., 999., 5., 6.]]))
    result = clean_proxy({"view": "cs"}, torch.ones(8), Scorer())
    assert result == dict(cpps=1., hnr=2., shimmer_percent=3., shimmer_db=None, slope=5., tilt=6.)
