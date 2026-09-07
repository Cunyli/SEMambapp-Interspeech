import hashlib

import numpy as np
import pytest
import torch

from model.avqi_route_c_candidate_e import exact_metric_branch_ste, exact_numpy_highpass_pcm16
from scripts.avqi_shimmer_exact_pcm_worker_v1 import export_engine_type


def test_authoritative_pcm_forward_and_existing_jacobian_agree():
    x = (torch.sin(torch.arange(16000, dtype=torch.float64) * 0.07) * 0.1).requires_grad_()
    expected, digest = exact_numpy_highpass_pcm16(x, peak_scale_required=False)
    codes = (expected * 32768).to(torch.int32).tolist()
    args = (x, torch.arange(x.numel()), 0)
    local, _ = exact_metric_branch_ste(*args, expected_highpass_pcm16_sha256=digest)
    actual, cert = exact_metric_branch_ste(*args, expected_highpass_pcm16_sha256=digest,
                                          authoritative_highpass_pcm16_codes=codes)
    assert torch.equal(actual, expected) and cert["exact_highpass_pcm16_sha256"] == digest
    weight = torch.cos(torch.arange(x.numel(), dtype=torch.float64) * 0.1)
    a = torch.autograd.grad((actual * weight).sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad((local * weight).sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.isfinite(a).all() and a.norm() > 0


def test_pcm_corruption_is_not_accepted():
    x = torch.zeros(100, dtype=torch.float64)
    codes = np.zeros(100, dtype="<i4")
    digest = hashlib.sha256(codes.tobytes()).hexdigest()
    codes[50] = 1
    with pytest.raises(ValueError, match="hash differs"):
        exact_metric_branch_ste(x, torch.arange(100), 0,
                               expected_highpass_pcm16_sha256=digest,
                               authoritative_highpass_pcm16_codes=codes.tolist())


@pytest.mark.parametrize("codes", [[0] * 9, [40000] * 10, [0.5] * 10])
def test_pcm_length_range_and_integer_contract(codes):
    with pytest.raises(ValueError, match="codes are invalid"):
        exact_metric_branch_ste(torch.zeros(10, dtype=torch.float64), torch.arange(10), 0,
                               expected_highpass_pcm16_sha256="0" * 64,
                               authoritative_highpass_pcm16_codes=codes)


def test_worker_exports_this_refresh_without_recomputing_metric():
    class Base:
        def metric_highpass(self, waveform, mode):
            self.calls += 1
            return waveform, {}

        def refresh_waveform(self, waveform):
            values, _ = self.metric_highpass(waveform, "test")
            codes = np.rint(values * 32768).astype("<i4")
            return dict(source_sample_count=values.size,
                        highpass_pcm16_sha256=hashlib.sha256(codes.tobytes()).hexdigest())

    engine = export_engine_type(Base, lambda x: np.rint(x * 32768).astype("<i4"))()
    engine.calls = 0
    for value in (0.1, 0.2):
        result = engine.refresh_waveform(np.full(16, value))
        assert result["highpass_pcm16_codes"] == [round(value * 32768)] * 16
    assert engine.calls == 2
