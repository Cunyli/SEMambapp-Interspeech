import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
from auraloss.freq import MultiResolutionSTFTLoss
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


class DummyUTMOS(nn.Module):
    def forward(self, pred, sample_rate):
        batch = pred.shape[0] if pred.ndim > 1 else 1
        device = pred.device
        return torch.full((batch,), float("nan"), device=device)

def load_modules(cfg, device):

    mrstft = MultiResolutionSTFTLoss(sample_rate=cfg["stft_cfg"]["sampling_rate"]).to(device)
    pesq = PerceptualEvaluationSpeechQuality(fs=cfg["stft_cfg"]["sampling_rate"], mode="wb").to(device)
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    except Exception as exc:
        print(f"Warning: failed to load UTMOS from torch.hub ({exc}). Validation will continue without UTMOS.")
        utmos = DummyUTMOS().to(device).eval()

    return mrstft, pesq, utmos


def compute_val_metrics(mrstft, pesq, utmos, clean, pred, cfg, skip_pesq=False):
    """
    clean, pred: torch.FloatTensor (B, T)
    """
    device = clean.device
    batch_size = clean.size(0)

    # STFT loss
    mrstft_loss = mrstft(pred.unsqueeze(1), clean.unsqueeze(1))

    # PESQ can fail on pathological or very low-energy utterances. Keep
    # validation running and rely on MRSTFT/UTMOS for those cases.
    if skip_pesq:
        pesq_score = torch.full((), float("nan"), device=device)
        pesq_valid = False
        pesq_skipped = True
    else:
        try:
            pesq_score = pesq(pred, clean)
            pesq_valid = True
            pesq_skipped = False
        except Exception as exc:
            print(f"Warning: PESQ failed during validation ({exc}). Excluding this sample from PESQ mean.")
            pesq_score = torch.full((), float("nan"), device=device)
            pesq_valid = False
            pesq_skipped = False


    # UTMOS
    with torch.no_grad():
        utmos_score = utmos(pred, cfg["stft_cfg"]["sampling_rate"])

    return {
        "mrstft_score": mrstft_loss.mean(),
        "pesq_score": pesq_score.mean(),
        "pesq_valid": pesq_valid,
        "pesq_skipped": pesq_skipped,
        "utmos_score": utmos_score.mean()
    }
