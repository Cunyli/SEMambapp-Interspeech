import torch
from auraloss.freq import MultiResolutionSTFTLoss


def load_modules(cfg, device):
    return MultiResolutionSTFTLoss(sample_rate=cfg["stft_cfg"]["sampling_rate"]).to(device)


def compute_val_metrics(mrstft, clean, pred):
    """
    clean, pred: torch.FloatTensor (B, T)
    """
    # STFT loss
    mrstft_loss = mrstft(pred.unsqueeze(1), clean.unsqueeze(1))


    return {"mrstft_score": mrstft_loss.mean()}
