"""Small differentiable models for AVQI-component experiments.

The models predict the six terms used by AVQI v03.01. Jitter is deliberately
excluded because it is diagnostic-only in the verified Praat implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


AVQI_COMPONENT_NAMES = (
    "cpps",
    "hnr",
    "shimmer_percent",
    "shimmer_db",
    "slope",
    "tilt",
)

# CPPS and HNR each represent one concept. The two shimmer terms and the two
# LTAS terms are correlated representations, so each member receives half
# weight instead of counting the same concept twice.
AVQI_COMPONENT_LOSS_WEIGHTS = (1.0, 1.0, 0.5, 0.5, 0.5, 0.5)


def avqi_v0301(components: torch.Tensor) -> torch.Tensor:
    """Compute the verified AVQI v03.01 scalar from six ordered components."""
    if components.shape[-1] != len(AVQI_COMPONENT_NAMES):
        raise ValueError(
            f"expected {len(AVQI_COMPONENT_NAMES)} AVQI components, "
            f"got shape {tuple(components.shape)}"
        )
    coefficients = components.new_tensor(
        (-0.177, -0.006, -0.037, 0.941, 0.01, 0.093)
    )
    return (4.152 + (components * coefficients).sum(dim=-1)) * 2.8902


def pool_shared_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Pool a SeMamba++ shared map without assuming a fixed utterance length."""
    if feature_map.ndim != 4:
        raise ValueError(
            "expected a [batch, channels, time, frequency] feature map, "
            f"got {tuple(feature_map.shape)}"
        )
    mean = feature_map.mean(dim=(-2, -1))
    std = feature_map.std(dim=(-2, -1), unbiased=False)
    return torch.cat((mean, std), dim=-1)


class SharedComponentHead(nn.Module):
    """A compact head for an encoder or late shared SeMamba++ feature map."""

    def __init__(self, feature_channels: int = 48, hidden_features: int = 64):
        super().__init__()
        pooled_features = feature_channels * 2
        self.regressor = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Linear(pooled_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, len(AVQI_COMPONENT_NAMES)),
        )

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        if pooled_features.ndim != 2:
            raise ValueError(
                f"expected a [batch, features] tensor, got {pooled_features.shape}"
            )
        return self.regressor(pooled_features)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.forward_pooled(pool_shared_feature_map(feature_map))


class WaveformComponentPredictor(nn.Module):
    """A small waveform-to-six-components predictor with a differentiable STFT."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        hidden_features: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, len(AVQI_COMPONENT_NAMES)),
        )

    def log_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                f"expected a [batch, time] waveform, got {tuple(waveform.shape)}"
            )
        if waveform.shape[-1] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[-1]))
        centered = waveform - waveform.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        normalized = centered / rms.clamp_min(1e-5)
        spectrum = torch.stft(
            normalized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=waveform.dtype),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        log_magnitude = torch.log1p(spectrum.abs())
        mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
        std = log_magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return ((log_magnitude - mean) / std.clamp_min(1e-5)).unsqueeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.log_spectrogram(waveform))
        mean = encoded.mean(dim=(-2, -1))
        std = encoded.std(dim=(-2, -1), unbiased=False)
        return self.regressor(torch.cat((mean, std), dim=-1))


def standardized_component_loss(
    normalized_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Concept-balanced Huber loss in training-split deviation units."""
    if normalized_prediction.shape != raw_target.shape:
        raise ValueError(
            "prediction and target shapes differ: "
            f"{normalized_prediction.shape} != {raw_target.shape}"
        )
    if normalized_prediction.shape[-1] != len(AVQI_COMPONENT_NAMES):
        raise ValueError(
            f"expected {len(AVQI_COMPONENT_NAMES)} AVQI components, "
            f"got shape {tuple(normalized_prediction.shape)}"
        )
    normalized_target = (raw_target - target_mean) / target_scale.clamp_min(1e-8)
    element_loss = F.smooth_l1_loss(
        normalized_prediction,
        normalized_target,
        reduction="none",
    )
    weights = element_loss.new_tensor(AVQI_COMPONENT_LOSS_WEIGHTS)
    return (element_loss * weights).sum(dim=-1).div(weights.sum()).mean()


def denormalize_components(
    normalized_prediction: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    return normalized_prediction * target_scale + target_mean


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
