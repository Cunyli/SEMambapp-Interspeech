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

TFGRID_COMPONENT_GROUPS = (
    ("periodicity", (0, 1)),
    ("amplitude_modulation", (2, 3)),
    ("spectral_shape", (4, 5)),
)


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


def pool_frequency_aware_shared_feature_map(
    feature_map: torch.Tensor,
    frequency_bins: int = 8,
) -> torch.Tensor:
    """Pool time while retaining a coarse shared-feature frequency profile."""
    if feature_map.ndim != 4:
        raise ValueError(
            "expected a [batch, channels, time, frequency] feature map, "
            f"got {tuple(feature_map.shape)}"
        )
    if frequency_bins < 1:
        raise ValueError("frequency_bins must be positive")
    temporal_mean = feature_map.mean(dim=-2)
    temporal_std = feature_map.std(dim=-2, unbiased=False)
    pooled_mean = F.adaptive_avg_pool1d(temporal_mean, frequency_bins)
    pooled_std = F.adaptive_avg_pool1d(temporal_std, frequency_bins)
    return torch.cat((pooled_mean, pooled_std), dim=-1).flatten(start_dim=1)


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


class FrequencyAwareSharedComponentHead(nn.Module):
    """A compact shared head that preserves coarse frequency structure."""

    def __init__(
        self,
        feature_channels: int = 48,
        frequency_bins: int = 8,
        hidden_features: int = 128,
    ):
        super().__init__()
        self.frequency_bins = frequency_bins
        pooled_features = feature_channels * frequency_bins * 2
        self.regressor = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Linear(pooled_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, len(AVQI_COMPONENT_NAMES)),
        )

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        if pooled_features.ndim != 2:
            raise ValueError(
                f"expected a [batch, features] tensor, got {pooled_features.shape}"
            )
        return self.regressor(pooled_features)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = pool_frequency_aware_shared_feature_map(
            feature_map,
            frequency_bins=self.frequency_bins,
        )
        return self.forward_pooled(pooled)


class TFGridNetBlock(nn.Module):
    """Compact full-band, sub-band, and frame-attention TF-GridNet block.

    Adapted from this workspace's Hybrid-UniSE discriminative branch, with a
    bounded grid and no complex-mask decoder for scalar component regression.
    """

    def __init__(
        self,
        channels: int,
        lstm_hidden: int,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels % attention_heads != 0:
            raise ValueError(
                f"attention_heads={attention_heads} must divide channels={channels}"
            )
        self.intra_norm = nn.GroupNorm(1, channels)
        self.intra_rnn = nn.LSTM(
            channels,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_proj = nn.Linear(lstm_hidden * 2, channels)
        self.sub_norm = nn.GroupNorm(1, channels)
        self.sub_rnn = nn.LSTM(
            channels,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.sub_proj = nn.Linear(lstm_hidden * 2, channels)
        self.attention_norm = nn.GroupNorm(1, channels)
        self.frame_attention = nn.MultiheadAttention(
            channels,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "expected a [batch, channels, frequency, time] tensor, "
                f"got {tuple(features.shape)}"
            )
        batch, channels, frequencies, frames = features.shape

        residual = features
        intra = self.intra_norm(features)
        intra = intra.permute(0, 3, 2, 1).reshape(
            batch * frames,
            frequencies,
            channels,
        )
        intra, _ = self.intra_rnn(intra)
        intra = self.intra_proj(intra)
        intra = intra.reshape(batch, frames, frequencies, channels).permute(
            0,
            3,
            2,
            1,
        )
        features = residual + self.dropout(intra)

        residual = features
        sub = self.sub_norm(features)
        sub = sub.permute(0, 2, 3, 1).reshape(
            batch * frequencies,
            frames,
            channels,
        )
        sub, _ = self.sub_rnn(sub)
        sub = self.sub_proj(sub)
        sub = sub.reshape(batch, frequencies, frames, channels).permute(
            0,
            3,
            1,
            2,
        )
        features = residual + self.dropout(sub)

        residual = features
        frame_tokens = self.attention_norm(features).mean(dim=2).transpose(1, 2)
        attended, _ = self.frame_attention(
            frame_tokens,
            frame_tokens,
            frame_tokens,
            need_weights=False,
        )
        attended = attended + self.dropout(self.attention_ffn(attended))
        attended = attended.transpose(1, 2).unsqueeze(2).expand(
            -1,
            -1,
            frequencies,
            -1,
        )
        return residual + self.dropout(attended)


class GroupedComponentRegressor(nn.Module):
    """Three small heads avoid forcing correlated AVQI concepts into one output."""

    def __init__(self, input_features: int, hidden_features: int = 64):
        super().__init__()
        self.normalization = nn.LayerNorm(input_features)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_features, hidden_features),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_features, len(indices)),
                )
                for name, indices in TFGRID_COMPONENT_GROUPS
            }
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(features)
        return torch.cat(
            [self.heads[name](normalized) for name, _ in TFGRID_COMPONENT_GROUPS],
            dim=-1,
        )


class CompactTFGridComponentEncoder(nn.Module):
    """A bounded TF-GridNet encoder shared by both comparison routes."""

    def __init__(
        self,
        input_channels: int,
        embedding: int = 24,
        lstm_hidden: int = 64,
        num_blocks: int = 2,
        attention_heads: int = 4,
        frequency_bins: int = 32,
        time_bins: int = 64,
    ):
        super().__init__()
        if embedding % 4 != 0:
            raise ValueError(f"embedding={embedding} must be divisible by 4")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if frequency_bins < 1 or time_bins < 1:
            raise ValueError("frequency_bins and time_bins must be positive")
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, embedding, kernel_size=3, padding=1),
            nn.GroupNorm(4, embedding),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TFGridNetBlock(
                    embedding,
                    lstm_hidden,
                    attention_heads=attention_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_features = embedding * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "expected a [batch, channels, frequency, time] tensor, "
                f"got {tuple(features.shape)}"
            )
        features = F.adaptive_avg_pool2d(
            features,
            (self.frequency_bins, self.time_bins),
        )
        features = self.input_projection(features)
        for block in self.blocks:
            features = block(features)
        mean = features.mean(dim=(-2, -1))
        std = features.std(dim=(-2, -1), unbiased=False)
        return torch.cat((mean, std), dim=-1)


class CompactTFGridSharedComponentHead(nn.Module):
    """TF-GridNet dual head attached to a SeMamba++ [B,C,T,F] feature map."""

    def __init__(self, feature_channels: int = 48):
        super().__init__()
        self.encoder = CompactTFGridComponentEncoder(feature_channels)
        self.regressor = GroupedComponentRegressor(self.encoder.output_features)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(
                "expected a [batch, channels, time, frequency] feature map, "
                f"got {tuple(feature_map.shape)}"
            )
        encoded = self.encoder(feature_map.transpose(-2, -1))
        return self.regressor(encoded)


class WaveformSpectrogramFrontend(nn.Module):
    """Differentiable, gain-normalized log-STFT frontend."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        time_bins: int = 256,
    ):
        super().__init__()
        if time_bins < 1:
            raise ValueError("time_bins must be positive")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.time_bins = time_bins
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

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
        normalized = ((log_magnitude - mean) / std.clamp_min(1e-5)).unsqueeze(1)
        return F.adaptive_avg_pool2d(
            normalized,
            (normalized.shape[-2], self.time_bins),
        )


class WaveformComponentPredictor(WaveformSpectrogramFrontend):
    """A small waveform-to-six-components predictor with a differentiable STFT."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        hidden_features: int = 64,
    ):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
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

    def forward_spectrogram(self, log_spectrogram: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(log_spectrogram)
        mean = encoded.mean(dim=(-2, -1))
        std = encoded.std(dim=(-2, -1), unbiased=False)
        return self.regressor(torch.cat((mean, std), dim=-1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_spectrogram(self.log_spectrogram(waveform))


class FrequencyAwareWaveformComponentPredictor(WaveformComponentPredictor):
    """A waveform predictor that keeps an eight-bin LTAS-like profile."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        frequency_bins: int = 8,
        hidden_features: int = 128,
    ):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            hidden_features=hidden_features,
        )
        self.frequency_bins = frequency_bins
        pooled_features = 32 * frequency_bins * 2
        self.regressor = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Linear(pooled_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, len(AVQI_COMPONENT_NAMES)),
        )

    def forward_spectrogram(self, log_spectrogram: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(log_spectrogram)
        temporal_mean = encoded.mean(dim=-1)
        temporal_std = encoded.std(dim=-1, unbiased=False)
        pooled_mean = F.adaptive_avg_pool1d(
            temporal_mean,
            self.frequency_bins,
        )
        pooled_std = F.adaptive_avg_pool1d(
            temporal_std,
            self.frequency_bins,
        )
        pooled = torch.cat((pooled_mean, pooled_std), dim=-1).flatten(start_dim=1)
        return self.regressor(pooled)


class CompactTFGridWaveformComponentPredictor(WaveformSpectrogramFrontend):
    """Compact TF-GridNet waveform predictor with three concept-group heads."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        self.encoder = CompactTFGridComponentEncoder(input_channels=1)
        self.regressor = GroupedComponentRegressor(self.encoder.output_features)

    def forward_spectrogram(self, log_spectrogram: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(log_spectrogram)
        return self.regressor(encoded)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_spectrogram(self.log_spectrogram(waveform))


class ComponentAffineCalibrator(nn.Module):
    """A fixed per-component affine calibration fitted outside this module."""

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        expected = (len(AVQI_COMPONENT_NAMES),)
        if tuple(scale.shape) != expected or tuple(bias.shape) != expected:
            raise ValueError(
                f"expected scale and bias with shape {expected}, "
                f"got {tuple(scale.shape)} and {tuple(bias.shape)}"
            )
        if not torch.isfinite(scale).all() or not torch.isfinite(bias).all():
            raise ValueError("calibration parameters must be finite")
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, raw_components: torch.Tensor) -> torch.Tensor:
        if raw_components.shape[-1] != len(AVQI_COMPONENT_NAMES):
            raise ValueError(
                f"expected six components, got {tuple(raw_components.shape)}"
            )
        return raw_components * self.scale + self.bias


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


def enable_recurrent_input_gradients(module: nn.Module) -> None:
    """Enable cuDNN RNN input backward without activating surrounding dropout."""
    for child in module.modules():
        if isinstance(child, (nn.RNN, nn.GRU, nn.LSTM)):
            if child.dropout != 0.0:
                raise ValueError(
                    "recurrent input-gradient mode requires zero internal dropout"
                )
            child.train()


def freeze_module_for_input_gradient(module: nn.Module) -> None:
    """Freeze weights while retaining CUDA gradients with respect to inputs."""
    freeze_module(module)
    enable_recurrent_input_gradients(module)
