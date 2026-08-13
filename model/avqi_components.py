"""Small differentiable models for AVQI-component experiments.

The models predict the six terms used by AVQI v03.01. Jitter is deliberately
excluded because it is diagnostic-only in the verified Praat implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class PretrainedFullTFGridWaveformComponentPredictor(nn.Module):
    """Full Hybrid-UniSE TF-GridNet backbone adapted to component regression.

    The encoder and all eight TF-GridNet blocks match the discriminative branch
    used by Hybrid-UniSE. The pretrained prefix is frozen; the final block and
    a grouped six-component regressor are adapted on the bounded TAU label bank.
    """

    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 160,
        time_bins: int = 64,
        embedding: int = 64,
        lstm_hidden: int = 256,
        num_blocks: int = 8,
        adaptation_blocks: int = 1,
        attention_heads: int = 4,
    ):
        super().__init__()
        if n_fft < 2 or n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even integer")
        if hop_length < 1 or time_bins < 1:
            raise ValueError("hop_length and time_bins must be positive")
        if not 1 <= adaptation_blocks <= num_blocks:
            raise ValueError(
                "adaptation_blocks must be between one and num_blocks"
            )
        if embedding % attention_heads != 0:
            raise ValueError(
                f"attention_heads={attention_heads} must divide embedding={embedding}"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.time_bins = time_bins
        self.frequency_bins = n_fft // 2 + 1
        self.prefix_blocks = num_blocks - adaptation_blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embedding, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TFGridNetBlock(
                    embedding,
                    lstm_hidden,
                    attention_heads=attention_heads,
                    dropout=0.0,
                )
                for _ in range(num_blocks)
            ]
        )
        self.regressor = GroupedComponentRegressor(embedding * 2)
        self.register_buffer(
            "window",
            torch.hann_window(n_fft, periodic=True),
            persistent=False,
        )
        self.pretrained_backbone_receipt: dict[str, Any] | None = None

    def spectrogram_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                f"expected a [batch, time] waveform, got {tuple(waveform.shape)}"
            )
        if waveform.shape[-1] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[-1]))
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        normalized = waveform * (0.5 / peak)
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
        features = torch.stack(
            (spectrum.real, spectrum.imag, spectrum.abs()),
            dim=1,
        )
        return features

    def encode_frozen_prefix(self, spectrogram_features: torch.Tensor) -> torch.Tensor:
        features = self.encoder(spectrogram_features)
        for block in self.blocks[: self.prefix_blocks]:
            features = block(features)
        # Preserve the input grid through the frozen pretrained prefix. Pooling
        # only here keeps the checkpoint's feature extraction intact while
        # bounding the one trainable block and its cache.
        return F.adaptive_avg_pool2d(
            features,
            (self.frequency_bins, self.time_bins),
        )

    def forward_cached_prefix(self, prefix_features: torch.Tensor) -> torch.Tensor:
        features = prefix_features
        for block in self.blocks[self.prefix_blocks :]:
            features = block(features)
        mean = features.mean(dim=(-2, -1))
        std = features.std(dim=(-2, -1), unbiased=False)
        return self.regressor(torch.cat((mean, std), dim=-1))

    def forward_spectrogram(self, spectrogram_features: torch.Tensor) -> torch.Tensor:
        prefix = self.encode_frozen_prefix(spectrogram_features)
        return self.forward_cached_prefix(prefix)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_spectrogram(self.spectrogram_features(waveform))

    def freeze_pretrained_prefix(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.blocks[: self.prefix_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)

    def load_hybrid_discriminative_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        architecture = checkpoint.get("hybrid_architecture_config", {}).get(
            "discriminative",
            {},
        )
        expected_architecture = {
            "num_blocks": len(self.blocks),
            "embedding": self.encoder[0].out_channels,
            "lstm_hidden": self.blocks[0].intra_rnn.hidden_size,
            "attention_heads": self.blocks[0].frame_attention.num_heads,
            "dropout": 0.0,
        }
        if architecture and architecture != expected_architecture:
            raise ValueError(
                "Hybrid-UniSE TF-GridNet architecture mismatch: "
                f"{architecture} != {expected_architecture}"
            )
        mapped: dict[str, torch.Tensor] = {}
        prefix = "discriminative."
        replacements = {
            ".attn_norm.": ".attention_norm.",
            ".frame_attn.": ".frame_attention.",
            ".attn_ffn.": ".attention_ffn.",
        }
        for raw_key, value in state_dict.items():
            if not raw_key.startswith(prefix):
                continue
            key = raw_key[len(prefix) :]
            if key.startswith("mask_head."):
                continue
            for source, target in replacements.items():
                key = key.replace(source, target)
            mapped[key] = value
        expected_keys = {
            key
            for key in self.state_dict()
            if key.startswith("encoder.") or key.startswith("blocks.")
        }
        if set(mapped) != expected_keys:
            raise ValueError(
                "pretrained TF-GridNet key mismatch: "
                f"missing={sorted(expected_keys - set(mapped))[:10]}, "
                f"extra={sorted(set(mapped) - expected_keys)[:10]}"
            )
        load_result = self.load_state_dict(mapped, strict=False)
        unexpected = list(load_result.unexpected_keys)
        missing = [
            key
            for key in load_result.missing_keys
            if not key.startswith("regressor.")
        ]
        if missing or unexpected:
            raise ValueError(
                f"pretrained TF-GridNet load mismatch: {missing=} {unexpected=}"
            )
        receipt = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_stage": checkpoint.get("hybrid_stage"),
            "architecture": expected_architecture,
            "loaded_tensor_count": len(mapped),
            "loaded_parameter_count": sum(value.numel() for value in mapped.values()),
            "adaptation_blocks": len(self.blocks) - self.prefix_blocks,
            "time_pool_position": "after_frozen_prefix",
        }
        self.pretrained_backbone_receipt = receipt
        self.freeze_pretrained_prefix()
        return receipt


class DifferentiableAVQIComponentEstimator(nn.Module):
    """Direct, exact-inspired approximations of all six AVQI components.

    This route has no neural predictor. It uses soft voicing, autocorrelation,
    cepstral prominence, adjacent-frame amplitude variation, and LTAS band
    statistics. A positive affine map fitted on the training split only aligns
    each approximation to the corresponding exact Praat label.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_length: int = 640,
        hop_length: int = 160,
        n_fft: int = 1024,
        max_frames: int = 256,
        peak_temperature: float = 30.0,
    ):
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_length <= 1 or hop_length <= 0:
            raise ValueError("frame and hop lengths must be positive")
        if n_fft < frame_length:
            raise ValueError("n_fft must be at least frame_length")
        if max_frames < 2 or peak_temperature <= 0.0:
            raise ValueError("max_frames and peak_temperature must be positive")
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_frames = max_frames
        self.peak_temperature = peak_temperature
        self.register_buffer(
            "window",
            torch.hann_window(frame_length, periodic=True),
            persistent=False,
        )
        self.register_buffer(
            "alignment_scale",
            torch.ones(len(AVQI_COMPONENT_NAMES)),
        )
        self.register_buffer(
            "alignment_bias",
            torch.zeros(len(AVQI_COMPONENT_NAMES)),
        )

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.reshape(-1)
        if waveform.numel() < self.frame_length + self.hop_length:
            waveform = F.pad(
                waveform,
                (0, self.frame_length + self.hop_length - waveform.numel()),
            )
        waveform = waveform - waveform.mean()
        spectrum = torch.fft.rfft(waveform)
        frequencies = torch.fft.rfftfreq(
            waveform.numel(),
            d=1.0 / self.sample_rate,
            device=waveform.device,
        )
        response = torch.sigmoid((frequencies - 34.0) / 4.0)
        response = response * (frequencies > 0.0).to(response.dtype)
        highpassed = torch.fft.irfft(
            spectrum * response,
            n=waveform.numel(),
        )
        rms = highpassed.square().mean().sqrt().clamp_min(1e-5)
        return highpassed / rms

    def _frames(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = waveform.unfold(
            0,
            self.frame_length,
            self.hop_length,
        )
        if frames.shape[0] > self.max_frames:
            indices = torch.linspace(
                0,
                frames.shape[0] - 1,
                self.max_frames,
                device=frames.device,
            ).round().long()
            frames = frames.index_select(0, indices)
        return frames

    def _raw_one(self, waveform: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare(waveform)
        frames = self._frames(prepared)
        centered = frames - frames.mean(dim=-1, keepdim=True)
        frame_power = centered.square().mean(dim=-1).clamp_min(1e-10)
        relative_power_db = 10.0 * torch.log10(
            frame_power / frame_power.mean().clamp_min(1e-10)
        )
        power_weight = torch.sigmoid((relative_power_db + 5.0) / 2.0)
        normalized_frames = centered / frame_power.sqrt().unsqueeze(-1)
        adjacent_product = (
            normalized_frames[:, 1:] * normalized_frames[:, :-1]
        )
        crossing_probability = torch.sigmoid(-20.0 * adjacent_product)
        crossing_rate_hz = crossing_probability.mean(dim=-1) * self.sample_rate
        zcr_weight = torch.sigmoid((3_000.0 - crossing_rate_hz) / 500.0)
        voicing_weight = (power_weight * zcr_weight).clamp_min(1e-4)

        windowed = centered * self.window.to(dtype=waveform.dtype)
        spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
        power = spectrum.abs().square().clamp_min(1e-12)
        autocorrelation = torch.fft.irfft(power, n=self.n_fft, dim=-1)
        autocorrelation = autocorrelation / autocorrelation[:, :1].clamp_min(1e-10)
        lag_min = max(1, int(self.sample_rate / 600.0))
        lag_max = min(
            self.n_fft // 2 - 1,
            int(self.sample_rate / 60.0),
        )
        pitch_correlation = autocorrelation[:, lag_min : lag_max + 1]
        pitch_weights = torch.softmax(
            self.peak_temperature * pitch_correlation,
            dim=-1,
        )
        periodicity = (pitch_weights * pitch_correlation).sum(dim=-1)
        periodicity = periodicity.clamp(1e-4, 0.9999)
        frame_hnr = 10.0 * torch.log10(periodicity / (1.0 - periodicity))
        hnr = self._weighted_mean(frame_hnr, voicing_weight)

        cepstrum = torch.fft.irfft(torch.log(power), n=self.n_fft, dim=-1)
        quefrency = torch.arange(
            self.n_fft,
            device=waveform.device,
            dtype=waveform.dtype,
        ) / float(self.sample_rate)
        quefrency_mask = (quefrency >= 1.0 / 330.0) & (
            quefrency <= 1.0 / 60.0
        )
        cepstral_band = cepstrum[:, quefrency_mask]
        quefrency_band = quefrency[quefrency_mask]
        cepstral_weights = torch.softmax(
            self.peak_temperature * cepstral_band,
            dim=-1,
        )
        soft_peak = (cepstral_weights * cepstral_band).sum(dim=-1)
        soft_peak_quefrency = (
            cepstral_weights * quefrency_band.unsqueeze(0)
        ).sum(dim=-1)
        centered_quefrency = quefrency_band - quefrency_band.mean()
        baseline_slope = (
            cepstral_band * centered_quefrency.unsqueeze(0)
        ).sum(dim=-1) / centered_quefrency.square().sum().clamp_min(1e-10)
        baseline_at_peak = cepstral_band.mean(dim=-1) + baseline_slope * (
            soft_peak_quefrency - quefrency_band.mean()
        )
        frame_cpps = 10.0 * (soft_peak - baseline_at_peak) / torch.log(
            waveform.new_tensor(10.0)
        )
        cpps = self._weighted_mean(frame_cpps, voicing_weight)

        amplitude = frame_power.sqrt()
        pair_weight = (
            voicing_weight[1:] * voicing_weight[:-1]
        ).sqrt().clamp_min(1e-4)
        amplitude_first = amplitude[:-1]
        amplitude_second = amplitude[1:]
        shimmer_percent_frames = (
            200.0
            * (amplitude_second - amplitude_first).abs()
            / (amplitude_second + amplitude_first).clamp_min(1e-8)
        )
        shimmer_db_frames = (
            20.0
            * torch.log10(
                amplitude_second.clamp_min(1e-8)
                / amplitude_first.clamp_min(1e-8)
            ).abs()
        )
        shimmer_percent = self._weighted_mean(
            shimmer_percent_frames,
            pair_weight,
        )
        shimmer_db = self._weighted_mean(shimmer_db_frames, pair_weight)

        mean_power = (
            power * voicing_weight.unsqueeze(-1)
        ).sum(dim=0) / voicing_weight.sum().clamp_min(1e-8)
        frequencies = torch.fft.rfftfreq(
            self.n_fft,
            d=1.0 / self.sample_rate,
            device=waveform.device,
        )
        low_band = (frequencies >= 34.0) & (frequencies < 1_000.0)
        high_band = (frequencies >= 1_000.0) & (
            frequencies <= self.sample_rate / 2.0
        )
        low_energy = mean_power[low_band].mean().clamp_min(1e-12)
        high_energy = mean_power[high_band].mean().clamp_min(1e-12)
        slope = 10.0 * torch.log10(high_energy / low_energy)

        trend_band = (frequencies >= 34.0) & (
            frequencies <= self.sample_rate / 2.0
        )
        trend_frequency = frequencies[trend_band] / (self.sample_rate / 2.0)
        trend_power_db = 10.0 * torch.log10(mean_power[trend_band].clamp_min(1e-12))
        centered_frequency = trend_frequency - trend_frequency.mean()
        trend_slope = (
            centered_frequency
            * (trend_power_db - trend_power_db.mean())
        ).sum() / centered_frequency.square().sum().clamp_min(1e-10)
        trend_line_db = trend_power_db.mean() + trend_slope * centered_frequency
        trend_linear = torch.pow(
            waveform.new_tensor(10.0),
            trend_line_db.clamp(-120.0, 120.0) / 10.0,
        )
        trend_boundary = 1_000.0 / (self.sample_rate / 2.0)
        trend_low = trend_linear[trend_frequency < trend_boundary]
        trend_high = trend_linear[trend_frequency >= trend_boundary]
        tilt = 10.0 * torch.log10(
            trend_high.mean().clamp_min(1e-12)
            / trend_low.mean().clamp_min(1e-12)
        )
        components = torch.stack(
            (cpps, hnr, shimmer_percent, shimmer_db, slope, tilt)
        )
        if not torch.isfinite(components).all():
            raise ValueError("non-finite differentiable AVQI components")
        return components

    def raw_components(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                f"expected a [batch, time] waveform, got {tuple(waveform.shape)}"
            )
        return torch.stack([self._raw_one(row) for row in waveform])

    def fit_alignment(
        self,
        raw_features: torch.Tensor,
        raw_targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> dict[str, list[float]]:
        if raw_features.shape != raw_targets.shape:
            raise ValueError(
                f"feature and target shapes differ: {raw_features.shape} != "
                f"{raw_targets.shape}"
            )
        normalized_target = (raw_targets - target_mean) / target_scale.clamp_min(1e-8)
        centered_features = raw_features - raw_features.mean(dim=0)
        centered_targets = normalized_target - normalized_target.mean(dim=0)
        variance = centered_features.square().mean(dim=0).clamp_min(1e-8)
        scale = (
            (centered_features * centered_targets).mean(dim=0) / variance
        ).clamp_min(1e-4)
        bias = normalized_target.mean(dim=0) - scale * raw_features.mean(dim=0)
        if not torch.isfinite(scale).all() or not torch.isfinite(bias).all():
            raise ValueError("non-finite direct-component alignment")
        self.alignment_scale.copy_(scale.detach())
        self.alignment_bias.copy_(bias.detach())
        return {
            "scale": scale.detach().cpu().tolist(),
            "bias": bias.detach().cpu().tolist(),
        }

    def forward_proxy_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        return raw_features * self.alignment_scale + self.alignment_bias

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_proxy_features(self.raw_components(waveform))


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
