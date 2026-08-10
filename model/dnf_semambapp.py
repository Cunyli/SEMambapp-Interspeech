import torch
from einops import rearrange
from torch import nn

from .encdec import DenseEncoder, MagDecoder, PhaseDecoder
from .semambapp import SEMambapp_bottleneck


class DNFSEMambapp(nn.Module):
    """Shared-trunk, two-output SeMamba++ model for DNF experiments.

    The paper requires a two-output network but does not prescribe how the
    backbone is shared. Sharing the encoder and Mamba blocks keeps the DNF
    comparison close to the capacity of the one-output SeMamba++ baseline.
    Paper-specific losses and projection subtraction live in ``dnf_paper``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        num_blocks = cfg["model_cfg"].get("num_tfmamba")
        self.num_tscblocks = int(num_blocks) if num_blocks is not None else 4

        self.dense_encoder = DenseEncoder(cfg)
        self.TSMamba = nn.ModuleList(
            [SEMambapp_bottleneck(cfg) for _ in range(self.num_tscblocks)]
        )
        self.speech_mag_decoder = MagDecoder(cfg)
        self.speech_phase_decoder = PhaseDecoder(cfg)
        self.noise_mag_decoder = MagDecoder(cfg)
        self.noise_phase_decoder = PhaseDecoder(cfg)

    @staticmethod
    def _decode_branch(
        features: torch.Tensor,
        mag_decoder: MagDecoder,
        phase_decoder: PhaseDecoder,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude = rearrange(
            mag_decoder(features),
            "b c t f -> b f t c",
        ).squeeze(-1)
        phase = rearrange(
            phase_decoder(features),
            "b c t f -> b f t c",
        ).squeeze(-1)
        complex_components = torch.stack(
            (magnitude * torch.cos(phase), magnitude * torch.sin(phase)),
            dim=-1,
        )
        return magnitude, phase, complex_components

    def forward(
        self,
        noisy_mag: torch.Tensor,
        noisy_pha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        magnitude_features = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)
        phase_features = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)
        features = self.dense_encoder(
            torch.cat((magnitude_features, phase_features), dim=1)
        )
        for block in self.TSMamba:
            features = block(features)

        speech_mag, speech_pha, speech_com = self._decode_branch(
            features,
            self.speech_mag_decoder,
            self.speech_phase_decoder,
        )
        noise_mag, noise_pha, noise_com = self._decode_branch(
            features,
            self.noise_mag_decoder,
            self.noise_phase_decoder,
        )
        return {
            "speech_mag": speech_mag,
            "speech_pha": speech_pha,
            "speech_com": speech_com,
            "noise_mag": noise_mag,
            "noise_pha": noise_pha,
            "noise_com": noise_com,
        }
