import argparse
import os

import torch
import torchaudio

from model.semambapp import SEMambapp
from model.stfts import mag_phase_stft, mag_phase_istft
from utils import load_config


def load_generator(checkpoint_path, cfg, device):
    model = SEMambapp(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("generator", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_audio(audio_path, target_sr):
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sr)
    return audio


@torch.no_grad()
def enhance_audio(model, audio, cfg, device):
    audio = audio.to(device)
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    audio = audio.unsqueeze(0)

    stft_cfg = cfg["stft_cfg"]
    compress_factor = cfg["model_cfg"]["compress_factor"]
    mag, pha, _ = mag_phase_stft(
        audio,
        stft_cfg["n_fft"],
        stft_cfg["hop_size"],
        stft_cfg["win_size"],
        compress_factor,
    )

    enhanced_mag, enhanced_pha, _ = model(mag, pha)
    enhanced_audio = mag_phase_istft(
        enhanced_mag,
        enhanced_pha,
        stft_cfg["n_fft"],
        stft_cfg["hop_size"],
        stft_cfg["win_size"],
        compress_factor,
    )

    enhanced_audio = enhanced_audio.squeeze(0).cpu()
    peak = enhanced_audio.abs().max()
    if peak > 1.0:
        enhanced_audio = enhanced_audio / peak
    return enhanced_audio.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SEMambapp inference on a single audio file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint.")
    parser.add_argument("--input", required=True, help="Path to input noisy wav/flac.")
    parser.add_argument("--output", required=True, help="Path to output enhanced wav.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sample_rate = cfg["stft_cfg"]["sampling_rate"]
    device = torch.device(args.device)

    model = load_generator(args.checkpoint, cfg, device)
    audio = load_audio(args.input, sample_rate)
    enhanced_audio = enhance_audio(model, audio, cfg, device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torchaudio.save(args.output, enhanced_audio, sample_rate)
    print(f"Saved enhanced audio to {args.output}")


if __name__ == "__main__":
    main()
