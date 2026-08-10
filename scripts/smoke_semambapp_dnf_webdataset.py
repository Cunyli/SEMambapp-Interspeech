import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DNF_REPO = Path(
    os.environ.get("DNF_USE_ROOT", REPO_ROOT.parent / "DNF_USE")
).expanduser().resolve()
if not DNF_REPO.is_dir():
    raise FileNotFoundError(
        f"DNF_USE checkout not found at {DNF_REPO}; set DNF_USE_ROOT"
    )
for path_entry in (str(DNF_REPO), str(REPO_ROOT)):
    if path_entry in sys.path:
        sys.path.remove(path_entry)
sys.path.insert(0, str(REPO_ROOT))

from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft

sys.path.insert(1, str(DNF_REPO))
from dataloader.dnf_webdataset_protocol import DNFHybridStreamDataset, dnf_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test SeMamba++ on DNF WebDatasetStream batches.")
    parser.add_argument("--config", default="configs/train/semambapp_shifted_anechoic_online_v1.yaml")
    parser.add_argument(
        "--split-root",
        default="/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10",
    )
    parser.add_argument(
        "--simulation-config",
        default=str(DNF_REPO / "conf/simulation_train_shifted_anechoic.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=4)
    parser.add_argument("--max-shards-per-role", type=int, default=1)
    parser.add_argument("--clean-shuffle-buffer", type=int, default=2)
    parser.add_argument("--noise-buffer-size", type=int, default=2)
    parser.add_argument("--rir-buffer-size", type=int, default=1)
    parser.add_argument("--cut-duration", type=float, default=0.5)
    parser.add_argument("--added-noise-snr", nargs=2, type=float, default=(0.0, 15.0))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--tiny-hid-feature", type=int, default=16)
    parser.add_argument("--tiny-num-tfmamba", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_semambapp_cfg(path, args):
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    cfg["env_setting"]["seed"] = int(args.seed)
    cfg["training_cfg"]["batch_size"] = int(args.batch_size)
    cfg["training_cfg"]["segment_size"] = int(round(float(args.cut_duration) * int(cfg["stft_cfg"]["sampling_rate"])))
    if not args.full_model:
        cfg["model_cfg"]["hid_feature"] = int(args.tiny_hid_feature)
        cfg["model_cfg"]["num_tfmamba"] = int(args.tiny_num_tfmamba)
    return cfg


def build_loader(args, sample_rate):
    dataset = DNFHybridStreamDataset(
        split_root=args.split_root,
        simulation_config=args.simulation_config,
        target_sample_rate=sample_rate,
        cut_duration=args.cut_duration,
        samples_per_epoch=args.samples_per_epoch,
        clean_shuffle_buffer=args.clean_shuffle_buffer,
        noise_buffer_size=args.noise_buffer_size,
        rir_buffer_size=args.rir_buffer_size,
        shard_shuffle_seed=args.seed,
        max_shards_per_role=args.max_shards_per_role,
        added_noise_snr=args.added_noise_snr,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=dnf_collate,
    )


def stft_features(wav, cfg):
    return mag_phase_stft(
        wav,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
        addeps=True,
    )


def istft_waveform(mag, pha, cfg):
    return mag_phase_istft(
        mag,
        pha,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
    )


def crop_like(reference, estimate):
    length = min(reference.size(-1), estimate.size(-1))
    return reference[..., :length], estimate[..., :length]


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def finite_grad_summary(model):
    total = 0
    finite = 0
    max_abs = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += grad.numel()
        finite += int(torch.isfinite(grad).sum().item())
        if grad.numel() > 0:
            max_abs = max(max_abs, float(grad.abs().max().item()))
    return {"finite": finite, "total": total, "max_abs": max_abs}


class DNFSEMambappNoGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.speech = SEMambapp(cfg)
        self.noise = SEMambapp(cfg)
        self.cfg = cfg

    def forward(self, degraded_mag, degraded_pha):
        speech_mag, speech_pha, speech_com = self.speech(degraded_mag, degraded_pha)
        noise_mag, noise_pha, noise_com = self.noise(degraded_mag, degraded_pha)
        speech_wav = istft_waveform(speech_mag, speech_pha, self.cfg)
        noise_wav = istft_waveform(noise_mag, noise_pha, self.cfg)
        speech_wav, noise_wav = crop_like(speech_wav, noise_wav)
        final_wav = speech_wav - noise_wav
        return {
            "speech_mag": speech_mag,
            "speech_pha": speech_pha,
            "speech_com": speech_com,
            "speech_wav": speech_wav,
            "noise_mag": noise_mag,
            "noise_pha": noise_pha,
            "noise_com": noise_com,
            "noise_wav": noise_wav,
            "final_wav": final_wav,
        }


def standard_semambapp_smoke(cfg, degraded_mag, degraded_pha, s_noisy_wav, s_noisy_mag, device):
    model = SEMambapp(cfg).to(device)
    model.train()
    mag, pha, _ = model(degraded_mag, degraded_pha)
    wav = istft_waveform(mag, pha, cfg)
    target, wav = crop_like(s_noisy_wav, wav)
    loss_time = F.l1_loss(wav, target)
    loss_mag = F.mse_loss(mag, s_noisy_mag)
    loss = loss_time + loss_mag
    assert_finite("standard_loss", loss)
    loss.backward()
    grad = finite_grad_summary(model)
    if grad["total"] == 0 or grad["finite"] != grad["total"]:
        raise ValueError(f"standard grad is not fully finite: {grad}")
    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_time": float(loss_time.detach().cpu().item()),
        "loss_mag": float(loss_mag.detach().cpu().item()),
        "wav_shape": tuple(wav.shape),
        "grad": grad,
    }


def dnf_semambapp_smoke(cfg, degraded_mag, degraded_pha, s_noisy_wav, s_noisy_mag, added_noise_wav, added_noise_mag, clean_wav, device):
    model = DNFSEMambappNoGAN(cfg).to(device)
    model.train()
    outputs = model(degraded_mag, degraded_pha)
    s_target, speech_wav = crop_like(s_noisy_wav, outputs["speech_wav"])
    n_target, noise_wav = crop_like(added_noise_wav, outputs["noise_wav"])
    clean_target, final_wav = crop_like(clean_wav, outputs["final_wav"])
    loss_speech_time = F.l1_loss(speech_wav, s_target)
    loss_noise_time = F.l1_loss(noise_wav, n_target)
    loss_speech_mag = F.mse_loss(outputs["speech_mag"], s_noisy_mag)
    loss_noise_mag = F.mse_loss(outputs["noise_mag"], added_noise_mag)
    loss = loss_speech_time + loss_noise_time + loss_speech_mag + loss_noise_mag
    final_l1_to_clean = F.l1_loss(final_wav, clean_target)
    assert_finite("dnf_loss", loss)
    assert_finite("dnf_final_l1_to_clean", final_l1_to_clean)
    loss.backward()
    grad = finite_grad_summary(model)
    if grad["total"] == 0 or grad["finite"] != grad["total"]:
        raise ValueError(f"dnf grad is not fully finite: {grad}")
    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_speech_time": float(loss_speech_time.detach().cpu().item()),
        "loss_noise_time": float(loss_noise_time.detach().cpu().item()),
        "loss_speech_mag": float(loss_speech_mag.detach().cpu().item()),
        "loss_noise_mag": float(loss_noise_mag.detach().cpu().item()),
        "final_l1_to_clean_diag": float(final_l1_to_clean.detach().cpu().item()),
        "speech_wav_shape": tuple(speech_wav.shape),
        "noise_wav_shape": tuple(noise_wav.shape),
        "final_wav_shape": tuple(final_wav.shape),
        "grad": grad,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_semambapp_cfg(args.config, args)
    device = torch.device(args.device)
    loader = build_loader(args, int(cfg["stft_cfg"]["sampling_rate"]))
    batch = next(iter(loader))
    degraded_wav = batch["degraded_wav"].to(device)
    s_noisy_wav = batch["s_noisy_wav"].to(device)
    added_noise_wav = batch["added_noise_wav"].to(device)
    clean_wav = batch["clean_wav"].to(device)

    degraded_mag, degraded_pha, _ = stft_features(degraded_wav, cfg)
    s_noisy_mag, _, _ = stft_features(s_noisy_wav, cfg)
    added_noise_mag, _, _ = stft_features(added_noise_wav, cfg)
    relation_residual = degraded_wav - s_noisy_wav - added_noise_wav

    for name, value in {
        "degraded_wav": degraded_wav,
        "s_noisy_wav": s_noisy_wav,
        "added_noise_wav": added_noise_wav,
        "clean_wav": clean_wav,
        "degraded_mag": degraded_mag,
        "s_noisy_mag": s_noisy_mag,
        "added_noise_mag": added_noise_mag,
        "relation_residual": relation_residual,
    }.items():
        assert_finite(name, value)

    print("stage=data")
    print("batch_shapes", {key: tuple(value.shape) for key, value in batch.items() if torch.is_tensor(value)})
    print("utterance_id", batch["utterance_id"][: min(3, len(batch["utterance_id"]))])
    print("relation_residual_mean_abs", float(relation_residual.abs().mean().detach().cpu().item()))
    print("relation_residual_max_abs", float(relation_residual.abs().max().detach().cpu().item()))
    print("model_cfg", {"hid_feature": cfg["model_cfg"]["hid_feature"], "num_tfmamba": cfg["model_cfg"]["num_tfmamba"]})
    if args.data_only:
        return

    standard = standard_semambapp_smoke(cfg, degraded_mag, degraded_pha, s_noisy_wav, s_noisy_mag, device)
    print("stage=standard_semambapp_no_gan")
    print("standard", standard)

    dnf = dnf_semambapp_smoke(
        cfg,
        degraded_mag,
        degraded_pha,
        s_noisy_wav,
        s_noisy_mag,
        added_noise_wav,
        added_noise_mag,
        clean_wav,
        device,
    )
    print("stage=dnf_semambapp_no_gan")
    print("dnf", dnf)
    print("stage=done")


if __name__ == "__main__":
    main()
