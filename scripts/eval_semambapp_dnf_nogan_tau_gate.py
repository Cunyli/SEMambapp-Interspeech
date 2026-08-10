import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.semambapp import SEMambapp
from model.stfts import mag_phase_istft, mag_phase_stft


AVQI_PYTHON = Path("/scratch/work/lil14/.conda_envs/avqi/bin/python")
AVQI_SCRIPT = Path("/scratch/work/lil14/Hybrid_Unise/scripts/validation_selected_tau_free_run.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Run no-GAN SeMamba++/DNF TAU AVQI and SV signal gate.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-csv", type=Path, required=True)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("auto", "standard", "dnf"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speaker-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def speaker_group(row: dict[str, str]) -> str:
    label = row.get("label", "")
    if label == "healthy":
        return "health"
    if label == "patient":
        return "pathology"
    speaker = row.get("speaker_id", "")
    return "health" if speaker.startswith(("V", "HC_")) else "pathology"


def limited_pair_rows(pair_csv: Path, speaker_limit: int) -> list[dict[str, str]]:
    rows = read_csv(pair_csv)
    if speaker_limit <= 0:
        return rows
    selected = []
    speakers = []
    for row in rows:
        speaker = row["speaker_id"]
        if speaker not in speakers:
            if len(speakers) >= speaker_limit:
                continue
            speakers.append(speaker)
        if speaker in speakers:
            selected.append(row)
    return selected


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sample_rate = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
    return wav


def stft_features(wav: torch.Tensor, cfg: dict):
    return mag_phase_stft(
        wav,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
        addeps=True,
    )


def istft_waveform(mag: torch.Tensor, pha: torch.Tensor, cfg: dict) -> torch.Tensor:
    return mag_phase_istft(
        mag,
        pha,
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
        cfg["model_cfg"]["compress_factor"],
    )


def crop_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(left.size(-1), right.size(-1))
    return left[..., :length], right[..., :length]


def strip_prefix_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix_dot = f"{prefix}."
    return {key.removeprefix(prefix_dot): value for key, value in state.items() if key.startswith(prefix_dot)}


def load_models(checkpoint_path: Path, requested_mode: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["cfg"]
    state = checkpoint["model"]
    mode = checkpoint.get("args", {}).get("mode", requested_mode)
    if requested_mode != "auto":
        mode = requested_mode
    if mode == "standard":
        model = SEMambapp(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return mode, cfg, {"standard": model}
    if mode == "dnf":
        speech = SEMambapp(cfg).to(device)
        noise = SEMambapp(cfg).to(device)
        speech.load_state_dict(strip_prefix_state(state, "speech"), strict=True)
        noise.load_state_dict(strip_prefix_state(state, "noise"), strict=True)
        speech.eval()
        noise.eval()
        return mode, cfg, {"speech": speech, "noise": noise}
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


@torch.inference_mode()
def enhance(audio: torch.Tensor, mode: str, cfg: dict, models: dict[str, SEMambapp], device: torch.device) -> torch.Tensor:
    audio = audio.to(device)
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    audio = audio.unsqueeze(0)

    eps = 1e-9
    input_peak = audio.abs().max()
    input_scale = 0.9 / (input_peak + eps)
    normalized = audio * input_scale
    mag, pha, _ = stft_features(normalized, cfg)
    if mode == "standard":
        out_mag, out_pha, _ = models["standard"](mag, pha)
        enhanced = istft_waveform(out_mag, out_pha, cfg)
    else:
        speech_mag, speech_pha, _ = models["speech"](mag, pha)
        noise_mag, noise_pha, _ = models["noise"](mag, pha)
        speech_wav = istft_waveform(speech_mag, speech_pha, cfg)
        noise_wav = istft_waveform(noise_mag, noise_pha, cfg)
        speech_wav, noise_wav = crop_pair(speech_wav, noise_wav)
        enhanced = speech_wav - noise_wav

    enhanced = enhanced / input_scale
    enhanced = enhanced.squeeze(0).cpu()
    peak = enhanced.abs().max()
    if peak > 1.0:
        enhanced = enhanced * (0.99 / peak)
    return enhanced.unsqueeze(0)


def signal_stats(reference: torch.Tensor, enhanced: torch.Tensor, threshold: float = 0.01) -> dict[str, float]:
    ref = reference.flatten().float()
    enh = enhanced.flatten().float()
    length = min(ref.numel(), enh.numel())
    ref = ref[:length]
    enh = enh[:length]
    ref_rms = float(torch.sqrt(torch.mean(ref**2) + 1e-12).item())
    enh_rms = float(torch.sqrt(torch.mean(enh**2) + 1e-12).item())
    ref_active = float((ref.abs() > threshold).float().mean().item())
    enh_active = float((enh.abs() > threshold).float().mean().item())
    return {
        "duration_seconds": length / 16000.0,
        "reference_rms": ref_rms,
        "enhanced_rms": enh_rms,
        "rms_ratio_to_reference": enh_rms / (ref_rms + 1e-12),
        "reference_active_ratio": ref_active,
        "enhanced_active_ratio": enh_active,
        "active_ratio_delta": enh_active - ref_active,
        "enhanced_peak": float(enh.abs().max().item()) if length else 0.0,
    }


def summarize_signal(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["sample_group"], row["task"])].append(row)
    summary = []
    for (sample_group, task), bucket in sorted(buckets.items()):
        for key in ("rms_ratio_to_reference", "active_ratio_delta", "enhanced_peak"):
            values = [float(row[key]) for row in bucket]
            summary.append(
                {
                    "sample_group": sample_group,
                    "task": task,
                    "metric": key,
                    "n": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.output_dir / "enhanced"
    degraded_dir = args.output_dir / "degraded"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    mode, cfg, models = load_models(args.checkpoint, args.mode, device)
    sample_rate = int(cfg["stft_cfg"]["sampling_rate"])
    rows = limited_pair_rows(args.pair_csv, args.speaker_limit)
    pair_csv = args.output_dir / "pair_manifest.csv"
    write_csv(pair_csv, rows, list(rows[0].keys()))

    signal_rows = []
    for index, row in enumerate(rows, start=1):
        audio = load_audio(Path(row["noisy_filepath"]), sample_rate)
        enhanced = enhance(audio, mode, cfg, models, device)
        torchaudio.save(str(enhanced_dir / f'{row["uid"]}.wav'), enhanced, sample_rate)
        target_degraded = degraded_dir / f'{row["uid"]}.wav'
        if target_degraded.exists() or target_degraded.is_symlink():
            target_degraded.unlink()
        try:
            target_degraded.symlink_to(Path(row["noisy_filepath"]))
        except OSError:
            shutil.copy2(row["noisy_filepath"], target_degraded)
        stats = signal_stats(audio, enhanced)
        signal_rows.append(
            {
                "uid": row["uid"],
                "speaker_id": row["speaker_id"],
                "task": row["task"],
                "sample_group": row.get("sample_group") or speaker_group(row),
                **stats,
            }
        )
        if index % 20 == 0 or index == len(rows):
            print(f"enhanced {index}/{len(rows)}", flush=True)

    signal_fields = [
        "uid",
        "speaker_id",
        "task",
        "sample_group",
        "duration_seconds",
        "reference_rms",
        "enhanced_rms",
        "rms_ratio_to_reference",
        "reference_active_ratio",
        "enhanced_active_ratio",
        "active_ratio_delta",
        "enhanced_peak",
    ]
    write_csv(args.output_dir / "signal_guard.csv", signal_rows, signal_fields)
    write_csv(
        args.output_dir / "signal_guard_summary.csv",
        summarize_signal(signal_rows),
        ["sample_group", "task", "metric", "n", "mean", "min", "max"],
    )

    score_command = [
        str(AVQI_PYTHON),
        str(AVQI_SCRIPT),
        "score",
        "--pair-csv",
        str(pair_csv),
        "--enhanced-dir",
        str(enhanced_dir),
        "--clean-cache",
        str(args.clean_cache),
        "--workers",
        str(args.workers),
    ]
    scored = subprocess.run(score_command, check=True, capture_output=True, text=True)
    metrics = json.loads(scored.stdout.strip().splitlines()[-1])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "mode": mode,
                "checkpoint": str(args.checkpoint),
                "pair_csv": str(pair_csv),
                "clean_cache": str(args.clean_cache),
                "enhanced_dir": str(enhanced_dir),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "mode": mode, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
