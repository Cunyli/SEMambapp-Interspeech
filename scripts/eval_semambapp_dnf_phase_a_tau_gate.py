import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.dnf_paper import dnf_output_eq14, sdr_loss_eq5, si_sdr_loss
from model.stfts import mag_phase_istft, mag_phase_stft


AVQI_PYTHON = Path("/scratch/work/lil14/.conda_envs/avqi/bin/python")
AVQI_SCRIPT = Path("/scratch/work/lil14/Hybrid_Unise/scripts/validation_selected_tau_free_run.py")
REQUIRED_SAMPLE_GROUPS = (
    "healthy_low",
    "pathological_mild",
    "pathological_severe",
)
REQUIRED_TASKS = ("cs", "sv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Phase-A Standard/DNF Eq.14 TAU AVQI and signal gate."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--controlled-comparison",
        type=Path,
        required=True,
        help="Passing Phase-A controlled comparison authorizing this TAU screen.",
    )
    parser.add_argument("--pair-csv", type=Path, required=True)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("auto", "standard", "dnf"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speakers-per-group", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def stable_rank(seed: int, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def stratified_pair_rows(
    pair_csv: Path,
    speakers_per_group: int,
    selection_seed: int,
) -> tuple[list[dict[str, str]], dict]:
    rows = read_csv(pair_csv)
    if not rows:
        raise ValueError(f"empty TAU pair manifest: {pair_csv}")
    if speakers_per_group <= 0:
        raise ValueError("speakers_per_group must be positive")
    speakers_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = row.get("sample_group") or speaker_group(row)
        if group in REQUIRED_SAMPLE_GROUPS:
            speakers_by_group[group].add(row["speaker_id"])
    selected_speakers = {}
    for group in REQUIRED_SAMPLE_GROUPS:
        speakers = sorted(
            speakers_by_group[group],
            key=lambda speaker: (
                stable_rank(selection_seed, group, speaker),
                speaker,
            ),
        )
        if len(speakers) < speakers_per_group:
            raise ValueError(
                f"TAU group {group!r} has {len(speakers)} speakers, "
                f"need {speakers_per_group}"
            )
        selected_speakers[group] = speakers[:speakers_per_group]
    selected = [
        row
        for row in rows
        if row["speaker_id"]
        in selected_speakers.get(
            row.get("sample_group") or speaker_group(row),
            [],
        )
    ]
    task_counts = defaultdict(int)
    for row in selected:
        group = row.get("sample_group") or speaker_group(row)
        task_counts[(group, row["task"])] += 1
    missing = [
        (group, task)
        for group in REQUIRED_SAMPLE_GROUPS
        for task in REQUIRED_TASKS
        if task_counts[(group, task)] == 0
    ]
    if missing:
        raise ValueError(f"stratified TAU selection misses strata: {missing}")
    receipt = {
        "schema_version": "dnf-phase-a-tau-selection-v1",
        "selection_seed": selection_seed,
        "speakers_per_group": speakers_per_group,
        "selected_speakers": selected_speakers,
        "required_sample_groups": list(REQUIRED_SAMPLE_GROUPS),
        "required_tasks": list(REQUIRED_TASKS),
        "row_count": len(selected),
        "stratum_counts": {
            f"{group}/{task}": task_counts[(group, task)]
            for group in REQUIRED_SAMPLE_GROUPS
            for task in REQUIRED_TASKS
        },
    }
    return selected, receipt


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


def load_models(
    checkpoint_path: Path,
    requested_mode: str,
    device: torch.device,
    controlled_comparison_path: Path,
):
    # Mamba is required only when a checkpoint model is actually constructed.
    from model.dnf_semambapp import DNFSEMambapp
    from model.semambapp import SEMambapp

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    state = checkpoint["model"]
    mode = checkpoint.get("args", {}).get("mode", requested_mode)
    if requested_mode != "auto":
        mode = requested_mode
    metadata = checkpoint.get("metadata", {})
    required_metadata = {
        "phase": "A",
        "mode": mode,
        "scratch_init": True,
        "init_checkpoint": None,
        "resume": None,
        "gan": False,
        "paper_mechanism_gate": True,
        "deployment_validation_input": "y=s+n1",
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": expected}
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if int(checkpoint.get("step", -1)) != int(metadata.get("max_steps", -2)):
        mismatches["checkpoint_step"] = {
            "observed": checkpoint.get("step"),
            "expected": metadata.get("max_steps"),
        }
    if checkpoint.get("scratch_init") is not True:
        mismatches["checkpoint_scratch_init"] = checkpoint.get("scratch_init")
    if checkpoint.get("resume") is not None:
        mismatches["checkpoint_resume"] = checkpoint.get("resume")
    if checkpoint.get("init_checkpoint") is not None:
        mismatches["checkpoint_init"] = checkpoint.get("init_checkpoint")
    if mismatches:
        raise ValueError(f"checkpoint violates Phase-A contract: {mismatches}")

    controlled = read_json(controlled_comparison_path)
    if not controlled.get("controlled_gate_pass", False):
        raise ValueError("TAU evaluation requires a passing controlled gate")
    if controlled.get("loss_variant") != metadata.get("loss_variant"):
        raise ValueError("controlled comparison and checkpoint loss variants differ")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_hash_key = f"{mode}_final_checkpoint_sha256"
    expected_sha256 = controlled["pair_integrity"].get(expected_hash_key)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint hash is not the arm authorized by the controlled gate"
        )
    receipt = {
        "schema_version": "dnf-phase-a-tau-checkpoint-v2",
        "mode": mode,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "loss_variant": metadata["loss_variant"],
        "train_manifest_sha256": metadata["train_manifest_sha256"],
        "valid_manifest_sha256": metadata["valid_manifest_sha256"],
        "canonical_speech_init_sha256": metadata[
            "canonical_speech_init_sha256"
        ],
        "contract_sha256": metadata["contract_sha256"],
        "model_config_sha256": metadata["model_config_sha256"],
        "code_surface_sha256": metadata["code_surface_sha256"],
        "controlled_comparison": str(controlled_comparison_path.resolve()),
        "controlled_comparison_sha256": sha256_file(
            controlled_comparison_path
        ),
    }
    if mode == "standard":
        model = SEMambapp(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return mode, cfg, {"standard": model}, receipt
    if mode == "dnf":
        model = DNFSEMambapp(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return mode, cfg, {"dnf": model}, receipt
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


@torch.inference_mode()
def enhance(
    audio: torch.Tensor,
    mode: str,
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    audio = audio.to(device)
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    audio = audio.unsqueeze(0)

    mag, pha, _ = stft_features(audio, cfg)
    if mode == "standard":
        out_mag, out_pha, _ = models["standard"](mag, pha)
        enhanced = istft_waveform(out_mag, out_pha, cfg)
        outputs = {"standard": enhanced}
    else:
        branch_outputs = models["dnf"](mag, pha)
        speech_wav = istft_waveform(
            branch_outputs["speech_mag"],
            branch_outputs["speech_pha"],
            cfg,
        )
        noise_wav = istft_waveform(
            branch_outputs["noise_mag"],
            branch_outputs["noise_pha"],
            cfg,
        )
        speech_wav, noise_wav = crop_pair(speech_wav, noise_wav)
        projection = dnf_output_eq14(speech_wav, noise_wav)
        if not projection.valid_mask.all():
            invalid = int((~projection.valid_mask).sum().item())
            raise RuntimeError(f"Eq.14 produced {invalid} invalid TAU sample(s)")
        outputs = {
            "eq14": projection.enhanced,
            "speech_head": speech_wav,
        }

    restored = {}
    for output_name, waveform in outputs.items():
        waveform = waveform.squeeze(0).cpu()
        if not torch.isfinite(waveform).all():
            raise RuntimeError(f"Non-finite {output_name} waveform")
        restored[output_name] = waveform.unsqueeze(0)
    return restored


def signal_stats(
    noisy_input: torch.Tensor,
    clean_reference: torch.Tensor,
    enhanced: torch.Tensor,
) -> dict[str, float]:
    noisy = noisy_input.flatten().float()
    clean = clean_reference.flatten().float()
    enh = enhanced.flatten().float()
    length = min(noisy.numel(), clean.numel(), enh.numel())
    if length <= 0:
        raise ValueError("TAU signal is empty")
    noisy = noisy[:length]
    clean = clean[:length]
    enh = enh[:length]
    noisy_rms = torch.sqrt(torch.mean(noisy**2) + 1e-12)
    clean_rms = torch.sqrt(torch.mean(clean**2) + 1e-12)
    enh_rms = torch.sqrt(torch.mean(enh**2) + 1e-12)
    activity_threshold = max(
        1.0e-5,
        float(clean.abs().max().item()) * 0.01,
    )
    clean_active_mask = clean.abs() >= activity_threshold
    if not bool(clean_active_mask.any()):
        raise ValueError("TAU clean reference has no active samples")
    noisy_active_rms = torch.sqrt(
        torch.mean(noisy[clean_active_mask] ** 2) + 1e-12
    )
    clean_active_rms = torch.sqrt(
        torch.mean(clean[clean_active_mask] ** 2) + 1e-12
    )
    enhanced_active_rms = torch.sqrt(
        torch.mean(enh[clean_active_mask] ** 2) + 1e-12
    )
    clean_active = float(clean_active_mask.float().mean().item())
    enhanced_active = float(
        (enh.abs() >= activity_threshold).float().mean().item()
    )
    input_si = si_sdr_loss(
        noisy.unsqueeze(0),
        clean.unsqueeze(0),
    )
    output_si = si_sdr_loss(
        enh.unsqueeze(0),
        clean.unsqueeze(0),
    )
    input_sdr = sdr_loss_eq5(
        noisy.unsqueeze(0),
        clean.unsqueeze(0),
    )
    output_sdr = sdr_loss_eq5(
        enh.unsqueeze(0),
        clean.unsqueeze(0),
    )
    if (
        not input_si.valid_mask.all()
        or not output_si.valid_mask.all()
        or not torch.isfinite(input_sdr).all()
        or not torch.isfinite(output_sdr).all()
    ):
        raise ValueError("TAU clean-reference metrics are invalid")
    return {
        "duration_seconds": length / 16000.0,
        "input_rms": float(noisy_rms.item()),
        "clean_rms": float(clean_rms.item()),
        "enhanced_rms": float(enh_rms.item()),
        "gain_db_to_input": 20.0
        * math.log10(float((enh_rms / noisy_rms).item())),
        "gain_db_to_clean": 20.0
        * math.log10(float((enh_rms / clean_rms).item())),
        "clean_active_gain_db_to_input": 20.0
        * math.log10(
            float((enhanced_active_rms / noisy_active_rms).item())
        ),
        "clean_active_gain_db_to_clean": 20.0
        * math.log10(
            float((enhanced_active_rms / clean_active_rms).item())
        ),
        "clean_activity_threshold": activity_threshold,
        "clean_active_ratio": clean_active,
        "enhanced_active_ratio": enhanced_active,
        "active_ratio_delta_to_clean": enhanced_active - clean_active,
        "si_sdri_db": float(
            (input_si.value - output_si.value).item()
        ),
        "sdri_db": float((input_sdr - output_sdr).item()),
        "si_sdr_db": float((-output_si.value).item()),
        "sdr_db": float((-output_sdr).item()),
        "enhanced_peak": float(enh.abs().max().item()) if length else 0.0,
    }


def summarize_signal(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["output_name"], row["sample_group"], row["task"])].append(row)
    summary = []
    for (output_name, sample_group, task), bucket in sorted(buckets.items()):
        for key in (
            "gain_db_to_input",
            "gain_db_to_clean",
            "clean_active_gain_db_to_input",
            "clean_active_gain_db_to_clean",
            "active_ratio_delta_to_clean",
            "si_sdri_db",
            "sdri_db",
            "si_sdr_db",
            "sdr_db",
            "enhanced_peak",
        ):
            values = [float(row[key]) for row in bucket]
            summary.append(
                {
                    "output_name": output_name,
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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    enhanced_dir = args.output_dir / "enhanced"
    degraded_dir = args.output_dir / "degraded"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    mode, cfg, models, checkpoint_receipt = load_models(
        args.checkpoint,
        args.mode,
        device,
        args.controlled_comparison,
    )
    (args.output_dir / "checkpoint_receipt.json").write_text(
        json.dumps(checkpoint_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample_rate = int(cfg["stft_cfg"]["sampling_rate"])
    rows, selection_receipt = stratified_pair_rows(
        args.pair_csv,
        args.speakers_per_group,
        args.selection_seed,
    )
    pair_csv = args.output_dir / "pair_manifest.csv"
    write_csv(pair_csv, rows, list(rows[0].keys()))
    (args.output_dir / "selection_receipt.json").write_text(
        json.dumps(selection_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    signal_rows = []
    output_names = ("standard",) if mode == "standard" else ("eq14", "speech_head")
    for output_name in output_names:
        (enhanced_dir / output_name).mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        audio = load_audio(Path(row["noisy_filepath"]), sample_rate)
        clean_audio = load_audio(Path(row["clean_filepath"]), sample_rate)
        outputs = enhance(audio, mode, cfg, models, device)
        for output_name, enhanced in outputs.items():
            torchaudio.save(
                str(enhanced_dir / output_name / f'{row["uid"]}.wav'),
                enhanced,
                sample_rate,
                encoding="PCM_F",
                bits_per_sample=32,
            )
            stats = signal_stats(audio, clean_audio, enhanced)
            signal_rows.append(
                {
                    "uid": row["uid"],
                    "speaker_id": row["speaker_id"],
                    "task": row["task"],
                    "sample_group": row.get("sample_group") or speaker_group(row),
                    "output_name": output_name,
                    **stats,
                }
            )
        torchaudio.save(
            str(degraded_dir / f'{row["uid"]}.wav'),
            audio,
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        if index % 20 == 0 or index == len(rows):
            print(f"enhanced {index}/{len(rows)}", flush=True)

    signal_fields = [
        "uid",
        "speaker_id",
        "task",
        "sample_group",
        "output_name",
        "duration_seconds",
        "input_rms",
        "clean_rms",
        "enhanced_rms",
        "gain_db_to_input",
        "gain_db_to_clean",
        "clean_active_gain_db_to_input",
        "clean_active_gain_db_to_clean",
        "clean_activity_threshold",
        "clean_active_ratio",
        "enhanced_active_ratio",
        "active_ratio_delta_to_clean",
        "si_sdri_db",
        "sdri_db",
        "si_sdr_db",
        "sdr_db",
        "enhanced_peak",
    ]
    write_csv(args.output_dir / "signal_guard.csv", signal_rows, signal_fields)
    write_csv(
        args.output_dir / "signal_guard_summary.csv",
        summarize_signal(signal_rows),
        ["output_name", "sample_group", "task", "metric", "n", "mean", "min", "max"],
    )

    metrics = {}
    input_score_command = [
        str(AVQI_PYTHON),
        str(AVQI_SCRIPT),
        "score",
        "--pair-csv",
        str(pair_csv),
        "--enhanced-dir",
        str(degraded_dir),
        "--clean-cache",
        str(args.clean_cache),
        "--workers",
        str(args.workers),
    ]
    input_scored = subprocess.run(
        input_score_command,
        check=True,
        capture_output=True,
        text=True,
    )
    metrics["input"] = json.loads(
        input_scored.stdout.strip().splitlines()[-1]
    )
    for output_name in output_names:
        score_command = [
            str(AVQI_PYTHON),
            str(AVQI_SCRIPT),
            "score",
            "--pair-csv",
            str(pair_csv),
            "--enhanced-dir",
            str(enhanced_dir / output_name),
            "--clean-cache",
            str(args.clean_cache),
            "--workers",
            str(args.workers),
        ]
        scored = subprocess.run(
            score_command,
            check=True,
            capture_output=True,
            text=True,
        )
        metrics[output_name] = json.loads(scored.stdout.strip().splitlines()[-1])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "mode": mode,
                "checkpoint": str(args.checkpoint),
                "checkpoint_receipt": checkpoint_receipt,
                "controlled_comparison": str(
                    args.controlled_comparison.resolve()
                ),
                "pair_csv": str(pair_csv),
                "clean_cache": str(args.clean_cache),
                "enhanced_dir": str(enhanced_dir),
                "dnf_final_output": "eq14" if mode == "dnf" else None,
                "inference_amplitude_policy": "raw_no_peak_normalization",
                "selection": selection_receipt,
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
