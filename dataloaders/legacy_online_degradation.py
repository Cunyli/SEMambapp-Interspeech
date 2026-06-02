import json
import random
import sys
from pathlib import Path

import librosa
import torch

from model.stfts import mag_phase_stft


def _add_use_simulation_to_path(root):
    root = Path(root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _validation_sort_key(file_path):
    base_name = Path(file_path).name
    stem = Path(base_name).stem
    suffix_digits = ""
    for char in reversed(stem):
        if not char.isdigit():
            break
        suffix_digits = char + suffix_digits
    if suffix_digits:
        return (0, int(suffix_digits), base_name)
    return (1, base_name)


def _peak_normalize(audio_tensor, eps=1e-9):
    return audio_tensor / (audio_tensor.abs().max() + eps)


def _load_json_file(path):
    with Path(path).expanduser().open() as handle:
        return json.load(handle)


class LegacyOnlineDegradationDataset(torch.utils.data.Dataset):
    """Legacy SEMamba++ pretraining dataset with on-the-fly train degradation."""

    def __init__(
        self,
        cfg,
        clean_json,
        noise_json,
        rir_json,
        clean_valid_json,
        degraded_valid_json,
        use_simulation_root="/scratch/work/lil14/USE_simulation",
        mode="Train",
        seed=None,
    ):
        _add_use_simulation_to_path(use_simulation_root)
        from simulate_degradation import apply_degradation, random_select_and_order

        self.apply_degradation = apply_degradation
        self.random_select_and_order = random_select_and_order
        self.cfg = cfg
        self.seed = seed
        self.mode = mode
        self.sampling_rate = int(cfg["stft_cfg"]["sampling_rate"])
        self.segment_size = int(cfg["training_cfg"]["segment_size"])
        self.n_fft = int(cfg["stft_cfg"]["n_fft"])
        self.hop_size = int(cfg["stft_cfg"]["hop_size"])
        self.win_size = int(cfg["stft_cfg"]["win_size"])
        self.compress_factor = float(cfg["model_cfg"]["compress_factor"])

        self.clean_wavs_path = _load_json_file(clean_json)
        self.noise_wavs_path = _load_json_file(noise_json)
        self.rir_wavs_path = _load_json_file(rir_json)
        clean_valid = sorted(_load_json_file(clean_valid_json), key=_validation_sort_key)
        degraded_valid = sorted(_load_json_file(degraded_valid_json), key=_validation_sort_key)

        valid_limit = int(cfg["training_cfg"].get("legacy_validation_limit", 100))
        self.clean_val_path = clean_valid[:valid_limit]
        self.degraded_val_path = degraded_valid[:valid_limit]

        if mode == "Train":
            if not self.clean_wavs_path:
                raise ValueError(f"No clean training files found in {clean_json}.")
            if not self.noise_wavs_path:
                raise ValueError(f"No noise training files found in {noise_json}.")
            if not self.rir_wavs_path:
                raise ValueError(f"No RIR training files found in {rir_json}.")
            random.shuffle(self.clean_wavs_path)
            random.shuffle(self.noise_wavs_path)
            random.shuffle(self.rir_wavs_path)
        elif len(self.clean_val_path) != len(self.degraded_val_path):
            raise ValueError(
                "Validation clean/degraded lists must have the same length after limiting: "
                f"{len(self.clean_val_path)} != {len(self.degraded_val_path)}"
            )

    def __getitem__(self, index):
        if self.mode == "Train":
            clean_path = self.clean_wavs_path[index]
            noise_path = random.choice(self.noise_wavs_path)
            rir_path = random.choice(self.rir_wavs_path)

            clean_audio = self._load_audio(clean_path)
            noise_audio = self._load_audio(noise_path)
            rir_audio = self._load_audio(rir_path)

            degrad_cfgs, selected_degrads = self.random_select_and_order(self.cfg, seed=self.seed)
            clean_audio, degraded_audio = self.apply_degradation(
                self.cfg,
                clean_audio,
                noise_audio,
                rir_audio,
                degrad_cfgs,
                selected_degrads,
                seed=self.seed,
            )
        else:
            clean_audio = self._load_audio(self.clean_val_path[index])
            degraded_audio = self._load_audio(self.degraded_val_path[index])

        clean_audio = _peak_normalize(torch.as_tensor(clean_audio, dtype=torch.float32))
        degraded_audio = _peak_normalize(torch.as_tensor(degraded_audio, dtype=torch.float32))
        clean_audio, degraded_audio = self._crop_or_pad_pair(clean_audio, degraded_audio)

        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio,
            self.n_fft,
            self.hop_size,
            self.win_size,
            self.compress_factor,
        )
        degraded_mag, degraded_pha, _ = mag_phase_stft(
            degraded_audio,
            self.n_fft,
            self.hop_size,
            self.win_size,
            self.compress_factor,
        )

        return (
            clean_audio.squeeze(),
            clean_mag.squeeze(),
            clean_pha.squeeze(),
            clean_com.squeeze(),
            degraded_audio.squeeze(),
            degraded_mag.squeeze(),
            degraded_pha.squeeze(),
        )

    def __len__(self):
        if self.mode == "Train":
            return len(self.clean_wavs_path)
        return len(self.clean_val_path)

    def sample_id(self, index):
        if self.mode == "Train":
            return Path(self.clean_wavs_path[index]).stem
        return Path(self.clean_val_path[index]).stem

    def _load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return audio.reshape(1, -1)

    def _crop_or_pad_pair(self, clean_audio, degraded_audio):
        length = min(clean_audio.size(1), degraded_audio.size(1))
        clean_audio = clean_audio[:, :length]
        degraded_audio = degraded_audio[:, :length]

        if length >= self.segment_size:
            if self.mode == "Train":
                start = random.randint(0, length - self.segment_size)
            else:
                start = 0
            return (
                clean_audio[:, start : start + self.segment_size],
                degraded_audio[:, start : start + self.segment_size],
            )

        pad = self.segment_size - length
        clean_audio = torch.nn.functional.pad(clean_audio, (0, pad), "constant")
        degraded_audio = torch.nn.functional.pad(degraded_audio, (0, pad), "constant")
        return clean_audio, degraded_audio
