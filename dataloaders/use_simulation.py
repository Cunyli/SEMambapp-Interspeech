import csv
import json
import sys
from pathlib import Path

import torch

from model.stfts import mag_phase_stft


def _add_use_simulation_to_path(root):
    root = Path(root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _string_metadata(row):
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def _load_pair_metadata(pair_manifest):
    path = Path(pair_manifest).expanduser()
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a list of pair rows.")
    metadata = {}
    for row in rows:
        key = row.get("uid") or row.get("id") or Path(row.get("noisy_filepath") or row.get("noisy_path", "")).stem
        if not key:
            raise ValueError(f"Pair row is missing uid/id/noisy path: {row}")
        metadata[str(key)] = _string_metadata(row)
    return metadata


class USESimulationSEMambaDataset(torch.utils.data.Dataset):
    """SEMamba adapter around USE_simulation fixed noisy/clean pair datasets."""

    def __init__(
        self,
        cfg,
        pair_manifest,
        use_simulation_root="../USE_simulation",
        mode="Train",
        random_start=True,
        normalize=True,
        seed=0,
    ):
        _add_use_simulation_to_path(use_simulation_root)
        from use_simulation_datasets import FixedPairDataset

        self.sampling_rate = int(cfg["stft_cfg"]["sampling_rate"])
        train_segment_size = int(cfg["training_cfg"]["segment_size"])
        if mode == "Train":
            self.segment_size = train_segment_size
        else:
            validation_seconds = cfg["training_cfg"].get("validation_segment_seconds")
            self.segment_size = int(
                cfg["training_cfg"].get(
                    "validation_segment_size",
                    round(float(validation_seconds) * self.sampling_rate)
                    if validation_seconds is not None
                    else train_segment_size,
                )
            )
        self.n_fft = int(cfg["stft_cfg"]["n_fft"])
        self.hop_size = int(cfg["stft_cfg"]["hop_size"])
        self.win_size = int(cfg["stft_cfg"]["win_size"])
        self.compress_factor = float(cfg["model_cfg"]["compress_factor"])
        self.mode = mode
        self.validation_random_start = bool(cfg["training_cfg"].get("validation_random_start", False))
        self.validation_prefer_active = bool(cfg["training_cfg"].get("validation_prefer_active", True))
        self.activity_threshold = float(cfg["training_cfg"].get("validation_activity_threshold", 0.01))
        self.min_active_ratio = float(cfg["training_cfg"].get("validation_min_active_ratio", 0.05))
        self.return_metadata = bool(
            cfg["training_cfg"].get("return_metadata", False)
            or cfg["training_cfg"].get("sv_guardrail", {}).get("enabled", False)
        )

        self.dataset = FixedPairDataset(
            pair_manifest=pair_manifest,
            wav_len=None,
            num_per_epoch=0,
            random_start=False,
            target_sample_rate=self.sampling_rate,
            mode="train" if mode == "Train" else "validation",
            normalize=normalize,
            seed=seed,
        )
        self.metadata_by_id = _load_pair_metadata(pair_manifest) if self.return_metadata else {}
        self.random_start = bool((random_start and mode == "Train") or (mode != "Train" and self.validation_random_start))

    def __getitem__(self, index):
        degraded_audio, clean_audio, item_info = self.dataset[index]
        clean_audio = torch.as_tensor(clean_audio, dtype=torch.float32).reshape(1, -1)
        degraded_audio = torch.as_tensor(degraded_audio, dtype=torch.float32).reshape(1, -1)

        clean_audio, degraded_audio = self._crop_or_pad_pair(clean_audio, degraded_audio)

        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )
        degraded_mag, degraded_pha, _ = mag_phase_stft(
            degraded_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )

        item = (
            clean_audio.squeeze(),
            clean_mag.squeeze(),
            clean_pha.squeeze(),
            clean_com.squeeze(),
            degraded_audio.squeeze(),
            degraded_mag.squeeze(),
            degraded_pha.squeeze(),
        )
        if not self.return_metadata:
            return item
        metadata = dict(self.metadata_by_id.get(str(item_info["id"]), {}))
        metadata.setdefault("uid", str(item_info["id"]))
        metadata.setdefault("sample_rate", str(item_info.get("sample_rate", "")))
        metadata.setdefault("length", str(item_info.get("length", "")))
        metadata.setdefault("noisy_filepath", str(item_info.get("noisy_path", "")))
        metadata.setdefault("clean_filepath", str(item_info.get("clean_path", "")))
        return (*item, metadata)

    def __len__(self):
        return len(self.dataset)

    def sample_id(self, index):
        return self.dataset.meta_selected[index]["id"]

    def _crop_or_pad_pair(self, clean_audio, degraded_audio):
        length = min(clean_audio.size(1), degraded_audio.size(1))
        clean_audio = clean_audio[:, :length]
        degraded_audio = degraded_audio[:, :length]

        if length >= self.segment_size:
            start = 0
            if self.random_start:
                start = int(torch.randint(0, length - self.segment_size + 1, (1,)).item())
            if self.mode != "Train" and self.validation_prefer_active:
                start = self._active_start(clean_audio, length, fallback=start)
            return (
                clean_audio[:, start : start + self.segment_size],
                degraded_audio[:, start : start + self.segment_size],
            )

        pad = self.segment_size - length
        clean_audio = torch.nn.functional.pad(clean_audio, (0, pad), "constant")
        degraded_audio = torch.nn.functional.pad(degraded_audio, (0, pad), "constant")
        return clean_audio, degraded_audio

    def _active_start(self, clean_audio, length, fallback):
        max_start = length - self.segment_size
        if max_start <= 0:
            return 0

        active = (clean_audio.squeeze(0).abs() > self.activity_threshold).float()
        prefix = torch.nn.functional.pad(torch.cumsum(active, dim=0), (1, 0))
        counts = prefix[self.segment_size:] - prefix[:-self.segment_size]
        valid = torch.nonzero(counts >= self.segment_size * self.min_active_ratio).flatten()
        if valid.numel() == 0:
            return fallback
        if self.random_start:
            return int(valid[torch.randint(0, valid.numel(), (1,)).item()].item())
        return int(valid[0].item())
