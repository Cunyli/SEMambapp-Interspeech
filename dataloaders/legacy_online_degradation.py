import bisect
import io
import json
import random
import sys
import tarfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import librosa
import numpy as np
import soundfile as sf
import torch

from model.stfts import mag_phase_stft


_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff"}


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


def _looks_like_webdataset_manifest(path):
    path = Path(path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return isinstance(row, dict) and "shard" in row and ("audio_member" in row or "_shard_dir" in row)
    except (OSError, json.JSONDecodeError):
        return False
    return False


class _TarShardAudioPool:
    def __init__(self, manifest_jsonl, shard_dir=None, tar_cache_size=8, member_cache_size=64):
        self.manifest_path = Path(manifest_jsonl).expanduser()
        self.shard_dir = Path(shard_dir).expanduser() if shard_dir else self.manifest_path.parent
        self.records = []
        self.shards = []
        self._counts = []
        self._cumulative_counts = []
        self._mode = None
        self._tar_cache_size = int(tar_cache_size)
        self._member_cache_size = int(member_cache_size)
        self._tar_cache = OrderedDict()
        self._member_cache = OrderedDict()
        self._raw_cache = OrderedDict()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self._append_manifest_row(json.loads(line))
        self._rebuild_cumulative_counts()
        if not self:
            raise ValueError(f"WebDataset shard manifest is empty: {self.manifest_path}")

    def __len__(self):
        if self._mode == "shard":
            return self._cumulative_counts[-1] if self._cumulative_counts else 0
        return len(self.records)

    @property
    def prefers_sequential_access(self):
        return self._mode == "shard"

    def __getitem__(self, index):
        if self._mode != "shard":
            return self.records[index]

        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)

        shard_pos = bisect.bisect_right(self._cumulative_counts, index)
        prev_count = self._cumulative_counts[shard_pos - 1] if shard_pos else 0
        local_index = index - prev_count
        members = self._members_for_shard(self.shards[shard_pos])
        return members[local_index % len(members)]

    def __setitem__(self, index, value):
        if self._mode == "shard":
            raise TypeError("Lazy shard-level manifests cannot be item-shuffled in place; use shuffle().")
        self.records[index] = value

    def shuffle(self, rng):
        if self._mode == "shard":
            paired = list(zip(self.shards, self._counts))
            rng.shuffle(paired)
            self.shards = [item[0] for item in paired]
            self._counts = [item[1] for item in paired]
            self._rebuild_cumulative_counts()
            return
        rng.shuffle(self.records)

    def close(self):
        for tar in self._tar_cache.values():
            tar.close()
        self._tar_cache.clear()
        for handle in self._raw_cache.values():
            handle.close()
        self._raw_cache.clear()
        self._member_cache.clear()

    def __del__(self):
        if hasattr(self, "_tar_cache"):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_cache"] = OrderedDict()
        state["_raw_cache"] = OrderedDict()
        state["_member_cache"] = OrderedDict()
        return state

    def _append_manifest_row(self, row):
        row_mode = "item" if "audio_member" in row else "shard"
        if self._mode is None:
            self._mode = row_mode
        elif self._mode != row_mode:
            raise ValueError(f"Mixed item-level and shard-level manifest rows are not supported: {self.manifest_path}")

        if row_mode == "item":
            record = dict(row)
            record.setdefault("_shard_dir", str(self.shard_dir))
            self.records.append(record)
            return

        shard_record = dict(row)
        shard_record.setdefault("_shard_dir", str(self.shard_dir))
        count = int(shard_record.get("sample_count") or 0)
        if count <= 0:
            count = len(self._members_for_shard(shard_record))
        shard_record["_sample_count"] = count
        self.shards.append(shard_record)
        self._counts.append(count)

    def _rebuild_cumulative_counts(self):
        self._cumulative_counts = []
        total = 0
        for count in self._counts:
            total += int(count)
            self._cumulative_counts.append(total)

    def _members_for_shard(self, shard_record):
        cache_key = str(self._shard_path(shard_record))
        records = self._member_cache.get(cache_key)
        if records is not None:
            self._member_cache.move_to_end(cache_key)
            return records

        records = list(self._iter_shard_records(shard_record))
        if not records:
            raise ValueError(f"No audio members found in shard: {cache_key}")
        self._member_cache[cache_key] = records
        self._member_cache.move_to_end(cache_key)
        while len(self._member_cache) > self._member_cache_size:
            self._member_cache.popitem(last=False)
        return records

    def _iter_shard_records(self, shard_record):
        tar = self._tar(shard_record)
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_path = PurePosixPath(member.name)
            if member_path.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            if member.size <= 0:
                continue
            record = dict(shard_record)
            record["_shard_dir"] = str(Path(shard_record.get("_shard_dir", self.shard_dir)).expanduser())
            record["shard"] = str(shard_record["shard"])
            record["audio_member"] = member.name
            record["json_member"] = str(member_path.with_suffix(".json"))
            record["key"] = str(shard_record.get("key", member_path.stem))
            record["_tar_offset_data"] = int(member.offset_data)
            record["_tar_size"] = int(member.size)
            yield record

    def _shard_path(self, record):
        shard_dir = Path(record.get("_shard_dir", self.shard_dir)).expanduser()
        return shard_dir / record["shard"]

    def _tar(self, record):
        tar_path = self._shard_path(record)
        cache_key = str(tar_path)
        tar = self._tar_cache.get(cache_key)
        if tar is not None:
            self._tar_cache.move_to_end(cache_key)
            return tar

        tar = tarfile.open(tar_path, "r:")
        self._tar_cache[cache_key] = tar
        self._tar_cache.move_to_end(cache_key)
        while len(self._tar_cache) > self._tar_cache_size:
            _, old_tar = self._tar_cache.popitem(last=False)
            old_tar.close()
        return tar

    def _raw_file(self, record):
        tar_path = self._shard_path(record)
        cache_key = str(tar_path)
        handle = self._raw_cache.get(cache_key)
        if handle is not None:
            self._raw_cache.move_to_end(cache_key)
            return handle

        handle = tar_path.open("rb")
        self._raw_cache[cache_key] = handle
        self._raw_cache.move_to_end(cache_key)
        while len(self._raw_cache) > self._tar_cache_size:
            _, old_handle = self._raw_cache.popitem(last=False)
            old_handle.close()
        return handle

    def _decode_audio_payload(self, payload, sampling_rate):
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        audio = audio[:, :1].T
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate, res_type="soxr_hq")
        return audio.reshape(1, -1)

    def read_audio(self, record, sampling_rate):
        if "_tar_offset_data" in record and "_tar_size" in record:
            handle = self._raw_file(record)
            handle.seek(int(record["_tar_offset_data"]))
            payload = handle.read(int(record["_tar_size"]))
            return self._decode_audio_payload(payload, sampling_rate)

        tar = self._tar(record)
        audio_member = record["audio_member"]
        extracted = tar.extractfile(audio_member)
        if extracted is None:
            raise FileNotFoundError(f"Missing {audio_member} in {self._shard_path(record)}")
        with extracted:
            payload = extracted.read()
        return self._decode_audio_payload(payload, sampling_rate)


def _load_training_collection(path):
    if _looks_like_webdataset_manifest(path):
        return _TarShardAudioPool(path)
    return _load_json_file(path)


@contextmanager
def _isolated_legacy_rng(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _rir_direct_path_delay(rir_audio, mode="argmax_abs"):
    if mode != "argmax_abs":
        raise ValueError(f"Unsupported rir_delay_mode: {mode}")

    rir_array = np.asarray(rir_audio)
    if rir_array.ndim == 0 or rir_array.size == 0:
        return 0
    if rir_array.ndim == 1:
        rir_magnitude = np.abs(rir_array)
    else:
        rir_magnitude = np.max(np.abs(rir_array), axis=0)
    return int(np.argmax(rir_magnitude))


def _shift_clean_by_delay(clean_audio, delay_samples):
    delay_samples = int(delay_samples)
    clean_array = np.asarray(clean_audio)
    if delay_samples <= 0:
        return clean_array.copy()

    shifted = np.zeros_like(clean_array)
    length = clean_array.shape[-1]
    if delay_samples < length:
        shifted[..., delay_samples:] = clean_array[..., : length - delay_samples]
    return shifted


def _target_audio_for_selected_degradations(
    clean_audio,
    rir_audio,
    selected_degradations,
    target_type="legacy_clean",
    rir_delay_mode="argmax_abs",
):
    if target_type == "legacy_clean":
        return np.asarray(clean_audio).copy()
    if target_type != "shifted_anechoic":
        raise ValueError(f"Unsupported target_cfg.type: {target_type}")

    if "reverb" not in selected_degradations:
        return np.asarray(clean_audio).copy()

    delay_samples = _rir_direct_path_delay(rir_audio, mode=rir_delay_mode)
    return _shift_clean_by_delay(clean_audio, delay_samples)


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
        self.seed = int(seed or 0)
        self.epoch = 0
        self.mode = mode
        self.sampling_rate = int(cfg["stft_cfg"]["sampling_rate"])
        self.segment_size = int(cfg["training_cfg"]["segment_size"])
        self.n_fft = int(cfg["stft_cfg"]["n_fft"])
        self.hop_size = int(cfg["stft_cfg"]["hop_size"])
        self.win_size = int(cfg["stft_cfg"]["win_size"])
        self.compress_factor = float(cfg["model_cfg"]["compress_factor"])
        target_cfg = cfg.get("target_cfg", {})
        self.target_type = target_cfg.get("type", "legacy_clean")
        self.rir_delay_mode = target_cfg.get("rir_delay_mode", "argmax_abs")

        self.clean_wavs_path = _load_training_collection(clean_json)
        self.noise_wavs_path = _load_training_collection(noise_json)
        self.rir_wavs_path = _load_training_collection(rir_json)
        self.prefers_sequential_sampler = any(
            getattr(collection, "prefers_sequential_access", False)
            for collection in (self.clean_wavs_path, self.noise_wavs_path, self.rir_wavs_path)
        )
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
            init_rng = random.Random(self.seed)
            for collection in (self.clean_wavs_path, self.noise_wavs_path, self.rir_wavs_path):
                if hasattr(collection, "shuffle"):
                    collection.shuffle(init_rng)
                else:
                    init_rng.shuffle(collection)
        elif len(self.clean_val_path) != len(self.degraded_val_path):
            raise ValueError(
                "Validation clean/degraded lists must have the same length after limiting: "
                f"{len(self.clean_val_path)} != {len(self.degraded_val_path)}"
            )

    def __getitem__(self, index):
        sample_rng = None
        if self.mode == "Train":
            sample_seed = self._sample_seed(index)
            sample_rng = random.Random(sample_seed)
            clean_path = self.clean_wavs_path[index]
            noise_path = sample_rng.choice(self.noise_wavs_path)
            rir_path = sample_rng.choice(self.rir_wavs_path)

            clean_audio = self._load_training_audio(self.clean_wavs_path, clean_path)
            noise_audio = self._load_training_audio(self.noise_wavs_path, noise_path)
            rir_audio = self._load_training_audio(self.rir_wavs_path, rir_path)

            with _isolated_legacy_rng(sample_seed):
                degrad_cfgs, selected_degrads = self.random_select_and_order(self.cfg, seed=sample_seed)
                clean_audio, degraded_audio = self.apply_degradation(
                    self.cfg,
                    clean_audio,
                    noise_audio,
                    rir_audio,
                    degrad_cfgs,
                    selected_degrads,
                    seed=sample_seed,
                )
            target_audio = _target_audio_for_selected_degradations(
                clean_audio,
                rir_audio,
                selected_degrads,
                target_type=getattr(self, "target_type", "legacy_clean"),
                rir_delay_mode=getattr(self, "rir_delay_mode", "argmax_abs"),
            )
        else:
            target_audio = self._load_audio(self.clean_val_path[index])
            degraded_audio = self._load_audio(self.degraded_val_path[index])

        target_audio = _peak_normalize(torch.as_tensor(target_audio, dtype=torch.float32))
        degraded_audio = _peak_normalize(torch.as_tensor(degraded_audio, dtype=torch.float32))
        target_audio, degraded_audio = self._crop_or_pad_pair(target_audio, degraded_audio, sample_rng)

        target_mag, target_pha, target_com = mag_phase_stft(
            target_audio,
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
            target_audio.squeeze(),
            target_mag.squeeze(),
            target_pha.squeeze(),
            target_com.squeeze(),
            degraded_audio.squeeze(),
            degraded_mag.squeeze(),
            degraded_pha.squeeze(),
        )

    def __len__(self):
        if self.mode == "Train":
            return len(self.clean_wavs_path)
        return len(self.clean_val_path)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.mode == "Train":
            epoch_rng = random.Random(self.seed + self.epoch * 1_000_003)
            for collection in (self.clean_wavs_path, self.noise_wavs_path, self.rir_wavs_path):
                if getattr(collection, "prefers_sequential_access", False):
                    collection.shuffle(epoch_rng)

    def _sample_seed(self, index):
        return self.seed + int(getattr(self, "epoch", 0)) * 1_000_003 + int(index)

    def sample_id(self, index):
        if self.mode == "Train":
            clean_record = self.clean_wavs_path[index]
            if isinstance(clean_record, dict):
                return clean_record.get("key", Path(clean_record.get("audio_member", f"train_{index}")).stem)
            return Path(clean_record).stem
        return Path(self.clean_val_path[index]).stem

    def _load_training_audio(self, collection, item):
        if isinstance(collection, _TarShardAudioPool):
            return collection.read_audio(item, self.sampling_rate)
        return self._load_audio(item)

    def _load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return audio.reshape(1, -1)

    def _crop_or_pad_pair(self, clean_audio, degraded_audio, rng=None):
        length = min(clean_audio.size(1), degraded_audio.size(1))
        clean_audio = clean_audio[:, :length]
        degraded_audio = degraded_audio[:, :length]

        if length >= self.segment_size:
            if self.mode == "Train":
                start = (rng or random).randint(0, length - self.segment_size)
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
