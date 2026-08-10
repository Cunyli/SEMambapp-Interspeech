"""Controlled additive WebDataset stream for the paper-faithful DNF test.

This module is intended to live in the SeMamba++ ``dataloaders`` package.  It
reuses DNF_USE's manifest, tar-item, audio-reader, and deterministic worker
helpers, but deliberately excludes the simulation/RIR/codec path.  The caller
must put ``/scratch/work/lil14/DNF_USE`` on ``sys.path`` before importing it.

The noisy-target training view exposes only ``x=s+n1+n2``, ``s+n1``, and
``n2``.  Clean speech is returned only when ``expose_clean_for_eval=True``.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataloader.dnf_webdataset_protocol import (
    _cycle_items,
    _iter_shard_items,
    _load_shard_records,
    _pop_random_item,
    _shard_source_counts,
)
from dataloader.gap_webdataset import gap_worker_init_fn, stable_uint32
from dataloader.hybrid_webdataset_protocol import (
    WebDatasetAudioReader,
    _checked_pad_or_crop,
    _duration_samples,
)


DEFAULT_CLEAN_SOURCES = ("EARS", "VCTK")
DEFAULT_NOISE_SOURCE = "WHAM_noise"
DEFAULT_SNR_DB_CHOICES = (0.0, 5.0, 10.0)


class ControlledSampleRejected(ValueError):
    """A decoded tuple violates the frozen controlled-mixture contract."""


@dataclass(frozen=True)
class ControlledAdditiveMixture:
    """Exact additive tensors and diagnostics for one controlled tuple."""

    degraded: np.ndarray
    noisy_speech_target: np.ndarray
    artificial_noise: np.ndarray
    clean_speech: np.ndarray
    noise1: np.ndarray
    diagnostics: dict


def _item_id(item: dict) -> str:
    member = item.get("audio_member") or item.get("key") or "<missing>"
    return f"{item.get('_shard_dir', '')}/{item.get('shard', '')}#{member}"


def _select_source_shards(
    records: Sequence[dict],
    allowed_sources: Sequence[str],
    role: str,
) -> list[dict]:
    """Select shards containing only the explicitly allowed normalized sources."""

    allowed = set(allowed_sources)
    selected = []
    for record in records:
        source_counts = _shard_source_counts(record)
        sources = {source for source, count in source_counts.items() if int(count or 0) > 0}
        if sources and sources.issubset(allowed):
            updated = dict(record)
            updated["_source_names"] = sorted(sources)
            updated["_route_category"] = "clean_strict" if role == "speech" else "noise_only"
            selected.append(updated)
    return selected


def _validate_mono_waveform(waveform: np.ndarray, role: str) -> np.ndarray:
    array = np.asarray(waveform)
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[-1] == 0:
        raise ControlledSampleRejected(
            f"{role} must have shape [1, time] with non-empty time; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ControlledSampleRejected(f"{role} contains non-finite samples")
    return array.astype(np.float64, copy=False)


def _energy(waveform: np.ndarray) -> float:
    return float(np.sum(np.square(waveform), dtype=np.float64))


def _absolute_correlation(first: np.ndarray, second: np.ndarray) -> float:
    denominator = math.sqrt(_energy(first) * _energy(second))
    if denominator <= 0.0:
        return math.inf
    return abs(float(np.sum(first * second, dtype=np.float64))) / denominator


def build_controlled_additive_mixture(
    clean_wav: np.ndarray,
    noise1_wav: np.ndarray,
    noise2_wav: np.ndarray,
    *,
    seed: int,
    target_sample_rate: int,
    cut_duration: Union[float, Sequence[float]],
    target_snr_db: float,
    max_abs_correlation: float = 0.05,
    silence_rms_threshold: float = 1e-6,
    peak_limit: float = 0.95,
) -> ControlledAdditiveMixture:
    """Construct ``s_noisy=s+n1`` and ``x=s+n1+n2`` without augmentation.

    ``n1`` and ``n2`` are independently scaled to half of the contracted total
    noise energy.  Peak protection, when needed, uses one common scalar for all
    components and targets.
    """

    if not math.isfinite(float(target_snr_db)):
        raise ValueError("target_snr_db must be finite")
    if not 0.0 <= float(max_abs_correlation) < 1.0:
        raise ValueError("max_abs_correlation must be in [0, 1)")
    if not math.isfinite(float(silence_rms_threshold)) or silence_rms_threshold < 0.0:
        raise ValueError("silence_rms_threshold must be finite and non-negative")
    if not 0.0 < float(peak_limit) <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")

    py_rng = random.Random(int(seed))
    rng = np.random.default_rng(int(seed))
    length = _duration_samples(cut_duration, int(target_sample_rate), py_rng)
    if length <= 0:
        raise ValueError(f"cut_duration produced invalid length {length}")

    clean_crop, _ = _checked_pad_or_crop(clean_wav, length, rng, "clean")
    noise1_crop, _ = _checked_pad_or_crop(noise1_wav, length, rng, "noise1")
    noise2_crop, _ = _checked_pad_or_crop(noise2_wav, length, rng, "noise2")
    clean = _validate_mono_waveform(clean_crop, "clean")
    noise1 = _validate_mono_waveform(noise1_crop, "noise1")
    noise2 = _validate_mono_waveform(noise2_crop, "noise2")

    clean = clean - clean.mean(axis=-1, keepdims=True)
    noise1 = noise1 - noise1.mean(axis=-1, keepdims=True)
    noise2 = noise2 - noise2.mean(axis=-1, keepdims=True)

    rms = {
        "clean": math.sqrt(_energy(clean) / length),
        "noise1": math.sqrt(_energy(noise1) / length),
        "noise2": math.sqrt(_energy(noise2) / length),
    }
    silent = [name for name, value in rms.items() if value <= silence_rms_threshold]
    if silent:
        raise ControlledSampleRejected(f"silent or near-silent component(s): {silent}")

    correlations = {
        "speech_noise1": _absolute_correlation(clean, noise1),
        "speech_noise2": _absolute_correlation(clean, noise2),
        "noise1_noise2": _absolute_correlation(noise1, noise2),
    }
    failed_correlations = {
        name: value
        for name, value in correlations.items()
        if value >= float(max_abs_correlation)
    }
    if failed_correlations:
        raise ControlledSampleRejected(
            f"absolute correlation exceeds {max_abs_correlation}: {failed_correlations}"
        )

    speech_energy = _energy(clean)
    contracted_noise_energy = speech_energy * (10.0 ** (-float(target_snr_db) / 10.0))
    half_noise_energy = 0.5 * contracted_noise_energy
    noise1_scale = math.sqrt(half_noise_energy / _energy(noise1))
    noise2_scale = math.sqrt(half_noise_energy / _energy(noise2))
    noise1 = noise1 * noise1_scale
    noise2 = noise2 * noise2_scale

    noisy_speech = clean + noise1
    degraded = noisy_speech + noise2
    peak_before_gain = float(np.max(np.abs(degraded)))
    common_gain = min(1.0, float(peak_limit) / max(peak_before_gain, np.finfo(np.float64).tiny))
    clean = clean * common_gain
    noise1 = noise1 * common_gain
    noise2 = noise2 * common_gain

    clean32 = clean.astype(np.float32)
    noise1_32 = noise1.astype(np.float32)
    noise2_32 = noise2.astype(np.float32)
    noisy_speech32 = (clean32 + noise1_32).astype(np.float32)
    degraded32 = (noisy_speech32 + noise2_32).astype(np.float32)

    actual_total_noise = noise1_32.astype(np.float64) + noise2_32.astype(np.float64)
    achieved_snr_db = 10.0 * math.log10(
        _energy(clean32.astype(np.float64)) / _energy(actual_total_noise)
    )
    diagnostics = {
        "length": int(length),
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(achieved_snr_db),
        "noise1_energy": _energy(noise1_32.astype(np.float64)),
        "noise2_energy": _energy(noise2_32.astype(np.float64)),
        "absolute_correlations": correlations,
        "peak_before_common_gain": peak_before_gain,
        "common_gain": common_gain,
        "peak_after_common_gain": float(np.max(np.abs(degraded32))),
        "max_additive_error": float(
            np.max(np.abs(degraded32 - noisy_speech32 - noise2_32))
        ),
    }
    return ControlledAdditiveMixture(
        degraded=degraded32,
        noisy_speech_target=noisy_speech32,
        artificial_noise=noise2_32,
        clean_speech=clean32,
        noise1=noise1_32,
        diagnostics=diagnostics,
    )


def controlled_dnf_collate(batch: Sequence[dict]) -> dict:
    """Collate the Eq. (13) training view, preserving the clean firewall."""

    eval_visibility = ["eval_clean" in item for item in batch]
    if any(eval_visibility) and not all(eval_visibility):
        raise ValueError("eval clean visibility must be consistent within a batch")
    collated = {
        "mode": "dnf_noisy_target_controlled",
        "degraded_wav": torch.from_numpy(
            np.concatenate([item["degraded"] for item in batch], axis=0)
        ).float(),
        "s_noisy_wav": torch.from_numpy(
            np.concatenate([item["s_noisy"] for item in batch], axis=0)
        ).float(),
        "added_noise_wav": torch.from_numpy(
            np.concatenate([item["added_noise"] for item in batch], axis=0)
        ).float(),
        "sample_rate": torch.LongTensor([int(item["sample_rate"]) for item in batch]),
        "length": torch.LongTensor([int(item["length"]) for item in batch]),
        "utterance_id": [str(item["uid"]) for item in batch],
        "info": [item["info"] for item in batch],
    }
    if all(eval_visibility):
        collated["eval_clean_wav"] = torch.from_numpy(
            np.concatenate([item["eval_clean"] for item in batch], axis=0)
        ).float()
    return collated


class ControlledDNFAdditiveStreamDataset(IterableDataset):
    """Deterministic EARS/VCTK + WHAM additive stream for Eq. (13)."""

    def __init__(
        self,
        split_root: Union[str, Path],
        *,
        split: str = "train",
        target_sample_rate: int = 16000,
        cut_duration: Union[float, Sequence[float]] = 3.0,
        samples_per_epoch: int = 30000,
        clean_shuffle_buffer: int = 128,
        noise_buffer_size: int = 128,
        shard_shuffle_seed: int = 1234,
        tar_cache_size: int = 8,
        snr_db_choices: Sequence[float] = DEFAULT_SNR_DB_CHOICES,
        max_abs_correlation: float = 0.05,
        silence_rms_threshold: float = 1e-6,
        peak_limit: float = 0.95,
        max_tuple_attempts: int = 64,
        expose_clean_for_eval: bool = False,
        clean_sources: Sequence[str] = DEFAULT_CLEAN_SOURCES,
        noise_source: str = DEFAULT_NOISE_SOURCE,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"split must be train, valid, or test; got {split}")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if not snr_db_choices:
            raise ValueError("snr_db_choices must be non-empty")
        if max_tuple_attempts <= 0:
            raise ValueError("max_tuple_attempts must be positive")

        self.split_root = Path(split_root)
        self.split = split
        split_dir = self.split_root / split
        speech_records = _load_shard_records(split_dir / "clean_shards.jsonl", None)
        noise_records = _load_shard_records(split_dir / "noise_shards.jsonl", None)
        self.speech_shards = _select_source_shards(speech_records, clean_sources, "speech")
        self.noise_shards = _select_source_shards(noise_records, (noise_source,), "noise")
        if not self.speech_shards:
            raise ValueError(f"no {list(clean_sources)} speech shards found in {split_dir}")
        if not self.noise_shards:
            raise ValueError(f"no {noise_source} noise shards found in {split_dir}")

        self.target_sample_rate = int(target_sample_rate)
        self.cut_duration = cut_duration
        self.samples_per_epoch = int(samples_per_epoch)
        self.clean_shuffle_buffer = max(1, int(clean_shuffle_buffer))
        self.noise_buffer_size = max(2, int(noise_buffer_size))
        self.seed = int(shard_shuffle_seed)
        self.epoch = 0
        self.snr_db_choices = tuple(float(value) for value in snr_db_choices)
        self.max_abs_correlation = float(max_abs_correlation)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.peak_limit = float(peak_limit)
        self.max_tuple_attempts = int(max_tuple_attempts)
        self.expose_clean_for_eval = bool(expose_clean_for_eval)
        self.noise_source = str(noise_source)
        self.reader = WebDatasetAudioReader(
            target_sample_rate=self.target_sample_rate,
            tar_cache_size=tar_cache_size,
        )
        self.route_summary = {
            "split": split,
            "clean_sources": list(clean_sources),
            "noise_source": noise_source,
            "speech_shards": len(self.speech_shards),
            "speech_samples": sum(int(row.get("sample_count") or 0) for row in self.speech_shards),
            "noise_shards": len(self.noise_shards),
            "noise_samples": sum(int(row.get("sample_count") or 0) for row in self.noise_shards),
            "expose_clean_for_eval": self.expose_clean_for_eval,
            "simulation": "none_pure_additive",
        }

    def __len__(self) -> int:
        return self.samples_per_epoch

    def reset_worker_state(self) -> None:
        self.reader.reset_worker_state()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        quota = self.samples_per_epoch // num_workers
        quota += int(worker_id < self.samples_per_epoch % num_workers)
        rng = random.Random(stable_uint32(self.seed, self.epoch, worker_id, "controlled-dnf"))
        speech_iter = _cycle_items(
            self.speech_shards,
            "controlled-speech",
            self.reader,
            worker_id,
            num_workers,
            self.seed,
            self.epoch,
        )
        noise_iter = _cycle_items(
            self.noise_shards,
            "controlled-noise",
            self.reader,
            worker_id,
            num_workers,
            self.seed,
            self.epoch,
        )
        speech_buffer = []
        noise_buffer = []
        emitted = 0
        attempts = 0
        last_rejection: Optional[ControlledSampleRejected] = None
        while emitted < quota:
            if attempts >= self.max_tuple_attempts:
                raise RuntimeError(
                    f"failed to construct controlled tuple after {attempts} attempts; "
                    f"last rejection: {last_rejection}"
                )
            attempts += 1
            speech_item = _pop_random_item(
                speech_buffer, speech_iter, self.clean_shuffle_buffer, rng
            )
            noise1_item = _pop_random_item(
                noise_buffer, noise_iter, self.noise_buffer_size, rng
            )
            noise2_item = _pop_random_item(
                noise_buffer, noise_iter, self.noise_buffer_size, rng
            )
            if _item_id(noise1_item) == _item_id(noise2_item):
                last_rejection = ControlledSampleRejected("n1 and n2 item IDs are identical")
                continue

            target_snr_db = rng.choice(self.snr_db_choices)
            tuple_seed = stable_uint32(
                self.seed,
                self.epoch,
                worker_id,
                emitted,
                attempts,
                _item_id(speech_item),
                _item_id(noise1_item),
                _item_id(noise2_item),
                target_snr_db,
            )
            try:
                mixture = build_controlled_additive_mixture(
                    self.reader.read(speech_item),
                    self.reader.read(noise1_item),
                    self.reader.read(noise2_item),
                    seed=tuple_seed,
                    target_sample_rate=self.target_sample_rate,
                    cut_duration=self.cut_duration,
                    target_snr_db=target_snr_db,
                    max_abs_correlation=self.max_abs_correlation,
                    silence_rms_threshold=self.silence_rms_threshold,
                    peak_limit=self.peak_limit,
                )
            except ControlledSampleRejected as exc:
                last_rejection = exc
                continue

            speech_id = _item_id(speech_item)
            noise1_id = _item_id(noise1_item)
            noise2_id = _item_id(noise2_item)
            uid = f"controlled-{stable_uint32(tuple_seed, speech_id, noise1_id, noise2_id):08x}"
            item = {
                "degraded": mixture.degraded,
                "s_noisy": mixture.noisy_speech_target,
                "added_noise": mixture.artificial_noise,
                "sample_rate": self.target_sample_rate,
                "length": int(mixture.diagnostics["length"]),
                "uid": uid,
                "info": {
                    "uid": uid,
                    "seed": int(tuple_seed),
                    "route_category": "noisy_eq13",
                    "speech_sources": list(speech_item.get("dataset", "").split(",")),
                    "noise_source": self.noise_source,
                    "speech_item_id": speech_id,
                    "noise1_item_id": noise1_id,
                    "noise2_item_id": noise2_id,
                    **mixture.diagnostics,
                },
            }
            if self.expose_clean_for_eval:
                item["eval_clean"] = mixture.clean_speech
            emitted += 1
            attempts = 0
            last_rejection = None
            yield item


class ControlledDNFAdditiveDataLoadIter:
    """Small compatibility wrapper matching the existing DNF loader pattern."""

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 1,
        prefetch: int = 2,
        persistent_workers: bool = False,
        **dataset_kwargs,
    ):
        self.dataset = ControlledDNFAdditiveStreamDataset(**dataset_kwargs)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self):
        self.dataset.set_epoch(self.epoch)
        self.epoch += 1
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(1, self.prefetch)
            kwargs["persistent_workers"] = self.persistent_workers
        return iter(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=controlled_dnf_collate,
                worker_init_fn=gap_worker_init_fn,
                **kwargs,
            )
        )

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


__all__ = [
    "ControlledAdditiveMixture",
    "ControlledDNFAdditiveDataLoadIter",
    "ControlledDNFAdditiveStreamDataset",
    "ControlledSampleRejected",
    "build_controlled_additive_mixture",
    "controlled_dnf_collate",
]
