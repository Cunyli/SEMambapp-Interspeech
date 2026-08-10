"""Frozen-manifest controlled data path for the Phase A DNF comparison.

Every training tuple is reconstructed as ``x = s + n1 + n2`` while the
held-out deployment view is ``y = s + n1``.  A deterministic 20-row block
contains 15 noisy-target rows, four regular clean-target rows, and one weak
(20 dB by default) clean-target row.  The collate function keeps clean and
noisy targets in indexed sub-batches so unavailable supervision is never
materialized as a full-batch tensor.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, get_worker_info


PHASE_A_SCHEMA_VERSION = "dnf_controlled_phase_a_v2"
TRAINING_INPUT_DEFINITION = "x=s+n1+n2"
DEPLOYMENT_INPUT_DEFINITION = "y=s+n1"
SNR_DEFINITION = "10log10(E_s/E_n1)"
ARTIFICIAL_NOISE_ENERGY_POLICY = "E_n2=E_n1"
ROUTE_NOISY = "noisy"
ROUTE_CLEAN_REGULAR = "clean_regular"
ROUTE_CLEAN_WEAK = "clean_weak"
ROUTE_TO_ID = {
    ROUTE_NOISY: 0,
    ROUTE_CLEAN_REGULAR: 1,
    ROUTE_CLEAN_WEAK: 2,
}
NOISE_FAMILIES = ("hvac", "fan", "vehicle_cabin")
NOISE_PAIRING_SAME_FAMILY_IID = "same_family_iid"
NOISE_PAIRING_CROSS_FAMILY_CYCLE = "cross_family_cycle"
NOISE_PAIRING_POLICIES = (
    NOISE_PAIRING_SAME_FAMILY_IID,
    NOISE_PAIRING_CROSS_FAMILY_CYCLE,
)
DEFAULT_NOISE_PAIRING_POLICY = NOISE_PAIRING_SAME_FAMILY_IID
SPEECH_PARTITION_DISJOINT = "disjoint_item_pools"
DEFAULT_SPEECH_PARTITION_POLICY = SPEECH_PARTITION_DISJOINT
DEFAULT_BLOCK_SIZE = 20
DEFAULT_NOISY_PER_BLOCK = 15
DEFAULT_REGULAR_CLEAN_PER_BLOCK = 4
DEFAULT_WEAK_CLEAN_PER_BLOCK = 1
DEFAULT_REGULAR_SNR_DB = (0.0, 5.0, 10.0)
DEFAULT_WEAK_SNR_DB = 20.0


class PhaseADataError(ValueError):
    """A manifest row or reconstructed waveform violates the Phase A contract."""


def stable_uint32(*parts: object) -> int:
    """Return a process-independent unsigned seed derived from arbitrary values."""

    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise PhaseADataError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def phase_a_route_schedule(
    row_count: int,
    *,
    seed: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    noisy_per_block: int = DEFAULT_NOISY_PER_BLOCK,
    regular_clean_per_block: int = DEFAULT_REGULAR_CLEAN_PER_BLOCK,
    weak_clean_per_block: int = DEFAULT_WEAK_CLEAN_PER_BLOCK,
) -> list[str]:
    """Build exact, independently shuffled 20-row route blocks."""

    counts = noisy_per_block + regular_clean_per_block + weak_clean_per_block
    if block_size <= 0 or counts != block_size:
        raise ValueError(
            "route counts must be positive and sum to block_size; "
            f"got block_size={block_size}, counts={counts}"
        )
    if row_count <= 0 or row_count % block_size:
        raise ValueError(
            f"row_count must be a positive multiple of block_size={block_size}, got {row_count}"
        )

    schedule = []
    template = (
        [ROUTE_NOISY] * noisy_per_block
        + [ROUTE_CLEAN_REGULAR] * regular_clean_per_block
        + [ROUTE_CLEAN_WEAK] * weak_clean_per_block
    )
    for block_index in range(row_count // block_size):
        block = list(template)
        random.Random(
            stable_uint32(seed, block_index, "phase-a-route-block")
        ).shuffle(block)
        schedule.extend(block)
    return schedule


def sample_noise_parameters(
    family: str,
    *,
    seed: int,
    sample_rate: int,
) -> dict[str, float | int | str]:
    """Sample deterministic parameters for one stable indoor-noise realization."""

    if family not in NOISE_FAMILIES:
        raise ValueError(f"unsupported noise family {family!r}; expected {NOISE_FAMILIES}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    rng = np.random.default_rng(int(seed))
    nyquist = 0.5 * sample_rate
    if family == "hvac":
        parameters = {
            "family": family,
            "color_alpha": float(rng.uniform(0.7, 1.3)),
            "highpass_hz": float(rng.uniform(12.0, 35.0)),
            "lowpass_hz": float(min(rng.uniform(350.0, 900.0), 0.9 * nyquist)),
            "fundamental_hz": float(rng.choice((50.0, 60.0))),
            "harmonic_count": int(rng.integers(2, 5)),
            "tone_gain": float(rng.uniform(0.02, 0.05)),
            "am_rate_hz": float(rng.uniform(0.08, 0.35)),
            "am_depth": float(rng.uniform(0.02, 0.08)),
        }
    elif family == "fan":
        parameters = {
            "family": family,
            "color_alpha": float(rng.uniform(0.2, 0.8)),
            "highpass_hz": float(rng.uniform(25.0, 70.0)),
            "lowpass_hz": float(min(rng.uniform(1200.0, 3200.0), 0.9 * nyquist)),
            "fundamental_hz": float(rng.uniform(72.0, 190.0)),
            "harmonic_count": int(rng.integers(3, 7)),
            "tone_gain": float(rng.uniform(0.05, 0.12)),
            "am_rate_hz": float(rng.uniform(0.3, 1.4)),
            "am_depth": float(rng.uniform(0.04, 0.14)),
        }
    else:
        parameters = {
            "family": family,
            "color_alpha": float(rng.uniform(1.0, 1.8)),
            "highpass_hz": float(rng.uniform(8.0, 25.0)),
            "lowpass_hz": float(min(rng.uniform(500.0, 1800.0), 0.9 * nyquist)),
            "fundamental_hz": float(rng.uniform(24.0, 58.0)),
            "harmonic_count": int(rng.integers(3, 8)),
            "tone_gain": float(rng.uniform(0.04, 0.10)),
            "am_rate_hz": float(rng.uniform(0.05, 0.25)),
            "am_depth": float(rng.uniform(0.03, 0.10)),
        }
    return parameters


def _unit_rms(waveform: np.ndarray, *, name: str) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float64)
    waveform = waveform - waveform.mean()
    rms = math.sqrt(float(np.mean(np.square(waveform), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= np.finfo(np.float64).tiny:
        raise PhaseADataError(f"{name} is silent or non-finite")
    return waveform / rms


def generate_parameterized_noise(
    *,
    family: str,
    seed: int,
    sample_rate: int,
    sample_count: int,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Generate a deterministic mono HVAC, fan, or vehicle-cabin waveform."""

    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("sample_count and sample_rate must be positive")
    params = dict(
        parameters
        if parameters is not None
        else sample_noise_parameters(family, seed=seed, sample_rate=sample_rate)
    )
    if params.get("family", family) != family:
        raise PhaseADataError("noise family and parameter family disagree")

    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal(sample_count)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate)
    safe_frequency = np.maximum(frequencies, 1.0)
    color = safe_frequency ** (-0.5 * float(params["color_alpha"]))
    highpass_hz = float(params["highpass_hz"])
    lowpass_hz = float(params["lowpass_hz"])
    highpass = frequencies / np.sqrt(np.square(frequencies) + highpass_hz**2)
    lowpass = 1.0 / np.sqrt(1.0 + (frequencies / lowpass_hz) ** 8)
    shaping = color * highpass * lowpass
    shaping[0] = 0.0
    colored = np.fft.irfft(spectrum * shaping, n=sample_count)
    colored = _unit_rms(colored, name=f"{family} broadband component")

    time = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    tones = np.zeros(sample_count, dtype=np.float64)
    fundamental = float(params["fundamental_hz"])
    harmonic_count = int(params["harmonic_count"])
    for harmonic in range(1, harmonic_count + 1):
        frequency = harmonic * fundamental
        if frequency >= 0.45 * sample_rate:
            break
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        tones += math.pow(harmonic, -1.2) * np.sin(
            2.0 * math.pi * frequency * time + phase
        )
    tones = _unit_rms(tones, name=f"{family} tonal component")

    tone_gain = float(params["tone_gain"])
    amplitude_modulation = 1.0 + float(params["am_depth"]) * np.sin(
        2.0 * math.pi * float(params["am_rate_hz"]) * time
        + float(rng.uniform(0.0, 2.0 * math.pi))
    )
    waveform = (colored + tone_gain * tones) * amplitude_modulation
    return _unit_rms(waveform, name=family)[None, :].astype(np.float32)


def _mono_float64(waveform: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(waveform)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[-1] == 0:
        raise PhaseADataError(f"{name} must have shape [1, time], got {array.shape}")
    if not np.isfinite(array).all():
        raise PhaseADataError(f"{name} contains non-finite samples")
    return array.astype(np.float64, copy=False)


def _crop_or_wrap(
    waveform: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    array = _mono_float64(waveform, name="clean speech")
    rng = np.random.default_rng(int(seed))
    length = array.shape[-1]
    if length >= sample_count:
        offset = int(rng.integers(0, length - sample_count + 1))
        return array[:, offset : offset + sample_count]
    offset = int(rng.integers(0, max(length, 1)))
    indices = (np.arange(sample_count) + offset) % length
    return array[:, indices]


def _energy(waveform: np.ndarray) -> float:
    return float(np.sum(np.square(waveform), dtype=np.float64))


def _absolute_correlation(first: np.ndarray, second: np.ndarray) -> float:
    denominator = math.sqrt(_energy(first) * _energy(second))
    if denominator <= 0.0:
        return math.inf
    return abs(float(np.sum(first * second, dtype=np.float64))) / denominator


def build_phase_a_mixture(
    clean_speech: np.ndarray,
    noise1: np.ndarray,
    noise2: np.ndarray,
    *,
    target_snr_db: float,
    peak_limit: float = 0.95,
    max_abs_correlation: float = 0.95,
    silence_rms_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Scale components and return exact float32 ``s``, targets, and ``x``.

    Statistical independence is established by frozen, role-specific seeds.
    The paper-mechanism manifest draws both noises from the same parameterized
    family so their roles are not semantically separable.  Finite one-second
    windows can still have non-trivial sample correlation, especially for
    periodic speech and machinery tones; the hard guard therefore catches
    near-duplicate components while the smoke report exposes the correlations
    for a separate approximate-orthogonality gate.
    """

    clean = _mono_float64(clean_speech, name="clean speech")
    n1 = _mono_float64(noise1, name="noise1")
    n2 = _mono_float64(noise2, name="noise2")
    if clean.shape != n1.shape or clean.shape != n2.shape:
        raise PhaseADataError(
            f"component shapes differ: clean={clean.shape}, n1={n1.shape}, n2={n2.shape}"
        )
    if not math.isfinite(float(target_snr_db)):
        raise ValueError("target_snr_db must be finite")
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")

    clean -= clean.mean(axis=-1, keepdims=True)
    n1 -= n1.mean(axis=-1, keepdims=True)
    n2 -= n2.mean(axis=-1, keepdims=True)
    sample_count = clean.shape[-1]
    rms = {
        "clean": math.sqrt(_energy(clean) / sample_count),
        "noise1": math.sqrt(_energy(n1) / sample_count),
        "noise2": math.sqrt(_energy(n2) / sample_count),
    }
    silent = [name for name, value in rms.items() if value <= silence_rms_threshold]
    if silent:
        raise PhaseADataError(f"silent or near-silent components: {silent}")

    correlations = {
        "speech_noise1": _absolute_correlation(clean, n1),
        "speech_noise2": _absolute_correlation(clean, n2),
        "noise1_noise2": _absolute_correlation(n1, n2),
    }
    failed = {
        name: value
        for name, value in correlations.items()
        if value >= max_abs_correlation
    }
    if failed:
        raise PhaseADataError(
            f"absolute correlation exceeds {max_abs_correlation}: {failed}"
        )

    # ``target_snr_db`` is the deployment SNR of y = s + n1.  The artificial
    # training noise n2 is an independent, equal-energy draw from the same
    # family.  Consequently x = y + n2 is about 3.01 dB harder than the
    # single-noise deployment input.  Keeping these two views explicit avoids
    # accidentally evaluating DNF on its double-noise training construction.
    contracted_noise_energy = _energy(clean) * 10.0 ** (
        -float(target_snr_db) / 10.0
    )
    n1 *= math.sqrt(contracted_noise_energy / _energy(n1))
    n2 *= math.sqrt(contracted_noise_energy / _energy(n2))
    degraded = clean + n1 + n2
    peak_before_gain = float(np.max(np.abs(degraded)))
    common_gain = min(
        1.0,
        peak_limit / max(peak_before_gain, np.finfo(np.float64).tiny),
    )
    clean *= common_gain
    n1 *= common_gain
    n2 *= common_gain

    clean32 = clean.astype(np.float32)
    noise1_32 = n1.astype(np.float32)
    noise2_32 = n2.astype(np.float32)
    noisy_speech32 = (clean32 + noise1_32).astype(np.float32)
    mixture_noise32 = (noise1_32 + noise2_32).astype(np.float32)
    degraded32 = (noisy_speech32 + noise2_32).astype(np.float32)
    clean_energy = _energy(clean32.astype(np.float64))
    noise1_energy = _energy(noise1_32.astype(np.float64))
    training_noise_energy = _energy(
        (noise1_32 + noise2_32).astype(np.float64)
    )
    return {
        "clean_speech": clean32,
        "noise1": noise1_32,
        "noise2": noise2_32,
        "noisy_speech_target": noisy_speech32,
        "mixture_noise": mixture_noise32,
        "model_input": degraded32,
        "diagnostics": {
            "target_deployment_snr_db": float(target_snr_db),
            "measured_deployment_snr_db": 10.0
            * math.log10(clean_energy / noise1_energy),
            "measured_training_tuple_snr_db": 10.0
            * math.log10(clean_energy / training_noise_energy),
            "absolute_correlations": correlations,
            "noise1_energy": noise1_energy,
            "noise2_energy": _energy(noise2_32.astype(np.float64)),
            "common_gain": float(common_gain),
            "peak_before_common_gain": peak_before_gain,
            "peak_after_common_gain": float(np.max(np.abs(degraded32))),
            "max_additive_error": float(
                np.max(np.abs(degraded32 - noisy_speech32 - noise2_32))
            ),
        },
    }


def build_phase_a_manifest_rows(
    speech_items: Sequence[Mapping[str, Any]],
    *,
    row_count: int,
    seed: int,
    split: str,
    sample_rate: int = 16000,
    sample_count: int = 16000,
    regular_snr_db: Sequence[float] = DEFAULT_REGULAR_SNR_DB,
    weak_snr_db: float = DEFAULT_WEAK_SNR_DB,
    noise_families: Sequence[str] = NOISE_FAMILIES,
    noise_pairing_policy: str = DEFAULT_NOISE_PAIRING_POLICY,
    speech_partition_policy: str = DEFAULT_SPEECH_PARTITION_POLICY,
) -> list[dict[str, Any]]:
    """Create frozen tuple recipes; waveform decoding happens in the Dataset."""

    if not speech_items:
        raise ValueError("speech_items must be non-empty")
    if not regular_snr_db:
        raise ValueError("regular_snr_db must be non-empty")
    if not noise_families or not set(noise_families).issubset(NOISE_FAMILIES):
        raise ValueError(f"noise_families must be drawn from {NOISE_FAMILIES}")
    if noise_pairing_policy not in NOISE_PAIRING_POLICIES:
        raise ValueError(
            "noise_pairing_policy must be one of "
            f"{NOISE_PAIRING_POLICIES}, got {noise_pairing_policy!r}"
        )
    if speech_partition_policy != SPEECH_PARTITION_DISJOINT:
        raise ValueError(
            "speech_partition_policy must be "
            f"{SPEECH_PARTITION_DISJOINT!r}, got {speech_partition_policy!r}"
        )
    schedule = phase_a_route_schedule(row_count, seed=seed)
    rng = random.Random(stable_uint32(seed, split, "phase-a-manifest"))
    ranked_speech = sorted(
        (dict(item) for item in speech_items),
        key=lambda item: stable_uint32(
            seed,
            split,
            "speech-partition",
            _canonical_json(item),
        ),
    )
    clean_pool_size = max(1, len(ranked_speech) // 4)
    clean_speech_pool = ranked_speech[:clean_pool_size]
    noisy_speech_pool = ranked_speech[clean_pool_size:]
    clean_row_count = sum(route != ROUTE_NOISY for route in schedule)
    noisy_row_count = sum(route == ROUTE_NOISY for route in schedule)
    if len(clean_speech_pool) < clean_row_count:
        raise ValueError(
            f"clean speech pool has {len(clean_speech_pool)} items but "
            f"{clean_row_count} rows are required"
        )
    if len(noisy_speech_pool) < noisy_row_count:
        raise ValueError(
            f"noisy speech pool has {len(noisy_speech_pool)} items but "
            f"{noisy_row_count} rows are required"
        )
    rng.shuffle(clean_speech_pool)
    rng.shuffle(noisy_speech_pool)
    clean_speech_index = 0
    noisy_speech_index = 0
    rows = []
    for row_index, route in enumerate(schedule):
        family = str(noise_families[row_index % len(noise_families)])
        if noise_pairing_policy == NOISE_PAIRING_SAME_FAMILY_IID:
            artificial_family = family
        else:
            artificial_family = str(
                noise_families[(row_index + 1) % len(noise_families)]
            )
        if route == ROUTE_NOISY:
            speech = dict(noisy_speech_pool[noisy_speech_index])
            speech_partition = "noisy_pool"
            noisy_speech_index += 1
        else:
            speech = dict(clean_speech_pool[clean_speech_index])
            speech_partition = "clean_pool"
            clean_speech_index += 1
        noise1_seed = stable_uint32(seed, split, row_index, "noise1")
        noise2_seed = stable_uint32(seed, split, row_index, "noise2")
        if noise1_seed == noise2_seed:
            noise2_seed = stable_uint32(seed, split, row_index, "noise2-distinct")
        target_snr_db = (
            float(weak_snr_db)
            if route == ROUTE_CLEAN_WEAK
            else float(rng.choice(tuple(float(value) for value in regular_snr_db)))
        )
        row_without_uid = {
            "schema_version": PHASE_A_SCHEMA_VERSION,
            "split": str(split),
            "row_index": int(row_index),
            "route": route,
            "weak_degradation": route == ROUTE_CLEAN_WEAK,
            "speech_source_category": "clean_strict",
            "speech": speech,
            "speech_partition_policy": speech_partition_policy,
            "speech_partition": speech_partition,
            "speech_crop_seed": stable_uint32(seed, split, row_index, "speech-crop"),
            "noise_pairing_policy": noise_pairing_policy,
            "noise_family": family,
            "artificial_noise_family": artificial_family,
            "noise1": {
                "source_id": f"{family}-n1-{noise1_seed:08x}",
                "family": family,
                "seed": int(noise1_seed),
                "parameters": sample_noise_parameters(
                    family,
                    seed=noise1_seed,
                    sample_rate=sample_rate,
                ),
            },
            "noise2": {
                "source_id": f"{artificial_family}-n2-{noise2_seed:08x}",
                "family": artificial_family,
                "seed": int(noise2_seed),
                "parameters": sample_noise_parameters(
                    artificial_family,
                    seed=noise2_seed,
                    sample_rate=sample_rate,
                ),
            },
            "target_snr_db": target_snr_db,
            "snr_definition": SNR_DEFINITION,
            "training_input": TRAINING_INPUT_DEFINITION,
            "deployment_validation_input": DEPLOYMENT_INPUT_DEFINITION,
            "artificial_noise_energy_policy": ARTIFICIAL_NOISE_ENERGY_POLICY,
            "sample_rate": int(sample_rate),
            "sample_count": int(sample_count),
            "manifest_seed": int(seed),
        }
        uid_digest = hashlib.sha256(
            _canonical_json(row_without_uid).encode("utf-8")
        ).hexdigest()
        rows.append({"uid": f"phase-a-{uid_digest[:24]}", **row_without_uid})
    return rows


class _ManifestSpeechReader:
    """Read filesystem or tar-member speech locators stored in manifest rows."""

    def __init__(self, root: str | Path | None, target_sample_rate: int):
        self.root = Path(root) if root else None
        self.target_sample_rate = int(target_sample_rate)
        self._tar_cache: dict[Path, tarfile.TarFile] = {}

    def reset_worker_state(self) -> None:
        for handle in self._tar_cache.values():
            handle.close()
        self._tar_cache = {}

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute() or self.root is None:
            return path
        return self.root / path

    def read(self, locator: Mapping[str, Any]) -> np.ndarray:
        if "npy_path" in locator:
            waveform = np.load(self._resolve(str(locator["npy_path"])))
            return _mono_float64(waveform, name="manifest speech").astype(np.float32)

        if "path" in locator or "audio_path" in locator:
            path = self._resolve(str(locator.get("path", locator.get("audio_path"))))
            waveform, sample_rate = torchaudio.load(path)
        else:
            shard_dir = self._resolve(str(locator["_shard_dir"]))
            tar_path = shard_dir / str(locator["shard"])
            tar = self._tar_cache.get(tar_path)
            if tar is None:
                tar = tarfile.open(tar_path, "r:")
                self._tar_cache[tar_path] = tar
            member = tar.extractfile(str(locator["audio_member"]))
            if member is None:
                raise FileNotFoundError(
                    f"{locator['audio_member']} not found in {tar_path}"
                )
            waveform, sample_rate = torchaudio.load(io.BytesIO(member.read()))

        waveform = waveform.float()
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                int(sample_rate),
                self.target_sample_rate,
            )
        return waveform.numpy().astype(np.float32, copy=False)


def _resolve_manifest_path(
    contract_path: str | Path,
    *,
    split: str,
) -> tuple[Path, dict[str, Any]]:
    contract = Path(contract_path)
    if contract.suffix.lower() == ".jsonl":
        return contract, {}
    with contract.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifests = payload.get("manifests", {})
    entry = manifests.get(split)
    if entry is None:
        raise PhaseADataError(f"contract {contract} has no manifest for split={split!r}")
    relative = entry.get("path") if isinstance(entry, dict) else entry
    manifest_path = Path(relative)
    if not manifest_path.is_absolute():
        manifest_path = contract.parent / manifest_path
    return manifest_path, payload


class PhaseAControlledStreamDataset(Dataset):
    """Map-style frozen-recipe Dataset with the historical stream-style name."""

    def __init__(
        self,
        split_root: str | Path | None,
        contract_path: str | Path,
        *,
        split: str = "train",
        samples_per_epoch: int | None = None,
        target_sample_rate: int = 16000,
        cut_duration: float = 1.0,
        seed: int = 1234,
        expose_clean_for_eval: bool = False,
        speech_reader: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    ):
        manifest_path, contract = _resolve_manifest_path(contract_path, split=split)
        rows = read_jsonl(manifest_path)
        if not rows:
            raise PhaseADataError(f"empty Phase A manifest: {manifest_path}")
        for row in rows:
            if row.get("schema_version") != PHASE_A_SCHEMA_VERSION:
                raise PhaseADataError(
                    f"unexpected schema_version in {manifest_path}: "
                    f"{row.get('schema_version')!r}"
                )
            if row.get("split") != split:
                raise PhaseADataError(
                    f"manifest row split {row.get('split')!r} != requested {split!r}"
                )
            if row.get("route") not in ROUTE_TO_ID:
                raise PhaseADataError(f"unsupported route {row.get('route')!r}")
            expected_semantics = {
                "snr_definition": SNR_DEFINITION,
                "training_input": TRAINING_INPUT_DEFINITION,
                "deployment_validation_input": DEPLOYMENT_INPUT_DEFINITION,
                "artificial_noise_energy_policy": ARTIFICIAL_NOISE_ENERGY_POLICY,
            }
            mismatches = {
                key: row.get(key)
                for key, expected in expected_semantics.items()
                if row.get(key) != expected
            }
            if mismatches:
                raise PhaseADataError(
                    f"manifest row has stale mixture semantics: {mismatches}"
                )
        requested = len(rows) if samples_per_epoch is None else int(samples_per_epoch)
        if requested <= 0 or requested > len(rows):
            raise PhaseADataError(
                f"samples_per_epoch must be in [1, {len(rows)}], got {requested}"
            )
        expected_samples = int(round(float(cut_duration) * int(target_sample_rate)))
        if expected_samples <= 0:
            raise ValueError("cut_duration must produce a positive sample count")
        contract_seed = contract.get("seed")
        if contract_seed is not None and int(contract_seed) != int(seed):
            raise PhaseADataError(
                f"contract seed {contract_seed} != requested seed {seed}"
            )

        self.rows = rows[:requested]
        self.split_root = Path(split_root) if split_root else None
        self.split = str(split)
        self.target_sample_rate = int(target_sample_rate)
        self.sample_count = expected_samples
        self.seed = int(seed)
        self.expose_clean_for_eval = bool(expose_clean_for_eval)
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_rows_sha256(self.rows)
        self.reader = _ManifestSpeechReader(
            self.split_root,
            self.target_sample_rate,
        )
        self.speech_reader = speech_reader
        self.route_summary = {
            "schema_version": PHASE_A_SCHEMA_VERSION,
            "split": self.split,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "sample_count": len(self.rows),
            "route_counts": {
                route: sum(row["route"] == route for row in self.rows)
                for route in ROUTE_TO_ID
            },
            "noise_families": sorted(
                {
                    str(spec["family"])
                    for row in self.rows
                    for spec in (row["noise1"], row["noise2"])
                }
            ),
            "noise_pairing_policies": sorted(
                {
                    str(row.get("noise_pairing_policy", "legacy_unlabeled"))
                    for row in self.rows
                }
            ),
            "speech_partition_policies": sorted(
                {
                    str(row.get("speech_partition_policy", "legacy_unlabeled"))
                    for row in self.rows
                }
            ),
            "expose_clean_for_eval": self.expose_clean_for_eval,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def reset_worker_state(self) -> None:
        self.reader.reset_worker_state()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["reader"] = _ManifestSpeechReader(
            self.split_root,
            self.target_sample_rate,
        )
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        clean_raw = (
            self.speech_reader(row["speech"])
            if self.speech_reader is not None
            else self.reader.read(row["speech"])
        )
        clean = _crop_or_wrap(
            clean_raw,
            sample_count=int(row["sample_count"]),
            seed=int(row["speech_crop_seed"]),
        )
        if int(row["sample_rate"]) != self.target_sample_rate:
            raise PhaseADataError(
                f"manifest sample rate {row['sample_rate']} != dataset "
                f"target_sample_rate {self.target_sample_rate}"
            )
        if int(row["sample_count"]) != self.sample_count:
            raise PhaseADataError(
                f"manifest sample_count {row['sample_count']} != expected {self.sample_count}"
            )

        noise1_spec = row["noise1"]
        noise2_spec = row["noise2"]
        if noise1_spec["source_id"] == noise2_spec["source_id"]:
            raise PhaseADataError("noise1 and noise2 source IDs must be distinct")
        noise1 = generate_parameterized_noise(
            family=str(noise1_spec["family"]),
            seed=int(noise1_spec["seed"]),
            sample_rate=self.target_sample_rate,
            sample_count=self.sample_count,
            parameters=noise1_spec["parameters"],
        )
        noise2 = generate_parameterized_noise(
            family=str(noise2_spec["family"]),
            seed=int(noise2_spec["seed"]),
            sample_rate=self.target_sample_rate,
            sample_count=self.sample_count,
            parameters=noise2_spec["parameters"],
        )
        mixture = build_phase_a_mixture(
            clean,
            noise1,
            noise2,
            target_snr_db=float(row["target_snr_db"]),
        )
        route = str(row["route"])
        item = {
            "model_input": mixture["model_input"],
            "route": route,
            "weak_degradation": bool(row["weak_degradation"]),
            "sample_rate": self.target_sample_rate,
            "sample_count": self.sample_count,
            "uid": str(row["uid"]),
            "info": {
                **row,
                "mixture_diagnostics": mixture["diagnostics"],
            },
        }
        if route in {ROUTE_CLEAN_REGULAR, ROUTE_CLEAN_WEAK}:
            item["clean_speech"] = mixture["clean_speech"]
            item["mixture_noise"] = mixture["mixture_noise"]
        else:
            item["noisy_speech_target"] = mixture["noisy_speech_target"]
            item["artificial_noise"] = mixture["noise2"]
        if self.expose_clean_for_eval:
            item["eval_clean_speech"] = mixture["clean_speech"]
            item["eval_model_input"] = mixture["noisy_speech_target"]
        return item


def _stack_numpy(items: Sequence[np.ndarray], *, sample_count: int) -> torch.Tensor:
    if not items:
        return torch.empty((0, sample_count), dtype=torch.float32)
    return torch.from_numpy(np.concatenate(items, axis=0)).float()


def phase_a_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate model inputs plus route-indexed supervision sub-batches."""

    if not batch:
        raise ValueError("batch must be non-empty")
    sample_counts = {int(item["sample_count"]) for item in batch}
    if len(sample_counts) != 1:
        raise PhaseADataError(f"mixed sample counts in batch: {sample_counts}")
    sample_count = next(iter(sample_counts))
    clean_indices = [
        index
        for index, item in enumerate(batch)
        if item["route"] in {ROUTE_CLEAN_REGULAR, ROUTE_CLEAN_WEAK}
    ]
    noisy_indices = [
        index for index, item in enumerate(batch) if item["route"] == ROUTE_NOISY
    ]
    if len(clean_indices) + len(noisy_indices) != len(batch):
        raise PhaseADataError("batch contains an unknown route")

    eval_visibility = ["eval_clean_speech" in item for item in batch]
    if any(eval_visibility) and not all(eval_visibility):
        raise PhaseADataError("eval clean visibility must be consistent within a batch")
    output = {
        "model_input_wav": _stack_numpy(
            [item["model_input"] for item in batch],
            sample_count=sample_count,
        ),
        "clean_indices": torch.tensor(clean_indices, dtype=torch.long),
        "clean_speech_wav": _stack_numpy(
            [batch[index]["clean_speech"] for index in clean_indices],
            sample_count=sample_count,
        ),
        "mixture_noise_wav": _stack_numpy(
            [batch[index]["mixture_noise"] for index in clean_indices],
            sample_count=sample_count,
        ),
        "noisy_indices": torch.tensor(noisy_indices, dtype=torch.long),
        "noisy_speech_target_wav": _stack_numpy(
            [batch[index]["noisy_speech_target"] for index in noisy_indices],
            sample_count=sample_count,
        ),
        "artificial_noise_wav": _stack_numpy(
            [batch[index]["artificial_noise"] for index in noisy_indices],
            sample_count=sample_count,
        ),
        "route_id": torch.tensor(
            [ROUTE_TO_ID[str(item["route"])] for item in batch],
            dtype=torch.long,
        ),
        "route": [str(item["route"]) for item in batch],
        "weak_mask": torch.tensor(
            [bool(item["weak_degradation"]) for item in batch],
            dtype=torch.bool,
        ),
        "sample_uid": [str(item["uid"]) for item in batch],
        "sample_rate": torch.tensor(
            [int(item["sample_rate"]) for item in batch],
            dtype=torch.long,
        ),
        "info": [item["info"] for item in batch],
    }
    if all(eval_visibility):
        output["eval_clean_wav"] = _stack_numpy(
            [item["eval_clean_speech"] for item in batch],
            sample_count=sample_count,
        )
        output["eval_model_input_wav"] = _stack_numpy(
            [item["eval_model_input"] for item in batch],
            sample_count=sample_count,
        )
    return output


def gap_worker_init_fn(worker_id: int) -> None:
    """Reset per-worker tar handles and seed standard RNGs deterministically."""

    worker = get_worker_info()
    if worker is None:
        return
    dataset = worker.dataset
    if hasattr(dataset, "reset_worker_state"):
        dataset.reset_worker_state()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(stable_uint32(worker_seed, worker_id, "python"))
    np.random.seed(stable_uint32(worker_seed, worker_id, "numpy"))


__all__ = [
    "ARTIFICIAL_NOISE_ENERGY_POLICY",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_REGULAR_SNR_DB",
    "DEFAULT_WEAK_SNR_DB",
    "DEPLOYMENT_INPUT_DEFINITION",
    "NOISE_FAMILIES",
    "PHASE_A_SCHEMA_VERSION",
    "PhaseAControlledStreamDataset",
    "PhaseADataError",
    "ROUTE_CLEAN_REGULAR",
    "ROUTE_CLEAN_WEAK",
    "ROUTE_NOISY",
    "ROUTE_TO_ID",
    "SNR_DEFINITION",
    "TRAINING_INPUT_DEFINITION",
    "build_phase_a_manifest_rows",
    "build_phase_a_mixture",
    "gap_worker_init_fn",
    "generate_parameterized_noise",
    "manifest_rows_sha256",
    "phase_a_collate",
    "phase_a_route_schedule",
    "read_jsonl",
    "sample_noise_parameters",
    "stable_uint32",
    "write_jsonl",
]
