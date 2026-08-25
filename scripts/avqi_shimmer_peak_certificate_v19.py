"""Fail-closed paired peak certificate for the Shimmer-dB metric high-pass.

The certificate is deliberately independent of pulse or source topology.  It
uses a base waveform peak upper bound plus a circular-convolution bound on the
PCM16 candidate-minus-base perturbation.  The only reusable state is a
read-only Stop-Hann response certificate keyed by FFT length.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


FROZEN_NUMPY_HIGHPASS_MODE = (
    "numpy_official_praat_6_1_38_stop_hann_0_34_0p1"
)
SINC70_ABSOLUTE_WEIGHT_BOUND = 5.2
PEAK_SCALE_TRIGGER = 0.999
PCM16_SCALE = 32768.0
FLOAT64_EPSILON = np.finfo(np.float64).eps
NUMERICAL_SAFETY_FACTOR = 4096.0
FFT_ROUNDOFF_TRANSFORM_COUNT = 2


def pcm16_roundtrip_values_to_codes(values: np.ndarray) -> np.ndarray:
    """Convert worker PCM16 roundtrip values to exact signed integer codes.

    The certificate must never silently quantize an arbitrary floating-point
    waveform.  Its caller first performs the worker's soundfile PCM16
    roundtrip, and this function proves that every supplied value is exactly
    representable by one signed PCM16 code before returning those codes.
    """

    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise ValueError("PCM16 roundtrip values must use worker float64 dtype")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("PCM16 roundtrip values must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError("PCM16 roundtrip values contain nonfinite samples")
    scaled = array * PCM16_SCALE
    rounded = np.rint(scaled)
    if np.any(rounded < -32768.0) or np.any(rounded > 32767.0):
        raise ValueError("PCM16 roundtrip values exceed signed PCM16 range")
    reconstructed = rounded / PCM16_SCALE
    if not np.array_equal(array, reconstructed):
        raise ValueError("input is not an exact worker PCM16 roundtrip")
    codes = np.ascontiguousarray(rounded.astype(np.int16))
    if not np.array_equal(codes.astype(np.float64) / PCM16_SCALE, array):
        raise ValueError("PCM16 code reconstruction drift")
    return codes


def _require_pcm16_codes(values: np.ndarray, label: str) -> np.ndarray:
    codes = np.asarray(values)
    if codes.dtype != np.dtype(np.int16):
        raise ValueError(f"{label} must contain exact signed PCM16 integer codes")
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError(f"{label} must be a nonempty PCM16 code vector")
    return np.ascontiguousarray(codes)


def _pcm16_codes_sha256(codes: np.ndarray) -> str:
    little_endian = codes.astype("<i2", copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def power_of_two_fft_length(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("Stop-Hann certificate requires a nonempty waveform")
    fft_length = 2
    while fft_length < sample_count:
        fft_length *= 2
    return fft_length


def stop_hann_impulse_l1_certificate(
    response: np.ndarray,
    fft_length: int,
) -> dict[str, Any]:
    if fft_length < 2 or fft_length & (fft_length - 1):
        raise ValueError("Stop-Hann certificate FFT length must be a power of two")
    values = np.asarray(response)
    expected_bins = fft_length // 2 + 1
    if values.dtype != np.dtype(np.float64):
        raise ValueError("Stop-Hann response must retain frozen float64 dtype")
    if values.ndim != 1 or values.size != expected_bins:
        raise ValueError("Stop-Hann response/FFT-length mismatch")
    if values.flags.writeable:
        raise ValueError("Stop-Hann response must be frozen read-only")
    if not np.isfinite(values).all():
        raise ValueError("Stop-Hann response contains nonfinite bins")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Stop-Hann response is outside the frozen [0, 1] range")
    impulse = np.fft.irfft(values, n=fft_length)
    raw_l1_longdouble = np.sum(
        np.abs(impulse).astype(np.longdouble),
        dtype=np.longdouble,
    )
    raw_l1 = float(raw_l1_longdouble)

    # A radix-2 inverse real FFT forms every output from the DC/Nyquist bins
    # and two contributions per interior bin.  Bounding the accumulated DFT
    # roundoff by eps * log2(N) times that weighted spectrum L1, then applying
    # a 4096x safety factor, covers construction of the impulse itself.  This
    # term is separate from the subsequent absolute-value/L1 summation error.
    weighted_response_l1 = float(
        np.abs(values[0])
        + np.abs(values[-1])
        + 2.0
        * np.sum(
            np.abs(values[1:-1]).astype(np.longdouble),
            dtype=np.longdouble,
        )
    )
    irfft_construction_epsilon = (
        NUMERICAL_SAFETY_FACTOR
        * FLOAT64_EPSILON
        * max(math.log2(fft_length), 1.0)
        * max(weighted_response_l1, 1.0)
    )
    summation_epsilon = (
        NUMERICAL_SAFETY_FACTOR
        * FLOAT64_EPSILON
        * fft_length
        * max(raw_l1 + irfft_construction_epsilon, 1.0)
    )
    upper_bound = float(
        np.nextafter(
            raw_l1 + irfft_construction_epsilon + summation_epsilon,
            math.inf,
        )
    )
    return {
        "fft_length": fft_length,
        "response_bin_count": int(values.size),
        "response_sha256": hashlib.sha256(
            values.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "response_weighted_spectrum_l1": weighted_response_l1,
        "impulse_l1_observed": raw_l1,
        "impulse_l1_irfft_construction_epsilon": (
            irfft_construction_epsilon
        ),
        "impulse_l1_summation_epsilon": summation_epsilon,
        "impulse_l1_upper_bound": upper_bound,
        "impulse_l1_bound_terms_recorded": True,
        "response_cache_key": f"stop_hann_fft_length_{fft_length}",
        "response_cache_waveform_dependent": False,
        "response_read_only": not values.flags.writeable,
    }


def base_sinc70_peak_upper_bound(
    highpass_timing: dict[str, Any],
) -> dict[str, Any]:
    highpass_mode = highpass_timing.get("highpass_mode")
    if highpass_mode != FROZEN_NUMPY_HIGHPASS_MODE:
        raise ValueError("base high-pass is not the frozen NumPy Stop-Hann mode")
    absolute_weight_bound = highpass_timing.get(
        "highpass_sinc70_absolute_weight_bound"
    )
    if absolute_weight_bound != SINC70_ABSOLUTE_WEIGHT_BOUND:
        raise ValueError("base high-pass Sinc70 absolute-weight bound drift")

    peak = highpass_timing.get("highpass_peak_value")
    local_upper = highpass_timing.get("highpass_sinc70_peak_upper_bound")
    mode = str(highpass_timing.get("highpass_peak_check_mode", ""))
    skipped = highpass_timing.get("highpass_sinc70_skipped")
    scaled = highpass_timing.get("highpass_peak_scaled")
    sample_abs_max = highpass_timing.get("highpass_sample_abs_max")
    if not isinstance(skipped, bool) or not isinstance(scaled, bool):
        raise ValueError("base high-pass skip/scale flags must be booleans")
    if sample_abs_max is None or local_upper is None:
        raise ValueError("base high-pass lacks NumPy Stop-Hann bound fields")
    sample_abs_max = float(sample_abs_max)
    local_upper = float(local_upper)
    if (
        not math.isfinite(sample_abs_max)
        or sample_abs_max < 0.0
        or not math.isfinite(local_upper)
        or local_upper < 0.0
    ):
        raise ValueError("base high-pass local Sinc70 bound is invalid")
    expected_local_upper = sample_abs_max * SINC70_ABSOLUTE_WEIGHT_BOUND
    if local_upper != expected_local_upper:
        raise ValueError("base high-pass sample maximum/local bound mismatch")

    if mode == "exact_praat_sinc70":
        if skipped or peak is None:
            raise ValueError("exact base peak-check timing combination is invalid")
        observed = float(peak)
        source = "base_exact_praat_sinc70_peak"
        if not math.isfinite(observed) or observed < 0.0:
            raise ValueError("invalid base exact Sinc70 peak")
        if observed > float(np.nextafter(local_upper, math.inf)):
            raise ValueError("base exact peak exceeds its Sinc70 absolute bound")
        if scaled != (observed > PEAK_SCALE_TRIGGER):
            raise ValueError("base exact peak/scale decision is inconsistent")
    elif mode == "proven_safe_sinc70_l1_upper_bound":
        if not skipped or peak is not None or scaled:
            raise ValueError("safe-bound base peak-check timing combination is invalid")
        if not local_upper < PEAK_SCALE_TRIGGER:
            raise ValueError("safe-bound base timing does not prove no scaling")
        observed = local_upper
        source = "base_proven_safe_sinc70_l1_upper_bound"
    else:
        raise ValueError("base high-pass peak-check mode is not certificate-compatible")

    numerical_epsilon = (
        NUMERICAL_SAFETY_FACTOR
        * FLOAT64_EPSILON
        * max(observed, 1.0)
    )
    upper_bound = float(
        np.nextafter(observed + numerical_epsilon, math.inf)
    )
    return {
        "base_peak_check_mode": mode,
        "base_peak_bound_source": source,
        "base_peak_observed_or_local_bound": observed,
        "base_peak_numerical_epsilon": numerical_epsilon,
        "base_peak_upper_bound": upper_bound,
        "base_highpass_mode": highpass_mode,
        "base_highpass_sinc70_skipped": skipped,
        "base_highpass_peak_scaled": scaled,
        "base_timing_certificate_compatible": True,
        "base_requires_exact_candidate_fallback": scaled,
    }


def paired_candidate_peak_certificate(
    base_input_pcm16_codes: np.ndarray,
    candidate_input_pcm16_codes: np.ndarray,
    base_highpass_timing: dict[str, Any],
    impulse_certificate: dict[str, Any],
) -> dict[str, Any]:
    base_codes = _require_pcm16_codes(
        base_input_pcm16_codes,
        "base input",
    )
    candidate_codes = _require_pcm16_codes(
        candidate_input_pcm16_codes,
        "candidate input",
    )
    base = base_codes.astype(np.float64) / PCM16_SCALE
    candidate = candidate_codes.astype(np.float64) / PCM16_SCALE
    if base.shape != candidate.shape or base.size == 0:
        raise ValueError("paired peak certificate waveform shape drift")
    fft_length = int(impulse_certificate["fft_length"])
    if power_of_two_fft_length(base.size) != fft_length:
        raise ValueError("paired peak certificate FFT length drift")
    if impulse_certificate.get("response_cache_waveform_dependent") is not False:
        raise ValueError("waveform-dependent Stop-Hann certificate cache is forbidden")
    if impulse_certificate.get("response_read_only") is not True:
        raise ValueError("Stop-Hann certificate response is not read-only")
    if impulse_certificate.get("impulse_l1_bound_terms_recorded") is not True:
        raise ValueError("Stop-Hann certificate does not record all bound terms")
    impulse_terms = (
        float(impulse_certificate["impulse_l1_observed"]),
        float(impulse_certificate["impulse_l1_irfft_construction_epsilon"]),
        float(impulse_certificate["impulse_l1_summation_epsilon"]),
        float(impulse_certificate["impulse_l1_upper_bound"]),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in impulse_terms):
        raise ValueError("Stop-Hann certificate contains invalid bound terms")
    if impulse_terms[-1] <= sum(impulse_terms[:-1]):
        raise ValueError("Stop-Hann certificate upper bound omits a bound term")

    difference_max_abs = float(np.max(np.abs(candidate - base)))
    signal_scale = max(
        float(np.max(np.abs(base))),
        float(np.max(np.abs(candidate))),
        1.0,
    )
    fft_roundoff_per_transform_epsilon = (
        NUMERICAL_SAFETY_FACTOR
        * FLOAT64_EPSILON
        * fft_length
        * max(math.log2(fft_length), 1.0)
        * signal_scale
    )
    # The paired inequality compares two separately executed Stop-Hann FFT
    # filters.  Their numerical errors can have opposite signs, so the bound
    # explicitly sums one conservative term for base and one for candidate.
    fft_roundoff_epsilon = (
        FFT_ROUNDOFF_TRANSFORM_COUNT * fft_roundoff_per_transform_epsilon
    )
    filtered_difference_upper_bound = float(
        np.nextafter(
            float(impulse_certificate["impulse_l1_upper_bound"])
            * difference_max_abs
            + fft_roundoff_epsilon,
            math.inf,
        )
    )
    interpolation_difference_upper_bound = float(
        np.nextafter(
            SINC70_ABSOLUTE_WEIGHT_BOUND
            * filtered_difference_upper_bound,
            math.inf,
        )
    )
    base_certificate = base_sinc70_peak_upper_bound(base_highpass_timing)
    candidate_peak_upper_bound = float(
        np.nextafter(
            float(base_certificate["base_peak_upper_bound"])
            + interpolation_difference_upper_bound,
            math.inf,
        )
    )
    scaled_base_fallback = bool(
        base_certificate["base_requires_exact_candidate_fallback"]
    )
    may_skip = (
        not scaled_base_fallback
        and candidate_peak_upper_bound < PEAK_SCALE_TRIGGER
    )
    return {
        **base_certificate,
        **impulse_certificate,
        "paired_input_sample_count": int(base.size),
        "paired_input_contract": "exact_worker_pcm16_roundtrip_int16_codes",
        "base_pcm16_codes_sha256": _pcm16_codes_sha256(base_codes),
        "candidate_pcm16_codes_sha256": _pcm16_codes_sha256(candidate_codes),
        "paired_pcm16_difference_max_abs": difference_max_abs,
        "fft_roundoff_per_transform_epsilon": (
            fft_roundoff_per_transform_epsilon
        ),
        "fft_roundoff_transform_count": FFT_ROUNDOFF_TRANSFORM_COUNT,
        "fft_roundoff_rationale": (
            "sum of independent base and candidate Stop-Hann FFT bounds"
        ),
        "fft_roundoff_epsilon": fft_roundoff_epsilon,
        "filtered_difference_upper_bound": filtered_difference_upper_bound,
        "sinc70_interpolation_difference_upper_bound": (
            interpolation_difference_upper_bound
        ),
        "candidate_sinc70_peak_upper_bound": candidate_peak_upper_bound,
        "peak_scale_trigger": PEAK_SCALE_TRIGGER,
        "candidate_sinc70_search_may_be_skipped": may_skip,
        "failure_mode": (
            "certified_skip"
            if may_skip
            else (
                "fallback_exact_praat_sinc70_scaled_base"
                if scaled_base_fallback
                else "fallback_exact_praat_sinc70_bound_not_below_trigger"
            )
        ),
    }
