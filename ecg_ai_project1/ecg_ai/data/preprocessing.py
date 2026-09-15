"""Lightweight ECG signal preprocessing (numpy-only, no scipy dependency).

Real-world ECG ingestion pipelines should do more (proper Butterworth
bandpass filtering, notch filtering for mains interference, lead-off
detection, etc.) -- these are lightweight, dependency-free approximations
suitable for this scaffold and for the synthetic-data demo.
"""
from __future__ import annotations

import numpy as np


def resample_signal(ecg: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample (leads, T) via linear interpolation."""
    if orig_fs == target_fs:
        return ecg
    leads, t = ecg.shape
    duration = t / orig_fs
    orig_times = np.linspace(0, duration, t, endpoint=False)
    n_target = int(round(duration * target_fs))
    target_times = np.linspace(0, duration, n_target, endpoint=False)
    resampled = np.stack([
        np.interp(target_times, orig_times, ecg[lead]) for lead in range(leads)
    ])
    return resampled


def remove_baseline_wander(ecg: np.ndarray, fs: float, window_sec: float = 0.6) -> np.ndarray:
    """Crude high-pass filter: subtract a moving-average baseline estimate."""
    n = ecg.shape[-1]
    window = max(1, int(round(window_sec * fs)))
    # np.convolve(..., mode="same") returns length max(len(signal), len(kernel));
    # a window longer than a short signal (e.g. a tightly-cropped digitized
    # ECG) would silently return baseline longer than ecg[lead], crashing the
    # subtraction below with a broadcast error. Never let the window exceed
    # the signal length; only decrement (not increment) to keep it odd, so
    # the cap is never violated.
    window = min(window, n)
    if window % 2 == 0:
        window = max(1, window - 1)
    kernel = np.ones(window) / window
    out = np.empty_like(ecg)
    for lead in range(ecg.shape[0]):
        baseline = np.convolve(ecg[lead], kernel, mode="same")
        out[lead] = ecg[lead] - baseline
    return out


def zscore_normalize(ecg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-lead zero-mean, unit-variance normalization."""
    mean = ecg.mean(axis=-1, keepdims=True)
    std = ecg.std(axis=-1, keepdims=True)
    return (ecg - mean) / (std + eps)


def pad_or_crop(ecg: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with edge values or center-crop to exactly `target_len` samples."""
    leads, t = ecg.shape
    if t == target_len:
        return ecg
    if t > target_len:
        start = (t - target_len) // 2
        return ecg[:, start:start + target_len]
    pad_total = target_len - t
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(ecg, ((0, 0), (pad_left, pad_right)), mode="edge")


def preprocess_ecg(
    ecg: np.ndarray, orig_fs: float, target_fs: float = 500.0, target_seconds: float = 10.0,
) -> np.ndarray:
    """Full preprocessing pipeline: resample -> de-wander -> pad/crop -> normalize."""
    ecg = resample_signal(ecg, orig_fs, target_fs)
    ecg = remove_baseline_wander(ecg, target_fs)
    ecg = pad_or_crop(ecg, int(round(target_seconds * target_fs)))
    ecg = zscore_normalize(ecg)
    return ecg.astype(np.float32)
