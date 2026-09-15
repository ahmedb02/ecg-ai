"""Synthetic 12-lead ECG generator.

There is no bundled real ECG dataset in this repository (see README for how
to plug in PTB-XL / MIMIC-IV-ECG / a private dataset). This module produces
*structurally plausible but not medically validated* synthetic waveforms so
that the model architecture, training loop, and end-to-end inference
pipeline (including the temporal/serial comparison) can be exercised and
unit-tested without real data.

Do not train a model intended for any real use on synthetic data alone --
it is a shape/plumbing generator, not a substitute for a labeled clinical
ECG dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ecg_ai.clinical.labels import LEAD_NAMES, REGION_LEADS

# Approximate, directionally-correct (not mV-calibrated) per-lead morphology
# scale factors: (P, Q, R, S, T).
LEAD_MORPHOLOGY: dict[str, tuple[float, float, float, float, float]] = {
    "I":   (0.10, -0.05, 0.80, -0.10, 0.25),
    "II":  (0.15, -0.05, 1.00, -0.10, 0.30),
    "III": (0.05, -0.05, 0.50, -0.10, 0.10),
    "aVR": (-0.10, 0.02, -0.60, 0.05, -0.20),
    "aVL": (0.05, -0.03, 0.40, -0.05, 0.10),
    "aVF": (0.10, -0.05, 0.60, -0.10, 0.20),
    "V1":  (0.05, 0.00, 0.20, -0.60, -0.10),
    "V2":  (0.08, 0.00, 0.40, -0.80, 0.30),
    "V3":  (0.10, -0.02, 0.90, -0.50, 0.40),
    "V4":  (0.10, -0.05, 1.20, -0.20, 0.40),
    "V5":  (0.12, -0.05, 1.10, -0.10, 0.35),
    "V6":  (0.10, -0.05, 0.90, -0.05, 0.30),
}

# (center_ms relative to R peak, width_ms) for each wave component.
WAVE_TIMING = {
    "P": (-160.0, 40.0),
    "Q": (-20.0, 8.0),
    "R": (0.0, 12.0),
    "S": (20.0, 10.0),
    "T": (250.0, 80.0),
}


@dataclass
class SyntheticECGParams:
    heart_rate: float = 70.0
    rr_irregularity: float = 0.0  # fraction of RR interval, std of jitter
    p_wave_present: bool = True
    qrs_width_mult: float = 1.0
    t_width_mult: float = 1.0
    t_center_shift_ms: float = 0.0  # QT prolongation when positive
    st_offset_mv: dict[str, float] = field(default_factory=dict)  # + = elevation
    t_invert_leads: set[str] = field(default_factory=set)
    t_peak_mult: float = 1.0  # tall + narrow T (hyperkalemia)
    q_depth_mult: dict[str, float] = field(default_factory=dict)  # >1 = deeper Q (old MI)
    lead_r_mult: dict[str, float] = field(default_factory=dict)
    lead_s_mult: dict[str, float] = field(default_factory=dict)
    pr_depression_mv: float = 0.0  # diffuse, pericarditis
    voltage_mult: float = 1.0
    paced: bool = False
    noise_std: float = 0.02


def _gaussian(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(t)
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_synthetic_ecg(
    params: SyntheticECGParams, fs: float = 500.0, duration_s: float = 10.0,
    seed: int | None = None,
) -> np.ndarray:
    """Returns a (12, T) float32 array."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t_ms = (np.arange(n) / fs) * 1000.0

    # Build beat onset times (R-peak positions) honoring rate + irregularity.
    mean_rr_ms = 60_000.0 / params.heart_rate
    r_times = []
    cursor = mean_rr_ms * 0.5
    while cursor < duration_s * 1000.0:
        r_times.append(cursor)
        jitter = 1.0 + rng.normal(0, params.rr_irregularity) if params.rr_irregularity > 0 else 1.0
        cursor += mean_rr_ms * max(0.3, jitter)
    r_times = np.array(r_times)

    ecg = np.zeros((len(LEAD_NAMES), n), dtype=np.float64)

    for lead_idx, lead in enumerate(LEAD_NAMES):
        p_s, q_s, r_s, s_s, t_s = LEAD_MORPHOLOGY[lead]
        r_s *= params.lead_r_mult.get(lead, 1.0)
        s_s *= params.lead_s_mult.get(lead, 1.0)
        q_s *= params.q_depth_mult.get(lead, 1.0)
        if lead in params.t_invert_leads:
            t_s *= -1.0
        t_s *= params.t_peak_mult

        signal = np.zeros(n, dtype=np.float64)
        for rp in r_times:
            if params.paced:
                # Narrow pacing spike ~40ms before the QRS complex.
                signal += _gaussian(t_ms, rp - 40.0, 2.0, 1.5 * r_s / max(abs(r_s), 1e-3))

            if params.p_wave_present:
                pc, pw = WAVE_TIMING["P"]
                pc_eff = rp + pc - (params.pr_depression_mv * 0)  # timing unaffected
                signal += _gaussian(t_ms, pc_eff, pw, p_s)
                if params.pr_depression_mv:
                    # Depress the PR segment between P offset and QRS onset.
                    seg_center = rp + (pc + WAVE_TIMING["Q"][0]) / 2
                    seg_width = abs(WAVE_TIMING["Q"][0] - pc) / 2
                    signal += _gaussian(t_ms, seg_center, max(seg_width, 5.0), -params.pr_depression_mv)

            qc, qw = WAVE_TIMING["Q"]
            qw *= params.qrs_width_mult
            signal += _gaussian(t_ms, rp + qc, qw, q_s)

            rc, rw = WAVE_TIMING["R"]
            rw *= params.qrs_width_mult
            signal += _gaussian(t_ms, rp + rc, rw, r_s)

            sc, sw = WAVE_TIMING["S"]
            sw *= params.qrs_width_mult
            signal += _gaussian(t_ms, rp + sc, sw, s_s)

            tc, tw = WAVE_TIMING["T"]
            tc = tc + params.t_center_shift_ms
            tw *= params.t_width_mult
            signal += _gaussian(t_ms, rp + tc, tw, t_s)

            st_off = params.st_offset_mv.get(lead, 0.0)
            if st_off:
                # Plateau spanning the ST segment (end of S to start of T).
                seg_center = rp + (sc + tc) / 2
                seg_width = max(abs(tc - sc) / 2, 10.0)
                signal += _gaussian(t_ms, seg_center, seg_width, st_off)

        ecg[lead_idx] = signal * params.voltage_mult

    ecg += rng.normal(0, params.noise_std, size=ecg.shape)
    return ecg.astype(np.float32)


# --- Named clinical scenarios ---------------------------------------------
# Each entry returns (params, global_finding_keys, regional_findings) where
# regional_findings is a list of (finding_key, region) pairs. Used both by
# the demo/test suite and by `training/train_spatial.py`'s synthetic mode.

def make_case(
    condition: str, region: str | None = None, seed: int | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    p = SyntheticECGParams()
    global_findings: list[str] = []
    regional_findings: list[tuple[str, str]] = []

    if condition == "normal":
        global_findings = ["nsr"]

    elif condition == "sinus_tach":
        p.heart_rate = 130
        global_findings = ["sinus_tach"]

    elif condition == "sinus_brady":
        p.heart_rate = 45
        global_findings = ["sinus_brady"]

    elif condition == "afib":
        p.heart_rate = 110
        p.rr_irregularity = 0.25
        p.p_wave_present = False
        global_findings = ["afib"]

    elif condition == "lbbb":
        p.qrs_width_mult = 2.2
        p.t_invert_leads = {"V5", "V6", "I", "aVL"}
        global_findings = ["lbbb"]

    elif condition == "rbbb":
        p.qrs_width_mult = 2.0
        p.lead_s_mult = {"I": 2.0, "V6": 2.0}
        p.lead_r_mult = {"V1": 1.8}
        global_findings = ["rbbb"]

    elif condition == "lvh":
        p.lead_r_mult = {"V5": 1.9, "V6": 1.7, "I": 1.4}
        p.lead_s_mult = {"V1": 1.8, "V2": 1.6}
        p.t_invert_leads = {"V5", "V6"}
        global_findings = ["lvh"]

    elif condition == "pericarditis":
        p.st_offset_mv = {l: 0.12 for l in LEAD_NAMES if l != "aVR"}
        p.pr_depression_mv = 0.08
        global_findings = ["pericarditis"]

    elif condition == "long_qt":
        p.t_center_shift_ms = 120.0
        global_findings = ["long_qt"]

    elif condition == "hyperkalemia":
        p.t_peak_mult = 2.2
        p.t_width_mult = 0.55
        p.p_wave_present = False
        global_findings = ["hyperkalemia"]

    elif condition == "low_voltage":
        p.voltage_mult = 0.25
        global_findings = ["low_voltage"]

    elif condition == "paced":
        p.paced = True
        global_findings = ["paced"]

    elif condition in ("stemi", "nstemi_ischemia", "old_mi"):
        assert region is not None and region in REGION_LEADS, "regionalized case needs a valid region"
        leads = REGION_LEADS[region]
        if condition == "stemi":
            p.st_offset_mv = {l: 0.25 for l in leads}
        elif condition == "nstemi_ischemia":
            p.st_offset_mv = {l: -0.15 for l in leads}
            p.t_invert_leads = set(leads)
        elif condition == "old_mi":
            p.q_depth_mult = {l: 4.0 for l in leads}
        regional_findings = [(condition, region)]

    else:
        raise ValueError(f"unknown synthetic condition: {condition}")

    ecg = generate_synthetic_ecg(p, seed=seed)
    return ecg, global_findings, regional_findings


ALL_NON_REGIONAL_CONDITIONS = [
    "normal", "sinus_tach", "sinus_brady", "afib", "lbbb", "rbbb", "lvh",
    "pericarditis", "long_qt", "hyperkalemia", "low_voltage", "paced",
]
ALL_REGIONAL_CONDITIONS = ["stemi", "nstemi_ischemia", "old_mi"]
