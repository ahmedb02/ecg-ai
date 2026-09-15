"""Loader for the PTB-XL 12-lead ECG dataset (real data, optional dependency).

PTB-XL (https://physionet.org/content/ptb-xl/) is the standard large public
12-lead ECG dataset and is mirrored as a Kaggle Dataset, which makes it a
convenient real-data source for running this repo's training scripts on
Kaggle (see `kaggle/train_ptbxl_on_kaggle.py`). This module is not imported
by the rest of `ecg_ai` by default -- it needs `pandas` and `wfdb`, which
are not in the base `requirements.txt` (see `requirements-kaggle.txt`).

Expected directory layout (the standard PTB-XL release layout, unchanged
by the common Kaggle mirrors):

    <root>/ptbxl_database.csv
    <root>/scp_statements.csv
    <root>/records100/...        (100 Hz waveforms)
    <root>/records500/...        (500 Hz waveforms)

The `scp_codes` -> finding mapping below (SCP_CODE_TO_FINDING) is a
best-effort clinical mapping from PTB-XL's SCP-ECG diagnostic statement
codes onto this repo's finding taxonomy (ecg_ai/clinical/labels.py). It is
not exhaustive and should be reviewed/extended by someone with ECG coding
expertise before being relied on -- PTB-XL's codes are finer-grained than,
and don't perfectly align with, this repo's regions and finding set.
"""
from __future__ import annotations

import ast
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_ai.clinical.labels import FINDINGS
from ecg_ai.config import DEFAULT_SIGNAL_CONFIG
from ecg_ai.data.dataset import labels_to_vectors
from ecg_ai.data.preprocessing import preprocess_ecg
from ecg_ai.models.temporal import EVOLUTION_LABELS

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ecg_ai.data.ptbxl requires pandas and wfdb. "
        "pip install -r requirements-kaggle.txt"
    ) from e

try:
    import wfdb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ecg_ai.data.ptbxl requires the 'wfdb' package to read PTB-XL "
        "waveform records. pip install -r requirements-kaggle.txt"
    ) from e


# SCP-ECG diagnostic code -> (finding_key, region | None). Region-specific
# codes are mapped onto the nearest region in ecg_ai.clinical.labels.REGIONS;
# several PTB-XL location codes (e.g. "inferolateral") don't map cleanly
# onto a single region and are approximated -- see the class docstring.
SCP_CODE_TO_FINDING: dict[str, tuple[str, str | None]] = {
    # Normal
    "NORM": ("nsr", None),
    # Rhythm
    "AFIB": ("afib", None),
    "AFLT": ("aflutter", None),
    "SVTAC": ("svt", None),
    "PSVT": ("svt", None),
    "STACH": ("sinus_tach", None),
    "SBRAD": ("sinus_brady", None),
    "PVC": ("pvc", None),
    "PAC": ("pac", None),
    "BIGU": ("pvc", None),
    "TRIGU": ("pvc", None),
    # Conduction
    "CLBBB": ("lbbb", None),
    "ILBBB": ("lbbb", None),
    "CRBBB": ("rbbb", None),
    "IRBBB": ("rbbb", None),
    "LAFB": ("lafb", None),
    "LPFB": ("lpfb", None),
    "1AVB": ("avb1", None),
    "2AVB": ("avb2_ii", None),
    "3AVB": ("avb3", None),
    "WPW": ("wpw", None),
    # Hypertrophy / enlargement
    "LVH": ("lvh", None),
    "RVH": ("rvh", None),
    "LAO/LAE": ("lae", None),
    "RAO/RAE": ("rae", None),
    # Myocardial infarction, by approximate region (PTB-XL codes are
    # generally chronic/unspecified-acuity -> mapped to old_mi)
    "IMI": ("old_mi", "inferior"),
    "AMI": ("old_mi", "anterior"),
    "ASMI": ("old_mi", "anteroseptal"),
    "ALMI": ("old_mi", "lateral"),
    "ILMI": ("old_mi", "inferior"),
    "IPLMI": ("old_mi", "inferior"),
    "LMI": ("old_mi", "lateral"),
    "IPMI": ("posterior_mi", None),
    "PMI": ("posterior_mi", None),
    # Ischemic ST-T changes / injury patterns, by approximate region
    "ISCAL": ("nstemi_ischemia", "lateral"),
    "ISCAN": ("nstemi_ischemia", "anterior"),
    "ISCAS": ("nstemi_ischemia", "anteroseptal"),
    "ISCIL": ("nstemi_ischemia", "inferior"),
    "ISCIN": ("nstemi_ischemia", "inferior"),
    "ISCLA": ("nstemi_ischemia", "lateral"),
    "INJAS": ("nstemi_ischemia", "anteroseptal"),
    "INJAL": ("nstemi_ischemia", "lateral"),
    "INJIL": ("nstemi_ischemia", "inferior"),
    "INJIN": ("nstemi_ischemia", "inferior"),
    "INJLA": ("nstemi_ischemia", "lateral"),
    # Other
    "LNGQT": ("long_qt", None),
    "PACE": ("paced", None),
    "LVOLT": ("low_voltage", None),
}

MI_KEYS = {f.key for f in FINDINGS if f.category == "ischemia"}
CONDUCTION_KEYS = {f.key for f in FINDINGS if f.category == "conduction"}
ARRHYTHMIA_KEYS = {f.key for f in FINDINGS if f.category == "rhythm" and f.key != "nsr"}


def find_ptbxl_root(search_dirs: list[str] | None = None) -> str:
    """Best-effort auto-detection of a PTB-XL root directory, for Kaggle
    input mounts where the exact dataset slug/subfolder name varies.
    """
    search_dirs = search_dirs or ["/kaggle/input", "."]
    for base in search_dirs:
        matches = glob.glob(os.path.join(base, "**", "ptbxl_database.csv"), recursive=True)
        if matches:
            return os.path.dirname(matches[0])
    raise FileNotFoundError(
        f"Could not find ptbxl_database.csv under {search_dirs}. "
        "Attach the PTB-XL Kaggle dataset, or pass `root=` explicitly."
    )


def load_ptbxl_metadata(root: str) -> "pd.DataFrame":
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def _record_path(root: str, row, use_high_res: bool) -> str:
    rel = row["filename_hr"] if use_high_res else row["filename_lr"]
    return os.path.join(root, rel)


def _scp_codes_to_findings(scp_codes: dict, likelihood_threshold: float = 0.0):
    global_findings: list[str] = []
    regional_findings: list[tuple[str, str]] = []
    for code, likelihood in scp_codes.items():
        if code not in SCP_CODE_TO_FINDING:
            continue
        if likelihood is not None and likelihood < likelihood_threshold:
            continue
        finding_key, region = SCP_CODE_TO_FINDING[code]
        if region is None:
            global_findings.append(finding_key)
        else:
            regional_findings.append((finding_key, region))
    return global_findings, regional_findings


class PTBXLSpatialDataset(Dataset):
    """Single-ECG dataset for training SpatialECGClassifier on PTB-XL.

    `folds`: PTB-XL's official 10-fold `strat_fold` column values to
    include -- the standard convention is folds 1-8 train, 9 validation,
    10 test.
    """

    def __init__(
        self, root: str | None = None, folds: list[int] | None = None,
        use_high_res: bool = True, likelihood_threshold: float = 0.0,
        df: "pd.DataFrame | None" = None,
    ):
        self.root = root or find_ptbxl_root()
        self.use_high_res = use_high_res
        self.likelihood_threshold = likelihood_threshold
        self.df = df if df is not None else load_ptbxl_metadata(self.root)
        if folds is not None:
            self.df = self.df[self.df["strat_fold"].isin(folds)]
        self.records = self.df.reset_index()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        path = _record_path(self.root, row, self.use_high_res)
        signal, meta = wfdb.rdsamp(path)  # signal: (T, 12)
        ecg = signal.T.astype(np.float32)  # (12, T)
        fs = meta["fs"]
        processed = preprocess_ecg(
            ecg, orig_fs=fs,
            target_fs=DEFAULT_SIGNAL_CONFIG.sampling_rate_hz,
            target_seconds=DEFAULT_SIGNAL_CONFIG.duration_seconds,
        )
        global_findings, regional_findings = _scp_codes_to_findings(
            row["scp_codes"], self.likelihood_threshold,
        )
        global_vec, region_vec = labels_to_vectors(global_findings, regional_findings)
        return torch.from_numpy(processed), torch.from_numpy(global_vec), torch.from_numpy(region_vec)


CHRONIC_MI_KEYS = {"old_mi", "posterior_mi"}


def infer_evolution_label(baseline_keys: set[str], latest_keys: set[str]) -> str:
    """Heuristic mapping from a (baseline, latest) finding-key set pair to
    one of TemporalECGModel's EVOLUTION_LABELS. Used to derive weak-label
    supervision for serial PTB-XL sequences, since PTB-XL has no explicit
    "evolution" annotation. This is a coarse approximation for
    demonstration/plumbing purposes, not a clinically validated labeling
    scheme -- see the module docstring.

    Checked in priority order (first match wins) so that, e.g., an
    ischemia-key that both appears and disappears across the same
    transition (STEMI resolving into a documented old MI code) is read as
    the single most clinically relevant event rather than two unrelated
    "new" and "resolved" MI-category findings.
    """
    new = latest_keys - baseline_keys
    resolved = baseline_keys - latest_keys

    if "stemi" in new:
        return "worsening_ischemia" if "nstemi_ischemia" in baseline_keys else "new_or_evolving_stemi"
    if "nstemi_ischemia" in new:
        return "worsening_ischemia"
    if new & CHRONIC_MI_KEYS:
        # A documented old/chronic MI code appearing where none was present
        # before is read as the (possibly unwitnessed) endpoint of an
        # infarct evolution, whether or not an intermediate STEMI/ischemia
        # code was also captured in this pair of studies.
        return "interval_infarct_evolution"
    if "nstemi_ischemia" in resolved and not (latest_keys & MI_KEYS):
        return "resolving_ischemia"
    if "stemi" in resolved and not (latest_keys & MI_KEYS):
        return "resolving_ischemia"
    if new & CONDUCTION_KEYS:
        return "new_conduction_abnormality"
    if resolved & CONDUCTION_KEYS:
        return "resolved_conduction_abnormality"
    if new & ARRHYTHMIA_KEYS:
        return "new_arrhythmia"
    if resolved & ARRHYTHMIA_KEYS:
        return "resolved_arrhythmia"
    return "no_significant_change"


@dataclass
class _SerialSequence:
    ecg_ids: list[int]
    paths: list[str]
    fs: list[float]
    hours: list[float]
    label: str


class PTBXLSerialDataset(Dataset):
    """Serial-ECG dataset for training TemporalECGModel on PTB-XL, built by
    grouping records with the same `patient_id` and ordering by
    `recording_date`. Only patients with >= 2 records are included.

    Evolution labels are derived heuristically via `infer_evolution_label`
    (baseline = earliest record in the sequence, latest = last) -- PTB-XL
    was not collected with paired serial evolution in mind, so treat models
    trained on this as a plumbing/fine-tuning demonstration, not a
    substitute for a dataset purpose-built for serial ECG comparison.
    """

    def __init__(
        self, root: str | None = None, folds: list[int] | None = None,
        use_high_res: bool = True, likelihood_threshold: float = 0.0,
        max_studies: int = 4, df: "pd.DataFrame | None" = None,
    ):
        self.root = root or find_ptbxl_root()
        self.use_high_res = use_high_res
        self.max_studies = max_studies
        full_df = df if df is not None else load_ptbxl_metadata(self.root)
        if folds is not None:
            full_df = full_df[full_df["strat_fold"].isin(folds)]
        full_df = full_df.reset_index()
        full_df["recording_date"] = pd.to_datetime(full_df["recording_date"], errors="coerce")

        self.sequences: list[_SerialSequence] = []
        for patient_id, group in full_df.groupby("patient_id"):
            group = group.sort_values("recording_date")
            if len(group) < 2:
                continue
            group = group.tail(max_studies)
            t0 = group.iloc[0]["recording_date"]
            hours = [
                (t - t0).total_seconds() / 3600.0 if pd.notnull(t) and pd.notnull(t0) else float(i)
                for i, t in enumerate(group["recording_date"])
            ]
            baseline_global, baseline_regional = _scp_codes_to_findings(
                group.iloc[0]["scp_codes"], likelihood_threshold,
            )
            latest_global, latest_regional = _scp_codes_to_findings(
                group.iloc[-1]["scp_codes"], likelihood_threshold,
            )
            baseline_keys = set(baseline_global) | {k for k, _ in baseline_regional}
            latest_keys = set(latest_global) | {k for k, _ in latest_regional}
            label = infer_evolution_label(baseline_keys, latest_keys)

            self.sequences.append(_SerialSequence(
                ecg_ids=group["ecg_id"].tolist(),
                paths=[_record_path(self.root, row, use_high_res) for _, row in group.iterrows()],
                fs=[500.0 if use_high_res else 100.0] * len(group),
                hours=hours,
                label=label,
            ))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        ecgs = []
        for path in seq.paths:
            signal, meta = wfdb.rdsamp(path)
            ecg = signal.T.astype(np.float32)
            processed = preprocess_ecg(
                ecg, orig_fs=meta["fs"],
                target_fs=DEFAULT_SIGNAL_CONFIG.sampling_rate_hz,
                target_seconds=DEFAULT_SIGNAL_CONFIG.duration_seconds,
            )
            ecgs.append(processed)
        stacked = torch.from_numpy(np.stack(ecgs, axis=0))
        time_deltas = torch.tensor(seq.hours, dtype=torch.float32)
        label = torch.tensor(EVOLUTION_LABELS.index(seq.label), dtype=torch.long)
        return stacked, time_deltas, label
