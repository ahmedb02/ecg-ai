"""PyTorch Dataset classes for the spatial classifier and the temporal model.

Includes synthetic on-the-fly datasets (see `synthetic.py`) so the full
training loops are runnable without a real ECG dataset. To train on real
data, implement a dataset that yields the same tensor shapes -- see the
"Using real data" section in the README -- and pass it to the training
scripts instead of `SyntheticSpatialDataset` / `SyntheticTemporalDataset`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_ai.clinical.labels import (
    NUM_FINDINGS, FINDING_INDEX, REGIONS, REGION_INDEX,
    REGIONALIZED_FINDING_KEYS, NUM_REGIONS,
)
from ecg_ai.data.synthetic import (
    make_case, ALL_NON_REGIONAL_CONDITIONS, ALL_REGIONAL_CONDITIONS,
)
from ecg_ai.models.temporal import EVOLUTION_LABELS

REGION_FINDING_INDEX = {k: i for i, k in enumerate(REGIONALIZED_FINDING_KEYS)}


def labels_to_vectors(
    global_findings: list[str], regional_findings: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """global_findings: list of finding keys. regional_findings: list of
    (finding_key, region) pairs. Returns (global_vec[NUM_FINDINGS],
    region_vec[NUM_REGIONS, num_regionalized_findings]).
    """
    global_vec = np.zeros(NUM_FINDINGS, dtype=np.float32)
    for key in global_findings:
        global_vec[FINDING_INDEX[key]] = 1.0

    region_vec = np.zeros((NUM_REGIONS, len(REGIONALIZED_FINDING_KEYS)), dtype=np.float32)
    for key, region in regional_findings:
        region_vec[REGION_INDEX[region], REGION_FINDING_INDEX[key]] = 1.0

    return global_vec, region_vec


class SyntheticSpatialDataset(Dataset):
    """Randomly generated single-study ECGs for training/testing the
    SpatialECGClassifier. Deterministic given `epoch_size` and a base seed,
    so re-running with the same seed reproduces the same dataset.
    """

    def __init__(self, epoch_size: int = 2000, seed: int = 0):
        self.epoch_size = epoch_size
        self.seed = seed
        conditions = ALL_NON_REGIONAL_CONDITIONS + [
            (c, r) for c in ALL_REGIONAL_CONDITIONS for r in
            ["inferior", "lateral", "anteroseptal", "septal", "anterior"]
        ]
        self._conditions = conditions

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        choice = self._conditions[rng.integers(len(self._conditions))]
        if isinstance(choice, tuple):
            condition, region = choice
        else:
            condition, region = choice, None
        ecg, global_findings, regional_findings = make_case(condition, region, seed=int(rng.integers(1 << 30)))
        global_vec, region_vec = labels_to_vectors(global_findings, regional_findings)
        return (
            torch.from_numpy(ecg),
            torch.from_numpy(global_vec),
            torch.from_numpy(region_vec),
        )


# Evolution scenario -> (baseline_condition, baseline_region, latest_condition, latest_region)
EVOLUTION_SCENARIOS: dict[str, tuple[str, str | None, str, str | None]] = {
    "no_significant_change": ("normal", None, "normal", None),
    "new_or_evolving_stemi": ("normal", None, "stemi", "inferior"),
    "resolving_ischemia": ("nstemi_ischemia", "anterior", "normal", None),
    "worsening_ischemia": ("nstemi_ischemia", "anterior", "stemi", "anterior"),
    "new_conduction_abnormality": ("normal", None, "lbbb", None),
    "resolved_conduction_abnormality": ("lbbb", None, "normal", None),
    "new_arrhythmia": ("normal", None, "afib", None),
    "resolved_arrhythmia": ("afib", None, "normal", None),
    "interval_infarct_evolution": ("stemi", "anterior", "old_mi", "anterior"),
}


class SyntheticTemporalDataset(Dataset):
    """Randomly generated serial-ECG sequences (2-4 studies) with an
    evolution label describing the change between the earliest and latest
    study, for training/testing TemporalECGModel.
    """

    def __init__(self, epoch_size: int = 1000, max_studies: int = 4, seed: int = 0):
        self.epoch_size = epoch_size
        self.max_studies = max_studies
        self.seed = seed
        self.scenario_names = list(EVOLUTION_SCENARIOS.keys())

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 7_919 + idx)
        scenario = self.scenario_names[rng.integers(len(self.scenario_names))]
        base_cond, base_region, latest_cond, latest_region = EVOLUTION_SCENARIOS[scenario]

        n_studies = int(rng.integers(2, self.max_studies + 1))
        ecgs = []
        time_deltas = [0.0]
        for i in range(n_studies - 1):
            ecg, _, _ = make_case(base_cond, base_region, seed=int(rng.integers(1 << 30)))
            ecgs.append(ecg)
            if i > 0:
                time_deltas.append(time_deltas[-1] + float(rng.uniform(0.5, 6.0)))
        latest_ecg, _, _ = make_case(latest_cond, latest_region, seed=int(rng.integers(1 << 30)))
        ecgs.append(latest_ecg)
        if n_studies > 1:
            time_deltas.append(time_deltas[-1] + float(rng.uniform(1.0, 48.0)))

        stacked = np.stack(ecgs, axis=0)  # (S, 12, T)
        label = EVOLUTION_LABELS.index(scenario)
        return (
            torch.from_numpy(stacked),
            torch.tensor(time_deltas, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def collate_temporal_batch(batch):
    """Pads variable-length serial-ECG sequences to the batch max length."""
    max_s = max(item[0].shape[0] for item in batch)
    leads, t = batch[0][0].shape[1:]

    ecgs = torch.zeros(len(batch), max_s, leads, t)
    time_deltas = torch.zeros(len(batch), max_s)
    study_mask = torch.zeros(len(batch), max_s, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, (ecg_seq, td, label) in enumerate(batch):
        s = ecg_seq.shape[0]
        ecgs[i, :s] = ecg_seq
        time_deltas[i, :s] = td
        study_mask[i, :s] = True
        labels[i] = label

    return ecgs, time_deltas, study_mask, labels
