"""Minimal runnable demo of the full pipeline using untrained models and
synthetic ECGs -- shows the plumbing (shapes, report structure) end to end.
Model weights are random here, so the *findings themselves* are meaningless;
train first (see README) for a model whose output is worth reading.

Run: python -m examples.demo
"""
from __future__ import annotations

import torch

from ecg_ai.data.synthetic import make_case
from ecg_ai.inference.predict import ECGInterpreter
from ecg_ai.models.spatial_cnn import SpatialECGClassifier
from ecg_ai.models.temporal import TemporalECGModel


def main():
    torch.manual_seed(0)

    spatial_classifier = SpatialECGClassifier(embed_dim=64, base_channels=16)
    temporal_model = TemporalECGModel(
        embed_dim=64, base_channels=16, num_heads=4, num_layers=2,
        spatial_encoder=spatial_classifier.encoder,  # share weights
    )
    interpreter = ECGInterpreter(spatial_classifier, temporal_model)

    print("=" * 70)
    print("SINGLE ECG: spatial interpretation + differential diagnosis")
    print("=" * 70)
    ecg, _, _ = make_case("stemi", region="inferior", seed=1)
    report = interpreter.interpret(ecg, fs=500)
    print(report.to_text())

    print()
    print("=" * 70)
    print("SERIAL ECGs: temporal evolution across studies")
    print("=" * 70)
    ecg_t0, _, _ = make_case("nstemi_ischemia", region="anterior", seed=2)
    ecg_t1, _, _ = make_case("stemi", region="anterior", seed=3)
    ecg_t2, _, _ = make_case("old_mi", region="anterior", seed=4)
    studies = [
        (ecg_t0, 500.0, 0.0),   # baseline (ED arrival)
        (ecg_t1, 500.0, 2.0),   # 2 hours later
        (ecg_t2, 500.0, 72.0),  # 3 days later
    ]
    latest_report, evolution_report = interpreter.interpret_serial(studies)
    print("-- Latest study interpretation --")
    print(latest_report.to_text())
    print()
    print("-- Evolution across studies --")
    print(evolution_report.to_text())


if __name__ == "__main__":
    main()
