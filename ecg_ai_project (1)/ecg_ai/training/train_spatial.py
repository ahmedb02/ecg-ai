"""Trains SpatialECGClassifier on synthetic data (default) or a plugged-in
real dataset with the same tensor shapes.

Usage:
    python -m ecg_ai.training.train_spatial --epochs 5 --out checkpoints/spatial.pt

To train on real data, swap `SyntheticSpatialDataset` below for a Dataset
that yields (ecg[12, T] float tensor, global_vec[NUM_FINDINGS], region_vec
[NUM_REGIONS, num_region_findings]) -- see data/dataset.py::labels_to_vectors
for how to build the label tensors from a list of finding keys.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset

from ecg_ai.config import DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG
from ecg_ai.data.dataset import SyntheticSpatialDataset
from ecg_ai.models.spatial_cnn import SpatialECGClassifier
from ecg_ai.training.losses import spatial_loss


def train(
    epochs: int = DEFAULT_TRAINING_CONFIG.epochs,
    batch_size: int = DEFAULT_TRAINING_CONFIG.batch_size,
    lr: float = DEFAULT_TRAINING_CONFIG.learning_rate,
    train_size: int = 2000,
    val_size: int = 200,
    out_path: str | None = None,
    device: str | None = None,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
    num_workers: int = DEFAULT_TRAINING_CONFIG.num_workers,
    model: SpatialECGClassifier | None = None,
) -> SpatialECGClassifier:
    """Train the spatial classifier.

    Defaults to synthetic data. Pass `train_dataset`/`val_dataset` (any
    Dataset yielding the (ecg, global_vec, region_vec) tuples described in
    the module docstring) to train on real data instead -- e.g. from
    `ecg_ai.data.ptbxl.PTBXLSpatialDataset` -- without duplicating this loop.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = train_dataset if train_dataset is not None else SyntheticSpatialDataset(epoch_size=train_size, seed=0)
    val_ds = val_dataset if val_dataset is not None else SyntheticSpatialDataset(epoch_size=val_size, seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model or SpatialECGClassifier(
        embed_dim=DEFAULT_MODEL_CONFIG.embed_dim,
        base_channels=DEFAULT_MODEL_CONFIG.base_channels,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=DEFAULT_TRAINING_CONFIG.weight_decay,
    )

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for ecg, global_vec, region_vec in train_loader:
            ecg, global_vec, region_vec = ecg.to(device), global_vec.to(device), region_vec.to(device)
            out = model(ecg)
            losses = spatial_loss(out["global_logits"], global_vec, out["region_logits"], region_vec)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"].item() * ecg.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ecg, global_vec, region_vec in val_loader:
                ecg, global_vec, region_vec = ecg.to(device), global_vec.to(device), region_vec.to(device)
                out = model(ecg)
                losses = spatial_loss(out["global_logits"], global_vec, out["region_logits"], region_vec)
                val_loss += losses["loss"].item() * ecg.size(0)
        val_loss /= len(val_ds)

        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({time.time() - t0:.1f}s)")

    if out_path:
        torch.save(model.state_dict(), out_path)
        print(f"saved checkpoint to {out_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        train_size=args.train_size, val_size=args.val_size, out_path=args.out,
    )


if __name__ == "__main__":
    main()
