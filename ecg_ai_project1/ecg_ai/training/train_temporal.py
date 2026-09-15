"""Trains TemporalECGModel on synthetic serial-ECG sequences (default), or a
plugged-in real dataset with the same tensor shapes (see
`collate_temporal_batch` in data/dataset.py for the expected batch format).

Usage:
    python -m ecg_ai.training.train_temporal \\
        --epochs 5 --spatial-checkpoint checkpoints/spatial.pt --out checkpoints/temporal.pt

If --spatial-checkpoint is given, the spatial encoder is initialized from a
pretrained SpatialECGClassifier (recommended: train_spatial.py first) so the
temporal model only has to learn the sequence-level evolution reasoning on
top of already-good per-study representations. Its weights are fine-tuned
jointly unless --freeze-spatial is passed.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset

from ecg_ai.config import DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG
from ecg_ai.data.dataset import SyntheticTemporalDataset, collate_temporal_batch
from ecg_ai.models.spatial_cnn import SpatialECGClassifier, SpatialECGEncoder
from ecg_ai.models.temporal import TemporalECGModel
from ecg_ai.training.losses import temporal_loss


def _load_spatial_encoder(checkpoint_path: str, model_cfg) -> SpatialECGEncoder:
    classifier = SpatialECGClassifier(
        embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    classifier.load_state_dict(state)
    return classifier.encoder


def train(
    epochs: int = DEFAULT_TRAINING_CONFIG.epochs,
    batch_size: int = DEFAULT_TRAINING_CONFIG.batch_size,
    lr: float = DEFAULT_TRAINING_CONFIG.learning_rate,
    train_size: int = 1500,
    val_size: int = 150,
    spatial_checkpoint: str | None = None,
    freeze_spatial: bool = False,
    out_path: str | None = None,
    device: str | None = None,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
    num_workers: int = DEFAULT_TRAINING_CONFIG.num_workers,
) -> TemporalECGModel:
    """Train the temporal (serial-ECG evolution) model.

    Defaults to synthetic data. Pass `train_dataset`/`val_dataset` (any
    Dataset yielding items compatible with `collate_temporal_batch`) to
    train on real data instead -- e.g. from
    `ecg_ai.data.ptbxl.PTBXLSerialDataset` -- without duplicating this loop.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = DEFAULT_MODEL_CONFIG

    train_ds = train_dataset if train_dataset is not None else SyntheticTemporalDataset(epoch_size=train_size, seed=0)
    val_ds = val_dataset if val_dataset is not None else SyntheticTemporalDataset(epoch_size=val_size, seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal_batch, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_batch, num_workers=num_workers)

    spatial_encoder = None
    if spatial_checkpoint:
        spatial_encoder = _load_spatial_encoder(spatial_checkpoint, model_cfg)
        if freeze_spatial:
            for p in spatial_encoder.parameters():
                p.requires_grad = False

    model = TemporalECGModel(
        embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
        num_heads=model_cfg.temporal_heads, num_layers=model_cfg.temporal_layers,
        spatial_encoder=spatial_encoder,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=DEFAULT_TRAINING_CONFIG.weight_decay)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        for ecgs, time_deltas, study_mask, labels in train_loader:
            ecgs, time_deltas = ecgs.to(device), time_deltas.to(device)
            study_mask, labels = study_mask.to(device), labels.to(device)
            out = model(ecgs, time_deltas, study_mask)
            losses = temporal_loss(out["evolution_logits"], labels)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"].item() * ecgs.size(0)
            correct += (out["evolution_logits"].argmax(dim=-1) == labels).sum().item()
        train_loss = running_loss / len(train_ds)
        train_acc = correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for ecgs, time_deltas, study_mask, labels in val_loader:
                ecgs, time_deltas = ecgs.to(device), time_deltas.to(device)
                study_mask, labels = study_mask.to(device), labels.to(device)
                out = model(ecgs, time_deltas, study_mask)
                losses = temporal_loss(out["evolution_logits"], labels)
                val_loss += losses["loss"].item() * ecgs.size(0)
                val_correct += (out["evolution_logits"].argmax(dim=-1) == labels).sum().item()
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}  ({time.time() - t0:.1f}s)")

    if out_path:
        torch.save(model.state_dict(), out_path)
        print(f"saved checkpoint to {out_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=150)
    parser.add_argument("--spatial-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-spatial", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        train_size=args.train_size, val_size=args.val_size,
        spatial_checkpoint=args.spatial_checkpoint, freeze_spatial=args.freeze_spatial,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
