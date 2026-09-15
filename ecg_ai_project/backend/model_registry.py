"""Loads the spatial/temporal models once at process startup.

Checkpoint paths come from env vars (SPATIAL_CHECKPOINT / TEMPORAL_CHECKPOINT
-- see ecg_ai/training/train_spatial.py and train_temporal.py, or
kaggle/train_ptbxl_on_kaggle.py, for how to produce them). If unset, the
app still starts with randomly-initialized weights so the whole
image-to-report pipeline is runnable/demoable end to end -- every response
is flagged accordingly (see `ModelStatus.trained`), on top of the
disclaimer every ECGReport/EvolutionReport already carries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ecg_ai.config import DEFAULT_MODEL_CONFIG
from ecg_ai.inference.predict import ECGInterpreter
from ecg_ai.models.spatial_cnn import SpatialECGClassifier
from ecg_ai.models.temporal import TemporalECGModel


@dataclass
class ModelStatus:
    spatial_trained: bool
    temporal_trained: bool
    temporal_available: bool


def build_interpreter() -> tuple[ECGInterpreter, ModelStatus]:
    spatial_checkpoint = os.environ.get("SPATIAL_CHECKPOINT")
    temporal_checkpoint = os.environ.get("TEMPORAL_CHECKPOINT")
    model_cfg = DEFAULT_MODEL_CONFIG

    spatial_classifier = SpatialECGClassifier(
        embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
    )
    spatial_trained = False
    if spatial_checkpoint and os.path.exists(spatial_checkpoint):
        import torch
        spatial_classifier.load_state_dict(torch.load(spatial_checkpoint, map_location="cpu"))
        spatial_trained = True

    temporal_model = TemporalECGModel(
        embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
        num_heads=model_cfg.temporal_heads, num_layers=model_cfg.temporal_layers,
    )
    temporal_trained = False
    if temporal_checkpoint and os.path.exists(temporal_checkpoint):
        import torch
        temporal_model.load_state_dict(torch.load(temporal_checkpoint, map_location="cpu"))
        temporal_trained = True

    interpreter = ECGInterpreter(spatial_classifier, temporal_model)
    status = ModelStatus(
        spatial_trained=spatial_trained,
        temporal_trained=temporal_trained,
        temporal_available=True,
    )
    return interpreter, status
