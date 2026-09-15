"""Central configuration for model architecture, signal format, and training."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalConfig:
    sampling_rate_hz: float = 500.0
    duration_seconds: float = 10.0

    @property
    def num_samples(self) -> int:
        return int(round(self.sampling_rate_hz * self.duration_seconds))


@dataclass
class ModelConfig:
    embed_dim: int = 128
    base_channels: int = 32
    lead_attention_heads: int = 4
    lead_attention_layers: int = 2
    temporal_heads: int = 4
    temporal_layers: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    finding_threshold: float = 0.5


DEFAULT_SIGNAL_CONFIG = SignalConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
