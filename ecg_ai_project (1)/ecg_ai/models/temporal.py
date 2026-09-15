"""Temporal model over serial ECGs.

Takes a sequence of ECGs from the same patient (e.g. ED arrival, 1hr,
3hr, next-day) plus the elapsed time between studies, and:

  1. Encodes each ECG independently with the *shared-weight* spatial
     encoder from `spatial_cnn.py` (so the same lead/region morphology
     detectors are reused, not re-learned).
  2. Feeds the resulting sequence of per-study embeddings through a
     Transformer with a continuous time encoding (irregular sampling --
     serial ECGs are not evenly spaced), producing a context-aware
     embedding per time step.
  3. Predicts, at the *latest* study, an evolution label describing how
     findings changed relative to the earlier studies (e.g. new/evolving
     STEMI, resolving ischemia, new conduction block, no significant change).
  4. Computes an interpretable per-lead / per-region "delta" signal
     (difference between consecutive study embeddings) that downstream
     rule logic in `clinical/evolution.py` turns into a plain-language
     evolution summary alongside the learned label.
"""
from __future__ import annotations

import torch
from torch import nn

from ecg_ai.clinical.labels import NUM_REGIONS
from ecg_ai.models.layers import SinusoidalTimeEncoding
from ecg_ai.models.spatial_cnn import SpatialECGEncoder

# Evolution categories predicted at the latest time step, relative to the
# prior study/studies in the sequence.
EVOLUTION_LABELS = [
    "no_significant_change",
    "new_or_evolving_stemi",
    "resolving_ischemia",
    "worsening_ischemia",
    "new_conduction_abnormality",
    "resolved_conduction_abnormality",
    "new_arrhythmia",
    "resolved_arrhythmia",
    "interval_infarct_evolution",  # e.g. hyperacute T -> ST elevation -> Q waves
]
NUM_EVOLUTION_LABELS = len(EVOLUTION_LABELS)


class TemporalECGModel(nn.Module):
    """Serial-ECG model: shared spatial encoder + temporal transformer."""

    def __init__(self, embed_dim: int = 128, base_channels: int = 32,
                 num_heads: int = 4, num_layers: int = 3,
                 spatial_encoder: SpatialECGEncoder | None = None):
        super().__init__()
        # Reuse a pretrained spatial encoder if provided, otherwise train
        # both stages jointly from scratch.
        self.spatial_encoder = spatial_encoder or SpatialECGEncoder(embed_dim, base_channels)
        self.embed_dim = embed_dim

        self.time_encoding = SinusoidalTimeEncoding(embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.evolution_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, NUM_EVOLUTION_LABELS),
        )

    def encode_sequence(self, ecgs: torch.Tensor, time_deltas: torch.Tensor,
                         study_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """ecgs: (B, S, num_leads, T); time_deltas: (B, S) hours since first
        study; study_mask: (B, S) bool, True for real (non-padding) studies.
        """
        b, s, leads, t = ecgs.shape
        flat = ecgs.reshape(b * s, leads, t)
        enc = self.spatial_encoder(flat)
        global_embed = enc["global_embed"].reshape(b, s, -1)
        region_embeds = enc["region_embeds"].reshape(b, s, NUM_REGIONS, -1)

        seq = global_embed + self.time_encoding(time_deltas)
        pad_mask = ~study_mask  # transformer expects True == ignore
        temporal_out = self.temporal_encoder(seq, src_key_padding_mask=pad_mask)

        return {
            "per_study_embed": global_embed,
            "per_study_region_embed": region_embeds,
            "temporal_embed": temporal_out,
        }

    def forward(self, ecgs: torch.Tensor, time_deltas: torch.Tensor,
                study_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encode_sequence(ecgs, time_deltas, study_mask)
        b, s, _ = enc["temporal_embed"].shape

        last_idx = study_mask.sum(dim=1).clamp(min=1) - 1  # index of latest real study
        batch_idx = torch.arange(b, device=ecgs.device)

        latest_temporal = enc["temporal_embed"][batch_idx, last_idx]  # context-aware
        latest_raw = enc["per_study_embed"][batch_idx, last_idx]      # study in isolation
        evolution_logits = self.evolution_head(
            torch.cat([latest_temporal, latest_raw], dim=-1)
        )

        # Per-region delta between the latest study and the first study in
        # the (padded) sequence -- a simple, interpretable "what changed"
        # signal used by clinical/evolution.py for the lead/region narrative.
        first_idx = torch.zeros(b, dtype=torch.long, device=ecgs.device)
        latest_region = enc["per_study_region_embed"][batch_idx, last_idx]
        first_region = enc["per_study_region_embed"][batch_idx, first_idx]
        region_delta = (latest_region - first_region).norm(dim=-1)  # (B, R)

        return {
            **enc,
            "evolution_logits": evolution_logits,
            "region_delta": region_delta,
            "last_idx": last_idx,
        }
