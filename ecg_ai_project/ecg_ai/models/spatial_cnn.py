"""Spatial CNN over a single 12-lead ECG.

Design: each lead is a 1D time series, but leads are not interchangeable
channels -- different leads "look at" the heart from different anatomical
angles (inferior, lateral, anteroseptal, ...). To exploit this spatial
structure instead of flattening leads into an unordered channel dimension:

  1. A shared per-lead 1D conv stack extracts a temporal feature map for
     every lead independently (weight sharing keeps the model small and
     lets it generalize the same waveform morphology detectors across leads).
  2. Per-lead feature maps are pooled over time into a per-lead embedding,
     giving a (num_leads, D) "spatial map" of the heart's electrical
     activity, analogous to a spatial feature map in an image CNN where
     leads play the role of spatial position.
  3. A lead-attention block lets leads attend to each other (e.g. reciprocal
     ST changes between inferior and lateral leads), producing a global
     ECG embedding.
  4. Anatomical region pooling (MaskedRegionPool) additionally produces
     region-level embeddings used for regionalized findings (e.g. "STEMI -
     inferior" vs "STEMI - anterior").

Outputs:
  - global finding logits (multi-label, whole-ECG findings)
  - per-region logits for regionalized findings (ischemia/infarct localization)
  - the pooled embedding (used as the per-study representation fed into the
    temporal model for serial-ECG comparison)
"""
from __future__ import annotations

import torch
from torch import nn

from ecg_ai.clinical.labels import (
    NUM_LEADS, NUM_FINDINGS, FINDING_INDEX, REGIONALIZED_FINDING_KEYS,
    NUM_REGIONS, region_lead_mask,
)
from ecg_ai.models.layers import ResidualConvBlock1D, MaskedRegionPool


class PerLeadEncoder(nn.Module):
    """Shared 1D CNN stack applied independently to every lead."""

    def __init__(self, embed_dim: int = 128, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Conv1d(1, c, kernel_size=15, padding=7, bias=False)
        self.stem_bn = nn.BatchNorm1d(c)
        self.stem_act = nn.GELU()
        self.stages = nn.Sequential(
            ResidualConvBlock1D(c, c * 2, stride=2),
            ResidualConvBlock1D(c * 2, c * 4, stride=2),
            ResidualConvBlock1D(c * 4, c * 4, stride=2),
            ResidualConvBlock1D(c * 4, embed_dim, stride=2),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * num_leads, 1, T)
        h = self.stem_act(self.stem_bn(self.stem(x)))
        h = self.stages(h)  # (B*L, D, T')
        avg = h.mean(dim=-1)
        mx = h.amax(dim=-1)
        return avg + mx  # (B*L, D) pooled per-lead embedding


class LeadAttention(nn.Module):
    """Multi-head self-attention across the 12 lead embeddings so the model
    can relate findings in one lead to reciprocal findings in another
    (e.g. inferior ST elevation with anterolateral reciprocal depression).
    """

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lead_pos_embed = nn.Parameter(torch.randn(1, NUM_LEADS, dim) * 0.02)

    def forward(self, lead_embeds: torch.Tensor) -> torch.Tensor:
        # lead_embeds: (B, num_leads, D)
        return self.encoder(lead_embeds + self.lead_pos_embed)


class SpatialECGEncoder(nn.Module):
    """Full spatial encoder: per-lead CNN -> lead attention -> pooled embedding.

    This is the module reused (with shared weights) by the temporal model
    to encode each ECG in a serial sequence.
    """

    def __init__(self, embed_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.per_lead_encoder = PerLeadEncoder(embed_dim, base_channels)
        self.lead_attention = LeadAttention(embed_dim)
        self.region_pool = MaskedRegionPool(region_lead_mask())
        self.embed_dim = embed_dim

    def forward(self, ecg: torch.Tensor) -> dict[str, torch.Tensor]:
        """ecg: (B, num_leads, T) float tensor, normalized per-lead."""
        b, leads, t = ecg.shape
        assert leads == NUM_LEADS, f"expected {NUM_LEADS} leads, got {leads}"
        flat = ecg.reshape(b * leads, 1, t)
        lead_embeds = self.per_lead_encoder(flat).reshape(b, leads, -1)
        lead_embeds = self.lead_attention(lead_embeds)  # (B, L, D)
        region_embeds = self.region_pool(lead_embeds)  # (B, R, D)
        global_embed = lead_embeds.mean(dim=1)  # (B, D)
        return {
            "lead_embeds": lead_embeds,
            "region_embeds": region_embeds,
            "global_embed": global_embed,
        }


class SpatialECGClassifier(nn.Module):
    """Spatial encoder + classification heads producing global and
    per-region finding logits for a single ECG.
    """

    def __init__(self, embed_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.encoder = SpatialECGEncoder(embed_dim, base_channels)
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, NUM_FINDINGS),
        )
        # Regionalized findings get one logit per (finding, region) pair,
        # computed from that region's pooled embedding.
        self.region_findings = REGIONALIZED_FINDING_KEYS
        self.region_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, len(self.region_findings)),
        )

    def forward(self, ecg: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encoder(ecg)
        global_logits = self.global_head(enc["global_embed"])  # (B, NUM_FINDINGS)
        region_logits = self.region_head(enc["region_embeds"])  # (B, R, num_region_findings)
        return {
            **enc,
            "global_logits": global_logits,
            "region_logits": region_logits,
        }
