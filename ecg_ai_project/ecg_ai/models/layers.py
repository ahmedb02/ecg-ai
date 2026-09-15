"""Reusable building blocks for the ECG CNN encoders."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Module):
    """Conv -> BatchNorm -> activation, with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        weights = self.fc(x.mean(dim=-1))  # (B, C)
        return x * weights.unsqueeze(-1)


class ResidualConvBlock1D(nn.Module):
    """Two conv blocks with a residual connection and SE gating."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block1 = ConvBlock1D(in_ch, out_ch, kernel_size=7, stride=stride)
        self.block2 = ConvBlock1D(out_ch, out_ch, kernel_size=7, stride=1)
        self.se = SEBlock1D(out_ch)
        self.residual = (
            nn.Identity() if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.se(out)
        return out + identity


class MaskedRegionPool(nn.Module):
    """Mean-pools per-lead embeddings into fixed anatomical region embeddings.

    Given per-lead feature vectors (B, num_leads, D) and a list of lead
    index groups (one per region), produces (B, num_regions, D).
    """

    def __init__(self, region_lead_indices: list[list[int]]):
        super().__init__()
        self.region_lead_indices = region_lead_indices

    def forward(self, lead_features: torch.Tensor) -> torch.Tensor:
        region_feats = []
        for indices in self.region_lead_indices:
            idx = torch.tensor(indices, device=lead_features.device)
            region_feats.append(lead_features.index_select(1, idx).mean(dim=1))
        return torch.stack(region_feats, dim=1)


class SinusoidalTimeEncoding(nn.Module):
    """Encodes a continuous time delta (e.g. hours/days since baseline ECG)
    into a vector added to the temporal sequence embeddings, so the
    transformer is aware of irregular spacing between serial ECGs.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        # time_deltas: (B, S) in hours since first study in the sequence
        half = self.dim // 2
        device = time_deltas.device
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32)
            * (torch.log(torch.tensor(self.max_period)) / half)
        )
        args = time_deltas.unsqueeze(-1).float() * freqs  # (B, S, half)
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if enc.shape[-1] < self.dim:
            enc = torch.nn.functional.pad(enc, (0, self.dim - enc.shape[-1]))
        return enc
