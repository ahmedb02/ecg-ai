"""Loss functions for the spatial classifier and the temporal model."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_loss(
    global_logits: torch.Tensor, global_targets: torch.Tensor,
    region_logits: torch.Tensor, region_targets: torch.Tensor,
    region_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Multi-label BCE over global findings + per-region findings.

    global_logits/targets: (B, NUM_FINDINGS)
    region_logits/targets: (B, NUM_REGIONS, num_region_findings)
    """
    global_bce = F.binary_cross_entropy_with_logits(global_logits, global_targets)
    region_bce = F.binary_cross_entropy_with_logits(region_logits, region_targets)
    total = global_bce + region_loss_weight * region_bce
    return {"loss": total, "global_bce": global_bce, "region_bce": region_bce}


def temporal_loss(evolution_logits: torch.Tensor, evolution_targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cross-entropy over the evolution label (single-label classification)."""
    loss = F.cross_entropy(evolution_logits, evolution_targets)
    return {"loss": loss}
