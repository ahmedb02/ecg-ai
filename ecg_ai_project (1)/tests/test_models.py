import torch

from ecg_ai.clinical.labels import (
    NUM_LEADS, NUM_FINDINGS, NUM_REGIONS, REGIONALIZED_FINDING_KEYS,
)
from ecg_ai.models.spatial_cnn import SpatialECGClassifier, SpatialECGEncoder
from ecg_ai.models.temporal import TemporalECGModel, NUM_EVOLUTION_LABELS


def test_spatial_encoder_output_shapes():
    encoder = SpatialECGEncoder(embed_dim=32, base_channels=8)
    ecg = torch.randn(3, NUM_LEADS, 2000)
    out = encoder(ecg)
    assert out["lead_embeds"].shape == (3, NUM_LEADS, 32)
    assert out["region_embeds"].shape == (3, NUM_REGIONS, 32)
    assert out["global_embed"].shape == (3, 32)


def test_spatial_classifier_output_shapes():
    model = SpatialECGClassifier(embed_dim=32, base_channels=8)
    ecg = torch.randn(2, NUM_LEADS, 2000)
    out = model(ecg)
    assert out["global_logits"].shape == (2, NUM_FINDINGS)
    assert out["region_logits"].shape == (2, NUM_REGIONS, len(REGIONALIZED_FINDING_KEYS))


def test_spatial_classifier_gradients_flow():
    model = SpatialECGClassifier(embed_dim=16, base_channels=8)
    ecg = torch.randn(2, NUM_LEADS, 1000, requires_grad=False)
    out = model(ecg)
    loss = out["global_logits"].sum() + out["region_logits"].sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs


def test_temporal_model_output_shapes():
    model = TemporalECGModel(embed_dim=16, base_channels=8, num_heads=2, num_layers=1)
    b, s = 2, 3
    ecgs = torch.randn(b, s, NUM_LEADS, 1000)
    time_deltas = torch.tensor([[0.0, 2.0, 10.0], [0.0, 5.0, 0.0]])
    study_mask = torch.tensor([[True, True, True], [True, True, False]])

    out = model(ecgs, time_deltas, study_mask)
    assert out["evolution_logits"].shape == (b, NUM_EVOLUTION_LABELS)
    assert out["region_delta"].shape == (b, NUM_REGIONS)
    # last_idx should reflect the mask: batch 1 has only 2 real studies -> idx 1
    assert out["last_idx"][1].item() == 1
    assert out["last_idx"][0].item() == 2


def test_temporal_model_shares_spatial_encoder_weights():
    encoder = SpatialECGEncoder(embed_dim=16, base_channels=8)
    model = TemporalECGModel(embed_dim=16, base_channels=8, spatial_encoder=encoder)
    assert model.spatial_encoder is encoder
