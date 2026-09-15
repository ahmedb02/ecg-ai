"""End-to-end inference pipeline: raw ECG signal(s) -> clinical report.

`ECGInterpreter` ties together preprocessing, the spatial CNN (single-ECG
findings + DDx), and the temporal model (serial-ECG evolution) into two
entry points:

  - `interpret(ecg, fs)` -> ECGReport for a single 12-lead ECG.
  - `interpret_serial(studies)` -> (latest ECGReport, EvolutionReport) for
    a time-ordered sequence of 12-lead ECGs from the same patient.
"""
from __future__ import annotations

import numpy as np
import torch

from ecg_ai.clinical.ddx import build_finding_results, generate_ddx
from ecg_ai.clinical.evolution import build_evolution_report
from ecg_ai.clinical.interpretation import ECGReport, build_report
from ecg_ai.clinical.labels import FINDINGS, REGIONALIZED_FINDING_KEYS, REGIONS
from ecg_ai.config import DEFAULT_MODEL_CONFIG, DEFAULT_SIGNAL_CONFIG, DEFAULT_TRAINING_CONFIG
from ecg_ai.data.preprocessing import preprocess_ecg
from ecg_ai.models.spatial_cnn import SpatialECGClassifier
from ecg_ai.models.temporal import EVOLUTION_LABELS, TemporalECGModel

FINDING_LABELS = {f.key: f.label for f in FINDINGS}
FINDING_CATEGORY = {f.key: f.category for f in FINDINGS}


class ECGInterpreter:
    """High-level, checkpoint-agnostic inference API.

    Construct with an already-instantiated (optionally pretrained)
    SpatialECGClassifier and, for serial-ECG support, a TemporalECGModel
    that shares (or was fine-tuned from) that classifier's encoder.
    """

    def __init__(
        self,
        spatial_classifier: SpatialECGClassifier,
        temporal_model: TemporalECGModel | None = None,
        finding_threshold: float = DEFAULT_TRAINING_CONFIG.finding_threshold,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spatial_classifier = spatial_classifier.to(self.device).eval()
        self.temporal_model = temporal_model.to(self.device).eval() if temporal_model else None
        self.finding_threshold = finding_threshold

    @classmethod
    def from_checkpoints(
        cls, spatial_checkpoint: str, temporal_checkpoint: str | None = None,
        device: str | None = None,
    ) -> "ECGInterpreter":
        model_cfg = DEFAULT_MODEL_CONFIG
        spatial_classifier = SpatialECGClassifier(
            embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
        )
        spatial_classifier.load_state_dict(torch.load(spatial_checkpoint, map_location="cpu"))

        temporal_model = None
        if temporal_checkpoint:
            temporal_model = TemporalECGModel(
                embed_dim=model_cfg.embed_dim, base_channels=model_cfg.base_channels,
                num_heads=model_cfg.temporal_heads, num_layers=model_cfg.temporal_layers,
            )
            temporal_model.load_state_dict(torch.load(temporal_checkpoint, map_location="cpu"))

        return cls(spatial_classifier, temporal_model, device=device)

    def _preprocess(self, ecg: np.ndarray, fs: float, target_seconds: float | None = None) -> torch.Tensor:
        # Model layers pool over time rather than assuming a fixed length,
        # so a shorter-than-default input (e.g. a digitized printed ECG,
        # which typically captures only ~2.5s per lead) is kept at its
        # natural duration instead of being edge-padded out to the full
        # 10s default -- padding that heavily would dilute the average
        # pooling in PerLeadEncoder with mostly-flat filler. A
        # longer-than-default input is still center-cropped to the default,
        # preserving prior behavior for normal-length recordings. Callers
        # that need several studies at a *common* length (interpret_serial,
        # which stacks them into one tensor) pass `target_seconds` explicitly.
        if target_seconds is None:
            natural_seconds = ecg.shape[-1] / fs
            target_seconds = max(min(natural_seconds, DEFAULT_SIGNAL_CONFIG.duration_seconds), 2.0)
        processed = preprocess_ecg(
            ecg, orig_fs=fs,
            target_fs=DEFAULT_SIGNAL_CONFIG.sampling_rate_hz,
            target_seconds=target_seconds,
        )
        return torch.from_numpy(processed).unsqueeze(0)  # (1, 12, T)

    def _findings_from_spatial_output(self, out: dict[str, torch.Tensor]):
        global_probs_all = torch.sigmoid(out["global_logits"][0]).tolist()
        global_probs = {
            f.key: p for f, p in zip(FINDINGS, global_probs_all)
            if f.key not in REGIONALIZED_FINDING_KEYS
        }

        region_probs_tensor = torch.sigmoid(out["region_logits"][0])  # (R, num_region_findings)
        region_probs: dict[str, dict[str, float]] = {k: {} for k in REGIONALIZED_FINDING_KEYS}
        for r_idx, region in enumerate(REGIONS):
            for f_idx, key in enumerate(REGIONALIZED_FINDING_KEYS):
                region_probs[key][region] = region_probs_tensor[r_idx, f_idx].item()

        results = build_finding_results(
            global_probs, region_probs, FINDING_LABELS, threshold=self.finding_threshold,
        )
        for r in results:
            r.category = FINDING_CATEGORY[r.key]
        return results

    @torch.no_grad()
    def interpret(self, ecg: np.ndarray, fs: float) -> ECGReport:
        """ecg: (12, num_samples) raw signal. fs: sampling rate in Hz."""
        tensor = self._preprocess(ecg, fs).to(self.device)
        out = self.spatial_classifier(tensor)
        findings = self._findings_from_spatial_output(out)
        ddx = generate_ddx(findings)
        return build_report(findings, ddx)

    @torch.no_grad()
    def interpret_serial(
        self, studies: list[tuple[np.ndarray, float, float]],
    ):
        """studies: time-ordered list of (ecg[12, T], fs, timestamp_hours).
        Returns (latest ECGReport, EvolutionReport).

        Requires the interpreter to have been constructed with a
        `temporal_model` (see `from_checkpoints` / `__init__`).
        """
        if self.temporal_model is None:
            raise ValueError("interpret_serial requires a temporal_model")
        if len(studies) < 2:
            raise ValueError("interpret_serial needs at least 2 studies to assess change")

        studies = sorted(studies, key=lambda s: s[2])
        # All studies must share one length to stack into a single tensor;
        # use the shortest study's natural duration (capped at the default)
        # so a short digitized study isn't padded out to a much longer one.
        common_seconds = min(
            max(min(ecg.shape[-1] / fs, DEFAULT_SIGNAL_CONFIG.duration_seconds), 2.0)
            for ecg, fs, _ in studies
        )
        tensors = [self._preprocess(ecg, fs, target_seconds=common_seconds) for ecg, fs, _ in studies]
        stacked = torch.stack(tensors, dim=1).to(self.device)  # (1, S, 12, T)
        t0 = studies[0][2]
        time_deltas = torch.tensor([[s[2] - t0 for s in studies]], dtype=torch.float32, device=self.device)
        study_mask = torch.ones(1, len(studies), dtype=torch.bool, device=self.device)

        temporal_out = self.temporal_model(stacked, time_deltas, study_mask)
        evolution_probs = {
            label: torch.softmax(temporal_out["evolution_logits"][0], dim=-1)[i].item()
            for i, label in enumerate(EVOLUTION_LABELS)
        }
        region_delta = {
            region: temporal_out["region_delta"][0, i].item() for i, region in enumerate(REGIONS)
        }

        # Per-study findings for the narrative diff use the spatial
        # classifier's heads, run on the baseline and latest study directly
        # (independent of the temporal transformer context).
        baseline_out = self.spatial_classifier(tensors[0].to(self.device))
        latest_out = self.spatial_classifier(tensors[-1].to(self.device))
        baseline_findings = self._findings_from_spatial_output(baseline_out)
        latest_findings = self._findings_from_spatial_output(latest_out)

        latest_report = build_report(latest_findings, generate_ddx(latest_findings))
        evolution_report = build_evolution_report(
            evolution_probs, region_delta, baseline_findings, latest_findings,
            time_span_hours=studies[-1][2] - studies[0][2],
        )
        return latest_report, evolution_report
