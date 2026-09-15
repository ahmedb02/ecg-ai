import numpy as np
import torch

from ecg_ai.clinical.ddx import FindingResult, generate_ddx
from ecg_ai.clinical.evolution import build_evolution_report, diff_findings
from ecg_ai.clinical.interpretation import build_report
from ecg_ai.clinical.labels import LEAD_NAMES, NUM_LEADS
from ecg_ai.data.dataset import (
    SyntheticSpatialDataset, SyntheticTemporalDataset, collate_temporal_batch,
    labels_to_vectors,
)
from ecg_ai.data.preprocessing import preprocess_ecg
from ecg_ai.data.synthetic import generate_synthetic_ecg, make_case, SyntheticECGParams
from ecg_ai.inference.predict import ECGInterpreter
from ecg_ai.models.spatial_cnn import SpatialECGClassifier
from ecg_ai.models.temporal import TemporalECGModel


def test_synthetic_generator_shape():
    ecg = generate_synthetic_ecg(SyntheticECGParams(), fs=500, duration_s=10)
    assert ecg.shape == (NUM_LEADS, 5000)
    assert np.isfinite(ecg).all()


def test_make_case_stemi_region_labels():
    ecg, global_findings, regional_findings = make_case("stemi", region="inferior", seed=1)
    assert ecg.shape[0] == NUM_LEADS
    assert regional_findings == [("stemi", "inferior")]
    assert global_findings == []


def test_preprocess_ecg_normalizes_and_resizes():
    ecg = generate_synthetic_ecg(SyntheticECGParams(), fs=250, duration_s=8)
    out = preprocess_ecg(ecg, orig_fs=250, target_fs=500, target_seconds=10)
    assert out.shape == (NUM_LEADS, 5000)
    # roughly zero-mean, unit-ish variance per lead after z-score normalization
    assert abs(out.mean()) < 0.5


def test_synthetic_spatial_dataset_item_shapes():
    ds = SyntheticSpatialDataset(epoch_size=5, seed=42)
    ecg, global_vec, region_vec = ds[0]
    assert ecg.shape[0] == NUM_LEADS
    assert global_vec.ndim == 1
    assert region_vec.ndim == 2


def test_synthetic_temporal_dataset_and_collate():
    ds = SyntheticTemporalDataset(epoch_size=6, max_studies=4, seed=7)
    batch = [ds[i] for i in range(4)]
    ecgs, time_deltas, study_mask, labels = collate_temporal_batch(batch)
    assert ecgs.shape[0] == 4
    assert ecgs.shape[2] == NUM_LEADS
    assert time_deltas.shape == study_mask.shape
    assert labels.shape == (4,)
    # every real study should be within the mask, padding should be zeroed
    for i in range(4):
        s = batch[i][0].shape[0]
        assert study_mask[i, :s].all()
        if s < ecgs.shape[1]:
            assert not study_mask[i, s:].any()


def test_generate_ddx_ranks_stemi_highest_for_matching_evidence():
    findings = [
        FindingResult(key="stemi", label="ST-elevation myocardial infarction pattern",
                      probability=0.95, region="inferior", category="ischemia"),
        FindingResult(key="sinus_tach", label="Sinus tachycardia", probability=0.6, category="rhythm"),
    ]
    ddx = generate_ddx(findings, top_k=5)
    assert len(ddx) > 0
    assert "inferior STEMI" in ddx[0].diagnosis or "inferior" in ddx[0].diagnosis.lower()
    assert ddx[0].score >= ddx[-1].score


def test_generate_ddx_empty_findings_gives_no_abnormality():
    ddx = generate_ddx([], top_k=5)
    assert len(ddx) == 1
    assert "No acute abnormality" in ddx[0].diagnosis


def test_build_report_text_contains_disclaimer():
    findings = [FindingResult(key="afib", label="Atrial fibrillation", probability=0.8, category="rhythm")]
    ddx = generate_ddx(findings)
    report = build_report(findings, ddx)
    text = report.to_text()
    assert "Atrial fibrillation" in text
    assert "Not a validated diagnostic device" in text
    d = report.to_dict()
    assert d["findings"][0]["finding"] == "Atrial fibrillation"


def test_diff_findings_new_resolved_persistent():
    baseline = [
        FindingResult(key="nstemi_ischemia", label="ST depression / T-wave ischemia",
                      probability=0.7, region="anterior"),
        FindingResult(key="sinus_tach", label="Sinus tachycardia", probability=0.6),
    ]
    latest = [
        FindingResult(key="stemi", label="ST-elevation myocardial infarction pattern",
                      probability=0.9, region="anterior"),
        FindingResult(key="sinus_tach", label="Sinus tachycardia", probability=0.55),
    ]
    new, resolved, persistent = diff_findings(baseline, latest)
    assert [f.key for f in new] == ["stemi"]
    assert [f.key for f in resolved] == ["nstemi_ischemia"]
    assert [f.key for f in persistent] == ["sinus_tach"]


def test_build_evolution_report_narrative():
    baseline = [FindingResult(key="nstemi_ischemia", label="ST depression / T-wave ischemia",
                               probability=0.7, region="anterior")]
    latest = [FindingResult(key="stemi", label="ST-elevation myocardial infarction pattern",
                             probability=0.9, region="anterior")]
    evolution_probs = {
        "no_significant_change": 0.05, "new_or_evolving_stemi": 0.1,
        "resolving_ischemia": 0.02, "worsening_ischemia": 0.7,
        "new_conduction_abnormality": 0.03, "resolved_conduction_abnormality": 0.02,
        "new_arrhythmia": 0.03, "resolved_arrhythmia": 0.03, "interval_infarct_evolution": 0.02,
    }
    region_delta = {"inferior": 0.1, "lateral": 0.2, "anteroseptal": 0.3, "septal": 0.1, "anterior": 0.9,
                     "posterior_reciprocal": 0.2}
    report = build_evolution_report(evolution_probs, region_delta, baseline, latest, time_span_hours=6.0)
    assert report.evolution_label == "worsening_ischemia"
    assert "anterior" in report.narrative.lower()
    assert report.new_findings[0].key == "stemi"
    assert report.resolved_findings[0].key == "nstemi_ischemia"


def test_end_to_end_single_ecg_interpretation_runs():
    torch.manual_seed(0)
    classifier = SpatialECGClassifier(embed_dim=16, base_channels=8)
    interpreter = ECGInterpreter(classifier, finding_threshold=0.5)

    ecg, _, _ = make_case("normal", seed=3)
    report = interpreter.interpret(ecg, fs=500)
    assert report.summary
    assert isinstance(report.findings, list)
    assert isinstance(report.differential_diagnosis, list)


def test_end_to_end_serial_interpretation_runs():
    torch.manual_seed(0)
    classifier = SpatialECGClassifier(embed_dim=16, base_channels=8)
    temporal_model = TemporalECGModel(embed_dim=16, base_channels=8, num_heads=2, num_layers=1)
    interpreter = ECGInterpreter(classifier, temporal_model, finding_threshold=0.5)

    ecg1, _, _ = make_case("normal", seed=1)
    ecg2, _, _ = make_case("stemi", region="anterior", seed=2)
    studies = [(ecg1, 500.0, 0.0), (ecg2, 500.0, 3.0)]

    latest_report, evolution_report = interpreter.interpret_serial(studies)
    assert latest_report.summary
    assert evolution_report.evolution_label in {
        "no_significant_change", "new_or_evolving_stemi", "resolving_ischemia",
        "worsening_ischemia", "new_conduction_abnormality", "resolved_conduction_abnormality",
        "new_arrhythmia", "resolved_arrhythmia", "interval_infarct_evolution",
    }
    assert evolution_report.time_span_hours == 3.0
    d = evolution_report.to_dict()
    assert "narrative" in d
