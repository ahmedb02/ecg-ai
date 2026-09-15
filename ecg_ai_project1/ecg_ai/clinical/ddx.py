"""Rule-based differential-diagnosis (DDx) generation from model findings.

The CNN outputs probabilities over ECG *findings* (ST elevation pattern,
rhythm, conduction abnormality, etc.), regionalized where relevant. This
module is a small, transparent clinical knowledge base that maps those
findings (individually and in combination) onto a ranked list of candidate
diagnoses with a stated rationale, the way an ECG teaching reference would.

This is intentionally rule-based rather than a second learned model: DDx
generation needs to be auditable by a clinician (why did the system suggest
this?), and the finding-to-diagnosis mappings are well-established medical
knowledge rather than something to be learned from a training set. The CNN
does the hard perceptual task (did the ECG show this pattern, and where);
this module does the reasoning step on top of that.

DISCLAIMER: This is a research/educational scaffold, not a validated
diagnostic device. Output must never be used for real clinical
decision-making without independent physician review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ecg_ai.clinical.labels import FINDING_INDEX, REGIONS


@dataclass
class FindingResult:
    key: str
    label: str
    probability: float
    region: str | None = None
    category: str = ""

    @property
    def display_name(self) -> str:
        if self.region:
            return f"{self.label} ({self.region.replace('_', ' ')})"
        return self.label


@dataclass
class DDxEntry:
    diagnosis: str
    rationale: str
    supporting_findings: list[str]
    score: float  # heuristic confidence, not a calibrated probability


# --- Knowledge base -------------------------------------------------------
# Non-regionalized findings -> candidate diagnoses.
FINDING_DDX_KB: dict[str, list[tuple[str, str]]] = {
    "afib": [
        ("Primary/idiopathic atrial fibrillation", "Irregularly irregular rhythm without P waves."),
        ("Atrial fibrillation secondary to hyperthyroidism", "Common reversible cause; check TSH."),
        ("Atrial fibrillation secondary to valvular disease", "Consider echo if new-onset."),
        ("Atrial fibrillation with sepsis/acute illness", "Consider in acutely unwell patients."),
    ],
    "aflutter": [
        ("Typical atrial flutter (cavotricuspid isthmus-dependent)", "Sawtooth flutter waves, often ~150 bpm with 2:1 block."),
        ("Atypical atrial flutter", "Consider if prior cardiac surgery/ablation."),
    ],
    "svt": [
        ("AV nodal reentrant tachycardia (AVNRT)", "Most common regular narrow-complex SVT."),
        ("AV reentrant tachycardia (AVRT, e.g. via accessory pathway)", "Consider if pre-excitation on baseline ECG."),
        ("Atrial tachycardia", "Consider if abnormal P-wave axis."),
    ],
    "vt": [
        ("Monomorphic ventricular tachycardia", "Wide-complex regular tachycardia; treat as VT until proven otherwise."),
        ("SVT with aberrancy", "Less likely given wide-complex morphology; use Brugada/Vereckei criteria to distinguish."),
    ],
    "vfib": [
        ("Ventricular fibrillation - cardiac arrest", "Chaotic, no organized complexes. Immediate defibrillation indicated."),
    ],
    "lbbb": [
        ("New left bundle branch block", "If new and symptomatic, evaluate for acute coronary occlusion (Sgarbossa criteria)."),
        ("Chronic left bundle branch block", "Consider prior ECGs / structural heart disease, hypertension."),
    ],
    "rbbb": [
        ("Right bundle branch block, isolated conduction disease", "Can be a normal variant."),
        ("Right bundle branch block secondary to RV strain (e.g. PE)", "Correlate with clinical context, S1Q3T3."),
    ],
    "avb3": [
        ("Complete heart block", "AV dissociation; evaluate for need for pacing."),
    ],
    "wpw": [
        ("Wolff-Parkinson-White pattern", "Delta wave, short PR; risk of pre-excited AFib/AVRT."),
    ],
    "pericarditis": [
        ("Acute pericarditis", "Diffuse concave ST elevation with PR depression, no reciprocal changes."),
        ("Early repolarization (mimic)", "Consider in young patients without chest pain/friction rub."),
    ],
    "long_qt": [
        ("Congenital long QT syndrome", "Consider if young, syncope, or family history of sudden death."),
        ("Drug-induced QT prolongation", "Review QT-prolonging medications."),
        ("Electrolyte-related QT prolongation", "Check K+, Mg2+, Ca2+."),
    ],
    "brugada": [
        ("Brugada pattern / syndrome", "Coved ST elevation V1-V2; risk of sudden cardiac death, consider EP referral."),
    ],
    "hyperkalemia": [
        ("Hyperkalemia", "Peaked T waves; if progressive, risk of sine-wave pattern and arrest."),
    ],
    "lvh": [
        ("Left ventricular hypertrophy, likely hypertensive", "Voltage criteria met; correlate with BP history/echo."),
        ("Hypertrophic cardiomyopathy", "Consider if disproportionate voltage or family history."),
    ],
    "posterior_mi": [
        ("Posterior myocardial infarction", "Tall R / ST depression V1-V3 as a mirror-image of posterior ST elevation."),
    ],
}

# Regionalized findings (STEMI / ischemia / old MI) -> candidate diagnoses
# and the culprit vessel typically implicated, by anatomical region.
REGIONAL_ISCHEMIA_DDX: dict[str, list[tuple[str, str]]] = {
    "inferior": [
        ("Acute inferior STEMI (RCA or LCx occlusion)", "ST elevation in II, III, aVF; check right-sided/posterior leads."),
        ("Pericarditis (mimic)", "Consider if diffuse rather than regional, with PR depression."),
    ],
    "anterior": [
        ("Acute anterior STEMI (LAD occlusion)", "ST elevation V3-V4; high-risk territory, large myocardial area at risk."),
        ("Left ventricular aneurysm (mimic)", "Consider if persistent ST elevation with known old anterior MI."),
    ],
    "anteroseptal": [
        ("Acute anteroseptal STEMI (proximal LAD occlusion)", "ST elevation V1-V4; assess for concurrent conduction block."),
    ],
    "septal": [
        ("Acute septal STEMI (septal perforator occlusion)", "ST elevation V1-V2."),
    ],
    "lateral": [
        ("Acute lateral STEMI (LCx or diagonal branch occlusion)", "ST elevation I, aVL, V5-V6."),
    ],
}


def _region_display(region: str) -> str:
    return region.replace("_", " ")


def build_finding_results(
    global_probs: dict[str, float],
    region_probs: dict[str, dict[str, float]],
    finding_labels: dict[str, str],
    threshold: float = 0.5,
) -> list[FindingResult]:
    """Turn raw probability dicts into thresholded FindingResult objects.

    global_probs: {finding_key: probability} for non-regionalized findings.
    region_probs: {finding_key: {region: probability}} for regionalized findings.
    """
    results: list[FindingResult] = []
    for key, prob in global_probs.items():
        if prob >= threshold:
            results.append(FindingResult(key=key, label=finding_labels[key], probability=prob))
    for key, by_region in region_probs.items():
        for region, prob in by_region.items():
            if prob >= threshold:
                results.append(FindingResult(
                    key=key, label=finding_labels[key], probability=prob, region=region,
                ))
    results.sort(key=lambda r: r.probability, reverse=True)
    return results


def generate_ddx(findings: list[FindingResult], top_k: int = 5) -> list[DDxEntry]:
    """Rank candidate diagnoses given the thresholded findings.

    Each finding contributes its knowledge-base diagnoses, weighted by the
    model's probability for that finding; diagnoses suggested by multiple
    findings (e.g. STEMI pattern + reciprocal ST depression) are boosted.
    """
    scored: dict[str, DDxEntry] = {}

    for f in findings:
        if f.region and f.key in ("stemi", "nstemi_ischemia", "old_mi"):
            candidates = REGIONAL_ISCHEMIA_DDX.get(f.region, [])
        else:
            candidates = FINDING_DDX_KB.get(f.key, [])

        for diagnosis, rationale in candidates:
            weight = f.probability
            if diagnosis in scored:
                entry = scored[diagnosis]
                entry.score = min(1.0, entry.score + weight * 0.3)
                entry.supporting_findings.append(f.display_name)
            else:
                scored[diagnosis] = DDxEntry(
                    diagnosis=diagnosis,
                    rationale=rationale,
                    supporting_findings=[f.display_name],
                    score=weight,
                )

    if not scored:
        scored["No acute abnormality suggested by model findings"] = DDxEntry(
            diagnosis="No acute abnormality suggested by model findings",
            rationale="No finding exceeded the reporting threshold; correlate clinically.",
            supporting_findings=[],
            score=0.0,
        )

    ranked = sorted(scored.values(), key=lambda e: e.score, reverse=True)
    return ranked[:top_k]
