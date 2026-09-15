"""Clinical label taxonomy for the ECG model.

Findings are multi-label (an ECG can show several simultaneously) and are
grouped by clinical category. Anatomical region groupings are used both by
the model's spatial fusion module and by the rule-based localization logic
in `ddx.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}
NUM_LEADS = len(LEAD_NAMES)

# Anatomical regions represented by contiguous / physiologically related
# lead groups. Used to localize ischemia/infarct and to fuse per-lead
# features spatially rather than treating leads as an unordered channel set.
REGION_LEADS = {
    "inferior": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "anteroseptal": ["V1", "V2", "V3", "V4"],
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "posterior_reciprocal": ["V1", "V2", "V3"],
}
REGIONS = list(REGION_LEADS.keys())


@dataclass(frozen=True)
class Finding:
    key: str
    label: str
    category: str
    regionalized: bool = False  # True if this finding is reported per-region


FINDINGS: list[Finding] = [
    # Rhythm
    Finding("nsr", "Normal sinus rhythm", "rhythm"),
    Finding("sinus_brady", "Sinus bradycardia", "rhythm"),
    Finding("sinus_tach", "Sinus tachycardia", "rhythm"),
    Finding("afib", "Atrial fibrillation", "rhythm"),
    Finding("aflutter", "Atrial flutter", "rhythm"),
    Finding("svt", "Supraventricular tachycardia", "rhythm"),
    Finding("vt", "Ventricular tachycardia", "rhythm"),
    Finding("vfib", "Ventricular fibrillation", "rhythm"),
    Finding("pac", "Premature atrial complexes", "rhythm"),
    Finding("pvc", "Premature ventricular complexes", "rhythm"),
    # Conduction
    Finding("lbbb", "Left bundle branch block", "conduction"),
    Finding("rbbb", "Right bundle branch block", "conduction"),
    Finding("lafb", "Left anterior fascicular block", "conduction"),
    Finding("lpfb", "Left posterior fascicular block", "conduction"),
    Finding("avb1", "First-degree AV block", "conduction"),
    Finding("avb2_i", "Second-degree AV block (Mobitz I)", "conduction"),
    Finding("avb2_ii", "Second-degree AV block (Mobitz II)", "conduction"),
    Finding("avb3", "Third-degree (complete) AV block", "conduction"),
    Finding("wpw", "Wolff-Parkinson-White pattern", "conduction"),
    # Ischemia / infarction (regionalized -> localized per anatomical zone)
    Finding("stemi", "ST-elevation myocardial infarction pattern", "ischemia", regionalized=True),
    Finding("nstemi_ischemia", "ST depression / T-wave ischemia", "ischemia", regionalized=True),
    Finding("old_mi", "Old myocardial infarction (pathologic Q waves)", "ischemia", regionalized=True),
    Finding("posterior_mi", "Posterior myocardial infarction (reciprocal pattern)", "ischemia"),
    # Hypertrophy / enlargement
    Finding("lvh", "Left ventricular hypertrophy", "hypertrophy"),
    Finding("rvh", "Right ventricular hypertrophy", "hypertrophy"),
    Finding("lae", "Left atrial enlargement", "hypertrophy"),
    Finding("rae", "Right atrial enlargement", "hypertrophy"),
    # Other
    Finding("pericarditis", "Pericarditis pattern", "other"),
    Finding("long_qt", "Prolonged QT interval", "other"),
    Finding("brugada", "Brugada pattern", "other"),
    Finding("hyperkalemia", "Peaked T waves / hyperkalemia pattern", "other"),
    Finding("low_voltage", "Low QRS voltage", "other"),
    Finding("paced", "Paced rhythm", "other"),
]

FINDING_KEYS = [f.key for f in FINDINGS]
NUM_FINDINGS = len(FINDINGS)
FINDING_INDEX = {f.key: i for i, f in enumerate(FINDINGS)}

REGIONALIZED_FINDING_KEYS = [f.key for f in FINDINGS if f.regionalized]
NUM_REGIONS = len(REGIONS)
REGION_INDEX = {r: i for i, r in enumerate(REGIONS)}


def region_lead_mask() -> list[list[int]]:
    """Boolean-style lead index list per region, for masked pooling."""
    return [[LEAD_INDEX[l] for l in leads] for leads in REGION_LEADS.values()]
