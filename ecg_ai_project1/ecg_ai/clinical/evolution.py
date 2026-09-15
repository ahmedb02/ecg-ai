"""Temporal comparison of serial ECGs: turns the temporal model's outputs
plus per-study finding sets into a plain-language "what changed" report.

Two complementary signals are combined:
  1. The learned evolution label (`TemporalECGModel.evolution_logits`),
     which captures patterns the transformer picked up across the whole
     sequence (e.g. the trajectory hyperacute-T -> ST-elevation -> Q-wave
     that indicates an evolving infarct even if no single snapshot looks
     dramatic).
  2. A transparent set-difference between the discrete findings detected
     at the baseline vs. latest study (new / resolved / persistent), plus
     the per-region embedding delta, so a clinician can see *why* the
     model called an evolution -- not just trust the label.
"""
from __future__ import annotations

from dataclasses import dataclass

from ecg_ai.clinical.ddx import FindingResult
from ecg_ai.clinical.interpretation import DISCLAIMER
from ecg_ai.models.temporal import EVOLUTION_LABELS

EVOLUTION_DESCRIPTIONS = {
    "no_significant_change": "No clinically significant interval change.",
    "new_or_evolving_stemi": "New or evolving ST-elevation pattern suggesting an active/evolving infarct.",
    "resolving_ischemia": "Resolving ST depression / T-wave changes.",
    "worsening_ischemia": "Worsening ST depression / T-wave changes.",
    "new_conduction_abnormality": "New conduction abnormality (bundle branch block / AV block).",
    "resolved_conduction_abnormality": "Previously seen conduction abnormality has resolved.",
    "new_arrhythmia": "New arrhythmia relative to prior study.",
    "resolved_arrhythmia": "Previously seen arrhythmia has resolved.",
    "interval_infarct_evolution": "Findings consistent with the expected temporal evolution of infarction "
                                    "(e.g. hyperacute T waves -> ST elevation -> Q wave formation).",
}


@dataclass
class EvolutionReport:
    evolution_label: str
    evolution_probability: float
    description: str
    new_findings: list[FindingResult]
    resolved_findings: list[FindingResult]
    persistent_findings: list[FindingResult]
    most_changed_regions: list[tuple[str, float]]
    time_span_hours: float
    narrative: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        def fmt(fs):
            return [{"finding": f.display_name, "probability": round(f.probability, 3)} for f in fs]

        return {
            "evolution_label": self.evolution_label,
            "evolution_probability": round(self.evolution_probability, 3),
            "description": self.description,
            "new_findings": fmt(self.new_findings),
            "resolved_findings": fmt(self.resolved_findings),
            "persistent_findings": fmt(self.persistent_findings),
            "most_changed_regions": [
                {"region": r, "change_score": round(s, 3)} for r, s in self.most_changed_regions
            ],
            "time_span_hours": round(self.time_span_hours, 1),
            "narrative": self.narrative,
            "disclaimer": self.disclaimer,
        }

    def to_text(self) -> str:
        return self.narrative + f"\n\n** {self.disclaimer} **"


def _finding_id(f: FindingResult) -> tuple[str, str | None]:
    return (f.key, f.region)


def diff_findings(
    baseline: list[FindingResult], latest: list[FindingResult]
) -> tuple[list[FindingResult], list[FindingResult], list[FindingResult]]:
    baseline_ids = {_finding_id(f): f for f in baseline}
    latest_ids = {_finding_id(f): f for f in latest}

    new = [f for fid, f in latest_ids.items() if fid not in baseline_ids]
    resolved = [f for fid, f in baseline_ids.items() if fid not in latest_ids]
    persistent = [f for fid, f in latest_ids.items() if fid in baseline_ids]
    return new, resolved, persistent


def build_evolution_report(
    evolution_probs: dict[str, float],
    region_delta: dict[str, float],
    baseline_findings: list[FindingResult],
    latest_findings: list[FindingResult],
    time_span_hours: float,
    top_n_regions: int = 3,
) -> EvolutionReport:
    top_label = max(evolution_probs, key=evolution_probs.get)
    top_prob = evolution_probs[top_label]

    new, resolved, persistent = diff_findings(baseline_findings, latest_findings)

    ranked_regions = sorted(region_delta.items(), key=lambda kv: kv[1], reverse=True)[:top_n_regions]

    narrative_parts = [
        f"Serial ECG comparison over {time_span_hours:.1f} hours: "
        f"{EVOLUTION_DESCRIPTIONS[top_label]} (model confidence {top_prob:.2f})."
    ]
    if new:
        narrative_parts.append(
            "New since prior study: " + ", ".join(f.display_name for f in new) + "."
        )
    if resolved:
        narrative_parts.append(
            "Resolved since prior study: " + ", ".join(f.display_name for f in resolved) + "."
        )
    if persistent:
        narrative_parts.append(
            "Unchanged/persistent: " + ", ".join(f.display_name for f in persistent) + "."
        )
    if ranked_regions:
        region_txt = ", ".join(f"{r.replace('_', ' ')} (Δ={s:.2f})" for r, s in ranked_regions)
        narrative_parts.append(f"Regions with the largest electrical change: {region_txt}.")

    return EvolutionReport(
        evolution_label=top_label,
        evolution_probability=top_prob,
        description=EVOLUTION_DESCRIPTIONS[top_label],
        new_findings=new,
        resolved_findings=resolved,
        persistent_findings=persistent,
        most_changed_regions=ranked_regions,
        time_span_hours=time_span_hours,
        narrative=" ".join(narrative_parts),
    )
