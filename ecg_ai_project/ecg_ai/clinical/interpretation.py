"""Turns model findings + DDx into a structured, human-readable report."""
from __future__ import annotations

from dataclasses import dataclass, field

from ecg_ai.clinical.ddx import DDxEntry, FindingResult

DISCLAIMER = (
    "Research/educational output only. Not a validated diagnostic device; "
    "must be independently reviewed and confirmed by a qualified clinician "
    "before any clinical decision is made."
)


@dataclass
class ECGReport:
    findings: list[FindingResult]
    differential_diagnosis: list[DDxEntry]
    summary: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "findings": [
                {
                    "finding": f.display_name,
                    "probability": round(f.probability, 3),
                    "category": f.category,
                }
                for f in self.findings
            ],
            "differential_diagnosis": [
                {
                    "diagnosis": d.diagnosis,
                    "rationale": d.rationale,
                    "supporting_findings": d.supporting_findings,
                    "score": round(d.score, 3),
                }
                for d in self.differential_diagnosis
            ],
            "disclaimer": self.disclaimer,
        }

    def to_text(self) -> str:
        lines = [self.summary, ""]
        lines.append("Findings:")
        if self.findings:
            for f in self.findings:
                lines.append(f"  - {f.display_name} (p={f.probability:.2f})")
        else:
            lines.append("  - No findings exceeded the reporting threshold.")
        lines.append("")
        lines.append("Differential diagnosis (ranked):")
        for i, d in enumerate(self.differential_diagnosis, start=1):
            support = ", ".join(d.supporting_findings) or "n/a"
            lines.append(f"  {i}. {d.diagnosis} [score={d.score:.2f}]")
            lines.append(f"     Rationale: {d.rationale}")
            lines.append(f"     Supporting findings: {support}")
        lines.append("")
        lines.append(f"** {self.disclaimer} **")
        return "\n".join(lines)


def _summarize(findings: list[FindingResult]) -> str:
    if not findings:
        return "ECG interpretation: No abnormal findings identified above threshold."
    top = findings[0]
    urgent_keys = {"stemi", "vt", "vfib", "avb3", "brugada"}
    urgency = " URGENT FINDING." if any(f.key in urgent_keys for f in findings) else ""
    others = ", ".join(f.display_name for f in findings[1:4])
    tail = f"; also: {others}" if others else ""
    return f"ECG interpretation: {top.display_name} (p={top.probability:.2f}){tail}.{urgency}"


def build_report(findings: list[FindingResult], ddx: list[DDxEntry]) -> ECGReport:
    return ECGReport(
        findings=findings,
        differential_diagnosis=ddx,
        summary=_summarize(findings),
    )
