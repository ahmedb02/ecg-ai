"""Pydantic models documenting the FastAPI JSON response shapes.

Reference/OpenAPI-docs only -- endpoints in app.py return the underlying
ECGReport/EvolutionReport.to_dict() output directly rather than being
bound via `response_model`, since a couple of nested shapes diverge
slightly between contexts (e.g. evolution new/resolved/persistent findings
omit `category`). Keep this file in sync with app.py's actual dict shapes
when either changes.
"""
from __future__ import annotations

from pydantic import BaseModel


class ModelStatusOut(BaseModel):
    spatial_trained: bool
    temporal_trained: bool
    message: str


class FindingOut(BaseModel):
    finding: str
    probability: float
    category: str


class DDxEntryOut(BaseModel):
    diagnosis: str
    rationale: str
    supporting_findings: list[str]
    score: float


class DigitizationInfoOut(BaseModel):
    calibration_method: str
    px_per_mm: tuple[float, float] | None
    duration_seconds: float
    warnings: list[str]
    overlay_image_data_url: str | None = None


class ReportOut(BaseModel):
    summary: str
    findings: list[FindingOut]
    differential_diagnosis: list[DDxEntryOut]
    disclaimer: str
    digitization: DigitizationInfoOut
    model_status: ModelStatusOut


class EvolutionOut(BaseModel):
    evolution_label: str
    evolution_probability: float
    description: str
    new_findings: list[FindingOut]
    resolved_findings: list[FindingOut]
    persistent_findings: list[FindingOut]
    most_changed_regions: list[dict]
    time_span_hours: float
    narrative: str
    disclaimer: str


class SerialReportOut(BaseModel):
    latest_report: ReportOut
    evolution: EvolutionOut
    per_study_digitization: list[DigitizationInfoOut]
