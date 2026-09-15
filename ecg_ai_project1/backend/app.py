"""FastAPI backend: upload ECG image(s) -> digitize -> interpret -> DDx / evolution.

Run locally:
    uvicorn backend.app:app --reload --port 8000

Then open http://localhost:8000 (the frontend/ static site is served at "/").
Set SPATIAL_CHECKPOINT / TEMPORAL_CHECKPOINT env vars to use trained
weights (see ecg_ai/training/ or kaggle/train_ptbxl_on_kaggle.py); without
them the app still runs, on randomly-initialized weights, for demoing the
end-to-end pipeline (every response says so via `model_status`, on top of
the "not a validated diagnostic device" disclaimer already baked into every
report).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from backend.model_registry import ModelStatus, build_interpreter
from ecg_ai.clinical.evolution import EvolutionReport
from ecg_ai.clinical.interpretation import ECGReport
from ecg_ai.data.synthetic import make_case
from ecg_ai.imaging.digitize import DigitizationResult, digitize_ecg_image
from ecg_ai.imaging.render_synthetic import render_ecg_image

logger = logging.getLogger("ecg_ai.backend")

app = FastAPI(title="ecg_ai", description="Upload an ECG image, get findings + differential diagnosis.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Without this, an unhandled exception (e.g. a model-inference edge
    # case) surfaces to the frontend as a bare non-JSON 500 -- the fetch
    # call's JSON-parse of the error body fails, so the UI can only report
    # "HTTP 500" with no indication of what actually went wrong. This
    # guarantees a JSON body with the real exception, and logs the full
    # traceback server-side for platforms (Render/Fly/...) whose log
    # viewer is otherwise the only way to see it.
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )

INTERPRETER, MODEL_STATUS = build_interpreter()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB per image


def _model_status_out(status: ModelStatus) -> dict:
    if status.spatial_trained:
        message = "Using trained model weights."
    else:
        message = (
            "DEMO MODE: no trained checkpoint was found, so this is running "
            "on randomly-initialized weights. Findings/DDx below are "
            "plumbing output only, not real predictions -- see README for "
            "how to train (kaggle/train_ptbxl_on_kaggle.py is the fastest "
            "path to a real-data checkpoint) and set SPATIAL_CHECKPOINT / "
            "TEMPORAL_CHECKPOINT."
        )
    return {
        "spatial_trained": status.spatial_trained,
        "temporal_trained": status.temporal_trained,
        "message": message,
    }


def _report_out(report: ECGReport, digitization: dict) -> dict:
    d = report.to_dict()
    return {
        "summary": d["summary"],
        "findings": d["findings"],
        "differential_diagnosis": d["differential_diagnosis"],
        "disclaimer": d["disclaimer"],
        "digitization": digitization,
        "model_status": _model_status_out(MODEL_STATUS),
    }


def _evolution_out(evo: EvolutionReport) -> dict:
    return evo.to_dict()


def _build_overlay_data_url(result: DigitizationResult) -> str | None:
    """Draws the extracted per-lead trace back onto the working image, so
    the caller can visually sanity-check the digitization before trusting
    the interpretation -- important given this is a heuristic, non-learned
    digitizer (see ecg_ai/imaging/digitize.py docstring).
    """
    if result.working_image is None or result.panel_traces_px is None:
        return None
    img = Image.fromarray(result.working_image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for lead, (y0, y1, x0, x1) in result.panel_bboxes.items():
        trace = result.panel_traces_px.get(lead)
        if trace is None:
            continue
        for col, row in enumerate(trace):
            if not np.isfinite(row):
                continue
            px, py = x0 + col, y0 + row
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(0, 200, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _digitization_info_out(result: DigitizationResult) -> dict:
    return {
        "calibration_method": result.calibration_method,
        "px_per_mm": result.px_per_mm,
        "duration_seconds": result.ecg.shape[1] / result.fs,
        "warnings": result.warnings,
        "overlay_image_data_url": _build_overlay_data_url(result),
    }


def _parse_crop_box(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    try:
        box = json.loads(raw)
        if box is None:
            return None
        x0, y0, x1, y1 = (float(v) for v in box)
        return (x0, y0, x1, y1)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"invalid crop_box: {e}")


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 15 MB)")
    if not data:
        raise HTTPException(status_code=400, detail=f"empty upload: {file.filename}")
    return data


def _digitize_or_400(data: bytes, crop_box, px_per_mm_x: float | None, px_per_mm_y: float | None) -> DigitizationResult:
    manual = (px_per_mm_x, px_per_mm_y) if (px_per_mm_x and px_per_mm_y) else None
    try:
        return digitize_ecg_image(data, crop_box=crop_box, manual_px_per_mm=manual)
    except Exception as e:
        logger.exception("digitization failed")
        raise HTTPException(status_code=422, detail=f"could not digitize image: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_status": _model_status_out(MODEL_STATUS)}


@app.get("/api/example")
def example_image(condition: str = "stemi", region: str = "inferior"):
    """Returns a rendered synthetic example ECG image, so the app can be
    tried immediately without a real ECG photo on hand. NOT real patient
    data -- see ecg_ai/data/synthetic.py.
    """
    try:
        ecg, _, _ = make_case(condition, region if region else None, seed=7)
    except (ValueError, AssertionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    img = render_ecg_image(ecg, fs=500.0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/api/interpret")
async def interpret(
    image: UploadFile = File(...),
    crop_box: str | None = Form(None),
    px_per_mm_x: float | None = Form(None),
    px_per_mm_y: float | None = Form(None),
):
    data = await _read_upload(image)
    box = _parse_crop_box(crop_box)
    result = _digitize_or_400(data, box, px_per_mm_x, px_per_mm_y)

    report = INTERPRETER.interpret(result.ecg, fs=result.fs)
    return _report_out(report, _digitization_info_out(result))


@app.post("/api/interpret-serial")
async def interpret_serial(
    images: list[UploadFile] = File(...),
    timestamps: list[str] = Form(...),
    crop_boxes_json: str | None = Form(None),
    px_per_mm_x: float | None = Form(None),
    px_per_mm_y: float | None = Form(None),
):
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 ECG images to assess change over time")
    if len(images) != len(timestamps):
        raise HTTPException(status_code=400, detail="images and timestamps must have the same length")

    crop_boxes: list[tuple[float, float, float, float] | None] = [None] * len(images)
    if crop_boxes_json:
        try:
            raw_list = json.loads(crop_boxes_json)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"invalid crop_boxes_json: {e}")
        if len(raw_list) != len(images):
            raise HTTPException(status_code=400, detail="crop_boxes_json must have one entry per image")
        crop_boxes = [tuple(b) if b else None for b in raw_list]

    try:
        parsed_times = [datetime.fromisoformat(t) for t in timestamps]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid timestamp (expected ISO 8601): {e}")
    t0 = min(parsed_times)
    hours = [(t - t0).total_seconds() / 3600.0 for t in parsed_times]

    digitized: list[DigitizationResult] = []
    for upload, box in zip(images, crop_boxes):
        data = await _read_upload(upload)
        digitized.append(_digitize_or_400(data, box, px_per_mm_x, px_per_mm_y))

    studies = [(d.ecg, d.fs, h) for d, h in zip(digitized, hours)]
    latest_report, evolution_report = INTERPRETER.interpret_serial(studies)

    # digitization info list re-sorted to match interpret_serial's internal
    # chronological sort, so it lines up with what was actually compared.
    order = sorted(range(len(hours)), key=lambda i: hours[i])
    per_study_info = [_digitization_info_out(digitized[i]) for i in order]

    return {
        "latest_report": _report_out(latest_report, per_study_info[-1]),
        "evolution": _evolution_out(evolution_report),
        "per_study_digitization": per_study_info,
    }


_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
