import io
import json
from datetime import datetime, timedelta

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from ecg_ai.data.synthetic import make_case
from ecg_ai.imaging.render_synthetic import render_ecg_image
from backend.app import app

client = TestClient(app)


def _png_bytes(condition="normal", region=None, seed=1) -> bytes:
    ecg, _, _ = make_case(condition, region, seed=seed)
    img = render_ecg_image(ecg, fs=500.0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_status" in body


def test_example_endpoint_returns_png():
    resp = client.get("/api/example?condition=stemi&region=inferior")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


def test_interpret_single_image():
    png = _png_bytes("stemi", "anterior", seed=2)
    resp = client.post(
        "/api/interpret",
        files={"image": ("ecg.png", png, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert "findings" in body
    assert "differential_diagnosis" in body
    assert "disclaimer" in body
    assert "digitization" in body
    assert body["digitization"]["calibration_method"] in ("grid_autodetect", "fallback_autoscale")
    assert body["model_status"]["spatial_trained"] is False  # no checkpoint in test env


def test_interpret_with_crop_box():
    png = _png_bytes("normal", seed=3)
    resp = client.post(
        "/api/interpret",
        files={"image": ("ecg.png", png, "image/png")},
        data={"crop_box": json.dumps([0.0, 0.0, 1.0, 1.0])},
    )
    assert resp.status_code == 200


def test_interpret_rejects_non_image():
    resp = client.post(
        "/api/interpret",
        files={"image": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422


def test_interpret_serial_two_images():
    png1 = _png_bytes("nstemi_ischemia", "anterior", seed=4)
    png2 = _png_bytes("stemi", "anterior", seed=5)
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    t1 = t0 + timedelta(hours=3)

    resp = client.post(
        "/api/interpret-serial",
        files=[
            ("images", ("ecg1.png", png1, "image/png")),
            ("images", ("ecg2.png", png2, "image/png")),
        ],
        data={"timestamps": [t0.isoformat(), t1.isoformat()]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "latest_report" in body
    assert "evolution" in body
    assert body["evolution"]["time_span_hours"] == pytest.approx(3.0, abs=0.01)
    assert len(body["per_study_digitization"]) == 2


def test_interpret_serial_requires_at_least_two_images():
    png = _png_bytes("normal", seed=6)
    resp = client.post(
        "/api/interpret-serial",
        files=[("images", ("ecg.png", png, "image/png"))],
        data={"timestamps": [datetime.now().isoformat()]},
    )
    assert resp.status_code == 400


def test_interpret_serial_mismatched_lengths_rejected():
    png1 = _png_bytes("normal", seed=7)
    png2 = _png_bytes("normal", seed=8)
    resp = client.post(
        "/api/interpret-serial",
        files=[
            ("images", ("ecg1.png", png1, "image/png")),
            ("images", ("ecg2.png", png2, "image/png")),
        ],
        data={"timestamps": [datetime.now().isoformat()]},
    )
    assert resp.status_code == 400
