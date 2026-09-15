import io

import numpy as np

from ecg_ai.clinical.labels import LEAD_NAMES, LEAD_INDEX
from ecg_ai.data.synthetic import SyntheticECGParams, generate_synthetic_ecg
from ecg_ai.imaging.digitize import STANDARD_3X4_LAYOUT, digitize_ecg_image
from ecg_ai.imaging.render_synthetic import render_ecg_image


def test_render_produces_expected_image_size():
    ecg = generate_synthetic_ecg(SyntheticECGParams(), fs=500, duration_s=10, seed=0)
    img = render_ecg_image(ecg, fs=500, panel_seconds=2.5, px_per_mm=6.0)
    assert img.size == (int(2.5 * 25 * 6.0) * 4, int(20 * 6.0) * 3)


def test_digitize_round_trip_shape_and_calibration():
    ecg = generate_synthetic_ecg(SyntheticECGParams(heart_rate=70), fs=500, duration_s=10, seed=0)
    img = render_ecg_image(ecg, fs=500, panel_seconds=2.5, px_per_mm=6.0)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = digitize_ecg_image(buf.getvalue(), target_fs=500.0)

    assert result.ecg.shape[0] == len(LEAD_NAMES)
    assert result.ecg.shape[1] > 0
    assert result.calibration_method in ("grid_autodetect", "fallback_autoscale")
    assert np.isfinite(result.ecg).all()


def test_digitize_grid_autodetect_recovers_known_px_per_mm():
    ecg = generate_synthetic_ecg(SyntheticECGParams(), fs=500, duration_s=10, seed=0)
    px_per_mm = 6.0
    img = render_ecg_image(ecg, fs=500, panel_seconds=2.5, px_per_mm=px_per_mm)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = digitize_ecg_image(buf.getvalue(), target_fs=500.0)

    assert result.calibration_method == "grid_autodetect"
    assert result.px_per_mm is not None
    px_x, px_y = result.px_per_mm
    # Grid self-calibration should recover the known small-box spacing
    # (1mm = px_per_mm pixels) within a reasonable tolerance.
    assert abs(px_x - px_per_mm) / px_per_mm < 0.25
    assert abs(px_y - px_per_mm) / px_per_mm < 0.25


def test_digitize_extracted_trace_correlates_with_source_signal():
    # Use a lead with clear, large-amplitude features (limb lead II, no
    # per-column time offset so the comparison window is trivial) to check
    # the pixel-tracing + calibration math actually recovers the waveform
    # shape, not just plausible output shapes.
    ecg = generate_synthetic_ecg(SyntheticECGParams(heart_rate=70), fs=500, duration_s=10, seed=0)
    img = render_ecg_image(ecg, fs=500, panel_seconds=2.5, px_per_mm=8.0, column_time_offset=False)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = digitize_ecg_image(buf.getvalue(), target_fs=500.0)

    lead_idx = LEAD_INDEX["II"]
    n = result.ecg.shape[1]
    original_segment = ecg[lead_idx, :n]
    digitized_segment = result.ecg[lead_idx]

    corr = np.corrcoef(original_segment, digitized_segment)[0, 1]
    assert corr > 0.8, f"expected strong correlation between source and digitized trace, got {corr}"


def test_digitize_respects_crop_box():
    ecg = generate_synthetic_ecg(SyntheticECGParams(), fs=500, duration_s=10, seed=0)
    img = render_ecg_image(ecg, fs=500, panel_seconds=2.5, px_per_mm=6.0)
    w, h = img.size

    # Pad the image with a fake "header" margin, then crop it back out.
    padded = img.copy()
    from PIL import Image as PILImage
    canvas = PILImage.new("RGB", (w, h + 100), (255, 255, 255))
    canvas.paste(padded, (0, 100))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    crop_box = (0.0, 100 / (h + 100), 1.0, 1.0)
    result = digitize_ecg_image(buf.getvalue(), crop_box=crop_box, target_fs=500.0)
    assert result.working_image.shape[0] == h
    assert not any("No crop region" in w_ for w_ in result.warnings)
