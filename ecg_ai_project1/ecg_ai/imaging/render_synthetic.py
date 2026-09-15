"""Renders a synthetic (12, T) ECG signal as a fake printed-ECG-style image
(grid + trace), matching the standard 3x4 panel layout with the same
per-column time offset a real printed ECG has (see `digitize.py`'s module
docstring). Test-only / demo-only: this is how `tests/test_digitize.py`
validates the digitizer round-trips a known signal without needing a real
scanned ECG image, and it's also usable to generate a "try it out" example
image for the web app.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ecg_ai.clinical.labels import LEAD_INDEX
from ecg_ai.imaging.digitize import (
    DEFAULT_PAPER_MV_MM, DEFAULT_PAPER_SPEED_MM_S, STANDARD_3X4_LAYOUT,
)


def render_ecg_image(
    ecg: np.ndarray,
    fs: float,
    panel_seconds: float = 2.5,
    layout: list[list[str]] | None = None,
    px_per_mm: float = 6.0,
    paper_speed_mm_s: float = DEFAULT_PAPER_SPEED_MM_S,
    paper_mv_mm: float = DEFAULT_PAPER_MV_MM,
    panel_height_mm: float = 20.0,
    column_time_offset: bool = True,
    line_color: tuple[int, int, int] = (20, 20, 30),
    grid_color: tuple[int, int, int] = (255, 190, 190),
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    layout = layout or STANDARD_3X4_LAYOUT
    rows, cols = len(layout), len(layout[0])
    panel_px_w = int(round(panel_seconds * paper_speed_mm_s * px_per_mm))
    panel_px_h = int(round(panel_height_mm * px_per_mm))
    width, height = panel_px_w * cols, panel_px_h * rows

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    small_box_px = px_per_mm
    x = 0.0
    i = 0
    while x < width:
        bold = i % 5 == 0
        draw.line([(x, 0), (x, height)], fill=grid_color, width=2 if bold else 1)
        x += small_box_px
        i += 1
    y = 0.0
    i = 0
    while y < height:
        bold = i % 5 == 0
        draw.line([(0, y), (width, y)], fill=grid_color, width=2 if bold else 1)
        y += small_box_px
        i += 1

    for r in range(rows):
        for c in range(cols):
            lead = layout[r][c]
            lead_idx = LEAD_INDEX[lead]
            panel_x0, panel_y0 = c * panel_px_w, r * panel_px_h
            baseline_y = panel_y0 + panel_px_h / 2.0
            t_start = (c * panel_seconds) if column_time_offset else 0.0

            points = []
            for px in range(panel_px_w):
                t = t_start + (px / panel_px_w) * panel_seconds
                sample_idx = int(round(t * fs))
                sample_idx = min(max(sample_idx, 0), ecg.shape[1] - 1)
                mv = ecg[lead_idx, sample_idx]
                y_px = baseline_y - mv * paper_mv_mm * px_per_mm
                points.append((panel_x0 + px, y_px))
            draw.line(points, fill=line_color, width=2)

    return img
