"""Digitizes a photo/scan of a printed 12-lead ECG into a (12, T) signal
array the rest of `ecg_ai` can consume.

Pipeline: load image -> optional user crop (to exclude headers/margins) ->
isolate ink pixels from the grid/background -> self-calibrate the ECG
paper's mm grid spacing directly from the image (via the dominant
periodicity of the grid lines -- this works regardless of the photo's
actual pixel resolution/DPI, since it never needs to know that) -> split
into the standard 3-row x 4-column lead panel layout -> trace each panel's
ink column-by-column -> convert pixel position to (time, amplitude) using
the calibration -> resample every lead to a common sampling rate.

Important limitation, stated once here and surfaced to the user in the web
UI: a standard printed 12-lead ECG shows each lead over a *different* few
seconds of real time (one column of panels per short interval), not all 12
leads simultaneously. So leads extracted this way are not perfectly
time-synchronized with each other, unlike leads from a real digital
12-lead acquisition (or `ecg_ai`'s synthetic generator). Per-lead
morphology and regional grouping -- most of this repo's findings -- still
work reasonably on that basis; anything relying on precise cross-lead
timing (subtle reciprocal-change timing) is weaker on digitized-from-print
input than on true simultaneous digital data. Digitization quality also
depends heavily on image contrast/cleanliness; noisy photos will digitize
poorly. This is a best-effort classical (non-learned) digitizer, not a
validated tool -- treat its output the same way as every other report this
repo produces: research/educational, not diagnostic.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

STANDARD_3X4_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

DEFAULT_PAPER_SPEED_MM_S = 25.0  # standard ECG paper speed
DEFAULT_PAPER_MV_MM = 10.0       # standard ECG paper gain (10 mm per mV)

MAX_IMAGE_DIM = 2200  # downscale very large photos for tractable processing


@dataclass
class DigitizationResult:
    ecg: np.ndarray  # (12, T) float32, millivolts
    fs: float
    panel_bboxes: dict[str, tuple[int, int, int, int]]  # lead -> (y0, y1, x0, x1) in the working image
    calibration_method: str  # "grid_autodetect" | "fallback_autoscale" | "manual"
    px_per_mm: tuple[float, float] | None  # (x, y), None if fallback/manual amplitude-only
    warnings: list[str] = field(default_factory=list)
    working_image: np.ndarray | None = None  # the (possibly cropped/resized) RGB image, for overlay/debug
    panel_traces_px: dict[str, np.ndarray] | None = None  # lead -> per-column row position (px, within-panel), for overlay/debug


def _load_image_array(image: bytes | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    else:
        if isinstance(image, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image))
        else:
            pil_img = image
        arr = np.array(pil_img.convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def _resize_max_dim(img: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return np.array(Image.fromarray(img).resize(new_size, Image.BILINEAR))


def _crop_normalized(img: np.ndarray, crop_box: tuple[float, float, float, float] | None) -> np.ndarray:
    if crop_box is None:
        return img
    h, w = img.shape[:2]
    x0f, y0f, x1f, y1f = crop_box
    x0, x1 = sorted((int(round(x0f * w)), int(round(x1f * w))))
    y0, y1 = sorted((int(round(y0f * h)), int(round(y1f * h))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
    return img[y0:y1, x0:x1]


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _ink_mask(gray: np.ndarray, rgb: np.ndarray, block: int = 25, abs_dark_threshold: float = 170.0) -> np.ndarray:
    """Adaptive local thresholding (numpy-only): a pixel is "ink" if it is
    noticeably darker than its local neighborhood (robust to uneven
    lighting in photographed, vs. flatbed-scanned, ECGs) AND darker than a
    fairly conservative absolute brightness cutoff AND not a reddish/pink
    grid-line color. All three checks matter: relative-only thresholding
    also flags grid lines as ink wherever they're the only non-background
    content in a mostly-white neighborhood (they're clearly lighter than
    the trace in absolute terms, just not lighter than *pure white*).
    """
    block = block if block % 2 == 1 else block + 1
    pad = block // 2
    padded = np.pad(gray, pad, mode="reflect")
    # Box-filtered local mean via cumulative sum (fast, no scipy dependency).
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = gray.shape
    area = block * block
    local_sum = (
        cs[block:block + h, block:block + w]
        - cs[0:h, block:block + w]
        - cs[block:block + h, 0:w]
        + cs[0:h, 0:w]
    )
    local_mean = local_sum / area
    relative_dark = gray < (local_mean - 12.0)
    absolute_dark = gray < abs_dark_threshold

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    reddish_grid_color = (r - g > 15) & (r - b > 15)

    return relative_dark & absolute_dark & ~reddish_grid_color


def _grid_color_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that look like an ECG grid line: distinctly colored (unlike
    the near-gray/near-white background) but not dark (unlike the ink
    trace). Used to self-calibrate grid spacing independently of whatever
    periodicity the trace itself happens to have -- using raw intensity
    gradients for this (an earlier approach) picks up QRS-complex/T-wave
    periodicity as a false grid signal, since those are periodic too.
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    saturation = (mx - mn) / np.clip(mx, 1.0, None)
    return (saturation > 0.12) & (mx > 120)


def _estimate_grid_spacing_px(grid_mask: np.ndarray, axis: int, min_spacing: float = 3.0, max_spacing: float = 80.0) -> float | None:
    """Self-calibrates ECG paper grid spacing (pixels per small 1mm box)
    from the dominant periodicity of `grid_mask` (see `_grid_color_mask`)
    along `axis`.
    axis=1 -> vertical grid lines -> horizontal (x) spacing (profile
    collapses rows, i.e. averaged over axis=0).
    axis=0 -> horizontal grid lines -> vertical (y) spacing (profile
    collapses columns, i.e. averaged over axis=1).
    """
    profile = grid_mask.mean(axis=1 - axis).astype(np.float64)
    if len(profile) < 16:
        return None
    profile = profile - profile.mean()
    windowed = profile * np.hanning(len(profile))
    power = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(profile))
    valid = (freqs > 1.0 / max_spacing) & (freqs < 1.0 / min_spacing)
    if not valid.any():
        return None
    masked_power = np.where(valid, power, -1.0)
    idx = int(np.argmax(masked_power))
    if masked_power[idx] <= 0:
        return None
    freq = freqs[idx]
    return float(1.0 / freq) if freq > 0 else None


def _panel_bboxes(shape: tuple[int, int], layout: list[list[str]]) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    rows, cols = len(layout), len(layout[0])
    row_edges = np.linspace(0, h, rows + 1).astype(int)
    col_edges = np.linspace(0, w, cols + 1).astype(int)
    bboxes = {}
    for r in range(rows):
        for c in range(cols):
            lead = layout[r][c]
            bboxes[lead] = (int(row_edges[r]), int(row_edges[r + 1]), int(col_edges[c]), int(col_edges[c + 1]))
    return bboxes


def _interpolate_nan(y: np.ndarray) -> np.ndarray:
    n = len(y)
    valid = ~np.isnan(y)
    if valid.sum() == 0:
        return np.zeros(n, dtype=np.float32)
    if valid.sum() == n:
        return y
    idx = np.arange(n)
    y_filled = np.interp(idx, idx[valid], y[valid])
    return y_filled.astype(np.float32)


def _resample_1d(y: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if orig_fs == target_fs or len(y) < 2:
        return y
    duration = len(y) / orig_fs
    orig_t = np.linspace(0, duration, len(y), endpoint=False)
    n_target = max(2, int(round(duration * target_fs)))
    target_t = np.linspace(0, duration, n_target, endpoint=False)
    return np.interp(target_t, orig_t, y).astype(np.float32)


def _extract_panel_trace(ink_mask: np.ndarray) -> np.ndarray:
    """Per-column mean ink row (pixels), NaN for columns with no ink."""
    h, w = ink_mask.shape
    rows = np.arange(h).reshape(-1, 1)
    weighted = np.where(ink_mask, rows, 0).sum(axis=0)
    counts = ink_mask.sum(axis=0)
    trace = np.full(w, np.nan, dtype=np.float32)
    has_ink = counts > 0
    trace[has_ink] = weighted[has_ink] / counts[has_ink]
    return trace


def digitize_ecg_image(
    image: bytes | np.ndarray | Image.Image,
    crop_box: tuple[float, float, float, float] | None = None,
    layout: list[list[str]] | None = None,
    target_fs: float = 500.0,
    paper_speed_mm_s: float = DEFAULT_PAPER_SPEED_MM_S,
    paper_mv_mm: float = DEFAULT_PAPER_MV_MM,
    manual_px_per_mm: tuple[float, float] | None = None,
) -> DigitizationResult:
    """Digitizes a printed/photographed 12-lead ECG image.

    `crop_box`: optional (x0, y0, x1, y1) in normalized [0, 1] image
    fractions, selecting just the lead-panel grid (excluding headers,
    patient info, margins). Strongly recommended for real photos --
    without it, the whole image is assumed to be the panel grid.

    `manual_px_per_mm`: (px_per_mm_x, px_per_mm_y) to force calibration
    instead of auto-detecting the paper grid spacing -- use this if grid
    autodetection fails or the image has no visible grid.
    """
    layout = layout or STANDARD_3X4_LAYOUT
    warnings: list[str] = []

    img = _load_image_array(image)
    img = _resize_max_dim(img)
    img = _crop_normalized(img, crop_box)
    if crop_box is None:
        warnings.append(
            "No crop region was provided; the full image was assumed to be "
            "the lead-panel grid. Crop out headers/margins for better results."
        )

    gray = _to_grayscale(img)
    ink = _ink_mask(gray, img)

    if manual_px_per_mm is not None:
        px_per_mm_x, px_per_mm_y = manual_px_per_mm
        calibration_method = "manual"
    else:
        grid_mask = _grid_color_mask(img)
        px_per_mm_x = _estimate_grid_spacing_px(grid_mask, axis=1)
        px_per_mm_y = _estimate_grid_spacing_px(grid_mask, axis=0)
        if px_per_mm_x and px_per_mm_y:
            calibration_method = "grid_autodetect"
        else:
            calibration_method = "fallback_autoscale"
            warnings.append(
                "Could not auto-detect the ECG paper grid spacing; falling "
                "back to amplitude auto-scaling. Amplitudes (and therefore "
                "voltage-dependent findings like hypertrophy/low-voltage) "
                "are less reliable for this image."
            )

    bboxes = _panel_bboxes(img.shape[:2], layout)
    panel_h = img.shape[0] // len(layout)
    panel_w = img.shape[1] // len(layout[0])

    raw_traces: dict[str, np.ndarray] = {}
    for lead, (y0, y1, x0, x1) in bboxes.items():
        panel_ink = ink[y0:y1, x0:x1]
        trace_px = _extract_panel_trace(panel_ink)
        trace_px = _interpolate_nan(trace_px)
        raw_traces[lead] = trace_px

    if calibration_method in ("grid_autodetect", "manual"):
        px_per_s = px_per_mm_x * paper_speed_mm_s
        mv_per_px = 1.0 / (px_per_mm_y * paper_mv_mm)
    else:
        px_per_s = panel_w / 2.5  # assume ~2.5s per panel, the common default
        mv_per_px = None  # determined per-lead below via autoscale

    ordered_leads = [l for row in layout for l in row]
    all_lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signals: list[np.ndarray] = []

    for lead in all_lead_names:
        trace_px = raw_traces[lead]
        baseline_px = panel_h / 2.0  # assume the panel's vertical center is the isoelectric baseline
        amplitude_px = baseline_px - trace_px  # image y grows downward; ECG amplitude grows upward

        if mv_per_px is not None:
            amplitude_mv = amplitude_px * mv_per_px
        else:
            spread = np.percentile(amplitude_px, 95) - np.percentile(amplitude_px, 5)
            spread = max(spread, 1e-3)
            amplitude_mv = amplitude_px * (2.0 / spread)  # autoscale to a ~2 mV-ish range

        resampled = _resample_1d(amplitude_mv, orig_fs=px_per_s, target_fs=target_fs)
        signals.append(resampled)

    min_len = min(len(s) for s in signals)
    if min_len < int(0.5 * target_fs):
        warnings.append(
            f"Digitized signal is very short ({min_len / target_fs:.2f}s per lead); "
            "results will be unreliable. Check the crop region and image quality."
        )
    ecg = np.stack([s[:min_len] for s in signals], axis=0).astype(np.float32)

    return DigitizationResult(
        ecg=ecg,
        fs=target_fs,
        panel_traces_px=raw_traces,
        panel_bboxes=bboxes,
        calibration_method=calibration_method,
        px_per_mm=(px_per_mm_x, px_per_mm_y) if calibration_method != "fallback_autoscale" else None,
        warnings=warnings,
        working_image=img,
    )
