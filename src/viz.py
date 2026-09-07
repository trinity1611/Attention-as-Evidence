"""Heatmap rendering for the app and the hero: one hue per finding.

Instead of a jet colormap (which reads as a thermal scale and cannot be
overlaid), each finding's CAM becomes a transparent-to-colour alpha ramp in that
finding's hue, plus a contour at 50 % of peak -- the same region the IoU metric
scores. Several findings can be composited on one film and still be told apart.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.palette import FILM_TINTS, colour, hex_to_rgb, pretty

CONTOUR_THR = 0.5


# ------------------------------------------------------------- film base ---
def tint_film(img_gray: Image.Image, theme: str | None) -> Image.Image:
    """Grayscale film -> RGB, optionally duotone-tinted (sepia / cyan / amber)."""
    g = np.asarray(img_gray.convert("L"), dtype=np.float32) / 255.0
    spec = FILM_TINTS.get(theme or "grayscale")
    if spec is None:
        rgb = np.stack([g, g, g], -1)
    else:
        lo, hi = np.array(spec[0]), np.array(spec[1])
        rgb = lo[None, None, :] + g[..., None] * (hi - lo)[None, None, :]
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


# -------------------------------------------------------------- overlays ---
def alpha_layer(cam: np.ndarray, hex_colour: str, alpha_max: float = 0.72,
                gamma: float = 1.6, floor: float = 0.15) -> Image.Image:
    """RGBA layer: colour everywhere, alpha rising with CAM intensity.

    Values below `floor` (as a fraction of peak) are fully transparent so the
    film is not washed out; `gamma` > 1 keeps the ramp concentrated on the peak.
    """
    c = np.nan_to_num(cam).astype(np.float32)
    c = c / c.max() if c.max() > 0 else c
    a = np.clip((c - floor) / (1 - floor), 0, 1) ** gamma * alpha_max
    r, g, b = hex_to_rgb(hex_colour)
    rgba = np.zeros((*c.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def contour_layer(cam: np.ndarray, hex_colour: str, thr: float = CONTOUR_THR,
                  width: int = 4) -> Image.Image:
    """Outline of the region >= thr * peak, drawn in the finding's colour."""
    c = np.nan_to_num(cam)
    mask = Image.fromarray(((c >= thr * c.max()) * 255).astype(np.uint8), "L") \
        if c.max() > 0 else Image.new("L", c.shape[::-1], 0)
    outer = mask.filter(ImageFilter.MaxFilter(width * 2 + 1))
    inner = mask.filter(ImageFilter.MinFilter(width * 2 + 1))
    edge = np.asarray(outer, dtype=np.int16) - np.asarray(inner, dtype=np.int16)
    r, g, b = hex_to_rgb(hex_colour)
    rgba = np.zeros((*edge.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
    rgba[..., 3] = np.where(edge > 0, 235, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def compose(img_gray: Image.Image, layers: Iterable[tuple[np.ndarray, str]],
            theme: str | None = None, alpha_max: float = 0.72, contours: bool = True,
            peak_marks: bool = True, box=None, box_colour: str = "#00E28A") -> Image.Image:
    """Film + any number of (cam, finding) layers -> RGB image at film size.

    `box` is an optional (x, y, w, h) radiologist box in image coordinates.
    """
    base = tint_film(img_gray, theme).convert("RGBA")
    W, H = base.size
    for cam, finding in layers:
        cam_r = _resize_cam(cam, (W, H))
        base = Image.alpha_composite(base, alpha_layer(cam_r, colour(finding), alpha_max))
        if contours:
            base = Image.alpha_composite(base, contour_layer(cam_r, colour(finding)))
    draw = ImageDraw.Draw(base)
    if peak_marks:
        for cam, finding in layers:
            cam_r = _resize_cam(cam, (W, H))
            py, px = np.unravel_index(int(np.argmax(cam_r)), cam_r.shape)
            s = max(6, W // 90)
            for dx, dy in ((-s, -s), (s, -s)):
                draw.line([(px + dx, py + dy), (px - dx, py - dy)], fill="white", width=max(2, W // 300))
    if box is not None:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline=box_colour, width=max(3, W // 250))
    return base.convert("RGB")


def _resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cam.shape[::-1] == size:
        return cam
    im = Image.fromarray((np.clip(np.nan_to_num(cam), 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


# --------------------------------------------------------------- legend ---
def legend_chip(finding: str, prob: float | None = None) -> str:
    """Inline HTML chip for Streamlit / the hero."""
    c = colour(finding)
    txt = pretty(finding) + (f" · {prob:.2f}" if prob is not None else "")
    return (f'<span style="display:inline-flex;align-items:center;gap:.4em;'
            f'padding:.18em .6em;border-radius:999px;background:{c}22;'
            f'border:1.5px solid {c};color:#1C2430;font-size:.85rem;">'
            f'<span style="width:.7em;height:.7em;border-radius:50%;background:{c};"></span>{txt}</span>')


def to_png_bytes(img: Image.Image, quality_downscale: int | None = None) -> bytes:
    if quality_downscale and max(img.size) > quality_downscale:
        img = img.copy()
        img.thumbnail((quality_downscale, quality_downscale))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def probability_bar_png(probs: np.ndarray, flagged: list[str], classes: list[str],
                        width_px: int = 900) -> bytes:
    """Static bar chart (for the PDF); the app uses Plotly for the live one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(probs)
    fig, ax = plt.subplots(figsize=(width_px / 100, 4.6), dpi=100)
    names = [pretty(classes[i]) for i in order]
    cols = [colour(classes[i]) for i in order]
    bars = ax.barh(names, probs[order], color=cols)
    for b, i in zip(bars, order):
        ax.text(probs[i] + 0.01, b.get_y() + b.get_height() / 2,
                f"{probs[i]:.2f}" + ("  *" if classes[i] in flagged else ""),
                va="center", fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("classifier score (uncalibrated; * = above threshold)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
