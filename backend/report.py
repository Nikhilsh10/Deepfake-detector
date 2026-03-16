"""
PDF report generator for the Deepfake Detection System.

Creates a single‑page PDF summarising an analysis result — including
the verdict, confidence, frame statistics, and a heatmap thumbnail.
Uses only the Python standard‑library + Pillow (no extra PDF libs
required) by building the PDF manually with a minimal writer.
"""

import io
import os
from datetime import datetime, timezone
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font; fall back to the default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def generate_pdf_report(
    result: dict[str, Any],
    filename: str,
    output_path: str,
    heatmap_path: Optional[str] = None,
) -> str:
    """Generate a PNG‑based analysis report (works everywhere without
    extra PDF libraries).

    Parameters
    ----------
    result : dict
        The full detection result dictionary.
    filename : str
        Original uploaded filename.
    output_path : str
        Where to save the report image.
    heatmap_path : str | None
        Optional path to a heatmap image to embed in the report.

    Returns
    -------
    str
        Path to the saved report image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Canvas ─────────────────────────────────────────────────────────────────
    width, height = 800, 1000
    bg_color = (15, 23, 42)  # navy-900
    text_color = (226, 232, 240)
    accent_red = (239, 68, 68)
    accent_green = (34, 197, 94)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts ──────────────────────────────────────────────────────────────────
    font_title = _load_font(28)
    font_heading = _load_font(20)
    font_body = _load_font(15)
    font_small = _load_font(12)

    y = 30
    is_fake = result.get("prediction") == "FAKE"
    verdict_color = accent_red if is_fake else accent_green

    # Title ──────────────────────────────────────────────────────────────────
    draw.text((30, y), "DeepGuard — Analysis Report", fill=(129, 140, 248), font=font_title)
    y += 45
    draw.line([(30, y), (width - 30, y)], fill=(51, 65, 85), width=1)
    y += 20

    # File info ──────────────────────────────────────────────────────────────
    draw.text((30, y), f"File: {filename}", fill=text_color, font=font_body)
    y += 25
    draw.text(
        (30, y),
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        fill=(148, 163, 184),
        font=font_small,
    )
    y += 35

    # Verdict ────────────────────────────────────────────────────────────────
    verdict_text = "🔴 DEEPFAKE DETECTED" if is_fake else "🟢 AUTHENTIC"
    draw.rounded_rectangle(
        [(30, y), (width - 30, y + 60)],
        radius=12,
        fill=(verdict_color[0] // 5, verdict_color[1] // 5, verdict_color[2] // 5),
        outline=verdict_color,
    )
    draw.text((60, y + 15), verdict_text, fill=verdict_color, font=font_heading)
    y += 80

    # Confidence & probabilities ─────────────────────────────────────────────
    conf = result.get("confidence", 0)
    fake_p = result.get("fake_probability", 0)
    real_p = result.get("real_probability", 0)

    draw.text((30, y), "Confidence", fill=text_color, font=font_heading)
    y += 30
    draw.text((50, y), f"{conf * 100:.1f}%", fill=verdict_color, font=font_title)
    y += 40

    draw.text((30, y), f"Fake probability:  {fake_p * 100:.1f}%", fill=accent_red, font=font_body)
    y += 25
    draw.text((30, y), f"Real probability:  {real_p * 100:.1f}%", fill=accent_green, font=font_body)
    y += 35

    # Frame stats (video) ───────────────────────────────────────────────────
    if result.get("total_frames_analyzed") is not None:
        draw.line([(30, y), (width - 30, y)], fill=(51, 65, 85), width=1)
        y += 15
        draw.text((30, y), "Frame Analysis", fill=text_color, font=font_heading)
        y += 30
        draw.text(
            (50, y),
            f"Total frames : {result['total_frames_analyzed']}",
            fill=text_color,
            font=font_body,
        )
        y += 25
        draw.text(
            (50, y),
            f"Fake frames  : {result.get('fake_frames', 0)}",
            fill=accent_red,
            font=font_body,
        )
        y += 25
        draw.text(
            (50, y),
            f"Real frames  : {result.get('real_frames', 0)}",
            fill=accent_green,
            font=font_body,
        )
        y += 25
        draw.text(
            (50, y),
            f"Processing   : {result.get('processing_time_seconds', 0):.2f}s",
            fill=(148, 163, 184),
            font=font_body,
        )
        y += 35

    # Heatmap thumbnail ─────────────────────────────────────────────────────
    if heatmap_path and os.path.isfile(heatmap_path):
        draw.line([(30, y), (width - 30, y)], fill=(51, 65, 85), width=1)
        y += 15
        draw.text((30, y), "Grad-CAM Heatmap", fill=text_color, font=font_heading)
        y += 30
        try:
            hm = Image.open(heatmap_path).convert("RGB")
            hm.thumbnail((350, 250))
            img.paste(hm, (30, y))
            y += hm.height + 20
        except Exception:
            pass

    # Footer ─────────────────────────────────────────────────────────────────
    draw.text(
        (30, height - 40),
        "Generated by DeepGuard — Real-Time Deepfake Detection System",
        fill=(71, 85, 105),
        font=font_small,
    )

    # Crop to content height + margin ───────────────────────────────────────
    final_height = min(y + 60, height)
    img = img.crop((0, 0, width, final_height))
    img.save(output_path, quality=95)
    return output_path
