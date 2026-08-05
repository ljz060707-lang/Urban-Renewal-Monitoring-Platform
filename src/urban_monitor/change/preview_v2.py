from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def _panel(image: Image.Image | None, title: str, pending: str | None = None, size=(420, 300)) -> Image.Image:
    if image is None:
        image = Image.new("RGB", size, (34, 39, 46))
        draw = ImageDraw.Draw(image)
        draw.text((20, size[1] // 2), f"PENDING\n{pending or title}", fill=(220, 225, 230))
    else:
        image = ImageOps.fit(image.convert("RGB"), size)
    canvas = Image.new("RGB", (size[0], size[1] + 38), "white")
    canvas.paste(image, (0, 38))
    ImageDraw.Draw(canvas).text((12, 12), title, fill=(20, 24, 31))
    return canvas


def _probability_image(array: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    values = np.clip(np.asarray(array, dtype=np.float32), 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        rgb[..., channel] = (values * value).astype(np.uint8)
    return Image.fromarray(rgb)


def build_preview(
    output_dir: str | Path,
    *,
    t1: Image.Image,
    t2: Image.Image,
    outputs: dict[str, np.ndarray],
    pending: dict[str, str],
    teacher_status: dict,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        ("T1 / Earlier", t1, None),
        ("T2 / Later", t2, None),
        ("Any change", _probability_image(outputs["any_change_probability"], (30, 120, 255)) if "any_change_probability" in outputs else None, pending.get("any_change_probability")),
        ("New building", _probability_image(outputs["new_building_candidate"], (40, 200, 80)) if "new_building_candidate" in outputs else None, pending.get("new_building_candidate")),
        ("Demolition", _probability_image(outputs["demolition_candidate"], (230, 60, 55)) if "demolition_candidate" in outputs else None, pending.get("demolition_candidate")),
        ("Semantic transition", None, pending.get("semantic_transition", "SECOND")),
        ("Construction candidate", None, pending.get("construction_candidate", "SECOND")),
        ("Building geometry", None, pending.get("building_geometry", "RSBuilding")),
        ("Final fusion", _probability_image(outputs["any_change_probability"], (255, 180, 40)) if "any_change_probability" in outputs else None, pending.get("any_change_probability")),
    ]
    panels = [_panel(image, title, note) for title, image, note in panel_specs]
    width, height = panels[0].size
    contact = Image.new("RGB", (width * 3, height * 3 + 54), (245, 246, 248))
    ImageDraw.Draw(contact).text((16, 18), "Urban Renewal Detection – First Round Partial Result", fill=(15, 22, 32))
    for index, panel in enumerate(panels):
        contact.paste(panel, ((index % 3) * width, 54 + (index // 3) * height))
    preview_path = root / "preview.png"
    contact.save(preview_path)

    status_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value.get('status', 'unknown')).upper())}</td>"
        f"<td>{html.escape(str(value.get('role', '')))}</td></tr>"
        for name, value in teacher_status.items()
    )
    pending_items = "".join(f"<li><b>{html.escape(name)}</b>: {html.escape(reason)}</li>" for name, reason in pending.items())
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Urban Renewal V2 Partial Result</title>
<style>body{{font:16px system-ui;margin:32px;background:#f5f6f8;color:#14181f}}
main{{max-width:1320px;margin:auto}}img{{max-width:100%;background:white;border-radius:10px}}
table{{border-collapse:collapse;background:white}}td,th{{padding:10px 16px;border:1px solid #ddd}}
.pending{{background:#fff5d6;padding:12px 20px;border-radius:8px}}</style>
<main><h1>Urban Renewal Detection – First Round Partial Result</h1>
<h2>Teacher status</h2><table><tr><th>Teacher</th><th>Status</th><th>Role</th></tr>{status_rows}</table>
<h2>Preview</h2><img src="preview.png" alt="V2 preview">
<h2>Pending outputs</h2><div class="pending"><ul>{pending_items or '<li>None</li>'}</ul></div></main></html>"""
    index_path = root / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return preview_path, index_path

