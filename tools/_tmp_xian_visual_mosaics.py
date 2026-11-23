from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


REMOTE_ROOT = Path("/data/xian_r4_100km2_20260914/output/13_visual_asset_repair_and_obj0008_identity_closure_20260916")
OUT = REMOTE_ROOT / "visual_qa_mosaics"


def make_mosaics(scope: str, pattern: str, chunk_size: int = 25) -> list[dict[str, object]]:
    folders = sorted(Path(REMOTE_ROOT / scope).glob(pattern))
    folders = [p for p in folders if (p / "contact_sheet_v2.png").is_file()]
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    rows = []
    for chunk_no in range(0, len(folders), chunk_size):
        chunk = folders[chunk_no:chunk_no + chunk_size]
        cells = []
        for folder in chunk:
            img = Image.open(folder / "contact_sheet_v2.png").convert("RGB")
            img.thumbnail((300, 210))
            cell = Image.new("RGB", (320, 245), "white")
            x = (320 - img.width) // 2
            cell.paste(img, (x, 25))
            draw = ImageDraw.Draw(cell)
            draw.text((6, 5), folder.name, fill="black", font=font)
            cells.append(cell)
        cols = 5
        rows_count = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * 320, rows_count * 245), "#dddddd")
        for idx, cell in enumerate(cells):
            canvas.paste(cell, ((idx % cols) * 320, (idx // cols) * 245))
        name = f"{scope.replace('/', '_')}_{chunk_no // chunk_size + 1:02d}.png"
        path = OUT / name
        canvas.save(path, "PNG", optimize=True)
        rows.append({"scope": scope, "mosaic": name, "count": len(chunk), "first": chunk[0].name, "last": chunk[-1].name})
    return rows


def main() -> None:
    rows = []
    rows += make_mosaics("repaired/strict_core", "SAMPLE_*")
    rows += make_mosaics("repaired/held_review", "SAMPLE_*")
    rows += make_mosaics("aba_visual_qa", "ABA_QA_*")
    with (OUT / "MOSAIC_INDEX.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scope", "mosaic", "count", "first", "last"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"mosaics={len(rows)} samples={sum(int(r['count']) for r in rows)}")


if __name__ == "__main__":
    main()
