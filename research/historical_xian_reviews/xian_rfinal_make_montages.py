from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dirs = sorted([p for p in (args.root / "samples").iterdir() if p.is_dir()])
    args.out.mkdir(parents=True, exist_ok=True)
    cols, rows, tile_w = 5, 4, 252
    tile_h, label_h = round(tile_w * 718 / 1008), 22
    font = ImageFont.load_default()
    per = cols * rows
    for no, start in enumerate(range(0, len(dirs), per), 1):
        batch = dirs[start : start + per]
        canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, d in enumerate(batch):
            with Image.open(d / "contact_sheet.png") as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
                x = (i % cols) * tile_w
                y = (i // cols) * (tile_h + label_h)
                canvas.paste(im, (x, y))
                draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
                draw.text((x + 3, y + tile_h + 3), d.name, fill="black", font=font)
        canvas.save(args.out / f"montage_{no:03d}.png", optimize=True)
    print({"samples": len(dirs), "montages": (len(dirs) + per - 1) // per, "out": str(args.out)})


if __name__ == "__main__":
    main()
