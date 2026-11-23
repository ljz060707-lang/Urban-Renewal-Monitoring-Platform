from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thumb-width", type=int, default=252)
    args = ap.parse_args()

    dirs = []
    for category_dir in [args.root / "state_samples", args.root / "held_2019_2020"]:
        if category_dir.exists():
            dirs.extend(sorted(p for p in category_dir.iterdir() if p.is_dir()))
    dirs.sort(key=lambda p: p.name)
    args.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w = args.thumb_width
    tile_h = round(tile_w * 718 / 1008)
    label_h = 22
    per = args.cols * args.rows
    for batch_no, start in enumerate(range(0, len(dirs), per), 1):
        batch = dirs[start : start + per]
        canvas = Image.new("RGB", (args.cols * tile_w, args.rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, sample_dir in enumerate(batch):
            p = sample_dir / "contact_sheet.png"
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
                x = (i % args.cols) * tile_w
                y = (i // args.cols) * (tile_h + label_h)
                canvas.paste(im, (x, y))
                draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
                draw.text((x + 3, y + tile_h + 3), sample_dir.name, fill="black", font=font)
        canvas.save(args.out / f"montage_{batch_no:03d}.png", optimize=True)

    print({"samples": len(dirs), "montages": (len(dirs) + per - 1) // per, "out": str(args.out)})


if __name__ == "__main__":
    main()
