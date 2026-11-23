#!/usr/bin/env python3
"""Render the non-result R5 Stage-1 architecture figure."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\14_figures\FIG_R5_STAGE1_ARCHITECTURE.png")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def wrapped(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], size: int = 24, fill: str = "#172033", bold: bool = False) -> None:
    x0, y0, x1, y1 = box
    fnt = font(size, bold)
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= x1 - x0 - 24:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = y0
    for line in lines:
        draw.text((x0 + 12, y), line, fill=fill, font=fnt)
        y += size + 8
        if y > y1 - size:
            break


def box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body: str, fill: str, title_fill: str = "#10253f") -> None:
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline="#30445e", width=3)
    x0, y0, x1, y1 = xy
    draw.text((x0 + 18, y0 + 16), title, fill=title_fill, font=font(28, True))
    wrapped(draw, body, (x0, y0 + 60, x1, y1), size=21)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#38516c") -> None:
    draw.line((*start, *end), fill=fill, width=7)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        tip = [(x1, y1), (x1 - 18 if x1 > x0 else x1 + 18, y1 - 12), (x1 - 18 if x1 > x0 else x1 + 18, y1 + 12)]
    else:
        tip = [(x1, y1), (x1 - 12, y1 - 18 if y1 > y0 else y1 + 18), (x1 + 12, y1 - 18 if y1 > y0 else y1 + 18)]
    draw.polygon(tip, fill=fill)


def main() -> None:
    image = Image.new("RGB", (1900, 1100), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "Xi'an R5 Stage-1 candidate architecture", fill="#10253f", font=font(44, True))
    draw.text((72, 98), "Identity-first off-nadir / merge-split upgrade; diagnostic design only, no production promotion", fill="#52647a", font=font(25))

    box(draw, (70, 180, 460, 395), "1  Source and date gate", "Read-only F:\\xian\\xian plus supplemental registry. Actual acquisition dates are retained; mixed scenes cannot become clean epochs.", "#e8f1fb")
    box(draw, (70, 490, 460, 705), "2  Frozen geometry", "B0/R4 formal geometry, threshold, tracking receipts, TIFs, and holdout boundary remain unchanged. Candidate outputs stay separate.", "#f1eee4")
    arrow(draw, (460, 286), (565, 286))
    arrow(draw, (460, 598), (565, 598))

    box(draw, (565, 180, 980, 395), "3  Ground registration evidence", "SIFT baseline now; LightGlue / ALIKED optional in an isolated environment. Compare translation, similarity, affine, and homography; retain stable-ground and parallax strata separately.", "#e5f5ef")
    box(draw, (565, 490, 980, 705), "4  Prompt ambiguity evidence", "RBF-SAM building / roof / footprint / noisy prompt contract and offset branches. Resolver-only: B0 remains the geometry owner.", "#f8e9ed")
    arrow(draw, (980, 286), (1085, 286))
    arrow(draw, (980, 598), (1085, 598))

    box(draw, (1085, 180, 1515, 395), "5  TBIA track query", "Masked semantic, boundary, shape, offset, neighborhood, and temporal groups. Uncertainty-aware radius, hard negatives, and explicit merge/split relations.", "#ede8fb")
    box(draw, (1085, 490, 1515, 705), "6  Temporal consistency", "Three-epoch cycle evidence, perturbation stress tests, and an unresolved-preserving cascade. No mask XOR to state conversion.", "#eaf1e7")
    arrow(draw, (1515, 286), (1600, 470))
    arrow(draw, (1515, 598), (1600, 470))

    box(draw, (1600, 300, 1830, 680), "7  Gate", "Fixed cases + random development A/B + false-change review. Scientific state may remain UNRESOLVED; operational forcing is flagged.", "#fff1d7")

    draw.line((80, 835, 1820, 835), fill="#a8b5c4", width=3)
    draw.text((80, 865), "Stage-1 status: alignment diagnostic recorded | RBF-SAM checkpoint absent | TBIA untrained prototype | fixed-case gate NOT RUN", fill="#8b2f39", font=font(25, True))
    draw.text((80, 925), "Promotion rule: no citywide or production output until identity-grounded acceptance evidence is complete.", fill="#30445e", font=font(27, True))
    draw.text((80, 990), "Figure is a methods architecture, not an accuracy or coverage claim.", fill="#52647a", font=font(23))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
