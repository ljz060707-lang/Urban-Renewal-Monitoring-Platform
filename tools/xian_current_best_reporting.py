"""Build the non-destructive reporting layer for the current Xi'an candidate.

This script does not alter Geometry, IdentityGraph, Presence, or State. It
only copies already reviewed visual assets and writes a clearly labelled
candidate report plus a static process figure.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916")
OUT = Path(r"F:\xian\西安_当前最佳候选交付_20260917")
CONTACTS = ROOT / "12_视觉QA" / "contact_sheets"
RECEIPT = ROOT / "14_最终报告与回执" / "FINAL_CANDIDATE_RECEIPT.json"
ALIGN = ROOT / "04_角度偏移与对齐" / "ALIGNMENT_DIAGNOSTICS.csv"


CASES = [
    {
        "display_id": "CASE_OBJ_0001",
        "category": "addition_candidate",
        "final_state": "ADDITION",
        "confidence": "0.85",
        "visual_verdict": "CORRECT",
        "note": "Clear 0-0-1 sequence: the target complex is absent in 2018/2019 and visibly built in 2020.",
    },
    {
        "display_id": "CASE_B01-P2-26",
        "category": "demolition_candidate",
        "final_state": "DEMOLITION_CANDIDATE",
        "confidence": "0.85",
        "visual_verdict": "CORRECT",
        "note": "The target footprint is visible in 2018/2019 and the 2020 scene is a cleared, non-building surface.",
    },
    {
        "display_id": "CASE_OBJ_0008",
        "category": "stable_candidate",
        "final_state": "STABLE_CANDIDATE",
        "confidence": "0.65",
        "visual_verdict": "PLAUSIBLE",
        "note": "The same apartment/complex footprint persists with viewpoint displacement; dense surroundings lower certainty.",
    },
    {
        "display_id": "CASE_B01-TEX02",
        "category": "stable_control",
        "final_state": "STABLE_CONTROL",
        "confidence": "0.65",
        "visual_verdict": "PLAUSIBLE",
        "note": "A persistent structure is visible at the same location; retained as a control because semantic interpretation is conservative.",
    },
    {
        "display_id": "SAMPLE_split_003",
        "category": "split_diagnostic",
        "final_state": "STABLE_DIAGNOSTIC",
        "confidence": "0.80",
        "visual_verdict": "CORRECT",
        "note": "A coherent building group remains spatially consistent while the instance diagnostic is split.",
    },
    {
        "display_id": "SAMPLE_merge_008",
        "category": "merge_diagnostic",
        "final_state": "STABLE_DIAGNOSTIC",
        "confidence": "0.85",
        "visual_verdict": "CORRECT",
        "note": "A compound footprint remains spatially coherent while the instance diagnostic is merge.",
    },
    {
        "display_id": "SAMPLE_split_merge_010",
        "category": "split_merge_diagnostic",
        "final_state": "STABLE_DIAGNOSTIC",
        "confidence": "0.60",
        "visual_verdict": "PLAUSIBLE",
        "note": "The elongated building chain is trackable across views, but the dense neighborhood remains a diagnostic case.",
    },
]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUT,
        "cases": OUT / "03_typical_tracking_cases",
        "figures": OUT / "04_process_figures",
        "stats": OUT / "02_area_statistics",
        "qa": OUT / "05_visual_qa_reference",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_case_selection(dirs: dict[str, Path]) -> None:
    out_csv = dirs["cases"] / "TYPICAL_CASE_SELECTION.csv"
    fields = [
        "display_id",
        "category",
        "final_state",
        "confidence",
        "visual_verdict",
        "source_contact_sheet",
        "copied_contact_sheet",
        "note",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in CASES:
            src = CONTACTS / f"{item['display_id']}.png"
            dst = dirs["cases"] / f"{item['display_id']}.png"
            copied = src.exists()
            if copied:
                shutil.copy2(src, dst)
            row = dict(item)
            row["source_contact_sheet"] = str(src)
            row["copied_contact_sheet"] = str(dst) if copied else "MISSING"
            writer.writerow(row)

    (dirs["cases"] / "README.md").write_text(
        "# Typical tracking cases\n\n"
        "These are visual-review candidates copied from the frozen candidate QA package. "
        "They are illustrative evidence, not a replacement for the formal uncertainty layer.\n\n"
        "Recommended first choices: 'CASE_OBJ_0001' for addition, 'CASE_B01-P2-26' "
        "for demolition, and 'CASE_OBJ_0008' or 'CASE_B01-TEX02' for stable/control.\n",
        encoding="utf-8",
    )


def read_interval_stats() -> list[dict[str, str]]:
    data = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for interval, info in data.get("interval_stats", {}).items():
        total = float(info.get("total_track_deduplicated_union_area_m2", 0.0))
        cats = info.get("area_m2", {})
        counts = info.get("counts", {})
        accounted = sum(float(v) for v in cats.values())
        for state in ["STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"]:
            area = float(cats.get(state, 0.0))
            rows.append(
                {
                    "interval": interval,
                    "state": state,
                    "object_count": str(counts.get(state, 0)),
                    "area_m2": f"{area:.6f}",
                    "area_km2": f"{area / 1_000_000:.6f}",
                    "share_of_track_union_pct": f"{(100 * area / total) if total else 0:.6f}",
                    "total_track_union_area_m2": f"{total:.6f}",
                    "category_area_sum_m2": f"{accounted:.6f}",
                    "denominator_gap_m2": f"{total - accounted:.6f}",
                    "scientific_status": str(data.get("scientific_status", "CANDIDATE_NOT_GROUND_TRUTH")),
                }
            )
    return rows


def write_statistics(dirs: dict[str, Path]) -> None:
    rows = read_interval_stats()
    fields = list(rows[0]) if rows else []
    with (dirs["stats"] / "CURRENT_BEST_INTERVAL_STATS.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    data = json.loads(RECEIPT.read_text(encoding="utf-8"))
    lines = [
        "# Xi'an current-best candidate area statistics",
        "",
        "This is a candidate report from the latest IdentityGraph rebuild. It is not a ground-truth certificate.",
        "Formal Unknown/Conflict is retained and shown explicitly; it is not silently added to Stable.",
        "",
        "| Interval | State | Count | Area (km²) | Share of track-union denominator |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['interval']} | {r['state']} | {r['object_count']} | {r['area_km2']} | {r['share_of_track_union_pct']}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The denominator is the deduplicated union area of physical tracks. The category area sum is reported separately because the current candidate contains unresolved identity/presence rows and its category areas do not exhaust the denominator.",
        "The current candidate receipt reports Unknown/Conflict at approximately 27.01% for 2018-2019 and 29.05% for 2019-2020. Therefore these numbers are suitable for review and method comparison, not for claiming an Unknown<5% final product.",
        "",
        f"Candidate gate status: {data.get('TRACKING_ACCURACY_GATE', 'REVIEW_REQUIRED')}; "
        f"Unknown gate: {data.get('UNKNOWN_LT5_GATE', 'FAIL')}; "
        f"scientific status: {data.get('scientific_status', 'CANDIDATE_NOT_GROUND_TRUTH')}.",
        "",
    ]
    (dirs["stats"] / "CURRENT_BEST_INTERVAL_STATS.md").write_text("\n".join(lines), encoding="utf-8")


def write_process_notes(dirs: dict[str, Path]) -> None:
    (dirs["root"] / "PROCESS_AND_PARAMETERS.md").write_text(
        "# Xi'an current-best candidate: plain-language method note\n\n"
        "## What this package is\n\n"
        "This is the current best candidate export from the available three-date main imagery. "
        "It is auditable and reviewable, but the latest receipt still marks it "
        "CANDIDATE_NOT_GROUND_TRUTH; Unknown/Conflict is retained rather than silently forced into Stable.\n\n"
        "The supplemental folder `F:\\xian\\XIAN补充` is not allowed to replace the date-specific Geometry owner. "
        "If its observations are not yet fully registered and audited, they are not claimed as already integrated in this package.\n\n"
        "## Fixed processing order\n\n"
        "1. Register actual acquisition dates, source IDs, CRS, transforms, footprints, quality and observability.\n"
        "2. Produce independent building Geometry for 2018-04-29, 2019-01-01 and 2020-12-31.\n"
        "3. Canonicalize overlaps and seams within each actual date.\n"
        "4. Record global/local alignment and bounded parallax evidence; polygons are not warped into new footprints.\n"
        "5. Build the Cross-Date IdentityGraph using alignment-aware distance, overlap, shape, area, neighbourhood, time continuity, split/merge evidence and uncertainty.\n"
        "6. Freeze IdentityGraph, then infer Presence as 1/0/U. Unclear visibility remains U.\n"
        "7. Decode STABLE, ADDITION and DEMOLITION only from reliable Presence transitions.\n"
        "8. Apply the metric `area_m2 >= 100` filter only at the product layer, after Geometry, Tracking, Presence and State.\n\n"
        "## Raster and area parameters\n\n"
        "- Source grid: WGS84 zoom-20 raster, 154,880 x 101,120 pixels.\n"
        "- Approximate pixel size near Xi'an: 0.247 m x 0.297 m, about 0.0734 m² per pixel.\n"
        "- A 100 m² building corresponds to roughly 1,362 source-grid pixels; this is a product threshold, not a tracking rule.\n"
        "- Boundary rasters are binary masks; ID rasters store a UInt32 numeric label with a CSV lookup to the stable physical track ID.\n"
        "- Area statistics use metric EPSG:32649 vector areas as the authority; rasters are delivery/visualization layers.\n"
        "- For event masks, Stable/Addition are represented on the target-date geometry, Demolition on the source-date geometry, and Unknown/Conflict preserves both sides where applicable.\n\n"
        "## Why alignment is shown\n\n"
        "A camera angle or off-nadir shift can move roof edges even when a building has not changed. "
        "The workflow therefore records displacement and uncertainty for scoring and review, while keeping each date's polygon frozen. "
        "The process figure shows the measured 2018→2019 displacement evidence and the rule that the evidence must not be mistaken for a new building footprint.\n\n"
        "## Interpretation guardrail\n\n"
        "The current candidate receipt reports Unknown/Conflict above the requested 5% gate. "
        "Therefore the supplied counts and areas are candidate statistics for review, not a claim that all unresolved cases have been correctly classified.\n",
        encoding="utf-8",
    )


def write_process_figure(dirs: dict[str, Path]) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    src = CONTACTS / "CASE_OBJ_0001.png"
    if not src.exists():
        return
    def font(name: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        candidates = [
            Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    image = Image.open(src).convert("RGB")
    width, height = image.size
    panel_w = width // 3
    raw_top, raw_bottom = 80, min(575, height // 2)
    overlay_top, overlay_bottom = 635, height
    raw = [image.crop((i * panel_w, raw_top, (i + 1) * panel_w, raw_bottom)) for i in range(3)]
    overlay = [image.crop((i * panel_w, overlay_top, (i + 1) * panel_w, overlay_bottom)) for i in range(3)]

    canvas = Image.new("RGB", (3200, 2200), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    title_font = font("times", 48, True)
    subtitle_font = font("times", 25)
    section_font = font("times", 28, True)
    body_font = font("times", 22)
    small_font = font("times", 19)
    draw.text((150, 70), "A transparent building-change workflow under viewpoint uncertainty", fill="#17212b", font=title_font)
    draw.text((150, 140), "Illustrative case: OBJ_0001 | clean addition candidate | presence 0-0-1 | track TRK_R4X_00346166", fill="#55616b", font=subtitle_font)

    def paste_panel(panel: Image.Image, x: int, y: int, w: int, h: int, label: str) -> None:
        p = ImageOps.fit(panel, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.48))
        canvas.paste(p, (x, y))
        draw.rectangle((x, y, x + w, y + h), outline="#274c77", width=3)
        draw.text((x, y - 38), label, fill="#17212b", font=body_font)

    positions = [180, 1110, 2040]
    for i, x in enumerate(positions):
        paste_panel(raw[i], x, 245, 820, 390, ["2018 raw observation", "2019 raw observation", "2020 raw observation"][i])
    draw.text((150, 675), "Frozen Geometry owner: polygons come from each date independently; later stages do not redraw them.", fill="#55616b", font=small_font)
    for i, x in enumerate(positions):
        paste_panel(overlay[i], x, 745, 820, 390, ["2018 geometry overlay", "2019 geometry overlay", "2020 geometry overlay"][i])

    draw.rounded_rectangle((150, 1325, 870, 1635), radius=18, fill="#ffffff", outline="#274c77", width=3)
    draw.text((180, 1350), "Alignment / parallax evidence", fill="#17212b", font=section_font)
    draw.rectangle((260, 1430, 510, 1540), outline="#1f77b4", width=7)
    draw.rectangle((390, 1475, 640, 1585), outline="#f28e2b", width=7)
    draw.line((525, 1475, 460, 1445), fill="#cc3d3d", width=7)
    draw.polygon([(460, 1445), (482, 1448), (468, 1465)], fill="#cc3d3d")
    draw.multiline_text((180, 1590), "2018->2019: dx = -5.94 px, dy = -14.48 px\nrotation = -0.039 deg, uncertainty = 7.50 px\nrecord for scoring; do not warp polygon", fill="#55616b", font=small_font, spacing=5)

    cards = [
        (1010, "Pre-processing", "date registry\nquality / nodata\nobservability"),
        (1510, "IdentityGraph", "1:1 / 1:N / N:1\nneighborhood + shape\nconfidence + conflict"),
        (2010, "Presence", "1 / 0 / U\nreliable absence\nconservative gaps"),
        (2510, "State + area", "STABLE / ADDITION / DEMOLITION\nmetric polygon area\n>=100 m2 product filter last"),
    ]
    for x, title, body in cards:
        draw.rounded_rectangle((x, 1370, x + 430, 1635), radius=18, fill="#ffffff", outline="#274c77", width=3)
        draw.text((x + 22, 1395), title, fill="#274c77", font=section_font)
        draw.multiline_text((x + 22, 1465), body, fill="#3d4852", font=small_font, spacing=6)
    for x in [920, 1420, 1920, 2420]:
        draw.line((x, 1500, x + 80, 1500), fill="#cc3d3d", width=6)
        draw.polygon([(x + 80, 1500), (x + 58, 1487), (x + 58, 1513)], fill="#cc3d3d")

    draw.text((150, 1800), "Area accounting is done in metric EPSG:32649 polygons. Alignment displacement is uncertainty evidence, not permission to create a warped footprint.", fill="#17212b", font=small_font)
    draw.text((150, 1850), "Source image and overlays: frozen visual QA contact sheet; candidate status: CANDIDATE_NOT_GROUND_TRUTH.", fill="#71808a", font=small_font)
    canvas.save(dirs["figures"] / "OBJ_0001_process_flow.png", dpi=(220, 220))
    canvas.save(dirs["figures"] / "OBJ_0001_process_flow.pdf", "PDF", resolution=220)


def main() -> None:
    dirs = ensure_dirs()
    write_case_selection(dirs)
    write_statistics(dirs)
    write_process_notes(dirs)
    write_process_figure(dirs)
    shutil.copy2(ROOT / "12_视觉QA" / "CATEGORY_SUMMARY.csv", dirs["qa"] / "CATEGORY_SUMMARY.csv")
    shutil.copy2(ROOT / "12_视觉QA" / "FIXED_CASE_VISUAL_SCREEN.csv", dirs["qa"] / "FIXED_CASE_VISUAL_SCREEN.csv")
    shutil.copy2(ROOT / "12_视觉QA" / "RANDOM_100_VISUAL_SCREEN.csv", dirs["qa"] / "RANDOM_100_VISUAL_SCREEN.csv")
    shutil.copy2(ALIGN, dirs["qa"] / "ALIGNMENT_DIAGNOSTICS.csv")
    (OUT / "README.md").write_text(
        "# Xi'an current-best candidate delivery\n\n"
        "This directory is an auditable reporting layer assembled from the latest available R4 candidate. "
        "It does not modify the formal Geometry, IdentityGraph, Presence, or State outputs. Unknown/Conflict remains explicit.\n\n"
        "The raster products are generated separately by the geometry export step; the statistics and typical cases in this directory are already available now.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "cases": len(CASES), "status": "REPORTING_COMPLETE"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
