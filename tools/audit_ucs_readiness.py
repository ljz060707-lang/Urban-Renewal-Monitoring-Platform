#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""URRS UCS formal-readiness audit.

Read-only toward source data. It never starts training and never reads Locked Test pixels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw

IMG_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}
LABEL_HINTS = {"label", "labels", "mask", "masks", "annotation", "annotations", "gt", "ground_truth"}
LICENSE_DECISIONS = {
    "confirmed_training_allowed",
    "confirmed_research_only",
    "confirmed_training_prohibited",
    "missing_license_information",
    "ambiguous_requires_human_confirmation",
}


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_stem(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    suffixes = (
        "_label", "-label", "_mask", "-mask", "_annotation", "-annotation",
        "_gt", "-gt", "_image", "-image", "_img", "-img", "_rgb", "-rgb",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                changed = True
    return re.sub(r"[_-]+", "_", value).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(out)


def looks_like_label(path: Path) -> bool:
    if path.suffix.lower() != ".png":
        return False
    parts = {part.lower() for part in path.parent.parts}
    return bool(parts & LABEL_HINTS) or any(x in path.stem.lower() for x in ("label", "mask", "_gt", "-gt"))


def discover(root: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    images, labels, archives, metadata = [], [], [], []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        low = path.name.lower()
        if looks_like_label(path):
            labels.append(path)
        elif path.suffix.lower() in IMG_EXT:
            images.append(path)
        if low.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar")):
            archives.append(path)
        if path.suffix.lower() in {".json", ".txt", ".md", ".yaml", ".yml"}:
            metadata.append(path)
    return sorted(images), sorted(labels), sorted(archives), sorted(metadata)


def find_split_manifests(project: Path) -> dict[str, Path]:
    patterns = {
        "train": ["UCS_TRAIN_MANIFEST*.csv", "*ucs*train*manifest*.csv"],
        "validation": ["UCS_VALIDATION_MANIFEST*.csv", "*ucs*validation*manifest*.csv", "*ucs*val*manifest*.csv"],
        "test": ["UCS_TEST_LOCKED_INVENTORY*.csv", "*ucs*test*locked*.csv"],
    }
    found: dict[str, Path] = {}
    for split, pats in patterns.items():
        candidates: list[Path] = []
        for pat in pats:
            candidates.extend(project.rglob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            found[split] = max(candidates, key=lambda p: p.stat().st_mtime)
    return found


def load_split_index(manifests: dict[str, Path]) -> tuple[dict[str, str], dict[str, Any]]:
    index: dict[str, str] = {}
    summary: dict[str, Any] = {}
    for split, path in manifests.items():
        rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                for key, value in row.items():
                    if value and any(token in key.lower() for token in ("path", "file", "image", "label", "stem", "id", "name")):
                        index.setdefault(norm_stem(Path(str(value)).stem), split)
        summary[split] = {"path": str(path), "rows": rows}
    return index, summary


def pair(labels: list[Path], images: list[Path], split_index: dict[str, str]) -> list[dict[str, Any]]:
    exact: dict[str, list[Path]] = defaultdict(list)
    normalized: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        exact[image.stem.lower()].append(image)
        normalized[norm_stem(image.stem)].append(image)
    rows = []
    for label in labels:
        matches = exact.get(label.stem.lower(), [])
        method, confidence = "", 0.0
        if len(matches) == 1:
            method, confidence = "exact_stem", 1.0
        elif not matches:
            matches = normalized.get(norm_stem(label.stem), [])
            if len(matches) == 1:
                method, confidence = "normalized_stem", 0.9
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: str(p).lower())[:1]
            method, confidence = "normalized_stem_ambiguous_best_effort", 0.6
        image = str(matches[0]) if matches else ""
        rows.append({
            "label_path": str(label),
            "label_stem": label.stem,
            "matched_image": image,
            "match_method": method or "unresolved",
            "match_confidence": confidence,
            "archive_source": "",
            "status": "matched" if image else "unresolved",
            "split": split_index.get(norm_stem(label.stem), "unknown"),
        })
    return rows


def archive_members(path: Path) -> tuple[str, list[str], str]:
    try:
        low = path.name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                return "listed", zf.namelist(), ""
        if low.endswith((".tar", ".tar.gz", ".tgz")):
            mode = "r:gz" if low.endswith((".tar.gz", ".tgz")) else "r"
            with tarfile.open(path, mode) as tf:
                return "listed", [m.name for m in tf.getmembers()], ""
        return "unsupported", [], "7z/rar requires an optional external listing tool."
    except Exception as exc:
        return "error", [], f"{type(exc).__name__}: {exc}"


def inspect_archives(archives: list[Path], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unresolved = {norm_stem(Path(row["label_path"]).stem): row for row in rows if row["status"] == "unresolved"}
    inventory = []
    hits: dict[str, list[str]] = defaultdict(list)
    for archive in archives:
        status, members, error = archive_members(archive)
        image_members = [m for m in members if Path(m).suffix.lower() in IMG_EXT]
        for member in image_members:
            key = norm_stem(Path(member).stem)
            if key in unresolved:
                hits[key].append(f"{archive}::{member}")
        inventory.append({
            "archive_path": str(archive),
            "size_bytes": archive.stat().st_size,
            "status": status,
            "members": len(members),
            "image_members": len(image_members),
            "error": error,
        })
    for row in rows:
        if row["status"] != "unresolved":
            continue
        candidates = hits.get(norm_stem(Path(row["label_path"]).stem), [])
        if len(candidates) == 1:
            row.update(status="available_in_archive", archive_source=candidates[0], match_method="archive_member", match_confidence=0.85)
        elif len(candidates) > 1:
            row.update(status="ambiguous_archive_match", archive_source=" | ".join(candidates), match_method="archive_member_ambiguous", match_confidence=0.5)
    return inventory, rows


def license_audit(metadata: list[Path], decision: str) -> dict[str, Any]:
    patterns = {
        "cc-by-4.0": re.compile(r"cc[-_ ]?by[-_ ]?4(?:\.0)?|creative commons attribution 4", re.I),
        "cc-by-nc-4.0": re.compile(r"cc[-_ ]?by[-_ ]?nc[-_ ]?4(?:\.0)?|noncommercial", re.I),
        "cc0-1.0": re.compile(r"\bcc0\b|public domain dedication", re.I),
        "mit": re.compile(r"\bmit license\b", re.I),
        "apache-2.0": re.compile(r"apache license[, ]+version 2", re.I),
    }
    evidence, identifiers = [], set()
    for path in metadata:
        if not any(x in path.name.lower() for x in ("license", "licence", "copying", "readme", "metadata", "citation")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:2_000_000]
        except Exception as exc:
            evidence.append({"path": str(path), "error": str(exc)})
            continue
        matched = [name for name, pattern in patterns.items() if pattern.search(text)]
        identifiers.update(matched)
        evidence.append({"path": str(path), "matched": matched, "sha256": sha256(path)})
    automatic = "ambiguous_requires_human_confirmation" if evidence else "missing_license_information"
    return {
        "recognized_identifiers": sorted(identifiers),
        "evidence": evidence,
        "automatic_status": automatic,
        "final_status": decision or automatic,
        "status_source": "explicit_cli_confirmation" if decision else "automatic_evidence_only",
    }


def raster_meta(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "path": str(path), "width": src.width, "height": src.height, "count": src.count,
            "dtype": list(src.dtypes), "descriptions": list(src.descriptions),
            "colorinterp": [str(x).split(".")[-1] for x in src.colorinterp],
            "crs": str(src.crs) if src.crs else None, "nodata": src.nodata,
        }


def infer_rgb(metas: list[dict[str, Any]]) -> dict[str, Any]:
    votes: Counter[tuple[int, int, int]] = Counter()
    evidence = []
    for meta in metas:
        mapping = {}
        for i, value in enumerate(meta.get("colorinterp", []), 1):
            if value.lower() in {"red", "green", "blue"}:
                mapping[value.lower()] = i
        if {"red", "green", "blue"} <= mapping.keys():
            order = (mapping["red"], mapping["green"], mapping["blue"])
            votes[order] += 1
            evidence.append({"path": meta["path"], "source": "colorinterp", "mapping": order})
    if votes:
        order, support = votes.most_common(1)[0]
        return {"status": "confirmed_by_embedded_metadata", "mapping": list(order), "support": support, "evidence": evidence}
    return {"status": "pending_human_confirmation", "mapping": None, "support": 0, "evidence": evidence}


def scale_band(array: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    valid = np.isfinite(array)
    vals = array[valid]
    if vals.size == 0:
        return np.zeros(array.shape, dtype=np.uint8), {"p2": 0.0, "p98": 0.0}
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((array - lo) / (hi - lo), 0, 1)
    out = (out * 255).astype(np.uint8)
    out[~valid] = 0
    return out, {"p2": lo, "p98": hi}


def render_rgb(path: Path, order: tuple[int, int, int]) -> tuple[Image.Image, dict[str, Any]]:
    with rasterio.open(path) as src:
        if max(order) > src.count:
            raise ValueError(f"Raster has {src.count} bands, requested {order}")
        scale = max(src.width / 800, src.height / 800, 1)
        width, height = max(1, round(src.width / scale)), max(1, round(src.height / scale))
        data = src.read(order, out_shape=(3, height, width), masked=True, resampling=rasterio.enums.Resampling.bilinear)
        channels, stats = [], []
        for band in data:
            rendered, stat = scale_band(band.filled(np.nan).astype(np.float32))
            channels.append(rendered); stats.append(stat)
        rgb = np.stack(channels, axis=-1)
        luminance = rgb.mean(axis=2)
        return Image.fromarray(rgb, "RGB"), {
            "valid_pixel_fraction": float(np.isfinite(data.filled(np.nan)).all(axis=0).mean()),
            "nonzero_fraction": float((rgb > 0).any(axis=2).mean()),
            "luminance_mean": float(luminance.mean()),
            "luminance_std": float(luminance.std()),
            "display_dynamic_range": float(np.percentile(luminance, 98) - np.percentile(luminance, 2)),
            "percentiles": stats,
        }


def render_label(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("P")
        out = img.convert("RGB") if img.getpalette() else img.convert("L").convert("RGB")
        return out.resize(size, Image.Resampling.NEAREST)


def deterministic_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["status"] == "matched" and row["split"].lower() not in {"test", "locked_test"}]
    if not eligible:
        eligible = [row for row in rows if row["status"] == "matched" and row["split"] == "unknown"]
    return sorted(eligible, key=lambda row: hashlib.sha256(row["label_path"].encode()).hexdigest())[:n]


def make_review(sample: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    previews, metas, size_fail, read_fail = [], [], 0, 0
    image_dir = out_dir / "previews"; image_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(sample, 1):
        image_path, label_path = Path(row["matched_image"]), Path(row["label_path"])
        try:
            meta = raster_meta(image_path); metas.append(meta)
            with Image.open(label_path) as lbl:
                label_size = list(lbl.size)
            if label_size != [meta["width"], meta["height"]]:
                size_fail += 1
            rgb, rgb_stats = render_rgb(image_path, (1, 2, 3))
            bgr, bgr_stats = render_rgb(image_path, (3, 2, 1))
            label = render_label(label_path, rgb.size)
            panel = Image.new("RGB", (rgb.width * 3, rgb.height + 48), "white")
            panel.paste(rgb, (0, 48)); panel.paste(bgr, (rgb.width, 48)); panel.paste(label, (rgb.width * 2, 48))
            draw = ImageDraw.Draw(panel)
            draw.text((10, 14), "RGB 1-2-3", fill="black")
            draw.text((rgb.width + 10, 14), "BGR 3-2-1", fill="black")
            draw.text((rgb.width * 2 + 10, 14), "LABEL", fill="black")
            target = image_dir / f"{i:03d}_{norm_stem(image_path.stem)}.jpg"
            panel.save(target, quality=92)
            previews.append({
                "image_path": str(image_path), "label_path": str(label_path), "split": row["split"],
                "preview_path": str(target), "rgb_stats": rgb_stats, "bgr_stats": bgr_stats,
                "image_size": [meta["width"], meta["height"]], "label_size": label_size,
                "size_match": label_size == [meta["width"], meta["height"]],
            })
        except Exception as exc:
            read_fail += 1
            previews.append({"image_path": str(image_path), "label_path": str(label_path), "split": row["split"], "error": f"{type(exc).__name__}: {exc}"})
    cards = []
    for item in previews:
        if "error" in item:
            cards.append(f"<article><h2>{html.escape(Path(item['image_path']).name)}</h2><pre>{html.escape(item['error'])}</pre></article>")
        else:
            rel = os.path.relpath(item["preview_path"], out_dir).replace(os.sep, "/")
            cards.append(f"<article><h2>{html.escape(Path(item['image_path']).name)}</h2><img src='{html.escape(rel)}'></article>")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>UCS RGB Review</title><style>body{{font-family:system-ui,'Microsoft YaHei';margin:24px;background:#f6f7f9}}header,main{{max-width:1500px;margin:auto}}article{{background:#fff;border:1px solid #ddd;padding:14px;margin:16px 0}}img{{width:100%;height:auto}}h2{{font-size:14px;word-break:break-all}}code{{background:#eee;padding:2px 5px}}</style><header><h1>UCS RGB 波段人工确认</h1><p>左：1-2-3；中：3-2-1；右：标签。确认后重新运行并传入 <code>--rgb-mapping 1,2,3</code>。</p></header><main>{''.join(cards)}</main></html>"""
    (out_dir / "UCS_RGB_BAND_REVIEW.html").write_text(page, encoding="utf-8")
    return previews, metas, size_fail, read_fail


SEED_INVENTORY_COLLECTION_KEYS = ("items", "samples", "inventory", "records", "seed_inventory")
SEED_INVENTORY_COUNT_KEYS = ("total", "count", "inventory_total", "seed_count")


def _seed_top_level_type(payload: Any) -> str:
    if isinstance(payload, list):
        return "array"
    if isinstance(payload, dict):
        return "object"
    if payload is None:
        return "null"
    return type(payload).__name__


def _seed_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_seed_inventory(payload: Any) -> dict[str, Any]:
    top_level_type = _seed_top_level_type(payload)
    top_level_keys = list(payload.keys()) if isinstance(payload, dict) else []
    if isinstance(payload, list):
        return {
            "parse_status": "SEED_INVENTORY_PARSED",
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
            "inventory_container_key": "$",
            "inventory_container_type": "array",
            "inventory_container_length": len(payload),
            "inventory_total": len(payload),
        }
    if not isinstance(payload, dict):
        return {
            "parse_status": "SEED_INVENTORY_PARSE_ERROR",
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
            "parse_error": "Top-level JSON value is neither an array nor an object.",
            "inventory_total": None,
        }

    for key in SEED_INVENTORY_COLLECTION_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, list):
            return {
                "parse_status": "SEED_INVENTORY_PARSED",
                "top_level_type": top_level_type,
                "top_level_keys": top_level_keys,
                "inventory_container_key": key,
                "inventory_container_type": "array",
                "inventory_container_length": len(value),
                "inventory_total": len(value),
            }
        return {
            "parse_status": "SEED_INVENTORY_PARSE_ERROR",
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
            "inventory_container_key": key,
            "inventory_container_type": type(value).__name__,
            "parse_error": f"Recognized inventory key '{key}' is not an array.",
            "inventory_total": None,
        }

    for key in SEED_INVENTORY_COUNT_KEYS:
        if key not in payload:
            continue
        total = _seed_count(payload[key])
        if total is not None:
            return {
                "parse_status": "SEED_INVENTORY_PARSED",
                "top_level_type": top_level_type,
                "top_level_keys": top_level_keys,
                "inventory_container_key": key,
                "inventory_container_type": type(payload[key]).__name__,
                "inventory_container_length": None,
                "inventory_total": total,
            }
        return {
            "parse_status": "SEED_INVENTORY_PARSE_ERROR",
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
            "inventory_container_key": key,
            "inventory_container_type": type(payload[key]).__name__,
            "parse_error": f"Recognized inventory count key '{key}' is not a non-negative integer.",
            "inventory_total": None,
        }

    return {
        "parse_status": "SEED_INVENTORY_PARSE_ERROR",
        "top_level_type": top_level_type,
        "top_level_keys": top_level_keys,
        "parse_error": "No supported inventory collection or count key was found.",
        "inventory_total": None,
    }


def seed_progress(root: Path) -> dict[str, Any]:
    inventories = list(root.rglob("SEED_LABEL_INVENTORY.json")) if root.exists() else []
    decisions = list(root.rglob("SEED_LABEL_DECISIONS.csv")) if root.exists() else []
    inv = max(inventories, key=lambda p: p.stat().st_mtime) if inventories else None
    dec = max(decisions, key=lambda p: p.stat().st_mtime) if decisions else None
    rows = []
    if dec and dec.stat().st_size:
        with dec.open("r", encoding="utf-8-sig", newline="") as fh: rows = list(csv.DictReader(fh))
    counter = Counter((row.get("transition_type") or row.get("transition") or row.get("verdict") or "unknown") for row in rows)
    if inv is None:
        return {
            "inventory_path": None, "decisions_path": str(dec) if dec else None,
            "inventory_total": None, "reviewed_count": len(rows), "remaining_count": None,
            "seed": None, "seed_status": "SEED_INVENTORY_NOT_FOUND", "status": "SEED_INVENTORY_NOT_FOUND",
            "top_level_type": None, "top_level_keys": [], "parse_status": "SEED_INVENTORY_NOT_FOUND",
            "parse_error": "SEED_LABEL_INVENTORY.json was not found.", "transition_counts": dict(counter),
        }
    try:
        payload = json.loads(inv.read_text(encoding="utf-8"))
        parsed = _parse_seed_inventory(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parsed = {
            "parse_status": "SEED_INVENTORY_PARSE_ERROR", "top_level_type": None, "top_level_keys": [],
            "inventory_total": None, "parse_error": f"Unable to parse inventory JSON: {type(exc).__name__}: {exc}",
        }
    if parsed["parse_status"] != "SEED_INVENTORY_PARSED":
        return {
            "inventory_path": str(inv), "decisions_path": str(dec) if dec else None,
            "inventory_total": None, "reviewed_count": len(rows), "remaining_count": None,
            "seed": None, "seed_status": "SEED_INVENTORY_PARSE_ERROR", "status": "SEED_INVENTORY_PARSE_ERROR",
            "transition_counts": dict(counter), **parsed,
        }
    total = parsed["inventory_total"]
    seed_status = "SEED_LABELING_COMPLETE" if total and len(rows) >= total else "PENDING_HUMAN_SEED_LABELS"
    return {
        "inventory_path": str(inv), "decisions_path": str(dec) if dec else None,
        "inventory_total": total, "reviewed_count": len(rows), "remaining_count": max(total-len(rows), 0),
        "seed": f"{len(rows)}/{total}", "seed_status": seed_status, "status": seed_status,
        "transition_counts": dict(counter), **parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ucs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/urrs/audits"))
    parser.add_argument("--seed-root", type=Path, default=Path("outputs/urrs/construction"))
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--rgb-mapping", default="")
    parser.add_argument("--license-decision", default="", choices=[""] + sorted(LICENSE_DECISIONS))
    args = parser.parse_args()

    project = args.project_root.resolve(); ucs = args.ucs_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else project / args.output_root
    seed_root = args.seed_root if args.seed_root.is_absolute() else project / args.seed_root
    if not project.exists(): raise SystemExit(f"Missing project root: {project}")
    if not ucs.exists(): raise SystemExit(f"Missing UCS root: {ucs}")

    out = output_root / f"ucs_formal_gate_v2_{stamp()}"; out.mkdir(parents=True)
    images, labels, archives, metadata = discover(ucs)
    manifests = find_split_manifests(project)
    split_index, split_summary = load_split_index(manifests)
    rows = pair(labels, images, split_index)
    archive_inventory, rows = inspect_archives(archives, rows)
    write_csv(out / "UCS_LABEL_IMAGE_MATCHES.csv", rows)
    write_csv(out / "UCS_UNRESOLVED.csv", [r for r in rows if r["status"] not in {"matched", "available_in_archive"}])
    write_csv(out / "UCS_ARCHIVE_INVENTORY.csv", archive_inventory)

    lic = license_audit(metadata, args.license_decision)
    write_json(out / "UCS_LICENSE_AUDIT.json", lic)

    sample = deterministic_sample(rows, args.sample_size)
    previews, metas, size_fail, read_fail = make_review(sample, out / "UCS_RGB_PAIR_REVIEW")
    write_json(out / "UCS_RGB_CANDIDATES.json", previews)
    metadata_rgb = infer_rgb(metas)
    if args.rgb_mapping:
        mapping = [int(x.strip()) for x in args.rgb_mapping.split(",")]
        if len(mapping) != 3 or min(mapping) < 1: raise SystemExit("--rgb-mapping must be like 1,2,3")
        rgb = {"status": "confirmed_by_human_cli", "mapping": mapping, "confirmed_at": iso_now(), "metadata_evidence": metadata_rgb}
    else:
        rgb = metadata_rgb
    write_json(out / "UCS_RGB_STATUS.json", rgb)

    seed = seed_progress(seed_root); write_json(out / "SEED_LABEL_PROGRESS.json", seed)
    matched = sum(r["status"] == "matched" for r in rows)
    archive_available = sum(r["status"] == "available_in_archive" for r in rows)
    unresolved = len(rows) - matched - archive_available
    rgb_ok = rgb["status"] in {"confirmed_by_embedded_metadata", "confirmed_by_human_cli"}
    license_ok = lic["final_status"] in {"confirmed_training_allowed", "confirmed_research_only"}
    pair_ok = bool(sample) and size_fail == 0 and read_fail == 0

    blockers = []
    if unresolved or archive_available: blockers.append({"code": "MISSING_OR_UNEXTRACTED_IMAGES", "detail": f"unresolved={unresolved}, archive_available={archive_available}"})
    if not rgb_ok: blockers.append({"code": "RGB_MAPPING_UNCONFIRMED", "detail": "Review UCS_RGB_BAND_REVIEW.html and rerun with --rgb-mapping."})
    if not license_ok: blockers.append({"code": "LICENSE_UNCONFIRMED_OR_PROHIBITED", "detail": lic["final_status"]})
    if not pair_ok: blockers.append({"code": "PAIR_VISUAL_AUDIT_FAILED", "detail": f"size_fail={size_fail}, read_fail={read_fail}"})
    if not manifests: blockers.append({"code": "SPLIT_MANIFESTS_NOT_FOUND", "detail": "Existing split manifests were not found."})

    status = "FORMAL_TRAINING_ALLOWED" if not blockers else "FORMAL_TRAINING_BLOCKED"
    codes = {b["code"] for b in blockers}
    if "MISSING_OR_UNEXTRACTED_IMAGES" in codes: status = "FORMAL_TRAINING_BLOCKED_MISSING_IMAGES"
    elif "LICENSE_UNCONFIRMED_OR_PROHIBITED" in codes: status = "FORMAL_TRAINING_BLOCKED_LICENSE"
    elif "RGB_MAPPING_UNCONFIRMED" in codes: status = "FORMAL_TRAINING_BLOCKED_RGB_MAPPING"
    elif "PAIR_VISUAL_AUDIT_FAILED" in codes: status = "FORMAL_TRAINING_BLOCKED_PAIR_AUDIT"
    elif "SPLIT_MANIFESTS_NOT_FOUND" in codes: status = "FORMAL_TRAINING_BLOCKED_SPLIT"

    gate = {
        "created_at": iso_now(), "project_root": str(project), "ucs_root": str(ucs), "audit_dir": str(out),
        "formal_training_status": status, "formal_training_started": False, "locked_test_pixel_reads": 0,
        "source_data_modified": 0, "images_discovered": len(images), "labels_discovered": len(labels),
        "matched_pairs": matched, "archive_available": archive_available, "unresolved_labels": unresolved,
        "sampled_train_validation_pairs": len(sample), "pair_size_mismatches": size_fail, "pair_read_failures": read_fail,
        "rgb_status": rgb, "license_status": lic["final_status"], "split_manifests": split_summary,
        "seed_progress": seed, "blockers": blockers,
        "safety": {"test_pixels_read": 0, "gpu_started": False, "training_started": False, "source_root_written": False, "split_rebuilt": False},
    }
    write_json(out / "FORMAL_TRAINING_GATE_V2.json", gate)
    write_json(out / "REMAINING_BLOCKERS.json", {"blockers": blockers})
    actions = []
    if archive_available: actions.append("Extract confirmed missing members to a new derived directory, never overwrite UCS root.")
    if unresolved: actions.append("Use UCS_UNRESOLVED.csv as the exact missing-image list.")
    if not rgb_ok: actions.append("Open UCS_RGB_PAIR_REVIEW/UCS_RGB_BAND_REVIEW.html and rerun with --rgb-mapping.")
    if not license_ok: actions.append("Confirm the license and rerun with --license-decision.")
    if status == "FORMAL_TRAINING_ALLOWED": actions.append("Prepare a training launch plan; do not start training automatically.")
    write_json(out / "NEXT_EXACT_ACTIONS.json", {"actions": actions})
    md = [
        "# UCS Formal Training Gate V2", "", f"- Status: **{status}**", f"- UCS root: `{ucs}`",
        f"- Images: {len(images)}", f"- Labels: {len(labels)}", f"- Matched: {matched}",
        f"- Archive-available: {archive_available}", f"- Unresolved: {unresolved}",
        f"- RGB: `{rgb['status']}` / `{rgb.get('mapping')}`", f"- License: `{lic['final_status']}`",
        f"- Seed labels: {seed.get('seed') or 'UNAVAILABLE'}", f"- Seed status: `{seed['seed_status']}`", "- Locked Test pixel reads: **0**",
        "- Training started: **No**", "", "## Blockers", "",
    ] + ([f"- `{b['code']}`: {b['detail']}" for b in blockers] or ["- None"])
    (out / "FORMAL_TRAINING_GATE_V2.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": status, "audit_dir": str(out), "matched": matched, "unresolved": unresolved, "archive_available": archive_available, "rgb": rgb["status"], "license": lic["final_status"], "seed": seed.get("seed"), "seed_status": seed["seed_status"], "seed_progress_status": seed["status"], "seed_top_level_keys": seed.get("top_level_keys", []), "locked_test_pixel_reads": 0, "training_started": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
