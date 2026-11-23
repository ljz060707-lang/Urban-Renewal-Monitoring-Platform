"""Read-only source audit and official Sentinel catalogue evidence, not registration."""
import csv
import hashlib
import json
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
import fiona
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform
from pyproj import Transformer

BASE = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923")
OUT = BASE / "19_ground_registration_evidence" / "B04_20260929_01"
PRIOR = BASE / "18_reconstruction" / "B04_TQA10_REVIEWED_REGISTRATION_CLOSURE_20260929_02"
PACKET = BASE / "18_reconstruction" / "B04_TQA10_alignment_review_grid2_4x4_20260929_02"
FOLDERS = ["00_audit", "01_sentinel_reference", "02_reference_registry", "03_b04_p1_remap", "04_manual_review", "05_accepted_cells", "06_registered_roof", "07_identity", "08_canonical", "09_presence", "10_registered_change", "11_tif", "12_qa", "13_reports", "14_receipts"]

def dump(path, data):
    with path.open("x", encoding="utf-8") as sink:
        json.dump(data, sink, indent=2, ensure_ascii=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def csvwrite(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as sink:
        w = csv.DictWriter(sink, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for folder in FOLDERS:
        (OUT / folder).mkdir()
    with (PRIOR / "B04_PRIORITY_ALIGNMENT_REVIEW.csv").open(encoding="utf-8-sig") as src:
        priority = [r for r in csv.DictReader(src) if r["priority_rank"] == "P1"]
    wanted = {(r["field_id"], r["cell_id"]) for r in priority}
    with fiona.open(PACKET / "absolute_base_review_template.gpkg", layer="reviewed_image_alignment_cells") as src:
        cells = {(r["properties"]["field_id"], r["properties"]["cell_id"]): shape(r["geometry"]) for r in src if (r["properties"]["field_id"], r["properties"]["cell_id"]) in wanted}
    region = unary_union(list(cells.values()))
    bbox = transform(Transformer.from_crs(32649, 4326, always_xy=True).transform, region).bounds
    audit = []
    # Explicitly bounded metadata scans; GE JPEG pyramids and training/test sets are not traversed.
    for directory in [r"F:\xian\xian底图", r"E:\Data Set\Satellite", r"E:\Data Set\External"]:
        command = ["rg", "--files", directory, "-g", "*.jp2", "-g", "*MSIL2A*", "-g", "*MTD*xml", "-g", "*S2*.zip"]
        try:
            p = subprocess.run(command, capture_output=True, timeout=20)
            names = p.stdout.decode("utf-8", errors="replace").splitlines()
            audit.append({"root": directory, "command": command, "returncode": p.returncode, "matches": names,
                          "stderr": p.stderr.decode("utf-8", errors="replace"), "status": "BOUNDED_SCAN_COMPLETED"})
        except subprocess.TimeoutExpired:
            audit.append({"root": directory, "status": "SCAN_TIMEOUT_NOT_PROOF_OF_ABSENCE", "timeout_s": 20})
    audit.append({"root": r"F:\xian\XIAN补充", "status": "TOP_LEVEL_GE_DATE_FOLDERS_ONLY_NO_SAFE_IDENTIFIED", "recursive_scan": "STOPPED_TO_AVOID_TRAVERSING_GE_TILE_PYRAMIDS"})
    dump(OUT / "00_audit" / "LOCAL_REFERENCE_SEARCH.json", {"scans": audit, "conclusion": "NO_QUALIFIED_LOCAL_REFERENCE_IDENTIFIED_WITHIN_AUDITED_SCOPE_NOT_GLOBAL_ABSENCE"})
    baseline = json.loads((PRIOR / "EVIDENCE_RECEIPT.json").read_text(encoding="utf-8"))
    protected = {str(PACKET / "absolute_base_review_template.gpkg"): sha(PACKET / "absolute_base_review_template.gpkg"),
                 str(PRIOR / "PRODUCT_MANIFEST.json"): sha(PRIOR / "PRODUCT_MANIFEST.json")}
    for p in [baseline["input_paths"]["r4_status"], *baseline["input_paths"].values()]:
        protected[p] = sha(p)
    dump(OUT / "00_audit" / "PROTECTED_INPUT_HASHES_BEFORE.json", protected)
    registry, candidates, probes = [], [], []
    for ge_date in sorted({r["moving_date"] for r in priority}):
        dt = date.fromisoformat(ge_date)
        parameters = {"collections": "sentinel-2-l2a", "bbox": ",".join(map(str, bbox)),
                      "datetime": f"{dt-timedelta(days=15)}T00:00:00Z/{dt+timedelta(days=15)}T23:59:59Z", "limit": 100}
        url = "https://stac.dataspace.copernicus.eu/v1/search?" + urllib.parse.urlencode(parameters)
        try:
            data = fetch_json(url)
        except Exception as exc:
            dump(OUT / "01_sentinel_reference" / f"{ge_date}_QUERY_ERROR.json", {"url": url, "error": str(exc)})
            continue
        raw_path = OUT / "01_sentinel_reference" / f"{ge_date}_STAC.json"
        dump(raw_path, {"query": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "response": data})
        entries = []
        for item in data.get("features", []):
            props = item["properties"]
            stamp = props.get("datetime", props.get("start_datetime"))
            gap = abs((date.fromisoformat(stamp[:10]) - dt).days)
            asset = item["assets"].get("B02_10m", {})
            covers = bool(asset.get("proj:bbox") and box(*asset["proj:bbox"]).covers(region) and asset.get("proj:code") == "EPSG:32649")
            cloud = props.get("eo:cloud_cover")
            entry = {"product_id": item["id"], "sensing_date": stamp[:10], "sensing_time": stamp,
                     "ge_actual_date": ge_date, "date_gap_days": gap, "cloud_percent_scene": cloud,
                     "support_fully_covered_metadata": covers, "local_cloud_visibility": "NOT_INSPECTED",
                     "catalogue_file": str(raw_path), "catalogue_sha256": sha(raw_path)}
            candidates.append(entry)
            entries.append((not covers, cloud is None or cloud > 20, gap, cloud if cloud is not None else 101, item, entry))
        entries.sort(key=lambda v: v[:4])
        if not entries:
            continue
        _, _, _, _, selected, selected_entry = entries[0]
        for band in ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]:
            asset = selected["assets"][band]
            checksum = asset.get("file:checksum", "")
            registry.append({"reference_id": selected["id"], "product_id": selected["id"],
                "sensing_date": selected_entry["sensing_date"], "sensing_time": selected_entry["sensing_time"],
                "mgrs_tile": selected["id"].split("_T")[1].split("_")[0], "processing_level": "L2A",
                "band": band, "crs": asset.get("proj:code"), "affine_transform": asset.get("proj:transform"),
                "width": asset.get("proj:shape", [None,None])[1], "height": asset.get("proj:shape", [None,None])[0],
                "bounds": asset.get("proj:bbox"), "pixel_size": asset.get("gsd"),
                "source_path": asset["href"], "sha256": None,
                "provider_sha256_unverified": checksum[4:] if checksum.startswith("1620") and len(checksum)==68 else None,
                "cloud_fraction_if_known": selected_entry["cloud_percent_scene"]/100 if selected_entry["cloud_percent_scene"] is not None else None,
                "ge_actual_date": ge_date, "date_gap_days": selected_entry["date_gap_days"],
                "selected_for_review": False, "selection_status": "SHORTLIST_ONLY_PIXELS_NOT_ACQUIRED",
                "https_url": asset["alternate"]["https"]["href"], "metadata_source": str(raw_path)})
        asset_url = selected["assets"]["B02_10m"]["alternate"]["https"]["href"]
        req = urllib.request.Request(asset_url, headers={"Range": "bytes=0-1023"})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                sample = response.read(1024)
                probe = {"http_status": response.status, "bytes_read": len(sample), "status": "ACCESSIBLE_REQUIRES_PIXEL_ACQUISITION"}
        except urllib.error.HTTPError as exc:
            probe = {"http_status": exc.code, "status": "AUTH_REQUIRED" if exc.code in (401,403) else "HTTP_ERROR"}
        except Exception as exc:
            probe = {"status": "NETWORK_ERROR", "error": str(exc)}
        probes.append({"ge_actual_date": ge_date, "product_id": selected["id"], "url": asset_url, **probe})
        print(json.dumps({"date": ge_date, "candidates": len(entries), "shortlist": selected_entry, "access": probe}, ensure_ascii=False), flush=True)
    dump(OUT / "02_reference_registry" / "SENTINEL_CATALOGUE_CANDIDATES.json", candidates)
    dump(OUT / "02_reference_registry" / "SENTINEL_REFERENCE_REGISTRY.json", {"status": "CATALOGUE_SHORTLIST_NOT_FROZEN_PIXEL_REFERENCE", "references": registry})
    if registry:
        csvwrite(OUT / "02_reference_registry" / "SENTINEL_REFERENCE_REGISTRY.csv", registry, list(registry[0]))
    dump(OUT / "14_receipts" / "OFFICIAL_ASSET_ACCESS_RECEIPT.json", probes)
    dump(OUT / "14_receipts" / "NATIVE_SENTINEL_GRID_RECEIPT.json", {"status": "METADATA_ONLY_NOT_PIXEL_VALIDATED", "native_grids": [{k:r[k] for k in ["reference_id", "crs", "affine_transform", "bounds", "width", "height", "pixel_size"]} for r in registry if r["band"] == "B02_10m"], "fine_grid_created": False, "old_outputs_resampled": False})
    remap = []
    for row in priority:
        ref = next((r for r in registry if r["ge_actual_date"] == row["moving_date"]), None)
        remap.append({"field_id": row["field_id"], "cell_id": row["cell_id"], "moving_date": row["moving_date"],
            "old_reference": row["reference_source"], "new_reference_id": ref["reference_id"] if ref else None,
            "new_reference_date": ref["sensing_date"] if ref else None, "date_gap_days": ref["date_gap_days"] if ref else None,
            "support_fully_covered": box(*ref["bounds"]).covers(cells[(row["field_id"], row["cell_id"])]) if ref else None,
            "coverage_basis": "CATALOGUE_GRID_EXTENT_ONLY_NOT_VALID_PIXEL_COVERAGE", "eligible_for_manual_review": False,
            "old_displacement_reused": False, "review_status": "UNRESOLVED_NOT_REVIEWED",
            "blocker_code": "OTHER_REFERENCE_PIXEL_ACCESS_BLOCKED", "fit_pairs": 0, "holdout_pairs": 0})
    csvwrite(OUT / "03_b04_p1_remap" / "B04_P1_REFERENCE_REMAP.csv", remap, list(remap[0]))
    dump(OUT / "04_manual_review" / "REVIEW_READINESS.json", {"p1_total": len(remap), "actual_visual_reviews": 0, "complete_fit_holdout_reviews": 0, "accepted": 0, "rejected": 0, "unresolved": len(remap), "reason": "Reference pixel access must be resolved before review; not a resolution-limit verdict"})
    after = {p: sha(p) for p in protected}
    dump(OUT / "14_receipts" / "PROTECTED_INPUT_HASH_RECHECK.json", {"before": protected, "after": after, "unchanged": after == protected, "scope": "named frozen packet, prior manifest, R4 status and fixed case/date source files"})
    print(str(OUT), flush=True)

if __name__ == "__main__":
    main()
