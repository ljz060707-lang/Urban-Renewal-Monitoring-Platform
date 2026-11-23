"""Replay the frozen R21 geometry and solve R22 cut selection against manual reference."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile

import cv2
import numpy as np
import pyogrio
import rasterio
from rasterio.features import shapes
from rasterio.io import MemoryFile
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import shapely
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer


def read(archive, name, expected=None):
    with archive.extractfile(name) as stream:
        data = stream.read()
    if expected and sha256(data).hexdigest() != expected.lower():
        raise ValueError(f"SHA mismatch: {name}")
    return data


def raster(data):
    with MemoryFile(data) as memory, memory.open() as dataset:
        return dataset.read(1).astype("int32"), dataset.transform, dataset.crs


def polygons(array, affine, project):
    groups = defaultdict(list)
    for geometry, label in shapes(array, mask=array > 0, transform=affine):
        groups[int(label)].append(transform(project, shapely.make_valid(shape(geometry))))
    return {key: shapely.union_all(parts) for key, parts in groups.items()}


def overlaps(geometry, resolution):
    ids, refs, index = resolution
    pairs = {}
    for j in index.query(geometry, predicate="intersects"):
        ref = refs[j]
        area = geometry.intersection(ref).area
        pairs[ids[j]] = (area / ref.area, area / (geometry.area + ref.area - area))
    return pairs


def edges(geometry, resolution):
    return {fid for fid, (_, iou) in overlaps(geometry, resolution).items() if iou >= .5}


def solve(options, protected):
    # options[parent] = [(candidate id, per-child reference edges)].
    col, lower, upper, matrix = [], [], [], []
    matched = defaultdict(list)
    actions = {}
    def constraint(items, lo, hi):
        matrix.extend((len(lower), i, value) for i, value in items)
        lower.append(lo)
        upper.append(hi)
    for parent, variants in options.items():
        choices = []
        for candidate, children in variants:
            x = len(col)
            col.append(.0001 if candidate is not None else 0)
            choices.append((x, 1))
            actions[x] = (parent, candidate)
            for links in children:
                assignments = []
                for fid in sorted(links):
                    y = len(col)
                    col.append(-1)
                    assignments.append((y, 1))
                    matched[fid].append((y, 1))
                if assignments:
                    constraint(assignments + [(x, -1)], -np.inf, 0)
        constraint(choices, 1, 1)
    if protected - matched.keys():
        raise ValueError("Missing frozen baseline match edges")
    for fid, assignments in matched.items():
        constraint(assignments, int(fid in protected), 1)
    if not col:
        return set(), {}
    row, column, value = zip(*matrix)
    arr = coo_matrix((value, (row, column)), shape=(len(lower), len(col))).tocsc()
    result = milp(col, integrality=np.ones(len(col)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(arr, lower, upper),
                  options={"mip_rel_gap": 0.0})
    if not result.success:
        raise RuntimeError(result.message)
    selected = {parent: cut for i, (parent, cut) in actions.items()
                if cut is not None and result.x[i] > .5}
    return {fid for fid, items in matched.items() if any(result.x[i] > .5 for i, _ in items)}, selected


def window(name, inputs, cache, row, shifts, baseline, coverage):
    native = PurePosixPath(row["image_path"]).parent.relative_to("/root")
    folder = f"xian20261111_r8/output/xian72_governance/{name}"
    label_file = "ROOF_GOVERNED_OBSERVATION_ID.tif"
    rec = json.loads(read(inputs, f"{folder}/WINDOW_RECEIPT.json"))
    digest = next(a["sha256"] for a in rec["artifacts"] if PurePosixPath(a["path"]).name == label_file)
    governed, affine, crs = raster(read(inputs, f"{folder}/{label_file}", digest))
    files = row["source_record"]["files"]
    support, support_affine, support_crs = raster(read(inputs, str(native/"SOURCE_SUPPORT.tif"), files["SOURCE_SUPPORT.tif"]["sha256"]))
    if support.shape != governed.shape or support_affine != affine or support_crs != crs:
        raise ValueError(f"{name}: support grid mismatch")
    project = Transformer.from_crs(crs, 32649, always_xy=True).transform
    bound = polygons((support > 0).astype("int32"), affine, project)[1]
    metadata, _, wkb, fields = pyogrio.raw.read(
        BytesIO(read(inputs, str(native/"PROVIDED_MAP_GT.gpkg"), files["PROVIDED_MAP_GT.gpkg"]["sha256"])),
        columns=["original_gt_fid", "fusion_role"],
        where="fusion_role = 'DESIGNATED_GT2019_ANCHOR'",
    )
    fids = dict(zip(metadata["fields"], fields))["original_gt_fid"]
    ref_project = Transformer.from_crs(metadata["crs"], 32649, always_xy=True).transform
    ref = {int(fid): transform(ref_project, shapely.make_valid(g)).intersection(bound)
           for fid, g in zip(fids, shapely.from_wkb(wkb))}
    ref = {fid: g for fid, g in ref.items() if g.area > 1.e-6}
    resolution = list(ref), list(ref.values()), shapely.STRtree(list(ref.values()))
    with np.load(BytesIO(read(inputs, f"compact/instances/{name}.npz"))) as f:
        raw = f["instances"].astype("int32")
    prefix = "xian20261111_r21_pair/selective_split/limited_fragmentation/XIAN"
    with np.load(BytesIO(read(inputs, f"{prefix}/{name}.npz"))) as f:
        split = f["SPLIT"].astype("int32")
    changed = (raw == governed) & (governed > 0) & (raw != split)
    variant = governed.copy()
    variant[changed] = split[changed]
    parent = dict(zip(*np.unique(np.stack([variant[governed > 0], governed[governed > 0]], axis=1), axis=0).T))
    def move(geom, owner):
        key = f"{name}_q{owner}"
        return translate(geom, *shifts.get(key, (0, 0))).intersection(bound)
    base = polygons(governed, affine, project)
    moved = {q: move(g, q) for q, g in base.items()}
    options = defaultdict(lambda: [(None, [])])
    for q, g in polygons(variant, affine, project).items():
        owner = int(parent[q])
        options[owner][0][1].append(edges(move(g, owner), resolution))
    frozen = {int(r["fid"]) for r in baseline if int(r["prior_match"])}
    reproduced, _ = solve(options, frozen)
    if len(reproduced) != len(frozen):
        raise ValueError(f"{name}: frozen R21 match count mismatch")
    with np.load(BytesIO(read(cache, f"xian20261111_r21_pair/data/{name}.npz"))) as f:
        boxes = f["boxes"]
    with np.load(BytesIO(read(cache, f"xian20261111_r22_cuts/data/{name}.npz"))) as f:
        groups, masks = f["groups"], f["masks"]
    preserved = set(np.unique(raw[raw != split]))
    variant_counts = Counter()
    for group in np.unique(groups):
        q, y0, y1, x0, x1 = map(int, boxes[group])
        if q in preserved or q not in moved:
            continue
        patch = np.s_[y0:y1, x0:x1]
        parent_pixels = raw[patch] == q
        affine_patch = affine * affine.translation(x0, y0)
        before = options[q][0][1][0]
        seen = set()
        for i in np.flatnonzero(groups == group):
            cut = cv2.resize(np.unpackbits(masks[i]).reshape(64, 64), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST).astype(bool) & parent_pixels
            if min(cut.sum(), (parent_pixels & ~cut).sum()) < max(32, .1 * parent_pixels.sum()):
                continue
            cut &= governed[patch] == q
            if not cut.any():
                continue
            shard = polygons(cut.astype("int32"), affine_patch, project)[1]
            a = move(shard, q).intersection(moved[q])
            b = moved[q].difference(a)
            aa, bb = overlaps(a, resolution), overlaps(b, resolution)
            ea = {fid for fid, (_, iou) in aa.items() if iou >= .5}
            eb = {fid for fid, (_, iou) in bb.items() if iou >= .5}
            harm = any(aa[fid][0] >= .1 and bb[fid][0] >= .1 for fid in aa.keys() & bb.keys())
            variant_counts["valid"] += 1
            variant_counts["fragmenting"] += harm
            signature = frozenset((frozenset(ea), frozenset(eb)))
            if harm or not (ea | eb) - before or signature in seen:
                continue
            seen.add(signature)
            options[q].append((int(i), [ea, eb]))
    found, selected = solve(options, frozen)
    candidates = list(moved.values())
    nearby_tree = shapely.STRtree(candidates)
    gaps = []
    for obj in coverage["details"]["AFTER_FACADE"]:
        if obj["covered_pixels"] or obj["fid"] not in ref:
            continue
        geometry = ref[obj["fid"]]
        nearest = nearby_tree.nearest(geometry)
        hits = nearby_tree.query(geometry, predicate="intersects")
        iou = max((geometry.intersection(candidates[j]).area / geometry.union(candidates[j]).area for j in hits), default=0.)
        corners = np.asarray(geometry.minimum_rotated_rectangle.exterior.coords)
        short = float(np.linalg.norm(np.diff(corners, axis=0), axis=1).min())
        gaps.append(dict(window=name, fid=obj["fid"], epoch=coverage["epoch"], site=coverage["site"],
                         date=coverage["date"], window_edge=obj["window_edge"], short_side_m=short,
                         native_ref_pixels=obj["supported_reference_pixels"], shifted_best_iou=iou,
                         shifted_candidate_distance_m=None if nearest is None else geometry.distance(candidates[nearest])))
    return dict(window=name, epoch=coverage["epoch"], site=coverage["site"], date=coverage["date"],
                baseline=len(frozen), selected=len(found), gained=len(found-frozen), lost=len(frozen-found),
                added_cuts=len(selected), reference_fragmenting_cuts=0, valid_cut_options=variant_counts["valid"],
                fragmenting_cut_options=variant_counts["fragmenting"],
                gain_fids=sorted(found-frozen), cut_by_parent=selected), gaps


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("r16", "r22", "r23", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--limit", type=int)
    args = cli.parse_args()
    coverage = {r["window"]: r for r in json.loads((args.r16/"xian_map_reference/BASELINE_OBJECTS.json").read_text())}
    baseline = defaultdict(list)
    with (args.r22/"REFERENCE_COMPARISON.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            baseline[r["window"]].append(r)
    args.output.mkdir(parents=True, exist_ok=False)
    windows, gaps = [], []
    with tarfile.open(args.r23/"INPUTS.tar") as inputs, tarfile.open(args.r23/"TRAIN_CACHE.tar") as cache:
        registry = {r["id"]: r for line in read(inputs, "xian20261111_r8/QUERY_CACHE_RECORDS.jsonl").splitlines() if (r := json.loads(line))["id"] in coverage}
        prefix = "xian20261111_r14_native_temporal/output/automatic_position"
        receipt = json.loads(read(inputs, f"{prefix}/PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json"))
        methods = json.loads(read(inputs, f"{prefix}/ALL_POSITION_PREDICTIONS.json", receipt["prediction_sha256"]))["methods"]
        shifts = {key: np.array(v) - methods["RAW"][key] for key, v in methods["GEOMETRY__DATE_FIELD"].items()}
        for name in list(coverage)[:args.limit]:
            result, missing = window(name, inputs, cache, registry[name], shifts, baseline[name], coverage[name])
            windows.append(result)
            gaps.extend(missing)
            print(name, result["baseline"], result["selected"], len(missing), flush=True)
    (args.output/"WINDOW_ORACLE.json").write_text(json.dumps(windows, indent=2))
    with (args.output/"ZERO_COVERAGE.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)


if __name__ == "__main__":
    main()
