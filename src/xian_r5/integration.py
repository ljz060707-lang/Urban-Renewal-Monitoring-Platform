"""Versioned handoff of existing R5 artifacts; evidence gates stay upstream.

This adapter replays native model products. It neither estimates displacement
nor selects SAM alternatives or certifies physical identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import fiona
import numpy as np
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .alignment.field_io import read_spatial_alignment_composition_fields
from .alignment.reviewed_field import attach_reviewed_alignment_cells, recheck_reviewed_alignment_sources
from .alignment.roof_candidate import apply_accepted_roof_grid_alignment
from .fraction_shadow import build_fraction_shadow
from .label_transfer import aggregate, fraction_supervision, supervision_eligibility
from .reconstruction.core import Observation, reconstruct
from .sam_source_candidates import sam_source_candidate
from .single_date_identity import validate_links
from .source_partition import SourceRegion

CONTRACT = "XIAN_R5_INTEGRATION_CONTRACT_V1"
INPUT_ROLES = {
    "MODULE_A_RECEIPT", "MODULE_A_CANDIDATES", "MODULE_A_REVIEW", "MODULE_B_RECEIPT", "MODULE_B_PLAN",
    "MODULE_B_PACKETS", "MODULE_B_PROMPTS", "MODULE_B_PREPARATION",
    "MODULE_B_GROUP_RECEIPT", "MODULE_B_NATIVE_MASK", "MODULE_B_NATIVE_RGB", "MODULE_B_WEIGHTS",
    "MODULE_I_RECEIPT", "MODULE_I_GRAPH", "MODULE_I_LINKS",
    "MODULE_I_ADJUDICATION", "MODULE_I_METRICS", "OBSERVED_CANDIDATE",
    "SOURCE_COVERAGE", "WINDOW_CONFIG", "GRID_TEMPLATE_ONLY", "CODE",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_artifacts(pins):
    """Only named, hash-pinned inference roles are readable by this entrypoint."""
    if not pins:
        raise ValueError("Empty artifact handoff")
    checked = {}
    for key, pin in pins.items():
        if pin["role"] not in INPUT_ROLES:
            raise ValueError("Forbidden automatic-inference input role: " + pin["role"])
        path = Path(pin["path"])
        if not path.is_absolute():
            raise ValueError("Absolute artifact paths required")
        digest = pin["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid SHA256: " + key)
        actual = sha256(path)
        if actual != digest:
            raise ValueError("Artifact changed: " + key)
        checked[key] = actual
    return checked


def adapt_native_components(path, tiles):
    """Retain original connected-component IDs and native geometry lineage."""
    tile_geometries = [shape(t["support_geometry"]) for t in tiles]
    if not tile_geometries or len({t["download_tile_id"] for t in tiles}) != len(tiles):
        raise ValueError("Unique source tiles required")
    tree = STRtree(tile_geometries)
    records = []
    with fiona.open(path, layer="native_positive_components") as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("Component geometry must use EPSG:32649")
        for feature in source:
            properties = dict(feature.properties)
            geometry = shape(feature.geometry)
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError("Invalid observed geometry; no automatic repair")
            hits = [int(i) for i in tree.query(geometry, predicate="intersects")
                    if geometry.intersection(tile_geometries[i]).area > 1e-6]
            source_dates = sorted({tiles[i]["actual_date"] for i in hits})
            for value in source_dates:
                date.fromisoformat(value)
            claimed = properties["source_actual_date"]
            # Do not silently replace a conflicting date with the nominal year.
            if source_dates and source_dates != [claimed]:
                raise ValueError("Observed/source actual-date conflict: " + properties["component_key"])
            support = unary_union([tile_geometries[i] for i in hits])
            uncovered = geometry.difference(support).area
            records.append({
                "contract_version": CONTRACT,
                "observation_id": properties["component_key"],
                "original_id": properties["component_key"],
                "geometry": mapping(geometry),
                "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
                "geometry_status": "NATIVE_ROOF_CANDIDATE_NOT_VERIFIED_INSTANCE",
                "crs": "EPSG:32649", "native_resampling": "NONE",
                "native_geometry_sha256": properties["native_sha256"],
                "source_path": str(Path(path)), "source_layer": source.name,
                "source_fid": feature.id,
                "actual_date": claimed if source_dates else None,
                "source_tile_ids": [tiles[i]["download_tile_id"] for i in sorted(hits)],
                "source_hashes": [tiles[i]["source_JPG_sha256"] for i in sorted(hits)],
                "source_paths": [tiles[i]["source_JPG"] for i in sorted(hits)],
                "source_support_uncovered_m2": uncovered,
                "source_support_complete": bool(hits) and uncovered <= 1e-6,
                "date_status": "REGISTRY_DATE_COHERENT" if source_dates else "UNKNOWN",
                "building_object_ids": [], "physical_id": None,
                "registered_geometry": None, "ground_footprint": None,
                "GT_USED_FOR_GEOMETRY": False,
            })
    ids = [r["observation_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate original observation ID")
    return records


def verify_source_dates(tiles, date_registry):
    """Match exact source-task/tile entries; year labels never supply dates."""
    expected = {(Path(t["source_JPG"]).parents[2].name, str(t["tile_x"]),
                 str(t["tile_y"]), str(t["zoom"])): t for t in tiles}
    found = set()
    with Path(date_registry).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["task_name"], row["tile_x"], row["tile_y"], row["zoom"])
            if key not in expected:
                continue
            tile = expected[key]
            if (key in found or row["date_conflict"] != "0" or row["tile_present"] != "1"
                    or row["actual_capture_date"] != tile["actual_date"]
                    or row["source_log"] != tile["date_evidence_source_log"]
                    or row["line_number"] != str(tile["date_evidence_line"])):
                raise ValueError("Conflicting source-date provenance: " + str(key))
            found.add(key)
    if found != set(expected):
        raise ValueError("Missing source-date provenance rows")
    return len(found)


def attach_single_date_links(records, ledger_path, graph_path, adjudications):
    """Revalidate the 53 source links, preserving all competing ledger rows."""
    if any(r["building_object_ids"] for r in records):
        raise ValueError("Link adapter requires original records without preattached object IDs")
    by_id = {r["observation_id"]: r for r in records}
    with fiona.open(graph_path, layer="building_object_reference") as source:
        object_ids = {f.properties["node_id"] for f in source}
    with fiona.open(graph_path, layer="roof_instance_reference") as source:
        original = {f.properties["node_id"]: (dict(f.properties), shape(f.geometry)) for f in source}
    if set(original) != set(by_id):
        raise ValueError("Module I observation set differs from native components")
    for key, row in by_id.items():
        properties, geometry = original[key]
        if (properties["original_id"] != key or properties["physical_id"] is not None
                or geometry.wkb != shape(row["geometry"]).wkb):
            raise ValueError("Module I original ID/geometry/physical-ID conflict")
    proofs = {}
    for document in adjudications:
        for row in document["records"]:
            if row["link_level"] == "TRACEABLE_SINGLE_DATE_LINK":
                key = (row["observation_id"], row["building_object_id"])
                if key in proofs:
                    raise ValueError("Duplicate source-link proof")
                proofs[key] = row
    with Path(ledger_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    linked = set()
    for row in rows:
        key = (row["observation_id"], row["building_object_id"])
        if row["physical_id"] != "" or row["GT_USED_FOR_GEOMETRY"] != "False":
            raise ValueError("Single-date CSV cannot contain physical ID or GT replacement")
        row["physical_id"] = None
        row["GT_USED_FOR_GEOMETRY"] = False
        if row["link_level"] == "TRACEABLE_SINGLE_DATE_LINK":
            if key in linked:
                raise ValueError("Duplicate traceable source-link ledger row")
            proof = proofs[key]
            if (proof["source_actual_date"] != by_id[key[0]]["actual_date"]
                    or row["source_actual_date"] != proof["source_actual_date"]
                    or proof["physical_id"] is not None or proof["GT_USED_FOR_GEOMETRY"] is not False):
                raise ValueError("Single-date link date/identity conflict")
            row["evidence"] = proof["evidence"]
            row["GT_USED_FOR_IDENTITY"] = proof["GT_USED_FOR_IDENTITY"]
            row["source_object_records"] = proof["source_object_records"]
            row["geometry_lineage"] = proof["geometry_lineage"]
            row["proof_scope"] = proof["proof_scope"]
            linked.add(key)
        validate_links([row], by_id, object_ids)
    if linked != set(proofs) or len({k[0] for k in linked}) != len(linked):
        raise ValueError("Single-date proofs must have one unique ledger link per observation")
    for row in rows:
        if row["link_level"] == "TRACEABLE_SINGLE_DATE_LINK":
            by_id[row["observation_id"]]["building_object_ids"].append(row["building_object_id"])
    # The full ledger, not just the closed subset, is returned for reverse queries.
    return rows, len(linked)


def adapt_sam_group(packet, prompt, receipt, arrays, checkpoint_sha256):
    """Decode original multi-mask outputs; never select by score or inherit ID."""
    if (receipt["group_id"] != prompt["group_id"]
            or receipt["component_key"] != prompt["component_key"]
            or receipt["image_sha256"] != packet["image_sha256"]
            or receipt["source_window"] != packet["source_window"]
            or receipt["source_geometry_wkb_sha256"] != prompt["source_geometry_wkb_sha256"]
            or receipt["selected_mask_index"] is not None
            or receipt["roof_or_full_instance_verified"] is not False
            or receipt["physical_id"] is not None):
        raise ValueError("SAM native handoff/selection conflict")
    masks = arrays["masks"]
    scores = arrays["predicted_mask_quality"]
    if (list(masks.shape) != receipt["original_mask_shape"] or masks.shape[0] != 3
            or not np.array_equal(masks.sum(axis=(1, 2)), receipt["mask_pixels"])
            or not np.array_equal(scores, np.asarray(receipt["predicted_mask_quality_diagnostic_only"]))):
        raise ValueError("Native SAM mask/score receipt mismatch")
    # Existing full-ROI prompts predate the small polygonizer's envelope fields.
    # These are joined from the hash-pinned native packet and source-only plan;
    # all actual prompt coordinates and labels remain unchanged.
    adapted = dict(prompt, source_parent_component=packet["source_parent_component"],
                   image_sha256=packet["image_sha256"], reference_or_target_positions_used=False)
    output = []
    for index in range(3):
        row, geometry = sam_source_candidate(masks[index], packet, adapted, index,
                                             float(scores[index]), checkpoint_sha256)
        row.update(contract_version=CONTRACT, observation_id=row["candidate_id"],
                   original_id=row["candidate_id"],
                   parent_observation_id=prompt["component_key"],
                   building_object_id=None, physical_id=None,
                   identity_link_status="NOT_INHERITED_FROM_PARENT_COMPONENT",
                   registered_geometry=None, geometry_status="UNSELECTED_NATIVE_SAM_CANDIDATE",
                   GT_USED_FOR_GEOMETRY=False)
        if geometry is not None:
            row["geometry"] = mapping(geometry)
        output.append(row)
    return output


def load_alignment_candidates(manifest_path, reference_path, reference_sha256):
    """Reuse the existing exact-support, hash-checked A interchange reader."""
    return read_spatial_alignment_composition_fields(
        Path(manifest_path), expected_base_raster=Path(reference_path),
        expected_base_sha256=reference_sha256)


def run_r5_state(records, alignment_receipt, reference_sources, *, candidate_fields=(), reviewed_layer=None):
    """Existing review/uncertainty adapter, whole-roof application, then R5.

    Candidate fields enter unreviewed. Only the existing review adapter can
    authorize their unchanged vectors/supports. A receipt count never does.
    """
    for name in ("ACCEPTED_LOCAL_REGIONS", "APPLIED_SHADOW_LOCAL_REGIONS"):
        if type(alignment_receipt.get(name)) is not int or alignment_receipt[name] < 0:
            raise ValueError("Invalid A counter: " + name)
    if alignment_receipt["APPLIED_SHADOW_LOCAL_REGIONS"] > alignment_receipt["ACCEPTED_LOCAL_REGIONS"]:
        raise ValueError("Applied A regions exceed accepted regions")
    fields = list(candidate_fields)
    if any(c.reviewed or c.accepted or c.applied for f in fields for c in f.cells):
        raise ValueError("A candidates must enter unreviewed; use the reviewed-field handoff")
    report = None
    if reviewed_layer is not None:
        fields, report = attach_reviewed_alignment_cells(fields, Path(reviewed_layer))
    accepted = sum(c.accepted for f in fields for c in f.cells)
    if accepted != alignment_receipt["ACCEPTED_LOCAL_REGIONS"]:
        raise ValueError("A acceptance count disagrees with deterministic reviewed-field handoff")
    observations = []
    for row in records:
        if (row["physical_id"] is not None or row["registered_geometry"] is not None
                or row["ground_footprint"] is not None or row["GT_USED_FOR_GEOMETRY"]):
            raise ValueError("Uncertified record cannot promote geometry or physical identity")
        roof = shape(row["geometry"])
        if hashlib.sha256(roof.wkb).hexdigest() != row["geometry_sha256"]:
            raise ValueError("Observed geometry changed")
        obs = Observation(uid=row["observation_id"], epoch_label="SOURCE_" + str(row["actual_date"]),
                          actual_date=row["actual_date"], tile_id=None, image_path=None,
                          roof=roof, source_ref=row["source_path"],
                          source_tile_ids=tuple(row["source_tile_ids"]),
                          source_image_paths=tuple(row["source_paths"]),
                          source_hashes=tuple(row["source_hashes"]),
                          date_status=row["date_status"])
        application = apply_accepted_roof_grid_alignment(
            roof, obs.actual_date, fields, reference_sources=reference_sources)
        # Complete objects are preserved. Missing source support cannot be
        # repaired by translating a clipped observation into an accepted cell.
        if application.applied and not row.get("source_support_complete", False):
            obs.image_alignment_status = "SOURCE_SUPPORT_INCOMPLETE_UNRESOLVED"
        elif application.applied:
            obs.registered_roof = application.geometry
            obs.image_alignment_status = application.status
            obs.image_alignment_field_id = application.field_id
            obs.image_alignment_cell_id = application.cell_id
            obs.image_alignment_dx_m = application.correction_dx_m
            obs.image_alignment_dy_m = application.correction_dy_m
            obs.image_alignment_positional_uncertainty_m = application.positional_uncertainty_m
            obs.image_alignment_relative_uncertainty_m = application.relative_registration_uncertainty_m
            obs.image_alignment_review_basis = application.review_basis
            obs.image_alignment_review_metadata = application.review_metadata
            row["registered_geometry"] = mapping(application.geometry)
            row["applied_vector_m"] = [application.correction_dx_m, application.correction_dy_m]
            row["alignment_evidence"] = application.review_metadata
        else:
            if application.geometry.wkb != roof.wkb:
                raise RuntimeError("Unqualified alignment changed native geometry")
            obs.image_alignment_status = application.status
        row["alignment_status"] = obs.image_alignment_status
        row.setdefault("applied_vector_m", None)
        observations.append(obs)
    result = reconstruct(observations)
    by_id = {r["uid"]: r for r in result.presence}
    for row in records:
        row["presence"] = by_id[row["observation_id"]]["presence"]
    if report is not None:
        recheck_reviewed_alignment_sources(report)
    return result, report


def build_gt10m_shadow(records, packets, source_support, grid, state_result):
    """Existing exact fraction builder + state/eligibility seams on one grid."""
    if {r["uid"] for r in state_result.presence} != {r["observation_id"] for r in records}:
        raise ValueError("GT10m/state object sets differ")
    observations = {o.uid: o for o in state_result.observations}
    registered = []
    for row in records:
        observation = observations[row["observation_id"]]
        if row.get("applied_vector_m") is not None:
            if (observation.registered_roof is None or row["registered_geometry"] is None
                    or shape(row["registered_geometry"]).wkb != observation.registered_roof.wkb
                    or row.get("alignment_evidence") != observation.image_alignment_review_metadata):
                raise ValueError("Qualified transformation handoff required for moved geometry")
            registered.append(observation.registered_roof)
        elif row["registered_geometry"] is not None or observation.registered_roof is not None:
            raise ValueError("Registered geometry without a qualified vector")
    objects = {r["observation_id"]: shape(r["geometry"]) for r in records}
    regions = [SourceRegion(p["region_id"], p["source_observed_date"],
                            shape(p["exact_support"]).intersection(source_support)) for p in packets]
    result = build_fraction_shadow(objects, regions, grid)
    if registered:
        known = unary_union(registered)
        projected = aggregate(grid, {"REGISTERED_ROOF_PROXY": known}, known)
        fraction = projected["fractions"][0]
        result["candidate_fraction"] = np.where(fraction > 0, fraction, -9999.)
        result["quality"][3] = fraction > 0
        result["quality"][0, fraction > 0] = 1  # Roof-instance semantics remain candidate.
        result["area_conservation_error_m2"]["candidate_area_m2"] = projected["area_conservation_error_m2"]
    shape2 = (grid.height, grid.width)
    no_evidence = np.zeros(shape2, dtype=bool)
    eligibility = supervision_eligibility(
        time_ok=no_evidence, geometry_ok=no_evidence, state_ok=no_evidence,
        interface_ok=np.ones(shape2, dtype=bool))
    # Observed positive candidate area is not a completeness/background proof.
    # No qualified roof supervision exists in this handoff, so its target is
    # explicitly all-unknown, independently of the nonzero diagnostic fraction.
    supervision = fraction_supervision(np.zeros((3, *shape2)),
                                      valid=np.zeros(shape2), unknown=np.ones(shape2),
                                      eligible=eligibility["eligible"])
    return result, eligibility, supervision
