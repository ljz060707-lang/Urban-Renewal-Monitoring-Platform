from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from affine import Affine
from shapely.geometry import Polygon, box

from urban_renewal.construction.area_accounting import audit_area
from urban_renewal.construction.datasets import UCSCodec, UCSManifestDataset
from urban_renewal.construction.parcel_features import calculate_parcel_features
from urban_renewal.construction.partial_labels import PartialLabelLoss
from urban_renewal.construction.scene_rules import classify_scene
from urban_renewal.construction.vectorization import repair_geometry, vectorize_mask
from urban_renewal.models.construction import ConstructionElementNet
from urban_renewal.review.construction_inventory import build_construction_inventory
from urban_renewal.review.construction_store import ConstructionReviewStore
from urban_renewal.temporal.construction_labeler import promote_human_label, pseudo_label
from urban_renewal.temporal.construction_mti import ConstructionMTI
from urban_renewal.temporal.construction_no_mti import ConstructionNoMTI
from urban_renewal.temporal.construction_state_graph import ConstructionStateGraph
from urban_renewal.temporal.construction_tfr import ConstructionTemporalFeatureService
from urban_renewal.tracking.construction_matching import match_construction_objects
from urban_renewal.tracking.split_merge import detect_split_merge


def test_ucs_codec_and_test_read_guard(tmp_path):
    codec = UCSCodec()
    encoded = codec.encode_array(np.array([[0, 1, 2, 3, 9]], dtype=np.uint8))
    assert encoded.tolist() == [[0, 1, 2, 3, 255]]
    manifest = tmp_path / "test.csv"
    manifest.write_text("sample_id,image_path,label_path,split\na,x,y,test\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Test pixel reads are locked"):
        UCSManifestDataset(manifest)


def test_dataset_specific_head_and_partial_label_loss_are_finite():
    model = ConstructionElementNet(width=4)
    image = torch.rand(2, 3, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    target[:, :2] = 255
    ucs = model(image, "ucs")
    assert "ucs_logits" in ucs and "cwld_logits" not in ucs
    assert ucs["evidence_presence_mask"].dtype == torch.bool
    losses = PartialLabelLoss()(ucs, target, "ucs")
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.ucs_native_head[-1].weight.grad is not None
    assert model.cwld_native_head[-1].weight.grad is None


def test_missing_native_evidence_is_not_background():
    model = ConstructionElementNet(width=4)
    output = model(torch.rand(1, 3, 16, 16), "cwld")
    presence = output["evidence_presence_mask"].cpu().tolist()
    assert presence[2] is False  # CWLD has no dustproof-net supervision.
    assert torch.count_nonzero(output["evidence_probabilities"][:, 2]) == 0


@pytest.mark.parametrize("timesteps", [2, 4])
def test_tfr_preserves_time_shape_and_missing_timestep(timesteps):
    module = ConstructionTemporalFeatureService(8, num_heads=2, num_layers=1)
    features = torch.randn(1, timesteps, 8, 5, 5, requires_grad=True)
    valid = torch.ones(1, timesteps, dtype=torch.bool)
    valid[:, -1] = False
    output = module(features, timestamps=torch.arange(timesteps)[None] * 31, temporal_valid_mask=valid)
    refined = output["refined_features"][0]
    assert refined.shape == features.shape
    assert torch.isfinite(refined).all()
    assert torch.count_nonzero(refined[:, -1]) == 0
    refined.sum().backward()


def test_no_mti_default_and_experimental_mti_disabled():
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    no_mti = ConstructionNoMTI()(probabilities)
    assert no_mti["integration"] == "no_mti"
    graph = ConstructionStateGraph({"states": ["stable_building", "cleared_site"], "transitions": []})
    experimental = ConstructionMTI(graph)
    assert experimental.available is True
    assert experimental.enabled is False
    assert experimental.selection_status == "experimental_unselected"


def test_state_graph_requires_demolition_evidence():
    graph = ConstructionStateGraph({"states": ["completed_building", "cleared_site"], "default_transition_cost": 4})
    without = graph.transition_cost("completed_building", "cleared_site", {"demolition": 0.0})
    with_evidence = graph.transition_cost("completed_building", "cleared_site", {"demolition": 1.0})
    assert without > with_evidence


def test_parcel_scene_vector_and_metric_area():
    polygon = box(0, 0, 10, 10)
    features = calculate_parcel_features(
        polygon,
        {"exposed_ground_evidence": 0.4, "dustproof_net_evidence": 0.1},
        distances={"roads": 12}, context={"full20_change_evidence": 0.8, "building_overlap": 0.3},
    )
    scene, reasons = classify_scene(features)
    assert scene == "demolition_renewal"
    assert reasons
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert repair_geometry(bowtie).is_valid
    mask = np.zeros((12, 12), dtype=np.uint8); mask[1:11, 1:11] = 1
    geometries = vectorize_mask(mask, Affine.translation(0, 12) * Affine.scale(1, -1), min_area=10)
    area = audit_area(mask, Affine.translation(0, 12) * Affine.scale(1, -1), geometries, "EPSG:32650")
    assert area.area_audit == "pass"
    assert area.raster_area_m2 == 100
    nonmetric = audit_area(mask, Affine.identity(), geometries, "EPSG:4326")
    assert nonmetric.raster_area_m2 is None


def test_object_matching_and_split_merge():
    previous = [{"track_id": "p", "geometry": box(0, 0, 10, 10), "scene_type": "active"}]
    current = [
        {"track_id": "c1", "geometry": box(0, 0, 5, 10), "scene_type": "active"},
        {"track_id": "c2", "geometry": box(5, 0, 10, 10), "scene_type": "active"},
    ]
    matched = match_construction_objects(previous, current, max_centroid_distance=30)
    assert matched["candidate_pairs"] == 2
    assert detect_split_merge(previous, current)[0]["event_type"] == "split"


def test_pseudo_label_requires_human_promotion():
    record = pseudo_label("t1", ["unknown_state"])
    assert record["label_source"] == "pseudo_rule"
    assert record["formal_metric_eligible"] is False
    verified = promote_human_label(record, reviewer="tester")
    assert verified["label_source"] == "human_verified"
    assert verified["formal_metric_eligible"] is True


def test_construction_review_store_is_independent(tmp_path):
    original = tmp_path / "review_decisions.csv"
    original.write_text("geometry,unchanged\n", encoding="utf-8")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    geometry_inventory = {"records": []}
    for date in ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16"):
        geometry_inventory["records"].append({
            "sample_id": f"geometry_{date}_00",
            "preview_paths": {"tile": f"previews/{date}.jpg"},
            "source_paths": {"overview": f"{date}.png"},
            "window_bounds": [0, 0, 10, 10],
        })
    inventory = build_construction_inventory(tmp_path, geometry_inventory)
    store = ConstructionReviewStore(tmp_path)
    payload = {
        "track_id": inventory["records"][0]["track_id"], "scene_type": "construction_site",
        "boundary_review": "accept", "confidence": 0.8, "reviewer": "tester",
        "inventory_hash": inventory["inventory_hash"],
    }
    store.save(payload)
    assert store.csv_path.exists() and store.jsonl_path.exists()
    assert json.loads(store.jsonl_path.read_text(encoding="utf-8"))["label_source"] == "human_verified"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
