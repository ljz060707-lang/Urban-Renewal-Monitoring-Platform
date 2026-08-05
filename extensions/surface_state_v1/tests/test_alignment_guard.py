from surface_state_v1.integration.overlay_runtime import alignment_guard


def test_alignment_guard_stops_mismatch():
    left = {"width": 10, "height": 10, "crs": None, "transform": None}
    right = {"width": 11, "height": 10, "crs": None, "transform": None}
    assert alignment_guard(left, right)["alignment_status"] == "registration_required_before_temporal_inference"
