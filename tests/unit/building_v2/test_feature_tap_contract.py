from urban_monitor.building_foundation_v2.diagnostics import oracle_by_stride


def test_feature_contract_uses_native_fusion_stride_not_a_new_backbone():
    rows = oracle_by_stride(__import__("numpy").pad(__import__("numpy").ones((16, 16), dtype="uint8"), 8))
    assert rows[2]["stride"] == 4

