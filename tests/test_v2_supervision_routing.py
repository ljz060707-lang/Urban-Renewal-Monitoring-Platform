from urban_monitor.change.supervision_v2 import active_heads, supervision_matrix


def test_explicit_supervision_routing():
    matrix = supervision_matrix(second_mapping_verified=False)
    assert active_heads("dsifn") == ("any_change",)
    assert matrix["s2looking"]["new_building"]
    assert not matrix["second"]["new_building"]
    assert supervision_matrix(second_mapping_verified=True)["second"]["demolition"]

