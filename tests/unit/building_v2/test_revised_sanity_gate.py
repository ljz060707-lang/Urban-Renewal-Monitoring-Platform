from urban_monitor.building_foundation_v2.diagnostics import revised_gate


def test_revised_gate_uses_b0_floor_and_oracle_fraction():
    gate = revised_gate(b0_iou=0.905, building_iou=0.904, boundary_loss_initial=2.0, boundary_loss_final=0.2, boundary_f1_at_2=0.9, oracle_bf1_at_2=1.0)
    assert gate["building_protection"] is True
    assert gate["boundary_loss_drop_ge_80pct"] is True
    assert gate["boundary_bf1_ge_90pct_oracle"] is True

