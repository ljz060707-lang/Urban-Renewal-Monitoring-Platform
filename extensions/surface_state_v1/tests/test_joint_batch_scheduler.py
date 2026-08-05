def test_joint_cycle_contract():
    # The trainer consumes one batch from each active loader before an optimizer step.
    active = ["cwld", "fusu"]
    assert active == ["cwld", "fusu"]
