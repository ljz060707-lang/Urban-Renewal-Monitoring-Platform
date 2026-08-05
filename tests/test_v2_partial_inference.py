import numpy as np

from urban_monitor.change.fusion_v2 import fuse_teacher_outputs


def test_partial_fusion_marks_missing_teachers_without_fake_arrays():
    change = np.full((8, 8), 0.8, dtype=np.float32)
    new = np.full((8, 8), 0.5, dtype=np.float32)
    result = fuse_teacher_outputs(any_change=change, new_building=new, demolition=None)
    assert np.allclose(result.outputs["new_building_candidate"], 0.4)
    assert "demolition_candidate" not in result.outputs
    assert result.pending_outputs["demolition_candidate"] == "pending_teacher"
    assert result.pending_outputs["semantic_transition"].startswith("pending_teacher")

