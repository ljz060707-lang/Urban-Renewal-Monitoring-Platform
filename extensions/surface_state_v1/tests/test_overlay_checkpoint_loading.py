import torch


def test_overlay_candidate_checkpoint_loads(mini_run):
    checkpoint = next((mini_run / "checkpoints").glob("*best_validation.pth"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["scientific_status"] == "validation_candidate"
    assert state["validation_metrics"]["cwld"]["test_read_count"] == 0
