from surface_state_v1.training.checkpointing import select_better


def test_best_checkpoint_uses_score_then_f1_then_earlier_epoch():
    best = {"selection_score": 0.5, "macro_F1": 0.6, "epoch": 1}
    assert select_better({"selection_score": 0.6, "macro_F1": 0.1, "epoch": 2}, best, "cwld")
    assert not select_better({"selection_score": 0.5, "macro_F1": 0.6, "epoch": 2}, best, "cwld")
