import json
import sys
from pathlib import Path


EXTENSIONS = Path(__file__).resolve().parents[1] / "extensions"
if str(EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(EXTENSIONS))

from surface_state_v1.training.full_trainer import _restore_selection_state


def test_restore_selection_and_patience(tmp_path):
    (tmp_path / "logs").mkdir()
    best = {"selection_score": 0.7, "macro_F1": 0.6, "epoch": 3}
    (tmp_path / "CHECKPOINT_SELECTION.json").write_text(
        json.dumps({"best": best}), encoding="utf-8"
    )
    rows = [
        {"epoch": 3, "improved": True},
        {"epoch": 4, "improved": False},
        {"epoch": 5, "improved": False},
    ]
    (tmp_path / "logs" / "epochs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    restored_best, patience = _restore_selection_state(tmp_path)

    assert restored_best == best
    assert patience == 2


def test_restore_selection_handles_warmstart_without_logs(tmp_path):
    assert _restore_selection_state(tmp_path) == (None, 0)