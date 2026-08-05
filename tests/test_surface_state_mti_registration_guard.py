import json
import sys
from pathlib import Path


EXTENSIONS = Path(__file__).resolve().parents[1] / "extensions"
if str(EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(EXTENSIONS))

from surface_state_v1.integration.registration_final import register_final_candidate
from surface_state_v1.training.mti_selection_guard import finalize_mti_selection


def test_regressed_mti_is_not_selected(tmp_path):
    source = tmp_path / "MTI_SELECTION.json"
    source.write_text(
        json.dumps(
            {
                "selected_parameters": {
                    "temporal_change_F1": 0.21,
                    "flicker_rate": 0.15,
                },
                "no_mti": {"f1": 0.23, "flicker_rate": 0.10},
                "test_reads": 0,
            }
        ),
        encoding="utf-8",
    )
    final = finalize_mti_selection(source, tmp_path / "MTI_SELECTION_FINAL.json")
    assert final["integration_mode"] == "no_mti"
    assert final["selected_parameters"] is None
    assert final["mti_status"] == "mti_calibrated_but_not_selected"
    assert final["test_reads"] == 0


def test_registration_supports_role_specific_key():
    assert "registry_key" in register_final_candidate.__annotations__