from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "run_tongzhou_construction_images_best.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("tongzhou_construction_images_best", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_has_only_four_frozen_dates_and_no_nanjing_paths():
    tool = load_tool()
    assert tool.DATES == ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
    assert all("nanjing" not in value.lower() for value in tool.SCENE_FILES.values())
    assert tool.EXPECTED_CHECKPOINT_SHA256 == "c522d5a4e6c5d6124f968d659004d7870fe4adc9d7381dc4c8917fe019f0f060"


def test_nanjing_guard_rejects_both_english_and_chinese_paths():
    tool = load_tool()
    for value in (r"D:\outputs\nanjing\x", r"D:\outputs\南京\x"):
        try:
            tool.ensure_safe_path(Path(value))
        except RuntimeError:
            pass
        else:
            raise AssertionError(value)


def test_recipe_registry_requires_validation_only(tmp_path):
    tool = load_tool()
    assert tool.RUN_PREFIX.startswith("beijing_tongzhou_")
    assert "test" not in tool.project_paths(tmp_path)["validation_manifest"].name.lower()
