from pathlib import Path


def test_package_script_exists():
    assert (Path(__file__).resolve().parents[3] / "tools" / "package_building_foundation_v2.py").exists()


def test_static_inference_tool_exists():
    assert (Path(__file__).resolve().parents[3] / "tools" / "infer_building_foundation_v2.py").exists()
