from pathlib import Path


def test_final_qa_entrypoint_exists():
    assert (Path(__file__).resolve().parents[3] / "tools" / "create_building_v2_qa_queue.py").exists()
