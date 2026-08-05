from pathlib import Path


def test_globfp_pairing_entrypoint_exists():
    assert (Path(__file__).resolve().parents[3] / "tools" / "pair_globfp_with_local_imagery.py").exists()
