from pathlib import Path


def test_globfp_audit_entrypoint_exists():
    assert (Path(__file__).resolve().parents[3] / "tools" / "audit_globfp_dataset.py").exists()
