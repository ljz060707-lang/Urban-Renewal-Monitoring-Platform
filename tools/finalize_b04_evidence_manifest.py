"""Freeze hashes for the completed, metadata-only B04 evidence acquisition run."""
import hashlib
import json
from pathlib import Path

root = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\19_ground_registration_evidence\B04_20260929_01")
target = root / "14_receipts" / "DELIVERY_MANIFEST_SHA256.json"
if target.exists():
    raise FileExistsError(target)
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path != target:
        h = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        files[str(path.relative_to(root))] = {"sha256": h.hexdigest(), "size_bytes": path.stat().st_size}
obj = {"run_id": "B04_20260929_01", "status": "CATALOGUE_EVIDENCE_ONLY_SCENE_PIXELS_NOT_ACQUIRED", "file_count": len(files), "files": files}
with target.open("x", encoding="utf-8") as sink:
    json.dump(obj, sink, ensure_ascii=False, indent=2)
print(json.dumps({"manifest": str(target), "file_count": len(files)}))
