from pathlib import Path

from urban_renewal.construction.audit_v2 import _collect_packages


def _package(root: Path, name: str):
    package = root / name
    (package / "images_tif").mkdir(parents=True)
    (package / "label").mkdir()
    return package


def test_package_merge_deduplicates_identical_and_keeps_complementary(tmp_path):
    first = _package(tmp_path, "part1")
    second = _package(tmp_path, "part2")
    (first / "images_tif" / "1.tif").write_bytes(b"same-image")
    (second / "images_tif" / "1.tif").write_bytes(b"same-image")
    (second / "images_tif" / "2.tif").write_bytes(b"new-image")
    (first / "label" / "1.png").write_bytes(b"same-label")
    (second / "label" / "1.png").write_bytes(b"same-label")
    (second / "label" / "2.png").write_bytes(b"new-label")
    images, labels, audit = _collect_packages([first, second])
    assert set(images) == {"1", "2"}
    assert set(labels) == {"1", "2"}
    assert audit["image_duplicate_stems"] == 1
    assert audit["image_conflicting_duplicate_stems"] == 0


def test_package_merge_reports_conflicting_same_stem(tmp_path):
    first = _package(tmp_path, "part1")
    second = _package(tmp_path, "part2")
    (first / "images_tif" / "1.tif").write_bytes(b"left")
    (second / "images_tif" / "1.tif").write_bytes(b"right")
    (first / "label" / "1.png").write_bytes(b"label")
    images, labels, audit = _collect_packages([first, second])
    assert set(images) == {"1"} and set(labels) == {"1"}
    assert audit["image_conflicting_duplicate_stems"] == 1


def test_review_page_declares_independent_construction_tab():
    root = Path(__file__).resolve().parents[3]
    html = (root / "web" / "urrs_review" / "index_v2.html").read_text(encoding="utf-8")
    script = (root / "web" / "urrs_review" / "review_v2.js").read_text(encoding="utf-8")
    assert "施工状态与生命周期" in html
    assert "PSEUDO LABEL — NOT GROUND TRUTH" in html
    assert "/api/construction/review" in script
    assert "construction_review_decisions" not in script  # paths remain server-side.
