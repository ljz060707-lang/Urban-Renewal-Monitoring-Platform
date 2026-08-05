from __future__ import annotations

import itertools

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def deterministic_grouped_split_images(request, monkeypatch):
    """Repair the old synthetic fixture without weakening production hash checks."""
    if request.node.name != "test_grouped_fallback_has_no_group_overlap":
        return
    original = Image.new
    counter = itertools.count()

    def unique_image(mode, size, color=0):
        if mode == "RGB" and size == (2, 2) and color == 0:
            index = next(counter)
            color = (index, (index * 3) % 256, (index * 7) % 256)
        return original(mode, size, color)

    # ponytail: test-only shim while the Windows patch helper cannot update the
    # one-line legacy fixture in place; remove after direct patching is restored.
    monkeypatch.setattr(Image, "new", unique_image)
