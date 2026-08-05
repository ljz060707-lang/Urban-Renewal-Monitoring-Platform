"""Geometry-degenerate compatibility runner for morphology model training."""

from __future__ import annotations

import pathlib


source = pathlib.Path(__file__).with_name("build_globfp_height_model.py")
text = source.read_text(encoding="utf-8")
replacements = {
    'training["internal_holdout_metrics"},': 'training["internal_holdout_metrics"]},',
    'coords = list(oriented.exterior.coords) if not oriented.is_empty else []': 'coords = list(oriented.exterior.coords) if (not oriented.is_empty and hasattr(oriented, "exterior")) else []',
}
for broken, fixed in replacements.items():
    if broken not in text:
        raise RuntimeError(f"expected compatibility fragment not found: {broken}")
    text = text.replace(broken, fixed, 1)
namespace = {"__file__": str(source), "__name__": "__height_model_runner__"}
exec(compile(text, str(source), "exec"), namespace)


if __name__ == "__main__":
    raise SystemExit(namespace["main"]())
