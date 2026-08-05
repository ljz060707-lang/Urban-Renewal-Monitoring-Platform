"""Compatibility entry point for the first height-model script revision.

The original runner contains one syntactic closing-brace typo.  This wrapper
applies that deterministic in-memory correction, without rewriting the
project file or source data, then executes the same main entry point.
"""

from __future__ import annotations

import pathlib


source = pathlib.Path(__file__).with_name("build_globfp_height_model.py")
text = source.read_text(encoding="utf-8")
broken = '"internal_holdout_metrics": training["internal_holdout_metrics"]},\n'
fixed = '"internal_holdout_metrics": training["internal_holdout_metrics"]}\n'
if broken not in text:
    raise RuntimeError("expected deterministic return-dict typo was not found")
namespace = {"__file__": str(source), "__name__": "__height_model_runner__"}
exec(compile(text.replace(broken, fixed, 1), str(source), "exec"), namespace)


if __name__ == "__main__":
    raise SystemExit(namespace["main"]())
