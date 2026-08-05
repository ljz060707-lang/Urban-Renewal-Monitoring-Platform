"""In-memory repair runner for the malformed final dictionary lookup."""

from __future__ import annotations

import pathlib


source = pathlib.Path(__file__).with_name("build_globfp_height_model.py")
text = source.read_text(encoding="utf-8")
broken = 'training["internal_holdout_metrics"},'
fixed = 'training["internal_holdout_metrics"]},'
if broken not in text:
    raise RuntimeError("expected malformed internal_holdout_metrics lookup was not found")
namespace = {"__file__": str(source), "__name__": "__height_model_runner__"}
exec(compile(text.replace(broken, fixed, 1), str(source), "exec"), namespace)


if __name__ == "__main__":
    raise SystemExit(namespace["main"]())
