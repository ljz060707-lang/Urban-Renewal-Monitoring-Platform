"""Compatibility runner for the final report builder's path-copy typo."""

from __future__ import annotations

from pathlib import Path


source = Path(__file__).with_name("build_globfp_final_reports.py")
text = source.read_text(encoding="utf-8")
broken = 'copy_audit(extract, height_inference_root / "GLOBFP_NANJING_EXTRACTION_AUDIT.json")'
fixed = 'copy_audit(data_root / "01_nanjing_extract" / "EXTRACTION_AUDIT.json", height_inference_root / "GLOBFP_NANJING_EXTRACTION_AUDIT.json")'
if broken not in text:
    raise RuntimeError("expected report builder compatibility fragment not found")
text = text.replace(broken, fixed, 1)
namespace = {"__file__": str(source), "__name__": "__report_builder__"}
exec(compile(text, str(source), "exec"), namespace)


if __name__ == "__main__":
    raise SystemExit(namespace["main"]())
