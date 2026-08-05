"""Resume report generation with the portable report chart schema enabled."""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCE = Path(__file__).with_name("build_globfp_final_reports.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--height-training-root", type=Path, required=True)
    parser.add_argument("--height-inference-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()

    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        """def copy_audit(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
""",
        """def copy_audit(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)
""",
    )
    source = source.replace(
        'copy_audit(extract, height_inference_root / "GLOBFP_NANJING_EXTRACTION_AUDIT.json")',
        'copy_audit(data_root / "01_nanjing_extract" / "EXTRACTION_AUDIT.json", height_inference_root / "GLOBFP_NANJING_EXTRACTION_AUDIT.json")',
    )
    source = source.replace(
        "def base_html(title: str, subtitle: str, content: str, badges: list[tuple[str, str, str]]) -> str:\n",
        "def base_html(title: str, subtitle: str, content: str, *badge_args) -> str:\n    badges = badge_args[-1]\n",
    )
    chart_block = "\n".join(
        [
            '            "charts": [',
            '                {',
            '                    "id": "height_model_mae_comparison",',
            '                    "type": "bar",',
            '                    "title": "Height model validation MAE (m)",',
            '                    "dataset": "height_models",',
            '                    "sourceId": "height_validation",',
            '                    "encodings": {',
            '                        "x": {"field": "model", "type": "nominal"},',
            '                        "y": {"field": "MAE_m", "type": "quantitative"},',
            '                        "tooltip": [',
            '                            {"field": "model"},',
            '                            {"field": "MAE_m"},',
            '                            {"field": "RMSE_m"},',
            '                        ],',
            '                    },',
            '                },',
            '            ],',
        ],
    )
    source = source.replace('            "charts": [],\n', chart_block + "\n")
    source = source.replace(
        '                {"id": "technical_summary", "type": "markdown", "body": f"## 鎶€鏈憳瑕乗n\\n{subtitle}"},\n',
        '                {"id": "technical_summary", "type": "markdown", "body": f"## 鎶€鏈憳瑕乗n\\n{subtitle}"},\n                {"id": "height_model_chart_block", "type": "chart", "chartId": "height_model_mae_comparison"},\n',
    )

    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    argv = [
        str(SOURCE),
        "--data-root",
        str(args.data_root),
        "--construction-root",
        str(args.construction_root),
        "--height-training-root",
        str(args.height_training_root),
        "--height-inference-root",
        str(args.height_inference_root),
        "--baseline-root",
        str(args.baseline_root),
    ]
    import sys

    old_argv = sys.argv
    try:
        sys.argv = argv
        exec(compile(source, str(SOURCE), "exec"), namespace)
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
