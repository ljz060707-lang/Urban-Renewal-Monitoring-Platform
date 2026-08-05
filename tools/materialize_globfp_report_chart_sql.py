"""Materialize the report chart rows through a recorded SQLite query."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


SQL = """SELECT model, count, MAE_m, RMSE_m, median_absolute_error_m,
       P90_absolute_error_m, mean_bias_m, R2
FROM height_model_validation_metrics
ORDER BY model;"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    (root / "REPORT_HEIGHT_MODEL_CHART.sql").write_text(SQL + "\n", encoding="utf-8")

    for name in ["artifact_main.json", "artifact_construction.json", "artifact_height.json"]:
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        input_rows = payload["snapshot"]["datasets"]["height_models"]
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                """CREATE TABLE height_model_validation_metrics (
                    model TEXT, count INTEGER, MAE_m REAL, RMSE_m REAL,
                    median_absolute_error_m REAL, P90_absolute_error_m REAL,
                    mean_bias_m REAL, R2 REAL
                )""",
            )
            connection.executemany(
                "INSERT INTO height_model_validation_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row.get("model"),
                        row.get("count"),
                        row.get("MAE_m"),
                        row.get("RMSE_m"),
                        row.get("median_absolute_error_m"),
                        row.get("P90_absolute_error_m"),
                        row.get("mean_bias_m"),
                        row.get("R2"),
                    )
                    for row in input_rows
                ],
            )
            columns = ["model", "count", "MAE_m", "RMSE_m", "median_absolute_error_m", "P90_absolute_error_m", "mean_bias_m", "R2"]
            rows = [dict(zip(columns, row)) for row in connection.execute(SQL)]
        payload["snapshot"]["datasets"]["height_models"] = rows
        for chart in payload["manifest"].get("charts", []):
            if chart.get("id") == "height_model_mae_comparison":
                chart["source"] = {
                    "path": "REPORT_HEIGHT_MODEL_CHART.sql",
                    "description": "SQLite query over saved HEIGHT_MODEL_VALIDATION_METRICS values",
                    "query": {"sql": SQL},
                }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT_CHART_SQL_MATERIALIZED", "output_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
