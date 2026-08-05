"""Deterministic identity wrapper for the Jiangsu slice.

The real schema has a source ``id_all`` field, but 243 extracted features
carry a missing ID.  Preserve that raw field and use a row-scoped fallback
only for missing IDs so records cannot silently collapse during matching.
"""

from __future__ import annotations

import pyogrio  # type: ignore
import rasterio  # type: ignore
import rasterio.features  # type: ignore


_read_dataframe = pyogrio.read_dataframe


def _read_without_arrow(*args, **kwargs):
    kwargs.pop("use_arrow", None)
    return _read_dataframe(*args, **kwargs)


pyogrio.read_dataframe = _read_without_arrow

try:
    setattr(rasterio.enums.MergeAlg, "max", rasterio.enums.MergeAlg.replace)
except Exception:
    pass

_rasterize = rasterio.features.rasterize


def _rasterize_compat(shapes, *args, **kwargs):
    if kwargs.get("dtype") == "float32" and kwargs.get("merge_alg") == rasterio.enums.MergeAlg.replace:
        shapes = sorted(list(shapes), key=lambda item: float(item[1]))
    return _rasterize(shapes, *args, **kwargs)


rasterio.features.rasterize = _rasterize_compat

import pandas as pd  # type: ignore  # noqa: E402
import globfp_nanjing_extract as pipeline  # noqa: E402


def unique_source_identity(frame, fields):
    if "id_all" in fields:
        values = []
        for ordinal, value in enumerate(frame["id_all"].tolist()):
            if pd.isna(value):
                values.append(f"China_3:jiangsu:missing_id_row_{ordinal:09d}")
            else:
                values.append(f"China_3:jiangsu:id_all_{value}")
        return pd.Series(values, index=frame.index, dtype="string"), "id_all_with_row_fallback_for_missing_ids"
    return pipeline.object_id_series(frame, fields)


pipeline.object_id_series = unique_source_identity


if __name__ == "__main__":
    raise SystemExit(pipeline.main())
