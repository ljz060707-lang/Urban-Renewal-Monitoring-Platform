"""Compatibility wrapper for pyarrow-free and older Rasterio environments."""

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
    merge_alg = kwargs.get("merge_alg")
    dtype = kwargs.get("dtype")
    if dtype == "float32" and merge_alg == rasterio.enums.MergeAlg.replace:
        shapes = sorted(list(shapes), key=lambda item: float(item[1]))
    return _rasterize(shapes, *args, **kwargs)


rasterio.features.rasterize = _rasterize_compat

from globfp_nanjing_extract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
