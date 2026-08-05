"""Compatibility wrapper for environments without pyarrow."""

from __future__ import annotations

import pyogrio  # type: ignore


_read_dataframe = pyogrio.read_dataframe


def _read_without_arrow(*args, **kwargs):
    kwargs.pop("use_arrow", None)
    return _read_dataframe(*args, **kwargs)


pyogrio.read_dataframe = _read_without_arrow

from globfp_nanjing_extract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
