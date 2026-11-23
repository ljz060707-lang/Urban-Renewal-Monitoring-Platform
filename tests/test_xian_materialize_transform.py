"""Regression tests for Xi'an VRT geotransform conversion."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "materialize_xian_reference_tiles.py"
_SPEC = spec_from_file_location("xian_materialize", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_affine_from_gdal_preserves_xy_origin_and_pixel_scales():
    transform = _MODULE.affine_from_gdal((108.75, 2.5e-6, 0.0, 34.43, 0.0, -2.5e-6))

    assert transform.c == 108.75
    assert transform.f == 34.43
    assert transform.a == 2.5e-6
    assert transform.e == -2.5e-6
    assert transform.d == 0.0
