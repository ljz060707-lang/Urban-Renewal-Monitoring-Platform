"""Urban Renewal Remote Sensing System (URRS) canonical package.

The package is deliberately non-invasive: existing Urban Monitor and
SurfaceState V1 entrypoints remain available, while new orchestration writes
only under ``outputs/urrs`` or timestamped candidate output directories.
"""

__version__ = "1.0.0-candidate"
__all__ = ["__version__"]
