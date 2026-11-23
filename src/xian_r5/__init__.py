"""Isolated R5 candidate components for Xi'an three-epoch identity work.

The package is deliberately independent from the frozen R4 production path.
It owns evidence, diagnostics, and candidate decisions only; it never edits
the R4 geometry, raster, checkpoint, or candidate directories.
"""

R5_VERSION = "0.1.0-stage1"
SCIENTIFIC_STATES = ("STABLE", "ADDITION", "DEMOLITION", "UNRESOLVED")
OPERATIONAL_STATES = ("STABLE", "ADDITION", "DEMOLITION")

__all__ = ["R5_VERSION", "SCIENTIFIC_STATES", "OPERATIONAL_STATES"]
