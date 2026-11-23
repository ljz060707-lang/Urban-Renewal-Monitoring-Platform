"""SemiBCD-inspired consistency metadata, not a replacement change detector."""

from .consistency import cycle_consistency, perturbation_catalog, resolve_unknown_cascade

__all__ = ["cycle_consistency", "perturbation_catalog", "resolve_unknown_cascade"]
