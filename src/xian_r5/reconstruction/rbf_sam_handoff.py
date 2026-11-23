"""Prepare a fixed-case RBF-SAM bundle for the reconstruction importer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .rbf_sam_native import convert_native_output_files


def prepare_rbf_sam_handoff(
    output: Path,
    *,
    prediction_bundle: Path | None,
    mosaic_diagnostic: Path | None,
    native_output: Path | None,
    inference_manifest: Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Validate the mutually exclusive input modes and produce the importer bundle.

    Native MMDetection JSON is converted to a sibling immutable candidate bundle.
    The exact-date mosaic diagnostic is mandatory in both modes. Conversion does
    not run inference and never promotes a roof proposal to ground truth.
    """
    native_requested = native_output is not None or inference_manifest is not None
    if native_requested:
        if native_output is None or inference_manifest is None:
            raise ValueError(
                "Native RBF-SAM input requires both --rbf-sam-native-output and "
                "--rbf-sam-inference-manifest")
        if prediction_bundle is not None:
            raise ValueError(
                "Supply either a prebuilt RBF-SAM prediction bundle or native output, not both")
        if mosaic_diagnostic is None:
            raise ValueError(
                "Native RBF-SAM conversion still requires the exact-date mosaic diagnostic")
        bundle_path = output.resolve().with_name(
            output.name + "_rbf_sam_prediction_bundle.json")
        if bundle_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite generated RBF-SAM candidate bundle: {bundle_path}")
        report = convert_native_output_files(
            native_output, inference_manifest, bundle_path)
        return bundle_path, report

    if (prediction_bundle is None) != (mosaic_diagnostic is None):
        raise ValueError(
            "RBF-SAM prediction import requires both the prediction bundle and its "
            "exact-date mosaic diagnostic")
    return prediction_bundle, None
