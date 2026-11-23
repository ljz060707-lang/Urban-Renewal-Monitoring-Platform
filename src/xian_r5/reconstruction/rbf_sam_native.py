"""Bridge upstream RBF-SAM JSON predictions into the audited R5 bundle.

The bridge does not run the model or accept its output as ground truth. It
requires an explicit fixed-case binding manifest, hashes the source/model
inputs, and only emits roof-mask/offset candidates for the existing review
gate.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..alignment.common import write_json
from .rbf_sam_candidates import (
    OFFSET_SEMANTICS,
    PREDICTION_SCHEMA,
    polygonize_rbf_sam_roof_mask,
)


NATIVE_MANIFEST_SCHEMA = "xian_r5.rbf_sam_native_inference_manifest.v1"
NATIVE_OUTPUT_FORMAT = "RBF_SAM_MMDET_JSON_PREDICTIONS"
FULL_TILE_PREPROCESS = "FULL_TILE_ASPECT_PRESERVING_RESIZE_NO_CROP"
_HASH = re.compile(r"[0-9a-fA-F]{64}")
_TILE_ID = re.compile(r"([0-9]+):([0-9]+)")
_PROMPT_SOURCES = {
    "OBSERVATION_ROOF",
    "DETECTED_ROOF_INSTANCE",
    "LOCAL_RESEGMENTED_ROOF",
}
_PROMPT_TYPES = {"building_box", "roof_box", "footprint_box", "noisy_box"}


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being hashed: {path}")
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[Any, str]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"JSON input changed while being read: {resolved}")
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON input: {resolved}") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _required_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return value.lower()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _actual_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact ISO acquisition date")
    try:
        normalized = date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label} must be an exact ISO acquisition date") from exc
    if normalized != value:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD format")
    return normalized


def _finite_pair(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain numeric values, not booleans or strings")
    pair = [float(item) for item in value]
    if not np.isfinite(pair).all():
        raise ValueError(f"{label} must contain finite values")
    return pair


def _finite_box(value: Any, label: str, *, width: int, height: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must be [x_min, y_min, x_max, y_max]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain numeric values, not booleans or strings")
    box = [float(item) for item in value]
    if not np.isfinite(box).all():
        raise ValueError(f"{label} must contain finite values")
    x_min, y_min, x_max, y_max = box
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(f"{label} must be a positive-area box inside the model input")
    return box


def _resolve_hashed_file(path_value: Any, expected_hash: Any, label: str) -> tuple[Path, str]:
    path_text = _required_text(path_value, f"{label}.path")
    try:
        path = Path(path_text).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label}.path cannot be resolved") from exc
    if not path.is_file():
        raise ValueError(f"{label}.path must point to a file")
    recorded = _required_hash(expected_hash, f"{label}.sha256")
    actual = _sha256_file(path)
    if actual != recorded:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, actual


def _image_size(path: Path, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"{label} is not a readable image") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid dimensions")
    return width, height


def _prepare_bindings(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if manifest.get("schema_version") != NATIVE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported RBF-SAM native inference manifest schema")
    if manifest.get("scope") != "FIXED_CASE_LOCAL_ONLY":
        raise ValueError("RBF-SAM native conversion is restricted to fixed-case inputs")

    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("name") != "RBF-SAM":
        raise ValueError("Manifest must identify the RBF-SAM model")
    checkpoint_path, checkpoint_hash = _resolve_hashed_file(
        model.get("checkpoint_path"), model.get("checkpoint_sha256"), "model.checkpoint")
    config_path, config_hash = _resolve_hashed_file(
        model.get("config_path"), model.get("config_sha256"), "model.config")
    model_receipt = {
        "name": "RBF-SAM",
        "source": _required_text(model.get("source"), "model.source"),
        "revision": _required_text(model.get("revision"), "model.revision"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "config_path": str(config_path),
        "config_sha256": config_hash,
    }

    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Manifest must contain image bindings")
    bindings: dict[str, dict[str, Any]] = {}
    seen_uids: set[str] = set()
    for index, image in enumerate(images):
        label = f"images[{index}]"
        if not isinstance(image, dict):
            raise ValueError(f"{label} must be an object")
        native_name = _required_text(image.get("native_file_name"), f"{label}.native_file_name")
        if native_name in bindings:
            raise ValueError(f"Duplicate native_file_name: {native_name}")

        if image.get("preprocess_semantics") != FULL_TILE_PREPROCESS:
            raise ValueError(
                f"{label}.preprocess_semantics must be {FULL_TILE_PREPROCESS}; "
                "cropped or padded model inputs need a separately audited affine")
        actual_date = _actual_date(image.get("actual_date"), f"{label}.actual_date")
        tile_id = _required_text(image.get("tile_id"), f"{label}.tile_id")
        match = _TILE_ID.fullmatch(tile_id)
        if match is None:
            raise ValueError(f"{label}.tile_id must be numeric x:y")
        source_path, source_hash = _resolve_hashed_file(
            image.get("source_image_path"), image.get("source_image_sha256"),
            f"{label}.source_image")
        model_input_path, model_input_hash = _resolve_hashed_file(
            image.get("model_input_path"), image.get("model_input_sha256"),
            f"{label}.model_input")
        width = _positive_int(image.get("model_input_width"), f"{label}.model_input_width")
        height = _positive_int(image.get("model_input_height"), f"{label}.model_input_height")
        source_width, source_height = _image_size(source_path, f"{label}.source_image")
        input_width, input_height = _image_size(model_input_path, f"{label}.model_input")
        if (input_width, input_height) != (width, height):
            raise ValueError(f"{label}.model_input dimensions disagree with the manifest")
        if source_width * height != source_height * width:
            raise ValueError(
                f"{label}.model_input aspect ratio differs from the full source tile")

        prompts = image.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError(f"{label}.prompts must be a list")
        prompt_map: dict[int, dict[str, str]] = {}
        for prompt_index, prompt in enumerate(prompts):
            prompt_label = f"{label}.prompts[{prompt_index}]"
            if not isinstance(prompt, dict):
                raise ValueError(f"{prompt_label} must be an object")
            annotation_id = prompt.get("native_annotation_id")
            if (isinstance(annotation_id, bool) or not isinstance(annotation_id, int)
                    or annotation_id <= 0):
                raise ValueError(f"{prompt_label}.native_annotation_id must be a positive integer")
            if annotation_id in prompt_map:
                raise ValueError(f"Duplicate native annotation ID in {native_name}: {annotation_id}")
            uid = _required_text(prompt.get("observation_uid"), f"{prompt_label}.observation_uid")
            if uid in seen_uids:
                raise ValueError(f"Observation UID is bound to more than one native prompt: {uid}")
            seen_uids.add(uid)
            prompt_source = prompt.get("prompt_source")
            if (not isinstance(prompt_source, str)
                    or prompt_source not in _PROMPT_SOURCES):
                raise ValueError(f"{prompt_label}.prompt_source is not a supported roof source")
            prompt_type = prompt.get("prompt_type")
            if (not isinstance(prompt_type, str)
                    or prompt_type not in _PROMPT_TYPES):
                raise ValueError(
                    f"{prompt_label}.prompt_type must explicitly name the RBF-SAM box prompt")
            prompt_box = _finite_box(
                prompt.get("prompt_box_xyxy_pixel"),
                f"{prompt_label}.prompt_box_xyxy_pixel",
                width=width,
                height=height,
            )
            prompt_map[annotation_id] = {
                "observation_uid": uid,
                "prompt_source": prompt_source,
                "prompt_type": prompt_type,
                "prompt_box_xyxy_pixel": prompt_box,
            }

        bindings[native_name] = {
            "actual_date": actual_date,
            "tile_id": tile_id,
            "source_image_path": str(source_path),
            "source_image_sha256": source_hash,
            "model_input_path": str(model_input_path),
            "model_input_sha256": model_input_hash,
            "model_input_width": width,
            "model_input_height": height,
            "prompts": prompt_map,
        }
    return model_receipt, bindings


def build_prediction_bundle(
    native_rows: Any,
    manifest: Mapping[str, Any],
    *,
    native_output_path: str | Path,
    native_output_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Convert upstream JSON rows; never treats predictions as reviewed ground truth."""
    if not isinstance(native_rows, list) or not native_rows:
        raise ValueError("RBF-SAM native output must be a nonempty list of prediction rows")
    model_receipt, bindings = _prepare_bindings(manifest)
    predictions = []
    seen_keys: set[tuple[str, int]] = set()
    seen_uids: set[str] = set()
    expected_keys = {
        (native_name, annotation_id)
        for native_name, binding in bindings.items()
        for annotation_id in binding["prompts"]
    }
    for index, row in enumerate(native_rows):
        label = f"native_rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        native_name = _required_text(row.get("file_name"), f"{label}.file_name")
        binding = bindings.get(native_name)
        if binding is None:
            raise ValueError(f"{label}.file_name has no exact manifest binding")
        annotation_id = row.get("id")
        if (isinstance(annotation_id, bool) or not isinstance(annotation_id, int)
                or annotation_id <= 0):
            raise ValueError(f"{label}.id must map to a positive native annotation ID")
        key = (native_name, annotation_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate native prediction key: {key}")
        seen_keys.add(key)
        prompt = binding["prompts"].get(annotation_id)
        if prompt is None:
            raise ValueError(f"{label}.id has no exact prompt binding for {native_name}")
        uid = prompt["observation_uid"]
        if uid in seen_uids:
            raise ValueError(f"Multiple native predictions map to observation {uid}")
        seen_uids.add(uid)

        mask = np.asarray(row.get("roof_mask"))
        mask = np.squeeze(mask)
        if mask.ndim != 2 or mask.shape != (
                binding["model_input_height"], binding["model_input_width"]):
            raise ValueError(f"{label}.roof_mask dimensions do not match the model input")
        roof_geometry = polygonize_rbf_sam_roof_mask(mask)
        offset = _finite_pair(row.get("offset"), f"{label}.offset")
        stable_id = hashlib.sha256(
            f"{native_name}\0{annotation_id}".encode("utf-8")).hexdigest()[:24]
        predictions.append({
            "prediction_id": f"p-{stable_id}",
            "observation_uid": uid,
            "actual_date": binding["actual_date"],
            "tile_id": binding["tile_id"],
            "source_image_path": binding["source_image_path"],
            "source_image_sha256": binding["source_image_sha256"],
            "model_input_path": binding["model_input_path"],
            "model_input_sha256": binding["model_input_sha256"],
            "model_input_width": binding["model_input_width"],
            "model_input_height": binding["model_input_height"],
            "roof_geometry_pixel": roof_geometry,
            "predicted_offset_xy_px": offset,
            "prompt_source": prompt["prompt_source"],
            "prompt_type": prompt["prompt_type"],
            "prompt_box_xyxy_pixel": prompt["prompt_box_xyxy_pixel"],
            "native_file_name": native_name,
            "native_annotation_id": annotation_id,
            "native_output_row": index,
        })

    missing_keys = expected_keys - seen_keys
    if missing_keys:
        examples = sorted(missing_keys)[:5]
        raise ValueError(
            f"Native output is missing {len(missing_keys)} declared prompt(s); "
            f"examples: {examples}")

    return {
        "schema_version": PREDICTION_SCHEMA,
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "offset_semantics": OFFSET_SEMANTICS,
        "model": model_receipt,
        "native_inference": {
            "format": NATIVE_OUTPUT_FORMAT,
            "native_output_path": str(Path(native_output_path).resolve()),
            "native_output_sha256": _required_hash(
                native_output_sha256, "native_output_sha256"),
            "binding_manifest_path": str(Path(manifest_path).resolve()),
            "binding_manifest_sha256": _required_hash(
                manifest_sha256, "manifest_sha256"),
            "mask_semantics": "UPSTREAM_POSTPROCESSED_BINARY_ROOF_MASK",
            "offset_source": "UPSTREAM_FINAL_DECODED_OFFSET",
            "preprocess_semantics": FULL_TILE_PREPROCESS,
            "predictions_are_ground_truth": False,
            "prompt_type_provenance": "EXPLICIT_FIXED_CASE_BINDING_MANIFEST",
            "declared_prompt_count": len(expected_keys),
            "predicted_prompt_count": len(predictions),
        },
        "predictions": predictions,
    }


def convert_native_output_files(
    native_output_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert two immutable receipts to a new prediction bundle without overwrite."""
    native_path = Path(native_output_path).resolve(strict=True)
    binding_path = Path(manifest_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    native_rows, native_hash = _read_json(native_path)
    manifest, manifest_hash = _read_json(binding_path)
    if not isinstance(manifest, dict):
        raise ValueError("RBF-SAM inference manifest root must be an object")
    bundle = build_prediction_bundle(
        native_rows, manifest,
        native_output_path=native_path,
        native_output_sha256=native_hash,
        manifest_path=binding_path,
        manifest_sha256=manifest_hash,
    )
    write_json(output, bundle)
    return {
        "status": "UNVERIFIED_CANDIDATE_BUNDLE_WRITTEN",
        "output_path": str(output),
        "prediction_count": len(bundle["predictions"]),
        "ground_footprints_confirmed": 0,
        "native_output_path": str(native_path),
        "native_output_sha256": native_hash,
        "binding_manifest_path": str(binding_path),
        "binding_manifest_sha256": manifest_hash,
        "checkpoint_sha256": bundle["model"]["checkpoint_sha256"],
        "config_sha256": bundle["model"]["config_sha256"],
    }
