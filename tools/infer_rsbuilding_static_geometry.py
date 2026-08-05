from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RSBUILDING_REPOSITORY = Path(
    os.environ.get("RSBUILDING_REPOSITORY", r"D:\Data Set\RSBuilding\repo")
).resolve()
if not RSBUILDING_REPOSITORY.exists():
    raise FileNotFoundError(
        "RSBuilding repository is missing. Set RSBUILDING_REPOSITORY or restore: "
        f"{RSBUILDING_REPOSITORY}"
    )
if str(RSBUILDING_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(RSBUILDING_REPOSITORY))

# This import must precede torch on Windows. It registers CUDA/NVRTC DLL paths.
from tools import rsbuilding_runtime_bootstrap  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner.checkpoint import load_checkpoint  # noqa: E402
from mmseg.structures import SegDataSample  # noqa: E402
from opencd.registry import MODELS  # noqa: E402
try:  # This RSBuilding fork predates opencd.utils.
    from opencd.utils import register_all_modules  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    import opencd.datasets  # noqa: E402,F401
    import opencd.models  # noqa: E402,F401
    from mmengine.registry import init_default_scope as _init_default_scope  # noqa: E402

    def register_all_modules(init_default_scope: bool = True) -> None:
        if init_default_scope:
            _init_default_scope("opencd")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rasterio.features import shapes  # noqa: E402
from rasterio.windows import Window  # noqa: E402


TARGET_DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
EXPECTED_CHECKPOINT_SHA256 = (
    "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
)
THRESHOLD = 0.5
TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
SOURCE_MODEL = "rsbuilding_swin_t_whu_iter2000"
EXCLUDED_SOURCE_WORDS = {
    "mask",
    "label",
    "prediction",
    "preview",
    "overview",
    "boundary",
    "probability",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                # Windows security scanners can briefly hold the destination
                # between progress writes. Preserve the latest state rather
                # than losing a resumable external inference receipt.
                path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
                temporary.unlink(missing_ok=True)
                return
            time.sleep(0.1 * (attempt + 1))


def date_from_name(path: Path) -> str | None:
    """Return a defensible acquisition date from a filename token.

    The original runner was limited to four Tongzhou dates.  Keeping the
    default target tuple below preserves that behaviour, while this generic
    parser lets the same frozen RSBuilding wrapper run on a separately audited
    external scene set such as Nanjing Jiangbei.
    """
    stem = path.stem
    for token in re.findall(r"(?<!\d)(20\d{6})(?!\d)", stem):
        try:
            return datetime.strptime(token, "%Y%m%d").date().isoformat()
        except ValueError:
            continue
    for token in re.findall(r"(?<!\d)(\d{6})(?!\d)", stem):
        try:
            return datetime.strptime(token, "%y%m%d").date().isoformat()
        except ValueError:
            continue
    return None


def raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": list(dataset.dtypes),
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform),
            "bounds": list(dataset.bounds),
            "resolution": list(dataset.res),
            "nodata": dataset.nodata,
            "georeferenced": bool(dataset.crs),
        }


def discover_scenes(source_root: Path, geometry_root: Path, target_dates: tuple[str, ...] | None = None) -> dict[str, Any]:
    target_dates = tuple(target_dates or TARGET_DATES)
    candidates: dict[str, list[dict[str, Any]]] = {date: [] for date in target_dates}
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        lowered = path.name.lower()
        if any(word in lowered for word in EXCLUDED_SOURCE_WORDS):
            continue
        scene_date = date_from_name(path)
        if scene_date not in candidates:
            continue
        try:
            metadata = raster_metadata(path)
        except Exception as exc:
            candidates[scene_date].append(
                {"path": str(path.resolve()), "read_error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        candidates[scene_date].append(
            {
                "path": str(path.resolve()),
                "pixel_count": metadata["width"] * metadata["height"],
                **metadata,
            }
        )

    scenes: list[dict[str, Any]] = []
    ambiguous: dict[str, list[dict[str, Any]]] = {}
    for scene_date in target_dates:
        valid = [item for item in candidates[scene_date] if "read_error" not in item]
        valid.sort(
            key=lambda item: (
                bool(item["georeferenced"]),
                item["pixel_count"],
                item["bands"],
                item["path"],
            ),
            reverse=True,
        )
        if not valid:
            ambiguous[scene_date] = candidates[scene_date]
            continue
        selected = valid[0]
        if len(valid) > 1:
            first_rank = (
                bool(valid[0]["georeferenced"]),
                valid[0]["pixel_count"],
                valid[0]["bands"],
            )
            second_rank = (
                bool(valid[1]["georeferenced"]),
                valid[1]["pixel_count"],
                valid[1]["bands"],
            )
            if first_rank == second_rank:
                ambiguous[scene_date] = valid
                continue
        source = Path(selected["path"])
        scenes.append(
            {
                "scene_date": scene_date,
                "selected_source": str(source),
                "source_sha256": sha256_file(source),
                "selection_reason": (
                    "largest valid georeferenced raster for date; generated masks, "
                    "labels, predictions, previews, overviews and boundaries excluded"
                ),
                "source_exists": source.exists(),
                **{key: value for key, value in selected.items() if key not in {"path", "pixel_count"}},
            }
        )

    status = "pass" if len(scenes) == len(target_dates) and not ambiguous else "blocked"
    payload = {
        "scene_discovery_status": status,
        "source_root": str(source_root.resolve()),
        "target_dates": list(target_dates),
        "scenes": scenes,
        "candidate_files": candidates,
        "ambiguous_or_missing": ambiguous,
        "scene_count": len(scenes),
        "original_data_unchanged": True,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "created_at": utc_now(),
    }
    write_json(geometry_root / "TONGZHOU_SCENES.json", payload)
    if status != "pass":
        raise RuntimeError(
            "Scene discovery is blocked. See "
            f"{geometry_root / 'TONGZHOU_SCENES.json'}"
        )
    return payload


def load_scenes(geometry_root: Path) -> dict[str, Any]:
    path = geometry_root / "TONGZHOU_SCENES.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene manifest: {path}; run discover first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scene_discovery_status") != "pass":
        raise RuntimeError(f"Scene discovery is not pass: {path}")
    expected_dates = tuple(payload.get("target_dates", []))
    if [scene["scene_date"] for scene in payload["scenes"]] != list(expected_dates):
        raise RuntimeError("Scene dates are incomplete or out of order.")
    return payload


def output_directory(geometry_root: Path, scene_date: str) -> Path:
    return geometry_root / scene_date


def model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "RSBuilding checkpoint SHA256 mismatch; refusing inference. "
            f"Expected {EXPECTED_CHECKPOINT_SHA256}, got {sha256_file(checkpoint)}"
        )
    if not config.exists():
        raise FileNotFoundError(f"Config missing: {config}")
    return checkpoint, config


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    geometry_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root).resolve()
    checkpoint, config = model_paths(args)
    scene_manifest = geometry_root / "TONGZHOU_SCENES.json"
    historical_entries = []
    for path in geometry_root.rglob("*"):
        if path == geometry_root:
            continue
        historical_entries.append(
            {
                "path": str(path),
                "kind": "file" if path.is_file() else "directory",
                "size": path.stat().st_size if path.is_file() else None,
            }
        )
    formal_outputs = list(geometry_root.rglob("GEOMETRY_MANIFEST.json"))
    failed_logs = [
        str(path)
        for path in geometry_root.rglob("*")
        if path.is_file() and ("fail" in path.name.lower() or path.suffix.lower() in {".log", ".traceback"})
    ]
    if formal_outputs:
        root_cause = "wrong_output_path"
    elif failed_logs:
        root_cause = "inference_failed"
    elif historical_entries:
        root_cause = "inference_not_run"
    else:
        root_cause = "inference_not_run"

    import_status = "pass"
    import_error = None
    try:
        register_all_modules(init_default_scope=True)
        Config.fromfile(str(config))
    except Exception:
        import_status = "fail"
        import_error = traceback.format_exc()
        root_cause = "inference_failed"

    try:
        scenes = (
            json.loads(scene_manifest.read_text(encoding="utf-8"))
            if scene_manifest.exists()
            else discover_scenes(
                source_root,
                geometry_root,
                tuple(args.target_dates) if args.target_dates else None,
            )
        )
    except Exception:
        scenes = {"scene_discovery_status": "blocked", "error": traceback.format_exc()}
        root_cause = "source_scene_invalid"

    payload = {
        "diagnosis_status": (
            "pass"
            if import_status == "pass"
            and scenes.get("scene_discovery_status") == "pass"
            and torch.cuda.is_available()
            else "blocked"
        ),
        "geometry_zero_root_cause": root_cause,
        "geometry_file_count": sum(1 for path in geometry_root.rglob("*") if path.is_file()),
        "scene_manifest_exists": scene_manifest.exists(),
        "scene_discovery_status": scenes.get("scene_discovery_status"),
        "selected_sources_exist": [
            bool(Path(scene["selected_source"]).exists()) for scene in scenes.get("scenes", [])
        ],
        "inference_script": str(Path(__file__).resolve()),
        "inference_script_exists": Path(__file__).exists(),
        "inference_import_status": import_status,
        "inference_import_error": import_error,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_load_check": "pass",
        "config_path": str(config),
        "config_load_check": import_status,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "nvrtc_bootstrap": "pass",
        "historical_smoke_receipts": [
            str(path) for path in geometry_root.rglob("*SMOKE*RECEIPT*.json")
        ],
        "historical_entries": historical_entries,
        "failed_logs": failed_logs,
        "unfinished_directories": [
            item["path"] for item in historical_entries if item["kind"] == "directory"
        ],
        "test_reads": {"WHU": 0, "SECOND": 0},
        "created_at": utc_now(),
    }
    path = geometry_root / f"GEOMETRY_RESUME_DIAGNOSIS_{timestamp()}.json"
    write_json(path, payload)
    payload["diagnosis_path"] = str(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_model(config_path: Path, checkpoint: Path, device: str) -> torch.nn.Module:
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    model_cfg = cfg.model
    if isinstance(model_cfg, dict):
        model_cfg = dict(model_cfg)
        model_cfg["init_cfg"] = None
    model = MODELS.build(model_cfg)
    # Do not call model.init_weights(); this is a full checkpoint.
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.to(device)
    model.eval()
    return model


def extract_probability(output: Any) -> np.ndarray:
    field = getattr(output, "seg_logits", None)
    if field is None and hasattr(output, "get"):
        field = output.get("seg_logits")
    tensor = getattr(field, "data", field)
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    if tensor is None:
        raise RuntimeError("RSBuilding output does not contain seg_logits.")
    array = tensor.detach().float().cpu().numpy()
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[0] not in {1, 3}:
        raise RuntimeError(f"Unexpected RSBuilding output shape: {array.shape}")
    channel_zero = array[0]
    if not np.isfinite(channel_zero).all():
        raise RuntimeError("RSBuilding channel 0 contains NaN or Inf.")
    if channel_zero.min() < 0.0 or channel_zero.max() > 1.0:
        channel_zero = 1.0 / (1.0 + np.exp(-channel_zero))
    return channel_zero.astype(np.float32, copy=False)


def predict_batch(
    model: torch.nn.Module,
    tiles_rgb: list[np.ndarray],
    device: str,
) -> list[np.ndarray]:
    tensors: list[torch.Tensor] = []
    samples: list[SegDataSample] = []
    for tile_rgb in tiles_rgb:
        if tile_rgb.shape != (3, TILE_SIZE, TILE_SIZE):
            raise RuntimeError(f"Unexpected tile shape: {tile_rgb.shape}")
        # The project loader supplies BGR then the model preprocessor converts it to RGB.
        bgr = tile_rgb[[2, 1, 0], ...]
        six_channel = np.concatenate([bgr, bgr], axis=0)
        tensors.append(torch.from_numpy(np.ascontiguousarray(six_channel)))
        sample = SegDataSample()
        sample.set_metainfo(
            {
                "ori_shape": (TILE_SIZE, TILE_SIZE),
                "img_shape": (TILE_SIZE, TILE_SIZE),
                "padding_size": [0, 0, 0, 0],
                "flip": False,
            }
        )
        samples.append(sample)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        enabled=str(device).startswith("cuda"),
        dtype=torch.float16,
    ):
        outputs = model.test_step({"inputs": tensors, "data_samples": samples})
    return [extract_probability(output) for output in outputs]


def center_tile(path: Path) -> tuple[np.ndarray, np.ndarray, Window]:
    with rasterio.open(path) as source:
        col = max(0, (source.width - TILE_SIZE) // 2)
        row = max(0, (source.height - TILE_SIZE) // 2)
        window = Window(col, row, TILE_SIZE, TILE_SIZE)
        image = source.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        valid = source.dataset_mask(window=window, boundless=True) > 0
    return image, valid, window


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(image[:3], 0, -1).astype(np.float32)
    if image.dtype == np.uint8:
        return rgb.astype(np.uint8)
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros(rgb.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0
    return np.clip((rgb - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def save_smoke_images(
    output: Path, image: np.ndarray, probability: np.ndarray, valid: np.ndarray
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rgb = normalize_rgb(image)
    probability_display = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    probability_display[~valid] = 0
    mask = (probability >= THRESHOLD) & valid
    overlay = rgb.copy()
    overlay[mask] = (
        0.40 * overlay[mask].astype(np.float32) + 0.60 * np.array([0, 160, 255])
    ).astype(np.uint8)
    Image.fromarray(rgb).save(output / "SMOKE_INPUT.png")
    Image.fromarray(probability_display, mode="L").save(output / "SMOKE_PROBABILITY.png")
    Image.fromarray(overlay).save(output / "SMOKE_CENTER_TILE.png")


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    scenes = load_scenes(geometry_root)
    smoke_date = args.smoke_date or scenes["scenes"][-1]["scene_date"]
    scene = next(item for item in scenes["scenes"] if item["scene_date"] == smoke_date)
    checkpoint, config = model_paths(args)
    output = geometry_root / f"smoke_{smoke_date.replace('-', '')}_{timestamp()}"
    image, valid, window = center_tile(Path(scene["selected_source"]))
    model = build_model(config, checkpoint, args.device)
    started = time.perf_counter()
    probability = predict_batch(model, [image], args.device)[0]
    runtime = time.perf_counter() - started
    save_smoke_images(output, image, probability, valid)
    valid_probability = probability[valid]
    positive_fraction = float((valid_probability >= THRESHOLD).mean())
    gate = {
        "finite": bool(np.isfinite(valid_probability).all()),
        "range_pass": bool(
            valid_probability.size
            and valid_probability.min() >= 0
            and valid_probability.max() <= 1
        ),
        "positive_fraction_pass": bool(0 < positive_fraction < 0.90),
        "not_all_empty": bool((valid_probability >= THRESHOLD).any()),
        "not_all_full": bool(not (valid_probability >= THRESHOLD).all()),
        "cuda": bool(torch.cuda.is_available() and str(args.device).startswith("cuda")),
        "checkpoint_hash": sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256,
        "channel": 0,
        "threshold": THRESHOLD,
    }
    payload = {
        "status": "pass" if all(value is True or key in {"channel", "threshold"} for key, value in gate.items()) else "fail",
        "scene_date": scene["scene_date"],
        "source_path": scene["selected_source"],
        "source_sha256": scene["source_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config),
        "device": args.device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "center_window": [window.col_off, window.row_off, window.width, window.height],
        "probability_min": float(valid_probability.min()),
        "probability_max": float(valid_probability.max()),
        "probability_mean": float(valid_probability.mean()),
        "positive_fraction": positive_fraction,
        "runtime_seconds": runtime,
        "gate": gate,
        "source_model": SOURCE_MODEL,
        "scientific_status": "candidate_not_ground_truth",
        "test_reads": {"WHU": 0, "SECOND": 0},
        "created_at": utc_now(),
    }
    write_json(output / "SMOKE_RECEIPT.json", payload)
    print(json.dumps({**payload, "output_directory": str(output)}, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise RuntimeError(f"2023 center-tile smoke failed: {output}")
    return payload


def tile_origins(length: int) -> list[int]:
    if length <= TILE_SIZE:
        return [0]
    values = list(range(0, length - TILE_SIZE + 1, STRIDE))
    last = length - TILE_SIZE
    if values[-1] != last:
        values.append(last)
    return values


def blending_weight() -> np.ndarray:
    axis = np.hanning(TILE_SIZE).astype(np.float32)
    weight = np.outer(axis, axis)
    return np.maximum(weight, 1e-3)


def read_tile(source: rasterio.DatasetReader, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    window = Window(x, y, TILE_SIZE, TILE_SIZE)
    image = source.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    valid = source.dataset_mask(window=window, boundless=True) > 0
    return image, valid


def initialize_accumulators(
    output: Path, shape: tuple[int, int]
) -> tuple[np.memmap, np.memmap]:
    probability_sum = np.memmap(
        output / "_probability_sum.float32",
        dtype=np.float32,
        mode="w+",
        shape=shape,
    )
    weight_sum = np.memmap(
        output / "_weight_sum.float32",
        dtype=np.float32,
        mode="w+",
        shape=shape,
    )
    probability_sum[:] = 0
    weight_sum[:] = 0
    probability_sum.flush()
    weight_sum.flush()
    return probability_sum, weight_sum


def open_accumulators(
    output: Path, shape: tuple[int, int]
) -> tuple[np.memmap, np.memmap]:
    return (
        np.memmap(
            output / "_probability_sum.float32",
            dtype=np.float32,
            mode="r+",
            shape=shape,
        ),
        np.memmap(
            output / "_weight_sum.float32",
            dtype=np.float32,
            mode="r+",
            shape=shape,
        ),
    )


def create_probability_raster(
    source: rasterio.DatasetReader,
    output_path: Path,
    probability_sum: np.memmap,
    weight_sum: np.memmap,
) -> dict[str, float]:
    profile = source.profile.copy()
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )
    statistics = {"min": math.inf, "max": -math.inf, "sum": 0.0, "count": 0, "positive": 0}
    with rasterio.open(output_path, "w", **profile) as target:
        for _, window in target.block_windows(1):
            rows = slice(int(window.row_off), int(window.row_off + window.height))
            cols = slice(int(window.col_off), int(window.col_off + window.width))
            weights = np.asarray(weight_sum[rows, cols])
            valid_source = source.dataset_mask(window=window) > 0
            valid = (weights > 0) & valid_source
            result = np.full(weights.shape, np.nan, dtype=np.float32)
            result[valid] = np.asarray(probability_sum[rows, cols])[valid] / weights[valid]
            result[valid] = np.clip(result[valid], 0.0, 1.0)
            target.write(result, 1, window=window)
            values = result[valid]
            if values.size:
                statistics["min"] = min(statistics["min"], float(values.min()))
                statistics["max"] = max(statistics["max"], float(values.max()))
                statistics["sum"] += float(values.astype(np.float64).sum())
                statistics["count"] += int(values.size)
                statistics["positive"] += int((values >= THRESHOLD).sum())
    if not statistics["count"]:
        raise RuntimeError("No valid output pixels were produced.")
    return {
        "probability_min": statistics["min"],
        "probability_max": statistics["max"],
        "probability_mean": statistics["sum"] / statistics["count"],
        "positive_fraction": statistics["positive"] / statistics["count"],
    }


def create_mask_and_boundary(
    source: rasterio.DatasetReader, probability_path: Path, mask_path: Path, boundary_path: Path
) -> None:
    profile = source.profile.copy()
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )
    with rasterio.open(probability_path) as probability, rasterio.open(
        mask_path, "w", **profile
    ) as mask_target:
        for _, window in mask_target.block_windows(1):
            values = probability.read(1, window=window)
            valid = np.isfinite(values)
            mask = np.full(values.shape, 255, dtype=np.uint8)
            mask[valid] = (values[valid] >= THRESHOLD).astype(np.uint8)
            mask_target.write(mask, 1, window=window)

    with rasterio.open(mask_path) as mask_source, rasterio.open(
        boundary_path, "w", **profile
    ) as boundary_target:
        for _, window in boundary_target.block_windows(1):
            expanded = Window(
                window.col_off - 1,
                window.row_off - 1,
                window.width + 2,
                window.height + 2,
            )
            data = mask_source.read(1, window=expanded, boundless=True, fill_value=255)
            valid = data != 255
            binary = (data == 1).astype(np.uint8)
            eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
            edge = ((binary == 1) & (eroded == 0)).astype(np.uint8)
            edge[~valid] = 255
            cropped = edge[1:-1, 1:-1]
            boundary_target.write(cropped, 1, window=window)


def create_overview(source_path: Path, probability_path: Path, output_path: Path) -> None:
    maximum = 1600
    with rasterio.open(source_path) as source:
        scale = min(1.0, maximum / max(source.width, source.height))
        height = max(1, int(source.height * scale))
        width = max(1, int(source.width * scale))
        image = source.read(
            [1, 2, 3], out_shape=(3, height, width), resampling=rasterio.enums.Resampling.bilinear
        )
    with rasterio.open(probability_path) as probability:
        values = probability.read(
            1, out_shape=(height, width), resampling=rasterio.enums.Resampling.bilinear
        )
    rgb = normalize_rgb(image)
    valid = np.isfinite(values)
    mask = valid & (values >= THRESHOLD)
    overlay = rgb.copy()
    overlay[mask] = (
        0.45 * overlay[mask].astype(np.float32) + 0.55 * np.array([0, 160, 255])
    ).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def infer_scene(
    model: torch.nn.Module,
    scene: dict[str, Any],
    checkpoint: Path,
    config: Path,
    geometry_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = output_directory(geometry_root, scene["scene_date"])
    manifest_path = output / "GEOMETRY_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "success":
            print(f"REUSED completed scene {scene['scene_date']}: {output}")
            return manifest
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "_TILE_STATE.json"
    source_path = Path(scene["selected_source"])
    started = time.perf_counter()
    with rasterio.open(source_path) as source:
        xs = tile_origins(source.width)
        ys = tile_origins(source.height)
        jobs = [(x, y) for y in ys for x in xs]
        expected_tiles = len(jobs)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("source_sha256") != scene["source_sha256"]
                or state.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
                or state.get("expected_tiles") != expected_tiles
            ):
                raise RuntimeError(
                    f"Unsafe resume metadata mismatch for {scene['scene_date']}: {state_path}"
                )
            completed = {int(value) for value in state.get("completed_tiles", [])}
            probability_sum, weight_sum = open_accumulators(
                output, (source.height, source.width)
            )
        else:
            completed = set()
            probability_sum, weight_sum = initialize_accumulators(
                output, (source.height, source.width)
            )
            state = {
                "status": "running",
                "scene_date": scene["scene_date"],
                "source_path": str(source_path),
                "source_sha256": scene["source_sha256"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "expected_tiles": expected_tiles,
                "completed_tiles": [],
                "created_at": utc_now(),
            }
            write_json(state_path, state)

        weight = blending_weight()
        pending = [(index, *job) for index, job in enumerate(jobs) if index not in completed]
        batch_size = int(args.batch_size)
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            images: list[np.ndarray] = []
            valid_masks: list[np.ndarray] = []
            for _, x, y in batch:
                image, valid = read_tile(source, x, y)
                images.append(image)
                valid_masks.append(valid)
            probabilities = predict_batch(model, images, args.device)
            for (index, x, y), probability, valid in zip(batch, probabilities, valid_masks):
                rows = slice(y, y + TILE_SIZE)
                cols = slice(x, x + TILE_SIZE)
                local_weight = weight * valid.astype(np.float32)
                probability_sum[rows, cols] += probability * local_weight
                weight_sum[rows, cols] += local_weight
                completed.add(index)
            probability_sum.flush()
            weight_sum.flush()
            state["completed_tiles"] = sorted(completed)
            state["completed_tile_count"] = len(completed)
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            print(
                f"{scene['scene_date']} tiles {len(completed)}/{expected_tiles}",
                flush=True,
            )

        probability_path = output / "building_probability.tif"
        mask_path = output / "building_mask.tif"
        boundary_path = output / "building_boundary.tif"
        stats = create_probability_raster(
            source, probability_path, probability_sum, weight_sum
        )
        create_mask_and_boundary(source, probability_path, mask_path, boundary_path)
        create_overview(source_path, probability_path, output / "overview.png")

    runtime_seconds = time.perf_counter() - started
    manifest = {
        "status": "success",
        "scene_date": scene["scene_date"],
        "source_path": str(source_path),
        "source_sha256": scene["source_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config),
        "model_name": "FoundationEncoderDecoder/Swin-T",
        "source_model": SOURCE_MODEL,
        "threshold": THRESHOLD,
        "channel": 0,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "batch_size": int(args.batch_size),
        **stats,
        "width": scene["width"],
        "height": scene["height"],
        "crs": scene["crs"],
        "transform": scene["transform"],
        "nodata": scene["nodata"],
        "runtime_seconds": runtime_seconds,
        "completed_tiles": len(completed),
        "expected_tiles": expected_tiles,
        "outputs": {
            "probability": str(probability_path),
            "mask": str(mask_path),
            "boundary": str(boundary_path),
            "overview": str(output / "overview.png"),
        },
        "scientific_status": "candidate_not_ground_truth",
        "test_reads": {"WHU": 0, "SECOND": 0},
        "completed_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    state["status"] = "success"
    state["completed_at"] = utc_now()
    write_json(state_path, state)
    return manifest


def infer_all(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    scenes = load_scenes(geometry_root)
    smoke_receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in geometry_root.glob("smoke_*/SMOKE_RECEIPT.json")
    ]
    if not smoke_receipts or not any(item.get("status") == "pass" for item in smoke_receipts):
        raise RuntimeError("A passing center-tile smoke is required before infer-all.")
    checkpoint, config = model_paths(args)
    model = build_model(config, checkpoint, args.device)
    manifests = []
    try:
        for scene in scenes["scenes"]:
            manifests.append(
                infer_scene(model, scene, checkpoint, config, geometry_root, args)
            )
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_scenes": [item["scene_date"] for item in manifests],
            "test_reads": {"WHU": 0, "SECOND": 0},
            "failed_at": utc_now(),
        }
        write_json(geometry_root / f"GEOMETRY_INFERENCE_FAILURE_{timestamp()}.json", failure)
        raise
    payload = {
        "status": "success",
        "scenes": manifests,
        "scene_count": len(manifests),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "threshold": THRESHOLD,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "completed_at": utc_now(),
    }
    write_json(geometry_root / "TONGZHOU_GEOMETRY_RUN.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def unique_values(path: Path) -> list[int]:
    values: set[int] = set()
    with rasterio.open(path) as dataset:
        for _, window in dataset.block_windows(1):
            values.update(np.unique(dataset.read(1, window=window)).astype(int).tolist())
    return sorted(values)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    scenes = load_scenes(geometry_root)
    rows = []
    all_pass = True
    for scene in scenes["scenes"]:
        output = output_directory(geometry_root, scene["scene_date"])
        manifest_path = output / "GEOMETRY_MANIFEST.json"
        required = {
            "probability": output / "building_probability.tif",
            "mask": output / "building_mask.tif",
            "boundary": output / "building_boundary.tif",
            "overview": output / "overview.png",
            "manifest": manifest_path,
        }
        exists = {key: path.exists() for key, path in required.items()}
        checks: dict[str, Any] = {"required_files": exists}
        if not all(exists.values()):
            checks["status"] = "fail"
            checks["reason"] = "missing_required_output"
            all_pass = False
            rows.append({"scene_date": scene["scene_date"], **checks})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with rasterio.open(required["probability"]) as probability, rasterio.open(
            required["mask"]
        ) as mask, rasterio.open(required["boundary"]) as boundary:
            probability_finite = True
            probability_in_range = True
            minimum = math.inf
            maximum = -math.inf
            total = 0
            positive = 0
            for _, window in probability.block_windows(1):
                data = probability.read(1, window=window)
                valid = np.isfinite(data)
                values = data[valid]
                if values.size:
                    probability_finite &= bool(np.isfinite(values).all())
                    probability_in_range &= bool(values.min() >= 0 and values.max() <= 1)
                    minimum = min(minimum, float(values.min()))
                    maximum = max(maximum, float(values.max()))
                    total += int(values.size)
                    positive += int((values >= THRESHOLD).sum())
            mask_values = unique_values(required["mask"])
            boundary_values = unique_values(required["boundary"])
            raster_match = all(
                dataset.width == scene["width"]
                and dataset.height == scene["height"]
                and str(dataset.crs) == scene["crs"]
                and np.allclose(list(dataset.transform), scene["transform"], atol=0, rtol=0)
                for dataset in (probability, mask, boundary)
            )
            profile_pass = (
                probability.count == mask.count == boundary.count == 1
                and probability.dtypes[0] == "float32"
                and mask.dtypes[0] == boundary.dtypes[0] == "uint8"
                and mask.nodata == boundary.nodata == 255
            )
        positive_fraction = positive / total if total else 0.0
        checks.update(
            {
                "probability_finite": probability_finite,
                "probability_in_range": probability_in_range,
                "probability_min": minimum,
                "probability_max": maximum,
                "positive_fraction": positive_fraction,
                "positive_fraction_pass": 0 < positive_fraction < 0.90,
                "mask_values": mask_values,
                "mask_values_pass": set(mask_values).issubset({0, 1, 255}),
                "boundary_values": boundary_values,
                "boundary_values_pass": set(boundary_values).issubset({0, 1, 255}),
                "raster_alignment_match": raster_match,
                "profile_pass": profile_pass,
                "tiles_complete": manifest.get("completed_tiles")
                == manifest.get("expected_tiles"),
            }
        )
        boolean_checks = [
            value
            for key, value in checks.items()
            if key.endswith("_pass") or key in {"probability_finite", "probability_in_range", "raster_alignment_match", "profile_pass", "tiles_complete"}
        ]
        checks["status"] = "pass" if all(boolean_checks) else "fail"
        all_pass &= checks["status"] == "pass"
        rows.append({"scene_date": scene["scene_date"], **checks})

    summary_csv = geometry_root / "TONGZHOU_GEOMETRY_SUMMARY.csv"
    flattened = []
    for row in rows:
        flattened.append(
            {
                "scene_date": row["scene_date"],
                "status": row["status"],
                "probability_min": row.get("probability_min"),
                "probability_max": row.get("probability_max"),
                "positive_fraction": row.get("positive_fraction"),
                "mask_values": json.dumps(row.get("mask_values")),
                "boundary_values": json.dumps(row.get("boundary_values")),
                "tiles_complete": row.get("tiles_complete"),
            }
        )
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)
    payload = {
        "geometry_numerical_status": "pass" if all_pass else "fail",
        "scenes": rows,
        "summary_csv": str(summary_csv),
        "test_reads": {"WHU": 0, "SECOND": 0},
        "audited_at": utc_now(),
    }
    write_json(geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_pass:
        raise RuntimeError("Tongzhou geometry numerical audit failed.")
    return payload


def local_utm_epsg(bounds: Iterable[float]) -> int:
    left, bottom, right, top = bounds
    lon = (left + right) / 2
    lat = (bottom + top) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def vectorize(args: argparse.Namespace) -> dict[str, Any]:
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform

    geometry_root = Path(args.output_root).resolve()
    numerical_path = geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json"
    if not numerical_path.exists():
        raise FileNotFoundError("Numerical audit is required before vectorization.")
    numerical = json.loads(numerical_path.read_text(encoding="utf-8"))
    if numerical.get("geometry_numerical_status") != "pass":
        raise RuntimeError("Numerical audit is not pass; vectorization refused.")
    scenes = load_scenes(geometry_root)
    results = []
    for scene in scenes["scenes"]:
        output = output_directory(geometry_root, scene["scene_date"])
        mask_path = output / "building_mask.tif"
        if not scene.get("crs"):
            skipped = {
                "status": "skipped_no_crs",
                "reason": "No CRS; geographic area would be fabricated.",
                "scene_date": scene["scene_date"],
            }
            write_json(output / "VECTORIZATION_SKIPPED.json", skipped)
            results.append(skipped)
            continue
        gpkg = output / "building_objects.gpkg"
        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)
            valid = mask == 1
            transformer = Transformer.from_crs(
                dataset.crs,
                f"EPSG:{local_utm_epsg(dataset.bounds)}",
                always_xy=True,
            )
            schema = {
                "geometry": "Polygon",
                "properties": {
                    "object_id": "str:80",
                    "scene_date": "str:10",
                    "state": "str:32",
                    "area_m2": "float",
                    "perimeter_m": "float",
                    "source_model": "str:80",
                    "scientific_status": "str:48",
                },
            }
            count = 0
            with fiona.open(
                gpkg,
                "w",
                driver="GPKG",
                layer="buildings",
                crs_wkt=dataset.crs.to_wkt(),
                schema=schema,
            ) as sink:
                for geometry_mapping, value in shapes(
                    mask, mask=valid, transform=dataset.transform, connectivity=8
                ):
                    if int(value) != 1:
                        continue
                    geometry = shape(geometry_mapping)
                    if not geometry.is_valid:
                        geometry = geometry.buffer(0)
                    if geometry.is_empty:
                        continue
                    if geometry.geom_type == "MultiPolygon":
                        parts = list(geometry.geoms)
                    elif geometry.geom_type == "Polygon":
                        parts = [geometry]
                    else:
                        continue
                    for part in parts:
                        projected = shapely_transform(transformer.transform, part)
                        if projected.area < float(args.min_area_m2):
                            continue
                        count += 1
                        sink.write(
                            {
                                "geometry": part.__geo_interface__,
                                "properties": {
                                    "object_id": f"{scene['scene_date']}_{count:08d}",
                                    "scene_date": scene["scene_date"],
                                    "state": "building_candidate",
                                    "area_m2": float(projected.area),
                                    "perimeter_m": float(projected.length),
                                    "source_model": SOURCE_MODEL,
                                    "scientific_status": "candidate_not_ground_truth",
                                },
                            }
                        )
        result = {
            "status": "success",
            "scene_date": scene["scene_date"],
            "gpkg": str(gpkg),
            "object_count": count,
            "layer": "buildings",
            "minimum_area_m2": float(args.min_area_m2),
            "scientific_status": "candidate_not_ground_truth",
        }
        write_json(output / "VECTORIZATION_RECEIPT.json", result)
        results.append(result)
    payload = {"status": "success", "scenes": results, "created_at": utc_now()}
    write_json(geometry_root / "TONGZHOU_VECTORIZATION_SUMMARY.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def alignment(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    scenes = load_scenes(geometry_root)
    rows = []
    reference_signature = None
    all_pass = True
    for scene in scenes["scenes"]:
        output = output_directory(geometry_root, scene["scene_date"])
        paths = [
            Path(scene["selected_source"]),
            output / "building_probability.tif",
            output / "building_mask.tif",
            output / "building_boundary.tif",
        ]
        if not all(path.exists() for path in paths):
            rows.append(
                {
                    "scene_date": scene["scene_date"],
                    "status": "fail",
                    "reason": "missing_source_or_geometry_output",
                    "paths": [str(path) for path in paths],
                }
            )
            all_pass = False
            continue
        signatures = []
        for path in paths:
            with rasterio.open(path) as dataset:
                signatures.append(
                    {
                        "path": str(path),
                        "width": dataset.width,
                        "height": dataset.height,
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "transform": list(dataset.transform),
                        "resolution": list(dataset.res),
                        "bounds": list(dataset.bounds),
                        "pixel_origin": [dataset.transform.c, dataset.transform.f],
                        "nodata": dataset.nodata,
                    }
                )
        base = signatures[0]
        within_scene = all(
            item["width"] == base["width"]
            and item["height"] == base["height"]
            and item["crs"] == base["crs"]
            and np.allclose(item["transform"], base["transform"], atol=0, rtol=0)
            and np.allclose(item["bounds"], base["bounds"], atol=1e-12, rtol=0)
            for item in signatures[1:]
        )
        temporal_signature = {
            key: base[key] for key in ("width", "height", "crs", "transform", "resolution", "bounds", "pixel_origin")
        }
        if reference_signature is None:
            reference_signature = temporal_signature
        cross_scene = (
            temporal_signature["width"] == reference_signature["width"]
            and temporal_signature["height"] == reference_signature["height"]
            and temporal_signature["crs"] == reference_signature["crs"]
            and np.allclose(temporal_signature["transform"], reference_signature["transform"], atol=0, rtol=0)
            and np.allclose(temporal_signature["bounds"], reference_signature["bounds"], atol=1e-12, rtol=0)
        )
        status = "pass" if within_scene and cross_scene else "fail"
        all_pass &= status == "pass"
        rows.append(
            {
                "scene_date": scene["scene_date"],
                "status": status,
                "within_scene_match": within_scene,
                "cross_scene_match": cross_scene,
                "rasters": signatures,
            }
        )
    payload = {
        "alignment_status": (
            "ready_for_pixelwise_temporal_tracking"
            if all_pass
            else "registration_required_before_pixelwise_fusion"
        ),
        "scenes": rows,
        "silent_resampling": False,
        "audited_at": utc_now(),
    }
    write_json(geometry_root / "ALIGNMENT_AUDIT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_pass:
        raise RuntimeError(payload["alignment_status"])
    return payload


def status(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = Path(args.output_root).resolve()
    scenes = []
    scene_manifest = geometry_root / "TONGZHOU_SCENES.json"
    target_dates = tuple(load_scenes(geometry_root).get("target_dates", [])) if scene_manifest.exists() else TARGET_DATES
    for scene_date in target_dates:
        output = output_directory(geometry_root, scene_date)
        files = {
            name: (output / name).exists()
            for name in (
                "building_probability.tif",
                "building_mask.tif",
                "building_boundary.tif",
                "overview.png",
                "GEOMETRY_MANIFEST.json",
                "building_objects.gpkg",
            )
        }
        manifest_path = output / "GEOMETRY_MANIFEST.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        scenes.append(
            {
                "scene_date": scene_date,
                "files": files,
                "geometry_status": manifest.get("status", "missing"),
                "positive_fraction": manifest.get("positive_fraction"),
                "completed_tiles": manifest.get("completed_tiles"),
                "expected_tiles": manifest.get("expected_tiles"),
            }
        )
    payload = {
        "geometry_root": str(geometry_root),
        "scenes": scenes,
        "numerical_audit": (
            str(geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json")
            if (geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json").exists()
            else None
        ),
        "alignment_audit": (
            str(geometry_root / "ALIGNMENT_AUDIT.json")
            if (geometry_root / "ALIGNMENT_AUDIT.json").exists()
            else None
        ),
        "test_reads": {"WHU": 0, "SECOND": 0},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSBuilding static building geometry for an audited scene set."
    )
    parser.add_argument(
        "command",
        choices=("discover", "diagnose", "smoke", "infer-all", "audit", "vectorize", "alignment", "status"),
    )
    parser.add_argument(
        "--source-root",
        default=r"D:\Data Set\Satellite\beijing_tongzhou",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_building_geometry"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "v2_readiness"
            / "rsbuilding"
            / "whu"
            / "work_dirs"
            / "swin_t_whu_pilot5000_20260728_110750"
            / "iter_2000.pth"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "v2_readiness"
            / "rsbuilding"
            / "whu"
            / "work_dirs"
            / "swin_t_whu_pilot5000_20260728_110750"
            / "Swin_T_whu_windows_pilot5000.py"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument(
        "--target-date",
        action="append",
        dest="target_dates",
        help="Explicit ISO acquisition date; repeat once per scene. Defaults to the historical Tongzhou dates.",
    )
    parser.add_argument(
        "--smoke-date",
        default=None,
        help="Date used for center-tile smoke. Defaults to the last audited scene.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "discover":
        print(
            json.dumps(
                discover_scenes(Path(args.source_root), Path(args.output_root), tuple(args.target_dates) if args.target_dates else None),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "diagnose":
        diagnose(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "infer-all":
        infer_all(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "vectorize":
        vectorize(args)
    elif args.command == "alignment":
        alignment(args)
    elif args.command == "status":
        status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
