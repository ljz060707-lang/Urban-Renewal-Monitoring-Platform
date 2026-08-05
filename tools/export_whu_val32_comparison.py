from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner


try:
    from opencd.utils import register_all_modules

    register_all_modules(
        init_default_scope=True
    )
except Exception:
    import opencd  # noqa: F401


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export WHU Validation comparison without "
            "using the incompatible CDVisualizationHook."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--master",
        required=True,
    )

    parser.add_argument(
        "--pretrained",
        required=True,
    )

    parser.add_argument(
        "--selected",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()

    encodings = [
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "gb18030",
    ]

    for encoding in encodings:
        try:
            text = raw.decode(
                encoding,
                errors="strict",
            )

            return text.replace(
                "\x00",
                "",
            )
        except UnicodeDecodeError:
            continue

    return raw.decode(
        "utf-8",
        errors="ignore",
    ).replace(
        "\x00",
        "",
    )


def resolve_manifest_path(
    value: str,
    parent: Path,
) -> Path:

    value = value.strip().lstrip(
        "\ufeff"
    )

    path = Path(value)

    if not path.is_absolute():
        path = parent / path

    return path.resolve()


def read_manifest(
    master_path: Path,
) -> list[dict[str, str]]:

    master_text = read_text_auto(
        master_path
    )

    specific_lists = [
        line.strip()
        for line in master_text.splitlines()
        if line.strip()
    ]

    if not specific_lists:
        raise RuntimeError(
            f"Empty master manifest: {master_path}"
        )

    rows: list[dict[str, str]] = []

    for specific_value in specific_lists:
        specific_path = resolve_manifest_path(
            specific_value,
            master_path.parent,
        )

        lower = specific_path.as_posix().lower()

        if "/test/" in lower:
            raise RuntimeError(
                f"WHU Test leak: {specific_path}"
            )

        if not specific_path.exists():
            raise FileNotFoundError(
                specific_path
            )

        specific_text = read_text_auto(
            specific_path
        )

        for line_number, line in enumerate(
            specific_text.splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            fields = line.rstrip(
                "\r\n"
            ).split("\t")

            if len(fields) != 5:
                raise RuntimeError(
                    f"Expected five TAB fields at "
                    f"{specific_path}:{line_number}; "
                    f"got {len(fields)}"
                )

            image_path = Path(
                fields[0].strip()
            )

            label_path = Path(
                fields[3].strip()
            )

            combined = (
                image_path.as_posix().lower()
                + " "
                + label_path.as_posix().lower()
            )

            if "/test/" in combined:
                raise RuntimeError(
                    "WHU Test path found in row "
                    f"{specific_path}:{line_number}"
                )

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            if not label_path.exists():
                raise FileNotFoundError(
                    label_path
                )

            rows.append({
                "sample_id": (
                    f"{len(rows):03d}_"
                    f"{image_path.stem}"
                ),
                "image_path": str(
                    image_path
                ),
                "label_path": str(
                    label_path
                ),
            })

    return rows


def field_value(
    sample: Any,
    field_name: str,
) -> Any:

    if isinstance(sample, dict):
        return sample.get(
            field_name
        )

    if hasattr(
        sample,
        field_name,
    ):
        return getattr(
            sample,
            field_name,
        )

    try:
        return sample.get(
            field_name,
            None,
        )
    except Exception:
        return None


def tensor_from_field(
    value: Any,
) -> torch.Tensor | None:

    if value is None:
        return None

    if isinstance(
        value,
        torch.Tensor,
    ):
        return value

    if hasattr(
        value,
        "data",
    ):
        data = value.data

        if isinstance(
            data,
            torch.Tensor,
        ):
            return data

    if isinstance(
        value,
        (list, tuple),
    ) and value:

        return tensor_from_field(
            value[0]
        )

    return None


def prediction_mask(
    output: Any,
) -> np.ndarray:
    """Extract WHU static-building prediction.

    RSBuilding returns three independent binary channels:

    channel 0 = building A
    channel 1 = building B
    channel 2 = change

    WHU static-building evaluation uses building A. These channels
    must not be treated as mutually exclusive semantic classes.
    """

    tensor = None
    source_name = None

    pred = field_value(
        output,
        "pred_sem_seg",
    )

    tensor = tensor_from_field(
        pred
    )

    if tensor is not None:
        source_name = "pred_sem_seg"

    if tensor is None:
        for field_name in (
            "seg_logits",
            "building_logits",
            "pred_logits",
        ):
            tensor = tensor_from_field(
                field_value(
                    output,
                    field_name,
                )
            )

            if tensor is not None:
                source_name = field_name
                break

    if tensor is None:
        available = []

        if hasattr(
            output,
            "to_dict",
        ):
            try:
                available = list(
                    output.to_dict().keys()
                )
            except Exception:
                pass

        raise RuntimeError(
            "Could not find prediction tensor. "
            f"Available fields: {available}"
        )

    array = (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # Remove batch dimensions only.
    while (
        array.ndim > 3
        and array.shape[0] == 1
    ):
        array = array[0]

    if array.ndim == 3:
        channels = array.shape[0]

        if channels == 3:
            # RSBuilding channel order:
            # 0 building A, 1 building B, 2 change.
            building = array[0]

        elif channels == 1:
            building = array[0]

        else:
            raise RuntimeError(
                "Unexpected RSBuilding channel count: "
                f"{channels}; shape={array.shape}"
            )

    elif array.ndim == 2:
        building = array

    else:
        raise RuntimeError(
            "Unsupported prediction shape: "
            f"{array.shape}"
        )

    finite = building[
        np.isfinite(building)
    ]

    if finite.size == 0:
        raise RuntimeError(
            "Prediction contains no finite values."
        )

    minimum = float(
        finite.min()
    )

    maximum = float(
        finite.max()
    )

    # pred_sem_seg is already thresholded.
    # seg_logits in the postprocessed SegDataSample is already sigmoid.
    if minimum >= 0.0 and maximum <= 1.0:
        probability = building
    else:
        probability = 1.0 / (
            1.0
            + np.exp(-building)
        )

    mask = (
        probability >= 0.5
    ).astype(
        np.uint8
    )

    print(
        "prediction source =",
        source_name,
        "shape =",
        array.shape,
        "building min/max =",
        minimum,
        maximum,
        "positive pixels =",
        int(mask.sum()),
    )

    return mask


def build_runner(
    config_path: Path,
    checkpoint: Path,
    master: Path,
    work_dir: Path,
) -> Runner:

    cfg = Config.fromfile(
        str(config_path)
    )

    cfg.load_from = str(
        checkpoint
    )

    cfg.resume = False
    cfg.work_dir = str(
        work_dir
    )

    cfg.test_dataloader[
        "dataset"
    ][
        "data_list"
    ] = master.resolve().as_posix()

    cfg.test_dataloader[
        "batch_size"
    ] = 1

    cfg.test_dataloader[
        "num_workers"
    ] = 0

    cfg.test_dataloader[
        "persistent_workers"
    ] = False

    cfg.test_cfg = dict(
        type="TestLoop"
    )

    # Test-only runner.
    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None

    # Do not call the incompatible visualization hook.
    if (
        "default_hooks" in cfg
        and "visualization" in cfg.default_hooks
    ):
        cfg.default_hooks[
            "visualization"
        ][
            "draw"
        ] = False

    config_text = cfg.pretty_text.lower()

    for forbidden in (
        "data_list/whu/test.txt",
        "data_list\\whu\\test.txt",
        "\\test\\image",
        "/test/image",
        "\\test\\label",
        "/test/label",
    ):
        if forbidden in config_text:
            raise RuntimeError(
                "WHU Test leak in resolved config: "
                + forbidden
            )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = Runner.from_cfg(
        cfg
    )

    runner.load_or_resume()

    runner.model.eval()

    return runner


def infer_checkpoint(
    config_path: Path,
    checkpoint: Path,
    master: Path,
    rows: list[dict[str, str]],
    work_dir: Path,
) -> list[np.ndarray]:

    print()
    print("=" * 80)
    print(
        "INFERENCE CHECKPOINT:",
        checkpoint,
    )
    print("=" * 80)

    runner = build_runner(
        config_path=config_path,
        checkpoint=checkpoint,
        master=master,
        work_dir=work_dir,
    )

    dataloader = (
        runner.test_loop.dataloader
    )

    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for index, data_batch in enumerate(
            dataloader
        ):
            outputs = runner.model.test_step(
                data_batch
            )

            if not isinstance(
                outputs,
                (list, tuple),
            ):
                outputs = [
                    outputs
                ]

            if len(outputs) != 1:
                raise RuntimeError(
                    "Expected batch size one, "
                    f"got {len(outputs)} outputs."
                )

            mask = prediction_mask(
                outputs[0]
            )

            predictions.append(
                mask
            )

            print(
                f"[{index + 1:02d}/{len(rows):02d}] "
                f"{rows[index]['sample_id']} "
                f"prediction shape={mask.shape}"
            )

    if len(predictions) != len(rows):
        raise RuntimeError(
            "Prediction count mismatch: "
            f"{len(predictions)} vs {len(rows)}"
        )

    del runner

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    tp = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            np.logical_not(target),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(prediction),
            target,
        ).sum()
    )

    tn = int(
        np.logical_and(
            np.logical_not(prediction),
            np.logical_not(target),
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 1.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn
        else 1.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    colour: tuple[int, int, int],
) -> np.ndarray:

    output = image.copy()

    coloured = np.zeros_like(
        image
    )

    coloured[:] = colour

    selected = mask.astype(
        bool
    )

    output[selected] = (
        0.45
        * image[selected].astype(
            np.float32
        )
        + 0.55
        * coloured[selected].astype(
            np.float32
        )
    ).astype(
        np.uint8
    )

    return output


def error_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    tp = np.logical_and(
        prediction,
        target,
    )

    fp = np.logical_and(
        prediction,
        np.logical_not(target),
    )

    fn = np.logical_and(
        np.logical_not(prediction),
        target,
    )

    output = (
        image.astype(np.float32)
        * 0.45
    ).astype(
        np.uint8
    )

    # BGR colours:
    # TP green, FP red, FN blue.
    output[tp] = (
        0,
        190,
        0,
    )

    output[fp] = (
        0,
        0,
        255,
    )

    output[fn] = (
        255,
        0,
        0,
    )

    return output


def titled_panel(
    image: np.ndarray,
    title: str,
) -> np.ndarray:

    height, width = image.shape[:2]

    header = np.full(
        (
            42,
            width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        title,
        (
            10,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            20,
            20,
            20,
        ),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([
        header,
        image,
    ])


def safe_name(
    value: str,
) -> str:

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )


def relative_uri(
    path: Path,
    root: Path,
) -> str:

    return path.relative_to(
        root
    ).as_posix()


def main() -> None:
    args = parse_args()

    config_path = Path(
        args.config
    ).resolve()

    master_path = Path(
        args.master
    ).resolve()

    pretrained_checkpoint = Path(
        args.pretrained
    ).resolve()

    selected_checkpoint = Path(
        args.selected
    ).resolve()

    output_root = Path(
        args.output
    ).resolve()

    for required in (
        config_path,
        master_path,
        pretrained_checkpoint,
        selected_checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(
                required
            )

    rows = read_manifest(
        master_path
    )

    if len(rows) != 32:
        raise RuntimeError(
            "Expected 32 WHU Validation samples; "
            f"found {len(rows)}."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dir = (
        output_root
        / "samples"
    )

    mask_pre_dir = (
        output_root
        / "masks"
        / "pretrained"
    )

    mask_sel_dir = (
        output_root
        / "masks"
        / "iter_2000"
    )

    for folder in (
        sample_dir,
        mask_pre_dir,
        mask_sel_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    pretrained_predictions = infer_checkpoint(
        config_path=config_path,
        checkpoint=pretrained_checkpoint,
        master=master_path,
        rows=rows,
        work_dir=(
            output_root
            / "_runner_pretrained"
        ),
    )

    selected_predictions = infer_checkpoint(
        config_path=config_path,
        checkpoint=selected_checkpoint,
        master=master_path,
        rows=rows,
        work_dir=(
            output_root
            / "_runner_iter2000"
        ),
    )

    csv_rows = []

    aggregate = {
        "pretrained": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        },
        "iter_2000": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        },
    }

    html_cards = []

    for index, row in enumerate(
        rows
    ):
        image_path = Path(
            row["image_path"]
        )

        label_path = Path(
            row["label_path"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        target_raw = cv2.imread(
            str(label_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if target_raw is None:
            raise RuntimeError(
                f"Could not read label: {label_path}"
            )

        target = (
            target_raw >= 128
        ).astype(
            np.uint8
        )

        pretrained = pretrained_predictions[
            index
        ]

        selected = selected_predictions[
            index
        ]

        height, width = target.shape

        if image.shape[:2] != (
            height,
            width,
        ):
            image = cv2.resize(
                image,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        if pretrained.shape != target.shape:
            pretrained = cv2.resize(
                pretrained,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        if selected.shape != target.shape:
            selected = cv2.resize(
                selected,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        pretrained = (
            pretrained > 0
        ).astype(
            np.uint8
        )

        selected = (
            selected > 0
        ).astype(
            np.uint8
        )

        pre_metrics = binary_metrics(
            pretrained,
            target,
        )

        sel_metrics = binary_metrics(
            selected,
            target,
        )

        for key in (
            "tp",
            "fp",
            "fn",
            "tn",
        ):
            aggregate[
                "pretrained"
            ][key] += int(
                pre_metrics[key]
            )

            aggregate[
                "iter_2000"
            ][key] += int(
                sel_metrics[key]
            )

        sample_name = safe_name(
            row["sample_id"]
        )

        pre_mask_path = (
            mask_pre_dir
            / f"{sample_name}.png"
        )

        sel_mask_path = (
            mask_sel_dir
            / f"{sample_name}.png"
        )

        cv2.imwrite(
            str(pre_mask_path),
            pretrained * 255,
        )

        cv2.imwrite(
            str(sel_mask_path),
            selected * 255,
        )

        original_panel = titled_panel(
            image,
            "Original",
        )

        gt_panel = titled_panel(
            overlay_mask(
                image,
                target,
                (
                    255,
                    255,
                    0,
                ),
            ),
            "GT Building",
        )

        pre_panel = titled_panel(
            overlay_mask(
                image,
                pretrained,
                (
                    0,
                    255,
                    255,
                ),
            ),
            (
                "Pretrained Prediction "
                f"IoU={pre_metrics['iou']:.3f}"
            ),
        )

        sel_panel = titled_panel(
            overlay_mask(
                image,
                selected,
                (
                    0,
                    255,
                    255,
                ),
            ),
            (
                "Iter2000 Prediction "
                f"IoU={sel_metrics['iou']:.3f}"
            ),
        )

        pre_error_panel = titled_panel(
            error_overlay(
                image,
                pretrained,
                target,
            ),
            "Pretrained Error: TP green / FP red / FN blue",
        )

        sel_error_panel = titled_panel(
            error_overlay(
                image,
                selected,
                target,
            ),
            "Iter2000 Error: TP green / FP red / FN blue",
        )

        row_one = np.hstack([
            original_panel,
            gt_panel,
            pre_panel,
        ])

        row_two = np.hstack([
            sel_panel,
            pre_error_panel,
            sel_error_panel,
        ])

        comparison = np.vstack([
            row_one,
            row_two,
        ])

        comparison_path = (
            sample_dir
            / f"{sample_name}.jpg"
        )

        cv2.imwrite(
            str(comparison_path),
            comparison,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        csv_rows.append({
            "sample_id": row[
                "sample_id"
            ],
            "image_path": str(
                image_path
            ),
            "label_path": str(
                label_path
            ),
            "pretrained_iou": (
                pre_metrics["iou"]
            ),
            "pretrained_f1": (
                pre_metrics["f1"]
            ),
            "pretrained_precision": (
                pre_metrics["precision"]
            ),
            "pretrained_recall": (
                pre_metrics["recall"]
            ),
            "iter2000_iou": (
                sel_metrics["iou"]
            ),
            "iter2000_f1": (
                sel_metrics["f1"]
            ),
            "iter2000_precision": (
                sel_metrics["precision"]
            ),
            "iter2000_recall": (
                sel_metrics["recall"]
            ),
            "iou_delta_iter2000_minus_pretrained": (
                sel_metrics["iou"]
                - pre_metrics["iou"]
            ),
            "preferred": "",
            "boundary_quality": "",
            "false_positive_comment": "",
            "false_negative_comment": "",
            "fragmentation_comment": "",
            "notes": "",
            "comparison_image": str(
                comparison_path
            ),
        })

        html_cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(row["sample_id"])}</h2>
              <p>
                Pretrained IoU:
                {pre_metrics["iou"]:.4f}
                &nbsp;|&nbsp;
                Iter2000 IoU:
                {sel_metrics["iou"]:.4f}
                &nbsp;|&nbsp;
                Δ:
                {sel_metrics["iou"] - pre_metrics["iou"]:+.4f}
              </p>
              <a href="{html.escape(relative_uri(comparison_path, output_root))}">
                <img
                  loading="lazy"
                  src="{html.escape(relative_uri(comparison_path, output_root))}"
                  alt="{html.escape(row["sample_id"])} comparison"
                >
              </a>
            </section>
            """
        )

    def aggregate_metrics(
        counts: dict[str, int],
    ) -> dict[str, float | int]:

        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 1.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 1.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        iou = (
            tp / (tp + fp + fn)
            if tp + fp + fn
            else 1.0
        )

        return {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    pretrained_total = aggregate_metrics(
        aggregate["pretrained"]
    )

    selected_total = aggregate_metrics(
        aggregate["iter_2000"]
    )

    csv_path = (
        output_root
        / "MANUAL_REVIEW.csv"
    )

    fields = list(
        csv_rows[0].keys()
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    summary = {
        "review_split": (
            "WHU Validation smoke subset"
        ),
        "sample_count": len(rows),
        "whu_test_used": False,
        "whu_test_pixel_reads": 0,
        "checkpoint_selection_split": (
            "WHU Validation 1036"
        ),
        "pretrained": {
            "checkpoint": str(
                pretrained_checkpoint
            ),
            "sha256": sha256_file(
                pretrained_checkpoint
            ),
            "val32_pixel_metrics": (
                pretrained_total
            ),
        },
        "iter_2000": {
            "checkpoint": str(
                selected_checkpoint
            ),
            "sha256": sha256_file(
                selected_checkpoint
            ),
            "val32_pixel_metrics": (
                selected_total
            ),
        },
        "status": (
            "qualitative_review_pending"
        ),
        "inference_ready": False,
        "note": (
            "Val32 is for qualitative review only. "
            "Full WHU Validation 1036 was used for "
            "checkpoint selection."
        ),
    }

    summary_path = (
        output_root
        / "REVIEW_SUMMARY.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html_path = (
        output_root
        / "WHU_VAL32_COMPARISON.html"
    )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WHU Swin-T Validation Review</title>
<style>
body {{
  max-width: 1500px;
  margin: 0 auto;
  padding: 24px;
  background: #f2f2f5;
  color: #222;
  font-family: Arial, "Microsoft YaHei", sans-serif;
}}
header,
.card {{
  background: white;
  border-radius: 12px;
  padding: 18px;
  margin-bottom: 18px;
}}
.warning {{
  background: #fff4cc;
  border-radius: 8px;
  padding: 12px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(2,minmax(260px,1fr));
  gap: 14px;
}}
.metric {{
  border: 1px solid #ddd;
  border-radius: 9px;
  padding: 12px;
}}
img {{
  width: 100%;
  height: auto;
  display: block;
}}
@media (max-width:800px) {{
  .metrics {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<header>
<h1>WHU Swin-T Validation Review</h1>
<p class="warning">
仅使用WHU Validation的32张固定样本。
WHU Test未使用，pixel reads=0。
此页面只用于定性审查，不用于再次选择checkpoint。
</p>
<div class="metrics">
<div class="metric">
<h2>Pretrained</h2>
<p>IoU: {pretrained_total["iou"]:.4f}</p>
<p>F1: {pretrained_total["f1"]:.4f}</p>
<p>Precision: {pretrained_total["precision"]:.4f}</p>
<p>Recall: {pretrained_total["recall"]:.4f}</p>
</div>
<div class="metric">
<h2>WHU iter_2000</h2>
<p>IoU: {selected_total["iou"]:.4f}</p>
<p>F1: {selected_total["f1"]:.4f}</p>
<p>Precision: {selected_total["precision"]:.4f}</p>
<p>Recall: {selected_total["recall"]:.4f}</p>
</div>
</div>
<p>
错误图颜色：
TP=绿色，FP=红色，FN=蓝色。
</p>
</header>
{''.join(html_cards)}
</body>
</html>
"""

    html_path.write_text(
        document,
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("WHU VAL32 COMPARISON EXPORT: PASS")
    print("=" * 90)
    print("samples =", len(rows))
    print(
        "pretrained aggregate =",
        pretrained_total,
    )
    print(
        "iter2000 aggregate =",
        selected_total,
    )
    print("HTML =", html_path)
    print("CSV =", csv_path)
    print("SUMMARY =", summary_path)
    print("WHU TEST READS = 0")
    print("INFERENCE READY = FALSE")


if __name__ == "__main__":
    main()