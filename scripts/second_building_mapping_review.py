from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

LABEL_TOKENS = {
    "label",
    "labels",
    "mask",
    "masks",
    "gt",
    "semantic",
    "semantic_label",
    "semantic_labels",
}

TEST_TOKENS = {
    "test",
    "testing",
}

VAL_TOKENS = {
    "val",
    "valid",
    "validation",
}


def parts_lower(path: Path) -> set[str]:
    return {
        p.lower()
        for p in path.parts
    }


def is_test(path: Path) -> bool:
    return bool(
        parts_lower(path)
        & TEST_TOKENS
    )


def is_validation(path: Path) -> bool:
    return bool(
        parts_lower(path)
        & VAL_TOKENS
    )


def is_label_candidate(path: Path) -> bool:
    """
    Recognize SECOND semantic label directories/files such as:

    label
    labels
    label1
    label2
    label1_rgb
    label2_rgb
    semantic_label
    gt

    Does not inspect Test contents.
    """

    if is_test(path):
        return False

    parts = [
        p.lower()
        for p in path.parts
    ]

    stem = path.stem.lower()

    for part in parts:

        if re.search(
            r"(^|[_\-])labels?([12])?($|[_\-])",
            part,
        ):
            return True

        if re.search(
            r"(^|[_\-])(mask|masks|gt|semantic)([12])?($|[_\-])",
            part,
        ):
            return True

        # Common SECOND forms:
        # label1 / label2 / label1_rgb / label2_rgb
        if part.startswith(
            (
                "label1",
                "label2",
                "semantic1",
                "semantic2",
            )
        ):
            return True

    if re.search(
        r"(^|[_\-])(label|mask|semantic|gt)[_\-]?[12]?($|[_\-])",
        stem,
    ):
        return True

    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_label(path: Path) -> np.ndarray | None:
    """
    Read a SECOND semantic ID mask.

    Important:
    - Palette PNG (P mode) is preserved as its integer class IDs.
    - Single-channel masks are returned directly.
    - RGB labels are NOT mapped here, because we must not guess
      RGB -> semantic class mappings.
    - Test paths must already have been excluded before this call.
    """

    try:
        img = Image.open(path)

        # -----------------------------------------------------
        # Palette PNG
        #
        # PIL mode "P" is important:
        # np.asarray() returns palette indices, e.g. 0..6,
        # rather than converting them to RGB.
        # -----------------------------------------------------

        if img.mode == "P":
            arr = np.asarray(img)

            if arr.ndim == 2:
                return arr.astype(
                    np.int32,
                    copy=False,
                )

        # -----------------------------------------------------
        # Native grayscale / integer masks
        # -----------------------------------------------------

        arr = np.asarray(img)

        if arr.ndim == 2:
            return arr.astype(
                np.int32,
                copy=False,
            )

        if (
            arr.ndim == 3
            and arr.shape[2] == 1
        ):
            return arr[:, :, 0].astype(
                np.int32,
                copy=False,
            )

        # -----------------------------------------------------
        # RGB labels:
        # DO NOT invent RGB -> class ID mapping.
        # -----------------------------------------------------

        return None

    except Exception as exc:
        print(
            "[WARN] cannot read label:",
            path,
            exc,
        )

        return None

    except Exception as exc:

        print(
            "[WARN] cannot read label:",
            path,
            exc,
        )

        return None


def normalize_rgb(
    path: Path,
    target_size: tuple[int, int],
) -> Image.Image | None:

    try:
        img = Image.open(path)

        if img.mode not in {
            "RGB",
            "RGBA",
        }:
            img = img.convert(
                "RGB"
            )

        else:
            img = img.convert(
                "RGB"
            )

        img = img.resize(
            target_size,
            Image.Resampling.BILINEAR,
        )

        return img

    except Exception:
        return None


def overlay_class(
    image: Image.Image,
    mask: np.ndarray,
    class_id: int,
) -> Image.Image:
    """
    Highlight only the selected semantic class in red.

    - selected class: semi-transparent red overlay
    - other pixels: original image
    - mask is resized with nearest-neighbor
    """

    image = image.convert("RGB")

    width, height = image.size

    mask_img = Image.fromarray(
        ((mask == class_id).astype(np.uint8) * 255),
        mode="L",
    ).resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    red = Image.new(
        "RGB",
        (width, height),
        (255, 30, 30),
    )

    # Create red-tinted version of the source image.
    tinted = Image.blend(
        image,
        red,
        alpha=0.55,
    )

    # Only use tinted pixels where current semantic class exists.
    overlay = Image.composite(
        tinted,
        image,
        mask_img,
    )

    return overlay


def make_mask_preview(
    mask: np.ndarray,
    class_id: int,
    size: tuple[int, int],
) -> Image.Image:

    binary = (
        mask == class_id
    ).astype(
        np.uint8
    )

    out = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    out[
        binary == 1
    ] = [
        255,
        40,
        40,
    ]

    preview = Image.fromarray(
        out,
        mode="RGB",
    )

    return preview.resize(
        size,
        Image.Resampling.NEAREST,
    )


def find_source_images(
    label_path: Path,
    image_index: dict[str, list[Path]],
) -> list[Path]:

    candidates = []

    # Prefer exact filename.
    candidates.extend(
        image_index.get(
            label_path.name.lower(),
            [],
        )
    )

    # Then same stem.
    candidates.extend(
        image_index.get(
            label_path.stem.lower(),
            [],
        )
    )

    unique = []

    seen = set()

    for path in candidates:

        resolved = str(
            path.resolve()
        )

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        if is_test(path):
            continue

        if is_label_candidate(path):
            continue

        unique.append(
            path
        )

    # Prefer im1 / im2 style folders.
    def score(path: Path):
        text = str(
            path
        ).lower()

        if "im1" in text:
            return 0

        if "im2" in text:
            return 1

        if "image1" in text:
            return 2

        if "image2" in text:
            return 3

        if "a/" in text.replace("\\", "/"):
            return 4

        if "b/" in text.replace("\\", "/"):
            return 5

        return 10

    unique.sort(
        key=score
    )

    return unique[:2]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"D:\Data Set\SECOND"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/v2_readiness/"
            "second_mapping_review"
        ),
    )

    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-class-pixels",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "SECOND root:",
        root,
    )

    print(
        "Output:",
        output,
    )

    # --------------------------------------------------------
    # Discover files.
    #
    # HARD RULE:
    # Test files are excluded BEFORE opening them.
    # --------------------------------------------------------

    all_images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTS
        ):
            continue

        if is_test(path):
            continue

        all_images.append(
            path
        )

    label_files = [
        p
        for p in all_images
        if is_label_candidate(p)
    ]

    non_label_images = [
        p
        for p in all_images
        if not is_label_candidate(p)
    ]

    print(
        "Non-test image files:",
        len(all_images),
    )

    print(
        "Label candidates:",
        len(label_files),
    )

    # --------------------------------------------------------
    # Image lookup.
    # --------------------------------------------------------

    image_index = defaultdict(
        list
    )

    for path in non_label_images:

        image_index[
            path.name.lower()
        ].append(
            path
        )

        image_index[
            path.stem.lower()
        ].append(
            path
        )

    # --------------------------------------------------------
    # Class scan.
    # --------------------------------------------------------

    class_pixels = defaultdict(
        int
    )

    class_files = defaultdict(
        list
    )

    readable_label_count = 0

    for i, path in enumerate(
        label_files
    ):

        arr = read_label(
            path
        )

        if arr is None:
            continue

        readable_label_count += 1

        values, counts = np.unique(
            arr,
            return_counts=True,
        )

        for value, count in zip(
            values,
            counts,
        ):

            class_id = int(
                value
            )

            count = int(
                count
            )

            # Safety against accidentally treating RGB/photo
            # intensity as segmentation IDs.
            if class_id > 255:
                continue

            class_pixels[
                class_id
            ] += count

            if (
                count
                >= args.min_class_pixels
            ):

                class_files[
                    class_id
                ].append(
                    (
                        count,
                        path,
                    )
                )

        if (
            (i + 1) % 500
            == 0
        ):
            print(
                f"Scanned {i + 1}"
            )

    IGNORE_IDS = {255}

    observed_ids = sorted(
        class_id
        for class_id in class_pixels.keys()
        if class_id not in IGNORE_IDS
    )

    print()
    print(
        "Observed semantic IDs:",
        observed_ids,
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    stats_csv = (
        output
        / "SECOND_CLASS_STATS.csv"
    )

    total_pixels = sum(
        class_pixels.values()
    )

    with stats_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "class_id",
            "pixel_count",
            "fraction",
            "candidate_files",
        ])

        for class_id in observed_ids:

            writer.writerow([
                class_id,
                class_pixels[class_id],
                (
                    class_pixels[class_id]
                    / total_pixels
                    if total_pixels
                    else 0
                ),
                len(
                    class_files[
                        class_id
                    ]
                ),
            ])

    # --------------------------------------------------------
    # Generate contact sheets.
    # --------------------------------------------------------

    sheet_records = []

    panel_w = 256
    panel_h = 256
    caption_h = 32

    for class_id in observed_ids:

        candidates = sorted(
            class_files[
                class_id
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        candidates = candidates[
            : args.examples_per_class
        ]

        if not candidates:
            continue

        rows = []

        evidence = []

        for pixel_count, label_path in candidates:

            arr = read_label(
                label_path
            )

            if arr is None:
                continue

            source_images = (
                find_source_images(
                    label_path,
                    image_index,
                )
            )

            mask_preview = (
                make_mask_preview(
                    arr,
                    class_id,
                    (
                        panel_w,
                        panel_h,
                    ),
                )
            )

            panels = [
                (
                    "MASK",
                    mask_preview,
                )
            ]

            for idx, source in enumerate(
                source_images
            ):

                image = normalize_rgb(
                    source,
                    (
                        panel_w,
                        panel_h,
                    ),
                )

                if image is None:
                    continue

                overlay = overlay_class(
                    image,
                    arr,
                    class_id,
                )

                panels.append(
                    (
                        f"IMAGE {idx + 1}",
                        image,
                    )
                )

                panels.append(
                    (
                        f"OVERLAY {idx + 1}",
                        overlay,
                    )
                )

            # Always make five columns.
            while len(panels) < 5:

                panels.append(
                    (
                        "",
                        Image.new(
                            "RGB",
                            (
                                panel_w,
                                panel_h,
                            ),
                            (
                                245,
                                245,
                                245,
                            ),
                        ),
                    )
                )

            panels = panels[:5]

            row = Image.new(
                "RGB",
                (
                    panel_w * 5,
                    panel_h
                    + caption_h
                ),
                "white",
            )

            draw = ImageDraw.Draw(
                row
            )

            for j, (
                title,
                panel,
            ) in enumerate(
                panels
            ):

                x = j * panel_w

                row.paste(
                    panel,
                    (
                        x,
                        caption_h,
                    ),
                )

                draw.text(
                    (
                        x + 4,
                        6,
                    ),
                    title,
                    fill="black",
                )

            draw.text(
                (
                    650,
                    6,
                ),
                (
                    f"class={class_id} | "
                    f"pixels={pixel_count}"
                ),
                fill="black",
            )

            rows.append(
                row
            )

            evidence.append({
                "label":
                    str(
                        label_path
                    ),

                "label_sha256":
                    sha256_file(
                        label_path
                    ),

                "pixels":
                    pixel_count,

                "source_images": [
                    str(p)
                    for p in source_images
                ],
            })

        if not rows:
            continue

        sheet = Image.new(
            "RGB",
            (
                panel_w * 5,
                (
                    panel_h
                    + caption_h
                )
                * len(rows)
                + 48
            ),
            "white",
        )

        draw = ImageDraw.Draw(
            sheet
        )

        draw.text(
            (
                10,
                12,
            ),
            (
                f"SECOND semantic class "
                f"ID = {class_id}"
            ),
            fill="black",
        )

        y = 48

        for row in rows:

            sheet.paste(
                row,
                (
                    0,
                    y,
                ),
            )

            y += row.height

        sheet_path = (
            output
            / (
                f"class_{class_id}"
                "_review.jpg"
            )
        )

        sheet.save(
            sheet_path,
            quality=92,
        )

        sheet_records.append({
            "class_id":
                class_id,

            "sheet":
                sheet_path.name,

            "sheet_sha256":
                sha256_file(
                    sheet_path
                ),

            "evidence":
                evidence,
        })

    # --------------------------------------------------------
    # HTML.
    # --------------------------------------------------------

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>SECOND Mapping Review</title>",
        """
        <style>
        body {
          font-family: Arial, sans-serif;
          margin: 30px;
          max-width: 1400px;
        }
        img {
          max-width: 100%;
          border: 1px solid #bbb;
          margin-bottom: 30px;
        }
        .warning {
          background: #fff3cd;
          padding: 15px;
          border-left: 5px solid #ffbf00;
        }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>SECOND semantic mapping review</h1>",
        """
        <div class='warning'>
        TEST images were excluded before image reads.
        Do not infer the building class from numeric order.
        Identify it visually from building footprints/roofs.
        </div>
        """,
    ]

    for record in sheet_records:

        cid = record[
            "class_id"
        ]

        html.extend([
            f"<h2>Class ID {cid}</h2>",
            (
                f"<img src='"
                f"{record['sheet']}"
                f"'>"
            ),
        ])

    html.extend([
        "</body>",
        "</html>",
    ])

    html_path = (
        output
        / "index.html"
    )

    html_path.write_text(
        "\n".join(
            html
        ),
        encoding="utf-8",
    )

    audit = {
        "second_root":
            str(root),

        "readable_non_test_labels":
            readable_label_count,

        "observed_class_ids":
            observed_ids,

        "test_image_reads":
            0,

        "numeric_mapping_inferred":
            False,

        "mapping_verified":
            False,

        "review_html":
            str(
                html_path
            ),

        "class_stats_csv":
            str(
                stats_csv
            ),

        "sheets":
            sheet_records,
    }

    audit_path = (
        output
        / "SECOND_MAPPING_REVIEW_AUDIT.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )
    print(
        "SECOND MAPPING REVIEW COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        "HTML:",
        html_path,
    )

    print(
        "Stats:",
        stats_csv,
    )

    print(
        "Audit:",
        audit_path,
    )

    print(
        "Test image reads: 0"
    )


if __name__ == "__main__":
    main()