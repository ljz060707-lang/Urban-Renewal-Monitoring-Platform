import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


pytestmark = pytest.mark.train_smoke


def _tiny_coco(root: Path):
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        pytest.skip("pycocotools is not installed")
    root.mkdir(parents=True)
    for split in ("train", "val"):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = 180
        Image.fromarray(image).save(root / f"{split}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        payload = {
            "images": [{"id": 1 if split == "train" else 2, "file_name": f"{split}.png", "width": 64, "height": 64}],
            "annotations": [{
                "id": 1 if split == "train" else 2,
                "image_id": 1 if split == "train" else 2,
                "category_id": 1, "segmentation": rle, "bbox": [16, 16, 32, 32],
                "area": 1024, "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "building"}],
        }
        (root / f"annotations_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_tiny_cpu_training_best_reload_inference_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from urban_monitor.model import build_model
    from urban_monitor.training import train_teacher

    coco = tmp_path / "coco"
    output = tmp_path / "training"
    _tiny_coco(coco)
    best = train_teacher(
        coco, output, epochs=1, batch_size=1, learning_rate=.001, weight_decay=.0005,
        num_workers=0, pretrained=False, amp=False, resume=False, min_size=64, max_size=64,
        freeze_backbone=True, device_name="cpu", early_stopping_patience=2,
    )
    assert best.exists() and (output / "checkpoints" / "last.pth").exists()
    first = torch.load(output / "checkpoints" / "last.pth", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1
    model = build_model(checkpoint=best, pretrained=False, min_size=64, max_size=64).eval()
    with torch.inference_mode():
        prediction = model([torch.zeros((3, 64, 64), dtype=torch.float32)])[0]
    assert {"boxes", "labels", "scores", "masks"}.issubset(prediction)
    train_teacher(
        coco, output, epochs=2, batch_size=1, learning_rate=.001, weight_decay=.0005,
        num_workers=0, pretrained=False, amp=False, resume=True, min_size=64, max_size=64,
        freeze_backbone=True, device_name="cpu", early_stopping_patience=2,
    )
    resumed = torch.load(output / "checkpoints" / "last.pth", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    import rasterio
    from rasterio.transform import from_origin
    from urban_monitor.inference import infer_full_timeseries
    from urban_monitor.validation import evaluate_checkpoint

    prepared = tmp_path / "prepared"
    (prepared / "images").mkdir(parents=True)
    with rasterio.open(
        prepared / "images" / "2024-01-01.tif", "w", driver="GTiff", height=64, width=64,
        count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 64, 1, 1),
    ) as target:
        target.write(np.zeros((3, 64, 64), dtype=np.uint8))
    # COCO tiles lack geospatial fields in this unit fixture; add the prepared tile offsets expected by validation.
    for split in ("train", "val"):
        annotation_path = coco / f"annotations_{split}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        payload["images"][0].update(date="2024-01-01", row_off=0, col_off=0)
        annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    metrics = evaluate_checkpoint(
        coco, prepared, best, tmp_path / "evaluation", min_size=64, max_size=64,
        score_threshold=.99, device_name="cpu",
    )
    assert {"precision", "recall", "F1", "mean_matched_IoU", "Dice"}.issubset(metrics)
    assert (tmp_path / "evaluation" / "metrics.json").exists()
    assert len(list((tmp_path / "evaluation" / "previews").glob("good_*.jpg"))) == 10
    summary = infer_full_timeseries(
        prepared, ["2024-01-01"], tmp_path / "predictions", checkpoint=best,
        chip_size=64, overlap=0, score_threshold=0.99, mask_threshold=.5,
        nms_iou_threshold=.4, model_version="tiny-test", min_size=64, max_size=64,
    )
    assert summary.exists()
    assert (tmp_path / "predictions" / "raw" / "2024-01-01.gpkg").exists()
    assert (tmp_path / "predictions" / "regularized" / "2024-01-01.gpkg").exists()
