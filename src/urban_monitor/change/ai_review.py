from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
from PIL import Image, ImageDraw


SCHEMA = {
    "type": "object",
    "required": ["sample_id", "has_visible_building_change", "change_type", "prediction_assessment",
                 "ground_truth_assessment", "registration_assessment", "binary_prediction_score",
                 "new_prediction_score", "demolition_prediction_score", "confidence", "needs_human_review",
                 "error_reasons", "short_reason_cn"],
    "properties": {
        "sample_id": {"type": "string"}, "has_visible_building_change": {"type": "boolean"},
        "change_type": {"type": "array", "items": {"enum": ["new_building", "demolition"]}},
        "prediction_assessment": {"enum": ["correct", "false_negative", "false_positive", "mixed_error", "uncertain"]},
        "ground_truth_assessment": {"enum": ["plausible", "suspicious", "clearly_wrong", "uncertain"]},
        "registration_assessment": {"enum": ["aligned", "slight_shift", "severe_shift", "uncertain"]},
        "binary_prediction_score": {"type": "number", "minimum": 0, "maximum": 100},
        "new_prediction_score": {"type": "number", "minimum": 0, "maximum": 100},
        "demolition_prediction_score": {"type": "number", "minimum": 0, "maximum": 100},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_human_review": {"type": "boolean"}, "error_reasons": {"type": "array"},
        "short_reason_cn": {"type": "string"},
    },
}
PROMPT_VERSION = "v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contact_sheet(record: dict, review: Path) -> Path:
    """Create the only remote payload: an EXIF-free 2x3 contact sheet."""
    output = review / "ai_inputs"; output.mkdir(parents=True, exist_ok=True)
    safe = str(record["sample_id"]).replace(":", "_").replace("/", "_").replace("\\", "_")
    entries = [("T1 / \u65e9\u671f", "t1_path"), ("T2 / \u665a\u671f", "t2_path"),
               ("Ground Truth / \u771f\u503c", "gt_binary_path"), ("Binary / \u84dd\u8272", "binary_probability_path"),
               ("New / \u7eff\u8272", "new_probability_path"), ("Demolition / \u7ea2\u8272", "demolition_probability_path")]
    size, title_height = 320, 28
    canvas = Image.new("RGB", (size * 3, (size + title_height) * 2), "#20242c")
    draw = ImageDraw.Draw(canvas)
    for index, (label, key) in enumerate(entries):
        source = record.get(key, "")
        path = review / source if source and not Path(source).is_absolute() else Path(source)
        image = Image.open(path).convert("RGB") if path.is_file() else Image.new("RGB", (size, size), "#30343b")
        image.thumbnail((size, size))
        tile = Image.new("RGB", (size, size), "black")
        tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        x, y = (index % 3) * size, (index // 3) * (size + title_height)
        canvas.paste(tile, (x, y + title_height)); draw.text((x + 6, y + 7), label, fill="white")
    target = output / f"{safe}.jpg"
    canvas.save(target, quality=88, exif=b"")
    return target


def _message(sample_id: str) -> str:
    return f"""You are a second reviewer for building-change imagery. T1 is earlier and T2 later. Judge building changes only: roads, trees, shadows, season, illumination, and colour are not building changes. A building appearing is new_building; disappearing is demolition; global displacement is misregistration. Ground truth and model can both be wrong. Return uncertain when unclear; do not guess and do not generate pixel labels. GLM is not ground truth and cannot directly alter training labels. Return only a JSON object satisfying this schema. sample_id must equal {sample_id!r}.\n{json.dumps(SCHEMA, ensure_ascii=False)}"""


def _call(client, model: str, image: Path, sample_id: str) -> dict:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": encoded}},
                    {"type": "text", "text": _message(sample_id)},
                ]}],
                thinking={"type": "disabled"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            import jsonschema
            jsonschema.validate(result, SCHEMA)
            result["needs_human_review"] = True  # Never turn an AI opinion into a human decision.
            return result
        except Exception as exc:  # 429/timeouts/parser faults: bounded exponential retry.
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last_error))


def run_ai_review(run: str | Path, *, model: str = "glm-4.6v-flash", split: str = "validation", max_samples: int = 100,
                  dry_run: bool = False, force_refresh: bool = False) -> Path:
    run = Path(run); review = run / "review"
    if split == "test":
        raise ValueError("AI review on test is disabled: test remains completely isolated")
    samples = review / "review_samples.csv"
    frame = pd.read_csv(samples, dtype=str).fillna("").head(max_samples)
    cache = review / "ai_cache"; cache.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("ZAI_API_KEY") or os.environ.get("GLM_API_KEY")
    if not key and not dry_run:
        raise RuntimeError("GLM AI review is disabled: set ZAI_API_KEY or GLM_API_KEY, or use --dry-run")
    client = None
    if key and not dry_run:
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise RuntimeError('Install optional AI support: pip install -e ".[ai_review]"') from exc
        client = ZhipuAiClient(api_key=key)
    records, usage = [], []
    for row in frame.to_dict("records"):
        image = _contact_sheet(row, review)
        request_hash = hashlib.sha256(f"{row['sample_id']}:{_hash(image)}:{model}:{PROMPT_VERSION}:{row.get('threshold', '')}".encode()).hexdigest()
        cached = cache / f"{request_hash}.json"; started = time.perf_counter(); error = ""
        if cached.is_file() and not force_refresh:
            result, status = json.loads(cached.read_text(encoding="utf-8")), "cached"
        elif dry_run:
            result, status = {"sample_id": row["sample_id"], "dry_run": True, "needs_human_review": True,
                              "short_reason_cn": "\u4ec5\u751f\u6210\u8054\u7cfb\u56fe\uff0c\u672a\u8c03\u7528 API"}, "dry_run"
            cached.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            try:
                result, status = _call(client, model, image, row["sample_id"]), "ok"
                cached.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                status, error = "api_parse_error", str(exc)[:300]
                result = {"sample_id": row["sample_id"], "needs_human_review": True, "error": "api_parse_error"}
        records.append(result)
        usage.append({"request_id": request_hash, "sample_id": row["sample_id"], "model": model,
                      "prompt_version": PROMPT_VERSION, "status": status,
                      "latency_seconds": round(time.perf_counter() - started, 3), "cached": status == "cached",
                      "error": error, "timestamp": datetime.now(timezone.utc).isoformat()})
    with (review / "glm_scores.jsonl").open("w", encoding="utf-8") as handle:
        for value in records:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    pd.DataFrame(usage).to_csv(review / "ai_usage.csv", index=False, encoding="utf-8-sig")
    return review / "glm_scores.jsonl"
