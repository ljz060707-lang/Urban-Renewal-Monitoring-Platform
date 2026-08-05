from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
from PIL import Image
import numpy as np


_HUMAN_FN_TERMS = ("false_negative", "\u6f0f\u68c0", "\u672a\u68c0\u51fa")
_COLUMNS = [
    "sample_id", "dataset_name", "split", "human_decision", "human_error_type", "glm_decision",
    "glm_confidence", "gt_positive_ratio", "prediction_positive_ratio", "precision", "recall", "f1",
    "false_negative_pixels", "false_positive_pixels", "target_head", "error_type", "human_verified", "hard_reason", "sampling_weight",
    "false_negative_binary_mask_path", "false_negative_new_mask_path", "false_negative_demolition_mask_path",
    "false_positive_binary_mask_path", "false_positive_new_mask_path", "false_positive_demolition_mask_path",
]


def _human_false_negative(decision: dict) -> bool:
    return decision.get("decision") == "prediction_error" and any(
        term in str(decision.get("notes", "")).casefold() for term in _HUMAN_FN_TERMS
    )


def _target_head(row: dict) -> str:
    """Classify the verified error from GT direction without changing labels."""
    def positive(path: str) -> bool:
        try:
            with Image.open(path) as image:
                return bool((image.getbbox()))
        except Exception:
            return False
    new = positive(str(row.get("gt_new_path", "")))
    demolition = positive(str(row.get("gt_dem_path") or row.get("gt_demolition_path") or ""))
    if new and demolition: return "both"
    if new: return "new"
    if demolition: return "demolition"
    if positive(str(row.get("gt_cd_path", ""))): return "binary"
    return "unknown"


def _write_false_negative_masks(run: Path, item: dict, threshold: float = 0.5) -> dict[str, str]:
    """Materialize pixel-level FN regions from baseline probabilities and GT."""
    destination = run / "review" / "hard_masks"; destination.mkdir(parents=True, exist_ok=True)
    result = {}
    paths = {
        "binary": (item.get("gt_cd_path") or item.get("gt_binary_path"), item.get("binary_probability_path")),
        "new": (item.get("gt_new_path"), item.get("new_probability_path")),
        "demolition": (item.get("gt_dem_path") or item.get("gt_demolition_path"), item.get("demolition_probability_path")),
    }
    for head, (gt_path, probability_path) in paths.items():
        if not gt_path or not probability_path or not Path(str(gt_path)).is_file() or not Path(str(probability_path)).is_file():
            continue
        with Image.open(gt_path) as image:
            gt = np.asarray(image.convert("L")) > 0
        with Image.open(probability_path) as image:
            probability = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if probability.shape != gt.shape:
            probability = np.asarray(Image.fromarray((probability * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        mask = (gt & (probability < threshold)).astype(np.uint8) * 255
        safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", str(item["sample_id"]))
        path = destination / f"{safe_id}_{head}_fn.png"
        Image.fromarray(mask).save(path)
        result[f"false_negative_{head}_mask_path"] = str(path.resolve())
        fp_path = destination / f"{safe_id}_{head}_fp.png"
        Image.fromarray(((~gt) & (probability >= threshold)).astype(np.uint8) * 255).save(fp_path)
        result[f"false_positive_{head}_mask_path"] = str(fp_path.resolve())
    return result


def build_hard_examples(run: str | Path, review_csv: str | Path | None = None, *, low_recall: float = 0.25,
                        false_negative_weight: float = 3.0, positive_weight: float = 2.0,
                        ordinary_weight: float = 1.0) -> Path:
    """Create a train-only sampling index; validation/test stay diagnosis-only."""
    run = Path(run)
    sample_files = [path for path in (run / "review" / "review_samples.csv", run / "review_train" / "review_samples.csv") if path.is_file()]
    if not sample_files:
        raise FileNotFoundError("No review/review_samples.csv or review_train/review_samples.csv")
    samples = pd.concat([pd.read_csv(path, dtype=str).fillna("") for path in sample_files], ignore_index=True)
    samples = samples.drop_duplicates("sample_id", keep="last")
    index = pd.read_csv(run / "prepared" / "public_change_index.csv", dtype=str).fillna("")
    decisions = (pd.read_csv(review_csv, dtype=str).fillna("")
                 if review_csv and Path(review_csv).is_file()
                 else pd.DataFrame(columns=["sample_id", "decision", "notes"]))
    decision_map = decisions.drop_duplicates("sample_id", keep="last").set_index("sample_id").to_dict("index") if not decisions.empty else {}
    records: list[dict] = []
    for _, row in samples.iterrows():
        item, decision = row.to_dict(), decision_map.get(row["sample_id"], {})
        human_fn = _human_false_negative(decision)
        auto_fn = (float(item.get("recall") or 0) < low_recall and float(item.get("gt_positive_ratio") or 0) > 0
                   and int(float(item.get("false_negative_pixels") or 0)) > 0)
        glm_fn = (str(item.get("glm_prediction_assessment", "")) == "false_negative"
                  and float(item.get("glm_confidence") or 0) >= .8)
        reasons = (["human_false_negative"] if human_fn else []) + (["auto_low_recall"] if auto_fn else []) + (["glm_false_negative_candidate"] if glm_fn else [])
        if reasons:
            target_head = _target_head(item)
            item.update(human_decision=decision.get("decision", ""), human_verified=human_fn,
                        error_type="false_negative" if human_fn else "mixed", target_head=target_head,
                        human_error_type="false_negative" if human_fn else "", hard_reason=";".join(reasons),
                        sampling_weight=false_negative_weight if human_fn else positive_weight)
            item.update(_write_false_negative_masks(run, item))
            records.append(item)

    # A separately reviewed train sample can be promoted.  It is impossible for
    # a validation/test decision to cross this explicit split gate.
    known = {str(item["sample_id"]) for item in records}
    train_rows = index[index["split"].str.casefold() == "train"].set_index("sample_id")
    for sample_id, decision in decision_map.items():
        if _human_false_negative(decision) and sample_id in train_rows.index and sample_id not in known:
            item = train_rows.loc[sample_id].to_dict()
            item.update(sample_id=sample_id, human_decision="prediction_error", human_error_type="false_negative",
                        human_verified=True, error_type="false_negative", target_head=_target_head(item),
                        hard_reason="human_false_negative", sampling_weight=false_negative_weight)
            item.update(_write_false_negative_masks(run, item))
            records.append(item)

    output = pd.DataFrame(records)
    for column in _COLUMNS:
        if column not in output:
            output[column] = ""
    review = run / "review"; review.mkdir(parents=True, exist_ok=True)
    output.to_csv(review / "hard_examples.csv", index=False, encoding="utf-8-sig")
    with (review / "hard_examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in output.to_dict("records"):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    train_ids = set(index.loc[index["split"].str.casefold() == "train", "sample_id"])
    train_hard = output[output["sample_id"].isin(train_ids)].copy()
    weighted = index.copy()
    weights = dict(zip(train_hard["sample_id"], train_hard["sampling_weight"]))
    for column in ("target_head", "hard_reason", "false_negative_binary_mask_path", "false_negative_new_mask_path", "false_negative_demolition_mask_path", "false_positive_binary_mask_path", "false_positive_new_mask_path", "false_positive_demolition_mask_path"):
        mapping = dict(zip(train_hard["sample_id"], train_hard.get(column, pd.Series("", index=train_hard.index))))
        weighted[column] = [mapping.get(value, "") for value in weighted["sample_id"]]
    weighted["sample_weight"] = [str(weights.get(value, positive_weight if str(has).casefold() in {"true", "1", "yes"} else ordinary_weight))
                                 for value, has in zip(weighted["sample_id"], weighted["has_binary"])]
    weighted.to_csv(run / "prepared" / "public_change_index.hard.csv", index=False, encoding="utf-8-sig")
    direction_counts = output.get("target_head", pd.Series(dtype=str)).value_counts().to_dict()
    report = ["# Hard example summary", "", f"- diagnostic candidates: {len(output)}", f"- train-eligible: {len(train_hard)}",
              f"- new_false_negative: {direction_counts.get('new', 0)}", f"- demolition_false_negative: {direction_counts.get('demolition', 0)}", f"- both: {direction_counts.get('both', 0)}", f"- unknown: {direction_counts.get('unknown', 0)}",
              "- Validation/test candidates are diagnosis-only and were not added to the training index.",
              f"- weighted index: `{run / 'prepared' / 'public_change_index.hard.csv'}`"]
    reports = run / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "HARD_EXAMPLE_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return review / "hard_examples.csv"


def build_demolition_head_sampling_index(run: str | Path, hard_examples: str | Path, *,
                                         demolition_only_weight: float = 2.0,
                                         both_weight: float = 1.25,
                                         ordinary_weight: float = 1.0,
                                         max_hard_fraction: float = 0.10,
                                         max_both_fraction_within_hard: float = 0.40) -> tuple[Path, dict]:
    """Build a deterministic, train-only demolition-focused sampling index.

    New-only difficult examples intentionally retain ordinary weight.  ``both``
    examples are deterministically capped so a demolition-only fine-tune does
    not become a repeated pass over the much larger mixed-direction subset.
    """
    run = Path(run)
    index = pd.read_csv(run / "prepared" / "public_change_index.csv", dtype=str).fillna("")
    hard = pd.read_csv(hard_examples, dtype=str).fillna("")
    train = index[index["split"].str.casefold() == "train"].copy()
    train_ids = set(train["sample_id"])
    eligible = hard[hard["sample_id"].isin(train_ids)].copy()
    if "human_verified" in eligible:
        eligible = eligible[eligible["human_verified"].astype(str).str.casefold().isin({"true", "1", "yes"})].copy()
    demolition_only = eligible[eligible.get("target_head", "").str.casefold() == "demolition"].sort_values("sample_id")
    both = eligible[eligible.get("target_head", "").str.casefold() == "both"].sort_values("sample_id")
    # Keep mixed samples at most the configured share of selected hard samples.
    allowed_both = int(len(demolition_only) * max_both_fraction_within_hard / max(1e-9, 1.0 - max_both_fraction_within_hard))
    selected_both = both.head(allowed_both)
    selected = pd.concat([demolition_only, selected_both], ignore_index=True)
    # The cap is a safety assertion: regular train examples always dominate.
    if len(train) and len(selected) / len(train) > max_hard_fraction:
        selected = selected.sort_values(["target_head", "sample_id"]).head(int(len(train) * max_hard_fraction))
    weights = {str(row.sample_id): (demolition_only_weight if row.target_head == "demolition" else both_weight)
               for row in selected.itertuples(index=False)}
    output = index.copy()
    output["sample_weight"] = [str(weights.get(str(sample_id), ordinary_weight)) for sample_id in output["sample_id"]]
    demolition_training_pool = pd.concat([demolition_only, both], ignore_index=True)
    selected_by_id = demolition_training_pool.drop_duplicates("sample_id").set_index("sample_id")
    for column in ("target_head", "hard_reason", "false_negative_binary_mask_path", "false_negative_new_mask_path", "false_negative_demolition_mask_path"):
        mapping = selected_by_id[column].to_dict() if column in selected_by_id else {}
        output[column] = [mapping.get(str(sample_id), "") for sample_id in output["sample_id"]]
    all_eligible_by_id = eligible.drop_duplicates("sample_id").set_index("sample_id")
    for column in ("false_positive_binary_mask_path", "false_positive_new_mask_path", "false_positive_demolition_mask_path"):
        mapping = all_eligible_by_id[column].to_dict() if column in all_eligible_by_id else {}
        output[column] = [mapping.get(str(sample_id), "") for sample_id in output["sample_id"]]
    path = run / "prepared" / "public_change_index.demolition_head.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    summary = {
        "index_path": str(path), "train_samples": int(len(train)), "hard_selected": int(len(selected)),
        "demolition_only_selected": int(len(demolition_only)), "both_selected": int(len(selected_both)),
        "demolition_hard_pool": int(len(demolition_training_pool)),
        "new_only_excluded": int((eligible.get("target_head", "").str.casefold() == "new").sum()),
        "hard_fraction": float(len(selected) / max(1, len(train))),
        "weights": {"demolition_only": demolition_only_weight, "both": both_weight, "ordinary": ordinary_weight},
    }
    if summary["hard_fraction"] > max_hard_fraction + 1e-12:
        raise RuntimeError("Demolition hard-example cap exceeded max_hard_sample_fraction")
    return path, summary
