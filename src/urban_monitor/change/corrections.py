from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


TASK_COLUMNS = {"binary": "gt_cd_path", "new": "gt_new_path", "demolition": "gt_dem_path", "semantic_t1": "gt_t1_path", "semantic_t2": "gt_t2_path"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_correction(source: str | Path, run: str | Path, sample_id: str, task: str, *, reviewer: str,
                    notes: str = "", version: str = "0.5.0") -> Path:
    if task not in TASK_COLUMNS:
        raise ValueError(f"Unknown correction task: {task}")
    source, run = Path(source), Path(run)
    if not source.is_file(): raise FileNotFoundError(source)
    safe = sample_id.replace(":", "_").replace("/", "_")
    target = run / "review" / "corrections" / task / f"{safe}{source.suffix.casefold()}"
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    record = {"sample_id": sample_id, "task": task, "correction_path": str(target.resolve()), "correction_hash": sha256(target),
              "reviewer": reviewer, "review_time": datetime.now(timezone.utc).isoformat(), "notes": notes, "version": version}
    metadata = run / "review" / "corrections" / "corrections.jsonl"
    with metadata.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def apply_review_corrections(index_path: str | Path, decisions_path: str | Path, run: str | Path,
                             output_path: str | Path | None = None) -> Path:
    run = Path(run); frame = pd.read_csv(index_path, dtype=str).fillna("")
    decisions = pd.read_csv(decisions_path, dtype=str).fillna("")
    if decisions["sample_id"].duplicated().any():
        duplicates = decisions.loc[decisions["sample_id"].duplicated(False), "sample_id"].tolist()
        raise ValueError(f"Duplicate review decisions: {duplicates}")
    decision_map = decisions.set_index("sample_id").to_dict("index")
    metadata_path = run / "review" / "corrections" / "corrections.jsonl"
    corrections = {}
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line); corrections[(record["sample_id"], record["task"])] = record
    rows, audit = [], []
    for _, row in frame.iterrows():
        item = row.to_dict(); sample_id = item["sample_id"]; decision = decision_map.get(sample_id, {})
        status = decision.get("decision", "")
        available = {task: corrections[(sample_id, task)] for task in TASK_COLUMNS if (sample_id, task) in corrections}
        excluded = status in {"exclude", "misregistration"} or (status == "label_error" and not available)
        if excluded:
            audit.append({"sample_id": sample_id, "action": "excluded", "decision": status}); continue
        for task, record in available.items():
            column = TASK_COLUMNS[task]; original = item.get(column, "")
            item[column] = record["correction_path"]
            item[f"{task}_label_source"] = "human_corrected_label"
            item[f"{task}_original_hash"] = sha256(original) if original and Path(original).is_file() else ""
            item[f"{task}_correction_hash"] = record["correction_hash"]
        item["sample_weight"] = "2.0" if status == "prediction_error" else item.get("sample_weight", "1.0") or "1.0"
        item["review_decision"] = status; item["review_notes"] = decision.get("notes", "")
        rows.append(item); audit.append({"sample_id": sample_id, "action": "corrected" if available else "kept", "decision": status})
    output = Path(output_path) if output_path else run / "prepared" / "public_change_index.reviewed.csv"
    output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(run / "review" / "correction_audit.csv", index=False, encoding="utf-8-sig")
    report = ["# Review correction handoff", "", f"- reviewed decisions: {len(decisions)}", f"- output samples: {len(rows)}",
              f"- excluded samples: {sum(row['action']=='excluded' for row in audit)}", f"- corrected samples: {sum(row['action']=='corrected' for row in audit)}",
              "", "Human-corrected labels override original dataset labels; original files were not modified."]
    (run / "reports").mkdir(parents=True, exist_ok=True)
    (run / "reports" / "before_after_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
