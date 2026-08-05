"""Validation-only guard that prevents a regressed MTI result from being selected."""
from __future__ import annotations

import json
from pathlib import Path


def finalize_mti_selection(source: str | Path, output: str | Path) -> dict:
    source_path, output_path = Path(source), Path(output)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = payload["selected_parameters"]
    baseline = payload["no_mti"]
    f1_improved = float(candidate["temporal_change_F1"]) > float(baseline["f1"])
    flicker_improved = float(candidate["flicker_rate"]) < float(baseline["flicker_rate"])
    selected = f1_improved or flicker_improved
    final = {
        **payload,
        "mti_status": (
            "mti_sample_validation_calibrated_candidate"
            if selected
            else "mti_calibrated_but_not_selected"
        ),
        "integration_mode": "mti" if selected else "no_mti",
        "candidate_parameters": candidate,
        "selected_parameters": candidate if selected else None,
        "selection_gate": {
            "temporal_change_f1_improved": f1_improved,
            "flicker_rate_improved": flicker_improved,
            "selected": selected,
            "rule": "select MTI only if validation change F1 or flicker improves over no-MTI",
        },
        "source_selection": str(source_path.resolve()),
        "test_reads": 0,
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != final:
            raise FileExistsError(f"Refusing to overwrite a different final MTI selection: {output_path}")
        return existing
    output_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return final