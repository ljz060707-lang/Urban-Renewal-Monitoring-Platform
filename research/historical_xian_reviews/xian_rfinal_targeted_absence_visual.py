from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("/data/xian_r4_100km2_20260914")
    base = root / "output/19_rfinal_phase2_observability_diagnostic_20260916"
    audit = pd.read_csv(base / "phase2_candidate_audit.csv", low_memory=False)
    chosen = audit[
        audit["physical_track_id"].astype(str).eq("PTRK_R4V2_00008046")
        & audit["interval"].astype(str).eq("2018_2019")
        & audit["scene_key"].astype(str).eq("xa190101")
    ].copy()
    if chosen.empty:
        raise RuntimeError("targeted candidate row not found")
    chosen["promotion_eligible_zero_candidate"] = True
    target_audit = root / "output/23_rfinal_targeted_absence_visual_20260917/target_audit.csv"
    target_audit.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(target_audit, index=False, encoding="utf-8-sig")
    output = target_audit.parent / "assets"
    script = Path("/root/urrs_xian_r2/code/_tmp_xian_rfinal_phase2_visual_assets.py")
    subprocess.run([sys.executable, str(script), "--root", str(root), "--audit", str(target_audit), "--output", str(output)], check=True)


if __name__ == "__main__":
    main()
