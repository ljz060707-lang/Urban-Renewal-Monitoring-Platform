"""Generate the live RUN_STATUS.json / RUN_PROGRESS.md for the delivery.

The status is derived from observable state on disk (progress logs, raster
completeness gates, deliverable presence) rather than from in-process
bookkeeping, so it stays truthful even if a producer dies.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work" / "python-packages"))

DELIVERY = Path(r"F:\xian\西安_当前最佳候选交付_20260917")
RASTERS = DELIVERY / "01_raster_products"
RECEIPTS = DELIVERY / "06_receipts"
EXPECTED_COUNTS = {"2018": 195_140, "2019": 182_193, "2020": 281_417}

BOUNDARY = {y: RASTERS / f"building_boundary_{y}_ge100m2.tif" for y in ("2018", "2019", "2020")}
IDTIF = {y: RASTERS / f"building_id_{y}_ge100m2.tif" for y in ("2018", "2019", "2020")}
UNFILTERED = {y: RASTERS / f"building_boundary_{y}_unfiltered.tif" for y in ("2018", "2019", "2020")}
INTERVALS = ("2018_2019", "2019_2020")


def raster_processes() -> list[dict]:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
        "Where-Object {$_.CommandLine -like '*xian_raster*'} | "
        "Select-Object ProcessId,CommandLine,CreationDate,WorkingSetSize | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["pwsh", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=90
        ).stdout.strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "pid": d.get("ProcessId"),
                "command_line": (d.get("CommandLine") or "").strip(),
                "created": str(d.get("CreationDate")),
                "working_set_gb": round((d.get("WorkingSetSize") or 0) / 1024**3, 2),
            }
            for d in data
        ]
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def tail(path: Path, n: int = 400) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln for ln in lines if ln.strip()][-n:]


def parse_stripes(lines: list[str], pattern: str) -> dict:
    """Return the furthest stripe seen per key for a progress pattern."""
    best: dict[str, dict] = {}
    rx = re.compile(pattern)
    for ln in lines:
        m = rx.search(ln)
        if not m:
            continue
        key = m.group("key")
        cur = int(m.group("i"))
        tot = int(m.group("n"))
        if key not in best or cur >= best[key]["strip"]:
            best[key] = {"strip": cur, "total": tot, "pct": round(cur / tot * 100.0, 2), "line": ln.strip()}
    return best


def sha256_head(path: Path, limit: int | None = None) -> str:
    import hashlib

    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
            read += len(b)
            if limit and read >= limit:
                break
    return h.hexdigest()


def raster_is_complete(path: Path) -> tuple[bool, str]:
    """A raster only counts as done when its final 512-row stripe carries data.

    A killed writer leaves a valid, openable, all-zero GeoTIFF, so existence and
    readability are not sufficient evidence of completion.
    """
    if path.exists() and path.with_name(path.name + ".incomplete").exists():
        return False, "incomplete sibling present"
    tmp = path.with_name(path.name + ".incomplete")
    if not path.exists():
        return False, "being written" if tmp.exists() else "missing"
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(path) as src:
            row_off = max(0, src.height - 512)
            tail = src.read(1, window=Window(0, row_off, src.width, src.height - row_off))
            if int(np.count_nonzero(tail)) == 0:
                return False, "no data in final stripe (truncated)"
        return True, "ok"
    except Exception as exc:
        return False, f"unreadable ({type(exc).__name__})"


def main() -> None:
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    prog_lines = tail(RASTERS / "RASTER_PROGRESS.md", 600)
    state_lines = tail(RASTERS / "STATE_PROGRESS.md", 400)
    all_lines = prog_lines + state_lines

    epoch_progress = parse_stripes(all_lines, r"(?P<key>\d{4}): stripe (?P<i>\d+)/(?P<n>\d+)")
    state_progress = parse_stripes(
        all_lines, r"state masks (?P<key>\d{4}): stripe (?P<i>\d+)/(?P<n>\d+)"
    )
    finished = {
        m.group(1) for m in (re.search(r"^-\s[\d\-: ]+(\d{4}): PROMOTED", ln) for ln in all_lines) if m
    }

    procs = raster_processes()
    from datetime import datetime, timezone

    bnd_state = {y: raster_is_complete(BOUNDARY[y]) for y in ("2018", "2019", "2020")}
    id_state = {y: raster_is_complete(IDTIF[y]) for y in ("2018", "2019", "2020")}
    unf_state = {y: raster_is_complete(UNFILTERED[y]) for y in ("2018", "2019", "2020")}
    state_files = {
        f"{iv}/{name}": raster_is_complete(RASTERS / iv / name)
        for iv in INTERVALS
        for name in (
            "STABLE_ge100m2.tif",
            "ADDITION_ge100m2.tif",
            "DEMOLITION_ge100m2.tif",
            "UNKNOWN_CONFLICT_ge100m2.tif",
            "THREE_STATE_COMBINED_ge100m2.tif",
        )
    }

    status = {
        "current_phase": "RASTERIZATION_AND_PRODUCT_GENERATION",
        "current_epoch": None,
        "current_strip": None,
        "total_strips": None,
        "raster_process_pid": procs[0].get("pid") if procs and "pid" in procs[0] else None,
        "raster_processes": procs,
        "boundary_tifs_done": sum(1 for y in BOUNDARY if bnd_state[y][0]),
        "id_tifs_done": sum(1 for y in IDTIF if id_state[y][0]),
        "unfiltered_tifs_done": sum(1 for y in UNFILTERED if unf_state[y][0]),
        "state_tifs_done": sum(1 for v in state_files.values() if v[0]),
        "boundary_tifs_detail": {y: bnd_state[y][1] for y in BOUNDARY},
        "id_tifs_detail": {y: id_state[y][1] for y in IDTIF},
        "unfiltered_tifs_detail": {y: unf_state[y][1] for y in UNFILTERED},
        "state_tifs_detail": {k: v[1] for k, v in state_files.items()},
        "area_stats_done": (DELIVERY / "02_area_statistics" / "XIAN_TRANSITION_STATS_CURRENT_BEST.json").exists(),
        "tracking_cases_done": len(list((DELIVERY / "03_tracking_cases").glob("**/contact_sheet.png")))
        if (DELIVERY / "03_tracking_cases").exists()
        else 0,
        "publication_figures_done": len(list((DELIVERY / "04_publication_figures").glob("*.png")))
        if (DELIVERY / "04_publication_figures").exists()
        else 0,
        "documentation_done": len(list((DELIVERY / "05_documentation").glob("*.md")))
        if (DELIVERY / "05_documentation").exists()
        else 0,
        "last_update": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "current_blocker": None,
        "next_action": None,
        "epoch_raster_progress": epoch_progress,
        "state_raster_progress": state_progress,
        "epochs_promoted": sorted(finished),
    }

    active_epoch = None
    if procs and any("xian_raster" in (p.get("command_line") or "") for p in procs):
        running_state = any("--state" in (p.get("command_line") or "") for p in procs)
        table = state_progress if running_state else epoch_progress
        cand = [k for k in table if k not in finished]
        if cand:
            active_epoch = sorted(cand)[0]
            status["current_epoch"] = active_epoch
            status["current_strip"] = table[active_epoch]["strip"]
            status["total_strips"] = table[active_epoch]["total"]
        status["current_phase"] = "STATE_MASK_RASTERIZATION" if running_state else "RASTERIZATION"

    try:
        import shutil

        status["disk_free_gb"] = round(shutil.disk_usage("F:/").free / 1024**3, 1)
    except Exception:
        status["disk_free_gb"] = None

    missing = []
    for y in ("2018", "2019", "2020"):
        for p, ok in ((BOUNDARY[y], bnd_state[y][0]), (IDTIF[y], id_state[y][0]), (UNFILTERED[y], unf_state[y][0])):
            if not ok:
                missing.append(str(p))
    for key, (ok, _) in state_files.items():
        if not ok and key.endswith("THREE_STATE_COMBINED_ge100m2.tif"):
            missing.append(str(RASTERS / key))
    status["missing_expected_products"] = missing

    if missing:
        status["next_action"] = f"produce {len(missing)} pending raster product(s)"
    elif status["state_tifs_done"] < 10:
        status["next_action"] = "combine three-state products"
    else:
        status["next_action"] = "tracking cases, publication figures, documentation, QA receipts"

    (RECEIPTS / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# RUN PROGRESS — Xi'an current-best candidate delivery",
        "",
        f"- last_update: {status['last_update']}",
        f"- scientific_status: CANDIDATE_NOT_GROUND_TRUTH",
        f"- phase: {status['current_phase']}",
        f"- disk_free_gb: {status['disk_free_gb']}",
        f"- raster processes: {len(procs)}",
        "",
        "## Boundary / ID rasters",
        "",
    ]
    for y in ("2018", "2019", "2020"):
        b_ok, b_why = bnd_state[y]
        i_ok, i_why = id_state[y]
        lines.append(
            f"- {y}: boundary={'OK' if b_ok else 'PENDING (' + b_why + ')'} "
            f"{'(' + format(BOUNDARY[y].stat().st_size/1e6, '.1f') + ' MB)' if BOUNDARY[y].exists() else ''}"
        )
        lines.append(
            f"- {y}: id={'OK' if i_ok else 'PENDING (' + i_why + ')'} "
            f"{'(' + format(IDTIF[y].stat().st_size/1e6, '.1f') + ' MB)' if IDTIF[y].exists() else ''}"
            f" expected_objects={EXPECTED_COUNTS[y]:,}"
        )
    lines += ["", "## Live stripe progress", ""]
    if epoch_progress:
        for k, v in sorted(epoch_progress.items()):
            lines.append(f"- epoch {k}: {v['strip']}/{v['total']} ({v['pct']}%)")
    if state_progress:
        for k, v in sorted(state_progress.items()):
            lines.append(f"- state masks {k}: {v['strip']}/{v['total']} ({v['pct']}%)")
    lines += [
        "",
        "## Deliverable counters",
        "",
        f"- boundary_tifs_done: {status['boundary_tifs_done']}/3",
        f"- id_tifs_done: {status['id_tifs_done']}/3",
        f"- state_tifs_done: {status['state_tifs_done']}/10",
        f"- area_stats_done: {status['area_stats_done']}",
        f"- tracking_cases_done: {status['tracking_cases_done']}",
        f"- publication_figures_done: {status['publication_figures_done']}",
        f"- documentation_done: {status['documentation_done']}",
        f"- blocker: {status['current_blocker']}",
        f"- next_action: {status['next_action']}",
        "",
        "## Recent progress log",
        "",
        "```",
    ]
    lines += all_lines[-25:]
    lines += ["```", ""]
    (RECEIPTS / "RUN_PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: status[k] for k in (
        "current_phase", "current_epoch", "current_strip", "total_strips", "boundary_tifs_done",
        "id_tifs_done", "state_tifs_done", "disk_free_gb", "next_action")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
