#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.project_discovery import build_manifest, discover_project


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover images/labels and build manifest.generated.csv")
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project = discover_project(args.aoi, args.data_root)
    output = Path(args.output) if args.output else project.project_dir / "manifest.generated.csv"
    if output.exists() and not args.force:
        print(f"reused: {output}")
        return 0
    result = build_manifest(project, output)
    print(f"recomputed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
