#!/usr/bin/env python3
from urban_monitor.catalog import build_catalog


if __name__ == "__main__":
    result = build_catalog()
    print(result["gallery"])
