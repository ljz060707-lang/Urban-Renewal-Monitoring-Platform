"""Resolve candidate source dates from exact acquisition dates of source tiles.

This module only associates a geometry's intersected tile IDs with dates in a
tile registry. It does not prove that the geometry was traced from that scene,
and it never promotes a candidate date to verified capture provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping


TileId = tuple[int, int]


@dataclass(frozen=True)
class SourceTileDateAssociation:
    """Candidate date coverage for one source geometry."""

    status: str
    tile_ids: tuple[str, ...]
    tile_actual_dates: tuple[tuple[str, str | None], ...]
    candidate_dates: tuple[str, ...]
    unknown_tile_ids: tuple[str, ...]


def associate_source_tile_dates(
    tile_ids: Iterable[TileId],
    actual_dates_by_tile: Mapping[TileId, str | None],
) -> SourceTileDateAssociation:
    """Summarize exact-date registry coverage for every intersected source tile.

    A single candidate date is returned only when every intersected source tile
    has a valid date. Missing, conflicting, or out-of-scope tiles remain
    unresolved; neighboring dates are never used to fill them.
    """
    normalized: set[TileId] = set()
    for tile in tile_ids:
        if (not isinstance(tile, (tuple, list)) or len(tile) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in tile)):
            raise ValueError("Source tile IDs must be integer (x, y) pairs")
        normalized.add((tile[0], tile[1]))

    ordered_tiles = tuple(sorted(normalized))
    tile_labels = tuple(f"{x}:{y}" for x, y in ordered_tiles)
    if not ordered_tiles:
        return SourceTileDateAssociation(
            "NO_SOURCE_TILE_SUPPORT", (), (), (), ())

    dates: set[str] = set()
    unknown: list[str] = []
    tile_actual_dates: list[tuple[str, str | None]] = []
    for tile, label in zip(ordered_tiles, tile_labels):
        value = actual_dates_by_tile.get(tile)
        if value is None or value == "":
            unknown.append(label)
            tile_actual_dates.append((label, None))
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid actual date for source tile {label}: {value!r}")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid actual date for source tile {label}: {value!r}") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"Actual date must use YYYY-MM-DD for source tile {label}")
        dates.add(value)
        tile_actual_dates.append((label, value))

    if unknown:
        status = "INCOMPLETE_SOURCE_TILE_DATES"
    elif len(dates) == 1:
        status = "SINGLE_ACTUAL_DATE_CANDIDATE"
    else:
        status = "MULTIPLE_ACTUAL_DATES"
    return SourceTileDateAssociation(
        status=status,
        tile_ids=tile_labels,
        tile_actual_dates=tuple(tile_actual_dates),
        candidate_dates=tuple(sorted(dates)),
        unknown_tile_ids=tuple(unknown),
    )
