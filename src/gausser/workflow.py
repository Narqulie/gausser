"""Authoritative stage dependency and invalidation policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .scene import ScenePaths


Generation = Literal["images", "masks", "reconstruction"]


@dataclass(frozen=True)
class GenerationPolicy:
    """Canonical artifacts and latest-stage records superseded by a generation."""

    invalidated_stages: frozenset[str]
    required_destinations: frozenset[str]


_POLICIES: dict[Generation, GenerationPolicy] = {
    "images": GenerationPolicy(
        invalidated_stages=frozenset({
            "ingest", "dataset", "sharpframes", "sammer", "colmap",
            "brush", "msplat", "lod", "supersplat",
        }),
        required_destinations=frozenset({"images"}),
    ),
    "masks": GenerationPolicy(
        invalidated_stages=frozenset({"sammer", "colmap", "brush", "msplat", "lod", "supersplat"}),
        required_destinations=frozenset({"masks", "colmap/masks"}),
    ),
    "reconstruction": GenerationPolicy(
        invalidated_stages=frozenset({"colmap", "brush", "msplat", "lod", "supersplat"}),
        required_destinations=frozenset({"database.db", "sparse"}),
    ),
}


def generation_policy(generation: Generation) -> GenerationPolicy:
    return _POLICIES[generation]


def superseded_artifacts(paths: ScenePaths, generation: Generation) -> tuple[Path, ...]:
    """Return non-overlapping canonical paths retired by a new generation."""
    if generation == "images":
        return (
            paths.images,
            paths.brush_masks,
            paths.colmap,
            paths.database,
            paths.sparse,
            paths.runs,
        )
    if generation == "masks":
        return (
            paths.brush_masks,
            paths.colmap,
            paths.database,
            paths.sparse,
            paths.runs,
        )
    return (paths.database, paths.sparse, paths.runs)
