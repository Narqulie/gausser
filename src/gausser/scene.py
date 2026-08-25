"""Canonical paths and invariants for a Gausser scene."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .ui import is_image_file


class SceneError(ValueError):
    """Raised when a directory cannot be used as a Gausser scene."""


@dataclass(frozen=True)
class ScenePaths:
    """All managed paths derived from one resolved scene root."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "ScenePaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def brush_masks(self) -> Path:
        return self.root / "masks"

    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def colmap_masks(self) -> Path:
        return self.colmap / "masks"

    @property
    def database(self) -> Path:
        return self.root / "database.db"

    @property
    def sparse(self) -> Path:
        return self.root / "sparse"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def metadata(self) -> Path:
        return self.root / ".gausser"

    @property
    def manifest(self) -> Path:
        return self.metadata / "manifest.json"

    @property
    def staging(self) -> Path:
        return self.metadata / "staging"

    def training_run(self, backend: str, run_id: str) -> Path:
        self._validate_component(backend, "backend")
        self._validate_component(run_id, "run id")
        return self.runs / backend / run_id

    def staging_attempt(self, stage: str, attempt_id: str) -> Path:
        self._validate_component(stage, "stage")
        self._validate_component(attempt_id, "attempt id")
        return self.staging / f"{stage}-{attempt_id}"

    def new_run_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4().hex[:8]}"

    def relative_path(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise SceneError(f"Path escapes scene root: {path}") from exc

    def image_files(self) -> list[Path]:
        if not self.images.is_dir():
            raise SceneError(f"Image directory not found: {self.images}")

        nested_files = [path for path in self.images.rglob("*") if path.is_file() and path.parent != self.images]
        if nested_files:
            raise SceneError("images/ must be flat; nested image paths are not supported")

        files = sorted(path for path in self.images.iterdir() if is_image_file(path))
        stems: set[str] = set()
        for path in files:
            if path.stem in stems:
                raise SceneError(
                    f"Duplicate image stem {path.stem!r}; Brush masks would collide"
                )
            stems.add(path.stem)
        return files

    @staticmethod
    def _validate_component(value: str, name: str) -> None:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise SceneError(f"Invalid {name}: {value!r}")
