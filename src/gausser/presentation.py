"""Versioned, artifact-bound Scene Composition documents.

Presentation metadata is intentionally separate from immutable training runs.  It
stores an origin, navigation pivot, semantic focus points, and portable camera
poses in the selected artifact's local coordinates.  The artifact bytes are
never rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .artifacts import list_artifacts
from .manifest import fingerprint_file
from .scene import ScenePaths


_SCHEMA_VERSION = 1
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_ITEMS = 1_000
_MAX_COORDINATE = 1.0e12
_NATIVE_FORMATS = frozenset({"ply", "spz", "splat"})
_DOCUMENT_KEYS = frozenset({
    "version",
    "artifact",
    "revision",
    "frame",
    "default_pivot",
    "focus_points",
    "views",
    "home_view_id",
})


class PresentationError(ValueError):
    """Base class for a Scene Composition domain error."""


class PresentationValidationError(PresentationError):
    """The requested document is malformed or internally inconsistent."""


class PresentationChangedError(PresentationError):
    """Another writer saved a newer revision."""


class PresentationArtifactChangedError(PresentationError):
    """The artifact no longer has the bytes to which the document was bound."""


def load_presentation(paths: ScenePaths, artifact: str) -> dict[str, Any]:
    """Load one artifact's composition, or return an unpersisted default."""
    relative, artifact_hash = _resolve_artifact(paths, artifact)
    destination = _document_path(paths, relative)
    if not destination.is_file():
        return _default_document(relative, artifact_hash)

    document = _read_document(destination)
    normalized = _validate_document(document, relative, artifact_hash)
    stored_hash = normalized["artifact"]["sha256"]
    if stored_hash != artifact_hash:
        raise PresentationArtifactChangedError(
            "presentation artifact changed; create or select composition metadata for the new artifact bytes"
        )
    expected_revision = _revision_for(normalized)
    if normalized["revision"] != expected_revision:
        raise PresentationValidationError("presentation revision does not match its content")
    return normalized


def save_presentation(paths: ScenePaths, document: Mapping[str, object]) -> dict[str, Any]:
    """Validate and atomically save one complete document with CAS semantics.

    The caller must already hold the scene mutation lock.  ``revision`` is the
    expected current revision; a stale or missing value cannot overwrite a
    newer file.
    """
    relative = _document_artifact_path(document)
    relative, artifact_hash = _resolve_artifact(paths, relative)
    requested = _validate_document(document, relative, artifact_hash)
    if requested["artifact"]["sha256"] != artifact_hash:
        raise PresentationArtifactChangedError(
            "presentation artifact changed; reload it before editing"
        )

    current = load_presentation(paths, relative)
    if requested["revision"] != current["revision"]:
        raise PresentationChangedError(
            "presentation changed; reload it before editing"
        )

    saved = {key: value for key, value in requested.items() if key != "revision"}
    saved["revision"] = _revision_for(saved)
    # Keep the stable schema ordering rather than allowing revision to drift to
    # the end according to whichever client encoded the request.
    saved = _ordered_document(saved)
    encoded = _encode_document(saved)
    if len(encoded) > _MAX_DOCUMENT_BYTES:
        raise PresentationValidationError("presentation document is too large")
    _atomic_write(_document_path(paths, relative), encoded)
    return saved


def _default_document(relative: str, artifact_hash: str) -> dict[str, Any]:
    return {
        "version": _SCHEMA_VERSION,
        "artifact": {"relative_path": relative, "sha256": artifact_hash},
        "revision": None,
        "frame": {"origin": _vector(0.0, 0.0, 0.0)},
        "default_pivot": None,
        "focus_points": [],
        "views": [],
        "home_view_id": None,
    }


def _ordered_document(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": document["version"],
        "artifact": document["artifact"],
        "revision": document["revision"],
        "frame": document["frame"],
        "default_pivot": document["default_pivot"],
        "focus_points": document["focus_points"],
        "views": document["views"],
        "home_view_id": document["home_view_id"],
    }


def _validate_document(
    value: Mapping[str, object],
    relative: str,
    artifact_hash: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PresentationValidationError("presentation must be an object")
    if frozenset(value) != _DOCUMENT_KEYS:
        raise PresentationValidationError("presentation has missing or unsupported fields")
    if value.get("version") != _SCHEMA_VERSION:
        raise PresentationValidationError(
            f"presentation version must be {_SCHEMA_VERSION}"
        )

    artifact = _object(value.get("artifact"), "artifact")
    if frozenset(artifact) != {"relative_path", "sha256"}:
        raise PresentationValidationError("artifact has missing or unsupported fields")
    if artifact.get("relative_path") != relative:
        raise PresentationValidationError("presentation artifact path does not match the selected artifact")
    bound_hash = _string(artifact.get("sha256"), "artifact.sha256", maximum=64)
    if len(bound_hash) != 64 or any(character not in "0123456789abcdef" for character in bound_hash):
        raise PresentationValidationError("artifact.sha256 must be a lowercase SHA-256 digest")

    revision_value = value.get("revision")
    if revision_value is not None:
        revision = _string(revision_value, "revision", maximum=64)
        if len(revision) != 64 or any(character not in "0123456789abcdef" for character in revision):
            raise PresentationValidationError("revision must be a lowercase SHA-256 digest")
    else:
        revision = None

    frame = _object(value.get("frame"), "frame")
    if frozenset(frame) != {"origin"}:
        raise PresentationValidationError("frame has missing or unsupported fields")
    normalized_frame = {"origin": _validate_vector(frame.get("origin"), "frame.origin")}

    pivot_value = value.get("default_pivot")
    pivot = None if pivot_value is None else _validate_vector(pivot_value, "default_pivot")

    focus_values = _array(value.get("focus_points"), "focus_points")
    if len(focus_values) > _MAX_ITEMS:
        raise PresentationValidationError("focus_points contains too many items")
    focus_points = [
        _validate_focus_point(item, index)
        for index, item in enumerate(focus_values)
    ]
    _require_unique_ids(focus_points, "focus_points")

    view_values = _array(value.get("views"), "views")
    if len(view_values) > _MAX_ITEMS:
        raise PresentationValidationError("views contains too many items")
    views = [_validate_view(item, index) for index, item in enumerate(view_values)]
    _require_unique_ids(views, "views")

    home_value = value.get("home_view_id")
    home_view_id = None if home_value is None else _string(home_value, "home_view_id")
    if home_view_id is not None and home_view_id not in {view["id"] for view in views}:
        raise PresentationValidationError("home_view_id must reference an existing view")

    # ``artifact_hash`` is intentionally not substituted here: comparing the
    # client/stored binding to current bytes is how callers detect staleness.
    return {
        "version": _SCHEMA_VERSION,
        "artifact": {"relative_path": relative, "sha256": bound_hash},
        "revision": revision,
        "frame": normalized_frame,
        "default_pivot": pivot,
        "focus_points": focus_points,
        "views": views,
        "home_view_id": home_view_id,
    }


def _validate_focus_point(value: object, index: int) -> dict[str, Any]:
    name = f"focus_points[{index}]"
    item = _object(value, name)
    if frozenset(item) != {"id", "name", "position", "framing_radius"}:
        raise PresentationValidationError(f"{name} has missing or unsupported fields")
    radius = _number(item.get("framing_radius"), f"{name}.framing_radius")
    if radius <= 0:
        raise PresentationValidationError(f"{name}.framing_radius must be greater than zero")
    return {
        "id": _identifier(item.get("id"), f"{name}.id"),
        "name": _string(item.get("name"), f"{name}.name", maximum=200),
        "position": _validate_vector(item.get("position"), f"{name}.position"),
        "framing_radius": radius,
    }


def _validate_view(value: object, index: int) -> dict[str, Any]:
    name = f"views[{index}]"
    item = _object(value, name)
    expected = {"id", "name", "position", "target", "up", "vertical_fov_degrees"}
    if frozenset(item) != expected:
        raise PresentationValidationError(f"{name} has missing or unsupported fields")
    position = _validate_vector(item.get("position"), f"{name}.position")
    target = _validate_vector(item.get("target"), f"{name}.target")
    up = _validate_vector(item.get("up"), f"{name}.up")
    forward_length_squared = _distance_squared(position, target)
    if forward_length_squared <= 1.0e-18:
        raise PresentationValidationError(f"{name} position and target must differ")
    up_length_squared = _distance_squared(up, _vector(0.0, 0.0, 0.0))
    if up_length_squared <= 1.0e-18:
        raise PresentationValidationError(f"{name}.up must be non-zero")
    forward = _vector(
        target["x"] - position["x"],
        target["y"] - position["y"],
        target["z"] - position["z"],
    )
    if _cross_length_squared(forward, up) <= 1.0e-12 * forward_length_squared * up_length_squared:
        raise PresentationValidationError(f"{name}.up must not be parallel to the viewing direction")
    fov = _number(item.get("vertical_fov_degrees"), f"{name}.vertical_fov_degrees")
    if not 1.0 <= fov <= 179.0:
        raise PresentationValidationError(f"{name}.vertical_fov_degrees must be between 1 and 179")
    return {
        "id": _identifier(item.get("id"), f"{name}.id"),
        "name": _string(item.get("name"), f"{name}.name", maximum=200),
        "position": position,
        "target": target,
        "up": up,
        "vertical_fov_degrees": fov,
    }


def _validate_vector(value: object, name: str) -> dict[str, float]:
    item = _object(value, name)
    if frozenset(item) != {"x", "y", "z"}:
        raise PresentationValidationError(f"{name} must contain exactly x, y, and z")
    return {
        "x": _number(item.get("x"), f"{name}.x"),
        "y": _number(item.get("y"), f"{name}.y"),
        "z": _number(item.get("z"), f"{name}.z"),
    }


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PresentationValidationError(f"{name} must be an object")
    return value  # type: ignore[return-value]


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise PresentationValidationError(f"{name} must be an array")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresentationValidationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or abs(number) > _MAX_COORDINATE:
        raise PresentationValidationError(f"{name} must be finite and within supported bounds")
    return number


def _string(value: object, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PresentationValidationError(
            f"{name} must be a non-empty string of at most {maximum} characters"
        )
    return value


def _identifier(value: object, name: str) -> str:
    identifier = _string(value, name)
    if identifier in {".", ".."} or any(character in identifier for character in "/\\\0"):
        raise PresentationValidationError(f"{name} is not a valid identifier")
    return identifier


def _require_unique_ids(items: list[dict[str, Any]], name: str) -> None:
    identifiers = [item["id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise PresentationValidationError(f"{name} ids must be unique")


def _document_artifact_path(document: Mapping[str, object]) -> str:
    artifact = _object(document.get("artifact"), "artifact")
    return _string(artifact.get("relative_path"), "artifact.relative_path", maximum=1_024)


def _resolve_artifact(paths: ScenePaths, relative: str) -> tuple[str, str]:
    matches = [
        artifact
        for artifact in list_artifacts(paths)
        if artifact.relative_path == relative and artifact.format in _NATIVE_FORMATS
    ]
    if len(matches) != 1:
        raise PresentationValidationError(
            "artifact must be a scene-owned PLY, SPZ, or SPLAT export"
        )
    path = (paths.root / relative).resolve()
    return relative, _fingerprint_artifact(path)


@lru_cache(maxsize=64)
def _fingerprint_snapshot(path_value: str, size: int, modified_ns: int, inode: int) -> str:
    path = Path(path_value)
    before = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (size, modified_ns, inode):
        raise PresentationArtifactChangedError("The artifact changed before it could be fingerprinted.")
    digest = fingerprint_file(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (size, modified_ns, inode):
        raise PresentationArtifactChangedError("The artifact changed while it was being fingerprinted.")
    return digest


def _fingerprint_artifact(path: Path) -> str:
    stat = path.stat()
    return _fingerprint_snapshot(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _document_path(paths: ScenePaths, relative: str) -> Path:
    identifier = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:32]
    return paths.presentations / f"{identifier}.json"


def _read_document(path: Path) -> Mapping[str, object]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_DOCUMENT_BYTES:
            raise PresentationValidationError("presentation document has an invalid size")
        value = json.loads(path.read_text(encoding="utf-8"))
    except PresentationValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PresentationValidationError("presentation document is unreadable") from exc
    if not isinstance(value, dict):
        raise PresentationValidationError("presentation document must be an object")
    return value


def _revision_for(document: Mapping[str, Any]) -> str:
    content = {key: value for key, value in document.items() if key != "revision"}
    canonical = json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _encode_document(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _vector(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _distance_squared(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return sum((first[axis] - second[axis]) ** 2 for axis in ("x", "y", "z"))


def _cross_length_squared(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    cross_x = first["y"] * second["z"] - first["z"] * second["y"]
    cross_y = first["z"] * second["x"] - first["x"] * second["z"]
    cross_z = first["x"] * second["y"] - first["y"] * second["x"]
    return cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
