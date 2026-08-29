"""Bounded validation and atomic rotation for msplat v1 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from tempfile import NamedTemporaryFile
import time
from typing import Any, Mapping
from uuid import uuid4


_CHECKPOINT_MAGIC = 0x4C50534D
_CHECKPOINT_VERSION = 1
_HEADER = struct.Struct("<IIIIII8f")
_TENSOR_COUNT = 18
_GENERATIONS = ("a", "b")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CheckpointFormatError(ValueError):
    """Raised when a checkpoint or resume receipt is unsafe or inconsistent."""


@dataclass(frozen=True)
class CheckpointSummary:
    iteration: int
    num_active: int
    sh_degree: int
    adam_step_count: int
    tensor_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ResumeSelection:
    path: Path
    record: dict[str, Any]
    used_fallback: bool


@dataclass(frozen=True)
class CheckpointReceipt:
    document: dict[str, Any]
    path: Path
    iteration: int
    num_active: int
    size_bytes: int
    duration_seconds: float


def inspect_checkpoint(path: Path) -> CheckpointSummary:
    """Validate one v1.1.4 checkpoint without allocating from its dimensions.

    Upstream's loader trusts tensor dimensions and byte counts before allocating
    GPU buffers. This parser checks the entire file, including exact EOF, before
    Gausser permits ``load_checkpoint`` to see it.
    """
    checkpoint = Path(path)
    try:
        file_size = checkpoint.stat().st_size
    except OSError as exc:
        raise CheckpointFormatError(f"checkpoint is unavailable: {checkpoint}") from exc
    if file_size < _HEADER.size:
        raise CheckpointFormatError("checkpoint is truncated before its header")

    try:
        with checkpoint.open("rb") as stream:
            header = _read_exact(stream, _HEADER.size, "checkpoint header")
            (
                magic,
                version,
                iteration,
                num_active,
                sh_degree,
                adam_step_count,
                *learning_rates,
            ) = _HEADER.unpack(header)
            if magic != _CHECKPOINT_MAGIC:
                raise CheckpointFormatError("checkpoint magic is not MSPL")
            if version != _CHECKPOINT_VERSION:
                raise CheckpointFormatError(
                    f"unsupported checkpoint version: {version}"
                )
            if iteration < 1:
                raise CheckpointFormatError("checkpoint iteration must be positive")
            if num_active < 1 or num_active > file_size // 4:
                raise CheckpointFormatError("checkpoint active-Gaussian count is invalid")
            if sh_degree > 3:
                raise CheckpointFormatError("checkpoint SH degree exceeds v1.1.4 limits")
            if adam_step_count > iteration:
                raise CheckpointFormatError(
                    "checkpoint Adam step count exceeds its iteration"
                )
            if any(not math.isfinite(value) or value < 0 for value in learning_rates):
                raise CheckpointFormatError("checkpoint learning rates are invalid")

            shapes: list[tuple[int, ...]] = []
            for index in range(_TENSOR_COUNT):
                ndim = struct.unpack(
                    "<I", _read_exact(stream, 4, f"tensor {index} rank")
                )[0]
                if ndim < 1 or ndim > 3:
                    raise CheckpointFormatError(
                        f"tensor {index} rank {ndim} is outside the supported range"
                    )
                shape = struct.unpack(
                    f"<{ndim}q",
                    _read_exact(stream, ndim * 8, f"tensor {index} shape"),
                )
                if any(dimension < 1 or dimension > file_size // 4 for dimension in shape):
                    raise CheckpointFormatError(f"tensor {index} shape is invalid")
                byte_count = struct.unpack(
                    "<Q", _read_exact(stream, 8, f"tensor {index} byte count")
                )[0]
                expected_bytes = math.prod(shape) * 4
                if byte_count != expected_bytes:
                    raise CheckpointFormatError(
                        f"tensor {index} byte count does not match its shape"
                    )
                remaining = file_size - stream.tell()
                if byte_count > remaining:
                    raise CheckpointFormatError(f"tensor {index} payload is truncated")
                stream.seek(byte_count, os.SEEK_CUR)
                shapes.append(tuple(int(value) for value in shape))

            if stream.tell() != file_size:
                raise CheckpointFormatError("checkpoint contains trailing bytes")
    except OSError as exc:
        raise CheckpointFormatError(f"checkpoint cannot be read: {checkpoint}") from exc

    parameter_shapes = tuple(shapes[:6])
    expected_parameter_shapes = (
        (num_active, 3),
        (num_active, 3),
        (num_active, 4),
        (num_active, 3),
        (num_active, 15, 3),
        (num_active, 1),
    )
    if parameter_shapes != expected_parameter_shapes:
        raise CheckpointFormatError(
            "checkpoint parameter tensor shapes do not match msplat v1.1.4"
        )
    if tuple(shapes[6:12]) != parameter_shapes or tuple(shapes[12:18]) != parameter_shapes:
        raise CheckpointFormatError(
            "checkpoint optimizer tensor shapes do not match its parameters"
        )

    return CheckpointSummary(
        iteration=iteration,
        num_active=num_active,
        sh_degree=sh_degree,
        adam_step_count=adam_step_count,
        tensor_shapes=tuple(shapes),
    )


def initialize_resume_document(run_id: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "run_id": run_id,
        "identity": dict(identity),
        "current": None,
        "previous": None,
        "history": [],
    }


def load_resume_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError(f"invalid resume receipt: {path}") from exc
    _validate_resume_document(document)
    return document


def write_resume_document(path: Path, document: Mapping[str, Any]) -> None:
    materialized = dict(document)
    _validate_resume_document(materialized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f"{destination.name}-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(materialized, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint_generation(
    trainer: Any,
    checkpoints: Path,
    receipt_path: Path,
    document: Mapping[str, Any],
    *,
    iteration: int,
) -> CheckpointReceipt:
    """Save, fsync, structurally validate, promote, hash, then record one slot."""
    current = document.get("current")
    current_generation = current.get("generation") if isinstance(current, dict) else None
    generation = "b" if current_generation == "a" else "a"
    filename = f"checkpoint-{generation}.msplat"
    directory = Path(checkpoints)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = directory / f".{filename}.{uuid4().hex}.tmp"
    started = time.monotonic()
    try:
        trainer.save_checkpoint(str(temporary))
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        summary = inspect_checkpoint(temporary)
        if summary.iteration != iteration:
            raise CheckpointFormatError(
                f"checkpoint recorded iteration {summary.iteration}, expected {iteration}"
            )
        temporary.replace(destination)
        _fsync_directory(directory)
        size_bytes = destination.stat().st_size
        digest = _sha256(destination)
        duration = time.monotonic() - started
        record: dict[str, Any] = {
            "generation": generation,
            "filename": filename,
            "iteration": iteration,
            "num_active": summary.num_active,
            "size_bytes": size_bytes,
            "sha256": digest,
            "duration_seconds": duration,
        }
        updated = dict(document)
        updated["current"] = record
        updated["previous"] = dict(current) if isinstance(current, dict) else None
        history = updated.get("history")
        updated["history"] = (
            [dict(item) for item in history if isinstance(item, dict)]
            if isinstance(history, list)
            else []
        )
        updated["history"].append(dict(record))
        write_resume_document(receipt_path, updated)
        return CheckpointReceipt(
            document=updated,
            path=destination,
            iteration=iteration,
            num_active=summary.num_active,
            size_bytes=size_bytes,
            duration_seconds=duration,
        )
    finally:
        temporary.unlink(missing_ok=True)


def select_resume_generation(
    checkpoints: Path,
    document: Mapping[str, Any],
    *,
    maximum_iteration: int,
) -> ResumeSelection | None:
    """Return the newest safe generation, falling back exactly one slot."""
    records = (document.get("current"), document.get("previous"))
    if records == (None, None):
        return None
    failures: list[str] = []
    for index, record_value in enumerate(records):
        if not isinstance(record_value, dict):
            continue
        record = dict(record_value)
        path = Path(checkpoints) / str(record["filename"])
        try:
            if path.stat().st_size != record["size_bytes"]:
                raise CheckpointFormatError("recorded size does not match")
            if _sha256(path) != record["sha256"]:
                raise CheckpointFormatError("recorded hash does not match")
            summary = inspect_checkpoint(path)
            if summary.iteration != record["iteration"]:
                raise CheckpointFormatError("recorded iteration does not match")
            if summary.iteration > maximum_iteration:
                raise CheckpointFormatError("iteration exceeds the configured target")
            return ResumeSelection(path=path, record=record, used_fallback=index == 1)
        except (OSError, CheckpointFormatError) as exc:
            failures.append(f"{record.get('generation', '?')}: {exc}")
    raise CheckpointFormatError(
        "no structurally valid checkpoint generation remains (" + "; ".join(failures) + ")"
    )


def activate_fallback(
    receipt_path: Path,
    document: Mapping[str, Any],
    selection: ResumeSelection,
) -> dict[str, Any]:
    if not selection.used_fallback:
        return dict(document)
    updated = dict(document)
    updated["current"] = dict(selection.record)
    updated["previous"] = None
    write_resume_document(receipt_path, updated)
    return updated


def _validate_resume_document(document: object) -> None:
    if not isinstance(document, dict) or document.get("version") != 1:
        raise CheckpointFormatError("unsupported resume receipt")
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CheckpointFormatError("resume receipt has no run identity")
    if not isinstance(document.get("identity"), dict):
        raise CheckpointFormatError("resume receipt has no input/config identity")
    generations: list[str] = []
    for field in ("current", "previous"):
        record = document.get(field)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise CheckpointFormatError(f"resume receipt {field} generation is invalid")
        _validate_generation_record(record, field)
        generations.append(str(record["generation"]))
    if len(generations) != len(set(generations)):
        raise CheckpointFormatError("resume receipt generations must use distinct slots")
    history = document.get("history")
    if not isinstance(history, list):
        raise CheckpointFormatError("resume receipt checkpoint history is invalid")
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            raise CheckpointFormatError("resume receipt checkpoint history is invalid")
        _validate_generation_record(record, f"history item {index}")


def _validate_generation_record(record: Mapping[str, Any], label: str) -> None:
    generation = record.get("generation")
    filename = record.get("filename")
    iteration = record.get("iteration")
    num_active = record.get("num_active")
    size_bytes = record.get("size_bytes")
    digest = record.get("sha256")
    duration = record.get("duration_seconds")
    if generation not in _GENERATIONS or filename != f"checkpoint-{generation}.msplat":
        raise CheckpointFormatError(f"resume receipt {label} generation is invalid")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise CheckpointFormatError(f"resume receipt {label} iteration is invalid")
    if isinstance(num_active, bool) or not isinstance(num_active, int) or num_active < 1:
        raise CheckpointFormatError(f"resume receipt {label} splat count is invalid")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
        raise CheckpointFormatError(f"resume receipt {label} size is invalid")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise CheckpointFormatError(f"resume receipt {label} hash is invalid")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        raise CheckpointFormatError(f"resume receipt {label} duration is invalid")


def _read_exact(stream: Any, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise CheckpointFormatError(f"checkpoint is truncated in {label}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
