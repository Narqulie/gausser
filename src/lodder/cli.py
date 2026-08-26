"""Create validated, streamed PlayCanvas LOD artifacts from a scene-owned PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from gausser.manifest import fingerprint_file, record_stage
from gausser.process import run_command
from gausser.publish import replace_directory
from gausser.scene import SceneError, ScenePaths


TOOL = "splat-transform"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a scene-owned streamed PlayCanvas LOD artifact from an explicit PLY export.",
    )
    parser.add_argument("source", type=Path, help="Exact scene-owned .ply export to convert")
    parser.add_argument("--scene", type=Path, default=Path.cwd(), help="Scene root (default: current directory)")
    parser.add_argument("--chunk-count", type=int, default=512, help="Approximate splats per LOD chunk in K (default: 512)")
    parser.add_argument("--chunk-extent", type=float, default=16.0, help="Approximate LOD chunk extent in metres (default: 16)")
    parser.add_argument(
        "--lod-percentages", default="100,50,25", metavar="PCTS",
        help="Descending retained-splat percentages, starting with 100 (default: 100,50,25)",
    )
    return parser


def _parse_percentages(raw: str, parser: argparse.ArgumentParser) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError:
        parser.error("--lod-percentages must be comma-separated integers")
    if len(values) < 2 or values[0] != 100 or any(not 0 < value <= 100 for value in values):
        parser.error("--lod-percentages needs at least two values in 1..100, beginning with 100")
    if any(later >= earlier for earlier, later in zip(values, values[1:])):
        parser.error("--lod-percentages must be strictly descending")
    return values


def _tool_version(paths: ScenePaths) -> str:
    result = run_command([TOOL, "--version"], cwd=paths.root)
    if not result.succeeded:
        raise RuntimeError(f"{TOOL} --version failed")
    return next((line.strip() for line in result.output_tail if line.strip()), "unknown")


def _validate_output(staged: Path) -> dict[str, Any]:
    metadata = staged / "lod-meta.json"
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid LOD metadata: {metadata}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("lodLevels"), int):
        raise ValueError("LOD metadata lacks an integer lodLevels")
    filenames = document.get("filenames")
    if not isinstance(filenames, list) or not filenames or not all(isinstance(name, str) and name for name in filenames):
        raise ValueError("LOD metadata lacks a non-empty filenames list")
    root = staged.resolve()

    def checked_path(parent: Path, name: str) -> Path:
        candidate = (parent / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"LOD metadata references a path outside its artifact: {name}") from exc
        if not candidate.is_file():
            raise ValueError(f"LOD metadata references a missing file: {name}")
        return candidate

    for name in filenames:
        chunk_metadata = checked_path(staged, name)
        try:
            chunk = json.loads(chunk_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid LOD chunk metadata: {chunk_metadata}") from exc
        stack: list[Any] = [chunk]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                files = value.get("files")
                if files is not None:
                    if not isinstance(files, list) or not all(isinstance(file, str) and file for file in files):
                        raise ValueError(f"invalid files list in LOD chunk metadata: {chunk_metadata}")
                    for file in files:
                        checked_path(chunk_metadata.parent, file)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return document


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.chunk_count <= 0:
        parser.error("--chunk-count must be positive")
    if args.chunk_extent <= 0:
        parser.error("--chunk-extent must be positive")
    if shutil.which(TOOL) is None:
        parser.error(f"{TOOL} is not on PATH; install @playcanvas/splat-transform")
    percentages = _parse_percentages(args.lod_percentages, parser)

    paths = ScenePaths.from_root(args.scene)
    source = args.source.expanduser().resolve()
    if source.suffix.lower() != ".ply":
        parser.error(f"source must be a .ply export: {source}")
    if not source.is_file():
        parser.error(f"source PLY not found: {source}")
    try:
        source_relative = paths.relative_path(source)
    except SceneError as exc:
        parser.error(str(exc))

    source_sha256 = fingerprint_file(source)
    run_id = paths.new_run_id()
    staged = paths.staging_attempt("lod", run_id)
    output = paths.training_run("lod", run_id)
    staged.mkdir(parents=True)
    metadata = staged / "lod-meta.json"
    levels = staged / "levels"

    try:
        tool_version = _tool_version(paths)
        level_paths = [source]
        for level, percentage in enumerate(percentages[1:], start=1):
            level_path = levels / f"lod-{level}.ply"
            levels.mkdir(parents=True, exist_ok=True)
            command = [TOOL, str(source), "--decimate", f"{percentage}%", str(level_path)]
            print(f"Creating LOD {level} at {percentage}%…")
            result = run_command(command, cwd=paths.root, on_line=print)
            if not result.succeeded or not level_path.is_file():
                detail = "\n".join(result.output_tail[-20:])
                raise RuntimeError(f"{TOOL} failed to create LOD {level}" + (f"\n{detail}" if detail else ""))
            level_paths.append(level_path)

        command = [TOOL, "-C", str(args.chunk_count), "-X", f"{args.chunk_extent:g}"]
        for level, level_path in enumerate(level_paths):
            command.extend((str(level_path), "-l", str(level)))
        command.append(str(metadata))
        print(f"Bundling {len(level_paths)} LOD levels from {source_relative}…")
        result = run_command(command, cwd=paths.root, on_line=print)
        if not result.succeeded:
            detail = "\n".join(result.output_tail[-20:])
            raise RuntimeError(f"{TOOL} failed with exit code {result.returncode}" + (f"\n{detail}" if detail else ""))
        shutil.rmtree(levels)
        document = _validate_output(staged)
        replace_directory(staged, output)
        record_stage(
            paths,
            "lod",
            {
                "source": str(source_relative),
                "source_sha256": source_sha256,
                "tool": TOOL,
                "tool_version": tool_version,
                "chunk_count_k": args.chunk_count,
                "chunk_extent_m": args.chunk_extent,
                "lod_percentages": percentages,
                "lod_levels": document["lodLevels"],
                "output": str(paths.relative_path(output / "lod-meta.json")),
            },
        )
    except (RuntimeError, ValueError) as exc:
        print(f"❌ LOD creation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"✅ LOD artifact created: {output / 'lod-meta.json'}")


if __name__ == "__main__":
    main()
