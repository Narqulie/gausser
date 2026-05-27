"""COLMAP SfM pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .runner import LineCallback, ProgressCallback, run_colmap


class ColmapError(RuntimeError):
    pass


def _colmap(
    colmap_bin: str,
    subcommand: str,
    args: dict[str, str],
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> None:
    cmd = [colmap_bin, subcommand]
    for key, value in args.items():
        cmd += [f"--{key}", str(value)]
    rc = run_colmap(cmd, on_progress=on_progress, on_line=on_line, total_images=total_images)
    if rc != 0:
        raise ColmapError(f"`colmap {subcommand}` exited with code {rc}")


def feature_extraction(
    colmap_bin: str,
    database_path: Path,
    image_path: Path,
    use_gpu: bool = True,
    mask_path: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> None:
    opts: dict[str, str] = {
        "database_path": str(database_path),
        "image_path": str(image_path),
        "ImageReader.single_camera": "1",
        "FeatureExtraction.use_gpu": "1" if use_gpu else "0",
    }
    if mask_path is not None and mask_path.is_dir():
        opts["ImageReader.mask_path"] = str(mask_path)
    _colmap(colmap_bin, "feature_extractor", opts, on_progress=on_progress, on_line=on_line, total_images=total_images)


def feature_matching(
    colmap_bin: str,
    database_path: Path,
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
) -> None:
    _colmap(colmap_bin, f"{matcher}_matcher", {
        "database_path": str(database_path),
        "FeatureMatching.use_gpu": "1" if use_gpu else "0",
    }, on_progress=on_progress, on_line=on_line)


def sparse_reconstruction(
    colmap_bin: str,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    _colmap(colmap_bin, "mapper", {
        "database_path": str(database_path),
        "image_path": str(image_path),
        "output_path": str(output_path),
    }, on_progress=on_progress, on_line=on_line, total_images=total_images)
