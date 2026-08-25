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
    cwd: Path,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> None:
    cmd = [colmap_bin, subcommand]
    for key, value in args.items():
        cmd += [f"--{key}", str(value)]
    result = run_colmap(cmd, cwd=cwd, on_progress=on_progress, on_line=on_line, total_images=total_images)
    if not result.succeeded:
        tail = "\n".join(result.output_tail[-20:])
        details = f"\nLast output:\n{tail}" if tail else ""
        raise ColmapError(f"`colmap {subcommand}` exited with code {result.returncode}.{details}")


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
    _colmap(
        colmap_bin, "feature_extractor", opts, cwd=database_path.parent,
        on_progress=on_progress, on_line=on_line, total_images=total_images,
    )


def feature_matching(
    colmap_bin: str,
    database_path: Path,
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    vocab_tree_path: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
) -> None:
    opts = {
        "database_path": str(database_path),
        "FeatureMatching.use_gpu": "1" if use_gpu else "0",
    }
    if matcher == "vocab_tree":
        if vocab_tree_path is None:
            raise ColmapError("vocab_tree matcher requires a vocabulary tree path")
        opts["VocabTreeMatching.vocab_tree_path"] = str(vocab_tree_path)
    _colmap(
        colmap_bin, f"{matcher}_matcher", opts,
        cwd=database_path.parent, on_progress=on_progress, on_line=on_line,
    )


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
    _colmap(
        colmap_bin, "mapper", {
            "database_path": str(database_path),
            "image_path": str(image_path),
            "output_path": str(output_path),
        }, cwd=database_path.parent, on_progress=on_progress, on_line=on_line, total_images=total_images,
    )
