#!/usr/bin/env python3
"""Monitor Metashape cameras export, then run brusher (with optional colmapper fallback)."""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

from gausser.colmap_model import model_format, read_model_stats
from gausser.process import run_command

POLL_INTERVAL = 30        # check every 30s
STABILITY_WINDOW = 90     # wait 90s of no new/changed files before declaring done
INITIAL_START_DELAY = 600 # give ~10 min for export to begin (user said 15 min)

# ── Detection helpers ──────────────────────────────────────────────────────────

def _has_triplet(dir: Path) -> bool:
    """Check if a directory has a COLMAP model (3 required files)."""
    return model_format(dir) is not None


def has_colmap_structure(scene: Path) -> bool:
    """Check for COLMAP sparse model.

    Metashape can export in different layouts:
      - sparse/0/{cameras,images,points3D}.bin   (standard COLMAP)
      - 0/{cameras,images,points3D}.bin           (no sparse wrapper)
      - {cameras,images,points3D}.bin              (flat in export dir)
      - {cameras,images,points3D}.txt              (text format variant)
    """
    if _has_triplet(scene):
        return True
    sparse = scene / "sparse"
    if sparse.is_dir():
        if _has_triplet(sparse):
            return True
        for sub in sparse.iterdir():
            if sub.is_dir() and _has_triplet(sub):
                return True
    # Also check bare numbered dirs like <scene>/0/
    for d in scene.iterdir():
        if d.is_dir() and d.name.isdigit():
            if _has_triplet(d):
                return True
    return False


def find_files(scene: Path) -> set[Path]:
    """Recursively list all files (skipping .DS_Store)."""
    files = set()
    for f in scene.rglob("*"):
        if f.is_file() and f.name != ".DS_Store":
            files.add(f)
    return files


def get_mtime_snapshot(scene: Path) -> dict[Path, float]:
    """Get {path: mtime} for all files."""
    snap = {}
    for f in scene.rglob("*"):
        if f.is_file() and f.name != ".DS_Store":
            try:
                snap[f] = os.path.getmtime(f)
            except OSError:
                pass
    return snap


def count_images(scene: Path) -> int:
    """Count images in scene/images/."""
    img_dir = scene / "images"
    if not img_dir.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sum(1 for f in img_dir.rglob("*") if f.suffix.lower() in exts)


def scene_summary(scene: Path) -> str:
    """Quick summary of what's in the scene dir."""
    parts = []
    n_imgs = count_images(scene)
    parts.append(f"{n_imgs} images")
    if has_colmap_structure(scene):
        parts.append("COLMAP sparse ✓")
    elif (scene / "sparse").is_dir():
        parts.append("COLMAP sparse (incomplete)")
    if (scene / "transforms.json").exists():
        parts.append("transforms.json ✓")
    if (scene / "exports").is_dir():
        exports = list((scene / "exports").glob("*.ply"))
        parts.append(f"{len(exports)} exports")
    return ", ".join(parts) if parts else "(empty)"


def read_model_stats(model_dir: Path) -> tuple[int, int]:
    """Read registered images and 3D points from a COLMAP sparse model dir."""
    stats = read_model_stats(model_dir)
    return (stats.images, stats.points) if stats is not None else (0, 0)


def _find_model_dir(scene: Path) -> Path | None:
    """Find the directory containing the COLMAP model files."""
    candidates = [scene, scene / "sparse"]
    if (scene / "sparse").is_dir():
        for sub in (scene / "sparse").iterdir():
            if sub.is_dir():
                candidates.append(sub)
    for d in scene.iterdir():
        if d.is_dir() and d.name.isdigit():
            candidates.append(d)
    for c in candidates:
        if _has_triplet(c):
            return c
    return None


def validate_colmap(scene: Path) -> bool:
    """Validate the COLMAP sparse model is usable."""
    model_dir = _find_model_dir(scene)
    if model_dir is None:
        return False
    n_imgs, n_pts = read_model_stats(model_dir)
    total = count_images(scene)
    print(f"  ✓ COLMAP model: {n_imgs}/{total} registered, {n_pts:,} 3D points")
    return n_imgs > 0


# ── Monitoring ─────────────────────────────────────────────────────────────────

def wait_for_export(scene: Path) -> bool:
    """Poll for COLMAP export. Returns True if export detected."""
    print(f"┌─ Scene: {scene}")
    print(f"├─ {scene_summary(scene)}")
    print(f"├─ Polling every {POLL_INTERVAL}s, stability window {STABILITY_WINDOW}s")
    print(f"└─ Waiting up to 3 hours...")
    print()

    # Phase 1: Wait for any new files to appear (export to start)
    print("⏳ Phase 1: Waiting for Metashape export to begin...")
    initial_files = find_files(scene)
    started = time.monotonic()
    headroom = INITIAL_START_DELAY

    while True:
        elapsed = time.monotonic() - started
        if elapsed > 10800:  # 3h timeout
            print("❌ Timeout waiting for export after 3 hours.")
            return False

        current_files = find_files(scene)
        new_files = current_files - initial_files

        if new_files:
            print(f"  ✓ Export started! ({len(new_files)} new file(s) detected)")
            print(f"  New: {[str(f.relative_to(scene)) for f in sorted(new_files)[:5]]}")
            break
        if elapsed > headroom:
            if elapsed % 120 < POLL_INTERVAL:
                print(f"  ⏳ Still waiting... ({int(elapsed)}s elapsed)")

        # After initial delay, check if export already happened
        if elapsed > INITIAL_START_DELAY:
            if has_colmap_structure(scene):
                print(f"  ✓ COLMAP structure already present!")
                return True

        time.sleep(POLL_INTERVAL)

    # Phase 2: Wait for stability (no changes for STABILITY_WINDOW)
    print("\n⏳ Phase 2: Waiting for export to stabilize...")
    last_snapshot = get_mtime_snapshot(scene)
    stable_since = time.monotonic()

    while True:
        elapsed = time.monotonic() - started
        if elapsed > 10800:
            print("❌ Timeout.")
            return False

        time.sleep(POLL_INTERVAL)
        current_snapshot = get_mtime_snapshot(scene)

        # Check for changes
        changed = False
        for path, mtime in current_snapshot.items():
            old_mtime = last_snapshot.get(path)
            if old_mtime is None or abs(mtime - old_mtime) > 0.01:
                changed = True
                break
        # Check for new files
        if len(current_snapshot) != len(last_snapshot):
            changed = True

        if changed:
            stable_since = time.monotonic()
            last_snapshot = current_snapshot
            print(f"  ↻ File change detected, resetting stability timer ({int(elapsed)}s elapsed)")
            continue

        stable_for = time.monotonic() - stable_since
        if stable_for >= STABILITY_WINDOW:
            print(f"  ✓ Export stable for {STABILITY_WINDOW}s")
            break

        print(f"  · Stable for {int(stable_for)}s/{STABILITY_WINDOW}s...")

    print(f"\n📋 Final scene summary: {scene_summary(scene)}")
    return True


def run(cmd: list[str], desc: str, *, cwd: Path) -> bool:
    """Run a subprocess in its owning scene and print output live."""
    print(f"\n{'='*60}")
    print(f"▶  {desc}")
    print(f"   {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = run_command(cmd, cwd=cwd, on_line=print)
    if result.succeeded:
        print(f"\n✅ {desc} — success (exit code 0)")
    elif result.cancelled:
        print(f"\n⚠️  {desc} — interrupted.")
    else:
        print(f"\n❌ {desc} — failed (exit code {result.returncode})")
        if result.output_tail:
            print("Last output:")
            print("\n".join(result.output_tail[-20:]))
    return result.succeeded


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Wait for an optional Metashape export, validate COLMAP, then train with Brush.",
    )
    parser.add_argument("scene", type=Path, help="Scene root containing images/")
    parser.add_argument("--no-wait", action="store_true", help="Fail instead of waiting for a missing export")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive", "vocab_tree"], default="sequential")
    args = parser.parse_args(argv)
    scene = args.scene.expanduser().resolve()

    if not scene.is_dir():
        parser.error(f"scene directory not found: {scene}")

    ok = False
    if has_colmap_structure(scene):
        print(f"✅ COLMAP structure already present in {scene}")
        ok = validate_colmap(scene)
        if ok:
            print("\n✓ COLMAP model valid! Proceeding directly to brusher...")
        else:
            print("\n⚠️  COLMAP structure invalid. Reconstructing cameras with colmapper.")
    elif args.no_wait:
        print("❌ No COLMAP export found and --no-wait was requested.")
        sys.exit(1)
    else:
        print("⏳ No COLMAP export found; waiting for Metashape export.")
        if not wait_for_export(scene):
            sys.exit(1)
        ok = has_colmap_structure(scene) and validate_colmap(scene)

    if not ok:
        print("\n🔄 Running colmapper to reconstruct cameras from images...")
        ok = run(
            ["colmapper", "--scene", str(scene), "--matcher", args.matcher],
            "COLMAP SfM Reconstruction",
            cwd=scene,
        )
    if not ok:
        print("❌ Failed to get a valid COLMAP reconstruction. Aborting.")
        sys.exit(1)

    n_imgs = count_images(scene)
    print(f"\n🖌️  Running brusher on {n_imgs} images...")
    ok = run(
        [
            "brusher", str(scene),
            "--total-train-iters", "30000",
            "--export-every", "5000",
            "--eval-every", "2000",
            "--refine-every", "200",
            "--max-resolution", "1600",
            "--max-splats", "2000000",
        ],
        "Brush 3DGS Training",
        cwd=scene,
    )
    if not ok:
        print("\n❌ brusher failed. Check the scene directory for partial results.")
        sys.exit(1)

    print(f"\n🎉 Training completed. Run artifacts are under {scene / 'runs' / 'brush'}.")


if __name__ == "__main__":
    main()
