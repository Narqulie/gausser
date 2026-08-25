# Workflow failure baselines

These are observed pre-refactor behaviors. They define the initial regression
suite; no real COLMAP, Brush, or SAM3 installation is required to reproduce
them.

## CWD owns COLMAP artifacts

A fake `colmap` executable was invoked through `colmapper` from a directory
different from the scene while `--path <scene>/images` was supplied. The
feature extractor, matcher, and mapper received:

```text
--database_path <caller>/database.db
--image_path <scene>/images
--output_path <caller>/sparse
```

The database and sparse model therefore do not belong to the requested scene.

## CWD owns Brush exports

A fake `brush` executable was invoked through:

```text
brusher <scene> --total-train-iters 10 --export-every 5
```

from a different caller directory. The command received:

```text
--export-path <caller>/exports
```

rather than a scene-owned directory. Its stderr diagnostic was absent from the
CLI output because the runner routes stderr to `DEVNULL`.

## Invalid existing COLMAP output reaches training

With `has_colmap_structure(scene) == True` and `validate_colmap(scene) ==
False`, `monitor_and_run.main()` prints that COLMAP must be rerun but then
invokes `brusher`. The fallback is only reachable on the no-existing-model
branch.

## Text and partial sparse models are unsafe

A `cameras.txt`/`images.txt`/`points3D.txt` triplet is accepted by structural
detection but fails validation because model statistics only read binary
headers. A truncated `images.bin` raises `struct.error` while reading the first
8 bytes instead of producing a failed validation result.

## Remaining code-derived regression cases

- `sammer` considers one existing PNG in each mask directory complete; it does
  not compare outputs to the image inventory.
- Forced masking overwrites successful current masks but does not remove stale
  masks for images now missed or removed.
- A partial SAM exception returns success unless every image failed.
- `colmapper` defaults to GPU mode even on its documented Apple/Homebrew path.
- `brusher` accepts non-positive values and an export cadence at or above the
  total iteration count, although it cannot yield an expected checkpoint.

The pytest suite will reproduce these at the process/CLI boundary with temporary
1×1 images, minimal COLMAP fixtures, and fake executables.
