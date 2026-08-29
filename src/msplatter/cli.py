"""Explicitly unmasked experimental Msplat trainer backend."""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import shutil
import signal
import threading
from typing import Any

from rich.progress import MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from gausser.colmap_model import read_model_stats
from gausser.interactive import (
    BARE_COMMAND_EPILOG,
    InteractiveUsageError,
    prepare_stage_arguments,
)
from gausser.manifest import fingerprint_file, fingerprint_inputs
from gausser.publish import publish_stage
from gausser.scene import SceneError, ScenePaths
from gausser.scene_lock import SceneLockError, acquire_scene_lock
from gausser.ui import (
    NEON_CYAN,
    NEON_MAGENTA,
    NEON_YELLOW,
    completion_panel,
    console,
    neon_bar,
    neon_spinner,
    stage_header,
    status_text,
)
from msplatter.checkpoint import (
    CheckpointFormatError,
    activate_fallback,
    initialize_resume_document,
    inspect_checkpoint,
    load_resume_document,
    save_checkpoint_generation,
    select_resume_generation,
    write_resume_document,
)


_REQUIRED_MSPLAT_VERSION = "1.1.4"


class _TrainingInterrupted(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def _load_msplat() -> Any:
    try:
        import msplat
    except ImportError as exc:
        raise RuntimeError(
            "Msplat is not installed. Install Gausser's `msplat` extra."
        ) from exc
    version = getattr(msplat, "__version__", None)
    if version is None:
        try:
            version = metadata.version("msplat")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("The Msplat runtime does not report its version") from exc
    if version != _REQUIRED_MSPLAT_VERSION:
        raise RuntimeError(
            f"Msplat {_REQUIRED_MSPLAT_VERSION} is required; found {version}"
        )
    return msplat


def _has_masks(paths: ScenePaths) -> bool:
    return paths.brush_masks.is_dir() or paths.colmap_masks.is_dir()


def _prepare_dataset(paths: ScenePaths, run_dir: Path, images: list[Path]) -> Path:
    """Materialize Msplat's co-located COLMAP model/image input convention."""
    dataset = run_dir / "dataset"
    dataset.mkdir(parents=True)
    model_dir = paths.sparse / "0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(model_dir / name, dataset / name)
    for image in images:
        destination = dataset / image.name
        try:
            destination.symlink_to(image)
        except OSError:
            shutil.copy2(image, destination)
    return dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="msplatter",
        description="Run experimental, explicitly unmasked Msplat training on a Gausser scene.",
        epilog=BARE_COMMAND_EPILOG,
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Scene root (default: cwd)",
    )
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--num-downscales", type=int, default=0)
    parser.add_argument("--downscale-factor", type=float, default=1.0, metavar="N")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        metavar="N",
        help="Write and validate a resumable A/B checkpoint every N completed steps",
    )
    parser.add_argument(
        "--run-id",
        help="Stable run identity; rerun the identical command to resume its staging attempt",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Validate that Msplat imports in this interpreter, then exit",
    )
    parser.add_argument(
        "--allow-unmasked",
        action="store_true",
        help="Required acknowledgement: Msplat does not consume Gausser's SAM/Brush masks.",
    )
    try:
        prepared_argv = prepare_stage_arguments(
            "msplat", argv, program="msplatter"
        )
    except InteractiveUsageError as exc:
        parser.error(str(exc))
    if prepared_argv is None:
        return
    args = parser.parse_args(prepared_argv)
    if args.check_runtime:
        try:
            _load_msplat()
        except (ImportError, OSError, RuntimeError) as exc:
            console.print(status_text(str(exc), "error"))
            raise SystemExit(1) from exc
        console.print(status_text("Msplat runtime import is ready", "success"))
        return
    if args.iterations <= 0 or args.num_downscales < 0 or args.checkpoint_every <= 0:
        parser.error(
            "--iterations and --checkpoint-every must be positive and "
            "--num-downscales cannot be negative"
        )
    if args.downscale_factor < 1:
        parser.error("--downscale-factor must be at least 1")

    paths = ScenePaths.from_root(args.scene)
    if not paths.root.is_dir():
        parser.error(f"scene directory does not exist: {paths.root}")
    try:
        with acquire_scene_lock(paths.root):
            _run(args, paths, parser)
    except SceneLockError as exc:
        parser.error(str(exc))


def _run(
    args: argparse.Namespace, paths: ScenePaths, parser: argparse.ArgumentParser
) -> None:
    try:
        images = paths.image_files()
    except SceneError as exc:
        parser.error(str(exc))
    if not images:
        parser.error(f"No images found in {paths.images}")
    model = read_model_stats(paths.sparse / "0")
    if model is None:
        parser.error("Msplat requires a completed COLMAP sparse/0 model")
    if model.format != "binary":
        parser.error(
            "Msplat currently requires COLMAP binary sparse/0 output; text models are unsupported upstream"
        )
    if _has_masks(paths) and not args.allow_unmasked:
        parser.error(
            "This scene contains SAM masks, but Msplat has no mask-input support. "
            "Pass --allow-unmasked only to deliberately train without them."
        )
    if not args.allow_unmasked:
        parser.error(
            "Msplat is experimental and unmasked; pass --allow-unmasked to acknowledge this"
        )

    try:
        msplat = _load_msplat()
    except RuntimeError as exc:
        parser.error(str(exc))

    identity = _resume_identity(paths, images, args, msplat)
    run_id, attempt, resume_document, is_resume = _resolve_attempt(
        paths, images, args, identity, parser
    )
    staged_run = attempt / "run"
    destination_run = paths.training_run("msplat", run_id)
    dataset_path = staged_run / "dataset"
    checkpoints = attempt / "checkpoints"
    receipt_path = attempt / "resume.json"
    export = staged_run / "exports" / "scene.ply"
    checkpoint = staged_run / "checkpoint.msplat"
    checkpoint_receipt = staged_run / "checkpoint-receipt.json"

    console.print(
        stage_header(
            "MSPLATTER",
            "Experimental unmasked training signal",
            (
                ("Scene", str(paths.root)),
                ("Run", f"{run_id}{' · resume' if is_resume else ''}"),
                ("Images", f"{len(images)} files"),
                ("Reconstruction", f"{model.images} cameras · {model.points:,} points"),
                ("Iterations", f"{args.iterations:,}"),
                ("Checkpoint", f"every {args.checkpoint_every:,} completed steps"),
                ("Mask policy", "UNMASKED — explicit acknowledgement received"),
            ),
            accent=NEON_YELLOW,
        )
    )
    console.print()

    try:
        dataset = msplat.load_dataset(
            str(dataset_path),
            downscale_factor=args.downscale_factor,
            eval_mode=not args.no_eval,
        )
        config = msplat.TrainingConfig(
            iterations=args.iterations,
            num_downscales=args.num_downscales,
            downscale_factor=args.downscale_factor,
        )
        trainer = msplat.GaussianTrainer(dataset, config)
        selection = select_resume_generation(
            checkpoints, resume_document, maximum_iteration=args.iterations
        )
        current_iteration = 0
        if selection is not None:
            trainer.load_checkpoint(str(selection.path))
            current_iteration = int(trainer.iteration)
            if current_iteration != selection.record["iteration"]:
                raise CheckpointFormatError(
                    "Msplat loaded a different iteration than the validated receipt"
                )
            resume_document = activate_fallback(
                receipt_path, resume_document, selection
            )
            fallback = " using the previous A/B generation" if selection.used_fallback else ""
            console.print(
                status_text(
                    f"Resuming {run_id} at iteration {current_iteration:,}{fallback}",
                    "warning" if selection.used_fallback else "success",
                )
            )

        progress_display = Progress(
            neon_spinner(style=NEON_MAGENTA),
            TextColumn(
                "{task.description}", style=f"bold {NEON_MAGENTA}", markup=False
            ),
            neon_bar(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[splats]}", style=NEON_CYAN, markup=False),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        task_id = progress_display.add_task(
            "Training",
            total=args.iterations,
            completed=current_iteration,
            splats=f"{getattr(trainer, 'splat_count', 0):,} splats",
        )
        signal_request: dict[str, int | None] = {"signum": None}
        previous_handlers = _install_signal_handlers(signal_request)
        try:
            with progress_display:
                while current_iteration < args.iterations:
                    stats = trainer.step()
                    current_iteration = int(stats.iteration)
                    if current_iteration > args.iterations:
                        raise RuntimeError(
                            "Msplat stepped beyond the configured terminal iteration"
                        )
                    progress_display.update(
                        task_id,
                        completed=current_iteration,
                        splats=f"{stats.splat_count:,} splats",
                    )
                    requested_signum = signal_request["signum"]
                    interrupted = requested_signum is not None
                    if (
                        current_iteration % args.checkpoint_every == 0
                        or current_iteration == args.iterations
                        or interrupted
                    ):
                        receipt = save_checkpoint_generation(
                            trainer,
                            checkpoints,
                            receipt_path,
                            resume_document,
                            iteration=current_iteration,
                        )
                        resume_document = receipt.document
                        console.print(
                            status_text(
                                "Checkpoint "
                                f"{receipt.iteration:,} · {_format_bytes(receipt.size_bytes)} · "
                                f"{receipt.duration_seconds:.1f}s · "
                                f"slot {resume_document['current']['generation'].upper()}",
                                "muted",
                            )
                        )
                    if requested_signum is not None:
                        raise _TrainingInterrupted(requested_signum)
                progress_display.update(task_id, completed=args.iterations)
        finally:
            _restore_signal_handlers(previous_handlers)

        final_selection = select_resume_generation(
            checkpoints, resume_document, maximum_iteration=args.iterations
        )
        if final_selection is None or final_selection.record["iteration"] != args.iterations:
            raise RuntimeError(
                "Msplat reached the terminal iteration without a validated final checkpoint"
            )
        trainer.export_ply(str(export))
        if not export.is_file() or export.stat().st_size == 0:
            raise RuntimeError("Msplat produced an empty or missing PLY export")
        _copy_final_checkpoint(final_selection.path, checkpoint, args.iterations)
        final_summary = inspect_checkpoint(checkpoint)
        final_receipt_document = {
            **resume_document,
            "completed": {
                "iteration": final_summary.iteration,
                "num_active": final_summary.num_active,
                "checkpoint_sha256": final_selection.record["sha256"],
                "checkpoint_size_bytes": final_selection.record["size_bytes"],
            },
        }
        write_resume_document(checkpoint_receipt, final_receipt_document)
        metrics = {} if args.no_eval else dict(trainer.evaluate())
    except _TrainingInterrupted as exc:
        console.print(
            status_text(
                f"Msplat interrupted after a safe checkpoint; resume {run_id} with the identical command",
                "warning",
            )
        )
        raise SystemExit(128 + exc.signum) from exc
    except Exception as exc:
        console.print(
            status_text(
                f"Msplat failed; incomplete run retained at {attempt}: {exc}", "error"
            )
        )
        raise SystemExit(1) from exc

    final_dataset = destination_run / "dataset"
    final_export = destination_run / "exports" / "scene.ply"
    final_checkpoint = destination_run / "checkpoint.msplat"
    final_checkpoint_receipt = destination_run / "checkpoint-receipt.json"
    current_record = resume_document["current"]
    try:
        publish_stage(
            paths,
            "msplat",
            generation=None,
            replacements=((staged_run, destination_run),),
            data={
                "inputs": identity["inputs"]["images"],
                "parameters": {
                    **identity["config"],
                    "masked": False,
                    "run_id": run_id,
                    "msplat_version": _REQUIRED_MSPLAT_VERSION,
                },
                "outputs": {
                    "run": str(paths.relative_path(destination_run)),
                    "dataset": str(paths.relative_path(final_dataset)),
                    "export": str(paths.relative_path(final_export)),
                    "checkpoint": str(paths.relative_path(final_checkpoint)),
                    "checkpoint_receipt": str(paths.relative_path(final_checkpoint_receipt)),
                    "checkpoint_receipt_sha256": fingerprint_file(checkpoint_receipt),
                    "checkpoint_sha256": current_record["sha256"],
                    "checkpoint_size_bytes": current_record["size_bytes"],
                    "terminal_iteration": current_record["iteration"],
                    "num_active": current_record["num_active"],
                    "metrics": metrics,
                },
            },
        )
    except Exception as exc:
        console.print(
            status_text(f"Msplat result could not be published: {exc}", "error")
        )
        console.print(
            status_text(f"Completed staged run retained at: {attempt}", "muted")
        )
        raise SystemExit(1) from exc

    shutil.rmtree(attempt, ignore_errors=True)
    console.print(
        completion_panel(
            "Unmasked training complete",
            (
                ("Export", str(final_export)),
                ("Checkpoint", str(final_checkpoint)),
                ("Run", run_id),
                ("Mode", "experimental · unmasked"),
            ),
        )
    )


def _resume_identity(
    paths: ScenePaths,
    images: list[Path],
    args: argparse.Namespace,
    msplat: Any,
) -> dict[str, Any]:
    model_dir = paths.sparse / "0"
    sparse_files = [model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    return {
        "inputs": {
            "images": fingerprint_inputs(images),
            "sparse_model": {path.name: fingerprint_file(path) for path in sparse_files},
        },
        "runtime": {"msplat_version": str(msplat.__version__)},
        "config": {
            "iterations": args.iterations,
            "num_downscales": args.num_downscales,
            "downscale_factor": args.downscale_factor,
            "eval_mode": not args.no_eval,
            "test_every": 8,
            "checkpoint_every": args.checkpoint_every,
            "unmasked": True,
        },
    }


def _resolve_attempt(
    paths: ScenePaths,
    images: list[Path],
    args: argparse.Namespace,
    identity: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> tuple[str, Path, dict[str, Any], bool]:
    if args.run_id:
        run_id = args.run_id
        try:
            attempt = paths.staging_attempt("msplat", run_id)
            destination = paths.training_run("msplat", run_id)
        except SceneError as exc:
            parser.error(str(exc))
    else:
        candidates = (
            sorted(path for path in paths.staging.glob("msplat-*") if path.is_dir())
            if paths.staging.is_dir()
            else []
        )
        if len(candidates) > 1:
            parser.error(
                "Multiple Msplat staging attempts exist; select one with --run-id"
            )
        if candidates:
            attempt = candidates[0]
            run_id = attempt.name.removeprefix("msplat-")
            destination = paths.training_run("msplat", run_id)
        else:
            run_id = paths.new_run_id()
            attempt = paths.staging_attempt("msplat", run_id)
            destination = paths.training_run("msplat", run_id)

    if destination.exists():
        parser.error(f"Msplat run id has already been published: {run_id}")

    receipt_path = attempt / "resume.json"
    if attempt.exists():
        try:
            document = load_resume_document(receipt_path)
        except CheckpointFormatError as exc:
            parser.error(str(exc))
        if document["run_id"] != run_id or attempt.name != f"msplat-{run_id}":
            parser.error("Msplat staging run identity does not match its directory")
        if document["identity"] != identity:
            parser.error(
                "Msplat staging identity mismatch; inputs, sparse model, runtime, and full config must be unchanged"
            )
        _validate_staged_dataset(
            attempt / "run" / "dataset", images, identity, parser
        )
        return run_id, attempt, document, True

    attempt.mkdir(parents=True)
    staged_run = attempt / "run"
    (staged_run / "exports").mkdir(parents=True)
    _prepare_dataset(paths, staged_run, images)
    document = initialize_resume_document(run_id, identity)
    write_resume_document(receipt_path, document)
    return run_id, attempt, document, False


def _validate_staged_dataset(
    dataset: Path,
    images: list[Path],
    identity: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> None:
    sparse_paths = [
        dataset / name for name in ("cameras.bin", "images.bin", "points3D.bin")
    ]
    image_paths = [dataset / image.name for image in images]
    if not dataset.is_dir() or any(
        not path.is_file() for path in (*sparse_paths, *image_paths)
    ):
        parser.error("Msplat staging dataset is incomplete; refusing to overwrite it")
    try:
        staged_sparse = {path.name: fingerprint_file(path) for path in sparse_paths}
        staged_images = fingerprint_inputs(image_paths)
    except OSError as exc:
        parser.error(f"Msplat staging dataset cannot be read: {exc}")
    if (
        staged_sparse != identity["inputs"]["sparse_model"]
        or staged_images != identity["inputs"]["images"]
    ):
        parser.error(
            "Msplat staging dataset bytes do not match the recorded input identity"
        )


def _install_signal_handlers(
    request: dict[str, int | None],
) -> dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}

    def defer(signum: int, _frame: Any) -> None:
        request["signum"] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, defer)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _copy_final_checkpoint(source: Path, destination: Path, iteration: int) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        summary = inspect_checkpoint(temporary)
        if summary.iteration != iteration:
            raise CheckpointFormatError(
                "final checkpoint iteration does not match the completed run"
            )
        if fingerprint_file(temporary) != fingerprint_file(source):
            raise CheckpointFormatError("final checkpoint copy hash does not match")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
