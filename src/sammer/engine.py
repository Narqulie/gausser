"""
SAM 3 open-vocabulary predictor and mask writers.

Ported from Automask-GS (src/predictor.py + src/composer.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageFilter


# ── Device resolution ────────────────────────────────────────────────────────────


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── SAM 3 predictor ──────────────────────────────────────────────────────────────


def validate_sam3_runtime() -> None:
    """Raise a clear error unless the external SAM3 API is import-compatible."""
    try:
        from sam3.model_builder import build_sam3_image_model  # noqa: F401
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM 3 is not installed or is incompatible. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam3.git\n"
            "Then authenticate:\n"
            "  hf auth login"
        ) from exc


class SegmentResult(NamedTuple):
    mask: np.ndarray   # bool (H, W)
    box: np.ndarray    # float (4,) [x1, y1, x2, y2]
    score: float


class ImageResult(NamedTuple):
    image_name: str
    success: bool
    confidence: float | None
    error: str | None = None


class Sam3Predictor:
    """Wraps SAM 3's image processor for text-prompted segmentation."""

    def __init__(self, checkpoint: str | None = None, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self._load_model(checkpoint)

    def _load_model(self, checkpoint: str | None) -> None:
        validate_sam3_runtime()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        local_path = Path(checkpoint) if checkpoint else None
        use_local = (
            local_path is not None
            and local_path.exists()
            and local_path.suffix in (".pt", ".pth", ".safetensors")
        )

        if use_local:
            model = build_sam3_image_model(
                checkpoint_path=checkpoint, load_from_HF=False, device="cpu"
            )
        else:
            model = build_sam3_image_model(load_from_HF=True, device="cpu")

        model = model.to(self.device)
        model.eval()
        self._processor = Sam3Processor(model, device=self.device)

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        score_threshold: float = 0.5,
    ) -> list[SegmentResult]:
        image_rgb = image.convert("RGB")

        with torch.inference_mode():
            state = self._processor.set_image(image_rgb)
            output = self._processor.set_text_prompt(state=state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        results: list[SegmentResult] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue
            mask_t = masks[i]
            if hasattr(mask_t, "cpu"):
                mask_t = mask_t.cpu()
            mask = np.array(mask_t)
            if mask.ndim == 3:
                mask = mask[0]
            box_t = boxes[i]
            if hasattr(box_t, "cpu"):
                box_t = box_t.cpu()
            results.append(SegmentResult(
                mask=mask.astype(bool),
                box=np.array(box_t),
                score=score,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def merge_masks(results: list[SegmentResult]) -> np.ndarray | None:
        if not results:
            return None
        merged = results[0].mask.copy()
        for r in results[1:]:
            merged = merged | r.mask
        return merged


# ── Mask writers ─────────────────────────────────────────────────────────────────


def dilate_and_feather_mask(
    mask: np.ndarray,
    dilation_px: int = 8,
    feather_px: int = 12,
) -> np.ndarray:
    """Dilate a binary mask then feather its edges for cleaner training."""
    alpha_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if dilation_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.MaxFilter(dilation_px * 2 + 1))
    if feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return np.array(alpha_img).astype(np.float32) / 255.0


def save_brush_mask(
    mask: np.ndarray,
    source_path: Path,
    output_dir: Path,
    dilation_px: int = 0,
    feather_px: int = 0,
) -> None:
    """Write Brush training mask to <output_dir>/<stem>.png (white=subject)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (source_path.stem + ".png")

    if dilation_px > 0 or feather_px > 0:
        soft = dilate_and_feather_mask(mask, dilation_px=dilation_px, feather_px=feather_px)
        brush_mask = Image.fromarray((soft * 255).clip(0, 255).astype(np.uint8), mode="L")
    else:
        brush_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    brush_mask.save(str(output_path), format="PNG")


def save_colmap_mask(mask: np.ndarray, source_path: Path, output_dir: Path) -> None:
    """Write COLMAP feature mask to <output_dir>/<name><ext>.png (double-ext)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (source_path.name + ".png")
    colmap_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    colmap_mask.save(str(output_path), format="PNG")
