"""SAM3 text-prompted segmentation. Imported only by `gausser mask`."""

from __future__ import annotations

import os
from pathlib import Path

# Without this, SAM3 silently fails on every image on Apple MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

# A local checkpoint avoids the gated Hugging Face download (and its login).
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "gausser" / "sam3.pt"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Predictor:
    def __init__(self, device: str = "auto", checkpoint: str | None = None) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = resolve_device(device)
        if checkpoint is None and DEFAULT_CHECKPOINT.is_file():
            checkpoint = str(DEFAULT_CHECKPOINT)
        if checkpoint:
            model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False, device="cpu")
        else:
            model = build_sam3_image_model(load_from_HF=True, device="cpu")
        model = model.to(self.device)
        model.eval()
        self._processor = Sam3Processor(model, device=self.device)

    def predict(self, image: Image.Image, prompt: str, threshold: float, all_instances: bool) -> tuple[np.ndarray, float] | None:
        """Return the subject mask and best score, or None when nothing passes ``threshold``."""
        with torch.inference_mode():
            state = self._processor.set_image(image.convert("RGB"))
            output = self._processor.set_text_prompt(state=state, prompt=prompt)
        hits = []
        for mask, score in zip(output["masks"], output["scores"]):
            if float(score) >= threshold:
                array = np.array(mask.cpu() if hasattr(mask, "cpu") else mask)
                hits.append((float(score), array[0] if array.ndim == 3 else array))
        if not hits:
            return None
        hits.sort(key=lambda hit: hit[0], reverse=True)
        merged = hits[0][1].astype(bool)
        if all_instances:
            for _score, mask in hits[1:]:
                merged |= mask.astype(bool)
        return merged, hits[0][0]


def save_masks(mask: np.ndarray, image: Path, brush_dir: Path, colmap_dir: Path, dilation: int, feather: int) -> None:
    """Write the Brush mask (<stem>.png, softened) and the COLMAP mask (<name><ext>.png, hard)."""
    hard = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    hard.save(colmap_dir / f"{image.name}.png")
    soft = hard
    if dilation > 0:
        soft = soft.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    if feather > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(radius=feather))
    soft.save(brush_dir / f"{image.stem}.png")
