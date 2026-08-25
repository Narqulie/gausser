from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    root = tmp_path / "scene"
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (1, 1), color="white").save(images / "frame_0001.jpg")
    return root


def capture_cli_exit(fn, *args, **kwargs) -> int:
    with pytest.raises(SystemExit) as exc_info:
        fn(*args, **kwargs)
    return int(exc_info.value.code)
