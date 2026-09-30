from pathlib import Path
import pytest

pytest.importorskip("gradio")
from img_creator.ui import build_ui
from img_creator.config import Settings
from img_creator.generator import ImageGenerator
from test_generation import FakeBackend


def test_ui_build_and_generation_callback(tmp_path):
    service = ImageGenerator(Settings(output_dir=tmp_path), backend=FakeBackend())
    ui = build_ui(service)
    assert len(ui.config["components"]) >= 30
    callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "run")
    gallery, metadata, downloads = callback(
        "A tree", "1:1", "fast", "Native", 123, False, "neutral", 1, 0, -1, "lanczos", "png", "", 1, None
    )
    assert len(gallery) == 1 and metadata[0]["seed"] == 123
    assert all(Path(path).is_file() for path in downloads)
