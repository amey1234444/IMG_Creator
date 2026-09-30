import pytest
from pydantic import ValidationError
from img_creator.schemas import GenerateRequest
from img_creator.config import Settings


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": " "},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"steps": 0},
        {"guidance_scale": float("nan")},
        {"aspect_ratio": "4:5"},
        {"output_resolution": "8K"},
        {"num_images": 5},
        {"quality": "ultra"},
        {"lora": "../../private"},
        {"unknown": True},
    ],
)
def test_invalid_request(change):
    with pytest.raises(ValidationError):
        GenerateRequest(**({"prompt": "A tree"} | change))


def test_legacy_and_aliases():
    req = GenerateRequest(prompt="A tree", quality="4K", enhance_prompt=True)
    assert req.quality == "fast" and req.output_resolution == "4K" and req.prompt_enhancement
    assert GenerateRequest(prompt="a", output_resolution="2k").output_resolution == "2K"


def test_ultra_explicit_learned_backend():
    assert GenerateRequest(prompt="A tree", quality="ultra", upscaler="learned").quality == "ultra"


def test_invalid_device_and_tile():
    with pytest.raises(ValueError):
        Settings(device="magic")
    with pytest.raises(ValueError):
        Settings(tile_size=0)
