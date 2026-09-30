import pytest
from PIL import Image
from img_creator.upscale import tile_regions, upscale_to, Upscaler
from img_creator.config import Settings


def test_tile_cores_cover_every_pixel_once():
    width, height = 73, 57
    coverage = [[0] * width for _ in range(height)]
    for core, context in tile_regions(width, height, 16, 4):
        assert context[0] <= core[0] < core[2] <= context[2]
        assert context[1] <= core[1] < core[3] <= context[3]
        for y in range(core[1], core[3]):
            for x in range(core[0], core[2]):
                coverage[y][x] += 1
    assert all(n == 1 for row in coverage for n in row)


def test_resize_crops_instead_of_squashing():
    image = Image.new("RGB", (100, 50), "red")
    for x in range(25, 75):
        for y in range(50):
            image.putpixel((x, y), (0, 255, 0))
    result = upscale_to(image, (50, 50))
    assert result.getpixel((0, 25)) == (0, 255, 0)


def test_learned_does_not_silently_fallback():
    with pytest.raises(ValueError):
        Upscaler(Settings()).run(Image.new("RGB", (8, 8)), (16, 16), "learned")
