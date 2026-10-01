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


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("resolution", ["1K", "2K", "4K", "8K"])
@pytest.mark.parametrize("ratio", ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3"])
def test_sr_plan_covers_export_without_interpolation(scale, resolution, ratio):
    from img_creator.platform.catalog import Generation
    from img_creator.upscale import sr_plan, MAX_SR_PIXELS

    source, target = Generation(prompt="test", resolution=resolution, aspect_ratio=ratio).dimensions()
    current = source
    for size, output in sr_plan(source, target, scale):
        assert all(a <= b for a, b in zip(size, current))  # never upscale input
        assert output == tuple(n * scale for n in size)
        assert output[0] * output[1] <= MAX_SR_PIXELS
        current = output
    assert all(a >= b for a, b in zip(current, target))


def test_sr_plan_rejects_unbounded_export():
    from img_creator.upscale import sr_plan

    with pytest.raises(ValueError, match="budget"):
        sr_plan((1024, 1024), (16384, 16384), 4)
    assert sr_plan((4096, 4096), (1024, 1024), 4) == []
    # Do not throw away original detail by reducing a 1000px source to 275px.
    assert sr_plan((1000, 1000), (1100, 1100), 4) == [((1000, 1000), (4000, 4000))]


def test_tiled_halo_matches_full_image_filter(monkeypatch):
    """A spatial filter catches seams, missing borders and crop-coordinate bugs."""
    import numpy as np
    from PIL import ImageFilter
    from types import SimpleNamespace

    image = Image.fromarray(np.random.default_rng(19).integers(0, 256, (117, 159, 3), dtype=np.uint8))
    runner = Upscaler(Settings(tile_size=64, tile_pad=16))

    def infer(tile, model):
        return tile.filter(ImageFilter.GaussianBlur(2)).resize(
            (tile.width * model.scale, tile.height * model.scale), Image.Resampling.NEAREST
        )

    monkeypatch.setattr(runner, "_infer_tile", infer)
    model = SimpleNamespace(scale=2)
    progress = []
    output = runner._tiles(image, model, 64, progress.append)
    expected = infer(image, model)
    assert np.array_equal(np.asarray(output), np.asarray(expected))
    assert progress[-1] == {"tiles_done": 6, "tiles_total": 6}


def test_progressive_runner_8k_square(monkeypatch):
    """Exercise actual 8192-square allocation; fake predictor, no quality claim."""
    from types import SimpleNamespace
    import img_creator.device

    runner = Upscaler(Settings())
    model = SimpleNamespace(scale=4, architecture=SimpleNamespace(id="test"), to=lambda *_: None)
    monkeypatch.setattr(runner, "preflight", lambda *_: None)
    monkeypatch.setattr(runner, "_load", lambda: model)
    monkeypatch.setattr(img_creator.device, "resolve_device", lambda *_: "cpu")
    calls = []

    def predict(image, model, notify):
        calls.append(image.size)
        notify({"tiles_done": 1, "tiles_total": 1})
        return Image.new("RGB", (image.width * model.scale, image.height * model.scale), "red"), 256

    monkeypatch.setattr(runner, "_pass", predict)
    events = []
    result, metadata = runner.run(Image.new("RGB", (1984, 1984), "red"), (8192, 8192), "learned", events.append)
    assert result.size == (8192, 8192)
    assert calls == [(1984, 1984), (2048, 2048)]
    assert result.getpixel((8191, 8191)) == (255, 0, 0)
    assert len(metadata["stages"]) == 2
    assert not metadata["final_interpolation_upscale"]
    assert events[-1]["pass"] == 2
    result.close()


def test_cuda_oom_retries_only_sr_with_smaller_tiles(monkeypatch):
    import sys
    from types import SimpleNamespace

    class OutOfMemoryError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=lambda: None)),
    )
    runner = Upscaler(Settings(tile_size=256))
    calls = []

    def tiles(image, model, size, notify):
        calls.append(size)
        if size > 64:
            raise OutOfMemoryError()
        return image

    monkeypatch.setattr(runner, "_tiles", tiles)
    image = Image.new("RGB", (8, 8))
    assert runner._pass(image, None, lambda *_: None) == (image, 64)
    assert calls == [256, 128, 64]
