"""Optional CPU tensor integration. Random weights verify plumbing, not realism."""

import hashlib
from types import SimpleNamespace
import pytest
from PIL import Image
from img_creator.config import Settings
from img_creator.upscale import Upscaler

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("failure", ["shape", "nan"])
def test_rejects_invalid_model_tensor(failure):
    class InvalidModel:
        scale = 2
        device = "cpu"
        dtype = torch.float32

        def __call__(self, tensor):
            if failure == "shape":
                return tensor
            return torch.full((1, 3, 16, 16), float("nan"))

    with pytest.raises(ValueError, match="invalid image"):
        Upscaler(Settings(device="cpu"))._infer_tile(Image.new("RGB", (8, 8)), InvalidModel())


def test_spandrel_checkpoint_load_and_progressive_inference(tmp_path):
    pytest.importorskip("spandrel")
    from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet

    torch.set_num_threads(1)
    torch.manual_seed(5)
    checkpoint = tmp_path / "random-test-network.pth"
    torch.save(RRDBNet(num_filters=8, num_blocks=1, scale=2).state_dict(), checkpoint)
    runner = Upscaler(Settings(device="cpu", sr_weights=checkpoint, tile_size=64))
    runner.prepare("learned")
    output, metadata = runner.run(Image.new("RGB", (17, 19), "gray"), (68, 76), "learned")
    assert output.size == (68, 76)
    assert len(metadata["stages"]) == 2
    assert metadata["weights_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["architecture"] == "ESRGAN"
    assert not metadata["final_interpolation_upscale"]
    assert runner._model.device == torch.device("cpu")


def test_rgb_tensor_conversion_preserves_values():
    class NearestModel:
        scale = 2
        device = "cpu"
        dtype = torch.float32
        architecture = SimpleNamespace(id="test")

        def __call__(self, tensor):
            return torch.nn.functional.interpolate(tensor, scale_factor=2, mode="nearest")

    image = Image.new("RGB", (7, 9), (23, 101, 249))
    output = Upscaler(Settings(device="cpu"))._infer_tile(image, NearestModel())
    assert output.size == (14, 18)
    assert output.getpixel((13, 17)) == (23, 101, 249)
