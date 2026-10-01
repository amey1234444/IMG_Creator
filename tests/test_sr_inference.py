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


def test_local_vision_generation_contract(monkeypatch):
    import sys
    from img_creator.platform.vision import LocalVision, MODEL_REVISION
    from img_creator.platform.config import PlatformSettings

    observed = {}

    class Inputs(dict):
        def __getattr__(self, name):
            return self[name]

        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            observed["processor_config"] = kwargs
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            observed["messages"] = messages
            return "test prompt"

        def __call__(self, **kwargs):
            observed["images"] = len(kwargs["images"])
            return Inputs(input_ids=torch.ones((1, 5), dtype=torch.long), image_grid_thw=torch.tensor([[1, 2, 2]]))

        def batch_decode(self, ids, **kwargs):
            assert ids.shape == (1, 3)
            return ['{"test":true}']

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            observed["model_config"] = kwargs
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            observed["generation"] = kwargs
            return torch.ones((1, 8), dtype=torch.long)

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor, Qwen2_5_VLForConditionalGeneration=Model)
    )
    raw, metrics = LocalVision(PlatformSettings(vision_device="cpu")).analyze(Image.new("RGB", (800, 800)), "deep")
    assert raw == '{"test":true}' and metrics["output_tokens"] == 3
    assert observed["images"] == 10
    assert observed["model_config"]["revision"] == MODEL_REVISION
    assert observed["model_config"]["trust_remote_code"] is False
    assert observed["generation"]["do_sample"] is False
    assert observed["generation"]["max_new_tokens"] == 4096
    assert observed["generation"]["max_time"] == 600
