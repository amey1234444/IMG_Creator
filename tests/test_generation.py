import json
from threading import Event, Thread
from unittest.mock import Mock
import pytest
from PIL import Image
from img_creator.config import Settings
from img_creator.generator import ImageGenerator, BusyError
from img_creator.schemas import GenerateRequest
from img_creator.metadata import ArtifactStore


class FakeBackend:
    def __init__(self):
        self.pipe = Mock()
        self.device = "cpu"
        self.dtype = "float32"
        self.seeds = []

    def load(self, model):
        return self.pipe

    def generate(self, pipe, prompt, size, steps, guidance, seed, reference=None):
        self.seeds.append(seed)
        return Image.new("RGB", size, (seed % 256, 12, 34))

    def reset(self):
        self.pipe = None


@pytest.fixture
def service(tmp_path):
    return ImageGenerator(Settings(output_dir=tmp_path / "outputs"), backend=FakeBackend())


def test_batch_seeds_metadata_and_actual_size(service):
    items = service.generate_batch(
        GenerateRequest(prompt="Tree", seed=2**32 - 1, num_images=2, output_resolution="2K", output_format="webp")
    )
    assert service.backend.seeds == [2**32 - 1, 0]
    for m in items:
        assert m["upscale"]["method"] == "lanczos-resize"
        assert m["model_id"].endswith("klein-4B")
        assert service.store.read(m["generation_id"]) == m
        with Image.open(service.store.image_path(m["generation_id"])) as image:
            assert image.size == (2048, 2048)
    assert not list(service.settings.output_dir.glob(".*.tmp"))


def test_concurrent_calls_rejected_before_model_loading(service):
    entered, release = Event(), Event()
    original = service.backend.load

    def blocked(model):
        entered.set()
        assert release.wait(5)
        return original(model)

    service.backend.load = blocked
    errors = []

    def run():
        try:
            service.generate_batch(GenerateRequest(prompt="a"))
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(BusyError):
            service.generate_batch(GenerateRequest(prompt="b"))
    finally:
        release.set()
        thread.join()
    assert not errors


def test_failed_call_releases_lock(service):
    original = service.backend.generate
    service.backend.generate = Mock(side_effect=RuntimeError("OOM"))
    with pytest.raises(RuntimeError):
        service.generate_batch(GenerateRequest(prompt="a"))
    service.backend.generate = original
    assert service.generate_batch(GenerateRequest(prompt="b"))


def test_atomic_cleanup(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(Image.Image, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        store.save(Image.new("RGB", (4, 4)), {}, "png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["../secret", "a" * 31, "g" * 32, "/etc/passwd"])
def test_unsafe_id_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path).directory(value)


def test_lora_cleanup_after_failed_inference(tmp_path):
    weights = tmp_path / "lora.safetensors"
    weights.write_bytes(b"test-fixture")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"test": {"path": str(weights), "model": "black-forest-labs/FLUX.2-klein-4B"}}))
    backend = FakeBackend()
    backend.generate = Mock(side_effect=RuntimeError("failure"))
    service = ImageGenerator(Settings(output_dir=tmp_path / "out", lora_registry=registry), backend=backend)
    with pytest.raises(RuntimeError):
        service.generate_batch(GenerateRequest(prompt="a", lora="test"))
    backend.pipe.unload_lora_weights.assert_called_once()
