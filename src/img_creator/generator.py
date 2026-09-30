from __future__ import annotations
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
import secrets
import hashlib
from importlib.metadata import version, PackageNotFoundError
from PIL import Image
from .config import Settings
from .metadata import ArtifactStore
from .pipelines.flux_pipeline import FluxBackend
from .pipelines.lora_pipeline import LoraRegistry
from .presets import PRESETS, PROFILES
from .prompting import enhance_prompt
from .schemas import GenerateRequest, UpscaleRequest
from .upscale import Upscaler


class BusyError(RuntimeError):
    pass


def runtime_versions():
    result = {}
    for name in ("torch", "diffusers", "transformers", "peft", "pillow", "spandrel"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


class ImageGenerator:
    """One model owner per process. Loading, adapters, inference and SR share one lock."""

    def __init__(self, settings=None, backend=None, upscaler=None):
        self.settings = settings or Settings()
        self.backend = backend or FluxBackend(self.settings)
        self.upscaler = upscaler or Upscaler(self.settings)
        self.registry = LoraRegistry(self.settings.lora_registry)
        self.store = ArtifactStore(self.settings.output_dir)
        self._lock = Lock()

    @staticmethod
    def available_qualities(aspect_ratio):
        return list(PRESETS[aspect_ratio])

    def generate(self, **kwargs):
        """Single-image compatibility helper; use generate_batch for multiple images."""
        request = GenerateRequest(**kwargs)
        if request.num_images != 1:
            raise ValueError("Use generate_batch for multiple images")
        metadata = self.generate_batch(request)[0]
        with Image.open(self.store.image_path(metadata["generation_id"])) as image:
            return image.copy(), metadata

    def generate_batch(self, request: GenerateRequest, reference=None):
        if not self._lock.acquire(blocking=False):
            raise BusyError("A generation is already running; retry after it completes")
        try:
            return self._generate_locked(request, reference)
        finally:
            self._lock.release()

    def _generate_locked(self, request, reference):
        profile = PROFILES[request.quality]
        model = profile["model"]
        adapter = self.registry.resolve(request.lora, model)
        self.upscaler.preflight(request.upscaler)
        prompt = enhance_prompt(request.prompt, request.prompt_enhancement, request.style)
        preset = PRESETS[request.aspect_ratio][request.output_resolution]
        steps = request.steps if request.steps is not None else profile["steps"]
        guidance = request.guidance_scale if request.guidance_scale is not None else profile["guidance"]
        seed = secrets.randbits(32) if request.seed is None else request.seed
        started = perf_counter()
        if reference is not None and reference.width * reference.height > 16_000_000:
            raise ValueError("Reference image must be at most 16 megapixels")
        pipe = self.backend.load(model)
        results = []
        try:
            self.registry.apply(pipe, adapter, request.lora_weight)
            for index in range(request.num_images):
                item_start = perf_counter()
                item_seed = (seed + index) % 2**32
                base = self.backend.generate(pipe, prompt, preset.base_size, steps, guidance, item_seed, reference)
                image, sr = self.upscaler.run(base, preset.final_size, request.upscaler)
                metadata = {
                    "schema_version": 1,
                    "model_id": model,
                    "model_revision": self.settings.model_revision,
                    "prompt": request.prompt,
                    "effective_prompt": prompt,
                    "style": request.style,
                    "prompt_enhancement": request.prompt_enhancement,
                    "reference_image_used": reference is not None,
                    "reference_pixel_sha256": hashlib.sha256(reference.convert("RGB").tobytes()).hexdigest()
                    if reference is not None
                    else None,
                    "runtime_versions": runtime_versions(),
                    "aspect_ratio": request.aspect_ratio,
                    "quality": request.quality,
                    "output_resolution": request.output_resolution,
                    "seed": item_seed,
                    "steps": steps,
                    "guidance_scale": guidance,
                    "base_width": base.width,
                    "base_height": base.height,
                    "final_width": image.width,
                    "final_height": image.height,
                    "device": self.backend.device,
                    "dtype": self.backend.dtype,
                    "lora": {"id": adapter["id"], "sha256": adapter["sha256"], "weight": request.lora_weight}
                    if adapter
                    else None,
                    "upscale": sr,
                    "generation_seconds": round(perf_counter() - item_start, 3),
                    "batch_elapsed_seconds": round(perf_counter() - started, 3),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                results.append(self.store.save(image, metadata, request.output_format))
        finally:
            # A failed load can partially install an adapter; discard the whole pipeline if cleanup fails.
            if adapter:
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    self.backend.reset()
        return results

    def upscale(self, request: UpscaleRequest):
        if not self._lock.acquire(blocking=False):
            raise BusyError("A generation is already running; retry after it completes")
        try:
            started = perf_counter()
            previous = self.store.read(request.generation_id)
            ratio = previous["aspect_ratio"]
            if request.output_resolution not in PRESETS[ratio]:
                raise ValueError("Unsupported output resolution for the original aspect ratio")
            size = PRESETS[ratio][request.output_resolution].final_size
            with Image.open(self.store.image_path(request.generation_id)) as image:
                output, sr = self.upscaler.run(image.convert("RGB"), size, request.upscaler)
            metadata = {
                **previous,
                "parent_generation_id": request.generation_id,
                "final_width": size[0],
                "final_height": size[1],
                "output_resolution": request.output_resolution,
                "upscale": sr,
                "operation": "upscale",
                "generation_seconds": round(perf_counter() - started, 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata.pop("batch_elapsed_seconds", None)
            return self.store.save(output, metadata, request.output_format)
        finally:
            self._lock.release()
