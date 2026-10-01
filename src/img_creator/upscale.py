from __future__ import annotations
import hashlib
import math
import time
from pathlib import Path
from PIL import Image, ImageOps


def upscale_to(image: Image.Image, final_size: tuple[int, int]) -> Image.Image:
    """Center-crop to the requested aspect without stretching the source."""
    if image.size == final_size:
        return image.copy()
    return ImageOps.fit(image, final_size, method=Image.Resampling.LANCZOS)


def tile_regions(width, height, tile_size, pad=32):
    """Non-overlapping output cores with overlapping input context on every edge."""
    if min(width, height, tile_size) <= 0 or pad < 0:
        raise ValueError("Invalid tile geometry")
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            core = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            yield core, (max(0, x - pad), max(0, y - pad), min(width, core[2] + pad), min(height, core[3] + pad))


MAX_SR_PIXELS = 70_000_000


def sr_plan(source, target, scale):
    """Bounded learned passes. Resample DOWN before the last pass, never up.

    This avoids a 16K intermediate for an 8K export with a 4x checkpoint.
    Each entry is (input_size, learned_output_size).
    """
    if scale not in (2, 4) or min(*source, *target) < 1:
        raise ValueError("Invalid super-resolution geometry")
    if max(*target) > 8192 or math.prod(target) > MAX_SR_PIXELS:
        raise ValueError("Export exceeds the 8192-pixel / 70-megapixel budget")
    current, passes = tuple(source), []
    while current[0] < target[0] or current[1] < target[1]:
        required = tuple(math.ceil(n / scale) for n in target)
        # Preserve native detail whenever the full learned pass fits the budget.
        # Downsample only to avoid an oversized intermediate, never just because
        # a smaller input could reach the export dimensions.
        size = current
        output = tuple(n * scale for n in size)
        if max(output) > 8192 or math.prod(output) > MAX_SR_PIXELS:
            if not all(a >= b for a, b in zip(current, required)):
                raise ValueError("Super-resolution plan exceeds resource budget")
            size = required
            output = tuple(n * scale for n in size)
        if math.prod(output) > MAX_SR_PIXELS or len(passes) >= 8:
            raise ValueError("Super-resolution plan exceeds resource budget")
        passes.append((size, output))
        current = output
    return passes


class Upscaler:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._digest = None

    def preflight(self, name):
        if name == "learned":
            p = self.settings.sr_weights
            if p is None or not p.is_file():
                raise ValueError("Configure IMG_CREATOR_SR_WEIGHTS with trusted Real-ESRGAN or SwinIR weights first")
        elif name != "lanczos":
            raise ValueError("Unsupported upscaler")

    def _load(self):
        if self._model is None:
            try:
                from spandrel import ModelLoader, ImageModelDescriptor, ModelTiling
            except ImportError as exc:
                raise RuntimeError('Install the upscale extra: pip install -e ".[upscale]"') from exc
            path = Path(self.settings.sr_weights)
            model = ModelLoader().load_from_file(str(path))
            if not isinstance(model, ImageModelDescriptor) or model.input_channels != 3 or model.output_channels != 3:
                raise ValueError("Upscaler must be an RGB image model")
            if model.scale not in (2, 4):
                raise ValueError("Use a 2x or 4x super-resolution model")
            if model.tiling != ModelTiling.SUPPORTED:
                raise ValueError("Use a super-resolution model that supports external tiling")
            self._model = model.eval()
            with path.open("rb") as f:
                self._digest = hashlib.file_digest(f, "sha256").hexdigest()
        return self._model

    def prepare(self, name):
        """Load and validate before submitting a potentially billable generation."""
        self.preflight(name)
        if name == "learned":
            from .device import resolve_device

            resolve_device(self.settings.device)
            self._load()

    def _infer_tile(self, image, model):
        import numpy as np
        import torch

        tile = np.asarray(image.convert("RGB")).copy()
        tensor = (
            torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device=model.device, dtype=model.dtype).div_(255)
        )
        # Spandrel handles checkpoint-specific minimum sizes and padding.
        with torch.inference_mode():
            result = model(tensor)
            expected = (1, 3, image.height * model.scale, image.width * model.scale)
            if tuple(result.shape) != expected or not torch.isfinite(result).all().item():
                raise ValueError("Super-resolution model returned an invalid image")
            result = result.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((result * 255).round().astype("uint8"))

    def _pass(self, image, model, notify):
        import torch

        tile_size = self.settings.tile_size
        while True:
            try:
                return self._tiles(image, model, tile_size, notify), tile_size
            except torch.cuda.OutOfMemoryError:
                # Restart only the deterministic SR pass, never the paid generation.
                if tile_size <= 64:
                    raise
                tile_size = max(64, tile_size // 2)
                torch.cuda.empty_cache()
                notify({"retry_tile_size": tile_size})

    def _tiles(self, image, model, tile_size, notify):
        scale, pad = model.scale, self.settings.tile_pad
        canvas = Image.new("RGB", (image.width * scale, image.height * scale))
        count = math.ceil(image.width / tile_size) * math.ceil(image.height / tile_size)
        try:
            for index, (core, context) in enumerate(tile_regions(image.width, image.height, tile_size, pad), 1):
                tile_out = self._infer_tile(image.crop(context), model)
                left, top = (core[0] - context[0]) * scale, (core[1] - context[1]) * scale
                right, bottom = left + (core[2] - core[0]) * scale, top + (core[3] - core[1]) * scale
                # Discard the halo: adjacent cores use overlapping source context.
                canvas.paste(tile_out.crop((left, top, right, bottom)), (core[0] * scale, core[1] * scale))
                notify({"tiles_done": index, "tiles_total": count})
            return canvas
        except BaseException:
            canvas.close()
            raise

    def run(self, image, final_size, name, progress=None):
        self.preflight(name)
        if name == "lanczos":
            method = "native" if image.size == final_size else "lanczos-resize"
            return upscale_to(image, final_size), {"method": method, "learned": False}
        model = self._load()
        plan = sr_plan(image.size, final_size, model.scale)
        if plan:
            from .device import resolve_device

            model.to(resolve_device(self.settings.device))
        current, stages = image, []
        for index, (size, output) in enumerate(plan, 1):
            began = time.monotonic()
            source_size = current.size
            prepared = upscale_to(current, size) if current.size != size else current

            def notify(detail):
                if progress:
                    progress({"stage": "super_resolution", "pass": index, "passes": len(plan), **detail})

            notify({"tiles_done": 0})
            try:
                enlarged, tile_size = self._pass(prepared, model, notify)
            finally:
                if prepared is not current:
                    prepared.close()
            if current is not image:
                current.close()
            current = enlarged
            stages.append(
                {
                    "source_size": list(source_size),
                    "input_size": list(size),
                    "output_size": list(output),
                    "tile_size": tile_size,
                    "seconds": round(time.monotonic() - began, 3),
                }
            )
        # Any final adjustment is crop/downsample, never interpolation enlargement.
        result = upscale_to(current, final_size)
        intermediate_size = list(current.size)
        if current is not image:
            current.close()
        return result, {
            "method": "progressive-tiled-sr" if plan else "native-or-downsample",
            "pipeline_version": 2,
            "learned": bool(plan),
            "scale": model.scale,
            "architecture": str(model.architecture.id),
            "weights_sha256": self._digest,
            "tile_pad": self.settings.tile_pad,
            "stages": stages,
            "intermediate_size": intermediate_size,
            "final_resample": tuple(intermediate_size) != final_size,
            "final_interpolation_upscale": False,
        }
