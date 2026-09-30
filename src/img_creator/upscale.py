from __future__ import annotations
import hashlib
from pathlib import Path
from PIL import Image, ImageOps


def upscale_to(image: Image.Image, final_size: tuple[int, int]) -> Image.Image:
    """Center-crop to the requested aspect without stretching the source."""
    if image.size == final_size:
        return image.copy()
    return ImageOps.fit(image, final_size, method=Image.Resampling.LANCZOS)


def tile_regions(width, height, tile_size, pad=32):
    """Non-overlapping output cores with overlapping input context on every edge."""
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            core = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            yield core, (max(0, x - pad), max(0, y - pad), min(width, core[2] + pad), min(height, core[3] + pad))


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
                from spandrel import ModelLoader, ImageModelDescriptor
            except ImportError as exc:
                raise RuntimeError('Install the upscale extra: pip install -e ".[upscale]"') from exc
            from .device import resolve_device

            path = Path(self.settings.sr_weights)
            model = ModelLoader().load_from_file(str(path))
            if not isinstance(model, ImageModelDescriptor) or model.input_channels != 3 or model.output_channels != 3:
                raise ValueError("Upscaler must be an RGB image model")
            if model.scale not in (2, 4):
                raise ValueError("Use a 2x or 4x super-resolution model")
            self._model = model.to(resolve_device(self.settings.device)).eval()
            with path.open("rb") as f:
                self._digest = hashlib.file_digest(f, "sha256").hexdigest()
        return self._model

    def run(self, image, final_size, name):
        self.preflight(name)
        if name == "lanczos":
            method = "native" if image.size == final_size else "lanczos-resize"
            return upscale_to(image, final_size), {"method": method, "learned": False}
        import numpy as np
        import torch

        model = self._load()
        scale = model.scale
        if image.width * image.height * scale * scale > 64_000_000:
            raise ValueError("Learned SR canvas exceeds 64 megapixels; start from the original native generation")
        # Full canvas stays in host RAM; only a padded tile lives on the accelerator.
        canvas = Image.new("RGB", (image.width * scale, image.height * scale))
        for core, context in tile_regions(image.width, image.height, self.settings.tile_size):
            tile = np.asarray(image.crop(context).convert("RGB")).copy()
            tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(model.device)
            with torch.inference_mode():
                result = model(tensor).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            tile_out = Image.fromarray((result * 255).round().astype("uint8"))
            left, top = (core[0] - context[0]) * scale, (core[1] - context[1]) * scale
            right, bottom = left + (core[2] - core[0]) * scale, top + (core[3] - core[1]) * scale
            canvas.paste(tile_out.crop((left, top, right, bottom)), (core[0] * scale, core[1] * scale))
        return upscale_to(canvas, final_size), {
            "method": "learned-tiled-sr",
            "learned": True,
            "scale": scale,
            "architecture": str(model.architecture.id),
            "weights_sha256": self._digest,
            "tile_size": self.settings.tile_size,
            "tile_pad": 32,
            "intermediate_size": list(canvas.size),
            "final_resample": canvas.size != final_size,
        }
