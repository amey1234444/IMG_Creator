from __future__ import annotations
import gc
import os
from ..device import resolve_device, preferred_dtype


class FluxBackend:
    def __init__(self, settings):
        self.settings = settings
        self.pipe = None
        self.model_id = None
        self.device = "not loaded"
        self.dtype = "not loaded"

    def reset(self):
        self.pipe = None
        self.model_id = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_id):
        if self.pipe is not None and self.model_id == model_id:
            return self.pipe
        if self.pipe is not None:
            self.reset()
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError("Install the inference extra and pinned Diffusers source; see README.") from exc
        self.device = resolve_device(self.settings.device)
        dtype = preferred_dtype(self.device)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision=self.settings.model_revision,
            token=os.getenv("HF_TOKEN") or None,
        )
        pipe.vae.enable_tiling()
        if self.device == "cuda" and self.settings.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self.pipe, self.model_id, self.dtype = pipe, model_id, str(dtype)
        return pipe

    def generate(self, pipe, prompt, size, steps, guidance, seed, reference=None):
        import torch

        kwargs = dict(
            prompt=prompt,
            width=size[0],
            height=size[1],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        if reference is not None:
            kwargs["image"] = reference.convert("RGB")
        with torch.inference_mode():
            return pipe(**kwargs).images[0]
