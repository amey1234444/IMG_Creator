"""Self-hosted GPU adapter with required input/output moderation and approved weights."""

import base64
import hashlib
from io import BytesIO
import tempfile
from threading import Event, Thread
from pathlib import Path
import httpx
from .catalog import STYLES
from .db import TrainingRun
from .provider import ProviderRejected


class Moderation:
    def __init__(self, key):
        if not key:
            raise ValueError("OPENAI_MODERATION_KEY is required for public self-hosted generation")
        self.key = key

    def check(self, prompt, image=None):
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            # Moderation copy only; preserve the original image for the actual output.
            thumbnail = image.copy()
            thumbnail.thumbnail((2048, 2048))
            data = BytesIO()
            thumbnail.save(data, format="JPEG", quality=90)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()},
                }
            )
        response = httpx.post(
            "https://api.openai.com/v1/moderations",
            timeout=60,
            headers={"Authorization": "Bearer " + self.key},
            json={"model": "omni-moderation-latest", "input": content},
        )
        response.raise_for_status()
        results = response.json()["results"]
        if not results or any(not isinstance(r.get("flagged"), bool) for r in results):
            raise RuntimeError("Invalid moderation response")
        if any(r["flagged"] for r in results):
            raise ProviderRejected("This request could not be served. Credits have been returned.")


class LocalProvider:
    def __init__(self, settings, db, storage):
        if not settings.local_models:
            raise ValueError("LOCAL_MODELS_ENABLED must be true")
        self.settings, self.db, self.storage = settings, db, storage
        self.moderation = Moderation(settings.moderation_key)
        self.pipe = None
        self.model = None

    def generate(self, request, heartbeat):
        stop = Event()
        errors = []

        def pulse():
            while not stop.wait(20):
                try:
                    heartbeat(None)
                except Exception as exc:
                    errors.append(exc)
                    return

        thread = Thread(target=pulse, daemon=True)
        thread.start()
        try:
            return self._generate(request, heartbeat, errors)
        finally:
            stop.set()
            thread.join(timeout=30)

    def _generate(self, request, heartbeat, errors):
        prompt = (request.prompt + " " + STYLES[request.style]).strip()
        self.moderation.check(prompt)
        model_id = "black-forest-labs/FLUX.2-klein-base-4B"
        if request.model == "flux-2-dev-32b":
            if not self.settings.dev_license:
                raise ProviderRejected("This model is not licensed for this deployment")
            model_id = "black-forest-labs/FLUX.2-dev"
        import torch
        from ..config import Settings
        from ..device import resolve_device, preferred_dtype

        local = Settings()
        device = resolve_device(local.device)
        if self.model != model_id:
            self.pipe = None
            self.model = None
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from diffusers import Flux2KleinPipeline, Flux2Pipeline

            pipeline = Flux2Pipeline if request.model == "flux-2-dev-32b" else Flux2KleinPipeline
            self.pipe = pipeline.from_pretrained(
                model_id, torch_dtype=preferred_dtype(device), revision=local.model_revision
            )
            self.pipe.vae.enable_tiling()
            if local.cpu_offload and device == "cuda":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(device)
            self.model = model_id
        adapter = None
        try:
            if request.training_run:
                with self.db.transaction() as s:
                    run = s.get(TrainingRun, request.training_run)
                    if not run or run.status != "approved" or not run.artifact_key:
                        raise ProviderRejected("The selected adapter is no longer available")
                data = self.storage.get(run.artifact_key)
                if hashlib.sha256(data).hexdigest() != run.artifact_sha256:
                    raise RuntimeError("Adapter integrity verification failed")
                adapter = run.id
                with tempfile.TemporaryDirectory() as directory:
                    (Path(directory) / "adapter.safetensors").write_bytes(data)
                    self.pipe.load_lora_weights(directory, weight_name="adapter.safetensors", adapter_name="studio")
                    self.pipe.set_adapters(["studio"], adapter_weights=[request.adapter_strength])
            native, _ = request.dimensions()
            steps = {"low": 20, "medium": 35, "high": 50}[request.effort]
            with torch.inference_mode():
                image = self.pipe(
                    prompt=prompt,
                    width=native[0],
                    height=native[1],
                    num_inference_steps=steps,
                    guidance_scale=4.0,
                    generator=torch.Generator("cpu").manual_seed(request.seed),
                ).images[0]
            if errors:
                raise errors[0]
            heartbeat(None)
            self.moderation.check(prompt, image)
            return image, {
                "provider": "self_hosted",
                "model_id": model_id,
                "adapter": adapter,
                "adapter_strength": request.adapter_strength if adapter else None,
                "steps": steps,
                "tokens": None,
                "provider_cost": None,
                "effective_prompt": prompt,
                "native_size": list(image.size),
                "moderation": "input_and_output_passed",
            }
        finally:
            if request.training_run:
                try:
                    self.pipe.unload_lora_weights()
                except Exception:
                    self.pipe, self.model = None, None
