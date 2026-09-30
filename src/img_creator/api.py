from __future__ import annotations
import logging
import secrets
from fastapi import Depends, FastAPI, Header, HTTPException, Path
from fastapi.responses import FileResponse
from .config import Settings
from .generator import ImageGenerator, BusyError
from .presets import PRESETS, PROFILES
from .schemas import GenerateRequest, UpscaleRequest

log = logging.getLogger(__name__)


def create_app(service=None):
    service = service or ImageGenerator(Settings())
    app = FastAPI(title="IMG Creator", version="0.2.0")
    app.state.generator = service

    def authorize(x_api_key: str | None = Header(default=None)):
        key = service.settings.api_key
        if key and not secrets.compare_digest(x_api_key or "", key):
            raise HTTPException(401, "Invalid API key")

    def execute(call):
        try:
            return call()
        except BusyError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "Artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            log.exception("Generation failed")
            raise HTTPException(
                503, "Generation unavailable. Check server logs, dependencies, model access and available memory."
            ) from exc

    def result(metadata):
        gid = metadata["generation_id"]
        return {"generation": metadata, "image_url": f"/images/{gid}", "metadata_url": f"/metadata/{gid}"}

    @app.get("/")
    def root():
        return {"application": "IMG Creator", "docs": "/docs", "version": "0.2.0"}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": service.backend.pipe is not None, "device": service.backend.device}

    @app.get("/models", dependencies=[Depends(authorize)])
    def models():
        return {
            "profiles": PROFILES,
            "loras": service.registry.list(),
            "learned_upscaler_configured": service.settings.sr_weights is not None,
        }

    @app.get("/presets")
    def presets():
        return {
            ratio: {name: {"base": p.base_size, "final": p.final_size} for name, p in choices.items()}
            for ratio, choices in PRESETS.items()
        }

    @app.post("/generate", dependencies=[Depends(authorize)])
    def generate(request: GenerateRequest):
        items = execute(lambda: service.generate_batch(request))
        response = {"images": [result(item) for item in items]}
        if len(items) == 1:
            response.update(result(items[0]))
        return response

    @app.post("/upscale", dependencies=[Depends(authorize)])
    def upscale(request: UpscaleRequest):
        return result(execute(lambda: service.upscale(request)))

    @app.get("/images/{generation_id}", dependencies=[Depends(authorize)])
    def image(generation_id: str = Path(pattern=r"^[0-9a-f]{32}$")):
        path = execute(lambda: service.store.image_path(generation_id))
        if not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, filename=f"{generation_id}{path.suffix}")

    @app.get("/metadata/{generation_id}", dependencies=[Depends(authorize)])
    def metadata(generation_id: str = Path(pattern=r"^[0-9a-f]{32}$")):
        return execute(lambda: service.store.read(generation_id))

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run("img_creator.api:app", host="127.0.0.1", port=8000, workers=1)
