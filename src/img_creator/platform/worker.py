"""CPU provider worker; optional learned upscaling needs the upscale extra and weights."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import signal
import time
from threading import Event, Thread
from .config import PlatformSettings
from .db import Database
from .storage import Storage
from .catalog import Generation
from .provider import BFLProvider, ProviderRejected, ProviderUncertain
from .jobs import claim, heartbeat, finish, recover_stale

log = logging.getLogger(__name__)


def store_png(storage, image, key):
    """Encode to disk and stream upload instead of duplicating an 8K PNG in RAM."""
    with TemporaryDirectory(prefix="image-export-") as directory:
        path = Path(directory) / "image.png"
        image.save(path, format="PNG")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        storage.put_file(key, path)
        return {"key": key, "size": list(image.size), "sha256": digest, "bytes": path.stat().st_size}


def run_one(db, storage, provider, models=None):
    job = claim(db, models)
    if not job:
        return False
    began = time.monotonic()
    stopped, lease_errors = Event(), []

    def pulse():
        # Cover model loading, provider calls, upscaling and storage writes.
        while not stopped.wait(20):
            try:
                heartbeat(db, job.id, job.lease_token)
            except Exception as exc:
                lease_errors.append(exc)
                return

    lease_thread = Thread(target=pulse, daemon=True)
    lease_thread.start()
    try:
        request = Generation(**job.request)

        state = {"pipeline_version": 2, "progress": {"stage": "preflight"}}
        last_progress = 0.0

        def callback(update=None):
            if update:
                state.update(update)
            heartbeat(db, job.id, job.lease_token, state)

        def progress(detail):
            nonlocal last_progress
            state["progress"] = detail
            now = time.monotonic()
            if now - last_progress >= 3 or detail.get("tiles_done") == detail.get("tiles_total"):
                callback()
                last_progress = now

        upscaler = None
        if request.upscale == "learned":
            from ..config import Settings
            from ..upscale import Upscaler

            upscaler = Upscaler(Settings())
            try:
                # Validate on CPU before any billable provider submission.
                upscaler.prepare("learned")
            except Exception as exc:
                log.exception("Super-resolution preflight failed for job %s", job.id)
                raise ProviderRejected("Super-resolution worker is not ready. Credits have been returned.") from exc
        callback({"progress": {"stage": "generation"}})
        image, metadata = provider.generate(request, callback)
        native = store_png(storage, image, f"images/{job.user_id}/{job.id}-native.png")
        native["image_url"] = f"/api/jobs/{job.id}/native"
        # Persist the paid source before attempting any GPU enhancement.
        callback({"native_artifact": native, "generation_metadata": metadata, "progress": {"stage": "source_saved"}})
        _, target = request.dimensions()
        if image.size == target:
            method = {"method": "native", "learned": False}
        elif upscaler:
            image, method = upscaler.run(image, target, "learned", progress=progress)
        else:
            from PIL import ImageOps, Image

            image = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS)
            method = {"method": "lanczos_resize", "adds_learned_detail": False}
        if image.size != target:
            raise ValueError("Export dimensions do not match the request")
        if lease_errors:
            raise lease_errors[0]
        callback({"progress": {"stage": "export"}})
        artifact = store_png(storage, image, f"images/{job.user_id}/{job.id}.png")
        result = {
            **metadata,
            **artifact,
            "pipeline_version": 2,
            "native_artifact": native,
            "image_url": f"/api/jobs/{job.id}/image",
            "export_size": list(target),
            "upscale": method,
            "seconds": round(time.monotonic() - began, 3),
            "seed": request.seed,
        }
        callback({"progress": {"stage": "complete"}})
        if not finish(db, job.id, job.lease_token, result=result):
            log.warning("Job %s completed after losing its lease; reconcile stored artifact", job.id)
    except ProviderRejected as exc:
        finish(db, job.id, job.lease_token, error=str(exc))
    except ProviderUncertain as exc:
        finish(db, job.id, job.lease_token, error=str(exc), uncertain=True)
    except Exception:
        log.exception("Job %s failed", job.id)
        # The paid provider may already have completed; don't hide costs or auto-resubmit.
        finish(db, job.id, job.lease_token, error="Generation interrupted. Owner review required.", uncertain=True)
    finally:
        stopped.set()
        lease_thread.join(timeout=30)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--backend", choices=["bfl", "local"], default="bfl")
    args = parser.parse_args()
    settings = PlatformSettings()
    if args.backend == "bfl" and not settings.bfl_key:
        raise SystemExit("BFL_API_KEY is required")
    db, storage = Database(settings.database_url), Storage(settings)
    if args.backend == "local":
        from .local_provider import LocalProvider

        provider = LocalProvider(settings, db, storage)
        models = ["custom-klein-4b", "flux-2-dev-32b"]
    else:
        provider = BFLProvider(settings.bfl_key)
        models = settings.enabled_models.split(",")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        recover_stale(db)
        worked = run_one(db, storage, provider, models)
        if args.once:
            break
        if not worked:
            time.sleep(2)


if __name__ == "__main__":
    main()
