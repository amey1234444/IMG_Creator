"""Run local visual annotation independently of web, diffusion and training workers."""

import argparse
import hashlib
import logging
import signal
import time
from io import BytesIO
from threading import Event, Thread
from PIL import Image
from sqlalchemy import select, update
from .config import PlatformSettings
from .db import Database, Asset, ImageAnalysis, AnalysisAttempt, uid
from .storage import Storage
from .vision import LocalVision, parse_report, MODEL_ID, MODEL_REVISION, PROMPT_VERSION

log = logging.getLogger(__name__)


def analyze_one(db, storage, analyzer):
    with db.transaction() as s:
        s.execute(
            update(ImageAnalysis)
            .where(ImageAnalysis.status == "running", ImageAnalysis.lease_until < time.time())
            .values(status="failed", error="Vision worker interrupted; retry explicitly", finished=time.time())
        )
        item = s.scalar(
            select(ImageAnalysis)
            .where(ImageAnalysis.status == "queued")
            .order_by(ImageAnalysis.created)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if not item:
            return False
        token = uid()
        if not s.execute(
            update(ImageAnalysis)
            .where(ImageAnalysis.id == item.id, ImageAnalysis.status == "queued")
            .values(status="running", lease_token=token, lease_until=time.time() + 180)
        ).rowcount:
            return False
        asset = s.get(Asset, item.asset_id)
        key = asset.key
    stopped, errors = Event(), []

    def pulse():
        while not stopped.wait(20):
            try:
                with db.transaction() as s:
                    if not s.execute(
                        update(ImageAnalysis)
                        .where(
                            ImageAnalysis.id == item.id,
                            ImageAnalysis.status == "running",
                            ImageAnalysis.lease_token == token,
                        )
                        .values(lease_until=time.time() + 180)
                    ).rowcount:
                        raise RuntimeError("Vision lease lost")
            except Exception as exc:
                errors.append(exc)
                return

    thread = Thread(target=pulse, daemon=True)
    thread.start()
    began, raw, metrics = time.monotonic(), None, {}
    try:
        if (item.config["model"], item.config["revision"], item.config["prompt_version"]) != (
            MODEL_ID,
            MODEL_REVISION,
            PROMPT_VERSION,
        ):
            raise ValueError("Queued analysis version differs from this worker")
        data = storage.get(key)
        if hashlib.sha256(data).hexdigest() != item.source_sha256:
            raise ValueError("Source integrity check failed")
        with Image.open(BytesIO(data)) as source:
            raw, metrics = analyzer.analyze(source.convert("RGB"), item.config["mode"])
        report = parse_report(raw).model_dump(mode="json")
        if errors:
            raise errors[0]
        values = {"status": "succeeded", "report": report, "error": None}
    except Exception:
        log.exception("Image analysis %s failed", item.id)
        values = {"status": "failed", "error": "Analysis failed validation or inference; inspect worker logs and retry"}
    finally:
        stopped.set()
        thread.join(timeout=30)
    values.update(
        raw_output=raw[:40000] if isinstance(raw, str) else None,
        metrics={**metrics, "seconds": round(time.monotonic() - began, 3)},
        finished=time.time(),
    )
    with db.transaction() as s:
        if s.execute(
            update(ImageAnalysis)
            .where(ImageAnalysis.id == item.id, ImageAnalysis.status == "running", ImageAnalysis.lease_token == token)
            .values(**values)
        ).rowcount:
            s.add(
                AnalysisAttempt(
                    analysis_id=item.id,
                    status=values["status"],
                    raw_output=values["raw_output"],
                    metrics=values["metrics"],
                    error=values.get("error"),
                )
            )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = PlatformSettings()
    if not settings.vision_enabled:
        raise SystemExit("VISION_ANALYSIS_ENABLED=true is required")
    db, storage, analyzer = Database(settings.database_url), Storage(settings), LocalVision(settings)
    stopping = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    while not stopping.is_set():
        worked = analyze_one(db, storage, analyzer)
        if args.once:
            break
        if not worked:
            stopping.wait(2)


if __name__ == "__main__":
    main()
