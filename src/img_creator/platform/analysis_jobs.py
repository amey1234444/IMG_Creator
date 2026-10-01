"""Owner-only annotation queue and explicit approval into the training dataset."""

import hashlib
import json
import math
from io import BytesIO
from typing import Literal
from PIL import Image
from fastapi import Depends, HTTPException, Response
from pydantic import Field
from sqlalchemy import select, func, update
from sqlalchemy.exc import IntegrityError
from .db import Asset, ImageAnalysis, AnalysisReview, AnalysisAttempt, Audit
from .security import rate_limit
from .vision import Strict, ImageReport, MODEL_ID, MODEL_REVISION, PROMPT_VERSION


class AnalyzeRequest(Strict):
    mode: Literal["overview", "detail", "deep"] = "detail"


class ReviewAnalysis(Strict):
    report: ImageReport
    caption: str = Field(min_length=10, max_length=1200)


def analysis_json(item):
    return {
        key: getattr(item, key)
        for key in (
            "id",
            "asset_id",
            "status",
            "config",
            "source_sha256",
            "report",
            "metrics",
            "error",
            "created",
            "finished",
        )
    }


def install_analysis_routes(app, owner, db, storage, settings):
    @app.post("/api/admin/assets/{asset_id}/analyze", status_code=202)
    def analyze(asset_id: str, body: AnalyzeRequest, user=Depends(owner)):
        if not settings.vision_enabled:
            raise HTTPException(503, "Enable and deploy the local vision worker first")
        rate_limit(db, "image-analysis", user.id, 60, 3600)
        try:
            with db.transaction() as s:
                asset = s.scalar(select(Asset).where(Asset.id == asset_id).with_for_update())
                if not asset or asset.kind != "image":
                    raise HTTPException(404, "Training image not found")
                config = {
                    "mode": body.mode,
                    "model": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "prompt_version": PROMPT_VERSION,
                }
                fingerprint = hashlib.sha256(
                    json.dumps({"asset": asset.id, "sha256": asset.sha256, **config}, sort_keys=True).encode()
                ).hexdigest()
                existing = s.scalar(select(ImageAnalysis).where(ImageAnalysis.fingerprint == fingerprint))
                if existing:
                    return analysis_json(existing)
                if (
                    s.scalar(
                        select(func.count())
                        .select_from(ImageAnalysis)
                        .where(ImageAnalysis.status.in_(["queued", "running"]))
                    )
                    >= 100
                ):
                    raise HTTPException(429, "Analysis queue is full")
                item = ImageAnalysis(
                    asset_id=asset.id, source_sha256=asset.sha256, config=config, fingerprint=fingerprint
                )
                s.add(item)
                s.flush()
                s.add(Audit(actor=user.id, action="image_analysis_queued", target=item.id))
                return analysis_json(item)
        except IntegrityError:
            with db.transaction() as s:
                existing = s.scalar(select(ImageAnalysis).where(ImageAnalysis.fingerprint == fingerprint))
                if existing:
                    return analysis_json(existing)
            raise

    @app.get("/api/admin/assets/{asset_id}/analyses")
    def analyses(asset_id: str, user=Depends(owner)):
        with db.transaction() as s:
            rows = []
            for a in s.scalars(
                select(ImageAnalysis)
                .where(ImageAnalysis.asset_id == asset_id)
                .order_by(ImageAnalysis.created.desc())
                .limit(10)
            ):
                review = s.scalar(
                    select(AnalysisReview)
                    .where(AnalysisReview.analysis_id == a.id)
                    .order_by(AnalysisReview.created.desc())
                    .limit(1)
                )
                rows.append(
                    {
                        **analysis_json(a),
                        "latest_review": None
                        if not review
                        else {
                            "id": review.id,
                            "report": review.report,
                            "caption": review.caption,
                            "reviewer": review.reviewer,
                            "created": review.created,
                        },
                    }
                )
            return rows

    @app.get("/api/admin/analysis-reviews/{review_id}")
    def reviewed_analysis(review_id: str, user=Depends(owner)):
        with db.transaction() as s:
            review = s.get(AnalysisReview, review_id)
            if not review:
                raise HTTPException(404, "Review not found")
            analysis = s.get(ImageAnalysis, review.analysis_id)
            return {
                "id": review.id,
                "asset_id": review.asset_id,
                "caption": review.caption,
                "report": review.report,
                "reviewer": review.reviewer,
                "created": review.created,
                "analysis": analysis_json(analysis),
            }

    @app.get("/api/admin/analyses/{analysis_id}/attempts")
    def attempts(analysis_id: str, user=Depends(owner)):
        with db.transaction() as s:
            return [
                {
                    "id": a.id,
                    "status": a.status,
                    "raw_output": a.raw_output,
                    "metrics": a.metrics,
                    "error": a.error,
                    "created": a.created,
                }
                for a in s.scalars(
                    select(AnalysisAttempt)
                    .where(AnalysisAttempt.analysis_id == analysis_id)
                    .order_by(AnalysisAttempt.created.desc())
                    .limit(20)
                )
            ]

    @app.post("/api/admin/analyses/{analysis_id}/retry")
    def retry(analysis_id: str, user=Depends(owner)):
        if not settings.vision_enabled:
            raise HTTPException(503, "Vision analysis is disabled")
        rate_limit(db, "image-analysis", user.id, 60, 3600)
        with db.transaction() as s:
            if not s.execute(
                update(ImageAnalysis)
                .where(ImageAnalysis.id == analysis_id, ImageAnalysis.status == "failed")
                .values(
                    status="queued",
                    error=None,
                    raw_output=None,
                    report=None,
                    metrics={},
                    finished=None,
                    lease_token=None,
                    lease_until=0,
                )
            ).rowcount:
                raise HTTPException(409, "Only failed analyses can be retried")
            s.add(Audit(actor=user.id, action="image_analysis_retry", target=analysis_id))
        return {"ok": True}

    @app.post("/api/admin/analyses/{analysis_id}/approve")
    def approve(analysis_id: str, body: ReviewAnalysis, user=Depends(owner)):
        caption = body.caption.strip()
        if len(caption) < 10:
            raise HTTPException(422, "Write a descriptive training caption")
        with db.transaction() as s:
            item = s.get(ImageAnalysis, analysis_id)
            if not item or item.status != "succeeded":
                raise HTTPException(409, "Review a completed analysis")
            asset = s.scalar(select(Asset).where(Asset.id == item.asset_id).with_for_update())
            if not asset or asset.sha256 != item.source_sha256:
                raise HTTPException(409, "The analyzed source has changed")
            review = AnalysisReview(
                analysis_id=item.id,
                asset_id=asset.id,
                reviewer=user.id,
                report=body.report.model_dump(mode="json"),
                caption=caption,
            )
            s.add(review)
            s.flush()
            asset.caption, asset.approved = caption, True
            s.add(
                Audit(
                    actor=user.id,
                    action="image_analysis_approved",
                    target=review.id,
                    detail={"analysis_id": item.id, "asset_id": asset.id},
                )
            )
            return {"review_id": review.id, "approved": True}

    @app.get("/api/admin/analyses/{analysis_id}/regions/{index}")
    def region(analysis_id: str, index: int, review_id: str | None = None, user=Depends(owner)):
        rate_limit(db, "analysis-region-download", user.id, 60, 60)
        with db.transaction() as s:
            item = s.get(ImageAnalysis, analysis_id)
            if not item or item.status != "succeeded":
                raise HTTPException(404, "Analysis not found")
            review = s.get(AnalysisReview, review_id) if review_id else None
            if review_id and (not review or review.analysis_id != item.id):
                raise HTTPException(404, "Review not found")
            report = ImageReport.model_validate(review.report if review else item.report)
            if not 0 <= index < len(report.objects):
                raise HTTPException(404, "Region not found")
            asset = s.get(Asset, item.asset_id)
            data = storage.get(asset.key)
            if hashlib.sha256(data).hexdigest() != item.source_sha256:
                raise HTTPException(409, "Source integrity check failed")
        with Image.open(BytesIO(data)) as image:
            x1, y1, x2, y2 = report.objects[index].bbox
            box = (
                math.floor(x1 * image.width),
                math.floor(y1 * image.height),
                math.ceil(x2 * image.width),
                math.ceil(y2 * image.height),
            )
            output = BytesIO()
            image.crop(box).save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")
