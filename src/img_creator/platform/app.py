from __future__ import annotations
import hmac
import secrets
import time
from pathlib import Path
from typing import Literal
import stripe
from fastapi import FastAPI, Depends, HTTPException, Request, Response, UploadFile, File, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update, func, delete
from sqlalchemy.exc import IntegrityError
from .config import PlatformSettings
from .db import Database, User, Session, Job, Ledger, Dataset, Asset, TrainingRun, Audit, Payment
from .security import password_hash, password_matches, email_address, digest, rate_limit
from .catalog import Generation, MODELS, RATIOS
from .storage import Storage
from .jobs import submit, serialize
from .datasets import ingest, snapshot_run
from . import billing
from .metrics import usage_rows, empty_usage
from .middleware import BodyLimitMiddleware, BodyTooLarge

STATIC = Path(__file__).parent / "static"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Credentials(Strict):
    email: str = Field(max_length=254)
    password: str = Field(min_length=12, max_length=128)


class Plan(Strict):
    plan: Literal["starter", "pro"]


class CreditGrant(Strict):
    amount: int = Field(ge=1, le=100000, strict=True)
    reason: str = Field(min_length=5, max_length=200)
    reference: str = Field(min_length=8, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")


class AccountState(Strict):
    enabled: bool


class NewDataset(Strict):
    name: str = Field(min_length=1, max_length=100)
    rights_note: str = Field(min_length=10, max_length=2000)


class ReviewAsset(Strict):
    caption: str = Field(default="", max_length=4000)
    approved: bool = False


class TrainConfig(Strict):
    dataset_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    steps: int = Field(default=500, ge=100, le=3000, strict=True)
    rank: Literal[8, 16, 32] = 16
    learning_rate: float = Field(default=0.0001, ge=0.000001, le=0.001)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class Evaluation(Strict):
    realism: float = Field(ge=1, le=5)
    prompt_adherence: float = Field(ge=1, le=5)
    artifacts: float = Field(ge=1, le=5, description="5 means minimal artifacts")
    sample_count: int = Field(ge=10, le=10000)
    report: str = Field(min_length=30, max_length=10000)
    decision: Literal["approve", "reject"]


def public_user(user):
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "credits": user.credits,
        "enabled": user.enabled,
        "plan": user.plan,
        "subscription_status": user.subscription_status,
    }


def asset_json(a):
    return {
        "id": a.id,
        "filename": a.filename,
        "kind": a.kind,
        "caption": a.caption,
        "approved": a.approved,
        "width": a.width,
        "height": a.height,
        "text_preview": a.text[:2000],
        "sha256": a.sha256,
    }


def run_json(r):
    return {
        "id": r.id,
        "dataset_id": r.dataset_id,
        "status": r.status,
        "config": r.config,
        "metrics": r.metrics,
        "evaluation": r.evaluation,
        "error": r.error,
        "image_count": len(r.snapshot),
        "created": r.created,
        "finished": r.finished,
        "artifact_sha256": r.artifact_sha256,
    }


def create_app(settings=None, db=None, storage=None):
    settings = settings or PlatformSettings()
    db = db or Database(settings.database_url)
    storage = storage or Storage(settings)
    app = FastAPI(title="IMG Creator Studio", version="0.3.0")
    app.state.db, app.state.settings, app.state.storage = db, settings, storage
    app.add_middleware(BodyLimitMiddleware, limit=settings.upload_limit + 1024 * 1024)

    @app.exception_handler(BodyTooLarge)
    async def too_large(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Request is too large"}, status_code=413)

    @app.middleware("http")
    async def headers(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != settings.public_url:
                return Response("Cross-origin request rejected", status_code=403)
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > settings.upload_limit + 1024 * 1024):
                return Response("Request is too large", status_code=413)
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",
                "X-Frame-Options": "DENY",
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            }
        )
        return response

    def current(request: Request):
        token = digest(request.cookies.get("img_session", ""))
        with db.transaction() as s:
            session = s.get(Session, token)
            if not session or session.expires < time.time():
                raise HTTPException(401, "Sign in to continue")
            user = s.get(User, session.user_id)
            if not user or not user.enabled:
                raise HTTPException(403, "Account is disabled")
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not hmac.compare_digest(
                session.csrf, request.headers.get("x-csrf-token", "")
            ):
                raise HTTPException(403, "Refresh the page and try again")
            request.state.session = session
            return user

    def owner(user=Depends(current)):
        if user.role != "owner":
            raise HTTPException(403, "Owner access required")
        return user

    def authenticate(response, user):
        raw, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with db.transaction() as s:
            s.execute(delete(Session).where(Session.expires < time.time()))
            s.add(Session(token=digest(raw), user_id=user.id, csrf=csrf, expires=time.time() + 86400))
        response.set_cookie(
            "img_session", raw, httponly=True, secure=settings.secure_cookies, samesite="lax", max_age=86400, path="/"
        )
        return {"user": public_user(user), "csrf": csrf}

    def ip(request):
        # Trust only the ASGI peer, never arbitrary X-Forwarded-For headers here.
        return request.client.host if request.client else "unknown"

    def model_enabled(model):
        if model in {"custom-klein-4b", "flux-2-dev-32b"}:
            return bool(
                settings.local_models
                and settings.moderation_key
                and (model != "flux-2-dev-32b" or settings.dev_license)
            )
        return bool(settings.bfl_key) and model in settings.enabled_models.split(",")

    def available(request):
        if not model_enabled(request.model):
            raise HTTPException(503, "This model is not enabled by the owner yet")
        if request.training_run:
            with db.transaction() as s:
                run = s.get(TrainingRun, request.training_run)
                if not run or run.status != "approved" or not run.artifact_key:
                    raise HTTPException(422, "Select a completed, owner-approved training run")
        if request.upscale == "learned":
            from ..config import Settings

            if not Settings().sr_weights:
                raise HTTPException(422, "Learned upscaling is not configured on the worker")

    @app.get("/health")
    def health():
        with db.transaction() as s:
            s.execute(select(1))
        return {"status": "ok"}

    @app.get("/api/catalog")
    def catalog():
        return {
            "models": [{"id": k, **v, "available": model_enabled(k)} for k, v in MODELS.items()],
            "ratios": list(RATIOS),
            "resolutions": ["1K", "2K", "4K", "8K"],
            "plans": [
                {"id": k, "credits": v["credits"], "available": bool(settings.stripe_key and v["price_id"])}
                for k, v in settings.plans.items()
            ],
        }

    @app.post("/api/auth/register")
    def register(body: Credentials, request: Request, response: Response):
        rate_limit(db, "register", ip(request), 5, 3600)
        email = email_address(body.email)
        try:
            with db.transaction() as s:
                user = User(email=email, password=password_hash(body.password))
                s.add(user)
                s.flush()
        except IntegrityError:
            raise HTTPException(409, "Unable to register this email. Try signing in.")
        return authenticate(response, user)

    @app.post("/api/auth/login")
    def login(body: Credentials, request: Request, response: Response):
        rate_limit(db, "login-ip", ip(request), 30, 900)
        email = email_address(body.email)
        rate_limit(db, "login-account", email, 10, 900)
        with db.transaction() as s:
            user = s.scalar(select(User).where(User.email == email))
            # Fixed-cost dummy hash avoids skipping the password work for unknown accounts.
            encoded = user.password if user else "scrypt$00000000000000000000000000000000$" + "0" * 128
            valid = password_matches(body.password, encoded)
            if not valid or not user or not user.enabled:
                raise HTTPException(401, "Incorrect credentials or unavailable account")
        return authenticate(response, user)

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response, user=Depends(current)):
        with db.transaction() as s:
            s.execute(delete(Session).where(Session.token == request.state.session.token))
        response.delete_cookie("img_session", path="/")
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request, user=Depends(current)):
        return {"user": public_user(user), "csrf": request.state.session.csrf}

    @app.get("/api/adapters")
    def adapters(user=Depends(current)):
        with db.transaction() as s:
            return [
                {"id": r.id, "name": f"Studio adapter {r.id[:8]}", "created": r.created}
                for r in s.scalars(
                    select(TrainingRun)
                    .where(TrainingRun.status == "approved")
                    .order_by(TrainingRun.created.desc())
                    .limit(100)
                )
            ]

    @app.post("/api/quote")
    def quote(body: Generation, user=Depends(current)):
        native, final = body.dimensions()
        return {
            "credits": body.quote(),
            "native": native,
            "export": final,
            "method": "native" if native == final else body.upscale,
            "note": "Export size is not native model resolution. Credits are not tokens.",
        }

    @app.post("/api/jobs", status_code=202)
    def generate(body: Generation, idempotency_key: str = Header(min_length=8, max_length=100), user=Depends(current)):
        available(body)
        rate_limit(db, "jobs", user.id, 60, 3600)
        return serialize(submit(db, user.id, body, idempotency_key))

    @app.get("/api/jobs")
    def jobs(user=Depends(current), offset: int = 0):
        if not 0 <= offset <= 100000:
            raise HTTPException(422, "Invalid offset")
        with db.transaction() as s:
            return [
                serialize(j)
                for j in s.scalars(
                    select(Job).where(Job.user_id == user.id).order_by(Job.created.desc()).offset(offset).limit(50)
                )
            ]

    def owned_job(s, job_id, user):
        job = s.get(Job, job_id)
        if not job or job.user_id != user.id:
            raise HTTPException(404, "Image not found")
        return job

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, user=Depends(current)):
        with db.transaction() as s:
            return serialize(owned_job(s, job_id, user))

    @app.get("/api/jobs/{job_id}/image")
    def image(job_id: str, user=Depends(current)):
        with db.transaction() as s:
            item = owned_job(s, job_id, user)
            if item.status != "succeeded":
                raise HTTPException(404, "Image is not ready")
            key = item.result["key"]
        return Response(
            storage.get(key),
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{item.id}.png"'},
        )

    @app.get("/api/usage")
    def usage(user=Depends(current)):
        with db.transaction() as s:
            entries = s.scalars(
                select(Ledger).where(Ledger.user_id == user.id).order_by(Ledger.created.desc()).limit(100)
            )
            return [{"delta": x.delta, "reason": x.reason, "created": x.created} for x in entries]

    @app.post("/api/billing/checkout")
    def checkout(body: Plan, user=Depends(current)):
        return billing.checkout(db, settings, user.id, body.plan)

    @app.post("/api/billing/portal")
    def portal(user=Depends(current)):
        if not user.stripe_customer:
            raise HTTPException(422, "No billing account yet")
        result = billing.client(settings).v1.billing_portal.sessions.create(
            {"customer": user.stripe_customer, "return_url": settings.public_url}
        )
        return {"url": result.url}

    @app.post("/api/billing/webhook")
    async def webhook(request: Request):
        if not settings.stripe_webhook_secret:
            raise HTTPException(503, "Billing is not configured")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1_000_000:
                raise HTTPException(413, "Webhook too large")
        try:
            event = stripe.Webhook.construct_event(
                bytes(body), request.headers.get("stripe-signature", ""), settings.stripe_webhook_secret
            )
        except (ValueError, stripe.SignatureVerificationError):
            raise HTTPException(400, "Invalid webhook signature")
        # Synchronous SDK/DB work is moved off the async event loop.
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(billing.process_event, db, settings, event)
        return {"received": True}

    @app.get("/api/admin/metrics")
    def metrics(user=Depends(owner)):
        with db.transaction() as s:
            counts = dict(s.execute(select(Job.status, func.count()).group_by(Job.status)).all())
            by_model = usage_rows(s, Job.request["model"].as_string())
            token_values = [m["provider_tokens"] for m in by_model.values() if m["provider_tokens"] is not None]
            return {
                "users": s.scalar(select(func.count()).select_from(User)),
                "subscribers": s.scalar(
                    select(func.count()).select_from(User).where(User.subscription_status == "active")
                ),
                "jobs": counts,
                "models": by_model,
                "credits_consumed": sum(m["credits"] for m in by_model.values()),
                "credits_outstanding": s.scalar(select(func.coalesce(func.sum(User.credits), 0))),
                "provider_tokens": sum(token_values) if token_values else None,
                "token_reporting_images": sum(m["token_reporting_images"] for m in by_model.values()),
                "provider_cost": None,
                "revenue_minor_units": dict(
                    s.execute(select(Payment.currency, func.sum(Payment.amount)).group_by(Payment.currency)).all()
                ),
                "note": "Tokens cover only reporting providers; null means unreported. Time is worker wall time, "
                "not GPU billing time. Provider cost is unreported. Revenue excludes refunds and disputes.",
            }

    @app.get("/api/admin/users")
    def users(user=Depends(owner), offset: int = 0):
        if not 0 <= offset <= 100000:
            raise HTTPException(422, "Invalid offset")
        with db.transaction() as s:
            page = list(s.scalars(select(User).order_by(User.created.desc()).offset(offset).limit(100)))
            totals = usage_rows(s, Job.user_id, [u.id for u in page])
            return [{**public_user(u), "usage": totals.get(u.id, empty_usage())} for u in page]

    @app.post("/api/admin/users/{user_id}/credits")
    def grant(user_id: str, body: CreditGrant, user=Depends(owner)):
        with db.transaction() as s:
            target = s.scalar(select(User).where(User.id == user_id).with_for_update())
            if not target:
                raise HTTPException(404, "User not found")
            ref = "owner-grant:" + body.reference
            existing = s.scalar(select(Ledger).where(Ledger.reference == ref))
            if existing:
                if existing.user_id != user_id or existing.delta != body.amount:
                    raise HTTPException(409, "Grant reference already used")
                return {"ok": True}
            target.credits += body.amount
            s.add(Ledger(user_id=user_id, delta=body.amount, reason="owner_grant", reference=ref))
            s.add(Audit(actor=user.id, action="credit_grant", target=user_id, detail=body.model_dump()))
        return {"ok": True}

    @app.patch("/api/admin/users/{user_id}")
    def account(user_id: str, body: AccountState, user=Depends(owner)):
        with db.transaction() as s:
            target = s.get(User, user_id)
            if not target:
                raise HTTPException(404, "User not found")
            if target.role == "owner":
                raise HTTPException(422, "Owner accounts cannot be disabled through this endpoint")
            target.enabled = body.enabled
            if not body.enabled:
                s.execute(delete(Session).where(Session.user_id == target.id))
            s.add(Audit(actor=user.id, action="account_state", target=user_id, detail=body.model_dump()))
        return {"ok": True}

    @app.get("/api/admin/jobs")
    def admin_jobs(user=Depends(owner)):
        with db.transaction() as s:
            return [
                {**serialize(j), "user_id": j.user_id, "provider_state": j.provider_state}
                for j in s.scalars(select(Job).order_by(Job.created.desc()).limit(100))
            ]

    @app.post("/api/admin/jobs/{job_id}/refund")
    def reconcile(job_id: str, user=Depends(owner)):
        with db.transaction() as s:
            j = s.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if not j or j.status != "uncertain":
                raise HTTPException(409, "Only uncertain jobs need reconciliation")
            j.status = "failed"
            j.error = "Owner refunded this interrupted generation"
            s.execute(update(User).where(User.id == j.user_id).values(credits=User.credits + j.credits))
            s.add(Ledger(user_id=j.user_id, delta=j.credits, reason="generation_refund", reference="refund:" + j.id))
            s.add(Audit(actor=user.id, action="job_refund", target=j.id))
        return {"ok": True}

    @app.post("/api/admin/datasets")
    def create_dataset(body: NewDataset, user=Depends(owner)):
        with db.transaction() as s:
            item = Dataset(**body.model_dump())
            s.add(item)
            s.flush()
            s.add(Audit(actor=user.id, action="dataset_create", target=item.id))
            return {"id": item.id, "name": item.name}

    @app.get("/api/admin/datasets")
    def datasets(user=Depends(owner)):
        with db.transaction() as s:
            return [
                {
                    "id": d.id,
                    "name": d.name,
                    "rights_note": d.rights_note,
                    "assets": s.scalar(select(func.count()).select_from(Asset).where(Asset.dataset_id == d.id)),
                }
                for d in s.scalars(select(Dataset).order_by(Dataset.created.desc()).limit(100))
            ]

    @app.post("/api/admin/datasets/{dataset_id}/assets")
    def upload(dataset_id: str, file: UploadFile = File(), user=Depends(owner)):
        data = file.file.read(settings.upload_limit + 1)
        if len(data) > settings.upload_limit:
            raise HTTPException(413, "File exceeds 25 MB; use the large-dataset CLI for bulk collections")
        try:
            with db.transaction() as s:
                if not s.get(Dataset, dataset_id):
                    raise HTTPException(404, "Dataset not found")
                a = ingest(s, storage, dataset_id, file.filename or "upload", data)
                s.add(Audit(actor=user.id, action="asset_upload", target=a.id))
                s.flush()
                return asset_json(a)
        except HTTPException:
            raise
        except IntegrityError:
            raise HTTPException(409, "Duplicate asset")
        except Exception:
            raise HTTPException(422, "Unable to read this file. Check the supported format and file limits.")

    @app.get("/api/admin/datasets/{dataset_id}/assets")
    def assets(dataset_id: str, user=Depends(owner)):
        with db.transaction() as s:
            return [
                asset_json(a)
                for a in s.scalars(
                    select(Asset).where(Asset.dataset_id == dataset_id).order_by(Asset.created).limit(2000)
                )
            ]

    @app.get("/api/admin/assets/{asset_id}/image")
    def asset_image(asset_id: str, user=Depends(owner)):
        with db.transaction() as s:
            a = s.get(Asset, asset_id)
            if not a or a.kind != "image":
                raise HTTPException(404, "Image not found")
            return Response(storage.get(a.key), media_type="image/png")

    @app.patch("/api/admin/assets/{asset_id}")
    def review_asset(asset_id: str, body: ReviewAsset, user=Depends(owner)):
        with db.transaction() as s:
            a = s.get(Asset, asset_id)
            if not a:
                raise HTTPException(404, "Asset not found")
            if a.kind == "image" and body.approved and not body.caption.strip():
                raise HTTPException(422, "Caption is required before approval")
            a.caption, a.approved = body.caption.strip(), body.approved
            s.add(Audit(actor=user.id, action="asset_review", target=a.id, detail=body.model_dump()))
        return {"ok": True}

    @app.post("/api/admin/training")
    def training(body: TrainConfig, user=Depends(owner)):
        with db.transaction() as s:
            if not s.get(Dataset, body.dataset_id):
                raise HTTPException(404, "Dataset not found")
            run = snapshot_run(s, body.dataset_id, body.model_dump())
            s.add(Audit(actor=user.id, action="training_queued", target=run.id))
            s.flush()
            return run_json(run)

    @app.get("/api/admin/training")
    def runs(user=Depends(owner)):
        with db.transaction() as s:
            return [run_json(r) for r in s.scalars(select(TrainingRun).order_by(TrainingRun.created.desc()).limit(100))]

    @app.post("/api/admin/training/{run_id}/evaluate")
    def evaluate(run_id: str, body: Evaluation, user=Depends(owner)):
        with db.transaction() as s:
            run = s.scalar(select(TrainingRun).where(TrainingRun.id == run_id).with_for_update())
            if not run or run.status not in {"awaiting_evaluation", "approved", "rejected"}:
                raise HTTPException(409, "Only completed runs can be evaluated")
            if body.decision == "approve" and min(body.realism, body.prompt_adherence, body.artifacts) < 3:
                raise HTTPException(422, "Approval requires all review scores of at least 3/5")
            run.evaluation = {
                **body.model_dump(),
                "reviewer": user.id,
                "reviewed_at": time.time(),
                "source": "owner_manual_review",
            }
            run.status = "approved" if body.decision == "approve" else "rejected"
            s.add(Audit(actor=user.id, action="training_evaluated", target=run.id, detail=run.evaluation))
        return {"ok": True}

    @app.get("/api/admin/training/{run_id}/artifact")
    def artifact(run_id: str, user=Depends(owner)):
        with db.transaction() as s:
            run = s.get(TrainingRun, run_id)
            if not run or not run.artifact_key:
                raise HTTPException(404, "Training artifact not ready")
            return Response(
                storage.get(run.artifact_key),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{run.id}.safetensors"'},
            )

    @app.get("/api/admin/audit")
    def audit(user=Depends(owner)):
        with db.transaction() as s:
            return [
                {"actor": a.actor, "action": a.action, "target": a.target, "detail": a.detail, "created": a.created}
                for a in s.scalars(select(Audit).order_by(Audit.created.desc()).limit(100))
            ]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
