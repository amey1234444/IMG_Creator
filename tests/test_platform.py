import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select, update, func

from img_creator.platform.app import create_app
from img_creator.platform.catalog import Generation
from img_creator.platform.config import PlatformSettings
from img_creator.platform.db import Base, Database, User, Job, Ledger, TrainingRun, Session
from img_creator.platform.security import password_hash
from img_creator.platform.storage import Storage
from img_creator.platform.jobs import submit, claim, finish, recover_stale
from img_creator.platform.provider import ProviderRejected, ProviderUncertain
from img_creator.platform.worker import run_one
from img_creator.platform.billing import process_event
from img_creator.platform.datasets import decode_upload

PASSWORD = "correct horse battery staple"


@pytest.fixture
def platform(tmp_path):
    settings = PlatformSettings(
        database_url=os.getenv("TEST_DATABASE_URL", "sqlite:///" + str(tmp_path / "test.db")),
        storage_dir=tmp_path / "data",
        secure_cookies=False,
        bfl_key="test-provider-key",
        starter_price="price_starter",
        pro_price="price_pro",
    )
    db = Database(settings.database_url)
    db.initialize()
    storage = Storage(settings)
    app = create_app(settings, db, storage)
    yield app, db, storage, settings
    Base.metadata.drop_all(db.engine)
    db.engine.dispose()


def account(platform, email, role="user", credits=100):
    app, db, _, _ = platform
    with db.transaction() as s:
        u = User(email=email, password=password_hash(PASSWORD), role=role, credits=credits)
        s.add(u)
        s.flush()
        user_id = u.id
    c = TestClient(app)
    result = c.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert result.status_code == 200, result.text
    c.headers["x-csrf-token"] = result.json()["csrf"]
    return c, user_id


def test_auth_csrf_owner_and_logout(platform):
    client, user_id = account(platform, "member@example.com")
    assert client.get("/api/me").json()["user"]["role"] == "user"
    assert client.get("/api/admin/metrics").status_code == 403
    assert (
        client.post("/api/auth/register", json={"email": "x@y.com", "password": PASSWORD, "role": "owner"}).status_code
        == 422
    )
    csrf = client.headers.pop("x-csrf-token")
    assert (
        client.post("/api/jobs", json={"prompt": "a cat"}, headers={"Idempotency-Key": "test-key"}).status_code == 403
    )
    client.headers["x-csrf-token"] = csrf
    assert (
        client.post("/api/quote", json={"prompt": "a cat"}, headers={"origin": "https://evil.example"}).status_code
        == 403
    )
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/me").status_code == 401
    with platform[1].transaction() as s:
        assert s.scalar(select(func.count()).select_from(Session).where(Session.user_id == user_id)) == 0


def test_login_rate_limit(platform):
    c = TestClient(platform[0])
    for _ in range(10):
        assert c.post("/api/auth/login", json={"email": "none@example.com", "password": PASSWORD}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "none@example.com", "password": PASSWORD}).status_code == 429


def test_idempotency_conflict_balance_and_isolation(platform):
    c, uid = account(platform, "one@example.com")
    other, _ = account(platform, "two@example.com")
    response = c.post("/api/jobs", json={"prompt": "a cat"}, headers={"Idempotency-Key": "first-key"})
    assert response.status_code == 202, response.text
    j = response.json()
    assert (
        c.post("/api/jobs", json={"prompt": "a cat"}, headers={"Idempotency-Key": "first-key"}).json()["id"] == j["id"]
    )
    assert c.post("/api/jobs", json={"prompt": "a dog"}, headers={"Idempotency-Key": "first-key"}).status_code == 409
    assert other.get("/api/jobs/" + j["id"]).status_code == 404
    assert other.get("/api/jobs/" + j["id"] + "/image").status_code == 404
    assert len(c.get("/api/jobs").json()) == 1
    assert c.get("/api/me").json()["user"]["credits"] == 100 - j["credits"]


def test_concurrent_credit_reservation_cannot_overspend(platform):
    _, uid = account(platform, "race@example.com", credits=16)
    db = platform[1]

    def operation(i):
        try:
            return submit(db, uid, Generation(prompt="a cat"), f"key-{i:08d}")
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(operation, range(6)))
    assert sum(not isinstance(x, int) for x in outcomes) == 1
    with db.transaction() as s:
        assert s.get(User, uid).credits == 0
        assert s.scalar(select(func.count()).select_from(Job)) == 1


def test_worker_success_and_private_download(platform):
    c, uid = account(platform, "image@example.com")
    other, _ = account(platform, "intruder@example.com")
    j = submit(platform[1], uid, Generation(prompt="a vase"), "image-key")

    class FakeProvider:
        def generate(self, request, heartbeat):
            heartbeat({"id": "provider-id"})
            return Image.new("RGB", (512, 512), "red"), {
                "tokens": None,
                "provider_cost": None,
                "native_size": [512, 512],
            }

    assert run_one(platform[1], platform[2], FakeProvider())
    out = c.get("/api/jobs/" + j.id).json()
    assert out["status"] == "succeeded"
    assert out["result"]["upscale"]["method"] == "lanczos_resize"
    data = c.get(out["result"]["image_url"])
    assert data.status_code == 200
    assert Image.open(BytesIO(data.content)).size == (1024, 1024)
    assert other.get(out["result"]["image_url"]).status_code == 404


@pytest.mark.parametrize(
    "error,status,refunded",
    [(ProviderRejected("rejected"), "failed", True), (ProviderUncertain("unknown"), "uncertain", False)],
)
def test_worker_refunds_only_known_failures(platform, error, status, refunded):
    _, uid = account(platform, "failure@example.com")
    j = submit(platform[1], uid, Generation(prompt="a vase"), "failure-key")

    class FakeProvider:
        def generate(self, *args):
            raise error

    run_one(platform[1], platform[2], FakeProvider())
    with platform[1].transaction() as s:
        job = s.get(Job, j.id)
        assert job.status == status
        assert s.get(User, uid).credits == (100 if refunded else 100 - j.credits)
    assert not finish(platform[1], j.id, job.lease_token, error="again")


def test_stale_jobs_and_admin_refund_once(platform):
    owner, _ = account(platform, "owner@example.com", role="owner")
    _, uid = account(platform, "stale@example.com")
    job = submit(platform[1], uid, Generation(prompt="a vase"), "stale-key")
    claimed = claim(platform[1])
    with platform[1].transaction() as s:
        s.execute(update(Job).where(Job.id == job.id).values(lease_until=time.time() - 1))
    recover_stale(platform[1])
    assert not finish(platform[1], job.id, claimed.lease_token, result={"key": "anything"})
    assert owner.post("/api/admin/jobs/" + job.id + "/refund").status_code == 200
    assert owner.post("/api/admin/jobs/" + job.id + "/refund").status_code == 409
    with platform[1].transaction() as s:
        assert s.get(User, uid).credits == 100


def test_worker_routes_do_not_claim_other_models(platform):
    _, uid = account(platform, "route@example.com")
    submit(platform[1], uid, Generation(prompt="a vase"), "route-key")
    assert claim(platform[1], ["flux-2-dev-32b"]) is None
    assert claim(platform[1], ["flux-2-pro"]) is not None


def test_admin_grant_disable_and_metrics(platform):
    owner, _ = account(platform, "owner@example.com", role="owner")
    member, uid = account(platform, "member@example.com")
    body = {"amount": 25, "reason": "Welcome trial", "reference": "welcome-trial"}
    assert owner.post("/api/admin/users/" + uid + "/credits", json=body).status_code == 200
    assert owner.post("/api/admin/users/" + uid + "/credits", json=body).status_code == 200
    assert member.get("/api/me").json()["user"]["credits"] == 125
    assert owner.patch("/api/admin/users/" + uid, json={"enabled": False}).status_code == 200
    assert member.get("/api/me").status_code == 401
    metrics = owner.get("/api/admin/metrics").json()
    assert metrics["users"] == 2 and metrics["provider_tokens"] is None
    assert owner.get("/api/admin/audit").json()


def test_model_validation_and_dimensions(platform):
    for ratio in ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3"]:
        for resolution in ["1K", "2K", "4K", "8K"]:
            native, final = Generation(prompt="a photo", aspect_ratio=ratio, resolution=resolution).dimensions()
            assert native[0] * native[1] <= 4_000_000 and all(x % 16 == 0 for x in native)
            assert max(final) == {"1K": 1024, "2K": 2048, "4K": 4096, "8K": 8192}[resolution]
    c, _ = account(platform, "model@example.com")
    assert (
        c.post(
            "/api/jobs",
            json={"prompt": "a photo", "model": "flux-2-pro", "effort": "high"},
            headers={"Idempotency-Key": "model-key"},
        ).status_code
        == 422
    )
    assert (
        c.post(
            "/api/jobs", json={"prompt": "a photo", "safety_tolerance": 6}, headers={"Idempotency-Key": "model-key"}
        ).status_code
        == 422
    )
    assert (
        c.post(
            "/api/jobs",
            json={"prompt": "a photo", "model": "flux-2-dev-32b", "effort": "high"},
            headers={"Idempotency-Key": "model-key"},
        ).status_code
        == 503
    )


def test_paid_invoice_replay_and_out_of_order_subscription(platform):
    _, uid = account(platform, "payer@example.com", credits=0)
    db, settings = platform[1], platform[3]
    with db.transaction() as s:
        s.get(User, uid).stripe_customer = "cus_123"
    obj = {
        "id": "in_123",
        "customer": "cus_123",
        "status": "paid",
        "billing_reason": "subscription_cycle",
        "subscription": "sub_123",
        "lines": {"data": [{"price": {"id": "price_starter"}, "quantity": 1}]},
        "amount_paid": 1000,
        "currency": "usd",
    }
    event = {"id": "evt_paid", "type": "invoice.paid", "created": 100, "data": {"object": obj}}
    sub = {
        "status": "active",
        "customer": "cus_123",
        "items": {"data": [{"price": {"id": "price_starter"}, "quantity": 1}]},
    }
    process_event(db, settings, event, lambda _: sub)
    process_event(db, settings, event, lambda _: sub)
    process_event(db, settings, {**event, "id": "evt_paid_duplicate"}, lambda _: sub)
    with db.transaction() as s:
        assert s.get(User, uid).credits == 500
        assert s.scalar(select(func.count()).select_from(Ledger).where(Ledger.user_id == uid)) == 1
    cancel = {
        "id": "evt_cancel",
        "created": 200,
        "type": "customer.subscription.deleted",
        "data": {"object": {"id": "sub_123", "customer": "cus_123", "status": "canceled"}},
    }
    process_event(db, settings, cancel)
    process_event(
        db,
        settings,
        {
            **cancel,
            "id": "evt_old",
            "created": 150,
            "type": "customer.subscription.updated",
            "data": {"object": {"id": "sub_123", "customer": "cus_123", "status": "active"}},
        },
    )
    with db.transaction() as s:
        assert s.get(User, uid).subscription_status == "canceled"


def test_invalid_webhook_signature(platform):
    app, db, storage, settings = platform
    client = TestClient(create_app(replace(settings, stripe_webhook_secret="whsec_test"), db, storage))
    assert client.post("/api/billing/webhook", content="{}", headers={"stripe-signature": "bad"}).status_code == 400


def test_owner_dataset_snapshot_and_training_gate(platform):
    owner, _ = account(platform, "owner@example.com", role="owner")
    member, _ = account(platform, "member@example.com")
    ds = owner.post(
        "/api/admin/datasets", json={"name": "Ceramics", "rights_note": "Original studio photographs owned by us"}
    ).json()["id"]
    assert member.post("/api/admin/datasets/" + ds + "/assets", files={"file": ("x.txt", b"abc")}).status_code == 403
    for i in range(5):
        content = BytesIO()
        Image.new("RGB", (256, 256), (i * 30, 10, 10)).save(content, format="PNG")
        response = owner.post(
            "/api/admin/datasets/" + ds + "/assets", files={"file": (f"image-{i}.png", content.getvalue())}
        )
        assert response.status_code == 200, response.text
        a = response.json()
        assert (
            owner.post(
                "/api/admin/datasets/" + ds + "/assets", files={"file": ("duplicate.png", content.getvalue())}
            ).status_code
            == 409
        )
        assert owner.patch("/api/admin/assets/" + a["id"], json={"approved": True}).status_code == 422
        assert (
            owner.patch(
                "/api/admin/assets/" + a["id"], json={"caption": "A ceramic vase " + str(i), "approved": True}
            ).status_code
            == 200
        )
    document = owner.post(
        "/api/admin/datasets/" + ds + "/assets", files={"file": ("notes.md", b"Describe lighting carefully.")}
    ).json()
    assert document["kind"] == "document"
    run = owner.post("/api/admin/training", json={"dataset_id": ds}).json()
    assert run["image_count"] == 5 and run["status"] == "queued"
    with platform[1].transaction() as s:
        r = s.get(TrainingRun, run["id"])
        assert len([a for a in r.snapshot if a["split"] == "train"]) == 4
        assert r.snapshot[0]["sha256"]
    assert (
        owner.post(
            "/api/admin/training/" + run["id"] + "/evaluate",
            json={
                "realism": 5,
                "prompt_adherence": 5,
                "artifacts": 5,
                "sample_count": 10,
                "report": "Reviewed ten held-out images against the base model.",
                "decision": "approve",
            },
        ).status_code
        == 409
    )
    assert not member.get("/api/adapters").json()


def test_invalid_uploads_storage_paths_and_document_extraction(platform):
    with pytest.raises(ValueError):
        decode_upload("script.exe", b"anything")
    with pytest.raises(Exception):
        decode_upload("fake.png", b"not an image")
    assert decode_upload("notes.txt", b"Original reference notes")[1] == "document"
    for key in ["../secret", "/absolute", "images/../../bad"]:
        with pytest.raises(ValueError):
            platform[2].get(key)


def test_approved_adapter_catalog_requires_completed_run(platform):
    c, _ = account(platform, "owner@example.com", role="owner")
    from img_creator.platform.db import Dataset

    with platform[1].transaction() as s:
        d = Dataset(name="test", rights_note="owned training images")
        s.add(d)
        s.flush()
        r = TrainingRun(
            dataset_id=d.id,
            config={},
            snapshot=[],
            status="awaiting_evaluation",
            artifact_key="training/adapter.safetensors",
            artifact_sha256=hashlib.sha256(b"weights").hexdigest(),
        )
        s.add(r)
        s.flush()
        rid = r.id
    body = {
        "realism": 4,
        "prompt_adherence": 4,
        "artifacts": 4,
        "sample_count": 10,
        "report": "Reviewed ten held-out images against the base model.",
        "decision": "approve",
    }
    assert c.post("/api/admin/training/" + rid + "/evaluate", json=body).status_code == 200
    assert c.get("/api/adapters").json()[0]["id"] == rid
    assert c.post("/api/admin/training/" + rid + "/evaluate", json={**body, "decision": "reject"}).status_code == 200
    assert not c.get("/api/adapters").json()


def test_usage_aggregates_separate_tokens_from_credits(platform):
    owner, _ = account(platform, "metrics-owner@example.com", role="owner")
    _, uid = account(platform, "metrics-user@example.com")
    job = submit(platform[1], uid, Generation(prompt="a vase"), "metrics-key")
    running = claim(platform[1])
    assert finish(platform[1], job.id, running.lease_token, result={"seconds": 2.5, "tokens": None})
    metrics = owner.get("/api/admin/metrics").json()
    assert metrics["credits_consumed"] == job.credits
    assert metrics["provider_tokens"] is None
    assert metrics["models"]["flux-2-pro"]["seconds"] == 2.5
    users = owner.get("/api/admin/users").json()
    member = next(u for u in users if u["id"] == uid)
    assert member["usage"]["images"] == 1
    assert member["usage"]["credits"] == job.credits
    assert member["usage"]["provider_tokens"] is None


def test_actual_provider_tokens_are_summed_without_filling_missing_data(platform):
    owner, uid = account(platform, "tokens-owner@example.com", role="owner")
    for index, tokens in enumerate([None, 150]):
        job = submit(platform[1], uid, Generation(prompt="a vase"), f"tokens-key-{index}")
        running = claim(platform[1])
        finish(platform[1], job.id, running.lease_token, result={"seconds": 1.0, "tokens": tokens})
    metrics = owner.get("/api/admin/metrics").json()
    assert metrics["provider_tokens"] == 150
    assert metrics["token_reporting_images"] == 1
    assert metrics["models"]["flux-2-pro"]["images"] == 2


def test_checkout_reuses_open_session_and_blocks_second_plan(platform, monkeypatch):
    from types import SimpleNamespace
    from img_creator.platform import billing
    from img_creator.platform.db import BillingCheckout

    _, uid = account(platform, "checkout@example.com", credits=0)
    with platform[1].transaction() as session:
        session.get(User, uid).stripe_customer = "cus_checkout"
    calls = []
    result = SimpleNamespace(id="cs_test", url="https://checkout.stripe.com/test", status="open")

    class Sessions:
        def create(self, params, options):
            calls.append(params)
            return result

        def retrieve(self, session_id):
            assert session_id == "cs_test"
            return result

    fake = SimpleNamespace(v1=SimpleNamespace(checkout=SimpleNamespace(sessions=Sessions())))
    monkeypatch.setattr(billing, "client", lambda _: fake)
    first = billing.checkout(platform[1], platform[3], uid, "starter")
    assert billing.checkout(platform[1], platform[3], uid, "starter") == first
    assert len(calls) == 1
    with pytest.raises(HTTPException) as error:
        billing.checkout(platform[1], platform[3], uid, "pro")
    assert error.value.status_code == 409
    result.status = "complete"
    with pytest.raises(HTTPException) as error:
        billing.checkout(platform[1], platform[3], uid, "starter")
    assert error.value.status_code == 409
    with platform[1].transaction() as session:
        assert session.get(BillingCheckout, uid).session_id == "cs_test"


def test_invoice_credits_use_purchased_price_not_later_plan(platform):
    _, uid = account(platform, "invoice-plan@example.com", credits=0)
    with platform[1].transaction() as session:
        session.get(User, uid).stripe_customer = "cus_plan"
    invoice = {
        "id": "in_old_plan",
        "customer": "cus_plan",
        "status": "paid",
        "billing_reason": "subscription_cycle",
        "subscription": "sub_plan",
        "lines": {"data": [{"pricing": {"price_details": {"price": "price_starter"}}, "quantity": 1}]},
        "amount_paid": 1000,
        "currency": "usd",
    }
    current = {
        "status": "active",
        "customer": "cus_plan",
        "items": {"data": [{"price": {"id": "price_pro"}, "quantity": 1}]},
    }
    process_event(
        platform[1],
        platform[3],
        {"id": "evt_old_plan", "type": "invoice.paid", "created": 100, "data": {"object": invoice}},
        lambda _: current,
    )
    with platform[1].transaction() as session:
        user = session.get(User, uid)
        assert user.credits == 500
        assert user.plan == "pro"


def test_streamed_body_limit_without_content_length():
    import asyncio
    from img_creator.platform.middleware import BodyLimitMiddleware

    output = []

    async def consuming_app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})

    chunks = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ]
    )

    async def receive():
        return next(chunks)

    async def send(message):
        output.append(message)

    asyncio.run(
        BodyLimitMiddleware(consuming_app, limit=5)({"type": "http", "method": "POST", "headers": []}, receive, send)
    )
    assert output[0]["status"] == 413


def test_hosted_provider_payload_and_download_credentials(monkeypatch):
    import httpx
    from img_creator.platform import provider

    data = BytesIO()
    Image.new("RGB", (64, 64), "green").save(data, format="PNG")
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            import json

            payload = json.loads(request.content)
            assert payload["steps"] == 50
            assert "safety_tolerance" not in payload
            assert payload["width"] * payload["height"] <= 4_000_000
            return httpx.Response(200, json={"id": "task123", "polling_url": "https://api.bfl.ai/poll/123"})
        if request.url.host == "api.bfl.ai":
            assert request.headers["x-key"] == "private-test-key"
            return httpx.Response(200, json={"status": "Ready", "result": {"sample": "https://images.example/image"}})
        assert "x-key" not in request.headers
        return httpx.Response(200, content=data.getvalue())

    original = httpx.Client
    monkeypatch.setattr(
        provider.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), trust_env=False, **kwargs),
    )
    monkeypatch.setattr(provider, "safe_url", lambda url, polling=False: url)
    monkeypatch.setattr(provider.time, "sleep", lambda _: None)
    image, metadata = provider.BFLProvider("private-test-key").generate(
        Generation(prompt="A ceramic vase", model="flux-2-flex", effort="high", resolution="4K"), lambda *_: None
    )
    assert image.size == (64, 64)
    assert metadata["tokens"] is None
    assert len(calls) == 3


def test_provider_rejects_private_and_untrusted_polling_urls(monkeypatch):
    from img_creator.platform import provider

    monkeypatch.setattr(provider.socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 443))])
    with pytest.raises(ValueError):
        provider.safe_url("https://api.bfl.ai/poll", polling=True)
    with pytest.raises(ValueError):
        provider.safe_url("https://attacker.example/poll", polling=True)
    with pytest.raises(ValueError):
        provider.safe_url("http://api.bfl.ai/poll", polling=True)


def test_schema_upgrade_is_additive_and_repeatable(platform):
    from img_creator.platform.db import (
        SchemaRevision,
        AssetSource,
        DatasetVersion,
        DatasetVersionItem,
        RunDataset,
        TrainingArtifact,
    )

    _, uid = account(platform, "migration@example.com")
    db = platform[1]
    for table in [RunDataset, DatasetVersionItem, DatasetVersion, AssetSource, TrainingArtifact, SchemaRevision]:
        table.__table__.drop(db.engine)
    db.initialize()
    db.initialize()
    with db.transaction() as s:
        assert s.get(User, uid).email == "migration@example.com"
        assert s.get(SchemaRevision, 2)


def test_versioned_datasets_preserve_captions_and_groups(platform):
    from img_creator.platform.db import Dataset, Asset, AssetSource, DatasetVersion, DatasetVersionItem
    from img_creator.platform.datasets import ingest, snapshot_run

    db, storage = platform[1:3]
    with db.transaction() as s:
        d = Dataset(name="Versioned", rights_note="Owned source photographs")
        s.add(d)
        s.flush()
        for i in range(10):
            data = BytesIO()
            Image.new("RGB", (256, 256), (i * 20, 12, 18)).save(data, format="PNG")
            a = ingest(s, storage, d.id, f"{i}.png", data.getvalue())
            a.caption = f"Vase {i}"
            a.approved = True
            s.flush()
            s.get(AssetSource, a.id).group_id = f"shoot-{i // 2}"
        cfg = {"steps": 100, "rank": 8, "seed": 42, "learning_rate": 0.0001}
        r = snapshot_run(s, d.id, cfg)
        s.flush()
        snapshot = r.snapshot
        version = r.config["dataset_version_id"]
        assert all(
            len({a["split"] for a in snapshot if a["group_id"] == group}) == 1
            for group in {a["group_id"] for a in snapshot}
        )
        assert sum(a["split"] == "validation" for a in snapshot) == 2
        second = snapshot_run(s, d.id, cfg)
        assert second.config["dataset_version_id"] == version
        s.get(Asset, snapshot[0]["id"]).caption = "Changed for a future experiment"
        s.flush()
        third = snapshot_run(s, d.id, cfg)
        assert third.config["dataset_version_id"] != version
        assert s.get(DatasetVersionItem, (version, snapshot[0]["id"])).caption == snapshot[0]["caption"]
        assert len(list(s.scalars(select(DatasetVersion)))) == 2
        source = s.get(AssetSource, snapshot[0]["id"])
        assert hashlib.sha256(storage.get(source.original_key)).hexdigest() == source.original_sha256


def test_api_limits_and_docs_default_off(platform):
    app, db, storage, settings = platform
    c, uid = account(platform, "rate@example.com")
    limited = TestClient(create_app(replace(settings, api_requests_per_minute=10), db, storage))
    limited.cookies.update(c.cookies)
    assert limited.get("/docs").status_code == 404
    assert limited.get("/openapi.json").status_code == 404
    for _ in range(10):
        assert limited.get("/api/me").status_code == 200
    r = limited.get("/api/me")
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_archive_restore_and_corruption_detection(platform, tmp_path):
    from img_creator.platform.db import Dataset, TrainingArtifact
    from img_creator.platform.artifacts import archive_outputs, restore_run

    db, storage = platform[1:3]
    with db.transaction() as s:
        ds = Dataset(name="Recover", rights_note="Owned reference images")
        s.add(ds)
        s.flush()
        r = TrainingRun(dataset_id=ds.id, config={}, snapshot=[])
        s.add(r)
        s.flush()
        rid = r.id
    root = tmp_path / "work"
    root.mkdir()
    (root / "snapshot.json").write_text("[]")
    for step in (100, 200):
        p = root / "output" / f"checkpoint-{step}"
        p.mkdir(parents=True)
        (p / "model.safetensors").write_bytes(f"weights-{step}".encode())
    assert archive_outputs(db, storage, rid, root, complete=False) == 2
    assert archive_outputs(db, storage, rid, root, complete=False) == 0
    with db.transaction() as s:
        files = list(s.scalars(select(TrainingArtifact).where(TrainingArtifact.run_id == rid)))
        assert not any("checkpoint-200" in a.path for a in files)
    assert archive_outputs(db, storage, rid, root, complete=True) == 1
    destination = tmp_path / "restored"
    assert restore_run(db, storage, rid, destination) == 3
    assert (destination / "RESTORE_VERIFIED").is_file()
    with db.transaction() as s:
        artifact = s.scalar(select(TrainingArtifact).where(TrainingArtifact.run_id == rid))
    storage.put(artifact.key, b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_run(db, storage, rid, tmp_path / "corrupt-restore")


def test_training_options_reach_pinned_launcher(monkeypatch, tmp_path):
    from img_creator.platform.app import TrainConfig
    from img_creator.platform.training_worker import training_command
    import training.train_lora

    captured = {}

    def fake_command(config, *args):
        captured.update(config)
        return ["python", "trainer.py", "--report_to", "none"]

    monkeypatch.setattr(training.train_lora, "command", fake_command)
    config = TrainConfig(dataset_id="a" * 32, resolution=768, lr_scheduler="linear", warmup_steps=20).model_dump()
    result = training_command(config, tmp_path, tmp_path, tmp_path)
    assert captured["resolution"] == 768
    assert captured["use_aspect_ratio_buckets"] is True
    assert captured["lr_scheduler"] == "linear"
    assert captured["lr_warmup_steps"] == 20
    assert captured["validation_prompt"] == config["validation_prompt"]
    assert result[-1] == "tensorboard"
    with pytest.raises(ValueError):
        TrainConfig(dataset_id="a" * 32, steps=100, warmup_steps=100)


def test_failed_training_keeps_workspace_and_archives_logs(platform, tmp_path, monkeypatch):
    import json
    import sys
    from img_creator.platform import training_worker
    from img_creator.platform.db import Dataset, TrainingArtifact

    db, storage = platform[1:3]
    with db.transaction() as s:
        d = Dataset(name="Failure recovery", rights_note="Owned source data")
        s.add(d)
        s.flush()
        r = TrainingRun(dataset_id=d.id, snapshot=[], config={"snapshot_sha256": hashlib.sha256(b"[]").hexdigest()})
        s.add(r)
        s.flush()
        rid = r.id
    monkeypatch.setattr(
        training_worker,
        "training_command",
        lambda config, checkout, dataset, output: [
            sys.executable,
            "-c",
            "print('diagnostic training failure'); raise SystemExit(1)",
        ],
    )
    monkeypatch.setattr(training_worker, "tensorboard_metrics", lambda _: {})
    with pytest.raises(RuntimeError, match="GPU training failed"):
        training_worker.train_one(db, storage, tmp_path, work_dir=tmp_path / "work")
    root = tmp_path / "work" / rid
    assert root.exists() and "diagnostic training failure" in (root / "train.log").read_text()
    assert json.loads((root / "snapshot.json").read_text()) == []
    with db.transaction() as s:
        assert s.get(TrainingRun, rid).status == "failed"
        files = list(s.scalars(select(TrainingArtifact).where(TrainingArtifact.run_id == rid)))
        assert {"train.log", "snapshot.json", "launch.json"}.issubset({f.path for f in files})


def test_adapter_strength_is_explicit_and_model_specific():
    req = Generation(
        prompt="A ceramic vase", model="custom-klein-4b", training_run="a" * 32, effort="high", adapter_strength=0.75
    )
    assert req.adapter_strength == 0.75
    with pytest.raises(ValueError):
        Generation(prompt="A vase", adapter_strength=0.75)
    with pytest.raises(ValueError):
        Generation(prompt="A vase", model="custom-klein-4b", training_run="a" * 32, effort="high", adapter_strength=2)
