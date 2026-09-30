import hashlib
import json
import secrets
import time
from fastapi import HTTPException
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError
from .db import Job, Ledger, User, uid


def submit(db, user_id, request, idempotency_key):
    data = request.model_dump()
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    try:
        with db.transaction() as s:
            # Serializes per-account mutations in Postgres. Conditional debit also protects SQLite.
            user = s.scalar(select(User).where(User.id == user_id).with_for_update())
            existing = s.scalar(select(Job).where(Job.user_id == user_id, Job.idempotency_key == idempotency_key))
            if existing:
                if existing.request_hash != fingerprint:
                    raise HTTPException(409, "Idempotency key was already used with different settings")
                return existing
            if not user.enabled:
                raise HTTPException(403, "Account is disabled")
            outstanding = s.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.user_id == user_id, Job.status.in_(["queued", "running", "uncertain"]))
            )
            if outstanding >= 3:
                raise HTTPException(429, "Wait for an existing generation to complete")
            charge = request.quote()
            if not s.execute(
                update(User).where(User.id == user_id, User.credits >= charge).values(credits=User.credits - charge)
            ).rowcount:
                raise HTTPException(402, "Not enough credits. Choose a plan or ask the owner for credits.")
            if data["seed"] is None:
                data["seed"] = secrets.randbits(32)
            job = Job(
                id=uid(),
                user_id=user_id,
                idempotency_key=idempotency_key,
                request_hash=fingerprint,
                request=data,
                credits=charge,
            )
            s.add(job)
            s.add(Ledger(user_id=user_id, delta=-charge, reason="generation_reserved", reference="reserve:" + job.id))
            return job
    except IntegrityError:
        with db.transaction() as s:
            existing = s.scalar(select(Job).where(Job.user_id == user_id, Job.idempotency_key == idempotency_key))
            if not existing:
                raise
            if existing.request_hash != fingerprint:
                raise HTTPException(409, "Idempotency key conflict")
            return existing


def claim(db, models=None):
    with db.transaction() as s:
        query = select(Job).where(Job.status == "queued")
        if models is not None:
            query = query.where(Job.request["model"].as_string().in_(models))
        job = s.scalar(query.order_by(Job.created).with_for_update(skip_locked=True).limit(1))
        if not job:
            return None
        token = uid()
        if not s.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == "queued")
            .values(status="running", lease_token=token, lease_until=time.time() + 120, started=time.time())
        ).rowcount:
            return None
        s.refresh(job)
        return job


def heartbeat(db, job_id, token, provider_state=None):
    with db.transaction() as s:
        values = {"lease_until": time.time() + 120}
        if provider_state is not None:
            values["provider_state"] = provider_state
        if not s.execute(
            update(Job).where(Job.id == job_id, Job.status == "running", Job.lease_token == token).values(**values)
        ).rowcount:
            raise RuntimeError("Worker lease lost")


def finish(db, job_id, token, result=None, error=None, uncertain=False):
    with db.transaction() as s:
        job = s.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if not job or job.status != "running" or job.lease_token != token:
            return False
        job.status = "uncertain" if uncertain else ("failed" if error else "succeeded")
        job.result, job.error, job.finished = result, error, time.time()
        if error and not uncertain:
            s.execute(update(User).where(User.id == job.user_id).values(credits=User.credits + job.credits))
            s.add(
                Ledger(user_id=job.user_id, delta=job.credits, reason="generation_refund", reference="refund:" + job.id)
            )
        return True


def recover_stale(db):
    # Never blindly retry paid provider submissions: a timeout may have created a billable image.
    with db.transaction() as s:
        s.execute(
            update(Job)
            .where(Job.status == "running", Job.lease_until < time.time())
            .values(
                status="uncertain", error="Worker interrupted. Owner reconciliation required.", finished=time.time()
            )
        )


def serialize(job):
    return {
        "id": job.id,
        "status": job.status,
        "credits": job.credits,
        "request": job.request,
        "result": job.result,
        "error": job.error,
        "created": job.created,
        "finished": job.finished,
    }
