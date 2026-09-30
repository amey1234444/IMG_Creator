from __future__ import annotations
from contextlib import contextmanager
import time
import uuid
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def uid():
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True, default=uid)
    email = Column(String(254), unique=True, nullable=False)
    password = Column(Text, nullable=False)
    role = Column(String(20), default="user", nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    credits = Column(Integer, default=0, nullable=False)
    stripe_customer = Column(String(100), unique=True)
    subscription = Column(String(100))
    plan = Column(String(32), default="free", nullable=False)
    subscription_status = Column(String(40), default="none")
    billing_updated = Column(Integer, default=0)
    created = Column(Float, default=time.time)


class Session(Base):
    __tablename__ = "sessions"
    token = Column(String(64), primary_key=True)
    user_id = Column(ForeignKey("users.id"), nullable=False, index=True)
    csrf = Column(String(64), nullable=False)
    expires = Column(Float, nullable=False, index=True)


class RateLimit(Base):
    __tablename__ = "rate_limits"
    key = Column(String(100), primary_key=True)
    count = Column(Integer, nullable=False, default=0)
    expires = Column(Float, nullable=False, index=True)


class Ledger(Base):
    __tablename__ = "credit_ledger"
    id = Column(String(32), primary_key=True, default=uid)
    user_id = Column(ForeignKey("users.id"), nullable=False, index=True)
    delta = Column(Integer, nullable=False)
    reason = Column(String(100), nullable=False)
    reference = Column(String(200), nullable=False, unique=True)
    created = Column(Float, default=time.time)


class Job(Base):
    __tablename__ = "generation_jobs"
    __table_args__ = (UniqueConstraint("user_id", "idempotency_key"),)
    id = Column(String(32), primary_key=True, default=uid)
    user_id = Column(ForeignKey("users.id"), nullable=False, index=True)
    idempotency_key = Column(String(100), nullable=False)
    request_hash = Column(String(64), nullable=False)
    request = Column(JSON, nullable=False)
    status = Column(String(30), default="queued", nullable=False, index=True)
    credits = Column(Integer, nullable=False)
    result = Column(JSON)
    provider_state = Column(JSON)
    error = Column(Text)
    lease_token = Column(String(32))
    lease_until = Column(Float, default=0, index=True)
    created = Column(Float, default=time.time, index=True)
    started = Column(Float)
    finished = Column(Float)


class BillingCheckout(Base):
    __tablename__ = "billing_checkouts"
    user_id = Column(ForeignKey("users.id"), primary_key=True)
    session_id = Column(String(100), nullable=False)
    plan = Column(String(32), nullable=False)


class BillingEvent(Base):
    __tablename__ = "billing_events"
    id = Column(String(100), primary_key=True)
    type = Column(String(100), nullable=False)
    created = Column(Float, default=time.time)


class Payment(Base):
    __tablename__ = "payments"
    invoice = Column(String(100), primary_key=True)
    user_id = Column(ForeignKey("users.id"), nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String(10), nullable=False)
    created = Column(Float, default=time.time)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(32), primary_key=True, default=uid)
    name = Column(String(100), nullable=False)
    rights_note = Column(Text, nullable=False)
    created = Column(Float, default=time.time)


class Asset(Base):
    __tablename__ = "dataset_assets"
    __table_args__ = (UniqueConstraint("dataset_id", "sha256"),)
    id = Column(String(32), primary_key=True, default=uid)
    dataset_id = Column(ForeignKey("datasets.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    key = Column(String(200), nullable=False)
    sha256 = Column(String(64), nullable=False)
    kind = Column(String(20), nullable=False)
    caption = Column(Text, default="")
    text = Column(Text, default="")
    width = Column(Integer)
    height = Column(Integer)
    approved = Column(Boolean, default=False, nullable=False)
    created = Column(Float, default=time.time)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String(32), primary_key=True, default=uid)
    dataset_id = Column(ForeignKey("datasets.id"), nullable=False)
    status = Column(String(30), default="queued", nullable=False, index=True)
    config = Column(JSON, nullable=False)
    snapshot = Column(JSON, nullable=False)
    metrics = Column(JSON, default=dict)
    evaluation = Column(JSON)
    artifact_key = Column(String(200))
    artifact_sha256 = Column(String(64))
    error = Column(Text)
    lease_until = Column(Float, default=0)
    lease_token = Column(String(32))
    created = Column(Float, default=time.time)
    finished = Column(Float)


class Audit(Base):
    __tablename__ = "audit_events"
    id = Column(String(32), primary_key=True, default=uid)
    actor = Column(String(32), nullable=False)
    action = Column(String(60), nullable=False)
    target = Column(String(100), nullable=False)
    detail = Column(JSON, default=dict)
    created = Column(Float, default=time.time)


class Database:
    def __init__(self, url):
        kwargs = {"connect_args": {"check_same_thread": False, "timeout": 30}} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def configure(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA journal_mode=WAL")

        self.factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self):
        # Initial schema only. Future schema revisions require an explicit migration.
        Base.metadata.create_all(self.engine)

    @contextmanager
    def transaction(self):
        with self.factory.begin() as session:
            yield session
