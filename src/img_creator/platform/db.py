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


class SchemaRevision(Base):
    __tablename__ = "schema_revisions"
    version = Column(Integer, primary_key=True)
    applied = Column(Float, default=time.time, nullable=False)


class AssetSource(Base):
    __tablename__ = "asset_sources"
    asset_id = Column(ForeignKey("dataset_assets.id"), primary_key=True)
    original_key = Column(String(250), nullable=False)
    original_sha256 = Column(String(64), nullable=False)
    original_bytes = Column(Integer, nullable=False)
    group_id = Column(String(120), nullable=False, index=True)
    quality = Column(JSON, nullable=False, default=dict)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "sha256"),)
    id = Column(String(32), primary_key=True, default=uid)
    dataset_id = Column(ForeignKey("datasets.id"), nullable=False, index=True)
    sha256 = Column(String(64), nullable=False)
    settings = Column(JSON, nullable=False)
    created = Column(Float, default=time.time, nullable=False)


class DatasetVersionItem(Base):
    __tablename__ = "dataset_version_items"
    version_id = Column(ForeignKey("dataset_versions.id"), primary_key=True)
    asset_id = Column(ForeignKey("dataset_assets.id"), primary_key=True)
    caption = Column(Text, nullable=False)
    group_id = Column(String(120), nullable=False)
    split = Column(String(20), nullable=False)
    key = Column(String(250), nullable=False)
    sha256 = Column(String(64), nullable=False)


class RunDataset(Base):
    __tablename__ = "run_datasets"
    run_id = Column(ForeignKey("training_runs.id"), primary_key=True)
    version_id = Column(ForeignKey("dataset_versions.id"), nullable=False, index=True)


class TrainingArtifact(Base):
    __tablename__ = "training_artifacts"
    __table_args__ = (UniqueConstraint("run_id", "path"),)
    id = Column(String(32), primary_key=True, default=uid)
    run_id = Column(ForeignKey("training_runs.id"), nullable=False, index=True)
    path = Column(String(500), nullable=False)
    key = Column(String(600), nullable=False)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    created = Column(Float, default=time.time, nullable=False)


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
        # Revision 2 is strictly additive: existing tables and data are not rewritten.
        # Only the operator's pre-deploy command runs this; workers never migrate.
        from sqlalchemy import select

        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                from sqlalchemy import text

                connection.execute(text("SELECT pg_advisory_xact_lock(4782202)"))
            Base.metadata.create_all(connection)
            if connection.scalar(select(SchemaRevision.version).where(SchemaRevision.version == 2)) is None:
                connection.execute(SchemaRevision.__table__.insert().values(version=2, applied=time.time()))

    @contextmanager
    def transaction(self):
        with self.factory.begin() as session:
            yield session
