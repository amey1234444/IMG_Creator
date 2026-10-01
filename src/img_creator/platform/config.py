from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class PlatformSettings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///platform.db"))
    public_url: str = field(default_factory=lambda: os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/"))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("PLATFORM_STORAGE_DIR", "platform-data")))
    secure_cookies: bool = field(default_factory=lambda: os.getenv("COOKIE_SECURE", "true").lower() == "true")
    bfl_key: str = field(default_factory=lambda: os.getenv("BFL_API_KEY", ""))
    stripe_key: str = field(default_factory=lambda: os.getenv("STRIPE_SECRET_KEY", ""))
    stripe_webhook_secret: str = field(default_factory=lambda: os.getenv("STRIPE_WEBHOOK_SECRET", ""))
    starter_price: str = field(default_factory=lambda: os.getenv("STRIPE_STARTER_PRICE_ID", ""))
    pro_price: str = field(default_factory=lambda: os.getenv("STRIPE_PRO_PRICE_ID", ""))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_endpoint: str | None = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL") or None)
    enabled_models: str = field(
        default_factory=lambda: os.getenv(
            "ENABLED_MODELS", "flux-2-klein-4b,flux-2-klein-9b,flux-2-pro,flux-2-flex,flux-2-max"
        )
    )
    local_models: bool = field(default_factory=lambda: os.getenv("LOCAL_MODELS_ENABLED", "false").lower() == "true")
    moderation_key: str = field(default_factory=lambda: os.getenv("OPENAI_MODERATION_KEY", ""))
    dev_license: bool = field(
        default_factory=lambda: os.getenv("FLUX_DEV_COMMERCIAL_LICENSE", "false").lower() == "true"
    )
    expose_api_docs: bool = field(default_factory=lambda: os.getenv("EXPOSE_API_DOCS", "false").lower() == "true")
    api_requests_per_minute: int = field(default_factory=lambda: int(os.getenv("API_REQUESTS_PER_MINUTE", "300")))
    training_work_dir: Path = field(default_factory=lambda: Path(os.getenv("TRAINING_WORK_DIR", "training-work")))
    vision_enabled: bool = field(
        default_factory=lambda: os.getenv("VISION_ANALYSIS_ENABLED", "false").lower() == "true"
    )
    vision_device: str = field(default_factory=lambda: os.getenv("VISION_DEVICE", "cuda"))
    upload_limit: int = 25 * 1024 * 1024

    def __post_init__(self):
        if self.vision_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Invalid VISION_DEVICE")
        if not 10 <= self.api_requests_per_minute <= 10000:
            raise ValueError("API_REQUESTS_PER_MINUTE must be between 10 and 10000")
        if self.database_url.startswith("postgres://"):
            object.__setattr__(
                self, "database_url", self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
            )
        elif self.database_url.startswith("postgresql://"):
            object.__setattr__(
                self, "database_url", self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
            )
        if self.public_url.startswith("https://") and not self.secure_cookies:
            raise ValueError("HTTPS deployments require secure cookies")

    @property
    def plans(self):
        return {
            "starter": {"credits": 500, "price_id": self.starter_price},
            "pro": {"credits": 2000, "price_id": self.pro_price},
        }
