from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from .presets import PRESETS as PRESETS, ResolutionPreset as ResolutionPreset

load_dotenv()


@dataclass(frozen=True)
class Settings:
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("IMG_CREATOR_OUTPUT_DIR", "outputs")))
    device: str = field(default_factory=lambda: os.getenv("IMG_CREATOR_DEVICE", "auto").lower())
    cpu_offload: bool = field(default_factory=lambda: os.getenv("IMG_CREATOR_CPU_OFFLOAD", "true").lower() == "true")
    model_revision: str | None = field(default_factory=lambda: os.getenv("IMG_CREATOR_MODEL_REVISION") or None)
    lora_registry: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["IMG_CREATOR_LORA_REGISTRY"]) if os.getenv("IMG_CREATOR_LORA_REGISTRY") else None
        )
    )
    sr_weights: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["IMG_CREATOR_SR_WEIGHTS"]) if os.getenv("IMG_CREATOR_SR_WEIGHTS") else None
        )
    )
    tile_size: int = field(default_factory=lambda: int(os.getenv("IMG_CREATOR_TILE_SIZE", "256")))
    api_key: str | None = field(default_factory=lambda: os.getenv("IMG_CREATOR_API_KEY") or None)

    def __post_init__(self):
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("IMG_CREATOR_DEVICE must be auto, cuda, mps, or cpu")
        if not 64 <= self.tile_size <= 1024:
            raise ValueError("IMG_CREATOR_TILE_SIZE must be between 64 and 1024")


OUTPUT_DIR = Path(os.getenv("IMG_CREATOR_OUTPUT_DIR", "outputs"))
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
