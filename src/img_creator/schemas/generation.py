from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from ..presets import PRESETS


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=4000)
    aspect_ratio: str = "1:1"
    quality: Literal["fast", "quality", "ultra"] = "fast"
    output_resolution: str = "Native"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    steps: int | None = Field(default=None, ge=1, le=100, strict=True)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    prompt_enhancement: bool = False
    style: Literal["neutral", "photographic", "illustration", "cinematic", "product", "watercolor"] = "neutral"
    num_images: int = Field(default=1, ge=1, le=4, strict=True)
    upscaler: Literal["lanczos", "learned"] = "lanczos"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    lora: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    lora_weight: float = Field(default=1.0, ge=0, le=2)

    @model_validator(mode="before")
    @classmethod
    def legacy(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Preserve starter requests that used quality as the export size.
        if data.get("quality") in {"Native", "Full HD", "1.5K", "2K", "3K", "4K", "5K", "8K"}:
            data.setdefault("output_resolution", data["quality"])
            data["quality"] = "fast"
        if "enhance_prompt" in data:
            data.setdefault("prompt_enhancement", data.pop("enhance_prompt"))
        value = data.get("output_resolution")
        if isinstance(value, str):
            names = {k.lower(): k for presets in PRESETS.values() for k in presets}
            data["output_resolution"] = names.get(value.lower(), value)
        return data

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value.strip()

    @model_validator(mode="after")
    def valid_combination(self):
        if self.aspect_ratio not in PRESETS or self.output_resolution not in PRESETS[self.aspect_ratio]:
            raise ValueError("Unsupported aspect ratio / output resolution combination")
        if self.quality == "ultra" and self.upscaler != "learned":
            raise ValueError("Ultra requires the learned upscaler and configured SR weights")
        return self


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    output_resolution: str = "4K"
    upscaler: Literal["lanczos", "learned"] = "learned"
    output_format: Literal["png", "jpeg", "webp"] = "png"
