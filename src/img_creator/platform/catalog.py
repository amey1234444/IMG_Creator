"""Versioned, server-owned capabilities; never accept provider URLs from users."""

import math
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

MODELS = {
    "custom-klein-4b": {
        "name": "Custom Klein 4B",
        "parameters_b": 4,
        "credits": 8,
        "efforts": ["low", "medium", "high"],
        "description": "Your approved studio-trained adapter",
    },
    "flux-2-dev-32b": {
        "name": "FLUX.2 Dev 32B",
        "parameters_b": 32,
        "credits": 20,
        "efforts": ["low", "medium", "high"],
        "description": "Self-hosted 32B model; licensed GPU deployment required",
    },
    "flux-2-klein-4b": {
        "name": "FLUX.2 Klein 4B",
        "parameters_b": 4,
        "credits": 2,
        "efforts": ["standard"],
        "description": "Quick concepts and everyday creation",
    },
    "flux-2-klein-9b": {
        "name": "FLUX.2 Klein 9B",
        "parameters_b": 9,
        "credits": 3,
        "efforts": ["standard"],
        "description": "More capacity with fast generation",
    },
    "flux-2-pro": {
        "name": "FLUX.2 Pro",
        "parameters_b": None,
        "credits": 8,
        "efforts": ["standard"],
        "description": "Detailed photography and prompt fidelity",
    },
    "flux-2-flex": {
        "name": "FLUX.2 Flex",
        "parameters_b": None,
        "credits": 12,
        "efforts": ["low", "medium", "high"],
        "description": "Adjustable sampling effort and fine detail",
    },
    "flux-2-max": {
        "name": "FLUX.2 Max",
        "parameters_b": None,
        "credits": 16,
        "efforts": ["standard"],
        "description": "Premium composition and image consistency",
    },
}
# Product credits, NOT vendor prices or tokens. Review margins before enabling paid plans.
EFFORTS = {"standard": 1, "low": 1, "medium": 1.5, "high": 2}
RATIOS = {"1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "3:2": (3, 2), "2:3": (2, 3), "4:3": (4, 3)}
STYLES = {
    "natural": "",
    "photographic": "Photograph, physically plausible lighting, natural textures, realistic material response.",
    "cinematic": "Cinematic photograph, intentional composition, subtle film grain, realistic lighting.",
    "product": "Studio product photograph, precise material textures, controlled lighting, clear subject.",
    "illustration": "Detailed editorial illustration, intentional shapes and color palette.",
}


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = "flux-2-pro"
    effort: Literal["standard", "low", "medium", "high"] = "standard"
    aspect_ratio: str = "1:1"
    resolution: Literal["1K", "2K", "4K", "8K"] = "1K"
    upscale: Literal["resize", "learned"] = "resize"
    style: Literal["natural", "photographic", "cinematic", "product", "illustration"] = "photographic"
    training_run: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)

    @model_validator(mode="after")
    def valid(self):
        self.prompt = self.prompt.strip()
        if not self.prompt or self.model not in MODELS or self.aspect_ratio not in RATIOS:
            raise ValueError("Invalid prompt, model or aspect ratio")
        if self.effort not in MODELS[self.model]["efforts"]:
            raise ValueError("This model does not expose that effort setting")
        if (self.model == "custom-klein-4b") != (self.training_run is not None):
            raise ValueError("Select an approved training run only for the custom model")
        return self

    def dimensions(self):
        a, b = RATIOS[self.aspect_ratio]
        edge = {"1K": 1024, "2K": 2048, "4K": 4096, "8K": 8192}[self.resolution]
        final = (round(edge * a / max(a, b)), round(edge * b / max(a, b)))
        # 2K square slightly exceeds 4 MP; stay within the provider's 4 MP bound.
        scale = min(1.0, math.sqrt(4_000_000 / (final[0] * final[1])), 2048 / max(final))
        native = tuple(max(64, math.floor(n * scale / 16) * 16) for n in final)
        return native, final

    def quote(self):
        native, final = self.dimensions()
        base = MODELS[self.model]["credits"]
        return math.ceil(base * EFFORTS[self.effort] * math.ceil(native[0] * native[1] / 1_000_000)) + (
            math.ceil(final[0] * final[1] / 1_000_000) if self.upscale == "learned" else 0
        )
