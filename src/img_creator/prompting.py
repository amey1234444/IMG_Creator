from __future__ import annotations

STYLES = {
    "neutral": "clear composition, coherent perspective, carefully resolved detail",
    "photographic": "photorealistic, physically plausible materials, natural lighting, realistic shadows, fine surface texture, accurate perspective",
    "illustration": "expressive illustration, deliberate linework, coherent shapes, balanced color palette",
    "cinematic": "cinematic composition, motivated lighting, controlled depth of field, rich material detail",
    "product": "professional product photography, precise edges, realistic material response, controlled studio lighting",
    "watercolor": "watercolor on textured paper, translucent washes, expressive brushwork, soft pigment transitions",
}


def enhance_prompt(prompt: str, enabled: bool, style: str = "photographic") -> str:
    prompt = " ".join(prompt.strip().split())
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if style not in STYLES:
        raise ValueError("Unknown style")
    return f"{prompt}. {STYLES[style]}." if enabled else prompt
