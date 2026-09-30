from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    import torch

    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if preference not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Unsupported device")
    if preference == "cuda" and not cuda or preference == "mps" and not mps:
        raise RuntimeError(f"Requested {preference} device is unavailable")
    return ("cuda" if cuda else "mps" if mps else "cpu") if preference == "auto" else preference


def preferred_dtype(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Conservative MPS precision: hardware/runtime-specific half support varies.
    return torch.float32
