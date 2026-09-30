"""Model-friendly generation sizes and independently chosen export dimensions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionPreset:
    label: str
    base_size: tuple[int, int]
    final_size: tuple[int, int]


_SIZES = {
    "1:1": (
        (1024, 1024),
        {"Native": (1024, 1024), "1.5K": (1536, 1536), "2K": (2048, 2048), "3K": (3072, 3072), "4K": (4096, 4096)},
    ),
    "16:9": (
        (1344, 768),
        {
            "Native": (1344, 768),
            "Full HD": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160),
            "5K": (5120, 2880),
            "8K": (7680, 4320),
        },
    ),
    "9:16": (
        (768, 1344),
        {
            "Native": (768, 1344),
            "Full HD": (1080, 1920),
            "2K": (1440, 2560),
            "4K": (2160, 3840),
            "5K": (2880, 5120),
            "8K": (4320, 7680),
        },
    ),
    "3:2": (
        (1216, 832),
        {"Native": (1216, 832), "1.5K": (1536, 1024), "2K": (2048, 1365), "3K": (3072, 2048), "4K": (4096, 2731)},
    ),
    "2:3": (
        (832, 1216),
        {"Native": (832, 1216), "1.5K": (1024, 1536), "2K": (1365, 2048), "3K": (2048, 3072), "4K": (2731, 4096)},
    ),
    "4:3": (
        (1152, 896),
        {"Native": (1152, 896), "Full HD": (1600, 1200), "2K": (2048, 1536), "3K": (3200, 2400), "4K": (4096, 3072)},
    ),
}
PRESETS = {
    ratio: {name: ResolutionPreset(name, base, size) for name, size in sizes.items()}
    for ratio, (base, sizes) in _SIZES.items()
}
PROFILES = {
    "fast": {"model": "black-forest-labs/FLUX.2-klein-4B", "steps": 4, "guidance": 1.0},
    "quality": {"model": "black-forest-labs/FLUX.2-klein-base-4B", "steps": 50, "guidance": 4.0},
    "ultra": {"model": "black-forest-labs/FLUX.2-klein-base-4B", "steps": 50, "guidance": 4.0},
}
