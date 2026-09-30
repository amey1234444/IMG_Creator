from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
from uuid import uuid4


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def directory(self, generation_id):
        if not re.fullmatch(r"[0-9a-f]{32}", generation_id):
            raise ValueError("Invalid generation ID")
        path = self.root / generation_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("Invalid artifact path")
        return path

    def save(self, image, metadata, output_format):
        generation_id = uuid4().hex
        target = self.directory(generation_id)
        staging = self.root / f".{generation_id}.tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        metadata = {
            **metadata,
            "generation_id": generation_id,
            "output_format": output_format,
            "image_file": f"image.{output_format}",
        }
        try:
            kwargs = {"quality": 95} if output_format in ("jpeg", "webp") else {}
            image.convert("RGB").save(staging / metadata["image_file"], format=output_format.upper(), **kwargs)
            (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return metadata

    def read(self, generation_id):
        path = self.directory(generation_id) / "metadata.json"
        if path.is_symlink():
            raise ValueError("Invalid artifact path")
        return json.loads(path.read_text(encoding="utf-8"))

    def image_path(self, generation_id):
        data = self.read(generation_id)
        filename = data["image_file"]
        if filename not in {"image.png", "image.jpeg", "image.webp"}:
            raise ValueError("Invalid artifact filename")
        path = self.directory(generation_id) / filename
        if path.is_symlink():
            raise ValueError("Invalid artifact path")
        return path
