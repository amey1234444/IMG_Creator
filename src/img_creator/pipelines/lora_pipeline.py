from __future__ import annotations
import hashlib
import json
from pathlib import Path


class LoraRegistry:
    """Administrator-configured adapters; requests can select IDs, never paths or URLs."""

    def __init__(self, path: Path | None):
        self.entries = json.loads(path.read_text()) if path else {}
        if not isinstance(self.entries, dict):
            raise ValueError("LoRA registry must be a JSON object")
        self.root = path.resolve().parent if path else Path.cwd()

    def list(self):
        return [{"id": key, "model": value["model"]} for key, value in self.entries.items()]

    def resolve(self, name, model):
        if name is None:
            return None
        if name not in self.entries:
            raise ValueError("Unknown LoRA ID; configure it in the server registry")
        entry = self.entries[name]
        if entry["model"] != model:
            raise ValueError("LoRA was registered for a different model")
        path = (self.root / entry["path"]).resolve()
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("LoRA must be an existing .safetensors file")
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return {"id": name, "path": path, "sha256": digest}

    @staticmethod
    def apply(pipe, adapter, weight):
        if adapter:
            pipe.load_lora_weights(
                str(adapter["path"].parent), weight_name=adapter["path"].name, adapter_name="selected"
            )
            pipe.set_adapters(["selected"], adapter_weights=[weight])
