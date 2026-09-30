"""Export an indexed dataset as bounded tar shards or a small HF ImageFolder pilot."""

from __future__ import annotations
import argparse
import hashlib
import io
import json
import sqlite3
import shutil
import tarfile
from pathlib import Path
from PIL import Image, ImageOps


def export(index: Path, output: Path, split="train", shard_size=1000, pilot_limit=None, shard_bytes=1_000_000_000):
    if split not in {"train", "validation"} or not 1 <= shard_size <= 10000:
        raise ValueError("Invalid split or shard size")
    if pilot_limit is not None and not 1 <= pilot_limit <= 2000:
        raise ValueError("Pilot size must be between 1 and 2000")
    if shard_bytes < 1024:
        raise ValueError("Shard byte limit must be at least 1024")
    db = sqlite3.connect(f"{index.resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    output.mkdir(parents=True, exist_ok=False)
    tar = None
    manifest = None
    count = 0
    shard_index = -1
    shard_count = 0
    current_bytes = 0
    completed = False
    try:
        query = "SELECT * FROM samples WHERE split=? ORDER BY id"
        if pilot_limit:
            query += " LIMIT ?"
        rows = db.execute(query, (split, pilot_limit) if pilot_limit else (split,))
        if pilot_limit:
            manifest = (output / "metadata.jsonl").open("w", encoding="utf-8")
        for row in rows:
            path = Path(row["path"])
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"Image changed since indexing: sample {row['id']}")
            key = f"{row['id']:09d}"
            with Image.open(path) as im:
                # Normalize orientation and remove EXIF/GPS; preserve aspect and original pixels.
                image = ImageOps.exif_transpose(im).convert("RGB")
                image.info.clear()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            record = dict(row)
            record.pop("path")
            if manifest:
                (output / f"{key}.png").write_bytes(image_bytes)
                manifest.write(json.dumps({"file_name": f"{key}.png", "text": row["caption"]}) + "\n")
            else:
                if tar is None or shard_count >= shard_size or current_bytes + len(image_bytes) > shard_bytes:
                    if tar:
                        tar.close()
                    shard_index += 1
                    shard_count = 0
                    current_bytes = 0
                    tar = tarfile.open(output / f"{split}-{shard_index:06d}.tar", "w")
                for suffix, data in [
                    ("png", image_bytes),
                    ("txt", row["caption"].encode()),
                    ("json", json.dumps(record).encode()),
                ]:
                    info = tarfile.TarInfo(f"{key}.{suffix}")
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                    current_bytes += ((len(data) + 511) // 512 + 1) * 512
                shard_count += 1
            count += 1
        if count == 0:
            raise ValueError(f"No {split} samples available")
        # Hidden summary keeps ImageFolder inference focused on images + metadata.jsonl.
        (output / (".export.json" if pilot_limit else "export.json")).write_text(
            json.dumps(
                {
                    "count": count,
                    "split": split,
                    "format": "imagefolder" if pilot_limit else "webdataset",
                    "shard_size": None if pilot_limit else shard_size,
                },
                indent=2,
            )
        )
        completed = True
        return count
    finally:
        if tar:
            tar.close()
        if manifest:
            manifest.close()
        db.close()
        if not completed:
            shutil.rmtree(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--pilot-limit", type=int)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    print(
        f"Exported {export(args.index, args.output, args.split, args.shard_size, args.pilot_limit, args.shard_bytes)} images"
    )


if __name__ == "__main__":
    main()
