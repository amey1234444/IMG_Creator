"""Stream caption rows into a disk-backed, deduplicated image inventory."""

from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
import warnings
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageStat
from training.buckets import choose_bucket


def prepare(images: Path, metadata: Path, output: Path, min_side=512, validation_fraction=0.05):
    if not 0 < validation_fraction < 0.5 or min_side < 1:
        raise ValueError("Invalid minimum size or validation fraction")
    images = images.resolve()
    # A new output directory prevents accidental replacement of a reviewed dataset version.
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output / "index.sqlite3")
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE samples (
      id INTEGER PRIMARY KEY, path TEXT NOT NULL, caption TEXT NOT NULL,
      sha256 TEXT UNIQUE NOT NULL, pixel_sha256 TEXT UNIQUE NOT NULL,
      width INTEGER, height INTEGER, bucket_width INTEGER, bucket_height INTEGER,
      split TEXT, group_id TEXT, license TEXT, source TEXT, edge_variance REAL, review_flags TEXT);
    CREATE INDEX sample_split ON samples(split);
    """)
    counts = Counter()
    try:
        with (
            metadata.open(encoding="utf-8") as rows,
            (output / "rejected.jsonl").open("w", encoding="utf-8") as rejected,
        ):
            for line_no, line in enumerate(rows, 1):
                if not line.strip():
                    continue
                counts["total"] += 1
                try:
                    if len(line) > 1_000_000:
                        raise ValueError("Metadata row exceeds 1 MB")
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Metadata row must be an object")
                    for key in ("file_name", "text", "license", "source"):
                        if not isinstance(row.get(key), str) or not row[key].strip():
                            raise ValueError(f"Missing {key}")
                    if len(row["text"]) > 4000:
                        raise ValueError("Caption exceeds 4000 characters")
                    path = (images / row["file_name"]).resolve()
                    if not path.is_relative_to(images) or not path.is_file():
                        raise ValueError("Image path escapes dataset root or is missing")
                    with path.open("rb") as stream:
                        sha = hashlib.file_digest(stream, "sha256").hexdigest()
                    if db.execute("SELECT 1 FROM samples WHERE sha256=?", (sha,)).fetchone():
                        raise ValueError("Duplicate file")
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(path) as source:
                            if getattr(source, "n_frames", 1) != 1:
                                raise ValueError("Animated or multi-frame images require manual conversion")
                            source.load()
                            image = ImageOps.exif_transpose(source).convert("RGB")
                    width, height = image.size
                    if min(width, height) < min_side:
                        raise ValueError("Image below minimum dimensions")
                    pixel_sha = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
                    if db.execute("SELECT 1 FROM samples WHERE pixel_sha256=?", (pixel_sha,)).fetchone():
                        raise ValueError("Duplicate decoded image")
                    sample = image.convert("L")
                    sample.thumbnail((256, 256))
                    edge = float(ImageStat.Stat(sample.filter(ImageFilter.FIND_EDGES)).var[0])
                    flags = ["low_edge_variance_review"] if edge < 100 else []
                    # Supply group_id for related frames, subjects, or near duplicates to prevent split leakage.
                    group = row.get("group_id") or pixel_sha
                    if not isinstance(group, str):
                        raise ValueError("group_id must be a string")
                    fraction = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 2**32
                    split = "validation" if fraction < validation_fraction else "train"
                    bw, bh = choose_bucket(width, height)
                    db.execute(
                        "INSERT INTO samples(path,caption,sha256,pixel_sha256,width,height,bucket_width,bucket_height,split,group_id,license,source,edge_variance,review_flags) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            str(path),
                            row["text"].strip(),
                            sha,
                            pixel_sha,
                            width,
                            height,
                            bw,
                            bh,
                            split,
                            group,
                            row["license"],
                            row["source"],
                            edge,
                            json.dumps(flags),
                        ),
                    )
                    counts["accepted"] += 1
                    counts[split] += 1
                    if flags:
                        counts["flagged_for_review"] += 1
                except (
                    ValueError,
                    TypeError,
                    OSError,
                    Image.DecompressionBombError,
                    Image.DecompressionBombWarning,
                ) as exc:
                    counts["rejected"] += 1
                    rejected.write(json.dumps({"line": line_no, "reason": str(exc)}) + "\n")
                if line_no % 1000 == 0:
                    db.commit()
            db.commit()
        report = {
            "schema_version": 1,
            "counts": dict(counts),
            "min_side": min_side,
            "validation_fraction": validation_fraction,
            "images_root": str(images),
            "manifest": str(metadata.resolve()),
            "notes": [
                "Exact file and decoded-pixel deduplication only.",
                "Review blur flags, captions, provenance, watermarks and content manually.",
                "Use group_id for related images before splitting.",
            ],
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.images, args.metadata, args.output, args.min_side, args.validation_fraction), indent=2)
    )


if __name__ == "__main__":
    main()
