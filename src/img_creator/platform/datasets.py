"""Bounded owner upload ingestion. Documents are references; only captioned images train."""

from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile
from PIL import Image, ImageOps
from fastapi import HTTPException
from sqlalchemy import select
from .db import Asset, TrainingRun, uid

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTS = {".pdf", ".docx", ".txt", ".md", ".csv", ".jsonl"}


def decode_upload(filename, data):
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTS:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 20_000_000 or min(image.size) < 256:
                raise ValueError("Training images must be at least 256px per side and at most 20 MP")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Animated and multipage images must be split into individual images")
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            output = BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue(), "image", "", normalized.size
    if suffix not in DOCUMENT_EXTS:
        raise ValueError("Supported: PNG, JPEG, WebP, BMP, TIFF, PDF, DOCX, TXT, MD, CSV and JSONL")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError("PDF must be unencrypted and at most 100 pages")
        text = "\n".join((page.extract_text() or "")[:20000] for page in reader.pages)
    elif suffix == ".docx":
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 50 * 1024 * 1024:
                raise ValueError("DOCX expands beyond 50 MB")
            if len(archive.infolist()) > 1000:
                raise ValueError("DOCX contains too many parts")
        from docx import Document

        text = "\n".join(p.text for p in Document(BytesIO(data)).paragraphs)
    else:
        text = data.decode("utf-8-sig")
    if len(text) > 1_000_000:
        raise ValueError("Document text exceeds one million characters")
    if not text.strip():
        raise ValueError("No text extracted. Scanned documents require OCR before upload")
    return data, "document", text, (None, None)


def ingest(s, storage, dataset_id, filename, data):
    data, kind, text, size = decode_upload(filename, data)
    sha = hashlib.sha256(data).hexdigest()
    if s.scalar(select(Asset).where(Asset.dataset_id == dataset_id, Asset.sha256 == sha)):
        raise HTTPException(409, "Duplicate asset in this dataset")
    asset_id = uid()
    key = f"datasets/{dataset_id}/{asset_id}{'.png' if kind == 'image' else Path(filename).suffix.lower()}"
    storage.put(key, data)
    asset = Asset(
        id=asset_id,
        dataset_id=dataset_id,
        filename=Path(filename).name[:255],
        key=key,
        sha256=sha,
        kind=kind,
        text=text,
        width=size[0],
        height=size[1],
    )
    s.add(asset)
    return asset


def snapshot_run(s, dataset_id, config):
    assets = s.scalars(
        select(Asset)
        .where(Asset.dataset_id == dataset_id, Asset.kind == "image", Asset.approved.is_(True))
        .order_by(Asset.id)
    ).all()
    if not 5 <= len(assets) <= 2000:
        raise HTTPException(422, "Pilot training needs 5–2000 approved, captioned images")
    snapshot = [
        {
            "id": a.id,
            "key": a.key,
            "sha256": a.sha256,
            "caption": a.caption,
            "split": "validation" if i % 10 == 0 else "train",
        }
        for i, a in enumerate(assets)
    ]
    if any(not a["caption"].strip() for a in snapshot):
        raise HTTPException(422, "Every training image requires a caption")
    config = {**config, "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()}
    run = TrainingRun(id=uid(), dataset_id=dataset_id, snapshot=snapshot, config=config)
    s.add(run)
    return run
