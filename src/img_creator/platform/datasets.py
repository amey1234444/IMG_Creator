"""Bounded owner upload ingestion. Documents are references; only captioned images train."""

from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile
from PIL import Image, ImageOps, ImageFilter, ImageStat
from fastapi import HTTPException
from sqlalchemy import select
from .db import Asset, AssetSource, DatasetVersion, DatasetVersionItem, RunDataset, TrainingRun, uid

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
            normalized.info.clear()
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
    original = data
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
    s.flush()
    original_key = f"sources/{dataset_id}/{asset_id}/original"
    storage.put(original_key, original)
    quality = {}
    if kind == "image":
        with Image.open(BytesIO(data)) as image:
            sample = image.convert("L")
            sample.thumbnail((256, 256))
            variance = float(ImageStat.Stat(sample.filter(ImageFilter.FIND_EDGES)).var[0])
        quality = {"edge_variance": round(variance, 3), "flags": []}
        if variance < 100:
            quality["flags"].append("low_edge_variance_review")
        if min(size) < 768:
            quality["flags"].append("low_resolution_for_detail_training")
    s.add(
        AssetSource(
            asset_id=asset.id,
            original_key=original_key,
            original_sha256=hashlib.sha256(original).hexdigest(),
            original_bytes=len(original),
            group_id=asset.sha256,
            quality=quality,
        )
    )
    return asset


def snapshot_run(s, dataset_id, config):
    assets = s.scalars(
        select(Asset)
        .where(Asset.dataset_id == dataset_id, Asset.kind == "image", Asset.approved.is_(True))
        .order_by(Asset.id)
    ).all()
    if not 5 <= len(assets) <= 2000:
        raise HTTPException(422, "Pilot training needs 5–2000 approved, captioned images")
    sources = {
        a.asset_id: a for a in s.scalars(select(AssetSource).where(AssetSource.asset_id.in_([a.id for a in assets])))
    }
    groups = {a.id: sources[a.id].group_id if a.id in sources else a.sha256 for a in assets}
    unique_groups = set(groups.values())
    if len(unique_groups) < 2:
        raise HTTPException(422, "Use at least two independent subject/shoot groups for validation")
    split_seed = config.get("split_seed", 42)
    fraction = config.get("validation_fraction", 0.1)
    ordered = sorted(unique_groups, key=lambda g: hashlib.sha256(f"{split_seed}:{g}".encode()).hexdigest())
    held_out = set(ordered[: max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))])
    snapshot = [
        {
            "id": a.id,
            "key": a.key,
            "sha256": a.sha256,
            "caption": a.caption,
            "group_id": groups[a.id],
            "split": "validation" if groups[a.id] in held_out else "train",
        }
        for a in assets
    ]
    if any(not a["caption"].strip() for a in snapshot):
        raise HTTPException(422, "Every training image requires a caption")
    if sum(a["split"] == "train" for a in snapshot) < 2:
        raise HTTPException(422, "Validation grouping leaves fewer than two training images")
    split_settings = {"split_seed": split_seed, "validation_fraction": fraction, "algorithm": "ranked-groups-v1"}
    sha = hashlib.sha256(
        json.dumps({"items": snapshot, "settings": split_settings}, sort_keys=True).encode()
    ).hexdigest()
    version = s.scalar(
        select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id, DatasetVersion.sha256 == sha)
    )
    if not version:
        version = DatasetVersion(dataset_id=dataset_id, sha256=sha, settings=split_settings)
        s.add(version)
        s.flush()
        for item in snapshot:
            s.add(
                DatasetVersionItem(
                    version_id=version.id,
                    asset_id=item["id"],
                    **{k: item[k] for k in ("caption", "group_id", "split", "key", "sha256")},
                )
            )
    config = {
        **config,
        "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        "dataset_version_id": version.id,
    }
    run = TrainingRun(id=uid(), dataset_id=dataset_id, snapshot=snapshot, config=config)
    s.add(run)
    s.flush()
    s.add(RunDataset(run_id=run.id, version_id=version.id))
    return run
