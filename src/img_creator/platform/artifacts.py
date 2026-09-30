"""Persist trusted trainer outputs with content hashes and bounded-memory uploads."""

import hashlib
from sqlalchemy import select
from .db import TrainingArtifact


def archive_outputs(db, storage, run_id, root, complete=False):
    checkpoints = sorted(
        (root / "output").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    # A subsequent checkpoint proves that the previous save returned. The current
    # directory may be incomplete if training is running or failed during a save.
    candidates = checkpoints if complete else checkpoints[:-1]
    files = [p for name in ("snapshot.json", "launch.json", "train.log") if (p := root / name).is_file()]
    for folder in candidates:
        files.extend(p for p in folder.rglob("*") if p.is_file())
    if complete:
        files.extend(
            p
            for p in (root / "output").rglob("*")
            if p.is_file() and not any(part.startswith("checkpoint-") for part in p.relative_to(root / "output").parts)
        )
    archived = 0
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Trainer artifact escapes workspace")
        relative = path.relative_to(root).as_posix()
        if len(relative) > 400:
            raise ValueError("Artifact path too long")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        size = path.stat().st_size
        with db.transaction() as s:
            previous = s.scalar(
                select(TrainingArtifact).where(TrainingArtifact.run_id == run_id, TrainingArtifact.path == relative)
            )
            if previous and previous.sha256 == sha:
                continue
        key = f"training/{run_id}/files/{sha}/{relative}"
        storage.put_file(key, path)
        # Only stopped files/checkpoints should be archived. Verify against any
        # accidental concurrent modification before recording an artifact.
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise ValueError("Artifact changed during archival")
        with db.transaction() as s:
            previous = s.scalar(
                select(TrainingArtifact).where(TrainingArtifact.run_id == run_id, TrainingArtifact.path == relative)
            )
            if previous:
                previous.key, previous.sha256, previous.size_bytes = key, sha, size
            else:
                s.add(TrainingArtifact(run_id=run_id, path=relative, key=key, sha256=sha, size_bytes=size))
        archived += 1
    return archived


def restore_run(db, storage, run_id, destination):
    """Export a run to a new directory; verify all files before the operator uses it."""
    import json
    from pathlib import Path
    from .db import TrainingRun

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with db.transaction() as s:
        run = s.get(TrainingRun, run_id)
        if not run:
            raise ValueError("Training run not found")
        files = list(s.scalars(select(TrainingArtifact).where(TrainingArtifact.run_id == run_id)))
    if not files:
        raise ValueError("No archived artifacts; check the persistent GPU workspace")
    for artifact in files:
        target = destination / artifact.path
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Artifact path escapes restore directory")
        storage.get_file(artifact.key, target)
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != artifact.sha256:
                raise ValueError("Archived artifact checksum mismatch")
    dataset = destination / "dataset"
    dataset.mkdir(exist_ok=True)
    manifest = []
    for item in run.snapshot:
        if item["split"] != "train":
            continue
        filename = item["id"] + ".png"
        target = dataset / filename
        if not target.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Invalid snapshot filename")
        storage.get_file(item["key"], target)
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError("Dataset checksum mismatch")
        manifest.append({"file_name": filename, "text": item["caption"]})
    (dataset / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in manifest) + "\n")
    (destination / "RESTORE_VERIFIED").write_text(run_id + "\n")
    return len(files)
