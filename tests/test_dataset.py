import json
import sqlite3
import tarfile
import pytest
from PIL import Image
from training.prepare_dataset import prepare
from training.export_dataset import export
from training.buckets import choose_bucket


def test_dataset_rejects_bad_rows_and_preserves_groups(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 32), "red").save(images / "a.png")
    (images / "duplicate.png").write_bytes((images / "a.png").read_bytes())
    Image.new("RGB", (64, 32), "blue").save(images / "b.png")
    Image.new("RGB", (8, 8), "black").save(images / "small.png")
    (images / "broken.png").write_text("not an image")
    base = {"text": "Factory motor", "license": "owned", "source": "internal camera", "group_id": "same-session"}
    rows = [
        dict(base, file_name=name)
        for name in ["a.png", "b.png", "duplicate.png", "small.png", "broken.png", "../outside.png"]
    ]
    rows.append({"file_name": "a.png", "text": "missing provenance"})
    manifest = tmp_path / "metadata.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot json\n")
    output = tmp_path / "prepared"
    report = prepare(images, manifest, output, min_side=16)
    assert report["counts"]["accepted"] == 2 and report["counts"]["rejected"] == 6
    with sqlite3.connect(output / "index.sqlite3") as db:
        records = db.execute("SELECT split,bucket_width,bucket_height FROM samples").fetchall()
    assert len(set(row[0] for row in records)) == 1
    split = records[0][0]
    count = export(output / "index.sqlite3", tmp_path / "shards", split=split, shard_size=1)
    assert count == 2
    shards = list((tmp_path / "shards").glob("*.tar"))
    assert len(shards) == 2
    with tarfile.open(shards[0]) as tar:
        assert len(tar.getnames()) == 3
        assert any(name.endswith(".txt") for name in tar.getnames())
    assert export(output / "index.sqlite3", tmp_path / "pilot", split=split, pilot_limit=1) == 1
    assert len((tmp_path / "pilot" / "metadata.jsonl").read_text().splitlines()) == 1
    Image.new("RGB", (64, 32), "green").save(images / "a.png")
    with pytest.raises(ValueError, match="changed since indexing"):
        export(output / "index.sqlite3", tmp_path / "changed", split=split)


def test_duplicate_pixels_across_encodings(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = Image.new("RGB", (64, 64), "red")
    image.save(images / "a.png")
    image.save(images / "b.bmp")
    manifest = tmp_path / "metadata.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"file_name": name, "text": "Red", "source": "test", "license": "owned"})
            for name in ["a.png", "b.bmp"]
        )
    )
    report = prepare(images, manifest, tmp_path / "out", min_side=16)
    assert report["counts"]["accepted"] == 1 and report["counts"]["rejected"] == 1


def test_bucket_orientation_and_invalid_size():
    assert choose_bucket(1600, 900) == (1344, 768)
    assert choose_bucket(900, 1600) == (768, 1344)
    with pytest.raises(ValueError):
        choose_bucket(0, 1)
