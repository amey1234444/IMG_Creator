"""Separate GPU worker for bounded Klein 4B LoRA pilots. No web-supplied shell commands."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
from sqlalchemy import select, update
from .config import PlatformSettings
from .db import Database, TrainingRun, uid
from .storage import Storage

log = logging.getLogger(__name__)


def training_command(config, checkout, dataset, output):
    from training.train_lora import command

    options = {
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
        "instance_prompt": "a detailed image",
        "resolution": 1024,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "learning_rate": config["learning_rate"],
        "max_train_steps": config["steps"],
        "checkpointing_steps": 100,
        "checkpoints_total_limit": 2,
        "mixed_precision": "bf16",
        "rank": config["rank"],
        "seed": config["seed"],
    }
    cmd = command(options, checkout, dataset, output)
    cmd[cmd.index("--report_to") + 1] = "tensorboard"
    return cmd


def tensorboard_metrics(output):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for event in output.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 100})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                metrics[tag] = [
                    {"step": v.step, "value": v.value} for v in values[-50:] if __import__("math").isfinite(v.value)
                ]
    return metrics


def train_one(db, storage, checkout, timeout_hours=12):
    with db.transaction() as s:
        # Interrupted training is failed for explicit resubmission, never invisibly restarted.
        s.execute(
            update(TrainingRun)
            .where(TrainingRun.status == "running", TrainingRun.lease_until < time.time())
            .values(status="failed", error="Training worker interrupted; submit a new run")
        )
        run = s.scalar(
            select(TrainingRun)
            .where(TrainingRun.status == "queued")
            .order_by(TrainingRun.created)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if not run:
            return False
        token = uid()
        if not s.execute(
            update(TrainingRun)
            .where(TrainingRun.id == run.id, TrainingRun.status == "queued")
            .values(status="running", lease_token=token, lease_until=time.time() + 180)
        ).rowcount:
            return False
        s.refresh(run)
    began = time.monotonic()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="img-train-") as temporary:
            root = Path(temporary)
            dataset, output = root / "dataset", root / "output"
            dataset.mkdir()
            output.mkdir()
            manifest = []
            for item in run.snapshot:
                if item["split"] != "train":
                    continue
                data = storage.get(item["key"])
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Dataset snapshot integrity check failed")
                filename = item["id"] + ".png"
                (dataset / filename).write_bytes(data)
                manifest.append({"file_name": filename, "text": item["caption"]})
            (dataset / "metadata.jsonl").write_text("\n".join(json.dumps(x) for x in manifest) + "\n")
            cmd = training_command(run.config, checkout, dataset, output)
            with (root / "train.log").open("wb") as log_file:
                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if time.monotonic() - began > timeout_hours * 3600:
                        raise TimeoutError("Training exceeded the configured runtime budget")
                    metrics = {
                        "elapsed_seconds": round(time.monotonic() - began, 1),
                        "scalars": tensorboard_metrics(output),
                        "train_images": len(manifest),
                        "validation_images": len(run.snapshot) - len(manifest),
                    }
                    with db.transaction() as s:
                        count = s.execute(
                            update(TrainingRun)
                            .where(
                                TrainingRun.id == run.id,
                                TrainingRun.status == "running",
                                TrainingRun.lease_token == token,
                            )
                            .values(metrics=metrics, lease_until=time.time() + 180)
                        ).rowcount
                        if not count:
                            raise RuntimeError("Training lease lost")
                    time.sleep(5)
            log_key = f"training/{run.id}/train.log"
            # Keep the end of the log, bounded to 2 MB.
            with (root / "train.log").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 2_000_000))
                storage.put(log_key, stream.read())
            if process.returncode:
                raise RuntimeError("GPU training failed; see the private training log")
            weights = output / "pytorch_lora_weights.safetensors"
            from safetensors import safe_open

            with safe_open(str(weights), framework="pt", device="cpu") as tensors:
                if not list(tensors.keys()):
                    raise ValueError("Empty adapter artifact")
            data = weights.read_bytes()
            key = f"training/{run.id}/adapter.safetensors"
            storage.put(key, data)
            metrics = {
                "elapsed_seconds": round(time.monotonic() - began, 1),
                "scalars": tensorboard_metrics(output),
                "train_images": len(manifest),
                "validation_images": len(run.snapshot) - len(manifest),
                "log_key": log_key,
            }
            with db.transaction() as s:
                s.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run.id, TrainingRun.status == "running", TrainingRun.lease_token == token)
                    .values(
                        status="awaiting_evaluation",
                        metrics=metrics,
                        artifact_key=key,
                        artifact_sha256=hashlib.sha256(data).hexdigest(),
                        finished=time.time(),
                    )
                )
    except BaseException:
        if process and process.poll() is None:
            import os
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.exception("Training run %s failed", run.id)
        with db.transaction() as s:
            s.execute(
                update(TrainingRun)
                .where(TrainingRun.id == run.id, TrainingRun.lease_token == token)
                .values(status="failed", error="Training failed. Inspect the GPU worker logs.", finished=time.time())
            )
        raise
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers-checkout", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=12)
    args = parser.parse_args()
    if not 0 < args.timeout_hours <= 48:
        parser.error("Training budget must be greater than 0 and at most 48 hours")
    settings = PlatformSettings()
    db, storage = Database(settings.database_url), Storage(settings)
    while True:
        try:
            worked = train_one(db, storage, args.diffusers_checkout, args.timeout_hours)
        except Exception:
            worked = True
        if args.once:
            break
        if not worked:
            time.sleep(5)


if __name__ == "__main__":
    main()
