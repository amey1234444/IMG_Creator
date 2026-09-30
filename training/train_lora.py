"""Validate and launch a small LoRA pilot with the official, pinned Diffusers trainer.

The upstream trainer holds image tensors in RAM. This wrapper deliberately caps it
at 2,000 images; large sharded datasets need a streaming training integration.
"""

from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys

DIFFUSERS_REVISION = "031b2798addadd1652db7cfba50eacc1079245cf"


def command(config: dict, checkout: Path, dataset: Path, output: Path, resume=None):
    script = checkout / "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
    if not script.is_file():
        raise ValueError("Pinned Diffusers training script not found; see training/README.md")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != DIFFUSERS_REVISION:
        raise ValueError(f"Diffusers checkout must be pinned to {DIFFUSERS_REVISION}")
    manifest = dataset / "metadata.jsonl"
    with manifest.open() as f:
        count = sum(1 for line in f if line.strip())
    if not 1 <= count <= 2000:
        raise ValueError("Pilot trainer accepts 1–2000 images; use a streaming trainer for larger datasets")
    allowed = {
        "pretrained_model_name_or_path",
        "instance_prompt",
        "resolution",
        "train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "learning_rate",
        "max_train_steps",
        "checkpointing_steps",
        "checkpoints_total_limit",
        "mixed_precision",
        "rank",
        "lora_alpha",
        "seed",
        "lr_scheduler",
        "lr_warmup_steps",
        "validation_prompt",
        "validation_epochs",
        "use_aspect_ratio_buckets",
        "aspect_ratio_buckets",
        "offload",
    }
    if unknown := set(config) - allowed:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    if config.get("pretrained_model_name_or_path") != "black-forest-labs/FLUX.2-klein-base-4B":
        raise ValueError("Pilot config must target the Klein base 4B model")
    if int(config.get("max_train_steps", 0)) < 1:
        raise ValueError("max_train_steps must be positive")
    # Validate flags against the actual pinned source without importing training dependencies.
    flags = {
        node.value
        for node in ast.walk(ast.parse(script.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")
    }
    options = {
        **config,
        "dataset_name": str(dataset.resolve()),
        "image_column": "image",
        "caption_column": "text",
        "output_dir": str(output.resolve()),
        "report_to": "none",
    }
    if resume:
        options["resume_from_checkpoint"] = resume
    cmd = [sys.executable, "-m", "accelerate.commands.launch", str(script.resolve())]
    for key, value in options.items():
        flag = "--" + key
        if flag not in flags:
            raise ValueError(f"Pinned trainer does not support {flag}")
        if value is True:
            cmd.append(flag)
        elif value is not False and value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers-checkout", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("training/configs/flux_lora.json"))
    parser.add_argument("--resume")
    parser.add_argument("--run", action="store_true", help="Launch GPU training; default only prints the command")
    args = parser.parse_args()
    cmd = command(json.loads(args.config.read_text()), args.diffusers_checkout, args.dataset, args.output, args.resume)
    print(shlex.join(cmd))
    if args.run:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "launch.json").write_text(
            json.dumps({"command": cmd, "diffusers_revision": DIFFUSERS_REVISION}, indent=2)
        )
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
