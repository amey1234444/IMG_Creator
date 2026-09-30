from __future__ import annotations
import argparse
import json
from pathlib import Path
from img_creator.generator import ImageGenerator
from img_creator.schemas import GenerateRequest


def main():
    parser = argparse.ArgumentParser(description="Generate fixed prompt/seed pairs for a human-reviewed benchmark")
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["fast", "quality", "ultra"], default="fast")
    parser.add_argument("--resolution", default="Native")
    parser.add_argument("--upscaler", choices=["lanczos", "learned"], default="lanczos")
    parser.add_argument("--lora")
    args = parser.parse_args()
    service = ImageGenerator()
    with args.output.open("x", encoding="utf-8") as report:
        for row in json.loads(args.prompts.read_text()):
            req = GenerateRequest(
                **row, quality=args.profile, output_resolution=args.resolution, upscaler=args.upscaler, lora=args.lora
            )
            metadata = service.generate_batch(req)[0]
            report.write(json.dumps(metadata) + "\n")
            report.flush()


if __name__ == "__main__":
    main()
