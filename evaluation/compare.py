"""Create a CSV for paired human scoring; no invented automatic quality score."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields = [
        "run",
        "generation_id",
        "prompt",
        "seed",
        "model_id",
        "quality",
        "generation_seconds",
        "prompt_adherence_1_5",
        "composition_1_5",
        "detail_1_5",
        "anatomy_1_5",
        "artifacts_0_5",
        "notes",
    ]
    with args.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report in args.reports:
            for line in report.read_text().splitlines():
                row = json.loads(line)
                writer.writerow({**{k: row.get(k, "") for k in fields}, "run": report.stem})


if __name__ == "__main__":
    main()
