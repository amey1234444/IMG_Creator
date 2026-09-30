#!/usr/bin/env bash
set -euo pipefail
uvicorn img_creator.api:app --host 127.0.0.1 --port 8000
