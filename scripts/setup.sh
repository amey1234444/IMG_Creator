#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[inference,ui]'
printf '%s\n' 'Activate with: source .venv/bin/activate' 'Then run: img-creator-ui'
