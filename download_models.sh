#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-t4_local}"
CACHE_DIR="${2:-${HF_HOME:-$HOME/.cache/huggingface}}"

python -m pip install -e ".[local-model,rag,image,speech]"
python download_resources.py --profile "${PROFILE}" --download --cache-dir "${CACHE_DIR}"
