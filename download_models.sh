#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-t4_local}"
CACHE_DIR="${2:-${HF_HOME:-$HOME/.cache/huggingface}}"

# The downloader only needs the local source tree, PyYAML and huggingface_hub.
# Avoid resolving the full local-model/RAG/image/speech stacks when those imports
# are already available in the current environment.
if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub
import yaml
PY
then
  python -m pip install --prefer-binary -e . 'huggingface-hub>=0.28,<2'
fi

python download_resources.py --profile "${PROFILE}" --download --cache-dir "${CACHE_DIR}"
