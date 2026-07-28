#!/usr/bin/env bash
set -euo pipefail

# Optional: the deterministic planner/generator/build pipeline works without it.
python -m pip install "transformers>=4.57,<6" "accelerate>=1.0,<2"
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir "${1:-./models/qwen3-4b-instruct-2507}"
