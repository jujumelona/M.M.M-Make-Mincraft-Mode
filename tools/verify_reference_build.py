from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minecraft_mod_ai import MinecraftModPipeline

REFERENCE_PROMPT = (
    "Create a frost boss with a 3D model, one item and one block"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()

    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan(REFERENCE_PROMPT)
    result = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=args.output,
        build=True,
        run_gametest=True,
        gradle_cache=args.cache or (args.output / ".cache"),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
