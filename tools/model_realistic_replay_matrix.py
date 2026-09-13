from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "tests" / "model_realistic_replay"


def discover() -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in sorted(REPLAY_DIR.glob("test_*.py"))]


def main() -> int:
    files = discover()
    if not files:
        raise SystemExit("no model-realistic replay tests discovered")
    print(json.dumps({"include": [{"id": Path(path).stem.removeprefix("test_"), "test": path} for path in files]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
