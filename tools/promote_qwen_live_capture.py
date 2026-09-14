from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("replay id must contain an alphanumeric character")
    return normalized


def _raw_text(capture: dict[str, Any]) -> str:
    if capture.get("provenance") != "real_capture":
        raise ValueError("only provenance=real_capture artifacts may be promoted")
    message = capture.get("raw_assistant_message")
    if not isinstance(message, dict):
        raise TypeError("capture has no raw_assistant_message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("capture assistant content is not text")
    return content


def render_replay_test(capture: dict[str, Any], replay_id: str) -> str:
    name = _identifier(replay_id)
    raw = _raw_text(capture)
    conditions = capture.get("conditions")
    condition_json = json.dumps(conditions, ensure_ascii=False, sort_keys=True)
    return (
        "import pytest\n\n"
        "from ._support import replay_text\n\n\n"
        f"# Promoted from a real model capture. Conditions: {condition_json}\n"
        f"def test_qwen35_real_{name}_remains_rejected():\n"
        f"    raw = {raw!r}\n"
        "    with pytest.raises(Exception):\n"
        "        replay_text(raw)\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a failing real Qwen capture into one deterministic replay test."
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/model_realistic_replay")
    )
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    name = _identifier(args.id)
    output = args.output_dir / f"test_qwen35_real_{name}.py"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replay: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_replay_test(capture, name), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
