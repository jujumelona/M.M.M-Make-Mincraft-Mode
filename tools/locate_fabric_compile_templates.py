from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "FABRIC_TEMPLATE_LOCATIONS.json"
PATTERNS = (
    "state.stop()",
    "validateTicker(",
    "ServerPlayerEntity",
    "PlayerEntity player",
    "reward",
)


def line_context(lines: list[str], index: int, radius: int = 8) -> dict[str, object]:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return {
        "line": index + 1,
        "start_line": start + 1,
        "end_line": end,
        "context": "\n".join(
            f"{number + 1:05d}: {lines[number]}"
            for number in range(start, end)
        ),
    }


def main() -> None:
    matches: dict[str, list[dict[str, object]]] = {pattern: [] for pattern in PATTERNS}
    for path in sorted((ROOT / "minecraft_mod_ai").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            for pattern in PATTERNS:
                if pattern in line:
                    matches[pattern].append(
                        {
                            "path": path.relative_to(ROOT).as_posix(),
                            **line_context(lines, index),
                        }
                    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(matches, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
