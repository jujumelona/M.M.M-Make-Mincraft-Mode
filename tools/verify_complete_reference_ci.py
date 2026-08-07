from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

if __package__:
    from .verify_complete_reference_build import build_reference
else:
    from verify_complete_reference_build import build_reference


def _collect_logs(output: Path) -> list[dict[str, object]]:
    evidence = output / "evidence" / "logs"
    evidence.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, object]] = []
    for source in sorted(output.rglob(".minecraft_ai/logs/*.log")):
        relative = source.relative_to(output).as_posix().replace("/", "__")
        target = evidence / relative
        shutil.copy2(source, target)
        lines = source.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        tail = lines[-160:]
        collected.append(
            {
                "source": str(source),
                "artifact": str(target),
                "line_count": len(lines),
                "tail": tail,
            }
        )
        print(f"\n===== {source} (last {len(tail)} lines) =====")
        print("\n".join(tail))
    manifest = output / "evidence" / "gradle-log-manifest.json"
    manifest.write_text(
        json.dumps(collected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return collected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        result = build_reference(output)
    except Exception:
        _collect_logs(output)
        traceback.print_exc()
        return 1
    _collect_logs(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
