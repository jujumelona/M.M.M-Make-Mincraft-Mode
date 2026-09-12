from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (ROOT / "minecraft_mod_ai",)

PATTERNS = {
    "executor": re.compile(r"\b(?:ThreadPoolExecutor|ProcessPoolExecutor)\b"),
    "future_result": re.compile(r"\.result\s*\("),
    "shutdown_wait_true": re.compile(r"\.shutdown\s*\(\s*wait\s*=\s*True"),
    "executor_map": re.compile(r"\.(?:map)\s*\("),
    "as_completed": re.compile(r"\bas_completed\s*\("),
    "wait_call": re.compile(r"\bwait\s*\("),
    "tool_round_cap": re.compile(
        r"(?:_agent_tool_round_limit|MMM_AGENT_TOOL_ROUNDS|DEFAULT_AGENT_TOOL_ROUNDS|round_limit)"
    ),
    "candidate_score_constant": re.compile(
        r"(?:1000\.0|candidate_count|scored\.append)"
    ),
}


def source_files() -> list[Path]:
    return sorted(path for root in SCAN_ROOTS for path in root.rglob("*.py"))


def main() -> int:
    files = source_files()
    print(f"PYTHON_FILES={len(files)}")

    for label, pattern in PATTERNS.items():
        print(f"=== {label} ===")
        count = 0
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                print(f"READ_ERROR {path.relative_to(ROOT)}: {exc}")
                continue
            for lineno, line in enumerate(lines, 1):
                if pattern.search(line):
                    count += 1
                    print(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")
        print(f"COUNT={count}")

    print("=== syntax ===")
    bad = 0
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            bad += 1
            lineno = int(getattr(exc, "lineno", 0) or 0)
            print(f"SYNTAX_ERROR {path.relative_to(ROOT)}:{lineno}:{exc}")
    print(f"SYNTAX_ERRORS={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
