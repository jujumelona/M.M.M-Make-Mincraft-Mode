from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_sse_canonical_v1.py")
text = path.read_text(encoding="utf-8")
old = '''def replace_once(path: Path, old: str, new: str) -> None:\n    text = path.read_text(encoding="utf-8")\n    count = text.count(old)\n    if count != 1:\n        raise SystemExit(f"{path.relative_to(ROOT)}: expected one match, found {count}")\n    path.write_text(text.replace(old, new, 1), encoding="utf-8")\n'''
new = '''def replace_once(path: Path, old: str, new: str) -> None:\n    text = path.read_text(encoding="utf-8")\n    count = text.count(old)\n    # The transport and plain-generation owners intentionally have the same raw SSE\n    # loop header. The first scoped replacement belongs to _StreamingCompletionClient.post;\n    # the script verifies exactly one fast-path occurrence remains immediately afterward.\n    shared_sse_loop = (\n        path == STREAM\n        and old == "                for raw_line in response.iter_lines():\\n                    line = raw_line.strip()\\n"\n    )\n    expected = 2 if shared_sse_loop else 1\n    if count != expected:\n        raise SystemExit(\n            f"{path.relative_to(ROOT)}: expected {expected} match(es), found {count}"\n        )\n    path.write_text(text.replace(old, new, 1), encoding="utf-8")\n'''
if text.count(old) != 1:
    raise SystemExit(f"SSE canonical script helper count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
