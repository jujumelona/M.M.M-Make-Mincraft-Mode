from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_sse_canonical_v1.py")
text = path.read_text(encoding="utf-8")
old = 'replace_once(BOOTSTRAP, "        llama_completion_liveness_contract,\\n", "")\n'
if text.count(old) != 1:
    raise SystemExit(f"SSE canonical bootstrap list cleanup count={text.count(old)}")
# Keep module-list entries untouched; only the obsolete SSE installer import/call is removed.
path.write_text(text.replace(old, "", 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
