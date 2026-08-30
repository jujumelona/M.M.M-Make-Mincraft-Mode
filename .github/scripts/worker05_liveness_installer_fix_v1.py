from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "minecraft_mod_ai/llama_completion_liveness_contract.py"
text = path.read_text(encoding="utf-8")
old = '''def _install_stream_progress_payload(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current = client_type.stream\n    if getattr(current, _STREAM_MARKER, False):\n        return\n'''
new = '''def _install_stream_progress_payload(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current = getattr(client_type, "stream", None)\n    if not callable(current) or getattr(current, _STREAM_MARKER, False):\n        return\n'''
count = text.count(old)
if count != 1:
    raise SystemExit(
        "llama_completion_liveness_contract.py: expected one strict stream installer, "
        f"found {count}"
    )
path.write_text(text.replace(old, new, 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
