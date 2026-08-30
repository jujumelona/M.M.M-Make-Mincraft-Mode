from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_liveness_reporter_canonical_v1.py")
text = path.read_text(encoding="utf-8")
anchor = '''replace_exact(\n    "minecraft_mod_ai/llama_completion_liveness_contract.py",\n    '_REPORTER_MARKER = "_mmm_no_slot_poll_completion_liveness_v1"\\n',\n    "",\n)\n'''
addition = '''replace_exact(\n    "minecraft_mod_ai/llama_completion_liveness_contract.py",\n    '''def _install_stream_progress_payload(stream_module: Any) -> None:\\n    client_type = stream_module._StreamingCompletionClient\\n    current = client_type.stream\\n    if getattr(current, _STREAM_MARKER, False):\\n        return\\n''',\n    '''def _install_stream_progress_payload(stream_module: Any) -> None:\\n    client_type = stream_module._StreamingCompletionClient\\n    current = getattr(client_type, "stream", None)\\n    if not callable(current) or getattr(current, _STREAM_MARKER, False):\\n        return\\n''',\n)\n\n'''
if text.count(anchor) != 1:
    raise SystemExit(f"worker05 liveness canonical installer anchor count={text.count(anchor)}")
path.write_text(text.replace(anchor, addition + anchor, 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
