from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_sse_canonical_v1.py")
text = path.read_text(encoding="utf-8")

anchor = '''STREAM.write_text(stream_text.replace(needle, replacement, 1), encoding="utf-8")\n\n# Bootstrap no longer composes a runtime SSE monkeypatch.\n'''
insert = '''STREAM.write_text(stream_text.replace(needle, replacement, 1), encoding="utf-8")\n\n# Remove exports for the deleted native slot-polling fallback.\nstream_text = STREAM.read_text(encoding="utf-8")\nfor stale_export in (\n    '    "_probe_native_tool_progress",\\n',\n    '    "_slot_progress_from_payload",\\n',\n    '    "_needs_native_tool_liveness_reporter",\\n',\n    '    "_native_tool_liveness_reporter",\\n',\n):\n    stream_text = stream_text.replace(stale_export, "")\nSTREAM.write_text(stream_text, encoding="utf-8")\n\n# Bootstrap no longer composes a runtime SSE monkeypatch.\n'''
if text.count(anchor) != 1:
    raise SystemExit(f"SSE canonical stream-export anchor count={text.count(anchor)}")
text = text.replace(anchor, insert, 1)

anchor = '''replace_once(\n    BOOTSTRAP,\n    ''' + "'''" + '''    install_sse_errors(\\n        llama_completion_liveness_contract,\\n        llama_stream_efficiency_contract,\\n    )\\n''' + "'''" + ''',\n    "",\n)\n\nif not OLD_SHIM.exists():\n'''
insert = '''replace_once(\n    BOOTSTRAP,\n    ''' + "'''" + '''    install_sse_errors(\\n        llama_completion_liveness_contract,\\n        llama_stream_efficiency_contract,\\n    )\\n''' + "'''" + ''',\n    "",\n)\n\n# Remove only the now-unused module-list import owned by _install_model_runtime_contracts.\nbootstrap_text = BOOTSTRAP.read_text(encoding="utf-8")\nstart = bootstrap_text.find("def _install_model_runtime_contracts() -> None:")\nend = bootstrap_text.find("\\ndef ", start + 1)\nif start < 0 or end < 0:\n    raise SystemExit("runtime_bootstrap.py: model-runtime stage boundaries not found")\nsection = bootstrap_text[start:end]\nunused_line = "        llama_completion_liveness_contract,\\n"\nif section.count(unused_line) != 1:\n    raise SystemExit(\n        "runtime_bootstrap.py: expected one model-runtime liveness module-list import, "\n        f"found {section.count(unused_line)}"\n    )\nsection = section.replace(unused_line, "", 1)\nBOOTSTRAP.write_text(bootstrap_text[:start] + section + bootstrap_text[end:], encoding="utf-8")\n\nif not OLD_SHIM.exists():\n'''
if text.count(anchor) != 1:
    raise SystemExit(f"SSE canonical bootstrap anchor count={text.count(anchor)}")
text = text.replace(anchor, insert, 1)

anchor = '''if OLD_SSE_TEST.exists():\n    OLD_SSE_TEST.unlink()\n\n# No stale runtime shim references may survive.\n'''
insert = '''if OLD_SSE_TEST.exists():\n    OLD_SSE_TEST.unlink()\n\n# Keep the generated regression test lint-clean without weakening its assertions.\nsse_test_text = SSE_TEST.read_text(encoding="utf-8")\nsse_test_text = sse_test_text.replace("from types import SimpleNamespace\\n\\n", "", 1)\nsse_test_text = sse_test_text.replace("        headers = {}\\n", "", 1)\nSSE_TEST.write_text(sse_test_text, encoding="utf-8")\n\n# No stale runtime shim references may survive.\n'''
if text.count(anchor) != 1:
    raise SystemExit(f"SSE canonical test-cleanup anchor count={text.count(anchor)}")
text = text.replace(anchor, insert, 1)

path.write_text(text, encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
