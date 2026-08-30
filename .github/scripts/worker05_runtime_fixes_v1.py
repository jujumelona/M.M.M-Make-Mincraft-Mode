from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# A wrapper must never advertise stream() when the wrapped test/custom client does not
# implement streaming. Production httpx.Client has stream(); minimal clients then fall
# through to the bounded POST compatibility path in _StreamingCompletionClient.
replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    '''def _wrap_raw_client(client: Any, stream_module: Any) -> Any:\n    if getattr(client, "_mmm_semantic_progress_client_v1", False):\n        return client\n    return _SemanticProgressClient(client, stream_module)\n''',
    '''def _wrap_raw_client(client: Any, stream_module: Any) -> Any:\n    if getattr(client, "_mmm_semantic_progress_client_v1", False):\n        return client\n    if not callable(getattr(client, "stream", None)):\n        return client\n    return _SemanticProgressClient(client, stream_module)\n''',
)
replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    '''import json\nimport time\nfrom collections.abc import Mapping\nfrom functools import wraps\nfrom typing import Any, Callable\n''',
    '''import json\nimport time\nfrom collections.abc import Callable, Mapping\nfrom functools import wraps\nfrom types import TracebackType\nfrom typing import Any\n''',
)
replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    '''    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:\n        return self._stream.__exit__(exc_type, exc, tb)\n''',
    '''    def __exit__(\n        self,\n        exc_type: type[BaseException] | None,\n        exc: BaseException | None,\n        tb: TracebackType | None,\n    ) -> Any:\n        return self._stream.__exit__(exc_type, exc, tb)\n''',
)

# A compacted-history ledger is part of semantic continuity. Protocol selection must
# also be closed over assistant tool calls and their tool results: OpenAI chat history
# must never contain an orphaned tool result or an assistant tool call without results.
context_path = ROOT / "minecraft_mod_ai/llama_context_safety_contract.py"
context_text = context_path.read_text(encoding="utf-8")
anchor = '''def _latest_mutation_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:\n'''
helper = '''def _latest_compaction_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:\n    latest: int | None = None\n    for index, message in enumerate(messages):\n        if str(message.get("role", "")) != "system":\n            continue\n        content = message.get("content")\n        if isinstance(content, str) and "HOST COMPACTED VERIFIED CONTEXT" in content:\n            latest = index\n    return {latest} if latest is not None else set()\n\n\ndef _close_tool_protocol_indices(\n    messages: Sequence[Mapping[str, Any]],\n    selected: set[int],\n) -> set[int]:\n    owners: dict[str, int] = {}\n    results: dict[str, set[int]] = {}\n    for index, message in enumerate(messages):\n        if str(message.get("role", "")) == "assistant":\n            calls = message.get("tool_calls")\n            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):\n                for call in calls:\n                    if not isinstance(call, Mapping):\n                        continue\n                    call_id = str(call.get("id", "")).strip()\n                    if call_id:\n                        owners[call_id] = index\n        elif str(message.get("role", "")) == "tool":\n            call_id = str(message.get("tool_call_id", "")).strip()\n            if call_id:\n                results.setdefault(call_id, set()).add(index)\n\n    closed = set(selected)\n    changed = True\n    while changed:\n        changed = False\n        for index in tuple(closed):\n            message = messages[index]\n            role = str(message.get("role", ""))\n            related: set[int] = set()\n            if role == "tool":\n                call_id = str(message.get("tool_call_id", "")).strip()\n                owner = owners.get(call_id)\n                if owner is not None:\n                    related.add(owner)\n            elif role == "assistant":\n                calls = message.get("tool_calls")\n                if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):\n                    for call in calls:\n                        if not isinstance(call, Mapping):\n                            continue\n                        call_id = str(call.get("id", "")).strip()\n                        related.update(results.get(call_id, ()))\n            before = len(closed)\n            closed.update(related)\n            changed = changed or len(closed) != before\n    return closed\n\n\n'''
if context_text.count(anchor) != 1:
    raise SystemExit("llama_context_safety_contract.py: mutation helper anchor mismatch")
context_text = context_text.replace(anchor, helper + anchor, 1)
old = '''        selected = _leading_authority_indices(original)\n        selected.update(_latest_mutation_indices(original))\n        selected.update(_latest_protocol_tail_indices(original, width=width))\n        candidate = tuple(original[index] for index in sorted(selected))\n'''
new = '''        selected = _leading_authority_indices(original)\n        selected.update(_latest_compaction_indices(original))\n        selected.update(_latest_mutation_indices(original))\n        selected.update(_latest_protocol_tail_indices(original, width=width))\n        selected = _close_tool_protocol_indices(original, selected)\n        candidate = tuple(original[index] for index in sorted(selected))\n'''
if context_text.count(old) != 1:
    raise SystemExit("llama_context_safety_contract.py: protocol selection mismatch")
context_text = context_text.replace(old, new, 1)
old = '''    mandatory = _leading_authority_indices(original)\n    mandatory.update(_latest_mutation_indices(original))\n    mandatory.update(_latest_protocol_tail_indices(original, width=1))\n    candidate = tuple(original[index] for index in sorted(mandatory))\n'''
new = '''    mandatory = _leading_authority_indices(original)\n    mandatory.update(_latest_compaction_indices(original))\n    mandatory.update(_latest_mutation_indices(original))\n    mandatory.update(_latest_protocol_tail_indices(original, width=1))\n    mandatory = _close_tool_protocol_indices(original, mandatory)\n    candidate = tuple(original[index] for index in sorted(mandatory))\n'''
if context_text.count(old) != 1:
    raise SystemExit("llama_context_safety_contract.py: mandatory selection mismatch")
context_text = context_text.replace(old, new, 1)
context_path.write_text(context_text, encoding="utf-8")

# Keep the SSE regression fixture lint-clean; this test is part of the worker05 gate.
replace_exact(
    "tests/test_llama_sse_error_contract.py",
    '''from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract\nfrom minecraft_mod_ai import llama_sse_error_contract as contract\n''',
    '''from minecraft_mod_ai import llama_sse_error_contract as contract\nfrom minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract\n''',
)
replace_exact(
    "tests/test_llama_sse_error_contract.py",
    '''class _FakeResponse:\n    status_code = 200\n    headers: dict[str, str] = {}\n\n    def __init__(self, lines: tuple[str, ...]) -> None:\n        self._lines = lines\n''',
    '''class _FakeResponse:\n    status_code = 200\n\n    def __init__(self, lines: tuple[str, ...]) -> None:\n        self.headers: dict[str, str] = {}\n        self._lines = lines\n''',
)

Path(__file__).unlink(missing_ok=True)
