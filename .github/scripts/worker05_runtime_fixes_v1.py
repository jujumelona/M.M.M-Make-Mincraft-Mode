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

# A compacted-history ledger is part of the semantic continuity proof. If a later hard
# fit has to reduce history further, retain the newest ledger together with authority,
# the authored task, mutation proof and recent protocol instead of orphaning the archive.
context_path = ROOT / "minecraft_mod_ai/llama_context_safety_contract.py"
context_text = context_path.read_text(encoding="utf-8")
anchor = '''def _latest_mutation_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:\n'''
helper = '''def _latest_compaction_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:\n    latest: int | None = None\n    for index, message in enumerate(messages):\n        if str(message.get("role", "")) != "system":\n            continue\n        content = message.get("content")\n        if (\n            isinstance(content, str)\n            and "HOST COMPACTED VERIFIED CONTEXT" in content\n        ):\n            latest = index\n    return {latest} if latest is not None else set()\n\n\n'''
if context_text.count(anchor) != 1:
    raise SystemExit("llama_context_safety_contract.py: mutation helper anchor mismatch")
context_text = context_text.replace(anchor, helper + anchor, 1)
old = '''        selected = _leading_authority_indices(original)\n        selected.update(_latest_mutation_indices(original))\n        selected.update(_latest_protocol_tail_indices(original, width=width))\n'''
new = '''        selected = _leading_authority_indices(original)\n        selected.update(_latest_compaction_indices(original))\n        selected.update(_latest_mutation_indices(original))\n        selected.update(_latest_protocol_tail_indices(original, width=width))\n'''
if context_text.count(old) != 1:
    raise SystemExit("llama_context_safety_contract.py: protocol selection mismatch")
context_text = context_text.replace(old, new, 1)
old = '''    mandatory = _leading_authority_indices(original)\n    mandatory.update(_latest_mutation_indices(original))\n'''
new = '''    mandatory = _leading_authority_indices(original)\n    mandatory.update(_latest_compaction_indices(original))\n    mandatory.update(_latest_mutation_indices(original))\n'''
if context_text.count(old) != 1:
    raise SystemExit("llama_context_safety_contract.py: mandatory selection mismatch")
context_text = context_text.replace(old, new, 1)
context_path.write_text(context_text, encoding="utf-8")

Path(__file__).unlink(missing_ok=True)
