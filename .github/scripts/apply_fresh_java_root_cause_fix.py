from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "minecraft_mod_ai" / "progress_aware_tool_loop.py"
TEST = ROOT / "tests" / "test_api_grounding_repair_installation.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_loop() -> None:
    text = LOOP.read_text(encoding="utf-8")

    helper_anchor = "\n\n@dataclass\nclass HostRunState:\n"
    helpers = r'''

_JAVA_API_EVIDENCE_RE = re.compile(
    r"(?:\b(?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)\.[A-Za-z0-9_.$]+"
    r"|\b(?:package|import)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+"
    r"|\b(?:class|interface|record|enum)\s+[A-Za-z_$][\w$]*)"
)
_ATOMIC_OUTPUT_RECOVERY_MARKER = "MMM_ATOMIC_OUTPUT_RECOVERY_V1"


def _fresh_java_context(context: TargetMutationContext | None) -> bool:
    if context is None or not context.is_new_file:
        return False
    return _canonical_mutation_path(context.target_path).casefold().endswith(".java")


def _mapping_schema(value: Any, schema: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if str(value.get("schema_version") or "").strip() == schema:
        return True
    for key in ("structured_content", "result", "data"):
        child = value.get(key)
        if isinstance(child, Mapping) and _mapping_schema(child, schema):
            return True
    return False


def _java_evidence_texts(value: Any) -> tuple[str, ...]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        for key in ("parsed_text", "text", "content", "snippet", "code", "source", "source_text", "body"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                texts.append(raw)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                texts.extend(str(item) for item in raw if isinstance(item, str) and item.strip())
        for key in ("hits", "results", "records", "documents", "chunks", "resources", "symbols", "evidence"):
            raw = value.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                for item in raw:
                    texts.extend(_java_evidence_texts(item))
        for key in ("structured_content", "result", "data"):
            child = value.get(key)
            if child is not None:
                texts.extend(_java_evidence_texts(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            texts.extend(_java_evidence_texts(item))
    return tuple(texts)


def _authoritative_java_evidence(value: Any) -> bool:
    """Return whether evidence is strong enough to authorize a fresh Java mutation."""
    if not isinstance(value, Mapping) or not value:
        return False

    # The built-in project RAG corpus is target-neutral context. It may inform the
    # model, but it is never an exact classpath/symbol proof for a fresh Java source.
    if _mapping_schema(value, "mmm/rag-result-v2"):
        return False

    if _mapping_schema(value, "mmm/java-symbols-v1"):
        def has_symbols(item: Any) -> bool:
            if isinstance(item, Mapping):
                symbols = item.get("symbols")
                if isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
                    if any(isinstance(symbol, Mapping) and bool(symbol) for symbol in symbols):
                        return True
                return any(has_symbols(child) for child in item.values())
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                return any(has_symbols(child) for child in item)
            return False
        return has_symbols(value)

    if _mapping_schema(value, "mmm/code-rag-result-v1"):
        return any(_JAVA_API_EVIDENCE_RE.search(text) for text in _java_evidence_texts(value))

    # Reviewed external MCP evidence only authorizes ACT when its payload actually
    # names Java/Minecraft/Fabric symbols or contains concrete mapping records.
    if any(_JAVA_API_EVIDENCE_RE.search(text) for text in _java_evidence_texts(value)):
        return True

    def has_mapping_records(item: Any) -> bool:
        if isinstance(item, Mapping):
            mappings = item.get("mappings")
            if isinstance(mappings, Mapping) and bool(mappings):
                return True
            if isinstance(mappings, Sequence) and not isinstance(mappings, (str, bytes, bytearray)):
                if any(isinstance(entry, Mapping) and bool(entry) for entry in mappings):
                    return True
            return any(has_mapping_records(child) for child in item.values())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return any(has_mapping_records(child) for child in item)
        return False

    return has_mapping_records(value)


def _target_evidence_ready(
    state: "HostRunState",
    *,
    require_rag: bool,
    fresh_java_target: bool,
) -> bool:
    if not require_rag:
        return True
    if fresh_java_target:
        return state.has_authoritative_java_evidence
    return state.has_fresh_evidence


def _completion_boundary_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        text = str(current).casefold()
        if name == "LlamaCompletionBoundaryError":
            return True
        if (
            "completion boundary" in text
            or "completion token limit" in text
            or "maximum completion" in text
            or ("finish_reason" in text and "length" in text)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
'''
    if "def _authoritative_java_evidence(" not in text:
        text = replace_once(text, helper_anchor, helpers + helper_anchor, "Java grounding helpers")

    text = replace_once(
        text,
        "    evidence_fingerprints: set[str] = field(default_factory=set)\n    mutation_context: TargetMutationContext | None = None\n",
        "    evidence_fingerprints: set[str] = field(default_factory=set)\n    authoritative_java_evidence_fingerprints: set[str] = field(default_factory=set)\n    mutation_context: TargetMutationContext | None = None\n",
        "authoritative evidence state",
    )

    text = replace_once(
        text,
        "    @property\n    def has_fresh_evidence(self) -> bool:\n        with self._lock:\n            return bool(self.evidence_fingerprints)\n\n    def record_query",
        "    @property\n    def has_fresh_evidence(self) -> bool:\n        with self._lock:\n            return bool(self.evidence_fingerprints)\n\n    @property\n    def has_authoritative_java_evidence(self) -> bool:\n        with self._lock:\n            return bool(self.authoritative_java_evidence_fingerprints)\n\n    def record_query",
        "authoritative evidence property",
    )

    text = replace_once(
        text,
        "            self.evidence_fingerprints.add(fp)\n            context = _extract_mutation_context_from_payload(value)\n",
        "            self.evidence_fingerprints.add(fp)\n            if _fresh_java_context(self.mutation_context) and _authoritative_java_evidence(value):\n                self.authoritative_java_evidence_fingerprints.add(fp)\n            context = _extract_mutation_context_from_payload(value)\n",
        "authoritative evidence recording",
    )

    text = replace_once(
        text,
        "            preferred = (\n                \"java_workspace_symbols\", \"search_project_rag\", \"external_mcp_call\", \"search_code_rag\"\n            )\n",
        "            preferred = (\n                \"search_code_rag\", \"external_mcp_call\", \"java_workspace_symbols\"\n            )\n",
        "fresh Java evidence route order",
    )

    old_recovery = '''    return adapter.generate_turn(turn_request)\n\n\ndef _sync_phase_tool_transcript(\n'''
    new_recovery = '''    try:\n        return adapter.generate_turn(turn_request)\n    except Exception as exc:\n        if not _completion_boundary_error(exc):\n            raise\n        already_recovered = any(\n            isinstance(message.get("content"), str)\n            and _ATOMIC_OUTPUT_RECOVERY_MARKER in str(message.get("content"))\n            for message in messages\n            if isinstance(message, Mapping)\n        )\n        if already_recovered:\n            raise\n        recovery_instruction = (\n            _ATOMIC_OUTPUT_RECOVERY_MARKER\n            + "\\n"\n            + _atomic_output_recovery_instruction(turn_request)\n        )\n        recovery_messages = [*messages, {"role": "system", "content": recovery_instruction}]\n        fitted_recovery = fit_messages_to_context(\n            recovery_messages, config=config, tools=request.tools\n        )\n        messages[:] = [dict(message) for message in fitted_recovery]\n        recovery_request = replace(\n            turn_request,\n            messages=tuple(messages),\n            media_paths=(),\n        )\n        emit_root_cause(\n            "atomic_output_boundary_recovery",\n            operation="generate_with_tools",\n            gate="completion_boundary",\n            result="RETRY",\n            reason=f"{type(exc).__name__}: {exc}",\n            details={"tool_choice": tool_choice},\n        )\n        return adapter.generate_turn(recovery_request)\n\n\ndef _sync_phase_tool_transcript(\n'''
    text = replace_once(text, old_recovery, new_recovery, "completion boundary recovery")

    text = replace_once(
        text,
        "        baseline_ready = state.has_fresh_evidence or not require_rag\n",
        "        baseline_ready = _target_evidence_ready(\n            state, require_rag=require_rag, fresh_java_target=fresh_java_target\n        )\n",
        "target evidence readiness",
    )

    text = replace_once(
        text,
        "            semantic_retrieval_choice=bool(require_rag and not state.has_fresh_evidence),\n",
        "            semantic_retrieval_choice=bool(require_rag and not baseline_ready),\n",
        "retrieval choice readiness",
    )

    text = replace_once(
        text,
        "            and not state.has_fresh_evidence\n",
        "            and not baseline_ready\n",
        "fresh Java guidance readiness",
    )

    text = replace_once(
        text,
        "                baseline_progress = bool(\n                    state.phase == LoopPhase.OBSERVE\n                    and require_rag\n                    and recorded\n                    and usable\n                )\n",
        "                baseline_progress = bool(\n                    state.phase == LoopPhase.OBSERVE\n                    and require_rag\n                    and recorded\n                    and usable\n                    and (\n                        not fresh_java_target\n                        or state.has_authoritative_java_evidence\n                    )\n                )\n",
        "authoritative baseline progress",
    )

    text = replace_once(
        text,
        "                        and (state.has_fresh_evidence or not require_rag)\n",
        "                        and _target_evidence_ready(\n                            state,\n                            require_rag=require_rag,\n                            fresh_java_target=fresh_java_target,\n                        )\n",
        "ACT authorization readiness",
    )

    LOOP.write_text(text, encoding="utf-8")


def rewrite_regression_test() -> None:
    TEST.write_text(r'''from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.progress_aware_tool_loop as loop
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse
from minecraft_mod_ai.model_router import _usable_rag_result


def _fresh_context() -> loop.TargetMutationContext:
    return loop.TargetMutationContext(
        target_path="src/main/java/dev/mmm/debugfixture/DebugToken.java",
        target_symbol="DebugToken",
        is_new_file=True,
        writable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        creatable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        target_pinned=True,
        evidence_source="host_task_authority",
    )


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_metadata_only_project_rag_from_trace_is_not_usable() -> None:
    payload = {
        "parsed_text": None,
        "resources": [],
        "structured_content": {
            "schema_version": "mmm/rag-result-v2",
            "sources": [{"source_id": "provenance-only", "version_scope": "26.2"}],
            "target": {"minecraft_version": "26.2", "loader": "fabric", "mappings": ""},
        },
        "text": [],
    }
    assert _usable_rag_result(payload) is False
    assert loop._authoritative_java_evidence(payload) is False


def test_generic_evidence_cannot_authorize_fresh_java_act() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    generic = {
        "schema_version": "other/evidence-v1",
        "content": "general project convention without exact API symbols",
    }
    assert state.record_evidence(generic, usable=True) is True
    assert state.has_fresh_evidence is True
    assert state.has_authoritative_java_evidence is False
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is False


def test_target_neutral_project_rag_never_authorizes_fresh_java() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    payload = {
        "schema_version": "mmm/rag-result-v2",
        "content": "Fabric items are registered during initialization.",
    }
    assert state.record_evidence(payload, usable=True) is True
    assert state.has_authoritative_java_evidence is False


def test_code_rag_with_concrete_minecraft_api_authorizes_fresh_java() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    payload = {
        "schema_version": "mmm/code-rag-result-v1",
        "hits": [
            {
                "source_path": "src/main/java/dev/mmm/ExistingItems.java",
                "text": "import net.minecraft.world.item.Item; final class ExistingItems {}",
            }
        ],
        "receipt": {"result_count": 1, "coverage_score": 1.0, "relevance_score": 1.0},
    }
    assert _usable_rag_result(payload) is True
    assert state.record_evidence(payload, usable=True) is True
    assert state.has_authoritative_java_evidence is True
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is True


def test_jdt_symbols_authorize_fresh_java_only_when_symbols_exist() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    empty = {"schema_version": "mmm/java-symbols-v1", "symbols": []}
    assert state.record_evidence(empty, usable=True) is True
    assert state.has_authoritative_java_evidence is False

    concrete = {
        "schema_version": "mmm/java-symbols-v1",
        "symbols": [{"name": "Item", "location": {"uri": "file:///workspace/Item.java"}}],
    }
    assert state.record_evidence(concrete, usable=True) is True
    assert state.has_authoritative_java_evidence is True


def test_fresh_java_observe_skips_target_neutral_project_rag() -> None:
    selected = loop._filter_tools_for_phase(
        (
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("external_mcp_call"),
            _tool("java_workspace_symbols"),
        ),
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=_fresh_context(),
        attempted_sources=frozenset(),
        semantic_retrieval_choice=True,
    )
    assert [item["function"]["name"] for item in selected] == ["search_code_rag"]


def test_completion_boundary_gets_one_in_state_recovery(monkeypatch) -> None:
    class LlamaCompletionBoundaryError(RuntimeError):
        pass

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise LlamaCompletionBoundaryError("completion token limit")
            return GenerationResponse(content="recovered")

    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )
    adapter = Adapter()
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        tools=(_tool("apply_source_edit"),),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    messages = [dict(message) for message in request.messages]
    result = loop._generate_turn_with_context_recovery(
        SimpleNamespace(),
        config=SimpleNamespace(),
        adapter=adapter,
        request=request,
        messages=messages,
        media_paths=(),
        tool_choice=request.tool_choice,
        parallel_tool_calls=False,
    )
    assert result.content == "recovered"
    assert len(adapter.requests) == 2
    recovery_text = str(adapter.requests[1].messages[-1]["content"])
    assert "MMM_ATOMIC_OUTPUT_RECOVERY_V1" in recovery_text
    assert "one small semantic edit" in recovery_text


def test_completion_boundary_recovery_does_not_loop(monkeypatch) -> None:
    class LlamaCompletionBoundaryError(RuntimeError):
        pass

    class Adapter:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            raise LlamaCompletionBoundaryError("completion token limit")

    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )
    request = GenerationRequest(
        messages=(
            {"role": "user", "content": "repair"},
            {"role": "system", "content": "MMM_ATOMIC_OUTPUT_RECOVERY_V1 already attempted"},
        ),
        tools=(_tool("apply_source_edit"),),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    with pytest.raises(LlamaCompletionBoundaryError):
        loop._generate_turn_with_context_recovery(
            SimpleNamespace(),
            config=SimpleNamespace(),
            adapter=Adapter(),
            request=request,
            messages=[dict(message) for message in request.messages],
            media_paths=(),
            tool_choice=request.tool_choice,
            parallel_tool_calls=False,
        )
''', encoding="utf-8")


def main() -> None:
    patch_loop()
    rewrite_regression_test()


if __name__ == "__main__":
    main()
