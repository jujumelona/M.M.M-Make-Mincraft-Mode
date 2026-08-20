from __future__ import annotations

"""Fast structural runtime checks that run before any production model decode.

These checks deliberately use synthetic tool schemas and fake adapters. They are
intended to catch Python/runtime composition regressions (bad wrapper replacement,
unhashable causal state, request-field loss, broken causal progression) before a
Colab user spends time loading multi-gigabyte models.
"""

import io
import json
import threading
from contextlib import redirect_stdout
from typing import Any, Mapping

_PREFLIGHT_LOCK = threading.RLock()
_PREFLIGHT_DONE = False


class RuntimePreflightError(RuntimeError):
    pass


def _schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _CaptureAdapter:
    def __init__(self) -> None:
        self.request: Any | None = None

    def generate_turn(self, request: Any) -> Any:
        from .model_adapters import GenerationResponse

        self.request = request
        return GenerationResponse(content="preflight-ok")


def _assert_wrapper_chain() -> None:
    from .model_router import ModelRouter

    current: Any = ModelRouter._generate_with_tools
    required = (
        "_mmm_coder_tool_route_integrity_v1",
        "_mmm_progress_aware_causal_composed",
        "_mmm_writable_coder_fail_closed",
        "_mmm_dynamic_causal_frontier",
    )
    missing = [name for name in required if getattr(current, name, False) is not True]
    if missing:
        raise RuntimePreflightError(
            "final ModelRouter tool-loop composition is missing contracts: "
            + ", ".join(missing)
        )

    seen: set[int] = set()
    depth = 0
    while callable(current):
        marker = id(current)
        if marker in seen:
            raise RuntimePreflightError("ModelRouter tool-loop __wrapped__ chain contains a cycle")
        seen.add(marker)
        depth += 1
        if depth > 64:
            raise RuntimePreflightError("ModelRouter tool-loop wrapper chain is unexpectedly deep")
        current = getattr(current, "__wrapped__", None)


def _assert_repair_causal_progression() -> None:
    from .causal_tool_graph import executable_frontier, verified_state_from_messages

    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    initial = executable_frontier(
        schemas,
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
        limit=3,
        max_depth=8,
    )
    if initial != ("search_code_rag",):
        raise RuntimePreflightError(
            f"repair causal entry frontier drifted: expected search_code_rag, got {initial!r}"
        )

    payload = {
        "ok": True,
        "tool": "search_code_rag",
        "result": {
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
            }
        },
    }
    state = verified_state_from_messages(
        (
            {
                "role": "tool",
                "name": "search_code_rag",
                "content": json.dumps(payload, separators=(",", ":")),
            },
        ),
        schemas,
    )
    after_evidence = executable_frontier(
        schemas,
        state=state,
        goals=("repair",),
        limit=3,
        max_depth=8,
    )
    if after_evidence != ("apply_source_patch",):
        raise RuntimePreflightError(
            "repair causal mutation frontier is unreachable after valid RAG evidence: "
            f"{after_evidence!r}"
        )


def _assert_per_turn_adapter() -> None:
    from .causal_frontier_adapter import CausalFrontierAdapter, _query, clear_current_frontier
    from .causal_tool_frontier_contract import goals_for_query
    from .model_adapters import GenerationRequest

    payload = {
        "phase": "implement_module",
        "task": "Implement the approved Minecraft/Fabric feature in the current project.",
        # Deliberately larger than the old 12 KiB tail routing window. The explicit
        # phase/task above must remain authoritative regardless of evidence size.
        "research_context": "e" * 20_000,
    }
    messages = ({"role": "user", "content": json.dumps(payload)},)
    query = _query(messages)
    if "implement_module" not in query:
        raise RuntimePreflightError("structured implementation phase was lost from routing query")
    if tuple(goals_for_query(query)) != ("repair",):
        raise RuntimePreflightError(
            f"structured implementation request routed to {goals_for_query(query)!r}, not repair"
        )

    capture = _CaptureAdapter()
    request = GenerationRequest(
        messages=messages,
        tools=(_schema("search_code_rag"), _schema("apply_source_patch")),
        tool_choice="auto",
        task="sentinel-task",
        prompt="sentinel-prompt",
        metadata={"sentinel": "metadata"},
    )
    adapter = CausalFrontierAdapter(
        capture,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
    )
    clear_current_frontier()
    try:
        with redirect_stdout(io.StringIO()):
            response = adapter.generate_turn(request)
    finally:
        clear_current_frontier()
    if response.content != "preflight-ok" or capture.request is None:
        raise RuntimePreflightError("per-turn causal adapter did not complete its synthetic turn")
    forwarded = capture.request
    if forwarded.task != request.task or forwarded.prompt != request.prompt:
        raise RuntimePreflightError("per-turn causal adapter dropped task/prompt fields")
    if dict(forwarded.metadata) != dict(request.metadata):
        raise RuntimePreflightError("per-turn causal adapter dropped request metadata")
    names = tuple(
        str(schema.get("function", {}).get("name", "")) for schema in forwarded.tools
    )
    if names != ("search_code_rag",):
        raise RuntimePreflightError(f"unexpected initial per-turn causal surface: {names!r}")


def _assert_compaction_clone() -> None:
    from . import small_model_compacting_adapter as compaction_module
    from .model_adapters import GenerationRequest

    capture = _CaptureAdapter()
    adapter = compaction_module.CompactingAdapter(capture)
    request = GenerationRequest(
        messages=({"role": "user", "content": "preflight"},),
        task="sentinel-task",
        prompt="sentinel-prompt",
        metadata={"sentinel": "metadata"},
    )
    original = compaction_module.compact_messages
    compaction_module.compact_messages = lambda messages: (
        *tuple(messages),
        {"role": "system", "content": "synthetic compacted context"},
    )
    try:
        adapter.generate_turn(request)
    finally:
        compaction_module.compact_messages = original
    if capture.request is None:
        raise RuntimePreflightError("compaction adapter did not forward its synthetic request")
    forwarded = capture.request
    if forwarded.task != request.task or forwarded.prompt != request.prompt:
        raise RuntimePreflightError("compaction adapter dropped task/prompt fields")
    if dict(forwarded.metadata) != dict(request.metadata):
        raise RuntimePreflightError("compaction adapter dropped request metadata")


def _assert_unordered_retrieval_canonicalization() -> None:
    from .retrieval_progress import _stable_value, evidence_fingerprint

    stable = _stable_value({"facts": {"b", "a"}}, drop_volatile=False)
    if stable != {"facts": ["a", "b"]}:
        raise RuntimePreflightError(f"set-valued retrieval state is not canonical: {stable!r}")
    left = evidence_fingerprint({"facts": {"a", "b"}})
    right = evidence_fingerprint({"facts": frozenset(("b", "a"))})
    if not left or left != right:
        raise RuntimePreflightError("equivalent unordered retrieval evidence fingerprints diverged")


def run_runtime_preflight() -> None:
    """Fail fast on structural agent regressions before expensive model loading."""

    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        return
    with _PREFLIGHT_LOCK:
        if _PREFLIGHT_DONE:
            return
        checks = (
            ("wrapper-chain", _assert_wrapper_chain),
            ("repair-causal-progression", _assert_repair_causal_progression),
            ("per-turn-adapter", _assert_per_turn_adapter),
            ("compaction-clone", _assert_compaction_clone),
            ("retrieval-canonicalization", _assert_unordered_retrieval_canonicalization),
        )
        for name, check in checks:
            try:
                check()
            except RuntimePreflightError:
                raise
            except BaseException as exc:
                raise RuntimePreflightError(
                    f"runtime preflight {name!r} crashed: {type(exc).__name__}: {exc}"
                ) from exc
        _PREFLIGHT_DONE = True
        print("runtime preflight: PASS", flush=True)


__all__ = ["RuntimePreflightError", "run_runtime_preflight"]
