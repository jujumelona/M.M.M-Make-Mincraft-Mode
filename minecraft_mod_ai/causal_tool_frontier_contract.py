from __future__ import annotations

"""Expose only host-authorized tools on the verified causal next-action frontier."""

import os
import sys
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import (
    CausalFrontierAdapter,
    FrontierExecutionGate,
    authorized_tool_preference,
    authorized_tools,
    clear_current_frontier,
    remember_authorized_tools,
)
from .causal_tool_graph import executable_frontier, infer_verified_state


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _contains(value: str, markers: Sequence[str]) -> bool:
    return any(marker in value for marker in markers)


def goals_for_query(query: str) -> tuple[str, ...]:
    """Resolve one terminal target; preconditions supply intermediate states.

    Outcome verbs must outrank references to a transport or supporting capability.
    A coder request such as "implement X using external MCP" is an implementation
    goal; external MCP is only one possible means to reach it. Treating the means as
    the terminal goal can strand the coder on an observation-only frontier.
    """

    value = query.casefold()
    external = (
        "external mcp",
        "external tool",
        "mcp server",
        "외부 mcp",
        "외부 도구",
    )
    runtime_verify = (
        "runtime assertion",
        "playtest verify",
        "runtime verify",
        "런타임 검증",
        "플레이테스트 검증",
    )
    release = ("package", "release", "jar", "배포", "패키지")
    repair = (
        "repair",
        "fix",
        "patch",
        "bugfix",
        "고쳐",
        "고치",
        "수리",
        "복구",
    )
    generate = (
        "generate",
        "create",
        "new project",
        "new mod",
        "만들",
        "생성",
        "새 모드",
        "새 프로젝트",
    )
    act = (
        "modify",
        "write",
        "implement",
        "edit",
        "change",
        "수정",
        "구현",
        "변경",
    )
    verify = (
        "error",
        "fail",
        "compile",
        "diagnostic",
        "verify",
        "test",
        "validation",
        "검증",
        "오류",
        "실패",
        "테스트",
    )
    runtime = (
        "runtime",
        "playtest",
        "server",
        "client",
        "screenshot",
        "런타임",
        "플레이테스트",
        "서버",
        "클라이언트",
    )
    evidence = (
        "api",
        "version",
        "mapping",
        "yarn",
        "research",
        "evidence",
        "source search",
        "검색",
        "근거",
        "버전",
    )
    plan = ("plan", "planning", "계획", "플랜")

    if _contains(value, runtime_verify):
        return ("runtime_verify",)
    if _contains(value, release):
        return ("release",)
    if _contains(value, repair):
        return ("repair",)
    if _contains(value, generate):
        return ("generate",)
    if _contains(value, act):
        return ("act",)
    if _contains(value, verify):
        return ("verify",)
    if _contains(value, runtime):
        return ("runtime",)
    if _contains(value, evidence):
        return ("evidence",)
    if _contains(value, plan):
        return ("plan",)
    if _contains(value, external):
        return ("external",)
    return ("observe",)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _lexical_preference(
    owner: Any,
    query: str,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Rank already-authorized tools without invoking an embedding/reranker model."""

    token_fn = getattr(owner, "_tokens", None)
    document_fn = getattr(owner, "_tool_document", None)
    if not callable(token_fn) or not callable(document_fn):
        return {
            _name(schema): index
            for index, schema in enumerate(tool_schemas)
            if _name(schema)
        }

    query_tokens = set(token_fn(query))
    rows: list[tuple[float, int, str]] = []
    for index, schema in enumerate(tool_schemas):
        name = _name(schema)
        if not name:
            continue
        document = str(document_fn(schema))
        document_tokens = set(token_fn(document))
        name_tokens = set(token_fn(name.replace("_", " ")))
        score = 0.0
        if query_tokens and document_tokens:
            score += len(query_tokens & document_tokens) / max(1, len(query_tokens))
        if query_tokens and name_tokens:
            score += 2.5 * len(query_tokens & name_tokens) / max(1, len(name_tokens))
        rows.append((-score, index, name))
    rows.sort()
    return {name: rank for rank, (_score, _index, name) in enumerate(rows)}


class _FrontierRuntimeProxy:
    """Enforce the exact schemas shown on the current model turn."""

    def __init__(self, inner: Any, execution_gate: FrontierExecutionGate) -> None:
        self._inner = inner
        self._execution_gate = execution_gate

    def _require_visible(self, name: str) -> None:
        visible = self._execution_gate.visible_names()
        if visible is None:
            raise RuntimeError("No causal frontier has been published for execution.")
        if name not in visible:
            raise RuntimeError(
                f"Tool {name!r} was not exposed on the current causal frontier."
            )

    def call(self, stage: str, name: str, arguments: Mapping[str, Any]) -> Any:
        self._require_visible(name)
        return self._inner.call(stage, name, arguments)

    def call_scoped(
        self,
        stage: str,
        name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        self._require_visible(name)
        method = getattr(self._inner, "call_scoped", None)
        if callable(method):
            return method(stage, name, arguments, **kwargs)
        return self._inner.call(stage, name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _install_live_loop() -> None:
    model_router_module = sys.modules.get("minecraft_mod_ai.model_router")
    if model_router_module is None:
        return
    cls = model_router_module.ModelRouter
    current_loop = cls._generate_with_tools
    if getattr(current_loop, "_mmm_dynamic_causal_frontier", False):
        return

    @wraps(current_loop)
    def causal_tool_loop(
        self: Any,
        *,
        config: Any,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        complete_surface = tuple(authorized_tools(request.tools))
        complete_preference = dict(authorized_tool_preference())
        host_request = replace(
            request,
            tools=complete_surface,
            tool_validation_schemas=complete_surface,
            tool_choice="auto" if complete_surface else None,
            parallel_tool_calls=True if complete_surface else False,
        )
        execution_gate = FrontierExecutionGate()
        wrapped = CausalFrontierAdapter(
            adapter,
            stage=stage,
            role=role,
            require_fresh_evidence=bool(
                getattr(self, "_agent_require_fresh_evidence", False)
            ),
            frontier_limit=_env_int(
                "MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=3
            ),
            execution_gate=execution_gate,
            authorized_surface=complete_surface,
            preference=complete_preference,
            request_template=host_request,
        )
        clear_current_frontier()
        try:
            return current_loop(
                self,
                config=config,
                adapter=wrapped,
                request=host_request,
                runtime=_FrontierRuntimeProxy(runtime, execution_gate),
                stage=stage,
                role=role,
            )
        finally:
            execution_gate.clear()
            clear_current_frontier()

    causal_tool_loop._mmm_dynamic_causal_frontier = True  # type: ignore[attr-defined]
    causal_tool_loop.__wrapped__ = current_loop  # type: ignore[attr-defined]
    cls._generate_with_tools = causal_tool_loop


def install(max_agent_owner: Any) -> None:
    module = max_agent_owner
    if not hasattr(module, "select_tool_schemas"):
        module = sys.modules[str(getattr(max_agent_owner, "__module__", ""))]
    current = module.select_tool_schemas
    if not getattr(current, "_mmm_causal_tool_frontier", False):

        @wraps(current)
        def causal_frontier(
            router: Any,
            *,
            role: str,
            query: str,
            tool_schemas: Sequence[Mapping[str, Any]],
            require_fresh_evidence: bool = False,
        ) -> tuple[Mapping[str, Any], ...]:
            del router
            available = tuple(tool_schemas)
            remember_authorized_tools(available, {})
            if not available:
                return ()

            rank = _lexical_preference(module, query, available)
            remember_authorized_tools(available, rank)
            state = infer_verified_state(
                query=query,
                tool_schemas=available,
                require_fresh_evidence=require_fresh_evidence,
            )
            goals = goals_for_query(query)
            names = executable_frontier(
                available,
                state=state,
                goals=goals,
                limit=_env_int(
                    "MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=3
                ),
                max_depth=_env_int(
                    "MMM_CAUSAL_TOOL_MAX_DEPTH", 8, minimum=1, maximum=10
                ),
                preference=rank,
            )
            by_name = {_name(schema): schema for schema in available if _name(schema)}
            result = tuple(by_name[name] for name in names if name in by_name)[:3]
            print(
                "causal tool frontier:",
                f"role={role}",
                f"state={','.join(sorted(state))}",
                f"goals={','.join(goals)}",
                f"tools={','.join(_name(item) for item in result)}",
                file=sys.stderr,
                flush=True,
            )
            return result

        causal_frontier._mmm_causal_tool_frontier = True  # type: ignore[attr-defined]
        causal_frontier._mmm_no_model_tool_rerank = True  # type: ignore[attr-defined]
        causal_frontier.__wrapped__ = current  # type: ignore[attr-defined]
        module.select_tool_schemas = causal_frontier

    _install_live_loop()


__all__ = ["goals_for_query", "install"]
