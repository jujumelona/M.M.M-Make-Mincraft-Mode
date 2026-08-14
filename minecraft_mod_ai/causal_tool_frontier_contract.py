from __future__ import annotations

"""Expose only host-authorized tools on the verified causal next-action frontier."""

import os
import sys
from functools import wraps
from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import CausalFrontierAdapter, remember_authorized_tools
from .causal_tool_graph import executable_frontier, infer_verified_state


def _name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _contains(value: str, markers: Sequence[str]) -> bool:
    return any(marker in value for marker in markers)


def goals_for_query(query: str) -> tuple[str, ...]:
    """Resolve the request to one terminal causal target, not a keyword AND-bag.

    Intermediate requirements such as evidence/project observation are encoded in
    transition preconditions and are therefore reached by the shortest-path solver.
    Choosing one terminal state prevents generic words such as ``inspect`` from
    accidentally requiring an unrelated second terminal state.
    """

    value = query.casefold()
    external = ("external mcp", "mcp server", "capability", "외부 mcp")
    runtime_verify = (
        "runtime assertion",
        "playtest verify",
        "runtime verify",
        "런타임 검증",
        "플레이테스트 검증",
    )
    release = ("package", "release", "jar", "배포", "패키지")
    act = (
        "generate",
        "create",
        "patch",
        "modify",
        "fix",
        "write",
        "implement",
        "repair",
        "만들",
        "생성",
        "수정",
        "고쳐",
        "구현",
        "수리",
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

    # Terminal intents are ordered by the state the user ultimately wants.  A
    # mutation request that also mentions errors/API still targets project_changed;
    # its causal path acquires the required evidence first.  Verification requests
    # without mutation words target verified.  This keeps verification from becoming
    # stale merely because both words appeared in the same prompt.
    if _contains(value, external):
        return ("external",)
    if _contains(value, runtime_verify):
        return ("runtime_verify",)
    if _contains(value, release):
        return ("release",)
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
    return ("observe",)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


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
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        wrapped = CausalFrontierAdapter(
            adapter,
            stage=stage,
            role=role,
            require_fresh_evidence=bool(getattr(self, "_agent_require_fresh_evidence", False)),
            frontier_limit=_env_int("MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=3),
        )
        return current_loop(
            self,
            adapter=wrapped,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

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
            available = tuple(tool_schemas)
            remember_authorized_tools(available)
            if not available:
                return ()

            # Semantic/tool relevance remains a tie-breaker only after the causal
            # policy has established which next transitions are executable.
            ranked = tuple(
                current(
                    router,
                    role=role,
                    query=query,
                    tool_schemas=available,
                    require_fresh_evidence=require_fresh_evidence,
                )
            )
            rank = {_name(schema): index for index, schema in enumerate(ranked)}
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
                limit=_env_int("MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=3),
                max_depth=_env_int("MMM_CAUSAL_TOOL_MAX_DEPTH", 8, minimum=1, maximum=10),
            )
            by_name = {_name(schema): schema for schema in available if _name(schema)}
            selected = [by_name[name] for name in names if name in by_name]
            selected.sort(key=lambda schema: (rank.get(_name(schema), len(rank)), _name(schema)))
            result = tuple(selected[:3])
            print(
                "causal tool frontier:",
                f"role={role}",
                f"state={','.join(sorted(state))}",
                f"goals={','.join(goals)}",
                f"tools={','.join(_name(item) for item in result)}",
                flush=True,
            )
            return result

        causal_frontier._mmm_causal_tool_frontier = True  # type: ignore[attr-defined]
        causal_frontier.__wrapped__ = current  # type: ignore[attr-defined]
        module.select_tool_schemas = causal_frontier

    _install_live_loop()


__all__ = ["goals_for_query", "install"]
