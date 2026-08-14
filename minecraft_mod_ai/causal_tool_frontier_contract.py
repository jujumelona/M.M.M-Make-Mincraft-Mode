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
    """Resolve one terminal target; preconditions supply intermediate states."""

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

    if _contains(value, external):
        return ("external",)
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
            remember_authorized_tools(available, {})
            if not available:
                return ()

            # This prior selector is relevance-only. Its order is retained as a
            # preference map, but causal legality/minimum total cost is computed over
            # the COMPLETE authorized surface below.
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
                limit=_env_int("MMM_CAUSAL_TOOL_FRONTIER_MAX", 3, minimum=1, maximum=3),
                max_depth=_env_int("MMM_CAUSAL_TOOL_MAX_DEPTH", 8, minimum=1, maximum=10),
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
                flush=True,
            )
            return result

        causal_frontier._mmm_causal_tool_frontier = True  # type: ignore[attr-defined]
        causal_frontier.__wrapped__ = current  # type: ignore[attr-defined]
        module.select_tool_schemas = causal_frontier

    _install_live_loop()


__all__ = ["goals_for_query", "install"]
