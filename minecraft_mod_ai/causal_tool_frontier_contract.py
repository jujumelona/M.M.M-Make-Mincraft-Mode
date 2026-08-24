from __future__ import annotations

"""Compatibility surface for the retired causal tool router.

Tool selection now has one owner: ``small_model_max_agent_contract.select_tool_schemas``.
The live model/tool loop receives that stable security-filtered set and lets the model
choose among those tools with ordinary function-calling semantics. This module keeps
only intent classification helpers required by older imports; it no longer wraps the
selector, rewrites the live tool surface, or forces per-turn actions.
"""

from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import CausalFrontierAdapter


def _contains(value: str, markers: Sequence[str]) -> bool:
    return any(marker in value for marker in markers)


def goals_for_query(query: str) -> tuple[str, ...]:
    value = str(query).casefold()
    if "implement_module" in value:
        return ("repair",)
    if _contains(value, ("runtime assertion", "playtest verify", "runtime verify", "런타임 검증", "플레이테스트 검증")):
        return ("runtime_verify",)
    if _contains(value, ("package", "release", "jar", "배포", "패키지")):
        return ("release",)
    if _contains(value, ("repair", "fix", "patch", "bugfix", "고쳐", "고치", "수리", "복구")):
        return ("repair",)
    if _contains(value, ("generate", "create", "new project", "new mod", "만들", "생성", "새 모드", "새 프로젝트")):
        return ("generate",)
    if _contains(value, ("modify", "write", "implement", "edit", "change", "수정", "구현", "변경")):
        return ("act",)
    if _contains(value, ("error", "fail", "compile", "diagnostic", "verify", "test", "validation", "검증", "오류", "실패", "테스트")):
        return ("verify",)
    if _contains(value, ("runtime", "playtest", "server", "client", "screenshot", "런타임", "플레이테스트", "서버", "클라이언트")):
        return ("runtime",)
    if _contains(value, ("api", "version", "mapping", "yarn", "research", "evidence", "source search", "검색", "근거", "버전")):
        return ("evidence",)
    if _contains(value, ("plan", "planning", "계획", "플랜")):
        return ("plan",)
    if _contains(value, ("external mcp", "external tool", "mcp server", "외부 mcp", "외부 도구")):
        return ("external",)
    return ("observe",)


class _FrontierRuntimeProxy:
    """Legacy pass-through retained only for import compatibility."""

    def __init__(self, inner: Any, execution_gate: Any | None = None) -> None:
        self._inner = inner
        self._execution_gate = execution_gate

    def call(self, stage: str, name: str, arguments: Mapping[str, Any]) -> Any:
        return self._inner.call(stage, name, arguments)

    def call_scoped(self, stage: str, name: str, arguments: Mapping[str, Any], **kwargs: Any) -> Any:
        method = getattr(self._inner, "call_scoped", None)
        if callable(method):
            return method(stage, name, arguments, **kwargs)
        return self._inner.call(stage, name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install(max_agent_owner: Any) -> None:
    """Retired compatibility hook; intentionally performs no runtime mutation."""

    del max_agent_owner


__all__ = ["_FrontierRuntimeProxy", "goals_for_query", "install"]
