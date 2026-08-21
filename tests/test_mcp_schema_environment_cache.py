from __future__ import annotations

import threading

from minecraft_mod_ai.mcp_schema_integrity_contract import ensure_schema_environment


class _Runtime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._schema_cache = {}
        self._allowed_tool_cache = {}
        self.env = {"FEATURE": "a"}

    @staticmethod
    def _stage(stage: str) -> str:
        return stage.strip().lower()

    def _child_env(self, stage: str) -> dict[str, str]:
        return {"MMM_MCP_STAGE": stage, **self.env}


def test_unchanged_environment_keeps_hot_schema_cache() -> None:
    runtime = _Runtime()
    first = ensure_schema_environment(runtime, "generation")
    runtime._schema_cache["generation"] = ("cached",)
    runtime._allowed_tool_cache["generation"] = frozenset({"cached"})

    second = ensure_schema_environment(runtime, "generation")

    assert second == first
    assert runtime._schema_cache["generation"] == ("cached",)
    assert runtime._allowed_tool_cache["generation"] == frozenset({"cached"})


def test_environment_drift_invalidates_schema_and_allowed_tool_caches() -> None:
    runtime = _Runtime()
    first = ensure_schema_environment(runtime, "generation")
    runtime._schema_cache["generation"] = ("old",)
    runtime._allowed_tool_cache["generation"] = frozenset({"old"})

    runtime.env["FEATURE"] = "b"
    second = ensure_schema_environment(runtime, "generation")

    assert second != first
    assert "generation" not in runtime._schema_cache
    assert "generation" not in runtime._allowed_tool_cache


def test_preexisting_cache_is_invalidated_when_environment_owner_is_first_installed() -> None:
    runtime = _Runtime()
    runtime._schema_cache["generation"] = ("pre-install",)
    runtime._allowed_tool_cache["generation"] = frozenset({"pre-install"})

    ensure_schema_environment(runtime, "generation")

    assert "generation" not in runtime._schema_cache
    assert "generation" not in runtime._allowed_tool_cache
