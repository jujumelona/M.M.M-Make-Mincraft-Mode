from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime, AgentToolRuntimeError


class _SessionContext:
    def __init__(self, raw) -> None:
        self._raw = raw

    async def __aenter__(self):
        raw = self._raw

        class _Session:
            async def call_tool(self, name: str, *, arguments):
                del name, arguments
                return raw

        return _Session()

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb


class _Runtime(AgentToolRuntime):
    def __init__(self, raw) -> None:
        super().__init__(profile="test")
        self._raw = raw

    def _session(self, stage: str):
        assert stage == "generation"
        return _SessionContext(self._raw)


def _invoke(runtime: AgentToolRuntime) -> None:
    anyio.run(
        runtime._call_tool_async,
        "generation",
        "apply_source_patch",
        {},
    )


def test_mcp_error_preserves_workspace_impact_marker_and_redacts_secrets() -> None:
    raw = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=(
            SimpleNamespace(
                text=(
                    "[mmm-workspace-impact:unchanged] source precondition rejected; "
                    "api_key=supersecretvalue"
                )
            ),
        ),
    )

    with pytest.raises(AgentToolRuntimeError) as raised:
        _invoke(_Runtime(raw))

    message = str(raised.value)
    assert "[mmm-workspace-impact:unchanged]" in message
    assert "[workspace_impact=unchanged]" in message
    assert "api_key=[REDACTED]" in message
    assert "supersecretvalue" not in message


def test_mcp_error_collapses_markers_before_bounded_diagnostic_truncation() -> None:
    raw = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=(
            SimpleNamespace(
                text=(
                    "[mmm-workspace-impact:unchanged] "
                    + ("diagnostic " * 900)
                    + "[mmm-workspace-impact:drift]"
                )
            ),
        ),
    )

    with pytest.raises(AgentToolRuntimeError) as raised:
        _invoke(_Runtime(raw))

    message = str(raised.value)
    assert len(message) < 9 * 1024
    assert message.endswith("[workspace_impact=drift]")


def test_mcp_error_without_detail_remains_generic_and_fail_closed() -> None:
    raw = SimpleNamespace(
        is_error=True,
        structured_content=None,
        content=(),
    )

    with pytest.raises(
        AgentToolRuntimeError,
        match=r"^MCP tool 'apply_source_patch' returned an error$",
    ):
        _invoke(_Runtime(raw))
