from __future__ import annotations

from mcp_gateway import (
    AuthContext,
    DomainMCPServerRegistry,
    ExecutionLimits,
    MCPRequestEnvelope,
)


class _ProductionService:
    def __init__(self) -> None:
        self.seen_timeout = None
        self.calls = 0

    def java_diagnostics(
        self,
        project_root: str,
        relative_files=None,
        timeout_seconds: int = 60,
    ):
        self.calls += 1
        self.seen_timeout = timeout_seconds
        return {"project_root": project_root, "timeout_seconds": timeout_seconds}


def _request(timeout_s: int, **inputs):
    return MCPRequestEnvelope(
        project_id="project",
        plan_version=1,
        artifact_revision="rev",
        request_id="request",
        auth_context=AuthContext(),
        limits=ExecutionLimits(timeout_s=timeout_s),
        tool_name="java_diagnostics",
        input={"project_root": "project", **inputs},
    )


def test_envelope_timeout_is_injected_into_cooperative_tool() -> None:
    production = _ProductionService()
    registry = DomainMCPServerRegistry(
        service=object(),
        production_service=production,
    )
    response = registry.dispatch(_request(7))

    assert response.status == "succeeded"
    assert production.seen_timeout == 7


def test_tool_timeout_cannot_exceed_envelope_timeout() -> None:
    production = _ProductionService()
    registry = DomainMCPServerRegistry(
        service=object(),
        production_service=production,
    )
    response = registry.dispatch(_request(5, timeout_seconds=99))

    assert response.status == "succeeded"
    assert production.seen_timeout == 5


def test_invalid_envelope_timeout_fails_before_tool_execution() -> None:
    production = _ProductionService()
    registry = DomainMCPServerRegistry(
        service=object(),
        production_service=production,
    )
    response = registry.dispatch(_request(0))

    assert response.status == "failed"
    assert "positive integer" in str(response.error)
    assert production.calls == 0
