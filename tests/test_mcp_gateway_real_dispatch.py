from mcp_gateway import (
    AuthContext,
    DomainMCPServerRegistry,
    ExecutionLimits,
    MCPRequestEnvelope,
)


class FakeService:
    def search_project_rag(self, **kwargs):
        return {"query": kwargs["query"], "sources": ["real-result"]}


def request(tool_name: str, input: dict):
    return MCPRequestEnvelope(
        project_id="test_project",
        plan_version=1,
        artifact_revision="r1",
        request_id="req-1",
        auth_context=AuthContext(),
        limits=ExecutionLimits(),
        tool_name=tool_name,
        input=input,
    )


def test_dispatch_returns_actual_result() -> None:
    gateway = DomainMCPServerRegistry(service=FakeService())
    response = gateway.dispatch(request("search_project_rag", {"query": "gametest"}))
    assert response.status == "succeeded"
    assert response.result == {"query": "gametest", "sources": ["real-result"]}


def test_dispatch_rejects_unknown_tool() -> None:
    gateway = DomainMCPServerRegistry(service=FakeService())
    response = gateway.dispatch(request("pretend_success", {}))
    assert response.status == "failed"
    assert "disallowed" in response.error
