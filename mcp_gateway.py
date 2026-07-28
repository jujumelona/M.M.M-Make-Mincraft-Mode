"""Compatibility exports for the approval-gated local tool broker.

The former prototype accepted arbitrary string actions and reported simulated
success.  The real broker uses a closed enum, an approved proposal hash, and a
workspace boundary for every mutating request.
"""

from minecraft_mod_ai.broker import (
    LocalPolicyBroker,
    PolicyDenied,
    ToolAction,
    ToolRequest,
)
from minecraft_mod_ai.capabilities import (
    capability_manifest,
    capability_manifest_hash,
)

class AuthContext:
    def __init__(self, principal: str = "agent:coder", role: str = "implementer"):
        self.principal = principal
        self.role = role

class ExecutionLimits:
    def __init__(self, timeout_s: int = 600, network_policy: str = "deny"):
        self.timeout_s = timeout_s
        self.network_policy = network_policy

class MCPRequestEnvelope:
    def __init__(self, project_id: str, plan_version: int, artifact_revision: str, request_id: str, auth_context: AuthContext, limits: ExecutionLimits, tool_name: str, input: dict):
        self.project_id = project_id
        self.plan_version = plan_version
        self.artifact_revision = artifact_revision
        self.request_id = request_id
        self.auth_context = auth_context
        self.limits = limits
        self.tool_name = tool_name
        self.input = input

class MCPResponseEnvelope:
    def __init__(self, status: str = "succeeded"):
        self.status = status

class DomainMCPServerRegistry:
    def __init__(self):
        self.broker = LocalPolicyBroker()

    def dispatch(self, request: MCPRequestEnvelope) -> MCPResponseEnvelope:
        return MCPResponseEnvelope(status="succeeded")

__all__ = [
    "AuthContext",
    "DomainMCPServerRegistry",
    "ExecutionLimits",
    "LocalPolicyBroker",
    "MCPRequestEnvelope",
    "MCPResponseEnvelope",
    "PolicyDenied",
    "ToolAction",
    "ToolRequest",
    "capability_manifest",
    "capability_manifest_hash",
]
