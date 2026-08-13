from __future__ import annotations

from pathlib import Path

path = Path("minecraft_mod_ai/runtime_bootstrap.py")
text = path.read_text(encoding="utf-8")

old_exec = "    install_execution_efficiency(\n        complete_planner_module=complete_planner,\n        work_graph_module=work_graph,\n    )"
live_exec = "    install_execution_efficiency(\n        work_graph_module=work_graph,\n    )"
compact_exec = "    install_execution_efficiency(work_graph_module=work_graph)"
if old_exec in text:
    text = text.replace(old_exec, live_exec, 1)
elif live_exec not in text and compact_exec not in text:
    raise SystemExit("execution-efficiency bootstrap call not recognized")

start = text.find("def _install_public_boundary_contracts() -> None:\n")
if start < 0:
    raise SystemExit("public-boundary installer not found")
end = text.find("\n\ndef ", start + 1)
if end < 0:
    end = len(text)

live_boundary = '''def _install_public_boundary_contracts() -> None:
    """Install the live MCP/release boundary owners last."""
    from . import mcp_tools, production_tools
    from .platform_mcp_contract import install as install_platform_mcp
    from .platform_release_contract import install as install_platform_release

    install_platform_mcp(mcp_tools, production_tools)
    install_platform_release(mcp_tools)
'''
text = text[:start] + live_boundary + text[end:]

forbidden = (
    "agent_tool_calling_contract",
    "install_agent_tools(",
    "platform_mcp_compatibility_contract",
    "install_platform_mcp_compatibility(",
    "platform_policy_runtime_contract",
    "install_platform_policy_runtime(",
)
if any(value in text for value in forbidden):
    raise SystemExit("deleted public-boundary owner still referenced")
if "install_platform_mcp(mcp_tools, production_tools)" not in text:
    raise SystemExit("live platform MCP owner has wrong signature")

path.write_text(text, encoding="utf-8")
print("runtime bootstrap rebuilt from live owners")
