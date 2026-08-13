from pathlib import Path

path = Path("tests/test_runtime_bootstrap_clean.py")
text = path.read_text(encoding="utf-8")

stale_calls = (
    '        "install_runtime_helpers(",\n',
    '        "install_runtime_helper_json_deadline(",\n',
    '        "install_mcp_target_validation(",\n',
    '        "install_external_mcp_bridge_safety(",\n',
    '        "install_mcp_runtime(",\n',
    '        "install_mcp_federation(",\n',
    '        "install_mcp_repair_batch(",\n',
    '        "install_mcp_repair_diagnostic_shape(",\n',
    '        "install_skill_policy(",\n',
)
for call in stale_calls:
    if call in text:
        text = text.replace(call, "", 1)

anchor = '        "install_planner_production_page(",\n'
if '        "install_platform_mcp(",\n' not in text:
    if anchor not in text:
        raise SystemExit("live bootstrap required_once anchor not found")
    text = text.replace(
        anchor,
        anchor + '        "install_platform_mcp(",\n        "install_platform_release(",\n',
        1,
    )

old_order = '''    assert source.index("install_mcp_target_validation(") < source.index(
        "install_external_mcp_bridge_safety("
    )
'''
if old_order in text:
    text = text.replace(old_order, "", 1)

stale_owner_assert_anchor = '    assert "platform_mcp_compatibility_contract" not in source\n'
extra = '''    assert "agent_tool_calling_contract" not in source
    assert "platform_policy_runtime_contract" not in source
'''
# Add explicit stale-owner checks to the runtime-bootstrap test only.
marker = '''    assert "integrated_contract_bootstrap" not in source
    assert "final_architecture_contract" not in source
    assert "platform_mcp_compatibility_contract" not in source
'''
if extra.strip() not in text:
    if marker not in text:
        raise SystemExit("runtime bootstrap stale-owner assertion marker not found")
    text = text.replace(marker, marker + extra, 1)

for stale in (
    "install_runtime_helpers(",
    "install_runtime_helper_json_deadline(",
    "install_mcp_target_validation(",
    "install_external_mcp_bridge_safety(",
    "install_mcp_runtime(",
    "install_mcp_federation(",
    "install_mcp_repair_batch(",
    "install_mcp_repair_diagnostic_shape(",
    "install_skill_policy(",
):
    if stale in text:
        raise SystemExit(f"stale installer expectation remains: {stale}")

path.write_text(text, encoding="utf-8")
print("bootstrap contract test now follows live owners")
