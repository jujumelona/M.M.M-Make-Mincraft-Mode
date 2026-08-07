from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_exact(path: str, old: str, new: str) -> None:
    text = read(path)
    if old in text:
        write(path, text.replace(old, new))
        return
    if new and new in text:
        return
    raise RuntimeError(f"Expected reconciliation pattern missing: {path}: {old!r}")


def remove_regex(path: str, pattern: str, replacement: str, *, marker: str) -> None:
    text = read(path)
    if marker not in text:
        return
    updated, count = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one reconciliation block in {path}; found {count}"
        )
    write(path, updated)


def reconcile_mcp_server() -> None:
    path = "minecraft_mod_ai/mcp_server.py"
    replace_exact(
        path,
        '    "generate_world_ir": frozenset({"generation"}),\n'
        '    "compile_world_ir": frozenset({"generation"}),\n',
        "",
    )
    replace_exact(
        path,
        'mcp = MCPServer("M.M.M Minecraft Mod AI", version="0.7.0")',
        'mcp = MCPServer("M.M.M Minecraft Mod AI", version="0.8.0")',
    )
    remove_regex(
        path,
        r"\n@_stage_tool\(\)\ndef generate_world_ir\(.*?"
        r"\n@_stage_tool\(\)\ndef generate_geckolib_entity\(",
        "\n\n@_stage_tool()\ndef generate_geckolib_entity(",
        marker="def generate_world_ir(",
    )


def reconcile_core_service() -> None:
    remove_regex(
        "minecraft_mod_ai/mcp_tools.py",
        r"\n    def generate_world_ir\(.*?\n    def run_static_validation\(",
        "\n    def run_static_validation(",
        marker="    def generate_world_ir(",
    )


def reconcile_gateway() -> None:
    path = "mcp_gateway.py"
    replace_exact(path, '        "generate_world_ir",\n', "")
    replace_exact(path, '        "compile_world",\n', "")


def reconcile_skill_catalog() -> None:
    path = "minecraft_mod_ai/skill_catalog.py"
    replace_exact(
        path,
        '    "generate_world_ir": frozenset({"generation"}),\n'
        '    "compile_world_ir": frozenset({"generation"}),\n',
        "",
    )
    replace_exact(path, '        "generate_world_ir",\n', "")
    replace_exact(path, '        "compile_world_ir",\n', "")


def reconcile_skill_sources() -> None:
    payload_path = ROOT / "minecraft_mod_ai/packaged_skills.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    skills = payload.get("skills")
    if not isinstance(skills, dict):
        raise RuntimeError("packaged_skills.json has no skills object")
    for name in ("plan-game-design", "generate-worldgen"):
        source = ROOT / "skills" / name / "SKILL.md"
        skills[name] = source.read_text(encoding="utf-8")
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    remove_regex(
        "minecraft_mod_ai/__init__.py",
        r"\n# Remove standalone map tools from source and packaged Skill policy\.\n"
        r"from \. import skill_catalog as _skill_catalog_module\n"
        r"from \.skill_scope_contract import install as _install_skill_scope_contract\n\n"
        r"_install_skill_scope_contract\(_skill_catalog_module\)\n",
        "",
        marker="_install_skill_scope_contract",
    )
    obsolete = ROOT / "minecraft_mod_ai/skill_scope_contract.py"
    if obsolete.exists():
        obsolete.unlink()


def reconcile_external_registry() -> None:
    path = "minecraft_mod_ai/config/external_mcp_registry.yaml"
    text = read(path)
    generation_block = (
        "  mmm-generation:\n"
        "    status: enabled\n"
        "    transport: stdio\n"
        "    command: [python, -m, minecraft_mod_ai.mcp_server]\n"
        "    env: {MMM_MCP_STAGE: generation}\n"
        "    target_versions: [\"1.20.1\"]\n"
        "    trust: first_party\n"
    )
    replacement = (
        "  mmm-generation:\n"
        "    status: enabled\n"
        "    transport: stdio\n"
        "    command: [python, -m, minecraft_mod_ai.mod_generation_mcp_server]\n"
        "    target_versions: [\"1.20.1\"]\n"
        "    trust: first_party\n"
        "    purpose: mod-only Fabric project generation; no standalone map artifacts\n"
    )
    if generation_block in text:
        write(path, text.replace(generation_block, replacement))
    elif replacement not in text:
        raise RuntimeError("mmm-generation external registry block is unexpected")


def create_public_surface_test() -> None:
    path = ROOT / "tests" / "test_mod_only_public_surface.py"
    source = """from __future__ import annotations

import json
from pathlib import Path

import mcp_gateway
from minecraft_mod_ai import mcp_server
from minecraft_mod_ai.mcp_tools import MMMToolService
from minecraft_mod_ai.skill_catalog import MUTATING_TOOLS, REVIEWED_TOOL_STAGES


ROOT = Path(__file__).resolve().parents[1]
REMOVED = {"generate_world_ir", "compile_world_ir"}


def test_primary_mcp_has_no_standalone_map_tools() -> None:
    assert REMOVED.isdisjoint(mcp_server._TOOL_STAGES)
    assert REMOVED.isdisjoint(mcp_server._tool_names_for_stage("all"))


def test_compatibility_gateway_has_no_map_compiler_surface() -> None:
    assert "generate_world_ir" not in mcp_gateway._CORE_TOOLS
    assert "compile_world" not in mcp_gateway._PRODUCTION_TOOLS
    assert not hasattr(MMMToolService, "generate_world_ir")


def test_skill_policy_physically_excludes_removed_tools() -> None:
    assert REMOVED.isdisjoint(REVIEWED_TOOL_STAGES)
    assert REMOVED.isdisjoint(MUTATING_TOOLS)
    packaged = json.loads(
        (ROOT / "minecraft_mod_ai/packaged_skills.json").read_text(
            encoding="utf-8"
        )
    )
    selected = "\n".join(
        packaged["skills"][name]
        for name in ("plan-game-design", "generate-worldgen")
    )
    assert "generate_world_ir" not in selected
    assert "compile_world_ir" not in selected


def test_all_generation_registries_use_mod_only_server() -> None:
    root_config = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    plugin_config = json.loads(
        (ROOT / "plugins/mmm-minecraft-mod-ai/.mcp.json").read_text(
            encoding="utf-8"
        )
    )
    expected = ["-m", "minecraft_mod_ai.mod_generation_mcp_server"]
    for config in (root_config, plugin_config):
        args = config["mcpServers"]["mmm-generation"]["args"]
        assert args == expected
    registry = (
        ROOT / "minecraft_mod_ai/config/external_mcp_registry.yaml"
    ).read_text(encoding="utf-8")
    assert "minecraft_mod_ai.mod_generation_mcp_server" in registry
    assert "env: {MMM_MCP_STAGE: generation}" not in registry
"""
    path.write_text(source, encoding="utf-8")


def main() -> None:
    reconcile_mcp_server()
    reconcile_core_service()
    reconcile_gateway()
    reconcile_skill_catalog()
    reconcile_skill_sources()
    reconcile_external_registry()
    create_public_surface_test()


if __name__ == "__main__":
    main()
