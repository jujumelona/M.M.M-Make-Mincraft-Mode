
from minecraft_mod_ai.blockbench_client import allowed_blockbench_operations
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS, validate_skill_catalog
from minecraft_mod_ai.system_pack_generator import supported_system_packs


def test_external_mcp_registry_is_version_locked() -> None:
    registry = ExternalMCPRegistry().public_dict()["servers"]
    assert registry["mmm-frontdoor"]["env"]["MMM_MCP_STAGE"] == "frontdoor"
    generation = registry["mmm-generation"]
    assert generation["command"] == [
        "python",
        "-m",
        "minecraft_mod_ai.mcp_stdio_entrypoint",
        "minecraft_mod_ai.mod_generation_mcp_server",
    ]
    assert "env" not in generation
    minecraft_dev = registry["minecraft-dev"]
    assert minecraft_dev["status"] == "enabled"
    assert minecraft_dev["version_policy"] == "provider_reported"
    assert minecraft_dev["command"][-1] == "@mcdxai/minecraft-dev-mcp"
    assert "fabric" in minecraft_dev["loaders"]
    assert registry["playwright"]["command"][2] == "@playwright/mcp@0.0.78"
    assert registry["gdmc"]["status"] == "incompatible_by_default"


def test_canonical_skill_catalog_is_complete() -> None:
    report = validate_skill_catalog()
    assert report["passed"], report["findings"]
    assert set(report["contracts"]) == set(CANONICAL_SKILLS)
    assert len(CANONICAL_SKILLS) == len(set(CANONICAL_SKILLS))


def test_restricted_blockbench_tools_have_no_shell_or_script() -> None:
    operations = set(allowed_blockbench_operations())
    assert {"validate_uv", "render_preview", "export_geckolib"} <= operations
    assert not operations & {"shell", "script", "eval", "arbitrary_file_write"}


def test_system_pack_ids_are_explicit() -> None:
    assert set(supported_system_packs()) == {
        "quest-system",
        "class-skill-system",
        "economy-shop",
        "gui-networking",
        "party-guild",
    }
