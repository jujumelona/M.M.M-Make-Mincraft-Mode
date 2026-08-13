from __future__ import annotations

from pathlib import Path


MIGRATION = Path(".github/scripts/upgrade_adaptive_production_agent.py")
CAPS = Path("minecraft_mod_ai/agent_capability_context.py")


def patch_migration_regex() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    source = source.replace(
        "rf'(\"{re.escape(name)}\"\\s*:\\s*frozenset\\()(?P<set>\\{{[^}}]+\\}})(\\))'",
        "rf'(\"{re.escape(name)}\"\\s*:\\s*frozenset\\()\\s*(?P<set>\\{{[^}}]+\\}})\\s*(\\))'",
    )
    source = source.replace(
        "rf'(\"{name}\"\\s*:\\s*frozenset\\()(?P<set>\\{{[^}}]+\\}})(\\))'",
        "rf'(\"{name}\"\\s*:\\s*frozenset\\()\\s*(?P<set>\\{{[^}}]+\\}})\\s*(\\))'",
    )
    source = source.replace(
        "\nimport json\nimport threading\n",
        "\nimport threading\n",
        1,
    )
    MIGRATION.write_text(source, encoding="utf-8")


def patch_routing_policy() -> None:
    text = CAPS.read_text(encoding="utf-8")
    start = text.find('        "routing_policy": (\n')
    if start < 0:
        raise SystemExit("routing_policy field not found")
    end_marker = "        ),\n    }"
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit("routing_policy end not found")
    end += len("        ),\n")
    desired = '''        "routing_policy": (
            "Choose every relevant Skill route, not every route indiscriminately. "
            "During production use an adaptive loop: retrieve fresh project/API evidence; "
            "inspect RAG receipt quality; reformulate or switch evidence source when weak; "
            "generate or repair; consume compiler/JDT/runtime feedback; then retrieve again "
            "when new uncertainty or errors appear. Never guess exact Minecraft/Fabric/API "
            "facts from parametric memory when reviewed evidence tools can resolve them. "
            "Use model_tools directly. host_owned_tools belong to the durable host pipeline "
            "and must not be recreated recursively. For an external MCP capability, use "
            "external_mcp_schema when its live arguments are unknown, then external_mcp_call. "
            "Run independent read-only evidence routes in parallel when useful; keep state "
            "changes ordered and skip unrelated tools."
        ),
'''
    CAPS.write_text(text[:start] + desired + text[end:], encoding="utf-8")


def main() -> None:
    patch_migration_regex()
    patch_routing_policy()


if __name__ == "__main__":
    main()
