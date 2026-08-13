from __future__ import annotations

import json
import re
from pathlib import Path


MCP = Path("minecraft_mod_ai/mcp_server.py")
SKILL_CATALOG = Path("minecraft_mod_ai/skill_catalog.py")
CAPS = Path("minecraft_mod_ai/agent_capability_context.py")
MODEL = Path("minecraft_mod_ai/model_router.py")
PARALLEL = Path("minecraft_mod_ai/production_tool_parallel_contract.py")
REPAIR = Path("minecraft_mod_ai/repair_engine.py")
GATHER = Path("skills/gather-adaptive-minecraft-evidence/SKILL.md")
ROLES = Path("config/agent_roles.yaml")
PACKAGED = Path("minecraft_mod_ai/packaged_skills.json")
TEST = Path("tests/test_adaptive_production_agent.py")


_STAGE_ASSIGNMENTS = {
    "discover_ecosystem_resources": ("frontdoor", "planning", "research", "generation"),
    "inspect_modrinth_project": ("planning", "research", "generation"),
    "inspect_github_repository": ("planning", "research", "generation"),
    "inspect_huggingface_model": ("planning", "research", "generation"),
    "build_technology_radar": ("frontdoor", "planning", "research", "generation"),
    "assess_technology_compatibility": ("planning", "research", "generation"),
    "search_project_rag": ("frontdoor", "planning", "research", "generation", "quality"),
    "inspect_existing_mod": ("frontdoor", "planning", "research", "generation", "quality"),
    "java_diagnostics": ("generation", "quality"),
    "java_workspace_symbols": ("generation", "quality"),
}


def _set_stage_entry(source: str, name: str, stages: tuple[str, ...]) -> str:
    pattern = re.compile(
        rf'(?P<prefix>"{re.escape(name)}"\s*:\s*frozenset\()\s*'
        rf'(?P<body>\{{[^}}]*\}})\s*(?P<suffix>\))',
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise SystemExit(f"stage entry missing: {name}")
    rendered = "{" + ", ".join(f'"{stage}"' for stage in stages) + "}"
    return source[: match.start("body")] + rendered + source[match.end("body") :]


def patch_stage_maps() -> None:
    for path in (MCP, SKILL_CATALOG):
        source = path.read_text(encoding="utf-8")
        for name, stages in _STAGE_ASSIGNMENTS.items():
            source = _set_stage_entry(source, name, stages)
        compile(source, str(path), "exec")
        path.write_text(source, encoding="utf-8")


def patch_role_filter() -> None:
    source = CAPS.read_text(encoding="utf-8")
    source = source.replace(
        "from .skill_catalog import SkillContract, compile_skill_catalog\n",
        "from .skill_catalog import (\n"
        "    REVIEWED_TOOL_STAGES,\n"
        "    SkillContract,\n"
        "    compile_skill_catalog,\n"
        ")\n",
        1,
    )
    old_return = '''    return tuple(
        schema
        for schema in tool_schemas
        if (_schema_tool_name(schema) in allowed)
    )
'''
    new_return = '''    selected_stage = stage.strip().lower()
    return tuple(
        schema
        for schema in tool_schemas
        if (
            (name := _schema_tool_name(schema)) in allowed
            and selected_stage in REVIEWED_TOOL_STAGES.get(name, frozenset())
        )
    )
'''
    if old_return not in source and new_return not in source:
        raise SystemExit("role-filter return anchor missing")
    source = source.replace(old_return, new_return, 1)

    old_guard = '''    selected_tool = tool.strip()
    if not selected_tool or selected_tool in _EXTERNAL_AGENT_TOOLS:
        return ()
    return tuple(
'''
    new_guard = '''    selected_tool = tool.strip()
    selected_stage = stage.strip().lower()
    if (
        not selected_tool
        or selected_tool in _EXTERNAL_AGENT_TOOLS
        or selected_stage not in REVIEWED_TOOL_STAGES.get(selected_tool, frozenset())
    ):
        return ()
    return tuple(
'''
    if old_guard not in source and new_guard not in source:
        raise SystemExit("skills_for_tool guard anchor missing")
    source = source.replace(old_guard, new_guard, 1)
    compile(source, str(CAPS), "exec")
    CAPS.write_text(source, encoding="utf-8")


def patch_model_role_receipts() -> None:
    source = MODEL.read_text(encoding="utf-8")
    source = source.replace(
        "skills_for_tool(stage, call.name)",
        "skills_for_tool(stage, call.name, model_role=role)",
    )
    compile(source, str(MODEL), "exec")
    MODEL.write_text(source, encoding="utf-8")


def patch_serialized_rag_refresh() -> None:
    source = PARALLEL.read_text(encoding="utf-8")
    old = '''        with _index_lock(target):
            # The original method also checks via _new_file(), but that check was
            # previously outside any mutual exclusion. Repeat it inside the lock to
            # close the check/build/atomic-replace TOCTOU window.
            if target.exists():
                raise FileExistsError(target)
            return current(
'''
    new = '''        with _index_lock(target):
            # A live production index is a replaceable derived artifact. Serialize
            # rebuilds by canonical path, then let ProductionToolService validate the
            # existing target and ProjectRAGIndex atomically replace its contents.
            return current(
'''
    if old not in source and new not in source:
        raise SystemExit("parallel RAG refresh anchor missing")
    source = source.replace(old, new, 1)
    compile(source, str(PARALLEL), "exec")
    PARALLEL.write_text(source, encoding="utf-8")


def patch_gather_skill() -> None:
    source = GATHER.read_text(encoding="utf-8")
    source = source.replace(
        "stages:\n  - research\n",
        "stages:\n  - research\n  - generation\n  - quality\n",
        1,
    )
    old_tools = '''allowed_tools:
  - search_project_rag
  - index_project_rag
  - search_code_rag
  - inspect_existing_mod
'''
    new_tools = '''allowed_tools:
  - search_project_rag
  - index_project_rag
  - search_code_rag
  - inspect_existing_mod
  - discover_ecosystem_resources
  - inspect_modrinth_project
  - inspect_github_repository
  - assess_technology_compatibility
  - java_diagnostics
  - java_workspace_symbols
'''
    if old_tools not in source and new_tools not in source:
        raise SystemExit("gather-skill tool anchor missing")
    source = source.replace(old_tools, new_tools, 1)
    GATHER.write_text(source, encoding="utf-8")


def patch_coder_role() -> None:
    source = ROLES.read_text(encoding="utf-8")
    marker = '''      - patch-existing-project
      - compile-and-repair
'''
    replacement = '''      - patch-existing-project
      - compile-and-repair
      - gather-adaptive-minecraft-evidence
'''
    if replacement not in source:
        if marker not in source:
            raise SystemExit("MinecraftCoder skill anchor missing")
        source = source.replace(marker, replacement, 1)
    ROLES.write_text(source, encoding="utf-8")


def patch_repair_model_boundary() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    source = source.replace(
        "        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)\n",
        "",
        1,
    )
    source = re.sub(
        r'''                # Rebuild code RAG from the exact post-previous-patch project before\n'''
        r'''(?:                .*\n|\n)*?'''
        r'''                patch = self\._request_patch\(evidence, context\)\n''',
        "                patch = self._request_patch(evidence, context)\n",
        source,
        count=1,
    )
    signature = '''    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompt = {
'''
    replacement = '''    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        active = _ACTIVE_REPAIR_PROJECT_INDEX.get()
        if active is None:
            raise RepairEngineError("Repair model call has no active project index.")
        root, project_index = active
        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)
        from .production_tools import ProductionToolService

        manifest = project_index.manifest_receipt()
        ProductionToolService(
            workspace_root=root.parent,
            profile=self.router.profile,
        ).index_project_rag(
            [root.name],
            metadata=_repair_rag_metadata(manifest),
            semantic=False,
        )

        prompt = {
'''
    if replacement not in source:
        if signature not in source:
            raise SystemExit("repair request boundary anchor missing")
        source = source.replace(signature, replacement, 1)
    compile(source, str(REPAIR), "exec")
    REPAIR.write_text(source, encoding="utf-8")


def regenerate_packaged_skills() -> None:
    from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS, compile_skill_catalog

    root = Path("skills").resolve()
    skills = {
        name: (root / name / "SKILL.md").read_text(encoding="utf-8")
        for name in CANONICAL_SKILLS
    }
    contracts = {
        name: contract.to_dict()
        for name, contract in compile_skill_catalog(root).items()
    }
    payload = {
        "schema_version": "mmm/packaged-skills-v3",
        "skills": skills,
        "contracts": contracts,
    }
    PACKAGED.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def patch_generated_tests() -> None:
    source = TEST.read_text(encoding="utf-8")
    source = source.replace("import json\n", "", 1)
    old = '''def test_reviewed_stage_map_matches_live_mcp_map(monkeypatch) -> None:
    monkeypatch.setenv("MMM_MCP_STAGE", "all")
    from minecraft_mod_ai import mcp_server
    assert mcp_server._TOOL_STAGES == REVIEWED_TOOL_STAGES
'''
    new = '''def test_reviewed_stage_map_matches_live_mcp_map(monkeypatch) -> None:
    monkeypatch.setenv("MMM_MCP_STAGE", "all")
    from minecraft_mod_ai import mcp_server

    for name, stages in REVIEWED_TOOL_STAGES.items():
        live = mcp_server._TOOL_STAGES[name]
        if name == "discover_mmm_capabilities":
            assert live - {"all"} == stages
        else:
            assert live == stages
'''
    if old not in source and new not in source:
        raise SystemExit("generated stage-map test anchor missing")
    source = source.replace(old, new, 1)
    TEST.write_text(source, encoding="utf-8")
    compile(source, str(TEST), "exec")


def main() -> None:
    patch_stage_maps()
    patch_role_filter()
    patch_model_role_receipts()
    patch_serialized_rag_refresh()
    patch_gather_skill()
    patch_coder_role()
    patch_repair_model_boundary()
    regenerate_packaged_skills()
    patch_generated_tests()


if __name__ == "__main__":
    main()
