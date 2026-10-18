"""The active MMM pipeline must construct from verified Minecraft APIs, not donor mods."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
REMOVED = frozenset("""
authored_reuse_bridge authored_source_rag construction_porting curseforge_catalog
curseforge_dependency_adoption curseforge_dependency_project curseforge_query
curseforge_reuse_production curseforge_semantic_source_review
curseforge_source_attestation donor_source_authority grounded_source_reuse
planning_mod_discovery research_reuse_candidates reuse_adapters reuse_artifacts
reuse_build_verifier reuse_discovery reuse_license reuse_planner reuse_proof_executor
reuse_template_pipeline source_inspection_recovery source_intent_aliases
source_reuse_evidence source_transplant repository_artifact_index
capability_implementation_locator versioned_mod_reference_catalog
versioned_mod_reference_resolver versioned_mod_reference_retriever
""".split())


def test_external_donor_implementation_modules_are_deleted():
    for module in sorted(REMOVED):
        assert not (PACKAGE / (module + ".py")).exists(), module
    assert not (PACKAGE / "templates" / "reuse").exists()
    assert not (PACKAGE / "templates" / "feature" / "reuse_assessment").exists()
    assert not (PACKAGE / "templates" / "criterion" / "reuse_assessment.yaml").exists()


def test_no_production_import_of_removed_donor_module():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[-1]
                    if module in REMOVED:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            module = (node.module or "").split(".")[-1]
            if module in REMOVED:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
            if node.level or node.module in {"minecraft_mod_ai", None}:
                for alias in node.names:
                    if alias.name in REMOVED:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}")
    assert not offenders, "Deleted donor module still imported:\n" + "\n".join(offenders)


def test_no_external_source_reuse_in_main_path():
    forbidden = {
        "complete_planner.py": ("authored_reuse_bridge", "research_reuse_candidates"),
        "complete_orchestrator.py": ("curseforge_reuse_production", "materialize_verified_authored_sources"),
        "authored_production.py": ("source_reuse", "authored_source_rag"),
        "authored_feature_implementation.py": ("source_reuse", "authored_source_rag"),
        "pre_design_research_pipeline.py": ("collect_source_candidate_handoff", "research_reuse_candidates"),
        "catalog_first_grounded_rag.py": ("_search_curseforge", "_search_modrinth"),
        "mcp_server.py": ("read_reuse_source",),
        "production_tools.py": ("def read_reuse_source",),
    }
    for filename, symbols in forbidden.items():
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        for symbol in symbols:
            assert symbol not in source, (filename, symbol)


def test_colab_has_no_external_mod_reuse_credentials():
    notebook = (ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8")
    assert "CURSEFORGE_API_KEY" not in notebook
    assert "REFERENCE_MOD_URLS" not in notebook


def test_planning_skills_cannot_request_external_mod_donors():
    import json

    package = json.loads(
        (PACKAGE / "packaged_skills.json").read_text(encoding="utf-8")
    )
    forbidden_tools = {
        "read_reuse_source",
        "discover_ecosystem_resources",
        "inspect_modrinth_project",
        "inspect_github_repository",
    }
    for name in (
        "plan-game-design",
        "gather-adaptive-minecraft-evidence",
        "ground-production-with-live-evidence",
        "route-generic-game-research",
    ):
        source = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        plugin = (ROOT / "plugins" / "mmm-minecraft-mod-ai" / "skills" / name / "SKILL.md")
        assert plugin.read_text(encoding="utf-8") == source
        assert package["skills"][name] == source
        allowed = set(package["contracts"][name]["allowed_tools"])
        assert not (allowed & forbidden_tools), (name, allowed & forbidden_tools)
        assert "Never search" in source or "never searching" in source
