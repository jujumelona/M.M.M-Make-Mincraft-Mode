from __future__ import annotations

import pytest

from minecraft_mod_ai.authored_existing_localization import (
    AuthoredExistingLocalizationError,
    localize_existing_authored_module,
)
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    compile_direct_task_mutation_authority,
)
from minecraft_mod_ai.mutation_authority import MutationAuthorityMode
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.small_model_task_capsule_contract import (
    _CURRENT_CAPSULE,
    compile_task_capsule,
    task_capsule_generation_scope,
)
from minecraft_mod_ai.small_model_write_scope_enforcement import generation_authority_scoped


class _Router:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, description):
        del role, messages, description
        self.calls.append(tool_name)
        if tool_name == "derive_authored_repository_search_terms":
            return {"terms": ["economy", "trade", "ship", "upgrade"]}
        if tool_name == "freeze_existing_authored_targets":
            if self.invalid:
                return {"primary_path": "src/main/java/outside/Invented.java", "supporting_paths": []}
            return {
                "primary_path": parameters["properties"]["primary_path"]["enum"][0],
                "supporting_paths": [],
            }
        raise AssertionError(tool_name)


def _module():
    adapter = adapter_for_target("1.21.11", "fabric")
    plan = AuthoredPlan(
        "Add economy and ship upgrades",
        "# Economy\nTrade credits.\n# Ships\nUpgrade ship systems.\n",
        existing_input_sha256="a" * 64,
    )
    return ProductionModule(
        module_id="authored_design",
        kind="custom_java",
        config={
            "implementation": "custom",
            "authored_plan": plan.to_dict(),
            "authored_localization_required": True,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
        required_gates=("target_compile", "project build"),
    )


def _project(root):
    main = root / "src/main/java/example/SpaceMod.java"
    economy = root / "src/main/java/example/EconomySystem.java"
    main.parent.mkdir(parents=True)
    main.write_text("package example; public final class SpaceMod { EconomySystem e; }\n", encoding="utf-8")
    economy.write_text("package example; public final class EconomySystem { int credits; }\n", encoding="utf-8")


def test_localizer_freezes_exact_authority(tmp_path):
    _project(tmp_path)
    router = _Router()
    localized = localize_existing_authored_module(router, tmp_path, _module())
    assert router.calls == [
        "derive_authored_repository_search_terms",
        "freeze_existing_authored_targets",
    ]
    capsule = compile_task_capsule(localized)
    assert capsule is not None
    assert 1 <= len(capsule.writable_paths) <= 6
    authority = compile_direct_task_mutation_authority(localized)
    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    assert authority.writable_paths == capsule.writable_paths
    error = authority.mutation_authority.mutation_error(
        "src/main/java/other/NotOwned.java", operation="create_file"
    )
    assert error and error.startswith("MUTATION_TARGET_DRIFT")


def test_generation_boundary_localizes_before_exact_authority(tmp_path):
    _project(tmp_path)
    router = _Router()

    class Generator:
        def __init__(self):
            self.router = router

        @task_capsule_generation_scope
        @generation_authority_scoped
        def generate(
            self,
            project_root,
            *,
            module,
            research_modules=(),
            minecraft_version=None,
            loader=None,
            mappings=None,
            execution_feedback=None,
        ):
            del project_root, research_modules, minecraft_version, loader, mappings
            del execution_feedback
            authority = _CURRENT_AUTHORITY.get()
            capsule = _CURRENT_CAPSULE.get()
            assert authority is not None
            assert capsule is not None
            assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
            assert authority.writable_paths == capsule.writable_paths
            assert "evidence_task" in module.config
            assert module.config["_authored_localization"]["primary_path"] == capsule.primary_path
            return capsule.primary_path

    primary = Generator().generate(tmp_path, module=_module())
    assert primary.endswith(".java")
    assert router.calls == [
        "derive_authored_repository_search_terms",
        "freeze_existing_authored_targets",
    ]
    assert _CURRENT_AUTHORITY.get() is None
    assert _CURRENT_CAPSULE.get() is None


def test_localization_receipt_prevents_reselection(tmp_path):
    _project(tmp_path)
    first = localize_existing_authored_module(_Router(), tmp_path, _module())

    class NoCall:
        def generate_tool_decision(self, *args, **kwargs):
            raise AssertionError("locator must be cached")

    second = localize_existing_authored_module(NoCall(), tmp_path, _module())
    assert second.config["_authored_localization"] == first.config["_authored_localization"]


def test_localizer_rejects_non_candidate_path(tmp_path):
    _project(tmp_path)
    with pytest.raises(AuthoredExistingLocalizationError, match="PRIMARY_INVALID"):
        localize_existing_authored_module(_Router(invalid=True), tmp_path, _module())
