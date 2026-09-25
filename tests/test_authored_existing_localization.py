from __future__ import annotations

import json
from dataclasses import replace

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
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.mutation_authority import MutationAuthorityMode
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.small_model_task_capsule_contract import (
    _CURRENT_CAPSULE,
    compile_task_capsule,
    task_capsule_generation_scope,
)
from minecraft_mod_ai.small_model_write_scope_enforcement import (
    generation_authority_scoped,
)


def _native_router(monkeypatch, respond):
    """Keep the production router/schema boundary; replace only model inference."""
    requests = []

    class Adapter:
        def generate_turn(self, request):
            requests.append(request)
            function = request.tools[0]["function"]
            return GenerationResponse(tool_calls=(ToolCall(
                id=str(len(requests)), name=function["name"],
                arguments=respond(function, request),
            ),))

    router = ModelRouter(profile="fast_test")
    config = AdapterConfig(role="coder", adapter="llama_cpp", max_context=32768)
    monkeypatch.setattr(router, "_generation_adapter", lambda role: (config, Adapter()))
    return router, requests


@pytest.mark.parametrize("stage", ["terms", "targets"])
def test_localizer_requests_cross_production_router_boundary(monkeypatch, stage):
    from minecraft_mod_ai.authored_existing_localization import _search_terms, _select

    router, requests = _native_router(monkeypatch, lambda function, request: (
        {"terms": ["ship"], "done": True} if stage == "terms" else
        {"primary_path": "src/main/java/example/SpaceMod.java",
         "supporting_paths": [], "done": True}
    ))
    if stage == "terms":
        assert _search_terms(router, {}) == ("ship",)
    else:
        assert _select(router, {}, ["ship"], [{
            "path": "src/main/java/example/SpaceMod.java", "size_bytes": 10,
            "snippet": "class SpaceMod {}",
        }]) == ("src/main/java/example/SpaceMod.java", ())
    assert len(requests) == 1


def test_coherent_generation_preserves_bounded_creation_authority(tmp_path):
    class NoLocalization:
        def generate_tool_decision(self, *args, **kwargs):
            pytest.fail("new coherent design was routed into existing-file localization")

    module = _module()
    module = replace(module, config={
        **{k: v for k, v in module.config.items() if k != "authored_localization_required"},
        "authored_execution_mode": "bounded_coherent", "authored_bounded_scope": True,
        "authored_java_package": "example.space", "authored_mod_id": "space",
        "authored_plan": AuthoredPlan("Space mod", "# Ships\nBuild ships.").to_dict(),
    })

    class Generator:
        router = NoLocalization()

        @task_capsule_generation_scope
        @generation_authority_scoped
        def generate(self, project_root, *, module, **kwargs):
            authority = _CURRENT_AUTHORITY.get()
            assert _CURRENT_CAPSULE.get() is None
            assert "evidence_task" not in module.config
            assert authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
            assert authority.mutation_authority.authorizes(
                "src/main/java/example/space/Ship.java", operation="create_file"
            )
            assert not authority.mutation_authority.authorizes(
                "src/main/java/outside/Ship.java", operation="create_file"
            )

    Generator().generate(tmp_path, module=module)
    assert not (tmp_path / ".minecraft_ai/authored-localization").exists()


def test_paged_localization_keeps_full_budget_through_native_router(monkeypatch, tmp_path):
    paths = [f"src/main/java/example/Part{i}.java" for i in range(6)]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"package example; class {target.stem} {{}}", encoding="utf-8")
    terms = ["economy", "trade", "ship", "upgrade", "crew", "planet", "ore", "colony"]

    def respond(function, request):
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        props = function["parameters"]["properties"]
        if function["name"] == "derive_authored_repository_search_terms":
            previous = payload["previous_terms"]
            assert previous == terms[:len(previous)]
            assert props["terms"]["maxItems"] == 4
            return {"terms": terms[len(previous):len(previous) + 4], "done": False}
        selected = payload["selected_supporting_paths"]
        assert selected == paths[1:1 + len(selected)]
        cap = props["supporting_paths"]["maxItems"]
        assert cap == (4 if not selected else 1)
        if selected:
            assert props["primary_path"]["enum"] == [paths[0]]
            assert props["supporting_paths"]["items"]["enum"] == [paths[-1]]
        return {
            "primary_path": paths[0],
            "supporting_paths": paths[1 + len(selected):1 + len(selected) + cap],
            "done": False,
        }

    router, requests = _native_router(monkeypatch, respond)
    localized = localize_existing_authored_module(router, tmp_path, _module())
    assert len(requests) == 4  # Stops at the host budget even when done remains false.
    assert localized.config["_authored_localization"]["selected_paths"] == paths
    authority = compile_direct_task_mutation_authority(localized)
    assert set(authority.writable_paths) == set(paths)
    receipt = json.loads((tmp_path / ".minecraft_ai/authored-localization/authored_design.json").read_text())
    assert receipt["search_terms"] == terms
    assert receipt["selected_paths"] == paths
    assert localize_existing_authored_module(router, tmp_path, _module()) == localized
    assert len(requests) == 4


@pytest.mark.parametrize("failure", ["stalled_terms", "duplicate_terms", "stalled_support", "primary_drift", "oversized_terms"])
def test_paged_localization_rejects_invalid_continuations(monkeypatch, tmp_path, failure):
    _project(tmp_path)
    if failure == "primary_drift":
        (tmp_path / "src/main/java/example/Ship.java").write_text("class Ship {}", encoding="utf-8")

    def respond(function, request):
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if function["name"] == "derive_authored_repository_search_terms":
            if failure == "oversized_terms":
                return {"terms": ["ship", "crew", "ore", "trade", "colony"], "done": True}
            if failure in {"stalled_terms", "duplicate_terms"}:
                page = ["ship"] if not payload["previous_terms"] or failure == "duplicate_terms" else []
                return {"terms": page, "done": False}
            return {"terms": ["ship"], "done": True}
        candidates = function["parameters"]["properties"]["primary_path"]["enum"]
        primary = payload["primary_path"] or candidates[0]
        if failure == "stalled_support":
            return {"primary_path": primary, "supporting_paths": [], "done": False}
        if not payload["primary_path"]:
            return {"primary_path": candidates[0], "supporting_paths": [candidates[1]], "done": False}
        # Changing the primary to another real candidate on page two is still drift.
        return {"primary_path": payload["selected_supporting_paths"][0], "supporting_paths": [], "done": True}

    router, requests = _native_router(monkeypatch, respond)
    with pytest.raises(AuthoredExistingLocalizationError):
        localize_existing_authored_module(router, tmp_path, _module())
    assert len(requests) <= 3
    assert not (tmp_path / ".minecraft_ai/authored-localization").exists()


class _Router:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, description):
        del role, messages, description
        self.calls.append(tool_name)
        if tool_name == "derive_authored_repository_search_terms":
            return {"terms": ["economy", "trade", "ship", "upgrade"], "done": True}
        if tool_name == "freeze_existing_authored_targets":
            if self.invalid:
                return {"primary_path": "src/main/java/outside/Invented.java", "supporting_paths": [], "done": True}
            return {
                "primary_path": parameters["properties"]["primary_path"]["enum"][0],
                "supporting_paths": [],
                "done": True,
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
    with pytest.raises(AuthoredExistingLocalizationError, match="DECISION_INVALID"):
        localize_existing_authored_module(_Router(invalid=True), tmp_path, _module())
