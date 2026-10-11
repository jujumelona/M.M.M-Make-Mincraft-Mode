from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import api
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.complete_orchestrator import CompleteExecutionOptions


def _session(tmp_path):
    session = api.CompleteModAISession.__new__(api.CompleteModAISession)
    session.output_root = tmp_path
    session.workspace_root = tmp_path
    session.existing_input = None
    session.router = SimpleNamespace()
    session.orchestrator = SimpleNamespace()
    return session


def test_fabric_prepare_failure_stops_before_production_model_calls(monkeypatch, tmp_path):
    session = _session(tmp_path)
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    monkeypatch.setattr(api, "_bound_platform_for_preflight", lambda *_: None)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Fabric CLI unavailable before model work")

    monkeypatch.setattr(api, "_prepare_production_scaffold", unavailable, raising=False)
    session.planner = SimpleNamespace(compile_for_production=lambda *_args, **_kwargs:
        pytest.fail("production model must not run before Fabric bootstrap"))
    with pytest.raises(RuntimeError, match="Fabric CLI unavailable before model work"):
        session.build(plan, options=CompleteExecutionOptions(source_only=True, run_jdt=False))


def test_project_identity_is_stable_while_production_ir_is_authored():
    from minecraft_mod_ai.authored_production import authored_base_proposal

    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    compiled = replace(plan, typed_plan_ir={"functions": []}, text="Enriched design")
    assert authored_base_proposal(plan).spec == authored_base_proposal(compiled).spec


def test_build_prepares_scaffold_before_authoring_and_execution(monkeypatch, tmp_path):
    session = _session(tmp_path)
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    compiled = SimpleNamespace(modules=(), calculate_hash=lambda: "proposal")
    events = []
    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    monkeypatch.setattr(api, "_bound_platform_for_preflight", lambda *_: None)
    monkeypatch.setattr(api, "_prepare_production_scaffold", lambda *_:
        events.append("prepare"))
    session.planner = SimpleNamespace(compile_for_production=lambda *_args, **_kwargs:
        events.append("author") or compiled)
    session.orchestrator.execute = lambda *_args, **_kwargs: events.append("execute") or "result"
    assert session.build(plan, options=CompleteExecutionOptions(source_only=True, run_jdt=False)) == "result"
    assert events == ["prepare", "author", "execute"]


def test_existing_project_does_not_create_fresh_fabric_scaffold(tmp_path):
    session = _session(tmp_path)
    session.existing_input = tmp_path / "existing.zip"
    platform = object()
    plan = AuthoredPlan(requested_prompt="Revise existing", text="Existing project")
    assert api._prepare_production_scaffold(session, plan, platform) is platform
    assert not (tmp_path / ".cache").exists()


def test_session_target_adapter_is_locked_before_fabric_prepare(monkeypatch, tmp_path):
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai import platform_live_execution_contract as contract

    adapter = adapter_for_target("26.2", "fabric")
    session = _session(tmp_path)
    session.planner = SimpleNamespace()
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    monkeypatch.setattr(contract, "prepare_official_fabric_project", lambda *_args, **_kwargs:
        tmp_path / "official")
    locked = api._prepare_production_scaffold(session, plan, adapter)
    assert locked.minecraft_version == "26.2"
    assert locked.receipt_sha256
    assert session.orchestrator._prepared_fabric_project == tmp_path / "official"


def test_prepared_fabric_project_is_copied_without_second_cli_call(monkeypatch, tmp_path):
    from minecraft_mod_ai import platform_live_execution_contract as contract

    cached = tmp_path / "prepared"
    cached.mkdir()
    (cached / "build.gradle").write_text("official scaffold")
    base = SimpleNamespace(
        spec=SimpleNamespace(mod_id="demo"), approve=lambda _: None, calculate_hash=lambda: "base",
    )
    owner = SimpleNamespace(
        _prepared_fabric_project=cached,
        _write_base_proposal=lambda *_: None,
    )
    monkeypatch.setattr(contract, "_matches_prepared_scaffold", lambda root, *_: Path(root).is_dir())
    monkeypatch.setattr(contract, "bootstrap_fabric_project", lambda **_: pytest.fail("CLI must not run twice"))
    result = contract.prepare_official_fabric_project(
        owner, SimpleNamespace(base_proposal=base), run_root=tmp_path / "run",
        adapter=object(), error_type=RuntimeError,
    )
    assert result != cached
    assert (result / "build.gradle").read_text() == "official scaffold"
    (result / "build.gradle").write_text("generated production code")
    assert (cached / "build.gradle").read_text() == "official scaffold"


@pytest.mark.parametrize("failure", [None, "prepare", "author"])
@pytest.mark.parametrize("user_constraints", [False, True])
def test_build_target_binding_is_scoped_and_preserves_user_settings(
    monkeypatch, tmp_path, failure, user_constraints,
):
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai import platform_live_execution_contract as contract

    adapter = adapter_for_target("26.2", "fabric")
    session = _session(tmp_path)
    if user_constraints:
        session.router._mmm_requested_minecraft_version = "26.2"
        session.router._mmm_requested_loader = "fabric"
        session.router._mmm_target_adapter = adapter
        session.router._mmm_deterministic_module_kinds = frozenset({"original"})
    original_router = vars(session.router).copy()
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    compiled = SimpleNamespace(modules=(), calculate_hash=lambda: "proposal")

    def prepare(*_args, **_kwargs):
        if failure == "prepare":
            raise RuntimeError("prepare failed")
        return tmp_path / "official"

    def author(*_args, **_kwargs):
        assert session.planner.adapter == adapter
        assert session.router._mmm_target_adapter == adapter
        if failure == "author":
            raise RuntimeError("author failed")
        return compiled

    session.planner = SimpleNamespace(
        adapter=None, deterministic_module_kinds=None, compile_for_production=author,
    )
    session.orchestrator.execute = lambda *_args, **_kwargs: "result"
    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    monkeypatch.setattr(api, "_bound_platform_for_preflight", lambda *_: adapter)
    monkeypatch.setattr(contract, "prepare_official_fabric_project", prepare)
    options = CompleteExecutionOptions(source_only=True, run_jdt=False)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            session.build(plan, options=options)
    else:
        assert session.build(plan, options=options) == "result"
    assert vars(session.router) == original_router
    assert session.planner.adapter is None
    assert session.planner.deterministic_module_kinds is None


def test_production_binding_consumes_exact_prepared_receipt(monkeypatch, tmp_path):
    from minecraft_mod_ai import host_target_binding, platform_selection_pipeline
    from minecraft_mod_ai.authored_production import authored_base_proposal
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai import platform_live_execution_contract as contract

    adapter = adapter_for_target("26.2", "fabric")
    session = _session(tmp_path)
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    compiled = SimpleNamespace(modules=(), calculate_hash=lambda: "proposal")

    def author(*_args, **_kwargs):
        _design, base = host_target_binding.bind_platform(
            session.router, plan.requested_prompt, {}, authored_base_proposal(plan),
        )
        assert base.spec.platform == session.router._mmm_production_platform_selection.lock
        assert base.spec.platform.fabric_loader == adapter.fabric_loader
        return compiled

    session.planner = SimpleNamespace(compile_for_production=author)
    session.orchestrator.execute = lambda *_args, **_kwargs: "result"
    monkeypatch.setattr(api, "_validate_internal_engine_preflight", lambda: None)
    monkeypatch.setattr(api, "_bound_platform_for_preflight", lambda *_: adapter)
    monkeypatch.setattr(contract, "prepare_official_fabric_project", lambda *_args, **_kwargs:
        tmp_path / "official")
    monkeypatch.setattr(platform_selection_pipeline, "resolve_platform_fail_closed", lambda *_args, **_kwargs:
        pytest.fail("Do not resolve a second potentially changed platform receipt after authoring"))
    monkeypatch.setattr("minecraft_mod_ai.platform_target_research.target_research_callback",
        lambda _: lambda _adapter: {"status": "host_target_resolved"})
    assert session.build(plan, options=CompleteExecutionOptions(source_only=True, run_jdt=False)) == "result"
    assert not hasattr(session.router, "_mmm_production_platform_selection")


def test_prepared_receipt_still_checks_authored_module_support():
    from minecraft_mod_ai.host_target_binding import bind_platform
    from minecraft_mod_ai.authored_production import authored_base_proposal
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai.platform_resolver import PlatformSelection
    from minecraft_mod_ai.resolved_version_context import VersionContextError

    selection = PlatformSelection(
        adapter=adapter_for_target("26.2", "fabric"), source="production_scaffold",
        reason="Prepared before authoring", explicit_version=True, explicit_loader=True,
    )
    router = SimpleNamespace(_mmm_production_platform_selection=selection)
    plan = AuthoredPlan(requested_prompt="Make a crystal", text="Crystal design")
    with pytest.raises(VersionContextError, match="unsupported_test_kind"):
        bind_platform(router, plan.requested_prompt, {}, authored_base_proposal(plan),
            module_kinds=("unsupported_test_kind",))
