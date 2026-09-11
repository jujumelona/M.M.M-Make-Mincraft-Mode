"""Synthetic HOST fixtures verify boundaries, not real release compatibility."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json

import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_graph_executor import execute_artifact_graph
from minecraft_mod_ai.artifact_ports import PortConnectionError, PortKind, PortRegistry, TypedPort
from minecraft_mod_ai.platform_resolver import lock_from_adapter
from minecraft_mod_ai.resolved_version_context import ResolvedVersionContext, VersionContextError, VersionRequest, VersionResolver
from minecraft_mod_ai.target_contract import TargetContract
from minecraft_mod_ai.task_template_catalog import load_template
from minecraft_mod_ai.task_template_runner import execute_artifact_template
from minecraft_mod_ai.artifact_job import ArtifactJob


def target_fixture(revision="fixture-a"):
    template = load_template("fabric/item/register_basic")
    digest = "sha256:" + sha256(json.dumps(template, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    facts = {
        "host_revision": revision,
        "capabilities": {"REGISTER_ITEM": True, "UNSUPPORTED": False},
        "api_symbols": {"register_item": "Registry.register"},
        "schemas": {},
        "artifact_rules": {template["id"]: {"template_sha256": digest, "required_symbols": ["register_item"],
                                          "requires_capabilities": ["REGISTER_ITEM"]}},
        "dependency_coordinates": {"fabric_api": "fixture:api:1"},
        "repositories": ["https://example.invalid/host-fixture"],
        "replacements": {},
        "leaf_bindings": {},
    }
    return TargetContract(
        adapter_id="fixture", edition="java", loader="fabric", minecraft_version="26.2",
        java_version="25", yarn_mappings="", mappings_kind="", mappings_version="",
        fabric_loader="fixture-loader", fabric_api="fixture-api", fabric_loom="fixture-loom",
        gradle="fixture-gradle", gradle_sha256="a" * 64, data_pack_version="1",
        resource_pack_version="1", resource_pack_format=1,
        release_metadata_url="https://www.minecraft.net/test", source_api_family="fixture",
        deterministic_module_kinds=frozenset(), host_facts_json=json.dumps(facts),
    )


def job_fixture(context):
    return ArtifactJob(
        "ore.register_basic", "fabric/item/register_basic", "ore", context_id=context.context_id,
        deterministic_inputs={"minecraft_version": context.minecraft, "mod_id": "demo",
                              "package_path": "com/demo", "java_constant": "ORE", "registry_path": "ore"},
    )


def test_auto_and_pinned_resolve_the_same_whole_snapshot():
    ctx = target_fixture().version_context
    resolver = VersionResolver([ctx], auto_context_id=ctx.context_id)
    assert resolver.resolve(VersionRequest.from_input("auto")) is ctx
    assert resolver.resolve(VersionRequest.from_input("26.2")) is ctx
    assert "AUTO" not in ctx.snapshot_json
    assert ctx.to_dict()["source"] == "HOST"
    with pytest.raises(VersionContextError) as error:
        resolver.resolve(VersionRequest.from_input("1.20.1"))
    assert error.value.diagnostic == {"code": "UNSUPPORTED_MINECRAFT_VERSION", "requested": "1.20.1", "supported": ["26.2"]}


def test_nested_context_is_immutable_and_detached_from_receipts():
    ctx = target_fixture().version_context
    with pytest.raises(FrozenInstanceError):
        ctx.snapshot_json = "changed"
    with pytest.raises(TypeError):
        ctx.capabilities["REGISTER_ITEM"] = False
    public = ctx.to_dict()
    public["host_facts"]["api_symbols"]["register_item"] = "wrong"
    assert ctx.api_symbols["register_item"] == "Registry.register"
    with pytest.raises(VersionContextError, match="HASH_MISMATCH"):
        ResolvedVersionContext.from_dict(public)


def test_lock_and_provider_share_snapshot_without_requery():
    target = target_fixture()
    lock = lock_from_adapter(target)
    assert lock.version_context == target.version_context
    changed = replace(lock, host_facts_json=target_fixture("other").host_facts_json)
    with pytest.raises(ValueError, match="SHA-256"):
        changed.validate()


@pytest.mark.parametrize("field", ["fabric_loader", "fabric_api", "fabric_loom", "gradle"])
def test_floating_coordinates_cannot_enter_context(field):
    with pytest.raises(VersionContextError, match="UNRESOLVED_VERSION_COORDINATE"):
        replace(target_fixture(), **{field: "latest"}).version_context


def test_host_missing_facts_do_not_get_defaults():
    with pytest.raises(VersionContextError, match="HOST_BUNDLE_INCOMPLETE"):
        replace(target_fixture(), host_facts_json="").version_context
    ctx = target_fixture().version_context
    with pytest.raises(VersionContextError, match="HOST_FACT_UNAVAILABLE"):
        ctx.require_fact("api_symbols", "invented")
    with pytest.raises(VersionContextError, match="UNSUPPORTED_CAPABILITY"):
        ctx.require_capability("UNSUPPORTED")


def test_same_context_render_validation_and_ports():
    ctx = target_fixture().version_context
    registry = PortRegistry()
    result = execute_artifact_graph([job_fixture(ctx)], context={"resolved_version_context": ctx.to_dict()}, port_registry=registry)
    receipt = result["receipts"][0]
    assert receipt["context_id"] == ctx.context_id
    assert receipt["validations"][0]["context_id"] == ctx.context_id
    assert all(port.context_id == ctx.context_id for port in registry.all_ports().values())


def test_mixed_context_graph_rejects_before_any_execution(monkeypatch):
    ctx = target_fixture().version_context
    other = target_fixture("other").version_context
    import minecraft_mod_ai.artifact_graph_executor as graph
    monkeypatch.setattr(graph, "_execute_one", lambda *a, **k: pytest.fail("must reject before execution"))
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_MISMATCH"):
        execute_artifact_graph([job_fixture(other)], context={"resolved_version_context": ctx.to_dict()})


def test_existing_foreign_ports_are_rejected():
    registry = PortRegistry()
    registry.publish(TypedPort("external", PortKind.JAVA_SYMBOL, "Item", "Foreign.ITEM", "foreign"))
    ctx = target_fixture().version_context
    with pytest.raises(PortConnectionError, match="VERSION_CONTEXT_MISMATCH"):
        execute_artifact_graph([job_fixture(ctx)], context={"resolved_version_context": ctx.to_dict()}, port_registry=registry)


def test_job_cannot_override_host_fields():
    ctx = target_fixture().version_context
    job = job_fixture(ctx)
    job.deterministic_inputs["fabric_api"] = "invented"
    with pytest.raises(VersionContextError, match="HOST_FACT_OVERRIDE"):
        execute_artifact_template(job, context={"resolved_version_context": ctx.to_dict()})


def test_template_drift_requires_host_readmission():
    ctx = target_fixture().version_context
    template = load_template("fabric/item/register_basic")
    template["render"]["body"] += "WrongApi.call();"
    with pytest.raises(VersionContextError, match="HOST_TEMPLATE_NOT_ADMITTED"):
        ctx.admit_template(template)
    with pytest.raises(VersionContextError, match="INVALID_API_SYMBOL"):
        ctx.validate_artifact(load_template("fabric/item/register_basic"), "WrongApi.call();")


def test_reuse_rejects_missing_or_different_context():
    from minecraft_mod_ai.fact_source_reuse import FactReuseClassifier
    from minecraft_mod_ai.implementation_fact import ImplementationFact, FactType

    ctx = target_fixture().version_context
    classifier = FactReuseClassifier(project_index={"context_id": "foreign"}, version_context=ctx)
    with pytest.raises(VersionContextError, match="REUSE_VERSION_INCOMPATIBLE"):
        classifier.classify(ImplementationFact("ore.exists", FactType.ITEM_EXISTS, "ore"))


def test_host_catalog_has_no_live_component_fallback(tmp_path, monkeypatch):
    from minecraft_mod_ai.host_version_catalog import host_target, host_versions

    monkeypatch.delenv("MMM_VERSION_BUNDLE_CATALOG", raising=False)
    monkeypatch.setattr("minecraft_mod_ai.host_version_catalog.DEFAULT_CATALOG", tmp_path / "missing.json")
    with pytest.raises(VersionContextError, match="HOST_BUNDLE_CATALOG_INVALID"):
        host_target("26.2")
    ctx = target_fixture().version_context
    path = tmp_path / "bundles.json"
    path.write_text(json.dumps({"schema_version": "mmm/host-version-catalog-v1", "auto_context_id": ctx.context_id, "bundles": [ctx.to_dict()]}))
    monkeypatch.setenv("MMM_VERSION_BUNDLE_CATALOG", str(path))
    assert host_versions(10) == ("26.2",)
    assert host_target("auto").version_context == ctx


def test_lowering_rejects_conflicting_version():
    ctx = target_fixture().version_context
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_MISMATCH"):
        expand_facts_to_jobs([], mod_id="demo", package_name="com.demo", minecraft_version="1.20.1", version_context=ctx)


def test_real_selection_boundary_consumes_host_bundle_before_research(tmp_path, monkeypatch):
    from minecraft_mod_ai import platform_selection_pipeline as selection
    from minecraft_mod_ai.platform_catalog import PlatformProvider
    from minecraft_mod_ai.host_version_catalog import host_target, host_versions

    ctx = target_fixture().version_context
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema_version": "mmm/host-version-catalog-v1", "auto_context_id": ctx.context_id, "bundles": [ctx.to_dict()]}))
    monkeypatch.setenv("MMM_VERSION_BUNDLE_CATALOG", str(path))
    monkeypatch.setattr(selection, "provider_for_loader", lambda _: PlatformProvider(
        "fabric", "host-coherent-version-catalog-v1", host_versions, host_target))
    monkeypatch.setattr(selection, "optimize_platform_evidence", lambda *a, **k: pytest.fail("must not score version"))
    result = selection.resolve_platform_fail_closed("Add an item", target_research_fn=lambda *a: pytest.fail("must not research version"))
    assert result.adapter.version_context == ctx
    assert result.lock.version_context == ctx
    assert result.to_dict()["resolved_version_context"] == ctx.to_dict()
    with pytest.raises(VersionContextError, match="UNSUPPORTED_MINECRAFT_VERSION"):
        selection.resolve_platform_fail_closed("Minecraft 1.20.1")


def test_saved_host_snapshot_cannot_be_replaced(tmp_path):
    from minecraft_mod_ai import platform_catalog, platform_generation_contract

    target = target_fixture()
    platform_generation_contract._write_platform_lock(tmp_path, target)
    assert platform_catalog.adapter_from_project(tmp_path).version_context == target.version_context
    path = tmp_path / ".minecraft_ai/platform-lock.json"
    payload = json.loads(path.read_text())
    payload["context_id"] = target_fixture("foreign").version_context.context_id
    path.write_text(json.dumps(payload))
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_MISMATCH"):
        platform_catalog.adapter_from_project(tmp_path)


def test_research_projection_rejects_foreign_snapshot_and_removes_request_mode():
    from types import SimpleNamespace
    from minecraft_mod_ai.research_requirement_template import build_host_planning_context

    target = target_fixture()
    selection = {"target": target.public_dict(), "resolved_version_context": target.version_context.to_dict()}
    result = build_host_planning_context(SimpleNamespace(_mmm_requested_minecraft_version="auto"),
                                         {"_platform_selection": selection})
    assert "requested_constraints" not in result
    assert result["context_id"] == target.version_context.context_id
    selection["resolved_version_context"] = target_fixture("foreign").version_context.to_dict()
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_MISMATCH"):
        build_host_planning_context(SimpleNamespace(), {"_platform_selection": selection})


def test_selection_capability_check_is_host_owned():
    from minecraft_mod_ai.platform_resolver import _require_supported_kinds

    _require_supported_kinds(target_fixture(), ["REGISTER_ITEM"], explicit=True)
    with pytest.raises(VersionContextError, match="UNSUPPORTED_CAPABILITY"):
        _require_supported_kinds(target_fixture(), ["UNSUPPORTED"], explicit=True)


def test_catalog_rejects_two_bundles_for_one_minecraft_version():
    ctx = target_fixture().version_context
    with pytest.raises(VersionContextError, match="AMBIGUOUS_HOST_BUNDLE"):
        VersionResolver([ctx, target_fixture("other").version_context], auto_context_id=ctx.context_id)
