"""Every original-occurrence producer belongs to compiler admission, not inference."""
from __future__ import annotations

import json
import pytest

from minecraft_mod_ai.host_source_publication import source_publication_transaction
from minecraft_mod_ai.incremental_java_pipeline import candidate_data, load_candidate

_SYMBOLS = {
    "class": {"kind": "CLASS", "owner": "example.Transaction",
              "name": "Transaction", "access": 1},
    "purchase": {"kind": "METHOD", "owner": "example.Transaction",
                 "name": "applyPurchase", "parameters": ["java.lang.String", "int"],
                 "return_type": "boolean", "access": 1},
}
_ARGS = {"source_site_ref": "purchase", "behavior_id": "test_behavior",
         "package": "example.generated", "at": "HEAD", "side": "COMMON",
         "java_version": 25}


def _workspace(tmp_path):
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = '{"schemaVersion":1,"id":"custommod","mixins":["existing.json"]}'
    manifest.write_text(original)
    return manifest, original


def test_rejected_compiler_or_provenance_restores_manifest_mixin_and_receipt(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
    manifest, original = _workspace(tmp_path)
    plan = plan_verified_custom_event(symbols=_SYMBOLS, mod_id="custommod", **_ARGS)
    records = [{"kind": "custom", "args": dict(_ARGS)}]
    with pytest.raises(ValueError, match="javac symbol failure"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS, records=records):
            assert manifest.read_text() != original
            assert all((tmp_path / name).is_file() for name in plan["files"])
            raise ValueError("javac symbol failure")
    assert manifest.read_text() == original
    assert all(not (tmp_path / name).exists() for name in plan["files"])
    assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_accepted_compiler_and_proof_keep_installed_mixin(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
    manifest, original = _workspace(tmp_path)
    plan = plan_verified_custom_event(symbols=_SYMBOLS, mod_id="custommod", **_ARGS)
    records = [{"kind": "custom", "args": dict(_ARGS)}]
    with source_publication_transaction(
        project_root=tmp_path, symbols=_SYMBOLS, records=records):
        assert manifest.read_text() != original
    assert all((tmp_path / name).is_file() for name in plan["files"])
    assert list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_source_owned_candidate_roundtrip_reinstalls_exact_original_source_plan(tmp_path):
    manifest, original = _workspace(tmp_path)
    state = {"strategy": "host_source_owned", "native_mutation_receipts": [],
             "source_publications": [{"kind": "custom", "args": dict(_ARGS)}]}
    pending = candidate_data("package example; class Bridge {}", "compilation failed", state)
    restored = {}
    assert load_candidate(pending, restored, True) == pending
    assert restored["source_publications"] == state["source_publications"]
    with pytest.raises(RuntimeError, match="failed after replay"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS,
            records=restored["source_publications"]):
            assert manifest.read_text() != original
            raise RuntimeError("failed after replay")
    assert manifest.read_text() == original
    legacy = {key: value for key, value in pending.items()
              if key != "source_publications_json"}
    assert load_candidate(legacy, {}, True) == {}


def test_source_publication_invalid_plan_has_no_writes(tmp_path):
    manifest, original = _workspace(tmp_path)
    with pytest.raises(ValueError, match="HOST_SOURCE_PUBLICATION_CUSTOM_INVALID"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS,
            records=[{"kind": "custom", "args": {**_ARGS, "symbols": {}}}]):
            pytest.fail("invalid plan reached compilation")
    assert manifest.read_text() == original


def test_ui_producer_rollback_restores_original_manifest_and_all_files(tmp_path):
    from test_source_owned_ui_event import _plan, SYMBOLS
    planned = _plan()
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps({"id": "allocation", "entrypoints": {
        "main": ["example.Preexisting"], "client": []}})
    manifest.write_text(original)
    with pytest.raises(ValueError, match="semantic condition rejected"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=SYMBOLS,
            records=[{"kind": "ui", "plan": planned}]):
            assert all((tmp_path / name).exists() for name in planned["files"])
            raise ValueError("semantic condition rejected")
    assert manifest.read_text() == original
    assert all(not (tmp_path / name).exists() for name in planned["files"])
    assert not list((tmp_path / ".mmm/typed-ui-producers").glob("*.json"))


def test_source_owned_compiler_loads_qualified_external_producer_closure():
    from minecraft_mod_ai.java_source_snapshot import direct_source_dependencies
    assert direct_source_dependencies({"host_source_owned": True}) == {
        "include_qualified_sources": True}
    assert direct_source_dependencies({
        "host_direct_feature_source": False, "host_source_owned": True}) == {
        "include_qualified_sources": True}
    assert direct_source_dependencies({"host_direct_feature_source": True}) == {
        "include_qualified_sources": True}
    assert direct_source_dependencies({}) == {}


@pytest.mark.parametrize("compiler_accepts", [False, True])
def test_actual_host_construction_rollback_boundary(tmp_path, monkeypatch, compiler_accepts):
    import minecraft_mod_ai.construction_host_source as host
    import minecraft_mod_ai.incremental_java_pipeline as pipeline
    from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
    manifest, original = _workspace(tmp_path)
    plan = plan_verified_custom_event(symbols=_SYMBOLS, mod_id="custommod", **_ARGS)
    spec = {
        "language": "java", "target_path": "src/main/java/example/Bridge.java",
        "requirement": "implement original occurrence",
        "bindings": {"request_feature_entrypoint": True},
    }
    behavior = {"behavior_id": "test_behavior", "intent": "actual task",
                "trigger": "concrete source invocation", "effects": "real effect"}
    context = {
        "project_root": str(tmp_path), "java_classpath": ["test-only"],
        "target": {"loader": "fabric", "java_version": 25},
        "symbols": _SYMBOLS,
        "feature_work_order": {"acceptance": {"max_compiler_repair_attempts": 0}},
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "source-publication-boundary-integration-v1")
    monkeypatch.setattr(host, "_build_host_java_prompt",
        lambda *a, **kw: ({}, "original occurrence", [behavior], {
            "inspected_target_api": {}, "materialized_prerequisite_api": [],
            "host_state_work_order_scope": {}, "available_fabric_callbacks": [],
            "retained_typed_behavior": {}, "owned_runtime_records": [],
        }, []))
    monkeypatch.setattr(
        "minecraft_mod_ai.host_verified_entity_type.verified_entity_type_source",
        lambda *a, **kw: None)
    monkeypatch.setattr(
        "minecraft_mod_ai.host_verified_entity_registry.verified_entity_registry_source",
        lambda *a, **kw: None)
    monkeypatch.setattr(
        "minecraft_mod_ai.template_first_route.checked_rule_template_ready",
        lambda *a, **kw: False)
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *a, **kw: None)
    output = "package example; public class Bridge {}"
    def produce(router, spec, messages, *, construction_context, **kwargs):
        construction_context["incremental_state"]["strategy"] = "host_source_owned"
        construction_context["pending_source_publications"].append({
            "kind": "custom", "args": dict(_ARGS)})
        assert not (tmp_path / next(iter(plan["files"]))).exists()
        return output
    monkeypatch.setattr(pipeline, "compose_source", produce)
    observed = []
    def check(source, *args, **kwargs):
        assert source == output
        assert all((tmp_path / name).is_file() for name in plan["files"])
        observed.append(True)
        if not compiler_accepts:
            raise ValueError("mock javac error after event files were materialized")
    monkeypatch.setattr(host, "_verification_boundary", lambda: check)
    from minecraft_mod_ai.construction_host_source import compose_host_java_source
    def call():
        return compose_host_java_source(
            object(), spec, context, package="example", name="Bridge",
            owner="example.Bridge", behaviors=[behavior])
    if not compiler_accepts:
        with pytest.raises(ValueError, match="mock javac error"):
            call()
        assert manifest.read_text() == original
        assert all(not (tmp_path / name).exists() for name in plan["files"])
        assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))
    else:
        assert call() == output
        assert manifest.read_text() != original
        assert all((tmp_path / name).is_file() for name in plan["files"])
    assert len(observed) == 1


def test_transaction_rolls_back_acceptance_receipts_and_previous_authorship(tmp_path):
    manifest, initial = _workspace(tmp_path)
    old_authorship = tmp_path / ".mmm/construction_authorship/example.json"
    old_authorship.parent.mkdir(parents=True)
    old_authorship.write_text("old verified source")
    new_checkpoint = tmp_path / ".mmm/construction_host_files/generated.json"
    records = [{"kind": "custom", "args": dict(_ARGS)}]
    with pytest.raises(OSError, match="checkpoint failure"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS, records=records,
            tracked_paths=(old_authorship, new_checkpoint)):
            old_authorship.write_text("new uncommitted source")
            new_checkpoint.parent.mkdir(parents=True)
            new_checkpoint.write_text("not durable")
            raise OSError("checkpoint failure")
    assert old_authorship.read_text() == "old verified source"
    assert not new_checkpoint.exists()
    assert manifest.read_text() == initial
    assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_whole_feature_failure_restores_retired_custom_event_and_receipt(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import (
        materialize_verified_custom_event, plan_verified_custom_event)
    manifest, _ = _workspace(tmp_path)
    symbols = {**_SYMBOLS, "replacement": {
        "kind": "METHOD", "owner": "example.Transaction",
        "name": "finishPurchase", "parameters": ["int"],
        "return_type": "boolean", "access": 1,
    }}
    previous = materialize_verified_custom_event(
        project_root=tmp_path, symbols=symbols, **_ARGS)
    previous_manifest = manifest.read_bytes()
    previous_receipts = {
        path.name: path.read_bytes()
        for path in (tmp_path / ".mmm/source-producers").glob("*.json")
    }
    replacement_args = {**_ARGS, "source_site_ref": "replacement"}
    replacement = plan_verified_custom_event(
        symbols=symbols, mod_id="custommod", **replacement_args)
    with pytest.raises(RuntimeError, match="javac rejected replacement"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=symbols,
            records=[{"kind": "custom", "args": replacement_args}]):
            assert all(not (tmp_path / name).exists() for name in previous["files"])
            assert all((tmp_path / name).is_file() for name in replacement["files"])
            raise RuntimeError("javac rejected replacement")
    assert manifest.read_bytes() == previous_manifest
    assert all((tmp_path / name).read_text() == body
               for name, body in previous["files"].items())
    assert all(not (tmp_path / name).exists() for name in replacement["files"])
    assert previous_receipts == {
        path.name: path.read_bytes()
        for path in (tmp_path / ".mmm/source-producers").glob("*.json")
    }


def test_unrelated_compiler_failure_restores_checkpoint_without_event_plans(tmp_path):
    # A normal Java compilation can be interrupted after its authorship or
    # checkpoint is written, even when it did not create a new Fabric hook.
    receipt = tmp_path / ".mmm/construction_host_files/owner.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("last accepted owner")
    with pytest.raises(ValueError, match="persist failed"):
        with source_publication_transaction(
            project_root=tmp_path, symbols={}, records=[],
            tracked_paths=(receipt,)):
            receipt.write_text("unverified owner")
            raise ValueError("persist failed")
    assert receipt.read_text() == "last accepted owner"


def test_duplicate_original_source_producers_for_one_behavior_are_rejected_prepublication(tmp_path):
    manifest, original = _workspace(tmp_path)
    duplicate = [
        {"kind": "custom", "args": dict(_ARGS)},
        {"kind": "custom", "args": dict(_ARGS)},
    ]
    with pytest.raises(ValueError, match="HOST_SOURCE_PUBLICATION_DUPLICATE_BEHAVIOR"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS, records=duplicate):
            pytest.fail("conflicting producers were published")
    assert manifest.read_text() == original
    assert not (tmp_path / ".mmm/source-producers").exists()


def test_java_only_checkpoint_rollback_holds_cross_process_publication_lock(
    tmp_path, monkeypatch,
):
    from contextlib import contextmanager
    import minecraft_mod_ai.source_owned_custom_event as producer
    receipt = tmp_path / ".mmm/construction_host_files/owner.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("accepted")
    actual_lock = producer._producer_cross_process_lock
    entered = []
    @contextmanager
    def guarded_lock(path):
        with actual_lock(path):
            entered.append(path)
            yield
    monkeypatch.setattr(producer, "_producer_cross_process_lock", guarded_lock)
    with pytest.raises(ValueError, match="javac rejected"):
        with source_publication_transaction(
            project_root=tmp_path, symbols={}, records=[],
            tracked_paths=(receipt,),
        ):
            assert len(entered) == 1
            assert entered[0] == tmp_path / ".mmm/source-producer.lock"
            receipt.write_text("unverified")
            raise ValueError("javac rejected")
    assert receipt.read_text() == "accepted"


@pytest.mark.parametrize("compiler_ok", [False, True])
def test_rule_ir_is_not_persisted_before_target_compile_acceptance(
    tmp_path, monkeypatch, compiler_ok,
):
    import minecraft_mod_ai.construction_host_source as host
    import minecraft_mod_ai.host_rule_feature as rules
    from minecraft_mod_ai.construction_host_source import compose_host_java_source
    from minecraft_mod_ai.construction_direct_writer import _read_checkpoint
    manifest, _ = _workspace(tmp_path)
    behavior = {"behavior_id": "observation", "trigger": "inspected event",
                "intent": "respond to observation", "effects": "actual effect"}
    spec = {
        "language": "java", "target_path": "src/main/java/example/Bridge.java",
        "requirement": "implement callback",
        "bindings": {"request_feature_entrypoint": True},
    }
    ctx = {
        "project_root": str(tmp_path), "java_classpath": ["test-only"],
        "target": {"loader": "fabric", "java_version": 25},
        "symbols": {"owner": {"kind": "CLASS", "owner": "example.Bridge"}},
        "feature_work_order": {"acceptance": {"max_compiler_repair_attempts": 0}},
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "rule-proof-deferred-compile-contract-v1")
    monkeypatch.setattr(host, "_build_host_java_prompt",
        lambda *a, **kw: ({}, "compile checked rule", [behavior], {
            "inspected_target_api": {}, "materialized_prerequisite_api": [],
            "host_state_work_order_scope": {}, "available_fabric_callbacks": [],
            "retained_typed_behavior": {}, "owned_runtime_records": [],
        }, []))
    monkeypatch.setattr(
        "minecraft_mod_ai.host_verified_entity_type.verified_entity_type_source",
        lambda *a, **kw: None)
    monkeypatch.setattr(
        "minecraft_mod_ai.host_verified_entity_registry.verified_entity_registry_source",
        lambda *a, **kw: None)
    monkeypatch.setattr(
        "minecraft_mod_ai.template_first_route.checked_rule_template_ready",
        lambda *a, **kw: True)
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_coverage.record_authored_source_calls",
        lambda *a, **kw: None)
    rule_path = tmp_path / ".mmm/construction_rule_ir/rule-proof.json"
    source = "package example; public final class Bridge {}"
    def proposed_rule(_router, _spec, context, **kwargs):
        assert context["defer_rule_ir_persistence"]
        assert not rule_path.exists()
        context["pending_rule_ir"].append((
            rule_path, "rule-proof", {
                "schema": "mmm/host-observation-rules-v1",
                "target_compilation_verified": False,
                "runtime_verified": False,
            }))
        return source
    monkeypatch.setattr(rules, "compose_rule_feature", proposed_rule)
    checked = []
    def verify(candidate, *a, **kw):
        assert candidate == source
        assert not rule_path.exists(), "rule was stored before javac"
        checked.append(True)
        if not compiler_ok:
            raise ValueError("javac rejected rule")
    monkeypatch.setattr(host, "_verification_boundary", lambda: verify)
    def produce():
        return compose_host_java_source(
            object(), spec, ctx, package="example", name="Bridge",
            owner="example.Bridge", behaviors=[behavior])
    if compiler_ok:
        assert produce() == source
        assert _read_checkpoint(rule_path, "rule-proof")["runtime_verified"] is False
    else:
        with pytest.raises(ValueError, match="HOST_RULE_TEMPLATE_COMPILATION_FAILED"):
            produce()
        assert not rule_path.exists()
    assert checked == [True]


@pytest.mark.parametrize("symlink_target", ["lock_file", "lock_directory"])
def test_source_publication_rejects_external_symlinked_os_lock(
    tmp_path, symlink_target,
):
    manifest, original = _workspace(tmp_path)
    location = tmp_path / ".mmm/source-producer.lock"
    location.parent.mkdir(parents=True)
    if symlink_target == "lock_file":
        outside = tmp_path / "outside-lock-file"
        outside.write_bytes(b"")
        try:
            location.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
    else:
        outside = tmp_path / "outside-lock-directory"
        outside.mkdir()
        location.parent.rmdir()
        try:
            location.parent.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks unavailable")
    with pytest.raises(ValueError, match="SOURCE_PRODUCER_LOCK_PATH_UNSAFE"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS,
            records=[{"kind": "custom", "args": dict(_ARGS)}],
        ):
            pytest.fail("unsafe OS lock admitted a source write")
    assert manifest.read_text() == original
    assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))
