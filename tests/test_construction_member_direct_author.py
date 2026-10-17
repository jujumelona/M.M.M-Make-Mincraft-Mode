"""Canonical registry methods should be implemented as whole Java units."""
from __future__ import annotations

from minecraft_mod_ai.construction_task import ConstructionTask
from minecraft_mod_ai.construction_member_direct_author import (
    compose_method, eligible_method,
)


def _fixture(tmp_path):
    symbol = "demo.AlienRegistry#registerAliens()"
    task = ConstructionTask(
        "reg_alien", ("req:alien",), "logic",
        "Register alien entity in target registry",
        target_path="src/main/java/demo/AlienRegistry.java",
        owner_symbol=symbol,
        effects="Add the alien entity type to the game's entity registry",
    )
    row = {
        "kind": "METHOD", "owner": "demo.AlienRegistry", "name": "registerAliens",
        "return_type": "void", "parameters": [],
        "visibility": "private", "static": False, "final": False, "side": "COMMON",
    }
    context = {
        "project_root": str(tmp_path),
        "target": {"minecraft_version": "26.2", "loader": "fabric", "java_version": 25},
        "symbols": {symbol: row}, "prerequisite_hashes": {},
        "authoring_contract": {}, "source_pages": {},
        "evidence_index": object(),
    }
    return task, context


def test_direct_method_rejects_unowned_state_or_port_donor(tmp_path):
    task, ctx = _fixture(tmp_path)
    assert eligible_method(task, ctx["symbols"], ctx)
    assert not eligible_method(task, ctx["symbols"], ctx, owned_records=[{"record": 1}])
    assert not eligible_method(task, ctx["symbols"], ctx, port_source_ref="port")
    assert not eligible_method(task, ctx["symbols"], {
        **ctx, "donor_callable_symbols": {"required": ["demo.External#doThing"]},
    })
    from dataclasses import replace
    stateful = replace(task, state_reads=("alienPopulation",))
    assert not eligible_method(stateful, ctx["symbols"], ctx)


def test_direct_method_authors_one_whole_method_and_resumes(tmp_path, monkeypatch):
    from minecraft_mod_ai import construction_member_direct_author as module

    task, ctx = _fixture(tmp_path)
    monkeypatch.setattr(module, "_evidence",
                        lambda _i, _purpose, _source: {"target_api": [], "source": []})

    class Router:
        calls = 0

        def generate_java_source(self, role, messages, *, source_frame, **kwargs):
            self.calls += 1
            assert role == "coder"
            assert "real target" in messages[0]["content"]
            return source_frame["prefix"] + (
                'System.out.println("registered");'
            ) + source_frame["suffix"]

    router = Router()
    first = compose_method(router, task, ctx, owner_name="AlienRegistry")
    second = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert router.calls == 1
    assert first == second
    assert 'System.out.println("registered");' in first
    assert "registerAliens" in first


def test_direct_field_generates_whole_initialized_declaration(tmp_path, monkeypatch):
    from dataclasses import replace
    from minecraft_mod_ai import construction_member_direct_author as module

    task, ctx = _fixture(tmp_path)
    field_symbol = "demo.AlienRegistry#ALIEN_TYPE"
    ctx["symbols"][field_symbol] = {
        "kind": "FIELD", "owner": "demo.AlienRegistry", "name": "ALIEN_TYPE",
        "return_type": "java.lang.String", "static": True, "final": True,
        "visibility": "public", "side": "COMMON",
    }
    task = replace(task, owner_symbol=field_symbol, task_id="alien_type_field")
    assert eligible_method(task, ctx["symbols"], ctx)
    monkeypatch.setattr(module, "_evidence",
                        lambda _i, _purpose, _source: {"target_api": [], "source": []})

    class FieldRouter:
        calls = 0

        def generate_java_source(self, role, messages, *, source_frame, **kwargs):
            self.calls += 1
            assert source_frame["kind"] == "field"
            assert "empty_value" not in source_frame
            return source_frame["prefix"] + '"alien"' + source_frame["suffix"]

    router = FieldRouter()
    source = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert router.calls == 1
    assert 'public static final java.lang.String ALIEN_TYPE = "alien";' == source
