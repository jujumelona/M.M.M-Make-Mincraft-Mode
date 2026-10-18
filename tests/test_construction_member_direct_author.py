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

    # Production always passes pinned classpath; these author-only unit tests
    # mock compiler proof rather than requiring JDK 25 and a real Fabric JAR.
    from minecraft_mod_ai import javac_bridge
    ctx["java_classpath"] = ["/pinned/fabric.jar"]
    monkeypatch.setattr(javac_bridge, "analyze_java", lambda *_a, **_k: [])
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

    from minecraft_mod_ai import javac_bridge
    ctx["java_classpath"] = ["/pinned/fabric.jar"]
    monkeypatch.setattr(javac_bridge, "analyze_java", lambda *_a, **_k: [])
    router = FieldRouter()
    source = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert router.calls == 1
    assert 'public static final java.lang.String ALIEN_TYPE = "alien";' == source


def test_bad_java_types_never_checkpointed_before_compiler_proof(tmp_path, monkeypatch):
    from minecraft_mod_ai import construction_member_direct_author as module
    from minecraft_mod_ai import javac_bridge

    task, ctx = _fixture(tmp_path)
    ctx["java_classpath"] = ["/pinned/fabric-classpath.jar"]
    monkeypatch.setattr(module, "_evidence",
                        lambda *_args: {"target_api": [], "source": []})
    examined = []

    def fake_javac(source, *, classpath, java_version, filename):
        examined.append((source, tuple(classpath), java_version, filename))
        if "ImaginaryAlienEntity" in source:
            raise ValueError("JAVA_ANALYSIS_FAILED: cannot find symbol ImaginaryAlienEntity")
        return [{"name": "registerAliens", "kind": "METHOD"}]

    monkeypatch.setattr(javac_bridge, "analyze_java", fake_javac)

    class Router:
        def __init__(self):
            self.calls = 0

        def generate_java_source(self, _role, _messages, *, source_frame, **_kw):
            self.calls += 1
            body = ("new ImaginaryAlienEntity();" if self.calls == 1
                    else 'System.out.println("known Java API");')
            return source_frame["prefix"] + body + source_frame["suffix"]

    router = Router()
    source = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert router.calls == 2
    assert len(examined) == 2
    assert "ImaginaryAlienEntity" not in source
    assert examined[0][1] == ("/pinned/fabric-classpath.jar",)
    assert examined[0][2] == "25"
    assert examined[0][3] == "AlienRegistry.java"
    assert "\npublic class AlienRegistry" in examined[1][0]
    # Repeat must revalidate the exact cached source without more model work.
    again = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert again == source and router.calls == 2
    assert len(examined) == 3


def test_invalid_direct_method_never_publishes_checkpoint(tmp_path, monkeypatch):
    from minecraft_mod_ai import construction_member_direct_author as module
    from minecraft_mod_ai import javac_bridge
    import pytest

    task, ctx = _fixture(tmp_path)
    ctx["java_classpath"] = ["/pinned/fabric.jar"]
    monkeypatch.setattr(module, "_evidence",
                        lambda *_args: {"target_api": [], "source": []})

    def reject(*_args, **_kwargs):
        raise ValueError("JAVA_ANALYSIS_FAILED: unresolved entity class")

    monkeypatch.setattr(javac_bridge, "analyze_java", reject)

    class Router:
        def generate_java_source(self, _role, _messages, *, source_frame, **_kw):
            return source_frame["prefix"] + "new ImaginaryAlienEntity();" + source_frame["suffix"]

    with pytest.raises(ValueError, match="CONSTRUCTION_DIRECT_TARGET_SYMBOL_INVALID"):
        compose_method(Router(), task, ctx, owner_name="AlienRegistry")
    assert not list((tmp_path / ".mmm" / "construction_direct_methods").glob("*.json"))


def test_invalid_java_from_router_is_passed_to_repair_not_none(tmp_path, monkeypatch):
    import json
    from minecraft_mod_ai import construction_member_direct_author as module
    from minecraft_mod_ai import first_pass_java_declarations, javac_bridge
    from minecraft_mod_ai.java_region_parser import JavaRegionParseError

    task, ctx = _fixture(tmp_path)
    ctx["java_classpath"] = ["/pinned/fabric.jar"]
    monkeypatch.setattr(module, "_evidence",
                        lambda *_a: {"target_api": [], "source": []})
    monkeypatch.setattr(javac_bridge, "analyze_java", lambda *_a, **_kw: [])
    failures = []
    monkeypatch.setattr(first_pass_java_declarations, "_record_source_failure",
                        lambda source, **kwargs: failures.append((source, kwargs)))

    class Router:
        calls = 0
        def generate_java_source(self, _role, messages, *, source_frame, **_kw):
            self.calls += 1
            if self.calls == 1:
                bad = source_frame["prefix"] + "}" + source_frame["suffix"]
                error = JavaRegionParseError("syntax error at 6:1: '}'")
                error.failed_source = bad  # as attached by model_router
                raise error
            repair = json.loads(messages[-1]["content"])
            assert repair["previous_source"] == bad
            assert repair["diagnostic"] == "syntax error at 6:1: '}'"
            assert repair["required_suffix"] == source_frame["suffix"]
            return (source_frame["prefix"]
                    + 'System.out.println("fixed");'
                    + source_frame["suffix"])

    router = Router()
    fixed = compose_method(router, task, ctx, owner_name="AlienRegistry")
    assert router.calls == 2
    assert 'System.out.println("fixed");' in fixed
    assert len(failures) == 1
    assert failures[0][0].endswith("}" + "}")
    assert failures[0][1]["phase"] == "direct_author"
