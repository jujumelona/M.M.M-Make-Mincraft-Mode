"""Regression: generated dependency declarations must survive prompt boundaries."""
from __future__ import annotations

import json

from test_construction_host_source import SPEC, Router, _context

from minecraft_mod_ai.construction_host_source import compose_host_java_source
from minecraft_mod_ai.host_prerequisite_api import java_api_declarations


def _dependencies(tmp_path):
    sources = {
        "example/AuthoredProgram.java": (
            "package example; public final class AuthoredProgram {\n"
            + "private static final String INTERNAL = \"" + "x" * 125_000 + "\";\n"
            + "public static Object invoke(String action) { return null; } }"
        ),
        "example/AuthoredStateModel.java": """package example;
import java.util.Map;
public final class AuthoredStateModel {
    private static final String INTERNAL = "implementation detail";
    public static synchronized Object getState(String name) { return null; }
    public static synchronized Object getState(String name, Map<String, Object> context) { return null; }
    public static synchronized void setState(String name, Object value) {}
    public static synchronized boolean readState_ready() { return true; }
    public static synchronized void writeState_ready(boolean value) {}
    private static Object hidden() { return null; }
} """,
        "example/AuthoredHostCapabilities.java": """package example;
public final class AuthoredHostCapabilities { public static void sendMessage(Object player, String text) {} }
""",
        "example/generated/AlienEntity.java": """package example.generated;
public class AlienEntity extends java.lang.Object { public AlienEntity() {} }
""",
        "example/generated/AlienEntityRegistry.java": """package example.generated;
public final class AlienEntityRegistry {
    public static final Class<AlienEntity> ENTITY_TYPE = AlienEntity.class;
    public static void register() {}
}
""",
    }
    context = []
    for relative, source in sources.items():
        path = "src/main/java/" + relative
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8")
        context.append({"path": path})
    return context


def test_generation_and_compiler_repair_receive_every_dependency_api(tmp_path, monkeypatch):
    # The fifth dependency was omitted by the production four-file limit;
    # the first file exceeded the previous size cap. Neither is optional ABI.
    source_context = _dependencies(tmp_path)
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence", lambda *a, **k: {})
    monkeypatch.setattr("minecraft_mod_ai.construction_coverage.record_authored_source_calls", lambda *a, **k: None)

    def compile_source(source, *a, **k):
        if "return true;" not in source:
            line = next(i for i, text in enumerate(source.splitlines(), 1) if "return " in text)
            raise RuntimeError(f"Bridge.java:{line}: error: cannot find symbol")

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._verify_full_source", compile_source)

    class DependencyAwareRouter(Router):
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            payloads = [json.loads(m["content"]) for m in messages if m["role"] == "user"]
            payload = payloads[0]
            apis = payload["materialized_prerequisite_api"]
            assert len(apis) == 5
            evidence = json.dumps(apis)
            assert "boolean readState_ready()" in evidence
            assert "void writeState_ready(boolean value)" in evidence
            assert "getState(" not in evidence and "setState(" not in evidence
            assert "Class<AlienEntity> ENTITY_TYPE" in evidence
            assert "Object invoke(String action)" in evidence
            assert "implementation detail" not in evidence
            assert "hidden()" not in evidence
            assert len(evidence) < 6_000
            body = "return missing;" if data_name.startswith("submit_") else "return true;"
            return source_frame["prefix"] + body + source_frame["suffix"]

    result = compose_host_java_source(
        DependencyAwareRouter(), SPEC, _context(tmp_path), package="example", name="Bridge",
        owner="example.Bridge", source_context=source_context,
    )
    assert "return true;" in result


def test_enum_methods_and_private_members_with_adjacent_comments():
    api = java_api_declarations("""package example;
public enum State {
    READY, RUNNING;
    public boolean active() { return this == RUNNING; }
    private/*legal comment*/ String secret() { return "secret"; }
    private/*legal comment*/ class Hidden { public void leak() {} }
    public static class Details { public <T> T identity(T value) { return value; } }
}
""", path="State.java")
    text = json.dumps(api)
    assert "boolean active()" in text
    assert "READY" in text and "RUNNING" in text
    assert "public <T> T identity(T value)" in text
    assert "secret" not in text and "Hidden" not in text and "leak" not in text


def test_record_compact_constructor_preserves_canonical_parameters():
    api = java_api_declarations("""package example;
public record Result(String name, int value) {
    public Result { java.util.Objects.requireNonNull(name); }
}
""", path="Result.java")
    assert "public Result(String name, int value);" in api["types"][0]["members"]


def test_api_identity_uses_package_node_and_modifier_tokens():
    api = java_api_declarations('''package /* comment */
example.generated;
public class Demo {
    @Hint("a private method") public void visible() {}
    private/* comment */ void hidden() {}
}
''', path="Demo.java")
    assert api["types"][0]["owner"] == "example.generated.Demo"
    assert any("void visible()" in member for member in api["types"][0]["members"])
    assert "hidden" not in json.dumps(api)



def test_materialized_dependency_symbols_are_actual_public_source_declarations():
    from minecraft_mod_ai.host_prerequisite_api import (
        java_api_declarations, inspected_prerequisite_symbols)
    java = """package example;
import net.minecraft.world.entity.Entity;
public class AlienEntity extends Entity {
    public static final String LABEL = "alien";
    public int getThreatLevel() { return 2; }
    public static boolean isAlienPresent() { return true; }
    private boolean isSecretPresent() { return false; }
}
"""
    api = java_api_declarations(java, path="src/main/java/example/AlienEntity.java")
    symbols = inspected_prerequisite_symbols([api])
    assert symbols["example.AlienEntity"]["superclass"] == "net.minecraft.world.entity.Entity"
    assert symbols["materialized:example.AlienEntity#LABEL"]["return_type"] == "java.lang.String"
    threat = symbols["materialized:example.AlienEntity#getThreatLevel()"]
    assert threat["kind"] == "METHOD" and threat["return_type"] == "int"
    assert threat["static"] is False
    observed = symbols["materialized:example.AlienEntity#isAlienPresent()"]
    assert observed["static"] is True
    assert not any("isSecretPresent" in key for key in symbols)
    assert all(row["provenance"] == "materialized_prerequisite_declaration"
               for row in symbols.values())


def test_unresolved_generated_type_is_not_fabricated_as_java_abi():
    from minecraft_mod_ai.host_prerequisite_api import (
        java_api_declarations, inspected_prerequisite_symbols)
    source = """package example;
public class GeneratedOwner {
    public MissingEntity getMissing() { return null; }
    public long getTick() { return 1L; }
}
"""
    symbols = inspected_prerequisite_symbols([java_api_declarations(
        source, path="src/main/java/example/GeneratedOwner.java")])
    assert "materialized:example.GeneratedOwner#getTick()" in symbols
    assert "materialized:example.GeneratedOwner#getMissing()" not in symbols
