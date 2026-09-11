"""Real JVM fixtures plus failure-injection contracts for integrity admission."""
import json
import shutil
import subprocess
from zipfile import ZipFile

import pytest

from minecraft_mod_ai.canonical_schema_compiler import canonical_leaves, compile_all_schemas
from minecraft_mod_ai.implementation_identity import compute_content_hash
from minecraft_mod_ai.jar_api_extractor import inspect_jar, parse_class, ClassFormatError
from minecraft_mod_ai.tiny_mappings import TinyMappings, MappingError
from minecraft_mod_ai.java_symbol_resolver import JavaSymbolResolver, JavaSymbolResolverError


@pytest.fixture(scope="module")
def jvm_fixture(tmp_path_factory):
    if not shutil.which("javac"):
        pytest.skip("JDK required for real bytecode fixture")
    root = tmp_path_factory.mktemp("real-jvm")
    sources = {
        "net/fabricmc/api/EnvType.java": "package net.fabricmc.api; public enum EnvType {CLIENT,SERVER}",
        "net/fabricmc/api/Environment.java": """package net.fabricmc.api;
            import java.lang.annotation.*;
            @Retention(RetentionPolicy.CLASS) @Target({ElementType.TYPE,ElementType.METHOD,ElementType.FIELD})
            public @interface Environment { EnvType value(); }""",
        "api/Library.java": """package api;
            import net.fabricmc.api.*;
            public class Library {
                public static final long LARGE=1234567891011L;
                public int number;
                public static int choose(int n) { return n; }
                public static String choose(String s) { return s; }
                public String[] arrays(String[][] s) { return s[0]; }
                @Environment(EnvType.CLIENT) public void clientOnly() {}
            }""",
        "api/Client.java": "package api; @net.fabricmc.api.Environment(net.fabricmc.api.EnvType.CLIENT) public class Client {}",
    }
    paths = []
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    classes = root / "classes"
    subprocess.run([shutil.which("javac"), "--release", "17", "-d", str(classes), *paths], check=True, capture_output=True)
    jar = root / "api.jar"
    with ZipFile(jar, "w") as archive:
        for path in classes.rglob("*.class"):
            archive.write(path, path.relative_to(classes).as_posix())
    return root, jar, classes


def test_all_schemas_have_real_leaf_contracts():
    schemas = compile_all_schemas()
    assert len(canonical_leaves()) == 346
    assert len(schemas) == 692
    from jsonschema import Draft202012Validator, ValidationError
    for leaf in canonical_leaves():
        for direction in ("input", "output"):
            schema = schemas[f"{leaf}:{direction}"]
            assert schema["additionalProperties"] is False
            with pytest.raises(ValidationError):
                Draft202012Validator(schema).validate({})


def test_schema_registry_rejects_conflict_and_nested_errors():
    from minecraft_mod_ai.type_registry import TypeRegistry, TypeValidationError
    registry = TypeRegistry()
    schema = {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer", "minimum": 1}}, "additionalProperties": False}
    registry.register_json_schema("x", schema)
    schema["required"].clear()
    with pytest.raises(TypeValidationError):
        registry.validate_input("x", {})
    with pytest.raises(TypeValidationError):
        registry.validate_input("x", {"count": True})
    with pytest.raises(TypeValidationError, match="Conflicting"):
        registry.register_json_schema("x", {"type": "string"})


def test_real_class_descriptors_and_annotations(jvm_fixture):
    _, jar, _ = jvm_fixture
    classes = inspect_jar(jar, java_version=17)
    lib = classes["api/Library"]
    assert {m.descriptor for m in lib.members if m.name == "choose"} == {"(I)I", "(Ljava/lang/String;)Ljava/lang/String;"}
    assert next(m for m in lib.members if m.name == "arrays").descriptor == "([[Ljava/lang/String;)[Ljava/lang/String;"
    assert next(m for m in lib.members if m.name == "clientOnly").side == "CLIENT"
    assert classes["api/Client"].side == "CLIENT"
    with pytest.raises(ClassFormatError):
        parse_class(b"not a class")


def test_real_javac_overload_instance_and_field_resolution(jvm_fixture):
    _, jar, _ = jvm_fixture
    resolver = JavaSymbolResolver([jar], "17")
    source = 'import api.Library; class T { int f() { Library a=new Library(); a.number=4; return Library.choose(a.number); } }'
    symbols = resolver.extract_symbols_from_source(source)
    chosen = next(s for s in symbols if s.name == "choose")
    assert (chosen.owner, chosen.descriptor, chosen.is_static) == ("api/Library", "(I)I", True)
    assert any(s.name == "number" and s.descriptor == "I" and s.kind == "FIELD" for s in symbols)
    assert any(s.owner == "api/Library" and s.kind == "CONSTRUCTOR" for s in symbols)
    with pytest.raises(JavaSymbolResolverError):
        resolver.resolve_method_call("class T { void f() { Missing.call(); } }", "call")
    with pytest.raises(JavaSymbolResolverError, match="NOT_UNIQUE"):
        resolver.resolve_method_call('import api.Library; class T { void f() { Library.choose(1); Library.choose("a"); } }', "choose")


def test_client_field_and_type_rejected(jvm_fixture):
    from minecraft_mod_ai.integrity_validators import validate_side
    _, jar, _ = jvm_fixture
    for source in ("import api.Library; class T { void f() { new Library().clientOnly(); } }",
                   "class T { api.Client field; }"):
        with pytest.raises(ValueError, match="SIDE_VIOLATION"):
            validate_side(source, leaf_id="minecraft/entity/registry", side="COMMON", classpath=[jar], java_version="17")


def test_tiny_overloads_arrays_and_reverse_namespaces():
    mappings = TinyMappings.parse("tiny\t2\t0\tofficial\tintermediary\tnamed\n"
        "c\ta\tclass_1\tapi/Example\n\tm\t(La;[La;)La;\tx\tmethod_1\tconvert\n"
        "\tm\t(I)I\tx\tmethod_2\tconvert\n\tf\tLa;\tf\tfield_1\tinstance\n")
    assert mappings.member("a", "x", "(La;[La;)La;", "m", "official", "named") == (
        "api/Example", "convert", "(Lapi/Example;[Lapi/Example;)Lapi/Example;")
    assert mappings.member("api/Example", "convert", "(I)I", "m", "named", "official") == ("a", "x", "(I)I")
    with pytest.raises(MappingError):
        mappings.class_name("a", "missing", "named")
    with pytest.raises(MappingError):
        TinyMappings.parse("tiny\t2\t0\ta\ta")


def test_jar_epoch_changes_with_api_and_requires_inspection(jvm_fixture):
    from minecraft_mod_ai.api_epoch_catalog import inspect_api_epoch, determine_api_epoch, compile_epoch_templates
    from minecraft_mod_ai.implementation_registry import ImplementationRegistry
    root, jar, _ = jvm_fixture
    inspection = inspect_api_epoch("test", loader="fabric", namespace="named", java_version=17, jars=[jar])
    assert determine_api_epoch("test", inspection=inspection).startswith("fabric:sha256:")
    with pytest.raises(ValueError):
        determine_api_epoch("other", inspection=inspection)
    with pytest.raises(ValueError, match="MIXED_LOADER"):
        inspect_api_epoch("test", loader="neoforge", namespace="named", java_version=17, jars=[jar])
    templates = compile_epoch_templates(inspection, declarations=[{
        "canonical_leaf": "minecraft/item/registry", "render": "api.Library.choose(1)",
        "api_requirements": [{"owner": "api/Library", "name": "choose", "descriptor": "(I)I", "static": True}],
    }], output_dir=root / "templates", registry=ImplementationRegistry())
    assert templates[0]["api_epoch"] == inspection.epoch_id


def test_java_syntax_uses_parser():
    from minecraft_mod_ai.integrity_validators import validate_java_syntax
    validate_java_syntax("class T { int a=1; }", java_version="17", filename="T.java")
    with pytest.raises(ValueError, match="JAVA_ANALYSIS_FAILED"):
        validate_java_syntax("class T { int a=; }", java_version="17", filename="T.java")


def test_semantic_validator_does_not_accept_missing_contract():
    from minecraft_mod_ai.integrity_validators import validate_semantic_contract
    with pytest.raises(ValueError, match="SEMANTIC_BINDING"):
        validate_semantic_contract("anything", contract={}, context_id="ctx", leaf_id="leaf")


def test_evidence_rechecks_raw_blobs_and_exact_context(tmp_path):
    # Synthetic unit-test record, never used as Minecraft execution evidence.
    from minecraft_mod_ai.evidence_store import EvidenceStore
    from minecraft_mod_ai.integrity_evidence import _blob, verify_execution_evidence
    store = EvidenceStore(tmp_path)
    blob = _blob(store, b"unit fixture")
    expected = {"leaf_id": "fixture"}
    record = {"format": "mmm.execution-evidence.v1", "expected": expected,
              "project_files": {}, "classpath": {}, "artifacts": {"T.class": blob}, "reports": {"test.xml": blob},
              "gates": {"compile": {"status": "PASS", "exit_code": 0, "stdout": blob, "stderr": blob},
                        "gametest": {"status": "PASS", "exit_code": 0, "stdout": blob, "stderr": blob,
                                     "counts": {"tests": 0, "skipped": 0, "failures": 0, "errors": 0}}}}
    evidence_id = _blob(store, json.dumps(record).encode())
    with pytest.raises(ValueError, match="TESTS_MISSING"):
        verify_execution_evidence(store, evidence_id, expected=expected)

    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        verify_execution_evidence(store, evidence_id, expected={"leaf_id": "different"})
    (tmp_path / "blobs" / blob.split(":")[1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="BLOB_CORRUPT"):
        verify_execution_evidence(store, evidence_id, expected=expected)


def test_minecraft_nested_gametest_report_is_counted():
    from minecraft_mod_ai.integrity_evidence import _report_counts
    raw = b'<testsuite><testsuite time="1"><testcase name="registry"/></testsuite></testsuite>'
    assert _report_counts(raw) == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    failed = b'<testsuites><testsuite><testcase name="bad"><failure/></testcase></testsuite></testsuites>'
    assert _report_counts(failed)["failures"] == 1


def test_registry_rehashes_template_before_reuse(tmp_path):
    from minecraft_mod_ai.implementation_registry import ImplementationRegistry
    registry = ImplementationRegistry()
    path = tmp_path / "template.yaml"
    path.write_text("id: test\n")
    impl = registry.register_template("test", path)
    assert registry.verify_implementation_hash("test", impl.content_sha256)
    path.write_text("id: modified\n")
    assert not registry.verify_implementation_hash("test", impl.content_sha256)


def test_complete_bootstrap_and_candidate_catalog():
    from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.integrity_catalog import audit_reachability
    authority = bootstrap_integrity()
    assert len(authority.types.list_types()) == 692
    assert len(authority.executors) == 346
    facts = build_version_facts("1.21.5", base_facts={})
    report = audit_reachability(facts, authority=authority)
    assert report["status"] == "PASS", report
    assert all(row["state"] == "not_reviewed" for row in facts["leaf_bindings"].values())
    assert all("evidence_id" not in row["implementation"] for row in facts["leaf_bindings"].values())


def test_generator_executes_registered_callable_and_emits_host_receipt(monkeypatch):
    from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity
    from minecraft_mod_ai import fixed_template_generation
    leaf = "minecraft/entity/registry"
    authority = bootstrap_integrity()
    monkeypatch.setattr(fixed_template_generation, "generate_fixed_template_value", lambda *a, **k: "class EntityRegistration {}")
    spec = {"leaf_id": leaf, "context_id": "ctx", "requirement": "Create entity registration",
            "target_path": "src/main/java/EntityRegistration.java", "language": "java", "operation": "CREATE_FILE",
            "side": "COMMON", "bindings": {"entity": "example:entity"},
            "render_mold": "class EntityRegistration {}", "slots": [], "output_schema": {"type": "string", "pattern": "EntityRegistration"}}
    output = authority.executors["python_generator:" + leaf](
        {"entity_identity": "example:entity", "entity_registry_input": spec},
        leaf_id=leaf, router=object(), authority=authority)
    assert output["entity_registry_artifact"] == "class EntityRegistration {}"
    assert output["entity_registry_receipt"]["content_sha256"] == compute_content_hash(b"class EntityRegistration {}")


@pytest.mark.parametrize("field", ["implementation_sha256", "validator_sha256", "input_schema_sha256",
                                   "output_schema_sha256", "authority_sha256"])
def test_runtime_rejects_tampered_binding_before_execution(field):
    from types import SimpleNamespace
    from minecraft_mod_ai.populate_version_artifact_rules import make_implementation
    from minecraft_mod_ai.integrity_dispatcher import verify_job_binding
    from minecraft_mod_ai.artifact_job import ArtifactJob
    leaf = "minecraft/entity/registry"
    impl = make_implementation(leaf, "1.20.1")
    impl[field] = "sha256:" + "0" * 64
    resolved = SimpleNamespace(require_leaf_binding=lambda _: {"state": "admitted", "implementation": impl})
    job = ArtifactJob("entity", "", "owner", canonical_leaf=leaf, implementation_id=impl["implementation_id"])
    with pytest.raises(ValueError, match="RUNTIME_INTEGRITY_BINDING_MISMATCH"):
        verify_job_binding(job, resolved, {})


def test_no_context_cannot_execute_canonical_job():
    from minecraft_mod_ai.artifact_job import ArtifactJob
    from minecraft_mod_ai.task_template_runner import execute_artifact_template
    from minecraft_mod_ai.implementation_identity import ExecutorType
    job = ArtifactJob("entity", "", "owner", canonical_leaf="minecraft/entity/registry",
                      implementation_id="python_generator:minecraft/entity/registry", executor_type=ExecutorType.PYTHON_GENERATOR)
    with pytest.raises(ValueError, match="VERSION_CONTEXT_REQUIRED"):
        execute_artifact_template(job)
