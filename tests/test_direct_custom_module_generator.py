from __future__ import annotations

import json

import pytest
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai import custom_module_generator as direct


def _module(path: str, symbol: str) -> ProductionModule:
    task = {
        "task_id": "authored_feature_001",
        "semantic_outcome": "implement the approved feature",
        "implementation_obligations": ["implement the approved feature"],
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": f"{path}#{symbol}",
                "status": "existing",
            }
        ],
        "required_gates": ["target_compile"],
    }
    return ProductionModule(
        module_id="authored_feature_001",
        kind="custom_java",
        config={"implementation": "custom", "evidence_task": task},
        required_gates=("target_compile",),
    )


def _project(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "project"
    path = "src/main/java/example/AuthoredFeature001.java"
    symbol = "AuthoredFeature001"
    target = root / path
    target.parent.mkdir(parents=True)
    target.write_text(
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() {\n"
        "        // MMM_AUTHORED_FEATURE_BODY\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    return root, path, symbol


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(
        minecraft_version="1.21.1",
        loader="fabric",
        java_version=21,
        yarn_mappings="1.21.1+build.3",
    )


def test_invariant_failure_repairs_from_complete_current_source(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    bad = (
        "package example;\n\n"
        "public class AuthoredFeature001 {\n"
        "    public void initialize() {}\n"
        "}\n"
    )
    good = (
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() { System.out.println(\"ok\"); }\n"
        "}\n"
    )

    class Router:
        def generate_text(self, role, messages, **kwargs):
            calls.append((list(messages), dict(kwargs)))
            content = bad if len(calls) == 1 else good
            return content

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    generator = direct.CustomModuleGenerator(Router())
    result = generator.generate(
        root,
        module=_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert len(calls) == 2
    assert calls[0][1]["enable_tools"] is False
    assert calls[0][1]["response_format"] == "text"
    assert "response_schema" not in calls[0][1]
    assert "output_token_ceiling" not in calls[0][1]
    assert '"model_tool_choice_required": false' in calls[0][0][-1]["content"]
    assert '"resolved_before_first_coder_decode": true' in calls[0][0][-1]["content"]
    repair_prompt = calls[1][0][-1]["content"]
    assert bad in repair_prompt
    assert "public final class AuthoredFeature001" in repair_prompt
    assert (root / path).read_text(encoding="utf-8") == good


def test_compiler_failure_repairs_from_complete_source_and_exact_log(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls: list[list[dict[str, str]]] = []
    first = (
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() { MissingType.run(); }\n"
        "}\n"
    )
    second = (
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() { System.out.println(\"fixed\"); }\n"
        "}\n"
    )

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            calls.append(list(messages))
            content = first if len(calls) == 1 else second
            return content

    class Runner:
        attempts = 0

        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            Runner.attempts += 1
            if Runner.attempts == 1:
                log = project_root / ".minecraft_ai/logs/gradle-compile-java.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(
                    "AuthoredFeature001.java:5: error: cannot find symbol MissingType",
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    status="FAIL",
                    commands=(SimpleNamespace(log_path=str(log)),),
                    error="Gradle Java compilation failed.",
                )
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert len(calls) == 2
    repair_prompt = calls[1][-1]["content"]
    assert first in repair_prompt
    assert "cannot find symbol MissingType" in repair_prompt
    assert (root / path).read_text(encoding="utf-8") == second


def test_package_import_no_longer_bootstraps_runtime_mutation() -> None:
    init_text = (
        Path(__file__).resolve().parents[1]
        / "minecraft_mod_ai"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "runtime_bootstrap" not in init_text
    assert "initialize_runtime()" not in init_text


def test_removed_runtime_composition_files_do_not_exist() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    for name in (
        "runtime_bootstrap.py",
        "runtime_contract_composer.py",
        "runtime_contract_wrappers.py",
        "runtime_finalization.py",
        "runtime_wrapper_integrity.py",
        "runtime_composer_hardening.py",
    ):
        assert not (package / name).exists(), name


def test_invalid_or_truncated_model_output_is_not_blindly_retried(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    original = (root / path).read_bytes()
    calls: list[list[dict[str, str]]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            calls.append(list(messages))
            return "not valid Java source"

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            raise AssertionError("compiler must not run for malformed model output")

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    with pytest.raises(direct.CustomModuleGenerationError, match="DIRECT_CODER_COMPILE_FAILED"):
        direct.CustomModuleGenerator(Router()).generate(
            root,
            module=_module(path, symbol),
            minecraft_version="1.21.1",
            loader="fabric",
        )

    assert len(calls) == 1
    assert (root / path).read_bytes() == original


def test_host_reserved_missing_target_is_materialized_and_does_not_require_initialize(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = "src/main/java/example/FreshFeature.java"
    symbol = "FreshFeature"
    task = {
        "task_id": "fresh_feature",
        "semantic_outcome": "create one fresh exact source",
        "implementation_obligations": ["create the exact source"],
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": f"{path}#{symbol}",
                "status": "host_reserved",
                "ownership": "exclusive",
            }
        ],
        "required_gates": ["target_compile"],
    }
    module = ProductionModule(
        module_id="fresh_feature",
        kind="custom_java",
        config={"implementation": "custom", "evidence_task": task},
        required_gates=("target_compile",),
    )
    source = (
        "package example;\n\n"
        "public final class FreshFeature {\n"
        "    private FreshFeature() {}\n"
        "    public static final int VALUE = 1;\n"
        "}\n"
    )
    prompts: list[str] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            prompts.append(messages[-1]["content"])
            return source

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=module,
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert result["patch_receipt"]["operations"][0]["operation"] == "create"
    assert "MMM_AUTHORED_FEATURE_BODY" in prompts[0]
    assert (root / path).read_text(encoding="utf-8") == source


def _atomic_module(path: str, symbol: str) -> ProductionModule:
    base = _module(path, symbol)
    config = dict(base.config)
    config["implementation_ir_node"] = {
        "symbol": symbol,
        "public_api": ["public static void initialize()"],
        "activation": True,
    }
    config["implementation_section"] = "state_model"
    config["implementation_dependency_context"] = "[]"
    config["implementation_atomic_concerns"] = [
        {
            "sequence": 0,
            "identifier": "feature/state_model/variables",
            "concern": "variables",
            "task": "Resolve variables.",
            "rules": [],
            "record_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "sequence": 1,
            "identifier": "feature/state_model/invariants",
            "concern": "invariants",
            "task": "Resolve invariants.",
            "rules": [],
            "record_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]
    return ProductionModule(
        module_id=base.module_id,
        kind=base.kind,
        config=config,
        required_gates=base.required_gates,
    )


def test_ir_atomic_concerns_are_isolated_and_compiled_as_one_host_file(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append((concern, dict(kwargs)))
            if concern == "variables":
                content = (
                    "<<<MMM_CONCERN_MEMBERS>>>\n"
                    "private static int balance = 0;\n"
                    "<<<MMM_CONCERN_INITIALIZE>>>\n"
                    "<<<MMM_CONCERN_END>>>"
                )
            else:
                content = (
                    "<<<MMM_CONCERN_MEMBERS>>>\n"
                    "public static boolean valid() { return balance >= 0; }\n"
                    "<<<MMM_CONCERN_INITIALIZE>>>\n"
                    "<<<MMM_CONCERN_END>>>"
                )
            return content

    class Runner:
        def __init__(self, _cache):
            pass
        def compile_java(self, _root):
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)
    result = direct.CustomModuleGenerator(Router()).generate(
        root, module=_atomic_module(path, symbol),
        minecraft_version="1.21.1", loader="fabric",
    )

    source = (root / path).read_text(encoding="utf-8")
    assert [name for name, _kwargs in calls] == ["variables", "invariants"]
    assert all(kwargs["enable_tools"] is False for _name, kwargs in calls)
    assert all(kwargs["response_format"] == "text" for _name, kwargs in calls)
    assert all("response_schema" not in kwargs for _name, kwargs in calls)
    assert "private static int balance = 0;" in source
    assert "public static boolean valid()" in source
    assert "MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_START" in source
    assert "MMM_ATOMIC_CONCERN_INVARIANTS_MEMBERS_START" in source
    assert result["generation_verification"]["mode"] == "gradle_compile_java_atomic_concerns"
    assert result["generation_verification"]["atomic_concern_count"] == 2


def test_atomic_concern_compile_repair_reopens_only_localized_concern(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls: list[str] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append(concern)
            repairing = bool(payload.get("repair_failure"))
            if concern == "variables":
                member = (
                    "private static Object value = new MissingType();"
                    if not repairing
                    else "private static Object value = new Object();" 
                )
            else:
                member = "public static boolean valid() { return value != null; }"
            content = (
                "<<<MMM_CONCERN_MEMBERS>>>\n" + member + "\n"
                "<<<MMM_CONCERN_INITIALIZE>>>\n"
                "<<<MMM_CONCERN_END>>>"
            )
            return content

    class Runner:
        def __init__(self, _cache):
            pass
        def compile_java(self, project_root):
            source = (project_root / path).read_text(encoding="utf-8")
            if "MissingType" not in source:
                return SimpleNamespace(status="PASS", commands=(), error=None)
            line = next(
                index for index, text in enumerate(source.splitlines(), start=1)
                if "MissingType" in text
            )
            log = project_root / ".minecraft_ai/logs/atomic.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                f"{project_root / path}:{line}: error: cannot find symbol MissingType",
                encoding="utf-8",
            )
            return SimpleNamespace(
                status="FAIL",
                commands=(SimpleNamespace(log_path=str(log)),),
                error="compile failed",
            )

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)
    result = direct.CustomModuleGenerator(Router()).generate(
        root, module=_atomic_module(path, symbol),
        minecraft_version="1.21.1", loader="fabric",
    )

    assert calls == ["variables", "invariants", "variables"]
    source = (root / path).read_text(encoding="utf-8")
    assert "MissingType" not in source
    assert "new Object()" in source
    assert "public static boolean valid()" in source
    assert result["generation_verification"]["atomic_repair_count"] == 1


def test_nonintegration_atomic_concern_cannot_write_initialize_body() -> None:
    from minecraft_mod_ai.atomic_concern_source import parse_concern_content

    content = (
        "<<<MMM_CONCERN_MEMBERS>>>\nprivate static int x;\n"
        "<<<MMM_CONCERN_INITIALIZE>>>\nx = 1;\n"
        "<<<MMM_CONCERN_END>>>"
    )
    with pytest.raises(direct.CustomModuleGenerationError, match="only integration"):
        parse_concern_content(content, section="state_model")


def test_atomic_concern_cannot_redeclare_type_or_entrypoint() -> None:
    from minecraft_mod_ai.atomic_concern_source import parse_concern_content

    content = (
        "<<<MMM_CONCERN_MEMBERS>>>\npublic class Escape {}\n"
        "<<<MMM_CONCERN_INITIALIZE>>>\n"
        "<<<MMM_CONCERN_END>>>"
    )
    with pytest.raises(direct.CustomModuleGenerationError, match="SCOPE_ESCAPE"):
        parse_concern_content(content, section="state_model")
