from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai.complete_spec import ProductionModule


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


def test_package_import_has_one_live_runtime_bootstrap_owner() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    init_text = (package / "__init__.py").read_text(encoding="utf-8")
    bootstrap = (package / "runtime_bootstrap.py").read_text(encoding="utf-8")

    assert init_text.count(
        "from .runtime_bootstrap import initialize_runtime as _initialize_runtime"
    ) == 1
    assert init_text.count("_initialize_runtime()") == 1
    assert "def initialize_runtime() -> None:" in bootstrap
    assert "_INITIALIZED = False" in bootstrap
    assert "with _LOCK:" in bootstrap


def test_retired_runtime_composition_files_stay_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert (package / "runtime_bootstrap.py").is_file()
    for name in (
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

    assert len(calls) == 2
    repair_prompt = calls[1][-1]["content"]
    assert "top-level contract must be exactly" in repair_prompt
    assert "required `public static void initialize()` is missing" in repair_prompt
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
    """Generic semantic-Java atomic fixture; state_model is host-compiled elsewhere."""
    base = _module(path, symbol)
    config = dict(base.config)
    config["implementation_ir_node"] = {
        "symbol": symbol,
        "public_api": ["public static void initialize()"],
        "activation": True,
    }
    config["implementation_section"] = "algorithm"
    config["implementation_dependency_context"] = "[]"
    config["implementation_atomic_concerns"] = [
        {
            "sequence": 0,
            "identifier": "feature/algorithm/steps",
            "concern": "steps",
            "task": "Resolve deterministic algorithm steps.",
            "rules": [],
            "record_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "sequence": 1,
            "identifier": "feature/algorithm/branches",
            "concern": "branches",
            "task": "Resolve algorithm branches.",
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

def _native_field_parts(kind, name, initializer):
    return [
        {"part": "fields"}, {"type": kind, "name": name, "initializer": initializer},
        {"part": "done"},
    ]


def _native_method_parts(name, body):
    return [
        {"part": "methods"}, {"return_type": "boolean", "name": name},
        {"part": "body"}, {"statement": body},
        {"part": "done"}, {"part": "done"},
    ]


def test_ir_atomic_concerns_are_isolated_and_compiled_as_one_host_file(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter([
        "private static int balance = 0;",
        "private static boolean valid() { return balance >= 0; }",
    ])
    calls: list[tuple[str, dict[str, object]]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append((concern, dict(kwargs)))
            return next(responses)

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

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
    concern_transitions = [
        name
        for index, (name, _kwargs) in enumerate(calls)
        if index == 0 or calls[index - 1][0] != name
    ]
    assert concern_transitions == ["steps", "branches"]
    assert all(kwargs.get("enable_tools") is False for _name, kwargs in calls)
    assert "static int balance = 0;" in source
    assert "private static boolean valid()" in source
    assert "MMM_ATOMIC_CONCERN_STEPS_MEMBERS_START" in source
    assert "MMM_ATOMIC_CONCERN_BRANCHES_MEMBERS_START" in source
    assert result["generation_verification"]["mode"] == "gradle_compile_java_semantic_concerns"
    assert result["generation_verification"]["atomic_concern_count"] == 2

def test_production_tree_sitter_unwraps_accidental_outer_class(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter([
        """
package accidental.wrapper;
import java.util.List;

public final class AccidentalOuter {
    private AccidentalOuter() {}
    private static int balance = 0;
    public static void initialize() {}
}
""",
        "private static boolean valid() { return balance >= 0; }",
    ])

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            return next(responses)

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)
    direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    source = (root / path).read_text(encoding="utf-8")
    assert "package accidental.wrapper" not in source
    assert "import java.util.List" not in source
    assert "AccidentalOuter" not in source
    assert "private static int balance = 0;" in source
    assert "private static boolean valid()" in source


def test_atomic_concern_compile_repair_reopens_only_localized_concern(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter([
        "private static Object value = new Object(1);",
        "private static Object value = new Object();",
        "private static boolean valid() { return value != null; }",
    ])
    calls: list[tuple[str, bool]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append((concern, bool(payload.get("repair_failure"))))
            return next(responses)

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            source = (project_root / path).read_text(encoding="utf-8")
            if "new Object(1)" not in source:
                return SimpleNamespace(status="PASS", commands=(), error=None)
            line = next(
                index for index, text in enumerate(source.splitlines(), start=1)
                if "new Object(1)" in text
            )
            log = project_root / ".minecraft_ai/logs/atomic.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                f"{project_root / path}:{line}: error: constructor Object cannot be applied to given types",
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

    concern_transitions: list[tuple[str, bool]] = []
    for call in calls:
        if not concern_transitions or concern_transitions[-1] != call:
            concern_transitions.append(call)
    assert concern_transitions == [
        ("steps", False),
        ("steps", True),
        ("branches", False),
    ]
    source = (root / path).read_text(encoding="utf-8")
    assert "new Object(1)" not in source
    assert "new Object()" in source
    assert "private static boolean valid()" in source
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
