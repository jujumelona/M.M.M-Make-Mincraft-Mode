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



def test_invariant_failure_is_not_retried(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    original = (root / path).read_bytes()
    calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
    bad = (
        "package example;\n\n"
        "public class AuthoredFeature001 {\n"
        "    public void initialize() {}\n"
        "}\n"
    )

    class Router:
        def generate_text(self, role, messages, **kwargs):
            calls.append((list(messages), dict(kwargs)))
            return bad

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            raise AssertionError("compiler must not run for contract-invalid source")

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="DIRECT_CODER_FIRST_PASS_CONTRACT_FAILED",
    ):
        direct.CustomModuleGenerator(Router()).generate(
            root,
            module=_module(path, symbol),
            minecraft_version="1.21.1",
            loader="fabric",
        )

    assert len(calls) == 1
    assert calls[0][1]["enable_tools"] is False
    assert calls[0][1]["response_format"] == "text"
    assert (root / path).read_bytes() == original


def test_compiler_failure_is_not_sent_back_to_model(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    original = (root / path).read_bytes()
    calls: list[list[dict[str, str]]] = []
    first = (
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() { MissingType.run(); }\n"
        "}\n"
    )

    class Router:
        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            calls.append(list(messages))
            return first

    class Runner:
        attempts = 0

        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            Runner.attempts += 1
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

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="DIRECT_CODER_FIRST_PASS_COMPILE_FAILED",
    ):
        direct.CustomModuleGenerator(Router()).generate(
            root,
            module=_module(path, symbol),
            minecraft_version="1.21.1",
            loader="fabric",
        )

    assert len(calls) == 1
    assert Runner.attempts == 1
    assert (root / path).read_bytes() == original

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



def test_invalid_or_truncated_model_output_is_not_retried(
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

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="DIRECT_CODER_FIRST_PASS_CONTRACT_FAILED",
    ):
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


def test_every_execution_section_declares_platform_api_policy() -> None:
    schema = __import__(
        "minecraft_mod_ai.authored_execution_schema",
        fromlist=["SECTION_SPECS"],
    )
    assert schema.SECTION_SPECS
    for section, spec in schema.SECTION_SPECS.items():
        assert spec["platform_api_policy"] in {"forbidden", "host_grounded_only"}, section


def test_platform_api_policy_is_section_wide() -> None:
    atomic = __import__(
        "minecraft_mod_ai.atomic_concern_source",
        fromlist=["_validate_platform_api_admission"],
    )

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="ATOMIC_CONCERN_PLATFORM_API_FORBIDDEN",
    ):
        atomic._validate_platform_api_admission(
            (
                "private static Object encode(Object value) { "
                "return net.minecraft.nbt.NbtUtils.writeItemStack("
                "(net.minecraft.world.item.ItemStack) value); }"
            ),
            section="persistence",
            grounding={},
        )

    grounded = {
        "facts": [
            {
                "required_imports": [
                    "net.minecraft.resources.ResourceLocation",
                ],
                "api_symbols": {
                    "resource_location_factory": {
                        "owner": "net.minecraft.resources.ResourceLocation",
                    }
                },
            }
        ]
    }
    atomic._validate_platform_api_admission(
        (
            "private static Object id(String ns, String path) { "
            "return net.minecraft.resources.ResourceLocation.fromNamespaceAndPath(ns, path); }"
        ),
        section="resources_and_ui",
        grounding=grounded,
    )

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="ATOMIC_CONCERN_UNGROUNDED_PLATFORM_API",
    ):
        atomic._validate_platform_api_admission(
            (
                "private static Object bad(Object value) { "
                "return net.minecraft.nbt.NbtUtils.writeItemStack("
                "(net.minecraft.world.item.ItemStack) value); }"
            ),
            section="resources_and_ui",
            grounding=grounded,
        )


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

def _single_atomic_module(path: str, symbol: str) -> ProductionModule:
    base = _atomic_module(path, symbol)
    config = dict(base.config)
    config["implementation_atomic_concerns"] = [
        dict(config["implementation_atomic_concerns"][0])
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
    compile_calls = 0
    responses = iter([
        "private static int balance = 0;",
        "private static boolean valid() { return balance >= 0; }",
    ])
    calls: list[tuple[str, dict[str, object]]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append((concern, dict(kwargs)))
            return next(responses)

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            nonlocal compile_calls
            compile_calls += 1
            source = (project_root / path).read_text(encoding="utf-8")
            assert "static int balance = 0;" in source
            assert "private static boolean valid()" in source
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
    assert compile_calls == 1

def test_production_tree_sitter_unwraps_accidental_outer_class(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter([
        """
Here is the complete Java implementation for the requested concern.
package accidental.wrapper;
import java.util.List;

public final class AccidentalOuter {
    private AccidentalOuter() {}
    private static int balance = 0;
    public static void initialize() {}
}
This wrapper is complete.
""",
        "private static boolean valid() { return balance >= 0; }",
    ])

    calls = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            calls += 1
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
    assert calls == 2


def test_production_admits_reasoning_with_multiple_java_fences(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter([
        """
The user is asking me to implement the actors concern. Let me analyze it.

```java
public static void initialize() {}
```

I need to reconsider the design and provide only the final members.

```java
private static int balance = 0;
public static void initialize() { balance = 0; }
```
""",
        "private static boolean valid() { return balance >= 0; }",
    ])

    calls = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            calls += 1
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
    assert "private static int balance = 0;" in source
    assert "private static boolean valid()" in source
    assert "The user is asking" not in source
    assert "public static void initialize() { balance = 0; }" not in source
    assert calls == 2


def _actors_atomic_module(
    path: str,
    symbol: str,
    *,
    structured: bool,
) -> ProductionModule:
    base = _atomic_module(path, symbol)
    config = dict(base.config)
    config["implementation_section"] = "behavior_contract"
    config["implementation_atomic_concerns"] = [
        {
            "sequence": 0,
            "identifier": "feature/behavior_contract/actors",
            "concern": "actors",
            "task": "Resolve exactly one actors record for the supplied feature and acceptance criterion.",
            "rules": [],
            "record_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "authority": {"type": "string"},
                },
                "required": ["name", "role", "authority"],
                "additionalProperties": False,
            },
        }
    ]
    task = dict(config["evidence_task"])
    payload = {
        "instruction": json.dumps({"concern": "actors"}, ensure_ascii=False),
        "source_requirements": {
            "R1": (
                "- actors: Player(사용자, 소유권 및 실행 권한), "
                "Merchant NPC(상인, 거래 및 정보 제공), "
                "Ship AI(우주선, 항행 및 자동 방어), "
                "Server Authority(서버, 상태 검증 및 세계 데이터 관리)"
            )
        },
    }
    if structured:
        payload["structured_records"] = [
            {"name": "Player", "role": "사용자", "authority": "소유권 및 실행 권한"},
            {"name": "Merchant NPC", "role": "상인", "authority": "거래 및 정보 제공"},
        ]
    task["implementation_obligations"] = [
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    ]
    config["evidence_task"] = task
    return ProductionModule(
        module_id=base.module_id,
        kind=base.kind,
        config=config,
        required_gates=base.required_gates,
    )


def test_behavior_actors_are_host_compiled_without_model_java(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    compile_calls = 0

    class Router:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("actors must be host-compiled, not model-generated")

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("actors must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            nonlocal compile_calls
            compile_calls += 1
            source = (project_root / path).read_text(encoding="utf-8")
            assert "private static final class Actor" in source
            assert "java.util.List<Actor> ACTORS" in source
            assert 'new Actor("Player"' in source
            assert 'new Actor("Merchant NPC"' in source
            assert "createPlayer" not in source
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_actors_atomic_module(path, symbol, structured=True),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert compile_calls == 1
    assert result["generation_verification"]["atomic_concern_count"] == 1


def test_behavior_actors_recover_from_exact_requirement_when_structured_sections_missing(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)

    class Router:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("legacy actors requirement must still be host-compiled")

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("actors must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            source = (project_root / path).read_text(encoding="utf-8")
            assert 'new Actor("Player", "Player"' in source
            assert 'new Actor("Merchant NPC", "Merchant NPC"' in source
            assert 'new Actor("Ship AI", "Ship AI"' in source
            assert 'new Actor("Server Authority", "Server Authority"' in source
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_actors_atomic_module(path, symbol, structured=False),
        minecraft_version="1.21.1",
        loader="fabric",
    )


def _entry_conditions_atomic_module(path: str, symbol: str) -> ProductionModule:
    base = _atomic_module(path, symbol)
    config = dict(base.config)
    config["implementation_section"] = "behavior_contract"
    config["implementation_atomic_concerns"] = [
        {
            "sequence": 0,
            "identifier": "feature/behavior_contract/entry_conditions",
            "concern": "entry_conditions",
            "task": "Resolve exactly one host-requested entry conditions record.",
            "rules": [],
            "record_schema": {
                "type": "object",
                "properties": {
                    "trigger": {"type": "string"},
                    "owner": {"type": "string"},
                },
                "required": ["trigger", "owner"],
                "additionalProperties": False,
            },
        }
    ]
    task = dict(config["evidence_task"])
    task["implementation_obligations"] = [
        json.dumps(
            {
                "instruction": json.dumps(
                    {"concern": "entry_conditions"},
                    ensure_ascii=False,
                ),
                "source_requirements": {
                    "R1": (
                        "- entry_conditions: 플레이어가 도크 블록과 상호작용하거나 "
                        "상인과 대화할 때, 자금 또는 재료 보유 확인 시"
                    )
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    ]
    config["evidence_task"] = task
    return ProductionModule(
        module_id=base.module_id,
        kind=base.kind,
        config=config,
        required_gates=base.required_gates,
    )


def test_behavior_contract_generic_concern_is_host_compiled(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)

    class Router:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("behavior_contract concerns must not call the coder")

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("behavior_contract concerns must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            source = (project_root / path).read_text(encoding="utf-8")
            assert "CONTRACT_ENTRY_CONDITIONS" in source
            assert "entry_conditions" in source
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_entry_conditions_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )


def _stored_state_atomic_module(path: str, symbol: str) -> ProductionModule:
    base = _atomic_module(path, symbol)
    config = dict(base.config)
    config["implementation_section"] = "persistence"
    config["implementation_atomic_concerns"] = [
        {
            "sequence": 0,
            "identifier": "feature/persistence/stored_state",
            "concern": "stored_state",
            "task": "Resolve exactly one stored state record for the supplied feature.",
            "rules": [
                "Preserve supplied state ownership and scope.",
                "Do not implement neighboring persistence concerns.",
            ],
            "record_schema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "owner": {"type": "string"},
                    "scope": {"type": "string"},
                },
                "required": ["state", "owner", "scope"],
                "additionalProperties": False,
            },
        }
    ]
    return ProductionModule(
        module_id=base.module_id,
        kind=base.kind,
        config=config,
        required_gates=base.required_gates,
    )


def test_stored_state_receives_semantic_shape_and_compact_output_budget(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    seen: list[dict[str, object]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            assert kwargs.get("output_token_ceiling") == 4096
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]
            seen.append(concern)
            assert concern["name"] == "stored_state"
            assert concern["semantic_fields"] == ["state", "owner", "scope"]
            assert concern["java_shape"] == "declarations_only_fields_or_private_nested_types"
            assert "stored state record" in concern["task"]
            return "private static final java.util.Map<String, Object> storedState = new java.util.HashMap<>();"

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
        module=_stored_state_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    source = (root / path).read_text(encoding="utf-8")
    assert "storedState" in source
    assert len(seen) == 1



def test_stored_state_mixed_shape_is_projected_before_compile(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls = 0
    compiles = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            assert role == "coder"
            assert kwargs.get("output_token_ceiling") == 4096
            calls += 1
            return (
                "private static final java.util.Map<String, String> STATE_MAPPINGS = "
                "new java.util.HashMap<>();\n"
                "private static void registerStateMapping(String owner, String state) { "
                "STATE_MAPPINGS.put(owner, state); }\n"
                "private static String getStateForOwner(String owner) { "
                "return STATE_MAPPINGS.get(owner); }"
            )

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            nonlocal compiles
            compiles += 1
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_stored_state_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    source = (root / path).read_text(encoding="utf-8")
    assert "STATE_MAPPINGS" in source
    assert "registerStateMapping" not in source
    assert "getStateForOwner" not in source
    assert calls == 1
    assert compiles == 1


def test_atomic_first_candidate_canonicalizes_jdk_lock_semantics_before_compile(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls: list[str] = []
    compiles = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append(concern)
            return (
                "private static final Map<String, java.lang.Object> CACHE = "
                "new HashMap<>();\n"
                "private static final java.util.Map<String, java.lang.Object> CACHE_LOCK = "
                "new java.util.ReentrantLock();\n"
                "private static void loadCachedState() {\n"
                "    CACHE.put(\"ready\", java.lang.Boolean.TRUE);\n"
                "    CACHE_LOCK.lock();\n"
                "    try { } finally { CACHE_LOCK.unlock(); }\n"
                "}"
            )

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            nonlocal compiles
            compiles += 1
            source = (project_root / path).read_text(encoding="utf-8")
            assert (
                "java.util.Map<String, java.lang.Object> CACHE = "
                "new java.util.HashMap<>();"
            ) in source
            assert (
                "java.util.concurrent.locks.Lock CACHE_LOCK = "
                "new java.util.concurrent.locks.ReentrantLock();"
            ) in source
            assert "new HashMap<>()" not in source
            assert "java.util.ReentrantLock" not in source
            assert "java.util.Map<String, java.lang.Object> CACHE_LOCK" not in source
            assert "CACHE_LOCK.lock();" in source
            assert "CACHE_LOCK.unlock();" in source
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_single_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert result["generation_verification"]["atomic_repair_count"] == 0
    assert result["generation_verification"]["atomic_first_pass_rejection_count"] == 0
    assert result["generation_verification"]["atomic_first_compile_failure_count"] == 0
    assert calls == ["steps"]
    assert compiles == 1


def test_atomic_compile_failure_is_terminal_without_model_retry(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "4")
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "4")
    root, path, symbol = _project(tmp_path)
    calls: list[str] = []
    compiles = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            assert kwargs.get("force_non_thinking") is True
            assert kwargs.get("tool_stage") == "atomic_java"
            payload = json.loads(messages[-1]["content"])
            concern = payload["concern"]["name"]
            calls.append(concern)
            return "private static int value = missingSymbol;"

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            nonlocal compiles
            compiles += 1
            source = (project_root / path).read_text(encoding="utf-8")
            line = next(
                index for index, text in enumerate(source.splitlines(), start=1)
                if "missingSymbol" in text
            )
            log = project_root / ".minecraft_ai/logs/atomic.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                f"{project_root / path}:{line}: error: cannot find symbol\n  symbol:   variable missingSymbol",
                encoding="utf-8",
            )
            return SimpleNamespace(
                status="FAIL",
                commands=(SimpleNamespace(log_path=str(log)),),
                error="compile failed",
            )

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_COMPILE_FAILED",
    ):
        direct.CustomModuleGenerator(Router()).generate(
            root,
            module=_single_atomic_module(path, symbol),
            minecraft_version="1.21.1",
            loader="fabric",
        )

    source = (root / path).read_text(encoding="utf-8")
    assert "missingSymbol" not in source
    assert calls == ["steps"]
    assert compiles == 1


def test_production_atomic_response_failure_ignores_diagnostic_retry_env(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "4")
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "4")
    root, path, symbol = _project(tmp_path)
    calls = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            assert role == "coder"
            assert kwargs.get("tool_stage") == "atomic_java"
            calls += 1
            return "package escaped;"

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("production concern generation must not use scalar Java tools")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _project_root):
            raise AssertionError("host-invalid atomic output must fail before compile")

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    with pytest.raises(
        direct.CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID",
    ):
        direct.CustomModuleGenerator(Router()).generate(
            root,
            module=_single_atomic_module(path, symbol),
            minecraft_version="1.21.1",
            loader="fabric",
        )

    assert calls == 1


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


def test_atomic_sibling_map_generics_are_propagated_before_first_compile(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    responses = iter(
        [
            (
                "private static final java.util.Map<String, "
                "java.util.Map<String, java.lang.Object>> FAILURE_RULES = "
                "new java.util.HashMap<>();"
            ),
            (
                "private static boolean failClosed() {\n"
                "    for (java.util.Map.Entry<String, "
                "java.util.Map<String, java.lang.Object>> entry "
                ": FAILURE_RULES.entrySet()) {\n"
                "        java.util.Map<String, java.lang.String> condition = "
                "entry.getValue();\n"
                "        if (condition.isEmpty()) { return false; }\n"
                "    }\n"
                "    return true;\n"
                "}"
            ),
        ]
    )
    calls: list[str] = []
    compiles = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert kwargs.get("enable_tools") is False
            payload = json.loads(messages[-1]["content"])
            calls.append(payload["concern"]["name"])
            return next(responses)

        def generate_tool_decision(self, *_args, **_kwargs):
            raise AssertionError("atomic production must use complete Java regions")

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, project_root):
            nonlocal compiles
            compiles += 1
            source = (project_root / path).read_text(encoding="utf-8")
            if "failClosed()" in source:
                assert (
                    "java.util.Map<java.lang.String, java.lang.Object> condition = "
                    "entry.getValue();"
                ) in source
                assert (
                    "java.util.Map<java.lang.String, java.lang.String> condition"
                    not in source
                )
                assert (
                    "java.util.Map<String, java.lang.String> condition"
                    not in source
                )
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=_atomic_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert calls == ["steps", "branches"]
    assert compiles == 1
    assert result["generation_verification"]["atomic_repair_count"] == 0
    assert result["generation_verification"]["atomic_first_pass_rejection_count"] == 0
    assert result["generation_verification"]["atomic_first_compile_failure_count"] == 0

def test_graph_owned_atomic_leaf_defers_gradle_until_graph_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    root, path, symbol = _project(tmp_path)
    calls = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            calls += 1
            assert role == "coder"
            assert kwargs.get("force_non_thinking") is True
            return "private static int balance = 0;"

    class Runner:
        def __init__(self, _cache):
            pass

        @staticmethod
        def compile_java(_root):
            raise AssertionError(
                "implementation-graph leaf must defer Gradle to the graph transaction"
            )

    base = _single_atomic_module(path, symbol)
    config = dict(base.config)
    config["implementation_graph_deferred_compile"] = True
    module = ProductionModule(
        module_id="ir_authored_algorithm",
        kind=base.kind,
        config=config,
        required_gates=base.required_gates,
    )

    class ForbiddenCoarseLock:
        def __enter__(self):
            raise AssertionError(
                "graph-owned deferred leaf must not hold the coarse project lock during model decode"
            )

        def __exit__(self, exc_type, exc, tb):
            return False

    path_locks: list[tuple[str, ...]] = []

    class RecordingPathLock:
        def __init__(self, paths):
            self.paths = tuple(str(item) for item in paths)

        def __enter__(self):
            path_locks.append(self.paths)
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)
    monkeypatch.setattr(
        direct,
        "project_write_lock",
        lambda _root: ForbiddenCoarseLock(),
    )
    monkeypatch.setattr(
        direct,
        "project_path_write_locks",
        lambda _root, paths: RecordingPathLock(paths),
    )

    result = direct.CustomModuleGenerator(Router()).generate(
        root,
        module=module,
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert calls == 1
    assert path_locks == [(path,)]
    assert "private static int balance = 0;" in (
        root / path
    ).read_text(encoding="utf-8")
    verification = result["generation_verification"]
    assert verification["status"] == "PASS"
    assert verification["compile_deferred"] is True
    assert (
        verification["mode"]
        == "host_semantic_validation_deferred_to_implementation_graph"
    )

@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_pipeline_deferred_whole_file_generation_skips_gradle_but_commits_source(
    tmp_path: Path, monkeypatch, newline
) -> None:
    root, path, symbol = _project(tmp_path)
    target = root / path
    target.write_bytes(target.read_text(encoding="utf-8").replace("\n", newline).encode("utf-8"))
    source = (
        "package example;\n\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() { int ready = 1; }\n"
        "}\n"
    )

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            return source

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            raise AssertionError(
                "complete-production generation must defer compile to the final build gate"
            )

    class ForbiddenCoarseLock:
        def __enter__(self):
            raise AssertionError(
                "pipeline-deferred whole-file generation must not hold the coarse project lock"
            )

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(direct, "adapter_for_target", lambda *_args: _adapter())
    monkeypatch.setattr(direct, "GradleRunner", Runner)
    monkeypatch.setattr(
        direct,
        "project_write_lock",
        lambda _root: ForbiddenCoarseLock(),
    )

    result = direct.CustomModuleGenerator(
        Router(),
        defer_compile_to_pipeline=True,
    ).generate(
        root,
        module=_module(path, symbol),
        minecraft_version="1.21.1",
        loader="fabric",
    )

    assert (root / path).read_text(encoding="utf-8") == source
    assert result["generation_verification"]["status"] == "PASS"
    assert result["generation_verification"]["compile_deferred"] is True
    assert (
        result["generation_verification"]["mode"]
        == "host_source_validation_deferred_to_pipeline"
    )
