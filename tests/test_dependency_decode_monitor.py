from __future__ import annotations

import json

from minecraft_mod_ai import custom_generation_search_contract as generation_search
from minecraft_mod_ai import llama_server_hardware_policy as llama_hardware
from minecraft_mod_ai import research_code_context as research
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.dependency_decode_monitor import (
    activate_dependency_decode_monitor,
)

_TEST_MINECRAFT_VERSION = "1.21.11+mmm-test"
_TEST_LOADER = "fabric"
_TEST_MAPPINGS = "1.21.11+mmm-test+test-mappings"
_TEST_FABRIC_LOADER = "test-loader"
_TEST_FABRIC_API = "test-api"
_TEST_FABRIC_LOOM = "test-loom"


def _project(tmp_path):
    source = tmp_path / "src/main/java/example/Entry.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example;\n"
        "import java.util.List;\n"
        "public final class Entry { List<String> values; }\n",
        encoding="utf-8",
    )
    (tmp_path / "gradle.properties").write_text(
        "\n".join(
            (
                f"minecraft_version={_TEST_MINECRAFT_VERSION}",
                f"yarn_mappings={_TEST_MAPPINGS}",
                f"loader_version={_TEST_FABRIC_LOADER}",
                f"fabric_version={_TEST_FABRIC_API}",
                f"loom_version={_TEST_FABRIC_LOOM}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _monitor(tmp_path):
    activate_dependency_decode_monitor()
    _project(tmp_path)
    return research.DependencyMonitor(
        tmp_path,
        minecraft_version=_TEST_MINECRAFT_VERSION,
        loader=_TEST_LOADER,
        mappings=_TEST_MAPPINGS,
    )


def test_packmonitor_rejects_unknown_java_imports_and_accepts_authoritative_ones(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    good = json.dumps(
        {
            "operations": [
                {
                    "operation": "create",
                    "path": "src/main/java/example/Good.java",
                    "content": (
                        "package example;\n"
                        "import java.util.Map;\n"
                        "import net.fabricmc.api.ModInitializer;\n"
                        "public final class Good {}\n"
                    ),
                }
            ]
        }
    )
    assert monitor.validate_model_output(good) == ()

    bad = json.dumps(
        {
            "operations": [
                {
                    "operation": "create",
                    "path": "src/main/java/example/Bad.java",
                    "content": (
                        "package example;\n"
                        "import evil.fake.NonexistentClient;\n"
                        "public final class Bad {}\n"
                    ),
                }
            ]
        }
    )
    violations = monitor.validate_model_output(bad)
    assert any(
        item.kind == "java_import" and item.value == "evil.fake.NonexistentClient"
        for item in violations
    )


def test_decode_monitor_defers_partial_coordinate_then_blocks_completed_unknown(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    partial = '"evil.fake:nonexistent:1.0'
    assert monitor.stream_violations(partial, final=False) == ()
    completed = partial + '"'
    violations = monitor.stream_violations(completed, final=False)
    assert any(
        item.kind == "package" and item.value == "evil.fake:nonexistent"
        for item in violations
    )

    admitted_partial = (
        '"net.fabricmc.fabric-api:fabric-api:'
        + _TEST_FABRIC_API[:-2]
    )
    assert monitor.stream_violations(admitted_partial, final=False) == ()
    admitted_complete = (
        '"net.fabricmc.fabric-api:fabric-api:' + _TEST_FABRIC_API + '"'
    )
    assert monitor.stream_violations(admitted_complete, final=False) == ()


def test_decode_monitor_is_installed_on_actual_research_and_llama_hot_paths() -> None:
    activate_dependency_decode_monitor()
    assert getattr(
        research.DependencyMonitor,
        "_mmm_decode_time_packmonitor",
        False,
    )
    assert getattr(
        generation_search._ResearchEvidenceRouter.generate_text,
        "_mmm_dependency_decode_scope",
        False,
    )
    assert getattr(
        llama_hardware._strict_server_generate,
        "_mmm_dependency_decode_monitor",
        False,
    )
    assert getattr(
        llama_hardware._stream_delta_parts,
        "_mmm_dependency_decode_monitor",
        False,
    )


def test_custom_generator_target_defaults_are_disabled_after_runtime_install() -> None:
    defaults = CustomModuleGenerator.generate.__kwdefaults__ or {}
    assert defaults.get("minecraft_version") is None
    assert defaults.get("loader") is None
    assert defaults.get("mappings") is None
