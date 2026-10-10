"""Server GameTest runtime route: positive and adversarial evidence tests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.managed_content_gametest import (
    eligible_for_managed_runtime,
    install_managed_content_gametest,
    independently_verified_managed_runtime,
)
from minecraft_mod_ai.quality_evidence import _managed_content_runtime_evidence


def _approved(*, version="26.2", modules=None):
    facts = {
        "api_symbols": {
            "builtin_item_registry": {
                "owner": "net.minecraft.core.registries.BuiltInRegistries",
                "name": "ITEM",
            },
            "builtin_block_registry": {
                "owner": "net.minecraft.registry.BuiltInRegistries",
                "name": "BLOCK",
            },
        },
    }
    default = (("crystal_fragment", "item"), ("crystal_block", "block"),
               ("crystal_block_recipe", "recipe"), ("authored_typed_plan", "typed_host"))
    return SimpleNamespace(
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(
                mod_id="mmm_debug_crystal",
                package_name="ai.minecraft.generated.mmm_debug_crystal",
                platform=SimpleNamespace(
                    loader="fabric", minecraft_version=version,
                    mappings_kind="", host_facts_json=json.dumps(facts),
                ),
            ),
        ),
        modules=tuple(
            SimpleNamespace(
                module_id=name, kind=kind,
                config=(
                    {"typed_plan_ir": {
                        "event_bindings": [], "functions": [],
                        "initialize": [], "platform_modules": [],
                    }} if kind == "typed_host" else {}
                ),
            )
            for name, kind in (modules if modules is not None else default)
        ),
    )


def _project(tmp_path: Path) -> tuple[Path, Path]:
    package = "ai/minecraft/generated/mmm_debug_crystal"
    path = (tmp_path / "src/gametest/java" / package / "MmmDebugCrystalModGameTests.java")
    path.parent.mkdir(parents=True)
    path.write_text(
        "public final class MmmDebugCrystalModGameTests {\n"
        "    public void generatedRegistriesAreLive(Object context) {\n"
        "        context.succeed();\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "src/main/java" / package / "registry"
    source_dir.mkdir(parents=True, exist_ok=True)
    # These are the compiler-visible registration owners; a host-facts catalog
    # can contain a different alias and must not silently override them.
    (source_dir / "ModItems.java").write_text(
        "import net.minecraft.core.registries.BuiltInRegistries;\\n"
        "class ModItems { void run() { var registry = BuiltInRegistries.ITEM; } }\\n",
        encoding="utf-8",
    )
    (source_dir / "ModBlocks.java").write_text(
        "import net.minecraft.core.registries.BuiltInRegistries;\\n"
        "class ModBlocks { void run() { var registry = BuiltInRegistries.BLOCK; } }\\n",
        encoding="utf-8",
    )
    resources = tmp_path / "src/main/resources"
    for directory, name, payload in (
        ("assets/mmm_debug_crystal/items", "crystal_fragment.json", {"model": {"type": "minecraft:model"}}),
        ("assets/mmm_debug_crystal/blockstates", "crystal_block.json", {"variants": {}}),
        ("data/mmm_debug_crystal/recipe", "crystal_block_recipe.json", {
            "type": "minecraft:crafting_shaped",
            "pattern": ["FF", "FF"],
            "key": {"F": "mmm_debug_crystal:crystal_fragment"},
            "result": {"id": "mmm_debug_crystal:crystal_block", "count": 1}}),
    ):
        dest = resources / directory / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path, path


def test_generated_content_gametest_uses_target_host_facts_and_real_recipe_resource(tmp_path):
    root, path = _project(tmp_path)
    approved = _approved()
    assert eligible_for_managed_runtime(approved)
    installed = install_managed_content_gametest(root, approved)
    assert installed and installed["status"] == "INSTALLED"
    assert installed["recipes"] == ["crystal_block_recipe"]
    source = path.read_text(encoding="utf-8")
    assert 'Class.forName("net.minecraft.core.registries.BuiltInRegistries")' in source
    assert 'Class.forName("net.minecraft.core.registries.BuiltInRegistries")' in source
    assert "GameTest missing live item" in source
    assert "GameTest missing live block" in source
    assert 'getMethod("setBlock", blockPosClass, blockClass)' in source
    assert 'getMethod("assertBlockPresent", blockClass, blockPosClass)' in source
    assert "GameTest RecipeManager did not load actual recipe: mmm_debug_crystal:crystal_block_recipe" in source
    assert "createInput.invoke(null, 2, 2, craftedInputs)" in source
    assert "getMethod(\"matches\", inputClass" in source
    assert "getMethod(\"assemble\", inputClass)" in source
    assert "GameTest crafted item or count incorrect" in source
    assert 'getMethod("byKey", resourceKeyClass)' in source
    assert source.count("MMM_MANAGED_CONTENT_GAMETEST_V1 START") == 1
    # Re-installing must not silently append old assertions.
    assert install_managed_content_gametest(root, approved)
    assert path.read_text(encoding="utf-8").count("MMM_MANAGED_CONTENT_GAMETEST_V1 START") == 1


def test_runtime_receipt_requires_matching_executed_native_testcase(tmp_path):
    root, path = _project(tmp_path)
    approved = _approved()
    assert install_managed_content_gametest(root, approved)
    report = root / "build/gametest-report.xml"
    report.parent.mkdir(parents=True)
    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="MmmDebugCrystalModGameTests" name="generatedRegistriesAreLive"/>'
        '</testsuite>', encoding="utf-8",
    )
    build = {"gametest_report": str(report)}
    assert independently_verified_managed_runtime(root, approved, build, gametest_passed=False) is None
    receipt = independently_verified_managed_runtime(root, approved, build, gametest_passed=True)
    assert receipt and receipt["runtime_kind"] == "live_minecraft_server_gametest"
    assert receipt["recipes"] == ["crystal_block_recipe"]
    good = (["fabric-test"], [{"status": "PASS"}])
    assert _managed_content_runtime_evidence(receipt, good, good, good, good, build)
    report.write_text('<testsuite><testcase name="fake"/></testsuite>', encoding="utf-8")
    assert independently_verified_managed_runtime(root, approved, build, gametest_passed=True) is None
    assert _managed_content_runtime_evidence(receipt, good, good, good, good, build) is None


def test_runtime_receipt_rejects_mutated_test_source_and_recipe(tmp_path):
    root, path = _project(tmp_path)
    approved = _approved()
    assert install_managed_content_gametest(root, approved)
    report = root / "build/gametest-report.xml"
    report.parent.mkdir(parents=True)
    report.write_text(
        '<testsuite><testcase name="MmmDebugCrystalModGameTests.generatedRegistriesAreLive"/></testsuite>',
        encoding="utf-8",
    )
    build = {"gametest_report": str(report)}
    path.write_text(path.read_text().replace('GameTest missing live item', 'fake success'), encoding="utf-8")
    assert independently_verified_managed_runtime(root, approved, build, gametest_passed=True) is None
    assert not install_managed_content_gametest(root, _approved(modules=(("extra_entity", "entity"),)))


def test_dynamic_features_or_wrong_platform_never_inherit_game_test_shortcut(tmp_path):
    root, _ = _project(tmp_path)
    assert not eligible_for_managed_runtime(_approved(version="1.21.1"))
    assert not eligible_for_managed_runtime(_approved(modules=(("npc", "entity"),)))
    assert install_managed_content_gametest(root, _approved(version="1.21.1")) is None
    custom_item = _approved(modules=(("crystal_fragment", "item"),))
    custom_item.modules[0].config["on_use_script"] = "grant_flight"
    assert not eligible_for_managed_runtime(custom_item)
    custom_typed = _approved()
    custom_typed.modules[-1].config["typed_plan_ir"]["event_bindings"] = [
        {"event": "on_tick"}
    ]
    assert not eligible_for_managed_runtime(custom_typed)


def test_emitted_server_gametest_java_compiles_with_real_javac(tmp_path):
    root, path = _project(tmp_path)
    assert install_managed_content_gametest(root, _approved())
    source = path.read_text(encoding="utf-8")
    body = source[source.index("// MMM_MANAGED_CONTENT_GAMETEST_V1 START"):
                  source.index("// MMM_MANAGED_CONTENT_GAMETEST_V1 END")
                  + len("// MMM_MANAGED_CONTENT_GAMETEST_V1 END")]
    target = tmp_path / "ManagedGameTestSyntax.java"
    target.write_text(
        "class GameTestHelper { void succeed() {} }\n"
        "public class ManagedGameTestSyntax {\n"
        "  public void test(GameTestHelper context) {\n"
        "    " + body + "\n"
        "    context.succeed();\n"
        "  }\n"
        "}\n", encoding="utf-8",
    )
    result = subprocess.run(
        ["javac", "--release", "21", "-encoding", "UTF-8", str(target)],
        cwd=tmp_path, check=False, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_colab_content_only_runtime_preflight_uses_real_gametest_route() -> None:
    from minecraft_mod_ai.complete_orchestrator import (
        CompleteExecutionOptions,
        _quality_execution_preflight_gaps,
    )
    approved = _approved()
    approved.game_design = {
        "_production_contract": {
            "quality_dimension_catalog": [{"dimension_id": "runtime"}],
        },
    }
    options = CompleteExecutionOptions(
        run_gametest=True, run_runtime=False, run_client=False,
        run_mineflayer=False, server_launcher=None, playtest_actions=(),
    )
    assert _quality_execution_preflight_gaps(approved, options) == {}
    without_real_server_test = CompleteExecutionOptions(
        run_gametest=False, run_runtime=False, run_mineflayer=False,
        server_launcher=None, playtest_actions=(),
    )
    assert "runtime" in _quality_execution_preflight_gaps(approved, without_real_server_test)
    different_target = _approved(version="1.21.1")
    different_target.game_design = approved.game_design
    assert "runtime" in _quality_execution_preflight_gaps(different_target, options)
