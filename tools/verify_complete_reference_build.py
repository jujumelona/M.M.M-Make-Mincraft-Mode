from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.runner import GradleRunner
from minecraft_mod_ai.scalable_pipeline import ScalableMinecraftModPipeline
from minecraft_mod_ai.validator import validate_jar


class _ReferenceCoderRouter:
    """Deterministic CI coder that exercises the real custom-patch pipeline.

    The reference workflow verifies orchestration, source patching, Gradle,
    GameTest and JAR validation. It must not download or depend on a production
    7-23GB local coding model merely to prove those host-side contracts. Real model
    backends are covered by their own configuration/runtime contracts.
    """

    def generate_text(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        media_paths: Sequence[str | Path] = (),
        response_format: str = "text",
    ) -> str:
        if role != "coder":
            raise RuntimeError(
                f"Reference coder received unexpected model role: {role}"
            )
        if media_paths:
            raise RuntimeError("Reference coder does not accept media inputs.")
        if not messages:
            raise RuntimeError("Reference coder received no request messages.")

        request = json.loads(str(messages[-1]["content"]))
        module = request.get("module", {})
        module_id = str(module.get("module_id", "reference_custom"))
        digest = hashlib.sha256(module_id.encode("utf-8")).hexdigest()[:16]
        class_name = "ReferenceCustom" + digest.upper()
        path = f"src/main/java/mmm/reference/generated/{class_name}.java"
        java_module_id = json.dumps(module_id, ensure_ascii=False)
        content = (
            "package mmm.reference.generated;\n\n"
            f"public final class {class_name} {{\n"
            f"    private {class_name}() {{}}\n\n"
            "    public static String moduleId() {\n"
            f"        return {java_module_id};\n"
            "    }\n"
            "}\n"
        )
        return json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "path": path,
                        "content": content,
                    }
                ],
                "runtime_tests": [
                    f"Reference custom module {module_id} compiles in the generated Fabric project"
                ],
                "complete": True,
                "next_cursor": "",
                "context_page_complete": True,
            },
            ensure_ascii=False,
        )


def build_reference(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base = ScalableMinecraftModPipeline(
        planner=HeuristicPlanner()
    ).plan("Create one frost crystal item")

    modules = (
        ProductionModule(
            "reference_blade",
            "weapon",
            {
                "display_name_en": "Reference Blade",
                "display_name_ko": "참조 검",
                "attack_damage": 7.0,
                "attack_speed": -2.4,
            },
        ),
        ProductionModule(
            "reference_machine",
            "machine",
            {
                "display_name_en": "Reference Machine",
                "display_name_ko": "참조 기계",
                "input_item": "minecraft:iron_ingot",
                "output_item": "minecraft:gold_ingot",
                "output_count": 1,
                "processing_ticks": 20,
            },
        ),
        ProductionModule(
            "break_stone",
            "quest",
            {
                "objective": "break",
                "target": "minecraft:stone",
                "required": 1,
                "reward_item": "minecraft:diamond",
                "reward_count": 1,
                "reward_currency": 5.0,
            },
            ("reference_blade",),
        ),
        ProductionModule(
            "sentinel",
            "class",
            {"display_name": "Sentinel"},
        ),
        ProductionModule(
            "sentinel_speed",
            "skill",
            {
                "required_class": "sentinel",
                "effect": "minecraft:speed",
                "duration_ticks": 100,
                "amplifier": 1,
                "cooldown_ticks": 40,
            },
            ("sentinel",),
        ),
        ProductionModule(
            "coins",
            "economy",
            {"initial_balance": 20.0},
        ),
        ProductionModule(
            "starter_shop",
            "shop",
            {
                "entries": [
                    {
                        "id": "buy_bread",
                        "item": "minecraft:bread",
                        "count": 2,
                        "price": 3.0,
                    }
                ]
            },
            ("coins",),
        ),
        ProductionModule(
            "reference_menu",
            "gui",
            {
                "template": "read_only_menu",
                "title": "Reference Menu",
                "rows": 1,
                "entries": [
                    {
                        "slot": 0,
                        "item": "minecraft:book",
                        "count": 1,
                    }
                ],
            },
        ),
        ProductionModule(
            "reference_channel",
            "networking",
            {
                "template": "validated_action_channel",
                "actions": [
                    {
                        "id": "show_help",
                        "type": "message",
                        "message": "Reference action accepted",
                    },
                    {
                        "id": "claim_book",
                        "type": "grant_item",
                        "item": "minecraft:book",
                        "count": 1,
                    },
                ],
            },
            ("reference_menu",),
        ),
        ProductionModule(
            "reference_party",
            "party",
            {},
        ),
        ProductionModule(
            "reference_guild",
            "guild",
            {},
        ),
        ProductionModule(
            "reference_guard",
            "entity",
            {
                "display_name_en": "Reference Guard",
                "display_name_ko": "참조 수호자",
                "max_health": 40.0,
                "attack_damage": 5.0,
                "movement_speed": 0.25,
                "follow_range": 24.0,
                "archetype": "biped",
                "behavior": "hostile_melee",
                "entity_width": 0.7,
                "entity_height": 1.9,
            },
        ),
        ProductionModule(
            "reference_hall",
            "structure",
            {},
        ),
    )

    proposal = complete_proposal_from_parts(
        requested_prompt=(
            "Build a complete deterministic Fabric reference mod with an "
            "explicit village structure"
        ),
        base_proposal=base,
        game_design={
            "title": "Complete Reference",
            "pitch": "CI reference build",
            "core_loop": ["craft", "quest", "explore"],
            "progression": ["start", "complete"],
            "combat": {
                "player_verbs": ["attack"],
                "enemy_roles": ["guard"],
            },
            "mod_context": {"vanilla_integration": ["server lifecycle"]},
            "modules": [{"plugin_id": "complete-orchestrator", "status": "implemented", "reason": "CI"}],
            "assets": [],
            "acceptance_tests": ["reference build compiles"],
        },
        modules=modules,
        acceptance_tests=(
            "All generated registries load in GameTest",
            "All generated Java compiles on Fabric 1.20.1",
            "The built JAR passes independent validation",
        ),
    )

    source_result = CompleteProductionOrchestrator(
        workspace_root=output / "orchestrator",
        router_factory=_ReferenceCoderRouter,
    ).execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="reference",
        options=CompleteExecutionOptions(
            source_only=True,
            run_jdt=False,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
        ),
    )
    if source_result.status != "SOURCE_READY":
        raise RuntimeError(
            f"Reference source generation failed: {source_result.to_dict()}"
        )

    project_root = Path(source_result.project_root)
    build = GradleRunner(output / "gradle-cache").build(
        project_root,
        run_gametest=True,
    )
    if not build.passed or not build.jar_path:
        raise RuntimeError(
            "Reference Gradle/GameTest failed:\n"
            + json.dumps(build.to_dict(), ensure_ascii=False, indent=2)
        )

    jar = Path(build.jar_path)
    jar_report = validate_jar(jar, base.spec)
    if not jar_report.passed:
        raise RuntimeError(
            "Reference JAR validation failed:\n"
            + json.dumps(jar_report.to_dict(), ensure_ascii=False, indent=2)
        )

    result = {
        "status": "PASS",
        "project_root": str(project_root),
        "jar_path": str(jar),
        "source_result": source_result.to_dict(),
        "build": build.to_dict(),
        "jar_validation": jar_report.to_dict(),
    }
    (output / "reference-build-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference-build-output"),
    )
    args = parser.parse_args()
    result = build_reference(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
