from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .complete_spec import ProductionModule
from .extended_content_generator import generate_extended_content
from .gametest_discovery import discovered_gametest_root_java
from .generator import FabricProjectGenerator, GeneratedProject
from .project_edit import inspect_fabric_project, write_text_files
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file
from .spec import ContentKind, ModSpec


class ScalableFabricProjectGenerator:
    """Generate the bootstrap project and shard arbitrary content counts.

    The legacy generator remains the versioned resource compiler. Content registration
    is moved to independently discoverable GeneratedContentUnit classes so neither
    the bootstrap main class nor a registrar root grows with requested content.
    """

    def __init__(self, *, policy: ScalePolicy | None = None) -> None:
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def generate(self, spec: ModSpec, root: Path) -> GeneratedProject:
        spec.validate()
        generator = FabricProjectGenerator()
        skeleton = replace(spec, contents=())
        generator.generate(skeleton, root)

        # Reuse the pinned version-specific resource compiler without registering every
        # content field in the bootstrap initializer.
        for content in spec.contents:
            if content.kind is ContentKind.ITEM:
                generator._write_item(root, spec, content)
            elif content.kind is ContentKind.BLOCK:
                generator._write_block(root, spec, content)
            else:  # pragma: no cover - enum is closed
                raise ValueError(f"Unsupported bootstrap content kind: {content.kind}")

        modules = tuple(
            ProductionModule(
                module_id=content.content_id,
                kind=content.kind.value,
                config={
                    "display_name_en": content.display_name_en,
                    "display_name_ko": content.display_name_ko,
                    "color": content.color,
                },
                required_gates=("registry", "resource", "recipe" if content.recipe else "resource"),
            )
            for content in spec.contents
        )
        if modules:
            generate_extended_content(
                project_root=root,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                modules=modules,
                policy=self.policy,
            )
            self._write_gametest_shards(root, spec)

        files = tuple(
            sorted(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        )
        main_class = "".join(part.capitalize() for part in spec.mod_id.split("_")) + "Mod"
        return GeneratedProject(
            root=root.resolve(),
            files=files,
            main_class=f"{spec.package_name}.{main_class}",
        )

    def _write_gametest_shards(self, root: Path, spec: ModSpec) -> None:
        info = inspect_fabric_project(root)
        files: dict[str, str] = {}
        contents = list(spec.contents)
        package_path = spec.package_name.replace(".", "/")
        root_class_name = "ScalableContentGameTest"
        unit_prefix = root_class_name + "Unit"
        root_entrypoint = f"{spec.package_name}.{root_class_name}"
        files[
            f"src/main/java/{package_path}/{root_class_name}.java"
        ] = discovered_gametest_root_java(
            package_name=spec.package_name,
            mod_id=spec.mod_id,
            root_class_name=root_class_name,
            unit_class_prefix=unit_prefix,
        )
        for offset in range(0, len(contents), self.policy.java_shard_size):
            shard = contents[offset : offset + self.policy.java_shard_size]
            index = offset // self.policy.java_shard_size
            class_name = f"{unit_prefix}{index:04d}"
            relative = (
                "src/main/java/"
                + package_path
                + f"/{class_name}.java"
            )
            files[relative] = _gametest_java(
                spec,
                class_name,
                shard,
            )
        write_text_files(info, files, replace_existing=True)

        metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
        entrypoints = metadata.setdefault("entrypoints", {})
        if not isinstance(entrypoints, dict):
            raise ValueError("fabric.mod.json entrypoints must be an object")
        gametest = entrypoints.setdefault("fabric-gametest", [])
        if not isinstance(gametest, list):
            raise ValueError("fabric-gametest entrypoints must be a list")
        generated_prefix = f"{spec.package_name}.ScalableContentGameTest"

        def entrypoint_value(item) -> str | None:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                value = item.get("value")
                return value if isinstance(value, str) else None
            return None

        retained = [
            item
            for item in gametest
            if not (
                (value := entrypoint_value(item))
                and value.startswith(generated_prefix)
            )
        ]
        replacement = retained + [root_entrypoint]
        if replacement != gametest:
            entrypoints["fabric-gametest"] = replacement
            TransactionalSourcePatcher(root).apply(
                [
                    {
                        "operation": "replace",
                        "path": "src/main/resources/fabric.mod.json",
                        "expected_sha256": sha256_file(info.fabric_mod_json),
                        "content": json.dumps(
                            metadata,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    }
                ]
            )


def _gametest_java(spec: ModSpec, class_name: str, contents: list) -> str:
    checks: list[str] = []
    for content in contents:
        registry = "ITEM" if content.kind is ContentKind.ITEM else "BLOCK"
        checks.append(
            f'''        require(Registries.{registry}.containsId(new Identifier("{spec.mod_id}", "{content.content_id}")), "{content.content_id} registry entry missing");'''
        )
        if content.recipe:
            checks.append(
                f'''        require(context.getWorld().getServer().getRecipeManager().get(new Identifier("{spec.mod_id}", "{content.content_id}")).isPresent(), "{content.content_id} recipe missing");'''
            )
    return f'''package {spec.package_name};

import net.minecraft.registry.Registries;
import net.minecraft.test.TestContext;
import net.minecraft.util.Identifier;

public final class {class_name} {{
    public static void run(TestContext context) {{
{chr(10).join(checks)}
    }}

    private static void require(boolean condition, String message) {{
        if (!condition) throw new AssertionError(message);
    }}
}}
'''
