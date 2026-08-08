from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .spec import BossSpec, ContentKind, ContentSpec, ModSpec
from .toolchain_contract import fabric_dependency_predicates


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedProject:
    root: Path
    files: tuple[Path, ...]
    main_class: str


def _java_class_name(mod_id: str) -> str:
    return "".join(part.capitalize() for part in mod_id.split("_")) + "Mod"


def _constant_name(content_id: str) -> str:
    return content_id.upper()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


class FabricProjectGenerator:
    """Compile a validated ModSpec into a pinned Fabric 1.20.1 project."""

    def generate(self, spec: ModSpec, root: Path) -> GeneratedProject:
        spec.validate()
        root = root.resolve()
        if root.exists() and any(root.iterdir()):
            raise GenerationError(f"Refusing to generate into non-empty directory: {root}")
        root.mkdir(parents=True, exist_ok=True)

        self._written: list[Path] = []
        package_path = Path(*spec.package_name.split("."))
        main_class = _java_class_name(spec.mod_id)

        self._write_text(root, "settings.gradle", self._settings_gradle(spec))
        self._write_text(root, "build.gradle", self._build_gradle(spec))
        self._write_text(root, "gradle.properties", self._gradle_properties(spec))
        self._write_text(root, ".gitignore", _GITIGNORE)
        self._write_text(root, "LICENSE", _MIT_LICENSE)
        self._write_text(root, "README.md", self._project_readme(spec))
        self._write_text(
            root,
            package_path / f"{main_class}.java",
            self._main_java(spec, main_class),
            source=True,
        )
        self._write_text(
            root,
            Path("src/test/java") / package_path / "GeneratedContractTest.java",
            self._contract_test_java(spec),
            raw=True,
        )
        self._write_text(
            root,
            package_path / f"{main_class}GameTests.java",
            self._gametest_java(spec, main_class),
            source=True,
        )

        resource_root = Path("src/main/resources")
        self._write_text(
            root,
            resource_root / "fabric.mod.json",
            _json_text(self._fabric_mod_json(spec, main_class)),
            raw=True,
        )
        self._write_text(
            root,
            resource_root / "pack.mcmeta",
            _json_text(
                {
                    "pack": {
                        "pack_format": 15,
                        "description": f"{spec.mod_name} resources",
                    }
                }
            ),
            raw=True,
        )

        self._write_lang_files(root, spec)
        for content in spec.contents:
            if content.kind is ContentKind.ITEM:
                self._write_item(root, spec, content)
            elif content.kind is ContentKind.BLOCK:
                self._write_block(root, spec, content)
            else:
                raise GenerationError(f"Unsupported content kind: {content.kind}")

        if spec.boss is not None:
            self._write_boss(root, spec, main_class, spec.boss)
        self._write_tags(root, spec)
        self._write_contract(root, spec)
        return GeneratedProject(
            root=root,
            files=tuple(sorted(self._written)),
            main_class=f"{spec.package_name}.{main_class}",
        )

    def _resolve(self, root: Path, relative: Path | str, *, source: bool = False, raw: bool = False) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise GenerationError(f"Unsafe generated path: {relative}")
        if source:
            relative = Path("src/main/java") / relative
        elif not raw and relative.parts and relative.parts[0] not in {
            "src",
            "gradle",
            ".minecraft_ai",
        }:
            relative = Path(relative)
        destination = (root / relative).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise GenerationError(f"Generated path escaped the project root: {relative}") from exc
        return destination

    def _write_text(
        self,
        root: Path,
        relative: Path | str,
        content: str,
        *,
        source: bool = False,
        raw: bool = False,
    ) -> None:
        destination = self._resolve(root, relative, source=source, raw=raw)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        self._written.append(destination)

    def _write_bytes(self, root: Path, relative: Path | str, content: bytes) -> None:
        destination = self._resolve(root, relative, raw=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self._written.append(destination)

    def _settings_gradle(self, spec: ModSpec) -> str:
        return f"""pluginManagement {{
    repositories {{
        maven {{ url = 'https://maven.fabricmc.net/' }}
        mavenCentral()
        gradlePluginPortal()
    }}
}}

rootProject.name = '{spec.mod_id}'
"""

    def _build_gradle(self, spec: ModSpec) -> str:
        required_resources = [
            f"src/main/resources/assets/{spec.mod_id}/lang/en_us.json",
            f"src/main/resources/assets/{spec.mod_id}/lang/ko_kr.json",
            "src/main/resources/fabric.mod.json",
        ]
        for content in spec.contents:
            if content.kind is ContentKind.ITEM:
                required_resources.extend(
                    [
                        f"src/main/resources/assets/{spec.mod_id}/models/item/{content.content_id}.json",
                        f"src/main/resources/assets/{spec.mod_id}/textures/item/{content.content_id}.png",
                    ]
                )
            else:
                required_resources.extend(
                    [
                        f"src/main/resources/assets/{spec.mod_id}/blockstates/{content.content_id}.json",
                        f"src/main/resources/assets/{spec.mod_id}/models/block/{content.content_id}.json",
                        f"src/main/resources/assets/{spec.mod_id}/models/item/{content.content_id}.json",
                        f"src/main/resources/assets/{spec.mod_id}/textures/block/{content.content_id}.png",
                        f"src/main/resources/data/{spec.mod_id}/loot_tables/blocks/{content.content_id}.json",
                    ]
                )
            if content.recipe:
                required_resources.append(
                    f"src/main/resources/data/{spec.mod_id}/recipes/{content.content_id}.json"
                )
        if spec.boss is not None:
            required_resources.extend(
                [
                    f"src/main/resources/assets/{spec.mod_id}/textures/entity/{spec.boss.entity_id}.png",
                    f"src/main/resources/assets/{spec.mod_id}/models/item/{spec.boss.entity_id}_spawn_egg.json",
                    f"src/main/resources/data/{spec.mod_id}/loot_tables/entities/{spec.boss.entity_id}.json",
                    f".minecraft_ai/art_sources/{spec.boss.entity_id}.bbmodel",
                    f".minecraft_ai/art_sources/{spec.boss.entity_id}.obj",
                ]
            )
        required_groovy = ",\n        ".join(f"'{path}'" for path in required_resources)
        return f"""plugins {{
    id 'fabric-loom' version "${{loom_version}}"
    id 'maven-publish'
}}

version = project.mod_version
group = project.maven_group

base {{
    archivesName = project.archives_base_name
}}

repositories {{
}}

loom {{
    runs {{
        gameTestServer {{
            server()
            name = 'Game Test Server'
            runDir = 'build/run-gametest'
            vmArg '-Dfabric-api.gametest'
            vmArg "-Dfabric-api.gametest.report-file=${{file('build/gametest-report.xml').absolutePath}}"
        }}
    }}
}}

dependencies {{
    minecraft "com.mojang:minecraft:${{project.minecraft_version}}"
    mappings "net.fabricmc:yarn:${{project.yarn_mappings}}:v2"
    modImplementation "net.fabricmc:fabric-loader:${{project.loader_version}}"
    modImplementation "net.fabricmc.fabric-api:fabric-api:${{project.fabric_version}}"

    testImplementation platform('org.junit:junit-bom:5.10.2')
    testImplementation 'org.junit.jupiter:junit-jupiter'
}}

processResources {{
    inputs.property 'version', project.version
    filesMatching('fabric.mod.json') {{
        expand 'version': project.version
    }}
}}

tasks.withType(JavaCompile).configureEach {{
    options.release = 17
    options.encoding = 'UTF-8'
}}

test {{
    useJUnitPlatform()
}}

java {{
    withSourcesJar()
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}}

tasks.register('verifyGeneratedResources') {{
    group = 'verification'
    description = 'Validates that the versioned schema compiler emitted every required resource.'
    doLast {{
        def required = [
        {required_groovy}
        ]
        def missing = required.findAll {{ !file(it).isFile() }}
        if (!missing.isEmpty()) {{
            throw new GradleException("Missing generated resources: " + missing.join(', '))
        }}
        fileTree('src/main/resources').matching {{
            include '**/*.json'
            include '**/*.mcmeta'
        }}.files.each {{ candidate ->
            new groovy.json.JsonSlurper().parse(candidate)
        }}
    }}
}}

check.dependsOn verifyGeneratedResources

jar {{
    from('LICENSE') {{
        rename {{ "${{it}}_{spec.mod_id}" }}
    }}
}}
"""

    def _gradle_properties(self, spec: ModSpec) -> str:
        platform = spec.platform
        return f"""org.gradle.jvmargs=-Xmx2G -Dfile.encoding=UTF-8
org.gradle.parallel=false
org.gradle.configuration-cache=false

minecraft_version={platform.minecraft_version}
yarn_mappings={platform.yarn_mappings}
loader_version={platform.fabric_loader}
loom_version={platform.fabric_loom}
fabric_version={platform.fabric_api}

mod_version={spec.version}
maven_group={spec.package_name}
archives_base_name={spec.mod_id}
"""

    def _main_java(self, spec: ModSpec, main_class: str) -> str:
        declarations: list[str] = []
        registrations: list[str] = []
        creative_entries: list[str] = []
        for content in spec.contents:
            constant = _constant_name(content.content_id)
            if content.kind is ContentKind.ITEM:
                declarations.append(f"    public static Item {constant};")
                registrations.append(
                    f'        {constant} = registerItem("{content.content_id}", '
                    "new Item(new FabricItemSettings()));"
                )
            else:
                declarations.append(f"    public static Block {constant};")
                registrations.append(
                    f'        {constant} = registerBlock("{content.content_id}", '
                    "new Block(FabricBlockSettings.copyOf(Blocks.STONE).strength(3.0f, 6.0f)));"
                )
            creative_entries.append(f"            entries.add({constant});")

        boss_imports = ""
        boss_declarations = ""
        boss_registrations = ""
        if spec.boss is not None:
            boss = spec.boss
            boss_class = _java_class_name(boss.entity_id) + "Entity"
            primary = int(boss.primary_color[1:], 16)
            secondary = int(boss.secondary_color[1:], 16)
            boss_imports = f"""import {spec.package_name}.entity.{boss_class};
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricDefaultAttributeRegistry;
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityTypeBuilder;
import net.minecraft.entity.EntityDimensions;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnGroup;
import net.minecraft.item.SpawnEggItem;
"""
            boss_declarations = f"""
    public static EntityType<{boss_class}> {boss.entity_id.upper()};
    public static Item {boss.entity_id.upper()}_SPAWN_EGG;
"""
            boss_registrations = f"""
        {boss.entity_id.upper()} = Registry.register(
            Registries.ENTITY_TYPE,
            new Identifier(MOD_ID, "{boss.entity_id}"),
            FabricEntityTypeBuilder.create(SpawnGroup.MONSTER, {boss_class}::new)
                .dimensions(EntityDimensions.fixed(0.9f, 2.1f))
                .trackRangeBlocks(10)
                .build()
        );
        FabricDefaultAttributeRegistry.register(
            {boss.entity_id.upper()},
            {boss_class}.createBossAttributes()
        );
        {boss.entity_id.upper()}_SPAWN_EGG = registerItem(
            "{boss.entity_id}_spawn_egg",
            new SpawnEggItem(
                {boss.entity_id.upper()},
                0x{primary:06X},
                0x{secondary:06X},
                new FabricItemSettings()
            )
        );
"""
            creative_entries.append(f"            entries.add({boss.entity_id.upper()}_SPAWN_EGG);")

        declarations_text = "\n".join(declarations)
        registrations_text = "\n".join(registrations)
        entries_text = "\n".join(creative_entries)
        return f"""package {spec.package_name};

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.item.v1.FabricItemSettings;
import net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents;
import net.fabricmc.fabric.api.object.builder.v1.block.FabricBlockSettings;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.item.BlockItem;
import net.minecraft.item.Item;
import net.minecraft.item.ItemGroups;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
{boss_imports}

public final class {main_class} implements ModInitializer {{
    public static final String MOD_ID = "{spec.mod_id}";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

{declarations_text}
{boss_declarations}

    @Override
    public void onInitialize() {{
{registrations_text}
{boss_registrations}

        ItemGroupEvents.modifyEntriesEvent(ItemGroups.INGREDIENTS).register(entries -> {{
{entries_text}
        }});
        LOGGER.info("Initialized {{}} with {{}} generated content entries", MOD_ID, GeneratedContent.CONTENT_IDS.length);
    }}

    private static Item registerItem(String name, Item item) {{
        return Registry.register(Registries.ITEM, new Identifier(MOD_ID, name), item);
    }}

    private static Block registerBlock(String name, Block block) {{
        Identifier id = new Identifier(MOD_ID, name);
        Registry.register(Registries.BLOCK, id, block);
        Registry.register(Registries.ITEM, id, new BlockItem(block, new FabricItemSettings()));
        return block;
    }}
}}
"""

    def _contract_test_java(self, spec: ModSpec) -> str:
        contract_ids = [content.content_id for content in spec.contents]
        if spec.boss is not None:
            contract_ids.append(spec.boss.entity_id)
        ids = ", ".join(f'"{content_id}"' for content_id in contract_ids)
        generated_content = f"""package {spec.package_name};

import java.util.Set;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class GeneratedContractTest {{
    private static final Pattern VALID_ID = Pattern.compile("^[a-z][a-z0-9_]{{1,63}}$");

    @Test
    void generatedIdsAreUniqueAndNamespacedSafely() {{
        String[] ids = new String[] {{{ids}}};
        assertEquals(ids.length, Set.of(ids).size(), "content ids must be unique");
        for (String id : ids) {{
            assertTrue(VALID_ID.matcher(id).matches(), "invalid generated id: " + id);
        }}
    }}
}}
"""
        return generated_content

    def _gametest_java(self, spec: ModSpec, main_class: str) -> str:
        registry_checks: list[str] = []
        recipe_checks: list[str] = []
        for content in spec.contents:
            registry = "ITEM" if content.kind is ContentKind.ITEM else "BLOCK"
            registry_checks.append(
                f'        require(Registries.{registry}.containsId(new Identifier('
                f'{main_class}.MOD_ID, "{content.content_id}")), '
                f'"{content.content_id} registry entry missing");'
            )
            if content.recipe:
                recipe_checks.append(
                    f'        require(context.getWorld().getServer().getRecipeManager().get('
                    f'new Identifier({main_class}.MOD_ID, "{content.content_id}")).isPresent(), '
                    f'"{content.content_id} recipe was not loaded");'
                )
        if spec.boss is not None:
            registry_checks.append(
                f'        require(Registries.ENTITY_TYPE.containsId(new Identifier('
                f'{main_class}.MOD_ID, "{spec.boss.entity_id}")), '
                f'"{spec.boss.entity_id} entity entry missing");'
            )
            registry_checks.append(
                f'        require(Registries.ITEM.containsId(new Identifier('
                f'{main_class}.MOD_ID, "{spec.boss.entity_id}_spawn_egg")), '
                f'"{spec.boss.entity_id} spawn egg entry missing");'
            )
            registry_checks.extend(
                [
                    f"        var boss = {main_class}.{spec.boss.entity_id.upper()}.create("
                    "context.getWorld());",
                    '        require(boss != null, "boss factory returned null");',
                    f"        require(Math.abs(boss.getMaxHealth() - "
                    f'{spec.boss.max_health:.1f}f) < 0.01f, '
                    '"boss max-health attribute mismatch");',
                    "        var bossPosition = context.getAbsolutePos("
                    "new net.minecraft.util.math.BlockPos(0, 2, 0));",
                    "        boss.refreshPositionAndAngles(bossPosition, 0.0f, 0.0f);",
                    '        require(context.getWorld().spawnEntity(boss), '
                    '"boss could not be spawned in the server world");',
                    f"        require(boss.isAlive() && boss.getType() == "
                    f"{main_class}.{spec.boss.entity_id.upper()}, "
                    '"spawned boss has the wrong runtime type");',
                    "        var probeBossUuid = boss.getUuid();",
                    "        boss.discard();",
                ]
            )
        checks = "\n".join((*registry_checks, *recipe_checks))
        return f"""package {spec.package_name};

import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.registry.Registries;
import net.minecraft.test.GameTest;
import net.minecraft.test.TestContext;
import net.minecraft.util.Identifier;

public final class {main_class}GameTests {{
    @GameTest(templateName = FabricGameTest.EMPTY_STRUCTURE)
    public void generatedRegistriesAreLive(TestContext context) {{
{checks}
        context.complete();
    }}

    private static void require(boolean condition, String message) {{
        if (!condition) {{
            throw new AssertionError(message);
        }}
    }}
}}
"""

    def _fabric_mod_json(self, spec: ModSpec, main_class: str) -> dict[str, object]:
        entrypoints: dict[str, list[str]] = {
            "main": [f"{spec.package_name}.{main_class}"],
            "fabric-gametest": [f"{spec.package_name}.{main_class}GameTests"],
        }
        if spec.boss is not None:
            entrypoints["client"] = [f"{spec.package_name}.client.{main_class}Client"]
        return {
            "schemaVersion": 1,
            "id": spec.mod_id,
            "version": "${version}",
            "name": spec.mod_name,
            "description": spec.summary,
            "authors": [
                "Generated with M.M.M Make Mincraft Mode; review before redistribution"
            ],
            "license": "MIT",
            "environment": "*",
            "entrypoints": entrypoints,
            "depends": fabric_dependency_predicates(spec.platform),
        }

    def _write_lang_files(self, root: Path, spec: ModSpec) -> None:
        english: dict[str, str] = {}
        korean: dict[str, str] = {}
        for content in spec.contents:
            prefix = "item" if content.kind is ContentKind.ITEM else "block"
            key = f"{prefix}.{spec.mod_id}.{content.content_id}"
            english[key] = content.display_name_en
            korean[key] = content.display_name_ko
        if spec.boss is not None:
            boss = spec.boss
            english[f"entity.{spec.mod_id}.{boss.entity_id}"] = boss.display_name_en
            korean[f"entity.{spec.mod_id}.{boss.entity_id}"] = boss.display_name_ko
            english[f"item.{spec.mod_id}.{boss.entity_id}_spawn_egg"] = (
                f"{boss.display_name_en} Spawn Egg"
            )
            korean[f"item.{spec.mod_id}.{boss.entity_id}_spawn_egg"] = (
                f"{boss.display_name_ko} 생성 알"
            )
        base = Path("src/main/resources/assets") / spec.mod_id / "lang"
        self._write_text(root, base / "en_us.json", _json_text(english), raw=True)
        self._write_text(root, base / "ko_kr.json", _json_text(korean), raw=True)

    def _write_item(self, root: Path, spec: ModSpec, content: ContentSpec) -> None:
        assets = Path("src/main/resources/assets") / spec.mod_id
        data = Path("src/main/resources/data") / spec.mod_id
        self._write_text(
            root,
            assets / "models/item" / f"{content.content_id}.json",
            _json_text(
                {
                    "parent": "minecraft:item/generated",
                    "textures": {
                        "layer0": f"{spec.mod_id}:item/{content.content_id}"
                    },
                }
            ),
            raw=True,
        )
        self._write_bytes(
            root,
            assets / "textures/item" / f"{content.content_id}.png",
            make_texture_png(content.color, content.content_id, kind="item"),
        )
        if content.recipe:
            self._write_text(
                root,
                data / "recipes" / f"{content.content_id}.json",
                _json_text(
                    {
                        "type": "minecraft:crafting_shaped",
                        "category": "misc",
                        "pattern": [" A ", "ASA", " A "],
                        "key": {
                            "A": {"item": "minecraft:amethyst_shard"},
                            "S": {"item": "minecraft:stick"},
                        },
                        "result": {"item": f"{spec.mod_id}:{content.content_id}"},
                    }
                ),
                raw=True,
            )

    def _write_block(self, root: Path, spec: ModSpec, content: ContentSpec) -> None:
        assets = Path("src/main/resources/assets") / spec.mod_id
        data = Path("src/main/resources/data") / spec.mod_id
        model_id = f"{spec.mod_id}:block/{content.content_id}"
        self._write_text(
            root,
            assets / "blockstates" / f"{content.content_id}.json",
            _json_text({"variants": {"": {"model": model_id}}}),
            raw=True,
        )
        self._write_text(
            root,
            assets / "models/block" / f"{content.content_id}.json",
            _json_text(
                {
                    "parent": "minecraft:block/cube_all",
                    "textures": {"all": f"{spec.mod_id}:block/{content.content_id}"},
                }
            ),
            raw=True,
        )
        self._write_text(
            root,
            assets / "models/item" / f"{content.content_id}.json",
            _json_text({"parent": model_id}),
            raw=True,
        )
        self._write_bytes(
            root,
            assets / "textures/block" / f"{content.content_id}.png",
            make_texture_png(content.color, content.content_id, kind="block"),
        )
        self._write_text(
            root,
            data / "loot_tables/blocks" / f"{content.content_id}.json",
            _json_text(
                {
                    "type": "minecraft:block",
                    "pools": [
                        {
                            "bonus_rolls": 0.0,
                            "rolls": 1.0,
                            "entries": [
                                {
                                    "type": "minecraft:item",
                                    "name": f"{spec.mod_id}:{content.content_id}",
                                }
                            ],
                            "conditions": [
                                {"condition": "minecraft:survives_explosion"}
                            ],
                        }
                    ],
                }
            ),
            raw=True,
        )
        if content.recipe:
            self._write_text(
                root,
                data / "recipes" / f"{content.content_id}.json",
                _json_text(
                    {
                        "type": "minecraft:crafting_shaped",
                        "category": "building",
                        "pattern": ["SSS", "SAS", "SSS"],
                        "key": {
                            "S": {"item": "minecraft:stone"},
                            "A": {"item": "minecraft:amethyst_shard"},
                        },
                        "result": {"item": f"{spec.mod_id}:{content.content_id}"},
                    }
                ),
                raw=True,
            )

    def _write_boss(
        self,
        root: Path,
        spec: ModSpec,
        main_class: str,
        boss: BossSpec,
    ) -> None:
        package_path = Path(*spec.package_name.split("."))
        boss_class = _java_class_name(boss.entity_id) + "Entity"
        renderer_class = _java_class_name(boss.entity_id) + "Renderer"
        entity_java = f"""package {spec.package_name}.entity;

import net.minecraft.entity.EntityType;
import net.minecraft.entity.attribute.DefaultAttributeContainer;
import net.minecraft.entity.attribute.EntityAttributes;
import net.minecraft.entity.boss.BossBar;
import net.minecraft.entity.boss.ServerBossBar;
import net.minecraft.entity.mob.ZombieEntity;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.world.World;

public final class {boss_class} extends ZombieEntity {{
    private final ServerBossBar bossBar = new ServerBossBar(
        Text.translatable("entity.{spec.mod_id}.{boss.entity_id}"),
        BossBar.Color.BLUE,
        BossBar.Style.PROGRESS
    );

    public {boss_class}(EntityType<? extends ZombieEntity> entityType, World world) {{
        super(entityType, world);
        this.experiencePoints = 80;
        this.setBaby(false);
    }}

    public static DefaultAttributeContainer.Builder createBossAttributes() {{
        return ZombieEntity.createZombieAttributes()
            .add(EntityAttributes.GENERIC_MAX_HEALTH, {boss.max_health:.1f})
            .add(EntityAttributes.GENERIC_ATTACK_DAMAGE, {boss.attack_damage:.1f})
            .add(EntityAttributes.GENERIC_MOVEMENT_SPEED, {boss.movement_speed:.3f})
            .add(EntityAttributes.GENERIC_FOLLOW_RANGE, 48.0)
            .add(EntityAttributes.GENERIC_KNOCKBACK_RESISTANCE, 0.65);
    }}

    @Override
    protected boolean burnsInDaylight() {{
        return false;
    }}

    @Override
    public void tick() {{
        super.tick();
        if (!this.getWorld().isClient) {{
            this.bossBar.setPercent(Math.max(0.0f, this.getHealth() / this.getMaxHealth()));
        }}
    }}

    @Override
    public void onStartedTrackingBy(ServerPlayerEntity player) {{
        super.onStartedTrackingBy(player);
        this.bossBar.addPlayer(player);
    }}

    @Override
    public void onStoppedTrackingBy(ServerPlayerEntity player) {{
        super.onStoppedTrackingBy(player);
        this.bossBar.removePlayer(player);
    }}
}}
"""
        client_java = f"""package {spec.package_name}.client;

import {spec.package_name}.{main_class};
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;

public final class {main_class}Client implements ClientModInitializer {{
    @Override
    public void onInitializeClient() {{
        EntityRendererRegistry.register(
            {main_class}.{boss.entity_id.upper()},
            {renderer_class}::new
        );
    }}
}}
"""
        renderer_java = f"""package {spec.package_name}.client;

import {spec.package_name}.entity.{boss_class};
import net.minecraft.client.render.entity.EntityRendererFactory;
import net.minecraft.client.render.entity.MobEntityRenderer;
import net.minecraft.client.render.entity.model.EntityModelLayers;
import net.minecraft.client.render.entity.model.ZombieEntityModel;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.util.Identifier;

public final class {renderer_class}
        extends MobEntityRenderer<{boss_class}, ZombieEntityModel<{boss_class}>> {{
    private static final Identifier TEXTURE =
        new Identifier("{spec.mod_id}", "textures/entity/{boss.entity_id}.png");

    public {renderer_class}(EntityRendererFactory.Context context) {{
        super(
            context,
            new ZombieEntityModel<>(context.getPart(EntityModelLayers.ZOMBIE)),
            0.72f
        );
    }}

    @Override
    public Identifier getTexture({boss_class} entity) {{
        return TEXTURE;
    }}

    @Override
    protected void scale({boss_class} entity, MatrixStack matrices, float amount) {{
        matrices.scale({boss.scale:.3f}f, {boss.scale:.3f}f, {boss.scale:.3f}f);
    }}
}}
"""
        self._write_text(
            root,
            Path("src/main/java") / package_path / "entity" / f"{boss_class}.java",
            entity_java,
            raw=True,
        )
        self._write_text(
            root,
            Path("src/main/java") / package_path / "client" / f"{main_class}Client.java",
            client_java,
            raw=True,
        )
        self._write_text(
            root,
            Path("src/main/java") / package_path / "client" / f"{renderer_class}.java",
            renderer_java,
            raw=True,
        )

        assets = Path("src/main/resources/assets") / spec.mod_id
        data = Path("src/main/resources/data") / spec.mod_id
        self._write_bytes(
            root,
            assets / "textures/entity" / f"{boss.entity_id}.png",
            make_texture_png(
                boss.primary_color,
                boss.entity_id,
                kind="entity",
                size=64,
            ),
        )
        self._write_text(
            root,
            assets / "models/item" / f"{boss.entity_id}_spawn_egg.json",
            _json_text({"parent": "minecraft:item/template_spawn_egg"}),
            raw=True,
        )
        loot_item = (
            f"{spec.mod_id}:{spec.contents[0].content_id}"
            if spec.contents
            else "minecraft:diamond"
        )
        self._write_text(
            root,
            data / "loot_tables/entities" / f"{boss.entity_id}.json",
            _json_text(
                {
                    "type": "minecraft:entity",
                    "pools": [
                        {
                            "rolls": 1,
                            "entries": [
                                {
                                    "type": "minecraft:item",
                                    "name": loot_item,
                                    "functions": [
                                        {
                                            "function": "minecraft:set_count",
                                            "count": {"min": 2.0, "max": 5.0},
                                        }
                                    ],
                                }
                            ],
                            "conditions": [
                                {"condition": "minecraft:killed_by_player"}
                            ],
                        }
                    ],
                }
            ),
            raw=True,
        )
        self._write_text(
            root,
            Path(".minecraft_ai/art_sources") / f"{boss.entity_id}.bbmodel",
            _json_text(_boss_bbmodel(spec, boss)),
            raw=True,
        )
        self._write_text(
            root,
            Path(".minecraft_ai/art_sources") / f"{boss.entity_id}.obj",
            _boss_obj(boss),
            raw=True,
        )
        self._write_text(
            root,
            Path(".minecraft_ai/art_sources") / f"{boss.entity_id}.mtl",
            (
                f"newmtl {boss.entity_id}_material\n"
                f"Kd {int(boss.primary_color[1:3], 16) / 255:.4f} "
                f"{int(boss.primary_color[3:5], 16) / 255:.4f} "
                f"{int(boss.primary_color[5:7], 16) / 255:.4f}\n"
                f"map_Kd ../../src/main/resources/assets/{spec.mod_id}/textures/entity/{boss.entity_id}.png\n"
            ),
            raw=True,
        )

    def _write_tags(self, root: Path, spec: ModSpec) -> None:
        blocks = [
            f"{spec.mod_id}:{content.content_id}"
            for content in spec.contents
            if content.kind is ContentKind.BLOCK
        ]
        if not blocks:
            return
        tag_root = Path("src/main/resources/data/minecraft/tags/blocks")
        self._write_text(
            root,
            tag_root / "mineable/pickaxe.json",
            _json_text({"replace": False, "values": blocks}),
            raw=True,
        )
        self._write_text(
            root,
            tag_root / "needs_iron_tool.json",
            _json_text({"replace": False, "values": blocks}),
            raw=True,
        )

    def _write_contract(self, root: Path, spec: ModSpec) -> None:
        package_path = Path(*spec.package_name.split("."))
        contract_ids = [content.content_id for content in spec.contents]
        if spec.boss is not None:
            contract_ids.append(spec.boss.entity_id)
        ids = ", ".join(f'"{content_id}"' for content_id in contract_ids)
        self._write_text(
            root,
            Path("src/main/java") / package_path / "GeneratedContent.java",
            f"""package {spec.package_name};

public final class GeneratedContent {{
    public static final String[] CONTENT_IDS = new String[] {{{ids}}};

    private GeneratedContent() {{
    }}
}}
""",
            raw=True,
        )
        manifest_contents: list[dict[str, object]] = [
            {
                "content_id": content.content_id,
                "kind": content.kind.value,
                "trace_edges": {
                    "implementation": [
                        f"src/main/java/{spec.package_name.replace('.', '/')}/{_java_class_name(spec.mod_id)}.java"
                    ],
                    "assets": [
                        f"src/main/resources/assets/{spec.mod_id}/models/item/{content.content_id}.json",
                        (
                            f"src/main/resources/assets/{spec.mod_id}/textures/item/{content.content_id}.png"
                            if content.kind is ContentKind.ITEM
                            else f"src/main/resources/assets/{spec.mod_id}/textures/block/{content.content_id}.png"
                        ),
                    ],
                    "world_placement": {
                        "status": "NOT_APPLICABLE",
                        "reason": "This item/block is registry content, not an automatic world edit.",
                    },
                    "tests": [
                        f"src/main/java/{spec.package_name.replace('.', '/')}/{_java_class_name(spec.mod_id)}GameTests.java",
                        f"src/test/java/{spec.package_name.replace('.', '/')}/GeneratedContractTest.java",
                    ],
                    "documentation": ["README.md"],
                },
            }
            for content in spec.contents
        ]
        if spec.boss is not None:
            boss_class = _java_class_name(spec.boss.entity_id) + "Entity"
            manifest_contents.append(
                {
                    "content_id": spec.boss.entity_id,
                    "kind": "boss",
                    "trace_edges": {
                        "implementation": [
                            f"src/main/java/{spec.package_name.replace('.', '/')}/entity/{boss_class}.java",
                            f"src/main/java/{spec.package_name.replace('.', '/')}/client/{_java_class_name(spec.boss.entity_id)}Renderer.java",
                        ],
                        "assets": [
                            f"src/main/resources/assets/{spec.mod_id}/textures/entity/{spec.boss.entity_id}.png",
                            f".minecraft_ai/art_sources/{spec.boss.entity_id}.bbmodel",
                            f".minecraft_ai/art_sources/{spec.boss.entity_id}.obj",
                        ],
                        "world_placement": {
                            "status": "EXPLICIT_ONLY",
                            "reason": "Spawn egg only; no automatic natural spawn.",
                        },
                        "tests": [
                            f"src/main/java/{spec.package_name.replace('.', '/')}/{_java_class_name(spec.mod_id)}GameTests.java"
                        ],
                        "documentation": ["README.md"],
                    },
                }
            )
        manifest = {
            "schema_version": "minecraft-mod-ai/content-manifest-v1",
            "mod_id": spec.mod_id,
            "contents": manifest_contents,
        }
        self._write_text(
            root,
            ".minecraft_ai/generation_contract.json",
            _json_text(manifest),
            raw=True,
        )

    def _project_readme(self, spec: ModSpec) -> str:
        content_lines = "\n".join(
            f"- `{content.content_id}` ({content.kind.value}): "
            f"{content.display_name_en} / {content.display_name_ko}"
            for content in spec.contents
        )
        if spec.boss is not None:
            content_lines += (
                f"\n- `{spec.boss.entity_id}` (boss): "
                f"{spec.boss.display_name_en} / {spec.boss.display_name_ko}; "
                "server boss bar, loot, spawn egg, editable Blockbench/OBJ source"
            )
        return f"""# {spec.mod_name}

M.M.M Make Mincraft Mode가 생성한 Fabric 프로젝트입니다.

- Minecraft: {spec.platform.minecraft_version}
- Java: {spec.platform.java_version}
- Fabric Loader: {spec.platform.fabric_loader}
- Fabric API: {spec.platform.fabric_api}

## 생성 콘텐츠

{content_lines}

## 빌드

Gradle Wrapper가 있으면 Linux/macOS에서 `./gradlew clean build`, Windows에서
`gradlew.bat clean build`를 실행하세요. Wrapper가 아직 없다면 상위
M.M.M Make Mincraft Mode의 `build` 과정이 검증된 Gradle 배포본을 받아
wrapper를 만든 뒤 실제 빌드를 수행합니다.

성공한 JAR는 `build/libs/{spec.mod_id}-{spec.version}.jar`입니다.
"""


def _boss_bbmodel(spec: ModSpec, boss: BossSpec) -> dict[str, object]:
    cubes = (
        ("head", [4, 24, 4], [12, 32, 12], "11111111-1111-4111-8111-111111111111"),
        ("body", [4, 12, 6], [12, 24, 10], "22222222-2222-4222-8222-222222222222"),
        ("left_arm", [12, 12, 6], [16, 24, 10], "33333333-3333-4333-8333-333333333333"),
        ("right_arm", [0, 12, 6], [4, 24, 10], "44444444-4444-4444-8444-444444444444"),
        ("left_leg", [8, 0, 6], [12, 12, 10], "55555555-5555-4555-8555-555555555555"),
        ("right_leg", [4, 0, 6], [8, 12, 10], "66666666-6666-4666-8666-666666666666"),
    )
    faces = {
        direction: {"uv": [0, 0, 8, 8], "texture": 0}
        for direction in ("north", "east", "south", "west", "up", "down")
    }
    elements = [
        {
            "name": name,
            "box_uv": True,
            "rescale": False,
            "locked": False,
            "light_emission": 0,
            "render_order": "default",
            "allow_mirror_modeling": True,
            "from": start,
            "to": end,
            "autouv": 0,
            "color": index,
            "origin": [8, 12, 8],
            "faces": faces,
            "type": "cube",
            "uuid": uuid,
        }
        for index, (name, start, end, uuid) in enumerate(cubes)
    ]
    return {
        "meta": {
            "format_version": "4.10",
            "model_format": "free",
            "box_uv": True,
        },
        "name": boss.entity_id,
        "model_identifier": f"{spec.mod_id}:{boss.entity_id}",
        "visible_box": [2, 3, 0],
        "variable_placeholders": "",
        "variable_placeholder_buttons": [],
        "resolution": {"width": 64, "height": 64},
        "elements": elements,
        "outliner": [uuid for _, _, _, uuid in cubes],
        "textures": [
            {
                "path": (
                    f"../../src/main/resources/assets/{spec.mod_id}/textures/entity/"
                    f"{boss.entity_id}.png"
                ),
                "name": f"{boss.entity_id}.png",
                "folder": "textures/entity",
                "namespace": spec.mod_id,
                "id": "0",
                "width": 64,
                "height": 64,
                "uv_width": 64,
                "uv_height": 64,
                "particle": False,
                "render_mode": "default",
                "render_sides": "auto",
                "frame_time": 1,
                "frame_order_type": "loop",
                "frame_order": "",
                "frame_interpolate": False,
                "visible": True,
                "internal": False,
                "saved": True,
                "uuid": "77777777-7777-4777-8777-777777777777",
                "relative_path": f"textures/entity/{boss.entity_id}.png",
            }
        ],
    }


def _boss_obj(boss: BossSpec) -> str:
    parts = (
        ("head", (-4.0, 24.0, -4.0), (4.0, 32.0, 4.0)),
        ("body", (-4.0, 12.0, -2.0), (4.0, 24.0, 2.0)),
        ("left_arm", (4.0, 12.0, -2.0), (8.0, 24.0, 2.0)),
        ("right_arm", (-8.0, 12.0, -2.0), (-4.0, 24.0, 2.0)),
        ("left_leg", (0.0, 0.0, -2.0), (4.0, 12.0, 2.0)),
        ("right_leg", (-4.0, 0.0, -2.0), (0.0, 12.0, 2.0)),
    )
    lines = [
        f"# Deterministic cuboid model for {boss.entity_id}",
        f"mtllib {boss.entity_id}.mtl",
        f"usemtl {boss.entity_id}_material",
    ]
    vertex_offset = 1
    for name, minimum, maximum in parts:
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        vertices = (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
        lines.append(f"o {name}")
        lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in vertices)
        faces = (
            (1, 2, 3, 4),
            (5, 8, 7, 6),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 8, 4),
            (5, 1, 4, 8),
        )
        for face in faces:
            indices = " ".join(str(vertex_offset + index - 1) for index in face)
            lines.append(f"f {indices}")
        vertex_offset += 8
    return "\n".join(lines) + "\n"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def make_texture_png(color: str, seed: str, *, kind: str, size: int = 16) -> bytes:
    """Create a deterministic, license-clean RGBA Minecraft texture."""
    red, green, blue = _hex_to_rgb(color)
    seed_value = sum((index + 1) * ord(char) for index, char in enumerate(seed))
    rows: list[bytes] = []
    center = (size - 1) / 2
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            checker = ((x // 4) + (y // 4) + seed_value) % 2
            distance = math.sqrt((x - center) ** 2 + (y - center) ** 2)
            delta = 24 if checker == 0 else -16
            if kind == "item" and distance > size * 0.43:
                row.extend((0, 0, 0, 0))
                continue
            highlight = 28 if (x + y + seed_value) % 7 == 0 else 0
            row.extend(
                (
                    max(0, min(255, red + delta + highlight)),
                    max(0, min(255, green + delta + highlight)),
                    max(0, min(255, blue + delta + highlight)),
                    255,
                )
            )
        rows.append(bytes(row))
    raw = b"".join(rows)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


_GITIGNORE = """.gradle/
build/
run/
*.log
"""

_MIT_LICENSE = """MIT License

Copyright (c) 2026 M.M.M Make Mincraft Mode user

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
