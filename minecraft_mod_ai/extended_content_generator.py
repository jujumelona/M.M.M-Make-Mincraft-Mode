from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .complete_spec import ProductionModule
from .generator import make_texture_png
from .project_edit import ensure_main_initializer_call, inspect_fabric_project, write_text_files
from .scale_policy import ScalePolicy


class ExtendedContentError(RuntimeError):
    pass


_SUPPORTED = frozenset(
    {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "machine",
        "effect",
        "enchantment",
        "command",
        "recipe",
        "advancement",
        "loot",
    }
)
_JAVA_KINDS = _SUPPORTED - {"recipe", "advancement", "loot"}
_CATALOG_SCHEMA = "mmm/extended-module-catalog-v1"
_DIRECTORY_CATALOG_SCHEMA = "mmm/extended-module-directory-v1"
_CATALOG_NODE_SCHEMA = "mmm/extended-module-catalog-node-v1"
_CATALOG_SHARD_SCHEMA = "mmm/extended-module-shard-v1"


def generate_extended_content(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    modules: Iterable[ProductionModule],
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ExtendedContentError("Extended content target does not match fabric.mod.json.")
    selected = tuple(module for module in modules if module.kind in _SUPPORTED)
    if not selected:
        return {"schema_version": "mmm/extended-content-v2", "status": "SKIPPED", "modules": []}
    for module in selected:
        module.validate(policy=policy)

    existing = {
        str(item["module_id"]): item
        for item in iter_extended_module_records(info.root)
    }
    for module in selected:
        existing[module.module_id] = {
            "module_id": module.module_id,
            "kind": module.kind,
            "config": module.config,
            "depends_on": list(module.depends_on),
            "required_gates": list(module.required_gates),
        }
    ordered = [existing[key] for key in sorted(existing)]
    catalog_path = info.root / ".minecraft_ai/extended-modules.json"
    already_directory = False
    if catalog_path.is_file() and not catalog_path.is_symlink():
        try:
            already_directory = (
                json.loads(catalog_path.read_text(encoding="utf-8")).get(
                    "schema_version"
                )
                == _DIRECTORY_CATALOG_SCHEMA
            )
        except (json.JSONDecodeError, OSError, AttributeError):
            already_directory = False
    selected_records = [existing[module.module_id] for module in selected]
    generation_records = (
        selected_records if already_directory else ordered
    )
    files = _extended_directory_catalog_files(
        selected_records if already_directory else ordered,
        module_count=len(ordered),
    )
    lang_en: dict[str, str] = {}
    lang_ko: dict[str, str] = {}
    generated_binary: list[str] = []

    for item in generation_records:
        module_id = item["module_id"]
        kind = item["kind"]
        config = item["config"]
        display_en = str(config.get("display_name_en", module_id.replace("_", " ").title()))
        display_ko = str(config.get("display_name_ko", display_en))
        if kind in {"block", "crop", "machine"}:
            lang_en[f"block.{mod_id}.{module_id}"] = display_en
            lang_ko[f"block.{mod_id}.{module_id}"] = display_ko
            files.update(_block_resources(mod_id, module_id, kind, config))
            texture_path = info.root / f"src/main/resources/assets/{mod_id}/textures/block/{module_id}.png"
            texture_path.parent.mkdir(parents=True, exist_ok=True)
            texture_path.write_bytes(
                make_texture_png(str(config.get("color", "#748cab")), module_id, kind="block", size=16)
            )
            generated_binary.append(str(texture_path))
            if kind == "crop":
                lang_en[f"item.{mod_id}.{module_id}_seeds"] = display_en + " Seeds"
                lang_ko[f"item.{mod_id}.{module_id}_seeds"] = display_ko + " 씨앗"
                seed_texture = info.root / f"src/main/resources/assets/{mod_id}/textures/item/{module_id}_seeds.png"
                seed_texture.parent.mkdir(parents=True, exist_ok=True)
                seed_texture.write_bytes(
                    make_texture_png(
                        str(config.get("seed_color", config.get("color", "#8fbf5f"))),
                        module_id + "_seeds",
                        kind="item",
                        size=16,
                    )
                )
                generated_binary.append(str(seed_texture))
        elif kind in {"effect", "enchantment"}:
            prefix = "effect" if kind == "effect" else "enchantment"
            lang_en[f"{prefix}.{mod_id}.{module_id}"] = display_en
            lang_ko[f"{prefix}.{mod_id}.{module_id}"] = display_ko
        elif kind in {"recipe", "advancement", "loot"}:
            files.update(_data_only_resource(mod_id, module_id, kind, config))
        elif kind == "command":
            pass
        else:
            lang_en[f"item.{mod_id}.{module_id}"] = display_en
            lang_ko[f"item.{mod_id}.{module_id}"] = display_ko
            files.update(_item_resources(mod_id, module_id, kind, config))
            texture_path = info.root / f"src/main/resources/assets/{mod_id}/textures/item/{module_id}.png"
            texture_path.parent.mkdir(parents=True, exist_ok=True)
            texture_path.write_bytes(
                make_texture_png(str(config.get("color", "#74c7ec")), module_id, kind="item", size=16)
            )
            generated_binary.append(str(texture_path))

    java_items = [
        item for item in generation_records if item["kind"] in _JAVA_KINDS
    ]
    for item in java_items:
        class_name = _unit_class_name(str(item["module_id"]))
        files[_java_path(package_name, class_name)] = _shard_java(
            package_name,
            mod_id,
            class_name,
            [item],
        )
    files[_java_path(package_name, "GeneratedExtendedContent")] = _root_java(
        package_name,
        mod_id,
    )

    _merge_lang(info.root / f"src/main/resources/assets/{mod_id}/lang/en_us.json", lang_en)
    _merge_lang(info.root / f"src/main/resources/assets/{mod_id}/lang/ko_kr.json", lang_ko)
    receipt = write_text_files(info, files, replace_existing=True)
    binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.extended.GeneratedExtendedContent",
        call_line="GeneratedExtendedContent.register()",
        marker="extended:content",
    )
    return {
        "schema_version": "mmm/extended-content-v2",
        "status": "GENERATED",
        "modules": [item["module_id"] for item in generation_records],
        "catalog_module_count": len(ordered),
        "shard_count": len(java_items),
        "shard_size": policy.java_shard_size,
        "registrar_dispatch_count": 0,
        "files": [str(info.root / path) for path in files] + generated_binary,
        "source_receipt": receipt,
        "binding_receipt": binding,
        "required_gates": ["JDT", "Gradle", "GameTest", "runtime interaction tests"],
    }


def iter_extended_module_records(
    project_root: str | Path,
) -> Iterator[dict[str, Any]]:
    """Read both legacy monoliths and the bounded catalog tree."""

    root = Path(project_root).expanduser().resolve()
    catalog = root / ".minecraft_ai/extended-modules.json"
    if not catalog.is_file() or catalog.is_symlink():
        return
    raw = json.loads(catalog.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ExtendedContentError("Extended module catalog must be an object.")
    legacy = raw.get("modules")
    if isinstance(legacy, list):
        for item in legacy:
            if not isinstance(item, dict) or not item.get("module_id"):
                raise ExtendedContentError(
                    "Legacy extended module catalog contains an invalid module."
                )
            yield item
        return
    if raw.get("schema_version") == _DIRECTORY_CATALOG_SCHEMA:
        relative = raw.get("directory")
        expected = raw.get("module_count")
        if not isinstance(relative, str) or type(expected) is not int:
            raise ExtendedContentError(
                "Extended module directory catalog is invalid."
            )
        normalized = PurePosixPath(relative.replace("\\", "/"))
        if (
            normalized.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            raise ExtendedContentError(
                "Extended module directory path is unsafe."
            )
        directory = (root / Path(*normalized.parts)).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise ExtendedContentError(
                "Extended module directory escaped the project."
            ) from exc
        if not directory.is_dir() or directory.is_symlink():
            raise ExtendedContentError(
                "Extended module directory is missing or unsafe."
            )
        yielded = 0
        for path in sorted(directory.glob("*.json")):
            if not path.is_file() or path.is_symlink():
                raise ExtendedContentError(
                    "Extended module record is unsafe."
                )
            item = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(item, dict)
                or not item.get("module_id")
                or path.stem != str(item["module_id"])
            ):
                raise ExtendedContentError(
                    "Extended module record is invalid."
                )
            yielded += 1
            yield item
        if yielded != expected:
            raise ExtendedContentError(
                "Extended module directory count does not match."
            )
        return
    if raw.get("schema_version") != _CATALOG_SCHEMA:
        raise ExtendedContentError("Unsupported extended module catalog schema.")
    entry = raw.get("root")
    if not isinstance(entry, str) or not entry:
        raise ExtendedContentError("Extended module catalog root is invalid.")

    pending = [entry]
    visited: set[str] = set()
    yielded = 0
    while pending:
        relative = pending.pop()
        if relative in visited:
            raise ExtendedContentError("Extended module catalog contains a cycle.")
        visited.add(relative)
        node_path = _catalog_target(root, relative)
        node = json.loads(node_path.read_text(encoding="utf-8"))
        if not isinstance(node, dict):
            raise ExtendedContentError("Extended module catalog node is invalid.")
        schema = node.get("schema_version")
        if schema == _CATALOG_NODE_SCHEMA:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                raise ExtendedContentError(
                    "Extended module catalog index has no children."
                )
            if not all(isinstance(value, str) and value for value in children):
                raise ExtendedContentError(
                    "Extended module catalog child path is invalid."
                )
            pending.extend(reversed(children))
            continue
        if schema != _CATALOG_SHARD_SCHEMA:
            raise ExtendedContentError(
                "Unsupported extended module catalog node schema."
            )
        modules = node.get("modules")
        if not isinstance(modules, list) or not modules:
            raise ExtendedContentError("Extended module shard is empty or invalid.")
        for item in modules:
            if not isinstance(item, dict) or not item.get("module_id"):
                raise ExtendedContentError(
                    "Extended module shard contains an invalid module."
                )
            yielded += 1
            yield item
    expected = raw.get("module_count")
    if type(expected) is not int or expected != yielded:
        raise ExtendedContentError("Extended module catalog count does not match.")


def _extended_directory_catalog_files(
    modules: list[dict[str, Any]],
    *,
    module_count: int,
) -> dict[str, str]:
    directory = ".minecraft_ai/extended-module-records"
    files = {
        f"{directory}/{item['module_id']}.json": _json_text(item)
        for item in modules
    }
    files[".minecraft_ai/extended-modules.json"] = _json_text(
        {
            "schema_version": _DIRECTORY_CATALOG_SCHEMA,
            "module_count": module_count,
            "directory": directory,
        }
    )
    return files


def _extended_catalog_files(
    modules: list[dict[str, Any]],
    *,
    shard_size: int,
) -> dict[str, str]:
    base = ".minecraft_ai/extended-modules"
    files: dict[str, str] = {}
    leaves: list[str] = []
    for offset in range(0, len(modules), shard_size):
        relative = f"{base}/shards/modules-{offset // shard_size:08d}.json"
        leaves.append(relative)
        files[relative] = _json_text(
            {
                "schema_version": _CATALOG_SHARD_SCHEMA,
                "modules": modules[offset : offset + shard_size],
            }
        )
    current = leaves
    fanout = _bounded_fanout(shard_size)
    level = 0
    while len(current) > 1:
        parents: list[str] = []
        for offset in range(0, len(current), fanout):
            relative = (
                f"{base}/index/level-{level:04d}-"
                f"{offset // fanout:08d}.json"
            )
            parents.append(relative)
            files[relative] = _json_text(
                {
                    "schema_version": _CATALOG_NODE_SCHEMA,
                    "children": current[offset : offset + fanout],
                }
            )
        current = parents
        level += 1
    files[".minecraft_ai/extended-modules.json"] = _json_text(
        {
            "schema_version": _CATALOG_SCHEMA,
            "module_count": len(modules),
            "shard_size": shard_size,
            "root": current[0],
        }
    )
    return files


def _catalog_target(root: Path, relative: str) -> Path:
    normalized = PurePosixPath(relative.replace("\\", "/"))
    if (
        normalized.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ExtendedContentError("Extended module catalog path is unsafe.")
    candidate = root / Path(*normalized.parts)
    if candidate.is_symlink():
        raise ExtendedContentError("Extended module catalog path is a symlink.")
    target = candidate.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ExtendedContentError(
            "Extended module catalog path escaped the project."
        ) from exc
    if not target.is_file():
        raise ExtendedContentError(
            f"Extended module catalog node is missing: {relative}"
        )
    return target


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _bounded_fanout(shard_size: int) -> int:
    return max(2, min(shard_size, 32))


def _java_path(package_name: str, class_name: str) -> str:
    return (
        "src/main/java/"
        + package_name.replace(".", "/")
        + "/extended/"
        + class_name
        + ".java"
    )


def _unit_class_name(module_id: str) -> str:
    return (
        "GeneratedContentUnit"
        + hashlib.sha256(module_id.encode("utf-8")).hexdigest()[:20]
    )


def _root_java(
    package_name: str,
    mod_id: str,
) -> str:
    return f'''package {package_name}.extended;

import net.fabricmc.loader.api.FabricLoader;
import net.fabricmc.fabric.api.object.builder.v1.block.entity.FabricBlockEntityTypeBuilder;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.BlockWithEntity;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.block.entity.BlockEntityTicker;
import net.minecraft.block.entity.BlockEntityType;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.inventory.Inventories;
import net.minecraft.item.ItemStack;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.Identifier;
import net.minecraft.util.collection.DefaultedList;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.nio.file.Files;
import java.nio.file.Path;

public final class GeneratedExtendedContent {{
    private static final String MOD_ID = "{mod_id}";
    public static BlockEntityType<GeneratedMachineBlockEntity> MACHINE_ENTITY_TYPE;
    private static boolean registered;

    private GeneratedExtendedContent() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        List<Block> machineBlocks = registerGeneratedUnits();
        if (!machineBlocks.isEmpty()) {{
            MACHINE_ENTITY_TYPE = Registry.register(
                Registries.BLOCK_ENTITY_TYPE,
                new Identifier(MOD_ID, "generated_machine"),
                FabricBlockEntityTypeBuilder.create(
                    GeneratedMachineBlockEntity::new,
                    machineBlocks.toArray(Block[]::new)
                ).build()
            );
        }}
    }}

    @SuppressWarnings("unchecked")
    private static List<Block> registerGeneratedUnits() {{
        Set<String> classes = new TreeSet<>();
        String relative = "{package_name.replace(".", "/")}/extended";
        FabricLoader.getInstance().getModContainer(MOD_ID).orElseThrow().getRootPaths().forEach(root -> {{
            Path directory = root.resolve(relative);
            if (!Files.isDirectory(directory)) return;
            try (var paths = Files.list(directory)) {{
                paths.filter(path -> {{
                    String name = path.getFileName().toString();
                    return name.startsWith("GeneratedContentUnit") && name.endsWith(".class");
                }}).forEach(path -> {{
                    String name = path.getFileName().toString();
                    classes.add("{package_name}.extended." + name.substring(0, name.length() - 6));
                }});
            }} catch (java.io.IOException error) {{
                throw new IllegalStateException("Could not enumerate generated content units", error);
            }}
        }});
        List<Block> machineBlocks = new ArrayList<>();
        for (String className : classes) {{
            try {{
                Class<?> unit = Class.forName(
                    className,
                    true,
                    GeneratedExtendedContent.class.getClassLoader()
                );
                unit.getMethod("register").invoke(null);
                Object value = unit.getMethod("machineBlocks").invoke(null);
                if (value instanceof List<?> blocks) {{
                    for (Object block : blocks) {{
                        if (block instanceof Block typed) machineBlocks.add(typed);
                    }}
                }}
            }} catch (ReflectiveOperationException error) {{
                throw new IllegalStateException(
                    "Could not register generated content unit " + className,
                    error
                );
            }}
        }}
        return machineBlocks;
    }}

    public record MachineDefinition(
        Identifier input,
        Identifier output,
        int outputCount,
        int processingTicks
    ) {{}}

    public static final class GeneratedMachineBlock extends BlockWithEntity {{
        private final MachineDefinition definition;

        public GeneratedMachineBlock(Settings settings, MachineDefinition definition) {{
            super(settings);
            this.definition = definition;
        }}

        public MachineDefinition definition() {{ return definition; }}

        @Override
        public BlockEntity createBlockEntity(BlockPos pos, BlockState state) {{
            return new GeneratedMachineBlockEntity(pos, state);
        }}

        @Override
        public ActionResult onUse(
            BlockState state,
            World world,
            BlockPos pos,
            PlayerEntity player,
            Hand hand,
            BlockHitResult hit
        ) {{
            if (world.isClient) return ActionResult.SUCCESS;
            BlockEntity raw = world.getBlockEntity(pos);
            if (!(raw instanceof GeneratedMachineBlockEntity machine)) return ActionResult.PASS;
            ItemStack held = player.getStackInHand(hand);
            if (held.isEmpty()) {{
                ItemStack output = machine.takeOutput();
                if (!output.isEmpty()) player.giveItemStack(output);
                return ActionResult.CONSUME;
            }}
            if (Registries.ITEM.getId(held.getItem()).equals(definition.input()) && machine.insertInput(held)) {{
                player.sendMessage(Text.literal("Machine started"), true);
                return ActionResult.CONSUME;
            }}
            return ActionResult.PASS;
        }}

        @Override
        public <T extends BlockEntity> BlockEntityTicker<T> getTicker(
            World world,
            BlockState state,
            BlockEntityType<T> type
        ) {{
            if (world.isClient || MACHINE_ENTITY_TYPE == null) return null;
            return checkType(type, MACHINE_ENTITY_TYPE, GeneratedMachineBlockEntity::tick);
        }}
    }}

    public static final class GeneratedMachineBlockEntity extends BlockEntity {{
        private final DefaultedList<ItemStack> items = DefaultedList.ofSize(2, ItemStack.EMPTY);
        private int progress;

        public GeneratedMachineBlockEntity(BlockPos pos, BlockState state) {{
            super(MACHINE_ENTITY_TYPE, pos, state);
        }}

        public boolean insertInput(ItemStack source) {{
            if (!items.get(0).isEmpty()) return false;
            items.set(0, source.split(1));
            progress = 0;
            markDirty();
            return true;
        }}

        public ItemStack takeOutput() {{
            ItemStack result = items.get(1);
            items.set(1, ItemStack.EMPTY);
            markDirty();
            return result;
        }}

        public static void tick(World world, BlockPos pos, BlockState state, GeneratedMachineBlockEntity machine) {{
            if (!(state.getBlock() instanceof GeneratedMachineBlock block)) return;
            MachineDefinition definition = block.definition();
            ItemStack input = machine.items.get(0);
            if (input.isEmpty() || !machine.items.get(1).isEmpty()) {{
                machine.progress = 0;
                return;
            }}
            if (!Registries.ITEM.getId(input.getItem()).equals(definition.input())) return;
            machine.progress++;
            if (machine.progress < definition.processingTicks()) return;
            machine.items.set(0, ItemStack.EMPTY);
            machine.items.set(
                1,
                new ItemStack(Registries.ITEM.get(definition.output()), definition.outputCount())
            );
            machine.progress = 0;
            machine.markDirty();
        }}

        @Override
        protected void writeNbt(NbtCompound nbt) {{
            super.writeNbt(nbt);
            Inventories.writeNbt(nbt, items);
            nbt.putInt("Progress", progress);
        }}

        @Override
        public void readNbt(NbtCompound nbt) {{
            super.readNbt(nbt);
            Inventories.readNbt(nbt, items);
            progress = nbt.getInt("Progress");
        }}
    }}
}}
'''


def _registrar_tree_files(
    package_name: str,
    leaf_names: list[str],
    *,
    fanout: int,
) -> tuple[str | None, dict[str, str]]:
    if not leaf_names:
        return None, {}
    current = list(leaf_names)
    files: dict[str, str] = {}
    level = 0
    while len(current) > 1:
        parents: list[str] = []
        for offset in range(0, len(current), fanout):
            children = current[offset : offset + fanout]
            class_name = (
                f"GeneratedContentDispatchL{level:03d}"
                f"N{len(parents):08d}"
            )
            parents.append(class_name)
            files[_java_path(package_name, class_name)] = _dispatch_java(
                package_name,
                class_name,
                children,
            )
        current = parents
        level += 1
    return current[0], files


def _dispatch_java(
    package_name: str,
    class_name: str,
    children: list[str],
) -> str:
    calls = "\n".join(f"        {name}.register();" for name in children)
    collectors = "\n".join(
        f"        blocks.addAll({name}.machineBlocks());"
        for name in children
    )
    return f'''package {package_name}.extended;

import net.minecraft.block.Block;

import java.util.ArrayList;
import java.util.List;

final class {class_name} {{
    private {class_name}() {{}}

    static void register() {{
{calls}
    }}

    static List<Block> machineBlocks() {{
        List<Block> blocks = new ArrayList<>();
{collectors}
        return blocks;
    }}
}}
'''


def _shard_java(
    package_name: str,
    mod_id: str,
    class_name: str,
    modules: list[dict[str, Any]],
) -> str:
    fields: list[str] = []
    registrations: list[str] = []
    commands: list[str] = []
    creative: list[str] = []
    machine_fields: list[str] = []

    for item in modules:
        module_id, kind, config = item["module_id"], item["kind"], item["config"]
        constant = _constant(module_id)
        if kind == "item":
            fields.append(f"    public static Item {constant};")
            registrations.append(
                f'        {constant} = item("{module_id}", new Item(new FabricItemSettings()));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "food":
            hunger = max(0, int(config.get("hunger", 4)))
            saturation = float(config.get("saturation", 0.4))
            fields.append(f"    public static Item {constant};")
            registrations.append(
                f'        {constant} = item("{module_id}", new Item(new FabricItemSettings().food(new FoodComponent.Builder().hunger({hunger}).saturationModifier({saturation:.3f}f).build())));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "weapon":
            damage = int(config.get("attack_damage", 4))
            speed = float(config.get("attack_speed", -2.4))
            fields.append(f"    public static Item {constant};")
            registrations.append(
                f'        {constant} = item("{module_id}", new SwordItem(ToolMaterials.IRON, {damage}, {speed:.3f}f, new FabricItemSettings()));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "tool":
            damage = int(config.get("attack_damage", 1))
            speed = float(config.get("attack_speed", -2.8))
            fields.append(f"    public static Item {constant};")
            registrations.append(
                f'        {constant} = item("{module_id}", new PickaxeItem(ToolMaterials.IRON, {damage}, {speed:.3f}f, new FabricItemSettings()));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "armor":
            armor_type = str(config.get("slot", "chestplate")).upper()
            if armor_type not in {"HELMET", "CHESTPLATE", "LEGGINGS", "BOOTS"}:
                raise ExtendedContentError(f"Invalid armor slot for {module_id}: {armor_type}")
            fields.append(f"    public static Item {constant};")
            registrations.append(
                f'        {constant} = item("{module_id}", new ArmorItem(ArmorMaterials.IRON, ArmorItem.Type.{armor_type}, new FabricItemSettings()));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "block":
            fields.append(f"    public static Block {constant};")
            registrations.append(
                f'        {constant} = block("{module_id}", new Block(FabricBlockSettings.copyOf(Blocks.STONE).strength({float(config.get("hardness", 3.0)):.2f}f)));'
            )
            creative.append(f"            entries.add({constant});")
        elif kind == "machine":
            input_id = _identifier(config.get("input_item", "minecraft:iron_ingot"), module_id)
            output_id = _identifier(config.get("output_item", "minecraft:gold_ingot"), module_id)
            output_count = max(1, int(config.get("output_count", 1)))
            ticks = max(1, int(config.get("processing_ticks", 100)))
            fields.append(f"    public static Block {constant};")
            registrations.append(
                f'''        {constant} = block("{module_id}", new GeneratedExtendedContent.GeneratedMachineBlock(
            FabricBlockSettings.copyOf(Blocks.IRON_BLOCK).strength(3.5f),
            new GeneratedExtendedContent.MachineDefinition(
                new Identifier("{input_id}"),
                new Identifier("{output_id}"),
                {output_count},
                {ticks}
            )
        ));'''
            )
            machine_fields.append(constant)
            creative.append(f"            entries.add({constant});")
        elif kind == "crop":
            fields.extend([f"    public static Item {constant}_SEEDS;", f"    public static Block {constant};"])
            registrations.extend(
                [
                    f'        {constant} = Registry.register(Registries.BLOCK, id("{module_id}"), new CropBlock(FabricBlockSettings.copyOf(Blocks.WHEAT)) {{ @Override protected ItemConvertible getSeedsItem() {{ return {constant}_SEEDS; }} }});',
                    f'        {constant}_SEEDS = item("{module_id}_seeds", new AliasedBlockItem({constant}, new FabricItemSettings()));',
                ]
            )
            creative.append(f"            entries.add({constant}_SEEDS);")
        elif kind == "effect":
            color = int(str(config.get("color", "#74c7ec")).lstrip("#"), 16)
            fields.append(f"    public static StatusEffect {constant};")
            registrations.append(
                f'        {constant} = Registry.register(Registries.STATUS_EFFECT, id("{module_id}"), new GeneratedEffect(0x{color:06X}));'
            )
        elif kind == "enchantment":
            max_level = max(1, int(config.get("max_level", 3)))
            fields.append(f"    public static Enchantment {constant};")
            registrations.append(
                f'        {constant} = Registry.register(Registries.ENCHANTMENT, id("{module_id}"), new GeneratedEnchantment({max_level}));'
            )
        elif kind == "command":
            literal = re.sub(r"[^a-z0-9_]", "", str(config.get("literal", module_id))) or module_id
            message = _java_string(str(config.get("message", module_id.replace("_", " "))))
            permission = max(0, int(config.get("permission_level", 0)))
            commands.append(
                f'        dispatcher.register(CommandManager.literal("{literal}").requires(source -> source.hasPermissionLevel({permission})).executes(context -> {{ context.getSource().sendFeedback(() -> Text.literal("{message}"), false); return 1; }}));'
            )

    command_registration = ""
    if commands:
        command_registration = '''
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
%s
        });''' % "\n".join(commands)

    machine_list = (
        "List.of(" + ", ".join(machine_fields) + ")" if machine_fields else "List.of()"
    )
    return f'''package {package_name}.extended;

import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.item.v1.FabricItemSettings;
import net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents;
import net.fabricmc.fabric.api.object.builder.v1.block.FabricBlockSettings;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.block.CropBlock;
import net.minecraft.enchantment.Enchantment;
import net.minecraft.enchantment.EnchantmentTarget;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.entity.effect.StatusEffect;
import net.minecraft.entity.effect.StatusEffectCategory;
import net.minecraft.item.AliasedBlockItem;
import net.minecraft.item.ArmorItem;
import net.minecraft.item.ArmorMaterials;
import net.minecraft.item.BlockItem;
import net.minecraft.item.FoodComponent;
import net.minecraft.item.Item;
import net.minecraft.item.ItemConvertible;
import net.minecraft.item.ItemGroups;
import net.minecraft.item.PickaxeItem;
import net.minecraft.item.SwordItem;
import net.minecraft.item.ToolMaterials;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.List;

public final class {class_name} {{
    private static final String MOD_ID = "{mod_id}";
    private static boolean registered;
{chr(10).join(fields)}

    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
{chr(10).join(registrations)}
{command_registration}
        ItemGroupEvents.modifyEntriesEvent(ItemGroups.INGREDIENTS).register(entries -> {{
{chr(10).join(creative)}
        }});
    }}

    public static List<Block> machineBlocks() {{
        return {machine_list};
    }}

    private static Identifier id(String path) {{ return new Identifier(MOD_ID, path); }}
    private static Item item(String path, Item value) {{
        return Registry.register(Registries.ITEM, id(path), value);
    }}
    private static Block block(String path, Block value) {{
        Registry.register(Registries.BLOCK, id(path), value);
        Registry.register(Registries.ITEM, id(path), new BlockItem(value, new FabricItemSettings()));
        return value;
    }}

    private static final class GeneratedEffect extends StatusEffect {{
        private GeneratedEffect(int color) {{ super(StatusEffectCategory.BENEFICIAL, color); }}
    }}

    private static final class GeneratedEnchantment extends Enchantment {{
        private final int maxLevel;
        private GeneratedEnchantment(int maxLevel) {{
            super(Rarity.UNCOMMON, EnchantmentTarget.BREAKABLE, new EquipmentSlot[] {{EquipmentSlot.MAINHAND}});
            this.maxLevel = maxLevel;
        }}
        @Override public int getMaxLevel() {{ return maxLevel; }}
    }}
}}
'''


def _block_resources(mod_id: str, module_id: str, kind: str, config: dict[str, Any]) -> dict[str, str]:
    assets = f"src/main/resources/assets/{mod_id}"
    data = f"src/main/resources/data/{mod_id}"
    if kind == "crop":
        variants = {
            f"age={age}": {"model": f"{mod_id}:block/{module_id}_stage{min(age, 7)}"}
            for age in range(8)
        }
        files = {
            f"{assets}/blockstates/{module_id}.json": json.dumps({"variants": variants}, indent=2) + "\n",
            f"{assets}/models/item/{module_id}_seeds.json": json.dumps(
                {
                    "parent": "minecraft:item/generated",
                    "textures": {"layer0": f"{mod_id}:item/{module_id}_seeds"},
                },
                indent=2,
            )
            + "\n",
            f"{data}/loot_tables/blocks/{module_id}.json": json.dumps(
                {
                    "type": "minecraft:block",
                    "pools": [
                        {
                            "rolls": 1,
                            "entries": [
                                {"type": "minecraft:item", "name": f"{mod_id}:{module_id}_seeds"}
                            ],
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
        }
        for stage in range(8):
            files[f"{assets}/models/block/{module_id}_stage{stage}.json"] = json.dumps(
                {
                    "parent": "minecraft:block/crop",
                    "textures": {"crop": f"{mod_id}:block/{module_id}"},
                },
                indent=2,
            ) + "\n"
        return files
    return {
        f"{assets}/blockstates/{module_id}.json": json.dumps(
            {"variants": {"": {"model": f"{mod_id}:block/{module_id}"}}}, indent=2
        )
        + "\n",
        f"{assets}/models/block/{module_id}.json": json.dumps(
            {
                "parent": "minecraft:block/cube_all",
                "textures": {"all": f"{mod_id}:block/{module_id}"},
            },
            indent=2,
        )
        + "\n",
        f"{assets}/models/item/{module_id}.json": json.dumps(
            {"parent": f"{mod_id}:block/{module_id}"}, indent=2
        )
        + "\n",
        f"{data}/loot_tables/blocks/{module_id}.json": json.dumps(
            {
                "type": "minecraft:block",
                "pools": [
                    {
                        "rolls": 1,
                        "entries": [{"type": "minecraft:item", "name": f"{mod_id}:{module_id}"}],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    }


def _item_resources(mod_id: str, module_id: str, kind: str, config: dict[str, Any]) -> dict[str, str]:
    assets = f"src/main/resources/assets/{mod_id}"
    data = f"src/main/resources/data/{mod_id}"
    files = {
        f"{assets}/models/item/{module_id}.json": json.dumps(
            {
                "parent": "minecraft:item/handheld" if kind in {"tool", "weapon"} else "minecraft:item/generated",
                "textures": {"layer0": f"{mod_id}:item/{module_id}"},
            },
            indent=2,
        )
        + "\n"
    }
    ingredients = config.get("ingredients")
    if isinstance(ingredients, list) and ingredients:
        files[f"{data}/recipes/{module_id}.json"] = json.dumps(
            {
                "type": "minecraft:crafting_shapeless",
                "ingredients": [{"item": str(value)} for value in ingredients],
                "result": {"item": f"{mod_id}:{module_id}"},
            },
            indent=2,
        )
        + "\n"
    return files


def _data_only_resource(mod_id: str, module_id: str, kind: str, config: dict[str, Any]) -> dict[str, str]:
    if kind == "recipe":
        path = f"src/main/resources/data/{mod_id}/recipes/{module_id}.json"
    elif kind == "advancement":
        path = f"src/main/resources/data/{mod_id}/advancements/{module_id}.json"
    else:
        path = f"src/main/resources/data/{mod_id}/loot_tables/{module_id}.json"
    payload = config.get("json", config)
    return {path: json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"}


def _merge_lang(path: Path, additions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(current, dict):
        raise ExtendedContentError(f"Language file must be an object: {path}")
    current.update(additions)
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _constant(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value).upper()
    if not result or result[0].isdigit():
        result = "M_" + result
    return result


def _identifier(value: Any, module_id: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[a-z0-9_.-]+:[a-z0-9_./-]+", text):
        raise ExtendedContentError(f"Invalid namespaced item id for machine {module_id}: {text}")
    return text


def _java_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
