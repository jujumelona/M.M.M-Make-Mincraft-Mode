from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .complete_spec import ProductionModule
from .generator import make_texture_png
from .project_edit import ensure_main_initializer_call, inspect_fabric_project, write_text_files


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


def generate_extended_content(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    modules: Iterable[ProductionModule],
) -> dict[str, Any]:
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ExtendedContentError("Extended content target does not match fabric.mod.json.")
    selected = tuple(module for module in modules if module.kind in _SUPPORTED)
    if not selected:
        return {"schema_version": "mmm/extended-content-v1", "status": "SKIPPED", "modules": []}
    for module in selected:
        module.validate()

    manifest_path = info.root / ".minecraft_ai/extended-modules.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {str(item["module_id"]): item for item in raw.get("modules", [])}
    for module in selected:
        existing[module.module_id] = {
            "module_id": module.module_id,
            "kind": module.kind,
            "config": module.config,
            "depends_on": list(module.depends_on),
            "required_gates": list(module.required_gates),
        }
    ordered = [existing[key] for key in sorted(existing)]
    manifest = {"schema_version": "mmm/extended-modules-v1", "modules": ordered}

    files: dict[str, str] = {
        ".minecraft_ai/extended-modules.json": json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        _java_path(package_name): _java(package_name, mod_id, ordered),
    }
    lang_en: dict[str, str] = {}
    lang_ko: dict[str, str] = {}
    generated_binary: list[str] = []
    for item in ordered:
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
            texture_path.write_bytes(make_texture_png(str(config.get("color", "#748cab")), module_id, kind="block", size=16))
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
        elif kind == "command":
            pass
        elif kind in {"recipe", "advancement", "loot"}:
            files.update(_data_only_resource(mod_id, module_id, kind, config))
        else:
            lang_en[f"item.{mod_id}.{module_id}"] = display_en
            lang_ko[f"item.{mod_id}.{module_id}"] = display_ko
            files.update(_item_resources(mod_id, module_id, kind, config))
            texture_path = info.root / f"src/main/resources/assets/{mod_id}/textures/item/{module_id}.png"
            texture_path.parent.mkdir(parents=True, exist_ok=True)
            texture_path.write_bytes(make_texture_png(str(config.get("color", "#74c7ec")), module_id, kind="item", size=16))
            generated_binary.append(str(texture_path))

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
        "schema_version": "mmm/extended-content-v1",
        "status": "GENERATED",
        "modules": [item["module_id"] for item in ordered],
        "files": [str(info.root / path) for path in files] + generated_binary,
        "source_receipt": receipt,
        "binding_receipt": binding,
        "required_gates": ["JDT", "Gradle", "GameTest", "runtime interaction tests"],
    }


def _java_path(package_name: str) -> str:
    return "src/main/java/" + package_name.replace(".", "/") + "/extended/GeneratedExtendedContent.java"


def _java(package_name: str, mod_id: str, modules: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    registrations: list[str] = []
    commands: list[str] = []
    nested: list[str] = []
    creative: list[str] = []
    for item in modules:
        module_id, kind, config = item["module_id"], item["kind"], item["config"]
        constant = module_id.upper()
        if kind == "item":
            fields.append(f"    public static Item {constant};")
            registrations.append(f'        {constant} = item("{module_id}", new Item(new FabricItemSettings()));')
            creative.append(f"            entries.add({constant});")
        elif kind == "food":
            hunger = int(config.get("hunger", 4))
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
            registrations.append(f'        {constant} = item("{module_id}", new SwordItem(ToolMaterials.IRON, {damage}, {speed:.3f}f, new FabricItemSettings()));')
            creative.append(f"            entries.add({constant});")
        elif kind == "tool":
            damage = int(config.get("attack_damage", 1))
            speed = float(config.get("attack_speed", -2.8))
            fields.append(f"    public static Item {constant};")
            registrations.append(f'        {constant} = item("{module_id}", new PickaxeItem(ToolMaterials.IRON, {damage}, {speed:.3f}f, new FabricItemSettings()));')
            creative.append(f"            entries.add({constant});")
        elif kind == "armor":
            armor_type = str(config.get("slot", "chestplate")).upper()
            if armor_type not in {"HELMET", "CHESTPLATE", "LEGGINGS", "BOOTS"}:
                armor_type = "CHESTPLATE"
            fields.append(f"    public static Item {constant};")
            registrations.append(f'        {constant} = item("{module_id}", new ArmorItem(ArmorMaterials.IRON, ArmorItem.Type.{armor_type}, new FabricItemSettings()));')
            creative.append(f"            entries.add({constant});")
        elif kind in {"block", "machine"}:
            fields.append(f"    public static Block {constant};")
            block_class = f"{_class_name(module_id)}MachineBlock" if kind == "machine" else "Block"
            constructor = f"new {block_class}(FabricBlockSettings.copyOf(Blocks.IRON_BLOCK).strength(3.5f))" if kind == "machine" else "new Block(FabricBlockSettings.copyOf(Blocks.STONE).strength(3.0f))"
            registrations.append(f'        {constant} = block("{module_id}", {constructor});')
            creative.append(f"            entries.add({constant});")
            if kind == "machine":
                nested.append(_machine_nested(module_id))
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
            registrations.append(f'        {constant} = Registry.register(Registries.STATUS_EFFECT, id("{module_id}"), new GeneratedEffect(0x{color:06X}));')
        elif kind == "enchantment":
            fields.append(f"    public static Enchantment {constant};")
            registrations.append(f'        {constant} = Registry.register(Registries.ENCHANTMENT, id("{module_id}"), new GeneratedEnchantment());')
        elif kind == "command":
            literal = re.sub(r"[^a-z0-9_]", "", str(config.get("literal", module_id))) or module_id
            message = str(config.get("message", module_id.replace("_", " "))).replace('"', '\\"')
            commands.append(
                f'        dispatcher.register(CommandManager.literal("{literal}").executes(context -> {{ context.getSource().sendFeedback(() -> Text.literal("{message}"), false); return 1; }}));'
            )

    command_registration = ""
    if commands:
        command_registration = '''
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
%s
        });''' % "\n".join(commands)
    return f'''package {package_name}.extended;

import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.item.v1.FabricItemSettings;
import net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents;
import net.fabricmc.fabric.api.object.builder.v1.block.FabricBlockSettings;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
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
import net.minecraft.item.ItemStack;
import net.minecraft.item.PickaxeItem;
import net.minecraft.item.SwordItem;
import net.minecraft.item.ToolMaterials;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.Identifier;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.entity.player.PlayerEntity;

public final class GeneratedExtendedContent {{
    private static final String MOD_ID = "{mod_id}";
    private static boolean registered;
{chr(10).join(fields)}

    private GeneratedExtendedContent() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
{chr(10).join(registrations)}
{command_registration}
        ItemGroupEvents.modifyEntriesEvent(ItemGroups.INGREDIENTS).register(entries -> {{
{chr(10).join(creative)}
        }});
    }}

    private static Identifier id(String path) {{ return new Identifier(MOD_ID, path); }}
    private static Item item(String path, Item value) {{ return Registry.register(Registries.ITEM, id(path), value); }}
    private static Block block(String path, Block value) {{
        Registry.register(Registries.BLOCK, id(path), value);
        Registry.register(Registries.ITEM, id(path), new BlockItem(value, new FabricItemSettings()));
        return value;
    }}

    private static final class GeneratedEffect extends StatusEffect {{
        private GeneratedEffect(int color) {{ super(StatusEffectCategory.BENEFICIAL, color); }}
    }}

    private static final class GeneratedEnchantment extends Enchantment {{
        private GeneratedEnchantment() {{ super(Rarity.UNCOMMON, EnchantmentTarget.BREAKABLE, new EquipmentSlot[] {{EquipmentSlot.MAINHAND}}); }}
        @Override public int getMaxLevel() {{ return 3; }}
    }}

{chr(10).join(nested)}
}}
'''


def _machine_nested(module_id: str) -> str:
    class_name = _class_name(module_id) + "MachineBlock"
    return f'''    private static final class {class_name} extends Block {{
        private {class_name}(Settings settings) {{ super(settings); }}
        @Override
        public ActionResult onUse(BlockState state, World world, BlockPos pos, PlayerEntity player, Hand hand, BlockHitResult hit) {{
            if (!world.isClient) player.sendMessage(Text.literal("Machine active: {module_id}"), false);
            return ActionResult.success(world.isClient);
        }}
    }}'''


def _block_resources(mod_id: str, module_id: str, kind: str, config: dict[str, Any]) -> dict[str, str]:
    assets = f"src/main/resources/assets/{mod_id}"
    data = f"src/main/resources/data/{mod_id}"
    files = {
        f"{assets}/blockstates/{module_id}.json": json.dumps({"variants": {"": {"model": f"{mod_id}:block/{module_id}"}}}, indent=2) + "\n",
        f"{assets}/models/block/{module_id}.json": json.dumps({"parent": "minecraft:block/cube_all", "textures": {"all": f"{mod_id}:block/{module_id}"}}, indent=2) + "\n",
        f"{assets}/models/item/{module_id}.json": json.dumps({"parent": f"{mod_id}:block/{module_id}"}, indent=2) + "\n",
        f"{data}/loot_tables/blocks/{module_id}.json": json.dumps({"type": "minecraft:block", "pools": [{"rolls": 1, "entries": [{"type": "minecraft:item", "name": f"{mod_id}:{module_id}"}]}]}, indent=2) + "\n",
    }
    if kind == "crop":
        files[f"{assets}/models/item/{module_id}_seeds.json"] = json.dumps({"parent": "minecraft:item/generated", "textures": {"layer0": f"{mod_id}:item/{module_id}_seeds"}}, indent=2) + "\n"
    return files


def _item_resources(mod_id: str, module_id: str, kind: str, config: dict[str, Any]) -> dict[str, str]:
    assets = f"src/main/resources/assets/{mod_id}"
    data = f"src/main/resources/data/{mod_id}"
    files = {
        f"{assets}/models/item/{module_id}.json": json.dumps({"parent": "minecraft:item/handheld" if kind in {"tool", "weapon"} else "minecraft:item/generated", "textures": {"layer0": f"{mod_id}:item/{module_id}"}}, indent=2) + "\n"
    }
    ingredients = config.get("ingredients")
    if isinstance(ingredients, list) and ingredients:
        files[f"{data}/recipes/{module_id}.json"] = json.dumps({"type": "minecraft:crafting_shapeless", "ingredients": [{"item": str(value)} for value in ingredients], "result": {"item": f"{mod_id}:{module_id}"}}, indent=2) + "\n"
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
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_name(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))
