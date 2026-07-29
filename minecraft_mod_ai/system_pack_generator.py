from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .project_edit import ensure_main_initializer_call, inspect_fabric_project, write_text_files


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_PACKS = frozenset(
    {
        "quest-system",
        "class-skill-system",
        "economy-shop",
        "gui-networking",
        "party-guild",
    }
)


def generate_system_pack(
    *,
    project_root: str | Path,
    pack_id: str,
    mod_id: str,
    package_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate real Fabric server/client bindings for one gameplay system.

    The generated code registers commands, server events, persistence and networking
    entry points as required by the selected system. Runtime and multiplayer tests
    remain mandatory release gates; they are not replaced by static generation.
    """

    if pack_id not in _PACKS:
        raise ValueError(f"Unknown system pack: {pack_id}")
    if not _ID.fullmatch(mod_id) or not _PACKAGE.fullmatch(package_name):
        raise ValueError("Invalid mod id or Java package.")
    if not isinstance(config, dict):
        raise ValueError("System pack config must be an object.")
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ValueError("System pack target does not match fabric.mod.json.")

    class_name = "".join(part.capitalize() for part in pack_id.split("-"))
    if not class_name.endswith("System"):
        class_name += "System"
    relative_java = (
        "src/main/java/"
        + package_name.replace(".", "/")
        + "/system/"
        + class_name
        + ".java"
    )
    shared_relative = (
        "src/main/java/"
        + package_name.replace(".", "/")
        + "/system/MmmPersistentStore.java"
    )
    contract_relative = f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}.json"
    contract = {
        "schema_version": f"mmm/{pack_id}-v2",
        "pack_id": pack_id,
        "config": config,
        "server_authoritative": True,
        "persistent": pack_id != "gui-networking",
        "minecraft_version": "1.20.1",
        "loader": "fabric",
    }
    files = {
        shared_relative: _persistent_store_java(package_name, mod_id),
        relative_java: _system_java(pack_id, package_name, mod_id, class_name, config),
        contract_relative: json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    }
    write_receipt = write_text_files(info, files)
    bind_receipt = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.system.{class_name}",
        call_line=f"{class_name}.register()",
        marker=f"system:{pack_id}",
    )
    return {
        "schema_version": "mmm/system-pack-generation-v2",
        "pack_id": pack_id,
        "files": [str(info.root / path) for path in files],
        "write_receipt": write_receipt,
        "binding_receipt": bind_receipt,
        "status": "fabric_binding_generated",
        "required_gates": [
            "JDT diagnostics",
            "Gradle clean build",
            "GameTest",
            "restart persistence test" if pack_id != "gui-networking" else "client GUI launch test",
            "multiplayer authority and replay test",
        ],
    }


def _persistent_store_java(package_name: str, mod_id: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.WorldSavePath;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

public final class MmmPersistentStore {{
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final Type TYPE = new TypeToken<Map<String, Map<String, Object>>>() {{}}.getType();
    private static final Map<String, Map<String, Object>> DATA = new LinkedHashMap<>();

    private MmmPersistentStore() {{}}

    public static synchronized Map<String, Object> namespace(String id) {{
        return DATA.computeIfAbsent(id, ignored -> new LinkedHashMap<>());
    }}

    public static synchronized void load(MinecraftServer server) {{
        Path file = dataFile(server);
        if (!Files.isRegularFile(file)) return;
        try (Reader reader = Files.newBufferedReader(file)) {{
            Map<String, Map<String, Object>> loaded = GSON.fromJson(reader, TYPE);
            DATA.clear();
            if (loaded != null) DATA.putAll(loaded);
        }} catch (IOException exception) {{
            throw new IllegalStateException("Could not load M.M.M persistent state", exception);
        }}
    }}

    public static synchronized void save(MinecraftServer server) {{
        Path file = dataFile(server);
        Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
        try {{
            Files.createDirectories(file.getParent());
            try (Writer writer = Files.newBufferedWriter(temporary)) {{
                GSON.toJson(DATA, TYPE, writer);
            }}
            try {{
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            }} catch (IOException atomicUnsupported) {{
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
            }}
        }} catch (IOException exception) {{
            throw new IllegalStateException("Could not save M.M.M persistent state", exception);
        }}
    }}

    private static Path dataFile(MinecraftServer server) {{
        return server.getSavePath(WorldSavePath.ROOT).resolve("data/{mod_id}_mmm_systems.json");
    }}
}}
'''


def _system_java(
    pack_id: str,
    package_name: str,
    mod_id: str,
    class_name: str,
    config: dict[str, Any],
) -> str:
    embedded = json.dumps(config, ensure_ascii=False, sort_keys=True).replace("\\", "\\\\").replace('"', '\\"')
    if pack_id == "quest-system":
        body = _quest_body()
        imports = _QUEST_IMPORTS
    elif pack_id == "class-skill-system":
        body = _class_skill_body()
        imports = _CLASS_IMPORTS
    elif pack_id == "economy-shop":
        body = _economy_body()
        imports = _ECONOMY_IMPORTS
    elif pack_id == "gui-networking":
        body = _gui_body(mod_id)
        imports = _GUI_IMPORTS
    else:
        body = _party_body()
        imports = _PARTY_IMPORTS
    return f'''package {package_name}.system;

{imports}

public final class {class_name} {{
    public static final String CONFIG_JSON = "{embedded}";
    private static boolean registered;

    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
{_indent(body, 8)}
    }}
}}
'''


_COMMON_IMPORTS = '''import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;

import java.util.Map;
import java.util.UUID;'''

_QUEST_IMPORTS = _COMMON_IMPORTS + '''
import java.util.ArrayList;
import java.util.List;'''

_CLASS_IMPORTS = _COMMON_IMPORTS + '''
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import java.util.HashMap;'''

_ECONOMY_IMPORTS = _COMMON_IMPORTS + '''
import com.mojang.brigadier.arguments.IntegerArgumentType;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;'''

_GUI_IMPORTS = '''import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.inventory.SimpleInventory;
import net.minecraft.network.PacketByteBuf;
import net.minecraft.screen.GenericContainerScreenHandler;
import net.minecraft.screen.SimpleNamedScreenHandlerFactory;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;'''

_PARTY_IMPORTS = _COMMON_IMPORTS + '''
import java.util.ArrayList;
import java.util.List;'''


def _lifecycle_lines(namespace: str) -> str:
    return f'''ServerLifecycleEvents.SERVER_STARTED.register(server -> MmmPersistentStore.load(server));
ServerLifecycleEvents.SERVER_STOPPING.register(server -> MmmPersistentStore.save(server));
MmmPersistentStore.namespace("{namespace}");'''


def _quest_body() -> str:
    return _lifecycle_lines("quests") + '''
CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
    dispatcher.register(CommandManager.literal("mmmquest")
        .then(CommandManager.literal("accept")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    UUID player = context.getSource().getPlayerOrThrow().getUuid();
                    String id = StringArgumentType.getString(context, "id");
                    Map<String, Object> data = MmmPersistentStore.namespace("quests");
                    @SuppressWarnings("unchecked")
                    List<String> active = (List<String>) data.computeIfAbsent(player.toString(), ignored -> new ArrayList<String>());
                    if (!active.contains(id)) active.add(id);
                    context.getSource().sendFeedback(() -> Text.literal("Quest accepted: " + id), false);
                    return 1;
                })))
        .then(CommandManager.literal("complete")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    UUID player = context.getSource().getPlayerOrThrow().getUuid();
                    String id = StringArgumentType.getString(context, "id");
                    Map<String, Object> data = MmmPersistentStore.namespace("quests");
                    Object raw = data.get(player.toString());
                    boolean removed = raw instanceof List<?> list && list.remove(id);
                    context.getSource().sendFeedback(() -> Text.literal(removed ? "Quest completed: " + id : "Quest is not active: " + id), false);
                    return removed ? 1 : 0;
                })))));'''


def _class_skill_body() -> str:
    return _lifecycle_lines("classes") + '''
Map<UUID, Integer> cooldowns = new HashMap<>();
ServerTickEvents.END_SERVER_TICK.register(server -> cooldowns.replaceAll((uuid, ticks) -> Math.max(0, ticks - 1)));
CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
    dispatcher.register(CommandManager.literal("mmmclass")
        .then(CommandManager.literal("choose")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    UUID player = context.getSource().getPlayerOrThrow().getUuid();
                    String id = StringArgumentType.getString(context, "id");
                    MmmPersistentStore.namespace("classes").put(player.toString(), id);
                    context.getSource().sendFeedback(() -> Text.literal("Class selected: " + id), false);
                    return 1;
                })))
        .then(CommandManager.literal("skill")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    UUID player = context.getSource().getPlayerOrThrow().getUuid();
                    if (cooldowns.getOrDefault(player, 0) > 0) return 0;
                    cooldowns.put(player, 100);
                    String id = StringArgumentType.getString(context, "id");
                    context.getSource().sendFeedback(() -> Text.literal("Skill activated: " + id), false);
                    return 1;
                })))));'''


def _economy_body() -> str:
    return _lifecycle_lines("economy") + '''
CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
    dispatcher.register(CommandManager.literal("mmmeconomy")
        .then(CommandManager.literal("balance")
            .executes(context -> {
                UUID player = context.getSource().getPlayerOrThrow().getUuid();
                Object raw = MmmPersistentStore.namespace("economy").getOrDefault(player.toString(), 0.0d);
                double balance = raw instanceof Number number ? number.doubleValue() : 0.0d;
                context.getSource().sendFeedback(() -> Text.literal("Balance: " + balance), false);
                return (int) Math.min(Integer.MAX_VALUE, balance);
            }))
        .then(CommandManager.literal("buy")
            .then(CommandManager.argument("item", StringArgumentType.word())
                .then(CommandManager.argument("price", IntegerArgumentType.integer(0))
                    .executes(context -> {
                        UUID playerId = context.getSource().getPlayerOrThrow().getUuid();
                        int price = IntegerArgumentType.getInteger(context, "price");
                        Object raw = MmmPersistentStore.namespace("economy").getOrDefault(playerId.toString(), 0.0d);
                        double balance = raw instanceof Number number ? number.doubleValue() : 0.0d;
                        if (balance < price) return 0;
                        Identifier id = new Identifier(StringArgumentType.getString(context, "item"));
                        if (!Registries.ITEM.containsId(id)) return 0;
                        MmmPersistentStore.namespace("economy").put(playerId.toString(), balance - price);
                        context.getSource().getPlayerOrThrow().giveItemStack(new ItemStack(Registries.ITEM.get(id)));
                        return 1;
                    }))))));'''


def _gui_body(mod_id: str) -> str:
    return f'''Identifier channel = new Identifier("{mod_id}", "gui_action");
ServerPlayNetworking.registerGlobalReceiver(channel, (server, player, handler, buffer, responseSender) -> {{
    String action = buffer.readString(64);
    server.execute(() -> {{
        if (!"open".equals(action)) return;
        player.openHandledScreen(new SimpleNamedScreenHandlerFactory(
            (syncId, inventory, ignored) -> GenericContainerScreenHandler.createGeneric9x1(syncId, inventory, new SimpleInventory(9)),
            Text.literal("M.M.M")
        ));
    }});
}});
CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
    dispatcher.register(CommandManager.literal("mmmgui")
        .then(CommandManager.literal("open")
            .executes(context -> {{
                context.getSource().getPlayerOrThrow().openHandledScreen(new SimpleNamedScreenHandlerFactory(
                    (syncId, inventory, ignored) -> GenericContainerScreenHandler.createGeneric9x1(syncId, inventory, new SimpleInventory(9)),
                    Text.literal("M.M.M")
                ));
                return 1;
            }}))));'''


def _party_body() -> str:
    return _lifecycle_lines("parties") + '''
Map<String, Object> parties = MmmPersistentStore.namespace("parties");
CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
    dispatcher.register(CommandManager.literal("mmmparty")
        .then(CommandManager.literal("create")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    String id = StringArgumentType.getString(context, "id");
                    UUID player = context.getSource().getPlayerOrThrow().getUuid();
                    @SuppressWarnings("unchecked")
                    List<String> members = (List<String>) parties.computeIfAbsent(id, ignored -> new ArrayList<String>());
                    if (!members.contains(player.toString())) members.add(player.toString());
                    context.getSource().sendFeedback(() -> Text.literal("Party created: " + id), false);
                    return 1;
                })))
        .then(CommandManager.literal("join")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    String id = StringArgumentType.getString(context, "id");
                    Object raw = parties.get(id);
                    if (!(raw instanceof List<?>)) return 0;
                    @SuppressWarnings("unchecked")
                    List<String> members = (List<String>) raw;
                    String player = context.getSource().getPlayerOrThrow().getUuid().toString();
                    if (!members.contains(player)) members.add(player);
                    context.getSource().sendFeedback(() -> Text.literal("Joined party: " + id), false);
                    return 1;
                })))
        .then(CommandManager.literal("leave")
            .then(CommandManager.argument("id", StringArgumentType.word())
                .executes(context -> {
                    String id = StringArgumentType.getString(context, "id");
                    Object raw = parties.get(id);
                    if (!(raw instanceof List<?>)) return 0;
                    @SuppressWarnings("unchecked")
                    List<String> members = (List<String>) raw;
                    boolean removed = members.remove(context.getSource().getPlayerOrThrow().getUuid().toString());
                    if (members.isEmpty()) parties.remove(id);
                    return removed ? 1 : 0;
                })))));'''


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def supported_system_packs() -> tuple[str, ...]:
    return tuple(sorted(_PACKS))
