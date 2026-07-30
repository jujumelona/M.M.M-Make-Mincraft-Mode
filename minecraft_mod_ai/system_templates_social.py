from __future__ import annotations


def _gui_java(package_name: str, mod_id: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.inventory.SimpleInventory;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.screen.GenericContainerScreenHandler;
import net.minecraft.screen.ScreenHandler;
import net.minecraft.screen.SimpleNamedScreenHandlerFactory;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class {class_name} {{
    public static final Identifier CHANNEL = new Identifier("{mod_id}", "system_action");

    private record MenuEntry(int slot, Identifier item, int count) {{}}
    private record MenuDefinition(String id, String title, int rows, List<MenuEntry> entries) {{}}
    private record ActionDefinition(
        String id,
        String type,
        String message,
        Identifier item,
        int count,
        Identifier effect,
        int durationTicks,
        int amplifier
    ) {{}}

    private static final Map<String, MenuDefinition> MENUS = new LinkedHashMap<>();
    private static final Map<String, ActionDefinition> ACTIONS = new LinkedHashMap<>();
    private static boolean registered;

    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        ServerLifecycleEvents.SERVER_STARTED.register(server -> loadDefinitions());
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal("mmmgui")
                .then(CommandManager.argument("id", com.mojang.brigadier.arguments.StringArgumentType.word())
                    .executes(context -> openMenu(
                        context.getSource().getPlayerOrThrow(),
                        com.mojang.brigadier.arguments.StringArgumentType.getString(context, "id")
                    )))
            )
        );
        ServerPlayNetworking.registerGlobalReceiver(CHANNEL, (server, player, handler, buffer, responseSender) -> {{
            String actionId = buffer.readString(128);
            server.execute(() -> executeAction(player, actionId));
        }});
    }}

    private static int openMenu(ServerPlayerEntity player, String id) {{
        MenuDefinition definition = MENUS.get(id);
        if (definition == null) return 0;
        SimpleInventory inventory = new SimpleInventory(definition.rows() * 9);
        for (MenuEntry entry : definition.entries()) {{
            if (!Registries.ITEM.containsId(entry.item())) continue;
            inventory.setStack(
                entry.slot(),
                new ItemStack(Registries.ITEM.get(entry.item()), entry.count())
            );
        }}
        player.openHandledScreen(
            new SimpleNamedScreenHandlerFactory(
                (syncId, playerInventory, ignored) -> createHandler(
                    definition.rows(),
                    syncId,
                    playerInventory,
                    inventory
                ),
                Text.literal(definition.title())
            )
        );
        return 1;
    }}

    private static ScreenHandler createHandler(
        int rows,
        int syncId,
        net.minecraft.entity.player.PlayerInventory playerInventory,
        SimpleInventory inventory
    ) {{
        return switch (rows) {{
            case 1 -> GenericContainerScreenHandler.createGeneric9x1(syncId, playerInventory, inventory);
            case 2 -> GenericContainerScreenHandler.createGeneric9x2(syncId, playerInventory, inventory);
            case 3 -> GenericContainerScreenHandler.createGeneric9x3(syncId, playerInventory, inventory);
            case 4 -> GenericContainerScreenHandler.createGeneric9x4(syncId, playerInventory, inventory);
            case 5 -> GenericContainerScreenHandler.createGeneric9x5(syncId, playerInventory, inventory);
            case 6 -> GenericContainerScreenHandler.createGeneric9x6(syncId, playerInventory, inventory);
            default -> throw new IllegalStateException("Unsupported menu rows: " + rows);
        }};
    }}

    private static void executeAction(ServerPlayerEntity player, String actionId) {{
        ActionDefinition definition = ACTIONS.get(actionId);
        if (definition == null) {{
            player.networkHandler.disconnect(Text.literal("Rejected unapproved M.M.M action"));
            return;
        }}
        switch (definition.type()) {{
            case "message" -> player.sendMessage(Text.literal(definition.message()), false);
            case "grant_item" -> {{
                if (!Registries.ITEM.containsId(definition.item())) return;
                player.giveItemStack(
                    new ItemStack(Registries.ITEM.get(definition.item()), definition.count())
                );
            }}
            case "status_effect" -> {{
                if (!Registries.STATUS_EFFECT.containsId(definition.effect())) return;
                player.addStatusEffect(
                    new StatusEffectInstance(
                        Registries.STATUS_EFFECT.get(definition.effect()),
                        definition.durationTicks(),
                        definition.amplifier()
                    )
                );
            }}
            default -> throw new IllegalStateException(
                "Unsupported validated action type: " + definition.type()
            );
        }}
    }}

    private static void loadDefinitions() {{
        MENUS.clear();
        ACTIONS.clear();
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            String kind = module.get("kind").getAsString();
            String moduleId = module.get("module_id").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            if ("gui".equals(kind)) {{
                int rows = config.get("rows").getAsInt();
                String title = config.get("title").getAsString();
                List<MenuEntry> entries = new ArrayList<>();
                config.getAsJsonArray("entries").forEach(rawEntry -> {{
                    JsonObject entry = rawEntry.getAsJsonObject();
                    entries.add(new MenuEntry(
                        entry.get("slot").getAsInt(),
                        new Identifier(entry.get("item").getAsString()),
                        entry.has("count") ? entry.get("count").getAsInt() : 1
                    ));
                }});
                MENUS.put(moduleId, new MenuDefinition(moduleId, title, rows, List.copyOf(entries)));
            }} else if ("networking".equals(kind)) {{
                config.getAsJsonArray("actions").forEach(rawAction -> {{
                    JsonObject action = rawAction.getAsJsonObject();
                    String id = action.get("id").getAsString();
                    String type = action.get("type").getAsString();
                    ACTIONS.put(id, new ActionDefinition(
                        id,
                        type,
                        action.has("message") ? action.get("message").getAsString() : "",
                        action.has("item") ? new Identifier(action.get("item").getAsString()) : null,
                        action.has("count") ? action.get("count").getAsInt() : 1,
                        action.has("effect") ? new Identifier(action.get("effect").getAsString()) : null,
                        action.has("duration_ticks") ? action.get("duration_ticks").getAsInt() : 100,
                        action.has("amplifier") ? action.get("amplifier").getAsInt() : 0
                    ));
                }});
            }}
        }});
    }}
}}
'''


def _party_java(package_name: str, class_name: str) -> str:
    return f'''package {package_name}.system;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.command.argument.EntityArgumentType;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;

import java.util.Map;

public final class {class_name} {{
    private static boolean registered;
    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        MmmPersistentStore.registerLifecycle();
        registerGroup("mmmparty", "parties");
        registerGroup("mmmguild", "guilds");
    }}

    private static void registerGroup(String command, String namespace) {{
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal(command)
                .then(CommandManager.literal("create")
                    .then(CommandManager.argument("name", StringArgumentType.word()).executes(context -> {{
                        ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                        String name = StringArgumentType.getString(context, "name");
                        Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                        if (data.containsKey("owner|" + name)
                            || data.containsKey("member|" + player.getUuidAsString())) return 0;
                        data.put("owner|" + name, player.getUuidAsString());
                        data.put("member|" + player.getUuidAsString(), name);
                        MmmPersistentStore.save(player.getServer());
                        return 1;
                    }})))
                .then(CommandManager.literal("invite")
                    .then(CommandManager.argument("player", EntityArgumentType.player()).executes(context -> {{
                        ServerPlayerEntity owner = context.getSource().getPlayerOrThrow();
                        ServerPlayerEntity target = EntityArgumentType.getPlayer(context, "player");
                        Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                        Object group = data.get("member|" + owner.getUuidAsString());
                        if (!(group instanceof String name)
                            || !owner.getUuidAsString().equals(data.get("owner|" + name))
                            || data.containsKey("member|" + target.getUuidAsString())) return 0;
                        data.put("invite|" + target.getUuidAsString(), name);
                        MmmPersistentStore.save(owner.getServer());
                        target.sendMessage(Text.literal("Invited to " + name), false);
                        return 1;
                    }})))
                .then(CommandManager.literal("accept").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                    if (data.containsKey("member|" + player.getUuidAsString())) return 0;
                    Object invited = data.remove("invite|" + player.getUuidAsString());
                    if (!(invited instanceof String name) || !data.containsKey("owner|" + name)) return 0;
                    data.put("member|" + player.getUuidAsString(), name);
                    MmmPersistentStore.save(player.getServer());
                    return 1;
                }}))
                .then(CommandManager.literal("kick")
                    .then(CommandManager.argument("player", EntityArgumentType.player()).executes(context -> {{
                        ServerPlayerEntity owner = context.getSource().getPlayerOrThrow();
                        ServerPlayerEntity target = EntityArgumentType.getPlayer(context, "player");
                        Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                        Object ownerGroup = data.get("member|" + owner.getUuidAsString());
                        Object targetGroup = data.get("member|" + target.getUuidAsString());
                        if (!(ownerGroup instanceof String name)
                            || !name.equals(targetGroup)
                            || !owner.getUuidAsString().equals(data.get("owner|" + name))
                            || owner.getUuid().equals(target.getUuid())) return 0;
                        data.remove("member|" + target.getUuidAsString());
                        data.remove("invite|" + target.getUuidAsString());
                        MmmPersistentStore.save(owner.getServer());
                        target.sendMessage(Text.literal("Removed from " + name), false);
                        return 1;
                    }})))
                .then(CommandManager.literal("leave").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                    Object group = data.get("member|" + player.getUuidAsString());
                    if (group instanceof String name
                        && player.getUuidAsString().equals(data.get("owner|" + name))) return 0;
                    Object removed = data.remove("member|" + player.getUuidAsString());
                    data.remove("invite|" + player.getUuidAsString());
                    MmmPersistentStore.save(player.getServer());
                    return removed == null ? 0 : 1;
                }}))
                .then(CommandManager.literal("disband").executes(context -> {{
                    ServerPlayerEntity owner = context.getSource().getPlayerOrThrow();
                    Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                    Object group = data.get("member|" + owner.getUuidAsString());
                    if (!(group instanceof String name)
                        || !owner.getUuidAsString().equals(data.get("owner|" + name))) return 0;
                    data.entrySet().removeIf(entry ->
                        (entry.getKey().startsWith("member|") || entry.getKey().startsWith("invite|"))
                            && name.equals(entry.getValue())
                    );
                    data.remove("owner|" + name);
                    MmmPersistentStore.save(owner.getServer());
                    return 1;
                }}))
                .then(CommandManager.literal("status").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Object group = MmmPersistentStore.namespace(namespace)
                        .get("member|" + player.getUuidAsString());
                    context.getSource().sendFeedback(
                        () -> Text.literal(group == null ? "No membership" : group.toString()),
                        false
                    );
                    return group == null ? 0 : 1;
                }}))
            )
        );
    }}
}}
'''
