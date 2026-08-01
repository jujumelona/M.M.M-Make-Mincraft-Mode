from __future__ import annotations


def _gui_java(package_name: str, mod_id: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.inventory.Inventory;
import net.minecraft.inventory.SimpleInventory;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.screen.ScreenHandler;
import net.minecraft.screen.ScreenHandlerType;
import net.minecraft.screen.SimpleNamedScreenHandlerFactory;
import net.minecraft.screen.slot.Slot;
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
    private static final Map<String, Long> LAST_USE_TICK = new LinkedHashMap<>();
    private static final int NETWORK_COOLDOWN_TICKS = 10;
    private static boolean registered;

    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        MmmPersistentStore.registerLifecycle();
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
                (syncId, playerInventory, ignored) -> new ReadOnlyMenuHandler(
                    syncId,
                    playerInventory,
                    inventory,
                    definition.rows()
                ),
                Text.literal(definition.title())
            )
        );
        return 1;
    }}

    private static void executeAction(ServerPlayerEntity player, String actionId) {{
        ActionDefinition definition = ACTIONS.get(actionId);
        if (definition == null) {{
            player.networkHandler.disconnect(Text.literal("Rejected unapproved M.M.M action"));
            return;
        }}
        String playerAction = player.getUuidAsString() + "|" + actionId;
        long now = player.getServerWorld().getTime();
        long previous = LAST_USE_TICK.getOrDefault(playerAction, Long.MIN_VALUE / 2);
        if (now - previous < NETWORK_COOLDOWN_TICKS) return;
        LAST_USE_TICK.put(playerAction, now);

        Map<String, Object> usage = MmmPersistentStore.namespace("network_actions");
        boolean oneShot = "grant_item".equals(definition.type())
            || "status_effect".equals(definition.type());
        if (oneShot && Boolean.TRUE.equals(usage.get(playerAction))) return;

        switch (definition.type()) {{
            case "message" -> player.sendMessage(Text.literal(definition.message()), false);
            case "grant_item" -> {{
                if (!Registries.ITEM.containsId(definition.item())) return;
                player.giveItemStack(
                    new ItemStack(Registries.ITEM.get(definition.item()), definition.count())
                );
                usage.put(playerAction, true);
                MmmPersistentStore.save(player.getServer());
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
                usage.put(playerAction, true);
                MmmPersistentStore.save(player.getServer());
            }}
            default -> throw new IllegalStateException(
                "Unsupported validated action type: " + definition.type()
            );
        }}
    }}

    private static void loadDefinitions() {{
        MENUS.clear();
        ACTIONS.clear();
        LAST_USE_TICK.clear();
        MmmSystemConfig.forEachModule("{resource}", module -> {{
            String kind = module.get("kind").getAsString();
            String moduleId = module.get("module_id").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            if ("gui".equals(kind)) {{
                int rows = config.has("rows") ? config.get("rows").getAsInt() : 3;
                String title = config.has("title") ? config.get("title").getAsString() : "M.M.M";
                List<MenuEntry> entries = new ArrayList<>();
                JsonArray rawEntries = config.has("entries")
                    ? config.getAsJsonArray("entries")
                    : new JsonArray();
                rawEntries.forEach(rawEntry -> {{
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

    private static ScreenHandlerType<?> handlerType(int rows) {{
        return switch (rows) {{
            case 1 -> ScreenHandlerType.GENERIC_9X1;
            case 2 -> ScreenHandlerType.GENERIC_9X2;
            case 3 -> ScreenHandlerType.GENERIC_9X3;
            case 4 -> ScreenHandlerType.GENERIC_9X4;
            case 5 -> ScreenHandlerType.GENERIC_9X5;
            case 6 -> ScreenHandlerType.GENERIC_9X6;
            default -> throw new IllegalStateException("Unsupported menu rows: " + rows);
        }};
    }}

    private static final class ReadOnlyMenuHandler extends ScreenHandler {{
        private final Inventory inventory;
        private final int menuSlots;

        private ReadOnlyMenuHandler(
            int syncId,
            PlayerInventory playerInventory,
            Inventory inventory,
            int rows
        ) {{
            super(handlerType(rows), syncId);
            checkSize(inventory, rows * 9);
            this.inventory = inventory;
            this.menuSlots = rows * 9;
            inventory.onOpen(playerInventory.player);
            for (int row = 0; row < rows; row++) {{
                for (int column = 0; column < 9; column++) {{
                    int slot = column + row * 9;
                    addSlot(new Slot(inventory, slot, 8 + column * 18, 18 + row * 18) {{
                        @Override public boolean canInsert(ItemStack stack) {{ return false; }}
                        @Override public boolean canTakeItems(PlayerEntity player) {{ return false; }}
                    }});
                }}
            }}
            int playerY = 31 + rows * 18;
            for (int row = 0; row < 3; row++) {{
                for (int column = 0; column < 9; column++) {{
                    addSlot(new Slot(
                        playerInventory,
                        column + row * 9 + 9,
                        8 + column * 18,
                        playerY + row * 18
                    ));
                }}
            }}
            for (int column = 0; column < 9; column++) {{
                addSlot(new Slot(
                    playerInventory,
                    column,
                    8 + column * 18,
                    playerY + 58
                ));
            }}
        }}

        @Override
        public ItemStack quickMove(PlayerEntity player, int slot) {{
            return ItemStack.EMPTY;
        }}

        @Override
        public boolean canUse(PlayerEntity player) {{
            return inventory.canPlayerUse(player);
        }}

        @Override
        public void onClosed(PlayerEntity player) {{
            super.onClosed(player);
            inventory.onClose(player);
        }}
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
