from __future__ import annotations


def _gui_java(package_name: str, mod_id: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.inventory.SimpleInventory;
import net.minecraft.screen.GenericContainerScreenHandler;
import net.minecraft.screen.SimpleNamedScreenHandlerFactory;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.LinkedHashSet;
import java.util.Set;

public final class {class_name} {{
    public static final Identifier CHANNEL = new Identifier("{mod_id}", "system_action");
    private static final Set<String> ALLOWED_ACTIONS = new LinkedHashSet<>();
    private static boolean registered;
    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        ServerLifecycleEvents.SERVER_STARTED.register(server -> loadDefinitions());
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal("mmmgui").executes(context -> {{
                SimpleInventory inventory = new SimpleInventory(27);
                context.getSource().getPlayerOrThrow().openHandledScreen(
                    new SimpleNamedScreenHandlerFactory(
                        (syncId, playerInventory, player) -> GenericContainerScreenHandler.createGeneric9x3(syncId, playerInventory, inventory),
                        Text.literal("M.M.M")
                    )
                );
                return 1;
            }}))
        );
        ServerPlayNetworking.registerGlobalReceiver(CHANNEL, (server, player, handler, buffer, responseSender) -> {{
            String action = buffer.readString(128);
            server.execute(() -> {{
                if (!ALLOWED_ACTIONS.contains(action)) {{
                    player.networkHandler.disconnect(Text.literal("Rejected unapproved M.M.M action"));
                    return;
                }}
                player.sendMessage(Text.literal("Action accepted: " + action), true);
            }});
        }});
    }}

    private static void loadDefinitions() {{
        ALLOWED_ACTIONS.clear();
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            JsonObject config = module.getAsJsonObject("config");
            if (config.has("actions")) {{
                config.getAsJsonArray("actions").forEach(value -> ALLOWED_ACTIONS.add(value.getAsString()));
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
                        if (data.containsKey("owner|" + name) || data.containsKey("member|" + player.getUuidAsString())) return 0;
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
                        if (!(group instanceof String name) || !owner.getUuidAsString().equals(data.get("owner|" + name))) return 0;
                        data.put("invite|" + target.getUuidAsString(), name);
                        MmmPersistentStore.save(owner.getServer());
                        target.sendMessage(Text.literal("Invited to " + name), false);
                        return 1;
                    }})))
                .then(CommandManager.literal("accept").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                    Object invited = data.remove("invite|" + player.getUuidAsString());
                    if (!(invited instanceof String name) || !data.containsKey("owner|" + name)) return 0;
                    data.put("member|" + player.getUuidAsString(), name);
                    MmmPersistentStore.save(player.getServer());
                    return 1;
                }}))
                .then(CommandManager.literal("leave").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Map<String, Object> data = MmmPersistentStore.namespace(namespace);
                    Object group = data.get("member|" + player.getUuidAsString());
                    if (group instanceof String name && player.getUuidAsString().equals(data.get("owner|" + name))) return 0;
                    Object removed = data.remove("member|" + player.getUuidAsString());
                    MmmPersistentStore.save(player.getServer());
                    return removed == null ? 0 : 1;
                }}))
                .then(CommandManager.literal("status").executes(context -> {{
                    ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                    Object group = MmmPersistentStore.namespace(namespace).get("member|" + player.getUuidAsString());
                    context.getSource().sendFeedback(() -> Text.literal(group == null ? "No membership" : group.toString()), false);
                    return group == null ? 0 : 1;
                }}))
            )
        );
    }}
}}
'''
