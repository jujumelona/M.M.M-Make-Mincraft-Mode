from __future__ import annotations


def _party_java(package_name: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
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

        boolean partyEnabled = false;
        boolean guildEnabled = false;
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        for (JsonElement element : modules) {{
            JsonObject module = element.getAsJsonObject();
            String kind = module.get("kind").getAsString();
            if ("party".equals(kind)) partyEnabled = true;
            if ("guild".equals(kind)) guildEnabled = true;
        }}
        if (!partyEnabled && !guildEnabled) {{
            throw new IllegalStateException("Party-guild pack contains no supported definitions");
        }}
        if (partyEnabled) registerGroup("mmmparty", "parties");
        if (guildEnabled) registerGroup("mmmguild", "guilds");
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
