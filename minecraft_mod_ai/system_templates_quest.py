from __future__ import annotations


def _quest_java(package_name: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

public final class {class_name} {{
    private record QuestDefinition(
        String id, String objective, String target, int required,
        String rewardItem, int rewardCount, double rewardCurrency
    ) {{}}
    private static final Map<String, QuestDefinition> DEFINITIONS = new LinkedHashMap<>();
    private static boolean registered;
    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        MmmPersistentStore.registerLifecycle();
        ServerLifecycleEvents.SERVER_STARTED.register(server -> loadDefinitions());
        ServerLivingEntityEvents.AFTER_DEATH.register((entity, source) -> {{
            if (source.getAttacker() instanceof ServerPlayerEntity player) {{
                progress(player, "kill", Registries.ENTITY_TYPE.getId(entity.getType()).toString(), 1);
            }}
        }});
        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, blockEntity) -> {{
            if (player instanceof ServerPlayerEntity serverPlayer) {{
                progress(serverPlayer, "break", Registries.BLOCK.getId(state.getBlock()).toString(), 1);
            }}
        }});
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal("mmmquest")
                .then(CommandManager.literal("list").executes(context -> {{
                    context.getSource().sendFeedback(
                        () -> Text.literal("Quests: " + String.join(", ", DEFINITIONS.keySet())),
                        false
                    );
                    return DEFINITIONS.size();
                }}))
                .then(CommandManager.literal("accept")
                    .then(CommandManager.argument("id", StringArgumentType.word()).executes(context -> {{
                        ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                        String id = StringArgumentType.getString(context, "id");
                        if (!DEFINITIONS.containsKey(id)) return 0;
                        Map<String, Object> data = MmmPersistentStore.namespace("quests");
                        if (Boolean.TRUE.equals(data.get(key(player.getUuid(), id, "active")))
                            || Boolean.TRUE.equals(data.get(key(player.getUuid(), id, "completed")))) return 0;
                        data.put(key(player.getUuid(), id, "active"), true);
                        data.put(key(player.getUuid(), id, "progress"), 0.0d);
                        MmmPersistentStore.save(player.getServer());
                        return 1;
                    }})))
                .then(CommandManager.literal("status")
                    .then(CommandManager.argument("id", StringArgumentType.word()).executes(context -> {{
                        ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                        String id = StringArgumentType.getString(context, "id");
                        QuestDefinition definition = DEFINITIONS.get(id);
                        if (definition == null) return 0;
                        Object raw = MmmPersistentStore.namespace("quests")
                            .getOrDefault(key(player.getUuid(), id, "progress"), 0.0d);
                        int value = raw instanceof Number number ? number.intValue() : 0;
                        context.getSource().sendFeedback(
                            () -> Text.literal(id + ": " + value + "/" + definition.required()),
                            false
                        );
                        return value;
                    }})))
                .then(CommandManager.literal("progress")
                    .requires(source -> source.hasPermissionLevel(2))
                    .then(CommandManager.argument("id", StringArgumentType.word())
                        .then(CommandManager.argument("amount", IntegerArgumentType.integer(1)).executes(context ->
                            progress(
                                context.getSource().getPlayerOrThrow(),
                                "manual",
                                StringArgumentType.getString(context, "id"),
                                IntegerArgumentType.getInteger(context, "amount")
                            )
                        ))))
            )
        );
    }}

    private static void loadDefinitions() {{
        DEFINITIONS.clear();
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            if (!"quest".equals(module.get("kind").getAsString())) return;
            String id = module.get("module_id").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            DEFINITIONS.put(id, new QuestDefinition(
                id,
                string(config, "objective", "manual"),
                string(config, "target", id),
                integer(config, "required", 1),
                string(config, "reward_item", ""),
                integer(config, "reward_count", 1),
                decimal(config, "reward_currency", 0.0d)
            ));
        }});
    }}

    public static int progress(
        ServerPlayerEntity player,
        String objective,
        String target,
        int amount
    ) {{
        int changed = 0;
        Map<String, Object> data = MmmPersistentStore.namespace("quests");
        for (QuestDefinition definition : DEFINITIONS.values()) {{
            if (!definition.objective().equals(objective)) continue;
            if ("manual".equals(objective)) {{
                if (!definition.id().equals(target)) continue;
            }} else if (!definition.target().equals(target)) {{
                continue;
            }}
            String activeKey = key(player.getUuid(), definition.id(), "active");
            if (!Boolean.TRUE.equals(data.get(activeKey))) continue;
            String progressKey = key(player.getUuid(), definition.id(), "progress");
            Object raw = data.getOrDefault(progressKey, 0.0d);
            int value = raw instanceof Number number ? number.intValue() : 0;
            value = Math.min(definition.required(), value + amount);
            data.put(progressKey, (double) value);
            changed++;
            if (value >= definition.required()) complete(player, definition, data);
        }}
        if (changed > 0) MmmPersistentStore.save(player.getServer());
        return changed;
    }}

    private static void complete(
        ServerPlayerEntity player,
        QuestDefinition definition,
        Map<String, Object> data
    ) {{
        data.put(key(player.getUuid(), definition.id(), "active"), false);
        data.put(key(player.getUuid(), definition.id(), "completed"), true);
        if (!definition.rewardItem().isBlank()) {{
            Identifier id = new Identifier(definition.rewardItem());
            if (Registries.ITEM.containsId(id)) {{
                player.giveItemStack(
                    new ItemStack(Registries.ITEM.get(id), definition.rewardCount())
                );
            }}
        }}
        if (definition.rewardCurrency() != 0.0d) {{
            Map<String, Object> economy = MmmPersistentStore.namespace("economy");
            String economyKey = player.getUuid().toString();
            Object raw = economy.getOrDefault(economyKey, 0.0d);
            double current = raw instanceof Number number ? number.doubleValue() : 0.0d;
            economy.put(economyKey, current + definition.rewardCurrency());
        }}
        player.sendMessage(Text.literal("Quest completed: " + definition.id()), false);
    }}

    private static String key(UUID player, String quest, String field) {{
        return player + "|" + quest + "|" + field;
    }}

    private static String string(JsonObject value, String key, String fallback) {{
        return value.has(key) ? value.get(key).getAsString() : fallback;
    }}

    private static int integer(JsonObject value, String key, int fallback) {{
        return value.has(key) ? Math.max(1, value.get(key).getAsInt()) : fallback;
    }}

    private static double decimal(JsonObject value, String key, double fallback) {{
        return value.has(key) ? value.get(key).getAsDouble() : fallback;
    }}
}}
'''
