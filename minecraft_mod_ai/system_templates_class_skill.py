from __future__ import annotations


def _class_skill_java(package_name: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.entity.effect.StatusEffect;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.registry.Registries;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

public final class {class_name} {{
    private record SkillDefinition(
        String id,
        String requiredClass,
        Identifier effect,
        int duration,
        int amplifier,
        int cooldown
    ) {{}}

    private static final Map<String, String> CLASSES = new LinkedHashMap<>();
    private static final Map<String, SkillDefinition> SKILLS = new LinkedHashMap<>();
    private static final Map<UUID, Map<String, Integer>> COOLDOWNS = new HashMap<>();
    private static boolean registered;
    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        MmmPersistentStore.registerLifecycle();
        ServerLifecycleEvents.SERVER_STARTED.register(server -> loadDefinitions());
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> COOLDOWNS.clear());
        ServerTickEvents.END_SERVER_TICK.register(server -> tickCooldowns());
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {{
            dispatcher.register(CommandManager.literal("mmmclass")
                .then(CommandManager.literal("list").executes(context -> {{
                    context.getSource().sendFeedback(
                        () -> Text.literal("Classes: " + String.join(", ", CLASSES.keySet())),
                        false
                    );
                    return CLASSES.size();
                }}))
                .then(CommandManager.literal("choose")
                    .then(CommandManager.argument("id", StringArgumentType.word()).executes(context -> {{
                        ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                        String id = StringArgumentType.getString(context, "id");
                        if (!CLASSES.containsKey(id)) return 0;
                        MmmPersistentStore.namespace("classes").put(player.getUuidAsString(), id);
                        COOLDOWNS.remove(player.getUuid());
                        MmmPersistentStore.save(player.getServer());
                        player.sendMessage(Text.literal("Class selected: " + id), false);
                        return 1;
                    }}))));
            dispatcher.register(CommandManager.literal("mmmskill")
                .then(CommandManager.literal("list").executes(context -> {{
                    context.getSource().sendFeedback(
                        () -> Text.literal("Skills: " + String.join(", ", SKILLS.keySet())),
                        false
                    );
                    return SKILLS.size();
                }}))
                .then(CommandManager.literal("cast")
                    .then(CommandManager.argument("id", StringArgumentType.word()).executes(context -> cast(
                        context.getSource().getPlayerOrThrow(),
                        StringArgumentType.getString(context, "id")
                    )))));
        }});
    }}

    private static void loadDefinitions() {{
        CLASSES.clear();
        SKILLS.clear();
        COOLDOWNS.clear();
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            String kind = module.get("kind").getAsString();
            String id = module.get("module_id").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            if ("class".equals(kind)) {{
                if (CLASSES.putIfAbsent(id, string(config, "display_name", id)) != null) {{
                    throw new IllegalStateException("Duplicate class: " + id);
                }}
            }}
        }});
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            if (!"skill".equals(module.get("kind").getAsString())) return;
            String id = module.get("module_id").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            String requiredClass = string(config, "required_class", "");
            if (!requiredClass.isBlank() && !CLASSES.containsKey(requiredClass)) {{
                throw new IllegalStateException(
                    "Skill " + id + " references unknown class " + requiredClass
                );
            }}
            Identifier effect = new Identifier(string(config, "effect", "minecraft:speed"));
            if (!Registries.STATUS_EFFECT.containsId(effect)) {{
                throw new IllegalStateException("Unknown skill effect: " + effect);
            }}
            SkillDefinition definition = new SkillDefinition(
                id,
                requiredClass,
                effect,
                integer(config, "duration_ticks", 100),
                integerZero(config, "amplifier", 0),
                integer(config, "cooldown_ticks", 100)
            );
            if (SKILLS.putIfAbsent(id, definition) != null) {{
                throw new IllegalStateException("Duplicate skill: " + id);
            }}
        }});
    }}

    private static int cast(ServerPlayerEntity player, String id) {{
        SkillDefinition skill = SKILLS.get(id);
        if (skill == null) return 0;
        Object selectedRaw = MmmPersistentStore.namespace("classes")
            .get(player.getUuidAsString());
        String selected = selectedRaw instanceof String text ? text : "";
        if (!skill.requiredClass().isBlank()
            && !skill.requiredClass().equals(selected)) return 0;
        Map<String, Integer> cooldown = COOLDOWNS.computeIfAbsent(
            player.getUuid(),
            ignored -> new HashMap<>()
        );
        if (cooldown.getOrDefault(id, 0) > 0) return 0;
        StatusEffect effect = Registries.STATUS_EFFECT.get(skill.effect());
        player.addStatusEffect(
            new StatusEffectInstance(effect, skill.duration(), skill.amplifier())
        );
        cooldown.put(id, skill.cooldown());
        return 1;
    }}

    private static void tickCooldowns() {{
        Iterator<Map.Entry<UUID, Map<String, Integer>>> players = COOLDOWNS.entrySet().iterator();
        while (players.hasNext()) {{
            Map<String, Integer> cooldowns = players.next().getValue();
            cooldowns.replaceAll((id, ticks) -> Math.max(0, ticks - 1));
            cooldowns.entrySet().removeIf(entry -> entry.getValue() <= 0);
            if (cooldowns.isEmpty()) players.remove();
        }}
    }}

    private static String string(JsonObject value, String key, String fallback) {{
        return value.has(key) ? value.get(key).getAsString() : fallback;
    }}

    private static int integer(JsonObject value, String key, int fallback) {{
        return value.has(key) ? Math.max(1, value.get(key).getAsInt()) : fallback;
    }}

    private static int integerZero(JsonObject value, String key, int fallback) {{
        return value.has(key) ? Math.max(0, value.get(key).getAsInt()) : fallback;
    }}
}}
'''
