from __future__ import annotations


def _economy_java(package_name: str, class_name: str, resource: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

import java.util.LinkedHashMap;
import java.util.Map;

public final class {class_name} {{
    private record ShopEntry(String id, Identifier item, int count, double price) {{}}
    private static final Map<String, ShopEntry> CATALOG = new LinkedHashMap<>();
    private static double initialBalance;
    private static boolean registered;
    private {class_name}() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        MmmPersistentStore.registerLifecycle();
        ServerLifecycleEvents.SERVER_STARTED.register(server -> loadDefinitions());
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal("mmmeconomy")
                .then(CommandManager.literal("balance").executes(context -> {{
                    double value = balance(context.getSource().getPlayerOrThrow());
                    context.getSource().sendFeedback(() -> Text.literal("Balance: " + value), false);
                    return (int) Math.min(Integer.MAX_VALUE, value);
                }}))
                .then(CommandManager.literal("grant").requires(source -> source.hasPermissionLevel(2))
                    .then(CommandManager.argument("amount", DoubleArgumentType.doubleArg(0.0d)).executes(context -> {{
                        ServerPlayerEntity player = context.getSource().getPlayerOrThrow();
                        credit(player, DoubleArgumentType.getDouble(context, "amount"));
                        return 1;
                    }})))
            )
        );
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(CommandManager.literal("mmmshop")
                .then(CommandManager.literal("list").executes(context -> {{
                    context.getSource().sendFeedback(() -> Text.literal("Shop: " + String.join(", ", CATALOG.keySet())), false);
                    return CATALOG.size();
                }}))
                .then(CommandManager.literal("buy")
                    .then(CommandManager.argument("id", StringArgumentType.word()).executes(context ->
                        buy(context.getSource().getPlayerOrThrow(), StringArgumentType.getString(context, "id"))
                    )))
            )
        );
    }}

    private static void loadDefinitions() {{
        CATALOG.clear();
        initialBalance = 0.0d;
        JsonArray modules = MmmSystemConfig.load("{resource}").getAsJsonArray("modules");
        modules.forEach(element -> {{
            JsonObject module = element.getAsJsonObject();
            String kind = module.get("kind").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            if ("economy".equals(kind) && config.has("initial_balance")) initialBalance = config.get("initial_balance").getAsDouble();
            if (!"shop".equals(kind)) return;
            JsonArray entries = config.has("entries") ? config.getAsJsonArray("entries") : new JsonArray();
            entries.forEach(raw -> {{
                JsonObject entry = raw.getAsJsonObject();
                String id = entry.get("id").getAsString();
                Identifier item = new Identifier(entry.get("item").getAsString());
                if (!Registries.ITEM.containsId(item)) throw new IllegalStateException("Unknown shop item: " + item);
                CATALOG.put(id, new ShopEntry(
                    id,
                    item,
                    entry.has("count") ? Math.max(1, entry.get("count").getAsInt()) : 1,
                    Math.max(0.0d, entry.get("price").getAsDouble())
                ));
            }});
        }});
    }}

    private static double balance(ServerPlayerEntity player) {{
        Map<String, Object> data = MmmPersistentStore.namespace("economy");
        Object raw = data.computeIfAbsent(player.getUuidAsString(), ignored -> initialBalance);
        return raw instanceof Number number ? number.doubleValue() : initialBalance;
    }}

    public static void credit(ServerPlayerEntity player, double amount) {{
        MmmPersistentStore.namespace("economy").put(player.getUuidAsString(), balance(player) + amount);
        MmmPersistentStore.save(player.getServer());
    }}

    private static int buy(ServerPlayerEntity player, String id) {{
        ShopEntry entry = CATALOG.get(id);
        if (entry == null) return 0;
        double current = balance(player);
        if (current < entry.price()) return 0;
        MmmPersistentStore.namespace("economy").put(player.getUuidAsString(), current - entry.price());
        player.giveItemStack(new ItemStack(Registries.ITEM.get(entry.item()), entry.count()));
        MmmPersistentStore.save(player.getServer());
        return 1;
    }}
}}
'''
