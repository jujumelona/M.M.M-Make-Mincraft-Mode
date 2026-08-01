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
                        return credit(player, DoubleArgumentType.getDouble(context, "amount")) ? 1 : 0;
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
        MmmSystemConfig.forEachModule("{resource}", module -> {{
            String kind = module.get("kind").getAsString();
            JsonObject config = module.getAsJsonObject("config");
            if ("economy".equals(kind) && config.has("initial_balance")) {{
                initialBalance = config.get("initial_balance").getAsDouble();
                if (!Double.isFinite(initialBalance) || initialBalance < 0.0d) {{
                    throw new IllegalStateException("Invalid initial economy balance");
                }}
            }}
            if (!"shop".equals(kind)) return;
            JsonArray entries = config.getAsJsonArray("entries");
            entries.forEach(raw -> {{
                JsonObject entry = raw.getAsJsonObject();
                String id = entry.get("id").getAsString();
                Identifier item = new Identifier(entry.get("item").getAsString());
                if (!Registries.ITEM.containsId(item)) {{
                    throw new IllegalStateException("Unknown shop item: " + item);
                }}
                int count = entry.has("count") ? entry.get("count").getAsInt() : 1;
                double price = entry.get("price").getAsDouble();
                if (count < 1 || !Double.isFinite(price) || price < 0.0d) {{
                    throw new IllegalStateException("Invalid shop entry: " + id);
                }}
                if (CATALOG.putIfAbsent(id, new ShopEntry(id, item, count, price)) != null) {{
                    throw new IllegalStateException("Duplicate shop entry: " + id);
                }}
            }});
        }});
    }}

    private static double balance(ServerPlayerEntity player) {{
        Map<String, Object> data = MmmPersistentStore.namespace("economy");
        Object raw = data.computeIfAbsent(player.getUuidAsString(), ignored -> initialBalance);
        double value = raw instanceof Number number ? number.doubleValue() : initialBalance;
        if (!Double.isFinite(value) || value < 0.0d) {{
            data.put(player.getUuidAsString(), initialBalance);
            return initialBalance;
        }}
        return value;
    }}

    public static boolean credit(ServerPlayerEntity player, double amount) {{
        if (!Double.isFinite(amount) || amount < 0.0d) return false;
        double next = balance(player) + amount;
        if (!Double.isFinite(next)) return false;
        MmmPersistentStore.namespace("economy").put(player.getUuidAsString(), next);
        MmmPersistentStore.save(player.getServer());
        return true;
    }}

    private static int buy(ServerPlayerEntity player, String id) {{
        ShopEntry entry = CATALOG.get(id);
        if (entry == null) return 0;
        double current = balance(player);
        if (current < entry.price()) return 0;

        ItemStack delivery = new ItemStack(
            Registries.ITEM.get(entry.item()),
            entry.count()
        );
        double next = current - entry.price();
        MmmPersistentStore.namespace("economy").put(player.getUuidAsString(), next);
        MmmPersistentStore.save(player.getServer());

        if (!player.giveItemStack(delivery) && !delivery.isEmpty()) {{
            player.dropItem(delivery, false);
        }}
        return 1;
    }}
}}
'''
