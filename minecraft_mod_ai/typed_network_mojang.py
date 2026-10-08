from __future__ import annotations

"""Minecraft 26.1+ typed-state network renderer.

Changed 2026-10-08: 26.1 uses Mojang names and Fabric's registered,
object-based CustomPacketPayload transport; legacy PacketByteBufs +
Identifier channel callbacks are not valid on this target.
"""

import json
from collections.abc import Mapping


def render_mojang_network_policy_files(
    *,
    package_name: str,
    mod_id: str,
    field_types: Mapping[str, str],
    interval: int,
    max_bytes: int,
) -> dict[str, str]:
    if not field_types or interval < 1 or max_bytes < 1:
        raise ValueError("TYPED_MOJANG_NETWORK_CONFIG_INVALID")
    package_path = package_name.replace(".", "/")
    identifier = f"{mod_id}:typed_state_sync"
    write_lines = []
    guards = []
    for name, kind in field_types.items():
        lit = json.dumps(name, ensure_ascii=True)
        condition = {
            "int": "value instanceof Number",
            "long": "value instanceof Number",
            "double": "value instanceof Number",
            "boolean": "value instanceof Boolean",
            "string": "value instanceof String",
        }.get(kind)
        if condition is None:
            raise ValueError(f"TYPED_NETWORK_STATE_TYPE_INVALID: {name!r} -> {kind!r}")
        write_lines.extend([
            f"        value = AuthoredStateModel.getState({lit}, context);",
            f"        if (value != null && accepted({lit}, value)) {{",
            f"            payload.add({lit}, GSON.toJsonTree(value));",
            "        }",
        ])
        guards.append(f"            case {lit} -> {condition};")

    payload = f"""package {package_name};

import net.minecraft.network.RegistryFriendlyByteBuf;
import net.minecraft.network.codec.ByteBufCodecs;
import net.minecraft.network.codec.StreamCodec;
import net.minecraft.network.protocol.common.custom.CustomPacketPayload;
import net.minecraft.resources.Identifier;

// MMM:TYPED_NETWORK_PAYLOAD_OWNER
public record AuthoredStateSyncPayload(String json) implements CustomPacketPayload {{
    public static final Type<AuthoredStateSyncPayload> TYPE =
            new Type<>(Identifier.parse({json.dumps(identifier)}));
    public static final StreamCodec<RegistryFriendlyByteBuf, AuthoredStateSyncPayload> CODEC =
            StreamCodec.composite(ByteBufCodecs.STRING_UTF8,
                    AuthoredStateSyncPayload::json, AuthoredStateSyncPayload::new);

    @Override
    public Type<? extends CustomPacketPayload> type() {{
        return TYPE;
    }}
}}
"""

    server = f"""package {package_name};

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.networking.v1.PayloadTypeRegistry;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.server.level.ServerPlayer;
import java.nio.charset.StandardCharsets;
import java.util.Map;

// MMM:TYPED_NETWORK_SYNC_OWNER
public final class AuthoredNetworkSync {{
    private static final Gson GSON = new Gson();
    private static final int SYNC_INTERVAL_TICKS = {interval};
    private static final int MAX_PAYLOAD_BYTES = {max_bytes};
    private static boolean registered;
    private static int ticks;

    private AuthoredNetworkSync() {{}}

    public static synchronized void register() {{
        if (registered) return;
        PayloadTypeRegistry.clientboundPlay().register(
                AuthoredStateSyncPayload.TYPE, AuthoredStateSyncPayload.CODEC);
        ServerPlayConnectionEvents.JOIN.register(
                (handler, sender, server) -> sync(handler.player));
        ServerTickEvents.END_SERVER_TICK.register(server -> {{
            ticks++;
            if (ticks < SYNC_INTERVAL_TICKS) return;
            ticks = 0;
            for (ServerPlayer player : server.getPlayerList().getPlayers()) {{
                sync(player);
            }}
        }});
        registered = true;
    }}

    private static void sync(ServerPlayer player) {{
        JsonObject payload = new JsonObject();
        Map<String, Object> context = Map.of("player", player.getUUID().toString());
        Object value;
{chr(10).join(write_lines)}
        String encoded = GSON.toJson(payload);
        if (encoded.getBytes(StandardCharsets.UTF_8).length > MAX_PAYLOAD_BYTES) {{
            throw new IllegalStateException("Typed synchronization payload exceeds configured bound");
        }}
        ServerPlayNetworking.send(player, new AuthoredStateSyncPayload(encoded));
    }}

    private static boolean accepted(String field, Object value) {{
        return switch (field) {{
{chr(10).join(guards)}
            default -> false;
        }};
    }}
}}
"""
    client = f"""package {package_name};

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import java.nio.charset.StandardCharsets;

// MMM:TYPED_NETWORK_CLIENT_OWNER
public final class AuthoredNetworkClient implements ClientModInitializer {{
    private static final Gson GSON = new Gson();
    private static final int MAX_PAYLOAD_BYTES = {max_bytes};
    private static JsonObject snapshot = new JsonObject();

    @Override
    public void onInitializeClient() {{
        ClientPlayNetworking.registerGlobalReceiver(
                AuthoredStateSyncPayload.TYPE, (payload, context) -> {{
                    String encoded = payload.json();
                    if (encoded.getBytes(StandardCharsets.UTF_8).length > MAX_PAYLOAD_BYTES) {{
                        return;
                    }}
                    JsonObject parsed = JsonParser.parseString(encoded).getAsJsonObject();
                    context.client().execute(() -> update(parsed));
                }});
    }}

    private static synchronized void update(JsonObject value) {{
        snapshot = value.deepCopy();
    }}

    public static synchronized JsonObject snapshot() {{
        return snapshot.deepCopy();
    }}

    public static synchronized Object value(String key) {{
        if (!snapshot.has(key)) return null;
        return GSON.fromJson(snapshot.get(key), Object.class);
    }}
}}
"""
    return {
        f"src/main/java/{package_path}/AuthoredStateSyncPayload.java": payload,
        f"src/main/java/{package_path}/AuthoredNetworkSync.java": server,
        f"src/main/java/{package_path}/AuthoredNetworkClient.java": client,
    }


__all__ = ["render_mojang_network_policy_files"]
