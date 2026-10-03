from __future__ import annotations

"""Host-owned Minecraft capability bindings for Typed PlanIR.

The planner sees only capability IDs and scalar signatures. Java owners and API call
topology are fixed here and rendered by the host; production never asks a model to
invent Minecraft symbols.
"""

from copy import deepcopy
from typing import Any


_CAPABILITIES: dict[str, dict[str, Any]] = {
    "player.uuid": {
        "owner": "AuthoredHostCapabilities",
        "method": "playerUuid",
        "parameters": ["object"],
        "return_type": "string",
    },
    "player.send_message": {
        "owner": "AuthoredHostCapabilities",
        "method": "sendMessage",
        "parameters": ["object", "string"],
        "return_type": "void",
    },
    "player.grant_item": {
        "owner": "AuthoredHostCapabilities",
        "method": "grantItem",
        "parameters": ["object", "string", "int"],
        "return_type": "boolean",
    },
    "player.add_status_effect": {
        "owner": "AuthoredHostCapabilities",
        "method": "addStatusEffect",
        "parameters": ["object", "string", "int", "int"],
        "return_type": "boolean",
    },
    "command.has_permission": {
        "owner": "AuthoredHostCapabilities",
        "method": "hasPermission",
        "parameters": ["object", "int"],
        "return_type": "boolean",
    },
}


def typed_host_capability_contracts() -> dict[str, dict[str, Any]]:
    return deepcopy(_CAPABILITIES)


def render_typed_host_capabilities_java(package_name: str) -> str:
    package = str(package_name or "").strip()
    if not package:
        raise ValueError("TYPED_HOST_CAPABILITY_PACKAGE_REQUIRED")
    return f"""package {package};

import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;

// MMM:TYPED_HOST_CAPABILITIES_OWNER
public final class AuthoredHostCapabilities {{
    private AuthoredHostCapabilities() {{}}

    private static ServerPlayerEntity player(Object value) {{
        if (value instanceof ServerPlayerEntity player) {{
            return player;
        }}
        if (value instanceof ServerCommandSource source) {{
            return source.getPlayerOrThrow();
        }}
        throw new IllegalArgumentException(
                "Typed capability requires a server player or command source"
        );
    }}

    public static String playerUuid(Object value) {{
        return player(value).getUuidAsString();
    }}

    public static void sendMessage(Object value, String message) {{
        if (value instanceof ServerCommandSource source) {{
            source.sendFeedback(() -> Text.literal(message), false);
            return;
        }}
        player(value).sendMessage(Text.literal(message), false);
    }}

    public static boolean grantItem(Object value, String itemId, int count) {{
        if (count < 1) return false;
        Identifier id = new Identifier(itemId);
        if (!Registries.ITEM.containsId(id)) return false;
        return player(value).giveItemStack(
                new ItemStack(Registries.ITEM.get(id), count)
        );
    }}

    public static boolean addStatusEffect(
            Object value,
            String effectId,
            int durationTicks,
            int amplifier
    ) {{
        if (durationTicks < 1 || amplifier < 0 || amplifier > 255) return false;
        Identifier id = new Identifier(effectId);
        if (!Registries.STATUS_EFFECT.containsId(id)) return false;
        return player(value).addStatusEffect(
                new StatusEffectInstance(
                        Registries.STATUS_EFFECT.get(id),
                        durationTicks,
                        amplifier
                )
        );
    }}

    public static boolean hasPermission(Object value, int level) {{
        if (level < 0 || level > 4) return false;
        if (value instanceof ServerCommandSource source) {{
            return source.hasPermissionLevel(level);
        }}
        return false;
    }}
}}
"""


__all__ = [
    "render_typed_host_capabilities_java",
    "typed_host_capability_contracts",
]
