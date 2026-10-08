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
        "parameter_constraints": [{}],
        "return_type": "string",
    },
    "player.send_message": {
        "owner": "AuthoredHostCapabilities",
        "method": "sendMessage",
        "parameters": ["object", "string"],
        "parameter_constraints": [{}, {"maxLength": 256, "description": "Message text shown to the player, including notifications such as insufficient currency."}],
        "return_type": "void",
    },
    "player.grant_item": {
        "gameplay_mutation": True,
        "owner": "AuthoredHostCapabilities",
        "method": "grantItem",
        "parameters": ["object", "string", "int"],
        "parameter_constraints": [
            {},
            {
                "description": "Registered item identifier in namespace:path form.",
                "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                "maxLength": 64,
            },
            {"minimum": 1, "maximum": 2**31 - 1, "description": "Number of items to grant; strictly positive."},
        ],
        "return_type": "boolean",
    },
    "player.add_status_effect": {
        "gameplay_mutation": True,
        "owner": "AuthoredHostCapabilities",
        "method": "addStatusEffect",
        "parameters": ["object", "string", "int", "int"],
        "parameter_constraints": [
            {},
            {
                "description": "Registered Minecraft status-effect identifier, such as minecraft:speed; not a message or an invented status label.",
                "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
                "maxLength": 64,
            },
            {"minimum": 1, "maximum": 2**31 - 1, "description": "Effect duration in ticks (20 ticks = 1 second); strictly positive."},
            {"minimum": 0, "maximum": 255, "description": "Zero-based effect amplifier; 0 means level I, 1 means level II."},
        ],
        "return_type": "boolean",
    },
    "command.has_permission": {
        "owner": "AuthoredHostCapabilities",
        "method": "hasPermission",
        "parameters": ["object", "int"],
        "parameter_constraints": [
            {},
            {"minimum": 0, "maximum": 4},
        ],
        "return_type": "boolean",
    },
}


def typed_host_capability_contracts() -> dict[str, dict[str, Any]]:
    return deepcopy(_CAPABILITIES)


def _render_unobfuscated_capabilities(package: str) -> str:
    """Minecraft 26.1+ (unobfuscated Mojang names, not Yarn aliases).

    Changed by OpenAI 2026-10-08: keep this target-specific Java renderer
    separate from the legacy Yarn renderer. All methods must typecheck.
    """
    return f"""package {package};

import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.core.registries.Registries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.effect.MobEffect;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;

// MMM:TYPED_HOST_CAPABILITIES_OWNER
public final class AuthoredHostCapabilities {{
    private AuthoredHostCapabilities() {{}}

    private static ServerPlayer player(Object value) {{
        if (value instanceof ServerPlayer player) return player;
        if (value instanceof CommandSourceStack source) {{
            try {{
                return source.getPlayerOrException();
            }} catch (com.mojang.brigadier.exceptions.CommandSyntaxException error) {{
                throw new IllegalArgumentException("Typed capability requires a player", error);
            }}
        }}
        throw new IllegalArgumentException("Typed capability requires a server player or command source");
    }}

    public static String playerUuid(Object value) {{
        return player(value).getUUID().toString();
    }}

    public static void sendMessage(Object value, String message) {{
        if (value instanceof CommandSourceStack source) {{
            source.sendSuccess(() -> Component.literal(message), false);
        }} else {{
            player(value).sendSystemMessage(Component.literal(message));
        }}
    }}

    public static boolean grantItem(Object value, String itemId, int count) {{
        if (count < 1) return false;
        Identifier id = Identifier.parse(itemId);
        if (!BuiltInRegistries.ITEM.containsKey(id)) return false;
        Item item = BuiltInRegistries.ITEM.getValue(id);
        return player(value).getInventory().add(new ItemStack(item, count));
    }}

    public static boolean addStatusEffect(Object value, String effectId, int durationTicks, int amplifier) {{
        if (durationTicks < 1 || amplifier < 0 || amplifier > 255) return false;
        Identifier id = Identifier.parse(effectId);
        Holder<MobEffect> effect = BuiltInRegistries.MOB_EFFECT.get(ResourceKey.create(Registries.MOB_EFFECT, id)).orElse(null);
        if (effect == null) return false;
        return player(value).addEffect(new MobEffectInstance(effect, durationTicks, amplifier));
    }}

    public static boolean hasPermission(Object value, int level) {{
        if (level < 0 || level > 4) return false;
        if (!(value instanceof CommandSourceStack source)) return false;
        return switch (level) {{
            case 0 -> Commands.LEVEL_ALL.check(source.permissions());
            case 1 -> Commands.LEVEL_MODERATORS.check(source.permissions());
            case 2 -> Commands.LEVEL_MODERATORS.check(source.permissions());
            case 3 -> Commands.LEVEL_GAMEMASTERS.check(source.permissions());
            case 4 -> Commands.LEVEL_OWNERS.check(source.permissions());
            default -> false;
        }};
    }}
}}
"""


def render_typed_host_capabilities_java(
    package_name: str, *, minecraft_version: str = ""
) -> str:
    package = str(package_name or "").strip()
    if not package:
        raise ValueError("TYPED_HOST_CAPABILITY_PACKAGE_REQUIRED")
    if str(minecraft_version).strip().startswith("26."):
        return _render_unobfuscated_capabilities(package)
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
            try {{
                return source.getPlayerOrThrow();
            }} catch (com.mojang.brigadier.exceptions.CommandSyntaxException exception) {{
                throw new IllegalArgumentException(
                        "Typed capability command source has no player",
                        exception
                );
            }}
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
