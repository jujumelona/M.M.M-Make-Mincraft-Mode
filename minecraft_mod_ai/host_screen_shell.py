from __future__ import annotations

"""Host-owned Minecraft 26.1+ basic client-screen Java mold.

The host derives display strings from the approved semantic contract.
No model calls are needed for static screen descriptions. Imports, class shapes, source-set ownership, Fabric lifecycle wiring,
Screen callbacks, and client command registration come from this host module.

References: Fabric documentation for 26.1.2 CustomScreen.java and
ExampleModClientCommands.java. This implements GUI_EXISTS/screen registration,
NOT server-authoritative container/menu synchronization or gameplay actions.
"""

import hashlib
import re
from typing import Any


# Java identifiers and screen text are two separate ownership boundaries.
# For static UI copy, never insert user or model text as Java syntax.
def _host_display_literal(value: str, *, limit: int, fallback: str) -> str:
    """Build a bounded Java UTF-8 string literal from approved prose.

    The full requirement is retained by the production contract; only the
    one-line *visual label* is shortened to fit a 16px Minecraft sprite UI.
    Quotes, backslashes, braces (template markers), control, format and
    unpaired surrogates are excluded rather than rendering model syntax.
    """
    safe = "".join(
        ch if ch.isprintable() and ch not in {'"', "\\", "{", "}"} else " "
        for ch in str(value)
    )
    safe = " ".join(safe.split())
    if not safe:
        safe = fallback
    if len(safe) > limit:
        safe = safe[:limit - 1].rstrip() + "…"
    return '"' + safe + '"'

def screen_command_name(mod_id: str, subject: str) -> str:
    """Namespaced, short and collision-resistant command for basic UI access."""
    if not re.fullmatch(r"[a-z0-9_-]+", mod_id):
        raise ValueError("HOST_SCREEN_MOD_ID_INVALID")
    if not subject:
        raise ValueError("HOST_SCREEN_SUBJECT_REQUIRED")
    digest = hashlib.sha256((mod_id + ":" + subject).encode("utf-8")).hexdigest()[:16]
    return "mmm_" + digest


def basic_screen_candidate_contract(
    *,
    package_name: str,
    class_name: str,
    mod_id: str,
    subject: str,
    default_title: str,
    requirement: str = "",
    minecraft_version: str = "26.1.2",
) -> dict[str, Any]:
    """Return a complete deterministic Java mold with zero model slots."""
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*", package_name):
        raise ValueError("HOST_SCREEN_PACKAGE_INVALID")
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", class_name):
        raise ValueError("HOST_SCREEN_CLASS_INVALID")

    if not re.fullmatch(r"26\.(?:1|2)(?:\.[0-9]+)?", minecraft_version):
        raise ValueError("HOST_SCREEN_UNREVIEWED_MINECRAFT_EPOCH")
    cmd = screen_command_name(mod_id, subject)
    title_literal = _host_display_literal(
        default_title, limit=80, fallback=subject.replace("_", " "),
    )
    # Do not claim that currency balances, inventory widgets or game systems
    # work before their separate server/menu/interaction receipts are verified.
    description = "Planned: " + (requirement.strip() or default_title)
    body_literal = _host_display_literal(
        description, limit=140, fallback="Planned client screen",
    )
    # Fabric 26.2 moved screen control onto Minecraft.gui.
    open_screen = "client.gui.setScreen" if minecraft_version.startswith("26.2") else "client.setScreen"
    # Static copy is a host-reviewed quoted Java literal, not a model slot.
    mold = f"""package {package_name};

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

// MMM:HOST_26_SCREEN_SHELL
// Basic client GUI. Open with /{cmd}; gameplay/menu integration requires a
// separate verified server-side screen handler and network contract.
public final class {class_name} implements ClientModInitializer {{
    private static final String SCREEN_TITLE = {title_literal};
    private static final String SCREEN_BODY = {body_literal};

    @Override
    public void onInitializeClient() {{
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {{
            dispatcher.register(ClientCommands.literal("{cmd}").executes(context -> {{
                Minecraft client = Minecraft.getInstance();
                client.execute(() -> {open_screen}(
                        new GeneratedView(Component.literal(SCREEN_TITLE))));
                return 1;
            }}));
        }});
    }}

    private static final class GeneratedView extends Screen {{
        private GeneratedView(Component title) {{
            super(title);
        }}

        @Override
        protected void init() {{
            this.addRenderableWidget(
                    Button.builder(Component.literal("Close"), button -> this.onClose())
                            .bounds(this.width / 2 - 60, this.height / 2 + 25, 120, 20)
                            .build()
            );
        }}

        @Override
        public void extractRenderState(
                GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta
        ) {{
            super.extractRenderState(graphics, mouseX, mouseY, delta);
            graphics.text(this.font, SCREEN_TITLE,
                    this.width / 2 - 80, this.height / 2 - 35, 0xFFFFFFFF, true);
            graphics.text(this.font, SCREEN_BODY,
                    this.width / 2 - 80, this.height / 2 - 12, 0xFFFFFFFF, true);
        }}
    }}
}}
"""
    return {
        "render_mold": mold,
        "slots": [],
        "capability": "basic_client_screen",
        "command": cmd,
    }


__all__ = ["basic_screen_candidate_contract", "screen_command_name"]
