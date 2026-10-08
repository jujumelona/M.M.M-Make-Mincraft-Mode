from __future__ import annotations

"""Host-owned Minecraft 26.1+ basic client-screen Java mold.

The model supplies only two bounded, non-executable display strings.
All imports, class shapes, source-set ownership, Fabric lifecycle wiring,
Screen callbacks, and client command registration come from this host module.

References: Fabric documentation for 26.1.2 CustomScreen.java and
ExampleModClientCommands.java. This implements GUI_EXISTS/screen registration,
NOT server-authoritative container/menu synchronization or gameplay actions.
"""

import hashlib
import json
import re
from typing import Any


# Exclude Java string delimiters, escapes and control characters; the model
# cannot inject Java statements or change class/interface structure.
_DISPLAY_PATTERN = r'^[^"\\\x00-\x1f\x7f]{1,96}$'
_BODY_PATTERN = r'^[^"\\\x00-\x1f\x7f]{1,160}$'


def _literal(value: str) -> str:
    # Host-owned strings from the planning graph still require Java escaping.
    # Use UTF-8 source text for non-ASCII characters; Java translation of
    # \uXXXX escapes before lexing would otherwise create injection surprises.
    return json.dumps(str(value), ensure_ascii=False)


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
    minecraft_version: str = "26.1.2",
) -> dict[str, Any]:
    """Return the deterministic Java mold and two atomic model-fillable slots."""
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*", package_name):
        raise ValueError("HOST_SCREEN_PACKAGE_INVALID")
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", class_name):
        raise ValueError("HOST_SCREEN_CLASS_INVALID")

    if not re.fullmatch(r"26\.(?:1|2)(?:\.[0-9]+)?", minecraft_version):
        raise ValueError("HOST_SCREEN_UNREVIEWED_MINECRAFT_EPOCH")
    cmd = screen_command_name(mod_id, subject)
    title_fallback = _literal(default_title)
    # Fabric 26.2 moved screen control onto Minecraft.gui.
    open_screen = "client.gui.setScreen" if minecraft_version.startswith("26.2") else "client.setScreen"
    # Host controls the Java shell. The two placeholders are Java string
    # *contents* only; their schemas forbid quote, slash and control injection.
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
    private static final String SCREEN_TITLE = "{{{{ui_title}}}}";
    private static final String SCREEN_BODY = "{{{{ui_body}}}}";
    private static final String HOST_FALLBACK_TITLE = {title_fallback};

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
        "slots": [
            {
                "name": "ui_title",
                "description": (
                    "Output only the short human-readable heading for this "
                    "Minecraft client screen. No Java, quotes or escapes. "
                    f"Screen subject: {subject}. Suggested title: {default_title}."
                ),
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 96,
                    "pattern": _DISPLAY_PATTERN,
                },
            },
            {
                "name": "ui_body",
                "description": (
                    "Output one short on-screen description of the user's "
                    "requested interface. No Java, quotes or escapes. "
                    "Do not claim gameplay operations, database integration "
                    "or network synchronization already exist. "
                    f"Screen subject: {subject}."
                ),
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 160,
                    "pattern": _BODY_PATTERN,
                },
            },
        ],
        "capability": "basic_client_screen",
        "command": cmd,
    }


__all__ = ["basic_screen_candidate_contract", "screen_command_name"]
