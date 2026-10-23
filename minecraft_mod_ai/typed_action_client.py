"""Render a client form from the exact server action ABI, without model Java."""

from __future__ import annotations

from .typed_action_contract import action_bindings
from .typed_action_java import _lit


def render_action_client_java(plan, *, package, mod_id, minecraft_version):
    if str(minecraft_version) != "26.2":
        raise ValueError(
            "ACTION_CLIENT_TARGET_UNSUPPORTED: requires checked Minecraft 26.2 GUI ABI"
        )
    actions = action_bindings(plan)
    definitions, names, labels = [], [], []
    for action in actions:
        names.append(_lit(action["id"]))
        labels.append(_lit(action["label"]))
        inputs = []
        for row in action["inputs"]:
            kind = row["type"]
            label = row["label"]
            if kind in {"int", "long", "double"}:
                label += f" ({row['minimum']} .. {row['maximum']})"
                initial = str(row["minimum"])
            elif kind == "boolean":
                initial = "false"
            else:
                initial = row["choices"][0] if row["choices"] else ""
            choices = row.get("choices") or (
                ["false", "true"] if kind == "boolean" else []
            )
            encoded_choices = ", ".join(_lit(c) for c in choices)
            inputs.append(
                f"new Input({_lit(row['name'])}, {_lit(label)}, {_lit(initial)}, "
                f"{row.get('max_length', 64)}, new String[]{{{encoded_choices}}})"
            )
        definitions.append(
            f"            case {_lit(action['id'])} -> new Input[]{{{', '.join(inputs)}}};"
        )
    return f"""package {package};

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.EditBox;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

// MMM:TYPED_ACTION_CLIENT_OWNER
public final class AuthoredActionsClient implements ClientModInitializer {{
    private static final String[] ACTIONS = {{{", ".join(names)}}};
    private static final String[] LABELS = {{{", ".join(labels)}}};
    private record Input(String name, String label, String initial, int limit, String[] choices) {{}}

    @Override
    public void onInitializeClient() {{
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) ->
            dispatcher.register(ClientCommands.literal({_lit(mod_id + "_actions")}).executes(context -> {{
                Minecraft.getInstance().execute(AuthoredActionsClient::openMenu);
                return 1;
            }})));
    }}

    public static void openMenu() {{
        var client = Minecraft.getInstance();
        client.gui.setScreen(new Menu(client.gui.screen()));
    }}

    public static void open(String action) {{
        int index = java.util.Arrays.asList(ACTIONS).indexOf(action);
        if (index < 0) throw new IllegalArgumentException("unknown action");
        var client = Minecraft.getInstance();
        client.gui.setScreen(new Form(action, LABELS[index], client.gui.screen()));
    }}

    private static Input[] inputs(String action) {{
        return switch (action) {{
{chr(10).join(definitions)}
            default -> throw new IllegalArgumentException("unknown action");
        }};
    }}

    private static final class Menu extends Screen {{
        private final Screen parent;
        private int page;
        private Menu(Screen parent) {{ super(Component.literal("Actions")); this.parent = parent; }}
        @Override protected void init() {{
            int count = Math.max(1, (height - 90) / 24);
            int start = page * count;
            for (int i = start; i < Math.min(ACTIONS.length, start + count); i++) {{
                final int index = i;
                addRenderableWidget(Button.builder(Component.literal(LABELS[i]), button ->
                    minecraft.gui.setScreen(new Form(ACTIONS[index], LABELS[index], this)))
                    .bounds(width / 2 - 120, 30 + (i - start) * 24, 240, 20).build());
            }}
            if (page > 0) addRenderableWidget(Button.builder(Component.literal("Previous"), b -> {{ page--; rebuildWidgets(); }})
                .bounds(width / 2 - 120, height - 48, 100, 20).build());
            if (start + count < ACTIONS.length) addRenderableWidget(Button.builder(Component.literal("Next"), b -> {{ page++; rebuildWidgets(); }})
                .bounds(width / 2 + 20, height - 48, 100, 20).build());
            addRenderableWidget(Button.builder(Component.literal("Close"), b -> onClose())
                .bounds(width / 2 - 50, height - 24, 100, 20).build());
        }}
        @Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {{
            super.extractRenderState(graphics, mouseX, mouseY, delta);
            graphics.text(font, title, width / 2 - 120, 12, 0xFFFFFFFF, true);
        }}
        @Override public void onClose() {{ minecraft.gui.setScreen(parent); }}
        @Override public boolean isPauseScreen() {{ return false; }}
    }}

    private static final class Form extends Screen {{
        private final String action;
        private final Screen parent;
        private final Input[] fields;
        private final java.util.Map<String, String> raw = new java.util.LinkedHashMap<>();
        private final java.util.Map<String, EditBox> editors = new java.util.LinkedHashMap<>();
        private int page;
        private String error = "";
        private Form(String action, String title, Screen parent) {{
            super(Component.literal(title)); this.action = action; this.parent = parent;
            this.fields = inputs(action);
            for (Input field : fields) raw.put(field.name(), field.initial());
        }}
        @Override protected void init() {{
            editors.clear();
            int count = Math.max(1, (height - 130) / 40);
            int start = page * count;
            for (int i = start; i < Math.min(fields.length, start + count); i++) {{
                Input field = fields[i];
                int y = 46 + (i - start) * 40;
                if (field.choices().length > 0) {{
                    addRenderableWidget(Button.builder(Component.literal(raw.get(field.name())), button -> {{
                        int selected = java.util.Arrays.asList(field.choices()).indexOf(raw.get(field.name()));
                        String next = field.choices()[(selected + 1) % field.choices().length];
                        raw.put(field.name(), next); button.setMessage(Component.literal(next));
                    }}).bounds(width / 2 - 120, y, 240, 20).build());
                }} else {{
                    EditBox editor = new EditBox(font, width / 2 - 120, y, 240, 20, Component.literal(field.label()));
                    editor.setMaxLength(field.limit() * 2);
                    editor.setValue(raw.get(field.name()));
                    editor.setResponder(value -> raw.put(field.name(), value));
                    editors.put(field.name(), editor);
                    addRenderableWidget(editor);
                }}
            }}
            if (page > 0) addRenderableWidget(Button.builder(Component.literal("Previous"), b -> {{ save(); page--; rebuildWidgets(); }})
                .bounds(width / 2 - 120, height - 72, 100, 20).build());
            if (start + count < fields.length) addRenderableWidget(Button.builder(Component.literal("Next"), b -> {{ save(); page++; rebuildWidgets(); }})
                .bounds(width / 2 + 20, height - 72, 100, 20).build());
            addRenderableWidget(Button.builder(Component.literal("Submit"), b -> submit())
                .bounds(width / 2 - 120, height - 46, 100, 20).build());
            addRenderableWidget(Button.builder(Component.literal("Back"), b -> onClose())
                .bounds(width / 2 + 20, height - 46, 100, 20).build());
        }}
        private void save() {{ for (var entry : editors.entrySet()) raw.put(entry.getKey(), entry.getValue().getValue()); }}
        private void submit() {{
            save();
            try {{
                String command = AuthoredActions.commandFromText(action, raw);
                var connection = minecraft.getConnection();
                if (connection == null) {{ error = "Not connected to a server"; return; }}
                connection.sendCommand(command);
                error = "Request sent";
            }} catch (IllegalArgumentException invalid) {{ error = "Check the input values and allowed ranges"; }}
        }}
        @Override public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float delta) {{
            super.extractRenderState(graphics, mouseX, mouseY, delta);
            graphics.text(font, title, width / 2 - 120, 12, 0xFFFFFFFF, true);
            int count = Math.max(1, (height - 130) / 40);
            int start = page * count;
            for (int i = start; i < Math.min(fields.length, start + count); i++)
                graphics.text(font, fields[i].label(), width / 2 - 120, 32 + (i - start) * 40, 0xFFFFFFFF, false);
            graphics.text(font, error, width / 2 - 120, height - 20, 0xFFFFFFFF, false);
        }}
        @Override public void onClose() {{ minecraft.gui.setScreen(parent); }}
        @Override public boolean isPauseScreen() {{ return false; }}
    }}
}}
"""
