"""Host-written action argument codecs and command registration.

The command channel is Minecraft's existing client-to-server transport. Each
argument is one bounded base64url token; no client value becomes command syntax.
"""
from __future__ import annotations

import json

from .typed_action_contract import action_bindings


def _lit(value):
    return json.dumps(value, ensure_ascii=True)


def _validate_input(row, raw):
    name, kind = row["name"], row["type"]
    error = 'throw new IllegalArgumentException("invalid input: ' + name + '");'
    if kind in {"int", "long"}:
        cast = "int" if kind == "int" else "long"
        return [
            f"if (!({raw} instanceof Byte || {raw} instanceof Short || {raw} instanceof Integer || {raw} instanceof Long)) {error}",
            f"long number_{name} = ((Number) {raw}).longValue();",
            f"if (number_{name} < {row['minimum']}L || number_{name} > {row['maximum']}L) {error}",
            f"normalized.put({_lit(name)}, ({cast}) number_{name});",
        ]
    if kind == "double":
        return [
            f"if (!({raw} instanceof Number)) {error}",
            f"double number_{name} = ((Number) {raw}).doubleValue();",
            f"if (!Double.isFinite(number_{name}) || number_{name} < {float(row['minimum'])!r} || number_{name} > {float(row['maximum'])!r}) {error}",
            f"normalized.put({_lit(name)}, number_{name});",
        ]
    if kind == "boolean":
        return [f"if (!({raw} instanceof Boolean)) {error}",
                f"normalized.put({_lit(name)}, {raw});"]
    result = [f"if (!({raw} instanceof String text_{name}) || text_{name}.codePointCount(0, text_{name}.length()) > {row['max_length']}) {error}"]
    if row["choices"]:
        choices = ", ".join(_lit(c) for c in row["choices"])
        result.append(f"if (!java.util.List.of({choices}).contains(text_{name})) {error}")
    result.append(f"normalized.put({_lit(name)}, text_{name});")
    return result


def render_action_dispatch_java(plan, *, package):
    actions = action_bindings(plan)
    normalizers, invocations, commands, decoders = [], [], [], []
    for action in actions:
        identity = _lit(action["id"])
        normalizers.append(f"            case {identity}: {{")
        keys = ", ".join(_lit(r["name"]) for r in action["inputs"])
        normalizers.append(f"                if (!inputs.keySet().equals(java.util.Set.of({keys}))) throw new IllegalArgumentException(\"invalid action inputs\");")
        for index, row in enumerate(action["inputs"]):
            raw = "raw" + str(index)
            normalizers.append(f"                Object {raw} = inputs.get({_lit(row['name'])});")
            normalizers.extend("                " + line for line in _validate_input(row, raw))
        normalizers.extend(["                return normalized;", "            }"])
        invocations.extend([
            f"            case {identity}: {{",
            f"                values.put({_lit(action['actor_parameter'])}, actor);",
            f"                return ((Number) AuthoredProgram.invoke({_lit(action['function'])}, values)).intValue();",
            "            }",
        ])
        commands.extend([f"            case {identity}: {{",
                         f"                StringBuilder command = new StringBuilder({_lit(action['literal'])});"])
        for row in action["inputs"]:
            commands.append(f"                command.append(' ').append(encode(values.get({_lit(row['name'])}).toString()));")
        commands.extend(["                return command.toString();", "            }"])
        decoders.extend([f"            case {identity}: {{",
                         f"                if (tokens.size() != {len(action['inputs'])}) throw new IllegalArgumentException(\"invalid action arity\");"])
        for index, row in enumerate(action["inputs"]):
            raw = f"decode(tokens.get({index}))"
            value = {"int": f"Integer.valueOf({raw})", "long": f"Long.valueOf({raw})",
                     "double": f"Double.valueOf({raw})", "boolean": f"parseBoolean({raw})",
                     "string": raw}[row["type"]]
            decoders.append(f"                values.put({_lit(row['name'])}, {value});")
        decoders.extend([f"                return invoke({identity}, actor, values);", "            }"])
    return f'''package {package};

// MMM:TYPED_ACTION_DISPATCH_OWNER
public final class AuthoredActions {{
    private AuthoredActions() {{}}

    private static java.util.Map<String, Object> normalize(String action, java.util.Map<String, Object> inputs) {{
        if (action == null || inputs == null) throw new IllegalArgumentException("action and inputs required");
        var normalized = new java.util.LinkedHashMap<String, Object>();
        switch (action) {{
{chr(10).join(normalizers)}
            default: throw new IllegalArgumentException("unknown action");
        }}
    }}

    public static int invoke(String action, Object actor, java.util.Map<String, Object> inputs) {{
        if (actor == null) throw new IllegalArgumentException("server actor required");
        var values = normalize(action, inputs);
        switch (action) {{
{chr(10).join(invocations)}
            default: throw new IllegalArgumentException("unknown action");
        }}
    }}

    public static String command(String action, java.util.Map<String, Object> inputs) {{
        var values = normalize(action, inputs);
        switch (action) {{
{chr(10).join(commands)}
            default: throw new IllegalArgumentException("unknown action");
        }}
    }}

    public static int invokeEncoded(String action, Object actor, java.util.List<String> tokens) {{
        if (action == null || tokens == null) throw new IllegalArgumentException("action and tokens required");
        var values = new java.util.LinkedHashMap<String, Object>();
        switch (action) {{
{chr(10).join(decoders)}
            default: throw new IllegalArgumentException("unknown action");
        }}
    }}

    public static int submitEncoded(String action, Object actor, java.util.List<String> tokens) {{
        try {{ return invokeEncoded(action, actor, tokens); }}
        catch (IllegalArgumentException rejected) {{ return 0; }}
    }}

    private static boolean parseBoolean(String text) {{
        if (!text.equals("true") && !text.equals("false")) throw new IllegalArgumentException("invalid boolean");
        return text.equals("true");
    }}

    private static String encode(String text) {{
        return "x" + java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }}

    private static String decode(String token) {{
        if (token == null || token.length() > 24577 || !token.startsWith("x")) throw new IllegalArgumentException("invalid input encoding");
        try {{
            byte[] bytes = java.util.Base64.getUrlDecoder().decode(token.substring(1));
            return java.nio.charset.StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        }} catch (java.nio.charset.CharacterCodingException invalid) {{
            throw new IllegalArgumentException("invalid UTF-8", invalid);
        }}
    }}
}}
'''


def render_action_command_registration(binding, *, minecraft_version, indent="        "):
    config = binding["config"]
    action = config["action"]
    if minecraft_version.startswith("26."):
        commands = "net.minecraft.commands.Commands"
        permission = ("LEVEL_ALL", "LEVEL_MODERATORS", "LEVEL_MODERATORS",
                      "LEVEL_GAMEMASTERS", "LEVEL_OWNERS")[config.get("permission_level", 0)]
        requirement = f"{commands}.hasPermission({commands}.{permission})"
    else:
        commands = "net.minecraft.server.command.CommandManager"
        requirement = f"source -> source.hasPermissionLevel({config.get('permission_level', 0)})"
    values = ", ".join(f"com.mojang.brigadier.arguments.StringArgumentType.getString(context, {_lit(r['name'])})"
                       for r in action["inputs"])
    tail = (f".executes(context -> AuthoredActions.submitEncoded({_lit(action['id'])}, "
            f"context.getSource(), java.util.List.of({values})))")
    for row in reversed(action["inputs"]):
        tail = (f".then({commands}.argument({_lit(row['name'])}, "
                f"com.mojang.brigadier.arguments.StringArgumentType.word()){tail})")
    return [indent + "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback.EVENT."
            "register((dispatcher, registryAccess, environment) -> dispatcher.register("
            f"{commands}.literal({_lit(config['literal'])}).requires({requirement}){tail}));"]
