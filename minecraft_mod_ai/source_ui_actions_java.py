"""Render the fail-closed Fabric Event transport for source-authored UI bodies."""
from __future__ import annotations

import json

from .source_ui_actions import _source_ui_delegate_body, _source_ui_event_abi
from .typed_action_contract import action_bindings

SOURCE_UI_OWNER_MARKER = "// MMM:SOURCE_UI_ACTIONS_OWNER"


def _source_actions(plan):
    functions = {row["id"]: row for row in plan.get("functions", ())}
    return [
        action for action in action_bindings(plan)
        if action["actor_parameter"] == "source"
        and functions[action["function"]].get("body") == _source_ui_delegate_body(
            action["id"], action["inputs"])
    ]


def _event_declaration(abi):
    listener = abi["event_interface"].rsplit(".", 1)[-1]
    parameters = ", ".join(
        typ + " " + name for typ, name in zip(abi["parameters"], abi["parameter_names"])
    )
    arguments = ", ".join(abi["parameter_names"])
    return f'''    @FunctionalInterface
    public interface {listener} {{
        int onAction({parameters});
    }}

    public static final net.fabricmc.fabric.api.event.Event<{listener}> {abi["event_field"]} =
            net.fabricmc.fabric.api.event.EventFactory.createArrayBacked(
                    {listener}.class, listeners -> ({arguments}) -> {{
                        if (listeners.length != 1)
                            throw new IllegalStateException("source action requires exactly one listener");
                        return listeners[0].onAction({arguments});
                    }});
'''


def _input_value(row):
    boxed = {"boolean": "Boolean", "int": "Integer", "long": "Long",
             "double": "Double", "string": "String"}[row["type"]]
    return f"(({boxed}) inputs.get({json.dumps(row['name'])}))"


def _invocation_case(action, abi):
    arguments = ", ".join(["source", *(_input_value(row) for row in action["inputs"])])
    keys = ", ".join(json.dumps(row["name"]) for row in action["inputs"])
    return f'''            case {json.dumps(action["id"])}:
                if (!inputs.keySet().equals(java.util.Set.of({keys})))
                    throw new IllegalArgumentException("invalid source action inputs");
                return {abi["event_field"]}.invoker().onAction({arguments});'''


def render_source_ui_action_files(plan, *, package, minecraft_version) -> dict[str, str]:
    """Render only persisted exact host delegates, never guessed source effects."""
    actions = _source_actions(plan)
    if not actions:
        return {}
    declarations, invocations = [], []
    for action in actions:
        abi = _source_ui_event_abi(action["id"], action["inputs"], package)
        declarations.append(_event_declaration(abi))
        invocations.append(_invocation_case(action, abi))
    source = f'''package {package};

{SOURCE_UI_OWNER_MARKER}
public final class AuthoredSourceActions {{
    private AuthoredSourceActions() {{}}

{chr(10).join(declarations)}
    public static int invokeSourceAction(String action, Object actor, java.util.Map<String, Object> inputs) {{
        if (!(actor instanceof net.minecraft.commands.CommandSourceStack source))
            throw new IllegalArgumentException("server command source required");
        if (action == null || inputs == null)
            throw new IllegalArgumentException("action and inputs required");
        switch (action) {{
{chr(10).join(invocations)}
            default: throw new IllegalArgumentException("unknown source action");
        }}
    }}
}}
'''
    return {f"src/main/java/{package.replace('.', '/')}/AuthoredSourceActions.java": source}


def materialize_source_ui_action_files(plan, *, root, package, minecraft_version, assert_owned):
    """Apply the production backend's strict ownership check before replacement."""
    files = render_source_ui_action_files(plan, package=package, minecraft_version=minecraft_version)
    for path in files:
        assert_owned(root, path, marker=SOURCE_UI_OWNER_MARKER)
    return files
