"""Host-owned typed access to canonical state; no guessed keys or casts."""
from __future__ import annotations

import json
from collections import Counter

from .java_region_parser import _parse, _text


def render_typed_state_accessors(variables) -> str:
    """Lower already validated state names/types/defaults through the same IR ABI."""
    from .structured_state_runtime import _default_value
    from .typed_plan_java import _java_type, _state_read

    members = []
    context_type = "java.util.Map<String, Object>"
    for variable in variables:
        # Match the canonical runtime renderer's existing normalization.
        name, kind = str(variable["name"]).strip(), str(variable["type"]).strip()
        java_type = _java_type(kind)
        literal = json.dumps(name, ensure_ascii=True)
        default = _default_value(variable)
        members.append(f"""public static synchronized {java_type} readState_{name}() {{
    return readState_{name}(java.util.Map.of());
}}

public static synchronized {java_type} readState_{name}({context_type} context) {{
    Object value = getState({literal}, context);
    if (value == null) value = {default};
    return {_state_read(kind, 'value')};
}}

public static synchronized void writeState_{name}({java_type} value) {{
    setState({literal}, value);
}}

public static synchronized void writeState_{name}({java_type} value, {context_type} context) {{
    setState({literal}, value, context);
}}""")
    return "\n\n".join(members)


def state_access_contract(prerequisite_api) -> list[dict]:
    """Take callable names from actual declarations, never fabricate accessors."""
    contracts = []
    for api in prerequisite_api:
        for declared_type in api.get("types", ()):
            owner = declared_type.get("owner", "")
            if owner.rsplit(".", 1)[-1] != "AuthoredStateModel":
                continue
            methods = []
            has_state_api = False
            for header in declared_type.get("members", ()):
                raw, root = _parse("class MmmStateApi {\n" + header + "\n}")
                body = root.named_children[0].child_by_field_name("body")
                for member in body.named_children:
                    if member.type != "method_declaration":
                        continue
                    name = _text(raw, member.child_by_field_name("name"))
                    has_state_api |= name in {"getState", "setState"}
                    if name.startswith(("readState_", "writeState_")):
                        methods.append({"name": name, "signature": header})
            if has_state_api:
                contracts.append({"owner": owner, "methods": methods})
    return contracts


def validate_host_state_access(source: str, contracts) -> None:
    """Admit only declared typed state access in model-authored members.

    The host's typed PlanIR and state runtime still own their generic state
    bridge. This gate applies only to the explicitly framed Java coder path.
    Comments and string contents cannot satisfy or violate a callable contract.
    """
    if not contracts:
        return
    simple_counts = Counter(row["owner"].rsplit(".", 1)[-1] for row in contracts)
    owners = {}
    for row in contracts:
        owners[row["owner"]] = row
        simple = row["owner"].rsplit(".", 1)[-1]
        if simple_counts[simple] == 1:
            owners[simple] = row
    raw, root = _parse("class MmmStateMember {\n" + source + "\n}")

    def qualified_name(node):
        if node.type in {"identifier", "type_identifier"}:
            return _text(raw, node)
        if node.type in {"field_access", "scoped_identifier", "scoped_type_identifier"}:
            parts = [qualified_name(child) for child in node.named_children
                     if child.type not in {"line_comment", "block_comment"}]
            return ".".join(parts) if parts and all(parts) else ""
        return ""

    def visit(node):
        if node.type in {"method_invocation", "method_reference"}:
            receiver = (node.named_children[0] if node.type == "method_reference"
                        else node.child_by_field_name("object"))
            name = (node.named_children[-1] if node.type == "method_reference"
                    else node.child_by_field_name("name"))
            row = owners.get(qualified_name(receiver)) if receiver is not None else None
            if row is not None and name is not None:
                called = _text(raw, name)
                allowed = {method["name"] for method in row["methods"]}
                code = ("JAVA_HOST_RAW_STATE_ACCESS" if called in {"getState", "setState"}
                        else "JAVA_HOST_STATE_ACCESS_UNKNOWN" if (
                            called.startswith(("readState_", "writeState_")) and called not in allowed
                        ) else "")
                if code:
                    raise ValueError(
                        code + ": " + _text(raw, node) + ". Use only declared typed state accessors: "
                        + json.dumps(row, ensure_ascii=False)
                        + ". Undeclared state requires a canonical state declaration; do not invent keys, "
                          "UUID lists, overloads or Object casts. Context overlays use Map<String, Object>."
                    )
        for child in node.named_children:
            visit(child)

    visit(root)
