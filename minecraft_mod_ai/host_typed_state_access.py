"""Host-owned typed access to canonical state; no guessed keys or casts."""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from .java_region_parser import _parse, _text

_HOST_STATE_INTERNALS = frozenset({"getState", "setState", "exportState", "importState", "clearScopedState", "defaultStateSnapshot"})


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
        # Emit a host-owned ABI marker for scoped values. Prerequisite API
        # extraction retains declarations (not bodies), so this zero-arg
        # marker carries the verified scope into later callback compilation.
        if str(variable.get("owner") or "").strip() == "player":
            members.append(f"public static boolean mmmPlayerScoped_{name}() {{ return true; }}")
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
            player_scoped = set()
            has_state_api = False
            for header in declared_type.get("members", ()):
                raw, root = _parse("class MmmStateApi {\n" + header + "\n}")
                body = root.named_children[0].child_by_field_name("body")
                for member in body.named_children:
                    if member.type != "method_declaration":
                        continue
                    name = _text(raw, member.child_by_field_name("name"))
                    has_state_api |= name in {"getState", "setState"}
                    if name.startswith("mmmPlayerScoped_"):
                        player_scoped.add(name[len("mmmPlayerScoped_"):])
                    if name.startswith(("readState_", "writeState_")):
                        methods.append({"name": name, "signature": header})
            if has_state_api:
                declared_names = {
                    row["name"].split("_", 1)[1] for row in methods
                }
                if not player_scoped <= declared_names:
                    raise ValueError("STATE_ACCESS_SCOPED_MARKER_ORPHAN")
                contracts.append({
                    "owner": owner, "methods": methods,
                    "state_scopes": {name: "player" for name in sorted(player_scoped)},
                })
    return contracts



def verified_state_accessor_symbols(contracts) -> dict:
    """Index only public static accessors parsed from materialized Java source.

    The target classpath snapshot predates generated prerequisites. Therefore
    a method can be compile-valid in a prerequisite but absent from the
    872k-symbol target inventory. Never infer a method from a state name:
    this bridge accepts only previously extracted declaration headers.
    """
    result = {}
    common = {"String": "java.lang.String", "Object": "java.lang.Object",
              "Boolean": "java.lang.Boolean", "Integer": "java.lang.Integer",
              "Long": "java.lang.Long", "Double": "java.lang.Double"}

    def typename(source):
        source = source.strip()
        # The checked bytecode ABI uses erased parameter types. The declared
        # Java source retains generic arguments; erase only that type wrapper.
        depth, chars = 0, []
        for char in source:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif depth == 0:
                chars.append(char)
        erased = "".join(chars).strip().replace("...", "[]")
        suffix = ""
        while erased.endswith("[]"):
            erased, suffix = erased[:-2], "[]" + suffix
        return common.get(erased, erased) + suffix

    for contract in contracts:
        owner = contract.get("owner")
        if not isinstance(owner, str) or not owner:
            continue
        for declaration in contract.get("methods", ()):
            header = declaration.get("signature", "")
            if not isinstance(header, str):
                continue
            raw, root = _parse("class MmmCheckedSource {\n" + header + "\n}")
            owner_node = next((node for node in root.named_children
                               if node.type == "class_declaration"), None)
            if owner_node is None:
                continue
            methods = [node for node in owner_node.child_by_field_name("body").named_children
                       if node.type == "method_declaration"]
            if len(methods) != 1:
                continue
            method = methods[0]
            name_node = method.child_by_field_name("name")
            type_node = method.child_by_field_name("type")
            if name_node is None or type_node is None:
                continue
            method_name = _text(raw, name_node)
            if (method_name != declaration.get("name")
                    or not method_name.startswith(("readState_", "writeState_"))):
                continue
            modifiers = next((node for node in method.named_children
                              if node.type == "modifiers"), None)
            keywords = _text(raw, modifiers).split() if modifiers is not None else []
            if "public" not in keywords or "static" not in keywords:
                continue
            parameters = []
            for argument in method.child_by_field_name("parameters").named_children:
                if argument.type != "formal_parameter":
                    continue
                declared = argument.child_by_field_name("type")
                if declared is None:
                    break
                parameters.append(typename(_text(raw, declared)))
            else:
                result[owner] = {"kind": "CLASS", "owner": owner, "access": 1}
                key = "materialized:" + owner + "#" + method_name + "(" + ",".join(parameters) + ")"
                result[key] = {
                    "kind": "METHOD", "owner": owner, "name": method_name,
                    "access": 9, "static": True, "parameters": parameters,
                    "return_type": typename(_text(raw, type_node)),
                    "provenance": "materialized_prerequisite_declaration",
                }
    return result


def state_work_order_scope(work_order, contracts):
    """Expose planning-to-Java state coverage without inventing new storage.

    A feature work order can name logical state that is not declared in the
    immutable state model. Those names are NOT callable Java accessors. The
    coder must use verified world/event state when appropriate, or the work
    order must be revised upstream; it may never write into an unrelated key.
    This is provenance, not an inferred state-variable type or default.
    """
    import re

    read_names, write_names = set(), set()
    for row in contracts:
        for method in row.get("methods", ()):
            name = method.get("name", "") if isinstance(method, dict) else ""
            if name.startswith("readState_"):
                read_names.add(name[len("readState_"):])
            elif name.startswith("writeState_"):
                write_names.add(name[len("writeState_"):])
    plan = work_order if isinstance(work_order, dict) else {}
    identifier = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
    referenced = {
        kind: sorted({name for raw in plan.get(kind, ())
                      if isinstance(raw, str)
                      and (name := raw.strip())
                      and identifier.fullmatch(name)})
        for kind in ("state_reads", "state_writes")
    }
    return {
        "declared_readable": sorted(read_names),
        "declared_writable": sorted(write_names),
        "undeclared_plan_reads": [
            name for name in referenced["state_reads"] if name not in read_names
        ],
        "undeclared_plan_writes": [
            name for name in referenced["state_writes"] if name not in write_names
        ],
        "contract": (
            "Only declared accessors are callable. An undeclared work-order "
            "state name is not an invitation to invent an accessor or "
            "overwrite an unrelated state key. Resolve persistent state "
            "through the canonical state model; use checked event/world "
            "inputs when behavior explicitly permits it."
        ),
    }

def state_prompt_api(prerequisite_api, evidence, contracts):
    """Expose the coder's state ABI, keeping the raw bridge internal to the host.

    Showing getState/setState implementations alongside a rule forbidding
    those calls teaches small coders the wrong API. Project source, compiler
    inputs and canonical state declarations remain authoritative and intact.
    """
    if not contracts:
        return prerequisite_api, evidence
    owners = {row['owner'] for row in contracts}
    apis, inspected = deepcopy(prerequisite_api), deepcopy(evidence)
    state_paths = set()
    for api in apis:
        for declared in api.get('types', ()):
            if declared.get('owner') not in owners:
                continue
            state_paths.add(api.get('path'))
            members = []
            for header in declared.get('members', ()):
                raw, root = _parse('class MmmStatePrompt {\n' + header + '\n}')
                body = root.named_children[0].child_by_field_name('body')
                blocked = False
                scoped = next(
                    (row.get("state_scopes") or {} for row in contracts
                     if row.get("owner") == declared.get("owner")), {})
                for member in body.named_children:
                    if member.type != "method_declaration":
                        continue
                    method_name = _text(raw, member.child_by_field_name("name"))
                    if method_name in _HOST_STATE_INTERNALS or method_name.startswith("mmmPlayerScoped_"):
                        blocked = True
                        break
                    if method_name.startswith(("readState_", "writeState_")):
                        state_name = method_name.split("_", 1)[1]
                        if scoped.get(state_name) == "player":
                            arguments = [param for param in
                                member.child_by_field_name("parameters").named_children
                                if param.type == "formal_parameter"]
                            required = 1 if method_name.startswith("readState_") else 2
                            if len(arguments) < required:
                                blocked = True
                                break
                if not blocked:
                    members.append(header)
            declared['members'] = members

    def allowed(row, owner=''):
        return not (row.get('owner', owner) in owners
                    and row.get('name') in _HOST_STATE_INTERNALS)

    inspected['source'] = [row for row in inspected.get('source', ())
                           if row.get('path') not in state_paths]
    inspected['target_api'] = [row for row in inspected.get('target_api', ()) if allowed(row)]
    for row in inspected.get('class_members', ()):
        row['members'] = [member for member in row.get('members', ())
                          if allowed(member, row.get('verified_class', ''))]
    for row in inspected.get('typed_input_producers', ()):
        row['verified_producers'] = [member for member in row.get('verified_producers', ())
                                    if allowed(member)]
    return apis, inspected



def validate_behavior_state_provenance(source: str, contracts, behavior) -> None:
    """Reject a state call that targets a different behavior's state value.

    A method compiling successfully is insufficient: a spacecraft placement
    handler writing 'player_credits' is silent feature corruption even though
    both symbol and signature exist. Enforce the authoritatively assigned
    read/write identifiers, not hand-written prompt-specific substitutions.
    Non-identifier descriptive state names remain non-callable until planning
    gives them a typed canonical declaration.
    """
    if not isinstance(behavior, dict) or not contracts:
        return
    from .java_region_parser import _parse, _text
    import re

    identifier = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
    permitted = {
        "readState_": set(),
        "writeState_": set(),
    }
    for name in behavior.get("state_reads", ()):
        if isinstance(name, str) and identifier.fullmatch(name):
            permitted["readState_"].add(name)
    for name in behavior.get("state_writes", ()):
        if isinstance(name, str) and identifier.fullmatch(name):
            permitted["writeState_"].add(name)
            # Read-modify-write legitimately reads the same target.
            permitted["readState_"].add(name)

    owner_names = {str(row["owner"]) for row in contracts}
    owner_names.update({name.rsplit(".", 1)[-1] for name in tuple(owner_names)})
    raw, root = _parse("class MmmStateProvenance {\n" + source + "\n}")
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type == "method_invocation":
            receiver = node.child_by_field_name("object")
            name = node.child_by_field_name("name")
            if receiver is not None and name is not None:
                owner = _text(raw, receiver)
                called = _text(raw, name)
                state_name = called.split("_", 1)[1] if called.startswith(
                    ("readState_", "writeState_")) else ""
                scope = next((
                    (contract.get("state_scopes") or {}).get(state_name)
                    for contract in contracts
                    if contract.get("owner") in owner_names
                    and (contract.get("state_scopes") or {}).get(state_name)
                ), None)
                if (owner in owner_names and scope == "player"
                        and called.startswith(("readState_", "writeState_"))):
                    argument_node = node.child_by_field_name("arguments")
                    arity = len(argument_node.named_children) if argument_node is not None else 0
                    minimum = 1 if called.startswith("readState_") else 2
                    if arity < minimum:
                        raise ValueError(
                            "JAVA_PLAYER_STATE_CONTEXT_REQUIRED: " + called)
                for method_prefix, accepted in permitted.items():
                    if (owner in owner_names and called.startswith(method_prefix)
                            and called[len(method_prefix):] not in accepted):
                        raise ValueError(
                            "JAVA_BEHAVIOR_STATE_PROVENANCE_MISMATCH: "
                            + called + " is not a declared " + method_prefix
                            + " target of behavior "
                            + str(behavior.get("behavior_id") or "")
                            + "; permitted=" + repr(sorted(accepted))
                            + ". Repair canonical state planning or use a "
                              "verified non-state world API; never redirect to "
                              "an unrelated existing state variable."
                        )
        pending.extend(node.named_children)

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
                code = ("JAVA_HOST_RAW_STATE_ACCESS" if called in _HOST_STATE_INTERNALS
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
