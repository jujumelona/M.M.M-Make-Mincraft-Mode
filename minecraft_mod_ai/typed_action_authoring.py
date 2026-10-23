"""Bounded UI input contracts, then one typed server body per action.

The client cannot choose a function, actor, or state key. Those identities come
from the host's persisted binding, not an English dispatcher string.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import re

from .typed_action_contract import SCALARS, action_bindings, validate_action_contract


def preflight_state_guard_semantics(sections):
    """Validate typed state guard inputs before any method-body generation.

    A comparison of a variable with itself proves no prerequisite, and an
    ordering comparison against string-typed currency cannot implement a
    numeric purchase or launch gate. Neither is an actionable game contract.
    """
    from .authored_structured_design import active_concern_records

    model = active_concern_records(sections, "state_model")
    types = {
        row["name"]: row.get("type")
        for row in model.get("variables", ())
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }

    def visit(node, section):
        if isinstance(node, Mapping):
            if node.get("kind") == "compare":
                op = node.get("op")
                left = node.get("left")
                right = node.get("right")
                if (op in {"==", "!=", "<", "<=", ">", ">="}
                        and isinstance(left, Mapping)
                        and isinstance(right, Mapping) and left == right):
                    raise ValueError(
                        "STATE_GUARD_TAUTOLOGY: "
                        + section + " compares an expression with itself"
                    )
                if op in {"<", "<=", ">", ">="}:
                    for expr in (left, right):
                        if (isinstance(expr, Mapping)
                                and expr.get("kind") == "state_ref"
                                and types.get(expr.get("name")) == "string"):
                            raise ValueError(
                                "STATE_GUARD_NONNUMERIC_ORDERING: "
                                + section + " orders string state "
                                + str(expr.get("name"))
                            )
            for value in node.values():
                visit(value, section)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value, section)

    for collection in ("transitions", "invariants"):
        for index, row in enumerate(model.get(collection, ())):
            visit(row, f"state_model.{collection}[{index}]")

    # Reusing the same state ref on the right of a plain assignment leaves the
    # world unchanged. Such rows used to pass the typed state schema and later
    # masquerade as successful gameplay updates.
    mutation_collections = {
        "transitions": "mutation",
        "updates": "mutation",
        "initialization": "initial_state",
        "cleanup": "action",
    }
    for collection, field in mutation_collections.items():
        for index, row in enumerate(model.get(collection, ())):
            if not isinstance(row, Mapping):
                continue
            for assignment in row.get(field, ()) or ():
                if not isinstance(assignment, Mapping):
                    continue
                value = assignment.get("value")
                if (assignment.get("operator", "=") == "="
                        and isinstance(value, Mapping)
                        and value.get("kind") == "state_ref"
                        and value.get("name") == assignment.get("target")):
                    raise ValueError(
                        "STATE_IDENTITY_MUTATION: "
                        f"state_model.{collection}[{index}].{field} "
                        f"assigns {assignment.get('target')!r} to itself"
                    )


def preflight_ui_action_effects(sections, *, strict_source=False):
    """Reject presentation-only command bodies without a source-owned effect.

    A UI label and a typed text field do not define a purchase, craft, trade or
    launch. The caller must supply an executable state transition or an
    explicitly owned gameplay capability before the host can author a server
    command. Reject fabricated self-writes before hundreds of model decisions.
    """
    from .authored_structured_design import active_concern_records

    rows = active_concern_records(sections, "resources_and_ui").get(
        "interactions", ()
    )
    state = active_concern_records(sections, "state_model")
    state_vars = {
        str(item.get("name")) for item in state.get("variables", ())
        if isinstance(item, Mapping) and item.get("name")
    }
    actual_writes = set()
    invalid_writes = []
    for collection in ("updates", "transitions"):
        for row in state.get(collection, ()):
            if not isinstance(row, Mapping):
                continue
            for mutation in row.get("mutation", ()) or ():
                if not isinstance(mutation, Mapping):
                    continue
                target = mutation.get("target")
                value = mutation.get("value")
                same_value = (
                    mutation.get("operator") == "="
                    and isinstance(value, Mapping)
                    and value.get("kind") == "state_ref"
                    and value.get("name") == target
                )
                if target in state_vars and not same_value:
                    actual_writes.add(str(target))
                elif same_value:
                    invalid_writes.append(str(target))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = row.get("server_action")
        if not isinstance(action, Mapping):
            continue
        intent = str(action.get("intent") or "")
        # A request to display a screen is not yet a server-side transaction.
        # Keep gameplay commands with explicit action verbs eligible even when
        # their UI title contains 'menu' or 'panel'.
        display_noun = re.search(
            r"\b(?:ui|menu|screen|interface|panel|dialog|window)\b",
            intent, re.I,
        )
        gameplay_verb = re.search(
            r"\b(?:buy|purchase|sell|trade|mine|craft|build|construct|"
            r"upgrade|launch|collect|harvest|equip|claim|fight|attack|"
            r"spawn|create|change|set|save|withdraw|deposit)\b",
            intent, re.I,
        )
        # Only a transaction with the same trigger can justify this command.
        # A launch-time currency reset cannot implement a shipyard purchase,
        # even if both mention the same state variable.
        trigger = str(row.get("trigger") or "").strip().casefold()
        matching_writes = [
            mutation
            for collection in ("updates", "transitions")
            for change in state.get(collection, ())
            if isinstance(change, Mapping)
            and str(change.get("trigger") or "").strip().casefold() == trigger
            for mutation in (change.get("mutation") or ())
            if isinstance(mutation, Mapping)
            and mutation.get("target") in actual_writes
        ]
        if not matching_writes and (strict_source or (display_noun and not gameplay_verb)):
            raise ValueError(
                ("ACTION_UI_EFFECT_UNBOUND: " if actual_writes
                 else "ACTION_DESIGN_NO_EXECUTABLE_EFFECT: ") + intent[:160]
                + "; no state mutation belongs to this UI trigger. "
                "The UI opening is not a purchase, and an unrelated state "
                "write cannot implement the requested server transaction. "
                "Author a source-owned input-bound update or transition, "
                "or separate the read-only screen from mutating actions."
            )


def _checked_action_target(router, minecraft_version=""):
    version = str(
        minecraft_version
        or getattr(router, "_mmm_requested_minecraft_version", "")
        or getattr(
            getattr(router, "_mmm_target_adapter", None), "minecraft_version", ""
        )
        or getattr(router, "_mmm_existing_minecraft_version", "")
    )
    return not version or version == "26.2"


def action_identity(metadata):
    encoded = json.dumps(
        metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return "action_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]


def action_metadata_schema():
    common = {
        "name": {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"},
        "label": {"type": "string", "minLength": 1, "maxLength": 160},
    }
    branches = []
    for kind in SCALARS:
        fields = {**common, "type": {"type": "string", "const": kind}}
        if kind in {"int", "long", "double"}:
            value_type = "number" if kind == "double" else "integer"
            fields.update(
                {"minimum": {"type": value_type}, "maximum": {"type": value_type}}
            )
        elif kind == "string":
            fields.update(
                {
                    "max_length": {"type": "integer", "minimum": 0, "maximum": 4096},
                    "choices": {
                        "type": "array",
                        "maxItems": 32,
                        "items": {"type": "string", "maxLength": 160},
                    },
                }
            )
        branches.append(
            {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "minLength": 1, "maxLength": 160},
            "inputs": {"type": "array", "maxItems": 16, "items": {"oneOf": branches}},
        },
        "required": ["intent", "inputs"],
        "additionalProperties": False,
    }


def validate_action_metadata(metadata):
    if not isinstance(metadata, Mapping) or set(metadata) != {"intent", "inputs"}:
        raise ValueError("ACTION_METADATA_FIELDS")
    validate_action_contract(
        {
            "id": action_identity(metadata),
            "label": metadata["intent"],
            "inputs": metadata["inputs"],
        }
    )
    if any(row["name"] in {"source", "action_context"} for row in metadata["inputs"]):
        raise ValueError("ACTION_INPUT_RESERVED_ACTOR")
    return deepcopy(dict(metadata))


def _complete_action_input(ask, row, *, count):
    """Host-owned bounds and unrestricted text ABI for one admitted user field."""
    head = row
    if head["type"] in {"int", "long", "double"}:
        if head["type"] == "double":
            numeric = {
                "type": "string",
                "minLength": 1,
                "maxLength": 24,
                "pattern": r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,3})?$",
            }
        else:
            bits = 32 if head["type"] == "int" else 64
            numeric = {
                "type": "integer",
                "minimum": -(2 ** (bits - 1)),
                "maximum": 2 ** (bits - 1) - 1,
            }
        row.update(
            ask(
                "input_bounds",
                {"minimum": numeric, "maximum": numeric},
                input=row,
            )
        )
        if head["type"] == "double":
            row["minimum"], row["maximum"] = (
                float(row["minimum"]),
                float(row["maximum"]),
            )
    elif head["type"] == "string":
        # A list of legal server values is not present in this prose
        # interaction. Never ask the small model to manufacture choices:
        # it repeated a screen title eight times in production, and even
        # unique fabricated options would not be authoritative. Empty
        # choices means text entry, subject to server-side validation.
        # The host chooses a UTF-8/command-safe byte budget.
        row["max_length"] = min(
            160, (32767 - 128 - count * 50) * 3 // (16 * max(1, count))
        )
        row["choices"] = []
    return row


def author_action_metadata(router, record, *, budget=None):
    from .fixed_template_generation import _validate_native_arguments

    if isinstance(record.get("server_action"), Mapping):
        return validate_action_metadata(record["server_action"])

    def ask(name, properties, **context):
        if budget is not None:
            budget.consume("ui_action." + name)
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        response = router.generate_structured_data(
            "planner",
            [
                {
                    "role": "system",
                    "content": (
                        "Define the requested UI action's input ABI, one bounded decision at a time. "
                        "Only values the user actually supplies are client inputs. Currency, prices, "
                        "permissions and world facts remain server-owned. Do not add a different "
                        "gameplay trigger, Java, new features, or a notification-only replacement."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"interaction": record, **context}, ensure_ascii=False
                    ),
                },
            ],
            data_name="ui_action_" + name,
            parameters=schema,
        )
        return _validate_native_arguments(response, schema)

    title = ask(
        "label", {"label": {"type": "string", "minLength": 1, "maxLength": 160}}
    )["label"]
    # Free-form prose has no authoritative form schema. Let the model propose
    # a small set of actually player-supplied controls, not sixteen fields
    # merely because the decoder allows that maximum. Structured action
    # metadata still supports up to sixteen explicitly declared controls.
    count = ask(
        "input_count", {"count": {"type": "integer", "minimum": 0, "maximum": 4}}
    )["count"]
    inputs = []
    for index in range(count):
        head = ask(
            "input_type",
            {
                "type": {"type": "string", "enum": list(SCALARS)},
                "label": {"type": "string", "minLength": 1, "maxLength": 160},
            },
            input_index=index,
            accepted_inputs=inputs,
        )
        # The small model can repeat the entire screen title or an earlier
        # control when a requested input count exceeds the real form.
        # Do not synthesize duplicate inputs: request one grounded correction,
        # and treat a repeated prior control as the end of the list.
        previous = {entry["label"].strip().casefold() for entry in inputs}
        screen_label = title.strip().casefold()
        label = head["label"].strip().casefold()
        if label == screen_label or label in previous:
            head = ask(
                "input_type",
                {
                    "type": {"type": "string", "enum": list(SCALARS)},
                    "label": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                input_index=index,
                accepted_inputs=inputs,
                rejected_input=head,
                correction="Name a DISTINCT user-supplied field, not the screen title or a prior field. If no other field exists, do not invent an option.",
            )
            label = head["label"].strip().casefold()
        if label in previous:
            break
        if label == screen_label:
            raise ValueError(
                "ACTION_INPUT_NOT_A_PLAYER_VALUE: "
                + json.dumps({"input_index": index, "label": head["label"]}, ensure_ascii=False)
            )
        row = {"name": "input_" + str(len(inputs) + 1), **head}
        row = _complete_action_input(ask, row, count=count)
        inputs.append(row)
    # The screen title is not the transaction. Preserve the authored
    # server-side action, otherwise a purchase becomes a presentation-only UI.
    source_intent = str(record.get("server_action") or "").strip()
    return validate_action_metadata({
        "intent": source_intent or title,
        "inputs": inputs,
    })


def _nodes(value):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _nodes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nodes(item)


def _admit_body(function, metadata, state_types, capabilities):
    from .typed_plan_ir import _Validator

    nodes = list(_nodes(function["body"]))
    mutating = {
        key for key, row in capabilities.items() if row.get("gameplay_mutation")
    }
    if not any(
        row.get("op") == "state_set"
        or (row.get("op") == "capability" and row.get("id") in mutating)
        for row in nodes
    ):
        raise ValueError("ACTION_GAMEPLAY_MUTATION_REQUIRED")
    used = {row.get("name") for row in nodes if row.get("op") == "ref"}
    if any(row["name"] not in used for row in metadata["inputs"]):
        raise ValueError("ACTION_DECLARED_INPUT_UNUSED")
    validator = _Validator({}, capabilities)
    for node in nodes:
        if node.get("op") == "state_get":
            key = node["key"]
            if key.get("op") != "literal" or key.get("value") not in state_types:
                raise ValueError("ACTION_STATE_TARGET_NOT_DECLARED")
            if node["type"] != state_types[key["value"]]:
                raise ValueError("ACTION_STATE_TYPE_MISMATCH")

    def block(statements, env):
        for statement in statements:
            if statement["op"] == "let":
                env[statement["name"]] = statement["type"]
            if statement["op"] == "state_set":
                key = statement["key"]
                if key.get("op") != "literal" or key.get("value") not in state_types:
                    raise ValueError("ACTION_STATE_TARGET_NOT_DECLARED")
                actual = validator.expression(
                    statement["value"], env, "action.value", function["id"]
                )
                if actual != state_types[key["value"]]:
                    raise ValueError("ACTION_STATE_TYPE_MISMATCH")
            for branch in ("then", "else", "body"):
                if isinstance(statement.get(branch), list):
                    nested = dict(env)
                    if statement["op"] == "foreach" and branch == "body":
                        nested[statement["name"]] = statement["type"]
                    block(statement[branch], nested)

    block(function["body"], {p["name"]: p["type"] for p in function["parameters"]})


def _host_lower_ui_action_body(
    record, metadata, owned_refs, sections, state_types,
):
    """Lower an already authored, unambiguous state mutation without a model.

    This is a recovery path for a small model that loops while writing a
    recursive AST. Only source-owned canonical state mutations are eligible.
    Unsupported, unassigned or ambiguous effects must fail closed, rather
    than inventing a gameplay success.
    """
    from .authored_structured_design import active_concern_records
    from .typed_plan_authoring import (
        _decode_double_literal, _decode_int_literal, _decode_long_literal,
    )

    state = active_concern_records(sections, "state_model")
    interactions = active_concern_records(sections, "resources_and_ui").get(
        "interactions", ()
    )
    inputs = {row["name"]: row["type"] for row in metadata["inputs"]}
    available = []
    for concern in ("updates", "transitions"):
        for index, row in enumerate(state.get(concern, ())):
            if isinstance(row, Mapping):
                available.append((f"state_model.{concern}[{index}]", row))

    action_trigger = str(record.get("trigger") or "").strip().casefold()
    selected = [(ref, row) for ref, row in available if ref in owned_refs]
    if any(str(row.get("trigger") or "").strip().casefold() != action_trigger
           for _ref, row in selected):
        raise ValueError(
            "ACTION_HOST_OWNER_TRIGGER_MISMATCH: a runtime owner references "
            "a state rule belonging to a different UI trigger"
        )
    if not selected and not owned_refs:
        # Legacy single-action designs have no explicit runtime owner. Only
        # admit an exactly matching, unique state mutation in that case.
        action_rows = [
            row for row in interactions
            if isinstance(row, Mapping)
            and isinstance(row.get("server_action"), Mapping)
        ]
        if len(action_rows) == 1 and len(available) == 1:
            candidate = available[0]
            trigger = str(candidate[1].get("trigger") or "").strip().casefold()
            requested = str(record.get("trigger") or "").strip().casefold()
            if trigger and requested and trigger == requested:
                selected = [candidate]
    if not selected:
        raise ValueError(
            "ACTION_HOST_LOWERING_UNAVAILABLE: no uniquely assigned canonical "
            "state mutation for this UI action"
        )
    # Charging or applying the same persisted transaction twice is never
    # acceptable. Detect duplicate source rows before lowering.
    signatures = [
        json.dumps(row.get("mutation"), sort_keys=True, ensure_ascii=False)
        for _ref, row in selected
    ]
    if len(signatures) != len(set(signatures)):
        raise ValueError(
            "ACTION_HOST_DUPLICATE_MUTATION: identical transaction records "
            "were assigned to the same UI action"
        )

    def literal(typ, value):
        if typ == "int":
            value = _decode_int_literal(value, scope="ui_action.literal")
        elif typ == "long":
            value = _decode_long_literal(value, scope="ui_action.literal")
        elif typ == "double":
            value = _decode_double_literal(value, scope="ui_action.literal")
        elif typ == "boolean":
            if type(value) is not bool:
                raise ValueError("ACTION_HOST_LITERAL_TYPE: expected boolean")
        elif typ == "string":
            if not isinstance(value, str):
                raise ValueError("ACTION_HOST_LITERAL_TYPE: expected string")
        else:
            raise ValueError("ACTION_HOST_LITERAL_TYPE: unsupported " + str(typ))
        return {"op": "literal", "type": typ, "value": value}

    def expression(node, expected=None):
        if not isinstance(node, Mapping):
            raise ValueError("ACTION_HOST_EXPRESSION_REQUIRED: canonical state IR")
        kind = node.get("kind")
        if kind == "context_ref":
            name = str(node.get("name") or "")
            if name not in inputs or (expected and inputs[name] != expected):
                raise ValueError("ACTION_HOST_INPUT_UNBOUND: " + name)
            return {"op": "ref", "name": name}
        if kind == "state_ref":
            name = str(node.get("name") or "")
            if name not in state_types or (expected and state_types[name] != expected):
                raise ValueError("ACTION_HOST_STATE_UNBOUND: " + name)
            return {
                "op": "state_get",
                "key": literal("string", name),
                "type": state_types[name],
                "context": {"op": "map", "entries": []},
            }
        if kind in {"literal", "number"}:
            value = node.get("value")
            inferred = expected or (
                "boolean" if type(value) is bool else
                "double" if isinstance(value, float) or
                (kind == "number" and "." in str(value)) else
                "int" if kind == "number" or type(value) is int else
                "string" if isinstance(value, str) else None
            )
            return literal(inferred, value)
        if kind == "arithmetic":
            if expected not in {"int", "long", "double"}:
                raise ValueError("ACTION_HOST_ARITHMETIC_TYPE_UNBOUND")
            op = node.get("op")
            if op not in {"+", "-", "*", "/", "%"}:
                raise ValueError("ACTION_HOST_ARITHMETIC_OPERATOR")
            return {
                "op": "binary", "operator": op,
                "left": expression(node.get("left"), expected),
                "right": expression(node.get("right"), expected),
            }
        if kind == "compare":
            if expected not in {None, "boolean"}:
                raise ValueError("ACTION_HOST_COMPARE_TYPE")
            op = node.get("op")
            if op not in {"==", "!=", "<", "<=", ">", ">="}:
                raise ValueError("ACTION_HOST_COMPARE_OPERATOR")
            # State references and player inputs carry their declared types.
            left = node.get("left")
            right = node.get("right")
            hint = None
            for operand in (left, right):
                if isinstance(operand, Mapping):
                    if operand.get("kind") == "state_ref":
                        hint = state_types.get(operand.get("name"))
                        break
                    if operand.get("kind") == "context_ref":
                        hint = inputs.get(operand.get("name"))
                        break
            if hint is None:
                raise ValueError("ACTION_HOST_COMPARE_UNTYPED")
            return {
                "op": "binary", "operator": op,
                "left": expression(left, hint),
                "right": expression(right, hint),
            }
        if kind == "not":
            return {
                "op": "unary", "operator": "!",
                "value": expression(node.get("term") or node.get("operand"), "boolean"),
            }
        if kind in {"and", "or"}:
            terms = node.get("terms")
            if not isinstance(terms, list) or len(terms) < 2:
                raise ValueError("ACTION_HOST_BOOLEAN_TERMS")
            acc = expression(terms[0], "boolean")
            for term in terms[1:]:
                acc = {
                    "op": "binary",
                    "operator": "&&" if kind == "and" else "||",
                    "left": acc, "right": expression(term, "boolean"),
                }
            return acc
        raise ValueError("ACTION_HOST_EXPRESSION_UNSUPPORTED: " + str(kind))

    all_statements = []
    for ref, row in selected:
        mutations = row.get("mutation")
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("ACTION_HOST_MUTATION_REQUIRED: " + ref)
        writes = []
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                raise ValueError("ACTION_HOST_MUTATION_INVALID: " + ref)
            name = str(mutation.get("target") or "")
            if name not in state_types:
                raise ValueError("ACTION_HOST_STATE_UNBOUND: " + name)
            typ = state_types[name]
            op = mutation.get("operator", "=")
            right = expression(mutation.get("value"), typ)
            if op != "=":
                if op not in {"+=", "-=", "*=", "/="} or typ not in {
                    "int", "long", "double"
                }:
                    raise ValueError("ACTION_HOST_MUTATION_OPERATOR")
                right = {
                    "op": "binary", "operator": op[0],
                    "left": expression({"kind": "state_ref", "name": name}, typ),
                    "right": right,
                }
            writes.append({
                "op": "state_set",
                "key": literal("string", name),
                "value": right,
                "context": {"op": "map", "entries": []},
            })
        guard = row.get("guard")
        if guard is not None:
            all_statements.append({
                "op": "if",
                "condition": expression(guard, "boolean"),
                "then": writes,
                "else": [{"op": "return", "value": literal("int", 0)}],
            })
        else:
            all_statements.extend(writes)
    all_statements.append({"op": "return", "value": literal("int", 1)})
    return all_statements


def author_action_functions(
    router,
    source_text,
    sections,
    capabilities,
    *,
    first_index,
    budget=None,
    runtime_owners=(),
    prefer_grounded=False,
):
    from .authored_structured_design import active_concern_records
    from .authored_ui_action_bindings import action_record_identity
    from .typed_plan_authoring import (
        TypedOperationAuthor,
        _extract_state_variable_types,
    )
    from .typed_plan_ir import validate_typed_plan_ir

    rows = active_concern_records(sections, "resources_and_ui").get("interactions", ())
    state = active_concern_records(sections, "state_model")
    state_types = _extract_state_variable_types(sections)
    preflight_state_guard_semantics(sections)
    preflight_ui_action_effects(sections)
    functions, bindings = [], []
    for record in rows:
        if not isinstance(record.get("server_action"), Mapping):
            continue
        metadata = validate_action_metadata(record["server_action"])
        identity = action_record_identity(record)
        if any(binding["config"]["action"]["id"] == identity for binding in bindings):
            continue
        fn_id = "ui_" + identity
        owned_refs = [
            ref
            for owner in runtime_owners
            if owner.get("feature_id") == "__typed_ui__"
            and owner.get("ui_action") == identity
            for ref in owner.get("worksheet_record_refs", ())
        ]
        semantic_context = {
            "resources_and_ui": {"interactions": [deepcopy(record)]},
            "state_model": state,
        }
        if owned_refs:
            from .request_runtime_records import owner_records

            semantic_context["assigned_runtime_records"] = owner_records(
                sections,
                [
                    owner
                    for owner in runtime_owners
                    if owner.get("ui_action") == identity
                ],
                "__typed_ui__",
            )

        class ActionAuthor(TypedOperationAuthor):
            def _scope_semantic_context(self, scope):
                return semantic_context

            def expression(self, scope, env=None, **kwargs):
                if scope.endswith(".state_context"):
                    return deepcopy(action_context)
                return super().expression(scope, env, **kwargs)

        author = ActionAuthor(
            router, source_text, sections, capabilities, budget=budget, max_calls=64
        )
        parameters = [
            {"name": "source", "type": "object"},
            *({"name": row["name"], "type": row["type"]} for row in metadata["inputs"]),
        ]
        if "player.uuid" not in (capabilities or {}):
            raise ValueError("ACTION_HOST_PLAYER_CONTEXT_REQUIRED")
        action_context = {
            "op": "map",
            "entries": [
                {
                    "key": {"op": "literal", "type": "string", "value": "player"},
                    "value": {
                        "op": "capability",
                        "id": "player.uuid",
                        "args": [{"op": "ref", "name": "source"}],
                    },
                }
            ],
        }
        body = None
        if prefer_grounded:
            # A declared transaction must compile from its source-owned IR.
            # Do not spend dozens of model calls inventing a recursive body
            # when the source has no corresponding executable mutation.
            try:
                body = _host_lower_ui_action_body(
                    record, metadata, owned_refs, sections, state_types,
                )
            except ValueError as error:
                raise ValueError(
                    "ACTION_SOURCE_EFFECT_UNBOUND: " + str(error)
                ) from error
        if body is None:
            try:
                body = author.body(
                    "ui_action." + identity,
                    env={**{p["name"]: p["type"] for p in parameters}, "action_context": "map"},
                    return_type="int",
                    max_statements=12,
                )
            except ValueError as exc:
                if not str(exc).startswith((
                    "TYPED_PLAN_REPEATED_NO_PROGRESS:", "TYPED_PLAN_AUTHORING_LIMIT:",
                    "TYPED_PLAN_SEMANTIC_BUDGET_EXHAUSTED:",
                )):
                    raise
                # Never accept an incomplete model branch as a valid effect.
                try:
                    body = _host_lower_ui_action_body(
                        record, metadata, owned_refs, sections, state_types,
                    )
                except ValueError as lowering_error:
                    raise ValueError(
                        "ACTION_HOST_LOWERING_REQUIRED: model AST did not progress; "
                        + str(lowering_error)
                    ) from exc
        for node in list(_nodes(body)):
            if node.get("op") == "set" and node.get("name") in {
                "source",
                "action_context",
            }:
                raise ValueError("ACTION_HOST_CONTEXT_IS_IMMUTABLE")
            if node.get("op") in {"state_get", "state_set"}:
                node["context"] = deepcopy(action_context)
            if node.get("op") == "ref" and node.get("name") == "action_context":
                node.clear()
                node.update(deepcopy(action_context))
        function = {
            "id": fn_id,
            "parameters": parameters,
            "return_type": "int",
            "covers": sorted(
                {
                    "resources_and_ui.interactions",
                    *(ref.split("[", 1)[0] for ref in owned_refs),
                }
            ),
            "body": body,
        }
        _admit_body(function, metadata, state_types, capabilities or {})
        functions.append(function)
        bindings.append(
            {
                "event": "command",
                "function": fn_id,
                "entry_point_index": first_index + len(bindings),
                "config": {
                    "literal": "mmm_ui_" + identity.removeprefix("action_"),
                    "permission_level": 0,
                    "action": {
                        "id": identity,
                        "label": metadata["intent"],
                        "inputs": metadata["inputs"],
                    },
                },
            }
        )
    validate_typed_plan_ir(
        {
            "schema_version": "mmm/typed-plan-ir-v1",
            "source_sha256": "0" * 64,
            "functions": functions,
            "event_bindings": bindings,
            "initialize": [],
        },
        capabilities=capabilities,
    )
    return functions, bindings


def migrate_ui_action_plan(router, plan, *, minecraft_version=""):
    """Upgrade saved UI execution, preserving unrelated functions and events."""
    from .authored_structured_design import active_concern_records
    from .authored_ui_action_bindings import (
        _mutating_ui_actions,
        action_record_identity,
    )
    from .typed_host_capabilities import typed_host_capability_contracts
    from .typed_plan_ir import validate_typed_plan_ir

    rows = active_concern_records(plan.structured_sections, "resources_and_ui").get(
        "interactions", ()
    )
    if not rows:
        return plan
    if not _checked_action_target(router, minecraft_version):
        if any(isinstance(row.get("server_action"), Mapping) for row in rows):
            raise ValueError(
                "ACTION_CLIENT_TARGET_UNSUPPORTED: typed action forms require checked Minecraft 26.2 GUI ABI"
            )
        return plan
    # Read-only worksheets without a mutating UI contract retain their owner.
    if not _mutating_ui_actions(plan.structured_sections) and not any(
        isinstance(row.get("server_action"), Mapping) for row in rows
    ):
        return plan
    expected = {
        action_record_identity(row)
        for row in rows
        if isinstance(row.get("server_action"), Mapping)
    }
    existing = {row["id"] for row in action_bindings(plan.typed_plan_ir)}
    if (
        all(isinstance(row.get("server_action"), Mapping) for row in rows)
        and expected == existing
    ):
        return plan
    sections = deepcopy(plan.structured_sections)
    target_rows = sections["resources_and_ui"]["specification"]["interactions"]
    identities = {}
    for row in target_rows:
        old = row["server_action"]
        row["server_action"] = author_action_metadata(router, row)
        identities[str(old)] = action_record_identity(row)
    content = deepcopy(plan.content_design)
    owners = content.get("_request_runtime_owners", ())
    for owner in owners:
        if (
            owner.get("feature_id") == "__typed_ui__"
            and owner.get("ui_action") in identities
        ):
            owner["ui_action"] = identities[owner["ui_action"]]
    ir = deepcopy(plan.typed_plan_ir)

    def is_ui_binding(row):
        return "action" in row.get("config", {}) or (
            row.get("function", "").startswith("uiAction")
            and row.get("config", {}).get("literal", "").startswith("mmm_action_")
        )

    removed = {
        row["function"] for row in ir.get("event_bindings", ()) if is_ui_binding(row)
    }
    ir["event_bindings"] = [
        b for b in ir.get("event_bindings", ()) if not is_ui_binding(b)
    ]
    retained_roots = {b["function"] for b in ir["event_bindings"]}
    retained_roots.update(
        node.get("function")
        for node in _nodes([ir["functions"], ir.get("initialize", [])])
        if node.get("op") == "call"
    )
    removed -= retained_roots
    ir["functions"] = [fn for fn in ir["functions"] if fn["id"] not in removed]
    calls = {
        node.get("function")
        for node in _nodes([ir["functions"], ir.get("initialize", [])])
        if node.get("op") == "call"
    }
    if "ui_action_dispatch" not in calls:
        ir["functions"] = [
            fn for fn in ir["functions"] if fn["id"] != "ui_action_dispatch"
        ]
    first = max(
        [
            len(
                active_concern_records(sections, "integration").get("entry_points", ())
            ),
            *(b["entry_point_index"] + 1 for b in ir["event_bindings"]),
        ]
    )
    functions, bindings = author_action_functions(
        router,
        plan.text,
        sections,
        typed_host_capability_contracts(),
        first_index=first,
        runtime_owners=owners, prefer_grounded=True,
    )
    ir["functions"].extend(functions)
    ir["event_bindings"].extend(bindings)
    ir = validate_typed_plan_ir(ir, capabilities=typed_host_capability_contracts())
    return replace(
        plan, structured_sections=sections, typed_plan_ir=ir, content_design=content
    )


def promote_action_sections(router, sections, *, budget=None, minecraft_version=""):
    """Resolve concrete inputs before runtime ownership and function authoring."""
    from .authored_ui_action_bindings import _mutating_ui_actions

    if not _checked_action_target(router, minecraft_version):
        rows = (
            sections.get("resources_and_ui", {})
            .get("specification", {})
            .get("interactions", ())
        )
        if any(isinstance(row.get("server_action"), Mapping) for row in rows):
            raise ValueError(
                "ACTION_CLIENT_TARGET_UNSUPPORTED: typed action forms require checked Minecraft 26.2 GUI ABI"
            )
        return sections
    if not _mutating_ui_actions(sections):
        return sections
    result = deepcopy(sections)
    for record in (
        result.get("resources_and_ui", {})
        .get("specification", {})
        .get("interactions", ())
    ):
        record["server_action"] = author_action_metadata(router, record, budget=budget)
    return result
