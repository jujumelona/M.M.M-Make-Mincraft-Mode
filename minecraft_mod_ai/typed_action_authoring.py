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
from .ui_action_source_lowering import _expression, _host_lower_ui_action_body
from .request_record_ownership import same_action_source


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


def _effective_state_mutation(mutation, declared):
    """A source write must demonstrably change an owned canonical state."""
    if not isinstance(mutation, Mapping) or mutation.get("target") not in declared:
        return False
    value = mutation.get("value")
    if not isinstance(value, Mapping):
        return False
    op = mutation.get("operator", "=")
    if (op == "=" and value.get("kind") == "state_ref"
            and value.get("name") == mutation.get("target")):
        return False
    if op in {"+=", "-=", "*=", "/="} and value.get("kind") in {"literal", "number"}:
        from decimal import Decimal, InvalidOperation
        raw = value.get("value")
        try:
            numeric = Decimal(str(raw))
        except (InvalidOperation, ValueError, TypeError):
            return True  # Rejected separately by the typed state validator.
        if (op in {"+=", "-="} and numeric == 0) or (
            op in {"*=", "/="} and numeric == 1
        ):
            return False
    return True


def preflight_ui_action_effects(sections, *, strict_source=False):
    """Reject presentation-only command bodies without a source-owned effect.

    A UI label and a typed text field do not define a purchase, craft, trade or
    launch. The caller must supply an executable state transition or an
    explicitly owned gameplay capability before the host can author a server
    command. Reject fabricated self-writes before hundreds of model decisions.
    """
    from .authored_structured_design import active_concern_records
    from .authored_ui_action_bindings import is_presentation_only_interaction

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
                if _effective_state_mutation(mutation, state_vars):
                    actual_writes.add(str(target))
                elif same_value:
                    invalid_writes.append(str(target))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = row.get("server_action")
        if not isinstance(action, (Mapping, str)):
            continue
        if is_presentation_only_interaction(row, state):
            # A pure screen-open event has no server mutation. It must not
            # be promoted to a synthetic command; its separately authored
            # gameplay controls still require executable state effects.
            continue
        intent = str(action.get("intent") if isinstance(action, Mapping) else action).strip()
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
        matching_writes = [
            mutation
            for collection in ("updates", "transitions")
            for change in state.get(collection, ())
            if isinstance(change, Mapping)
            and same_action_source(change, row)
            for mutation in (change.get("mutation") or ())
            if _effective_state_mutation(mutation, state_vars)
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


_ACTION_INPUT_DECISIONS = {
    "label": "Name the action or screen. This is not an input field.",
    "input_count": (
        "Count only distinct player-supplied values needed by this action. "
        "Opening a screen does not itself supply a value. Use zero when "
        "there is no player choice or text/number entry."
    ),
    "input_type": (
        "Define one player-supplied value at input_index. The label names "
        "that value or choice, never the screen, service, action, displayed "
        "balance or result. Choose the scalar type of the supplied value."
    ),
    "input_bounds": (
        "Define the minimum and maximum for the one supplied numeric input, "
        "using its declared type and the action's actual constraints."
    ),
}


def _action_input_asker(router, record, budget, state_context, requested_prompt):
    from .fixed_template_generation import _validate_native_arguments

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
                        " Current decision: " + _ACTION_INPUT_DECISIONS[name]
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"decision": name, "interaction": record,
                         "requested_prompt": requested_prompt,
                         "state_context": state_context, **context}, ensure_ascii=False
                    ),
                },
            ],
            data_name="ui_action_" + name,
            parameters=schema,
        )
        return _validate_native_arguments(response, schema)


    return ask


def _author_prose_action_inputs(ask):
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
    return title, inputs


def author_action_metadata(router, record, *, budget=None, source_sections=None,
                           requested_prompt=""):
    if isinstance(record.get("server_action"), Mapping):
        return validate_action_metadata(record["server_action"])
    state_context = {}
    if source_sections is not None:
        from .authored_structured_design import active_concern_records
        from .ui_action_input_source import source_action_inputs

        state_context = active_concern_records(source_sections, "state_model")
        interactions = active_concern_records(source_sections, "resources_and_ui").get("interactions", ())
        inputs = source_action_inputs(record, state_context, interactions,
                                      validate_expression=_expression)
        if inputs is not None:
            return validate_action_metadata({
                "intent": str(record.get("server_action") or "").strip(),
                "inputs": inputs,
            })
    ask = _action_input_asker(router, record, budget, state_context, requested_prompt)
    title, inputs = _author_prose_action_inputs(ask)
    # The screen title is not the server transaction.
    return validate_action_metadata({
        "intent": str(record.get("server_action") or "").strip() or title,
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
    from .authored_ui_action_bindings import is_presentation_only_interaction

    functions, bindings = [], []
    for record in rows:
        if (not isinstance(record.get("server_action"), Mapping)
                or is_presentation_only_interaction(record, state)):
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
        is_presentation_only_interaction,
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
    state = active_concern_records(plan.structured_sections, "state_model")
    expected = {
        action_record_identity(row)
        for row in rows
        if (isinstance(row.get("server_action"), Mapping)
            and not is_presentation_only_interaction(row, state))
    }
    existing = {row["id"] for row in action_bindings(plan.typed_plan_ir)}
    if (
        all(isinstance(row.get("server_action"), Mapping)
            or is_presentation_only_interaction(row, state) for row in rows)
        and expected == existing
    ):
        return plan
    sections = deepcopy(plan.structured_sections)
    target_rows = sections["resources_and_ui"]["specification"]["interactions"]
    identities = {}
    for row in target_rows:
        if is_presentation_only_interaction(row, state):
            continue
        old = row["server_action"]
        row["server_action"] = author_action_metadata(
            router, row, source_sections=plan.structured_sections,
            requested_prompt=plan.requested_prompt,
        )
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


def promote_action_sections(router, sections, *, budget=None, minecraft_version="",
                            requested_prompt=""):
    """Resolve concrete inputs before runtime ownership and function authoring."""
    from .authored_ui_action_bindings import (
        _mutating_ui_actions, is_presentation_only_interaction,
    )

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
    state = (
        result.get("state_model", {})
        .get("specification", {})
    )
    for record in (
        result.get("resources_and_ui", {})
        .get("specification", {})
        .get("interactions", ())
    ):
        if is_presentation_only_interaction(record, state):
            continue
        record["server_action"] = author_action_metadata(
            router, record, budget=budget, source_sections=sections,
            requested_prompt=requested_prompt,
        )
    return result
