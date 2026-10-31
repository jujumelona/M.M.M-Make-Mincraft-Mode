"""Compile small semantic callback choices into host-owned Java statements.

Expressions are values, never source declarations or statement fragments. The
host owns local names, semicolons, block delimiters, predicates' return syntax
and completion. Effects have no return operation. Final javac admission still
checks actual receiver/argument types and does not certify gameplay behavior.
"""
from __future__ import annotations

import json

from .construction_source_repair import validate_generated_java_source
from .fixed_template_generation import _validate_native_arguments
from .java_operation_ir import identifier
from .java_region_parser import _parse, _text, class_body_token_identity
from .structured_output import StructuredOutputValidationError
from .host_verified_callback_ops import verified_callback_choices
from .structured_json_schema import closed_object_schema as _object


_EXPRESSION = {"type": "string", "minLength": 1, "maxLength": 2048}
_EXPRESSION_SCHEMA = _object({"expression": _EXPRESSION})
_SCHEMAS = {
    "call": _object({"receiver": _EXPRESSION,
                     "method": {"type": "string", "minLength": 1,
                                "maxLength": 128, "pattern": r"^[A-Za-z_$][A-Za-z0-9_$]*$"},
                     "arguments": {"type": "array", "maxItems": 16, "items": _EXPRESSION}}),
    "assign": _object({"target": _EXPRESSION,
                       "operator": {"type": "string", "enum": ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="]},
                       "value": _EXPRESSION}),
    "declare": _EXPRESSION_SCHEMA,
    "if": _EXPRESSION_SCHEMA,
    "while": _EXPRESSION_SCHEMA,
    "foreach": _EXPRESSION_SCHEMA,
}
_FORBIDDEN = frozenset({"lambda_expression", "class_body", "class_declaration",
                        "method_declaration", "assignment_expression", "update_expression",
                        "switch_expression", "block", "return_statement", "yield_statement"})


def parse_expression(value: str, *, predicate=False) -> str:
    """Admit exactly one Java value and reject statement/declaration escapes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CALLBACK_EXPRESSION_REQUIRED")
    raw, root = _parse("class MmmExpression { Object value() { return " + value + "; } }")
    owners = [n for n in root.named_children if n.type == "class_declaration"]
    if len(owners) != 1:
        raise ValueError("CALLBACK_EXPRESSION_SCOPE")
    body = owners[0].child_by_field_name("body")
    methods = [n for n in body.named_children if n.type == "method_declaration"]
    if len(methods) != 1:
        raise ValueError("CALLBACK_EXPRESSION_SCOPE")
    statements = [n for n in methods[0].child_by_field_name("body").named_children
                  if n.type not in {"line_comment", "block_comment"}]
    if len(statements) != 1 or statements[0].type != "return_statement":
        raise ValueError("CALLBACK_EXPRESSION_NOT_A_VALUE")
    values = [n for n in statements[0].named_children
              if n.type not in {"line_comment", "block_comment"}]
    if len(values) != 1 or _text(raw, values[0]).strip() != value.strip():
        raise ValueError("CALLBACK_EXPRESSION_NOT_A_VALUE")
    pending = [values[0]]
    while pending:
        node = pending.pop()
        if node.type in _FORBIDDEN:
            raise ValueError("CALLBACK_EXPRESSION_SIDE_EFFECT_OR_DECLARATION")
        if node.type == "identifier" and _text(raw, node).startswith("__mmmHostEffectApplied"):
            raise ValueError("CALLBACK_EXPRESSION_HOST_OWNED_RECEIPT")
        if predicate and node.type == "field_access":
            field = node.child_by_field_name("field")
            if field is not None and _text(raw, field) == "EVENT":
                # An event listener already runs inside the selected callback.
                # Calling its dispatcher from a condition is re-entrancy, not
                # a read of gameplay state. This is structural, not a version-
                # specific blacklist of Fabric method names.
                raise ValueError("CALLBACK_PREDICATE_EVENT_DISPATCH")
        if predicate and node.type == "method_invocation":
            name = _text(raw, node.child_by_field_name("name"))
            if name.startswith("writeState_") or name == "setState":
                raise ValueError("CALLBACK_PREDICATE_STATE_WRITE")
        pending.extend(node.named_children)
    return value.strip()


def parse_initializer(value: str, *, host_local: str) -> str:
    """Accept an expression or the exact single-local Java declaration form.

    The host has already chosen the local variable identity. A model can
    sometimes emit a complete typed declaration despite a value-only schema.
    Parse it structurally to recover its initializer instead of prompting the
    same model twice with an unworkable source-string correction.
    """
    try:
        return parse_expression(value)
    except ValueError as expression_error:
        try:
            raw, root = _parse("class MmmInitializer { void run() {\n"
                               + value + "\n} }")
        except ValueError:
            raise expression_error
        owner = next((n for n in root.named_children if n.type == "class_declaration"), None)
        if owner is None:
            raise expression_error
        body = owner.child_by_field_name("body")
        methods = [n for n in body.named_children if n.type == "method_declaration"]
        if len(methods) != 1:
            raise expression_error
        statements = [n for n in methods[0].child_by_field_name("body").named_children
                      if n.type not in {"line_comment", "block_comment"}]
        if len(statements) != 1 or statements[0].type != "local_variable_declaration":
            raise expression_error
        variables = [n for n in statements[0].named_children
                     if n.type == "variable_declarator"]
        if len(variables) != 1:
            raise expression_error
        name = variables[0].child_by_field_name("name")
        initializer = variables[0].child_by_field_name("value")
        if name is not None and _text(raw, name).startswith("__mmmHostEffectApplied"):
            # This local is host-owned evidence of an executed gameplay effect,
            # never a model-chosen value or a valid effect in its own right.
            raise ValueError("CALLBACK_HOST_RECEIPT_NOT_AN_EFFECT")
        if (name is None or _text(raw, name) != host_local
                or initializer is None):
            raise expression_error
        return parse_expression(_text(raw, initializer))


def _same_compiled_member(candidate, original):
    """Compare Java syntax tokens, not whitespace, for rejected candidates."""
    return class_body_token_identity(candidate) == class_body_token_identity(original)


def _validate_state(source, frame):
    from .host_typed_state_access import (
        validate_behavior_state_provenance,
        validate_host_state_access,
    )
    contracts = frame.get("host_state_access", ())
    validate_host_state_access(source, contracts)
    validate_behavior_state_provenance(source, contracts, frame.get("host_behavior_contract"))


def _require_canonical_state_effects(source, frame):
    """Prove every authored *canonical* state write is present in Java.

    Typed state storage is already generated by the host. Completing a
    callback with only a chat message, helper variable, or unrelated method
    is not enough when the behavior explicitly owns state_writes. The
    verified accessor owner and the exact target must both match.
    """
    behavior = frame.get("host_behavior_contract") or {}
    required = set(behavior.get("state_writes") or ())
    if not required:
        return
    contracts = frame.get("host_state_access") or ()
    owners = {str(row["owner"]) for row in contracts}
    owners.update(owner.rsplit(".", 1)[-1] for owner in tuple(owners))
    if not owners:
        raise ValueError(
            "CALLBACK_STATE_OWNER_UNAVAILABLE: "
            + ", ".join(sorted(required))
        )
    raw, root = _parse("class MmmEffectCoverage {\n" + source + "\n}")
    applied = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type == "method_invocation":
            name = node.child_by_field_name("name")
            receiver = node.child_by_field_name("object")
            if name is not None and receiver is not None:
                method = _text(raw, name)
                if (method.startswith("writeState_")
                        and _text(raw, receiver) in owners):
                    applied.add(method.removeprefix("writeState_"))
        pending.extend(node.named_children)
    missing = required - applied
    if missing:
        raise ValueError(
            "CALLBACK_CANONICAL_STATE_WRITES_MISSING: "
            + ", ".join(sorted(missing))
        )


def _checked_player_state_context(frame):
    """Bind one *real callback participant* to a player-owned state write.

    A generated player-scoped state throws at runtime when its context lacks
    a player identity. Never guess a player name, use the global no-context
    overload, or let the 9B model invent a UUID. Support the two checked
    Minecraft mappings; other event ABI shapes must first provide a verified
    player-producing source path.
    """
    from .host_verified_callback_ops import _callback_parameters

    parameters = _callback_parameters(frame)
    methods = {
        "net.minecraft.entity.player.PlayerEntity": "getUuid",
        "net.minecraft.server.network.ServerPlayerEntity": "getUuid",
        "net.minecraft.world.entity.player.Player": "getUUID",
        "net.minecraft.server.level.ServerPlayer": "getUUID",
    }
    matched = [(variable, methods[kind]) for kind, variable in parameters
               if kind in methods]
    if len(matched) != 1:
        raise ValueError("CALLBACK_PLAYER_STATE_OWNER_UNRESOLVED: "
                         "one verified Minecraft player callback parameter "
                         "is required, found " + str(len(matched)))
    variable, getter = matched[0]
    return 'java.util.Map.of("player", ' + variable + "." + getter + "())"


def _required_state_write_slots(frame):
    """Resolve exact, already-materialized typed writers before coding starts.

    A planner label is not a Java API. Never infer a state accessor from the
    model's prose or silently redirect a write to a different state owner.
    """
    behavior = frame.get("host_behavior_contract") or {}
    required = sorted(set(behavior.get("state_writes") or ()))
    if not required:
        return {}
    contracts = frame.get("host_state_access") or ()
    slots = {}
    for name in required:
        matches = []
        for contract in contracts:
            owner = str(contract.get("owner") or "")
            if not owner:
                continue
            for method in contract.get("methods") or ():
                if method.get("name") != "writeState_" + name:
                    continue
                signature = str(method.get("signature") or "")
                # Canonical state has both (value) and (value, context)
                # overloads. This lowering intentionally invokes (value)
                # and must NOT infer the value type from the context overload.
                try:
                    raw, parsed = _parse("class MmmStateSignature {\n" + signature + "\n}")
                except ValueError:
                    continue
                owner_node = next((row for row in parsed.named_children
                                   if row.type == "class_declaration"), None)
                if owner_node is None:
                    continue
                methods = [row for row in owner_node.child_by_field_name("body").named_children
                           if row.type == "method_declaration"]
                if len(methods) != 1:
                    continue
                parameters = [p for p in methods[0].child_by_field_name("parameters").named_children
                              if p.type == "formal_parameter"]
                if len(parameters) != 1:
                    continue
                param_type = parameters[0].child_by_field_name("type")
                if param_type is None:
                    continue
                scope = (contract.get("state_scopes") or {}).get(name)
                if scope not in (None, "player"):
                    raise ValueError("CALLBACK_REQUIRED_STATE_SCOPE_UNSUPPORTED: " + name)
                matches.append((owner, signature, _text(raw, param_type), scope))
        if not matches:
            raise ValueError("CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED: " + name)
        if len({(owner, typ, scope) for owner, _, typ, scope in matches}) != 1:
            raise ValueError("CALLBACK_REQUIRED_STATE_OWNER_AMBIGUOUS: " + name)
        owner, signature, value_type, scope = matches[0]
        slots[name] = {"owner": owner, "signature": signature,
                       "value_type": value_type, "scope": scope}
        if scope == "player":
            # The real callback ABI, not model prose, supplies the UUID.
            # Reject impossible ownership before any model choice is made.
            slots[name]["context"] = _checked_player_state_context(frame)
    return slots


def _complete_required_state_effects(
    router, scoped, *, data_name, frame, statements, semantic_statements,
    receipt_name, effect_receipt, slots, typed_values=False,
):
    """Lower each missing mandatory write through one verified typed-value slot.

    The author chooses only a value expression; the host owns state identity,
    receiver, invocation, order and Java syntax. Missing coverage never sends
    the model back to its unbounded free-form operation chooser.
    """
    existing = frame["prefix"] + "\n".join(statements) + frame["suffix"]
    try:
        _require_canonical_state_effects(existing, frame)
        return
    except ValueError as exc:
        prefix = "CALLBACK_CANONICAL_STATE_WRITES_MISSING: "
        if not str(exc).startswith(prefix):
            raise
        missing = [part.strip() for part in str(exc)[len(prefix):].split(",") if part.strip()]
    # A missing state mutation cannot be appended *after* a model-authored
    # conditional block: doing so bypasses the condition and mutates state
    # when the model intentionally skipped the effect. Fail closed instead
    # of generating plausible but behaviorally incorrect Java.
    if any(
        statement.lstrip().startswith(("if (", "while (", "for (", "} else", "switch ("))
        for statement in semantic_statements
    ):
        raise ValueError(
            "CALLBACK_REQUIRED_STATE_WRITE_CONTROL_FLOW_UNRESOLVED: "
            + ", ".join(sorted(missing))
        )
    for name in missing:
        slot = slots[name]
        owner = slot["owner"]
        if typed_values:
            from .host_typed_state_values import (
                lower_typed_java_value, typed_value_schema,
            )
        def lower(row, *, state_name=name, state_owner=owner):
            expression = (lower_typed_java_value(row, slot["value_type"])
                          if typed_values else parse_expression(row["expression"]))
            # Reading a variable and writing it back is a semantic no-op;
            # it cannot satisfy a requested gameplay state transition.
            if expression in {
                state_owner + ".readState_" + state_name + "()",
                state_owner.rsplit(".", 1)[-1] + ".readState_" + state_name + "()",
            }:
                raise ValueError("CALLBACK_CANONICAL_STATE_WRITE_NOOP: " + state_name)
            context_arg = ", " + slot["context"] if slot.get("context") else ""
            operation = (state_owner + ".writeState_" + state_name + "("
                         + expression + context_arg + ");")
            trial = frame["prefix"] + "\n".join([*statements, operation]) + frame["suffix"]
            validate_generated_java_source(trial, frame, frame["prefix"], frame["suffix"])
            _validate_state(trial, frame)
            return operation
        statement = _ask(
            router, scoped, name=data_name + "_required_state_" + name,
            schema=(typed_value_schema(slot["value_type"]) if typed_values
                    else _EXPRESSION_SCHEMA),
            context={
                "instruction": (
                    "Choose the exact typed JSON value for this canonical state "
                    "transition. Do not supply Java syntax, methods, getters, "
                    "statements or quoted Java literals. The host renders Java "
                    "from the JSON primitive. Never substitute an unrelated "
                    "gameplay effect for the requested state change."
                    if typed_values else
                    "Produce ONLY the value expression for this exact required "
                    "canonical state write. The host owns the call, state name "
                    "and receiver. Use current verified callback or state values "
                    "to implement the original assigned behavior, not a no-op."
                ),
                "state_name": name, "verified_writer": slot["signature"],
                "verified_writer_owner": owner,
                "required_java_value_type": slot["value_type"],
                "assigned_behavior": frame.get("host_behavior_contract"),
                "accepted_host_statements": semantic_statements,
            }, lower=lower,
        )
        statements.append(statement)
        semantic_statements.append(statement)
        if effect_receipt:
            statements.append(receipt_name + " = true;")


def _effect_operation(action, code):
    """Only a possible external mutation can satisfy an effect receipt.

    Console/log calls, pure getter queries and local declarations used to
    masquerade as the entire requested gameplay implementation. This gate
    does not claim semantic success; a real playtest remains mandatory.
    """
    import re
    _, root = _parse("class MmmEffect { void run() {\n" + code + "\n} }")
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type == "method_invocation" and action == "call":
            name_node = node.child_by_field_name("name")
            name = _text(_, name_node) if name_node is not None else ""
            expression = _text(_, node)
            # Effect receipts must be based on gameplay state changes, not
            # textual diagnostics. Identifiers here describe Java API kinds,
            # not user prompts or particular Minecraft mechanics.
            logger = bool(re.search(
                r"(?:\bSystem\.(?:out|err)\.|\b(?:logger|log)\.)",
                expression, re.IGNORECASE))
            passive = name.startswith(("get", "is", "has", "can", "peek", "read"))
            if not logger and not passive and name not in {
                    "print", "println", "printf", "log", "debug", "info",
                    "warn", "warning", "error", "trace"}:
                return True
        if action == "assign" and node.type == "assignment_expression":
            target = node.child_by_field_name("left")
            if target is not None and target.type in {"field_access", "array_access"}:
                return True
        pending.extend(node.named_children)
    return False


def _ask(router, messages, *, name, schema, context, lower):
    rejected, error = None, ""
    for attempt in range(2):
        payload = {**context, "last_error": error, "rejected_payload": rejected}
        response = None
        try:
            response = router.generate_structured_data(
                "coder", [*messages, {"role": "user", "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"))}],
                data_name=f"{name}_try_{attempt}", parameters=schema,
            )
            checked = _validate_native_arguments(response, schema)
            return lower(checked)
        except (ValueError, StructuredOutputValidationError) as exc:
            rejected, error = response if response is not None else getattr(exc, "output", None), str(exc)
            if attempt:
                failure = ValueError("CALLBACK_OPERATION_INVALID: " + error)
                failure.failed_response = json.dumps(response, ensure_ascii=False)
                raise failure from exc
    raise AssertionError("unreachable")


def lower_operation_from_host_verified_call(expression, frame, statements, blocks):
    """Admit a host-assembled verified invocation into its actual body context."""
    code = expression + ";"
    source = (frame["prefix"] + "\n".join([*statements, code])
              + "\n}" * len(blocks) + frame["suffix"])
    validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
    _validate_state(source, frame)
    return code


def _choose_verified_expression(router, scoped, *, name, choices, context, frame,
                                rejected_compiler_member=None):
    by_id = {row["id"]: row["expression"] for row in choices}

    def lower(row):
        expression = by_id[row["verified_call_id"]]
        if context.get("predicate"):
            validate_runtime_observation_predicate(scoped, expression)
            source = frame["prefix"] + "return " + expression + ";" + frame["suffix"]
        else:
            source = frame["prefix"] + expression + ";" + frame["suffix"]
        if rejected_compiler_member is not None and _same_compiled_member(source, rejected_compiler_member):
            raise ValueError("CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED")
        _validate_state(source, frame)
        return validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])

    return _ask(router, scoped, name=name, schema=_object({
        "verified_call_id": {"type": "string", "enum": list(by_id)}}),
        context={**context, "verified_candidates": choices}, lower=lower)


def validate_runtime_observation_predicate(scoped, expression):
    """Require a live callback receiver when a source producer is still pending.

    A constant predicate cannot certify an undefined world/UI observation. This
    check is only necessary grounding, not a semantic equivalence proof:
    real gameplay still requires integration tests.
    """
    pending = []
    parameters = []
    observed_source = ""
    for message in reversed(scoped):
        if message.get("role") != "user":
            continue
        try:
            task = json.loads(message.get("content") or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(task, dict):
            continue
        work = task.get("source_owned_work")
        if isinstance(work, dict):
            observed_source = str(task.get("observed_source_name") or "")
            pending = [row for row in work.get("source_tasks", ())
                       if row.get("kind") == "runtime_producer"
                       and row.get("status") != "verified"]
            metadata = task.get("declaration_metadata") or {}
            parameters = [row.get("name") for row in metadata.get("parameters", ())
                          if isinstance(row, dict) and isinstance(row.get("name"), str)]
            break
    if not pending:
        return
    # Access to a real event/world receiver is the minimum prerequisite for
    # producing a runtime value. Strings, constants and undeclared state keys
    # do not count, even if such Java syntax compiles.
    expression = parse_expression(expression, predicate=True)
    raw, root = _parse("class MmmObservation { boolean check() { return "
                       + expression + "; } }")
    found = False
    visit = [root]
    while visit:
        node = visit.pop()
        if node.type == "identifier" and _text(raw, node) in parameters:
            found = True
        visit.extend(node.named_children)
    if not found:
        names = sorted({str(row.get("name") or row.get("task_id") or "")
                        for row in pending})
        raise ValueError("SOURCE_RUNTIME_OBSERVATION_NOT_GROUNDED: "
                         + json.dumps(names, ensure_ascii=False))
    if observed_source:
        # A world-state producer must read something from the actual event
        # object, not simply return true or check a non-null parameter. This
        # is necessary, not sufficient: compiling and playing the generated
        # feature remains the final semantic acceptance test.
        visit = [root]
        accessed = False
        while visit:
            node = visit.pop()
            if node.type in {"method_invocation", "field_access", "array_access"}:
                accessed = True
            visit.extend(node.named_children)
        if not accessed:
            raise ValueError(
                "SOURCE_RUNTIME_PRODUCER_NOT_IMPLEMENTED: " + observed_source
            )


def _author_predicate(router, scoped, name, frame, correction, verified, rejected):
    if verified:
        return _choose_verified_expression(
            router, scoped, name=name + "_verified_predicate",
            choices=verified, context={**correction, "predicate": True},
            frame=frame, rejected_compiler_member=rejected)

    def lower(row):
        expression = parse_expression(row["expression"], predicate=True)
        validate_runtime_observation_predicate(scoped, expression)
        source = frame["prefix"] + "return " + expression + ";" + frame["suffix"]
        if rejected is not None and _same_compiled_member(source, rejected):
            raise ValueError("CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED")
        _validate_state(source, frame)
        return validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])

    return _ask(router, scoped, name=name + "_predicate", schema=_EXPRESSION_SCHEMA,
                context={**correction, "instruction": (
                    "One pure boolean expression for the original trigger and preconditions.")},
                lower=lower)


def _author_operation(router, scoped, name, index, action, context, verified_calls,
                      frame, statements, blocks, new_local, lower_operation):
    if action == "call" and verified_calls:
        expressions = {row["id"]: row["expression"] for row in verified_calls}
        selected = _ask(
            router, scoped, name=f"{name}_operation_{index}_verified_call",
            schema=_object({"verified_call_id": {
                "type": "string", "enum": list(expressions)}}),
            context={**context, "action": action,
                     "verified_candidates": verified_calls},
            lower=lambda row: expressions[row["verified_call_id"]],
        )
        return lower_operation_from_host_verified_call(selected, frame, statements, blocks)
    return _ask(router, scoped, name=f"{name}_operation_{index}_{action}",
                schema=_SCHEMAS[action], context={**context, "action": action,
                    "host_local_name": new_local if action in {"declare", "foreach"} else None},
                lower=lower_operation)


def _native_callback_choices(scoped, frame, runtime_targets, verified_calls):
    if not runtime_targets:
        return verified_calls, []
    if "host_verified_runtime_mutation_candidates" in frame:
        return [], frame["host_verified_runtime_mutation_candidates"]
    from .host_verified_callback_ops import verified_runtime_mutation_candidates
    return verified_calls, verified_runtime_mutation_candidates(scoped, frame)


def author_indexed_callback_body(router, messages, *, data_name, frame, api_index,
                                 task, predicate=False, effect_receipt=False):
    """Attach host authority after message serialization, then author one body."""
    if effect_receipt and frame["host_behavior_contract"].get("runtime_state_writes"):
        from .host_runtime_mutation_catalog import cached_runtime_mutation_catalog
        frame["host_verified_runtime_mutation_candidates"] = cached_runtime_mutation_catalog(
            api_index, task, frame)
    return author_callback_body(router, messages, data_name=data_name, frame=frame,
                                predicate=predicate, effect_receipt=effect_receipt)


def author_callback_body(router, messages, *, data_name, frame, predicate=False,
                         effect_receipt=False, rejected_compiler_member=None,
                         compiler_diagnostic="") -> str:
    """Author expressions and operation parameters, never Java statements."""
    from .construction_java_context import scoped_java_member_messages
    scoped = list(scoped_java_member_messages(messages, frame))
    # Remove old source/statement transport instructions, retaining their
    # authoritative user payloads and the single coherent compiler protocol.
    scoped = [m for m in scoped if m.get("role") != "system"]
    instruction = (
        "The host compiles your semantic choices into Java. Supply only the requested "
        "JSON values, never statements, semicolons, return keywords, braces, method "
        "declarations or imports. Use actual callback parameters, inspected APIs and "
        "typed canonical state accessors. Preserve the assigned trigger and effects. "
        "runtime_state_reads/runtime_state_writes and runtime_state_observations refer "
        "to actual event/world owners, never invented host state accessors; implement "
        "their reads and mutations through verified callback/world APIs. "
        "Each expression is one value; lambdas and embedded assignments are forbidden. "
        "A predicate executes INSIDE an event listener; never register or redispatch an event bus. "
        "For call choose receiver, method name and argument values; the host writes "
        "the invocation. For assign choose an existing field/array target, operator "
        "and value. Declare creates a host-named local from your initializer. "
        "if/while/foreach open host-owned blocks; end closes one block; else opens "
        "the alternative of an if. Complete only after ALL assigned effects are "
        "implemented. Callback results and returns are owned by the host and cannot "
        "be supplied as effects. Do not replace missing state with an unrelated key."
    )
    scoped.insert(0, {"role": "system", "content": instruction})
    correction = {
        "javac_rejected_member": rejected_compiler_member,
        "javac_diagnostic": compiler_diagnostic,
    } if rejected_compiler_member is not None else {}
    verified = verified_callback_choices(scoped, frame, predicate=predicate)
    if predicate:
        return _author_predicate(router, scoped, data_name, frame, correction,
                                 verified, rejected_compiler_member)

    # Resolve the mandatory state ABI before asking the model for ANY
    # free-form operations. Unmaterialized state is a planning defect, not
    # something 64 more small-model attempts can fix.
    # Only the host-owned effect method is responsible for state mutation.
    # Observer and predicate helpers for the same behavior must never be
    # forced to write a state just to finish their own scoped task.
    state_slots = _required_state_write_slots(frame) if effect_receipt else {}
    from .runtime_mutation_author import mutation_targets
    runtime_targets = (mutation_targets(frame.get("host_behavior_contract"))
                       if effect_receipt else ())
    statements, blocks, locals_ = [], [], []
    semantic_statements = []
    verified_calls = verified_callback_choices(scoped, frame, predicate=False)
    native_verified_calls, typed_mutation_candidates = _native_callback_choices(
        scoped, frame, runtime_targets, verified_calls)
    if runtime_targets and not (native_verified_calls or typed_mutation_candidates):
        # This is missing implementation authority, NOT permission to infer
        # arbitrary Java source or claim success with a disconnected scalar.
        raise ValueError(
            "CALLBACK_RUNTIME_MUTATION_VERIFIED_API_UNAVAILABLE: "
            + ", ".join(runtime_targets))
    # An inspected callback already has a concrete receiver/argument ABI.
    # Free-form receiver+method JSON cannot be trusted as a gameplay call:
    # production log 34 invented ItemEvents.register(... UseOnCallback ...).
    # In this mode only host-assembled, bytecode-indexed call IDs are admitted.
    evidence_bound = False
    for message in scoped:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("verified_event_callback"), dict
        ):
            evidence_bound = True
            break
    receipt_name = "__mmmHostEffectApplied"
    while receipt_name in frame["prefix"]:
        receipt_name += "_"
    if effect_receipt:
        statements.append(f"boolean {receipt_name} = false;")

    # Real, ABI-bound production callbacks with explicit write obligations
    # are lowered as host-owned operations, not through an unbounded choice
    # of unrelated Java calls. The previous path accepted repeated
    # source.getPlayer().removeAllEffects() as a "gameplay effect" before
    # asking a 9B model to supply a boolean using arbitrary Java source.
    if effect_receipt and evidence_bound and (state_slots or runtime_targets):
        completed = list(statements)
        if state_slots:
            _complete_required_state_effects(
                router, scoped, data_name=data_name, frame=frame,
                statements=completed, semantic_statements=semantic_statements,
                receipt_name=receipt_name, effect_receipt=True, slots=state_slots,
                typed_values=True,
            )
        if runtime_targets:
            from .runtime_mutation_author import author_runtime_mutation_calls
            author_runtime_mutation_calls(
                router, scoped, data_name=data_name, frame=frame,
                verified_calls=native_verified_calls,
                typed_candidates=typed_mutation_candidates,
                targets=runtime_targets, statements=completed,
                semantic_statements=semantic_statements,
                effect_receipt=True, receipt_name=receipt_name,
            )
        completed.append(f"return {receipt_name};")
        source = frame["prefix"] + "\n".join(completed) + frame["suffix"]
        if rejected_compiler_member is not None and _same_compiled_member(
            source, rejected_compiler_member
        ):
            raise ValueError("CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED")
        if state_slots:
            _require_canonical_state_effects(source, frame)
        return validate_generated_java_source(
            source, frame, frame["prefix"], frame["suffix"]
        )

    effects = 0
    blocked_operations: set[str] = set()
    for index in range(64):
        # A gameplay transition with mandatory writes is an atomic EFFECT.
        # Its activation and preconditions live in the separately authored
        # predicate/observer methods. Allowing the model to wrap required
        # mutations in local if/while branches caused false success or late
        # unconditional writes. Prohibit that unbounded control flow here.
        supported_actions = (
            ("call", "assign", "declare")
            if state_slots or runtime_targets else
            ("call", "assign", "declare", "if", "while", "foreach")
        )
        choices = [kind for kind in supported_actions
                   if kind not in blocked_operations
                   and (kind != "call" or verified_calls or not evidence_bound)]
        if blocks:
            choices.append("end")
            if blocks[-1] == "if":
                choices.append("else")
        if any(b in {"while", "foreach"} for b in blocks):
            choices.extend(["break", "continue"])
        if (effects or state_slots or runtime_targets) and not blocks:
            choices.append("complete")
        context = {**correction, "accepted_host_statements": semantic_statements, "host_locals": locals_,
                   "open_blocks": blocks, "next_operation": index,
                   "blocked_non_effect_operations": sorted(blocked_operations)}
        action = _ask(router, scoped, name=f"{data_name}_operation_{index}",
                      schema=_object({"action": {"type": "string", "enum": choices}}),
                      context=context, lower=lambda row: row["action"])
        if action == "complete":
            completed = [*statements]
            if state_slots:
                _complete_required_state_effects(
                    router, scoped, data_name=data_name, frame=frame,
                    statements=completed, semantic_statements=semantic_statements,
                    receipt_name=receipt_name, effect_receipt=effect_receipt,
                    slots=state_slots,
                )
            if runtime_targets:
                # Appending mutations after an authored branch would silently
                # execute writes even when its condition was false. This must
                # become a separate host-authored guarded effect work item,
                # not a source repair through unconditional invocation.
                if any(
                    statement.lstrip().startswith((
                        "if (", "while (", "for (", "} else", "switch ("))
                    for statement in semantic_statements
                ):
                    raise ValueError(
                        "CALLBACK_RUNTIME_MUTATION_CONTROL_FLOW_UNRESOLVED: "
                        + ", ".join(runtime_targets))
                from .runtime_mutation_author import author_runtime_mutation_calls
                author_runtime_mutation_calls(
                    router, scoped, data_name=data_name, frame=frame,
                    verified_calls=native_verified_calls,
                    typed_candidates=typed_mutation_candidates,
                    targets=runtime_targets,
                    statements=completed,
                    semantic_statements=semantic_statements,
                    effect_receipt=effect_receipt, receipt_name=receipt_name,
                )
            if effect_receipt:
                completed.append(f"return {receipt_name};")
            source = frame["prefix"] + "\n".join(completed) + frame["suffix"]
            if rejected_compiler_member is not None and _same_compiled_member(source, rejected_compiler_member):
                raise ValueError("CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED")
            if state_slots:
                _require_canonical_state_effects(source, frame)
            return validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
        new_local = f"mmmValue{len(locals_)}"

        def lower_operation(row, action=action, new_local=new_local):
            if action == "call":
                receiver = parse_expression(row["receiver"])
                method = identifier(row["method"])
                arguments = ", ".join(parse_expression(v) for v in row["arguments"])
                code = f"{receiver}.{method}({arguments});"
            elif action == "assign":
                target = parse_expression(row["target"])
                value = parse_expression(row["value"])
                code = f"{target} {row['operator']} {value};"
            else:
                expression = (parse_initializer(row["expression"], host_local=new_local)
                              if action == "declare" else parse_expression(
                                  row["expression"], predicate=action in {"if", "while"}))
                if action == "declare":
                    code = f"var {new_local} = {expression};"
                elif action == "foreach":
                    code = f"for (var {new_local} : {expression}) {{"
                else:
                    code = f"{action} ({expression}) {{"
            # Check a balanced trial, including all already accepted blocks.
            trial_blocks = len(blocks) + int(action in {"if", "while", "foreach"})
            source = (frame["prefix"] + "\n".join([*statements, code])
                      + "\n}" * trial_blocks + frame["suffix"])
            validate_generated_java_source(source, frame, frame["prefix"], frame["suffix"])
            _validate_state(source, frame)
            return code

        if action in _SCHEMAS:
            try:
                code = _author_operation(
                    router, scoped, data_name, index, action, context, verified_calls,
                    frame, statements, blocks, new_local, lower_operation)
            except ValueError as exc:
                if (action == "declare"
                        and "CALLBACK_HOST_RECEIPT_NOT_AN_EFFECT" in str(exc)):
                    # A captured 9B response repeatedly declared the private
                    # effect receipt instead of mutating game state. Repeating
                    # the same value-only slot cannot solve that ownership
                    # mismatch. Disallow this non-effect action and ask for a
                    # different semantic operation; never mark success here.
                    blocked_operations.add("declare")
                    continue
                raise
            statements.append(code)
            semantic_statements.append(code)
            # A local declaration is an intermediate value, never a
            # gameplay write. Previously `var temp = player.someMethod()`
            # could set the host success receipt and terminate the feature
            # without mutating its requested game state.
            effect_attempted = action in {"call", "assign"} and _effect_operation(action, code)
            effects += int(effect_attempted)
            if effect_receipt and effect_attempted:
                statements.append(f"{receipt_name} = true;")
            if action in {"declare", "foreach"}:
                locals_.append(new_local)
            if action in {"if", "while", "foreach"}:
                blocks.append(action)
        elif action == "end":
            blocks.pop()
            statements.append("}")
            semantic_statements.append("}")
        elif action == "else":
            blocks[-1] = "else"
            statements.append("} else {")
            semantic_statements.append("} else {")
        else:
            statements.append(action + ";")
            semantic_statements.append(action + ";")
    raise ValueError("CALLBACK_OPERATION_LIMIT: assigned effect requires decomposition")


__all__ = ["author_callback_body", "parse_expression"]

