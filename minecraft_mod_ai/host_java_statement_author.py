"""Host-owned statement AST authoring for Fabric callback implementations.

The small coder never authors a method declaration or compilation unit. It
chooses one Java statement at a time through a closed native JSON schema. The
host parses every statement with Tree-sitter, assembles the fixed signature and
checks typed state access before returning to the existing javac admission.
This deliberately replaces the free-form JSON 'implementation' blob for
request-feature callback handlers, not the entire Java toolchain.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .construction_source_repair import validate_generated_java_source
from .java_region_parser import _parse


_STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        # Empty is a terminal marker; source never contains this sentinel.
        "statement": {"type": "string", "maxLength": 2048},
    },
    "required": ["statement"],
    "additionalProperties": False,
}
_MAX_MEMBER_STATEMENTS = 32
_COMMENT_NODES = frozenset({"line_comment", "block_comment"})


def parse_one_java_statement(fragment: str) -> str:
    """Admit exactly ONE block-level Java statement, never other members.

    This uses the same tree-sitter-java grammar as the production source
    validator. Neither regex matching nor string suffix trimming can turn
    a method declaration into an authorized statement.
    """
    if not isinstance(fragment, str) or not fragment.strip():
        raise ValueError("JAVA_STATEMENT_REQUIRED")
    _, root = _parse("class MmmStatementFrame { void f() {\n" + fragment + "\n} }")
    owner = next((n for n in root.named_children if n.type == "class_declaration"), None)
    class_body = owner.child_by_field_name("body") if owner is not None else None
    methods = [n for n in class_body.named_children
               if n.type == "method_declaration"] if class_body is not None else []
    if len(methods) != 1:
        raise ValueError("JAVA_STATEMENT_HOST_METHOD_SCOPE")
    block = methods[0].child_by_field_name("body")
    statements = [n for n in block.named_children if n.type not in _COMMENT_NODES
                  ] if block is not None else []
    if len(statements) != 1:
        raise ValueError(
            f"JAVA_STATEMENT_ATOMICITY: expected one statement, got {len(statements)}")
    statement = statements[0]
    if statement.type in {
        "class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "method_declaration",
        "import_declaration", "package_declaration", "module_declaration",
    }:
        raise ValueError("JAVA_STATEMENT_HOST_DECLARATION_OWNED")
    # The parser's statement starts after the wrapper. Accept whitespace and
    # source comments without interpreting or mutating user Java expressions.
    return fragment.strip()



def _top_level_terminates_method(fragment: str) -> bool:
    """Stop after an unconditional, top-level return or throw.

    A return nested in an if/loop is conditional, so the caller may continue.
    The parser, not text prefix detection, owns the distinction.
    """
    _, root = _parse("class MmmStatementFrame { void f() {\n" + fragment + "\n} }")
    owner = next(n for n in root.named_children if n.type == "class_declaration")
    method = next(n for n in owner.child_by_field_name("body").named_children
                  if n.type == "method_declaration")
    statements = [n for n in method.child_by_field_name("body").named_children
                  if n.type not in _COMMENT_NODES]
    return len(statements) == 1 and statements[0].type in {
        "return_statement", "throw_statement",
    }


def _executable_effect(source: str) -> bool:
    """Prevent an effects handler from being admitted as an empty return stub."""
    _, root = _parse("class MmmEffectFrame { void f() {\n" + source + "\n} }")
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type in {
            "method_invocation", "assignment_expression", "update_expression",
            "object_creation_expression", "throw_statement",
        }:
            return True
        pending.extend(node.named_children)
    return False


def author_java_callback_statements(
    router: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    data_name: str,
    frame: Mapping[str, Any],
    require_effect: bool,
) -> str:
    """Generate a bounded AST statement sequence with a durable journal.

    Every source statement has a separate closed schema and checkpoint. A bad
    fragment receives ONE grammar-guided correction rather than replaying a
    complete invalid method with ever more project source.
    """
    from .construction_java_context import scoped_java_member_messages

    scoped = list(scoped_java_member_messages(messages, frame))
    # The context projector emits the old whole-member directive. Replace
    # rather than append it: the model must see ONE consistent transport
    # contract, with system-first chat template order preserved.
    scoped = [
        {"role": "system", "content": (
            "Author exactly ONE Java statement in an existing host-owned "
            "method; the JSON field is 'statement', NOT 'implementation'. "
            "Preserve the original assigned behavior, verified callback ABI "
            "and state ownership. Do not invent methods, imports or classes."
        )} if (message.get("role") == "system"
              and '"implementation"' in message.get("content", ""))
        else message
        for message in scoped
    ]
    statements: list[str] = []
    # A tiny model often emits the callback's PASS/FAIL result before doing
    # any effect. The host can postpone that terminal statement without
    # treating it as a completed implementation or discarding the request.
    # Neither the return nor a no-op can satisfy an effect contract.
    deferred_return: str | None = None
    premature_returns = 0
    for index in range(_MAX_MEMBER_STATEMENTS):
        last_error = ""
        for correction in range(2):
            prompt = [
                *scoped,
                {"role": "system", "content": (
                    "You are authoring ONE Java statement inside an EXISTING "
                    "method, not a class, method, field or import. Return a JSON "
                    "object with only the key 'statement'. The statement must "
                    "be a SINGLE complete Java block statement including its "
                    "semicolon or its matching braces. To finish, return an "
                    "EMPTY statement, only after every assigned behavioral "
                    "effect has been implemented. The immutable method signature "
                    "and event registration are owned by the host. Never output "
                    "method signatures, imports, nested class declarations or "
                    "additional independent statements. Do not invent state "
                    "accessors or replace one state with another. "
                    "For a behavior EFFECT handler, do not return PASS, FAIL, "
                    "SUCCESS or any other result before implementing the "
                    "specific effect using verified APIs. A return is not an "
                    "effect. A top-level return TERMINATES the method: the host "
                    "will not ask for any further statements after it. "
                    "If a return has been deferred by the host, write the "
                    "REAL effect as the NEXT statement, using verified APIs, "
                    "rather than another return or empty implementation. "
                    "Do not repeat a previous statement."
                )},
                {"role": "user", "content": json.dumps({
                    "immutable_signature": {
                        "kind": frame.get("kind"), "prefix": frame.get("prefix"),
                        "suffix": frame.get("suffix"),
                    },
                    "already_accepted_statements": statements,
                    "deferred_terminal": deferred_return,
                    "effect_required_before_terminal": bool(require_effect and deferred_return),
                    "next_statement_index": index,
                    "last_error": last_error,
                    "instruction": ("Correct the rejected statement only."
                                    if last_error else
                                    "Write the next atomic statement, or finish."),
                }, ensure_ascii=False, separators=(",", ":"))},
            ]
            response = router.generate_structured_data(
                "coder", prompt,
                data_name=f"{data_name}_statement_{index}_try_{correction}",
                parameters=_STATEMENT_SCHEMA,
            )
            candidate = response["statement"]
            if not candidate.strip():
                if not statements:
                    raise ValueError(
                        ("JAVA_CALLBACK_EFFECT_REQUIRED: no concrete gameplay "
                         "effect has been authored" if require_effect else
                         "JAVA_CALLBACK_EMPTY_IMPLEMENTATION: no behavior was authored")
                    )
                combined = "\n".join(statements)
                if require_effect and not _executable_effect(combined):
                    raise ValueError(
                        "JAVA_CALLBACK_EFFECT_REQUIRED: return-only or empty "
                        "implementation does not implement the assigned effect")
                if deferred_return is not None:
                    combined += "\n" + deferred_return
                source = frame["prefix"] + combined + frame["suffix"]
                return validate_generated_java_source(
                    source, frame, frame["prefix"], frame["suffix"])
            deferred_now = False
            try:
                statement = parse_one_java_statement(candidate)
                if statement in statements:
                    raise ValueError(
                        "JAVA_CALLBACK_NO_SEMANTIC_PROGRESS: duplicate statement; "
                        "choose a different verified action or finish")
                terminal = _top_level_terminates_method(statement)
                if (terminal and require_effect
                        and not _executable_effect("\n".join(statements))):
                    if deferred_return is None:
                        deferred_return = statement
                    premature_returns += 1
                    if premature_returns >= 3:
                        raise ValueError(
                            "JAVA_CALLBACK_EFFECT_NOT_AUTHORED: the model emitted "
                            "three effect-free returns instead of verified "
                            "world/state mutations; no no-op source is accepted")
                    deferred_now = True
                if deferred_now:
                    # Keep the host-validated terminal OUTSIDE executable code
                    # until the model actually authors the assigned effect.
                    # Continue the next statement slot with explicit ABI context.
                    pass
                # Reject a raw state accessor mistake before the next generated
                # statement; never allow it to drift into a different accessor.
                if not deferred_now:
                    trial = frame["prefix"] + "\n".join([*statements, statement]) + frame["suffix"]
                    from .host_typed_state_access import (
                        validate_host_state_access, validate_behavior_state_provenance,
                    )
                    contracts = frame.get("host_state_access", ())
                    validate_host_state_access(trial, contracts)
                    validate_behavior_state_provenance(
                        trial, contracts, frame.get("host_behavior_contract"),
                    )
            except ValueError as exc:
                last_error = (f"{type(exc).__name__}: {exc}; "
                              "one block-level Java statement only")
                if correction:
                    raise ValueError("JAVA_CALLBACK_ATOMIC_STATEMENT_INVALID: " + last_error) from exc
                continue
            if deferred_now:
                break
            statements.append(statement)
            if terminal:
                source = frame["prefix"] + "\n".join(statements) + frame["suffix"]
                return validate_generated_java_source(
                    source, frame, frame["prefix"], frame["suffix"])
            break
    raise ValueError(
        "JAVA_CALLBACK_STATEMENT_PLAN_TOO_LARGE: decompose the assigned "
        "behavior into separately declared host members instead of silently "
        "truncating Java output")


__all__ = ["parse_one_java_statement", "author_java_callback_statements"]
