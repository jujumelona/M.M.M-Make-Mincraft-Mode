"""Sequential source construction: the model authors operations, the host owns Java."""
from __future__ import annotations

import json
from collections import deque
from contextlib import contextmanager
from copy import deepcopy

from .incremental_java_model import initial_program, initial_scope, allocation_counters
from jsonschema import Draft202012Validator
from .incremental_java_context import IncrementalJavaContext
from .incremental_java_schema import COUNT, DECLARATION_OPS, SCHEMAS, choice, obj, BOOL, operation_options
from .incremental_java_render import render_program
from .incremental_java_history import operation_view, walk_nodes, lookup_work, PAGE

_INSTRUCTION = (
    "Implement only the current operation of the current work item. The host owns all Java "
    "syntax, class/lifecycle wrappers, member names, local names and block boundaries. "
    "Choose one operation, then fill its operand page. Never return a method, class, imports or a statement string. "
    "requested_page identifies exactly which choice/count/details to provide now; kind "
    "asks for the NEXT operation after accepted_operations, not a repetition of the previous one. "
    "Use field/method/type to CREATE new supporting code when needed; "
    "those declarations immediately enter generated_declarations and their bodies are queued. "
    "The host implements newly declared bodies before returning to their caller. "
    "last_declaration reports the actual created or reused signature and implementation status; "
    "use call and select the completed helper's member_id to connect the result. When next_action is "
    "implement_current_work, implement the current body directly instead of delegating it back to an active helper. "
    "Repeating a declaration does not implement a call. "
    "value_kind chooses a value's form; value_stack shows its pending operands and accepted choices. "
    "literal.value is typed data, never Java source. Calls and fields select member_id from member_candidates. "
    "new_helper creates and implements missing supporting work. type_stack identifies the type being "
    "constructed, its argument index and earlier arguments; @lookup discovers additional API types. "
    "A missing gameplay implementation is coding work, never SOURCE_REQUIRED. lookup reads "
    "exact platform/project API signatures; it is not a search for a completed feature. "
    "register installs an actual indexed event and creates a separately authored listener; "
    "that accepted registration already connects the completed listener; do not call it again to install it. "
    "its inputs need not have the same name as the gameplay concept. Implement the original "
    "trigger's conditions and effects in that listener. lambda creates a typed local value "
    "with its own body; pass its host name as an argument. Preserve existing UI/action "
    "ownership and all behavior conditions. Complete a block only after its assigned work "
    "is implemented. Reuse the host names shown in scope and generated_declarations. "
    "lookup @operations with an offset reads earlier accepted operations of this block; "
    "lookup @node:ID reads an accepted block's operands and paginated children. "
    "lookup @declarations with an offset reads all earlier generated signatures; "
    "@declarations:words searches their names, owners and purposes. Recent declarations "
    "are already visible; an older declaration on another page still exists and is callable. "
    "Earlier work remains in the host AST; do not repeat it just because it is on another page."
)


class IncrementalJavaBuilder:
    def __init__(self, router, spec, messages, context, behaviors, program=None):
        self.router, self.spec, self.context = router, spec, context
        state = context.setdefault("incremental_state", {})
        self.api = state.get("api") or IncrementalJavaContext(messages, spec, context, behaviors)
        state["api"] = self.api
        self.program = deepcopy(program) if program is not None else initial_program(spec, behaviors)
        self.pending = deque(self.program["members"] if program is None else ())
        self.serial, self.latest_api, self.rejection = 0, {}, {}
        self.counters = allocation_counters(self.program)
        self.member, self.scope, self.path, self.block = None, {}, [], []
        self.current_operation = ""
        self.drafts, self.declaration_draft = [], {}
        self.value_stack, self.type_stack = [], []
        self.last_declaration = {}
        self.history = {n["id"]: n for m in self.program["members"] for n in walk_nodes(m.get("body", []))}
        self.identities = {m["id"] for m in self.program["members"]} | self.history.keys()
        from .source_ui_callback_authoring import seed_source_ui_callbacks
        seed_source_ui_callbacks(self, initial=program is None)

    def allocate(self, prefix):
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        candidate = prefix + str(self.counters[prefix])
        while candidate in self.identities:
            self.counters[prefix] += 1
            candidate = prefix + str(self.counters[prefix])
        self.identities.add(candidate)
        return candidate

    def payload(self, extra=None):
        task = self.api.view(self.member, self.scope, self.program, position=len(self.block))
        task.update({"host_incremental_java": True, "current_operation": self.current_operation,
                     "operation_stack": [operation_view(node) for node in self.drafts],
                     "current_declaration": self.declaration_draft,
                     "block_path": self.path, "accepted_operations": [operation_view(n) for n in self.block[-PAGE:]],
                     "accepted_operations_total": len(self.block),
                     "accepted_operations_offset": max(0, len(self.block) - PAGE),
                     "requested_api_page": self.latest_api})
        task.update(value_stack=self.value_stack, type_stack=self.type_stack)
        from .incremental_java_progress import declaration_feedback
        task["last_declaration"] = declaration_feedback(self)
        task.update(self.rejection)
        task.update(extra or {})
        return task

    def ask(self, schema, suffix, extra=None):
        from .model_concurrency import raise_if_model_execution_cancelled
        raise_if_model_execution_cancelled()
        self.serial += 1
        messages = [{"role": "system", "content": _INSTRUCTION},
                    {"role": "user", "content": json.dumps(self.payload({"requested_page": suffix, **(extra or {})}), ensure_ascii=False, separators=(",", ":"))}]
        result = self.router.generate_structured_data("coder", messages,
            data_name=f"incremental_{self.member['id']}_{self.serial}_{suffix}", parameters=schema)
        Draft202012Validator(schema).validate(result)
        return result

    def checked(self, schema, suffix, validate, extra=None):
        previous = self.rejection
        for attempt in range(2):
            result = self.ask(schema, suffix, extra)
            try:
                validate(result)
                self.rejection = previous
                return result
            except ValueError as exc:
                self.rejection = {**previous, "operand_diagnostic": str(exc), "rejected_operand_page": result}
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def typed(self, schema, suffix, *fields, extra=None, **type_options):
        from .incremental_java_type_choices import author_typed_metadata
        return author_typed_metadata(self, schema, suffix, fields, extra=extra, **type_options)

    def select_java_type(self, suffix, **options):
        from .incremental_java_type_choices import select_java_type
        return select_java_type(self, suffix, **options)

    @contextmanager
    def declaration_scope(self, member):
        previous = self.member, self.scope, self.path, self.block, self.drafts, self.value_stack, self.type_stack
        self.member, self.scope, self.path, self.block, self.drafts = member, initial_scope(member), [], [], []
        self.value_stack, self.type_stack = [], []
        try:
            yield
        finally:
            self.member, self.scope, self.path, self.block, self.drafts, self.value_stack, self.type_stack = previous

    def expression(self, slot, expected="java.lang.Object", *, allow_update=False):
        from .incremental_java_values import author_value
        return author_value(self, slot, expected, allow_update=allow_update)

    def parameters(self):
        count = self.ask(COUNT, "parameters_count")["count"]
        result = []
        for index in range(count):
            part = self.typed(SCHEMAS["parameter"], "parameter", "type",
                              extra={"parameter_index": index}, allow_varargs=index == count - 1)
            result.append({"type": part["type"], "name": "arg" + str(index), "purpose": part["purpose"]})
        return result

    def arguments(self, node):
        from .incremental_java_calls import author_call_arguments
        author_call_arguments(self, node)
        return node["arguments"]

    def enabled(self, suffix):
        return self.ask(obj({"enabled": BOOL}), suffix)["enabled"]

    def lookup(self):
        query = self.ask(SCHEMAS["lookup"], "lookup")
        self.latest_api = lookup_work(self, query["query"], query["offset"])
        if self.latest_api is None:
            self.latest_api = self.api.lookup(query["query"], query["offset"])

    def new_method(self, *, owner, purpose, behavior, metadata, name=None, kind="method", reuse=True):
        from .incremental_java_progress import declare_method
        return declare_method(self, owner=owner, purpose=purpose, behavior=behavior, metadata=metadata,
                              name=name, kind=kind, reuse=reuse)

    def author_block(self, member, scope, path, body=None):
        from .incremental_java_operations import author_operation
        from .incremental_java_declarations import author_declaration
        from .incremental_java_progress import finish_new_members
        previous = self.member, self.scope, self.path, self.block, self.current_operation
        body = [] if body is None else body
        self.member, self.scope, self.path, self.block = member, scope, path, body
        blocked_declarations = set()
        while True:
            options = operation_options(member, scope, body, blocked_declarations)
            self.current_operation = ""
            operation = self.ask(choice("op", options), "kind")["op"]
            self.current_operation = operation
            if operation == "complete":
                break
            if operation == "lookup":
                self.lookup()
                continue
            if operation in {*DECLARATION_OPS, "constructor", "override"}:
                pending_before = len(self.pending)
                author_declaration(self, operation)
                finish_new_members(self, pending_before)
                if self.last_declaration.get("outcome") in {"reused", "active"}:
                    blocked_declarations.add(operation)
                continue
            pending_before = len(self.pending)
            node = author_operation(self, operation)
            if node is None:
                continue
            body.append(node)
            self.history[node["id"]] = node
            self.rejection = {}
            blocked_declarations.clear()
            finish_new_members(self, pending_before)
            if operation in {"return", "throw", "break", "continue"}:
                break
        self.member, self.scope, self.path, self.block, self.current_operation = previous
        return body

    def finish(self):
        from .incremental_java_progress import finish_member
        while self.pending:
            finish_member(self, self.pending.popleft())
        return self.render()

    def render(self):
        from .host_java_imports import host_add_verified_imports
        source, mapping = render_program(self.program)
        final = host_add_verified_imports(source, symbols=self.api.symbols,
            owner=self.program["package"] + "." + self.program["class_name"], target_path=self.spec["target_path"])
        offset = len(final.splitlines()) - len(source.splitlines())
        self.line_map = {line + (offset if line > 1 else 0): value for line, value in mapping.items()}
        return final, self.program


def compose_incremental_feature(router, spec, messages, *, construction_context, behaviors):
    if not behaviors:
        raise ValueError("JAVA_INCREMENTAL_BEHAVIORS_REQUIRED")
    return IncrementalJavaBuilder(router, spec, messages, construction_context, behaviors).finish()
