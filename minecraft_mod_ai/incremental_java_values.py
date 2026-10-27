"""Author Java values from closed choices; the host owns every source token."""
from __future__ import annotations

from .incremental_java_calls import _signature
from .incremental_java_model import method_metadata
from .incremental_java_progress import finish_new_members
from .incremental_java_render import validate_expression
from .incremental_java_schema import choice, obj
from .incremental_java_values_forms import ValueForms
from .incremental_java_values_literals import author_literal
from .incremental_java_values_members import select_call, select_member, select_receiver
from .incremental_java_values_model import JavaReceiver, JavaValue, current_owner, inner_type_name, request, value_frame

VALUE_KINDS = ("literal", "local", "this", "null", "field", "call", "new", "binary", "unary",
               "conditional", "cast", "instanceof", "index", "array", "class_literal",
               "method_reference", "new_helper", "lookup")


class ValueAuthor:
    def __init__(self, builder):
        self.builder = builder
        self.forms = ValueForms(self)

    @property
    def frame(self):
        return self.builder.value_stack[-1]

    def value(self, slot, expected="java.lang.Object", *, writable=False, allow_update=False):
        with value_frame(self.builder, slot, expected, writable=writable) as frame:
            frame["statement_expression"] = allow_update
            options = self._options(expected, writable=writable, allow_update=allow_update)
            while True:
                kind = request(self.builder, choice("kind", options), "value_kind")["kind"]
                frame["kind"] = kind
                if kind != "lookup":
                    break
                self.builder.lookup()
            value = self._render(kind, expected, writable=writable)
            validate_expression(value.source, allow_update=allow_update)
            if writable and not value.writable:
                raise ValueError("JAVA_STRUCTURED_VALUE_NOT_WRITABLE")
            return value

    def _options(self, expected, *, writable=False, allow_update=False):
        if allow_update and not writable:
            return ["update", "call", "new", "new_helper", "lookup"]
        options = ["local", "field", "index", "lookup"] if writable else list(VALUE_KINDS)
        excluded = set()
        if self.builder.scope.get("static"):
            excluded.add("this")
        if expected in {"boolean", "byte", "short", "char", "int", "long", "float", "double"}:
            excluded.add("null")
        if not self._visible_locals(writable=writable):
            excluded.add("local")
        return [kind for kind in options if kind not in excluded]

    def _render(self, kind, expected, *, writable=False):
        if kind == "literal":
            return author_literal(self.builder, expected)
        if kind == "local":
            return self._local(writable=writable)
        if kind == "this":
            return JavaValue("this", current_owner(self.builder))
        if kind == "null":
            return JavaValue("null", expected)
        if kind == "field":
            return self._field(writable=writable)
        if kind == "call":
            return self._call()
        if kind == "new":
            return self._new(expected)
        if kind == "new_helper":
            return self._helper()
        if kind == "method_reference":
            return self._reference(expected)
        return getattr(self.forms, kind)(expected)

    def _visible_locals(self, *, writable=False):
        visible = {}
        for row in self.builder.scope.get("locals", ()):
            visible[row["name"]] = row
        if writable:
            visible = {name: row for name, row in visible.items() if not row.get("final")}
        return visible

    def _local(self, *, writable=False):
        visible = self._visible_locals(writable=writable)
        names, offset = list(visible), 0
        while True:
            page = {name: visible[name] for name in names[offset:offset + 10]}
            options = list(page)
            if offset + 10 < len(names):
                options.append("next_page")
            if offset:
                options.append("previous_page")
            name = request(self.builder, choice("name", options), "local_value", {
                "local_candidates": page, "local_offset": offset, "local_total": len(names)})["name"]
            if name in page:
                return JavaValue(name, page[name]["type"], writable=not page[name].get("final", False))
            offset += 10 if name == "next_page" else -10

    def _field(self, *, writable=False):
        while True:
            receiver = select_receiver(self, "field_receiver")
            self.frame.update(receiver=receiver.source, receiver_type=receiver.java_type)
            selected = select_member(self.builder, receiver, "FIELD", "field_member", writable=writable)
            if selected != "change_receiver":
                break
        self.frame["field_contract"] = selected
        java_type = selected.get("generic_return_type", selected["return_type"])
        return JavaValue(receiver.source + "." + selected["name"], java_type, writable=True)

    def _invoke(self, node):
        self.builder.arguments(node)
        source = (node["receiver"] + "." if node.get("receiver") else "") + node["method"]
        contract = node["call_contract"]
        return JavaValue(source + "(" + ", ".join(node["arguments"]) + ")",
                         contract.get("generic_return_type", contract.get("return_type", "java.lang.Object")))

    def _call(self):
        select_call(self, self.frame, nonvoid=not self.frame.get("statement_expression", False))
        return self._invoke(self.frame)

    def _new(self, expected):
        options = {"non_array": True} if self.frame.get("statement_expression") else {}
        while True:
            java_type = self.builder.select_java_type("new_type", expected=expected, reference_only=True, **options)
            self.frame["new_type"] = java_type
            if java_type.endswith("[]"):
                return self.forms.array(expected, java_type=java_type)
            receiver = JavaReceiver(java_type, java_type, static=True)
            selected = select_member(self.builder, receiver, "CONSTRUCTOR", "constructor_member")
            if selected != "change_receiver":
                break
        self.frame.update(receiver=java_type, receiver_type=java_type,
                          method=selected["name"], call_contract=_signature(selected))
        self.builder.arguments(self.frame)
        prefix = "new " + java_type
        if selected.get("enclosing_instance_type"):
            outer = self.value("enclosing_instance", selected["enclosing_instance_type"])
            prefix = "(" + outer.source + ").new " + inner_type_name(java_type)
        return JavaValue(prefix + "(" + ", ".join(self.frame["arguments"]) + ")", java_type)

    def helper_contract(self, *, nonvoid=True, owner=None, static=None):
        builder = self.builder
        java_type = builder.select_java_type("helper_return_type", expected=self.frame.get("expected_type", ""),
                                            allow_void=not nonvoid)
        static_schema = {"type": "boolean"} if static is None else {"type": "boolean", "enum": [static]}
        details = request(builder, obj({"purpose": {"type": "string", "minLength": 1}, "static": static_schema}), "value_helper")
        self.frame["helper"] = {"return_type": java_type, **details}
        metadata = method_metadata(java_type, builder.parameters(), static=details["static"])
        owner = self._generated_owner(owner or current_owner(builder))
        before = len(builder.pending)
        member = builder.new_method(owner=owner, purpose=details["purpose"], behavior=builder.member.get("behavior", {}), metadata=metadata)
        self.frame["helper"]["member_id"] = member["id"]
        previous = builder.value_stack
        try:
            builder.value_stack = []
            finish_new_members(builder, before)
        finally:
            builder.value_stack = previous
        return {"kind": "METHOD", "owner": member["owner"], "name": member["name"], **member["metadata"]}

    def _generated_owner(self, owner):
        from .incremental_java_values_catalog import generated_type
        declared = generated_type(self.builder, owner)
        if declared is None:
            raise ValueError("JAVA_STRUCTURED_HELPER_OWNER_MUST_BE_GENERATED: " + owner)
        return declared["name"]

    def _helper(self):
        contract = self.helper_contract(nonvoid=not self.frame.get("statement_expression", False))
        receiver = contract["owner"] if contract.get("static") else "this"
        self.frame.update(receiver=receiver, receiver_type=contract["owner"], method=contract["name"], call_contract=_signature(contract))
        return self._invoke(self.frame)

    def _reference(self, expected):
        kind = request(self.builder, choice("reference_kind", ("method", "constructor")), "reference_kind")["reference_kind"]
        if kind == "method":
            receiver = select_call(self, self.frame, reference=True)
            return JavaValue(receiver.source + "::" + self.frame["method"], expected)
        java_type = self.builder.select_java_type("reference_type", reference_only=True)
        if not java_type.endswith("[]"):
            receiver = JavaReceiver(java_type, java_type, static=True)
            contract = select_member(self.builder, receiver, "CONSTRUCTOR", "constructor_member")
            if contract == "change_receiver":
                return self._reference(expected)
            self.frame["call_contract"] = _signature(contract)
        return JavaValue(java_type + "::new", expected)


def author_value(builder, slot, expected="java.lang.Object", *, allow_update=False) -> str:
    """Render one structured value; assignment targets expose only writable forms."""
    value = ValueAuthor(builder).value(slot, expected or "java.lang.Object", writable=slot == "target",
                                       allow_update=allow_update)
    builder.last_value_type = value.java_type
    return value.source


def author_call_head(builder, node) -> None:
    """Select a receiver and exact method, leaving arguments and binding to the host."""
    with value_frame(builder, "call", ""):
        select_call(ValueAuthor(builder), node)
