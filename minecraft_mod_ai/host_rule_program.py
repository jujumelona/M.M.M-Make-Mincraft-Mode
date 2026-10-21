"""Typed rule expressions and scoped effects, rendered entirely by the host.

The model supplies symbol/value IDs and JSON literals. The existing
JavaOperations ABI/type/access checker decides what they mean; it never
interprets a model-supplied Java expression or declaration.
"""
from __future__ import annotations

from collections import ChainMap
import re
from types import SimpleNamespace

from .java_operation_ir import JavaOperations, identifier, java_literal, parameter_types


class RuleProgram:
    def __init__(self, symbols, parameters, behavior, *, owner, target_path, state_owners=()):
        self.behavior, self.state_owners = behavior, set(state_owners)
        method = owner + "#__mmmRule"
        extra = {
            owner: {"kind": "CLASS", "owner": owner, "access": 1},
            method: {"kind": "METHOD", "owner": owner, "name": "__mmmRule",
                     "return_type": "void", "parameters": parameters, "visibility": "private"},
        }
        self.symbols = ChainMap(extra, symbols)
        task = SimpleNamespace(owner_symbol=method, target_path=target_path)
        self.checker = JavaOperations(task, (), self.symbols, partial=True)
        self.values, self.active, self.statements, self.blocks, self.ir = {}, set(), [], [], []
        self.effects = 0
        self.written_states = set()
        self.read_states = set()
        self.inputs = tuple(self._store("parameter", [], {"name": identifier(row["name"])},
            row["type"], identifier(row["name"])) for row in parameters)

    def _store(self, op, refs, attrs, kind, source, *, generic="", purpose=""):
        key = "v" + str(len(self.values))
        row = {"node_id": key, "op": op, "input_refs": list(refs),
               "typed_result": kind, "attributes": attrs}
        self.checker.nodes[key] = row
        self.checker.types[key] = kind
        self.values[key] = {"source": source, "type": kind, "generic": generic, "purpose": purpose}
        self.active.add(key)
        self.ir.append(row)
        return key

    def source(self, key):
        if key not in self.active:
            raise ValueError("RULE_VALUE_SCOPE: " + str(key))
        return self.values[key]["source"]

    def kind(self, key):
        self.source(key)
        return self.values[key]["type"]

    def _state(self, row, *, effect):
        if row["owner"] not in self.state_owners:
            return
        name = row.get("name", "")
        prefix = "writeState_" if effect else "readState_"
        if not name.startswith(prefix):
            raise ValueError("RULE_STATE_AUTHORITY: " + name)
        state = name[len(prefix):]
        allowed = self.behavior.get("state_writes" if effect else "state_reads", ())
        if state not in allowed:
            raise ValueError("RULE_STATE_AUTHORITY: " + state)

    def _symbol(self, reference, *, effect=False, constructor=False):
        row = self.checker.symbol(reference)
        self._state(row, effect=effect)
        if (row["owner"].startswith("net.fabricmc.fabric.api.event.")
                or row["owner"] == "net.fabricmc.fabric.api.event.Event"):
            raise ValueError("RULE_EVENT_DISPATCH_FORBIDDEN")
        if row.get("kind") == "METHOD" and not effect:
            if not str(row.get("name", "")).startswith(("get", "is", "has", "can", "should", "read", "peek")):
                raise ValueError("RULE_OBSERVATION_CALL_REQUIRED: " + str(row.get("name")))
        if constructor and row.get("kind") != "CONSTRUCTOR":
            raise ValueError("RULE_CONSTRUCTOR_REQUIRED")
        if any(self.checker.checked_exception(kind) for kind in row.get("throws", ())):
            raise ValueError("RULE_CHECKED_EXCEPTION_REQUIRES_ADAPTER")
        return row

    def _infer(self, op, refs, attrs, claimed="int"):
        for key in refs:
            self.source(key)
        node = {"op": op, "attributes": attrs, "input_refs": list(refs), "typed_result": claimed}
        return self.checker.infer(node)

    def _invocation(self, op, row, refs):
        sources = [self.source(key) for key in refs]
        if op == "new":
            if row.get("enclosing_instance_type"):
                raise ValueError("RULE_INNER_CONSTRUCTOR_REQUIRES_ADAPTER")
            receiver = "new " + row["owner"]
        elif row.get("static") or int(row.get("access", 0)) & 8:
            receiver = row["owner"]
        else:
            receiver = "((" + row["owner"] + ") " + sources.pop(0) + ")"
        if op == "field":
            return receiver + "." + identifier(row["name"])
        args = ", ".join("((" + kind + ") " + value + ")"
                         for kind, value in zip(parameter_types(row), sources))
        return receiver + ("" if op == "new" else "." + identifier(row["name"])) + "(" + args + ")"

    def value(self, op, refs=(), *, purpose="", **attrs):
        generic = ""
        if op == "literal":
            if set(attrs) != {"type", "value"}:
                raise ValueError("RULE_LITERAL_FIELDS")
            kind = self.checker.nonvoid(attrs.pop("type"))
            source = java_literal(attrs["value"], kind)
        elif op in {"call", "field", "new"}:
            if set(attrs) != {"symbol_ref"}:
                raise ValueError("RULE_SYMBOL_FIELDS")
            row = self._symbol(attrs["symbol_ref"], constructor=op == "new")
            kind = self._infer(op, refs, attrs)
            if kind == "void":
                raise ValueError("RULE_VALUE_VOID")
            source = self._invocation(op, row, refs)
            generic = row.get("generic_return_type", "")
            if row["owner"] in self.state_owners:
                self.read_states.add(row["name"][len("readState_"):])
        elif op in {"binary", "unary", "array_get", "array_length", "cast", "ternary"}:
            required = {"binary": {"operator"}, "unary": {"operator"}, "cast": {"type_ref"}}.get(op, set())
            if set(attrs) != required:
                raise ValueError("RULE_OPERATOR_FIELDS")
            claimed = (attrs["type_ref"] if op == "cast" else self.kind(refs[1])
                       if op == "ternary" and len(refs) == 3 else "int")
            kind = self._infer(op, refs, attrs, claimed)
            sources = [self.source(key) for key in refs]
            if op == "binary":
                if attrs["operator"] in {"==", "!="} and all(self.kind(key) == "java.lang.String" for key in refs):
                    source = ("!" if attrs["operator"] == "!=" else "") + "java.util.Objects.equals(" + ", ".join(sources) + ")"
                else:
                    source = "(" + sources[0] + " " + attrs["operator"] + " " + sources[1] + ")"
            elif op == "unary":
                source = "(" + attrs["operator"] + sources[0] + ")"
            elif op == "cast":
                source = "((" + kind + ") " + sources[0] + ")"
            elif op == "array_get":
                source = "(" + sources[0] + ")[" + sources[1] + "]"
            elif op == "array_length":
                source = "(" + sources[0] + ").length"
            else:
                source = "(" + sources[0] + " ? " + sources[1] + " : " + sources[2] + ")"
        else:
            raise ValueError("RULE_VALUE_OPERATION: " + str(op))
        return self._store(op, refs, attrs, kind, source, generic=generic, purpose=purpose)

    def effect(self, op, refs=(), **attrs):
        if op == "call":
            if set(attrs) != {"symbol_ref"}:
                raise ValueError("RULE_SYMBOL_FIELDS")
            row = self._symbol(attrs["symbol_ref"], effect=True)
            self._infer(op, refs, attrs)
            statement = self._invocation(op, row, refs) + ";"
            if row["owner"] in self.state_owners:
                self.written_states.add(row["name"][len("writeState_"):])
        elif op == "assign":
            if len(refs) != 2 or self.checker.nodes.get(refs[0], {}).get("op") not in {"field", "array_get"}:
                raise ValueError("RULE_EFFECT_ASSIGNMENT_TARGET")
            self._infer(op, refs, attrs)
            statement = self.source(refs[0]) + " " + attrs.get("operator", "=") + " " + self.source(refs[1]) + ";"
        else:
            raise ValueError("RULE_EFFECT_OPERATION: " + str(op))
        self.statements.append(statement)
        self.statements.append("__mmmRuleApplied = true;")
        self.effects += 1
        self.ir.append({"effect": op, "input_refs": list(refs), "attributes": attrs})

    def begin_if(self, condition):
        if self.kind(condition) != "boolean":
            raise ValueError("RULE_BOOLEAN_REQUIRED")
        self.blocks.append(set(self.active))
        self.statements.append("if (" + self.source(condition) + ") {")
        self.ir.append({"control": "if", "value": condition})

    def iteration_type(self, iterable):
        kind = self.kind(iterable)
        if kind.endswith("[]"):
            return kind[:-2]
        if not self.checker.subtype(kind, "java.lang.Iterable"):
            raise ValueError("RULE_ITERABLE_REQUIRED")
        generic = self.values[iterable]["generic"]
        match = re.fullmatch(r"[^<>]+<\s*(?:\? extends )?([\w.$]+)\s*>", generic)
        if not match:
            raise ValueError("RULE_INSPECTED_ITERABLE_ELEMENT_REQUIRED")
        return self.checker.nonvoid(match[1])

    def begin_each(self, iterable):
        element = self.iteration_type(iterable)
        source = self.source(iterable)
        self.blocks.append(set(self.active))
        name = "__mmmItem" + str(len(self.values))
        # Cast each erased Iterable element only to its inspected generic type.
        if self.kind(iterable).endswith("[]"):
            self.statements.append(f"for ({element} {name} : {source}) {{")
        else:
            self.statements.append(f"for (Object {name}Raw : {source}) {{")
            self.statements.append(f"{element} {name} = ({element}) {name}Raw;")
        self.ir.append({"control": "foreach", "value": iterable, "element_type": element})
        self.checker.locals[name].add(element)
        return self._store("local", [], {"name": name}, element, name, purpose="Current iteration element")

    def end(self):
        if not self.blocks:
            raise ValueError("RULE_BLOCK_UNDERFLOW")
        self.active = self.blocks.pop()
        self.statements.append("}")
        self.ir.append({"control": "end"})

    def render(self):
        if self.blocks:
            raise ValueError("RULE_UNCLOSED_BLOCK")
        if not self.effects:
            raise ValueError("RULE_EFFECT_REQUIRED")
        return "boolean __mmmRuleApplied = false;\n" + "\n".join(self.statements) + "\nreturn __mmmRuleApplied;"

    def menu(self):
        return [{"id": key, "type": self.values[key]["type"],
                 "purpose": self.values[key]["purpose"] or (
                     "Callback input " + self.values[key]["source"] if key in self.inputs else "")}
                for key in self.values if key in self.active]
