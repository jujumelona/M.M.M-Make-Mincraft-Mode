"""Small semantic decisions over host-owned values and inspected API pages."""
from __future__ import annotations

import re
import hashlib
import json

from .host_rule_observation import PAGE_SIZE, choose_rule_data
from .java_operation_ir import PRIMITIVES, parameter_types


_TYPES = ("boolean", "int", "long", "double", "java.lang.String")
_BINARY = ("+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "&", "|", "^")
_INSTRUCTION = (
    "Implement only the current rule subgoal using host-owned typed values. "
    "Choose one operation or existing value ID. No Java expressions, names, "
    "declarations or statements. Build intermediates before using them. "
    "Preserve the original behavior including every OR branch and state obligation. "
    "A true constant, arbitrary API call or callback return cannot substitute "
    "for the original gameplay condition/effects. The host checks types and "
    "assembles all syntax; it retains older values and shows bounded pages."
)


class RuleAuthor:
    def __init__(self, router, program, *, data_pages=None):
        self.router, self.p = router, program
        self.decisions = 0
        self.data_pages = dict(data_pages or {})

    def ask(self, name, purpose, properties, **context):
        self.decisions += 1
        if self.decisions > 512:
            raise ValueError("RULE_DECISION_BUDGET: " + str(self.p.behavior.get("behavior_id")))
        menu = self.p.menu()
        return choose_rule_data(self.router, name, {
            "behavior": self.p.behavior, "subgoal": purpose,
            "recent_values": menu[-PAGE_SIZE:], "value_count": len(menu),
            "decision": self.decisions,
            "data_catalog": [{"id": key, "meaning": "Authoritative planning/prerequisite data"}
                             for key in sorted(self.data_pages)], **context,
        }, properties, _INSTRUCTION)

    def data_value(self, purpose):
        data_id = self.ask("rule_data_page", purpose,
            {"data_id": {"type": "string", "enum": sorted(self.data_pages)}})["data_id"]
        pending, scalars = [("", self.data_pages[data_id])], []
        while pending:
            path, value = pending.pop()
            if isinstance(value, dict):
                pending.extend((path + "/" + str(key).replace("~", "~0").replace("/", "~1"), value[key])
                               for key in reversed(sorted(value)))
            elif isinstance(value, (list, tuple)):
                pending.extend((path + "/" + str(i), value[i]) for i in reversed(range(len(value))))
            elif type(value) in {bool, int, float, str}:
                scalars.append((path, value))
        for offset in range(0, len(scalars), PAGE_SIZE):
            page = {"d" + str(i): entry for i, entry in enumerate(scalars[offset:offset + PAGE_SIZE])}
            selected = self.ask("rule_data_value", purpose,
                {"field_id": {"type": "string", "enum": [*page, "next_page"]}},
                data_id=data_id, fields=[{"id": key, "pointer": path, "value": value}
                                        for key, (path, value) in page.items()])["field_id"]
            if selected == "next_page":
                continue
            pointer, value = page[selected]
            kind = ("boolean" if type(value) is bool else "java.lang.String" if type(value) is str
                    else "double" if type(value) is float else "int" if -(2**31) <= value < 2**31 else "long")
            key = self.p.value("literal", type=kind, value=value, purpose=data_id + pointer)
            self.p.ir.append({"value_ref": key, "data_origin": {"data_id": data_id, "pointer": pointer,
                "sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()}})
            return key
        raise ValueError("RULE_DATA_VALUE_REQUIRED: " + data_id)

    def ref(self, purpose, *, expected=None, candidates=None, depth=0):
        keys = [row["id"] for row in self.p.menu()
                if (expected is None or self.p.checker.assignable(row["type"], expected))
                and (candidates is None or row["id"] in candidates)]
        if not keys:
            if candidates is not None:
                raise ValueError("RULE_COMPATIBLE_OPERAND_REQUIRED: " + purpose)
            return self.value(purpose, expected=expected, depth=depth + 1)
        for offset in range(0, len(keys), PAGE_SIZE):
            page = keys[offset:offset + PAGE_SIZE]
            options = [*page, *( ["next_page"] if offset + len(page) < len(keys) else []),
                       *( ["build"] if candidates is None else [])]
            key = self.ask("rule_operand", purpose,
                {"value_ref": {"type": "string", "enum": options}},
                expected_type=expected,
                values=[row for row in self.p.menu() if row["id"] in page])["value_ref"]
            if key == "build":
                return self.value(purpose, expected=expected, depth=depth + 1)
            if key != "next_page":
                return key
        raise ValueError("RULE_OPERAND_REQUIRED: " + purpose)

    def symbol(self, purpose, kind, *, effect=False):
        for _search in range(3):
            query = self.ask("rule_api_query", purpose, {
                "query": {"type": "string", "minLength": 1, "maxLength": 120}},
                instruction="Describe the required API operation/type in a few search words.")["query"]
            words = re.findall(r"[\w]+", query.casefold())
            pool = []
            for key, row in self.p.symbols.items():
                if row.get("kind") != kind:
                    continue
                text = (str(row.get("owner", "")) + "." + str(row.get("name", ""))).casefold()
                score = sum(word in text for word in words)
                if not score:
                    continue
                if (kind != "CONSTRUCTOR"
                        and not (row.get("static") or int(row.get("access", 0)) & 8)):
                    if not any(self.p.checker.assignable(value["type"], row["owner"])
                               for key2, value in self.p.values.items() if key2 in self.p.active):
                        continue
                if effect and str(row.get("name", "")).startswith(("get", "is", "has", "can", "should", "read", "peek")):
                    continue
                try:
                    self.p._symbol(key, effect=effect, constructor=kind == "CONSTRUCTOR")
                except ValueError:
                    continue
                pool.append((-score, str(row.get("name")), key, row))
            pool.sort(key=lambda item: item[:3])
            if not pool:
                continue
            retry = False
            for offset in range(0, len(pool), PAGE_SIZE):
                page = pool[offset:offset + PAGE_SIZE]
                index = {"s" + str(i): (key, row) for i, (_, _, key, row) in enumerate(page)}
                selected = self.ask("rule_api_page", purpose,
                    {"symbol": {"type": "string", "enum": [*index, "next_page", "search_again"]}},
                    query=query, symbols=[{
                        "id": key, "owner": row["owner"], "name": row.get("name"),
                        "parameters": parameter_types(row), "returns": row.get("return_type"),
                        "static": bool(row.get("static")),
                    } for key, (_ref, row) in index.items()],
                    remaining=len(pool) - offset - len(page))["symbol"]
                if selected == "search_again":
                    retry = True
                    break
                if selected != "next_page":
                    return index[selected]
            if not retry:
                break
        raise ValueError("RULE_CHECKED_CAPABILITY_REQUIRED: " + purpose)

    def operands(self, row, purpose, *, constructor=False, depth=0):
        refs = []
        if not constructor and not row.get("static"):
            refs.append(self.ref("Receiver for " + purpose, expected=row["owner"], depth=depth))
        for index, kind in enumerate(parameter_types(row)):
            refs.append(self.ref("Argument " + str(index) + " for " + purpose,
                                 expected=kind, depth=depth))
        return refs

    def _literal(self, purpose, expected):
        options = list(dict.fromkeys([*_TYPES,
            *[row["type"] for row in self.p.menu() if row["type"] not in PRIMITIVES]]))
        if expected and expected != "void":
            options = [expected, *[value for value in options if value != expected]]
        kind = self.ask("rule_literal_type", purpose,
            {"type": {"type": "string", "enum": options}})["type"]
        bits = {"byte": 8, "short": 16, "int": 32, "long": 64}.get(kind)
        schema = ({"type": "boolean"} if kind == "boolean" else
                  {"anyOf": [{"type": "string", "maxLength": 512}, {"type": "null"}]} if kind == "java.lang.String" else
                  {"type": "string", "minLength": 1, "maxLength": 1} if kind == "char" else
                  {"type": "number"} if kind in {"float", "double"} else
                  {"type": "integer", "minimum": -(2 ** (bits - 1)), "maximum": 2 ** (bits - 1) - 1}
                  if bits else {"type": "null"})
        literal = self.ask("rule_literal", purpose, {"value": schema})["value"]
        return self.p.value("literal", type=kind, value=literal, purpose=purpose)

    def value(self, purpose, *, expected=None, depth=0):
        if depth > 8:
            raise ValueError("RULE_VALUE_DEPTH: " + purpose)
        diagnostic = ""
        for _step in range(48):
            usable = [row["id"] for row in self.p.menu()
                      if expected is None or self.p.checker.assignable(row["type"], expected)]
            operations = ["literal", "call", "field", "new", "binary", "unary",
                          "ternary", "array_get", "array_length", "cast"]
            if self.data_pages:
                operations.append("data_value")
            if usable:
                operations.insert(0, "use")
            op = self.ask("rule_value_operation", purpose,
                {"operation": {"type": "string", "enum": operations}},
                expected_type=expected, last_diagnostic=diagnostic)["operation"]
            try:
                if op == "use":
                    return self.ref(purpose, expected=expected, candidates=usable, depth=depth)
                if op == "literal":
                    self._literal(purpose, expected)
                elif op == "data_value":
                    self.data_value(purpose)
                elif op == "cast":
                    types = sorted({*(_TYPES[:-1]),
                                    *[row["type"] for row in self.p.menu()],
                                    *([expected] if expected and expected != "void" else [])})
                    target = self.ask("rule_cast_type", purpose,
                        {"type": {"type": "string", "enum": types}})["type"]
                    valid = [row["id"] for row in self.p.menu()
                             if self.p.checker.castable(row["type"], target)]
                    source = self.ref("Value to cast to " + target, candidates=valid, depth=depth)
                    self.p.value("cast", [source], type_ref=target, purpose=purpose)
                elif op in {"call", "field", "new"}:
                    reference, row = self.symbol(purpose, {"call": "METHOD", "field": "FIELD", "new": "CONSTRUCTOR"}[op])
                    refs = self.operands(row, purpose, constructor=op == "new", depth=depth)
                    self.p.value(op, refs, symbol_ref=reference, purpose=row.get("name", purpose))
                elif op in {"binary", "unary"}:
                    operator = self.ask("rule_" + op + "_operator", purpose,
                        {"operator": {"type": "string", "enum": list(_BINARY if op == "binary" else ("!", "-", "+", "~"))}})["operator"]
                    left = self.ref("Left operand of " + operator, depth=depth)
                    refs = [left]
                    if op == "binary":
                        valid = []
                        for row in self.p.menu():
                            try:
                                self.p._infer(op, [left, row["id"]], {"operator": operator})
                            except ValueError:
                                continue
                            valid.append(row["id"])
                        # A first-time binary expression can require a new
                        # literal or verified call even if no old RHS is usable.
                        if valid:
                            right = self.ref("Right operand of " + operator,
                                             candidates=valid, depth=depth)
                        else:
                            right = self.value("Build right operand of " + operator,
                                               expected=self.p.kind(left),
                                               depth=depth + 1)
                        refs.append(right)
                    self.p.value(op, refs, operator=operator, purpose=operator + " on " + ", ".join(refs))
                elif op == "ternary":
                    condition = self.ref("Selection condition", expected="boolean", depth=depth)
                    yes = self.ref("Value when condition is true", expected=expected, depth=depth)
                    no = self.ref("Value when condition is false", expected=self.p.kind(yes), depth=depth)
                    self.p.value(op, [condition, yes, no], purpose=purpose)
                else:
                    arrays = [row["id"] for row in self.p.menu() if row["type"].endswith("[]")]
                    array = self.ref("Array for " + purpose, candidates=arrays, depth=depth)
                    refs = [array]
                    if op == "array_get":
                        refs.append(self.ref("Array index", expected="int", depth=depth))
                    self.p.value(op, refs, purpose=purpose)
                diagnostic = ""
            except ValueError as exc:
                diagnostic = str(exc)
        raise ValueError("RULE_VALUE_UNFINISHED: " + purpose + "; " + diagnostic)

    def effects(self):
        diagnostic = ""
        for _step in range(96):
            options = ["call", "assign", "if", "foreach"]
            if self.p.blocks:
                options.append("end")
            elif self.p.effects and set(self.p.behavior.get("state_writes", ())) <= self.p.written_states:
                options.append("complete")
            action = self.ask("rule_effect_operation", str(self.p.behavior.get("effects", "")),
                {"operation": {"type": "string", "enum": options}},
                open_blocks=len(self.p.blocks), required_state_writes=self.p.behavior.get("state_writes", []),
                written_states=sorted(self.p.written_states), last_diagnostic=diagnostic)["operation"]
            try:
                if action == "complete":
                    return self.p.render()
                if action == "call":
                    reference, row = self.symbol(str(self.p.behavior.get("effects", "")), "METHOD", effect=True)
                    refs = self.operands(row, row["name"])
                    self.p.effect("call", refs, symbol_ref=reference)
                elif action == "assign":
                    target = self.value("Choose or construct the exact field/array effect target")
                    value = self.ref("Value to assign", expected=self.p.kind(target))
                    operator = self.ask("rule_assignment_operator", "Choose assignment semantics",
                        {"operator": {"type": "string", "enum": ["=", "+=", "-=", "*=", "/=", "%="]}})["operator"]
                    self.p.effect("assign", [target, value], operator=operator)
                elif action == "if":
                    self.p.begin_if(self.value("Condition for only the next effect block", expected="boolean"))
                elif action == "foreach":
                    self.p.begin_each(self.value("Array or inspected generic Iterable to affect"))
                else:
                    self.p.end()
                diagnostic = ""
            except ValueError as exc:
                diagnostic = str(exc)
        raise ValueError("RULE_EFFECTS_UNFINISHED: " + diagnostic)
