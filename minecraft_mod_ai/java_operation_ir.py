"""Typed Java syntax operations. Gameplay meaning remains authored work."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict, deque

PRIMITIVES = frozenset({"boolean", "byte", "short", "int", "long", "char", "float", "double", "void"})
NUMERIC = ("byte", "short", "char", "int", "long", "float", "double")
INTEGRAL = frozenset({"byte", "short", "char", "int", "long"})
KEYWORDS = frozenset(("abstract assert boolean break byte case catch char class const continue default "
    "do double else enum extends final finally float for goto if implements import instanceof int "
    "interface long native new package private protected public return short static strictfp super "
    "switch synchronized this throw throws transient try void volatile while true false null record "
    "sealed permits yield var _").split())
ATTRIBUTES = {
    "literal": {"value"}, "parameter": {"name"}, "local": {"name"}, "this": set(),
    "declare": {"name", "final"}, "field": {"symbol_ref"}, "call": {"symbol_ref"},
    "new": {"symbol_ref"}, "binary": {"operator"}, "unary": {"operator"},
    "assign": {"operator"}, "cast": {"type_ref"}, "instanceof": {"type_ref"},
    "array_get": set(), "array_length": set(), "new_array": {"element_type", "sized"},
    "ternary": set(), "return": set(), "block": set(), "if": set(), "while": set(),
    "for": set(), "do_while": set(), "foreach": {"name", "element_type"},
    "lambda": {"parameters"}, "switch": {"cases"}, "throw": set(),
    "try": {"catches", "finally"}, "synchronized": set(), "break": set(), "continue": set(),
}
OPERATIONS = tuple(ATTRIBUTES)
EXPRESSIONS = frozenset({"literal", "parameter", "local", "this", "field", "call", "new", "binary",
    "unary", "assign", "cast", "instanceof", "array_get", "array_length", "new_array", "ternary", "lambda"})
STATEMENTS = frozenset(set(ATTRIBUTES) - EXPRESSIONS) | {"call", "new", "assign"}
LEAVES = frozenset({"literal", "parameter", "local", "this"})
REQUIRED = {op: fields - {"final", "sized", "operator", "catches", "finally"} for op, fields in ATTRIBUTES.items()}
REQUIRED.update(binary={"operator"}, unary={"operator"})
WIDENING = {"byte": {"short", "int", "long", "float", "double"}, "short": {"int", "long", "float", "double"},
    "char": {"int", "long", "float", "double"}, "int": {"long", "float", "double"},
    "long": {"float", "double"}, "float": {"double"}}
BOXING = {"boolean": "java.lang.Boolean", "byte": "java.lang.Byte", "short": "java.lang.Short",
    "int": "java.lang.Integer", "long": "java.lang.Long", "char": "java.lang.Character",
    "float": "java.lang.Float", "double": "java.lang.Double"}


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", value) or value in KEYWORDS:
        raise ValueError("CONSTRUCTION_IDENTIFIER_INVALID")
    return value


def qualified_name(value):
    if not isinstance(value, str):raise ValueError("CONSTRUCTION_TYPE_INVALID")
    for part in value.split("."):identifier(part)
    return value


def parameter_records(row):
    return [p if isinstance(p, dict) else {"type": p, "name": "arg" + str(i)}
            for i, p in enumerate(row.get("parameters", ()))]


def parameter_types(row):
    return [p["type"] if isinstance(p, dict) else p for p in row.get("parameters", ())]


def java_literal(value, type_name):
    if value is None and type_name not in PRIMITIVES:return "null"
    if type_name == "boolean" and type(value) is bool:return "true" if value else "false"
    if type_name in {"byte", "short", "int", "long"} and type(value) is int:
        bits = {"byte": 8, "short": 16, "int": 32, "long": 64}[type_name]
        if not -(2**(bits - 1)) <= value < 2**(bits - 1):raise ValueError("CONSTRUCTION_LITERAL_RANGE")
        if type_name == "long":return str(value) + "L"
        return ("(" + type_name + ")" if type_name in {"byte", "short"} else "") + str(value)
    if type_name in {"float", "double"} and type(value) in {int, float}:
        try:number = float(value)
        except OverflowError as exc:raise ValueError("CONSTRUCTION_LITERAL_RANGE") from exc
        if not math.isfinite(number):raise ValueError("CONSTRUCTION_LITERAL_RANGE")
        if type_name == "float" and (abs(number) > 3.4028234663852886e38 or 0 < abs(number) <= 2**-150):
            raise ValueError("CONSTRUCTION_LITERAL_RANGE")
        return repr(number) + ("F" if type_name == "float" else "D")
    if type_name == "char" and isinstance(value, str) and len(value) == 1 and ord(value) < 65536:
        text = json.dumps(value, ensure_ascii=True)[1:-1].replace("'", "\\'")
        return "'" + text + "'"
    if type_name in {"java.lang.String", "String"} and isinstance(value, str):return json.dumps(value, ensure_ascii=True)
    raise ValueError("CONSTRUCTION_LITERAL_TYPE: " + type_name)


class JavaOperations:
    def __init__(self, task, nodes, symbols):
        self.task, self.symbols = task, symbols
        if not isinstance(nodes, (list, tuple)) or any(not isinstance(n, dict) for n in nodes):
            raise ValueError("CONSTRUCTION_NODE_FIELDS")
        self.nodes = {}
        for n in nodes:
            key = n.get("node_id")
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", key):raise ValueError("CONSTRUCTION_NODE_ID_INVALID")
            if key in self.nodes:raise ValueError("CONSTRUCTION_NODE_DUPLICATE")
            self.nodes[key] = n
        if task.owner_symbol not in symbols:raise ValueError("CONSTRUCTION_SYMBOL_MISSING: " + task.owner_symbol)
        self.declaration = self.symbol(task.owner_symbol)
        self.types, self.locals, self.parameters, self.lambda_returns = {}, defaultdict(set), defaultdict(set), {}
        self.known_types = set(symbols)
        for row in symbols.values():
            for value in [row.get("owner"), row.get("return_type"), row.get("functional_return_type"),
                          row.get("superclass"), *row.get("interfaces", ()), *parameter_types(row),
                          *row.get("functional_parameters", ())]:
                if isinstance(value, str):self.known_types.add(value.split("[")[0])
        self.type_name(self.declaration["owner"])
        for p in parameter_records(self.declaration):
            name = identifier(p["name"])
            if name in self.parameters:raise ValueError("CONSTRUCTION_PARAMETER_DUPLICATE")
            self.parameters[name].add(self.nonvoid(p["type"]))
        self.check_class_header()
        self.member_parameters = {p["name"]: p["type"] for p in parameter_records(self.declaration)}
        if self.declaration["kind"] == "FIELD":self.nonvoid(self.declaration["return_type"])
        elif self.declaration["kind"] == "METHOD":self.type_name(self.declaration.get("return_type", "void"))
        elif self.declaration["kind"] != "CONSTRUCTOR":raise ValueError("CONSTRUCTION_DECLARATION_KIND")
        for key, n in self.nodes.items():
            op = n.get("op")
            if not isinstance(op, str) or op not in ATTRIBUTES:raise ValueError("CONSTRUCTION_OPERATION_UNSUPPORTED: " + str(op))
            if set(n) != {"node_id", "op", "input_refs", "typed_result", "attributes"}:raise ValueError("CONSTRUCTION_NODE_FIELDS")
            attrs = n["attributes"]
            if not isinstance(attrs, dict) or set(attrs) - ATTRIBUTES[op] or REQUIRED[op] - set(attrs):
                raise ValueError("CONSTRUCTION_OPERATION_ATTRIBUTES")
            if "operator" in attrs and not isinstance(attrs["operator"], str):raise ValueError("CONSTRUCTION_OPERATOR_INVALID")
            if "symbol_ref" in attrs and not isinstance(attrs["symbol_ref"], str):raise ValueError("CONSTRUCTION_SYMBOL_MISSING")
            for flag in {"final", "sized", "finally"} & set(attrs):
                if type(attrs[flag]) is not bool:raise ValueError("CONSTRUCTION_OPERATION_ATTRIBUTES")
            if not isinstance(n["input_refs"], list) or any(not isinstance(x, str) or x not in self.nodes for x in n["input_refs"]):
                raise ValueError("CONSTRUCTION_NODE_REFERENCE_MISSING")
            self.type_name(n["typed_result"])
            if op in {"declare", "foreach"}:
                self.locals[identifier(attrs["name"])].add(self.nonvoid(attrs.get("element_type", n["typed_result"])))
            if op == "lambda":
                functional = self.symbols.get(n["typed_result"], {})
                types, names = functional.get("functional_parameters"), attrs["parameters"]
                if types is None or not isinstance(names, list) or len(names) != len(types):raise ValueError("CONSTRUCTION_LAMBDA_SIGNATURE_REQUIRED")
                for name in names:identifier(name)
                if len(set(names)) != len(names):raise ValueError("CONSTRUCTION_LAMBDA_PARAMETER_COLLISION")
                for name, type_ in zip(names, types):self.parameters[identifier(name)].add(self.nonvoid(type_))
                self.lambda_returns[key] = self.type_name(functional["functional_return_type"])
            if op == "try":
                catches = attrs.get("catches", [])
                if not isinstance(catches, list):raise ValueError("CONSTRUCTION_CATCH_INVALID")
                for catch in catches:
                    if not isinstance(catch, dict) or set(catch) != {"name", "type_ref"}:raise ValueError("CONSTRUCTION_CATCH_INVALID")
                    self.locals[identifier(catch["name"])].add(self.nonvoid(catch["type_ref"]))
        # Linear Kahn traversal; forward references and arbitrary graph counts are legal.
        dependencies = {key: set(n["input_refs"]) for key, n in self.nodes.items()}
        users = defaultdict(list)
        for key, refs in dependencies.items():
            for ref in refs:users[ref].append(key)
        pending = {key: len(refs) for key, refs in dependencies.items()}
        ready = deque(key for key, count in pending.items() if not count)
        self.order = []
        while ready:
            key = ready.popleft()
            actual = self.infer(self.nodes[key])
            expected = self.type_name(self.nodes[key]["typed_result"])
            if actual != expected:raise ValueError("CONSTRUCTION_NODE_TYPE: " + key + ": " + actual + " != " + expected)
            self.types[key] = actual
            self.order.append(key)
            for user in users[key]:
                pending[user] -= 1
                if not pending[user]:ready.append(user)
        if len(self.order) != len(self.nodes):raise ValueError("CONSTRUCTION_NODE_CYCLE")
        consumed = Counter(r for n in self.nodes.values() for r in n["input_refs"])
        self.roots = [key for key in self.nodes if not consumed[key]]
        if not self.roots:raise ValueError("CONSTRUCTION_BODY_EMPTY")
        if self.declaration["kind"] == "FIELD":
            if len(self.roots) != 1:raise ValueError("CONSTRUCTION_FIELD_EXPRESSION_REQUIRED")
            self.require_expression(self.roots[0])
            if not self.assignable(self.types[self.roots[0]], self.declaration["return_type"]):raise ValueError("CONSTRUCTION_FIELD_TYPE")
        else:
            for key in self.roots:self.require_statement(key)
        # Authors must explicitly store and reference shared computed values.
        for key, count in consumed.items():
            if count > 1 and self.nodes[key]["op"] not in LEAVES:raise ValueError("CONSTRUCTION_SHARED_EVALUATION: " + key)
        self.check_contexts()
        self.check_completion()

    def nonvoid(self, value):
        value = self.type_name(value)
        if value == "void":raise ValueError("CONSTRUCTION_TYPE_VOID")
        return value

    def require_expression(self, key):
        if self.nodes[key]["op"] not in EXPRESSIONS:raise ValueError("CONSTRUCTION_EXPRESSION_REQUIRED: " + key)

    def require_statement(self, key):
        if self.nodes[key]["op"] not in STATEMENTS:raise ValueError("CONSTRUCTION_STATEMENT_REQUIRED: " + key)

    def type_name(self, value):
        if not isinstance(value, str):raise ValueError("CONSTRUCTION_TYPE_INVALID")
        base = value
        while base.endswith("[]"):base = base[:-2]
        if base in PRIMITIVES:
            if base == "void" and base != value:raise ValueError("CONSTRUCTION_TYPE_VOID")
            return value
        qualified_name(base)
        if base not in self.known_types:raise ValueError("CONSTRUCTION_TYPE_NOT_INSPECTED: " + base)
        if self.symbols.get(base, {}).get("kind") == "CLASS":
            self.symbol(base)
        return value

    def symbol(self, key):
        if not isinstance(key, str) or key not in self.symbols:raise ValueError("CONSTRUCTION_SYMBOL_MISSING: " + str(key))
        row = self.symbols[key]
        qualified_name(row["owner"])
        if row.get("name") != "<init>":identifier(row.get("name", row["owner"].rsplit(".", 1)[-1]))
        if row.get("side") == "CLIENT" and not self.task.target_path.startswith("src/client/"):raise ValueError("CONSTRUCTION_SIDE_MISMATCH")
        if row.get("descriptor") and row["kind"] != "CLASS":
            from .construction_context import descriptor_types
            try:parameters, returns = descriptor_types(row["descriptor"])
            except (ValueError, IndexError, TypeError) as exc:raise ValueError("CONSTRUCTION_SYMBOL_DESCRIPTOR") from exc
            if parameters != parameter_types(row) or returns != row.get("return_type"):raise ValueError("CONSTRUCTION_SYMBOL_DESCRIPTOR")
        current = self.symbols[self.task.owner_symbol]["owner"]
        if key != self.task.owner_symbol and "access" in row:
            access = row["access"]
            same_package = row["owner"].rpartition(".")[0] == current.rpartition(".")[0]
            if access & 2 and row["owner"] != current:raise ValueError("CONSTRUCTION_SYMBOL_ACCESS")
            if not access & 7 and not same_package:raise ValueError("CONSTRUCTION_SYMBOL_ACCESS")
            if access & 4 and not same_package and not self.subtype(current, row["owner"]):raise ValueError("CONSTRUCTION_SYMBOL_ACCESS")
        return row

    def check_class_header(self):
        row = self.declaration
        if row["kind"] == "CONSTRUCTOR" and row.get("static"):raise ValueError("CONSTRUCTION_CLASS_HEADER")
        if row.get("class_superclass"):
            type_ = self.nonvoid(row["class_superclass"])
            definition = self.symbols.get(type_, {})
            if definition.get("kind") != "CLASS" or definition.get("access", 0) & (0x10 | 0x200):raise ValueError("CONSTRUCTION_CLASS_HEADER")
            self.symbol(type_)
        interfaces = row.get("class_interfaces", [])
        if not isinstance(interfaces, (list, tuple)):raise ValueError("CONSTRUCTION_CLASS_HEADER")
        for type_ in interfaces:
            self.nonvoid(type_)
            definition = self.symbols.get(type_, {})
            if definition.get("kind") != "CLASS" or not definition.get("access", 0) & 0x200:raise ValueError("CONSTRUCTION_CLASS_HEADER")
            self.symbol(type_)
        if len(set(interfaces)) != len(interfaces):raise ValueError("CONSTRUCTION_CLASS_HEADER")
        for type_ in row.get("throws", []):
            self.nonvoid(type_)
            if not self.subtype(type_, "java.lang.Throwable"):raise ValueError("CONSTRUCTION_THROW_TYPE")

    def reference(self, value):
        return value not in PRIMITIVES

    def subtype(self, actual, expected):
        if actual == expected:return True
        if not self.reference(actual) or not self.reference(expected):return False
        if expected == "java.lang.Object":return True
        if actual.endswith("[]"):
            if expected in {"java.lang.Cloneable", "java.io.Serializable"}:return True
            if expected.endswith("[]"):
                a, b = actual[:-2], expected[:-2]
                return a == b or (self.reference(a) and self.reference(b) and self.subtype(a, b))
            return False
        seen, frontier = set(), [actual]
        while frontier:
            current = frontier.pop()
            if current == expected:return True
            if current in seen:continue
            seen.add(current)
            row = self.symbols.get(current, {})
            if current == self.symbols[self.task.owner_symbol]["owner"]:
                generated = self.symbols[self.task.owner_symbol]
                frontier.extend(x for x in [generated.get("class_superclass"), *generated.get("class_interfaces", [])] if x)
            frontier.extend(x for x in [row.get("superclass"), *row.get("interfaces", ())] if x)
        return False

    def assignable(self, actual, expected):
        if "void" in {actual, expected}:return False
        if actual == expected or expected in WIDENING.get(actual, set()):return True
        if actual in BOXING:return self.subtype(BOXING[actual], expected)
        unboxed = {v: k for k, v in BOXING.items()}.get(actual)
        if unboxed and (unboxed == expected or expected in WIDENING.get(unboxed, set())):return True
        return self.subtype(actual, expected)

    def castable(self, actual, expected):
        if "void" in {actual, expected}:return False
        if actual in NUMERIC and expected in NUMERIC:return True
        if actual == expected:return True
        if actual in PRIMITIVES or expected in PRIMITIVES:return self.assignable(actual, expected) or self.assignable(expected, actual)
        if self.subtype(actual, expected) or self.subtype(expected, actual):return True
        if actual.endswith("[]") or expected.endswith("[]"):return False
        a, b = self.symbols.get(actual, {}), self.symbols.get(expected, {})
        return bool((a.get("access", 0) & 0x200 and not b.get("access", 0) & 0x10) or
                    (b.get("access", 0) & 0x200 and not a.get("access", 0) & 0x10))

    def arity(self, n, count):
        if len(n["input_refs"]) != count:raise ValueError("CONSTRUCTION_OPERATION_ARITY: " + n["node_id"])

    def infer(self, n):
        op, attrs, refs = n["op"], n["attributes"], n["input_refs"]
        types = [self.types[r] for r in refs]
        claimed = self.type_name(n["typed_result"])
        expression_indices = {
            "literal": (), "parameter": (), "local": (), "this": (), "declare": range(len(refs)),
            "field": range(len(refs)), "call": range(len(refs)), "new": range(len(refs)),
            "binary": range(len(refs)), "unary": range(len(refs)), "assign": range(len(refs)),
            "cast": range(len(refs)), "instanceof": range(len(refs)), "array_get": range(len(refs)),
            "array_length": range(len(refs)), "new_array": range(len(refs)), "ternary": range(len(refs)),
            "return": range(len(refs)), "throw": range(len(refs)), "if": (0,), "while": (0,),
            "do_while": (0,), "for": (1,), "foreach": (0,), "switch": (0,), "synchronized": (0,),
            "lambda": (), "block": (), "try": (), "break": (), "continue": ()}
        for index in expression_indices[op]:
            if index < len(refs):
                self.require_expression(refs[index])
                if types[index] == "void":raise ValueError("CONSTRUCTION_EXPRESSION_VOID")
        if op in {"if", "while", "do_while", "foreach", "switch", "synchronized", "lambda", "block", "try"}:
            for index, ref in enumerate(refs):
                if index not in expression_indices[op]:self.require_statement(ref)
        if op == "for" and len(refs) == 4:
            if self.nodes[refs[0]]["op"] not in {"declare", "assign", "call", "new"} or self.nodes[refs[2]]["op"] not in {"assign", "call", "new"}:
                raise ValueError("CONSTRUCTION_FOR_HEADER")
            self.require_statement(refs[3])
        if op == "literal":self.arity(n, 0);java_literal(attrs["value"], claimed);return claimed
        if op == "parameter":
            self.arity(n, 0);name = identifier(attrs["name"])
            if name not in self.parameters:raise ValueError("CONSTRUCTION_PARAMETER_MISSING")
            if claimed not in self.parameters[name]:raise ValueError("CONSTRUCTION_PARAMETER_TYPE")
            return claimed
        if op == "local":
            self.arity(n, 0);name = identifier(attrs["name"])
            if name not in self.locals:raise ValueError("CONSTRUCTION_LOCAL_MISSING")
            if claimed not in self.locals[name]:raise ValueError("CONSTRUCTION_LOCAL_TYPE")
            return claimed
        if op == "this":
            self.arity(n, 0)
            if self.declaration.get("static"):raise ValueError("CONSTRUCTION_THIS_IN_STATIC")
            return self.declaration["owner"]
        if op == "declare":
            if len(types) > 1 or (types and not self.assignable(types[0], claimed)):raise ValueError("CONSTRUCTION_LOCAL_INITIALIZER_TYPE")
            if claimed == "void":raise ValueError("CONSTRUCTION_LOCAL_VOID")
            return claimed
        if op in {"call", "new", "field"}:
            row = self.symbol(attrs["symbol_ref"])
            self.type_name(row["owner"])
            required_kind = "CONSTRUCTOR" if op == "new" else "FIELD" if op == "field" else "METHOD"
            if row["kind"] != required_kind:raise ValueError("CONSTRUCTION_SYMBOL_KIND")
            if op == "new" and self.symbols.get(row["owner"], {}).get("access", 0) & (0x200 | 0x400):raise ValueError("CONSTRUCTION_CONSTRUCTOR_ABSTRACT")
            if op == "new" and row.get("access", 0) & 4 and row["owner"].rpartition(".")[0] != self.declaration["owner"].rpartition(".")[0]:
                raise ValueError("CONSTRUCTION_SYMBOL_ACCESS")
            args = types
            if op != "new" and not row.get("static"):
                if not args or not self.assignable(args[0], row["owner"]):raise ValueError("CONSTRUCTION_RECEIVER_TYPE")
                current = self.declaration["owner"]
                if row.get("access", 0) & 4 and row["owner"].rpartition(".")[0] != current.rpartition(".")[0] and not self.subtype(args[0], current):
                    raise ValueError("CONSTRUCTION_SYMBOL_ACCESS")
                args = args[1:]
            expected = [] if op == "field" else parameter_types(row)
            for type_ in expected:self.nonvoid(type_)
            if len(args) != len(expected):raise ValueError("CONSTRUCTION_CALL_ARITY")
            if any(not self.assignable(a, b) for a, b in zip(args, expected)):raise ValueError("CONSTRUCTION_CALL_PARAMETER_TYPE")
            return row["owner"] if op == "new" else self.type_name(row["return_type"])
        if op == "binary":
            self.arity(n, 2);operator = attrs["operator"]
            if operator in {"==", "!="}:
                compatible = all(t in NUMERIC for t in types) or types == ["boolean", "boolean"] or (all(self.reference(t) for t in types) and self.castable(*types))
                if not compatible:raise ValueError("CONSTRUCTION_COMPARISON_TYPE")
                return "boolean"
            if operator in {"&&", "||"}:
                if types != ["boolean", "boolean"]:raise ValueError("CONSTRUCTION_BOOLEAN_REQUIRED")
                return "boolean"
            if operator == "+" and "java.lang.String" in types:return "java.lang.String"
            if operator in {"&", "|", "^"} and types == ["boolean", "boolean"]:return "boolean"
            if operator not in {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "&", "|", "^", "<<", ">>", ">>>"}:raise ValueError("CONSTRUCTION_OPERATOR_INVALID")
            if operator in {"&", "|", "^", "<<", ">>", ">>>"} and any(t not in INTEGRAL for t in types):raise ValueError("CONSTRUCTION_INTEGRAL_REQUIRED")
            if any(t not in NUMERIC for t in types):raise ValueError("CONSTRUCTION_NUMERIC_REQUIRED")
            if operator in {"<", ">", "<=", ">="}:return "boolean"
            if operator in {"<<", ">>", ">>>"}:return "long" if types[0] == "long" else "int"
            return max(["int", *types], key=NUMERIC.index)
        if op == "unary":
            self.arity(n, 1)
            if attrs["operator"] == "!" and types == ["boolean"]:return "boolean"
            if attrs["operator"] == "~" and types[0] not in INTEGRAL:raise ValueError("CONSTRUCTION_INTEGRAL_REQUIRED")
            if attrs["operator"] in {"+", "-", "~"} and types[0] in NUMERIC:return "int" if types[0] in {"byte", "short", "char"} else types[0]
            raise ValueError("CONSTRUCTION_UNARY_TYPE")
        if op == "assign":
            self.arity(n, 2)
            if self.nodes[refs[0]]["op"] not in {"local", "field", "array_get"}:raise ValueError("CONSTRUCTION_ASSIGNMENT_TARGET")
            operator = attrs.get("operator", "=")
            if operator not in {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>="}:raise ValueError("CONSTRUCTION_ASSIGNMENT_OPERATOR")
            if operator == "=" and not self.assignable(types[1], types[0]):raise ValueError("CONSTRUCTION_ASSIGNMENT_TYPE")
            if operator != "=":
                binary_op = operator[:-1]
                valid = (binary_op == "+" and types[0] == "java.lang.String" or binary_op in {"&", "|", "^"} and types == ["boolean", "boolean"] or
                    all(t in (INTEGRAL if binary_op in {"&", "|", "^", "<<", ">>", ">>>"} else NUMERIC) for t in types))
                if not valid:raise ValueError("CONSTRUCTION_ASSIGNMENT_TYPE")
            target = self.nodes[refs[0]]
            if target["op"] == "field":
                field = self.symbol(target["attributes"]["symbol_ref"])
                if field.get("final") or field.get("access", 0) & 0x10:raise ValueError("CONSTRUCTION_FINAL_ASSIGNMENT")
            return types[0]
        if op in {"cast", "instanceof"}:
            self.arity(n, 1);target = self.type_name(attrs["type_ref"])
            if op == "instanceof":
                if not self.reference(types[0]) or not self.reference(target) or not self.castable(types[0], target):raise ValueError("CONSTRUCTION_INSTANCEOF_TYPE")
            elif not self.castable(types[0], target):raise ValueError("CONSTRUCTION_CAST_TYPE")
            return "boolean" if op == "instanceof" else target
        if op == "array_get":
            self.arity(n, 2)
            if not types[0].endswith("[]") or types[1] not in {"int", "byte", "short", "char"}:raise ValueError("CONSTRUCTION_ARRAY_TYPE")
            return types[0][:-2]
        if op == "array_length":
            self.arity(n, 1)
            if not types[0].endswith("[]"):raise ValueError("CONSTRUCTION_ARRAY_TYPE")
            return "int"
        if op == "new_array":
            element = self.nonvoid(attrs["element_type"])
            if attrs.get("sized"):
                if element.endswith("[]") or not types or any(t not in {"int", "byte", "short", "char"} for t in types):raise ValueError("CONSTRUCTION_ARRAY_SIZE")
                return element + "[]" * len(types)
            if any(not self.assignable(t, element) for t in types):raise ValueError("CONSTRUCTION_ARRAY_ELEMENT_TYPE")
            return element + "[]"
        if op == "ternary":
            self.arity(n, 3)
            if types[0] != "boolean" or not all(self.assignable(t, claimed) for t in types[1:]):raise ValueError("CONSTRUCTION_TERNARY_TYPE")
            return claimed
        if op == "lambda":self.arity(n, 1);return claimed
        if op == "return":
            if len(types) > 1:raise ValueError("CONSTRUCTION_RETURN_ARITY")
            return "void"
        if op == "if":
            if len(types) not in {2, 3} or types[0] != "boolean":raise ValueError("CONSTRUCTION_CONDITION_TYPE")
        elif op in {"while", "do_while"}:
            self.arity(n, 2)
            if types[0] != "boolean":raise ValueError("CONSTRUCTION_CONDITION_TYPE")
        elif op == "for":
            self.arity(n, 4)
            if types[1] != "boolean":raise ValueError("CONSTRUCTION_CONDITION_TYPE")
        elif op == "foreach":
            self.arity(n, 2);element = self.nonvoid(attrs["element_type"])
            if not types[0].endswith("[]") or not self.assignable(types[0][:-2], element):raise ValueError("CONSTRUCTION_FOREACH_TYPE")
        elif op == "switch":
            if not isinstance(attrs["cases"], list) or not refs or len(refs) != len(attrs["cases"]) + 1:raise ValueError("CONSTRUCTION_SWITCH_CASES")
            if types[0] not in {"byte", "short", "char", "int", "java.lang.String"}:raise ValueError("CONSTRUCTION_SWITCH_TYPE")
            for value in attrs["cases"]:
                if value != "default":java_literal(value, types[0])
            if len({json.dumps(x, sort_keys=True) for x in attrs["cases"]}) != len(attrs["cases"]):raise ValueError("CONSTRUCTION_SWITCH_DUPLICATE")
        elif op == "throw":
            self.arity(n, 1)
            if not self.subtype(types[0], "java.lang.Throwable"):raise ValueError("CONSTRUCTION_THROW_TYPE")
        elif op == "synchronized":
            self.arity(n, 2)
            if not self.reference(types[0]):raise ValueError("CONSTRUCTION_SYNCHRONIZED_TYPE")
        elif op == "try":
            if len(refs) != 1 + len(attrs.get("catches", ())) + bool(attrs.get("finally")):raise ValueError("CONSTRUCTION_TRY_ARITY")
            if not attrs.get("catches") and not attrs.get("finally"):raise ValueError("CONSTRUCTION_TRY_HANDLER_REQUIRED")
            seen = []
            for catch in attrs.get("catches", []):
                t = self.type_name(catch["type_ref"])
                if not self.subtype(t, "java.lang.Throwable"):raise ValueError("CONSTRUCTION_CATCH_TYPE")
                if any(self.subtype(t, previous) for previous in seen):raise ValueError("CONSTRUCTION_CATCH_UNREACHABLE")
                seen.append(t)
        elif op in {"break", "continue"}:self.arity(n, 0)
        elif op != "block":raise ValueError("CONSTRUCTION_OPERATION_UNSUPPORTED")
        return "void"

    def check_contexts(self):
        """Iterative lexical use-site validation; inference cannot prove name scope."""
        returns = "void" if self.declaration["kind"] == "CONSTRUCTOR" else self.declaration.get("return_type", "void")
        initial = {name: (type_, True, False, "parameter:" + name) for name, type_ in self.member_parameters.items()}
        initialized = {"parameter:" + name for name in initial}
        declaration_depth = {binding: 0 for binding in initialized}
        mutations, captured = Counter(), set()
        stack = [("visit", key, initial, returns, 0, 0, False) for key in reversed(self.roots)]
        while stack:
            action, key, env, returns, loops, switches, write = stack.pop()
            n = self.nodes[key];op, refs, attrs = n["op"], n["input_refs"], n["attributes"]
            if action == "for_body":stack.append(("visit", key, dict(env), returns, loops, switches, False));continue
            if action == "declare":
                name = attrs["name"]
                if name in env:raise ValueError("CONSTRUCTION_LOCAL_DUPLICATE")
                env[name] = (n["typed_result"], bool(refs), attrs.get("final", False), key)
                declaration_depth[key] = loops
                if refs:initialized.add(key)
                continue
            if action == "assigned":
                target = self.nodes[refs[0]]
                if target["op"] == "local":
                    name = target["attributes"]["name"]
                    type_, assigned, final, binding = env[name]
                    if final and (assigned or loops > declaration_depth[binding]):raise ValueError("CONSTRUCTION_FINAL_ASSIGNMENT")
                    env[name] = (type_, True, final, binding)
                    mutations[binding] += 2 if loops else 1
                continue
            if op in {"local", "parameter"}:
                name = attrs["name"]
                if name not in env:raise ValueError("CONSTRUCTION_" + op.upper() + "_SCOPE")
                type_, assigned, _, binding = env[name]
                if binding in env.get("@captures", set()):captured.add(binding)
                if type_ != n["typed_result"]:raise ValueError("CONSTRUCTION_LOCAL_SCOPE_TYPE")
                if not assigned and not write:raise ValueError("CONSTRUCTION_LOCAL_UNINITIALIZED")
            if op == "return":
                if (returns == "void" and refs) or (returns != "void" and (not refs or not self.assignable(self.types[refs[0]], returns))):raise ValueError("CONSTRUCTION_RETURN_TYPE")
            if op == "break" and not loops and not switches:raise ValueError("CONSTRUCTION_BREAK_SCOPE")
            if op == "continue" and not loops:raise ValueError("CONSTRUCTION_CONTINUE_SCOPE")
            def push(ref, scope=env, result=returns, loop=loops, switch=switches, writing=False):
                stack.append(("visit", ref, scope, result, loop, switch, writing))
            if op == "declare":
                stack.append(("declare", key, env, returns, loops, switches, False))
                for ref in reversed(refs):push(ref)
            elif op == "block":
                child = dict(env)
                for ref in reversed(refs):push(ref, child)
            elif op == "lambda":
                child = dict(env);signature = self.symbols[n["typed_result"]]
                child["@captures"] = {value[3] for name, value in env.items() if name != "@captures"}
                for name, type_ in zip(attrs["parameters"], signature["functional_parameters"]):
                    if name in child:raise ValueError("CONSTRUCTION_LAMBDA_PARAMETER_COLLISION")
                    child[name] = (type_, True, False, key + ":" + name)
                    initialized.add(key + ":" + name)
                push(refs[0], child, self.lambda_returns[key], 0, 0)
            elif op == "if":
                for ref in reversed(refs[1:]):push(ref, dict(env))
                push(refs[0])
            elif op in {"while", "do_while"}:
                push(refs[1], dict(env), loop=loops + 1);push(refs[0])
            elif op == "for":
                child = dict(env)
                push(refs[2], child, loop=loops + 1)
                stack.append(("for_body", refs[3], child, returns, loops + 1, switches, False))
                push(refs[1], child);push(refs[0], child)
            elif op == "foreach":
                child = dict(env);name = attrs["name"]
                if name in child:raise ValueError("CONSTRUCTION_LOCAL_DUPLICATE")
                child[name] = (attrs["element_type"], True, False, key)
                initialized.add(key)
                push(refs[1], child, loop=loops + 1);push(refs[0])
            elif op == "switch":
                for ref in reversed(refs[1:]):push(ref, dict(env), switch=switches + 1)
                push(refs[0])
            elif op == "try":
                if attrs.get("finally"):push(refs[-1], dict(env))
                for catch, ref in reversed(list(zip(attrs.get("catches", []), refs[1:]))):
                    child = dict(env);name = catch["name"]
                    if name in child:raise ValueError("CONSTRUCTION_LOCAL_DUPLICATE")
                    child[name] = (catch["type_ref"], True, False, key + ":" + name)
                    initialized.add(key + ":" + name);push(ref, child)
                push(refs[0], dict(env))
            elif op == "synchronized":push(refs[1], dict(env));push(refs[0])
            elif op == "assign":
                stack.append(("assigned", key, env, returns, loops, switches, False))
                push(refs[1]);push(refs[0], writing=attrs.get("operator", "=") == "=")
            else:
                for ref in reversed(refs):push(ref)
        if any(mutations[binding] and (binding in initialized or mutations[binding] > 1) for binding in captured):raise ValueError("CONSTRUCTION_LAMBDA_CAPTURE_MUTABLE")

    def check_completion(self):
        """Conservative structured completion; target javac remains authoritative."""
        completes, breaks, continues = {}, {}, {}
        for key in self.order:
            n = self.nodes[key];op, refs, attrs = n["op"], n["input_refs"], n["attributes"]
            complete, escaping_break, escaping_continue = True, False, False
            if op in {"return", "throw", "break", "continue"}:complete, escaping_break, escaping_continue = False, op == "break", op == "continue"
            elif op == "block":
                for ref in refs:
                    if not complete:raise ValueError("CONSTRUCTION_STATEMENT_UNREACHABLE")
                    complete = completes[ref];escaping_break |= breaks[ref];escaping_continue |= continues[ref]
            elif op == "if":
                complete = len(refs) == 2 or any(completes[r] for r in refs[1:])
                escaping_break = any(breaks[r] for r in refs[1:]);escaping_continue = any(continues[r] for r in refs[1:])
            elif op in {"while", "for", "do_while"}:
                condition, body = (refs[1], refs[3]) if op == "for" else (refs[0], refs[1])
                c = self.nodes[condition]
                if op != "do_while" and c["op"] == "literal" and c["attributes"]["value"] is False:raise ValueError("CONSTRUCTION_STATEMENT_UNREACHABLE")
                forever = c["op"] == "literal" and c["attributes"]["value"] is True
                complete = not forever or breaks[body]
                if op == "do_while" and not completes[body] and not breaks[body] and not continues[body]:complete = False
            elif op == "switch":
                complete = "default" not in attrs["cases"] or any(completes[r] or breaks[r] for r in refs[1:])
                escaping_continue = any(continues[r] for r in refs[1:])
            elif op == "try":
                bodies = refs[:-1] if attrs.get("finally") else refs
                complete = any(completes[r] for r in bodies)
                escaping_break = any(breaks[r] for r in bodies);escaping_continue = any(continues[r] for r in bodies)
                if attrs.get("finally"):
                    complete &= completes[refs[-1]]
                    escaping_break = breaks[refs[-1]] or (escaping_break and completes[refs[-1]])
                    escaping_continue = continues[refs[-1]] or (escaping_continue and completes[refs[-1]])
            elif op == "synchronized":complete, escaping_break, escaping_continue = completes[refs[1]], breaks[refs[1]], continues[refs[1]]
            elif op == "lambda":
                if self.lambda_returns[key] != "void" and completes[refs[0]]:raise ValueError("CONSTRUCTION_RETURN_MISSING")
            completes[key], breaks[key], continues[key] = complete, escaping_break, escaping_continue
        if self.declaration["kind"] == "FIELD":return
        complete = True
        for key in self.roots:
            if not complete:raise ValueError("CONSTRUCTION_STATEMENT_UNREACHABLE")
            complete = completes[key]
        if self.declaration["kind"] == "METHOD" and self.declaration.get("return_type", "void") != "void" and complete:raise ValueError("CONSTRUCTION_RETURN_MISSING")


def validate_java_operations(task, nodes, symbols):
    JavaOperations(task, nodes, symbols)
