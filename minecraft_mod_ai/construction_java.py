"""Host-owned Java syntax from typed operations; no model source fallback.

Templates are ropes, flattened once, so deeply nested graphs do not require
Python recursion or quadratic storage for intermediate source strings.
"""
from __future__ import annotations
from .construction_type_contracts import owner_package

from .java_operation_ir import (JavaOperations, EXPRESSIONS, STATEMENTS, identifier,
                                qualified_name, parameter_records, parameter_types, java_literal)


def _join(separator, values):
    parts = []
    for value in values:
        if parts:
            parts.append(separator)
        parts.append(value)
    return tuple(parts)


def _flatten(template):
    parts, stack = [], [template]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            parts.append(value)
        else:
            stack.extend(reversed(value))
    return "".join(parts)


def render_task_member(task, nodes, symbols):
    graph = JavaOperations(task, nodes, symbols)
    expressions, statements, raw_calls = {}, {}, {}

    def expr(key):
        if key not in expressions:
            raise ValueError("CONSTRUCTION_EXPRESSION_REQUIRED: " + key)
        return expressions[key]

    def stmt(key):
        if key not in statements:
            raise ValueError("CONSTRUCTION_STATEMENT_REQUIRED: " + key)
        return statements[key]

    def build_expr(key):
        n = graph.nodes[key]
        op, attrs, refs = n["op"], n["attributes"], n["input_refs"]
        values = [expr(r) for r in refs] if op != "lambda" else []
        if op == "literal":
            if attrs["value"] is None:
                return ("((", n["typed_result"], ") null)")
            return java_literal(attrs["value"], n["typed_result"])
        if op in {"parameter", "local"}:
            return identifier(attrs["name"])
        if op == "this":
            return "this"
        if op in {"call", "field", "new"}:
            row = graph.symbol(attrs["symbol_ref"])
            if op == "new":
                if row.get('enclosing_instance_type'):
                    enclosing = values.pop(0)
                    receiver = ('((', row['enclosing_instance_type'], ') ', enclosing, ').new ', identifier(row['owner'].rsplit('.', 1)[-1]))
                else:
                    receiver = ("new ", qualified_name(row["owner"]))
            elif row.get("static"):
                receiver = qualified_name(row["owner"])
            else:
                value = values.pop(0)
                # Resolve against the inspected owner, retaining subclass types where protected
                # access requires them. Formal argument casts fix overload selection.
                protected = row.get("access", 0) & 4 and owner_package(symbols, row.get('declaring_owner') or row['owner']) != owner_package(symbols, graph.declaration['owner'])
                receiver = ("(", value, ")") if protected else ("((", row["owner"], ") ", value, ")")
            if op == "field":
                return (receiver, ".", identifier(row["name"]))
            args = [("((", type_, ") ", value, ")") for type_, value in zip(parameter_types(row), values)]
            call = (receiver, "" if op == "new" else "." + identifier(row["name"]), "(", _join(", ", args), ")")
            if op == "new":
                return call
            raw_calls[key] = call
            # Keep the descriptor's erased result type despite generic source inference.
            return call if row["return_type"] == "void" else ("((", row["return_type"], ") ", call, ")")
        if op == "binary":
            return ("(", values[0], " " + attrs["operator"] + " ", values[1], ")")
        if op == "unary":
            return ("(", attrs["operator"], values[0], ")")
        if op == "assign":
            return ("(", values[0], " " + attrs.get("operator", "=") + " ", values[1], ")")
        if op == "cast":
            return ("((", graph.type_name(attrs["type_ref"]), ") ", values[0], ")")
        if op == "instanceof":
            return ("(", values[0], " instanceof ", graph.type_name(attrs["type_ref"]), ")")
        if op == "array_get":
            return ("(", values[0], ")[", values[1], "]")
        if op == "array_length":
            return ("(", values[0], ").length")
        if op == "new_array":
            tail = tuple(("[", value, "]") for value in values) if attrs.get("sized") else ("[] {", _join(", ", values), "}")
            return ("new ", graph.type_name(attrs["element_type"]), tail)
        if op == "ternary":
            return ("((", n["typed_result"], ") (", values[0], " ? ", values[1], " : ", values[2], "))")
        if op == "lambda":
            names = ", ".join(identifier(x) for x in attrs["parameters"])
            return ("((", n["typed_result"], ") ((", names, ") -> {\n", stmt(refs[0]), "\n}))")
        raise ValueError("CONSTRUCTION_EXPRESSION_REQUIRED: " + op)

    def build_stmt(key):
        n = graph.nodes[key]
        op, attrs, refs = n["op"], n["attributes"], n["input_refs"]
        if op == "block":
            return ("{\n", _join("\n", (stmt(r) for r in refs)), "\n}")
        if op == "declare":
            return ("final " if attrs.get("final") else "", n["typed_result"], " ", identifier(attrs["name"]),
                                   (" = ", expr(refs[0])) if refs else "", ";")
        if op == 'constructor_call':
            row = graph.symbol(attrs['symbol_ref'])
            values = [expr(ref) for ref in refs]
            receiver = attrs['target']
            if row.get('enclosing_instance_type'):
                receiver = ('((', row['enclosing_instance_type'], ') ', values.pop(0), ').', receiver)
            arguments = [('((', type_, ') ', value, ')') for type_, value in zip(parameter_types(row), values)]
            return (receiver, '(', _join(', ', arguments), ');')
        if op == "return":
            return ("return", (" ", expr(refs[0])) if refs else "", ";")
        if op == "if":
            return ("if (", expr(refs[0]), ") {\n", stmt(refs[1]), "\n}",
                               (" else {\n", stmt(refs[2]), "\n}") if len(refs) == 3 else "")
        if op == "while":
            return ("while (", expr(refs[0]), ") {\n", stmt(refs[1]), "\n}")
        if op == "do_while":
            return ("do {\n", stmt(refs[1]), "\n} while (", expr(refs[0]), ");")
        if op == "for":
            return ("for (", stmt(refs[0])[:-1], "; ", expr(refs[1]), "; ", stmt(refs[2])[:-1], ") {\n", stmt(refs[3]), "\n}")
        if op == "foreach":
            return ("for (", graph.type_name(attrs["element_type"]), " ", identifier(attrs["name"]),
                                    " : ", expr(refs[0]), ") {\n", stmt(refs[1]), "\n}")
        if op == "switch":
            cases = []
            for value, body in zip(attrs["cases"], refs[1:]):
                label = "default" if value['kind'] == "default" else "case " + java_literal(value['value'], graph.types[refs[0]])
                cases.append((label, " -> {\n", stmt(body), "\n}"))
            return ("switch (", expr(refs[0]), ") {\n", _join("\n", cases), "\n}")
        if op == "throw":
            return ("throw ", expr(refs[0]), ";")
        if op == "synchronized":
            return ("synchronized (", expr(refs[0]), ") {\n", stmt(refs[1]), "\n}")
        if op == "try":
            parts = ["try {\n", stmt(refs[0]), "\n}"]
            for catch, body in zip(attrs.get("catches", ()), refs[1:]):
                parts.extend((" catch (", graph.type_name(catch["type_ref"]), " ", identifier(catch["name"]), ") {\n", stmt(body), "\n}"))
            if attrs.get("finally"):
                parts.extend((" finally {\n", stmt(refs[-1]), "\n}"))
            return tuple(parts)
        if op in {"break", "continue"}:
            return (op, ";")
        if op == "assign":
            return (*expr(key)[1:-1], ";")
        if op == "call":
            return (raw_calls[key], ";")
        if op == "new":
            return (expr(key), ";")
        raise ValueError("CONSTRUCTION_STATEMENT_REQUIRED: " + op)

    for key in graph.order:
        op = graph.nodes[key]["op"]
        if op in EXPRESSIONS:
            expressions[key] = build_expr(key)
        if op in STATEMENTS:
            statements[key] = build_stmt(key)
    row = graph.declaration
    visibility = row.get("visibility", "public")
    if visibility not in {"public", "protected", "private", ""}:
        raise ValueError("CONSTRUCTION_VISIBILITY")
    modifiers = " ".join(x for x in (visibility, "static" if row.get("static") else "",
                                    'final' if row['kind'] == 'METHOD' and row.get('final') else '') if x)
    prefix = modifiers + " " if modifiers else ""
    name = identifier(row["name"]) if row["kind"] != "CONSTRUCTOR" else None
    if row["kind"] == "FIELD":
        return _flatten((prefix, "final " if row.get("final") else "", graph.type_name(row["return_type"]), " ", name, " = ", expr(graph.roots[0]), ";"))
    params = ", ".join(graph.type_name(p["type"]) + " " + identifier(p["name"]) for p in parameter_records(row))
    if row["kind"] == "CONSTRUCTOR":
        header = prefix + identifier(row["owner"].rsplit(".", 1)[-1]) + "(" + params + ")"
    else:
        header = prefix + graph.type_name(row.get("return_type", "void")) + " " + name + "(" + params + ")"
    annotations = row.get("annotations", ())
    for annotation in annotations:
        qualified_name(annotation)
    if annotations:
        header = "\n".join("@" + a for a in annotations) + "\n" + header
    if row.get("throws"):
        header += " throws " + ", ".join(graph.type_name(t) for t in row["throws"])
    return _flatten((header, " {\n", _join("\n", (stmt(root) for root in graph.body_roots)), "\n}"))


def render_task_java(task, nodes, symbols):
    row = symbols.get(task.owner_symbol)
    if row is None:
        raise ValueError("CONSTRUCTION_SYMBOL_MISSING: " + task.owner_symbol)
    owner = qualified_name(row["owner"])
    package, _, class_name = owner.rpartition(".")
    member = render_task_member(task, nodes, symbols)
    header = "public class " + class_name
    if row.get("class_superclass"):
        header += " extends " + row["class_superclass"]
    if row.get("class_interfaces"):
        header += " implements " + ", ".join(row["class_interfaces"])
    source = ("package " + package + ";\n\n" if package else "") + header + " {\n" + member + "\n}\n"
    from .java_region_parser import _parse
    _parse(source)
    return source
