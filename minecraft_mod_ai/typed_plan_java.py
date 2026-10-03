from __future__ import annotations

"""Deterministic Java backend for typed PlanIR."""

import math
from collections.abc import Mapping
from typing import Any

from .typed_plan_ir import validate_typed_plan_ir

_JAVA_TYPES = {
    "void": "void",
    "boolean": "boolean",
    "int": "int",
    "long": "long",
    "double": "double",
    "string": "String",
    "object": "Object",
}


def _string_expr(value: str) -> str:
    if value == "":
        return '""'
    raw = value.encode("utf-16-be", "surrogatepass")
    units = [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw), 2)]
    return "new String(new char[]{" + ", ".join(f"(char){unit}" for unit in units) + "})"


def _java_type(kind: str) -> str:
    if kind in _JAVA_TYPES:
        return _JAVA_TYPES[kind]
    if kind.startswith("list<"):
        return "java.util.List<?>"
    if kind == "map":
        return "java.util.Map<String, Object>"
    raise ValueError(f"unsupported Java type {kind!r}")


def _boxed_type(kind: str) -> str:
    return {
        "boolean": "Boolean", "int": "Integer", "long": "Long", "double": "Double",
        "string": "String", "object": "Object",
    }[kind]


def _unbox(kind: str, expression: str) -> str:
    if kind == "boolean":
        return f"((Boolean) {expression}).booleanValue()"
    if kind == "int":
        return f"((Integer) {expression}).intValue()"
    if kind == "long":
        return f"((Long) {expression}).longValue()"
    if kind == "double":
        return f"((Double) {expression}).doubleValue()"
    if kind == "string":
        return f"((String) {expression})"
    if kind == "object":
        return expression
    raise ValueError(f"cannot unbox {kind!r}")


def _default_check(kind: str, expression: str) -> str:
    if kind == "object":
        return "true"
    return f"{expression} instanceof {_boxed_type(kind)}"


class _Renderer:
    def __init__(self, plan: Mapping[str, Any], capabilities: Mapping[str, Any] | None) -> None:
        self.plan = plan
        self.capabilities = dict(capabilities or {})
        self.counter = 0

    def unique(self, stem: str) -> str:
        self.counter += 1
        return f"__mmm_{stem}_{self.counter}"

    def expr(self, node: Mapping[str, Any]) -> str:
        op = node["op"]
        if op == "literal":
            kind = node["type"]
            value = node["value"]
            if kind == "string":
                return _string_expr(value)
            if kind == "boolean":
                return "true" if value else "false"
            if kind == "long":
                return f"{value}L"
            if kind == "double":
                number = float(value)
                if math.isnan(number) or math.isinf(number):
                    raise ValueError("double literal must be finite")
                return repr(number) + "d"
            if kind == "int":
                return str(value)
            if kind == "object":
                if value is None:
                    return "null"
                if isinstance(value, str):
                    return _string_expr(value)
                if isinstance(value, bool):
                    return "Boolean.TRUE" if value else "Boolean.FALSE"
                if isinstance(value, int):
                    return f"Integer.valueOf({value})"
                if isinstance(value, float):
                    return f"Double.valueOf({repr(float(value))}d)"
                raise ValueError("unsupported object literal")
        if op == "ref":
            return str(node["name"])
        if op == "unary":
            return f"({node['operator']}{self.expr(node['value'])})"
        if op == "binary":
            return f"({self.expr(node['left'])} {node['operator']} {self.expr(node['right'])})"
        if op == "list":
            return "java.util.List.of(" + ", ".join(self.expr(item) for item in node["items"]) + ")"
        if op == "map":
            entries = node["entries"]
            if not entries:
                return "java.util.Map.<String, Object>of()"
            rendered = ", ".join(
                "java.util.Map.entry(" + self.expr(entry["key"]) + ", (Object) (" + self.expr(entry["value"]) + "))"
                for entry in entries
            )
            return "java.util.Map.<String, Object>ofEntries(" + rendered + ")"
        if op == "call":
            return "fn_" + node["function"] + "(" + ", ".join(self.expr(arg) for arg in node["args"]) + ")"
        if op == "capability":
            contract = self.capabilities[node["id"]]
            return f"{contract['owner']}.{contract['method']}(" + ", ".join(self.expr(arg) for arg in node["args"]) + ")"
        if op == "state_get":
            call = "AuthoredStateModel.getState(" + self.expr(node["key"]) + ", " + self.expr(node["context"]) + ")"
            return _unbox(node["type"], call)
        raise ValueError(f"unsupported expression op {op!r}")

    def block(self, body: list[Mapping[str, Any]], indent: str = "        ") -> list[str]:
        lines: list[str] = []
        for statement in body:
            op = statement["op"]
            if op == "let":
                lines.append(f"{indent}{_java_type(statement['type'])} {statement['name']} = {self.expr(statement['value'])};")
            elif op == "set":
                lines.append(f"{indent}{statement['name']} = {self.expr(statement['value'])};")
            elif op == "return":
                if "value" in statement and statement["value"] is not None:
                    lines.append(f"{indent}return {self.expr(statement['value'])};")
                else:
                    lines.append(f"{indent}return;")
            elif op == "assert":
                lines.append(
                    f"{indent}if (!({self.expr(statement['condition'])})) "
                    + "throw new IllegalStateException(" + _string_expr(statement["message"]) + ");"
                )
            elif op == "if":
                lines.append(f"{indent}if ({self.expr(statement['condition'])}) {{")
                lines.extend(self.block(statement["then"], indent + "    "))
                lines.append(f"{indent}}} else {{")
                lines.extend(self.block(statement["else"], indent + "    "))
                lines.append(f"{indent}}}")
            elif op == "while":
                counter = self.unique("loop")
                lines.append(f"{indent}int {counter} = 0;")
                lines.append(f"{indent}while ({self.expr(statement['condition'])}) {{")
                lines.append(
                    f"{indent}    if ({counter}++ >= {int(statement['max_iterations'])}) "
                    f"throw new IllegalStateException(\"typed loop bound exceeded\");"
                )
                lines.extend(self.block(statement["body"], indent + "    "))
                lines.append(f"{indent}}}")
            elif op == "foreach":
                raw_name = self.unique("item")
                lines.append(f"{indent}for (Object {raw_name} : {self.expr(statement['collection'])}) {{")
                lines.append(
                    f"{indent}    {_java_type(statement['type'])} {statement['name']} = "
                    f"{_unbox(statement['type'], raw_name)};"
                )
                lines.extend(self.block(statement["body"], indent + "    "))
                lines.append(f"{indent}}}")
            elif op == "state_set":
                lines.append(
                    f"{indent}AuthoredStateModel.setState({self.expr(statement['key'])}, "
                    f"{self.expr(statement['value'])}, {self.expr(statement['context'])});"
                )
            elif op == "expr":
                lines.append(f"{indent}{self.expr(statement['value'])};")
            else:
                raise ValueError(f"unsupported statement op {op!r}")
        return lines

    def dispatcher_case(self, fn: Mapping[str, Any]) -> list[str]:
        function_id = str(fn["id"])
        lines = [f"            case \"{function_id}\": {{"]
        params = list(fn["parameters"])
        required_keys = [str(param["name"]) for param in params]
        if required_keys:
            predicates = [f"args.containsKey(\"{name}\")" for name in required_keys]
            lines.append(
                f"                if (args == null || args.size() != {len(required_keys)} || !({' && '.join(predicates)})) "
                "throw new IllegalArgumentException(\"invalid arguments for " + function_id + "\");"
            )
        else:
            lines.append(
                "                if (args == null || !args.isEmpty()) "
                "throw new IllegalArgumentException(\"invalid arguments for " + function_id + "\");"
            )
        arg_names: list[str] = []
        for index, param in enumerate(params):
            raw = f"__arg_{index}"
            kind = str(param["type"])
            lines.append(f"                Object {raw} = args.get(\"{param['name']}\");")
            lines.append(
                f"                if (!({_default_check(kind, raw)})) "
                f"throw new IllegalArgumentException(\"invalid type for {param['name']}\");"
            )
            arg_names.append(_unbox(kind, raw))
        call = f"fn_{function_id}(" + ", ".join(arg_names) + ")"
        if fn["return_type"] == "void":
            lines.append(f"                {call};")
            lines.append("                return null;")
        else:
            lines.append(f"                return {call};")
        lines.append("            }")
        return lines

    def render(self, package: str) -> str:
        lines: list[str] = []
        if package:
            lines.extend([f"package {package};", ""])
        lines.extend([
            "// MMM:TYPED_PLAN_OWNER",
            "public final class AuthoredProgram {",
            "    private AuthoredProgram() {}",
            "",
            "    public static Object invoke(String function, java.util.Map<String, Object> args) {",
            "        if (function == null) throw new IllegalArgumentException(\"function is required\");",
            "        switch (function) {",
        ])
        for fn in self.plan["functions"]:
            lines.extend(self.dispatcher_case(fn))
        lines.extend([
            "            default: throw new IllegalArgumentException(\"unknown function: \" + function);",
            "        }",
            "    }",
            "",
            "    public static void initialize() {",
        ])
        lines.extend(self.block(self.plan["initialize"], "        "))
        lines.extend(["    }", ""])
        for fn in self.plan["functions"]:
            params = ", ".join(
                f"{_java_type(param['type'])} {param['name']}" for param in fn["parameters"]
            )
            lines.append(
                f"    private static {_java_type(fn['return_type'])} fn_{fn['id']}({params}) {{"
            )
            lines.extend(self.block(fn["body"], "        "))
            lines.extend(["    }", ""])
        lines.append("}")
        return "\n".join(lines) + "\n"


def render_typed_plan_java(
    plan: Mapping[str, Any], *, package: str = "",
    capabilities: Mapping[str, Any] | None = None,
) -> str:
    validated = validate_typed_plan_ir(plan, capabilities=capabilities)
    return _Renderer(validated, capabilities).render(str(package or "").strip())


__all__ = ["render_typed_plan_java"]
