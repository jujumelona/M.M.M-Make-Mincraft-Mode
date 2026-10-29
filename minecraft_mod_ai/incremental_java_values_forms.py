"""Host punctuation for recursively authored Java operators and array values."""
from __future__ import annotations

from .incremental_java_context import _erase
from .incremental_java_schema import choice, obj
from .incremental_java_values_model import JavaValue, count, request

_BINARY = ("+", "-", "*", "/", "%", "&&", "||", "&", "|", "^", "==", "!=", "<", "<=", ">", ">=", "<<", ">>", ">>>")
_BOOLEAN = frozenset({"&&", "||", "==", "!=", "<", "<=", ">", ">="})

# A large runtime array is still implementable with allocation and loops.
# Model-driven literal element authoring must never request unbounded calls.
MAX_LITERAL_ARRAY_ELEMENTS = 255


class ValueForms:
    def __init__(self, author):
        self.author, self.builder = author, author.builder

    @property
    def frame(self):
        return self.author.frame

    def operand(self, slot, expected="java.lang.Object", *, writable=False):
        value = self.author.value(slot, expected, writable=writable)
        self.frame[slot] = value.source
        return value

    def binary(self, expected):
        operator = request(self.builder, choice("operator", _BINARY), "binary_operator")["operator"]
        self.frame["operator"] = operator
        hint = "boolean" if operator in {"&&", "||"} else "java.lang.Object" if operator in _BOOLEAN else expected
        left = self.operand("binary:left", hint)
        right = self.operand("binary:right", hint)
        result_type = "boolean" if operator in _BOOLEAN else left.java_type
        if operator == "+" and "java.lang.String" in {left.java_type, right.java_type}:
            result_type = "java.lang.String"
        return JavaValue("(" + left.source + " " + operator + " " + right.source + ")", result_type)

    def unary(self, expected):
        operator = request(self.builder, choice("operator", ("+", "-", "!", "~")), "unary_operator")["operator"]
        self.frame["operator"] = operator
        value = self.operand("unary:value", "boolean" if operator == "!" else expected)
        return JavaValue("(" + operator + "(" + value.source + "))", "boolean" if operator == "!" else value.java_type)

    def update(self, expected):
        page = request(self.builder, obj({"operator": {"type": "string", "enum": ["++", "--"]},
            "position": {"type": "string", "enum": ["prefix", "postfix"]}}), "update_operator")
        self.frame.update(page)
        value = self.operand("update:target", expected, writable=True)
        code = page["operator"] + value.source if page["position"] == "prefix" else value.source + page["operator"]
        return JavaValue(code, value.java_type)

    def conditional(self, expected):
        condition = self.operand("condition", "boolean")
        then = self.operand("then", expected)
        otherwise = self.operand("else", expected)
        return JavaValue("(" + condition.source + " ? " + then.source + " : " + otherwise.source + ")", then.java_type)

    def cast(self, expected):
        java_type = self.builder.select_java_type("cast_type", expected=expected)
        self.frame["cast_type"] = java_type
        value = self.operand("cast:value")
        return JavaValue("((" + java_type + ") (" + value.source + "))", java_type)

    def instanceof(self, expected):
        value = self.operand("instanceof:value")
        java_type = self.builder.select_java_type("instanceof_type", reference_only=True)
        return JavaValue("(" + value.source + " instanceof " + java_type + ")", "boolean")

    def index(self, expected):
        array = self.operand("array")
        index = self.operand("index", "int")
        java_type = array.java_type[:-2] if array.java_type.endswith("[]") else expected
        return JavaValue("(" + array.source + ")[" + index.source + "]", java_type, writable=True)

    def array(self, expected, *, java_type=None):
        java_type = java_type or self.builder.select_java_type("array_type", expected=expected)
        if not java_type.endswith("[]"):
            java_type += "[]"
        self.frame["array_type"] = java_type
        mode = request(self.builder, choice("mode", ("elements", "sizes")), "array_mode")["mode"]
        self.frame["mode"] = mode
        if mode == "elements":
            code = self._array_elements(java_type)
        else:
            code = self._array_sizes(java_type)
        if "<" in java_type:
            code = "((" + java_type + ") " + code + ")"
        return JavaValue("(" + code + ")", java_type)

    def _array_elements(self, java_type):
        length = count(self.builder, "array_elements_count", maximum=MAX_LITERAL_ARRAY_ELEMENTS)
        self.frame["elements"] = []
        for index in range(length):
            value = self.author.value("element:" + str(index), java_type[:-2])
            self.frame["elements"].append(value.source)
        return "new " + _erase(java_type) + " {" + ", ".join(self.frame["elements"]) + "}"

    def _array_sizes(self, java_type):
        base, dimensions = java_type, 0
        while base.endswith("[]"):
            base, dimensions = base[:-2], dimensions + 1
        initialized = 1 if dimensions == 1 else count(self.builder, "array_dimensions_count", minimum=1, maximum=dimensions)
        self.frame["sizes"] = []
        for index in range(initialized):
            value = self.author.value("size:" + str(index), "int")
            self.frame["sizes"].append(value.source)
        return "new " + _erase(base) + "".join("[" + value + "]" for value in self.frame["sizes"]) + "[]" * (dimensions - initialized)

    def class_literal(self, expected):
        java_type = _erase(self.builder.select_java_type("class_literal_type", allow_void=True))
        boxed = {"boolean": "java.lang.Boolean", "byte": "java.lang.Byte", "short": "java.lang.Short",
                 "char": "java.lang.Character", "int": "java.lang.Integer", "long": "java.lang.Long",
                 "float": "java.lang.Float", "double": "java.lang.Double", "void": "java.lang.Void"}
        return JavaValue(java_type + ".class", "java.lang.Class<" + boxed.get(java_type, java_type) + ">")
