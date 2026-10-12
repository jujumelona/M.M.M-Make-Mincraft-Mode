"""Build native Java fragment grammars from the vendored Tree-sitter grammar.

This is a resource compiler, not a model retry or a generated-code repair.
Nullable and unit productions are removed before a selective left-corner
transformation. Lexemes stay atomic; immediate tokens do not acquire whitespace.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from re import _constants as op, _parser as regex_parser

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools/grammars/tree-sitter-java-0.23.5.json"
OUTPUT = REPO / "minecraft_mod_ai/data/java_source_grammar.json"
KEYWORDS = set("abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while _ true false null".split())


def literal(text):
    return json.dumps(text, ensure_ascii=False)


def character(value):
    return f"\\x{value:02X}" if value <= 0xFF else f"\\u{value:04X}" if value <= 0xFFFF else f"\\U{value:08X}"


def pattern(source):
    def emit(items):
        parts = []
        for kind, value in items:
            if kind == op.LITERAL:
                parts.append(literal(chr(value)))
            elif kind == op.NOT_LITERAL:
                parts.append("[^" + character(value) + "]")
            elif kind == op.ANY:
                parts.append(r"[^\n]")
            elif kind == op.IN:
                text = ""
                for item, data in value:
                    if item == op.NEGATE:
                        text += "^"
                    elif item == op.LITERAL:
                        text += character(data)
                    elif item == op.RANGE:
                        text += character(data[0]) + "-" + character(data[1])
                    else:
                        raise ValueError(f"unsupported Java lexical class: {item}")
                parts.append("[" + text + "]")
            elif kind in {op.MAX_REPEAT, op.MIN_REPEAT}:
                minimum, maximum, child = value
                suffix = ("*" if minimum == 0 else "+" if minimum == 1 else f"{{{minimum},}}") if maximum == op.MAXREPEAT else (f"{{{minimum}}}" if minimum == maximum else f"{{{minimum},{maximum}}}")
                parts.append("(" + emit(child) + ")" + suffix)
            elif kind == op.SUBPATTERN:
                parts.append("(" + emit(value[-1]) + ")")
            elif kind == op.BRANCH:
                parts.append("(" + " | ".join(emit(branch) for branch in value[1]) + ")")
            else:
                raise ValueError(f"unsupported Java lexical pattern: {kind}")
        return " ".join(parts) or '""'
    return emit(regex_parser.parse(source, 0))


def identifier_rules():
    """Use the same ASCII symbol alphabet as the production metadata contract."""
    trie = {}
    for word in sorted(KEYWORDS):
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = None
    result = {}
    def build(node, first=False):
        name = "id" + str(len(result))
        result[name] = ""
        children = sorted(key for key in node if key)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$" + ("" if first else "0123456789")
        remaining = "".join(character(ord(char)) for char in alphabet if char not in children)
        choices = ["[" + remaining + "] [A-Za-z0-9_$]*"] if remaining else []
        for char in children:
            choices.append(literal(char) + " " + build(node[char]))
        if not first and "" not in node:
            choices.append('""')
        result[name] = " | ".join(choices)
        return name
    root = build(trie, first=True)
    return root, result


class Compiler:
    def __init__(self, source):
        self.source = source
        self.rules = {}
        self.tokens = {}
        self.symbols = {name: "g" + str(index) for index, name in enumerate(source["rules"])}
        self.counter = 0
        self.id_root, self.lexical = identifier_rules()

    def helper(self):
        self.counter += 1
        return "h" + str(self.counter)

    def regex_node(self, node):
        kind = node["type"]
        if kind == "STRING":
            return literal(node["value"])
        if kind == "PATTERN":
            if "XID_Start" in node["value"]:
                return self.id_root
            return pattern(node["value"])
        if kind == "SYMBOL":
            return self.regex_node(self.source["rules"][node["name"]])
        if kind in {"TOKEN", "IMMEDIATE_TOKEN", "FIELD", "ALIAS", "PREC", "PREC_LEFT", "PREC_RIGHT", "PREC_DYNAMIC"}:
            return self.regex_node(node["content"])
        if kind in {"SEQ", "CHOICE"}:
            separator = " " if kind == "SEQ" else " | "
            return "(" + separator.join(self.regex_node(item) for item in node["members"]) + ")"
        if kind in {"REPEAT", "REPEAT1"}:
            return "(" + self.regex_node(node["content"]) + ")" + ("*" if kind == "REPEAT" else "+")
        if kind == "BLANK":
            return '""'
        raise ValueError(f"unsupported Java token rule: {kind}")

    def token(self, expression, *, immediate=False):
        name = "t" + str(len(self.tokens))
        # Required separation prevents a keyword and following identifier from
        # merging into another identifier. Inside literals, the token is atomic.
        self.tokens[name] = expression if immediate else '" " ' + expression
        return name

    def lower(self, node):
        kind = node["type"]
        if kind == "SYMBOL":
            return self.symbols[node["name"]]
        if kind in {"FIELD", "ALIAS", "PREC", "PREC_LEFT", "PREC_RIGHT", "PREC_DYNAMIC"}:
            return self.lower(node["content"])
        if kind in {"STRING", "PATTERN", "TOKEN", "IMMEDIATE_TOKEN"}:
            return self.token(self.regex_node(node), immediate=kind == "IMMEDIATE_TOKEN")
        name = self.helper()
        if kind == "BLANK":
            self.rules[name] = {()}
        elif kind == "SEQ":
            self.rules[name] = {tuple(self.lower(child) for child in node["members"])}
        elif kind == "CHOICE":
            self.rules[name] = {(self.lower(child),) for child in node["members"]}
        elif kind in {"REPEAT", "REPEAT1"}:
            child = self.lower(node["content"])
            self.rules[name] = {(), (child, name)} if kind == "REPEAT" else {(child,), (child, name)}
        else:
            raise ValueError(f"unsupported Java production: {kind}")
        return name

    def load(self):
        # A literal is one lexeme. Threading token separation through its
        # children would insert spaces into the user's string values.
        # Java translates Unicode escapes before tokenizing the source. Emit
        # Unicode values directly so an escape cannot create a quote, newline,
        # or backslash outside this literal's structural grammar.
        self.lexical.update({
            "java-escape": r''' "\\" (["'\\btnfrs] | [0-3] [0-7]{2} | [0-7]{1,2}) ''',
            "java-string": r'"\"" ([^"\\\r\n] | java-escape)* "\""',
            "java-char": r''' "'" ([^'\\\r\n\U00010000-\U0010FFFF] | java-escape) "'" ''',
            "java-text-block": r'"\"\"\"" [ \t\x0C]* "\r"? "\n" ([^"\\] | java-escape | "\\" "\r"? "\n" | "\"" [^"\\] | "\"\"" [^"\\])* "\"\"\""',
        })
        for name, node in self.source["rules"].items():
            if name == "string_literal":
                child = self.token("(java-string | java-text-block)")
            elif name == "character_literal":
                child = self.token("java-char")
            else:
                child = self.lower(node)
            self.rules[self.symbols[name]] = {(child,)}

    def reachable(self, entries):
        found = set(entries)
        pending = list(entries)
        while pending:
            for production in self.rules.get(pending.pop(), ()):
                for symbol in production:
                    if symbol not in found:
                        found.add(symbol)
                        pending.append(symbol)
        self.rules = {name: rules for name, rules in self.rules.items() if name in found}
        self.tokens = {name: rules for name, rules in self.tokens.items() if name in found}

    def normalize(self, entries):
        nullable = set()
        while True:
            more = {name for name, productions in self.rules.items() if any(all(symbol in nullable for symbol in production) for production in productions)} - nullable
            if not more:
                break
            nullable.update(more)
        nullable_entries = set(entries) & nullable
        for name, productions in self.rules.items():
            expanded = set()
            for production in productions:
                options = [(symbol, None) if symbol in nullable else (symbol,) for symbol in production]
                for selection in itertools.product(*options):
                    value = tuple(symbol for symbol in selection if symbol is not None)
                    if value:
                        expanded.add(value)
            self.rules[name] = expanded
        for name in list(self.rules):
            closure = {name}
            pending = [name]
            while pending:
                for production in self.rules[pending.pop()]:
                    if len(production) == 1 and production[0] in self.rules and production[0] not in closure:
                        closure.add(production[0])
                        pending.append(production[0])
            self.rules[name] = {production for child in closure for production in self.rules[child]
                                if not (len(production) == 1 and production[0] in self.rules)}
        # Selective left-corner transformation keeps shared Java productions
        # factored instead of exponentially substituting all expression paths.
        edges = {name: {production[0] for production in productions if production and production[0] in self.rules}
                 for name, productions in self.rules.items()}
        index_of, low, active, stack, groups = {}, {}, set(), [], []
        def visit(name):
            index_of[name] = low[name] = len(index_of)
            stack.append(name)
            active.add(name)
            for child in sorted(edges[name]):
                if child not in index_of:
                    visit(child)
                    low[name] = min(low[name], low[child])
                elif child in active:
                    low[name] = min(low[name], index_of[child])
            if low[name] == index_of[name]:
                group = set()
                while True:
                    child = stack.pop()
                    active.remove(child)
                    group.add(child)
                    if child == name:
                        break
                if len(group) > 1 or name in edges[name]:
                    groups.append(group)
        for name in edges:
            if name not in index_of:
                visit(name)
        for group in groups:
            members = sorted(group)
            originals = {name: self.rules[name] for name in members}
            bases = {}
            for name in members:
                productions = {p for p in originals[name] if p and p[0] not in group}
                if productions:
                    base = self.helper()
                    bases[name] = base
                    self.rules[base] = productions
            continuations = {(target, corner): self.helper() for target in members for corner in members}
            for target in members:
                self.rules[target] = {(base, continuations[target, corner]) for corner, base in bases.items()}
                for corner in members:
                    self.rules[continuations[target, corner]] = ({()} if target == corner else set()) | {
                        production[1:] + (continuations[target, owner],)
                        for owner, productions in originals.items()
                        for production in productions if production and production[0] == corner
                    }
        for name in nullable_entries:
            self.rules[name].add(())
        self.reachable(entries)

    def emit(self):
        lines = []
        for name, productions in self.rules.items():
            if not productions:
                raise ValueError(f"unproductive Java production: {name}")
            lines.append(name + " ::= " + " | ".join(" ".join(production) or '""' for production in sorted(productions)))
        lines.extend(name + " ::= " + expression for name, expression in self.tokens.items())
        lines.extend(name + " ::= " + expression for name, expression in self.lexical.items())
        return "\n".join(lines) + "\n"


def build():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    # Tree-sitter accepts some non-Java literal spellings. Keep generation in
    # the Java lexeme language: octal starts with 0, and hex floats require p/P.
    octal_prefix = source["rules"]["octal_integer_literal"]["content"]["members"][0]
    octal_prefix["members"] = [item for item in octal_prefix["members"] if item.get("value") == "0"]
    hex_float = source["rules"]["hex_floating_point_literal"]["content"]["members"]
    hex_float[-1] = hex_float[-1]["members"][0]
    # String templates are not part of the non-preview Java target used by
    # modern Minecraft. Preserve ordinary strings and standard text blocks.
    source["rules"]["primary_expression"]["members"] = [item for item in source["rules"]["primary_expression"]["members"]
        if item.get("name") != "template_expression"]
    # Package/import/module declarations belong outside a method, even though
    # Tree-sitter reuses its declaration choice for top-level statements.
    declarations = source["rules"]["declaration"]["content"]["members"]
    source["rules"]["declaration"]["content"]["members"] = [item for item in declarations
        if item.get("name") not in {"package_declaration", "import_declaration", "module_declaration"}]
    for name, original in (("source_method_content", "block"), ("source_constructor_content", "constructor_body")):
        source["rules"][name] = {"type": "SEQ", "members": source["rules"][original]["members"][1:-1]}
    compiler = Compiler(source)
    compiler.load()
    entries = {"field": compiler.symbols["_variable_initializer"],
               "method": compiler.symbols["source_method_content"],
               "constructor": compiler.symbols["source_constructor_content"]}
    compiler.reachable(entries.values())
    print(f"Java grammar: lowering {len(compiler.rules)} productions", flush=True)
    compiler.normalize(entries.values())
    grammar = compiler.emit()
    payload = {"source": "tree-sitter/tree-sitter-java@0.23.5",
               "source_blob_sha": "2b2f18d9e063c8c8161b4335f49cd04d0b39a9d7",
               "source_commit_sha": "94703d5a6bed02b98e438d7cad1136c01a60ba2c",
               "license": (SOURCE.with_name("tree-sitter-java-LICENSE")).read_text(encoding="utf-8"),
               "entries": entries, "grammar": grammar}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Java grammar: wrote {OUTPUT.name}, {len(grammar)} characters", flush=True)


if __name__ == "__main__":
    build()
