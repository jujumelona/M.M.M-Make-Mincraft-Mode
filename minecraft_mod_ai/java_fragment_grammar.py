"""Small native decoder for JSON-escaped Java fragments inside host frames.

This owns the JSON envelope and delimiter/lexical boundaries, not Java typing.
The Java parser, state checks and compiler still validate the assembled member.
There is no source, literal, statement or identifier length cap.
"""
from __future__ import annotations

import json
import string
from collections.abc import Mapping
from functools import lru_cache


_START = string.ascii_letters + "_$"
_CONTINUE = _START + string.digits
_UNIT_WORDS = ("package", "import", "public", "protected", "private")


def _identifier_grammar(forbidden):
    """A finite keyword trie; adjacent identifier pieces cannot bypass a word."""
    prefixes = {word[:index] for word in forbidden for index in range(len(word) + 1)}
    names = {prefix: f"word-{index}" for index, prefix in enumerate(sorted(prefixes))}
    rules = []
    for prefix in sorted(prefixes):
        children = {word[len(prefix)] for word in forbidden if word.startswith(prefix) and len(word) > len(prefix)}
        allowed = ''.join(char for char in (_CONTINUE if prefix else _START) if char not in children)
        options = [f'[{allowed}] id-continue*', r'[\u0080-\uffff] id-continue*']
        options.extend(json.dumps(char) + ' ' + names[prefix + char] for char in sorted(children))
        if prefix and prefix not in forbidden:
            options.append('""')
        rules.append(names[prefix] + ' ::= ' + ' | '.join(options))
    return '\n'.join(['identifier ::= ' + names[''], *rules])


def _lexical_rules():
    literals = {"jquote": '\\"', "jbackslash": '\\\\', "newline": '\\n', "carriage": '\\r',
                "tab": '\\t', "backspace": '\\b', "formfeed": '\\f', "slash-escape": '\\/',
                "unicode-start": '\\u'}
    rules = [name + ' ::= ' + json.dumps(value) for name, value in literals.items()]
    return '\n'.join(rules) + r'''
ws ::= [ \t\n\r]*
id-continue ::= [A-Za-z0-9_$] | [\u0080-\uffff]
any-id ::= ([A-Za-z_$] | [\u0080-\uffff]) id-continue*
hex ::= [0-9a-fA-F]
safe-unicode ::= [1-9a-fA-F] hex hex hex | "0" [1-9a-fA-F] hex hex | "00" [01346789a-fA-F] hex | "002" [01345689a-fA-F] | "005" [0-9aAbBdDeEfF]
unicode-escape ::= unicode-start safe-unicode
short-escape ::= newline | carriage | tab | backspace | formfeed | slash-escape
string-char ::= [^"\\\x00-\x1f]
char-char ::= [^'"\\\x00-\x1f]
java-escape ::= jbackslash (jquote | jbackslash | [^u"\\\x00-\x1f] | "u"+ safe-unicode)
java-string ::= jquote (string-char | java-escape | short-escape | unicode-escape)* jquote
java-char ::= "'" (char-char | jquote | java-escape | short-escape | unicode-escape)+ "'"
text-continuation ::= jbackslash (newline | carriage newline?)
text-non-quote ::= string-char | java-escape | short-escape | unicode-escape | text-continuation
text-block ::= jquote jquote jquote (text-non-quote | jquote text-non-quote | jquote jquote text-non-quote)* jquote jquote jquote
comment-escape ::= jquote | jbackslash (jbackslash | [^u"\\\x00-\x1f])
line-comment ::= "//" ([^"\\\x00-\x1f] | comment-escape | tab)*
block-char ::= [^*"\\\x00-\x1f] | comment-escape | newline | carriage | tab | backspace | formfeed
block-non-end ::= [^*/"\\\x00-\x1f] | comment-escape | newline | carriage | tab | backspace | formfeed
block-comment ::= "/*" (block-char | "*"+ block-non-end)* "*"+ "/"
blank ::= " " | newline | carriage | tab | line-comment | block-comment
punctuation ::= [!#%&*+,\-./:<=>?@^|~]
number ::= [0-9] [A-Za-z0-9_.]*
atom ::= punctuation | ".class"
round ::= "(" inner-code ")"
square ::= "[" inner-code "]"
curly ::= "{" inner-code "}"
inner-sep ::= blank | atom | round | square | curly | ";"
inner-value ::= any-id | number | java-string | java-char | text-block
inner-code ::= inner-sep* (inner-value inner-sep+)* inner-value?
'''


_FIELD_EXPRESSION = r'''
gap ::= blank*
space ::= blank+
type-argument ::= "?" (space ("extends" | "super") space type-reference)? | type-reference
type-arguments ::= "<" gap (type-argument (gap "," gap type-argument)*)? gap ">"
type-name ::= identifier (gap type-arguments)? (gap "." gap identifier (gap type-arguments)?)*
type-reference ::= type-name (gap "[" gap "]")*
array-initializer ::= "{" gap (array-value (gap "," gap array-value)* (gap ",")?)? gap "}"
array-value ::= expression | array-initializer
creation ::= "new" space (type-arguments gap)? type-name gap (round (gap curly)? | ("[" inner-code "]" gap)+ array-initializer?)
field-literal ::= number | "." [0-9] [A-Za-z0-9_.]* | java-string | java-char | text-block
type-method-reference ::= type-reference gap "::" gap (type-arguments gap)? (identifier | "new")
primary ::= field-literal | identifier | type-method-reference | "(" gap expression gap ")" | creation | "switch" gap round gap curly
selector ::= "." gap (type-arguments gap)? (identifier | "class" | "new" space type-name gap round (gap curly)?)
postfix ::= selector | round | square | "::" gap (type-arguments gap)? (identifier | "new")
cast ::= "(" gap type-reference (gap "&" gap type-reference)* gap ")" gap (unary | lambda)
unary ::= ("++" | "--" | "+" | "-" | "!" | "~") gap unary | cast | primary (gap postfix)* (gap ("++" | "--"))?
binary-operator ::= "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<=" | ">=" | "<" | ">" | "&&" | "||" | "&" | "|" | "^" | "<<" | ">>" | ">>>"
binary ::= unary (gap binary-operator gap unary | space "instanceof" space type-reference (space identifier)?)*
assignment-operator ::= "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" | ">>>="
lambda ::= (identifier | round) gap "->" gap (curly | expression)
expression ::= lambda | binary (gap "?" gap expression gap ":" gap expression)? (gap assignment-operator gap expression)?
'''


def _fragment_structure(kind, optional, semicolon):
    if kind == "field":
        # A lexical token bag let `package` become `packageai`, and let a
        # qualified identifier be followed by an outer class body. Require
        # expression operators/selectors; braces attach only to an array,
        # creation, lambda or switch. Nested Java remains AST/compiler checked.
        return _FIELD_EXPRESSION + f'\nfragment ::= gap (array-value){optional} gap\n'
    return f'''
first-sep ::= atom | round | square | curly
top-sep ::= blank | first-sep{semicolon}
top-value ::= identifier | number | java-string | java-char | text-block
code ::= top-sep* (top-value top-sep+)* top-value?
rest ::= top-sep+ code | ""
fragment ::= blank* (top-value rest | first-sep code){optional}
'''



@lru_cache(maxsize=6)
def native_java_implementation_grammar(kind, allow_empty=False):
    """Constrain a field expression or body without a full recursive Java PEG."""
    if kind not in {"field", "method", "constructor"}:
        raise ValueError("JAVA_IMPLEMENTATION_GRAMMAR_KIND_INVALID: " + str(kind))
    forbidden = _UNIT_WORDS
    semicolon = ''
    optional = '?'
    if kind == "field":
        forbidden += ("static", "final", "class", "interface", "enum", "void", "return")
        optional = '?' if allow_empty else ''
    else:
        semicolon = ' | ";"'
    envelope = ('root ::= ws "{" ws ' + json.dumps('"implementation"')
                + ' ws ":" ws ' + json.dumps('"') + ' fragment ' + json.dumps('"') + ' ws "}" ws')
    return '\n'.join((envelope, _identifier_grammar(forbidden), _lexical_rules(),
                      _fragment_structure(kind, optional, semicolon)))


def constrain_java_implementation(payload, request):
    """Use the same sampler contract in direct and wrapped native payloads."""
    metadata = getattr(request, "metadata", {})
    frame = metadata.get("mmm_source_declaration_frame") if isinstance(metadata, Mapping) else None
    if not isinstance(frame, Mapping) or metadata.get("mmm_source_implementation") is not True:
        return False
    payload.pop("response_format", None)
    payload.pop("json_schema", None)
    payload["grammar"] = native_java_implementation_grammar(frame["kind"], "empty_value" in frame)
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    return True
