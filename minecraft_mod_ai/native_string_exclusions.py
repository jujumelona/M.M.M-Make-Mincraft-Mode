"""Native-compatible string exclusions for host-owned authored choices.

For arbitrary text, the host keeps ordinary JSON Schema ``not/enum``. Native patterns describe
canonical wire spellings, because llama.cpp emits patterns inside JSON quotes.
Ordinary Unicode remains literal; quote, backslash and controls use exactly one
escape spelling, so alternative escapes cannot bypass decoded exclusions.

Finite identifier alphabets use bounded trie-complement patterns directly.
"""
from __future__ import annotations

from collections.abc import Mapping

BMP_SCALAR_PATTERN = r'^[\u0000-\ud7ff\ue000-\uffff]$'
DECODED_SCHEMA_KEY = 'mmm_decoded_string_schema'


def decoded_validation_schema(schema):
    """Keep JSON-wire patterns away from validators of decoded scalar values."""
    if isinstance(schema, Mapping):
        logical = schema.get(DECODED_SCHEMA_KEY)
        if schema.get('type') == 'string' and isinstance(logical, Mapping):
            if not supports_string_exclusions(logical):
                raise ValueError('NATIVE_STRING_LOGICAL_SCHEMA_INVALID')
            return decoded_validation_schema(logical)
        return {key: decoded_validation_schema(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return [decoded_validation_schema(value) for value in schema]
    return schema


def supports_string_exclusions(schema):
    excluded = schema.get('not')
    return (schema.get('type') == 'string' and isinstance(excluded, dict)
            and set(excluded) == {'enum'} and isinstance(excluded['enum'], list)
            and all(isinstance(value, str) for value in excluded['enum'])
            and schema.get('pattern') in {None, BMP_SCALAR_PATTERN})


def _point(value):
    if value <= 0xff:
        return '\\x' + format(value, '02x')
    if value <= 0xffff:
        return '\\u' + format(value, '04x')
    return '\\U' + format(value, '08x')


def _wire_character(char):
    value = ord(char)
    if char == '"':
        return r'[\x5c][\x22]'
    if char == '\\':
        return r'[\x5c][\x5c]'
    if value < 0x20:
        return r'[\x5c]u' + format(value, '04x')
    return '[' + _point(value) + ']'


def _character(excluded, maximum):
    blocked = sorted(ord(char) for char in excluded)
    intervals = []
    for lower, upper in ((0x20, 0x21), (0x23, 0x5b), (0x5d, 0xd7ff), (0xe000, maximum)):
        start = lower
        for value in blocked:
            if lower <= value <= upper:
                if start < value:
                    intervals.append((start, value - 1))
                start = value + 1
        if start <= upper:
            intervals.append((start, upper))
    options = []
    if intervals:
        options.append('[' + ''.join(_point(left) + ('-' + _point(right) if left != right else '')
                                    for left, right in intervals) + ']')
    for char in ('"', '\\'):
        if char not in excluded:
            options.append(_wire_character(char))
    controls = []
    for high in (0, 1):
        digits = ''.join(format(value, 'x') for value in range(16) if chr(high * 16 + value) not in excluded)
        if digits:
            controls.append(str(high) + '[' + digits + ']')
    if controls:
        options.append(r'[\x5c]u00(' + '|'.join(controls) + ')')
    return '(' + '|'.join(options) + ')' if options else None


def wire_string_schema(schema):
    if not supports_string_exclusions(schema):
        raise ValueError('NATIVE_STRING_EXCLUSION_SHAPE_UNSUPPORTED')
    maximum_length = schema['maxLength']
    minimum_length = schema.get('minLength', 0)
    maximum_point = 0xffff if schema.get('pattern') == BMP_SCALAR_PATTERN else 0x10ffff
    trie = {}
    for word in schema['not']['enum']:
        if not minimum_length <= len(word) <= maximum_length:
            continue
        if any(ord(char) > maximum_point or 0xd800 <= ord(char) <= 0xdfff for char in word):
            continue
        current = trie
        for char in word:
            current = current.setdefault(char, {})
        current[None] = True
    any_character = _character(set(), maximum_point)
    alternatives = []
    pending = [('', 0, trie)]
    while pending:
        prefix, length, branch = pending.pop()
        if length >= minimum_length and None not in branch:
            alternatives.append(prefix)
        if length == maximum_length:
            continue
        children = {char for char in branch if char is not None}
        different = _character(children, maximum_point)
        if different is not None:
            lower = max(0, minimum_length - length - 1)
            upper = maximum_length - length - 1
            alternatives.append(prefix + different + any_character + '{' + str(lower) + ',' + str(upper) + '}')
        for char in sorted(children, reverse=True):
            pending.append((prefix + _wire_character(char), length + 1, branch[char]))
    if not alternatives:
        raise ValueError('NATIVE_STRING_EXCLUSION_DOMAIN_EMPTY')
    result = {key: value for key, value in schema.items() if key not in {'not', 'pattern'}}
    result.update(pattern='^(' + '|'.join(alternatives) + ')$', maxLength=6 * maximum_length)
    result[DECODED_SCHEMA_KEY] = dict(schema)
    return result


# This limits regex factoring, not identifiers, queries or the excluded set.
# llama.cpp's pinned compiler accepts at most 100 nested pattern groups.
_MAX_TRIE_GROUP_DEPTH = 96


def _character_class(chars):
    def escape(char):
        # Python re.escape also escapes '~', '$', '.', etc. llama.cpp rejects
        # those class escapes and silently substitutes an unbounded string.
        return '\\x' + format(ord(char), '02x') if char in '\\[]^-' else char
    runs = []
    values = sorted(ord(char) for char in chars)
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        runs.append((start, end))
        start = end = value
    runs.append((start, end))
    parts = []
    for start, end in runs:
        if end - start >= 2:
            parts.append(escape(chr(start)) + '-' + escape(chr(end)))
        else:
            parts.extend(escape(chr(value)) for value in range(start, end + 1))
    return '[' + ''.join(parts) + ']'


def string_excluding(forbidden, *, alphabet, first=None, minimum=1, maximum=240,
                     fold_case=False, normalizer=None):
    """Finite trie complement; no unsupported lookahead/not or unbounded regex.

    Each branch either ends at an allowed prefix, takes a forbidden-word edge,
    or diverges from that trie and fills only its remaining length budget.
    """
    first = set(first or alphabet)
    alphabet = set(alphabet)
    normalize = normalizer or (str.casefold if fold_case else lambda value: value)
    trie = {}
    for value in forbidden:
        value = normalize(value)
        if not value:
            if minimum == 0:
                trie[''] = True
            continue
        if len(value) > maximum:
            continue
        if any(not any(normalize(char) == key for char in (first if i == 0 else alphabet))
               for i, key in enumerate(value)):
            continue
        node = trie
        for key in value:
            node = node.setdefault(key, {})
        node[''] = True
    tail = _character_class(alphabet)

    def choices(node, depth):
        options = [''] if depth >= minimum and '' not in node else []
        edges = []
        if depth < maximum:
            allowed = first if depth == 0 else alphabet
            other = {char for char in allowed if normalize(char) not in node}
            remaining = maximum - depth - 1
            if other and remaining >= max(0, minimum - depth - 1):
                option = _character_class(other)
                if remaining:
                    option += tail + '{' + str(max(0, minimum - depth - 1)) + ',' + str(remaining) + '}'
                options.append(option)
            for key in sorted(key for key in node if key):
                chars = {char for char in allowed if normalize(char) == key}
                if not chars:
                    continue
                edges.append((_character_class(chars), node[key]))
        return options, edges

    def combine(options):
        if not options:
            return None
        return options[0] if len(options) == 1 else '(?:' + '|'.join(options) + ')'

    def flat_suffix(node, depth):
        options = []
        pending = [(node, depth, '')]
        while pending:
            current, position, prefix = pending.pop()
            endings, edges = choices(current, position)
            options.extend(prefix + ending for ending in endings)
            pending.extend((child, position + 1, prefix + char)
                           for char, child in reversed(edges))
        return combine(options)

    def suffix(node, depth):
        if depth >= _MAX_TRIE_GROUP_DEPTH:
            # Flatten only the deep tail. Every excluded word and length bound
            # remains active without crossing the native nesting limit.
            return flat_suffix(node, depth)
        options, edges = choices(node, depth)
        for char, child in edges:
            rest = suffix(child, depth + 1)
            if rest is not None:
                options.append(char + rest)
        return combine(options)

    return {'type': 'string', 'minLength': minimum, 'maxLength': maximum,
            'pattern': '^' + suffix(trie, 0) + '$'}
