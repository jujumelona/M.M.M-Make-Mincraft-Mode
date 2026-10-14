"""Canonical JSON-wire complements for exact decoded string exclusions.

The host keeps ordinary JSON Schema ``not/enum``. Native patterns describe
canonical wire spellings, because llama.cpp emits patterns inside JSON quotes.
Ordinary Unicode remains literal; quote, backslash and controls use exactly one
escape spelling, so alternative escapes cannot bypass decoded exclusions.
"""
from __future__ import annotations


BMP_SCALAR_PATTERN = r'^[\u0000-\ud7ff\ue000-\uffff]$'


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
    return result
