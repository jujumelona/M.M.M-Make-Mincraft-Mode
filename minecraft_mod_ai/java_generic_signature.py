"""Render inspected JVMS Signature attributes without changing erased identities.

Class/method type bounds are evidence for source authors. They are never a
substitute for JVM descriptors or compiler verification.
"""
from __future__ import annotations

import re

_PRIMITIVES = {'Z': 'boolean', 'B': 'byte', 'S': 'short', 'I': 'int', 'J': 'long',
               'C': 'char', 'F': 'float', 'D': 'double', 'V': 'void'}


class _Signature:
    def __init__(self, text, class_name):
        self.text, self.pos, self.class_name = text, 0, class_name

    def peek(self):
        return self.text[self.pos:self.pos + 1]

    def take(self, expected=None):
        value = self.peek()
        if not value or expected is not None and value != expected:
            raise ValueError('CONSTRUCTION_GENERIC_SIGNATURE_INVALID')
        self.pos += 1
        return value

    def name(self, stops):
        start = self.pos
        while self.peek() and self.peek() not in stops:
            self.pos += 1
        if start == self.pos:
            raise ValueError('CONSTRUCTION_GENERIC_SIGNATURE_INVALID')
        return self.text[start:self.pos]

    def arguments(self):
        if self.peek() != '<':
            return ''
        self.take('<')
        values = []
        while self.peek() != '>':
            token = self.peek()
            if token == '*':
                self.take()
                values.append('?')
            elif token in {'+', '-'}:
                self.take()
                values.append(('? extends ' if token == '+' else '? super ') + self.type())
            else:
                values.append(self.type())
        self.take('>')
        return '<' + ', '.join(values) + '>'

    def type(self):
        token = self.take()
        if token in _PRIMITIVES:
            return _PRIMITIVES[token]
        if token == '[':
            return self.type() + '[]'
        if token == 'T':
            value = self.name(';')
            self.take(';')
            return value
        if token != 'L':
            raise ValueError('CONSTRUCTION_GENERIC_SIGNATURE_INVALID')
        binary = self.name('<.;')
        value = self.class_name(binary).replace('/', '.').replace('$', '.') + self.arguments()
        while self.peek() == '.':
            self.take('.')
            binary += '$' + self.name('<.;')
            nested = self.class_name(binary).replace('/', '.').replace('$', '.').rsplit('.', 1)[-1]
            value += '.' + nested + self.arguments()
        self.take(';')
        return value

    def formals(self):
        if self.peek() != '<':
            return []
        self.take('<')
        result = []
        while self.peek() != '>':
            name = self.name(':')
            self.take(':')
            bounds = []
            # ClassBound is ':' followed by an *optional* reference type.
            # The next formal may start with L/T too, so identify its colon
            # before attempting to consume a class or type-variable signature.
            next_formal = re.match(r'[^.;\[/<>:]+:', self.text[self.pos:])
            if self.peek() in {'L', 'T', '['} and not next_formal:
                bounds.append(self.type())
            while self.peek() == ':':
                self.take(':')
                bounds.append(self.type())
            result.append(name + (' extends ' + ' & '.join(bounds) if bounds else ''))
        self.take('>')
        return result


def generic_contract(signature: str, kind: str, *, class_name=lambda value: value) -> dict:
    """Return complete Java types for class, field or callable signatures."""
    if not signature:
        return {}
    parser = _Signature(signature, class_name)
    formals = parser.formals()
    result = {'type_parameters': formals} if formals else {}
    if kind == 'CLASS':
        result['generic_superclass'] = parser.type()
        interfaces = []
        while parser.peek():
            interfaces.append(parser.type())
        result['generic_interfaces'] = interfaces
    elif kind in {'METHOD', 'CONSTRUCTOR'}:
        parser.take('(')
        parameters = []
        while parser.peek() != ')':
            parameters.append(parser.type())
        parser.take(')')
        result.update(generic_parameters=parameters, generic_return_type=parser.type())
        exceptions = []
        while parser.peek() == '^':
            parser.take('^')
            exceptions.append(parser.type())
        if exceptions:
            result['generic_throws'] = exceptions
    else:
        result['generic_return_type'] = parser.type()
    if parser.peek():
        raise ValueError('CONSTRUCTION_GENERIC_SIGNATURE_TRAILING_DATA')
    return result
