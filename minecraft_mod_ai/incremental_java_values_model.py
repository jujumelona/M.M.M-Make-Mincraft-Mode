"""Small host-owned records and request context for structured Java values."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

from .incremental_java_schema import obj


@dataclass(frozen=True)
class JavaValue:
    source: str
    java_type: str
    writable: bool = False


@dataclass(frozen=True)
class JavaReceiver:
    source: str
    java_type: str
    static: bool = False
    unbound: bool = False


def request(builder, schema, suffix, extra=None):
    return builder.ask(schema, suffix, {"value_stack": deepcopy(builder.value_stack), **(extra or {})})


@contextmanager
def value_frame(builder, slot, expected, *, writable=False):
    if not hasattr(builder, "value_stack"):
        builder.value_stack = []
    previous = builder.value_stack
    frame = {"slot": slot, "expected_type": expected, "writable": writable}
    previous.append(frame)
    try:
        yield frame
    finally:
        builder.value_stack = previous
        previous.pop()


def count(builder, suffix, *, minimum=0, maximum=None):
    schema = {"type": "integer", "minimum": minimum}
    if maximum is not None:
        schema["maximum"] = maximum
    return request(builder, obj({"count": schema}), suffix)["count"]


def current_owner(builder):
    member = builder.member or {}
    return member.get("name") if member.get("kind") in {"class", "record", "enum"} else (
        member.get("owner") or builder.program["class_name"])


def receiver_source(value):
    import re
    return value.source if re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", value.source) else "(" + value.source + ")"


def inner_type_name(java_type):
    """Keep inner type arguments while skipping qualified argument names."""
    depth, start = 0, 0
    for index, char in enumerate(java_type):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char in {".", "$"} and depth == 0:
            start = index + 1
    return java_type[start:]
