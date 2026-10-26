"""Locate compiler-rejected Java members and their declared neighbor APIs."""
from __future__ import annotations

import re

from .java_region_parser import _parse


def _failing_member(source: str, diagnostic: str, *, filename: str):
    patterns = (
        rf"(?:^|\n)[^\n]*{re.escape(filename)}:(\d+):",
        r"\bline\s+(\d+)\b",
    )
    lines = []
    for pattern in patterns:
        lines = [int(n) for n in re.findall(pattern, diagnostic)]
        if lines:
            # javac reports multiple independent errors in one build. The
            # repair must target the FIRST diagnostic, not whichever method
            # happens to have the fewest bytes among all error locations.
            # Otherwise a later private-field error steals an earlier broken
            # constructor and prevents the superclass repair from running.
            lines = lines[:1]
            break
    if not lines:
        raise ValueError("JAVA_HOST_REPAIR_NO_SOURCE_LINE")
    raw, root = _parse(source)
    candidates = []

    def visit(node):
        if node.type in {"method_declaration", "constructor_declaration"}:
            body = node.child_by_field_name("body")
            if (body is not None and any(
                node.start_point.row + 1 <= line <= node.end_point.row + 1
                for line in lines
            )):
                candidates.append((node, body))
        for child in node.named_children:
            visit(child)

    visit(root)
    if not candidates:
        raise ValueError("JAVA_HOST_REPAIR_NOT_A_MEMBER_DIAGNOSTIC")
    node, body = min(candidates, key=lambda pair: (
        pair[0].end_byte - pair[0].start_byte))
    kind = ("constructor" if node.type == "constructor_declaration" else "method")
    prefix = raw[node.start_byte:body.start_byte + 1].decode("utf-8") + "\n"
    original = raw[node.start_byte:node.end_byte].decode("utf-8")
    frame = {"kind": kind, "prefix": prefix, "suffix": "\n}"}
    return raw, node, original, frame

def _first_compiler_error(diagnostic: str, *, filename: str) -> str:
    """Return one complete javac error including its required/found details.

    A single javac run can describe failures in several different members.
    Showing all of them to a fixed-frame repair causes the coder to edit the
    wrong member (or repeat a forbidden assignment). Preserve the original
    full diagnostic for the host's verified constructor repair; scope only
    the model's task to the first actual error.
    """
    expression = re.compile(
        rf"(?m)^[^\n]*{re.escape(filename)}:\d+:\s*error:"
    )
    matches = list(expression.finditer(diagnostic))
    if not matches:
        return diagnostic
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(diagnostic)
    return diagnostic[start:end].strip()

def _host_declared_neighbor_signatures(source: str, *, selected_name: str) -> list[str]:
    """Expose the actual generated owner ABI to a *single-member* compiler fix.

    Do not ask the 9B model to invent argument producers or neighboring method
    arities. These are parsed declarations from the candidate and are only
    candidates, NOT evidence that javac accepted the member bodies.
    """
    raw, root = _parse(source)
    declarations = []

    def visit(node):
        if node.type in {"method_declaration", "constructor_declaration"}:
            body = node.child_by_field_name("body")
            if body is not None:
                header = raw[node.start_byte:body.start_byte].decode("utf-8").strip()
                if header != selected_name:
                    declarations.append(header)
        elif node.type == "field_declaration":
            declarations.append(
                raw[node.start_byte:node.end_byte].decode("utf-8").strip()
            )
        if node.type not in {"method_declaration", "constructor_declaration"}:
            for child in node.named_children:
                visit(child)

    visit(root)
    return declarations
