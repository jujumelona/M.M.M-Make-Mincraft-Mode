"""Repair one Java member selected by an actual javac source diagnostic.

Tree-sitter supplies the source byte range. The model receives a fixed member
signature and compiler evidence; other checked members remain byte-identical.
"""
from __future__ import annotations

import hashlib
import json
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



def _comma_separated_java_types(text: str) -> list[str]:
    """Split javac parameter lists without splitting generic type arguments."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth < 0:
                return []
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth:
        return []
    return [*parts, text[start:].strip()] if text.strip() else []


def _javac_constructor_repair(source: str, diagnostic: str, *, filename: str) -> str | None:
    """Apply a javac-proven parent constructor call; never guess an overload.

    The verified source compiler runs again before the candidate is accepted.
    """
    match = re.search(
        r"constructor\s+([\w$]+)\s+in class\s+[\w$]+\s+cannot be applied"
        r"[\s\S]*?required:\s*([^\r\n]+)\s*"
        r"found:\s*no arguments",
        diagnostic,
    )
    if match is None:
        return None
    superclass = re.search(r"\bextends\s+([\w.$]+)", source)
    if superclass is None or superclass.group(1).split(".")[-1] != match.group(1):
        return None
    try:
        raw, node, _original, _frame = _failing_member(
            source, diagnostic, filename=filename)
    except ValueError:
        return None
    if node.type != "constructor_declaration":
        return None
    parameters_node = node.child_by_field_name("parameters")
    body = node.child_by_field_name("body")
    if parameters_node is None or body is None:
        return None
    declared = []
    for parameter in parameters_node.named_children:
        if parameter.type != "formal_parameter":
            return None
        declared_type = parameter.child_by_field_name("type")
        declared_name = parameter.child_by_field_name("name")
        if declared_type is None or declared_name is None:
            return None
        declared.append((
            raw[declared_type.start_byte:declared_type.end_byte].decode("utf-8"),
            raw[declared_name.start_byte:declared_name.end_byte].decode("utf-8"),
        ))
    required = _comma_separated_java_types(match.group(2))
    if len(required) != len(declared) or not required:
        return None
    def base_type(type_name: str) -> str:
        return re.sub(r"<.*", "", type_name).strip().split(".")[-1]
    if any(base_type(want) != base_type(have) for want, (have, _name) in zip(required, declared)):
        return None

    old_body = raw[body.start_byte:body.end_byte].decode("utf-8")
    if re.search(r"^\s*(?:super|this)\s*\(", old_body[1:-1]):
        return None
    remaining = old_body[1:-1]
    # Javac has proved these inherited fields inaccessible. Remove only
    # trivial assignments from constructor arguments or from that same field.
    invalid_fields = set(re.findall(
        r"\b([A-Za-z_$][\w$]*)\s+has private access in\s+[\w.$]+", diagnostic))
    parameter_names = {name for _type, name in declared}
    for field_name in invalid_fields:
        allowed_rhs = parameter_names | {"this." + field_name}
        assignment = re.compile(
            r"(?m)^[ \t]*this\." + re.escape(field_name)
            + r"\s*=\s*(?:" + "|".join(re.escape(name) for name in sorted(allowed_rhs))
            + r")\s*;[ \t]*(?:\r?\n)?"
        )
        remaining = assignment.sub("", remaining)
    new_body = ("{\n    super(" + ", ".join(name for _type, name in declared)
                + ");" + remaining + "}")
    if new_body == old_body:
        return None
    return (raw[:body.start_byte] + new_body.encode("utf-8")
            + raw[body.end_byte:]).decode("utf-8")


def repair_compiler_member(router, source: str, diagnostic: str, *,
                           filename: str, target_path: str, symbols: dict,
                           owner: str, api_evidence: dict, attempt: int) -> str:
    from .host_java_imports import host_add_verified_imports

    checked_constructor = _javac_constructor_repair(
        source, diagnostic, filename=filename)
    if checked_constructor is not None:
        return host_add_verified_imports(
            checked_constructor, symbols=symbols, owner=owner, target_path=target_path)

    raw, node, original, frame = _failing_member(
        source, diagnostic, filename=filename)
    messages = [{
        "role": "system",
        "content": (
            "Repair only the requested Java method or constructor body. Return "
            "JSON with one implementation string. The host preserves "
            "its exact signature and every assigned behavior. The host owns "
            "imports, class identity and other methods. Use only real APIs "
            "from the inspected target; no package statements, imports, "
            "placeholder bodies or new declarations."
        ),
    }, {
        "role": "user",
        "content": json.dumps({
            "target_path": target_path, "javac_diagnostic": diagnostic,
            "current_member": original, "response_source_frame": frame,
            "inspected_api": api_evidence,
        }, ensure_ascii=False, default=str),
    }]
    fingerprint = hashlib.sha256(
        (original + diagnostic).encode("utf-8")).hexdigest()[:16]
    revised = router.generate_java_source(
        "coder", messages,
        data_name=f"repair_compiler_member_{attempt}_{fingerprint}",
        source_frame=frame,
    )
    if revised == original:
        raise ValueError("JAVA_HOST_REPAIR_NO_MEMBER_PROGRESS")
    updated = (raw[:node.start_byte] + revised.encode("utf-8") +
               raw[node.end_byte:]).decode("utf-8")
    # The host reconstructs imports after a member edit; never delegate that
    # responsibility to the decoder or keep obsolete imports indefinitely.
    updated = re.sub(
        r"(?m)^import\s+(?:static\s+)?[\w.$*]+\s*;\s*\n",
        "", updated,
    )
    return host_add_verified_imports(
        updated, symbols=symbols, owner=owner, target_path=target_path)
