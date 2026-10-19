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


def repair_compiler_member(router, source: str, diagnostic: str, *,
                           filename: str, target_path: str, symbols: dict,
                           owner: str, api_evidence: dict, attempt: int) -> str:
    from .host_java_imports import host_add_verified_imports

    raw, node, original, frame = _failing_member(
        source, diagnostic, filename=filename)
    messages = [{
        "role": "system",
        "content": (
            "Repair only the requested Java method or constructor. Preserve "
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
