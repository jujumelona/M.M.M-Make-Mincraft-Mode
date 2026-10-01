"""Subdivide an exhausted Java region without moving state or changing its API.

The coder still emits ordinary Java. Pages contain complete declarations (or
initialize statements), never token slices. Accepted pages remain immutable until
the completed region passes the executor's ordinary semantic/ownership gates.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy

from .custom_module_errors import CustomModuleGenerationError
from .implementation_ir import OutputBudgetExhausted

MAX_REGION_PAGES = 64
MAX_ACCEPTED_CONTEXT_CHARS = 8192
MORE = "// MMM_REGION_MORE"
DONE = "// MMM_REGION_DONE"


def generate_region(
    call_coder: Callable[[Sequence[Mapping[str, str]]], str],
    messages: Sequence[Mapping[str, str]],
) -> str:
    """One normal region decode, then smaller semantic units on output pressure."""
    try:
        return call_coder(messages)
    except OutputBudgetExhausted:
        # A region is not a new class responsibility. Splitting its requirement IDs
        # into PartN.run() owners loses call signatures, state, and concern anchors.
        return _generate_pages(call_coder, messages)


def _generate_pages(call_coder, messages) -> str:
    from .atomic_concern_source import (
        _compact_prompt_member_contract,
        _member_declaration_symbols,
        _parse_region_content,
    )
    from .java_region_parser import (
        class_body_member_contracts,
        strict_initialize_statements,
        strict_member_chunks,
    )
    from .root_cause_trace import emit_root_cause

    payload = json.loads(messages[-1]["content"])
    region = payload["response_region"]
    emit_root_cause(
        "atomic_concern_region_subdivided", stage="production", result="START",
        details={"concern": payload["concern"]["name"], "region": region,
                 "owner": payload["host_selected_class"], "same_decode_retry": False},
    )
    parts: list[str] = []
    symbols: set[str] = set()
    for index in range(MAX_REGION_PAGES):
        accepted_source = "\n\n".join(parts)
        accepted = {
            "accepted_sha256": hashlib.sha256(accepted_source.encode("utf-8")).hexdigest(),
        }
        if region == "members":
            # Completed bodies stay host-side. The coder needs exact callable/type
            # contracts, not thousands of already accepted executable statements.
            accepted["accepted_api"] = [
                _compact_prompt_member_contract(row)
                for row in class_body_member_contracts(accepted_source)
            ]
            # A nested type name alone omits record components, constructors and
            # member APIs. Keep these complete, subject to the same context bound.
            accepted["accepted_nested_types"] = [
                part for part in parts
                if any(row.get("kind") == "type" for row in class_body_member_contracts(part))
            ]
        else:
            # Initialize statements can share local variables and depend on order;
            # eliding their bodies would silently change that authority.
            accepted["accepted_source"] = accepted_source
        if len(json.dumps(accepted, ensure_ascii=False)) > MAX_ACCEPTED_CONTEXT_CHARS:
            raise CustomModuleGenerationError(
                "ATOMIC_REGION_CONTEXT_LIMIT: exact accepted declarations/statements "
                "exceed the bounded continuation context"
            )
        page_payload = deepcopy(payload)
        page_payload["region_page"] = {
            "index": index,
            "remaining_pages": MAX_REGION_PAGES - index,
            **accepted,
            "unit": "one complete member" if region == "members" else "one complete statement or block",
            "rules": (
                "The previous complete-region request exceeded the output budget. "
                "Implement only the next small semantic unit of the SAME concern in the SAME class. "
                "Preserve task_authority and all frozen APIs. Accepted source is immutable; "
                "accepted_api lists completed declarations whose bodies are retained by the host. "
                "Reuse their exact signatures and never redeclare or replace them. "
                "Repeated initialize side effects are allowed only when the requirements need them. "
                "Declare shared backing state before methods. Keep each method small; "
                "factor complex logic into private methods within this same class. "
                "No new class owner, facade, run() wrapper, unrelated requirements, or tool protocol. "
                "Return one complete Java declaration (members) or statement/block (initialize). "
                "The FINAL line must be // MMM_REGION_MORE if work remains, or "
                "// MMM_REGION_DONE only after ALL selected concern requirements are implemented. "
                "A marker-only DONE is allowed after accepted pages; never return partial syntax."
            ),
        }
        page_messages = [dict(item) for item in messages]
        page_messages[0]["content"] += (
            "\nOUTPUT SUBDIVISION: region_page narrows this turn to ONE complete semantic "
            "unit. Its final Java comment is mandatory completion control."
        )
        page_messages[-1]["content"] = json.dumps(page_payload, ensure_ascii=False)
        try:
            output = str(call_coder(page_messages) or "").strip()
        except OutputBudgetExhausted as exc:
            # Do not cascade into graph repair or resubmit a completed owner.
            raise CustomModuleGenerationError(
                f"ATOMIC_REGION_UNIT_TOO_LARGE: {payload['host_selected_class']} "
                f"{payload['concern']['name']}:{region} page {index}; "
                "one semantic unit exhausted the output budget"
            ) from exc
        lines = output.splitlines()
        markers = [i for i, line in enumerate(lines) if line.strip() in {MORE, DONE}]
        if markers != [len(lines) - 1]:
            raise CustomModuleGenerationError("ATOMIC_REGION_COMPLETION_REQUIRED")
        done = lines[-1].strip() == DONE
        body = "\n".join(lines[:-1]).strip()
        if not body:
            if done and parts:
                return "\n\n".join(parts)
            raise CustomModuleGenerationError("ATOMIC_REGION_NO_PROGRESS: empty page")
        parsed = _parse_region_content(body, response_region=region)
        chunks = (strict_member_chunks(parsed) if region == "members"
                  else strict_initialize_statements(parsed))
        if len(chunks) != 1:
            raise CustomModuleGenerationError("ATOMIC_REGION_UNIT_CARDINALITY: expected one semantic unit")
        if region == "members" and parsed in parts:
            raise CustomModuleGenerationError("ATOMIC_REGION_NO_PROGRESS: repeated page")
        if region == "members":
            declared = set(_member_declaration_symbols(parsed))
            if declared & symbols:
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_OWNERSHIP_VIOLATION: accepted member redeclared: "
                    + ", ".join(sorted(declared & symbols))
                )
            symbols.update(declared)
        parts.append(parsed)
        if done:
            return "\n\n".join(parts)
    raise CustomModuleGenerationError(
        "ATOMIC_REGION_PAGE_LIMIT: bounded generation ended without explicit completion"
    )
