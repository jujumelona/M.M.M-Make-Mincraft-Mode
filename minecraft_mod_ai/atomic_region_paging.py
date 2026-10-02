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


def decide_region_completion(router, payload: Mapping) -> dict:
    """Read completion separately through the coder's native scalar tool contract.

    Java source remains ordinary text. Neither a missing comment nor Markdown
    layout may decide whether the implementation is complete.
    """
    from .model_adapters.base import ModelConfigurationError, NativeToolDecisionRejected
    from .model_output_atomicity_contract import assert_atomic_model_schema

    schema = {
        "type": "object",
        "properties": {
            "done": {"type": "boolean"},
            "next_work": {"type": "string", "maxLength": 256},
        },
        "required": ["done", "next_work"],
        "additionalProperties": False,
    }
    assert_atomic_model_schema(schema, surface="atomic region completion")
    context = deepcopy(dict(payload))
    context.pop("generation_recipe", None)
    feedback = ""
    for _ in range(2):
        request = {**context, "completion_feedback": feedback}
        try:
            decision = router.generate_tool_decision(
                "coder",
                [
                    {"role": "system", "content": (
                        "Assess completion of the selected production Java concern only. "
                        "Do not generate Java, change the design or inspect sibling requirements. "
                        "task_authority and region_correction (when present) define the work. "
                        "accepted_api/accepted_nested_types describe earlier completed pages; "
                        "current_page_source is the newly accepted page, and previous_next_work "
                        "is the work selected for that page. Set done=true only when all selected "
                        "requirements are implemented, with next_work empty. Otherwise done=false "
                        "and next_work describes the next small complete declaration or statement "
                        "needed in this same class. Do not declare completion merely because a "
                        "page parses or the model response ended."
                    )},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                tool_name="report_java_region_completion",
                parameters=schema,
                description="Report completion or the next semantic unit of this Java concern.",
                output_token_ceiling=1024,
                force_non_thinking=True,
            )
        except NativeToolDecisionRejected as exc:
            feedback = str(exc)
            continue
        except ModelConfigurationError as exc:
            if not str(exc).startswith("Native structured decision did not return"):
                raise
            feedback = str(exc)
            continue
        if (
            isinstance(decision, Mapping)
            and type(decision.get("done")) is bool
            and isinstance(decision.get("next_work"), str)
            and len(decision["next_work"]) <= 256
            and bool(decision["next_work"].strip()) is not decision["done"]
        ):
            return {"done": decision["done"], "next_work": decision["next_work"].strip()}
        feedback = "done must be a boolean; done=true requires empty next_work, done=false requires concrete next_work."
    raise CustomModuleGenerationError("ATOMIC_REGION_COMPLETION_DECISION_INVALID: " + feedback)


def generate_region(
    call_coder: Callable[[Sequence[Mapping[str, str]]], str],
    messages: Sequence[Mapping[str, str]],
    *,
    completion_decider: Callable[[Mapping], Mapping] | None = None,
) -> str:
    """One normal region decode, then smaller semantic units on output pressure."""
    try:
        return call_coder(messages)
    except OutputBudgetExhausted:
        # A region is not a new class responsibility. Splitting its requirement IDs
        # into PartN.run() owners loses call signatures, state, and concern anchors.
        pass
    # Leave the handled exception context before continuation: a later page
    # failure must not be reported as an unhandled transport output-limit error.
    return _generate_pages(call_coder, messages, completion_decider=completion_decider)


def _generate_pages(call_coder, messages, *, completion_decider=None) -> str:
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
    next_work = ""
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
            "next_work": next_work,
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
                "Return the next complete Java declaration (members) or statement/block (initialize). "
                "Follow next_work when supplied. Never return partial syntax. "
                + (
                    "Return Java source only. Completion is collected in a separate native tool turn; "
                    "do not add completion markers, a JSON envelope or a status explanation to Java."
                    if completion_decider is not None else
                    "Finish with // MMM_REGION_MORE or // MMM_REGION_DONE."
                )
            ),
        }
        page_messages = [dict(item) for item in messages]
        page_messages[0]["content"] += (
            "\nOUTPUT SUBDIVISION: region_page narrows this turn to ONE complete semantic "
            "unit. Keep complete declarations and the existing host owner."
        )
        page_messages[-1]["content"] = json.dumps(page_payload, ensure_ascii=False)
        output = ""
        try:
            try:
                output = str(call_coder(page_messages) or "").strip()
            except OutputBudgetExhausted as exc:
                raise CustomModuleGenerationError(
                    f"ATOMIC_REGION_UNIT_TOO_LARGE: {payload['host_selected_class']} "
                    f"{payload['concern']['name']}:{region} page {index}; "
                    "one semantic unit exhausted the output budget"
                ) from exc
            lines = output.splitlines()
            done = False
            if completion_decider is None:
                # Compatibility for explicitly marker-based callers. Production
                # always supplies the separate completion callback below.
                markers = [i for i, line in enumerate(lines) if line.strip() in {MORE, DONE}]
                if markers != [len(lines) - 1]:
                    raise CustomModuleGenerationError("ATOMIC_REGION_COMPLETION_REQUIRED")
                done = lines[-1].strip() == DONE
                body = "\n".join(lines[:-1]).strip()
            else:
                body = output
            if not body:
                if done and parts:
                    return "\n\n".join(parts)
                raise CustomModuleGenerationError("ATOMIC_REGION_NO_PROGRESS: empty page")
            try:
                parsed = _parse_region_content(body, response_region=region)
            except CustomModuleGenerationError as exc:
                # A paged turn promises one complete Java unit. Any parser rejection
                # is therefore a page-scope violation rather than a generic response
                # formatting error; keep the public taxonomy stable for recovery.
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_SCOPE_ESCAPE: page is not one admissible Java unit: "
                    + str(exc)
                ) from exc
            chunks = (strict_member_chunks(parsed) if region == "members"
                      else strict_initialize_statements(parsed))
            if not chunks:
                raise CustomModuleGenerationError("ATOMIC_REGION_NO_PROGRESS: page contains no executable unit")
            if len(chunks) != 1:
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_UNIT_CARDINALITY: region_page must contain exactly one "
                    f"complete semantic unit, got {len(chunks)}"
                )
            page_symbols: set[str] = set()
            if region == "members":
                for chunk in chunks:
                    declared = set(_member_declaration_symbols(chunk))
                    if declared & (symbols | page_symbols):
                        raise CustomModuleGenerationError(
                            "ATOMIC_REGION_OWNERSHIP_VIOLATION: accepted member redeclared: "
                            + ", ".join(sorted(declared & (symbols | page_symbols)))
                        )
                    page_symbols.update(declared)
            if completion_decider is not None:
                decision = completion_decider({
                    **payload,
                    **accepted,
                    "current_page_source": parsed,
                    "previous_next_work": next_work,
                    "page_index": index,
                })
                done = decision["done"]
                next_work = decision["next_work"]
            parts.extend(chunks)
            symbols.update(page_symbols)
            emit_root_cause(
                "atomic_concern_page_accepted", stage="production", result="PASS",
                details={"owner": payload["host_selected_class"], "region": region,
                         "concern": payload["concern"]["name"], "page": index,
                         "unit_count": len(chunks), "done": done, "next_work": next_work,
                         "source_sha256": hashlib.sha256(parsed.encode("utf-8")).hexdigest()},
            )
            if done:
                return "\n\n".join(parts)
        except Exception as exc:
            emit_root_cause(
                "atomic_concern_page_rejected", stage="production", result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={"owner": payload["host_selected_class"], "region": region,
                         "concern": payload["concern"]["name"], "page": index,
                         "accepted_units": len(parts), "rejected_response": output,
                         "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()},
            )
            raise
    raise CustomModuleGenerationError(
        "ATOMIC_REGION_PAGE_LIMIT: bounded generation ended without explicit completion"
    )
