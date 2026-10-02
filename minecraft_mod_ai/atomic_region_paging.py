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
from .execution_contract_policy import (
    ATOMIC_REGION_COMPLETION_PARAMETERS,
)
from .execution_contract_policy import (
    ATOMIC_REGION_MAX_ACCEPTED_CONTEXT_CHARS as MAX_ACCEPTED_CONTEXT_CHARS,
)
from .execution_contract_policy import (
    ATOMIC_REGION_MAX_OWNERSHIP_CORRECTIONS as MAX_OWNERSHIP_CORRECTIONS,
)
from .execution_contract_policy import (
    ATOMIC_REGION_MAX_PAGES as MAX_REGION_PAGES,
)
from .execution_contract_policy import (
    ATOMIC_REGION_MAX_UNIT_REFINEMENTS as MAX_UNIT_REFINEMENTS,
)
from .execution_contract_policy import (
    ATOMIC_REGION_MAX_UNITS as MAX_REGION_UNITS,
)
from .implementation_ir import OutputBudgetExhausted

MORE = "// MMM_REGION_MORE"
DONE = "// MMM_REGION_DONE"


def _member_page_delta(chunks, accepted_parts):
    """Project only provably identical echoes out of an append-only page.

    Symbol equality alone does not establish equality of implementations. Mixed
    declarators and changed bodies stay conflicts, with their exact source, so
    no accepted behavior can be replaced or silently discarded.
    """
    from .atomic_concern_source import _member_declaration_symbols
    from .java_region_parser import class_body_token_identity

    owners = {}
    for chunk in accepted_parts:
        for symbol in _member_declaration_symbols(chunk):
            owners[symbol] = chunk
    accepted_keys = set(owners)
    additions, conflicts = [], []
    echoed = set()
    for chunk in chunks:
        declared = set(_member_declaration_symbols(chunk))
        collisions = declared.intersection(owners)
        if collisions:
            originals = tuple(dict.fromkeys(owners[key] for key in sorted(collisions)))
            if (
                len(originals) == 1
                and declared == set(_member_declaration_symbols(originals[0]))
                and class_body_token_identity(chunk) == class_body_token_identity(originals[0])
            ):
                echoed.update(collisions)
                continue
            conflicts.append({
                "member_keys": sorted(collisions),
                "previously_accepted": collisions.issubset(accepted_keys),
                "immutable_source": "\n\n".join(originals),
                "rejected_source": chunk,
            })
            continue
        additions.append(chunk)
        for symbol in declared:
            owners[symbol] = chunk
    return tuple(additions), tuple(sorted(echoed)), conflicts


def _validated_decision(decision: Mapping, *, selecting: bool = False) -> dict:
    """Normalize the semantic next-work signal instead of validating protocol cosmetics."""
    if not isinstance(decision, Mapping):
        raise CustomModuleGenerationError(
            "ATOMIC_REGION_COMPLETION_DECISION_INVALID: completion decision must be an object"
        )
    raw_next = decision.get("next_work", "")
    if not isinstance(raw_next, str):
        raise CustomModuleGenerationError(
            "ATOMIC_REGION_COMPLETION_DECISION_INVALID: next_work must be text"
        )
    next_work = raw_next.strip()
    if selecting and not next_work:
        raise CustomModuleGenerationError(
            "ATOMIC_REGION_COMPLETION_DECISION_INVALID: unfinished selection needs semantic next_work"
        )
    # Production uses next_work as the single signal. Legacy callbacks may still
    # include done; contradictory booleans are ignored rather than promoted to a
    # fatal formatting gate.
    return {"done": not bool(next_work), "next_work": next_work}


def decide_region_completion(router, payload: Mapping) -> dict:
    """Read completion separately through the coder's native scalar tool contract.

    Java source remains ordinary text. Neither a missing comment nor Markdown
    layout may decide whether the implementation is complete.
    """
    from .model_adapters.base import ModelConfigurationError, NativeToolDecisionRejected
    from .model_output_atomicity_contract import assert_atomic_model_schema

    schema = deepcopy(ATOMIC_REGION_COMPLETION_PARAMETERS)
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
                        "Return only next_work: a concrete next semantic unit, or an empty string "
                        "when the selected concern is complete. Do not generate Java, change the design "
                        "or inspect sibling requirements. "
                        "task_authority and region_correction (when present) define the work. "
                        "accepted_api/accepted_nested_types describe earlier completed pages; "
                        "current_page_source is the newly accepted page, and previous_next_work "
                        "is the work selected for that page. Set done=true only when all selected "
                        "requirements are implemented, with next_work empty. Otherwise done=false "
                        "and next_work describes the next small complete declaration or statement "
                        "needed in this same class. Do not declare completion merely because a "
                        "page parses or the model response ended. next_work is the only completion signal."
                        " In select_next_unit phase no source has been accepted: choose the "
                        "first small declaration/statement, with done=false. In refine_next_unit "
                        "phase the selected work exhausted its output budget: choose a strictly "
                        "smaller prerequisite/helper in the SAME owner, with done=false. "
                        "Never repeat the exhausted work or move state to another owner. "
                        "next_work must name one concrete declaration/statement and its purpose, "
                        "using exact existing APIs. Source generation implements only that unit; "
                        "the host retains the complete task authority for later units."
                        " accepted_member_keys are already implemented and immutable. Never "
                        "select their implementation again. An echoed declaration adds no work; "
                        "evaluate the actual accepted implementation, not the rejected proposal."
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
        try:
            return _validated_decision(
                decision,
                selecting=context.get("completion_phase") in {"select_next_unit", "refine_next_unit"},
            )
        except CustomModuleGenerationError as exc:
            feedback = str(exc)
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
    page_correction = None
    pending_correction = None
    ownership_corrections = 0
    rejected_fingerprints: set[str] = set()
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
            accepted["accepted_member_keys"] = sorted(symbols)
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
        if len(parts) >= MAX_REGION_UNITS:
            raise CustomModuleGenerationError(
                "ATOMIC_REGION_UNIT_LIMIT: bounded generation ended without explicit completion"
            )
        if completion_decider is not None and not next_work:
            decision = _validated_decision(completion_decider({
                **payload, **accepted,
                "completion_phase": "select_next_unit",
                "current_page_source": "",
                "previous_next_work": "",
                "page_index": index,
            }), selecting=True)
            next_work = decision["next_work"]
        page_payload = deepcopy(payload)
        # Whole-region recipes otherwise keep telling a small coder to reproduce
        # the complete concern even after next_work selects an incremental unit.
        page_payload["phase"] = "append_atomic_concern_units"
        page_payload.setdefault("scope", {})["generation_mode"] = "append_only"
        recipe = page_payload.setdefault("generation_recipe", {})
        recipe["first_pass_goal"] = "Implement only region_page.next_work as new complete Java units."
        recipe["declare_plan_local_domain_type_rule"] = (
            "Reuse accepted, sibling and dependency types. Add an authorized local type "
            "only if absent from those inventories; never regenerate accepted members."
        )
        recipe["mechanical_type_authority_repair_rule"] = (
            "Apply supplied mechanical edits only to new units in this page. "
            "Accepted member declarations and bodies are immutable."
        )
        if page_correction is not None:
            page_payload["page_correction"] = page_correction
        page_payload["region_page"] = {
            "index": index,
            "remaining_pages": MAX_REGION_PAGES - index,
            "remaining_units": MAX_REGION_UNITS - len(parts),
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
                "The host parses complete declarations/statements in source order; if a small "
                "cohesive response contains several, all count toward remaining_units. "
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
            "unit. Keep complete declarations and the existing host owner. This is APPEND ONLY: "
            "the response region consists only of new declarations, never the whole concern. "
            "Call existing accepted methods; do not emit their definitions. If page_correction "
            "is present, return its pending_declarations corrected to use immutable_source, "
            "plus only the new units needed for next_work. Never return conflicting definitions."
        )
        page_messages[-1]["content"] = json.dumps(page_payload, ensure_ascii=False)
        output = ""
        try:
            exhausted_work: set[str] = set()
            for refinement in range(MAX_UNIT_REFINEMENTS + 1):
                try:
                    output = str(call_coder(page_messages) or "").strip()
                    break
                except OutputBudgetExhausted as exc:
                    if completion_decider is None or refinement >= MAX_UNIT_REFINEMENTS:
                        raise CustomModuleGenerationError(
                            f"ATOMIC_REGION_UNIT_TOO_LARGE: {payload['host_selected_class']} "
                            f"{payload['concern']['name']}:{region} page {index}; "
                            "bounded same-owner unit refinement exhausted"
                        ) from exc
                    exhausted_work.add(next_work)
                    decision = _validated_decision(completion_decider({
                        **payload, **accepted,
                        "completion_phase": "refine_next_unit",
                        "current_page_source": "",
                        "previous_next_work": next_work,
                        "exhausted_work": sorted(exhausted_work),
                        "page_index": index,
                    }), selecting=True)
                    if decision["next_work"] in exhausted_work:
                        raise CustomModuleGenerationError(
                            "ATOMIC_REGION_NO_PROGRESS: refinement repeated exhausted work"
                        ) from exc
                    next_work = decision["next_work"]
                    page_payload["region_page"]["next_work"] = next_work
                    page_payload["region_page"]["exhausted_work"] = sorted(exhausted_work)
                    page_messages[-1]["content"] = json.dumps(page_payload, ensure_ascii=False)
                    emit_root_cause(
                        "atomic_concern_unit_refined", stage="production", result="RETRY",
                        details={"owner": payload["host_selected_class"], "region": region,
                                 "concern": payload["concern"]["name"], "page": index,
                                 "refinement": refinement + 1, "next_work": next_work,
                                 "accepted_units": len(parts)},
                    )
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
            echoed = ()
            if region == "members":
                additions, echoed, conflicts = _member_page_delta(chunks, parts)
                if conflicts:
                    reason = "ATOMIC_REGION_OWNERSHIP_VIOLATION: accepted member redeclared: " + ", ".join(
                        sorted({key for conflict in conflicts for key in conflict["member_keys"]})
                    )
                    fingerprint = hashlib.sha256(parsed.encode("utf-8")).hexdigest()
                    if (
                        completion_decider is None
                        or any(not conflict["previously_accepted"] for conflict in conflicts)
                        or ownership_corrections >= MAX_OWNERSHIP_CORRECTIONS
                        or fingerprint in rejected_fingerprints
                    ):
                        raise CustomModuleGenerationError(reason)
                    from .atomic_region_correction import RegionCorrection

                    # Keep the nonconflicting candidate declarations provisional.
                    # A later correction cannot quietly omit their identities/APIs.
                    if pending_correction is None and additions:
                        pending_correction = RegionCorrection(
                            additions, frozenset(range(len(additions))),
                        )
                    page_correction = {
                        "immutable_conflicts": conflicts,
                        "pending_declarations": list(pending_correction.chunks) if pending_correction else [],
                        "rules": (
                            "The previous page was rejected; none of it was committed. "
                            "Keep every pending declaration identity and public API. "
                            "Implement pending behavior against immutable_source; never replace "
                            "or repeat a conflicting definition. This corrects only the new page."
                        ),
                    }
                    if len(json.dumps(page_correction, ensure_ascii=False)) > MAX_ACCEPTED_CONTEXT_CHARS:
                        raise CustomModuleGenerationError(
                            reason + "; exact conflict context exceeds the bounded page context"
                        )
                    ownership_corrections += 1
                    rejected_fingerprints.add(fingerprint)
                    emit_root_cause(
                        "atomic_concern_page_conflict_localized", stage="production", result="RETRY",
                        reason=reason,
                        details={"owner": payload["host_selected_class"], "region": region,
                                 "concern": payload["concern"]["name"], "page": index,
                                 "accepted_units": len(parts), "next_work": next_work,
                                 "correction": page_correction},
                    )
                    continue
                chunks = additions
                if pending_correction is not None:
                    pending_keys = {
                        key for chunk in pending_correction.chunks
                        for key in _member_declaration_symbols(chunk)
                    }
                    # Validate that pending identities/APIs survived, while
                    # allowing genuinely new helper declarations in source order.
                    pending_correction.merge("\n\n".join(
                        chunk for chunk in chunks
                        if pending_keys.intersection(_member_declaration_symbols(chunk))
                    ))
                parsed = "\n\n".join(chunks)
            if len(parts) + len(chunks) > MAX_REGION_UNITS:
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_UNIT_LIMIT: page exceeds remaining semantic unit budget; "
                    f"got {len(chunks)}, remaining {MAX_REGION_UNITS - len(parts)}"
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
                decision = _validated_decision(completion_decider({
                    **payload,
                    **accepted,
                    "completion_phase": "assess_completion",
                    "current_page_source": parsed,
                    "echoed_member_keys": list(echoed),
                    "previous_next_work": next_work,
                    "page_index": index,
                }))
                done = decision["done"]
                next_work = decision["next_work"]
            if not chunks and not done:
                raise CustomModuleGenerationError(
                    "ATOMIC_REGION_NO_PROGRESS: page only repeated immutable declarations"
                )
            parts.extend(chunks)
            symbols.update(page_symbols)
            page_correction = None
            pending_correction = None
            ownership_corrections = 0
            rejected_fingerprints.clear()
            emit_root_cause(
                "atomic_concern_page_accepted", stage="production", result="PASS",
                details={"owner": payload["host_selected_class"], "region": region,
                         "concern": payload["concern"]["name"], "page": index,
                         "unit_count": len(chunks), "done": done, "next_work": next_work,
                         "echoed_member_keys": list(echoed),
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
