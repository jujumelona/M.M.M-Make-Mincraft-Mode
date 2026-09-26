from __future__ import annotations

"""Force live custom-module coding into one-obligation model turns.

The planner may keep a multi-step host task for dependency and verification purposes, but
one small coder decode must never be responsible for every implementation obligation at
once. This runtime contract slices the already-approved task at the custom-module coder
call seam, keeps each decode on one obligation, preserves the canonical context required
to execute that obligation, and preserves the shared staged workspace between slices.
The outer custom-module generator still owns the final transaction and gates.
"""

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from functools import wraps
from typing import Any

from .implementation_template_contract import SCHEMA as CODER_EXECUTION_SCHEMA

_MARKER = "_mmm_small_model_atomic_coder"
_ATOMIC_SCHEMA = "mmm/atomic-coder-step"
_MAX_INITIAL_SOURCE_BYTES = 4 * 1024
_MAX_APPROVED_REUSE_BYTES = 4 * 1024
_MAX_SUMMARY_CHARS_PER_STEP = 1024
_MAX_AUTHORED_FRAGMENT_BYTES = 2 * 1024
_AUTHORED_FRAGMENT_SCHEMA = "mmm/authored-plan-fragment-v1"


class AtomicCoderContractError(RuntimeError):
    """Canonical coder authority could not be lowered safely to one obligation."""


def _fail(reason: str) -> None:
    raise AtomicCoderContractError(f"CODER_CONTRACT_LOWERING_FAILED: {reason}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence_copy(value: Any) -> list[Any]:
    if not _is_sequence(value):
        return []
    return copy.deepcopy(list(value))


def _steps(contract: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = contract.get("implementation_steps")
    if not _is_sequence(raw) or not raw:
        _fail("canonical coder execution contract has no implementation steps")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            _fail(f"implementation step {index + 1} is not an object")
        obligation = str(item.get("obligation") or "").strip()
        if not obligation:
            _fail(f"implementation step {index + 1} has no obligation")
        result.append(dict(item))
    return tuple(result)


def _task_ref(contract: Mapping[str, Any], evidence_task: Mapping[str, Any]) -> str:
    return str(contract.get("task_ref") or evidence_task.get("task_id") or "").strip()


def _objective(contract: Mapping[str, Any]) -> str:
    return str(contract.get("semantic_outcome") or "").strip()


def _dataflow(contract: Mapping[str, Any], name: str) -> list[str]:
    nested = contract.get("dataflow")
    if isinstance(nested, Mapping) and _is_sequence(nested.get(name)):
        return [str(item).strip() for item in nested[name] if str(item).strip()]
    return []


def _step_target_refs(step: Mapping[str, Any]) -> list[str]:
    return [
        str(item).strip()
        for item in _sequence_copy(step.get("target_refs"))
        if str(item).strip()
    ]


def _step_targets(
    contract: Mapping[str, Any],
    step: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_targets = contract.get("targets")
    if not _is_sequence(raw_targets):
        return []
    targets = [
        copy.deepcopy(dict(item)) for item in raw_targets if isinstance(item, Mapping)
    ]
    refs = frozenset(_step_target_refs(step))
    if not refs:
        return targets
    return [
        target
        for target in targets
        if str(target.get("locator") or "").strip() in refs
    ]


def _atomic_step(step: Mapping[str, Any], *, index: int, count: int) -> dict[str, Any]:
    obligation = str(step.get("obligation") or "").strip()
    target_refs = _step_target_refs(step)
    checklist = _sequence_copy(step.get("execution_checklist"))
    done_when = str(step.get("done_when") or "").strip()
    if not obligation:
        _fail("atomic step has no obligation")
    if not target_refs:
        _fail(f"atomic step {index + 1} has no target_refs")
    if not checklist:
        _fail(f"atomic step {index + 1} has no execution_checklist")
    if not done_when:
        _fail(f"atomic step {index + 1} has no completion condition")
    return {
        "index": index + 1,
        "count": count,
        "sequence": step.get("sequence", index),
        "obligation": obligation,
        "target_refs": target_refs,
        "consumes": _sequence_copy(step.get("consumes")),
        "must_provide": _sequence_copy(step.get("must_provide")),
        "execution_checklist": checklist,
        "done_when": done_when,
    }


def _step_execution_signature(step: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Identify steps that operate on the same host-owned state transition."""

    return (
        tuple(_step_target_refs(step)),
        tuple(str(item).strip() for item in _sequence_copy(step.get("consumes")) if str(item).strip()),
        tuple(str(item).strip() for item in _sequence_copy(step.get("must_provide")) if str(item).strip()),
    )


def _merge_coowned_steps(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(group) == 1:
        return copy.deepcopy(dict(group[0]))

    first = copy.deepcopy(dict(group[0]))
    obligations = [str(step.get("obligation") or "").strip() for step in group]
    done_when = [str(step.get("done_when") or "").strip() for step in group]
    checklist: list[Any] = []
    seen_checklist: set[str] = set()
    for step in group:
        for item in _sequence_copy(step.get("execution_checklist")):
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if marker in seen_checklist:
                continue
            seen_checklist.add(marker)
            checklist.append(copy.deepcopy(item))

    first["obligation"] = (
        "Satisfy all co-owned obligations in this one host-owned state transition:\n- "
        + "\n- ".join(obligations)
    )
    first["execution_checklist"] = checklist
    first["done_when"] = (
        "Every co-owned obligation in this grouped state transition is satisfied: "
        + " | ".join(done_when)
    )
    return first


def _coalesce_execution_steps(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Avoid reopening one target for constraint-only siblings with identical dataflow."""

    groups: list[list[Mapping[str, Any]]] = []
    for step in steps:
        signature = _step_execution_signature(step)
        if groups and _step_execution_signature(groups[-1][0]) == signature:
            groups[-1].append(step)
        else:
            groups.append([step])
    return tuple(_merge_coowned_steps(group) for group in groups)


def _atomic_contract(
    contract: Mapping[str, Any],
    evidence_task: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    index: int,
    count: int,
) -> dict[str, Any]:
    """Build one obligation contract without discarding canonical execution authority."""

    if contract.get("schema_version") != CODER_EXECUTION_SCHEMA:
        _fail(
            "canonical schema mismatch: expected "
            f"{CODER_EXECUTION_SCHEMA!r}, got {contract.get('schema_version')!r}"
        )
    task_ref = _task_ref(contract, evidence_task)
    if not task_ref:
        _fail("coder contract has no task identity")
    if task_ref != str(evidence_task.get("task_id") or "").strip():
        _fail("coder contract task_ref disagrees with evidence_task.task_id")

    source_sha = str(
        contract.get("task_sha256_input") or evidence_task.get("task_sha256") or ""
    ).strip()
    source_contract_sha = str(contract.get("contract_sha256") or "").strip()
    objective = _objective(contract)
    execution_role = str(contract.get("execution_role") or "").strip()
    target_constraints = copy.deepcopy(_mapping(contract.get("target_constraints")))
    targets = _step_targets(contract, step)
    worksheet = copy.deepcopy(contract.get("engineering_worksheet"))
    protected_boundaries = copy.deepcopy(_mapping(contract.get("protected_boundaries")))
    verification_plan = _sequence_copy(contract.get("verification_plan"))
    completion_predicate = copy.deepcopy(_mapping(contract.get("completion_predicate")))

    required = {
        "source_contract_sha256": source_contract_sha,
        "semantic_outcome": objective,
        "execution_role": execution_role,
        "target_constraints": target_constraints,
        "targets": targets,
        "engineering_worksheet": worksheet,
        "protected_boundaries": protected_boundaries,
        "verification_plan": verification_plan,
        "completion_predicate": completion_predicate,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        _fail("canonical coder authority is incomplete: " + ", ".join(missing))

    return {
        "schema_version": _ATOMIC_SCHEMA,
        "task_ref": task_ref,
        "source_task_sha256": source_sha,
        "source_contract_sha256": source_contract_sha,
        "semantic_outcome": objective,
        "execution_role": execution_role,
        "requirement_refs": _sequence_copy(contract.get("requirement_refs")),
        "target_constraints": target_constraints,
        "targets": targets,
        "depends_on": _sequence_copy(contract.get("depends_on")),
        "consumes": _dataflow(contract, "consumes"),
        "provides": _dataflow(contract, "provides"),
        "engineering_worksheet": worksheet,
        "artifacts": _sequence_copy(contract.get("artifacts")),
        "reuse_refs": _sequence_copy(contract.get("reuse_refs")),
        "protected_boundaries": protected_boundaries,
        "verification_plan": verification_plan,
        "completion_predicate": completion_predicate,
        "step": _atomic_step(step, index=index, count=count),
        "scope_policy": (
            "Execute only this obligation. Do not start, pre-implement, redesign, or summarize "
            "sibling obligations. Preserve the complete canonical execution authority carried "
            "in this atomic contract; exact writable paths are additionally enforced by host tools."
        ),
    }


def _implementation_request(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            request = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(request, dict) and request.get("phase") == "implement_module":
            return index, request
    return None


def _authored_implementation_request(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            request = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(request, dict) and request.get("phase") == "implement_authored_design":
            module = request.get("module")
            authored = module.get("authored_plan") if isinstance(module, Mapping) else None
            if isinstance(authored, Mapping):
                return index, request
    return None


def _split_authored_text(text: str, *, max_bytes: int) -> tuple[str, ...]:
    """Split exact authored text into deterministic UTF-8-safe bounded fragments."""

    encoded = text.encode("utf-8")
    limit = max(256, int(max_bytes))
    if len(encoded) <= limit:
        return (text,)

    fragments: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(len(encoded), start + limit)
        if end < len(encoded):
            while end > start and (encoded[end] & 0xC0) == 0x80:
                end -= 1
        if end <= start:
            _fail("authored design could not be split on a UTF-8 boundary")
        fragment = encoded[start:end].decode("utf-8")
        if end < len(encoded):
            newline = fragment.rfind("\n")
            if newline >= len(fragment) // 2:
                preferred = fragment[: newline + 1]
                preferred_bytes = preferred.encode("utf-8")
                if preferred_bytes:
                    fragment = preferred
                    end = start + len(preferred_bytes)
        fragments.append(fragment)
        start = end

    if "".join(fragments) != text:
        _fail("authored design fragmentation changed the approved text")
    return tuple(fragments)


def _atomicize_authored_coder_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    user_index: int,
    request: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    module = request.get("module")
    if not isinstance(module, Mapping):
        _fail("implement_authored_design request has no module object")
    authored = module.get("authored_plan")
    if not isinstance(authored, Mapping):
        _fail("implement_authored_design request has no authored_plan object")
    text = authored.get("text")
    if not isinstance(text, str):
        _fail("implement_authored_design authored_plan.text is not a string")

    if str(module.get("authored_execution_mode") or "").strip() == "bounded_coherent":
        # A worksheet-shaped authored design is one semantic contract. Splitting it by
        # byte range makes the first model turn architect from state_model alone before
        # it can see integration, persistence, UI, failure, and verification facets.
        # Keep the complete contract in one canonical tool loop; context budgeting may
        # trim repository observations, but it must never fragment the approved design.
        return (tuple(dict(message) for message in messages),)

    fragments = _split_authored_text(text, max_bytes=_MAX_AUTHORED_FRAGMENT_BYTES)
    if len(fragments) == 1:
        return (tuple(dict(message) for message in messages),)

    source_sha256 = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    batches: list[tuple[dict[str, Any], ...]] = []
    start_byte = 0
    count = len(fragments)
    for index, fragment in enumerate(fragments):
        fragment_bytes = fragment.encode("utf-8")
        end_byte = start_byte + len(fragment_bytes)
        current = copy.deepcopy(dict(request))
        current_module = _mapping(current.get("module"))
        current_authored = _mapping(current_module.get("authored_plan"))
        current_authored["text"] = fragment
        current_authored["fragment_contract"] = {
            "schema_version": _AUTHORED_FRAGMENT_SCHEMA,
            "fragment_index": index + 1,
            "fragment_count": count,
            "source_text_sha256": source_sha256,
            "start_byte": start_byte,
            "end_byte": end_byte,
            "source_bytes": len(text.encode("utf-8")),
            "policy": (
                "This text is an exact ordered fragment of the approved authored design. "
                "Implement it without redesigning or discarding behavior from earlier fragments."
            ),
        }
        current_module["authored_plan"] = current_authored
        current["module"] = current_module
        current["task"] = (
            "Implement only this bounded authored-design fragment in the preserved staged "
            "workspace. Treat earlier fragments as already-approved behavior that must remain "
            "intact; do not reinterpret this fragment as a standalone replacement design."
        )
        rules = [str(item) for item in current.get("rules", ()) if str(item).strip()]
        current["rules"] = [
            "authored_plan.text is one exact host-scheduled fragment; preserve its wording and semantics.",
            "Use the fragment_contract byte range and source_text_sha256 as integrity metadata; never invent omitted authored text.",
            "Inspect the current staged workspace before editing so later fragments extend rather than overwrite earlier work.",
            *rules,
        ]
        if index > 0 and "initial_exact_source_context" in current:
            current["initial_exact_source_context"] = {
                "mode": "retrieve_current_authored_fragment_with_tools",
                "reason": (
                    "Earlier authored fragments may have changed the staged workspace; "
                    "do not replay the stale pre-fragment source page."
                ),
            }
        current["authored_execution"] = {
            "schema_version": _AUTHORED_FRAGMENT_SCHEMA,
            "fragment_index": index + 1,
            "fragment_count": count,
            "source_text_sha256": source_sha256,
            "policy": "one_model_call_one_bounded_authored_design_fragment",
        }

        batch = [dict(message) for message in messages]
        batch[user_index] = {
            **batch[user_index],
            "content": json.dumps(current, ensure_ascii=False, separators=(",", ":")),
        }
        batches.append(tuple(batch))
        start_byte = end_byte
    return tuple(batches)


def atomicize_coder_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Return one canonical-context message batch per approved implementation obligation.

    Non-implementation messages pass through unchanged. Once a request declares
    ``phase=implement_module``, every host-owned contract component is mandatory and lowering
    fails closed rather than falling back to the original large coder request.
    """

    parsed = _implementation_request(messages)
    if parsed is None:
        authored = _authored_implementation_request(messages)
        if authored is None:
            return (tuple(dict(message) for message in messages),)
        user_index, request = authored
        return _atomicize_authored_coder_messages(
            messages,
            user_index=user_index,
            request=request,
        )
    user_index, request = parsed
    module = request.get("module")
    if not isinstance(module, Mapping):
        _fail("implement_module request has no module object")
    evidence_task = module.get("evidence_task")
    if not isinstance(evidence_task, Mapping):
        _fail("implement_module request has no evidence_task object")
    contract = evidence_task.get("coder_execution_contract")
    if not isinstance(contract, Mapping):
        _fail("implement_module request has no coder_execution_contract")
    if contract.get("schema_version") != CODER_EXECUTION_SCHEMA:
        _fail(
            "implement_module request carries a non-canonical coder execution schema: "
            f"{contract.get('schema_version')!r}"
        )
    raw_steps = _steps(contract)
    steps = _coalesce_execution_steps(raw_steps)

    batches: list[tuple[dict[str, Any], ...]] = []
    for step_index, step in enumerate(steps):
        current = copy.deepcopy(request)
        current_module = _mapping(current.pop("module", None))
        current_evidence: dict[str, Any] = {
            "task_id": _task_ref(contract, evidence_task),
            "coder_execution_contract": _atomic_contract(
                contract,
                evidence_task,
                step,
                index=step_index,
                count=len(steps),
            ),
        }
        task_sha = str(evidence_task.get("task_sha256") or "").strip()
        if task_sha:
            current_evidence["task_sha256"] = task_sha
        for key in ("owned_anchors", "production_bindings", "required_gates"):
            if key in evidence_task:
                current_evidence[key] = copy.deepcopy(evidence_task[key])
        current_module = {
            "module_id": str(
                current_module.get("module_id") or current_evidence["task_id"]
            ),
            "kind": str(current_module.get("kind") or "custom_java"),
            "evidence_task": current_evidence,
        }

        current["task"] = (
            "Implement only the atomic host-owned state transition declared in "
            "module.evidence_task.coder_execution_contract.step."
        )
        rules = [str(item) for item in current.get("rules", ()) if str(item).strip()]
        current["rules"] = [
            "Work on this atomic state transition only; independent sibling state transitions are host-scheduled later.",
            "Read and obey the atomic contract's engineering worksheet, exact targets, execution checklist, protected boundaries, and verification plan before editing.",
            "Read only the exact source chunk needed for this obligation; use bounded retrieval tools for additional source.",
            *rules,
        ]
        if step_index > 0 and "initial_exact_source_context" in current:
            current["initial_exact_source_context"] = {
                "mode": "retrieve_current_atomic_step_with_tools",
                "reason": (
                    "Earlier atomic steps may have changed the staged workspace; do not replay "
                    "the stale pre-step source page."
                ),
            }

        current["module"] = current_module
        current["atomic_execution"] = {
            "schema_version": _ATOMIC_SCHEMA,
            "step_index": step_index + 1,
            "step_count": len(steps),
            "policy": "one_model_call_one_host_owned_state_transition",
        }
        batch = [dict(message) for message in messages]
        batch[user_index] = {
            **batch[user_index],
            "content": json.dumps(current, ensure_ascii=False, separators=(",", ":")),
        }
        batches.append(tuple(batch))
    return tuple(batches)


def _bounded_initial_observations(
    index: Any,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Capture only the strongest initial exact-source page.

    The original collector walks every ProjectIndex page to build a complete ledger even
    though generation sends only the first bounded observation page and already exposes
    RAG/source tools for follow-up reads. That makes coder startup O(project source size).
    For atomic coding, one relevance-ranked exact page is sufficient bootstrap evidence;
    later source is retrieved on demand from the same indexed project.
    """

    # These helpers are owned by source_observation_context after the source-observation
    # extraction. Import them lazily to avoid the module cycle created by that module's
    # @bounded_initial_observations decorator.
    from .custom_module_errors import CustomModuleGenerationError
    from .source_observation_context import (
        append_observation,
        exact_observation,
        json_size,
        update_digest,
    )

    page_budget = max(1024, min(int(byte_budget), _MAX_INITIAL_SOURCE_BYTES))
    page = index.select_page(
        query=query,
        diagnostic_paths=diagnostic_paths,
        byte_budget=page_budget,
        cursor="",
    )
    if json_size(page) > page_budget:
        raise CustomModuleGenerationError(
            "Host project context page exceeded its byte budget."
        )

    project_sha256 = str(page["project_sha256"])
    query_sha256 = str(page["query_sha256"])
    source_page_digest = hashlib.sha256()
    page_commitment = {
        "page_index": page["page_index"],
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "start_position": page["start_position"],
        "start_offset": page["start_offset"],
        "next_cursor": page["next_cursor"],
        "files": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "content_start_bytes": item["content_start_bytes"],
                "content_end_bytes": item["content_end_bytes"],
            }
            for item in page["files"]
        ],
    }
    update_digest(source_page_digest, page_commitment)

    records: list[dict[str, Any]] = []
    record_keys: set[tuple[str, int, int]] = set()
    for item in page.get("files", []):
        if not (
            isinstance(item, dict)
            and "path" in item
            and ("content" in item or "text" in item)
        ):
            continue
        content_str = str(item.get("content", item.get("text", "")))
        append_observation(
            records,
            record_keys,
            exact_observation(
                path=str(item["path"]),
                sha256=str(item.get("sha256", "")),
                start=int(item.get("content_start_bytes", 0)),
                content=content_str.encode("utf-8"),
                source_page=int(page.get("page_index", 0)),
            ),
        )

    observation_digest = hashlib.sha256()
    for record in records:
        update_digest(observation_digest, record)
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v1",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "source_page_count": 1,
        "observation_count": len(records),
        "source_pages_sha256": "sha256:" + source_page_digest.hexdigest(),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "policy": {
            "exact_source_quotes": True,
            "path_sha256_byte_range_bound": True,
            "initial_page_only": True,
            "source_page_complete": bool(page.get("complete", False)),
            "supplemental_retrieval_available": True,
        },
    }
    return {
        "schema_version": "mmm/source-observation-ledger-v1",
        "receipt": receipt,
        "records": records,
    }


def _bounded_reuse_context(
    current: Any,
    project_root: Any,
    module: Any,
    *,
    byte_budget: int = _MAX_APPROVED_REUSE_BYTES,
) -> Any:
    return current(
        project_root,
        module,
        byte_budget=max(1024, min(int(byte_budget), _MAX_APPROVED_REUSE_BYTES)),
    )


def atomic_coder_call(func: Any) -> Any:
    """Attach atomic state-transition scheduling at the reviewed coder seam."""

    @wraps(func)
    def generate_text(
        router: Any,
        role: str,
        messages: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if str(role).strip().casefold() not in {"coder", "coder_safe"}:
            return func(router, role, messages, *args, **kwargs)
        if not _is_sequence(messages):
            return func(router, role, messages, *args, **kwargs)

        batches = atomicize_coder_messages(messages)
        if len(batches) == 1:
            return func(router, role, batches[0], *args, **kwargs)

        from .model_response_templates import (
            parse_response_text,
            response_schema,
            serialize_response,
        )

        coder_summary_schema = response_schema("coder_summary")
        structured_summary = kwargs.get("response_schema") == coder_summary_schema
        summary_max_length = int(
            coder_summary_schema["properties"]["summary"]["maxLength"]
        )
        summaries: list[str] = []
        raw_summaries: list[str] = []
        contract_results: list[bool] = []
        for index, batch in enumerate(batches, start=1):
            result = func(router, role, batch, *args, **kwargs)
            try:
                parsed = parse_response_text("coder_summary", result)
            except ValueError:
                if structured_summary:
                    raise
                summary = str(result or "").strip()
                contract_results.append(False)
            else:
                summary = str(parsed["summary"]).strip()
                contract_results.append(True)
            if len(summary) > _MAX_SUMMARY_CHARS_PER_STEP:
                summary = summary[:_MAX_SUMMARY_CHARS_PER_STEP] + "…"
            raw_summaries.append(summary)
            summaries.append(f"atomic step {index}/{len(batches)}: {summary}")
        combined = "\n".join(summaries)
        if contract_results and all(contract_results):
            if len(combined) > summary_max_length and len(set(raw_summaries)) == 1:
                combined = (
                    f"atomic steps 1-{len(batches)}/{len(batches)}: "
                    f"{raw_summaries[0]}"
                )
            if len(combined) > summary_max_length:
                ellipsis = "…" if summary_max_length > 0 else ""
                combined = combined[: max(0, summary_max_length - len(ellipsis))] + ellipsis
            return serialize_response("coder_summary", {"summary": combined})
        if any(contract_results):
            raise AtomicCoderContractError(
                "CODER_SUMMARY_TRANSPORT_MIXED: atomic coder steps returned a mixture "
                "of fixed coder_summary JSON and free text"
            )
        return combined

    setattr(generate_text, _MARKER, True)
    return generate_text


def bounded_initial_observations(
    func: Any,
    *,
    generator_module: Any | None = None,
) -> Any:
    """Own the bounded bootstrap source page directly instead of late rebinding."""

    del generator_module

    @wraps(func)
    def collect_initial_observations(
        index: Any,
        *,
        query: str,
        byte_budget: int,
        diagnostic_paths: Iterable[str] = (),
    ) -> dict[str, Any]:
        return _bounded_initial_observations(
            index,
            query=query,
            byte_budget=byte_budget,
            diagnostic_paths=diagnostic_paths,
        )

    setattr(collect_initial_observations, _MARKER, True)
    # The old runtime order first installed adaptive repository grounding, then
    # replaced it with this bounded atomic collector while copying this marker.
    # Publish the final ownership directly so adaptive_retrieval does not install
    # a dead wrapper that is immediately discarded later.
    setattr(
        collect_initial_observations,
        "__mmm_repository_grounding_live_context__",
        True,
    )
    return collect_initial_observations


def bounded_reuse_context(func: Any) -> Any:
    """Cap approved donor materialization at the source-owned helper boundary."""

    @wraps(func)
    def materialize_owned_reuse_context(
        project_root: Any,
        module: Any,
        *,
        byte_budget: int = _MAX_APPROVED_REUSE_BYTES,
    ) -> Any:
        return _bounded_reuse_context(
            func,
            project_root,
            module,
            byte_budget=byte_budget,
        )

    setattr(materialize_owned_reuse_context, _MARKER, True)
    return materialize_owned_reuse_context


def install(*, custom_module_generator_module: Any, model_router_module: Any) -> None:
    """Install only compatibility seams that still exist on the supplied module.

    The production whole-file coder no longer exposes the legacy text-call,
    observation-page, or donor-materialization helpers.  Do not recreate those
    retired surfaces merely to support a wrapper.
    """

    del model_router_module
    wrappers = (
        ("_generate_coder_text", atomic_coder_call),
        (
            "_collect_initial_observations",
            lambda current: bounded_initial_observations(
                current,
                generator_module=custom_module_generator_module,
            ),
        ),
        ("_materialize_owned_reuse_context", bounded_reuse_context),
    )
    for name, wrapper in wrappers:
        current = getattr(custom_module_generator_module, name, None)
        if current is None or getattr(current, _MARKER, False):
            continue
        setattr(custom_module_generator_module, name, wrapper(current))


def assert_installed(*, custom_module_generator_module: Any, model_router_module: Any) -> None:
    del model_router_module
    existing = [
        getattr(custom_module_generator_module, name)
        for name in (
            "_generate_coder_text",
            "_collect_initial_observations",
            "_materialize_owned_reuse_context",
        )
        if hasattr(custom_module_generator_module, name)
    ]
    if not all(getattr(value, _MARKER, False) for value in existing):
        raise RuntimeError("Small-model atomic coder execution contract is not active.")


__all__ = [
    "AtomicCoderContractError",
    "assert_installed",
    "atomic_coder_call",
    "atomicize_coder_messages",
    "bounded_initial_observations",
    "bounded_reuse_context",
    "install",
]