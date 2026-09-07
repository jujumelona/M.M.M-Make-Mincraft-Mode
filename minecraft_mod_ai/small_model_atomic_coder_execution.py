from __future__ import annotations

"""Force live custom-module coding into one-obligation model turns.

The planner may keep a multi-step host task for dependency and verification purposes, but
one small coder decode must never be responsible for every implementation obligation at
once. This runtime contract slices the already-approved task at the ModelRouter boundary,
keeps each decode on one obligation, preserves the canonical context required to execute
that obligation, and preserves the shared staged workspace between slices. The outer
custom-module generator still owns the final transaction and gates.
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
    if not _is_sequence(raw):
        return ()
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        obligation = str(item.get("obligation") or "").strip()
        if obligation:
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
        return (tuple(dict(message) for message in messages),)
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
    steps = _steps(contract)
    if not steps:
        _fail("canonical coder execution contract has no implementation steps")

    batches: list[tuple[dict[str, Any], ...]] = []
    for step_index, step in enumerate(steps):
        current = copy.deepcopy(request)
        current_module = _mapping(current.pop("module", None))
        current_evidence = {
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
        current_module = {
            "module_id": str(
                current_module.get("module_id") or current_evidence["task_id"]
            ),
            "kind": str(current_module.get("kind") or "custom_java"),
            "evidence_task": current_evidence,
        }

        current["task"] = (
            "Implement only the atomic obligation declared in "
            "module.evidence_task.coder_execution_contract.step."
        )
        rules = [str(item) for item in current.get("rules", ()) if str(item).strip()]
        current["rules"] = [
            "Work on this atomic obligation only; sibling obligations are host-scheduled later.",
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
            "policy": "one_model_call_one_implementation_obligation",
        }
        batch = [dict(message) for message in messages]
        batch[user_index] = {
            **batch[user_index],
            "content": json.dumps(current, ensure_ascii=False, separators=(",", ":")),
        }
        batches.append(tuple(batch))
    return tuple(batches)


def _bounded_initial_observations(
    generator_module: Any,
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

    page_budget = max(1024, min(int(byte_budget), _MAX_INITIAL_SOURCE_BYTES))
    page = index.select_page(
        query=query,
        diagnostic_paths=diagnostic_paths,
        byte_budget=page_budget,
        cursor="",
    )
    if generator_module._json_size(page) > page_budget:
        raise generator_module.CustomModuleGenerationError(
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
    generator_module._update_digest(source_page_digest, page_commitment)

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
        generator_module._append_observation(
            records,
            record_keys,
            generator_module._exact_observation(
                path=str(item["path"]),
                sha256=str(item.get("sha256", "")),
                start=int(item.get("content_start_bytes", 0)),
                content=content_str.encode("utf-8"),
                source_page=int(page.get("page_index", 0)),
            ),
        )

    observation_digest = hashlib.sha256()
    for record in records:
        generator_module._update_digest(observation_digest, record)
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


def install(*, custom_module_generator_module: Any, model_router_module: Any) -> None:
    """Install the atomic coder boundary after task/write authority contracts."""

    Router = model_router_module.ModelRouter
    if getattr(Router.generate_text, _MARKER, False):
        return

    original_generate_text = Router.generate_text
    original_reuse = custom_module_generator_module._materialize_owned_reuse_context

    @wraps(original_generate_text)
    def generate_text(
        self: Any,
        role: str,
        messages: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if str(role).strip().casefold() not in {"coder", "coder_safe"}:
            return original_generate_text(self, role, messages, *args, **kwargs)
        if not _is_sequence(messages):
            return original_generate_text(self, role, messages, *args, **kwargs)

        batches = atomicize_coder_messages(messages)
        if len(batches) == 1:
            return original_generate_text(self, role, batches[0], *args, **kwargs)

        from .model_response_templates import response_schema

        structured_summary = kwargs.get("response_schema") == response_schema("coder_summary")
        summaries: list[str] = []
        for index, batch in enumerate(batches, start=1):
            result = original_generate_text(self, role, batch, *args, **kwargs)
            summary = (
                json.loads(result)["summary"]
                if structured_summary
                else str(result or "").strip()
            )
            if len(summary) > _MAX_SUMMARY_CHARS_PER_STEP:
                summary = summary[:_MAX_SUMMARY_CHARS_PER_STEP] + "…"
            summaries.append(f"atomic step {index}/{len(batches)}: {summary}")
        combined = "\n".join(summaries)
        return (
            json.dumps({"summary": combined}, ensure_ascii=False)
            if structured_summary
            else combined
        )

    def collect_initial_observations(
        index: Any,
        *,
        query: str,
        byte_budget: int,
        diagnostic_paths: Iterable[str] = (),
    ) -> dict[str, Any]:
        return _bounded_initial_observations(
            custom_module_generator_module,
            index,
            query=query,
            byte_budget=byte_budget,
            diagnostic_paths=diagnostic_paths,
        )

    @wraps(original_reuse)
    def materialize_owned_reuse_context(
        project_root: Any,
        module: Any,
        *,
        byte_budget: int = _MAX_APPROVED_REUSE_BYTES,
    ) -> Any:
        return _bounded_reuse_context(
            original_reuse,
            project_root,
            module,
            byte_budget=byte_budget,
        )

    setattr(generate_text, _MARKER, True)
    setattr(collect_initial_observations, _MARKER, True)
    setattr(materialize_owned_reuse_context, _MARKER, True)
    Router.generate_text = generate_text
    custom_module_generator_module._collect_initial_observations = collect_initial_observations
    custom_module_generator_module._materialize_owned_reuse_context = materialize_owned_reuse_context


def assert_installed(*, custom_module_generator_module: Any, model_router_module: Any) -> None:
    checks = (
        getattr(model_router_module.ModelRouter.generate_text, _MARKER, False),
        getattr(custom_module_generator_module._collect_initial_observations, _MARKER, False),
        getattr(custom_module_generator_module._materialize_owned_reuse_context, _MARKER, False),
    )
    if not all(checks):
        raise RuntimeError("Small-model atomic coder execution contract is not active.")


__all__ = [
    "AtomicCoderContractError",
    "assert_installed",
    "atomicize_coder_messages",
    "install",
]
