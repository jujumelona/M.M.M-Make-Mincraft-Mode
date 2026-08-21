from __future__ import annotations

"""Incremental host-owned causal state for append-only agent tool loops."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import causal_tool_graph as _graph
from .causal_tool_graph import transition_for_schema, verified_state_from_messages
from .grounding_policy import host_baseline_causal_facts

_SOURCE_MUTATION_NAMES = frozenset(
    {
        "apply_source_patch",
        "apply_source_edit",
        "apply_java_operations",
        "repair_project",
    }
)
_STALE_EVIDENCE_FACTS = frozenset(
    {
        "code_evidence",
        "project_evidence",
        "ecosystem_evidence",
        "external_observation",
        "evidence_ready",
    }
)
_FRESH_EVIDENCE_FACTS = frozenset(
    {
        "code_evidence",
        "project_evidence",
        "ecosystem_evidence",
        "external_observation",
    }
)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _schema_signature(schemas: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    # Causal transitions are registered by canonical tool name. Descriptions and JSON
    # argument schemas do not affect host state transitions, so do not reserialize them.
    return tuple(sorted(name for schema in schemas if (name := _tool_name(schema))))


def _message_anchor(message: Mapping[str, Any]) -> str:
    """Bounded continuity fingerprint for the first/last processed transcript item."""

    payload = json.dumps(message, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=12).hexdigest()


def _is_failed_source_mutation(message: Mapping[str, Any]) -> bool:
    if str(message.get("role", "")).strip().casefold() != "tool":
        return False
    if str(message.get("name", "")).strip() not in _SOURCE_MUTATION_NAMES:
        return False
    payload = _graph._payload(message)
    if payload is None:
        return False
    return payload.get("ok") is not True or _graph._has_explicit_semantic_failure(payload)


def _latest_failed_source_mutation(messages: Sequence[Mapping[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _is_failed_source_mutation(messages[index]):
            return index
    return None


def _transitions(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        transition.name: transition
        for schema in schemas
        if (transition := transition_for_schema(schema)).name
    }


def _advance_verified_state(
    state: frozenset[str],
    messages: Sequence[Mapping[str, Any]],
    schemas: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Advance only new observations while reusing causal_tool_graph semantics."""

    transitions = _transitions(schemas)
    facts = set(state)
    for message in messages:
        if str(message.get("role", "")).casefold() != "tool":
            continue
        payload = _graph._payload(message)
        if payload is None or payload.get("ok") is not True:
            continue
        name = str(message.get("name", "")).strip()
        transition = transitions.get(name)
        if transition is None or not transition.reviewed:
            continue
        if not transition.preconditions.issubset(facts):
            continue
        facts.update(_graph._semantic_effects(name, payload, set(transition.effects)))
    return frozenset(facts)


def _fresh_evidence_ready(state: frozenset[str]) -> bool:
    return "evidence_ready" in state and bool(_FRESH_EVIDENCE_FACTS.intersection(state))


@dataclass(frozen=True)
class CausalStateSnapshot:
    state: frozenset[str]
    query: str
    replayed_full_transcript: bool
    processed_suffix_messages: int


class CausalStateLedger:
    """Cache verified causal facts across one append-only live tool loop.

    The model router appends assistant/tool observations between rounds. Replaying the
    complete transcript on every round makes causal planning O(n^2) over a long loop.
    This ledger performs one full replay, then advances only the newly appended suffix.
    Any continuity or tool-surface change fails safe to a complete replay.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._processed = 0
        self._first_anchor = ""
        self._last_anchor = ""
        self._schema_signature: tuple[str, ...] = ()
        self._verified_state = frozenset({"workspace_bound"})
        self._baseline_state = frozenset()
        self._query = ""
        self._refresh_required = False
        self._fresh_state = frozenset({"workspace_bound"})

    def resolve(
        self,
        messages: Sequence[Mapping[str, Any]],
        schemas: Sequence[Mapping[str, Any]],
        *,
        require_fresh_evidence: bool,
        query_fn: Callable[[Sequence[Mapping[str, Any]]], str],
    ) -> CausalStateSnapshot:
        signature = _schema_signature(schemas)
        if not self._can_advance(messages, signature):
            return self._rebuild(
                messages,
                schemas,
                signature=signature,
                require_fresh_evidence=require_fresh_evidence,
                query_fn=query_fn,
            )

        suffix = messages[self._processed :]
        if any(
            str(message.get("role", "")).strip().casefold() in {"system", "user"}
            for message in suffix
        ):
            return self._rebuild(
                messages,
                schemas,
                signature=signature,
                require_fresh_evidence=require_fresh_evidence,
                query_fn=query_fn,
            )

        for message in suffix:
            self._verified_state = _advance_verified_state(
                self._verified_state,
                (message,),
                schemas,
            )
            if _is_failed_source_mutation(message):
                self._refresh_required = True
                self._fresh_state = frozenset({"workspace_bound"})
                continue
            if self._refresh_required:
                self._fresh_state = _advance_verified_state(
                    self._fresh_state,
                    (message,),
                    schemas,
                )
                if _fresh_evidence_ready(self._fresh_state):
                    self._refresh_required = False

        self._remember_continuity(messages, signature)
        return self._snapshot(replayed=False, suffix_count=len(suffix))

    def _can_advance(
        self,
        messages: Sequence[Mapping[str, Any]],
        signature: tuple[str, ...],
    ) -> bool:
        if not self._initialized or signature != self._schema_signature:
            return False
        if len(messages) < self._processed:
            return False
        if self._processed == 0:
            return True
        if not messages:
            return False
        if _message_anchor(messages[0]) != self._first_anchor:
            return False
        return _message_anchor(messages[self._processed - 1]) == self._last_anchor

    def _rebuild(
        self,
        messages: Sequence[Mapping[str, Any]],
        schemas: Sequence[Mapping[str, Any]],
        *,
        signature: tuple[str, ...],
        require_fresh_evidence: bool,
        query_fn: Callable[[Sequence[Mapping[str, Any]]], str],
    ) -> CausalStateSnapshot:
        self._verified_state = verified_state_from_messages(
            messages,
            schemas,
            require_fresh_evidence=require_fresh_evidence,
        )
        self._baseline_state = frozenset(host_baseline_causal_facts(messages))
        self._query = query_fn(messages)
        failed_at = _latest_failed_source_mutation(messages)
        if failed_at is None:
            self._refresh_required = False
            self._fresh_state = frozenset({"workspace_bound"})
        else:
            self._fresh_state = verified_state_from_messages(
                messages[failed_at + 1 :],
                schemas,
                require_fresh_evidence=True,
            )
            self._refresh_required = "evidence_ready" not in self._fresh_state
        self._initialized = True
        self._remember_continuity(messages, signature)
        return self._snapshot(replayed=True, suffix_count=len(messages))

    def _remember_continuity(
        self,
        messages: Sequence[Mapping[str, Any]],
        signature: tuple[str, ...],
    ) -> None:
        self._processed = len(messages)
        self._schema_signature = signature
        self._first_anchor = _message_anchor(messages[0]) if messages else ""
        self._last_anchor = _message_anchor(messages[-1]) if messages else ""

    def _snapshot(self, *, replayed: bool, suffix_count: int) -> CausalStateSnapshot:
        facts = set(self._verified_state)
        facts.update(self._baseline_state)
        if self._refresh_required:
            facts.difference_update(_STALE_EVIDENCE_FACTS)
        return CausalStateSnapshot(
            state=frozenset(facts),
            query=self._query,
            replayed_full_transcript=replayed,
            processed_suffix_messages=suffix_count,
        )


__all__ = ["CausalStateLedger", "CausalStateSnapshot"]
