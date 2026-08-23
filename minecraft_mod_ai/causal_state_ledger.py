from __future__ import annotations

"""Incremental host-owned causal state for append-only agent tool loops."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import causal_tool_graph as _graph
from .causal_tool_graph import transition_for_schema
from .grounding_policy import host_baseline_causal_facts
from .source_mutation_contract import SOURCE_MUTATION_NAMES, mutation_payload_applied

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
_WORKSPACE_IMPACTS = frozenset({"unchanged", "rolled_back", "drift", "uncertain"})
_INVALIDATING_WORKSPACE_IMPACTS = frozenset({"drift", "uncertain"})
_SAFE_RETRY_WORKSPACE_IMPACTS = frozenset({"unchanged", "rolled_back"})
_VOLATILE_FAILURE_KEYS = frozenset(
    {
        "elapsed_ms",
        "duration_ms",
        "latency_ms",
        "timing_ms",
        "request_id",
        "trace_id",
        "timestamp",
    }
)
_WORKSPACE_IMPACT_MARKERS = (
    "[workspace_impact=",
    "[mmm-workspace-impact:",
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
    name = str(message.get("name", "")).strip()
    if name not in SOURCE_MUTATION_NAMES:
        return False
    payload = _graph._payload(message)
    return payload is not None and not mutation_payload_applied(name, payload)


def _workspace_impact_from_error(error: str) -> str:
    """Recover a normalized transaction fact from the text-only runtime boundary."""

    impacts = _workspace_impacts_from_error(error)
    if impacts:
        return _conservative_workspace_impact(impacts)
    # Spec/schema/resource-policy failures are raised by the tool service before the
    # transactional patcher is invoked, so they cannot have mutated the workspace.
    if error.lstrip().startswith("SpecValidationError:"):
        return "unchanged"
    # Unknown transport/runtime failures are conservative: without a transaction fact
    # the host cannot prove that the prior evidence still names the current snapshot.
    return "uncertain"


def _workspace_impacts_from_error(error: str) -> tuple[str, ...]:
    """Parse all marker claims without inventing an impact for marker-free text."""

    impacts: list[str] = []
    for marker in _WORKSPACE_IMPACT_MARKERS:
        search_at = 0
        while True:
            marker_at = error.find(marker, search_at)
            if marker_at < 0:
                break
            value_at = marker_at + len(marker)
            end = error.find("]", value_at)
            if end < 0:
                # A malformed transaction marker is not evidence of an unchanged
                # workspace. Treat it like an unknown impact below.
                impacts.append("unknown")
                break
            impacts.append(error[value_at:end].strip().casefold() or "unknown")
            search_at = end + 1
    return tuple(impacts)


def _conservative_workspace_impact(impacts: Sequence[str]) -> str:
    """Collapse conflicting transaction facts without trusting first-marker order."""

    normalized = tuple(str(value).strip().casefold() for value in impacts)
    # ``applied`` means the old snapshot is definitely stale. Unknown values and an
    # explicit uncertain marker cannot prove rollback, so normalize both to uncertain.
    if any(
        value in {"applied", "uncertain"} or value not in _WORKSPACE_IMPACTS
        for value in normalized
    ):
        return "uncertain"
    if "drift" in normalized:
        return "drift"
    if "rolled_back" in normalized:
        return "rolled_back"
    return "unchanged"


def _failed_source_mutation_workspace_impact(message: Mapping[str, Any]) -> str:
    if not _is_failed_source_mutation(message):
        return ""
    payload = _graph._payload(message)
    if payload is None:
        return "uncertain"
    structured_impacts = [
        str(result.get("workspace_impact", "")).strip().casefold()
        for result in _graph._result_mappings(payload)
        if "workspace_impact" in result
    ]
    error = str(payload.get("error", ""))
    marker_impacts = _workspace_impacts_from_error(error)
    if structured_impacts or marker_impacts:
        return _conservative_workspace_impact((*structured_impacts, *marker_impacts))
    return _workspace_impact_from_error(error)


def _stable_failure_value(value: Any) -> Any:
    """Drop transport telemetry while preserving mutation failure identity."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_failure_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold() not in _VOLATILE_FAILURE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_failure_value(item) for item in value]
    return value


def _failed_source_mutation_signature(message: Mapping[str, Any]) -> str:
    """Stable route/error identity for deterministic unchanged-workspace failures."""

    if not _is_failed_source_mutation(message):
        return ""
    payload = _graph._payload(message)
    if payload is None:
        return ""
    stable = {
        "tool": str(message.get("name", "")).strip(),
        "workspace_impact": _failed_source_mutation_workspace_impact(message),
        "error": str(payload.get("error", "")).strip(),
        "result": _stable_failure_value(payload.get("result")),
    }
    encoded = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


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
    """Advance only new observations while reusing causal-tool semantics."""

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
        if name in SOURCE_MUTATION_NAMES and not mutation_payload_applied(name, payload):
            continue
        facts.update(_graph._semantic_effects(name, payload, set(transition.effects)))
    return frozenset(facts)


def verified_state_from_messages(
    messages: Sequence[Mapping[str, Any]],
    schemas: Sequence[Mapping[str, Any]],
    *,
    require_fresh_evidence: bool = False,
) -> frozenset[str]:
    """Canonical full replay using the same mutation proof as incremental updates."""

    state = set(
        _advance_verified_state(
            frozenset({"workspace_bound"}),
            messages,
            schemas,
        )
    )
    if require_fresh_evidence and not _FRESH_EVIDENCE_FACTS.intersection(state):
        state.discard("evidence_ready")
    return frozenset(state)


def _fresh_evidence_ready(state: frozenset[str]) -> bool:
    return "evidence_ready" in state and bool(_FRESH_EVIDENCE_FACTS.intersection(state))


def supplies_fresh_evidence(
    message: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether one reviewed observation starts a usable evidence epoch."""

    state = _advance_verified_state(
        frozenset({"workspace_bound"}),
        (message,),
        schemas,
    )
    return _fresh_evidence_ready(state)


@dataclass(frozen=True)
class CausalStateSnapshot:
    state: frozenset[str]
    query: str
    replayed_full_transcript: bool
    processed_suffix_messages: int
    blocked_mutation_tools: frozenset[str] = frozenset()


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
        self._safe_failure_seen: set[str] = set()
        self._safe_failure_recovered: set[str] = set()
        self._pending_recovery_failures: set[str] = set()
        self._blocked_mutation_tools: set[str] = set()

    def _reset_refresh_window(self) -> None:
        self._refresh_required = False
        self._fresh_state = frozenset({"workspace_bound"})
        self._pending_recovery_failures.clear()

    def _reset_mutation_epoch(self) -> None:
        """Forget stuck history only after the workspace may actually have changed."""

        self._reset_refresh_window()
        self._safe_failure_seen.clear()
        self._safe_failure_recovered.clear()
        self._blocked_mutation_tools.clear()

    def _observe_refresh_policy(
        self,
        message: Mapping[str, Any],
        schemas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Track semantic mutation recovery without treating a round count as progress."""

        role = str(message.get("role", "")).strip().casefold()
        name = str(message.get("name", "")).strip()
        payload = _graph._payload(message) if role == "tool" else None
        if (
            role == "tool"
            and name in SOURCE_MUTATION_NAMES
            and payload is not None
            and mutation_payload_applied(name, payload)
        ):
            # A reviewed source diff is objective world-state progress. Only this (or
            # an uncertain/drift failure below) starts a new mutation epoch.
            self._reset_mutation_epoch()
            return

        impact = _failed_source_mutation_workspace_impact(message)
        if impact:
            if impact in _INVALIDATING_WORKSPACE_IMPACTS:
                # We cannot prove the workspace is the same, so old stuck identities
                # are no longer sound. Re-observe before another mutation.
                self._reset_mutation_epoch()
                self._refresh_required = True
                return
            if impact in _SAFE_RETRY_WORKSPACE_IMPACTS:
                signature = _failed_source_mutation_signature(message)
                if signature and signature in self._safe_failure_recovered:
                    # Same deterministic route/failure after an evidence recovery and
                    # no workspace change is a causal fixed point. Hide the route from
                    # the small model instead of asking it to invent an escape.
                    self._blocked_mutation_tools.add(name)
                    return
                if signature and signature in self._safe_failure_seen:
                    # A repeated semantic failure with no intervening recovery is not
                    # progress. Force a fresh evidence transition exactly for this
                    # failure identity rather than consuming a generic retry budget.
                    self._refresh_required = True
                    self._fresh_state = frozenset({"workspace_bound"})
                    self._pending_recovery_failures.add(signature)
                    return
                if signature:
                    self._safe_failure_seen.add(signature)
                return

        if self._refresh_required:
            self._fresh_state = _advance_verified_state(
                self._fresh_state,
                (message,),
                schemas,
            )
            if _fresh_evidence_ready(self._fresh_state):
                self._safe_failure_recovered.update(self._pending_recovery_failures)
                self._reset_refresh_window()
            return

        if supplies_fresh_evidence(message, schemas):
            # Retrieval is observation, not world-state progress. It may satisfy a
            # recovery, but must never erase the failure identity it is recovering.
            self._safe_failure_recovered.update(self._safe_failure_seen)

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
            self._observe_refresh_policy(message, schemas)

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
        self._reset_mutation_epoch()
        for message in messages:
            self._observe_refresh_policy(message, schemas)
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
            blocked_mutation_tools=frozenset(self._blocked_mutation_tools),
        )


__all__ = [
    "CausalStateLedger",
    "CausalStateSnapshot",
    "supplies_fresh_evidence",
    "verified_state_from_messages",
]
