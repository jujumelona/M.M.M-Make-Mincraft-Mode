from __future__ import annotations

"""Pure retrieval progress primitives shared by the host execution loop."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .model_adapters import ModelConfigurationError

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "with",
})
_VOLATILE_EVIDENCE_KEYS = frozenset({
    "coverage_score", "correction", "elapsed_ms", "generated_at", "latency_ms",
    "normalized_query", "query", "relevance_score", "request_id", "result_count",
    "timestamp", "trace_id",
})


class RetrievalState(Protocol):
    attempted_queries: set[str]
    attempted_sources: set[str]
    evidence_fingerprints: set[str]
    has_fresh_evidence: bool

    def record_query(self, tool_name: str, arguments: Mapping[str, Any]) -> bool: ...
    def record_evidence(self, value: Any, *, usable: bool = True) -> bool: ...
    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
    ) -> str | None: ...


def normalize_retrieval_query(value: Any) -> str:
    text = str(value or "").casefold()
    tokens = re.findall(r"[a-z0-9_.$:/+-]+", text)
    return " ".join(sorted({token for token in tokens if token not in _STOPWORDS}))


def retrieval_source_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    if name in {"external_mcp_schema", "external_mcp_call"}:
        capability = str(arguments.get("capability", "")).strip()
        return f"{name}:{capability}" if capability else name
    return name


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    parts = [name]
    if name in {"external_mcp_schema", "external_mcp_call"}:
        capability = str(arguments.get("capability", "")).strip().casefold()
        if capability:
            parts.append(f"capability={capability}")
    query = normalize_retrieval_query(arguments.get("query"))
    for key in ("index_path", "path", "file", "target_path", "symbol", "symbol_name"):
        value = str(arguments.get(key) or "").strip().casefold()
        if value:
            label = "target" if key in {"index_path", "path", "file", "target_path"} else key
            parts.append(f"{label}={value}")
    if query:
        parts.append(f"q={query}")
    cursor = arguments.get("cursor") or arguments.get("offset_bytes")
    if cursor not in (None, "", 0, "0"):
        parts.append(f"cursor={cursor}")
    return ":".join(parts)


def _stable_value(value: Any, *, drop_volatile: bool = True) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if drop_volatile and key.casefold() in _VOLATILE_EVIDENCE_KEYS:
                continue
            stable = _stable_value(child, drop_volatile=drop_volatile)
            if stable not in (None, "", [], {}):
                out[key] = stable
        return out
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item, drop_volatile=drop_volatile) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item, drop_volatile=drop_volatile) for item in value]
    return value


def evidence_fingerprint(value: Any) -> str | None:
    stable = _stable_value(value)
    if stable in (None, "", [], {}):
        return None
    canonical = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RetrievalNoProgressError(ModelConfigurationError):
    pass


class RetrievalProgress:
    def __init__(
        self,
        state: RetrievalState,
        *,
        execute_decision: Any,
        duplicate_decision: Any,
        fresh_observation: Any,
        duplicate_observation: Any,
        weak_observation: Any,
        no_progress_limit: int | None = None,
    ) -> None:
        self._state = state
        self._execute_decision = execute_decision
        self._duplicate_decision = duplicate_decision
        self._fresh_observation = fresh_observation
        self._duplicate_observation = duplicate_observation
        self._weak_observation = weak_observation
        self.attempted_queries = state.attempted_queries
        self.attempted_sources = state.attempted_sources
        self.evidence_fingerprints = state.evidence_fingerprints
        self.no_progress_observations = 0
        self._no_progress_limit = no_progress_limit

    def begin(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        return (
            self._execute_decision
            if self._state.record_query(tool_name, arguments)
            else self._duplicate_decision
        )

    def observe(self, *args: Any, usable: bool = True, **kwargs: Any) -> Any:
        value = args[2] if len(args) >= 3 else (args[0] if args else kwargs.get("value"))
        if not usable:
            self.no_progress_observations += 1
            if (
                self._no_progress_limit is not None
                and self.no_progress_observations >= self._no_progress_limit
            ):
                raise RetrievalNoProgressError("no novel usable evidence")
            return self._weak_observation
        if self._state.record_evidence(value, usable=True):
            self.no_progress_observations = 0
            return self._fresh_observation
        return self._duplicate_observation

    @property
    def has_fresh_evidence(self) -> bool:
        return self._state.has_fresh_evidence

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
    ) -> str | None:
        return self._state.next_untried_internal_tool(exposed_tools, preferred=preferred)
