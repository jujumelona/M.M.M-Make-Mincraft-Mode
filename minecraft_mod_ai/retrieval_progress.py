from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class RetrievalDecision(str, Enum):
    EXECUTE = "execute"
    DUPLICATE_QUERY = "duplicate_query"


class RetrievalObservation(str, Enum):
    FRESH = "fresh"
    WEAK = "weak"
    DUPLICATE_EVIDENCE = "duplicate_evidence"


class RetrievalNoProgressError(RuntimeError):
    """Raised when repeated successful retrievals still add no usable evidence."""


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
_DEFAULT_NO_PROGRESS_LIMIT = 8


def normalize_retrieval_query(value: Any) -> str:
    """Canonicalize retrieval intent without relying on incidental model phrasing."""
    text = str(value or "").casefold()
    tokens = re.findall(r"[a-z0-9_.$:/+-]+", text)
    material = sorted({token for token in tokens if token not in _STOPWORDS})
    return " ".join(material)


def retrieval_source_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    if name == "external_mcp_call":
        capability = str(arguments.get("capability", "")).strip()
        return f"{name}:{capability}" if capability else name
    return name


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    source = retrieval_source_key(tool_name, arguments)
    data = dict(arguments)
    query = normalize_retrieval_query(data.pop("query", ""))
    if tool_name == "external_mcp_call":
        nested = data.get("arguments")
        if isinstance(nested, Mapping):
            nested_data = dict(nested)
            nested_query = normalize_retrieval_query(nested_data.pop("query", ""))
            if nested_query:
                query = nested_query
            data["arguments"] = nested_data
    canonical = json.dumps(
        {"source": source, "query": query, "scope": _stable_value(data, drop_volatile=False)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_fingerprint(value: Any) -> str | None:
    stable = _stable_value(value, drop_volatile=True)
    if stable in (None, "", [], {}):
        return None
    canonical = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_value(value: Any, *, drop_volatile: bool) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if drop_volatile and key.casefold() in _VOLATILE_EVIDENCE_KEYS:
                continue
            stable_child = _stable_value(child, drop_volatile=drop_volatile)
            if stable_child not in (None, "", [], {}):
                result[key] = stable_child
        return result
    if isinstance(value, (set, frozenset)):
        # Tool/runtime results are normally JSON-like, but host-side helpers can
        # legitimately return set-valued facts. Serializing a raw set through
        # ``default=str`` makes fingerprints depend on hash iteration order. Convert
        # unordered containers into a canonically sorted JSON-compatible sequence.
        items = [_stable_value(item, drop_volatile=drop_volatile) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item, drop_volatile=drop_volatile) for item in value]
    return value


@dataclass
class RetrievalProgress:
    """Host-owned novelty state for one retrieve/act/generate episode.

    Retrieval progression is driven by novel queries and novel evidence. Duplicate
    evidence is local to the current observation and must never blacklist an entire
    source, because a later distinct query can still produce new evidence. A bounded
    no-progress streak prevents arbitrarily many distinct weak queries from growing
    the live model transcript until it reaches the physical context boundary.
    """

    attempted_queries: set[str] = field(default_factory=set)
    attempted_sources: set[str] = field(default_factory=set)
    evidence_fingerprints: set[str] = field(default_factory=set)
    no_progress_limit: int = _DEFAULT_NO_PROGRESS_LIMIT
    no_progress_observations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin(self, tool_name: str, arguments: Mapping[str, Any]) -> RetrievalDecision:
        source = retrieval_source_key(tool_name, arguments)
        signature = retrieval_query_signature(tool_name, arguments)
        with self._lock:
            if signature in self.attempted_queries:
                return RetrievalDecision.DUPLICATE_QUERY
            self.attempted_queries.add(signature)
            self.attempted_sources.add(source)
            return RetrievalDecision.EXECUTE

    def observe(self, value: Any, *, usable: bool) -> RetrievalObservation:
        if not usable:
            return self._record_no_progress(RetrievalObservation.WEAK)
        fingerprint = evidence_fingerprint(value)
        if fingerprint is None:
            return self._record_no_progress(RetrievalObservation.WEAK)
        with self._lock:
            if fingerprint in self.evidence_fingerprints:
                observation = RetrievalObservation.DUPLICATE_EVIDENCE
            else:
                self.evidence_fingerprints.add(fingerprint)
                self.no_progress_observations = 0
                return RetrievalObservation.FRESH
        return self._record_no_progress(observation)

    def _record_no_progress(
        self,
        observation: RetrievalObservation,
    ) -> RetrievalObservation:
        with self._lock:
            self.no_progress_observations += 1
            count = self.no_progress_observations
            limit = max(1, int(self.no_progress_limit))
        if count >= limit:
            raise RetrievalNoProgressError(
                "retrieval produced no novel usable evidence for "
                f"{count} consecutive successful observations"
            )
        return observation

    @property
    def has_fresh_evidence(self) -> bool:
        with self._lock:
            return bool(self.evidence_fingerprints)

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
    ) -> str | None:
        exposed = set(exposed_tools)
        with self._lock:
            for name in preferred:
                if name in exposed and name not in self.attempted_sources:
                    return name
        return None
