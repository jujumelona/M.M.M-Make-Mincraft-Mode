from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping, Sequence

_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.@+-]+)(?![A-Za-z0-9_])")
_SHA = re.compile(r"(?:sha256:)?\b[0-9a-fA-F]{64}\b")
_RESOURCE = re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b")
_VERSION = re.compile(r"\b(?:\d+\.){1,3}\d+(?:[-+._][A-Za-z0-9]+)*\b")


def _ledger(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rendered = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, default=str)
    observations: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role", "")) != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(str(message["content"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        item: dict[str, Any] = {
            "tool": value.get("tool") or message.get("name"),
            "ok": value.get("ok"),
        }
        if value.get("error") is not None:
            item["error"] = str(value.get("error"))[:1200]
        result = value.get("result")
        if isinstance(result, Mapping):
            receipt = result.get("receipt")
            if isinstance(receipt, Mapping):
                item["receipt"] = {
                    key: receipt.get(key)
                    for key in (
                        "route",
                        "result_count",
                        "coverage_score",
                        "relevance_score",
                        "relation_expansions",
                        "warnings",
                    )
                    if receipt.get(key) is not None
                }
        observations.append(item)
    return {
        "schema_version": "mmm/agent-context-compaction-v1",
        "dropped_transcript_sha256": "sha256:" + hashlib.sha256(rendered.encode()).hexdigest(),
        "paths": sorted(set(_PATH.findall(rendered)))[:256],
        "sha256": sorted({value.casefold() for value in _SHA.findall(rendered)})[:256],
        "resource_ids": sorted(set(_RESOURCE.findall(rendered.casefold())))[:256],
        "versions": sorted(set(_VERSION.findall(rendered)))[:128],
        "verified_observations": observations[-32:],
    }


def compact_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    original = tuple(messages)
    try:
        budget = int(os.environ.get("MMM_SMALL_AGENT_CONTEXT_BYTES", 96 * 1024))
    except ValueError:
        budget = 96 * 1024
    budget = max(24 * 1024, min(512 * 1024, budget))
    if len(json.dumps(original, ensure_ascii=False, sort_keys=True, default=str).encode()) <= budget:
        return original
    assistants = [i for i, item in enumerate(original) if str(item.get("role", "")) == "assistant"]
    if len(assistants) < 2:
        return original
    first = assistants[0]
    for keep in (3, 2, 1):
        if len(assistants) <= keep:
            continue
        start = assistants[-keep]
        dropped = original[first:start]
        if not dropped:
            continue
        context = {
            "role": "system",
            "content": "HOST COMPACTED VERIFIED CONTEXT. Exact facts/tool outcomes are authoritative; omitted prose is not.\n"
            + json.dumps(_ledger(dropped), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        }
        compacted: tuple[Mapping[str, Any], ...] = (*original[:first], context, *original[start:])
        if len(json.dumps(compacted, ensure_ascii=False, sort_keys=True, default=str).encode()) <= budget:
            return compacted
    return original


__all__ = ["compact_messages"]
