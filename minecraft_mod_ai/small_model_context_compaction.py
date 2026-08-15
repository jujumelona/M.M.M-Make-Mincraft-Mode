from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.@+-]+)(?![A-Za-z0-9_])")
_SHA = re.compile(r"(?:sha256:)?\b[0-9a-fA-F]{64}\b")
_RESOURCE = re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b")
_VERSION = re.compile(r"\b(?:\d+\.){1,3}\d+(?:[-+._][A-Za-z0-9]+)*\b")
_VERIFICATION_KEYS = frozenset(
    {
        "status",
        "passed",
        "overall_status",
        "jdt_status",
        "jdt_error_count",
        "build_status",
        "result_count",
        "coverage_score",
        "relevance_score",
        "assertion_count",
        "interaction_count",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _archive_root() -> Path:
    configured = os.environ.get("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    workspace = os.environ.get("MMM_WORKSPACE", "mmm-output").strip() or "mmm-output"
    return Path(workspace).expanduser().resolve() / ".minecraft_ai" / "context-memory"


def _archive_preview(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = _canonical_bytes(list(messages))
    digest = _sha256_bytes(payload)
    target = _archive_root() / f"{digest.removeprefix('sha256:')}.json"
    return {
        "available": True,
        "sha256": digest,
        "bytes": len(payload),
        "path": str(target),
        "format": "canonical-json",
    }


def _archive_transcript(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Persist exact dropped history and return a content-addressed recovery pointer."""

    payload = _canonical_bytes(list(messages))
    digest = _sha256_bytes(payload)
    root = _archive_root()
    target = root / f"{digest.removeprefix('sha256:')}.json"
    try:
        if root.exists() and root.is_symlink():
            raise OSError("context archive root is a symlink")
        root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.is_symlink():
                raise OSError("context archive target is not a regular file")
            if target.read_bytes() != payload:
                raise OSError("context archive hash collision or corrupted payload")
        else:
            fd, temporary_name = tempfile.mkstemp(prefix=".context-", suffix=".json", dir=root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return {
            "available": True,
            "sha256": digest,
            "bytes": len(payload),
            "path": str(target),
            "format": "canonical-json",
        }
    except OSError as exc:
        # Compaction must not crash a model turn solely because durable archive I/O is
        # unavailable. In that case callers keep the uncompressed transcript instead.
        return {
            "available": False,
            "sha256": digest,
            "bytes": len(payload),
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def _small_verification_facts(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 4 or not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        lowered = key.casefold()
        if lowered in _VERIFICATION_KEYS and (
            raw_value is None or isinstance(raw_value, (str, int, float, bool))
        ):
            result[key] = raw_value
        elif isinstance(raw_value, Mapping):
            nested = _small_verification_facts(raw_value, depth=depth + 1)
            if nested:
                result[key] = nested
    return result


def _remaining_facts(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 4 or not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        lowered = key.casefold()
        if lowered.startswith("remaining") and (
            raw_value is None or isinstance(raw_value, (str, int, float, bool))
        ):
            result[key] = raw_value
        elif isinstance(raw_value, Mapping):
            nested = _remaining_facts(raw_value, depth=depth + 1)
            if nested:
                result[key] = nested
    return result


def _ledger(
    messages: Sequence[Mapping[str, Any]],
    *,
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    rendered = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, default=str)
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", ""))
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
                for raw_call in raw_calls:
                    if not isinstance(raw_call, Mapping):
                        continue
                    function = raw_call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    arguments = str(function.get("arguments", ""))
                    actions.append(
                        {
                            "tool": str(function.get("name", "")),
                            "arguments_sha256": _sha256_bytes(arguments.encode("utf-8")),
                        }
                    )
            continue
        if role != "tool" or not isinstance(message.get("content"), str):
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
            error = str(value.get("error"))[:1200]
            item["error"] = error
            errors.append({"tool": item["tool"], "error": error})
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
            verified = _small_verification_facts(result)
            if verified:
                verifications.append({"tool": item["tool"], "facts": verified})
            remaining_facts = _remaining_facts(result)
            if remaining_facts:
                remaining.append({"tool": item["tool"], "facts": remaining_facts})
        observations.append(item)

    paths = sorted({value for value in _PATH.findall(rendered) if len(value) <= 512})[:256]
    resources = sorted({value for value in _RESOURCE.findall(rendered.casefold()) if len(value) <= 256})[:256]
    return {
        "schema_version": "mmm/agent-context-compaction-v2",
        "raw_history": dict(archive),
        "paths": paths,
        "sha256": sorted({value.casefold() for value in _SHA.findall(rendered)})[:256],
        "resource_ids": resources,
        "versions": sorted(set(_VERSION.findall(rendered)))[:128],
        "tool_actions": actions[-32:],
        "errors": errors[-32:],
        "verifications": verifications[-32:],
        "remaining_host_state": remaining[-32:],
        "verified_observations": observations[-32:],
        "policy": {
            "raw_history_recoverable": bool(archive.get("available")),
            "model_prose_is_not_promoted_to_verified_fact": True,
            "compaction_is_adaptive_to_byte_budget": True,
        },
    }


def compact_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    original = tuple(messages)
    try:
        budget = int(os.environ.get("MMM_SMALL_AGENT_CONTEXT_BYTES", 96 * 1024))
    except ValueError:
        budget = 96 * 1024
    budget = max(24 * 1024, min(512 * 1024, budget))
    if len(_canonical_bytes(original)) <= budget:
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
        archive = _archive_preview(dropped)
        context = {
            "role": "system",
            "content": "HOST COMPACTED VERIFIED CONTEXT. Exact facts/tool outcomes are authoritative; omitted prose is recoverable from raw_history and is not itself verified.\n"
            + json.dumps(
                _ledger(dropped, archive=archive),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        compacted: tuple[Mapping[str, Any], ...] = (*original[:first], context, *original[start:])
        if len(_canonical_bytes(compacted)) > budget:
            continue
        persisted_archive = _archive_transcript(dropped)
        if not persisted_archive.get("available"):
            # The research contract is lossless: if raw history cannot be recovered,
            # skip compaction instead of silently replacing it with only a hash.
            return original
        if persisted_archive != archive:
            context = {
                "role": "system",
                "content": "HOST COMPACTED VERIFIED CONTEXT. Exact facts/tool outcomes are authoritative; omitted prose is recoverable from raw_history and is not itself verified.\n"
                + json.dumps(
                    _ledger(dropped, archive=persisted_archive),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            compacted = (*original[:first], context, *original[start:])
            if len(_canonical_bytes(compacted)) > budget:
                return original
        return compacted
    return original


__all__ = ["compact_messages"]
