from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CONTROL = re.compile(r"\b(if|else|for|while|switch|try|catch|finally|return|throw)\b")
_REGISTER = re.compile(r"\b(register\w*|registry\w*|bootstrap\w*|initialize\w*)\b", re.IGNORECASE)
_PERSIST = re.compile(r"\b(load\w*|save\w*|persist\w*|serialize\w*|deserialize\w*|codec\w*)\b", re.IGNORECASE)
_NETWORK = re.compile(r"\b(send\w*|receive\w*|packet\w*|payload\w*|encode\w*|decode\w*|sync\w*)\b", re.IGNORECASE)
_EVENT = re.compile(r"\b(event\w*|callback\w*|listener\w*|tick\w*|subscribe\w*)\b", re.IGNORECASE)
_VERIFY = re.compile(r"\b(assert\w*|verify\w*|check\w*|validate\w*|test\w*)\b", re.IGNORECASE)
_IGNORE_CALLS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "new", "super", "this",
    "println", "print", "format", "valueof", "tostring", "hashcode", "equals",
})


@dataclass(frozen=True)
class ProcedurePlan:
    steps: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/procedure-plan-v1",
            "source": self.source,
            "step_count": len(self.steps),
            "steps": list(self.steps),
        }


def decompose_task_procedure(query: str) -> ProcedurePlan:
    """Decompose one repository task into ordered implementation operations.

    This is deliberately host-owned and deterministic.  It extracts task semantics
    from the approved request and never turns retrieved examples into authority.
    """
    raw = str(query or "").strip()
    payload = _json_object(raw)
    material = raw
    if payload is not None:
        material = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    lowered = material.casefold()
    steps: list[str] = ["locate existing contract"]

    if any(word in lowered for word in ("network", "packet", "payload", "sync", "client", "server")):
        steps.extend(
            [
                "locate packet or networking contract",
                "preserve server authoritative state",
                "encode and decode payload",
                "register networking handler",
                "apply synchronized behavior",
            ]
        )
    if any(word in lowered for word in ("persist", "save", "load", "state", "component", "nbt", "codec")):
        steps.extend(
            [
                "locate persistence contract",
                "load existing state",
                "apply state mutation",
                "serialize or save state",
            ]
        )
    if any(word in lowered for word in ("register", "registry", "item", "block", "entity", "recipe")):
        steps.extend(
            [
                "locate registry convention",
                "construct target value",
                "register stable identifier",
                "bind required resources",
            ]
        )
    if any(word in lowered for word in ("event", "callback", "listener", "tick", "lifecycle")):
        steps.extend(
            [
                "locate lifecycle hook",
                "register callback",
                "guard side effects",
                "execute event behavior",
            ]
        )
    if any(word in lowered for word in ("error", "failed", "failure", "cannot find", "diagnostic", "gradle", "gametest")):
        steps.extend(
            [
                "localize failing symbol or region",
                "preserve unaffected behavior",
                "apply minimal compatible correction",
            ]
        )

    task_tokens = _meaningful_tokens(material)
    action_tokens = [
        token for token in task_tokens
        if token.startswith(
            (
                "register", "load", "save", "sync", "spawn", "create", "build", "generate",
                "update", "apply", "validate", "serialize", "deserialize", "send", "receive",
            )
        )
    ][:8]
    for token in action_tokens:
        steps.append(f"perform {token}")
    steps.append("verify observable contract")
    return ProcedurePlan(tuple(dict.fromkeys(steps)), "host_deterministic_task_decomposition")


def extract_code_procedure(text: str) -> tuple[str, ...]:
    """Extract an ordered procedural trace from source without identifier matching."""
    source = str(text or "")
    events: list[tuple[int, str]] = []
    for match in _REGISTER.finditer(source):
        events.append((match.start(), "register contract"))
    for match in _PERSIST.finditer(source):
        value = match.group(1).casefold()
        if value.startswith(("load", "deserialize")):
            label = "load state"
        elif value.startswith(("save", "persist", "serialize")):
            label = "save state"
        else:
            label = "state codec"
        events.append((match.start(), label))
    for match in _NETWORK.finditer(source):
        value = match.group(1).casefold()
        if value.startswith(("encode", "decode")):
            label = "encode decode payload"
        elif value.startswith(("send", "receive", "sync")):
            label = "synchronize payload"
        else:
            label = "network packet"
        events.append((match.start(), label))
    for match in _EVENT.finditer(source):
        events.append((match.start(), "event callback"))
    for match in _CONTROL.finditer(source):
        events.append((match.start(), f"control {match.group(1).casefold()}"))
    for match in _VERIFY.finditer(source):
        events.append((match.start(), "verify contract"))
    for match in _CALL.finditer(source):
        name = match.group(1).casefold()
        if name in _IGNORE_CALLS:
            continue
        events.append((match.start(), f"call {_verb_shape(name)}"))
    events.sort(key=lambda item: item[0])
    return tuple(dict.fromkeys(label for _offset, label in events))[:48]


def procedure_similarity(plan: Sequence[str], code_steps: Sequence[str]) -> float:
    """Order-aware procedure similarity using semantic step tokens and LCS alignment."""
    target = tuple(_step_signature(step) for step in plan if str(step).strip())
    observed = tuple(_step_signature(step) for step in code_steps if str(step).strip())
    if not target or not observed:
        return 0.0
    scores = [[_token_jaccard(left, right) for right in observed] for left in target]
    # Weighted LCS: only ordered step matches above a conservative semantic threshold count.
    rows, cols = len(target), len(observed)
    dp = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            pair = scores[i - 1][j - 1]
            match = dp[i - 1][j - 1] + pair if pair >= 0.20 else -1.0
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], match)
    ordered = dp[rows][cols] / max(1.0, float(rows))
    coverage = sum(max(row) >= 0.20 for row in scores) / max(1.0, float(rows))
    return max(0.0, min(1.0, 0.72 * ordered + 0.28 * coverage))


def procedural_region_score(plan: ProcedurePlan, text: str) -> tuple[float, tuple[str, ...]]:
    observed = extract_code_procedure(text)
    return procedure_similarity(plan.steps, observed), observed


def _json_object(raw: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _meaningful_tokens(value: str) -> list[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "into", "true", "false"}
    return [
        token.casefold() for token in _TOKEN.findall(value)
        if token.casefold() not in stop and len(token) >= 3
    ]


def _verb_shape(value: str) -> str:
    token = str(value).casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _step_signature(value: str) -> frozenset[str]:
    aliases = {
        "registry": "register",
        "registration": "register",
        "handler": "callback",
        "listener": "callback",
        "persist": "save",
        "serialize": "save",
        "deserialize": "load",
        "synchronize": "sync",
        "synchronized": "sync",
        "networking": "network",
        "packet": "payload",
        "identifier": "id",
        "correction": "fix",
        "failing": "fail",
    }
    result: set[str] = set()
    for token in _meaningful_tokens(value):
        shaped = _verb_shape(token)
        result.add(aliases.get(shaped, shaped))
    return frozenset(result)


def _token_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


__all__ = [
    "ProcedurePlan",
    "decompose_task_procedure",
    "extract_code_procedure",
    "procedural_region_score",
    "procedure_similarity",
]
