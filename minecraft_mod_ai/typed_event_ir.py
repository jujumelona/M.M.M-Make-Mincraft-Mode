from __future__ import annotations

"""Host-owned event binding contract for Typed PlanIR."""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

EVENT_PARAMETERS: dict[str, tuple[tuple[str, str], ...]] = {
    "server_started": (("server", "object"),),
    "server_stopping": (("server", "object"),),
    "server_tick": (("server", "object"),),
    "player_join": (("player", "object"),),
    "player_disconnect": (("player", "object"),),
    "player_respawn": (
        ("oldPlayer", "object"),
        ("newPlayer", "object"),
        ("alive", "boolean"),
    ),
    "command": (("source", "object"),),
}
EVENT_SIGNATURES: dict[str, tuple[tuple[str, ...], str]] = {
    event: (
        tuple(type_name for _name, type_name in parameters),
        "int" if event == "command" else "void",
    )
    for event, parameters in EVENT_PARAMETERS.items()
}

_MOD_INITIALIZE_TRIGGER = re.compile(
    r"^(?:"
    r"mod[ _-]?(?:init|initialize|initialization)|"
    r"oninitialize|initialize|initialization|"
    r"모드[ _-]?초기화|초기화"
    r")$",
    re.IGNORECASE,
)
_COMMAND_LITERAL = re.compile(r"^[a-z0-9_]{1,64}$")


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def is_mod_initialize_trigger(value: Any) -> bool:
    normalized = re.sub(r"[\s:/\\]+", "_", _compact(value)).strip("_")
    return bool(_MOD_INITIALIZE_TRIGGER.fullmatch(normalized))


_EVENT_HINTS: dict[str, tuple[re.Pattern[str], ...]] = {
    "server_started": (
        re.compile(r"\bserver\s*(?:start|started|startup)\b", re.IGNORECASE),
        re.compile(r"\bstartup\b", re.IGNORECASE),
        re.compile(r"서버\s*시작"),
    ),
    "server_stopping": (
        re.compile(r"\bserver\s*(?:stop|stopping|shutdown)\b", re.IGNORECASE),
        re.compile(r"\bshutdown\b", re.IGNORECASE),
        re.compile(r"서버\s*(?:종료|중지)"),
    ),
    "server_tick": (
        re.compile(r"\bserver\s*tick\b", re.IGNORECASE),
        re.compile(r"\btick\b", re.IGNORECASE),
        re.compile(r"(?:서버\s*)?틱"),
    ),
    "player_join": (
        re.compile(r"\bplayer\s*(?:join|login|connect|reconnect)\b", re.IGNORECASE),
        re.compile(r"\b(?:join|login|reconnect)\b", re.IGNORECASE),
        re.compile(r"플레이어\s*(?:접속|입장|재접속)"),
    ),
    "player_disconnect": (
        re.compile(r"\bplayer\s*(?:disconnect|leave|logout)\b", re.IGNORECASE),
        re.compile(r"\b(?:disconnect|logout)\b", re.IGNORECASE),
        re.compile(r"플레이어\s*(?:퇴장|접속\s*종료|로그아웃)"),
    ),
    "player_respawn": (
        re.compile(r"\bplayer\s*respawn\b", re.IGNORECASE),
        re.compile(r"\brespawn\b", re.IGNORECASE),
        re.compile(r"플레이어\s*리스폰|부활"),
    ),
    "command": (
        re.compile(r"\bcommand\b", re.IGNORECASE),
        re.compile(r"명령어|커맨드"),
        re.compile(r"^/"),
    ),
}


def infer_event_type(value: Any) -> str | None:
    """Return a host-owned event only when the trigger text is unambiguous."""

    text = _compact(value)
    if not text or is_mod_initialize_trigger(text):
        return None
    matches = [
        event
        for event, patterns in _EVENT_HINTS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
    return matches[0] if len(matches) == 1 else None


def infer_event_config(event: str, trigger: Any) -> dict[str, Any] | None:
    if event != "command":
        return {}
    text = _compact(trigger)
    match = re.search(
        r"(?:^|\\s)/(?:\\s*)?([a-z0-9_]{1,64})(?:\\b|$)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\\b(?:command|명령어|커맨드)\\s*[:=]?\\s*([a-z0-9_]{1,64})\\b",
            text,
            re.IGNORECASE,
        )
    if match is None:
        return None
    return {
        "literal": match.group(1).lower(),
        "permission_level": 0,
    }


def event_config_schema(event: str) -> dict[str, Any]:
    if event == "command":
        return {
            "type": "object",
            "properties": {
                "literal": {
                    "type": "string",
                    "pattern": r"^[a-z0-9_]{1,64}$",
                },
                "permission_level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
            },
            "required": ["literal"],
            "additionalProperties": False,
        }
    if event in EVENT_SIGNATURES:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    raise ValueError(f"TYPED_EVENT_UNSUPPORTED: {event!r}")


def validate_event_bindings(
    raw_bindings: Any,
    *,
    signatures: Mapping[str, tuple[tuple[str, ...], str]],
) -> list[dict[str, Any]]:
    if raw_bindings is None:
        return []
    if not isinstance(raw_bindings, Sequence) or isinstance(
        raw_bindings,
        (str, bytes, bytearray),
    ):
        raise ValueError("typed_plan_ir.event_bindings: expected array")

    result: list[dict[str, Any]] = []
    seen_entry_points: set[int] = set()
    for index, raw in enumerate(raw_bindings):
        where = f"typed_plan_ir.event_bindings[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: expected object")
        if set(raw) != {
            "event",
            "function",
            "entry_point_index",
            "config",
        }:
            raise ValueError(f"{where}: invalid fields")

        event = str(raw["event"])
        expected = EVENT_SIGNATURES.get(event)
        if expected is None:
            raise ValueError(f"{where}: unsupported event {event!r}")

        function = str(raw["function"])
        actual = signatures.get(function)
        if actual is None:
            raise ValueError(f"{where}: unknown function {function!r}")
        if actual != expected:
            raise ValueError(
                f"{where}: function {function!r} has signature {actual!r}, "
                f"expected {expected!r} for {event!r}"
            )

        entry_point_index = raw["entry_point_index"]
        if type(entry_point_index) is not int or entry_point_index < 0:
            raise ValueError(
                f"{where}.entry_point_index: expected nonnegative integer"
            )
        if entry_point_index in seen_entry_points:
            raise ValueError(
                f"{where}: duplicate entry_point_index {entry_point_index}"
            )
        seen_entry_points.add(entry_point_index)

        config = raw["config"]
        if not isinstance(config, Mapping):
            raise ValueError(f"{where}.config: expected object")
        config = deepcopy(dict(config))
        if event == "command":
            if set(config) - {"literal", "permission_level"}:
                raise ValueError(f"{where}.config: invalid command fields")
            literal = str(config.get("literal") or "")
            if not _COMMAND_LITERAL.fullmatch(literal):
                raise ValueError(f"{where}.config.literal: invalid command literal")
            permission = config.get("permission_level", 0)
            if type(permission) is not int or not 0 <= permission <= 4:
                raise ValueError(
                    f"{where}.config.permission_level: expected integer 0-4"
                )
            config["permission_level"] = permission
        elif config:
            raise ValueError(f"{where}.config: event {event!r} takes no config")

        result.append({
            "event": event,
            "function": function,
            "entry_point_index": entry_point_index,
            "config": config,
        })
    return result


__all__ = [
    "EVENT_PARAMETERS",
    "EVENT_SIGNATURES",
    "event_config_schema",
    "infer_event_config",
    "infer_event_type",
    "is_mod_initialize_trigger",
    "validate_event_bindings",
]
