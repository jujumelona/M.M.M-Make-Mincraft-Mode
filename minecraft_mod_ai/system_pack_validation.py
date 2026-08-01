from __future__ import annotations

import math
import re
from typing import Any

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_PACK_KINDS = {
    "quest-system": {"quest"},
    "class-skill-system": {"class", "skill"},
    "economy-shop": {"economy", "shop"},
    "gui-networking": {"gui", "networking"},
    "party-guild": {"party", "guild"},
}
_QUEST_OBJECTIVES = {"kill", "break", "manual"}
_ACTION_TYPES = {"message", "grant_item", "status_effect"}


def validate_system_modules(pack_id: str, modules: list[Any]) -> None:
    if pack_id not in _PACK_KINDS:
        raise ValueError(f"Unknown system pack: {pack_id}")
    expected_kinds = _PACK_KINDS[pack_id]
    seen_modules: set[str] = set()
    typed: list[tuple[str, str, dict[str, Any]]] = []

    for item in modules:
        if not isinstance(item, dict):
            raise ValueError("System module must be an object.")
        allowed_fields = {
            "module_id",
            "kind",
            "config",
            "depends_on",
            "required_gates",
        }
        if set(item) != allowed_fields:
            raise ValueError(
                f"System module fields are invalid: {sorted(set(item))}"
            )
        module_id = str(item["module_id"])
        if not _ID.fullmatch(module_id) or module_id in seen_modules:
            raise ValueError(
                f"Invalid or duplicate system module id: {module_id!r}"
            )
        seen_modules.add(module_id)
        kind = str(item["kind"])
        if kind not in expected_kinds:
            raise ValueError(
                f"System pack {pack_id} cannot contain kind {kind!r}."
            )
        config = item["config"]
        if not isinstance(config, dict):
            raise ValueError(
                f"System module config must be an object: {module_id}"
            )
        if config.get("implementation") == "custom":
            raise ValueError(
                f"Custom module {module_id} must not be sent to built-in system pack."
            )
        typed.append((module_id, kind, config))

        if kind == "quest":
            _validate_quest(module_id, config)
        elif kind == "class":
            _validate_class(module_id, config)
        elif kind == "skill":
            _validate_skill(module_id, config)
        elif kind == "economy":
            _validate_economy(module_id, config)
        elif kind == "shop":
            _validate_shop(module_id, config)
        elif kind == "gui":
            _validate_gui(module_id, config)
        elif kind == "networking":
            _validate_networking(module_id, config)
        elif kind in {"party", "guild"}:
            _validate_social(module_id, config)

    _validate_cross_module_semantics(pack_id, typed)


def _validate_cross_module_semantics(
    pack_id: str,
    modules: list[tuple[str, str, dict[str, Any]]],
) -> None:
    if pack_id == "class-skill-system":
        class_ids = {module_id for module_id, kind, _ in modules if kind == "class"}
        for module_id, kind, config in modules:
            if kind != "skill":
                continue
            required = str(config.get("required_class", ""))
            if required and required not in class_ids:
                raise ValueError(
                    f"Skill {module_id} references missing class {required!r}."
                )

    elif pack_id == "economy-shop":
        economies = [module_id for module_id, kind, _ in modules if kind == "economy"]
        if len(economies) > 1:
            raise ValueError(
                "Built-in economy-shop provides one server-authoritative currency "
                "manager with any number of accounts, shops, and catalog entries. "
                "Route multiple independent currencies to custom_java so every "
                "currency has an explicit instance namespace."
            )
        entries: set[str] = set()
        for module_id, kind, config in modules:
            if kind != "shop":
                continue
            for entry in config["entries"]:
                entry_id = str(entry["id"])
                if entry_id in entries:
                    raise ValueError(
                        f"Shop entry ID is duplicated across catalogs: {entry_id!r}."
                    )
                entries.add(entry_id)

    elif pack_id == "gui-networking":
        actions: set[str] = set()
        for module_id, kind, config in modules:
            if kind != "networking":
                continue
            for action in config["actions"]:
                action_id = str(action["id"])
                if action_id in actions:
                    raise ValueError(
                        f"Network action ID is duplicated across channels: {action_id!r}."
                    )
                actions.add(action_id)

    elif pack_id == "party-guild":
        for kind in ("party", "guild"):
            matching = [module_id for module_id, current, _ in modules if current == kind]
            if len(matching) > 1:
                raise ValueError(
                    f"Built-in {kind} provides one manager that can create any "
                    f"number of runtime {kind} groups. Route multiple independent "
                    f"{kind} managers to custom_java so each manager has an "
                    "explicit instance namespace."
                )


def _validate_quest(module_id: str, config: dict[str, Any]) -> None:
    allowed = {
        "objective",
        "target",
        "required",
        "reward_item",
        "reward_count",
        "reward_currency",
    }
    _reject_unknown(module_id, config, allowed)
    objective = str(config.get("objective", "manual"))
    if objective not in _QUEST_OBJECTIVES:
        raise ValueError(
            f"Quest {module_id} objective {objective!r} is not built in; use custom_java."
        )
    target = str(config.get("target", module_id))
    if objective in {"kill", "break"} and not _RESOURCE_ID.fullmatch(target):
        raise ValueError(
            f"Quest {module_id} requires a namespaced target for {objective}."
        )
    if objective == "manual" and "target" in config and target != module_id:
        raise ValueError(
            f"Manual quest {module_id} target is its quest ID and may not be overridden."
        )
    _positive_int(config.get("required", 1), f"{module_id}.required")
    reward_item = str(config.get("reward_item", ""))
    if reward_item and not _RESOURCE_ID.fullmatch(reward_item):
        raise ValueError(
            f"Quest {module_id} reward_item must be namespaced."
        )
    _positive_int(
        config.get("reward_count", 1),
        f"{module_id}.reward_count",
    )
    reward_currency = _finite_number(
        config.get("reward_currency", 0.0),
        f"{module_id}.reward_currency",
    )
    if reward_currency < 0:
        raise ValueError(
            f"Quest {module_id} reward_currency must be nonnegative."
        )


def _validate_class(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(module_id, config, {"display_name"})
    display = str(config.get("display_name", module_id)).strip()
    if not display:
        raise ValueError(f"Class {module_id} display_name is empty.")


def _validate_skill(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(
        module_id,
        config,
        {
            "required_class",
            "effect",
            "duration_ticks",
            "amplifier",
            "cooldown_ticks",
        },
    )
    effect = str(config.get("effect", "minecraft:speed"))
    if not _RESOURCE_ID.fullmatch(effect):
        raise ValueError(f"Skill {module_id} effect must be namespaced.")
    required_class = str(config.get("required_class", ""))
    if required_class and not _ID.fullmatch(required_class):
        raise ValueError(
            f"Skill {module_id} required_class is invalid."
        )
    _positive_int(
        config.get("duration_ticks", 100),
        f"{module_id}.duration_ticks",
    )
    amplifier = _nonnegative_int(
        config.get("amplifier", 0),
        f"{module_id}.amplifier",
    )
    if amplifier > 255:
        raise ValueError(f"{module_id}.amplifier must be 0-255.")
    _positive_int(
        config.get("cooldown_ticks", 100),
        f"{module_id}.cooldown_ticks",
    )


def _validate_economy(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(module_id, config, {"initial_balance"})
    value = _finite_number(
        config.get("initial_balance", 0.0),
        f"{module_id}.initial_balance",
    )
    if value < 0:
        raise ValueError(
            f"Economy {module_id} initial_balance must be nonnegative."
        )


def _validate_shop(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(module_id, config, {"entries"})
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Shop {module_id} requires a non-empty entries list."
        )
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "id",
            "item",
            "count",
            "price",
        }:
            raise ValueError(
                f"Shop {module_id} contains an invalid entry."
            )
        if "id" not in entry or "item" not in entry or "price" not in entry:
            raise ValueError(
                f"Shop {module_id} entry requires id, item and price."
            )
        entry_id = str(entry["id"])
        if not _ID.fullmatch(entry_id) or entry_id in seen:
            raise ValueError(
                f"Shop {module_id} has invalid or duplicate entry {entry_id!r}."
            )
        seen.add(entry_id)
        if not _RESOURCE_ID.fullmatch(str(entry["item"])):
            raise ValueError(
                f"Shop {module_id}/{entry_id} item must be namespaced."
            )
        _positive_int(
            entry.get("count", 1),
            f"{module_id}.{entry_id}.count",
        )
        price = _finite_number(
            entry["price"],
            f"{module_id}.{entry_id}.price",
        )
        if price < 0:
            raise ValueError(
                f"Shop {module_id}/{entry_id} price must be nonnegative."
            )


def _validate_gui(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(
        module_id,
        config,
        {"template", "title", "rows", "entries"},
    )
    if config.get("template") != "read_only_menu":
        raise ValueError(
            f"GUI {module_id} must use template=read_only_menu or custom_java."
        )
    title = str(config.get("title", "M.M.M")).strip()
    if not title or len(title) > 128:
        raise ValueError(f"GUI {module_id} title is invalid.")
    rows = config.get("rows", 3)
    if type(rows) is not int or not 1 <= rows <= 6:
        raise ValueError(f"GUI {module_id} rows must be 1-6.")
    entries = config.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"GUI {module_id} entries must be a list.")
    slots: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "slot",
            "item",
            "count",
        }:
            raise ValueError(f"GUI {module_id} has an invalid menu entry.")
        if "slot" not in entry or "item" not in entry:
            raise ValueError(
                f"GUI {module_id} menu entry requires slot and item."
            )
        slot = entry["slot"]
        if type(slot) is not int or not 0 <= slot < rows * 9 or slot in slots:
            raise ValueError(
                f"GUI {module_id} has invalid or duplicate slot {slot!r}."
            )
        slots.add(slot)
        if not _RESOURCE_ID.fullmatch(str(entry["item"])):
            raise ValueError(
                f"GUI {module_id} slot {slot} item must be namespaced."
            )
        _positive_int(
            entry.get("count", 1),
            f"{module_id}.slot_{slot}.count",
        )


def _validate_networking(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(module_id, config, {"template", "actions"})
    if config.get("template") != "validated_action_channel":
        raise ValueError(
            f"Networking {module_id} must use template=validated_action_channel or custom_java."
        )
    actions = config.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(
            f"Networking {module_id} requires non-empty server-defined actions."
        )
    seen: set[str] = set()
    for action in actions:
        if not isinstance(action, dict) or "id" not in action or "type" not in action:
            raise ValueError(
                f"Networking {module_id} action fields are invalid."
            )
        action_id = str(action["id"])
        if not _ID.fullmatch(action_id) or action_id in seen:
            raise ValueError(
                f"Networking {module_id} has invalid or duplicate action {action_id!r}."
            )
        seen.add(action_id)
        action_type = str(action["type"])
        if action_type not in _ACTION_TYPES:
            raise ValueError(
                f"Networking {module_id}/{action_id} type {action_type!r} is not built in; use custom_java."
            )
        if action_type == "message":
            _require_exact_fields(action, {"id", "type", "message"}, module_id, action_id)
            if not str(action["message"]).strip():
                raise ValueError(
                    f"Networking {module_id}/{action_id} message is empty."
                )
        elif action_type == "grant_item":
            _require_exact_or_optional(
                action,
                required={"id", "type", "item"},
                optional={"count"},
                module_id=module_id,
                action_id=action_id,
            )
            if not _RESOURCE_ID.fullmatch(str(action["item"])):
                raise ValueError(
                    f"Networking {module_id}/{action_id} item must be namespaced."
                )
            _positive_int(
                action.get("count", 1),
                f"{module_id}.{action_id}.count",
            )
        else:
            _require_exact_or_optional(
                action,
                required={"id", "type", "effect"},
                optional={"duration_ticks", "amplifier"},
                module_id=module_id,
                action_id=action_id,
            )
            if not _RESOURCE_ID.fullmatch(str(action["effect"])):
                raise ValueError(
                    f"Networking {module_id}/{action_id} effect must be namespaced."
                )
            _positive_int(
                action.get("duration_ticks", 100),
                f"{module_id}.{action_id}.duration_ticks",
            )
            amplifier = _nonnegative_int(
                action.get("amplifier", 0),
                f"{module_id}.{action_id}.amplifier",
            )
            if amplifier > 255:
                raise ValueError(
                    f"{module_id}.{action_id}.amplifier must be 0-255."
                )


def _validate_social(module_id: str, config: dict[str, Any]) -> None:
    _reject_unknown(module_id, config, {"display_name"})
    if "display_name" in config and not str(config["display_name"]).strip():
        raise ValueError(
            f"Social module {module_id} display_name is empty."
        )


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    module_id: str,
    action_id: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"Networking {module_id}/{action_id} fields must be exactly {sorted(expected)}."
        )


def _require_exact_or_optional(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    module_id: str,
    action_id: str,
) -> None:
    fields = set(value)
    if not required <= fields or fields - required - optional:
        raise ValueError(
            f"Networking {module_id}/{action_id} fields are invalid."
        )


def _reject_unknown(
    module_id: str,
    config: dict[str, Any],
    allowed: set[str],
) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            f"Built-in module {module_id} has unsupported fields {sorted(unknown)}; use custom_java."
        )


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite JSON number.")
    return float(value)
