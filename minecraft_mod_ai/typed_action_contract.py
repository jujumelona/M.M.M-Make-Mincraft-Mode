"""One authoritative ABI for a UI request and its actual server function."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re


SCALARS = ("boolean", "int", "long", "double", "string")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAX_COMMAND_LENGTH = 32767


def action_command_bound(inputs, *, literal_length=128):
    """Upper bound for every valid input, including four-byte Unicode text."""
    total = literal_length
    for row in inputs:
        raw_bytes = (row["max_length"] * 4 if row["type"] == "string"
                     else {"int": 11, "long": 20, "double": 25, "boolean": 5}[row["type"]])
        total += 2 + (raw_bytes * 4 + 2) // 3  # space, x marker, unpadded base64
    return total


def validate_action_contract(raw, *, parameters=None):
    if not isinstance(raw, Mapping) or set(raw) != {"id", "label", "inputs"}:
        raise ValueError("ACTION_CONTRACT_FIELDS")
    if not isinstance(raw["id"], str) or not _NAME.fullmatch(raw["id"]):
        raise ValueError("ACTION_ID_INVALID")
    if not isinstance(raw["label"], str) or not 1 <= len(raw["label"]) <= 160:
        raise ValueError("ACTION_LABEL_INVALID")
    inputs = raw["inputs"]
    if not isinstance(inputs, (list, tuple)) or len(inputs) > 16:
        raise ValueError("ACTION_INPUTS_INVALID")
    names = set()
    for row in inputs:
        if not isinstance(row, Mapping):
            raise ValueError("ACTION_INPUT_FIELDS")
        name, kind = row.get("name"), row.get("type")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError("ACTION_INPUT_NAME_INVALID_OR_DUPLICATE")
        names.add(name)
        if kind not in SCALARS:
            raise ValueError("ACTION_INPUT_TYPE_INVALID")
        if not isinstance(row.get("label"), str) or not 1 <= len(row["label"]) <= 160:
            raise ValueError("ACTION_INPUT_LABEL_INVALID")
        fields = {"name", "type", "label"}
        if kind in {"int", "long", "double"}:
            fields |= {"minimum", "maximum"}
            low, high = row.get("minimum"), row.get("maximum")
            if any(type(value) not in {int, float} or not math.isfinite(value)
                   for value in (low, high)) or low > high:
                raise ValueError("ACTION_INPUT_RANGE_INVALID")
            if kind != "double":
                bits = 32 if kind == "int" else 64
                if (type(low) is not int or type(high) is not int
                        or low < -(2 ** (bits - 1)) or high >= 2 ** (bits - 1)):
                    raise ValueError("ACTION_INPUT_INTEGER_RANGE_INVALID")
        elif kind == "string":
            fields |= {"max_length", "choices"}
            length, choices = row.get("max_length"), row.get("choices")
            if type(length) is not int or not 0 <= length <= 4096:
                raise ValueError("ACTION_INPUT_STRING_BOUND_INVALID")
            if (not isinstance(choices, list) or len(choices) > 32
                    or any(not isinstance(c, str) or len(c) > length for c in choices)
                    or len(set(choices)) != len(choices)):
                raise ValueError("ACTION_INPUT_CHOICES_INVALID")
        if set(row) != fields:
            raise ValueError("ACTION_INPUT_FIELDS")
        if kind == "string":
            try:
                for choice in row["choices"]:
                    choice.encode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ValueError("ACTION_INPUT_CHOICES_INVALID_UNICODE") from error
    if action_command_bound(inputs) > MAX_COMMAND_LENGTH:
        raise ValueError("ACTION_COMMAND_TOO_LARGE")
    if parameters is not None:
        expected = [{"name": row["name"], "type": row["type"]} for row in inputs]
        if (not isinstance(parameters, Sequence) or len(parameters) != len(inputs) + 1
                or parameters[0].get("type") != "object"
                or parameters[0].get("name") in names
                or list(parameters[1:]) != expected):
            raise ValueError("ACTION_FUNCTION_SIGNATURE_MISMATCH")
    return deepcopy(dict(raw))


def action_bindings(plan):
    functions = {fn["id"]: fn for fn in plan.get("functions", ())}
    result, ids = [], set()
    literals = [row.get("config", {}).get("literal")
                for row in plan.get("event_bindings", ()) if row.get("event") == "command"]
    for binding in plan.get("event_bindings", ()):
        config = binding.get("config", {})
        if "action" not in config:
            continue
        fn = functions.get(binding.get("function"))
        if fn is None or fn.get("return_type") != "int":
            raise ValueError("ACTION_FUNCTION_TARGET_REQUIRED")
        action = validate_action_contract(config["action"], parameters=fn["parameters"])
        if action_command_bound(action["inputs"], literal_length=len(config["literal"])) > MAX_COMMAND_LENGTH:
            raise ValueError("ACTION_COMMAND_TOO_LARGE")
        if action["id"] in ids or literals.count(config.get("literal")) != 1:
            raise ValueError("ACTION_ID_OR_TRANSPORT_DUPLICATE")
        ids.add(action["id"])
        result.append({**action, "function": fn["id"],
                       "actor_parameter": fn["parameters"][0]["name"],
                       "literal": config["literal"]})
    return result
