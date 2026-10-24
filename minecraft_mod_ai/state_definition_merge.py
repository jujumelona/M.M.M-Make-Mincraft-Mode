"""Join already-authored state declarations without changing their contracts."""
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def merge_state_definitions(
    prior: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in (*prior, *additions):
        name = str(row["name"])
        if name in result and result[name] != row:
            raise ValueError(f"REQUEST_STATE_DEFINITION_CONFLICT: {name}")
        result[name] = deepcopy(dict(row))
    return list(result.values())
