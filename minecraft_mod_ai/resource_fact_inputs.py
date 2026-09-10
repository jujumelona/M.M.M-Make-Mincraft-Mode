"""Strict deterministic resource fact inputs, with no model-authored JSON files."""

import re
from math import isfinite

from .prompt_fact_types import FactType

_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")


def resource_inputs(fact, mod_id):
    value = fact.value
    if not isinstance(value, dict):
        raise TypeError("RESOURCE_FACT_VALUE: expected structured host inputs")
    references = []

    def identifier(raw):
        if isinstance(raw, str) and re.fullmatch(r"[a-z][a-z0-9_]*", raw):
            raw = mod_id + ":" + raw
        if not isinstance(raw, str) or not _ID.fullmatch(raw):
            raise ValueError(f"RESOURCE_IDENTIFIER: {raw!r}")
        references.append(raw)
        return raw

    def count(raw):
        if type(raw) is not int or not 1 <= raw <= 64:
            raise ValueError("RESOURCE_COUNT: explicit integer 1..64 required")
        return raw

    result = dict(value)
    if fact.fact_type == FactType.REGISTRY_TAG:
        if set(value) != {"registry_kind", "members"} or value["registry_kind"] not in {
            "item",
            "block",
            "entity_type",
        }:
            raise ValueError(
                "RESOURCE_TAG_REGISTRY: item, block, or entity_type required"
            )
        if not isinstance(value["members"], list) or not value["members"]:
            raise ValueError("RESOURCE_TAG_MEMBERS: nonempty explicit members required")
        result["members"] = [identifier(x) for x in value["members"]]
        kind = "tag/registry"
    elif fact.fact_type == FactType.CRAFTING_RECIPE:
        mode = value.get("kind")
        result["result_id"] = identifier(value.get("result_id"))
        count(value.get("count"))
        if mode == "shapeless":
            if (
                set(value) != {"kind", "ingredients", "result_id", "count"}
                or not isinstance(value["ingredients"], list)
                or not 1 <= len(value["ingredients"]) <= 9
            ):
                raise ValueError(
                    "RESOURCE_RECIPE_INGREDIENTS: one to nine explicit ingredients required"
                )
            result["ingredients"] = [identifier(x) for x in value["ingredients"]]
        elif mode == "shaped":
            if set(value) != {"kind", "pattern", "key", "result_id", "count"}:
                raise ValueError("RESOURCE_RECIPE_SHAPED: unexpected or missing fields")
            pattern, key = value["pattern"], value["key"]
            if (
                not isinstance(pattern, list)
                or not 1 <= len(pattern) <= 3
                or any(
                    not isinstance(row, str) or not 1 <= len(row) <= 3
                    for row in pattern
                )
                or len({len(row) for row in pattern}) != 1
            ):
                raise ValueError(
                    "RESOURCE_RECIPE_PATTERN: rectangular 1..3 pattern required"
                )
            symbols = set("".join(pattern)) - {" "}
            if not isinstance(key, dict) or not symbols or set(key) != symbols:
                raise ValueError("RESOURCE_RECIPE_KEY: exact pattern symbols required")
            result["key"] = {k: identifier(v) for k, v in key.items()}
        else:
            raise ValueError("RESOURCE_RECIPE_KIND: shaped or shapeless required")
        kind = "recipe/" + mode
    elif fact.fact_type == FactType.SMELTING_RECIPE:
        if set(value) != {
            "cooking_type",
            "ingredient",
            "result_id",
            "experience",
            "cookingtime",
        } or value["cooking_type"] not in {"smelting", "blasting"}:
            raise ValueError(
                "RESOURCE_COOKING_TYPE: explicit smelting or blasting inputs required"
            )
        result["ingredient"] = identifier(value["ingredient"])
        result["result_id"] = identifier(value["result_id"])
        exp, time = value["experience"], value["cookingtime"]
        if (
            type(exp) not in {float, int}
            or not isfinite(exp)
            or exp < 0
            or type(time) is not int
            or time <= 0
        ):
            raise ValueError(
                "RESOURCE_COOKING_VALUES: nonnegative experience and positive integer time required"
            )
        kind = "recipe/smelting"
    else:
        raise ValueError("RESOURCE_FACT_TYPE: unsupported")
    result["resource_references"] = list(dict.fromkeys(references))
    return "fabric/" + kind, result
