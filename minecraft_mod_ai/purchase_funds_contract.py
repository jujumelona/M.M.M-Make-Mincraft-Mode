"""Repair an authored purchase's missing funds dependency before content work.

Only an authored, typed currency balance can satisfy a purchase. The repair
creates one additional canonical state variable; it never guesses a state key,
replaces an unrelated variable, or allows the purchase to debit an arbitrary
numeric gameplay counter.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .authored_structured_design import (
    active_concern_records,
    normalize_structured_sections,
)

_NUMERIC = frozenset({"int", "long", "double"})
_PURCHASE_TERMS = ("purchase", "buy", "구매", "결제")
_CURRENCY_TERMS = (
    "balance", "credits", "currency", "coins", "money", "funds",
    "wallet", "cash", "gold", "emerald", "token", "dollar",
    "잔액", "잔고", "크레딧", "화폐", "골드", "코인",
    "에메랄드", "자금", "보유금", "토큰",
)


def is_purchase_interaction(record: Mapping[str, Any] | None) -> bool:
    if not isinstance(record, Mapping):
        return False
    text = " ".join(
        str(record.get(field) or "")
        for field in ("server_action", "trigger", "description", "label")
    ).casefold()
    return any(term in text for term in _PURCHASE_TERMS)


def currency_state_keys(
    structured_sections: Mapping[str, Any] | None,
) -> set[str]:
    result: set[str] = set()
    for record in active_concern_records(
        structured_sections, "state_model"
    ).get("variables", ()):
        if not isinstance(record, Mapping):
            continue
        key = str(record.get("name") or "").strip()
        if record.get("type") not in _NUMERIC or not key:
            continue
        meaning = " ".join(
            str(record.get(field) or "")
            for field in ("name", "unit", "domain")
        ).casefold()
        if any(word in meaning for word in _CURRENCY_TERMS):
            result.add(key)
    return result


def repair_missing_purchase_funds(
    router: Any,
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *,
    budget: Any = None,
) -> dict[str, Any]:
    """Repair a missing currency declaration without altering existing state.

    Fail before expensive content authoring when the state model is missing or
    already at the host's record limit. Never manufacture a funds key inside
    the executable dispatch lowering.
    """
    sections = normalize_structured_sections(structured_sections)
    interactions = active_concern_records(
        sections, "resources_and_ui"
    ).get("interactions", ())
    purchases = [record for record in interactions if is_purchase_interaction(record)]
    if not purchases or currency_state_keys(sections):
        return sections

    existing = active_concern_records(sections, "state_model").get("variables", [])
    if not isinstance(sections.get("state_model", {}).get("specification"), dict):
        raise ValueError(
            "PURCHASE_FUNDS_STATE_REAUTHOR_REQUIRED: purchase exists without "
            "an authored state_model; cannot invent a balance in executable IR"
        )

    from .execution_contract_policy import PLANNER_CONCERN_MAX_RECORDS
    if len(existing) >= PLANNER_CONCERN_MAX_RECORDS:
        raise ValueError(
            "PURCHASE_FUNDS_STATE_REAUTHOR_REQUIRED: all state variable "
            "record slots are occupied without an identified currency balance; "
            "re-author state_model rather than replacing an unrelated variable"
        )

    from .fixed_template_generation import generate_fixed_template_value
    from .model_output_atomicity_contract import structured_output_token_ceiling

    # Two bounded 3-field responses respect the atomic local-model contract,
    # while preserving all six fields of the canonical state variable record.
    schema_identity = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string", "minLength": 1, "maxLength": 64,
                "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$",
            },
            "type": {"type": "string", "enum": ["int", "long", "double"]},
            "default": {
                "type": "string", "minLength": 1, "maxLength": 32,
                "pattern": r"^(?:[0-9]+(?:\.[0-9]+)?)$",
            },
        },
        "required": ["name", "type", "default"],
        "additionalProperties": False,
    }
    schema_semantics = {
        "type": "object",
        "properties": {
            name: {"type": "string", "minLength": 1, "maxLength": 160}
            for name in ("owner", "unit", "domain")
        },
        "required": ["owner", "unit", "domain"],
        "additionalProperties": False,
    }
    inputs = {
        "user_request": requested_prompt,
        "purchase_interactions": purchases,
        "existing_variables": existing,
        "state_transitions": active_concern_records(
            sections, "state_model"
        ).get("transitions", ()),
        "instruction": (
            "A purchase needs one player-owned, persisted numeric currency "
            "balance. Author exactly one MISSING variable compatible with the "
            "actual game's economy. Use a unique identifier and a valid "
            "numeric default from the game design. Do not repurpose health, "
            "fuel, energy, experience, or item-count state as money. Do not "
            "claim a funding path or initial amount not supported by the design."
        ),
    }
    if budget is not None:
        budget.consume("typed.purchase_funds_state_repair.identity")
    identity = generate_fixed_template_value(
        router, "planner",
        (
            {"role": "system", "content": (
                "Complete the numeric currency variable identity for an authored "
                "Minecraft purchase, using only the fixed three-field schema."
            )},
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
        ),
        response_schema=schema_identity,
        enable_tools=False,
        description="Repair missing purchase funds state identity",
        output_token_ceiling=structured_output_token_ceiling(schema_identity),
    )
    if not isinstance(identity, Mapping):
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_INVALID: missing identity")
    name = str(identity.get("name") or "")
    kind = identity.get("type")
    default = str(identity.get("default") or "")
    if name in {str(record.get("name") or "") for record in existing}:
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_DUPLICATE: " + name)
    if kind not in _NUMERIC or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", default):
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_INVALID: non-numeric default")
    if kind in {"int", "long"} and not default.isdecimal():
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_INVALID: integer default required")

    if budget is not None:
        budget.consume("typed.purchase_funds_state_repair.semantics")
    semantics = generate_fixed_template_value(
        router, "planner",
        (
            {"role": "system", "content": (
                "Complete the owner, unit and domain of the SAME numeric currency "
                "variable. Do not alter its identity or invent an external API."
            )},
            {"role": "user", "content": json.dumps(
                {**inputs, "authored_identity": dict(identity)},
                ensure_ascii=False,
            )},
        ),
        response_schema=schema_semantics,
        enable_tools=False,
        description="Repair missing purchase funds state semantics",
        output_token_ceiling=structured_output_token_ceiling(schema_semantics),
    )
    if not isinstance(semantics, Mapping):
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_INVALID: missing semantics")
    record = {**dict(identity), **dict(semantics)}
    if not any(
        term in " ".join(str(record.get(field) or "") for field in
                         ("name", "unit", "domain")).casefold()
        for term in _CURRENCY_TERMS
    ):
        raise ValueError(
            "PURCHASE_FUNDS_STATE_REPAIR_INVALID: no currency meaning"
        )

    repaired = deepcopy(sections)
    repaired["state_model"]["specification"]["variables"] = [
        *deepcopy(existing), record,
    ]
    validated = normalize_structured_sections(repaired)
    if not currency_state_keys(validated):
        raise ValueError("PURCHASE_FUNDS_STATE_REPAIR_INVALID: currency not bound")
    return validated


__all__ = [
    "is_purchase_interaction", "currency_state_keys",
    "repair_missing_purchase_funds",
]
