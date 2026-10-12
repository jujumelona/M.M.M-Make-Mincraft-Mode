"""Assign validated worksheet records on measured small-model slots."""
from __future__ import annotations

import json

from .request_runtime_records import TYPED_UI_OWNER as TYPED_UI_OWNER
from .request_runtime_records import _runtime_records
from .request_runtime_records import feature_module_id as feature_module_id
from .request_runtime_records import (
    generated_owner_source_context as generated_owner_source_context,
)
from .request_runtime_records import owned_record_refs as owned_record_refs
from .request_runtime_records import owner_records as owner_records
from .request_runtime_records import runtime_sections as runtime_sections
from .request_runtime_records import ui_record_action_guards as ui_record_action_guards


def _choose_runtime_owner(router, item, features, actions, consume):
    from .fixed_template_generation import generate_fixed_template_value

    ref, record = item
    available = [row["requirement_id"] for row in features]
    if actions and not ref.startswith("integration."):
        available.extend(TYPED_UI_OWNER + ":" + action for action in actions)
    if len(available) == 1:
        owner = available[0]
    else:
        if consume is not None:
            consume("request_scope.runtime_owner")
        owner = generate_fixed_template_value(router, "planner", (
            {"role": "system", "content": (
                "Assign this existing worksheet record to exactly one original requested feature. "
                "Do not add mechanics or change the record. Choose __typed_ui__:ACTION only for a "
                "transaction/state rule already implemented by one of the listed UI actions. "
                "Custom gameplay triggers belong to their actual feature, not player join or an invented command."
            )},
            {"role": "user", "content": json.dumps({
                "record_ref": ref, "record": record,
                "features": [{"id": row["requirement_id"], "statement": row["statement"]} for row in features],
                "existing_ui_actions": actions,
            }, ensure_ascii=False, separators=(",", ":"))},
        ), response_schema={"type": "string", "enum": available}, enable_tools=False)
    return ref, owner


def author_runtime_ownership(router, scope, sections, *, budget=None):
    """Choose one owner per runtime record using bounded feature identities.

    Infrastructure remains in the Typed host. Existing mutating UI transactions
    may retain their dedicated host dispatcher, never a second generic handler.
    """
    from .authored_structured_design import normalize_structured_sections
    from .authored_ui_action_bindings import _mutating_ui_actions
    from .parallel_model_tasks import deterministic_model_map, serialized_callback

    features = [row for row in scope.get("requirements", ())
                if row.get("implementation_role", "gameplay") != "interface"]
    if not features:
        return []
    sections = normalize_structured_sections(sections)
    actions = _mutating_ui_actions(sections)
    consume = serialized_callback(budget.consume) if budget is not None else None
    assignments = deterministic_model_map(
        router, _runtime_records(sections),
        lambda item: _choose_runtime_owner(router, item, features, actions, consume),
        thread_name_prefix="mmm-request-runtime-owner",
    )
    groups = {}
    for ref, owner in assignments:
        action = ""
        if owner.startswith(TYPED_UI_OWNER + ":"):
            owner, action = TYPED_UI_OWNER, owner[len(TYPED_UI_OWNER) + 1:]
        group = groups.setdefault((owner, action), {
            "feature_id": owner,
            "owner_module": "authored_typed_plan" if owner == TYPED_UI_OWNER else feature_module_id(owner),
            "worksheet_record_refs": [],
            **({"ui_action": action} if action else {}),
        })
        group["worksheet_record_refs"].append(ref)
    return list(groups.values())
