"""Assign validated worksheet records on measured small-model slots."""
from __future__ import annotations

import json
from .request_record_ownership import SOURCE_REQUIREMENT, same_action_source

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



def _choose_runtime_owner_batch(router, items, features, actions, consume):
    """One small bounded owner decision for up to four actual records.

    The host owns all record identities and validates every returned value.
    No semantic record is dropped merely because the model omitted a field.
    """
    from .fixed_template_generation import generate_fixed_template_value

    if not 1 <= len(items) <= 4:
        raise ValueError("RUNTIME_OWNER_BATCH_SIZE_INVALID")
    available = [row["requirement_id"] for row in features]
    schema = {
        "type": "object",
        "properties": {
            f"r{i}": {"type": "string", "enum": (
                available + ([TYPED_UI_OWNER + ":" + action for action in actions]
                             if actions and not ref.startswith("integration.") else [])
            )}
            for i, (ref, _record) in enumerate(items)
        },
        "required": [f"r{i}" for i in range(len(items))],
        "additionalProperties": False,
    }
    if not all(schema["properties"][key]["enum"] for key in schema["required"]):
        raise ValueError("RUNTIME_OWNER_BATCH_EMPTY_DOMAIN")
    if all(len(schema["properties"][key]["enum"]) == 1 for key in schema["required"]):
        result = {key: schema["properties"][key]["enum"][0]
                  for key in schema["required"]}
    else:
        if consume is not None:
            consume("request_scope.runtime_owner_batch")
        result = generate_fixed_template_value(
            router, "planner", (
                {"role": "system", "content": (
                    "Assign each existing worksheet record to exactly one of "
                    "the enumerated original feature IDs or an already implemented "
                    "typed UI action. Do not add, remove or merge records. "
                    "Assign custom gameplay events to their actual feature."
                )},
                {"role": "user", "content": json.dumps({
                    "records": [{"key": f"r{i}", "record_ref": ref, "record": record}
                                for i, (ref, record) in enumerate(items)],
                    "features": [{"id": row["requirement_id"],
                                  "statement": row["statement"]} for row in features],
                    "existing_ui_actions": actions,
                }, ensure_ascii=False, separators=(",", ":"))},
            ), response_schema=schema, enable_tools=False,
        )
    if not isinstance(result, dict) or set(result) != set(schema["required"]):
        raise ValueError("RUNTIME_OWNER_BATCH_RESPONSE_INVALID")
    out = []
    for i, (ref, _record) in enumerate(items):
        key = f"r{i}"
        if result[key] not in schema["properties"][key]["enum"]:
            raise ValueError("RUNTIME_OWNER_BATCH_ASSIGNMENT_INVALID")
        out.append((ref, result[key]))
    return out


def _source_runtime_owner(ref, record, feature_ids, interactions):
    from .authored_ui_action_bindings import action_record_identity

    source = record.get(SOURCE_REQUIREMENT)
    if source not in feature_ids:
        return None
    if not ref.startswith("state_model."):
        return source
    actions = [action_record_identity(row) for row in interactions
               if same_action_source(record, row)]
    if len(actions) > 1:
        raise ValueError("REQUEST_UI_EFFECT_AMBIGUOUS: " + ref)
    return TYPED_UI_OWNER + ":" + actions[0] if actions else source


def _partition_source_owners(records, features, sections):
    feature_ids = {row["requirement_id"] for row in features}
    interactions = sections.get("resources_and_ui", {}).get("specification", {}).get("interactions", ())
    assigned, remaining = [], []
    for ref, record in records:
        owner = _source_runtime_owner(ref, record, feature_ids, interactions)
        if owner is None:
            remaining.append((ref, record))
        else:
            assigned.append((ref, owner))
    return assigned, remaining


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
    records = list(_runtime_records(sections))
    known_assignments, records = _partition_source_owners(records, features, sections)
    if len(records) >= 3 and len(features) > 1:
        # Group only four records per bounded inference; a dozen one-field
        # owner questions otherwise cost a dozen model roundtrips.
        batches = [records[i:i + 4] for i in range(0, len(records), 4)]
        assigned_batches = deterministic_model_map(
            router, batches,
            lambda items: _choose_runtime_owner_batch(
                router, items, features, actions, consume),
            thread_name_prefix="mmm-request-runtime-owner-batch",
        )
        assignments = [assignment for batch in assigned_batches for assignment in batch]
    else:
        assignments = deterministic_model_map(
            router, records,
            lambda item: _choose_runtime_owner(router, item, features, actions, consume),
            thread_name_prefix="mmm-request-runtime-owner",
        )
    assignments = [*known_assignments, *assignments]
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
