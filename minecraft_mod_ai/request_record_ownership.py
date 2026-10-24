"""Host-owned source identity for accumulated design records and UI effects."""
from copy import deepcopy
import hashlib
import json

SOURCE_REQUIREMENT = "source_requirement_id"


def accumulated_record_schema(schema, concerns):
    for concern in concerns:
        records = schema["properties"]["specification"]["properties"][concern]
        records.pop("maxItems", None)
        records["items"]["properties"][SOURCE_REQUIREMENT] = {
            "type": "string", "minLength": 1,
        }
    return schema


def bind_request_records(sections, requirement_id):
    result = deepcopy(sections)
    for section, value in result.items():
        for concern, rows in value["specification"].items():
            if concern == "inapplicable_concerns" or (section, concern) == ("state_model", "variables"):
                continue
            for row in rows:
                previous = row.get(SOURCE_REQUIREMENT, requirement_id)
                if previous != requirement_id:
                    raise ValueError("REQUEST_RECORD_SOURCE_CONFLICT: " + str(previous))
                row[SOURCE_REQUIREMENT] = requirement_id
    return result


def source_trigger_identity(record):
    """Canonical host event key, bounded to the state-model trigger ABI.

    The worksheet's descriptive UI trigger can exceed the state DSL's 128
    character limit. Hashing the exact approved description and source owner
    preserves identity without asking the model to copy unrepresentable text.
    Ordinary short triggers are kept untouched for existing contracts.
    """
    source = str(record.get("trigger") or "").strip().casefold()
    if len(source) <= 128:
        return source
    payload = json.dumps(
        [str(record.get(SOURCE_REQUIREMENT) or ""), source],
        ensure_ascii=False, separators=(",", ":"),
    )
    return "ui_evt_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def same_action_source(left, right):
    return (
        source_trigger_identity(left) == source_trigger_identity(right)
        and left.get(SOURCE_REQUIREMENT) == right.get(SOURCE_REQUIREMENT)
    )


def action_identity_payload(record, metadata):
    result = {"contract": metadata, "trigger": str(record.get("trigger") or "")}
    if SOURCE_REQUIREMENT in record:
        result[SOURCE_REQUIREMENT] = record[SOURCE_REQUIREMENT]
    return result
