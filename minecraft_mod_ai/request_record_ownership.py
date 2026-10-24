"""Host-owned source identity for accumulated design records and UI effects."""
from copy import deepcopy

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


def same_action_source(left, right):
    return (
        str(left.get("trigger") or "").strip().casefold()
        == str(right.get("trigger") or "").strip().casefold()
        and left.get(SOURCE_REQUIREMENT) == right.get(SOURCE_REQUIREMENT)
    )


def action_identity_payload(record, metadata):
    result = {"contract": metadata, "trigger": str(record.get("trigger") or "")}
    if SOURCE_REQUIREMENT in record:
        result[SOURCE_REQUIREMENT] = record[SOURCE_REQUIREMENT]
    return result
