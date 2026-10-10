"""Bounded authored repair of missing player-invoked gameplay entry points.

A saved design whose only Fabric callback is player_join cannot implement
interactive purchase/upgrade/launch actions. Repair the *design* with a
model-authored explicit command, rather than forging lifecycle handlers in
the Java lowering stage. This adds a callable server entry point; it does
not claim an unrelated GUI button is already wired to that command.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .authored_structured_design import (
    active_concern_records,
    normalize_structured_sections,
)
from .typed_event_ir import infer_event_type, is_mod_initialize_trigger

_BOOTSTRAP = frozenset({"player_join", "server_started"})


def _compact_duplicate_bootstrap_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce identical source hooks, preserving every authored owner/boundary.

    Multiple player_join records do not create multiple distinct invocation
    mechanisms. Each uses the same event/dispatch. Combining their metadata
    keeps the original obligations while releasing redundant record slots for
    an executable player action. Do not merge unlike hooks.
    """
    result: list[dict[str, Any]] = []
    by_trigger: dict[str, int] = {}
    for row in rows:
        trigger = str(row["trigger"])
        key = "mod_initialize" if is_mod_initialize_trigger(trigger) else trigger
        if key not in by_trigger:
            by_trigger[key] = len(result)
            result.append(deepcopy(row))
            continue
        prior = result[by_trigger[key]]
        for field in ("boundary", "owner"):
            previous = str(prior[field])
            new = str(row[field])
            if new != previous:
                # Include the full original description, not just its event ID.
                parts = previous.split(" | ")
                if new not in parts:
                    prior[field] = previous + " | " + new
    return result



def repair_missing_gameplay_entrypoint(
    router: Any,
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *,
    budget: Any = None,
) -> dict[str, Any]:
    """Author a player command without discarding existing source-hook duties.

    An overfilled integration page can hold duplicate player-join hooks. Merge
    those duplicate declarations and retain all boundary/owner descriptions.
    The command is not a substitute for the feature's required GUI.
    """
    sections = normalize_structured_sections(structured_sections)
    algorithm = active_concern_records(sections, "algorithm")
    if not (algorithm.get("steps") and algorithm.get("atomic_mutations")):
        return sections
    existing = active_concern_records(sections, "integration").get("entry_points", [])
    events = {
        infer_event_type(row.get("trigger"))
        for row in existing if isinstance(row, Mapping)
        and not is_mod_initialize_trigger(row.get("trigger"))
    }
    # A server tick, shutdown or player respawn is not a purchase/upgrade
    # invocation. Only an explicitly named command is a bound player action
    # in this contract; non-command lifecycle events cannot satisfy it.
    if "command" in events:
        return sections

    from .execution_contract_policy import PLANNER_CONCERN_MAX_RECORDS
    if len(existing) >= PLANNER_CONCERN_MAX_RECORDS:
        available = _compact_duplicate_bootstrap_entries(existing)
        if len(available) >= PLANNER_CONCERN_MAX_RECORDS:
            raise ValueError(
                "GAMEPLAY_ENTRYPOINT_REAUTHOR_REQUIRED: all entry-point slots "
                "describe distinct lifecycle hooks, none of which is a "
                "player-invoked action. The authored integration section "
                "needs bounded re-authoring rather than inserting a fake "
                "hook or discarding an unrelated lifecycle obligation."
            )
        existing = available

    from .task_template_catalog import load_record_template
    from .fixed_template_generation import generate_fixed_template_value
    from .model_output_atomicity_contract import structured_output_token_ceiling

    schema = deepcopy(
        load_record_template("feature/integration/entry_points")["record_schema"]
    )
    # The host supports many events, but the missing requirement is a player
    # action. Do not permit a fake server tick or respawn as its replacement.
    schema["properties"]["trigger"] = {
        "type": "string",
        "pattern": r"^command:[a-z0-9_]{1,64}$",
        "maxLength": 72,
    }
    if budget is not None:
        budget.consume("typed.gameplay_entrypoint_repair")
    inputs = {
        "request": requested_prompt,
        "existing_entry_points": existing,
        "ordered_algorithm_steps": algorithm["steps"],
        "atomic_mutations": algorithm["atomic_mutations"],
        "ui_interactions": active_concern_records(
            sections, "resources_and_ui"
        ).get("interactions", []),
        "instructions": (
            "Author ONE additional concrete command:<literal> Fabric entry "
            "point that lets a player invoke a genuine gameplay action from "
            "this design. Do not choose a lifecycle event, do not claim a GUI "
            "button is wired, and do not replace existing behavior. Choose a "
            "command with a meaningful name; this is an explicit design "
            "decision, not a guessed API hook."
        ),
    }
    record = generate_fixed_template_value(
        router,
        "planner",
        (
            {
                "role": "system",
                "content": (
                    "Author one bounded missing player-interaction entry point "
                    "for the supplied Minecraft game design. Output one record "
                    "matching the exact host schema. No invented Fabric APIs."
                ),
            },
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
        ),
        response_schema=schema,
        enable_tools=False,
        description="Repair missing real gameplay invocation during planning",
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(record, Mapping):
        raise ValueError("GAMEPLAY_ENTRYPOINT_REPAIR_INVALID: expected record")
    trigger = str(record.get("trigger") or "")
    if infer_event_type(trigger) != "command":
        raise ValueError(
            "GAMEPLAY_ENTRYPOINT_REPAIR_INVALID: model did not supply "
            "an explicit command entry point."
        )
    repaired = deepcopy(sections)
    if "integration" not in repaired:
        raise ValueError("GAMEPLAY_ENTRYPOINT_REPAIR_NO_INTEGRATION")
    spec = repaired["integration"].get("specification")
    if not isinstance(spec, dict):
        raise ValueError("GAMEPLAY_ENTRYPOINT_REPAIR_NO_SPEC")
    spec["entry_points"] = [*deepcopy(existing), dict(record)]
    return normalize_structured_sections(repaired)


__all__ = ["repair_missing_gameplay_entrypoint"]
