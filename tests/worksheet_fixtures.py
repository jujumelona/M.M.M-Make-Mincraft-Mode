"""Structured design fixtures shared by planning integration tests."""
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def specification(section):
    if section == "state_model":
        return {
            "variables": [{
                "name": "credits",
                "owner": "server",
                "type": "Double",
                "unit": "credits",
                "default": "0",
                "domain": "economy",
            }],
            "transitions": [{
                "from_state": "idle",
                "trigger": "buy",
                "guard": "credits >= cost",
                "mutation": "credits -= cost",
                "to_state": "done",
            }],
            "invariants": [{
                "condition": "credits >= 0",
                "enforcement": "reject negative balance",
            }],
            "initialization": [{
                "owner": "server",
                "trigger": "server_start",
                "initial_state": "credits = 0",
            }],
            "updates": [{
                "trigger": "reward",
                "mutation": "credits += amount",
                "owner": "server",
            }],
            "cleanup": [{
                "event": "reset",
                "action": "credits = 0",
                "retained_state": "none",
            }],
            "concurrency": [{
                "entry_path": "state_runtime",
                "ownership": "server",
                "reentrancy_rule": "serialized",
            }],
            "inapplicable_concerns": [],
        }
    return {
        **{
            concern: [{field: f"{section} {concern} {field}: server owns the observable outcome."
                       for field in columns.split()}]
            for concern, columns in DETAIL_RECORDS[section].items()
        },
        "inapplicable_concerns": [],
    }


def row(section, refs=()):
    return {"specification": specification(section), "constraint_evidence_refs": list(refs)}
