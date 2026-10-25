"""Complete host-side callback API authority, independent of prompt projections.

The normal callback prompt is a relevance-ranked sample. It cannot establish
that a native mutator is absent. This catalogue retains inspected instance
ABIs and every inspected one-hop receiver path; the author pages the callable
choices after binding their operands. No full catalogue goes to the model.
"""
from __future__ import annotations

from .construction_file_api_evidence import prompt_symbol_record
from .construction_type_contracts import canonical_type_name


def runtime_mutation_api_records(evidence, owners):
    roots = list(dict.fromkeys(canonical_type_name(name) for name in owners))
    roots = [name for name in roots if evidence._accessible(name)]
    records, options = {}, {}

    def include(owner):
        if owner in records:
            return
        options[owner] = evidence._member_options(owner)
        members = [row for _, row in options[owner]
                   if row.get("kind") == "METHOD"
                   and not (row.get("static") or int(row.get("access", 0)) & 8)
                   and row.get("owner") != "java.lang.Object"]
        records[owner] = {
            "verified_class": owner,
            "members": [prompt_symbol_record(row) for row in members],
            "member_count": len(members), "members_truncated": False,
            "evidence": "actual_target_symbol_index", "behavior_verified": False,
        }

    for root in roots:
        include(root)
    for root in roots:
        for key, row in options[root]:
            if (row.get("kind") != "METHOD" or row.get("parameters")
                    or row.get("static") or int(row.get("access", 0)) & 8
                    or row.get("throws")):
                continue
            name = str(row.get("name") or "")
            if not name or name.startswith(("set", "register", "wait", "notify")):
                continue
            target = canonical_type_name(row.get("return_type") or "")
            if (not evidence._accessible(target)
                    or target.startswith(("java.", "javax.", "jdk.", "sun.",
                                          "com.sun.", "org.lwjgl."))):
                continue
            include(target)
            records[target].setdefault("verified_receiver_paths", []).append({
                "source_type": root, "source_method": name,
                "destination_type": target, "inspected_source_ref": key,
            })
    return list(records.values())


def bind_runtime_mutation_catalog(evidence, task, frame):
    """Bind complete inspected ABIs to one exact host-owned callback frame."""
    from .host_verified_callback_ops import verified_runtime_mutation_candidates

    event = task["verified_event_callback"]
    native = {
        "verified_event_callback": event,
        "declaration_metadata": task.get("declaration_metadata", {}),
        "callback_input_api": runtime_mutation_api_records(
            evidence, event.get("parameters", ())),
    }
    return verified_runtime_mutation_candidates([], frame, api_task=native)
