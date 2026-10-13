"""Bounded, host-owned state shared by declaration and operation authors.

Planned declarations are usable within their owner, but are not evidence of
already installed bytecode. The complete inventory and accepted plan stay on
the host; every decision receives its phase, cursor and lookup provenance.
"""
from __future__ import annotations

from collections import deque


PROTOCOL_VERSION = "mmm/construction-state-v2"
PAGE_SIZE = 8
RECENT_SIZE = 4


def symbol_record(key, row):
    visible = {k: v for k, v in row.items()
               if k not in {"code", "source", "source_path"}}
    provenance = row.get("provenance")
    origin = ("planned_declaration" if provenance == "construction_plan" else
              "materialized_source" if provenance == "materialized_source" else
              "inspected_bytecode")
    return {**visible, "symbol_id": key, "origin": origin}


def page_result(rows, offset, *, field):
    if type(offset) is not int or offset < 0 or (offset and offset >= len(rows)):
        raise ValueError("CONSTRUCTION_LOOKUP_OFFSET_INVALID")
    selected = rows[offset:offset + PAGE_SIZE]
    return {field: selected, "offset": offset, "total_count": len(rows),
            "status": "matches" if selected else "no_matches",
            "next_offset": offset + PAGE_SIZE if len(rows) > offset + PAGE_SIZE else None}


def symbol_page(symbols, query, offset):
    query = query.strip()
    if len(query) < 2:
        raise ValueError("CONSTRUCTION_LOOKUP_QUERY_REQUIRED")
    folded = query.casefold()
    matches = sorted(key for key, row in symbols.items()
                     if folded in key.casefold() or folded == str(row.get("name", "")).casefold())
    # Materialize only this page, never the whole inspected inventory.
    result = page_result(matches, offset, field="matches")
    result["matches"] = [symbol_record(key, symbols[key]) for key in result["matches"]]
    return {"kind": "symbols", "query": query, **result}


STATE_INSTRUCTION = (
    "construction_state is the host's current accepted plan. A new owner and its "
    "planned_declaration symbols are being created, not missing installed source. "
    "Its class header and recent declarations are already shown; use declarations "
    "to page older accepted members when that action is available. Do not search "
    "for an existing implementation of a new owner. An empty lookup means no match "
    "for that query, not missing context. A lookup's next_offset=null means its "
    "results are exhausted. Never repeat a consumed query and offset in the same "
    "phase and revision; choose another dependency query or advance the current "
    "declaration/operation. Planned members with a body_owner are already assigned; "
    "add only missing declarations. Select only available_actions."
)


class ConstructionState:
    def __init__(self, symbols, owner, target_path, *, declaration_ids=(), revision=0):
        self.symbols = symbols
        self.owner = owner
        self.target_path = target_path
        self.declaration_ids = list(declaration_ids)
        self.revision = revision
        self.phase = ""
        self.lookup = {}
        self.history = deque(maxlen=PAGE_SIZE)
        self.requests = set()

    def enter(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.requests.clear()

    def advance(self):
        self.revision += 1
        self.lookup = {}
        self.requests.clear()

    def add_declaration(self, key):
        if key in self.declaration_ids:
            raise ValueError("CONSTRUCTION_DECLARATION_DUPLICATE: " + key)
        self.declaration_ids.append(key)
        self.advance()

    def payload(self, available_actions, **cursor):
        row = self.symbols.get(self.owner, {})
        planned = row.get("provenance") == "construction_plan"
        return {"construction_state": {
            "protocol": PROTOCOL_VERSION, "phase": self.phase, "revision": self.revision,
            "owner": {"symbol_id": self.owner, "target_path": self.target_path,
                      "status": "under_construction" if planned else "existing_inspected",
                      "header": symbol_record(self.owner, row)},
            "declarations": {"total_count": len(self.declaration_ids),
                "recent": [symbol_record(key, self.symbols[key])
                           for key in self.declaration_ids[-RECENT_SIZE:]]},
            "lookups_since_progress": len(self.requests),
            "recent_lookups": list(self.history),
            "available_actions": list(available_actions), **cursor,
        }, "lookup": self.lookup}

    def consume_lookup(self, kind, query, offset, result):
        normalized = query.strip().casefold() if kind in {"symbols", "sources"} else query.strip()
        if kind == "nodes": normalized = ",".join(sorted(key.strip() for key in query.split(',')))
        request = (kind, normalized, offset)
        if request in self.requests:
            raise ValueError("CONSTRUCTION_AUTHOR_NO_PROGRESS: repeated " + kind + " lookup")
        self.requests.add(request)
        self.lookup = {**result, "kind": kind, "query": query.strip(), "offset": offset}
        self.history.append({"phase": self.phase, "revision": self.revision,
            "kind": kind, "query": query.strip(), "offset": offset,
            "status": result["status"], "total_count": result["total_count"],
            "next_offset": result["next_offset"]})
        return self.lookup

    def lookup_symbols(self, query, offset):
        return self.consume_lookup("symbols", query, offset,
                                   symbol_page(self.symbols, query, offset))

    def lookup_declarations(self, offset):
        rows = [symbol_record(key, self.symbols[key]) for key in self.declaration_ids]
        return self.consume_lookup("declarations", "", offset,
                                   page_result(rows, offset, field="declarations"))
