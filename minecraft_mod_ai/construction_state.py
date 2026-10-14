"""Bounded, host-owned state shared by declaration and operation authors.

Planned declarations are usable within their owner, but are not evidence of
already installed bytecode. The complete inventory and accepted plan stay on
the host; every decision receives its phase, cursor and lookup provenance.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy


# Body lookup choices and retained API pages changed. Old journals/cursors may
# not be replayed against them; declaration-bound accepted graphs stay reusable.
PROTOCOL_VERSION = "mmm/construction-state-v7"
PAGE_SIZE = 8
RECENT_SIZE = 4
LOOKUP_KINDS = frozenset({'symbols', 'declarations', 'nodes', 'node_index', 'source', 'sources', 'bindings'})
TYPE_LOOKUP_PHASES = frozenset({'class_header', 'member_return_type', 'member_parameter_type'})


def symbol_query_key(query):
    # Bytecode uses '$' for nested owners; inspected source/type names use '.'.
    # They are one selector, including for consumed-query bookkeeping.
    return query.strip().casefold().replace('$', '.')


def _lookup_request(kind, query, offset):
    if kind not in LOOKUP_KINDS or not isinstance(query, str) or type(offset) is not int or offset < 0:
        raise ValueError('CONSTRUCTION_LOOKUP_REQUEST_INVALID')
    normalized = (symbol_query_key(query) if kind == 'symbols' else
                  query.strip().casefold() if kind == 'sources' else query.strip())
    if kind == 'nodes':
        normalized = ','.join(sorted(key.strip() for key in query.split(',')))
    if kind == 'declarations':
        normalized = ''
    return kind, normalized, offset


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


def symbol_page(symbols, query, offset, *, kinds=None):
    query = query.strip()
    if len(query) < 2:
        raise ValueError("CONSTRUCTION_LOOKUP_QUERY_REQUIRED")
    folded = symbol_query_key(query)

    def rank(key):
        row = symbols[key]
        owner = symbol_query_key(str(row.get('owner', '')))
        name = symbol_query_key(str(row.get('name', '')))
        simple_owner = owner.rsplit('.', 1)[-1].rsplit('$', 1)[-1]
        if symbol_query_key(key) == folded or (row.get('kind') == 'CLASS'
                and (folded in {name, simple_owner} or owner.endswith('.' + folded))):
            return 0, key
        if name == folded:
            return 1, key
        if folded in {owner, simple_owner}:
            return 2, key
        return 3, key

    # The first page must expose an exact type before unrelated fields or
    # method descriptors that merely mention it. Keep every remaining result
    # and a stable order so subsequent pages still exhaust the same inventory.
    matches = sorted((key for key, row in symbols.items()
                      if row.get('provenance') != 'construction_plan'
                      and (kinds is None or row.get('kind') in kinds)
                      and (folded in symbol_query_key(key)
                           or folded == symbol_query_key(str(row.get('name', ''))))),
                     key=rank)
    # Materialize only this page, never the whole inspected inventory.
    result = page_result(matches, offset, field="matches")
    result["matches"] = [symbol_record(key, symbols[key]) for key in result["matches"]]
    return {"kind": "symbols", "query": query, **result,
            **({'symbol_kinds': sorted(kinds)} if kinds is not None else {})}


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
    "add only missing declarations. authoring_contract defines the artifact's responsibility, "
    "identities and implementation requirements. Use bindings with an exact binding_id from "
    "authoring_contract.binding_pages to inspect shared configuration before choosing declarations "
    "or types. binding_lookup retains the latest complete configuration page across phases. "
    "Select only available_actions."
)


class ConstructionState:
    def __init__(self, symbols, owner, target_path, *, declaration_ids=(), revision=0,
                 authoring_contract=None, binding_pages=None):
        self.symbols = symbols
        self.owner = owner
        self.target_path = target_path
        self.authoring_contract = dict(authoring_contract or {})
        self.binding_pages = dict(binding_pages or {})
        self.declaration_ids = list(declaration_ids)
        self.revision = revision
        self.phase = ""
        self.lookup = {}
        self.symbol_lookup = {}
        self.binding_lookup = {}
        self.history = deque(maxlen=PAGE_SIZE)
        self.requests = set()
        self.lookup_cursors = {}

    def enter(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.requests.clear()
            self.lookup_cursors.clear()

    def advance(self, *, preserve_symbols=False):
        self.revision += 1
        self.lookup = {}
        if not preserve_symbols:
            self.symbol_lookup = {}
        self.requests.clear()
        self.lookup_cursors.clear()

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
        }, "lookup": self.lookup, "authoring_contract": self.authoring_contract,
            "binding_lookup": self.binding_lookup,
            "symbol_lookup": self.symbol_lookup if self.lookup.get('kind') != 'symbols' else {}}

    def _remember_cursor(self, kind, query, offset, result):
        request = _lookup_request(kind, query, offset)
        self.lookup_cursors[request[:2]] = {
            'kind': kind, 'query': query.strip(), 'offset': offset,
            'next_offset': result['next_offset'],
        }

    def lookup_successors(self, kind):
        pages = {key: row for key, row in self.lookup_cursors.items() if key[0] == kind}
        # A real dependency page remains usable after an accepted graph node.
        # Its exact successor is still resolved from the same target inventory.
        if kind == 'symbols' and self.symbol_lookup:
            row = self.symbol_lookup
            key = _lookup_request(kind, row['query'], row['offset'])[:2]
            pages.setdefault(key, row)
        return [row for row in pages.values() if row['next_offset'] is not None
                and _lookup_request(kind, row['query'], row['next_offset']) not in self.requests]

    def consume_lookup(self, kind, query, offset, result):
        request = _lookup_request(kind, query, offset)
        if request in self.requests:
            raise ValueError("CONSTRUCTION_AUTHOR_NO_PROGRESS: repeated " + kind + " lookup")
        self.requests.add(request)
        self.lookup = {**result, "kind": kind, "query": query.strip(), "offset": offset}
        self._remember_cursor(kind, query, offset, result)
        if kind == 'symbols':
            self.symbol_lookup = self.lookup
        if kind == 'bindings':
            self.binding_lookup = self.lookup
        self.history.append({"phase": self.phase, "revision": self.revision,
            "kind": kind, "query": query.strip(), "offset": offset,
            "status": result["status"], "total_count": result["total_count"],
            "next_offset": result["next_offset"]})
        return self.lookup

    def export(self):
        # Save selectors, never cached source or API rows as fresh authority.
        last = {key: self.lookup[key] for key in ('kind', 'query', 'offset')} if self.lookup else None
        binding = {key: self.binding_lookup[key] for key in ('kind', 'query', 'offset')} if self.binding_lookup else None
        symbol = {key: self.symbol_lookup[key] for key in ('kind', 'query', 'offset')} if self.symbol_lookup else None
        return {'protocol': PROTOCOL_VERSION, 'phase': self.phase, 'revision': self.revision,
                'requests': [list(row) for row in sorted(self.requests)],
                'history': deepcopy(list(self.history)), 'last_lookup': last,
                'last_binding_lookup': binding, 'last_symbol_lookup': symbol,
                'lookup_cursors': [{key: row[key] for key in ('kind', 'query', 'offset')}
                                   for _, row in sorted(self.lookup_cursors.items())]}

    def restore(self, record, *, resolve_lookup=None):
        if (not isinstance(record, dict) or record.get('protocol') != PROTOCOL_VERSION
                or record.get('phase') != self.phase or type(record.get('revision')) is not int
                or record['revision'] != self.revision):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_BINDING_MISMATCH')
        requests, history = record.get('requests'), record.get('history')
        if (not isinstance(requests, list) or not isinstance(history, list)
                or len(history) > PAGE_SIZE
                or any(not isinstance(row, list) or len(row) != 3 for row in requests)
                or any(not isinstance(row, dict) for row in history)):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        restored = {_lookup_request(*row) for row in requests}
        if len(restored) != len(requests):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        cursors = record.get('lookup_cursors')
        if (not isinstance(cursors, list) or any(
                not isinstance(row, dict) or set(row) != {'kind', 'query', 'offset'}
                or _lookup_request(row['kind'], row['query'], row['offset']) not in restored
                for row in cursors)):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        cursor_requests = [_lookup_request(row['kind'], row['query'], row['offset']) for row in cursors]
        latest = {}
        for kind, query, offset in restored:
            latest[(kind, query)] = max(offset, latest.get((kind, query), -1))
        if {(kind, query): offset for kind, query, offset in cursor_requests} != latest or len(cursors) != len(latest):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        if cursors and not callable(resolve_lookup):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_LOOKUP_RESOLVER_REQUIRED')
        last = record.get('last_lookup')
        if last is not None and (not isinstance(last, dict) or set(last) != {'kind', 'query', 'offset'}
                or _lookup_request(last['kind'], last['query'], last['offset']) not in restored):
            raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
        self.requests = restored
        self.history = deque(deepcopy(history), maxlen=PAGE_SIZE)
        self.lookup_cursors = {}
        for row in cursors:
            result = resolve_lookup(row['kind'], row['query'], row['offset'])
            self._remember_cursor(row['kind'], row['query'], row['offset'], result)
        # The caller re-resolves this selector against current inspected inputs.
        self.lookup = {}
        self.symbol_lookup = {}
        binding = record.get('last_binding_lookup')
        if binding is not None:
            if (not isinstance(binding, dict) or set(binding) != {'kind', 'query', 'offset'}
                    or binding['kind'] != 'bindings'):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(binding['kind'], binding['query'], binding['offset'])
            self.binding_lookup = {**self.binding_result(binding['query'], binding['offset']), **binding}
        else:
            self.binding_lookup = {}
        symbol = record.get('last_symbol_lookup')
        if symbol is not None:
            if (not isinstance(symbol, dict) or set(symbol) != {'kind', 'query', 'offset'}
                    or symbol['kind'] != 'symbols'):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
            _lookup_request(symbol['kind'], symbol['query'], symbol['offset'])
            self.symbol_lookup = {**self.symbol_result(symbol['query'], symbol['offset']), **symbol}
        return last

    def symbol_result(self, query, offset):
        kinds = {'CLASS'} if self.phase in TYPE_LOOKUP_PHASES else None
        return symbol_page(self.symbols, query, offset, kinds=kinds)

    def lookup_symbols(self, query, offset):
        return self.consume_lookup("symbols", query, offset,
                                   self.symbol_result(query, offset))

    def next_declaration_offset(self):
        # The schema and reader share this cursor. Every page remains reachable,
        # but an already consumed page cannot be offered as a fresh decision.
        for offset in range(0, max(1, len(self.declaration_ids)), PAGE_SIZE):
            if ('declarations', '', offset) not in self.requests:
                return offset
        return None

    def lookup_declarations(self, offset):
        rows = [symbol_record(key, self.symbols[key]) for key in self.declaration_ids]
        return self.consume_lookup("declarations", "", offset,
                                   page_result(rows, offset, field="declarations"))

    def binding_result(self, query, offset):
        if query not in self.binding_pages:
            raise ValueError('CONSTRUCTION_BINDING_MISSING')
        return page_result(self.binding_pages[query], offset, field='bindings')

    def lookup_bindings(self, query, offset):
        return self.consume_lookup('bindings', query, offset, self.binding_result(query, offset))
