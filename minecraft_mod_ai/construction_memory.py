"""Exact, recoverable evidence with a bounded active view for fresh requests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .construction_state import PAGE_SIZE, page_result

_FIELDS = {'symbols': 'matches', 'scopes': 'scopes', 'declarations': 'declarations',
           'nodes': 'nodes', 'source': 'source', 'bindings': 'bindings'}

# All retrieved facts remain in records/checkpoints. Only the currently displayed
# evidence is bounded: otherwise a few large Java source observations can make a
# single JSON request exceed the active 32K llama.cpp context even after the
# generic message fitter has run.
_MAX_ACTIVE_EVIDENCE_BYTES = 16 * 1024


def _identity(selector, row):
    kind = selector['kind']
    if kind in {'symbols', 'declarations'}:
        identity = row['symbol_id']
    elif kind == 'nodes':
        identity = row['node_id']
    else:
        identity = [selector['query'], selector['offset'] + selector['index']]
    digest = hashlib.sha256(json.dumps([kind, identity], ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()[:24]
    return 'evidence:' + digest


class ConstructionMemory:
    def __init__(self):
        self.records = {}
        self.active = []
        self.index_offset = 0
        self.visited_pages = set()
        self.require_recall = False

    def reset_navigation(self):
        # New accepted work may need the same facts. Only navigation within
        # one phase/revision is deduplicated; exact evidence is never deleted.
        self.visited_pages.clear()
        self.require_recall = False

    def invalidate_nodes(self, node_ids):
        """A revised node/query is no longer evidence about the current graph."""
        node_ids = set(node_ids)
        stale = {key for key, view in self.records.items()
                 if view['selector']['kind'] == 'nodes'
                 and node_ids.intersection(part.strip() for part in view['selector']['query'].split(','))}
        if not stale:
            return
        self.records = {key: value for key, value in self.records.items() if key not in stale}
        self.active = [key for key in self.active if key not in stale]
        self.index_offset = ((len(self.records) - 1) // PAGE_SIZE) * PAGE_SIZE if self.records else 0
        self.reset_navigation()

    def _page_identity(self, offset):
        return tuple(list(self.records)[offset:offset + PAGE_SIZE])

    def _active_bytes(self):
        return sum(len(json.dumps(self.records[key], ensure_ascii=False,
                                  separators=(',', ':')).encode('utf-8'))
                   for key in self.active)

    def _enforce_active_budget(self):
        # Evicted evidence remains addressable through memory_index/recall and
        # preserved by export/restore. A single oversized fact is kept intact
        # rather than being silently truncated or replaced with a false summary.
        while len(self.active) > 1 and self._active_bytes() > _MAX_ACTIVE_EVIDENCE_BYTES:
            self.active.pop(0)

    def _activate(self, key):
        if key in self.active:
            self.active.remove(key)
        self.active.append(key)
        self.active = self.active[-PAGE_SIZE:]
        self._enforce_active_budget()

    def remember(self, kind, query, offset, result):
        field = _FIELDS.get(kind)
        if field is None:
            return
        for index, row in enumerate(result[field]):
            selector = {'kind': kind, 'query': query.strip(), 'offset': offset, 'index': index}
            key = _identity(selector, row)
            self.records[key] = {'evidence_id': key, 'selector': selector, 'record': deepcopy(row)}
            self._activate(key)
        if self.records:
            self.index_offset = ((len(self.records) - 1) // PAGE_SIZE) * PAGE_SIZE
            self.require_recall = False

    def recall(self, key):
        if key not in self.records or key in self.active:
            raise ValueError('CONSTRUCTION_MEMORY_RECALL_INVALID')
        self._activate(key)
        self.require_recall = False

    def catalogue(self):
        result = page_result(list(self.records), self.index_offset, field='records')
        headers = []
        for key in result['records']:
            view = self.records[key]
            row, selector = view['record'], view['selector']
            identity = (row.get('symbol_id') or row.get('scope_id') or row.get('node_id') or row.get('signature')
                        if isinstance(row, dict) else '')
            headers.append({'evidence_id': key, **selector, 'identity': identity or selector['query'],
                            'active': key in self.active})
        result['records'] = headers
        return result

    def index_choices(self):
        if self.require_recall:
            return []
        keys = tuple(self.records)
        unread = [offset for offset in range(0, len(self.records), PAGE_SIZE)
                  if offset != self.index_offset
                  and any(key not in self.active for key in keys[offset:offset + PAGE_SIZE])]
        # A directory page containing only active facts adds no information.
        # Revisited pages remain reachable, but must lead to an actual recall
        # before another directory-only action can repeat the cursor cycle.
        before = [offset for offset in unread if offset < self.index_offset]
        after = [offset for offset in unread if offset > self.index_offset]
        return ([before[-1]] if before else []) + ([after[0]] if after else [])

    def select_index(self, offset):
        if offset not in self.index_choices():
            raise ValueError('CONSTRUCTION_MEMORY_INDEX_INVALID')
        self.visited_pages.add(self._page_identity(self.index_offset))
        self.index_offset = offset
        self.require_recall = self._page_identity(offset) in self.visited_pages
        self.visited_pages.add(self._page_identity(offset))

    def payload(self):
        return {'working_set': [deepcopy(self.records[key]) for key in self.active],
                'catalogue': self.catalogue(), 'index_offsets': self.index_choices(),
                'recall_before_more_navigation': self.require_recall,
                'meaning': 'Exact previously inspected facts; recall navigates memory, not a new search.'}

    def visible_symbols(self):
        return [view['record'] for key in self.active
                if (view := self.records[key])['selector']['kind'] in {'symbols', 'declarations'}]

    def visible_scopes(self):
        pages = {}
        for key in reversed(self.active):
            view = self.records[key]
            selector = view['selector']
            if selector['kind'] == 'scopes':
                page = (selector['query'], selector['offset'])
                pages.setdefault(page, []).append(view)
        return [view['record'] for rows in pages.values()
                for view in sorted(rows, key=lambda row: row['selector']['index'])]

    def export(self):
        return {'records': [{'evidence_id': key, **view['selector']} for key, view in self.records.items()],
                'active': list(self.active), 'index_offset': self.index_offset,
                'visited_pages': [list(page) for page in sorted(self.visited_pages)],
                'require_recall': self.require_recall}

    def restore(self, record, resolve_lookup):
        if (not isinstance(record, dict) or set(record) != {
                'records', 'active', 'index_offset', 'visited_pages', 'require_recall'}
                or not isinstance(record['records'], list) or not isinstance(record['active'], list)
                or len(record['active']) > PAGE_SIZE
                or (record['records'] and not callable(resolve_lookup))):
            raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
        self.records = {}
        pages = {}
        for view in record['records']:
            if (not isinstance(view, dict) or set(view) != {'evidence_id', 'kind', 'query', 'offset', 'index'}
                    or view['kind'] not in _FIELDS or not isinstance(view['query'], str)
                    or type(view['offset']) is not int or view['offset'] < 0
                    or type(view['index']) is not int or view['index'] < 0):
                raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
            selector = {key: view[key] for key in ('kind', 'query', 'offset', 'index')}
            page_key = (view['kind'], view['query'], view['offset'])
            if page_key not in pages:
                pages[page_key] = resolve_lookup(*page_key)
            rows = pages[page_key][_FIELDS[view['kind']]]
            if view['index'] >= len(rows):
                raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
            row = rows[view['index']]
            key = _identity(selector, row)
            if key != view['evidence_id'] or key in self.records:
                raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
            self.records[key] = {'evidence_id': key, 'selector': selector, 'record': deepcopy(row)}
        active = record['active']
        if (any(not isinstance(key, str) or key not in self.records for key in active)
                or len(set(active)) != len(active)):
            raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
        offset = record['index_offset']
        if (type(offset) is not int or offset < 0 or offset % PAGE_SIZE
                or (offset and offset >= len(self.records))):
            raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
        pages = record['visited_pages']
        keys = list(self.records)
        positions = {key: index for index, key in enumerate(keys)}
        if (not isinstance(pages, list) or any(
                not isinstance(page, list) or not 1 <= len(page) <= PAGE_SIZE
                or any(not isinstance(key, str) or key not in positions for key in page)
                or positions[page[0]] % PAGE_SIZE
                or keys[positions[page[0]]:positions[page[0]] + len(page)] != page
                for page in pages)
                or len({tuple(page) for page in pages}) != len(pages)
                or type(record['require_recall']) is not bool):
            raise ValueError('CONSTRUCTION_MEMORY_CHECKPOINT_INVALID')
        self.active, self.index_offset = list(active), offset
        self.visited_pages = {tuple(page) for page in pages}
        self.require_recall = record['require_recall']
        # Older checkpoints could contain a large eight-record active view.
        # Preserve those records, but do not re-inject their entire contents
        # into the first resumed llama.cpp request.
        self._enforce_active_budget()
