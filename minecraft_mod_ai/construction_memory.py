"""Exact, recoverable evidence with a bounded active view for fresh requests."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .construction_state import PAGE_SIZE, page_result


_FIELDS = {'symbols': 'matches', 'declarations': 'declarations',
           'nodes': 'nodes', 'source': 'source', 'bindings': 'bindings'}


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

    def _activate(self, key):
        if key in self.active:
            self.active.remove(key)
        self.active.append(key)
        self.active = self.active[-PAGE_SIZE:]

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

    def recall(self, key):
        if key not in self.records or key in self.active:
            raise ValueError('CONSTRUCTION_MEMORY_RECALL_INVALID')
        self._activate(key)

    def catalogue(self):
        result = page_result(list(self.records), self.index_offset, field='records')
        headers = []
        for key in result['records']:
            view = self.records[key]
            row, selector = view['record'], view['selector']
            identity = (row.get('symbol_id') or row.get('node_id') or row.get('signature')
                        if isinstance(row, dict) else '')
            headers.append({'evidence_id': key, **selector, 'identity': identity or selector['query'],
                            'active': key in self.active})
        result['records'] = headers
        return result

    def index_choices(self):
        offsets = []
        if self.index_offset:
            offsets.append(self.index_offset - PAGE_SIZE)
        if self.index_offset + PAGE_SIZE < len(self.records):
            offsets.append(self.index_offset + PAGE_SIZE)
        return offsets

    def select_index(self, offset):
        if offset not in self.index_choices():
            raise ValueError('CONSTRUCTION_MEMORY_INDEX_INVALID')
        self.index_offset = offset

    def payload(self):
        return {'working_set': [deepcopy(self.records[key]) for key in self.active],
                'catalogue': self.catalogue(), 'index_offsets': self.index_choices(),
                'meaning': 'Exact previously inspected facts; recall navigates memory, not a new search.'}

    def visible_symbols(self):
        return [view['record'] for key in self.active
                if (view := self.records[key])['selector']['kind'] in {'symbols', 'declarations'}]

    def export(self):
        return {'records': [{'evidence_id': key, **view['selector']} for key, view in self.records.items()],
                'active': list(self.active), 'index_offset': self.index_offset}

    def restore(self, record, resolve_lookup):
        if (not isinstance(record, dict) or set(record) != {'records', 'active', 'index_offset'}
                or not isinstance(record['records'], list) or not isinstance(record['active'], list)
                or len(record['active']) > PAGE_SIZE or not callable(resolve_lookup)):
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
        self.active, self.index_offset = list(active), offset
