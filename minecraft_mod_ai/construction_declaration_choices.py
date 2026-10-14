"""Native declaration choices derived from the inspected page and owner plan.

An instruction is not a type/ownership constraint. Keep action-specific
schemas closed and expose only choices the current phase can actually accept.
"""
from __future__ import annotations
from .construction_type_contracts import owner_package

import string

from .construction_author import ID, obj, ask
from .construction_state import PAGE_SIZE, STATE_INSTRUCTION, symbol_query_key
from .java_operation_ir import KEYWORDS, PRIMITIVES

# This limits regex factoring, not identifiers, queries or the excluded set.
# llama.cpp's pinned compiler accepts at most 100 nested pattern groups.
_MAX_TRIE_GROUP_DEPTH = 96
# Pattern atoms are emitted directly inside a quoted JSON value by llama.cpp.
# Queries are symbol selectors, so raw JSON delimiters/escapes are unnecessary.
_SYMBOL_QUERY_ALPHABET = ''.join(chr(i) for i in range(33, 127)
                                if chr(i) not in {'"', '\\'})


def _character_class(chars):
    def escape(char):
        # Python re.escape also escapes '~', '$', '.', etc. llama.cpp rejects
        # those class escapes and silently substitutes an unbounded string.
        return '\\x' + format(ord(char), '02x') if char in '\\[]^-' else char
    runs = []
    values = sorted(ord(char) for char in chars)
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        runs.append((start, end))
        start = end = value
    runs.append((start, end))
    parts = []
    for start, end in runs:
        if end - start >= 2:
            parts.append(escape(chr(start)) + '-' + escape(chr(end)))
        else:
            parts.extend(escape(chr(value)) for value in range(start, end + 1))
    return '[' + ''.join(parts) + ']'


def string_excluding(forbidden, *, alphabet, first=None, minimum=1, maximum=240,
                     fold_case=False, normalizer=None):
    """Finite trie complement; no unsupported lookahead/not or unbounded regex.

    Each branch either ends at an allowed prefix, takes a forbidden-word edge,
    or diverges from that trie and fills only its remaining length budget.
    """
    first = set(first or alphabet)
    alphabet = set(alphabet)
    normalize = normalizer or (str.casefold if fold_case else lambda value: value)
    trie = {}
    for value in forbidden:
        value = normalize(value)
        if not value:
            if minimum == 0:
                trie[''] = True
            continue
        if len(value) > maximum:
            continue
        if any(not any(normalize(char) == key for char in (first if i == 0 else alphabet))
               for i, key in enumerate(value)):
            continue
        node = trie
        for key in value:
            node = node.setdefault(key, {})
        node[''] = True
    tail = _character_class(alphabet)

    def choices(node, depth):
        options = [''] if depth >= minimum and '' not in node else []
        edges = []
        if depth < maximum:
            allowed = first if depth == 0 else alphabet
            other = {char for char in allowed if normalize(char) not in node}
            remaining = maximum - depth - 1
            if other and remaining >= max(0, minimum - depth - 1):
                option = _character_class(other)
                if remaining:
                    option += tail + '{' + str(max(0, minimum - depth - 1)) + ',' + str(remaining) + '}'
                options.append(option)
            for key in sorted(key for key in node if key):
                chars = {char for char in allowed if normalize(char) == key}
                if not chars:
                    continue
                edges.append((_character_class(chars), node[key]))
        return options, edges

    def combine(options):
        if not options:
            return None
        return options[0] if len(options) == 1 else '(?:' + '|'.join(options) + ')'

    def flat_suffix(node, depth):
        options = []
        pending = [(node, depth, '')]
        while pending:
            current, position, prefix = pending.pop()
            endings, edges = choices(current, position)
            options.extend(prefix + ending for ending in endings)
            pending.extend((child, position + 1, prefix + char)
                           for char, child in reversed(edges))
        return combine(options)

    def suffix(node, depth):
        if depth >= _MAX_TRIE_GROUP_DEPTH:
            # Flatten only the deep tail. Every excluded word and length bound
            # remains active without crossing the native nesting limit.
            return flat_suffix(node, depth)
        options, edges = choices(node, depth)
        for char, child in edges:
            rest = suffix(child, depth + 1)
            if rest is not None:
                options.append(char + rest)
        return combine(options)

    return {'type': 'string', 'minLength': minimum, 'maxLength': maximum,
            'pattern': '^' + suffix(trie, 0) + '$'}


def identifier_schema(excluded=()):
    first = string.ascii_letters + '_$'
    return string_excluding(set(excluded) | KEYWORDS,
                            alphabet=first + string.digits, first=first)


def action_schema(action, properties=None):
    return obj({'action': {'const': action}, **(properties or {})})


def choice_actions(schema, *, default_action=None):
    actions = []
    for branch in schema.get('oneOf', [schema]):
        properties = branch['properties']
        if 'action' in properties:
            shape = properties['action']
            actions.extend([shape['const']] if 'const' in shape else shape['enum'])
        elif 'complete' in properties:
            actions.append('complete' if properties['complete']['const'] else 'parameter')
        elif default_action:
            actions.append(default_action)
        else:
            raise ValueError('CONSTRUCTION_CHOICE_ACTION_REQUIRED')
    return list(dict.fromkeys(actions))


def symbol_lookup_schemas(state):
    if state.evidence_index is not None:
        goal = state.symbol_goal
        branches = []
        if goal is not None and ('symbols', symbol_query_key(goal), 0) not in state.requests:
            branches.append(action_schema('symbols', {'query': {'const': goal}, 'offset': {'const': 0}}))
        for row in state.symbol_page_successors():
            branches.append(action_schema('more_symbols', {
                'query': {'const': row['query']}, 'offset': {'const': row['offset']}}))
        candidates = state.scope_candidates()
        if candidates:
            branches.append(action_schema('scopes', {
                'query': {'enum': [row['goal_id'] for row in candidates]}, 'offset': {'const': 0}}))
        descriptor = state.evidence_index.goals.get(goal)
        for row in state.lookup_successors('scopes'):
            scope_descriptor = state.evidence_index.goals[row['query']]
            if descriptor and {**scope_descriptor, 'scope': 'package:'} == {**descriptor, 'scope': 'package:'}:
                branches.append(action_schema('scopes', {
                    'query': {'const': row['query']}, 'offset': {'const': row['next_offset']}}))
        return branches
    consumed = {query for kind, query, _offset in state.requests if kind == 'symbols'}
    # The plan/own declarations already have their own view. Dependency lookup
    # cannot manufacture another instance of this uninstalled class.
    blocked = consumed | {symbol_query_key(state.owner), symbol_query_key(state.owner.rsplit('.', 1)[-1])}
    for key in state.declaration_ids:
        blocked.update((symbol_query_key(key), symbol_query_key(key.rsplit('#', 1)[-1])))
    query = string_excluding(blocked, alphabet=_SYMBOL_QUERY_ALPHABET,
                             minimum=2, normalizer=symbol_query_key)
    branches = [action_schema('symbols', {'query': query})]
    for row in state.lookup_successors('symbols'):
        branches.append(action_schema('more_symbols', {
            'query': {'const': row['query']}, 'offset': {'const': row['next_offset']}}))
    return branches


def binding_lookup_schemas(state):
    branches = []
    for key, rows in sorted(state.binding_pages.items()):
        consumed = {offset for kind, query, offset in state.requests
                    if kind == 'bindings' and query == key}
        consumed.update(offset for query, offset in state.binding_lookups if query == key)
        offset = max(consumed) + PAGE_SIZE if consumed else 0
        # First page and then its exact successor; consumed pages and arbitrary
        # offsets never become model choices. Empty bindings have one empty page.
        if offset and offset >= len(rows):
            continue
        branches.append(action_schema('bindings', {'query': {'const': key}, 'offset': {'const': offset}}))
    return branches


def with_binding_lookups(state, schema):
    branches = binding_lookup_schemas(state)
    return {'oneOf': [*schema.get('oneOf', [schema]), *branches]} if branches else schema


def consume_lookup_choice(state, choice):
    if choice.get('action') == 'scopes':
        state.lookup_scopes(choice['query'], choice['offset'])
        return True
    if choice.get('action') in {'symbols', 'more_symbols'}:
        state.lookup_symbols(choice['query'], choice.get('offset', 0))
        return True
    if choice.get('action') == 'bindings':
        state.lookup_bindings(choice['query'], choice['offset'])
        return True
    return False


def member_schema(state, *, reserved_names, body_candidates):
    known_names = {state.symbols[key]['name'] for key in state.declaration_ids}
    helpers = sorted({state.symbols[key]['name'] for key in state.declaration_ids
                      if state.symbols[key]['kind'] == 'METHOD'
                      and state.symbols[key]['name'] not in reserved_names})
    branches = [obj({'action': {'enum': ['method', 'field']},
                     'name': identifier_schema(known_names | set(reserved_names)), 'purpose': ID}),
                action_schema('constructor', {'purpose': ID}),
                *binding_lookup_schemas(state)]
    # Choosing which helper is missing does not need dependency signatures.
    # Resolve its types after the declaration is selected, and its calls while
    # authoring the body. An open dependency browser here can consume the whole
    # inventory without ever selecting a declaration or an assigned body.
    declaration_offset = state.next_declaration_offset()
    if declaration_offset is not None:
        branches.append(action_schema('declarations', {'offset': {'const': declaration_offset}}))
    if helpers:
        branches.append(action_schema('overload', {'name': {'enum': helpers}, 'purpose': ID}))
    if body_candidates:
        branches.append(action_schema('begin_bodies', {'symbol_ref': {'enum': body_candidates}}))
    return {'oneOf': branches}


def parameter_schema(parameters, *, state, can_complete=True):
    branches = [obj({'complete': {'const': False},
                     'name': identifier_schema(p['name'] for p in parameters)})]
    if can_complete:
        branches.append(obj({'complete': {'const': True}}))
    return with_binding_lookups(state, {'oneOf': branches})


def type_candidates(state):
    types = set(PRIMITIVES) - {'void'}
    visible = [state.symbols[state.owner], *state.symbol_lookup.get('matches', ())]
    for row in visible:
        names = [row.get('owner', ''), row.get('return_type', '')]
        if row.get('kind') == 'CLASS':
            names.append(row.get('symbol_id', row.get('owner', '')))
        for name in names:
            while name.endswith('[]'):
                name = name[:-2]
            if name in types or accessible_type(state, state.symbols.get(name, {})):
                types.add(name)
    return sorted(types)


def choose_type(router, state, payload, *, phase, allow_void=False):
    state.enter(phase)
    state.prepare_symbols('\n'.join(str(payload[key]) for key in
        ('intent', 'purpose', 'name', 'parameter_name', 'parameters', 'effects', 'state_reads', 'state_writes')
        if payload.get(key)), kinds=('CLASS',))
    instruction = ('Choose only the current declaration type. No source code. Use an intrinsic primitive '
        'or inspected type_candidates. The host ranked real types from this member purpose and source/type '
        'relationships. scopes follows the real package/type structure by meaning; more_symbols reads '
        'the literal next member/type page if needed. Do not invent a type query. '
        'type binds an actual base type and array dimensions (zero means scalar). void is available '
        'only for a method result. This decision does not add another member or finish a body. ' + STATE_INSTRUCTION)
    while True:
        types = type_candidates(state)
        branches = [action_schema('type', {'base_type': {'type': 'string', 'enum': types},
                                          'array_dimensions': {'type': 'integer', 'minimum': 0, 'maximum': 255}}),
                    *symbol_lookup_schemas(state), *binding_lookup_schemas(state)]
        # JVMS 4.11 bounds array dimensions; void is never an array/component.
        if allow_void:
            branches.append(action_schema('void'))
        schema = {'oneOf': branches}
        choice = ask(router, 'construction_type_next',
            {**payload, **state.payload(choice_actions(schema), type_candidates=types)}, schema, instruction)
        if consume_lookup_choice(state, choice):
            continue
        state.advance()
        if choice['action'] == 'void':
            return 'void'
        return choice['base_type'] + '[]' * choice['array_dimensions']


def accessible_type(state, row):
    if row.get('kind') != 'CLASS':
        return False
    if row.get('side') == 'CLIENT' and not state.target_path.startswith('src/client/'):
        return False
    access = row.get('access', 0)
    same_package = owner_package(state.symbols, row.get('owner', '')) == owner_package(state.symbols, state.owner)
    return bool(access & 1 or (same_package and not access & 2))


def header_candidates(state):
    candidates = {'superclass': [], 'interface': []}
    header = state.symbols[state.owner]
    for match in state.symbol_lookup.get('matches', ()):
        key = match['symbol_id']
        row = state.symbols.get(key, {})
        if (key == state.owner or row.get('provenance') == 'construction_plan'
                or not accessible_type(state, row)):
            continue
        access = row.get('access', 0)
        if access & 0x0200:
            if key not in header['interfaces']:
                candidates['interface'].append(key)
        elif not access & 0x0010 and not header['superclass']:
            candidates['superclass'].append(key)
    return {kind: sorted(set(keys)) for kind, keys in candidates.items()}


def header_schema(state, candidates):
    branches = [*symbol_lookup_schemas(state), *binding_lookup_schemas(state),
                action_schema('finish_header')]
    for kind, keys in candidates.items():
        if keys:
            branches.append(action_schema(kind, {'type_ref': {'type': 'string', 'enum': keys}}))
    return {'oneOf': branches}
