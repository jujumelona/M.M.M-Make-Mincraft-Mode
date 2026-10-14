"""Keep every accepted streamed subdecision inspectable without a long prompt."""
from __future__ import annotations

from .construction_author import ask
from .construction_state import PAGE_SIZE, page_result
from .construction_declaration_choices import action_schema, choice_actions
from .construction_operation_choices import operation_context_branches


def sequence_item(router, state, name, current, schema, values, *, nodes, source_units,
                  pinned_source, read_lookup, save_author, instruction):
    offset = ((len(values) - 1) // PAGE_SIZE) * PAGE_SIZE if values else 0
    while True:
        page = page_result(values, offset, field='accepted')
        offsets = ([offset - PAGE_SIZE] if offset else [])
        if page['next_offset'] is not None:
            offsets.append(page['next_offset'])
        branches = [*schema.get('oneOf', [schema]),
                    *[action_schema('sequence_page', {'offset': {'const': value}}) for value in offsets],
                    *operation_context_branches(state, nodes=nodes, source_units=source_units,
                                                pinned_source=pinned_source)]
        decision_schema = {'oneOf': branches}
        cursor = {key: current['construction_state'][key]
                  for key in ('current_member', 'graph', 'sources') if key in current.get('construction_state', {})}
        choice = ask(router, name,
            {**current, **state.payload(choice_actions(decision_schema), **cursor), 'accepted_sequence': page},
            decision_schema, instruction + ' Every accepted entry is retained; sequence_page reads its exact ordered page. ')
        if choice.get('action') == 'sequence_page':
            offset = choice['offset']
            continue
        if choice.get('action') in {'symbols', 'more_symbols', 'scopes', 'bindings', 'source',
                                    'sources', 'declarations', 'nodes', 'node_index', 'memory_index', 'recall'}:
            read_lookup(choice)
            save_author()
            continue
        return choice


def unused_integer_labels(minimum, maximum, values):
    """Positive disjoint intervals; native decoding cannot repeat a label."""
    ranges, start = [], minimum
    for value in sorted(set(values)):
        if value < minimum or value > maximum:
            continue
        if start < value:
            ranges.append({'type': 'integer', 'minimum': start, 'maximum': value - 1})
        start = value + 1
    if start <= maximum:
        ranges.append({'type': 'integer', 'minimum': start, 'maximum': maximum})
    return {'oneOf': ranges} if len(ranges) > 1 else ranges[0] if ranges else None
