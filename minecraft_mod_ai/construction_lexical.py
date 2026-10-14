"""Project actual lexical names at partial graph use-sites, including holes."""
from __future__ import annotations

from collections import Counter
from .java_operation_ir import parameter_records


def lexical_sites(nodes, declaration, symbols, *, slot=None):
    """Yield (node id, environment, write target) in Java execution order.

    Environments carry type, category, initialization and declaration identity.
    Missing forward references are holes, not invented declarations. A shared
    graph value is visited separately in every actual lexical use-site.
    """
    by_id = {node['node_id']: node for node in nodes}
    referenced = {ref for node in nodes for ref in node['input_refs']}
    roots = [key for key in by_id if key not in referenced]
    if slot is not None and slot not in by_id and slot not in referenced:
        roots.append(slot)
    initial = {row['name']: (row['type'], 'parameter', True, 'parameter:' + row['name'], False, 0)
               for row in parameter_records(declaration)}
    effects = {'mutations': Counter(), 'initialized': {value[3] for value in initial.values()}}
    captured, visited = set(), set()
    returns = 'void' if declaration['kind'] == 'CONSTRUCTOR' else declaration.get('return_type', 'void')
    stack = [('visit', key, initial, False, frozenset(), 0, frozenset(), returns, 0)
             for key in reversed(roots)]
    while stack:
        action, key, env, writing, ancestors, loops, captures, returns, switches = stack.pop()
        node = by_id.get(key)
        if action == 'for_body':
            stack.append(('visit', key, dict(env), False, ancestors, loops, captures, returns, switches))
            continue
        if action == 'declare':
            name = node['attributes']['name']
            if name in env:
                raise ValueError('CONSTRUCTION_LOCAL_DUPLICATE')
            assigned = bool(node['input_refs'])
            env[name] = (node['typed_result'], 'local', assigned, key, node['attributes'].get('final', False), loops)
            if assigned:
                effects['initialized'].add(key)
            continue
        if action == 'assigned':
            target = by_id.get(node['input_refs'][0])
            if target is not None and target['op'] == 'local':
                name = target['attributes']['name']
                if name in env:
                    type_, kind, assigned, binding, final, depth = env[name]
                    if final and (assigned or loops > depth):
                        raise ValueError('CONSTRUCTION_FINAL_ASSIGNMENT')
                    env[name] = (type_, kind, True, binding, final, depth)
                    effects['mutations'][binding] += 2 if loops else 1
            continue
        if key in ancestors:
            raise ValueError('CONSTRUCTION_NODE_CYCLE')
        if node is not None:
            visited.add(key)
            if node['op'] in {'local', 'parameter'}:
                binding = env.get(node['attributes']['name'])
                if binding and binding[3] in captures:
                    captured.add(binding[3])
        yield key, dict(env), writing, {'loops': loops, 'captures': captures, 'effects': effects,
                                      'returns': returns, 'switches': switches}
        if node is None:
            continue
        ancestors = ancestors | {key}
        op, refs, attrs = node['op'], node['input_refs'], node['attributes']

        def push(ref, scope=env, write=False, loop=loops, captured_names=captures, result=returns, switch=switches):
            stack.append(('visit', ref, scope, write, ancestors, loop, captured_names, result, switch))

        def bind(scope, name, type_, kind, binding):
            if name in scope:
                raise ValueError('CONSTRUCTION_LOCAL_DUPLICATE')
            scope[name] = (type_, kind, True, binding, False, loops)
            effects['initialized'].add(binding)

        if op == 'declare':
            stack.append(('declare', key, env, False, ancestors, loops, captures, returns, switches))
            for ref in reversed(refs):
                push(ref)
        elif op == 'block':
            child = dict(env)
            for ref in reversed(refs):
                push(ref, child)
        elif op == 'lambda':
            child = dict(env)
            types = symbols.get(node['typed_result'], {}).get('functional_parameters', ())
            for name, type_ in zip(attrs['parameters'], types):
                bind(child, name, type_, 'parameter', key + ':' + name)
            for ref in reversed(refs):
                push(ref, child, loop=0, switch=0,
                     captured_names=frozenset(value[3] for value in env.values()),
                     result=symbols[node['typed_result']]['functional_return_type'])
        elif op in {'if', 'switch'}:
            for ref in reversed(refs[1:]):
                push(ref, dict(env), switch=switches + (op == 'switch'))
            if refs:
                push(refs[0])
        elif op in {'while', 'do_while'}:
            push(refs[1], dict(env), loop=loops + 1)
            push(refs[0], loop=loops + 1)
        elif op == 'for':
            child = dict(env)
            push(refs[2], child, loop=loops + 1)
            stack.append(('for_body', refs[3], child, False, ancestors, loops + 1, captures, returns, switches))
            push(refs[1], child, loop=loops + 1)
            push(refs[0], child)
        elif op == 'foreach':
            child = dict(env)
            bind(child, attrs['name'], attrs['element_type'], 'local', key)
            push(refs[1], child, loop=loops + 1)
            push(refs[0])
        elif op == 'try':
            if attrs.get('finally'):
                push(refs[-1], dict(env))
            for catch, ref in reversed(list(zip(attrs.get('catches', ()), refs[1:]))):
                child = dict(env)
                bind(child, catch['name'], catch['type_ref'], 'local', key + ':' + catch['name'])
                push(ref, child)
            push(refs[0], dict(env))
        elif op == 'synchronized':
            push(refs[1], dict(env))
            push(refs[0])
        elif op == 'assign':
            stack.append(('assigned', key, env, False, ancestors, loops, captures, returns, switches))
            push(refs[1])
            push(refs[0], write=attrs.get('operator', '=') == '=')
        else:
            for ref in reversed(refs):
                push(ref)
    if visited != set(by_id):
        raise ValueError('CONSTRUCTION_NODE_CYCLE')
    if any(_capture_mutable(binding, effects) for binding in captured):
        raise ValueError('CONSTRUCTION_LAMBDA_CAPTURE_MUTABLE')


def _capture_mutable(binding, effects):
    count = effects['mutations'][binding]
    return bool(count and (binding in effects['initialized'] or count > 1))


def names_at_slot(nodes, declaration, symbols, slot):
    visible = None
    sites = [row for row in lexical_sites(nodes, declaration, symbols, slot=slot) if row[0] == slot]
    for key, env, writing, context in sites:
        if key != slot:
            continue
        bindings = {(name, type_, kind, identity)
                    for name, (type_, kind, assigned, identity, final, depth) in env.items()
                    if (assigned or writing)
                    and not (writing and final and (assigned or context['loops'] > depth))
                    and not (identity in context['captures'] and _capture_mutable(identity, context['effects']))}
        visible = bindings if visible is None else visible & bindings
    result = {'parameter': [], 'local': []}
    for name, type_, kind, _identity in sorted(visible or ()):
        result[kind].append((name, type_))
    return result


def declared_names_at_slot(nodes, declaration, symbols, slot):
    return {name for key, env, _writing, _context in lexical_sites(nodes, declaration, symbols, slot=slot)
            if key == slot for name in env}


def check_name_use(node, env, writing):
    if node['op'] not in {'parameter', 'local'}:
        return
    binding = env.get(node['attributes']['name'])
    if binding is None:
        raise ValueError('CONSTRUCTION_' + node['op'].upper() + '_SCOPE')
    if binding[0] != node['typed_result']:
        raise ValueError('CONSTRUCTION_LOCAL_SCOPE_TYPE')
    if not binding[2] and not writing:
        raise ValueError('CONSTRUCTION_LOCAL_UNINITIALIZED')


def check_partial_names(nodes, declaration, symbols):
    """Never checkpoint a name that contradicts its already known use-site."""
    by_id = {node['node_id']: node for node in nodes}
    for key, env, writing, _context in lexical_sites(nodes, declaration, symbols):
        node = by_id.get(key)
        if node is not None:
            check_name_use(node, env, writing)
