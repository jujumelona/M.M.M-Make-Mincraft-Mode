"""Project actual lexical names at partial graph use-sites, including holes."""
from __future__ import annotations

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
    initial = {row['name']: (row['type'], 'parameter', True, 'parameter:' + row['name'])
               for row in parameter_records(declaration)}
    stack = [('visit', key, initial, False, frozenset()) for key in reversed(roots)]
    while stack:
        action, key, env, writing, ancestors = stack.pop()
        node = by_id.get(key)
        if action == 'declare':
            name = node['attributes']['name']
            if name in env:
                raise ValueError('CONSTRUCTION_LOCAL_DUPLICATE')
            env[name] = (node['typed_result'], 'local', bool(node['input_refs']), key)
            continue
        if action == 'assigned':
            target = by_id.get(node['input_refs'][0])
            if target is not None and target['op'] == 'local':
                name = target['attributes']['name']
                if name in env:
                    type_, kind, _assigned, binding = env[name]
                    env[name] = (type_, kind, True, binding)
            continue
        if key in ancestors:
            raise ValueError('CONSTRUCTION_NODE_CYCLE')
        yield key, dict(env), writing
        if node is None:
            continue
        ancestors = ancestors | {key}
        op, refs, attrs = node['op'], node['input_refs'], node['attributes']

        def push(ref, scope=env, write=False):
            stack.append(('visit', ref, scope, write, ancestors))

        def bind(scope, name, type_, kind, binding):
            if name in scope:
                raise ValueError('CONSTRUCTION_LOCAL_DUPLICATE')
            scope[name] = (type_, kind, True, binding)

        if op == 'declare':
            stack.append(('declare', key, env, False, ancestors))
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
                push(ref, child)
        elif op in {'if', 'switch'}:
            for ref in reversed(refs[1:]):
                push(ref, dict(env))
            if refs:
                push(refs[0])
        elif op in {'while', 'do_while'}:
            push(refs[1], dict(env))
            push(refs[0])
        elif op == 'for':
            child = dict(env)
            push(refs[2], child)
            push(refs[3], child)
            push(refs[1], child)
            push(refs[0], child)
        elif op == 'foreach':
            child = dict(env)
            bind(child, attrs['name'], attrs['element_type'], 'local', key)
            push(refs[1], child)
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
            stack.append(('assigned', key, env, False, ancestors))
            push(refs[1])
            push(refs[0], write=attrs.get('operator', '=') == '=')
        else:
            for ref in reversed(refs):
                push(ref)


def names_at_slot(nodes, declaration, symbols, slot):
    visible = None
    for key, env, writing in lexical_sites(nodes, declaration, symbols, slot=slot):
        if key != slot:
            continue
        bindings = {(name, type_, kind, identity) for name, (type_, kind, assigned, identity) in env.items()
                    if assigned or writing}
        visible = bindings if visible is None else visible & bindings
    result = {'parameter': [], 'local': []}
    for name, type_, kind, _identity in sorted(visible or ()):
        result[kind].append((name, type_))
    return result


def check_partial_names(nodes, declaration, symbols):
    """Never checkpoint a name that contradicts its already known use-site."""
    by_id = {node['node_id']: node for node in nodes}
    for key, env, writing in lexical_sites(nodes, declaration, symbols):
        node = by_id.get(key)
        if node is None or node['op'] not in {'parameter', 'local'}:
            continue
        binding = env.get(node['attributes']['name'])
        if binding is None:
            raise ValueError('CONSTRUCTION_' + node['op'].upper() + '_SCOPE')
        if binding[0] != node['typed_result']:
            raise ValueError('CONSTRUCTION_LOCAL_SCOPE_TYPE')
        if not binding[2] and not writing:
            raise ValueError('CONSTRUCTION_LOCAL_UNINITIALIZED')
