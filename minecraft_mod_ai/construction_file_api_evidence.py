"""Exact-owner API context for complete Java source authoring and repair.

An imported nested Builder is not every Builder on the classpath. Selection
uses inspected identities before applying any prompt-size boundary, and keeps
callable signatures instead of handing the model a bare class name.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .construction_type_contracts import canonical_type_name, class_accessible
from .java_operation_ir import parameter_types


def prompt_symbol_record(row):
    """Expose a Java contract, not the host's complete bytecode-index record.

    Hashes, binary descriptors, classpath/debug receipts and repeated symbol IDs
    stay in the authoritative host index. Never shorten a type or signature.
    """
    fields = ('kind', 'owner', 'name', 'access', 'return_type', 'static', 'varargs',
              'generic_signature', 'signature', 'type_parameters', 'throws',
              'generic_parameters', 'generic_return_type', 'generic_superclass',
              'generic_interfaces', 'generic_throws',
              'superclass', 'interfaces', 'enclosing_owner', 'enclosing_instance_type',
              'functional_parameters', 'functional_return_type', 'functional_throws')
    answer = {key: row[key] for key in fields if key in row}
    if row.get('kind') in {'METHOD', 'CONSTRUCTOR'}:
        answer['parameters'] = parameter_types(row)
    return answer


def scoped_body_api_records(evidence):
    """Keep selected calls and the callable contracts of selected class types.

    A CLASS anchor is not a method named after the class. Filtering its
    members by (owner, CLASS, name) erased every actual method and all builder
    dependencies. Follow only inspected type edges within this bounded view.
    """
    selected = {(row.get('owner'), row.get('kind'), row.get('name'))
                for row in evidence.get('target_api', ()) if isinstance(row, dict)}
    full_owners = {owner for owner, kind, _ in selected if kind == 'CLASS'}
    records = {row.get('verified_class'): row for row in evidence.get('class_members', ())
               if isinstance(row, dict) and isinstance(row.get('verified_class'), str)}
    pending = sorted({owner for owner, _, _ in selected})
    answer, seen = {}, set()
    while pending:
        owner = pending.pop(0)
        mode = owner in full_owners
        if (owner, mode) in seen or owner not in records:
            continue
        seen.add((owner, mode))
        entry = records[owner]
        members = [row for row in entry.get('members', ()) if isinstance(row, dict)
                   and (mode or
                        (row.get('owner'), row.get('kind'), row.get('name')) in selected)]
        if not members and not mode:
            continue
        answer[owner] = {**entry, 'members': members, 'body_context_selects_only_target_api': True}
        dependencies = set()
        if mode:
            # Static nested factories are accessed as types, even when no
            # root method returns them. Use inspected enclosure, not names.
            dependencies.update(name for name, nested in records.items()
                if canonical_type_name((nested.get('declaration') or {}).get('enclosing_owner', '')) == owner)
        for row in members:
            types = [row.get('return_type', ''), *row.get('parameters', ()),
                     row.get('generic_return_type', ''), *row.get('generic_parameters', ())]
            for type_ in types:
                dependencies.update(canonical_type_name(name) for name in re.findall(
                    r'[A-Za-z_$][\w$]*(?:[./][A-Za-z_$][\w$]*)*', str(type_)))
        for name in sorted(dependencies & records.keys() - full_owners):
            full_owners.add(name)
            pending.append(name)
    return [answer[name] for name in sorted(answer)]



class FileAPIEvidence:
    def __init__(self, symbols, *, target_path, owner='', by_owner=None):
        self.symbols = symbols
        self.target_path, self.owner = target_path, owner
        self.classes = {}
        self.superclasses = set()
        self.simple = defaultdict(list)
        self.children = defaultdict(list)
        self.members = defaultdict(list) if by_owner is None else by_owner
        for key, row in symbols.items():
            if row.get('provenance') == 'construction_plan':
                continue
            if by_owner is None:
                self.members[row.get('owner', '')].append(key)
            if row.get('kind') != 'CLASS':
                continue
            name = canonical_type_name(row.get('owner') or key)
            self.classes[name] = key
            self.simple[name.rsplit('.', 1)[-1]].append(name)
            enclosing = canonical_type_name(row.get('enclosing_owner') or '')
            if enclosing:
                self.children[enclosing].append(name)

    def _accessible(self, name):
        key = self.classes.get(name)
        if key is None:
            return False
        row = self.symbols[key]
        return class_accessible(self.symbols, row['owner'], self.owner, self.target_path)

    def _resolve(self, reference):
        """Resolve the longest real class prefix of a qualified expression."""
        reference = canonical_type_name(reference)
        while reference:
            if reference in self.classes:
                return reference if self._accessible(reference) else None
            reference = reference.rpartition('.')[0]
        return None

    def source_owners(self, source, diagnostic):
        imports = {}
        static_members, static_scopes = {}, []
        package = re.search(r'\bpackage\s+([\w.]+)\s*;', source)
        package_name = package.group(1) if package else ''
        wildcard_scopes = []
        pattern = r'\bimport\s+(static\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(\.\*)?\s*;'
        for is_static, reference, wildcard in re.findall(pattern, source):
            if wildcard:
                if not is_static or self._resolve(reference):
                    wildcard_scopes.append(reference)
                    if is_static:
                        static_scopes.append(self._resolve(reference))
                continue
            resolved = self._resolve(reference)
            if not resolved:
                # Recover a wrong mappings import only when the actual target
                # has one unambiguous counterpart in that same API family.
                family = '.'.join(reference.split('.')[:2]) + '.'
                candidates = [name for name in self.simple.get(reference.rsplit('.', 1)[-1], ())
                              if name.startswith(family) and self._accessible(name)]
                resolved = candidates[0] if len(candidates) == 1 else None
            if resolved:
                imports[resolved.rsplit('.', 1)[-1]] = resolved
                if is_static and canonical_type_name(reference) != resolved:
                    static_members[reference.rsplit('.', 1)[-1]] = resolved
        selected = set()
        body = re.sub(pattern, '', source)
        names = set(re.findall(r'\b[A-Za-z_$][\w$]*', body))
        selected.update(owner for name, owner in static_members.items() if name in names)
        for scope in static_scopes:
            if any(row.get('static') and row.get('name') in names
                   for _, row in self._member_options(scope)):
                selected.add(scope)
        superclass_refs = set(re.findall(
            r'\bclass\s+[\w$]+(?:\s*<[^{};]+>)?\s+extends\s+([\w.$]+)', body))
        for reference in re.findall(r'\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*', body):
            first, _, rest = reference.partition('.')
            resolved = self._resolve(imports[first] + ('.' + rest if rest else '')) if first in imports else self._resolve(reference)
            if not resolved and first[:1].isupper():
                # Java's package/import scopes take precedence over unrelated
                # library classes with the same simple name.
                if package_name:
                    resolved = self._resolve(package_name + '.' + reference)
                if not resolved:
                    candidates = {self._resolve(scope + '.' + reference) for scope in wildcard_scopes}
                    candidates.discard(None)
                    if len(candidates) == 1:
                        resolved = next(iter(candidates))
                if not resolved:
                    candidates = [name for name in self.simple.get(first, ()) if self._accessible(name)]
                    if len(candidates) == 1:
                        resolved = self._resolve(candidates[0] + ('.' + rest if rest else ''))
            if resolved:
                selected.add(resolved)
                if first in imports:
                    selected.add(imports[first])
                if reference in superclass_refs:
                    self.superclasses.add(resolved)
        locations = set(re.findall(r'\blocation:\s+(?:class|interface|enum)\s+([\w.$]+)', diagnostic))
        locations.update(re.findall(r'\blocation:\s+variable\s+[\w$]+\s+of type\s+([\w.$]+)', diagnostic))
        return sorted(selected, key=lambda name: (
            not any(name == loc or name.endswith('.' + loc) for loc in locations), name))

    def _signature_accessible(self, types):
        references = {canonical_type_name(value)
                      for type_ in types
                      for value in re.findall(r'[A-Za-z_$][\w$]*(?:[./][A-Za-z_$][\w$]*)*', type_)}
        return all(value not in self.classes or self._accessible(value) for value in references)

    def _member_options(self, name, *, subclass=False):
        """Keep true declaring owners; inheritance never fabricates a member."""
        pending, seen, signatures, options = [name], set(), set(), []
        while pending:
            ancestor = pending.pop(0)
            if ancestor in seen or ancestor not in self.classes:
                continue
            seen.add(ancestor)
            declaration = self.symbols[self.classes[ancestor]]
            pending.extend(canonical_type_name(value) for value in
                (declaration.get('superclass', ''), *declaration.get('interfaces', ())) if value)
            if not self._accessible(ancestor):
                continue
            for key in sorted(self.members.get(declaration['owner'], ())):
                row = self.symbols[key]
                kind = row.get('kind')
                if kind not in {'METHOD', 'FIELD', 'CONSTRUCTOR'} or row.get('provenance') == 'construction_plan':
                    continue
                if ancestor != name and (kind == 'CONSTRUCTOR' or
                    kind == 'METHOD' and row.get('static') and declaration.get('access', 0) & 0x0200):
                    continue
                access = row.get('access', 1)
                if access & 2 or row.get('visibility') == 'private':
                    continue
                same_package = declaration.get('package', declaration['owner'].rpartition('.')[0]) == self.owner.rpartition('.')[0]
                if not access & 1 and not same_package and not (subclass and access & 4):
                    continue
                if row.get('side') == 'CLIENT' and not self.target_path.startswith('src/client/'):
                    continue
                signature_types = [row.get('return_type', ''), *parameter_types(row),
                    row.get('generic_return_type', ''), *row.get('generic_parameters', ()),
                    *row.get('type_parameters', ()), *row.get('generic_throws', ()),
                    *row.get('throws', ()), *row.get('functional_parameters', ()),
                    row.get('functional_return_type', ''), *row.get('functional_throws', ())]
                if not self._signature_accessible(signature_types):
                    continue
                signature = (kind, row.get('name'), tuple(parameter_types(row)) if kind != 'FIELD' else ())
                if signature in signatures:
                    continue
                signatures.add(signature)
                options.append((key, row))
        return options

    @staticmethod
    def _rank_member(name, used, missing, item):
        member_key, row = item
        member_name = str(row.get('name', ''))
        factory = (row.get('kind') == 'CONSTRUCTOR' or
                   row.get('static') and canonical_type_name(row.get('return_type', '')) == name)
        # Actual callable methods must precede incidental FIELD constants.
        # The prior != FIELD test ranked fields FIRST, so the 16-record cap
        # offered a combat callback only Player/Entity constants and no
        # damage/attack methods. This is a generic API-index ordering defect,
        # independent of the Minecraft prompt or any particular method name.
        return (member_name not in missing, member_name not in used,
                not factory, member_name != 'build', member_name != 'register',
                row.get('kind') == 'FIELD', member_name, member_key)

    def _selected_members(self, name, used, missing):
        # Preserve referenced names and factory/build roles before incidental overloads.
        options = [(key, row) for key, row in self._member_options(name, subclass=name in self.superclasses)
                   if name == 'java.lang.Object' or row.get('owner') != 'java.lang.Object'
                   or row.get('name') in used or row.get('name') in missing]
        options.sort(key=lambda item: self._rank_member(name, used, missing, item))
        covered, representatives = set(), []
        for item in options:
            _, row = item
            member_name = row.get('name')
            factory = (row.get('kind') == 'CONSTRUCTOR' or
                       row.get('static') and canonical_type_name(row.get('return_type', '')) == name)
            if member_name not in covered and (member_name in used or member_name in missing or
                factory or member_name in {'build', 'register'}):
                representatives.append(item)
                covered.add(member_name)
        selected_keys = {key for key, _ in representatives}
        chosen = (representatives + [item for item in options if item[0] not in selected_keys])[
            :max(16, len(representatives))]
        return options, chosen

    def _public_declaration(self, declaration):
        record = prompt_symbol_record(declaration)
        callback_types = [*declaration.get('functional_parameters', ()),
                          declaration.get('functional_return_type', ''),
                          *declaration.get('functional_throws', ())]
        if not self._signature_accessible(callback_types):
            record = {field: value for field, value in record.items()
                      if not field.startswith('functional_')}
        return record

    def _class_record(self, name, declaration, options, chosen):
        chosen_names = {row.get('name') for _, row in chosen}
        return {'verified_class': name, 'declaration': self._public_declaration(declaration),
                'members': [prompt_symbol_record(row) for _, row in chosen],
                'member_count': len(options), 'members_truncated': len(chosen) < len(options),
                'overload_counts': {member_name: count for member_name, count in
                    sorted(Counter(row.get('name') for _, row in options).items())
                    if member_name in chosen_names},
                'evidence': 'actual_target_symbol_index', 'behavior_verified': False}

    def _append_signature_dependencies(self, pending, name, chosen, seen):
        # A builder's exact parameter types (ResourceKey, Identifier, factory)
        # are more useful than unrelated helper/JDK owners. Make them visible
        # before the finite discovery view fills with incidental nested types.
        dependencies = []
        for _, row in chosen:
            for type_ in parameter_types(row):
                related = self._resolve(type_.split('<', 1)[0].removesuffix('[]'))
                if related and related not in seen and related not in dependencies:
                    dependencies.append(related)
        actionable = [name for name in dependencies
                      if not name.startswith(('java.', 'javax.', 'com.sun.', 'jdk.'))]
        incidental = [name for name in dependencies if name not in actionable]
        pending[:0] = actionable
        pending.extend(sorted(self.children.get(name, ())))
        pending.extend(incidental)
        return actionable

    def entity_type_evidence(self):
        """A concrete subclass needs super constructors, not a registry builder.

        Every declaration is still resolved from the selected target inventory.
        Protected members here are usable only through subclass inheritance.
        """
        bases = ('net.minecraft.world.entity.Entity', 'net.minecraft.world.entity.Mob',
                 'net.minecraft.world.entity.monster.Monster')
        roots = (*bases, 'net.minecraft.world.entity.EntityType', 'net.minecraft.world.level.Level')
        records, declarations = [], []
        for name in roots:
            if not self._accessible(name):
                continue
            declaration = self.symbols[self.classes[name]]
            declarations.append(prompt_symbol_record(declaration))
            options = self._member_options(name, subclass=name in bases)
            chosen = [item for item in options if name in bases and (
                item[1].get('kind') == 'CONSTRUCTOR' or
                item[1].get('kind') == 'METHOD' and item[1].get('access', 0) & 0x0400)]
            record = self._class_record(name, declaration, options, chosen)
            if name in bases:
                record['access_context'] = 'generated_subclass_super_constructor_and_overrides'
            records.append(record)
        return {'target_api': declarations, 'class_members': records, 'source': [],
                'typed_input_producers': []}

    def records(self, owners, *, source='', diagnostic='', intent=''):
        roots = list(dict.fromkeys(canonical_type_name(name) for name in owners))
        roots = [name for name in roots if self._accessible(name)]
        # Nested builders/factories are actionable API, not incidental text matches.
        # Required owner contracts cannot be displaced by the first root's
        # incidental parameter types. Prioritize inspected nested factories,
        # then navigate supplemental signature dependencies separately.
        children = list(dict.fromkeys(child for name in roots
                         for child in sorted(self.children.get(name, ()))))
        factories = [child for child in children if
            self.symbols[self.classes[child]].get('functional_parameters') is not None or any(
                row.get('name') == 'build' or row.get('static') and
                canonical_type_name(str(row.get('return_type', '')).split('<', 1)[0]) == child
                for _, row in self._member_options(child))]
        required = list(dict.fromkeys([*roots, *factories]))
        pending = [child for child in children if child not in required]
        answer, seen, builder_inputs = [], set(), set()
        used = set(re.findall(r'\b[A-Za-z_$][\w$]*', source + '\n' + diagnostic + '\n' + intent))
        missing = set(re.findall(r'\bsymbol:\s+(?:method|variable|class)\s+([\w$]+)', diagnostic))
        missing_at = defaultdict(set)
        for member, location in re.findall(
                r'\bsymbol:\s+(?:method|variable)\s+([\w$]+)'
                r'(?:(?!\bsymbol:).)*?\blocation:\s+'
                r'(?:(?:class|interface|enum)\s+|variable\s+[\w$]+\s+of type\s+)([\w.$]+)',
                diagnostic, re.DOTALL):
            missing_at[canonical_type_name(location)].add(member)
        while (required or pending) and len(answer) < max(12, len(roots) + len(factories)):
            name = required.pop(0) if required else pending.pop(0)
            if name in seen or not self._accessible(name):
                continue
            seen.add(name)
            declaration = self.symbols[self.classes[name]]
            options, chosen = self._selected_members(name, used, missing)
            record = self._class_record(name, declaration, options, chosen)
            owner_missing = set().union(*(names for location, names in missing_at.items()
                if name == location or name.endswith('.' + location)))
            absent = owner_missing - {row.get('name') for _, row in options}
            if absent:
                record['unavailable_requested_members'] = sorted(absent)
            answer.append(record)
            if name in roots or canonical_type_name(declaration.get('enclosing_owner', '')) in roots:
                builder_inputs.update(self._append_signature_dependencies(pending, name, chosen, seen))
            elif name in builder_inputs:
                # Exactly one additional hop exposes the constructors/factories
                # of a required parameter type (e.g. Identifier for ResourceKey).
                self._append_signature_dependencies(pending, name, chosen, seen)
        return answer


    def callback_member_records(self, owners, *, intent, predicate=False, limit=28):
        """Search the full verified ABI and retain essential typed input APIs.

        Relevance ranking may prioritize specific operations, but lexical
        overlap is never a prerequisite for exposing a callback's basic
        observers or its typed callback-result constants. Every input owner
        remains visible, even if no method is currently selected.
        """
        roots = list(dict.fromkeys(canonical_type_name(name) for name in owners))
        roots = [name for name in roots if self._accessible(name)]
        words = set(re.findall(r"[a-z]{3,}", re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])", " ", str(intent)).lower()))
        pool = []
        all_options = {}
        for name in roots:
            options = self._member_options(name)
            all_options[name] = options
            for key, row in options:
                kind = row.get("kind")
                member_name = str(row.get("name") or "")
                name_words = set(re.findall(r"[a-z]{3,}", re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])", " ", member_name).lower()))
                overlap = len(name_words & words)
                is_static = bool(row.get("static") or int(row.get("access", 0)) & 8)
                params = parameter_types(row) if kind == "METHOD" else []
                observer = (kind == "METHOD" and not is_static and not params
                            and member_name.startswith((
                                "get", "is", "has", "can", "should", "read", "peek")))
                result_constant = (kind == "FIELD" and is_static
                                   and canonical_type_name(row.get("return_type", "")) == name)
                if kind == "METHOD":
                    if len(params) > 3 or any(
                        canonical_type_name(t) not in roots for t in params
                    ):
                        # A verified signature without any typed producer is
                        # documentation, not an executable callback choice.
                        continue
                    # Predicate code can directly call only zero-argument
                    # boolean observers; keep result constants separately.
                    if predicate and (row.get("return_type") != "boolean" or params):
                        if not observer:
                            continue
                    if not predicate and observer and not overlap:
                        pass  # Keep a small baseline; it is not a gameplay effect.
                    elif not overlap and not observer:
                        continue
                    rank = (0 if overlap else 2, -overlap, False,
                            len(params), name, member_name, key)
                elif result_constant:
                    rank = (1, 0, False, 0, name, member_name, key)
                elif kind == "FIELD" and overlap:
                    rank = (1, -overlap, False, 0, name, member_name, key)
                else:
                    continue
                pool.append((rank, name, key, row))
        pool.sort(key=lambda item: item[0])
        # A global top-N list can be filled by hundreds of Player getters and
        # starve the Level or callback-result owner completely. Reserve one
        # relevant, inspected member per callback owner before filling the
        # remaining capacity by global task relevance.
        chosen, selected = [], set()
        for root in roots[:limit]:
            first = next((row for row in pool if row[1] == root), None)
            if first is not None:
                chosen.append(first)
                selected.add((first[1], first[2]))
        for item in pool:
            if len(chosen) >= limit:
                break
            identity = (item[1], item[2])
            if identity not in selected:
                chosen.append(item)
                selected.add(identity)
        by_root = defaultdict(list)
        for _score, root, key, row in chosen:
            by_root[root].append((key, row))
        return [
            self._class_record(root, self.symbols[self.classes[root]],
                               all_options[root], by_root[root])
            for root in roots
        ]
