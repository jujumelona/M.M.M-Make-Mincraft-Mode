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
              'superclass', 'interfaces', 'enclosing_owner', 'enclosing_instance_type',
              'functional_parameters', 'functional_return_type', 'functional_throws')
    answer = {key: row[key] for key in fields if key in row}
    if row.get('kind') in {'METHOD', 'CONSTRUCTOR'}:
        answer['parameters'] = parameter_types(row)
    return answer


class FileAPIEvidence:
    def __init__(self, symbols, *, target_path, owner='', by_owner=None):
        self.symbols = symbols
        self.target_path, self.owner = target_path, owner
        self.classes = {}
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
        package = re.search(r'\bpackage\s+([\w.]+)\s*;', source)
        package_name = package.group(1) if package else ''
        wildcard_scopes = []
        pattern = r'\bimport\s+(static\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(\.\*)?\s*;'
        for is_static, reference, wildcard in re.findall(pattern, source):
            if wildcard:
                if not is_static or self._resolve(reference):
                    wildcard_scopes.append(reference)
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
        selected = set(imports.values())
        for reference in re.findall(r'\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*', source):
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
        locations = set(re.findall(r'\blocation:\s+(?:class|interface|enum)\s+([\w.$]+)', diagnostic))
        locations.update(re.findall(r'\blocation:\s+variable\s+[\w$]+\s+of type\s+([\w.$]+)', diagnostic))
        return sorted(selected, key=lambda name: (
            not any(name == loc or name.endswith('.' + loc) for loc in locations), name))

    def _signature_accessible(self, types):
        references = {canonical_type_name(value)
                      for type_ in types
                      for value in re.findall(r'[A-Za-z_$][\w$]*(?:[./][A-Za-z_$][\w$]*)*', type_)}
        return all(value not in self.classes or self._accessible(value) for value in references)

    def _member_options(self, name):
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
                if not access & 1 and not same_package:
                    continue
                if row.get('side') == 'CLIENT' and not self.target_path.startswith('src/client/'):
                    continue
                signature_types = [row.get('return_type', ''), *parameter_types(row),
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
        return (member_name not in missing, member_name not in used,
                not factory, member_name != 'build', member_name != 'register',
                row.get('kind') != 'FIELD', member_name, member_key)

    def _selected_members(self, name, used, missing):
        # Preserve referenced names and factory/build roles before incidental overloads.
        options = [(key, row) for key, row in self._member_options(name)
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

    def records(self, owners, *, source='', diagnostic='', intent=''):
        roots = list(dict.fromkeys(canonical_type_name(name) for name in owners))
        roots = [name for name in roots if self._accessible(name)]
        # Nested builders/factories are actionable API, not incidental text matches.
        pending = [*roots, *(child for name in roots
                             for child in sorted(self.children.get(name, ())))]
        answer, seen, builder_inputs = [], set(), set()
        used = set(re.findall(r'\b[A-Za-z_$][\w$]*', source + '\n' + diagnostic + '\n' + intent))
        missing = set(re.findall(r'\bsymbol:\s+(?:method|variable|class)\s+([\w$]+)', diagnostic))
        while pending and len(answer) < 12:
            name = pending.pop(0)
            if name in seen or not self._accessible(name):
                continue
            seen.add(name)
            declaration = self.symbols[self.classes[name]]
            options, chosen = self._selected_members(name, used, missing)
            answer.append(self._class_record(name, declaration, options, chosen))
            if name in roots or canonical_type_name(declaration.get('enclosing_owner', '')) in roots:
                builder_inputs.update(self._append_signature_dependencies(pending, name, chosen, seen))
            elif name in builder_inputs:
                # Exactly one additional hop exposes the constructors/factories
                # of a required parameter type (e.g. Identifier for ResourceKey).
                self._append_signature_dependencies(pending, name, chosen, seen)
        return answer
