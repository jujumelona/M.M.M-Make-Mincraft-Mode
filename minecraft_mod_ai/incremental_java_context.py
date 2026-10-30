"""Small semantic work views and paginated, inspected Java declaration lookup.

The complete inputs are decoded and indexed once. A statement author sees its
work order, lexical values and generated signatures; framework and saved project
declarations are requested separately. Declaration discovery makes no judgment
about whether a requested feature already has an implementation.
"""
from __future__ import annotations

from collections import ChainMap, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import json
import re

from .construction_file_api_evidence import FileAPIEvidence, prompt_symbol_record
from .construction_type_contracts import canonical_type_name
from .first_pass_java_declarations import declaration_header
from .java_operation_ir import parameter_types


_STATE_FIELDS = ("state_reads", "state_writes", "runtime_state_reads", "runtime_state_writes")
_FUNCTION_FIELDS = ("id", "covers", "name", "owner", "signature", "return_type", "parameters",
                    "params", "type_parameters", "scope", "static", "visibility", "access",
                    "throws", "annotations")
_LANG = frozenset(("String Object Class Boolean Integer Long Double Float Byte Short Character "
                   "Number Exception RuntimeException Throwable Void Enum Iterable Comparable "
                   "CharSequence StringBuilder Math System Thread Runnable AutoCloseable").split())
_MEMBER_FLAGS = (("static", 8), ("final", 0x10), ("synchronized", 0x20), ("volatile", 0x40),
                 ("transient", 0x80), ("native", 0x100), ("strictfp", 0x800))


def _words(value):
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", str(value))
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _parts(value):
    """Split Java generic arguments without splitting a nested parameterization."""
    result, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    if value[start:].strip():
        result.append(value[start:].strip())
    return result


def _erase(value):
    depth, answer = 0, []
    for char in value:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0:
            answer.append(char)
    return "".join(answer).replace("...", "[]").strip()


def _formals(value):
    return _parts(value.strip("<>")) if isinstance(value, str) else list(value or ())


def _ground_argument(actual, formal):
    wildcard = re.fullmatch(r"\?\s+(?:extends|super)\s+(.+)", actual)
    if wildcard:
        return wildcard[1].strip()
    if actual == "?":
        bounds = str(formal).split(" extends ", 1)
        return bounds[1].split(" & ", 1)[0] if len(bounds) == 2 else "java.lang.Object"
    return actual


def _type_mapping(declaration, java_type, *, ground=False):
    formals = _formals(declaration.get("type_parameters"))
    arguments = _parts(str(java_type).split("<", 1)[1].rsplit(">", 1)[0]) if "<" in str(java_type) else []
    return {str(formal).split()[0]: _ground_argument(actual, formal) if ground else actual
            for formal, actual in zip(formals, arguments)}


def _substitute(value, mapping):
    return re.sub(r"(?<![\w$.])[A-Za-z_$][\w$]*(?![\w$.])",
                  lambda match: mapping.get(match[0], match[0]), str(value))


def _bound_method(row, mapping):
    """Class bindings never capture a method's independently declared variables."""
    shadowed = {str(formal).split()[0] for formal in _formals(row.get("type_parameters"))}
    mapping = {name: value for name, value in mapping.items() if name not in shadowed}
    answer = prompt_symbol_record(row)
    for field in ("generic_parameters", "generic_throws", "type_parameters"):
        if field in answer:
            value = answer[field]
            answer[field] = (_substitute(value, mapping) if isinstance(value, str) else
                             [_substitute(item, mapping) for item in value])
    if "generic_return_type" in answer:
        answer["generic_return_type"] = _substitute(answer["generic_return_type"], mapping)
    return answer


def _parent_types(declaration):
    superclass = declaration.get("generic_superclass") or declaration.get("superclass")
    interfaces = declaration.get("generic_interfaces") or declaration.get("interfaces") or ()
    return ([superclass] if superclass else []) + list(interfaces)


def _state_names(behavior):
    if not isinstance(behavior, Mapping):
        return set()
    return {name for field in _STATE_FIELDS for name in behavior.get(field) or ()
            if isinstance(name, str)}


def _retained(value):
    if not isinstance(value, Mapping):
        return deepcopy(value)
    answer = deepcopy({key: item for key, item in value.items() if key != "functions"})
    if "functions" in value:
        answer["functions"] = [{key: deepcopy(row[key]) for key in _FUNCTION_FIELDS if key in row}
                               for row in value.get("functions") or () if isinstance(row, Mapping)]
    return answer


def _generated(member):
    parts = member.get("metadata") or {}
    answer = {key: deepcopy(member[key]) for key in ("id", "owner", "kind", "name", "purpose", "implementation_status")
              if key in member}
    if member.get("kind") and member.get("name"):
        answer["signature"] = declaration_header(member["kind"], member["name"], parts)
    for key in ("type", "parameters", "return_type", "type_parameters", "throws", "visibility",
                "static", "final", "superclass", "interfaces", "components", "annotations"):
        if key in parts:
            answer[key] = deepcopy(parts[key])
    return answer


def _visibility(words, *, interface=False):
    if "private" in words:
        return 2
    if "protected" in words:
        return 4
    return 1 if "public" in words or interface else 0


def _parse_source_header(header, simple):
    from .java_region_parser import _parse

    stub = header.strip().rstrip(";")
    constructor = bool(re.search(r"\b" + re.escape(simple) + r"\s*\(", stub))
    try:
        raw, root = _parse("class " + simple + " {\n" + stub + (" {}" if constructor else ";") + "\n}")
    except (TypeError, ValueError):
        return b"", []
    return raw, root.named_children[0].child_by_field_name("body").named_children


class _SourceType:
    """One materialized type's declaration context, without its implementation."""

    def __init__(self, page, declared, owners):
        self.page, self.declared, self.owners = page, declared, owners
        self.owner = canonical_type_name(declared["owner"])
        self.simple = self.owner.rsplit(".", 1)[-1]
        self.enclosing = max((name for name in owners if self.owner.startswith(name + ".")), key=len, default="")
        self.header = str(declared.get("declaration") or "")
        self.interface = bool(re.search(r"\binterface\b", self.header))
        self.aliases = {}
        for statement in page.get("imports") or ():
            match = re.fullmatch(r"\s*import\s+([\w.$]+)\s*;\s*", str(statement))
            if match:
                self.aliases[match[1].rsplit(".", 1)[-1]] = canonical_type_name(match[1])

    def resolve(self, value):
        return re.sub(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", self._token, str(value))

    def _token(self, match):
        name = match[0]
        if "." in name or name in {"extends", "super"}:
            return name
        if name in self.aliases:
            return self.aliases[name]
        if name in _LANG:
            return "java.lang." + name
        local = (self.enclosing or self.owner.rpartition(".")[0]) + "." + name
        return local if local in self.owners else name

    def class_symbol(self):
        words = set(re.findall(r"[A-Za-z_]+", self.header))
        access = _visibility(words)
        access |= sum(flag for word, flag in (("static", 8), ("final", 0x10), ("abstract", 0x400)) if word in words)
        row = {"kind": "CLASS", "owner": self.owner, "name": self.simple, "access": access,
               "signature": self.header, "package": (self.enclosing or self.owner).rpartition(".")[0],
               "provenance": "materialized_prerequisite_declaration"}
        if self.interface:
            row["access"] |= 0x600
        if self.enclosing:
            row["enclosing_owner"] = self.enclosing
        if str(self.page.get("path") or "").startswith("src/client/"):
            row["side"] = "CLIENT"
        self._hierarchy(row)
        return row

    def _hierarchy(self, row):
        from .java_region_parser import _parse, _text

        try:
            raw, root = _parse(self.header + " {}")
            node = root.named_children[0]
        except (TypeError, ValueError, IndexError):
            return
        formals = node.child_by_field_name("type_parameters")
        if formals is not None:
            row["type_parameters"] = [self.resolve(_text(raw, child)) for child in formals.named_children]
        superclass = node.child_by_field_name("superclass")
        if superclass is not None and superclass.named_children:
            parent = self.resolve(_text(raw, superclass.named_children[0]))
            row.update(superclass=_erase(parent), generic_superclass=parent)
        interfaces = node.child_by_field_name("interfaces")
        if interfaces is None:
            interfaces = next((child for child in node.named_children if child.type == "extends_interfaces"), None)
        if interfaces is not None and interfaces.named_children:
            parents = [self.resolve(_text(raw, child)) for child in interfaces.named_children[0].named_children]
            row.update(interfaces=[_erase(parent) for parent in parents], generic_interfaces=parents)

    def _base_member(self, node, raw, header):
        from .java_region_parser import _text

        modifiers = next((child for child in node.named_children if child.type == "modifiers"), None)
        words = set(_text(raw, modifiers).split()) if modifiers is not None else set()
        access = _visibility(words, interface=self.interface)
        access |= sum(flag for word, flag in _MEMBER_FLAGS if word in words)
        abstract = "abstract" in words or self.interface and not words & {"static", "default", "private"}
        access |= 0x400 if abstract else 0
        return {"owner": self.owner, "access": access, "static": "static" in words,
                "signature": header, "provenance": "materialized_prerequisite_declaration"}

    def _fields(self, node, raw, row):
        from .java_region_parser import _text

        type_ = node.child_by_field_name("type")
        if type_ is None:
            return []
        answer = []
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            name = child.child_by_field_name("name")
            if name is not None:
                answer.append({**row, "kind": "FIELD", "name": _text(raw, name),
                               "return_type": _erase(self.resolve(_text(raw, type_))),
                               "generic_return_type": self.resolve(_text(raw, type_))})
        return answer

    def _parameters(self, node, raw):
        from .java_region_parser import _text

        types, varargs = [], False
        for param in node.named_children:
            type_ = param.child_by_field_name("type")
            if param.type == "spread_parameter":
                varargs = True
                type_ = next((child for child in param.named_children
                              if child.type not in {"modifiers", "variable_declarator", "annotation"}), None)
            if type_ is None:
                continue
            dimensions = param.child_by_field_name("dimensions")
            suffix = _text(raw, dimensions) if dimensions is not None else ""
            types.append(self.resolve(_text(raw, type_)) + ("[]" if param.type == "spread_parameter" else suffix))
        return types, varargs

    def _callable(self, node, raw, row):
        from .java_region_parser import _text

        name, params = node.child_by_field_name("name"), node.child_by_field_name("parameters")
        if name is None or params is None:
            return []
        generic_params, varargs = self._parameters(params, raw)
        returns = node.child_by_field_name("type")
        generic_return = self.resolve(_text(raw, returns)) if returns is not None else "void"
        formals = node.child_by_field_name("type_parameters")
        throws = next((child for child in node.named_children if child.type == "throws"), None)
        constructor = node.type == "constructor_declaration"
        row.update(kind="CONSTRUCTOR" if constructor else "METHOD", name="<init>" if constructor else _text(raw, name),
                   parameters=[_erase(value) for value in generic_params], generic_parameters=generic_params,
                   return_type=_erase(generic_return), generic_return_type=generic_return, varargs=varargs,
                   throws=[self.resolve(_text(raw, child)) for child in throws.named_children] if throws is not None else [])
        row["access"] |= 0x80 if varargs else 0
        if formals is not None:
            row["type_parameters"] = [self.resolve(_text(raw, child)) for child in formals.named_children]
        return [row]

    def member_symbols(self, header):
        if isinstance(header, Mapping):
            return [{"owner": self.owner, **header}]
        if not isinstance(header, str):
            return []
        raw, nodes = _parse_source_header(header, self.simple)
        rows = []
        for node in nodes:
            if node.type not in {"method_declaration", "constructor_declaration", "field_declaration"}:
                continue
            base = self._base_member(node, raw, header)
            rows.extend(self._fields(node, raw, base) if node.type == "field_declaration" else self._callable(node, raw, base))
        return rows


def _source_declarations(pages):
    return [(page, item) for page in pages if isinstance(page, Mapping)
            for item in page.get("types") or () if isinstance(item, Mapping) and isinstance(item.get("owner"), str)]


def _source_sam(class_row, members):
    abstract = [row for row in members if row.get("kind") == "METHOD"
                and int(row.get("access", 0)) & 0x400 and not int(row.get("access", 0)) & 8]
    if len(abstract) == 1:
        method = abstract[0]
        class_row.update(functional_parameters=method.get("generic_parameters", parameter_types(method)),
                         functional_return_type=method.get("generic_return_type", method.get("return_type", "void")),
                         functional_throws=method.get("throws", []))


def _source_symbols(pages):
    """Parse retained headers, never source implementations, into full callable ABI."""
    declarations = _source_declarations(pages)
    owners = {canonical_type_name(item["owner"]) for _, item in declarations}
    symbols = {}
    for page, declared in declarations:
        context = _SourceType(page, declared, owners)
        class_row = context.class_symbol()
        symbols[context.owner] = class_row
        members = [row for header in declared.get("members") or () for row in context.member_symbols(header)]
        for index, row in enumerate(members):
            symbols[f"!materialized:{context.owner}#{index}"] = row
        if context.interface:
            _source_sam(class_row, members)
    return symbols


def _combine_checked_events(native_events, source_events):
    """Expose verified original-source callbacks without inventing event ABIs."""
    merged = {}
    for event in (*native_events, *source_events):
        key = str(event.get("event_id") or "")
        if not key:
            raise ValueError("JAVA_CALLBACK_EVENT_ID_REQUIRED")
        if key in merged and merged[key] != event:
            raise ValueError("JAVA_CALLBACK_EVENT_ABI_CONFLICT: " + key)
        merged[key] = deepcopy(event)
    return list(merged.values())


class IncrementalJavaContext:
    """Authoritative host context shared by all incremental actions in one file."""

    page_size = 10

    def __init__(self, messages, spec, construction_context, behaviors):
        self.spec, self.context, self.behaviors = spec, construction_context or {}, behaviors
        self.payload = self._read_messages(messages)
        self._load_semantics()
        self._configure_target()
        self._load_index()
        self._callbacks = self._load_callbacks()

    @staticmethod
    def _read_messages(messages):
        payload = {}
        for message in messages:
            if message.get("role") != "user":
                continue
            try:
                value = json.loads(message.get("content") or "")
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                payload.update(value)
        return payload

    def _bound(self, key, encoded, default):
        if key in self.payload:
            return self.payload[key]
        if key in self.context:
            return self.context[key]
        try:
            return json.loads((self.spec.get("bindings") or {}).get(encoded) or json.dumps(default))
        except (TypeError, ValueError):
            return default

    def _load_semantics(self):
        self.owned_records = self._bound("owned_runtime_records", "owned_runtime_records_json", [])
        self.retained = _retained(self._bound("retained_typed_behavior", "retained_typed_behavior_json", {}))
        self.bindings = self._bound("runtime_state_bindings", "runtime_state_bindings_json", {})
        self.state_access = self.payload.get("host_state_access", self.context.get("host_state_access", [])) or []
        self._state_methods = defaultdict(list)
        for owner_index, owner in enumerate(self.state_access):
            for method_index, method in enumerate(owner.get("methods") or ()):
                self._state_methods[method.get("name")].append((owner_index, method_index, method))

    def _configure_target(self):
        bindings = self.spec.get("bindings") or {}
        self.target_path = str(self.spec.get("target_path") or self.payload.get("output_path") or "src/main/java/")
        package = str(bindings.get("package_name") or self.payload.get("package") or "")
        name = str(bindings.get("class_name") or self.payload.get("public_class") or "")
        self.owner = ".".join(part for part in (package, name) if part)
        self.side = str(self.spec.get("side") or self.payload.get("side") or "COMMON")
        if self.side.upper() == "CLIENT" and not self.target_path.startswith("src/client/"):
            self.target_path = "src/client/java/" + self.target_path

    def _load_index(self):
        raw_symbols = self.context.get("symbols") or {}
        self.prerequisites = self.payload.get("materialized_prerequisite_api", self.context.get("materialized_prerequisite_api", [])) or []
        supplemental = _source_symbols(self.prerequisites)
        materialized_owners = {row["owner"] for row in supplemental.values() if row.get("kind") == "CLASS"}
        raw_classes = {canonical_type_name(row.get("owner") or key) for key, row in raw_symbols.items()
                       if row.get("kind") == "CLASS" and row.get("provenance") != "construction_plan"}
        selected = self._inspect_supplemental(supplemental, raw_classes | materialized_owners)
        self.symbols = ChainMap(supplemental, raw_symbols)
        by_owner = self._index_symbols(supplemental, materialized_owners)
        self.evidence = FileAPIEvidence(self.symbols, target_path=self.target_path,
                                       owner=self.owner, by_owner=by_owner)
        self._options, self._queries = {}, {}
        self.selected = self._select_initial(selected)

    def _add_inspected_type(self, supplemental, record, excluded):
        owner = canonical_type_name(record.get("verified_class") or "")
        if not owner or owner in excluded:
            return
        supplemental.setdefault(owner, {"kind": "CLASS", "owner": owner, "access": 1,
                                        **(record.get("declaration") or {})})
        for index, member in enumerate(record.get("members") or ()):
            supplemental[f"!inspected:{owner}#{index}"] = member

    def _inspect_supplemental(self, supplemental, excluded):
        selected = []
        for field in ("inspected_target_api", "inspected_api"):
            api = self.payload.get(field) or {}
            if not isinstance(api, Mapping):
                continue
            for record in api.get("class_members") or ():
                self._add_inspected_type(supplemental, record, excluded)
            selected.extend(row for row in api.get("target_api") or () if isinstance(row, Mapping))
        return selected

    def _index_symbols(self, supplemental, materialized_owners):
        by_owner = defaultdict(list)
        self._names, self._tokens = defaultdict(list), defaultdict(set)
        for key, row in self.symbols.items():
            if row.get("provenance") == "construction_plan":
                continue
            owner = canonical_type_name(row.get("owner") or "")
            if key in supplemental or owner not in materialized_owners:
                by_owner[row.get("owner", "")].append(key)
                name = str(row.get("name") or owner.rsplit(".", 1)[-1])
                folded = name.casefold()
                self._names[folded].append(key)
                for token in _words(name):
                    self._tokens[token].add(folded)
        return by_owner

    def _selected_member(self, owner, row):
        return next((actual for _, actual in self._member_options(owner)
                     if actual.get("kind") == row.get("kind") and actual.get("name") == row.get("name")
                     and ("parameters" not in row or parameter_types(actual) == parameter_types(row))), None)

    def _select_initial(self, rows):
        selected = []
        for row in rows:
            owner = self._owner(row.get("owner", ""))
            if not owner:
                continue
            if row.get("kind") == "CLASS":
                selected.append(self.evidence._public_declaration(self.symbols[self.evidence.classes[owner]]))
            else:
                matching = self._selected_member(owner, row)
                if matching:
                    selected.append(prompt_symbol_record(matching))
            if len(selected) >= self.page_size:
                break
        return selected

    def _load_callbacks(self):
        source_events = self.context.get("verified_source_events") or ()
        if "available_fabric_callbacks" in self.payload:
            return _combine_checked_events(self.payload["available_fabric_callbacks"] or (), source_events)
        from .host_fabric_callback_abi import callback_interfaces
        return _combine_checked_events(callback_interfaces(self.symbols, side=self.side), source_events)

    def callbacks(self):
        """Return the original exact registration/SAM records without selection."""
        return deepcopy(self._callbacks)

    def _state_view(self, names):
        state_methods = defaultdict(list)
        for name in names:
            for prefix in ("readState_", "writeState_"):
                for owner_index, method_index, method in self._state_methods.get(prefix + name, ()):
                    state_methods[owner_index].append((method_index, method))
        state = [{"owner": self.state_access[owner_index]["owner"],
                  "methods": [deepcopy(method) for _, method in sorted(methods)]}
                 for owner_index, methods in sorted(state_methods.items())]
        return state

    def view(self, member, scope, program, *, position=0):
        behavior = member.get("behavior") or {}
        names = _state_names(behavior)
        answer = {
            "requirement": self.payload.get("requirement", self.spec.get("requirement", "")),
            "original_feature_goal": self.payload.get("original_feature_goal", self.spec.get("requirement", "")),
            "assigned_behavior": deepcopy(behavior),
            "owned_runtime_records": deepcopy(self.owned_records),
            "retained_typed_behavior": deepcopy(self.retained),
            "target": deepcopy(self.payload.get("target", self.context.get("target", {}))),
            "side": self.side, "target_path": self.spec.get("target_path", self.target_path),
            "implementation_requirements": self.payload.get("implementation_requirements",
                (self.spec.get("bindings") or {}).get("java_implementation_requirements", "")),
            "runtime_state_bindings": {key: deepcopy(value) for key, value in self.bindings.items()
                if key in names or isinstance(value, Mapping) and value.get("binding") == "typed_input"},
            "host_state_access": self._state_view(names),
            "host_state_work_order_scope": deepcopy(self.payload.get(
                "host_state_work_order_scope", self.context.get("host_state_work_order_scope", {}))),
            "current_member": _generated(member),
            "generated_declarations": [_generated(row) for row in (program.get("members") or [])[-self.page_size:]],
            "generated_declarations_total": len(program.get("members") or ()),
            "generated_declarations_offset": max(0, len(program.get("members") or ()) - self.page_size),
            "scope": deepcopy(scope), "position": position,
        }
        if self.selected:
            answer["selected_target_api"] = [prompt_symbol_record(row) for row in self.selected[:self.page_size]]
        if "event" in member:
            answer["current_member"]["event"] = deepcopy(member["event"])
        return answer

    def generated(self, program, query="", offset=0):
        terms = str(query).casefold().split()
        rows = [row for row in program.get("members", ()) if all(term in " ".join(
            str(row.get(key, "")) for key in ("id", "owner", "kind", "name", "purpose")).casefold() for term in terms)]
        return {"query": "@declarations:" + query, "offset": offset, "total": len(rows),
                "next_offset": offset + self.page_size if offset + self.page_size < len(rows) else None,
                "generated_declarations": [_generated(row) for row in rows[offset:offset + self.page_size]]}

    def _member_options(self, owner, *, subclass=False):
        identity = (owner, subclass)
        if identity not in self._options:
            self._options[identity] = self.evidence._member_options(owner, subclass=subclass)
        return self._options[identity]

    def _owner(self, java_type):
        plain = canonical_type_name(_erase(str(java_type)))
        if plain in self.evidence.classes:
            return plain if self.evidence._accessible(plain) else None
        candidates = [owner for owner in self.evidence.simple.get(plain, ()) if self.evidence._accessible(owner)]
        return candidates[0] if len(candidates) == 1 else None

    def _qualified_search(self, query):
        canonical = canonical_type_name(query)
        owner = self._owner(canonical)
        if owner:
            return [owner], self._member_options(owner)
        reference, separator, member_name = canonical.partition("#")
        if not separator:
            reference, separator, member_name = canonical.rpartition(".")
        owner = self._owner(reference) if separator else None
        if owner:
            member_name = member_name.split("(", 1)[0]
            return [owner], [(key, row) for key, row in self._member_options(owner)
                             if row.get("name") == member_name]
        return None

    def _searchable(self, key):
        row = self.symbols[key]
        owner = canonical_type_name(row.get("owner") or "")
        if row.get("kind") == "CLASS":
            return self.evidence._accessible(owner)
        if row.get("kind") not in {"METHOD", "FIELD", "CONSTRUCTOR"}:
            return False
        return any(candidate == key for candidate, _ in self._member_options(owner))

    @staticmethod
    def _search_rank(query, terms, item):
        key, row = item
        name = str(row.get("name") or "")
        return (name.casefold() != query.casefold(), -len(terms & _words(name)),
                canonical_type_name(row.get("owner") or ""), name, key)

    def _semantic_search(self, query):
        names = {query.casefold()} if query.casefold() in self._names else set()
        terms = _words(query)
        for term in terms:
            names.update(self._tokens.get(term, ()))
        candidates = {key for name in names for key in self._names[name]}
        members = [(key, self.symbols[key]) for key in candidates if self._searchable(key)]
        members.sort(key=lambda item: self._search_rank(query, terms, item))
        return [], members

    def _search(self, query):
        exact = self._qualified_search(query)
        return exact if exact is not None else self._semantic_search(query)

    def lookup(self, query, offset=0):
        """Page actual callable declarations; every last page remains reachable."""
        query = str(query or "").strip()
        if type(offset) is not int or offset < 0:
            raise ValueError("JAVA_API_LOOKUP_OFFSET_INVALID")
        if query not in self._queries:
            self._queries[query] = self._search(query) if query else ([], [])
        owners, rows = self._queries[query]
        chosen = rows[offset:offset + self.page_size]
        page_owners = sorted(set(owners) | {canonical_type_name(row.get("owner") or "") for _, row in chosen})
        declarations = [self.evidence._public_declaration(self.symbols[self.evidence.classes[owner]])
                        for owner in page_owners if owner in self.evidence.classes]
        total = len(rows)
        return {"query": query, "offset": offset, "page_size": self.page_size, "total": total,
                "next_offset": offset + self.page_size if offset + self.page_size < total else None,
                "declarations": declarations, "members": [prompt_symbol_record(row) for _, row in chosen
                                                            if row.get("kind") != "CLASS"],
                "guidance": ("Look up an exact owner, method name or words from an API name. "
                             "New helper methods, fields and types can be authored as generated declarations.")}

    def override(self, owner_type, method_name=None):
        owner = self._owner(owner_type)
        bindings = self._owner_bindings(owner_type)
        return [_bound_method(row, bindings.get(canonical_type_name(row.get("owner")), {}))
                for _, row in self._member_options(owner, subclass=True)
                if row.get("kind") == "METHOD" and (method_name is None or row.get("name") == method_name)
                and not row.get("static") and not int(row.get("access") or 0) & 0x18] if owner else []

    def _owner_bindings(self, java_type, *, ground=False):
        owner = self._owner(java_type)
        if not owner:
            return {}
        declaration = self.symbols[self.evidence.classes[owner]]
        bindings, pending = {owner: _type_mapping(declaration, java_type, ground=ground)}, [owner]
        while pending:
            current = pending.pop(0)
            declaration = self.symbols[self.evidence.classes[current]]
            for parent_type in _parent_types(declaration):
                parent_type = _substitute(parent_type, bindings[current])
                parent = self._owner(parent_type)
                if parent and parent not in bindings:
                    declaration = self.symbols[self.evidence.classes[parent]]
                    bindings[parent] = _type_mapping(declaration, parent_type, ground=ground)
                    pending.append(parent)
        return bindings

    def _sam_record(self, declaration):
        parameters = declaration["functional_parameters"]
        returns = declaration.get("functional_return_type", "void")
        throws = declaration.get("functional_throws") or []
        owner = declaration["owner"]
        for source in declaration.get("functional_contract_sources") or ():
            method = self.symbols.get(source)
            if method:
                parameters = method.get("generic_parameters", parameter_types(method))
                returns = method.get("generic_return_type", method.get("return_type", returns))
                throws = method.get("generic_throws", method.get("throws", throws))
                owner = method["owner"]
                break
        return {"owner": owner, "generic_parameters": parameters,
                "generic_return_type": returns, "generic_throws": throws}

    def sam(self, java_type):
        """Resolve an inspected functional contract, substituting actual type arguments."""
        owner = self._owner(java_type)
        if not owner:
            return {}
        declaration = self.symbols[self.evidence.classes[owner]]
        if "functional_parameters" not in self.evidence._public_declaration(declaration):
            return {}
        record = self._sam_record(declaration)
        bindings = self._owner_bindings(java_type, ground=True)
        record = _bound_method(record, bindings.get(canonical_type_name(record["owner"]), {}))
        return {"parameters": record["generic_parameters"],
                "return_type": record["generic_return_type"], "throws": record["generic_throws"]}
