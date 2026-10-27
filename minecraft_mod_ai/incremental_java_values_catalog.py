"""Exact generated, language-defined, and inspected inherited Java members."""
from __future__ import annotations

from copy import deepcopy

from .construction_type_contracts import canonical_type_name
from .incremental_java_context import _bound_method, _erase, _formals, _substitute, _type_mapping
from .incremental_java_values_model import current_owner
from .java_operation_ir import parameter_types


def _generated_identity(builder, declaration):
    names, owner, seen = [declaration["name"]], declaration["owner"], set()
    while owner and owner not in seen:
        seen.add(owner)
        names.insert(0, owner)
        parent = next((row for row in builder.program["members"]
                       if row["kind"] in {"class", "record", "enum"} and row["name"] == owner), None)
        owner = parent["owner"] if parent else ""
    return ".".join([builder.program["package"], *names])


def _generated_matches(builder, declaration, owner):
    owner = canonical_type_name(_erase(owner))
    qualified = _generated_identity(builder, declaration)
    return owner in {declaration["name"], qualified, qualified.removeprefix(builder.program["package"] + ".")}


def generated_type(builder, owner):
    declarations = [row for row in builder.program.get("members", ()) if row["kind"] in {"class", "record", "enum"}]
    name = builder.program["class_name"]
    declarations.append({"id": "root:" + name, "kind": "class", "name": name, "owner": "",
                         "metadata": {"visibility": "public", "interfaces": [builder.program["interface"]]}})
    return next((row for row in declarations if _generated_matches(builder, row, owner)), None)



def _type_identity(builder, java_type):
    declaration = generated_type(builder, java_type)
    if declaration is not None:
        return _generated_identity(builder, declaration)
    return builder.api._owner(java_type) or canonical_type_name(_erase(java_type))

def _parents(declaration, java_type):
    metadata = declaration["metadata"]
    defaults = {"class": "java.lang.Object", "record": "java.lang.Record",
                "enum": "java.lang.Enum<" + java_type + ">"}
    parents = [metadata.get("superclass") or defaults[declaration["kind"]], *metadata.get("interfaces", ())]
    mapping = _type_mapping(metadata, java_type)
    return [_substitute(parent, mapping) for parent in parents]


def _lineage(builder, java_type):
    """Walk generated parents, stopping each branch at its inspected API root."""
    pending, seen, generated, indexed = [java_type], set(), [], []
    while pending:
        current = pending.pop(0)
        identity = _type_identity(builder, current)
        if identity in seen:
            continue
        seen.add(identity)
        declaration = generated_type(builder, current)
        if declaration is None:
            indexed.append(current)
        else:
            generated.append((declaration, current))
            pending.extend(_parents(declaration, current))
    return generated, indexed


def _bind_generated(row, declaration, java_type):
    answer = deepcopy(row)
    shadowed = {str(formal).split()[0] for formal in _formals(row.get("type_parameters"))}
    mapping = {name: value for name, value in _type_mapping(declaration["metadata"], java_type).items()
               if name not in shadowed}
    if mapping:
        answer["generic_return_type"] = _substitute(row.get("generic_return_type", row.get("return_type", "void")), mapping)
        if row["kind"] in {"METHOD", "CONSTRUCTOR"}:
            answer["generic_parameters"] = [_substitute(value, mapping) for value in parameter_types(row)]
    return answer


def _declared_members(builder, declaration, kind):
    rows = []
    for member in builder.program.get("members", ()):
        if member["kind"].upper() != kind or not _generated_matches(builder, declaration, member["owner"]):
            continue
        row = {"id": member["id"], "kind": kind, "owner": member["owner"],
               "name": member["name"], "purpose": member.get("purpose", ""), **member["metadata"]}
        if kind == "FIELD":
            row["return_type"] = row["type"]
        rows.append(row)
    return rows


def _record_accessors(declaration):
    return [{"id": declaration["id"] + ":component:" + str(index), "kind": "METHOD",
             "owner": declaration["name"], "name": component["name"], "parameters": [],
             "return_type": component["type"].replace("...", "[]"), "access": 1, "implicit": True}
            for index, component in enumerate(declaration["metadata"].get("components", ()))]


def _enum_members(declaration, kind):
    owner, identifier = declaration["name"], declaration["id"]
    if kind == "FIELD":
        return [{"id": identifier + ":constant:" + str(index), "kind": "FIELD", "owner": owner,
                 "name": constant["name"], "return_type": owner, "type": owner, "access": 0x4019,
                 "static": True, "final": True, "implicit": True}
                for index, constant in enumerate(declaration["metadata"].get("constants", ()))]
    if kind == "METHOD":
        return [{"id": identifier + ":" + name, "kind": "METHOD", "owner": owner, "name": name,
                 "parameters": parameters, "return_type": returns, "static": True, "access": 9, "implicit": True}
                for name, parameters, returns in (("values", [], owner + "[]"),
                                                  ("valueOf", ["java.lang.String"], owner))]
    return []


def _implicit_members(declaration, kind):
    if declaration["kind"] == "record":
        if kind == "METHOD":
            return _record_accessors(declaration)
        if kind == "CONSTRUCTOR":
            return [_constructor_row(declaration, declaration["name"])]
    if declaration["kind"] == "enum":
        return _enum_members(declaration, kind)
    return []


def _enclosing_constructor(row, declaration):
    answer = deepcopy(row)
    if declaration["kind"] == "class" and declaration["owner"] and not declaration["metadata"].get("static"):
        answer["enclosing_instance_type"] = declaration["owner"]
    return answer


def _constructor_row(declaration, owner):
    return {"id": declaration["id"], "kind": "CONSTRUCTOR", "owner": declaration["name"],
            "name": declaration["name"], "parameters": declaration["metadata"].get("components", []),
            "return_type": owner, "implicit": True}


def implicit_constructor(builder, owner):
    declaration = generated_type(builder, owner)
    if declaration is None or declaration["kind"] == "enum":
        return []
    row = _enclosing_constructor(_constructor_row(declaration, owner), declaration)
    return [_bind_generated(row, declaration, owner)]


def superclass_type(builder):
    owner = current_owner(builder)
    declaration = generated_type(builder, owner)
    return _parents(declaration, owner)[0] if declaration else "java.lang.Object"


def _member_key(row):
    return row["kind"], row["name"], tuple(parameter_types(row)) if row["kind"] != "FIELD" else ()


def _inherited_generated(row):
    return row.get("visibility") != "private" and not int(row.get("access") or 0) & 2


def generated_members(builder, owner, kind):
    declarations, _ = _lineage(builder, owner)
    rows, seen = [], set()
    for index, (declaration, java_type) in enumerate(declarations):
        if kind == "CONSTRUCTOR" and index:
            continue
        candidates = _declared_members(builder, declaration, kind) + _implicit_members(declaration, kind)
        for row in candidates:
            key = _member_key(row)
            if key in seen or index and not _inherited_generated(row):
                continue
            seen.add(key)
            if kind == "CONSTRUCTOR":
                row = _enclosing_constructor(row, declaration)
            rows.append(_bind_generated(row, declaration, java_type))
    return rows


def _relatives(builder, java_type):
    declarations, roots = _lineage(builder, java_type)
    names = {_generated_identity(builder, declaration) for declaration, _ in declarations}
    for root in roots:
        names.update(canonical_type_name(owner) for owner in builder.api._owner_bindings(root))
    return names


def _is_subtype(builder, java_type, owner):
    return _type_identity(builder, owner) in _relatives(builder, java_type)


def _lexical_owners(builder):
    owner, seen = current_owner(builder), set()
    while owner and owner not in seen:
        seen.add(owner)
        yield owner
        declaration = generated_type(builder, owner)
        owner = declaration["owner"] if declaration else ""


def _flag(row, name, bit):
    return bool(row.get(name) or int(row.get("access") or 0) & bit)


def _protected_access(builder, receiver, row):
    if not int(row.get("access", 1)) & 4:
        return True
    declaration = builder.api.symbols[builder.api.evidence.classes[row["owner"]]]
    package = declaration.get("package", row["owner"].rpartition(".")[0])
    if package == builder.api.owner.rpartition(".")[0]:
        return True
    contexts = [owner for owner in _lexical_owners(builder) if _is_subtype(builder, owner, row["owner"])]
    if _flag(row, "static", 8):
        return bool(contexts)
    if receiver.source == "super":
        return current_owner(builder) in contexts
    return any(_is_subtype(builder, receiver.java_type, owner) for owner in contexts)


def _inherited_api(builder, row):
    if row.get("kind") == "CONSTRUCTOR":
        return False
    declaration = builder.api.symbols[builder.api.evidence.classes[row["owner"]]]
    return not (row.get("kind") == "METHOD" and _flag(row, "static", 8)
                and int(declaration.get("access") or 0) & 0x0200)


def indexed_members(builder, receiver, kind):
    declarations, roots = _lineage(builder, receiver.java_type)
    if kind == "CONSTRUCTOR" and declarations:
        return []
    rows, seen = [], {_member_key(row) for row in generated_members(builder, receiver.java_type, kind)}
    for java_type in roots:
        owner = builder.api._owner(java_type)
        if owner is None:
            continue
        bindings = builder.api._owner_bindings(java_type)
        candidates = builder.api._member_options(owner, subclass=kind != "CONSTRUCTOR")
        for _, row in candidates:
            if declarations and not _inherited_api(builder, row):
                continue
            if row.get("kind") != kind or not _protected_access(builder, receiver, row):
                continue
            key = _member_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(_bound_method(row, bindings.get(row["owner"], {})))
    return rows
