"""Inspected symbols and complete source units for one construction owner."""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
from .construction_type_contracts import canonical_type_name, annotate_functional_interfaces


_PRIMITIVE_DESCRIPTORS = {"Z": "boolean", "B": "byte", "S": "short", "I": "int",
                          "J": "long", "C": "char", "F": "float", "D": "double", "V": "void"}


def descriptor_types(descriptor):
    def read_type(offset):
        arrays = 0
        while descriptor[offset] == "[":
            arrays += 1
            offset += 1
        char = descriptor[offset]
        if char == "L":
            end = descriptor.index(";", offset)
            value, offset = canonical_type_name(descriptor[offset+1:end]), end+1
        elif char in _PRIMITIVE_DESCRIPTORS:
            value, offset = _PRIMITIVE_DESCRIPTORS[char], offset+1
        else:
            raise ValueError("CONSTRUCTION_DESCRIPTOR_INVALID")
        return value + "[]" * arrays, offset
    if not descriptor.startswith("("):
        result, end = read_type(0)
        if end != len(descriptor):
            raise ValueError("CONSTRUCTION_DESCRIPTOR_INVALID")
        return [], result
    parameters, offset = [], 1
    while descriptor[offset] != ")":
        value, offset = read_type(offset)
        parameters.append(value)
    result, end = read_type(offset+1)
    if end != len(descriptor):
        raise ValueError("CONSTRUCTION_DESCRIPTOR_INVALID")
    return parameters, result


def build_target_symbols(classpath, *, java_version: int, namespace: str, classpath_sides,
                         mappings_file=None, source_namespace=None):
    from .jar_api_extractor import inspect_jar, parse_class
    if namespace not in {"NAMED", "INTERMEDIARY", "OFFICIAL"}:
        raise ValueError("CONSTRUCTION_NAMESPACE_REQUIRED")
    result, class_hashes = {}, {}
    for raw_path in classpath:
        logical_path = Path(raw_path).expanduser()
        path = logical_path.resolve()
        side = classpath_sides.get(str(path), classpath_sides.get(str(raw_path), "COMMON"))
        if path.is_dir():
            classes = [parse_class(p.read_bytes(), default_side=side) for p in path.rglob("*.class")]
        elif logical_path.name == "modules" and logical_path.parent.name == "lib":
            from .jdk_runtime_inventory import inspect_jdk_runtime_image

            classes = inspect_jdk_runtime_image(logical_path, java_version=java_version, default_side=side)
        elif path.suffix == ".jmod":
            with ZipFile(path) as archive:
                classes = [parse_class(archive.read(n), default_side=side) for n in archive.namelist()
                           if n.startswith("classes/") and n.endswith(".class") and not n.endswith("module-info.class")]
        else:
            classes = inspect_jar(
                path, java_version=java_version, default_side=side, require_classes=False,
            ).values()
        for item in classes:
            # Directory classpaths can contain module descriptors as well.
            if item.name == "module-info":
                continue
            if item.name in class_hashes:
                if class_hashes[item.name] != item.content_hash:
                    raise ValueError("CONSTRUCTION_CLASSPATH_AMBIGUOUS: " + item.name)
                # Preserve classpath precedence and the original bytecode source;
                # a later dependency cannot relabel a JDK type as its own API.
                continue
            class_hashes[item.name] = item.content_hash
            owner = canonical_type_name(item.name)
            result[owner] = {"owner": owner, "name": owner.rsplit(".",1)[-1], "kind": "CLASS",
                             "side": item.side, "namespace": namespace, "source_hash": item.content_hash,
                             "superclass": canonical_type_name(item.superclass or ""),
                             "interfaces": [canonical_type_name(x) for x in item.interfaces],
                             "package": item.name.replace('/', '.').rpartition('.')[0],
                             'binary_owner': item.name,
                             'enclosing_owner': canonical_type_name(item.enclosing_class),
                             "access": item.inner_access if item.inner_access is not None else item.access,
                             'enclosing_instance_type': canonical_type_name(item.enclosing_class)
                                if item.enclosing_class and not (item.inner_access or 0) & 8 else '',
                             "source_path": str(path)}
            for member in item.members:
                # Visibility belongs to the actual caller, not inventory
                # construction. Same-package APIs must remain discoverable.
                parameters, returns = descriptor_types(member.descriptor)
                enclosing = result[owner]['enclosing_instance_type'] if member.kind == 'CONSTRUCTOR' else ''
                if enclosing:
                    if not parameters or parameters[0] != enclosing:
                        raise ValueError('CONSTRUCTION_ENCLOSING_CONSTRUCTOR_DESCRIPTOR')
                    parameters = parameters[1:]
                key = owner + "#" + member.name + member.descriptor
                result[key] = {"owner": owner, "name": member.name, "descriptor": member.descriptor,
                    "kind": member.kind, "static": member.is_static, "side": member.side,
                    "namespace": namespace, "parameters": parameters, "return_type": returns,
                    "access": member.access, "source_hash": item.content_hash, "source_path": str(path)}
                result[key]['package'] = result[owner]['package']
                result[key]['binary_owner'] = item.name
                result[key]['throws'] = [canonical_type_name(x) for x in member.exceptions]
                result[key]['enclosing_instance_type'] = enclosing
    if mappings_file is not None:
        from .api_symbol_extractor import APISymbol, MinecraftAPIExtractor
        from .tiny_mappings import TinyMappings
        if source_namespace is None:
            raise ValueError("CONSTRUCTION_SOURCE_NAMESPACE_REQUIRED")
        mapping = TinyMappings.parse(Path(mappings_file).read_text(encoding="utf-8"))
        original = {key: APISymbol(key, row['binary_owner'], row["name"],
                    row.get("descriptor", "L"+row["owner"].replace(".", "/")+";"),
                    row["kind"], row.get("static", False), bool(row["access"] & 1), row["side"],
                    source_namespace, row["source_path"]) for key, row in result.items()}
        mapped = MinecraftAPIExtractor(java_version=java_version).apply_mappings(original, mapping, namespace.lower())
        remapped = {}
        for before, after in zip(original.values(), mapped.values()):
            row = deepcopy(result[before.qualified_name])
            owner = canonical_type_name(after.owner)
            row.update(owner=owner, name=after.name, namespace=namespace,
                       binary_owner=after.owner,
                       package=after.owner.replace('/', '.').rpartition('.')[0])
            if after.kind != "CLASS":
                parameters, returns = descriptor_types(after.descriptor)
                if row.get('enclosing_instance_type') and after.kind == 'CONSTRUCTOR':
                    parameters = parameters[1:]
                row.update(descriptor=after.descriptor, parameters=parameters, return_type=returns)
            remapped[owner if after.kind=="CLASS" else owner+"#"+after.name+after.descriptor] = row
        owner_names = {canonical_type_name(before.owner): canonical_type_name(after.owner)
                       for before, after in zip(original.values(), mapped.values()) if before.kind == 'CLASS'}
        for row in remapped.values():
            row['superclass'] = owner_names.get(row.get('superclass', ''), row.get('superclass', ''))
            row['interfaces'] = [owner_names.get(value, value) for value in row.get('interfaces', ())]
            row['throws'] = [owner_names.get(value, value) for value in row.get('throws', ())]
            row['enclosing_instance_type'] = owner_names.get(row.get('enclosing_instance_type', ''), row.get('enclosing_instance_type', ''))
            row['enclosing_owner'] = owner_names.get(row.get('enclosing_owner', ''), row.get('enclosing_owner', ''))
        result = remapped
    # Model lambdas may target only inspected single-abstract-method interfaces.
    # Derive the callback shape from the resolved bytecode, never from Fabric names.
    annotate_functional_interfaces(result)
    return result


def java_source_units(source, *, path=""):
    """Parse whole declarations. Never slice a statement by a character allowance."""
    from .java_region_parser import _parse, _text, _modifiers
    raw, root = _parse(source)
    package = ""
    imports = []
    for node in root.named_children:
        if node.type == "package_declaration":
            package = _text(raw,node).removeprefix("package ").removesuffix(";").strip()
        if node.type == "import_declaration":
            imports.append(_text(raw,node).removeprefix("import ").removesuffix(";").strip())
    digest = hashlib.sha256(raw).hexdigest()
    result = []
    def visit(node, owner, enclosing_kind=''):
        enclosing_owner = owner if enclosing_kind in {'class', 'interface', 'enum', 'record'} else ''
        kind = {"class_declaration":"class", "interface_declaration":"interface",
                "record_declaration":"record", "enum_declaration":"enum"}.get(node.type)
        if kind:
            name = _text(raw,node.child_by_field_name("name"))
            owner = owner + "." + name if owner else name
        else:
            kind = {"method_declaration":"method", "constructor_declaration":"constructor",
                    "field_declaration":"field", "constant_declaration":"field"}.get(node.type)
        if kind:
            body = node.child_by_field_name("body")
            code = _text(raw,node)
            header = raw[node.start_byte:body.start_byte].decode().strip() if body else code
            name_node = node.child_by_field_name("name")
            parameters_node = node.child_by_field_name("parameters")
            parameters = []
            if parameters_node:
                for p in parameters_node.named_children:
                    ptype, pname = p.child_by_field_name("type"),p.child_by_field_name("name")
                    if p.type == 'spread_parameter':
                        declarator = next((c for c in p.named_children if c.type == 'variable_declarator'), None)
                        ptype = next((c for c in p.named_children if c.type not in {'modifiers', 'variable_declarator', 'annotation', 'marker_annotation'}), None)
                        pname = declarator.child_by_field_name('name') if declarator else None
                    if ptype and pname:
                        dimensions = p.child_by_field_name('dimensions')
                        parameters.append({"type":_text(raw,ptype) + ('[]' if p.type == 'spread_parameter' else _text(raw, dimensions) if dimensions else ''),"name":_text(raw,pname)})
            names = [_text(raw,c.child_by_field_name("name")) for c in node.named_children
                     if c.type=="variable_declarator" and c.child_by_field_name("name")]
            name = _text(raw,name_node) if name_node else ",".join(names)
            return_node = node.child_by_field_name("type")
            modifiers = _modifiers(node, raw)
            parent = node.child_by_field_name('superclass')
            interfaces_node = node.child_by_field_name('interfaces')
            if kind == 'interface':
                interfaces_node = next((c for c in node.named_children if c.type == 'extends_interfaces'), None)
            type_list = next((c for c in interfaces_node.named_children if c.type == 'type_list'), None) if interfaces_node else None
            throws_node = next((c for c in node.named_children if c.type == 'throws'), None)
            implicit_public = enclosing_kind == 'interface' and 'private' not in modifiers
            implicit_static = (enclosing_kind == 'interface' and kind in {'field', 'class', 'interface', 'enum', 'record'}) or (
                bool(enclosing_owner) and kind in {'interface', 'enum', 'record'})
            result.append({"path":path,"owner":owner,"kind":kind,"name":name,
                "signature":header,"parameters":parameters,
                "return_type":_text(raw,return_node) if return_node else "void",
                "code":code,"start_line":node.start_point[0]+1,"end_line":node.end_point[0]+1,
                "source_sha256":digest,"imports":imports, 'package': package,
                'modifiers': sorted(modifiers), 'has_body': body is not None,
                'enclosing_owner': enclosing_owner,
                'implicit_final': enclosing_kind == 'interface' and kind == 'field',
                'superclass': _text(raw, parent.named_children[-1]) if parent and parent.named_children else '',
                'interfaces': [_text(raw, c) for c in type_list.named_children] if type_list else [],
                'throws': [_text(raw, c) for c in throws_node.named_children] if throws_node else [],
                "static": 'static' in modifiers or implicit_static,
                "public": bool({'public', 'protected'} & modifiers) or implicit_public,
                'implicit_public': implicit_public,
                "side":"CLIENT" if path.startswith("src/client/") else "COMMON"})
        if node.type in {"program", "class_body", "interface_body", "enum_body", 'enum_body_declarations',
                         "class_declaration", "interface_declaration", "record_declaration", "enum_declaration"}:
            for child in node.named_children:
                visit(child,owner, kind if kind in {'class', 'interface', 'enum', 'record'} else enclosing_kind)
    visit(root,package)
    return result


def bind_task_context(task, source_evidence, project_symbols, target_symbols):
    available = {**target_symbols, **project_symbols}
    selected = {}
    for key in task.input_symbols:
        if key not in available:
            raise ValueError("CONSTRUCTION_SYMBOL_MISSING: " + key)
        row = available[key]
        if row.get("side") == "CLIENT" and not task.target_path.startswith("src/client/"):
            raise ValueError("CONSTRUCTION_SIDE_MISMATCH: " + key)
        selected[key] = deepcopy(row)
    units = []
    ref_ids = {r.get("evidence_id") for r in task.source_refs}
    for evidence in source_evidence:
        row = evidence.to_dict() if hasattr(evidence,"to_dict") else evidence
        if not ref_ids or row.get("evidence_id") not in ref_ids:
            continue
        for unit in row.get("source_units", ()):
            units.append(deepcopy(unit))
    return {"task_id":task.task_id,"intent":task.intent,"input_symbols":selected,
        "source_units":units,"state_reads":list(task.state_reads),"state_writes":list(task.state_writes),
        "trigger":task.trigger,"effects":task.effects,
        "output_contract":{"target_path":task.target_path,"owner_symbol":task.owner_symbol,
                           "output_symbols":list(task.output_symbols)},
        "prerequisite_hashes":{key:row.get("source_hash", "") for key,row in selected.items()}}
