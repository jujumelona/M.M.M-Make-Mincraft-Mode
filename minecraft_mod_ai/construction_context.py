"""Inspected symbols and complete source units for one construction owner."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile


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
            value, offset = descriptor[offset+1:end].replace("/", ".").replace("$", "."), end+1
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
            owner = item.name.replace("/", ".").replace("$", ".")
            result[owner] = {"owner": owner, "name": owner.rsplit(".",1)[-1], "kind": "CLASS",
                             "side": item.side, "namespace": namespace, "source_hash": item.content_hash,
                             "superclass": (item.superclass or "").replace("/", "."),
                             "interfaces": [x.replace("/", ".") for x in item.interfaces],
                             "access": item.access, "source_path": str(path)}
            for member in item.members:
                if not member.access & 5:
                    continue
                parameters, returns = descriptor_types(member.descriptor)
                key = owner + "#" + member.name + member.descriptor
                result[key] = {"owner": owner, "name": member.name, "descriptor": member.descriptor,
                    "kind": member.kind, "static": member.is_static, "side": member.side,
                    "namespace": namespace, "parameters": parameters, "return_type": returns,
                    "access": member.access, "source_hash": item.content_hash, "source_path": str(path)}
    if mappings_file is not None:
        from .api_symbol_extractor import APISymbol, MinecraftAPIExtractor
        from .tiny_mappings import TinyMappings
        if source_namespace is None:
            raise ValueError("CONSTRUCTION_SOURCE_NAMESPACE_REQUIRED")
        mapping = TinyMappings.parse(Path(mappings_file).read_text(encoding="utf-8"))
        original = {key: APISymbol(key, row["owner"].replace(".", "/"), row["name"],
                    row.get("descriptor", "L"+row["owner"].replace(".", "/")+";"),
                    row["kind"], row.get("static", False), bool(row["access"] & 1), row["side"],
                    source_namespace, row["source_path"]) for key, row in result.items()}
        mapped = MinecraftAPIExtractor(java_version=java_version).apply_mappings(original, mapping, namespace.lower())
        remapped = {}
        for before, after in zip(original.values(), mapped.values()):
            row = deepcopy(result[before.qualified_name])
            owner = after.owner.replace("/", ".")
            row.update(owner=owner, name=after.name, namespace=namespace)
            if after.kind != "CLASS":
                parameters, returns = descriptor_types(after.descriptor)
                row.update(descriptor=after.descriptor, parameters=parameters, return_type=returns)
            remapped[owner if after.kind=="CLASS" else owner+"#"+after.name+after.descriptor] = row
        result = remapped
    # Model lambdas may target only inspected single-abstract-method interfaces.
    # Derive the callback shape from the resolved bytecode, never from Fabric names.
    from collections import defaultdict
    abstract_by_owner = defaultdict(list)
    for member in result.values():
        if (member.get("kind") == "METHOD"
                and member.get("access", 0) & 0x0400
                and not member.get("access", 0) & (0x0040 | 0x1000)):
            abstract_by_owner[member["owner"]].append(member)
    for symbol in result.values():
        if symbol.get("kind") != "CLASS" or not symbol.get("access", 0) & 0x0200:
            continue
        abstract_members = abstract_by_owner.get(symbol["owner"], ())
        if len(abstract_members) == 1:
            member = abstract_members[0]
            symbol["functional_parameters"] = list(member["parameters"])
            symbol["functional_return_type"] = member["return_type"]
    return result


def java_source_units(source, *, path=""):
    """Parse whole declarations. Never slice a statement by a character allowance."""
    from .java_region_parser import _parse, _text
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
    def visit(node, owner):
        kind = {"class_declaration":"class", "interface_declaration":"interface",
                "record_declaration":"record", "enum_declaration":"enum"}.get(node.type)
        if kind:
            name = _text(raw,node.child_by_field_name("name"))
            owner = owner + "." + name if owner else name
        else:
            kind = {"method_declaration":"method", "constructor_declaration":"constructor",
                    "field_declaration":"field"}.get(node.type)
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
                    if ptype and pname:
                        parameters.append({"type":_text(raw,ptype),"name":_text(raw,pname)})
            names = [_text(raw,c.child_by_field_name("name")) for c in node.named_children
                     if c.type=="variable_declarator" and c.child_by_field_name("name")]
            name = _text(raw,name_node) if name_node else ",".join(names)
            return_node = node.child_by_field_name("type")
            result.append({"path":path,"owner":owner,"kind":kind,"name":name,
                "signature":header,"parameters":parameters,
                "return_type":_text(raw,return_node) if return_node else "void",
                "code":code,"start_line":node.start_point[0]+1,"end_line":node.end_point[0]+1,
                "source_sha256":digest,"imports":imports,
                "static": bool(re.search(r"\bstatic\b",header)),
                "public": bool(re.search(r"\b(public|protected)\b",header)),
                "side":"CLIENT" if path.startswith("src/client/") else "COMMON"})
        if node.type in {"program", "class_body", "interface_body", "enum_body",
                         "class_declaration", "interface_declaration", "record_declaration", "enum_declaration"}:
            for child in node.named_children:
                visit(child,owner)
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
