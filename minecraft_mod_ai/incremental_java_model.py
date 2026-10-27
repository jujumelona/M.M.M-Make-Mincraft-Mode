"""Host-owned program identity, declarations and lexical work-item defaults."""
from __future__ import annotations

import re

from .direct_feature_source import behavior_registrar_name


def method_metadata(return_type="void", parameters=(), *, static=False):
    return {"visibility": "private", "static": static, "return_type": return_type,
            "parameters": list(parameters), "type_parameters": "", "throws": [], "annotations": []}


def initial_program(spec, behaviors):
    bindings = spec.get("bindings") or {}
    class_name = bindings.get("class_name") or spec["target_path"].rsplit("/", 1)[-1][:-5]
    package = bindings.get("package_name") or spec["target_path"].split("/java/", 1)[1].rsplit("/", 1)[0].replace("/", ".")
    client = str(spec.get("side") or "COMMON").upper() == "CLIENT"
    program = {"schema": "mmm/incremental-java-v1", "package": package, "class_name": class_name,
               "interface": "net.fabricmc.api." + ("ClientModInitializer" if client else "ModInitializer"),
               "lifecycle": "onInitializeClient" if client else "onInitialize", "members": []}
    for index, behavior in enumerate(behaviors, 1):
        program["members"].append({"id": "m" + str(index), "owner": class_name, "kind": "method",
            "name": behavior_registrar_name(behavior["behavior_id"]), "metadata": method_metadata(),
            "purpose": "Install the activation and connect the implementation for: " + str(behavior.get("trigger") or behavior["intent"]),
            "behavior": dict(behavior), "registrar": True, "body": []})
    return program


def initial_scope(member):
    return {"locals": [dict(p) for p in member["metadata"].get("parameters", ())],
            "return_type": member["metadata"].get("return_type", "void"),
            "loop_depth": 0, "switch_depth": 0, "owner": member["owner"],
            "static": bool(member["metadata"].get("static"))}


def allocation_counters(program):
    from .incremental_java_history import walk_nodes
    counters = program.setdefault("allocations", {})
    def reserve(name):
        match = re.fullmatch(r"mmm(Value|LambdaArg|Caught)(\d+)", str(name or ""))
        if match:
            prefix = {"Value": "v", "LambdaArg": "p", "Caught": "e"}[match[1]]
            counters[prefix] = max(counters.get(prefix, 0), int(match[2]))
    for member in program["members"]:
        for node in walk_nodes(member.get("body", [])):
            reserve(node.get("name"))
            for parameter in [*node.get("parameters", ()), *node.get("catches", ())]:
                reserve(parameter.get("name"))
    return counters
