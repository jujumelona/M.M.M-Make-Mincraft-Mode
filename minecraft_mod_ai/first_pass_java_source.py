"""Construct Java source from small, independently completed model actions.

This is a first-pass production protocol, not an output-exhaustion retry.  The
host owns the package boundary, final closing brace, and source assembly.  The
model supplies typed metadata and completes fixed-prefix Java declarations. Required
Fabric entrypoint declarations are immutable host data, not model suggestions.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .fixed_template_generation import (
    _validate_native_arguments,
)
from .model_output_atomicity_contract import assert_atomic_model_schema
from .first_pass_java_declarations import (
    IDENTIFIER, MEMBER_KINDS, declaration_frame, declaration_header,
    metadata_schema, render_declaration, validate_declaration_metadata,
)


_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "imports": {"type": "array", "maxItems": 24, "items": {"type": "string", "maxLength": 240}},
        "superclass": {"type": "string", "maxLength": 240},
        "interfaces": {"type": "string", "maxLength": 480},
        "units": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(MEMBER_KINDS)},
                    "name": IDENTIFIER,
                    "purpose": {"type": "string", "maxLength": 160},
                },
                "required": ["kind", "name", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["imports", "superclass", "interfaces", "units"],
    "additionalProperties": False,
}

_QUALIFIED_JAVA_IMPORT = re.compile(
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+[A-Za-z_$][A-Za-z0-9_$]*|"
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+\*"
)


def _assemble_java_imports(entries: Sequence[str]) -> str:
    """Create Java import statements from model-supplied qualified type names.

    The model supplies dependency references, not Java declaration syntax.
    Legacy import statements are accepted as suggestions, but arbitrary prose
    and executable fragments never enter the generated source header.
    """
    statements: list[str] = []
    for suggestion in entries:
        if not isinstance(suggestion, str):
            continue
        for fragment in re.split(r"[;\r\n,]", suggestion):
            reference = fragment.strip()
            reference = re.sub(r"^import\s+", "", reference).strip()
            is_static = reference.startswith("static ")
            if is_static:
                reference = reference[len("static "):].strip()
            if not _QUALIFIED_JAVA_IMPORT.fullmatch(reference):
                continue
            statements.append(
                "import " + ("static " if is_static else "") + reference + ";"
            )
    return "\n".join(dict.fromkeys(statements))


def _one_structured_action(
    router: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    name: str,
    schema: Mapping[str, Any],
    page_structure: bool = True,
) -> Mapping[str, Any]:
    """One attempt for one small action. No schema-repair or source regeneration."""
    assert_atomic_model_schema(schema, surface=f"first-pass Java {name}")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError("JAVA_DATA_PAGE_REQUIRES_DECLARED_FIELDS")
    # Select page boundaries from the data structure, not an invented token cap.
    # Arrays are planned once by count and authored one entry at a time. Every
    # page is an independent first-pass request, never a failed-output replay.
    if page_structure and len(properties) > 1:
        merged: dict[str, Any] = {}
        for field, child in properties.items():
            page = {"type": "object", "properties": {field: child},
                    "required": [field], "additionalProperties": False}
            merged.update(_one_structured_action(
                router, [*messages, {"role": "user", "content": json.dumps(
                    {"current_field": field, "completed_plan_fields": merged},
                    ensure_ascii=False)}], name=f"{name}_{field}", schema=page,
            ))
        return _validate_native_arguments(merged, schema)

    field, child = next(iter(properties.items()))
    if page_structure and child.get("type") == "object":
        value = _one_structured_action(
            router, messages, name=f"{name}_{field}", schema=child,
        )
        return _validate_native_arguments({field: value}, schema)
    if page_structure and child.get("type") == "array":
        maximum = int(child["maxItems"])
        minimum = int(child.get("minItems", 0))
        count_schema = {"type": "object", "properties": {"count": {
            "type": "integer", "enum": list(range(minimum, maximum + 1))}},
            "required": ["count"], "additionalProperties": False}
        count = _one_structured_action(
            router, [*messages, {"role": "user", "content":
                f"Choose the number of necessary {field} entries; include all requirements."}],
            name=f"{name}_count", schema=count_schema,
        )["count"]
        values: list[Any] = []
        for index in range(count):
            page_schema = {"type": "object", "properties": {"value": child["items"]},
                           "required": ["value"], "additionalProperties": False}
            item = _one_structured_action(
                router, [*messages, {"role": "user", "content": json.dumps({
                    "field": field, "index": index, "count": count,
                    "previous_entries": values,
                    "instruction": "Return the next necessary entry without duplicates."},
                    ensure_ascii=False)}], name=f"{name}_{index + 1}", schema=page_schema,
            )
            values.append(item["value"])
        return _validate_native_arguments({field: values}, schema)

    response = router.generate_structured_data(
        "coder", messages, data_name=name,
        parameters=dict(schema),
    )
    if not isinstance(response, Mapping):
        raise ValueError(f"JAVA_FIRST_PASS_ACTION_MISSING: {name}")
    return _validate_native_arguments(response, schema)


def _java_identity(spec: Mapping[str, Any]) -> tuple[str, str] | None:
    """Infer only the declared compilation-unit identity, never a guessed API."""
    if str(spec.get("language", "")).casefold() != "java":
        return None
    if spec.get("java_prefix") or spec.get("java_suffix"):
        # Fragment/anchor contracts do not own their surrounding Java class.
        return None
    path = str(spec.get("target_path") or "").replace("\\", "/")
    source_root = next(
        (prefix for prefix in ("src/main/java/", "src/client/java/")
         if path.startswith(prefix)), None,
    )
    if source_root is None or not path.endswith(".java"):
        return None
    relative = PurePosixPath(path[len(source_root):])
    inferred_class = relative.stem
    inferred_package = ".".join(relative.parts[:-1])
    bindings = spec.get("bindings") or {}
    if not isinstance(bindings, Mapping):
        return None
    class_name = str(bindings.get("class_name") or inferred_class)
    package_name = str(bindings.get("package_name") or inferred_package)
    identifier = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
    if not identifier.fullmatch(class_name) or not package_name or any(
        not identifier.fullmatch(part) for part in package_name.split(".")
    ):
        return None
    if class_name != inferred_class or package_name != inferred_package:
        # Respect the actual target path rather than manufacturing a mismatched class.
        return None
    return package_name, class_name


def can_compose_first_pass_java(spec: Mapping[str, Any]) -> bool:
    return _java_identity(spec) is not None


def compose_first_pass_java(
    router: Any,
    spec: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    *,
    source_context: Sequence[Any] = (),
) -> str:
    identity = _java_identity(spec)
    if identity is None:
        raise ValueError("JAVA_FIRST_PASS_REQUIRES_COMPLETE_COMPILATION_UNIT")
    package_name, class_name = identity

    task = {
        "package": package_name,
        "class": class_name,
        "requirement": spec.get("requirement", ""),
        "bindings": spec.get("bindings", {}),
        "prerequisite_source": source_context,
        "side": spec.get("side"),
        "target_path": spec.get("target_path"),
        "implementation_requirements": spec.get("bindings", {}).get("java_implementation_requirements", ""),
    }
    bindings = task["bindings"]
    # Use the persisted owner roles which already select Fabric entrypoints.
    # COMMON request-feature jobs and CLIENT canonical jobs have fixed loader APIs.
    if bindings.get("request_feature_entrypoint"):
        entrypoint_interface = "net.fabricmc.api.ModInitializer"
        entrypoint_method = "onInitialize"
    elif str(spec.get("side") or "").upper() == "CLIENT":
        entrypoint_interface = "net.fabricmc.api.ClientModInitializer"
        entrypoint_method = "onInitializeClient"
    else:
        entrypoint_interface = entrypoint_method = ""
    plan_schema = deepcopy(_PLAN_SCHEMA)
    if entrypoint_interface:
        # Initializer inheritance, constructor and lifecycle are already selected
        # by the source contract. Do not ask the model to select them again.
        for field in ("superclass", "interfaces"):
            plan_schema["properties"].pop(field)
        unit_schema = plan_schema["properties"]["units"]["items"]
        unit_schema["properties"].pop("name")
        unit_schema["properties"]["kind"]["enum"] = [kind for kind in MEMBER_KINDS if kind != "constructor"]
        unit_schema["required"] = list(unit_schema["properties"])
        plan_schema["properties"]["units"]["minItems"] = 0
        plan_schema["required"] = list(plan_schema["properties"])
        task["host_entrypoint"] = {"interface": entrypoint_interface, "method": entrypoint_method}
    plan_messages = [
        *messages,
        {
            "role": "system",
            "content": (
                "Plan typed Java field/method/constructor/nested-type responsibilities. "
                "The host fixes every declaration from typed metadata; later complete "
                "one Java declaration from its immutable signature per request. "
                "Account for every required "
                "behavior and integration, including constructors and registration "
                "when required; never simplify the gameplay specification. "
                "List Java imports as an array of fully-qualified type references, "
                "for example ['java.util.List', 'java.util.Map']. "
                "Do not include import keywords, semicolons or package lines. "
                "Use 'static package.Type.member' only for a static import. "
                "The HOST writes the package and class header using the exact "
                f"class name {class_name}; do not generate those declarations. "
                "Provide superclass as just the optional Java type name (no "
                "'extends' keyword), and interfaces as optional comma-separated "
                "Java type names (no 'implements' keyword). Empty strings mean "
                "none. Preserve actual API inheritance/interface requirements. "
                "When host_entrypoint is present, its public constructor, interface and "
                "initialization method are already owned by the host. Plan only needed "
                "supporting members; do not plan a second lifecycle method. The host "
                "assigns internal helper names and later authors the required lifecycle body. "
                "Plan only real necessary supporting members, at most ten; "
                "each planned unit is one complete declaration with one coherent "
                "responsibility. No code in the plan units."
            ),
        },
        {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)},
    ]
    plan = _one_structured_action(
        router,
        plan_messages,
        name="submit_java_construction_plan",
        schema=plan_schema,
    )

    imports = _assemble_java_imports(plan["imports"])
    superclass = str(plan.get("superclass") or "").strip()
    interfaces = entrypoint_interface or str(plan.get("interfaces") or "").strip()
    # The model never authors identity, scope, or outer braces. It supplies only
    # optional Java type references used in a deterministic host-owned header.
    # Reject executable fragments, not legitimate inheritance relationships.
    for type_clause in (superclass, interfaces):
        if any(marker in type_clause for marker in ("{", "}", ";", chr(10), chr(13), "/*", "//")):
            raise ValueError("JAVA_FIRST_PASS_TYPE_REFERENCE_INVALID")
    declaration = (
        f"public {'final ' if entrypoint_interface else ''}class {class_name}"
        + (f" extends {superclass}" if superclass else "")
        + (f" implements {interfaces}" if interfaces else "")
        + " {"
    )
    units = []
    for index, item in enumerate(plan["units"], 1):
        unit = dict(item)
        if entrypoint_interface:
            # These helpers have no preexisting public API. Select their names
            # before inference, outside the reserved entrypoint namespace.
            unit["name"] = ("MmmMember" if unit["kind"] in {"class", "record", "enum"} else "mmmMember") + str(index)
        elif unit["kind"] == "constructor":
            unit["name"] = class_name
        units.append(unit)
    declarations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, unit in enumerate(units, 1):
        unit_task = {
            **task,
            "imports": imports,
            "class_declaration": declaration,
            "all_units": units,
            "current_unit": unit,
            "previous_declaration_metadata": declarations,
        }
        metadata_messages = [
            {
                "role": "system",
                "content": (
                    "Supply only typed declaration metadata for current_unit. The host "
                    "owns its kind and exact name. Types are Java types, parameters are "
                    "type/name records, and annotations are annotation names with optional "
                    "arguments. Do not place a complete declaration or class in a scalar. "
                    "Use implementation_requirements, side and verified prerequisite APIs. "
                    "Preserve exact override names for nested API implementations. "
                    "For a nested type supply typed field/constructor/method metadata. "
                    "Implementation is requested only after every signature is available."
                ),
            },
            {"role": "user", "content": json.dumps(unit_task, ensure_ascii=False, default=str)},
        ]
        kind, member_name = unit["kind"], unit["name"]
        nested_type = kind in {"class", "record", "enum"}
        parts = dict(_one_structured_action(
            router,
            metadata_messages,
            name=f"submit_java_declaration_{index}",
            schema=metadata_schema(kind, member_name),
            page_structure=nested_type,
        ))
        validate_declaration_metadata(kind, member_name, parts, owner=class_name)
        declarations.append((unit, parts))

    # All method/field signatures are fixed before any body is authored. Forward
    # calls therefore use actual parameter/return types rather than guessed APIs.
    signature_index: list[dict[str, str]] = []
    owner_types: dict[str, dict[str, Any]] = {
        class_name: {"kind": "class", "name": class_name, "declaration": declaration},
    }
    def index_declaration(kind, name, parts, owner):
        header = declaration_header(kind, name, parts)
        signature_index.append({"owner": owner, "kind": kind, "name": name,
                                "signature": header})
        if kind in {"class", "record", "enum"}:
            owner_types[name] = {"kind": kind, "name": name, "declaration": header}
            if kind == "enum":
                owner_types[name]["constants"] = deepcopy(parts["constants"])
                for constant in parts["constants"]:
                    signature_index.append({"owner": name, "kind": "enum_constant",
                                            "name": constant["name"], "signature": constant["name"]})
            elif kind == "record":
                owner_types[name]["components"] = deepcopy(parts["components"])
            for child in parts["fields"]:
                index_declaration("field", child["name"], child, name)
            for child in parts["constructors"]:
                index_declaration("constructor", name, child, name)
            for child in parts["methods"]:
                index_declaration("method", child["name"], child, name)
    for unit, parts in declarations:
        index_declaration(unit["kind"], unit["name"], parts, class_name)

    finished: list[str] = []
    def implement(kind, name, parts, *, owner, action, purpose):
        if kind in {"class", "record", "enum"}:
            for index, child in enumerate(parts["fields"], 1):
                implement("field", child["name"], child, owner=name, action=f"{action}_field_{index}", purpose=purpose)
            for index, child in enumerate(parts["constructors"], 1):
                implement("constructor", name, child, owner=name, action=f"{action}_constructor_{index}", purpose=purpose)
            for index, child in enumerate(parts["methods"], 1):
                implement("method", child["name"], child, owner=name, action=f"{action}_method_{index}", purpose=purpose)
            return
        # No static initializer blocks are authored by this protocol. A final
        # field needs an expression unless an instance constructor can assign it.
        allow_uninitialized = not parts.get("final") or (
            not parts.get("static") and any(
                member["kind"] == "constructor" and member["owner"] == owner
                for member in signature_index
            )
        )
        frame = declaration_frame(kind, name, parts, allow_uninitialized=allow_uninitialized)
        body_task = {**task, "imports": imports, "current_unit_purpose": purpose,
                     "class_declaration": declaration, "owner_type": owner_types[owner],
                     "owner": owner, "declaration_metadata": parts,
                     "host_signature": declaration_header(kind, name, parts),
                     "response_source_frame": frame,
                     "declared_member_api": signature_index, "previous_complete_members": finished}
        body_messages = [
            {"role": "system", "content": (
                "Return one COMPLETE raw Java field/method/constructor declaration. "
                "Start with response_source_frame.prefix and end with its suffix; the "
                "sampler fixes this signature. For an uninitialized field its exact "
                "empty_value is also allowed. Complete the expression or statements "
                "in that declaration. Do not repeat the signature inside its implementation. "
                "Never return package/import directives, an outer class or extra members. "
                "Respect implementation_requirements, "
                "side, current_unit_purpose and actual prerequisite/member APIs. "
                "declared_member_api contains the exact signatures of all helpers, "
                "including forward references. Implement the requested behavior "
                "without placeholders or TODOs."
            )},
            {"role": "user", "content": json.dumps(body_task, ensure_ascii=False, default=str)},
        ]
        parts["source"] = router.generate_java_source("coder", body_messages,
            data_name=action, source_frame=frame)
    for index, (unit, parts) in enumerate(declarations, 1):
        kind, member_name = unit["kind"], unit["name"]
        implement(kind, member_name, parts, owner=class_name,
                  action=f"submit_java_implementation_{index}", purpose=unit["purpose"])
        finished.append(render_declaration(kind, member_name, parts, owner=class_name))

    if entrypoint_interface:
        # The lifecycle body is generated after helpers exist, against the exact
        # assigned COMMON/CLIENT contract. Neither its name nor signature is AI data.
        lifecycle = {"visibility": "public", "static": False, "return_type": "void",
                     "parameters": [], "throws": [], "annotations": ["Override"],
                     "type_parameters": ""}
        frame = declaration_frame("method", entrypoint_method, lifecycle)
        lifecycle_messages = [
            {"role": "system", "content": (
                "Return the COMPLETE raw Java initialization method, starting "
                "with response_source_frame.prefix and ending with its suffix. The sampler "
                "fixes its signature; complete its body without repeating that signature. "
                "Register and implement the exact original feature using the "
                "verified APIs and completed helper declarations. Respect the assigned "
                "side and implementation_requirements; retain existing typed behavior "
                "without duplicating its effects. Return no package/import directives, "
                "class declaration or extra method."
            )},
            {"role": "user", "content": json.dumps({**task, "imports": imports,
                "host_signature": declaration_header("method", entrypoint_method, lifecycle),
                "response_source_frame": frame,
                "complete_members": finished}, ensure_ascii=False, default=str)},
        ]
        lifecycle["source"] = router.generate_java_source("coder", lifecycle_messages,
            data_name="submit_java_entrypoint_source", source_frame=frame)
        finished.append(render_declaration("method", entrypoint_method, lifecycle, owner=class_name))

    return (
        f"package {package_name};\n\n"
        + (imports + "\n\n" if imports else "")
        + declaration + "\n\n"
        + "\n\n".join(finished)
        + "\n}\n"
    )
