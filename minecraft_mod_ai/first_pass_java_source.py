"""Construct Java source from small, independently completed model actions.

This is a first-pass production protocol, not an output-exhaustion retry.  The
host owns the package boundary, final closing brace, and source assembly.  The
model authors a brief API-aware class plan followed by complete class members;
it is never asked to serialize a complete multi-kilobyte source file at once.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .fixed_template_generation import (
    _model_transport_schema,
    _validate_native_arguments,
)
from .model_output_atomicity_contract import assert_atomic_model_schema


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
                    "name": {"type": "string", "maxLength": 80},
                    "purpose": {"type": "string", "maxLength": 160},
                },
                "required": ["name", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["imports", "superclass", "interfaces", "units"],
    "additionalProperties": False,
}

_MEMBER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "member": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2600,
        },
    },
    "required": ["member"],
    "additionalProperties": False,
}


_QUALIFIED_JAVA_IMPORT = re.compile(
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\\.)+[A-Za-z_$][A-Za-z0-9_$]*|"
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\\.)+\\*"
)


def _assemble_java_imports(entries: Sequence[str]) -> str:
    """Create legal Java import statements from model-supplied type references.

    A model need not serialize import keywords, semicolons or line layout.
    Tolerate legacy complete import statements and collapsed lines as input,
    but never copy arbitrary model text into the Java compilation-unit header.
    Syntax-irrelevant or unusable suggestions are omitted, leaving semantic
    dependency errors to the real Java compiler rather than a formatting gate.
    """
    statements: list[str] = []
    for suggestion in entries:
        if not isinstance(suggestion, str):
            continue
        for fragment in re.split(r"[;\\r\\n,]", suggestion):
            reference = fragment.strip()
            reference = re.sub(r"^import\\s+", "", reference).strip()
            is_static = reference.startswith("static ")
            if is_static:
                reference = reference[len("static "):].strip()
            if not _QUALIFIED_JAVA_IMPORT.fullmatch(reference):
                continue
            statements.append(
                "import " + ("static " if is_static else "") + reference + ";"
            )
    return "\\n".join(dict.fromkeys(statements))


def _one_structured_action(
    router: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    name: str,
    schema: Mapping[str, Any],
    output_tokens: int,
) -> Mapping[str, Any]:
    """One attempt for one small action. No schema-repair or source regeneration."""
    assert_atomic_model_schema(schema, surface=f"first-pass Java {name}")
    response = router.generate_tool_decision(
        "coder",
        messages,
        tool_name=name,
        parameters=_model_transport_schema(dict(schema)),
        description="Return the exact requested Java construction page, once.",
        output_token_ceiling=output_tokens,
        force_non_thinking=True,
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
    }
    plan_messages = [
        *messages,
        {
            "role": "system",
            "content": (
                "Design the COMPLETE Java compilation unit as small, independently "
                "finishable class-member responsibilities. Account for every required "
                "behavior and integration, including constructors and registration "
                "when required; never simplify the gameplay specification. "
                "List Java imports as an array of fully-qualified type references, "
                "for example [\\\"java.util.List\\\", \\\"java.util.Map\\\"]. "
                "Do not include import keywords, semicolons or package lines. "
                "Use \\\"static package.Type.member\\\" only for a static import. "
                "The HOST writes the package and class header using the exact "
                f"class name {class_name}; do not generate those declarations. "
                "Provide superclass as just the optional Java type name (no "
                "'extends' keyword), and interfaces as optional comma-separated "
                "Java type names (no 'implements' keyword). Empty strings mean "
                "none. Preserve actual API inheritance/interface requirements. "
                "Plan only real necessary members, at most ten; "
                "each planned unit must fit in 2600 Java source characters, "
                "including complete method/field bodies. No code in the plan units."
            ),
        },
        {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)},
    ]
    plan = _one_structured_action(
        router,
        plan_messages,
        name="submit_java_construction_plan",
        schema=_PLAN_SCHEMA,
        output_tokens=1536,
    )

    imports = _assemble_java_imports(plan["imports"])
    superclass = str(plan["superclass"]).strip()
    interfaces = str(plan["interfaces"]).strip()
    # The model never authors identity, scope, or outer braces. It supplies only
    # optional Java type references used in a deterministic host-owned header.
    # Reject executable fragments, not legitimate inheritance relationships.
    for type_clause in (superclass, interfaces):
        if any(marker in type_clause for marker in ("{", "}", ";", chr(10), chr(13), "/*", "//")):
            raise ValueError("JAVA_FIRST_PASS_TYPE_REFERENCE_INVALID")
    declaration = (
        f"public class {class_name}"
        + (f" extends {superclass}" if superclass else "")
        + (f" implements {interfaces}" if interfaces else "")
        + " {"
    )
    finished: list[str] = []
    for index, unit in enumerate(plan["units"], 1):
        unit_task = {
            **task,
            "imports": imports,
            "declaration": declaration,
            "all_units": plan["units"],
            "current_unit": unit,
            "previous_complete_members": finished,
        }
        member_messages = [
            {
                "role": "system",
                "content": (
                    "Write ONLY the complete Java class-body field/constructor/method "
                    "declarations needed for current_unit. No package, imports, "
                    "outer class braces, prose, Markdown fences, stubs, TODOs, "
                    "or dangling partial methods. Use the listed import and class "
                    "API and the previously authored members as context. "
                    "Never rewrite or duplicate previous members. Implement actual "
                    "behavior rather than a placeholder. Keep this self-contained "
                    "member page within 2600 characters; the host assembles all pages."
                ),
            },
            {"role": "user", "content": json.dumps(unit_task, ensure_ascii=False, default=str)},
        ]
        result = _one_structured_action(
            router,
            member_messages,
            name=f"submit_java_member_{index}",
            schema=_MEMBER_SCHEMA,
            output_tokens=2304,
        )
        finished.append(str(result["member"]).strip())

    return (
        f"package {package_name};\n\n"
        + (imports + "\n\n" if imports else "")
        + declaration + "\n\n"
        + "\n\n".join(finished)
        + "\n}\n"
    )
