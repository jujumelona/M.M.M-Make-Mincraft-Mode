from __future__ import annotations

"""Single authority for execution, generation, parsing, and repair contracts.

Hard limits, model-facing payload classes, Java ownership, coder-facing constraints,
and recovery classification are defined here. Runtime consumers may expose compatibility
facades, but they must not independently redefine these rules.
"""

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


SCHEMA_CONTRACT_PROFILE_KEY = "x-mmm-contract-profile"
SCHEMA_STRING_CLASS_KEY = "x-mmm-string-class"

DEFAULT_SCHEMA_PROFILE = "atomic"
SOURCE_REPAIR_SCHEMA_PROFILE = "source_repair"

STRING_CLASS_GENERIC = "generic"
STRING_CLASS_SOURCE = "source"
STRING_CLASS_REPAIR_SPAN = "repair_span"


@dataclass(frozen=True)
class AtomicSchemaLimits:
    max_fields: int
    max_string_chars: int
    max_array_items: int
    max_schema_depth: int


DEFAULT_ATOMIC_SCHEMA_LIMITS = AtomicSchemaLimits(
    max_fields=3,
    max_string_chars=256,
    max_array_items=4,
    max_schema_depth=3,
)
SOURCE_REPAIR_ATOMIC_SCHEMA_LIMITS = AtomicSchemaLimits(
    max_fields=4,
    max_string_chars=256,
    max_array_items=4,
    max_schema_depth=4,
)

ATOMIC_SCHEMA_PROFILES = MappingProxyType({
    DEFAULT_SCHEMA_PROFILE: DEFAULT_ATOMIC_SCHEMA_LIMITS,
    SOURCE_REPAIR_SCHEMA_PROFILE: SOURCE_REPAIR_ATOMIC_SCHEMA_LIMITS,
})

SOURCE_REPAIR_MAX_SOURCE_CHARS = 16_384
SOURCE_REPAIR_MAX_SPAN_CHARS = 4_096
SOURCE_REPAIR_HARD_ATTEMPTS = 2

# Complete source snapshots passed directly from verifier diagnostics to the
# repair path are byte-bounded independently from model-facing character limits.
# Large files fall back to retrieval/localized repair instead of being truncated.
DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES = 12 * 1024

STRING_CLASS_LIMITS = MappingProxyType({
    STRING_CLASS_GENERIC: DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars,
    STRING_CLASS_SOURCE: SOURCE_REPAIR_MAX_SOURCE_CHARS,
    STRING_CLASS_REPAIR_SPAN: SOURCE_REPAIR_MAX_SPAN_CHARS,
})
PROFILE_STRING_CLASSES = MappingProxyType({
    DEFAULT_SCHEMA_PROFILE: frozenset({STRING_CLASS_GENERIC}),
    SOURCE_REPAIR_SCHEMA_PROFILE: frozenset({
        STRING_CLASS_GENERIC,
        STRING_CLASS_SOURCE,
        STRING_CLASS_REPAIR_SPAN,
    }),
})

MODEL_MAX_COMPLETION_TOKENS = 128

DEFAULT_REGION_ATTEMPT_LIMIT = 1
MAX_REGION_ATTEMPT_LIMIT = 4
DEFAULT_COMPILE_REPAIR_LIMIT = 0
MAX_COMPILE_REPAIR_LIMIT = 16

PRODUCTION_REGION_ATTEMPT_LIMIT = 3
PRODUCTION_RETRY_STRUCTURAL_REJECTIONS = True
PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS = True
PRODUCTION_COMPILE_REPAIR_LIMIT = 2
ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING = 4_096
JAVA_ATOMIC_ASSEMBLY_MAX_CALLS = 128
JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS = 32
JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES = 2_048


# ---- Java ownership and admission authority ---------------------------------

JAVA_FENCE_LANGUAGES = frozenset({"", "java", "javac"})
JAVA_EXPLICIT_JDK_IMPORT_PATTERN = (
    r"import\s+(java\.[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+)\s*;"
)
JAVA_QUALIFIABLE_JDK_IMPORT_USE_ROLES = frozenset({"type", "static_receiver"})

JAVA_REGION_RECOVERY_SHAPES = MappingProxyType({
    "members": (
        "direct_class_body",
        "jdk_imported_class_body",
        "outer_class_envelope",
    ),
    "initialize": (
        "direct_statements",
        "jdk_imported_statements",
        "initialize_wrapper",
        "jdk_imported_initialize_wrapper",
        "outer_class_initialize",
    ),
})
JAVA_INITIALIZE_WRAPPER_NAMES = frozenset({"initialize"})
JAVA_INITIALIZE_LOCALIZABLE_FIELD_MODIFIERS = frozenset(
    {"public", "protected", "private", "static", "final", "volatile", "transient"}
)
JAVA_INITIALIZE_PRESERVED_LOCAL_MODIFIERS = frozenset({"final"})

JAVA_NESTED_TYPE_NODE_TYPES = frozenset({
    "annotation_type_declaration",
    "class_declaration",
    "enum_declaration",
    "interface_declaration",
    "record_declaration",
})
JAVA_CONCERN_MEMBER_NODE_TYPES = (
    frozenset({"field_declaration", "method_declaration"})
    | JAVA_NESTED_TYPE_NODE_TYPES
)
JAVA_HOST_OWNED_MEMBER_NODE_TYPES = frozenset({
    "block",
    "constructor_declaration",
    "compact_constructor_declaration",
    "static_initializer",
})
JAVA_NESTED_TYPE_REQUIRED_VISIBILITY = "private"
JAVA_NESTED_TYPE_CANONICALIZABLE_VISIBILITIES = frozenset({"public", "protected"})
JAVA_HOST_INITIALIZE_NAME = "initialize"
JAVA_HOST_INITIALIZE_RETURN_TYPE = "void"
JAVA_HOST_INITIALIZE_PARAMETERS = "()"
JAVA_HOST_INITIALIZE_REQUIRED_MODIFIER = "static"

JAVA_DECLARATION_ONLY_CONCERNS = frozenset({"stored_state"})
JAVA_TYPE_OWNING_CONCERNS = frozenset(
    {"variables", "inputs", "outputs", "stored_state", "payloads"}
)
JAVA_DECLARATION_ONLY_MEMBER_KINDS = frozenset({
    "field_declaration",
    "annotation_type_declaration",
    "class_declaration",
    "enum_declaration",
    "interface_declaration",
    "record_declaration",
})


# ---- Atomic error/recovery taxonomy authority -------------------------------

@dataclass(frozen=True)
class AtomicErrorRecoveryRule:
    prefix: str
    recoverable: bool
    terminal_after_normalization: bool


ATOMIC_ERROR_RECOVERY_RULES = (
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_RESPONSE_INVALID:", True, False),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_SCOPE_ESCAPE:", True, True),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE:", True, False),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_SYMBOL_COLLISION:", True, True),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_OUTPUT_EXHAUSTED:", True, False),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_SEMANTIC_SHAPE_INVALID:", True, False),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_PLATFORM_API_FORBIDDEN:", True, True),
    AtomicErrorRecoveryRule("ATOMIC_CONCERN_UNGROUNDED_PLATFORM_API:", True, True),
)

RECOVERABLE_ATOMIC_ERROR_PREFIXES = tuple(
    rule.prefix for rule in ATOMIC_ERROR_RECOVERY_RULES if rule.recoverable
)
TERMINAL_AFTER_NORMALIZATION_PREFIXES = tuple(
    rule.prefix
    for rule in ATOMIC_ERROR_RECOVERY_RULES
    if rule.terminal_after_normalization
)


# ---- Coder-facing Java generation authority ---------------------------------

JAVA_COMPILER_FIRST_RULES = (
    "The first answer must compile as Java for the selected host JDK; do not rely on a later repair pass.",
    "Never guess a package or fully-qualified class name. Use only a JDK/external type whose canonical package and API are known from the supplied authority.",
    "For non-java.lang JDK types, prefer canonical fully-qualified names. If an explicit java.* import is emitted, the host will remove it and qualify both type uses and static class receivers before validation.",
    "A field declaration type must be assignment-compatible with its initializer, and every receiver method call must exist on that declared type. Never use Map/List/Object as a lock holder merely because the field also guards cached state.",
    "Every concern-local final field must be definitely assigned before any read. Prefer initialization at the declaration; use a blank final only when the same region performs exactly one unconditional assignment in a static initializer.",
    "Never reassign a final field. If the binding must change, declare a non-final field; if a final field holds a mutable container, mutate the container rather than rebinding the field.",
    "Nested runtime types live inside a host-owned outer class. Their visibility is host-owned and canonicalized to private; never depend on public/protected nested visibility.",
    "Respect available_sibling_api types, generic arguments, and mutability exactly; final sibling fields are read-only after declaration.",
    "Java generics are invariant. Never narrow Map<K,Object> to Map<K,String>, List<Object> to List<String>, or any sibling generic declaration to a different type argument.",
    "For Map<K,V>.entrySet(), declare the iterator element as Map.Entry<K,V>; entry.getKey() has exact type K and entry.getValue() has exact type V. Preserve those exact types in local variables.",
    "Keep generic types exact. When an API returns Object, never return it directly from a method with a narrower generic/container return type and never use an unchecked cast as a shortcut; narrow the individual Object value with instanceof/pattern matching and provide a type-compatible fallback.",
    "Avoid raw collections and unchecked operations when a parameterized type or runtime type check can express the contract.",
    "For java.util.concurrent locks, Lock and ReentrantLock are in java.util.concurrent.locks, not java.util.concurrent.",
)

JAVA_JDK_PACKAGE_ANCHORS = (
    ("collections_and_core_util", "java.util"),
    ("concurrency_executors_and_concurrent_collections", "java.util.concurrent"),
    ("locks", "java.util.concurrent.locks"),
    ("lock_interface", "java.util.concurrent.locks.Lock"),
    ("reentrant_lock", "java.util.concurrent.locks.ReentrantLock"),
    ("atomics", "java.util.concurrent.atomic"),
    ("time", "java.time"),
)

JAVA_PRE_EMIT_COMPILE_CHECKLIST = (
    "Before emitting Java, internally type-check every assignment: declared_type <- expression_type.",
    "For every field or local receiver.method(...), verify the method exists on the receiver's declared type.",
    "For every generic projection, preserve exact invariant type arguments from available_sibling_api.",
    "For every return statement, verify the expression type is assignable to the declared return type.",
    "For every constructor call, verify the canonical JDK/package owner and constructor arguments. If a JDK type is factory-owned, use its public static factory instead of inventing a constructor.",
    "The installed-JDK javap contract is authoritative; never assume that a public JDK class has a public constructor or method.",
    "Only after all checks pass, emit the final Java region with no reasoning prose.",
)

JAVA_FIELD_SHAPE_CONCERNS = JAVA_TYPE_OWNING_CONCERNS
JAVA_METHOD_SHAPE_CONCERNS = frozenset({
    "transitions",
    "invariants",
    "initialization",
    "updates",
    "cleanup",
    "concurrency",
    "preconditions",
    "success_postconditions",
    "rejection_postconditions",
    "security_checks",
    "synchronization",
    "bounds",
})

JAVA_MEMBERS_RESPONSE_CONTRACT = (
    "Return only compile-ready Java class-body source for this selected concern region. "
    "Do not return JSON, tool calls, Markdown, prose, package/import declarations, or the "
    "outer class wrapper. Emit complete semantic Java declarations: fields, methods, and "
    "only concern-owned private nested runtime types when genuinely required. "
    "Reuse available_sibling_api/dependency_api exactly; do not redeclare sibling state. "
    "If this concern needs state not present in available_sibling_api, declare the minimal "
    "private static concern-local backing field. Outer concern fields must not depend on "
    "constructor assignment: the host-owned outer constructor is private, so never emit a "
    "blank final outer field. Use an initialized constant or private static non-final backing "
    "state instead. Use fully-qualified JDK/external types when imports would otherwise be "
    "required. The existing outer class constructor and lifecycle are host-owned. Keep "
    "methods bounded and concern-local."
)
JAVA_DECLARATION_ONLY_RESPONSE_SUFFIX = (
    " This is a declaration-only data concern. Emit at least one concern-owned "
    "field and/or private nested data type. Do not emit methods, initialize(), "
    "onInitialize(), registration hooks, load/save lifecycle methods, or calls whose "
    "only purpose is to invoke another class lifecycle. Encode the authored runtime "
    "data requirements in task_authority as data declarations in this region."
)
JAVA_INTEGRATION_MEMBERS_RESPONSE_SUFFIX = (
    " Initialization statements are generated in a separate host-owned initialize "
    "region. Never emit an initialize() wrapper in members. If this concern needs no "
    "class-body declarations or helper methods, return exactly '// no members required'."
)
JAVA_INITIALIZE_RESPONSE_CONTRACT = (
    "Return only compile-ready Java statements or balanced control-flow blocks that belong "
    "inside the host-owned initialize() body. Do not return JSON, tool calls, Markdown, "
    "prose, package/import declarations, an initialize() wrapper, or the outer class. "
    "When no initialization is required, return exactly '// no initialization required'."
)



JAVA_ATOMIC_IDENTIFIER_PATTERN = r"^[A-Za-z_$][A-Za-z0-9_$]*$"
JAVA_ATOMIC_METHOD_NAME_PATTERN = r"^(?:<init>|[A-Za-z_$][A-Za-z0-9_$]*)$"
JAVA_ATOMIC_PARAMETER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java type. Use a primitive/java.lang type, a type declared in this same "
                "structured call, a supplied sibling/dependency type, or a fully-qualified "
                "external type. Common JDK collection names may be simple names because the "
                "host qualifies them."
            ),
        },
        "name": {
            "type": "string",
            "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN,
            "description": (
                "Semantic identifier. Java reserved words are accepted here because the host "
                "canonicalizes them consistently before rendering."
            ),
        },
    },
    "required": ["type", "name"],
    "additionalProperties": True,
}
JAVA_ATOMIC_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java field type only. Concern-owned domain types may be declared in the same "
                "records/enums/classes payload. Declaration modifiers are host-owned."
            ),
        },
        "name": {"type": "string", "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN},
        "initializer": {
            "type": "string",
            "description": (
                "Initializer expression only, without a trailing semicolon. "
                "Never instantiate an interface or abstract JDK collection directly; "
                "use a concrete implementation or a valid factory."
            ),
        },
    },
    "required": ["type", "name"],
    "additionalProperties": True,
}
JAVA_ATOMIC_METHOD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "return_type": {
            "type": "string",
            "minLength": 1,
            "description": "Java return type only. Declaration modifiers are host-owned.",
        },
        "name": {"type": "string", "pattern": JAVA_ATOMIC_METHOD_NAME_PATTERN},
        "parameters": {"type": "array", "items": JAVA_ATOMIC_PARAMETER_SCHEMA},
        "throws": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "body": {
            "type": "array",
            "items": {
                "type": "string",
                "description": (
                    "One Java statement or one complete control-flow block inside this method."
                ),
            },
        },
    },
    "required": ["return_type", "name"],
    "additionalProperties": True,
}
JAVA_ATOMIC_OUTER_METHOD_SCHEMA: dict[str, Any] = deepcopy(JAVA_ATOMIC_METHOD_SCHEMA)
JAVA_ATOMIC_OUTER_METHOD_SCHEMA["properties"]["name"] = {
    "type": "string",
    "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN,
    "description": (
        "Ordinary method name in the existing host-selected outer class. "
        "<init> is forbidden here because outer-class construction is host-owned."
    ),
}
JAVA_ATOMIC_CONSTRUCTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parameters": {"type": "array", "items": JAVA_ATOMIC_PARAMETER_SCHEMA},
        "throws": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "body": {"type": "array", "items": {"type": "string"}},
    },
    "required": [],
    "additionalProperties": True,
}
JAVA_ATOMIC_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN},
        "components": {"type": "array", "items": JAVA_ATOMIC_PARAMETER_SCHEMA},
        "constructors": {"type": "array", "items": JAVA_ATOMIC_CONSTRUCTOR_SCHEMA},
        "methods": {"type": "array", "items": JAVA_ATOMIC_METHOD_SCHEMA},
    },
    "required": ["name"],
    "additionalProperties": True,
}
JAVA_ATOMIC_ENUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN},
        "constants": {
            "type": "array",
            "items": {"type": "string", "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN},
            "uniqueItems": True,
        },
    },
    "required": ["name", "constants"],
    "additionalProperties": True,
}
JAVA_ATOMIC_CLASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": JAVA_ATOMIC_IDENTIFIER_PATTERN},
        "fields": {"type": "array", "items": JAVA_ATOMIC_FIELD_SCHEMA},
        "constructors": {"type": "array", "items": JAVA_ATOMIC_CONSTRUCTOR_SCHEMA},
        "methods": {"type": "array", "items": JAVA_ATOMIC_METHOD_SCHEMA},
    },
    "required": ["name"],
    "additionalProperties": True,
}
JAVA_ATOMIC_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": JAVA_ATOMIC_RECORD_SCHEMA},
        "enums": {"type": "array", "items": JAVA_ATOMIC_ENUM_SCHEMA},
        "classes": {"type": "array", "items": JAVA_ATOMIC_CLASS_SCHEMA},
        "fields": {"type": "array", "items": JAVA_ATOMIC_FIELD_SCHEMA},
        "methods": {"type": "array", "items": JAVA_ATOMIC_OUTER_METHOD_SCHEMA},
    },
    "required": [],
    "additionalProperties": True,
}
JAVA_ATOMIC_LOGIC_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {"type": "array", "items": JAVA_ATOMIC_FIELD_SCHEMA},
        "methods": {"type": "array", "items": JAVA_ATOMIC_OUTER_METHOD_SCHEMA},
    },
    "required": [],
    "additionalProperties": False,
}
JAVA_ATOMIC_DECLARATION_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": JAVA_ATOMIC_RECORD_SCHEMA},
        "enums": {"type": "array", "items": JAVA_ATOMIC_ENUM_SCHEMA},
        "classes": {"type": "array", "items": JAVA_ATOMIC_CLASS_SCHEMA},
        "fields": {"type": "array", "items": JAVA_ATOMIC_FIELD_SCHEMA},
    },
    "required": [],
    "additionalProperties": False,
}
JAVA_ATOMIC_INITIALIZE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
    "required": [],
    "additionalProperties": True,
}

JAVA_ATOMIC_ASSEMBLY_SYSTEM_PROMPT = (
    "Implement the host-selected concern through native emit_java_part calls. "
    "Fill only the current assembly.path using the supplied scalar schema. "
    "The host constructs objects/arrays; never serialize them into strings. "
    "When the current path is a sibling batch, emit one function call per sibling item. "
    "A batch part is single-use: when a part is selected, emit every needed sibling item "
    "for that part in the same native turn because the host closes that part immediately. "
    "Keep the authored requirements, dependency_api, available_sibling_api, and "
    "compiler_contract authoritative. The target is first-pass compilable Java, not code that "
    "expects a compiler-repair round. Reuse exact sibling declarations; do not redeclare them "
    "or change their types/defaults. Never reassign a final sibling or concern-local final field. "
    "If a field is final, initialize it at declaration time. If an authoritative API returns Object "
    "but this method needs a narrower generic/container type, narrow with an explicit runtime type "
    "check and a type-compatible fallback; never use a raw/unchecked cast as a shortcut. "
    "Use canonical JDK packages; Lock/ReentrantLock live in java.util.concurrent.locks. "
    "Accepted structure and enclosing declarations remain fixed. "
    "For part selection choose a needed part or done when this enclosing object is complete. "
    "A body value is one complete Java statement or balanced control-flow block, "
    "not a fragment of JSON or a partial brace. Split long logic into named helper methods. "
    "For a declaration, type/return_type contains only a Java type. "
    "All Java declaration modifiers and visibility are host-owned; the model never emits them. "
    "Static initializer blocks, package/import directives, outer type declarations, and lifecycle "
    "wrappers are also host-owned and must never be emitted inside executable body values. "
    "The host adds static to outer fields/methods and owns nested-type visibility. "
    "Omit unnecessary optional scalar values. "
    "A field marked final must have a declaration initializer and generated executable code must never "
    "rebind a final field. Preserve generic types exactly. If an authoritative API returns Object, "
    "do not directly return it from a narrower typed method; inspect/narrow the runtime value first. "
    "For JDK locks use java.util.concurrent.locks.Lock/ReentrantLock (or simple Lock/ReentrantLock, "
    "which the host canonicalizes), never java.util.concurrent.Lock/ReentrantLock. "
    "Do not invent Minecraft/Fabric APIs or metadata-only gameplay implementations."
)


def atomic_schema_limits(profile: str) -> AtomicSchemaLimits:
    try:
        return ATOMIC_SCHEMA_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(f"unknown atomic schema profile: {profile!r}") from exc


def string_limit_for_schema_class(profile: str, string_class: str) -> int:
    profile_name = str(profile)
    class_name = str(string_class)
    allowed = PROFILE_STRING_CLASSES.get(profile_name)
    if allowed is None:
        raise ValueError(f"unknown atomic schema profile: {profile_name!r}")
    if class_name not in allowed:
        raise ValueError(
            f"string class {class_name!r} is not allowed for atomic schema profile "
            f"{profile_name!r}"
        )
    try:
        return int(STRING_CLASS_LIMITS[class_name])
    except KeyError as exc:
        raise ValueError(f"unknown schema string class: {class_name!r}") from exc


def atomic_error_recoverable(reason: str) -> bool:
    return str(reason or "").startswith(RECOVERABLE_ATOMIC_ERROR_PREFIXES)


def atomic_error_terminal_after_normalization(reason: str) -> bool:
    return str(reason or "").startswith(TERMINAL_AFTER_NORMALIZATION_PREFIXES)



def java_atomic_parameters_for_request(
    payload: Mapping[str, Any],
    *,
    response_region: str,
) -> tuple[dict[str, Any], str]:
    if response_region == "initialize":
        return JAVA_ATOMIC_INITIALIZE_PARAMETERS, "initialize_statements"

    recipe = payload.get("generation_recipe")
    preferred = (
        str(recipe.get("preferred_shape") or "").strip()
        if isinstance(recipe, Mapping)
        else ""
    )
    concern = payload.get("concern")
    concern_name = (
        str(concern.get("name") or "").strip()
        if isinstance(concern, Mapping)
        else ""
    )
    authorized_nested = tuple(
        str(item).strip()
        for item in payload.get("authorized_nested_runtime_types") or ()
        if str(item).strip()
    )

    if concern_name in JAVA_DECLARATION_ONLY_CONCERNS:
        parameters = JAVA_ATOMIC_DECLARATION_MEMBERS_PARAMETERS
        shape = preferred or "declarations_only_fields_or_private_nested_types"
    elif (
        concern_name
        and concern_name not in JAVA_TYPE_OWNING_CONCERNS
        and not authorized_nested
    ):
        return (
            JAVA_ATOMIC_LOGIC_MEMBERS_PARAMETERS,
            preferred or "logic_fields_methods_only",
        )
    else:
        parameters = JAVA_ATOMIC_MEMBERS_PARAMETERS
        shape = preferred or "smallest_components"

    host_symbol = str(payload.get("host_selected_class") or "").strip()
    if host_symbol or authorized_nested:
        parameters = deepcopy(parameters)
        for category in ("records", "enums", "classes"):
            category_schema = parameters["properties"].get(category)
            if not isinstance(category_schema, Mapping):
                continue
            name_schema = category_schema["items"]["properties"]["name"]
            if authorized_nested:
                name_schema["enum"] = list(authorized_nested)
            if host_symbol:
                name_schema["not"] = {"enum": [host_symbol]}
            name_schema["description"] = (
                "Name of a requirement-owned nested runtime helper. "
                + (
                    f"{host_symbol} is the existing outer class and must not be declared again. "
                    if host_symbol
                    else ""
                )
                + "Place outer fields in fields, not in a class wrapper."
            )
    return parameters, shape


def java_atomic_assembly_system_prompt() -> str:
    return JAVA_ATOMIC_ASSEMBLY_SYSTEM_PROMPT


def java_region_recovery_shapes(region: str) -> tuple[str, ...]:
    return tuple(JAVA_REGION_RECOVERY_SHAPES.get(str(region or "").strip(), ()))


def java_initialize_wrapper_allowed(
    name: str,
    return_type: str,
    parameters: str,
) -> bool:
    return (
        str(name or "").strip() in JAVA_INITIALIZE_WRAPPER_NAMES
        and str(return_type or "").strip() == JAVA_HOST_INITIALIZE_RETURN_TYPE
        and str(parameters or "").strip() == JAVA_HOST_INITIALIZE_PARAMETERS
    )


def java_localize_initialize_field_modifiers(
    modifiers: str,
) -> tuple[str, ...] | None:
    text = str(modifiers or "").strip()
    if not text:
        return ()
    if "@" in text:
        return None
    tokens = tuple(item for item in text.split() if item)
    if any(item not in JAVA_INITIALIZE_LOCALIZABLE_FIELD_MODIFIERS for item in tokens):
        return None
    return tuple(
        item for item in tokens if item in JAVA_INITIALIZE_PRESERVED_LOCAL_MODIFIERS
    )


def java_member_jdk_import_allowed(fqcn: str) -> bool:
    value = str(fqcn or "").strip()
    return bool(value.startswith("java.") and "*" not in value)


def java_imported_jdk_use_can_be_qualified(role: str) -> bool:
    return str(role or "").strip() in JAVA_QUALIFIABLE_JDK_IMPORT_USE_ROLES


def java_nested_type_scope_error(name: str) -> str:
    return (
        f"nested type {str(name or '')!r} must be "
        f"{JAVA_NESTED_TYPE_REQUIRED_VISIBILITY} because outer type ownership is host-owned"
    )


def java_host_initialize_signature(
    name: str,
    return_type: str,
    parameters: str,
    modifiers: Any,
) -> bool:
    modifier_set = frozenset(str(item) for item in (modifiers or ()))
    return (
        str(name or "").strip() == JAVA_HOST_INITIALIZE_NAME
        and str(return_type or "").strip() == JAVA_HOST_INITIALIZE_RETURN_TYPE
        and str(parameters or "").strip() == JAVA_HOST_INITIALIZE_PARAMETERS
        and JAVA_HOST_INITIALIZE_REQUIRED_MODIFIER in modifier_set
    )


def authorized_concern_nested_type_symbols(authority: Any) -> tuple[str, ...]:
    """Extract explicit requirement-owned Java type names for visibility lowering."""

    if not isinstance(authority, dict):
        return ()

    names: set[str] = set()
    sources = authority.get("source_requirements")
    if isinstance(sources, dict):
        for raw in sources.values():
            text = str(raw or "")
            for match in re.finditer(
                r"`([A-Za-z_$][A-Za-z0-9_$]*)`",
                text,
            ):
                name = match.group(1)
                if name[:1].isupper():
                    names.add(name)
            for match in re.finditer(
                r"\b(?:ERROR|EXCEPTION|TYPE|CLASS|RECORD|ENUM|INTERFACE)\s*[:=]\s*"
                r"([A-Z][A-Za-z0-9_$]*)\b",
                text,
                flags=re.IGNORECASE,
            ):
                name = match.group(1)
                if name[:1].isupper():
                    names.add(name)

    structured = authority.get("structured_records")
    if isinstance(structured, (list, tuple)):
        for row in structured:
            if not isinstance(row, dict):
                continue
            for key in ("name", "symbol", "identifier", "purpose", "type"):
                raw = str(row.get(key) or "").strip()
                if (
                    raw[:1].isupper()
                    and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", raw)
                ):
                    names.add(raw)

    return tuple(sorted(names))


def java_generation_shared_recipe_policy() -> dict[str, Any]:
    return {
        "compiler_first_rules": list(JAVA_COMPILER_FIRST_RULES),
        "jdk_package_anchors": dict(JAVA_JDK_PACKAGE_ANCHORS),
        "jdk_runtime_patterns": {
            "exclusive_lock": {
                "declared_type": "java.util.concurrent.locks.Lock",
                "initializer": "new java.util.concurrent.locks.ReentrantLock()",
                "receiver_methods": ["lock", "unlock", "tryLock"],
            },
            "map": {
                "declared_type": "java.util.Map<K,V>",
                "initializer_example": "new java.util.HashMap<>()",
                "receiver_methods": ["get", "put", "remove", "containsKey"],
            },
            "atomic_counter": {
                "declared_type": "java.util.concurrent.atomic.AtomicLong",
                "initializer_example": "new java.util.concurrent.atomic.AtomicLong(0L)",
                "receiver_methods": ["get", "set", "incrementAndGet", "compareAndSet"],
            },
        },
        "pre_emit_compile_checklist": list(JAVA_PRE_EMIT_COMPILE_CHECKLIST),
    }


def java_generation_recipe_contract(concern_name: str) -> dict[str, Any]:
    name = str(concern_name or "").strip()
    recipe = {
        "first_pass_goal": (
            "produce the smallest compile-ready semantic Java source region in one response "
            "and finish well inside the finite output page"
        ),
        "declare_local_domain_types_first": True,
        "require_fully_qualified_external_types": True,
        "output_language": "java_source_region",
        "no_json_ast_protocol": True,
        "sibling_api_is_authoritative": True,
        "dependency_call_contract_is_exhaustive": True,
        "dependency_call_rule": (
            "Before emitting Owner.method(...), match owner, method, arity, static=true, and "
            "parameter_names against dependency_call_contract. Treat parameter_names as semantic "
            "roles. trigger_dispatch/event_dispatch methods are not key/value setters; direct "
            "state assignment requires a direct_state_write signature. If no exact row exists, "
            "do not emit that call. Dependency initialize/onInitialize hooks are host-orchestrated; "
            "concern regions must not call them."
        ),
        "never_mutate_final_sibling_fields": True,
        "declare_missing_concern_local_state": (
            "When this concern reads or writes state absent from available_sibling_api, "
            "declare a private static non-final backing field with a compatible runtime "
            "value type instead of referencing an undeclared symbol or inventing a metadata DTO."
        ),
        **java_generation_shared_recipe_policy(),
        "compile_ready_examples": [
            {
                "bad": "java.util.Map<String,Object> lock = new java.util.ReentrantLock();",
                "good": "java.util.concurrent.locks.Lock lock = new java.util.concurrent.locks.ReentrantLock();",
            },
            {
                "bad": "java.util.Map<String,String> value = entry.getValue(); // entry is Map.Entry<String,Map<String,Object>>",
                "good": "java.util.Map<String,Object> value = entry.getValue();",
            },
            {
                "bad": "String value = objectMap.get(key);",
                "good": "Object raw = objectMap.get(key); String value = raw instanceof String s ? s : \"\";",
            },
        ],
        "preferred_shape": (
            "fields_and_local_types"
            if name in JAVA_FIELD_SHAPE_CONCERNS
            else "methods_and_constants"
            if name in JAVA_METHOD_SHAPE_CONCERNS
            else "smallest_components_that_satisfy_this_concern"
        ),
    }
    return recipe


def java_region_response_contract(
    *,
    section: str,
    concern_name: str,
    response_region: str,
) -> str:
    region = str(response_region or "").strip()
    name = str(concern_name or "").strip()
    if region == "initialize":
        return JAVA_INITIALIZE_RESPONSE_CONTRACT
    if region != "members":
        raise ValueError(f"unsupported Java response region: {response_region!r}")

    contract = JAVA_MEMBERS_RESPONSE_CONTRACT
    if name in JAVA_DECLARATION_ONLY_CONCERNS:
        contract += JAVA_DECLARATION_ONLY_RESPONSE_SUFFIX
    if str(section or "").strip() == "integration":
        contract += JAVA_INTEGRATION_MEMBERS_RESPONSE_SUFFIX
    return contract


def java_region_system_prompt_contract(
    *,
    section: str,
    concern_name: str,
    response_region: str,
    platform_api_policy: str,
) -> str:
    response_contract = java_region_response_contract(
        section=section,
        concern_name=concern_name,
        response_region=response_region,
    )
    platform_rule = (
        "This section is pure Java domain logic. Do not reference net.minecraft.*, "
        "net.fabricmc.*, registries, resource identifiers, packets, lifecycle hooks, or "
        "game registration APIs. "
        if str(platform_api_policy or "").strip() == "forbidden"
        else (
            "This is a platform-bound section. You may reference net.minecraft.* or "
            "net.fabricmc.* only when the exact owner is present in implementation_authority "
            "or host_grounding. If no such owner is supplied, keep this concern platform-neutral "
            "and use only JDK/dependency APIs. "
        )
    )
    return (
        "Emit one final Java region only. Do not think aloud, explain, draft, reconsider, "
        "or emit multiple candidate implementations. "
        "Implement exactly one host-selected concern inside one already-selected Java class. "
        "Planning record schemas, planning task labels, and concern cardinality are host-owned "
        "metadata and are deliberately not exposed as Java source shapes. Source requirement labels "
        "such as owner/type/unit/default/domain/from_state/trigger/guard describe semantics; they "
        "are not a request to create a Java metadata record with those labels as components. "
        "Create a record/class only when the runtime gameplay implementation itself needs that data object. "
        "You do not choose files, classes, dependencies, architecture, tools, search routes, APIs, or sibling work. "
        + response_contract + " "
        "The task_authority source requirements are already host-sliced to this concern; "
        "current_selected_region_source is the only region you may replace. "
        "available_sibling_api contains authoritative compiled Java declarations from "
        "earlier concerns: use their exact symbol spelling, declared type, signature, "
        "generic arguments, and mutability. Generic arguments are invariant authority: "
        "never narrow Map<K,Object> to Map<K,String> or otherwise substitute a different "
        "generic argument. For Map<K,V>.entrySet(), Map.Entry is exactly Map.Entry<K,V>, "
        "getKey() is exactly K, and getValue() is exactly V. Never treat an object/record "
        "field as a primitive, never assign to a field declared final, and never invent a "
        "sibling symbol that is not listed. This candidate must pass host semantic validation "
        "and compilation. If repair_failure or region_correction is supplied, correct the "
        "actual rejected candidate using those diagnostics and preserve unrelated declarations. "
        "Resolve every supplied semantic/API fact before emitting source, and do not implement "
        "sibling concerns. "
        + platform_rule
        + "Use only supplied host grounding and dependency APIs; never invent a Minecraft/Fabric API. "
        "dependency_call_contract is exhaustive for dependency method calls: match owner, method, "
        "static=true, arity, parameter types, and parameter_names exactly. parameter_names are semantic "
        "roles: never use a trigger_dispatch/event_dispatch method as a key/value setter; use an exact "
        "direct_state_write signature for direct named-state assignment. If no exact row exists, do not "
        "emit the call. Never add arguments to a zero-arity method. Never call a dependency initialize/"
        "onInitialize lifecycle hook from a concern region."
    )


def java_region_scope_policy(
    *,
    failure: bool,
    sibling_concerns: Any,
) -> dict[str, Any]:
    return {
        "sibling_regions_immutable": True,
        "required_output_format": "plain_java_source",
        "model_tools_enabled": False,
        "sibling_concerns_out_of_scope": [
            str(item) for item in (sibling_concerns or ())
        ],
        "scope_rule": (
            "Implement only the selected concern and only the lines in "
            "task_authority.source_requirements. Do not pre-implement sibling concerns. "
            "The host owns declaration ownership and sibling bookkeeping; earlier declarations are immutable."
        ),
        "repair_structure_rule": (
            "Compiler repair may remove or edit existing nested types but must not add, "
            "rename, or change the kind of nested types."
            if failure
            else None
        ),
    }


def assert_execution_contract_consistent() -> None:
    failures: list[str] = []
    generic = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars
    if not (SOURCE_REPAIR_MAX_SOURCE_CHARS >= SOURCE_REPAIR_MAX_SPAN_CHARS >= generic):
        failures.append("source/repair/generic string limits are not monotonic")
    if SOURCE_REPAIR_HARD_ATTEMPTS < 1:
        failures.append("source repair must have at least one bounded attempt")
    if DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES < 1:
        failures.append("diagnostic repair inline source byte bound must be positive")
    if JAVA_ATOMIC_ASSEMBLY_MAX_CALLS < 1 or JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS < 1:
        failures.append("atomic Java assembly bounds must be positive")
    if JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES < 0:
        failures.append("atomic Java assembly context margin cannot be negative")
    if not JAVA_DECLARATION_ONLY_CONCERNS <= JAVA_TYPE_OWNING_CONCERNS:
        failures.append("declaration-only concerns must be type-owning concerns")
    if not (1 <= PRODUCTION_REGION_ATTEMPT_LIMIT <= MAX_REGION_ATTEMPT_LIMIT):
        failures.append("production region attempt limit is outside its hard bound")
    if not (1 <= PRODUCTION_COMPILE_REPAIR_LIMIT <= MAX_COMPILE_REPAIR_LIMIT):
        failures.append("production compile repair must be enabled and bounded")
    if not PRODUCTION_RETRY_STRUCTURAL_REJECTIONS:
        failures.append("production structural correction is disabled")
    if JAVA_NESTED_TYPE_REQUIRED_VISIBILITY in JAVA_NESTED_TYPE_CANONICALIZABLE_VISIBILITIES:
        failures.append("required nested visibility cannot also be a lowering source")
    recoverable = set(RECOVERABLE_ATOMIC_ERROR_PREFIXES)
    terminal = set(TERMINAL_AFTER_NORMALIZATION_PREFIXES)
    if not terminal <= recoverable:
        failures.append("terminal-after-normalization errors must also be recoverable")
    if JAVA_HOST_INITIALIZE_NAME not in JAVA_INITIALIZE_WRAPPER_NAMES:
        failures.append("host initialize ownership and wrapper recovery disagree")
    if not java_initialize_wrapper_allowed(
        JAVA_HOST_INITIALIZE_NAME,
        JAVA_HOST_INITIALIZE_RETURN_TYPE,
        JAVA_HOST_INITIALIZE_PARAMETERS,
    ):
        failures.append("initialize wrapper recovery disagrees with host lifecycle signature")
    if failures:
        raise RuntimeError(
            "EXECUTION_CONTRACT_INVALID: " + "; ".join(failures)
        )


assert_execution_contract_consistent()


__all__ = [
    "ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING",
    "ATOMIC_ERROR_RECOVERY_RULES",
    "ATOMIC_SCHEMA_PROFILES",
    "AtomicErrorRecoveryRule",
    "AtomicSchemaLimits",
    "DEFAULT_ATOMIC_SCHEMA_LIMITS",
    "DEFAULT_COMPILE_REPAIR_LIMIT",
    "DIAGNOSTIC_REPAIR_INLINE_SOURCE_MAX_BYTES",
    "DEFAULT_REGION_ATTEMPT_LIMIT",
    "DEFAULT_SCHEMA_PROFILE",
    "JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES",
    "JAVA_ATOMIC_CLASS_SCHEMA",
    "JAVA_ATOMIC_CONSTRUCTOR_SCHEMA",
    "JAVA_ATOMIC_DECLARATION_MEMBERS_PARAMETERS",
    "JAVA_ATOMIC_ENUM_SCHEMA",
    "JAVA_ATOMIC_FIELD_SCHEMA",
    "JAVA_ATOMIC_IDENTIFIER_PATTERN",
    "JAVA_ATOMIC_INITIALIZE_PARAMETERS",
    "JAVA_ATOMIC_LOGIC_MEMBERS_PARAMETERS",
    "JAVA_ATOMIC_MEMBERS_PARAMETERS",
    "JAVA_ATOMIC_METHOD_NAME_PATTERN",
    "JAVA_ATOMIC_METHOD_SCHEMA",
    "JAVA_ATOMIC_OUTER_METHOD_SCHEMA",
    "JAVA_ATOMIC_PARAMETER_SCHEMA",
    "JAVA_ATOMIC_RECORD_SCHEMA",
    "JAVA_ATOMIC_ASSEMBLY_MAX_CALLS",
    "JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS",
    "JAVA_ATOMIC_ASSEMBLY_SYSTEM_PROMPT",
    "JAVA_COMPILER_FIRST_RULES",
    "JAVA_CONCERN_MEMBER_NODE_TYPES",
    "JAVA_DECLARATION_ONLY_CONCERNS",
    "JAVA_DECLARATION_ONLY_MEMBER_KINDS",
    "JAVA_TYPE_OWNING_CONCERNS",
    "JAVA_EXPLICIT_JDK_IMPORT_PATTERN",
    "JAVA_FENCE_LANGUAGES",
    "JAVA_HOST_INITIALIZE_NAME",
    "JAVA_HOST_INITIALIZE_PARAMETERS",
    "JAVA_HOST_INITIALIZE_REQUIRED_MODIFIER",
    "JAVA_HOST_INITIALIZE_RETURN_TYPE",
    "JAVA_HOST_OWNED_MEMBER_NODE_TYPES",
    "JAVA_INITIALIZE_LOCALIZABLE_FIELD_MODIFIERS",
    "JAVA_INITIALIZE_PRESERVED_LOCAL_MODIFIERS",
    "JAVA_INITIALIZE_WRAPPER_NAMES",
    "JAVA_JDK_PACKAGE_ANCHORS",
    "JAVA_NESTED_TYPE_CANONICALIZABLE_VISIBILITIES",
    "JAVA_NESTED_TYPE_NODE_TYPES",
    "JAVA_NESTED_TYPE_REQUIRED_VISIBILITY",
    "JAVA_PRE_EMIT_COMPILE_CHECKLIST",
    "JAVA_QUALIFIABLE_JDK_IMPORT_USE_ROLES",
    "JAVA_REGION_RECOVERY_SHAPES",
    "MAX_COMPILE_REPAIR_LIMIT",
    "MAX_REGION_ATTEMPT_LIMIT",
    "MODEL_MAX_COMPLETION_TOKENS",
    "PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS",
    "PRODUCTION_COMPILE_REPAIR_LIMIT",
    "PRODUCTION_REGION_ATTEMPT_LIMIT",
    "PRODUCTION_RETRY_STRUCTURAL_REJECTIONS",
    "PROFILE_STRING_CLASSES",
    "RECOVERABLE_ATOMIC_ERROR_PREFIXES",
    "SCHEMA_CONTRACT_PROFILE_KEY",
    "SCHEMA_STRING_CLASS_KEY",
    "SOURCE_REPAIR_ATOMIC_SCHEMA_LIMITS",
    "SOURCE_REPAIR_HARD_ATTEMPTS",
    "SOURCE_REPAIR_MAX_SOURCE_CHARS",
    "SOURCE_REPAIR_MAX_SPAN_CHARS",
    "SOURCE_REPAIR_SCHEMA_PROFILE",
    "STRING_CLASS_GENERIC",
    "STRING_CLASS_LIMITS",
    "STRING_CLASS_REPAIR_SPAN",
    "STRING_CLASS_SOURCE",
    "TERMINAL_AFTER_NORMALIZATION_PREFIXES",
    "assert_execution_contract_consistent",
    "atomic_error_recoverable",
    "atomic_error_terminal_after_normalization",
    "atomic_schema_limits",
    "authorized_concern_nested_type_symbols",
    "java_atomic_assembly_system_prompt",
    "java_atomic_parameters_for_request",
    "java_generation_recipe_contract",
    "java_generation_shared_recipe_policy",
    "java_imported_jdk_use_can_be_qualified",
    "java_initialize_wrapper_allowed",
    "java_localize_initialize_field_modifiers",
    "java_member_jdk_import_allowed",
    "java_host_initialize_signature",
    "java_nested_type_scope_error",
    "java_region_recovery_shapes",
    "java_region_response_contract",
    "java_region_scope_policy",
    "java_region_system_prompt_contract",
    "string_limit_for_schema_class",
]
