from __future__ import annotations

"""Single authority for hard execution/generation/repair contract limits.

Runtime modules may expose compatibility aliases, but hard limits and production
recovery toggles must be defined here.  Model-schema admission, repair-schema
construction, Java generation, verifier repair windows, and production repair all
consume this module so their contracts cannot silently drift apart.
"""

from dataclasses import dataclass
from types import MappingProxyType


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
    # repair operation objects carry operation/path/hash/payload together
    max_fields=4,
    max_string_chars=256,
    max_array_items=4,
    # edit -> replacements[] -> replacement object adds one structural level
    max_schema_depth=4,
)

ATOMIC_SCHEMA_PROFILES = MappingProxyType({
    DEFAULT_SCHEMA_PROFILE: DEFAULT_ATOMIC_SCHEMA_LIMITS,
    SOURCE_REPAIR_SCHEMA_PROFILE: SOURCE_REPAIR_ATOMIC_SCHEMA_LIMITS,
})

SOURCE_REPAIR_MAX_SOURCE_CHARS = 16_384
SOURCE_REPAIR_MAX_SPAN_CHARS = 4_096
SOURCE_REPAIR_HARD_ATTEMPTS = 2

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


__all__ = [
    "ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING",
    "ATOMIC_SCHEMA_PROFILES",
    "AtomicSchemaLimits",
    "DEFAULT_ATOMIC_SCHEMA_LIMITS",
    "DEFAULT_COMPILE_REPAIR_LIMIT",
    "DEFAULT_REGION_ATTEMPT_LIMIT",
    "DEFAULT_SCHEMA_PROFILE",
    "MAX_COMPILE_REPAIR_LIMIT",
    "MAX_REGION_ATTEMPT_LIMIT",
    "MODEL_MAX_COMPLETION_TOKENS",
    "PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS",
    "PRODUCTION_COMPILE_REPAIR_LIMIT",
    "PRODUCTION_REGION_ATTEMPT_LIMIT",
    "PRODUCTION_RETRY_STRUCTURAL_REJECTIONS",
    "PROFILE_STRING_CLASSES",
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
    "atomic_schema_limits",
    "string_limit_for_schema_class",
]
