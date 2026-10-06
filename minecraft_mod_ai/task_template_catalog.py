"""Load host-owned task contracts from the single runtime template authority."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml
from jsonschema import Draft202012Validator

from .authored_section_ids import AUTHORING_SECTION_ORDER
from .model_output_atomicity_contract import assert_atomic_model_schema

RUNTIME_TEMPLATE_ROOT = Path(__file__).with_name("templates").resolve()
ROOT = RUNTIME_TEMPLATE_ROOT

CRITERION_SECTIONS = AUTHORING_SECTION_ORDER
_CRITERION_ALIASES = {f"feature/{section}": f"criterion/{section}" for section in CRITERION_SECTIONS}
_PROMPT_POLICY = "prompt/policy"

# Evidence authority is host-owned. Record-set cardinality/iteration is not a
# separate model or host protocol at all: models author semantic records only.
_EVIDENCE_BOUND_RECORD_TASKS = frozenset({
    "feature/integration/target_bindings",
})
_POSITIVE_HOST_CONTROL_PREFIXES = (
    "return done",
    "return not_applicable",
    "return blocked",
    "return only the next record",
)
_HOST_CONTROL_RULE = (
    "Return semantic record content only. Do not emit count, done, continuation, "
    "retry, ordinal, blocked, applicability, or other loop-control protocol."
)



def _canonical_identifier(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise ValueError("TEMPLATE_PATH: identifier must be a string")
    if not identifier or identifier != identifier.strip():
        raise ValueError("TEMPLATE_PATH: identifier must be non-empty and trimmed")
    if "\\" in identifier or identifier.endswith(".yaml"):
        raise ValueError("TEMPLATE_PATH: use a canonical slash-separated semantic id without .yaml")
    parsed = PurePosixPath(identifier)
    parts = parsed.parts
    if parsed.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts) or "//" in identifier:
        raise ValueError(f"TEMPLATE_PATH: non-canonical identifier {identifier!r}")
    canonical = parsed.as_posix()
    if canonical != identifier:
        raise ValueError(f"TEMPLATE_PATH: non-canonical identifier {identifier!r}")
    return canonical


def _template_path(identifier: str) -> Path:
    canonical = _canonical_identifier(identifier)
    path = (RUNTIME_TEMPLATE_ROOT / f"{canonical}.yaml").resolve()
    if not path.is_relative_to(RUNTIME_TEMPLATE_ROOT):
        raise ValueError("TEMPLATE_PATH: identifier escapes runtime catalog")
    if not path.is_file():
        raise ValueError(f"TEMPLATE_MISSING: no runtime template for {canonical}")
    return path


@lru_cache(maxsize=None)
def _load(identifier: str):
    identifier = _canonical_identifier(identifier)
    path = _template_path(identifier)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != identifier:
        raise ValueError(f"TEMPLATE_ID: invalid template {identifier}")
    from .template_contract_validation import validate_template_contract
    validate_template_contract(value)
    for key in ("record_schema", "input_schema", "output_schema"):
        if key in value:
            Draft202012Validator.check_schema(value[key])
    return value


def _apply_shared_policy(identifier: str, value: dict):
    if identifier.startswith("prompt/") and identifier not in {_PROMPT_POLICY, "prompt/workflow"}:
        policy = _load(_PROMPT_POLICY)
        merged = []
        for rule in tuple(policy.get("rules", ())) + tuple(value.get("rules", ())):
            if rule not in merged:
                merged.append(rule)
        value["rules"] = merged
    return value


def _runtime_concern_section(identifier: str):
    parts = identifier.split("/")
    if len(parts) == 3 and parts[0] == "feature" and parts[1] in CRITERION_SECTIONS:
        return parts[1]
    return None


def _record_requires_external_evidence(identifier: str) -> bool:
    section = _runtime_concern_section(identifier)
    if section is None:
        return False
    if section == "reuse_assessment":
        return True
    if section == "integration":
        return identifier in _EVIDENCE_BOUND_RECORD_TASKS
    return False


def _apply_record_host_policy(identifier: str, value: dict):
    return _architecture_impl__apply_record_host_policy((identifier, value))


def _materialize_atomic_record_schema(schema: dict) -> dict:
    """Return the logical schema without inventing global size constraints.

    Explicit domain bounds in the template are preserved. Generic string length,
    array cardinality and object width are not correctness contracts and are not
    synthesized by the host.
    """
    return deepcopy(schema)


def _canonical_state_record_schema(
    identifier: str,
    declared_schema: dict,
) -> dict:
    """Project a state template's declared fields onto the canonical state schema."""

    parts = identifier.split("/")
    if len(parts) != 3 or parts[:2] != ["feature", "state_model"]:
        return deepcopy(declared_schema)

    from .structured_state_runtime import state_concern_schema

    concern = parts[2]
    canonical = state_concern_schema(concern)
    canonical_properties = canonical.get("properties")
    declared_required = declared_schema.get("required")
    if (
        not isinstance(canonical_properties, dict)
        or not isinstance(declared_required, list)
    ):
        raise ValueError(
            f"TEMPLATE_RECORD_SCHEMA: invalid canonical state schema for {identifier}"
        )
    missing = [
        field
        for field in declared_required
        if field not in canonical_properties
    ]
    if missing:
        raise ValueError(
            f"TEMPLATE_RECORD_SCHEMA: {identifier} declares non-canonical state fields "
            f"{missing}"
        )
    return {
        "type": "object",
        "properties": {
            field: deepcopy(canonical_properties[field])
            for field in declared_required
        },
        "required": list(declared_required),
        "additionalProperties": False,
    }


def _compile_record_schema(identifier: str, schema: dict) -> dict:
    """Validate the canonical logical record without misclassifying it as one model call."""
    compiled = _materialize_atomic_record_schema(schema)
    Draft202012Validator.check_schema(compiled)
    assert_atomic_model_schema(
        compiled,
        surface=f"runtime record template {identifier!r}",
    )
    return compiled


def load_template(identifier: str):
    requested = _canonical_identifier(identifier)
    canonical = _CRITERION_ALIASES.get(requested, requested)
    return _apply_shared_policy(canonical, deepcopy(_load(canonical)))


def load_record_template(identifier: str):
    requested = _canonical_identifier(identifier)
    concrete = (RUNTIME_TEMPLATE_ROOT / f"{requested}.yaml").resolve()
    if concrete.is_relative_to(RUNTIME_TEMPLATE_ROOT) and concrete.is_file():
        value = _apply_shared_policy(requested, deepcopy(_load(requested)))
    else:
        value = load_template(requested)
    if "record_schema" not in value:
        raise ValueError(f"TEMPLATE_RECORD_SCHEMA: missing record schema for {requested}")
    value = _apply_record_host_policy(requested, value)
    value["record_schema"] = _canonical_state_record_schema(
        requested,
        value["record_schema"],
    )
    value["record_schema"] = _compile_record_schema(requested, value["record_schema"])
    return value


def _required_leaf_fields(schema: dict, *, identifier: str) -> tuple[str, ...]:
    """Flatten required nested record groups into the model-facing leaf field order."""

    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError(
            f"TEMPLATE_RECORD_SCHEMA: {identifier} must declare object properties and required fields"
        )

    leaves: list[str] = []
    seen: set[str] = set()
    for field in required:
        child = properties.get(field)
        if not isinstance(field, str) or not isinstance(child, dict):
            raise ValueError(
                f"TEMPLATE_RECORD_SCHEMA: {identifier} required field {field!r} is undeclared"
            )
        if child.get("type") == "object":
            nested = _required_leaf_fields(child, identifier=f"{identifier}.{field}")
            for leaf in nested:
                if leaf in seen:
                    raise ValueError(
                        f"TEMPLATE_RECORD_SCHEMA: {identifier} has duplicate leaf field {leaf!r}"
                    )
                seen.add(leaf)
                leaves.append(leaf)
            continue
        if field in seen:
            raise ValueError(
                f"TEMPLATE_RECORD_SCHEMA: {identifier} has duplicate leaf field {field!r}"
            )
        seen.add(field)
        leaves.append(field)
    return tuple(leaves)


def detail_records():
    records = {}
    for section in CRITERION_SECTIONS:
        manifest = load_template(f"criterion/{section}")
        if manifest.get("execution") != "sequence":
            raise ValueError(f"TEMPLATE_CRITERION: {section} must be a sequence")
        records[section] = {}
        for identifier in manifest["steps"]:
            schema = load_record_template(identifier)["record_schema"]
            fields = _required_leaf_fields(schema, identifier=identifier)
            records[section][identifier.rsplit("/", 1)[1]] = " ".join(fields)
    return records

def _architecture_impl__apply_record_host_policy(_ctx):
    (identifier, value) = _ctx
    if _runtime_concern_section(identifier) is None:
        return value

    # Legacy catalog metadata is deliberately inert and removed before the model sees
    # the template. Cardinality is represented by the records themselves, not a policy bit.
    value.pop("cardinality_blocking", None)

    rules = []
    for rule in value.get("rules", ()):
        text = str(rule)
        lowered = text.strip().lower()
        if lowered.startswith(_POSITIVE_HOST_CONTROL_PREFIXES):
            continue
        if (
            "host owns cardinality" in lowered
            or "host-requested ordinal" in lowered
            or "determines cardinality only" in lowered
        ):
            continue
        rules.append(text)
    if not any("loop-control protocol" in rule.lower() for rule in rules):
        rules.append(_HOST_CONTROL_RULE)
    if not _record_requires_external_evidence(identifier):
        rules.append(
            "You are the designer of this gameplay record. Choose unspecified mechanics, "
            "actors, values and interactions coherently with the requirement. Authored "
            "design choices need no external evidence or approval. Keep externally "
            "verifiable API and repository facts separate from those choices."
        )
    value["rules"] = rules
    return value

