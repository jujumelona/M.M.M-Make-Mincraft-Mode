"""Load host-owned task contracts from the single runtime template authority."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml
from jsonschema import Draft202012Validator

RUNTIME_TEMPLATE_ROOT = Path(__file__).with_name("templates").resolve()
ROOT = RUNTIME_TEMPLATE_ROOT

CRITERION_SECTIONS = (
    "behavior_contract", "state_model", "algorithm", "integration",
    "authority_and_network", "persistence", "resources_and_ui",
    "failure_and_limits", "reuse_assessment", "verification",
)
_CRITERION_ALIASES = {f"feature/{section}": f"criterion/{section}" for section in CRITERION_SECTIONS}
_PROMPT_POLICY = "prompt/policy"


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
    return value


def detail_records():
    records = {}
    for section in CRITERION_SECTIONS:
        manifest = load_template(f"criterion/{section}")
        if manifest.get("execution") != "sequence":
            raise ValueError(f"TEMPLATE_CRITERION: {section} must be a sequence")
        records[section] = {}
        for identifier in manifest["steps"]:
            schema = load_record_template(identifier)["record_schema"]
            records[section][identifier.rsplit("/", 1)[1]] = " ".join(schema["required"])
    return records