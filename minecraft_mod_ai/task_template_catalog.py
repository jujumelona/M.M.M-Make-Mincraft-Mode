"""Load host-owned task contracts. A record layout has one authoritative source."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).with_name("templates")

CRITERION_SECTIONS = (
    "behavior_contract", "state_model", "algorithm", "integration",
    "authority_and_network", "persistence", "resources_and_ui",
    "failure_and_limits", "reuse_assessment", "verification",
)
_CRITERION_ALIASES = {f"feature/{section}": f"criterion/{section}" for section in CRITERION_SECTIONS}


@lru_cache(maxsize=None)
def _load(identifier):
    path = (ROOT / (identifier + ".yaml")).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("TEMPLATE_PATH: identifier escapes catalog")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != identifier:
        raise ValueError(f"TEMPLATE_ID: invalid template {identifier}")
    for key in ("record_schema", "input_schema", "output_schema"):
        if key in value:
            Draft202012Validator.check_schema(value[key])
    return value


def load_template(identifier):
    """Load a manifest/template, including isolated criterion compatibility aliases."""
    return deepcopy(_load(_CRITERION_ALIASES.get(identifier, identifier)))


def load_record_template(identifier):
    """Load exactly the requested record template with no namespace aliasing."""
    value = deepcopy(_load(identifier))
    if "record_schema" not in value:
        raise ValueError(f"TEMPLATE_RECORD_SCHEMA: missing record schema for {identifier}")
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
