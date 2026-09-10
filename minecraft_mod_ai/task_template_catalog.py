"""Load host-owned task contracts. A record layout has one authoritative source."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).with_name("templates")


@lru_cache(maxsize=None)
def _load(identifier):
    path = (ROOT / (identifier + ".yaml")).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("TEMPLATE_PATH: identifier escapes catalog")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != identifier:
        raise ValueError(f"TEMPLATE_ID: invalid template {identifier}")
    if "record_schema" in value:
        Draft202012Validator.check_schema(value["record_schema"])
    return value


def load_template(identifier):
    return deepcopy(_load(identifier))


def detail_records():
    records = {}
    for path in sorted((ROOT / "feature").glob("*.yaml")):
        manifest = load_template(f"feature/{path.stem}")
        if manifest.get("execution") != "sequence":
            continue
        records[path.stem] = {}
        for identifier in manifest["steps"]:
            schema = load_template(identifier)["record_schema"]
            records[path.stem][identifier.rsplit("/", 1)[1]] = " ".join(schema["required"])
    return records
