from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "minecraft_mod_ai" / "templates"
FORBIDDEN_DESIGN_RULE_FRAGMENTS = (
    "Must cite concrete Minecraft interaction mechanics.",
    "Must stay strictly under 256 characters.",
    "Return not_applicable",
    "Return blocked",
)
PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"template must be a mapping: {path}"
    return data


def test_all_canonical_templates_parse_and_ids_match_paths() -> None:
    for path in sorted(TEMPLATES.rglob("*.yaml")):
        data = _load(path)
        template_id = data.get("id")
        if template_id is None:
            continue
        expected = path.relative_to(TEMPLATES).with_suffix("").as_posix()
        assert template_id == expected, f"id/path mismatch: {path}: {template_id!r} != {expected!r}"


def test_design_record_schemas_are_closed_and_internally_consistent() -> None:
    for path in sorted((TEMPLATES / "design").glob("*.yaml")):
        data = _load(path)
        schema = data.get("record_schema")
        if schema is None:
            continue
        assert schema.get("type") == "object", f"record_schema must be object: {path}"
        assert schema.get("additionalProperties") is False, f"record_schema must be closed: {path}"
        properties = schema.get("properties")
        required = schema.get("required")
        assert isinstance(properties, dict) and properties, f"missing properties: {path}"
        assert isinstance(required, list) and required, f"missing required list: {path}"
        unknown_required = set(required) - set(properties)
        assert not unknown_required, f"required fields missing schemas in {path}: {sorted(unknown_required)}"


def test_design_templates_do_not_reintroduce_invalid_generic_or_sentinel_rules() -> None:
    for path in sorted((TEMPLATES / "design").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_DESIGN_RULE_FRAGMENTS:
            assert fragment not in text, f"forbidden rule fragment {fragment!r} in {path}"


def test_asset_render_placeholders_have_explicit_required_input_contracts() -> None:
    for path in sorted((TEMPLATES / "asset").glob("*.yaml")):
        data = _load(path)
        render = data.get("render")
        if render is None:
            continue
        assert isinstance(render, dict), f"asset render must be a mapping: {path}"
        body = render.get("body")
        assert isinstance(body, str) and body, f"asset render body must be non-empty: {path}"
        inputs = data.get("inputs")
        placeholders = set(PLACEHOLDER.findall(body))
        if not placeholders:
            # HOST-owned static prompt fragments consume resolved authorities rather than
            # exposing model/backend knobs as string substitutions.
            requires = data.get("requires")
            assert isinstance(requires, list) and requires, f"static asset render must declare authorities: {path}"
            assert all(isinstance(value, str) and value for value in requires), path
            assert inputs in (None, {}), f"static asset render must not declare unused inputs: {path}"
            continue
        assert isinstance(inputs, dict), f"asset inputs must be declared: {path}"
        assert placeholders == set(inputs), f"asset placeholder/input mismatch: {path}"
        for name, contract in inputs.items():
            assert isinstance(contract, dict), f"asset input contract must be a mapping: {path}:{name}"
            assert contract.get("type") == "string", f"asset input must be a string: {path}:{name}"
            assert contract.get("required") is True, f"asset input must be required: {path}:{name}"
            assert contract.get("minLength", 0) >= 1, f"asset input must reject empty strings: {path}:{name}"


def test_translation_templates_type_every_declared_input_and_output() -> None:
    for path in sorted((TEMPLATES / "translation").glob("*.yaml")):
        data = _load(path)
        inputs = data.get("input")
        outputs = data.get("output")
        input_schema = data.get("input_schema")
        output_schema = data.get("output_schema")
        assert isinstance(inputs, dict) and inputs, f"translation inputs missing: {path}"
        assert isinstance(outputs, dict) and outputs, f"translation outputs missing: {path}"
        assert isinstance(input_schema, dict), f"translation input schema missing: {path}"
        assert isinstance(output_schema, dict), f"translation output schema missing: {path}"
        assert set(input_schema) == set(inputs), f"translation input/schema mismatch: {path}"
        assert set(output_schema) == set(outputs), f"translation output/schema mismatch: {path}"
        for name, type_spec in input_schema.items():
            assert isinstance(type_spec, str) and type_spec, f"untyped translation input: {path}:{name}"
        for name, type_spec in output_schema.items():
            assert isinstance(type_spec, str) and type_spec, f"untyped translation output: {path}:{name}"
