from copy import deepcopy
import json

import pytest
import yaml

from minecraft_mod_ai import atomic_slot_executor as slots
from minecraft_mod_ai.implementation_template_renderer import TemplateRenderError, render_template
from minecraft_mod_ai.task_template_catalog import load_template
from minecraft_mod_ai.template_contract_validation import validate_catalog, validate_template_contract


def _renderer_contract_fixture():
    return {
        "id": "test/renderer_contract",
        "inputs": {
            "visual_description": {
                "type": "string",
                "minLength": 1,
                "required": True,
            }
        },
        "render": {
            "language": "text",
            "body": "{{ visual_description }}",
        },
    }


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"visual_description": ""},
        {"visual_description": None},
        {"visual_description": "ice", "invented": "extra"},
    ],
)
def test_renderer_rejects_missing_blank_null_and_extra_declared_inputs(values):
    with pytest.raises(TemplateRenderError, match="RENDER_INPUT_CONTRACT"):
        render_template(_renderer_contract_fixture(), values)


def test_spaced_placeholders_use_the_same_contract_as_renderer():
    template = _renderer_contract_fixture()
    template["render"]["body"] = "{{ visual_description }}"
    assert render_template(template, {"visual_description": "ice"}) == "ice"
    template["inputs"]["unused"] = {"type": "string", "required": True}
    with pytest.raises(ValueError, match="TEMPLATE_PLACEHOLDERS"):
        render_template(template, {"visual_description": "ice", "unused": "extra"})


def test_item_sprite_is_static_framing_not_a_semantic_prompt_slot():
    template = load_template("asset/item_sprite")
    assert "inputs" not in template
    assert tuple(template["requires"]) == (
        "resolved_resource_contract",
        "resolved_generation_profile",
        "visual_spec",
    )
    body = template["render"]["body"]
    assert "{{" not in body
    assert "visual_description" not in body
    assert "model" not in body.casefold()
    assert "lora" not in body.casefold()


@pytest.mark.parametrize("mutation", ["missing_schema", "open_schema", "slot_default"])
def test_model_contracts_fail_before_generation(mutation):
    template = load_template("prompt/intent")
    if mutation == "missing_schema":
        del template["output_schema"]
    elif mutation == "open_schema":
        template["output_schema"]["additionalProperties"] = True
    else:
        template["ai_slots"] = [{"id": "value", "schema": {"type": "integer"}, "default": None}]
    with pytest.raises(Exception, match="TEMPLATE_|MODEL_JSON_TEMPLATE_REQUIRED"):
        validate_template_contract(template)


@pytest.mark.parametrize("steps,code", [(["missing"], "TEMPLATE_MISSING"),
                                       (["root"], "TEMPLATE_SEQUENCE_CYCLE"),
                                       (["leaf", "leaf"], "TEMPLATE_SEQUENCE_DUPLICATE")])
def test_catalog_rejects_broken_manifest_graph(tmp_path, steps, code):
    (tmp_path / "root.yaml").write_text(yaml.safe_dump({"id": "root", "execution": "sequence", "steps": steps}))
    (tmp_path / "leaf.yaml").write_text(yaml.safe_dump({"id": "leaf", "task": "fixture"}))
    with pytest.raises(ValueError, match=code):
        validate_catalog(tmp_path)


def test_catalog_requires_explicit_consumer_or_standalone(tmp_path):
    response_dir = tmp_path / "response"
    response_dir.mkdir()
    (response_dir / "contracts.json").write_text(
        json.dumps({"fixture": {"type": "object"}}),
        encoding="utf-8",
    )
    path = tmp_path / "leaf.yaml"
    path.write_text(yaml.safe_dump({"id": "leaf", "task": "fixture"}))
    with pytest.raises(ValueError, match="TEMPLATE_UNCONSUMED"):
        validate_catalog(tmp_path, consumer_roots=())
    with pytest.raises(ValueError, match="TEMPLATE_CONSUMER_MISSING"):
        validate_catalog(tmp_path, consumer_roots=("missing",))
    assert "leaf" in validate_catalog(tmp_path, consumer_roots=("leaf",))
    path.write_text(yaml.safe_dump({"id": "leaf", "task": "fixture", "standalone": True}))
    assert "leaf" in validate_catalog(tmp_path, consumer_roots=())


def test_slot_repair_keeps_only_latest_structured_error(monkeypatch):
    calls = []
    responses = iter([100, 99, 32])

    def generate(*args, **kwargs):
        calls.append(deepcopy(args[2]))
        return next(responses)

    monkeypatch.setattr(slots, "generate_fixed_template_value", generate)
    slot = slots.SlotDefinition("stack_limit", {"type": "integer", "minimum": 1, "maximum": 64})
    assert slots.fill_one_slot(object(), slot, {"evidence": "stack limit is 32"}) == 32
    assert [len(messages) for messages in calls] == [2, 3, 3]
    repair = json.loads(calls[-1][-1]["content"])
    assert repair["invalid_leaf"] == 99
    assert repair["validator_error"]["repair_scope"] == ["stack_limit"]
    assert repair["validator_error"]["expected"] == {"validator": "maximum", "constraint": 64}


def test_slot_does_not_treat_transport_failure_as_schema_repair(monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("transport")

    monkeypatch.setattr(slots, "generate_fixed_template_value", generate)
    with pytest.raises(TimeoutError):
        slots.fill_one_slot(object(), slots.SlotDefinition("x", {"type": "integer"}), {})
    assert calls == [1]


def test_exhausted_slot_returns_structured_diagnostic(monkeypatch):
    monkeypatch.setattr(slots, "generate_fixed_template_value", lambda *a, **k: 99)
    with pytest.raises(slots.SlotFillError) as failure:
        slots.fill_one_slot(object(), slots.SlotDefinition("x", {"type": "integer", "maximum": 2}), {}, max_retries=0)
    assert failure.value.diagnostic["actual"] == 99
    assert failure.value.diagnostic["repair_scope"] == ["x"]


def test_slot_context_does_not_coerce_unknown_objects():
    with pytest.raises(slots.SlotFillError, match="SLOT_CONTEXT_INVALID"):
        slots.fill_one_slot(object(), slots.SlotDefinition("x", {"type": "integer"}), {"unknown": object()})
