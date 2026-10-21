"""A state-authority change and its derived storage config are one plan change."""
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.construction_behavior_state import reconcile_behavior_state
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.production_state_compiler import compile_production_state_section
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.typed_host_generation_contract import (
    canonical_typed_state_store_config,
)
from minecraft_mod_ai.typed_plan_production import generate_typed_plan_module


def _section(name, **records):
    return {
        "specification": {**{key: [] for key in DETAIL_RECORDS[name]},
                          **records, "inapplicable_concerns": []},
        "constraint_evidence_refs": [],
    }


def _persistent_plan(*, variables=None, behaviors=None):
    variables = variables or [{"name": "credits", "owner": "player", "type": "long",
                               "unit": "credits", "default": "0", "domain": "nonnegative"}]
    text = "Saved state authority with persistent storage and synchronization"
    sections = {
        "state_model": _section("state_model", variables=variables),
        "persistence": _section(
            "persistence",
            stored_state=[{"state": variables[0]["name"], "owner": "player", "scope": "world"}],
            migration=[{"source_version": "legacy", "operation": "rename_key",
                        "source_key": "old_credits", "destination_key": variables[0]["name"],
                        "value": None}],
            transfers=[{"event": "respawn", "copy_rule": "copy player state", "ownership": "server"}],
        ),
        "authority_and_network": _section(
            "authority_and_network",
            synchronization=[{"state": variables[0]["name"], "recipients": ["owner"], "trigger": "change"}],
        ),
    }
    store = canonical_typed_state_store_config(
        structured_sections=sections, namespace="custom.player_state", transfer_required=True,
    )
    return AuthoredPlan(
        requested_prompt=text, text=text, structured_sections=sections,
        typed_plan_ir={
            "schema_version": "mmm/typed-plan-ir-v1",
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "functions": [], "initialize": [],
            "platform_modules": [
                {"module_id": "storage", "kind": "state_store", "config": store,
                 "covers": ["persistence.stored_state", "persistence.migration", "persistence.transfers"]},
                {"module_id": "sync", "kind": "network_sync",
                 "config": {"sync_interval_ticks": 40, "max_payload_bytes": 4096},
                 "covers": ["authority_and_network.synchronization"]},
            ],
        },
        request_scope={"requirements": [{"requirement_id": "r1", "statement": text}],
                       "behavior_requirements": behaviors or [{
                           "behavior_id": "b1", "requirement_id": "r1", "intent": "Unlock charge",
                           "trigger": "interaction", "effects": "Enable charge", "source_quote": "Enable charge",
                           "state_reads": [], "state_writes": ["charge"],
                       }]},
    )


class _Router:
    def __init__(self, answers=None):
        self.answers = answers or {"charge": {
            "binding": "new_scalar_state", "owner": "player", "type": "boolean",
            "unit": "enabled", "default": "false", "domain": "player",
        }}

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        context = json.loads(messages[-1]["content"])
        field = next(iter(parameters["properties"]))
        return {field: self.answers[context["original_state_reference"]][field]}


def _lower(plan):
    modules, manifest = _compile_new_authored_modules(
        plan, mod_id="rebinding", package_name="example.rebinding",
        target={"minecraft_version": "26.2"},
        production_state_section=compile_production_state_section(plan),
    )
    assert len(modules) == 1
    assert manifest["typed_program"]["state_store"] is True
    return modules[0]


def test_new_scalar_rebinds_storage_migrations_and_generated_sources(tmp_path):
    original = _persistent_plan()
    before = original.to_dict()
    old_config = original.typed_plan_ir["platform_modules"][0]["config"]
    result = reconcile_behavior_state(_Router(), original)
    module = _lower(result)
    store = module.config["typed_state_store"]

    assert store["schema_version"] != old_config["schema_version"]
    assert store["namespace"] == "custom.player_state"
    assert store["transfer_on_respawn"] is True
    assert store["migrations"] == [old_config["migrations"][0] | {"to_version": store["schema_version"]}]
    assert result.typed_plan_ir["platform_modules"][1] == original.typed_plan_ir["platform_modules"][1]
    assert original.to_dict() == before
    assert result.text == original.text

    spec = ModSpec(
        mod_id="rebinding", mod_name="Rebinding", package_name="example.rebinding",
        version="1.0.0", summary="State contract regression",
        contents=(ContentSpec(content_id="marker", kind=ContentKind.ITEM,
                              display_name_en="Marker", display_name_ko="마커"),),
    )
    FabricProjectGenerator().generate(spec, tmp_path)
    generate_typed_plan_module(tmp_path, module=module)
    package = tmp_path / "src/main/java/example/rebinding"
    state_source = (package / "AuthoredStateModel.java").read_text()
    persistence_source = (package / "AuthoredStatePersistence.java").read_text()
    network_source = (package / "AuthoredNetworkSync.java").read_text()
    assert "boolean readState_charge(" in state_source
    assert store["schema_version"] in persistence_source
    assert old_config["schema_version"] not in persistence_source
    assert '"charge"' in persistence_source
    assert '"charge"' in network_source


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unrelated-schema"),
    ("migrations", []),
    ("malformed_policy", "discard"),
    ("transfer_on_respawn", False),
    ("unknown", "unrecognized setting"),
])
def test_rebinding_rejects_preexisting_config_drift(field, value):
    original = _persistent_plan()
    typed = deepcopy(original.typed_plan_ir)
    typed["platform_modules"][0]["config"][field] = value
    original = replace(original, typed_plan_ir=typed)
    before = original.to_dict()
    with pytest.raises(ValueError, match="TYPED_STATE_STORE_CONFIG_(DRIFT|UNKNOWN_FIELDS)"):
        reconcile_behavior_state(_Router(), original)
    assert original.to_dict() == before


@pytest.mark.parametrize("field,value", [("namespace", " custom.player_state "),
                                       ("transfer_on_respawn", 1)])
def test_rebinding_cannot_sanitize_invalid_non_schema_settings(field, value):
    original = _persistent_plan()
    typed = deepcopy(original.typed_plan_ir)
    typed["platform_modules"][0]["config"][field] = value
    original = replace(original, typed_plan_ir=typed)
    with pytest.raises(ValueError, match="TYPED_PLATFORM_CONFIG_INVALID"):
        _lower(reconcile_behavior_state(_Router(), original))


def test_rebinding_cannot_fill_missing_required_migrations():
    original = _persistent_plan()
    typed = deepcopy(original.typed_plan_ir)
    del typed["platform_modules"][0]["config"]["migrations"]
    original = replace(original, typed_plan_ir=typed)
    with pytest.raises(ValueError, match="migration coverage requires typed migrations"):
        _lower(original)
    with pytest.raises(ValueError, match="migration coverage requires typed migrations"):
        _lower(reconcile_behavior_state(_Router(), original))


@pytest.mark.parametrize("binding", [
    {"binding": "canonical_state", "state_name": "credits"},
    {"binding": "world_state", "observation": "live entity energy"},
])
def test_alias_and_world_binding_preserve_existing_storage_contract(binding):
    original = _persistent_plan()
    result = reconcile_behavior_state(_Router({"charge": binding}), original)
    assert result.typed_plan_ir == original.typed_plan_ir
    _lower(result)


def test_state_store_rebinding_survives_checkpoint_and_saved_plan_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    original = _persistent_plan()
    first = reconcile_behavior_state(_Router(), original)
    cached = reconcile_behavior_state(_Router({"unused": {}}), original)
    restored = AuthoredPlan.from_dict(json.loads(json.dumps(first.to_dict())))
    repeated = reconcile_behavior_state(_Router({"unused": {}}), restored)
    assert first.to_dict() == cached.to_dict() == repeated.to_dict()
    assert _lower(cached).config == _lower(repeated).config


def test_captured_67_reference_state_binding_reaches_production_lowering():
    fixture = json.loads((Path(__file__).parent / "fixtures/behavior-state-bindings-67.json").read_text())
    original = _persistent_plan(variables=fixture["variables"], behaviors=fixture["behaviors"])
    result = reconcile_behavior_state(_Router(fixture["answers"]), original)
    module = _lower(result)
    bindings = result.request_scope["behavior_state_bindings"]
    assert len(bindings) == 67
    assert sum(row["binding"] == "world_state" for row in bindings.values()) == 7
    assert len(module.config["typed_plan_state_section"]["specification"]["variables"]) == 2
    assert module.config["typed_state_store"]["schema_version"] != original.typed_plan_ir["platform_modules"][0]["config"]["schema_version"]
