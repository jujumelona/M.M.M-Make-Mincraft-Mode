"""Direct writer must provide bounded target-verified producers of API inputs."""
from __future__ import annotations

from minecraft_mod_ai.construction_direct_writer import _evidence


class _Index:
    def __init__(self):
        self.symbols = {
            "example.Registry#register": {
                "kind": "METHOD", "owner": "example.Registry",
                "name": "register", "return_type": "void", "parameters": [
                    {"type": "example.EntityType", "name": "entity_type"},
                ],
            },
            "example.EntityType": {
                "kind": "CLASS", "owner": "example.EntityType", "name": "EntityType",
            },
            "example.EntityType#builder": {
                "kind": "METHOD", "owner": "example.EntityType",
                "name": "builder", "return_type": "example.EntityType",
                "parameters": [], "static": True,
            },
            "example.EntityType#ENTITY_TYPE": {
                "kind": "FIELD", "owner": "example.EntityType",
                "name": "ENTITY_TYPE", "return_type": "example.EntityType",
                "parameters": [], "static": True,
            },
            "example.EntityType#member": {
                "kind": "METHOD", "owner": "example.EntityType",
                "name": "member", "return_type": "example.EntityType",
                "parameters": [], "static": False,
            },
        }
        self.member_result_keys = {
            "example.EntityType": {
                "example.EntityType#builder",
                "example.EntityType#ENTITY_TYPE",
                "example.EntityType#member",
            }
        }
        self.by_owner = {}
        for key, row in self.symbols.items():
            self.by_owner.setdefault(row['owner'], []).append(key)

    def prepare(self, intent, *, kinds):
        assert intent
        return "goal"

    def page(self, goal, offset):
        assert goal == "goal" and offset == 0
        return {
            "matches": [{"symbol_id": "example.Registry#register"}],
            "source_candidates": [],
        }

    def accessible(self, key, constructor=False):
        return True


def test_typed_api_inputs_have_small_verified_producer_shortlist():
    result = _evidence(_Index(), "Register entity type", {})
    assert len(result["typed_input_producers"]) == 1
    entry = result["typed_input_producers"][0]
    assert entry["required_type"] == "example.EntityType"
    assert [x["symbol_id"] for x in entry["verified_producers"]] == [
        "example.EntityType#builder", "example.EntityType#ENTITY_TYPE"
    ]
    assert all("member" not in x["symbol_id"] for x in entry["verified_producers"])


def test_direct_evidence_excludes_unrelated_entity_registrars_and_client_rendering():
    index = _Index()
    wrong = [
        "net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityDataRegistry",
        "net.fabricmc.fabric.api.registry.fluid.EntityFluidInteractionRegistry",
        "net.minecraft.client.renderer.entity.EntityRenderers",
    ]
    for owner in wrong:
        index.symbols[owner + "#register"] = {
            "kind": "METHOD", "owner": owner, "name": "register",
            "return_type": "void", "parameters": [], "side": "CLIENT" if "client" in owner else "COMMON",
        }
    def candidates(_goal, _offset):
        return {
            "matches": [
                {"symbol_id": item + "#register"} for item in wrong
            ] + [{"symbol_id": "example.Registry#register"}],
            "source_candidates": [],
        }
    index.page = candidates
    result = _evidence(index, "Register alien entity type on the common side", {})
    assert result["target_api"] == [{"symbol_id": "example.Registry#register"}]
    assert result["rejected_unrelated_api_count"] == 3


def test_unimplemented_current_owner_is_not_presented_as_an_available_api():
    index = _Index()
    index.owner = "example.GeneratedAlienRegistry"
    own = index.owner + "#registerAlienCombatHostileEntity"
    index.symbols[own] = {
        "kind": "METHOD", "owner": index.owner,
        "name": "registerAlienCombatHostileEntity", "return_type": "void",
        "parameters": [], "provenance": "construction_plan",
    }
    original = index.page
    index.page = lambda goal, offset: {
        **original(goal, offset), "matches": [{"symbol_id": own},
                                               {"symbol_id": "example.Registry#register"}],
    }
    evidence = _evidence(index, "Register alien combat hostile entity", {})
    assert evidence["target_api"] == [{"symbol_id": "example.Registry#register"}]
