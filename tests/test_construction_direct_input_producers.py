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
