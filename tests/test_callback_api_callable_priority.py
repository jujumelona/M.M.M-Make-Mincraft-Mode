"""Real production failure: the callback evidence contained only class constants.

The method sampler must expose callable operations before incidental static
fields whenever the query has no exact overload name in the prompt.
"""
from __future__ import annotations

from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence


def test_bounded_java_callback_evidence_prioritizes_callable_methods():
    index = object.__new__(FileAPIEvidence)
    index.superclasses = set()
    owner = "proof.CallbackInput"
    members = [
        (f"field_{i}", {
            "kind": "FIELD", "owner": owner, "name": f"AAA_STATIC_{i:04d}",
            "return_type": "int", "static": True, "access": 25,
        })
        for i in range(40)
    ]
    members += [
        ("method_attack", {
            "kind": "METHOD", "owner": owner, "name": "attack",
            "parameters": ["proof.Target"], "return_type": "void",
            "static": False, "access": 1,
        }),
        ("method_apply", {
            "kind": "METHOD", "owner": owner, "name": "apply",
            "parameters": [], "return_type": "boolean",
            "static": False, "access": 1,
        }),
    ]
    index._member_options = lambda name, subclass=False: members
    _, chosen = index._selected_members(owner, set(), set())
    assert len(chosen) == 16
    assert {"attack", "apply"} <= {row["name"] for _, row in chosen}
    assert any(row["kind"] == "METHOD" for _, row in chosen)
    assert sum(row["kind"] == "FIELD" for _, row in chosen) == 14


def test_exact_verified_symbol_names_still_win_over_kind_ranking():
    index = object.__new__(FileAPIEvidence)
    index.superclasses = set()
    owner = "proof.CallbackInput"
    members = [
        ("field_0", {"kind": "FIELD", "owner": owner,
                     "name": "IMPORTANT_FIELD", "return_type": "int"}),
        ("method_0", {"kind": "METHOD", "owner": owner,
                      "name": "action", "return_type": "void", "parameters": []}),
    ]
    index._member_options = lambda name, subclass=False: members
    _, chosen = index._selected_members(owner, {"IMPORTANT_FIELD"}, set())
    assert chosen[0][1]["name"] == "IMPORTANT_FIELD"
