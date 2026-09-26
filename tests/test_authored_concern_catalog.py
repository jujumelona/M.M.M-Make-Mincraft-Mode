from __future__ import annotations

from minecraft_mod_ai.authored_execution_schema import (
    EXECUTION_SECTION_ORDER,
    concern_contracts,
    section_for_symbol,
)
from minecraft_mod_ai.task_template_catalog import load_template


def test_authored_concern_catalog_matches_canonical_manifest_order():
    for section in EXECUTION_SECTION_ORDER:
        manifest = load_template(f"criterion/{section}")
        contracts = concern_contracts(section)
        assert [item["identifier"] for item in contracts] == list(manifest["steps"])
        assert [item["sequence"] for item in contracts] == list(range(len(contracts)))
        assert all(item["task"] for item in contracts)
        assert all(isinstance(item["record_schema"], dict) for item in contracts)


def test_authored_concern_catalog_drops_record_loop_control_rules():
    for section in EXECUTION_SECTION_ORDER:
        for contract in concern_contracts(section):
            lowered = [rule.casefold() for rule in contract["rules"]]
            assert not any(rule.startswith("return ") for rule in lowered)


def test_refinement_part_symbols_inherit_only_their_canonical_authored_section():
    assert section_for_symbol("AuthoredStateModel") == "state_model"
    assert section_for_symbol("AuthoredStateModelPart1") == "state_model"
    assert section_for_symbol("AuthoredStateModelPart1Part2") == "state_model"
    assert section_for_symbol("BehaviorContractPart6") == ""
    assert section_for_symbol("AuthoredStateModelHelper1") == ""
